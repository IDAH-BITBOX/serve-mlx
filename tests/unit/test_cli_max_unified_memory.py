from __future__ import annotations

import argparse

import pytest
from mlx_moe_stream import cli
from mlx_moe_stream.cli import build_parser
from mlx_moe_stream.memory import MemoryBudgetManager, MemorySnapshot

_GIB = 1024**3


def _mac_mini_snapshot(*, include_os_metrics: bool = False) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=1.0,
        physical_memory_bytes=17_179_869_184,
        recommended_working_set_bytes=12_713_115_648,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=None,
        swap_used_bytes=None,
        swap_free_bytes=None,
        device_name="Apple M4",
    )


def _generate_args(**overrides) -> argparse.Namespace:
    parser = build_parser()
    argv = ["generate", "--manifest", "m.json", "--prompt", "hi"]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


def _serve_args(**overrides) -> argparse.Namespace:
    parser = build_parser()
    argv = ["serve", "--manifest", "m.json"]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


# Both serve and generate parse --max-unified-memory into the *same*
# cli._memory_config() function; this table drives the parametrized wiring
# tests below so a future change that only updates one subcommand's argv
# handling (splitting the shared path) fails loudly on the other.
_ARGS_BUILDERS = {"generate": _generate_args, "serve": _serve_args}


# --- _parse_max_unified_memory_bytes --------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("12", 12_000_000_000),
        ("12.5", 12_500_000_000),
        ("14GB", 14_000_000_000),
        ("14GiB", 14 * _GIB),
    ],
)
def test_parse_max_unified_memory_bytes(value, expected):
    assert cli._parse_max_unified_memory_bytes(value) == expected


def test_parse_max_unified_memory_bytes_rejects_invalid_values():
    with pytest.raises(ValueError):
        cli._parse_max_unified_memory_bytes("not-a-size")


# --- --max-unified-memory wiring into MemoryBudgetConfig ------------------


def test_max_unified_memory_defaults_to_unset(monkeypatch):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _generate_args()
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes is None


def test_max_unified_memory_bare_number_is_gigabytes(monkeypatch):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _generate_args(**{"max-unified-memory": "12"})
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes == 12_000_000_000


def test_max_unified_memory_accepts_existing_size_grammar(monkeypatch):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _generate_args(**{"max-unified-memory": "14GiB"})
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes == 14 * _GIB


def test_memory_safety_margin_adaptive_uses_adaptive_formula(monkeypatch):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _generate_args(**{"memory-safety-margin": "adaptive"})
    config = cli._memory_config(args)
    assert config.safety_margin_bytes == 1 * _GIB  # Mac mini M4/16GB: hits the 1GiB floor


def test_memory_safety_margin_auto_default_is_unchanged(monkeypatch):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _generate_args()
    config = cli._memory_config(args)
    assert config.safety_margin_bytes == 4 * _GIB  # unchanged automatic_safety_margin_bytes


# --- serve and generate must wire --max-unified-memory symmetrically -----
# Defect 1 was originally reported against serve only, but the argv -> config
# assembly (_memory_config) is shared by both subcommand parsers. These
# parametrized tests pin both paths so a future refactor that only rewires
# one subcommand fails here instead of shipping a serve/generate split again.


@pytest.mark.parametrize("command", ["generate", "serve"])
def test_max_unified_memory_defaults_to_unset_for_both_commands(monkeypatch, command):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _ARGS_BUILDERS[command]()
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes is None


@pytest.mark.parametrize("command", ["generate", "serve"])
def test_max_unified_memory_bare_number_is_gigabytes_for_both_commands(monkeypatch, command):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _ARGS_BUILDERS[command](**{"max-unified-memory": "12"})
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes == 12_000_000_000


@pytest.mark.parametrize("command", ["generate", "serve"])
def test_max_unified_memory_accepts_existing_size_grammar_for_both_commands(monkeypatch, command):
    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _ARGS_BUILDERS[command](**{"max-unified-memory": "14GiB"})
    config = cli._memory_config(args)
    assert config.explicit_working_set_bytes == 14 * _GIB


@pytest.mark.parametrize("command", ["generate", "serve"])
def test_max_unified_memory_reaches_the_budget_plan_for_both_commands(monkeypatch, command):
    """End-to-end: argv -> cli._memory_config() -> MemoryBudgetManager.plan().

    This is the actual failure mode behind defect 1: explicit_working_set_bytes
    parsed correctly out of argv but never reached plan() because it was
    dropped somewhere between _memory_config() and the manager (in this repo's
    case, inside memory_config_with_kv_reserve() -- see test_kv_cache.py for the
    dedicated regression test of that function). Here we drive it straight from
    argv with no kv_cache indirection to pin the CLI half of the path.
    """

    monkeypatch.setattr(cli, "collect_memory_snapshot", lambda **_: _mac_mini_snapshot())
    args = _ARGS_BUILDERS[command](**{"max-unified-memory": "14GiB"})
    config = cli._memory_config(args)

    manager = MemoryBudgetManager(config, snapshot_provider=lambda: _mac_mini_snapshot())
    decision = manager.plan(
        shell_bytes=1_000_000_000,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )

    assert decision.working_set_source == "explicit"
    assert decision.working_set_bytes == 14 * _GIB
