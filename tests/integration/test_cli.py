import json
from pathlib import Path

import pytest

from mlx_moe_stream import cli
from mlx_moe_stream.cli import build_parser, main
from mlx_moe_stream.routing import RouteEvent, RouteTracer


def test_simulate_command_reads_a_trace_and_writes_summary(tmp_path: Path, capsys):
    trace_path = tmp_path / "routes.jsonl"
    summary_path = tmp_path / "summary.json"
    with RouteTracer(request_id="cli", output_path=trace_path) as tracer:
        with tracer.model_call("prefill", 2):
            tracer.record_routes(0, [[[0], [0]]], [[[1.0], [1.0]]], num_experts=2)

    assert main(["simulate", "--trace", str(trace_path), "--output", str(summary_path)]) == 0
    assert json.loads(summary_path.read_text())["events"] == 2
    assert '"cache_simulation"' in capsys.readouterr().out


def test_serve_requires_a_manifest():
    with pytest.raises(SystemExit, match="2"):
        main(["serve"])


def test_serve_help_renders_the_auto_memory_safety_margin(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["serve", "--help"])

    assert "reserves 25% of physical memory" in capsys.readouterr().out


def test_kv_cache_options_parse_for_generate_and_serve():
    parser = build_parser()

    generate = parser.parse_args(
        [
            "generate",
            "--manifest",
            "prepared/manifest.json",
            "--prompt",
            "hello",
            "--kv-cache",
            "8bit",
            "--kv-max-context",
            "8192",
        ]
    )
    serve = parser.parse_args(
        [
            "serve",
            "--manifest",
            "prepared/manifest.json",
            "--kv-cache",
            "4bit",
            "--prefill-step-size",
            "256",
        ]
    )

    assert (generate.kv_cache, generate.kv_max_context) == ("8bit", 8192)
    assert serve.kv_cache == "4bit"
    assert serve.prefill_step_size == 256
    assert generate.memory_safety_margin == serve.memory_safety_margin == "auto"


def test_vision_serve_defaults_prioritize_unified_memory_headroom():
    parser = build_parser()
    vision = parser.parse_args(["serve", "--manifest", "prepared/manifest.json", "--vision"])
    text = parser.parse_args(["serve", "--manifest", "prepared/manifest.json"])

    assert vision.resident_budget is None
    assert vision.prefill_step_size is None
    assert cli._serve_resident_budget(None, vision=True) == (None, False)
    assert cli._serve_resident_budget(None, vision=False) == (None, True)
    assert cli._serve_prefill_step_size(None, vision=True) == 256
    assert cli._serve_prefill_step_size(None, vision=False) == 2_048
    assert cli._serve_prefill_step_size(512, vision=True) == 512
    assert text.resident_budget is None


def test_serve_rejects_non_loopback_hosts_before_loading_a_model(tmp_path: Path):
    assert main(["serve", "--manifest", str(tmp_path / "missing.json"), "--host", "0.0.0.0"]) == 2


def test_serve_registers_multiple_models_without_loading_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seen = []

    def stop_immediately(server):
        seen.append(server.service)

    class FakeServer:
        def __init__(self, host, port, service):
            del host, port
            self.service = service
            self.server_address = ("127.0.0.1", 8000)

    monkeypatch.setattr(cli, "run_local_server", stop_immediately)
    monkeypatch.setattr(cli, "LocalApiServer", FakeServer)
    assert (
        main(
            [
                "serve",
                "--model",
                f"qwen={tmp_path / 'qwen.json'}",
                "--model",
                f"gemma={tmp_path / 'gemma.json'}",
                "--model-id",
                "gemma",
            ]
        )
        == 0
    )
    assert seen[0].models()["data"][1]["id"] == "gemma"
    assert seen[0].registry.snapshot()["loads_total"] == 0


def test_train_predictor_command_writes_a_manifest_compatible_plan(tmp_path: Path, capsys):
    trace_path = tmp_path / "routes.jsonl"
    output_path = tmp_path / "predictor.json"
    events = [
        RouteEvent(
            request_id="test",
            phase="decode",
            token_index=0,
            layer_id=layer,
            expert_ids=(expert,),
            router_scores=(1.0,),
            timestamp="2026-01-01T00:00:00+00:00",
            num_experts=4,
            top_k=1,
        )
        for layer, expert in ((0, 1), (1, 2))
    ]
    trace_path.write_text("".join(json.dumps(event.to_dict()) + "\n" for event in events))

    assert (
        main(
            [
                "train-predictor",
                "--trace",
                str(trace_path),
                "--model-type",
                "qwen3_moe",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text())["transitions"]["0"]["1"][0]["expert"] == 2
    assert json.loads(capsys.readouterr().out)["source_experts"] == 1
