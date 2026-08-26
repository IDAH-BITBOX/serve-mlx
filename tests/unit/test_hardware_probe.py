from __future__ import annotations

import subprocess

from mlx_moe_stream.hardware import (
    HardwareProfile,
    _parse_vm_stat_compressor_stats,
    probe_hardware,
)
from mlx_moe_stream.memory import MemorySnapshot

# Captured verbatim from `vm_stat` on the reference M4 mac mini (16GB); field
# names and the page-size header are the real strings/values vm_stat prints.
_SAMPLE_VM_STAT_OUTPUT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               16277.
Pages active:                            448756.
Pages inactive:                          449009.
Pages speculative:                         3661.
Pages throttled:                              0.
Pages wired down:                         71810.
Pages purgeable:                            1807.
"Translation faults":                  59159607.
Pages copy-on-write:                    3220856.
Pages zero filled:                     29816857.
Pages reactivated:                     10173800.
Pages purged:                            2757594.
File-backed pages:                       429319.
Anonymous pages:                         472107.
Pages stored in compressor:              122733.
Pages occupied by compressor:             25010.
Decompressions:                          889692.
Compressions:                           1339978.
Pageins:                             2825892634.
Pageouts:                                 94407.
Swapins:                                      0.
Swapouts:                                     0.
"""


def _snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=1.0,
        physical_memory_bytes=17_179_869_184,
        recommended_working_set_bytes=12_713_115_648,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=0,
        swap_used_bytes=0,
        swap_free_bytes=0,
        device_name="Apple M4",
    )


class _CompletedProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_probe_hardware_reuses_the_given_snapshot_without_calling_device_info(monkeypatch):
    """probe_hardware must not call mx.device_info() itself; only OS probes run."""

    calls: list[list[str]] = []

    def fake_run(cmd, *, check, capture_output, text, timeout):
        calls.append(cmd)
        assert timeout == 1.0
        if cmd[0] == "ioreg":
            return _CompletedProcess('| | |   "gpu-core-count" = 10\n')
        if cmd[0] == "sysctl" and cmd[-2:] == [
            "hw.perflevel0.logicalcpu",
            "hw.perflevel1.logicalcpu",
        ]:
            return _CompletedProcess("4\n6\n")
        if cmd[0] == "sysctl" and cmd[-1] == "iogpu.wired_limit_mb":
            return _CompletedProcess("0\n")
        if cmd[0] == "vm_stat":
            return _CompletedProcess(_SAMPLE_VM_STAT_OUTPUT)
        raise AssertionError(f"unexpected command {cmd!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    profile = probe_hardware(_snapshot())

    assert profile == HardwareProfile(
        device_name="Apple M4",
        physical_memory_bytes=17_179_869_184,
        recommended_working_set_bytes=12_713_115_648,
        gpu_core_count=10,
        cpu_performance_cores=4,
        cpu_efficiency_cores=6,
        wired_limit_mb=0,
        disk_free_bytes=profile.disk_free_bytes,
        disk_total_bytes=profile.disk_total_bytes,
        vm_page_size_bytes=16384,
        compressor_pages_stored=122733,
        compressor_pages_occupied=25010,
        compressor_compressions=1339978,
        compressor_decompressions=889692,
    )
    # No mx.device_info() call means every subprocess invocation is one of the
    # four known OS probes, and the snapshot's own fields pass through untouched.
    assert len(calls) == 4


def test_probe_hardware_degrades_to_none_on_timeout(monkeypatch):
    def fake_run(cmd, *, check, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    profile = probe_hardware(_snapshot())

    assert profile.gpu_core_count is None
    assert profile.cpu_performance_cores is None
    assert profile.cpu_efficiency_cores is None
    assert profile.wired_limit_mb is None
    assert profile.vm_page_size_bytes is None
    assert profile.compressor_pages_stored is None
    assert profile.compressor_pages_occupied is None
    assert profile.compressor_compressions is None
    assert profile.compressor_decompressions is None
    # Snapshot-derived fields are unaffected by OS probe failures.
    assert profile.device_name == "Apple M4"
    assert profile.physical_memory_bytes == 17_179_869_184


def test_probe_hardware_degrades_to_none_when_binary_is_missing(monkeypatch):
    def fake_run(cmd, *, check, capture_output, text, timeout):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    profile = probe_hardware(_snapshot())

    assert profile.gpu_core_count is None
    assert profile.cpu_performance_cores is None
    assert profile.cpu_efficiency_cores is None
    assert profile.wired_limit_mb is None
    assert profile.vm_page_size_bytes is None
    assert profile.compressor_pages_stored is None
    assert profile.compressor_pages_occupied is None
    assert profile.compressor_compressions is None
    assert profile.compressor_decompressions is None


def test_probe_hardware_disk_usage_failure_degrades_to_none(monkeypatch):
    import shutil

    def fake_run(cmd, *, check, capture_output, text, timeout):
        return _CompletedProcess("0\n")

    def fake_disk_usage(path):
        raise OSError("no such volume")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    profile = probe_hardware(_snapshot(), disk_path="/does/not/exist")

    assert profile.disk_free_bytes is None
    assert profile.disk_total_bytes is None


def test_probe_hardware_populates_compressor_stats_from_vm_stat(monkeypatch):
    def fake_run(cmd, *, check, capture_output, text, timeout):
        if cmd == ["vm_stat"]:
            return _CompletedProcess(_SAMPLE_VM_STAT_OUTPUT)
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    profile = probe_hardware(_snapshot())

    assert profile.vm_page_size_bytes == 16384
    assert profile.compressor_pages_stored == 122733
    assert profile.compressor_pages_occupied == 25010
    assert profile.compressor_compressions == 1339978
    assert profile.compressor_decompressions == 889692


def test_probe_hardware_compressor_stats_degrade_to_none_on_vm_stat_timeout(monkeypatch):
    def fake_run(cmd, *, check, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    profile = probe_hardware(_snapshot())

    assert profile.vm_page_size_bytes is None
    assert profile.compressor_pages_stored is None
    assert profile.compressor_pages_occupied is None
    assert profile.compressor_compressions is None
    assert profile.compressor_decompressions is None


def test_parse_vm_stat_compressor_stats_returns_expected_ints():
    parsed = _parse_vm_stat_compressor_stats(_SAMPLE_VM_STAT_OUTPUT)

    assert parsed == {
        "page_size_bytes": 16384,
        "pages_stored": 122733,
        "pages_occupied": 25010,
        "compressions": 1339978,
        "decompressions": 889692,
    }


def test_parse_vm_stat_compressor_stats_degrades_to_none_on_missing_page_size_header():
    output = _SAMPLE_VM_STAT_OUTPUT.replace(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n", ""
    )

    assert _parse_vm_stat_compressor_stats(output) is None


def test_parse_vm_stat_compressor_stats_degrades_to_none_on_missing_field():
    lines = _SAMPLE_VM_STAT_OUTPUT.splitlines()
    output = "\n".join(line for line in lines if "Compressions:" not in line)

    assert _parse_vm_stat_compressor_stats(output) is None


def test_parse_vm_stat_compressor_stats_degrades_to_none_on_corrupted_number():
    output = _SAMPLE_VM_STAT_OUTPUT.replace(
        "Pages stored in compressor:              122733.",
        "Pages stored in compressor:              N/A.",
    )

    assert _parse_vm_stat_compressor_stats(output) is None


def test_parse_vm_stat_compressor_stats_degrades_to_none_on_empty_output():
    assert _parse_vm_stat_compressor_stats("") is None
