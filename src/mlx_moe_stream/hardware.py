"""Cheap, monkeypatchable OS/GPU probes used to size the M7 memory budget.

Every probe here is a millisecond-scale ``subprocess`` call bounded by a
short timeout; on any failure (missing binary, timeout, unparsable output)
the corresponding field degrades to ``None`` rather than raising, matching
the ``_swap_usage`` convention in :mod:`mlx_moe_stream.memory`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .memory import MemorySnapshot

DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware facts used to reason about M7 memory budgets."""

    device_name: str | None
    physical_memory_bytes: int
    recommended_working_set_bytes: int
    gpu_core_count: int | None
    cpu_performance_cores: int | None
    cpu_efficiency_cores: int | None
    wired_limit_mb: int | None
    disk_free_bytes: int | None
    disk_total_bytes: int | None
    vm_page_size_bytes: int | None
    compressor_pages_stored: int | None
    compressor_pages_occupied: int | None
    compressor_compressions: int | None
    compressor_decompressions: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_hardware(
    snapshot: MemorySnapshot,
    *,
    disk_path: Path | str = ".",
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> HardwareProfile:
    """Build a :class:`HardwareProfile` from an existing memory snapshot.

    ``snapshot`` must come from ``collect_memory_snapshot()``; this function
    never calls ``mx.device_info()`` itself so callers pay for that probe at
    most once per process. The remaining OS probes (``ioreg``, ``sysctl``,
    disk usage) are each bounded by ``timeout`` seconds and degrade to
    ``None`` on any failure instead of raising.
    """

    gpu_core_count = _gpu_core_count(timeout=timeout)
    performance_cores, efficiency_cores = _cpu_core_split(timeout=timeout)
    wired_limit_mb = _sysctl_int("iogpu.wired_limit_mb", timeout=timeout)
    disk_free_bytes, disk_total_bytes = _disk_usage(disk_path)
    compressor_stats = _vm_stat_compressor_stats(timeout=timeout)
    if compressor_stats is None:
        vm_page_size_bytes = None
        compressor_pages_stored = None
        compressor_pages_occupied = None
        compressor_compressions = None
        compressor_decompressions = None
    else:
        vm_page_size_bytes = compressor_stats["page_size_bytes"]
        compressor_pages_stored = compressor_stats["pages_stored"]
        compressor_pages_occupied = compressor_stats["pages_occupied"]
        compressor_compressions = compressor_stats["compressions"]
        compressor_decompressions = compressor_stats["decompressions"]
    return HardwareProfile(
        device_name=snapshot.device_name,
        physical_memory_bytes=snapshot.physical_memory_bytes,
        recommended_working_set_bytes=snapshot.recommended_working_set_bytes,
        gpu_core_count=gpu_core_count,
        cpu_performance_cores=performance_cores,
        cpu_efficiency_cores=efficiency_cores,
        wired_limit_mb=wired_limit_mb,
        disk_free_bytes=disk_free_bytes,
        disk_total_bytes=disk_total_bytes,
        vm_page_size_bytes=vm_page_size_bytes,
        compressor_pages_stored=compressor_pages_stored,
        compressor_pages_occupied=compressor_pages_occupied,
        compressor_compressions=compressor_compressions,
        compressor_decompressions=compressor_decompressions,
    )


def _gpu_core_count(*, timeout: float) -> int | None:
    try:
        result = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator", "-d1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', result.stdout)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _cpu_core_split(*, timeout: float) -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None, None
    try:
        return int(lines[0]), int(lines[1])
    except ValueError:
        return None, None


def _sysctl_int(name: str, *, timeout: float) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return int(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _disk_usage(path: Path | str) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
        return usage.free, usage.total
    except OSError:
        return None, None


# vm_stat's header line looks like:
#   Mach Virtual Memory Statistics: (page size of 16384 bytes)
_VM_STAT_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")

# Field label (as printed by vm_stat, quotes stripped) -> key used in the
# dict returned by _parse_vm_stat_compressor_stats. Only fields that actually
# appear in vm_stat's output on this platform are listed here; nothing here
# is invented.
_VM_STAT_COMPRESSOR_FIELDS = {
    "Pages stored in compressor": "pages_stored",
    "Pages occupied by compressor": "pages_occupied",
    "Compressions": "compressions",
    "Decompressions": "decompressions",
}


def _parse_vm_stat_compressor_stats(output: str) -> dict[str, int] | None:
    """Parse the memory-compressor counters out of raw ``vm_stat`` output.

    Returns a dict with keys ``page_size_bytes``, ``pages_stored``,
    ``pages_occupied``, ``compressions`` and ``decompressions`` -- all ints --
    or ``None`` if the page size header or any of the four compressor fields
    is missing or unparsable, so the caller can degrade the whole group at
    once (matching ``_cpu_core_split``'s all-or-nothing behavior).
    """
    size_match = _VM_STAT_PAGE_SIZE_RE.search(output)
    if size_match is None:
        return None
    try:
        page_size_bytes = int(size_match.group(1))
    except ValueError:
        return None

    values: dict[str, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        label, _, rest = line.partition(":")
        label = label.strip().strip('"')
        field = _VM_STAT_COMPRESSOR_FIELDS.get(label)
        if field is None:
            continue
        rest = rest.strip().rstrip(".")
        try:
            values[field] = int(rest.replace(",", ""))
        except ValueError:
            return None

    if len(values) != len(_VM_STAT_COMPRESSOR_FIELDS):
        return None

    values["page_size_bytes"] = page_size_bytes
    return values


def _vm_stat_compressor_stats(*, timeout: float) -> dict[str, int] | None:
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _parse_vm_stat_compressor_stats(result.stdout)
