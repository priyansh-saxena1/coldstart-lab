"""Environment probing and cold-cache helpers.

A cold-start benchmark is only meaningful if the OS page cache is actually
cold. On the second read of a file, Linux serves it straight from RAM and you
end up measuring memory bandwidth instead of storage bandwidth. The functions
here make the "cold" in cold-start real where the platform allows it, and fall
back to honest degradation (with a recorded warning) where it doesn't -- e.g.
an unprivileged Colab runtime cannot write to /proc/sys/vm/drop_caches.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class GpuInfo:
    name: str
    memory_total_mib: int
    driver_version: str


@dataclass
class SystemFingerprint:
    """Everything a reviewer needs to reproduce or discount a measurement."""

    python_version: str
    platform: str
    cpu_count: int
    total_ram_gib: float
    torch_version: Optional[str]
    cuda_available: bool
    gpus: List[GpuInfo] = field(default_factory=list)
    page_cache_control: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _read_gpus() -> List[GpuInfo]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, mem, driver = parts
        try:
            gpus.append(GpuInfo(name=name, memory_total_mib=int(float(mem)), driver_version=driver))
        except ValueError:
            continue
    return gpus


def probe() -> SystemFingerprint:
    warnings: List[str] = []

    torch_version: Optional[str] = None
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        warnings.append("torch is not installed; GPU experiments unavailable.")

    total_ram_gib = 0.0
    try:
        total_ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError):
        warnings.append("Could not read total RAM via sysconf.")

    fp = SystemFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        total_ram_gib=total_ram_gib,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpus=_read_gpus(),
        page_cache_control=_can_drop_caches(),
        warnings=warnings,
    )
    if not fp.page_cache_control:
        fp.warnings.append(
            "Cannot drop the OS page cache (needs root). Cold reads are "
            "emulated with posix_fadvise(DONTNEED), which is best-effort."
        )
    return fp


def _can_drop_caches() -> bool:
    return os.geteuid() == 0 and os.path.exists("/proc/sys/vm/drop_caches")


def drop_page_cache(path: Optional[str] = None) -> bool:
    """Evict cached file pages so the next read hits the storage device.

    Returns True if a system-wide drop succeeded. If we lack privileges we try
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` on the specific file, which is the
    portable, unprivileged fallback and is usually enough to defeat the read
    cache for a single checkpoint.
    """

    if _can_drop_caches():
        try:
            with open("/proc/sys/vm/drop_caches", "w") as fh:
                fh.write("3\n")
            return True
        except OSError:
            pass

    if path is not None and os.path.exists(path):
        _fadvise_dontneed(path)
    return False


def _fadvise_dontneed(path: str) -> None:
    # POSIX_FADV_DONTNEED == 4 on Linux. We call libc directly to avoid a hard
    # dependency on os.posix_fadvise being present for every file type.
    POSIX_FADV_DONTNEED = 4
    for root, files in _walk_files(path):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                fd = os.open(fpath, os.O_RDONLY)
            except OSError:
                continue
            try:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
                else:  # pragma: no cover - non-Linux fallback
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)


def _walk_files(path: str):
    if os.path.isfile(path):
        yield os.path.dirname(path), [os.path.basename(path)]
        return
    for root, _dirs, files in os.walk(path):
        yield root, files
