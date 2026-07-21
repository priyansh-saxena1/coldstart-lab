"""Staging model weights onto a storage tier, and dropping the page cache.

Pipeshift's stated setup keeps weights/caches on a shared filesystem attached to
the cluster. The single most relevant experiment we can reproduce on commodity
hardware is: does it matter whether the weights sit on network-attached storage
vs local disk? On Colab you can approximate "network-attached" with a mounted
Google Drive and "local NVMe" with the instance's own disk.

Two things have to be right for that comparison to mean anything:
  1. The bytes actually have to come off the target tier, not out of RAM. Hence
     drop_page_cache() below.
  2. The staging copy itself is a measurement (it's the "pull" phase), so we time
     it separately from the load.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def stage_model(model_id: str, dest_root: str, revision: Optional[str] = None) -> tuple[str, float, int]:
    """Download (once) then copy the model snapshot into dest_root.

    Returns (staged_dir, copy_seconds, bytes_copied). The download is cached by
    huggingface_hub so repeated runs don't re-hit the network; the copy into
    dest_root is what puts the bytes on the tier we want to measure.
    """
    src = snapshot_download(model_id, revision=revision)
    dest = Path(dest_root) / _slug(model_id)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    shutil.copytree(src, dest)
    copy_s = time.perf_counter() - t

    nbytes = _dir_size(dest)
    return str(dest), copy_s, nbytes


def drop_page_cache() -> bool:
    """Best-effort drop of the OS page cache. Returns True if it worked.

    Without this you measure a warm read on every run after the first and the
    "cold start" number is a lie. It needs root (Colab gives you sudo; a locked
    down CI box won't). If it can't, we return False and the caller should warn
    loudly rather than silently report warm numbers as cold.

    Caveat worth knowing: FUSE-backed mounts (Google Drive) don't honor this the
    same way a block device does, so Drive numbers include whatever caching the
    FUSE layer does. That's arguably realistic — it's what their shared FS would
    do too — but don't pretend it's a clean drop.
    """
    try:
        subprocess.run(["sync"], check=True)
        # 3 = drop pagecache + dentries + inodes
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, FileNotFoundError):
        # try sudo once; some Colab kernels need it
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True, capture_output=True, timeout=30,
            )
            return True
        except Exception:
            return False


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")
