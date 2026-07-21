"""Drives repeated, isolated cold-load measurements.

Everything funnels through run_config: it writes the worker config to a temp
file, spawns `python -m coldstart.worker` in a clean process, optionally drops
the page cache first, parses the one JSON line the worker prints, and repeats.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from .timing import PhaseRecord
from . import storage


class WorkerError(RuntimeError):
    pass


def run_config(
    cfg: dict,
    repeats: int = 3,
    drop_cache: bool = True,
    warmup: bool = True,
    timeout: int = 600,
) -> List[PhaseRecord]:
    """Run one worker config `repeats` times, return a record per successful run.

    warmup: the very first cold start on a box also pays for things we don't want
    to attribute to the model — first-ever CUDA context on the GPU, kernel autotune
    caches, HF hub metadata fetches. One discarded warmup run absorbs most of that
    so the measured runs compare like with like. Off by default in CI (tiny models,
    not worth the time), on for real sweeps.
    """
    records: List[PhaseRecord] = []
    n = repeats + (1 if warmup else 0)

    cache_warned = False
    for i in range(n):
        if drop_cache:
            ok = storage.drop_page_cache()
            if not ok and not cache_warned:
                sys.stderr.write(
                    "WARN: could not drop page cache (need root). "
                    "Post-first-run numbers are warm, not cold.\n"
                )
                cache_warned = True

        result = _spawn_worker(cfg, timeout=timeout)
        if "error" in result:
            # a failed run is data too, but we don't average it in. surface it.
            sys.stderr.write(f"run {i}: worker error: {result['error']}\n")
            if result.get("trace"):
                sys.stderr.write(result["trace"] + "\n")
            continue

        if warmup and i == 0:
            continue  # discard

        rec = PhaseRecord(phases=result["phases"], meta=result.get("meta", {}))
        records.append(rec)

    if not records:
        raise WorkerError(f"no successful runs for config: {cfg.get('model_id')}")
    return records


def _spawn_worker(cfg: dict, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "coldstart.worker", cfg_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"worker timed out after {timeout}s", "meta": cfg}
    finally:
        Path(cfg_path).unlink(missing_ok=True)

    # worker prints exactly one json line on stdout; libraries spam stderr.
    line = _last_json_line(proc.stdout)
    if line is None:
        return {
            "error": f"worker produced no json (exit {proc.returncode})",
            "stderr_tail": proc.stderr[-2000:],
        }
    return line


def _last_json_line(text: str) -> Optional[dict]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
