"""The distributed worker loop.

One worker == one Colab session. It claims a task, runs that experiment with
the same code path the single-machine CLI uses, commits the result, and repeats
until the queue drains. Nothing about the measurement changes when it runs
distributed -- only who decides which task to run next.

The heartbeat runs on a daemon thread because the work itself is a long
blocking call (a checkpoint pull, a vLLM engine init) with no convenient place
to poll from. If the main thread dies, the daemon dies with it and the lease
expires naturally, which is exactly the behaviour we want.
"""

from __future__ import annotations

import random
import shutil
import threading
import time
import traceback
from typing import Optional

from coldstart_lab import environment
from coldstart_lab.distributed import config as DC
from coldstart_lab.distributed.coordinator import Coordinator
from coldstart_lab.experiments import (
    EngineInitExperiment,
    FormatExperiment,
    StorageExperiment,
    StorageTier,
)
from coldstart_lab.fetch import fetch


def is_non_retryable(exc: BaseException) -> bool:
    """True for failures that will recur identically on every attempt.

    Retrying these is not merely wasteful: tasks are claimed
    longest-checkpoint-first, so a big model that always 401s is re-claimed
    ahead of real work and starves the queue. We classify by exception type
    where the hub library gives us one, and fall back to matching the HTTP
    status in the message for wrapped errors.
    """
    name = type(exc).__name__
    if name in {"GatedRepoError", "RepositoryNotFoundError", "EntryNotFoundError",
                "RevisionNotFoundError", "LocalEntryNotFoundError"}:
        return True
    if isinstance(exc, (KeyError, ValueError, FileNotFoundError)):
        # Unknown model key, missing search space, absent checkpoint file:
        # all deterministic given the same task definition.
        return True
    msg = str(exc)
    return any(code in msg for code in ("401 Client Error", "403 Client Error",
                                        "404 Client Error"))


class _Heartbeat:
    """Refreshes a lease on a background thread until stopped."""

    def __init__(self, coord: Coordinator, task_id: str, epoch: int,
                 interval_s: int, logger=None) -> None:
        self.coord = coord
        self.task_id = task_id
        self.epoch = epoch
        self.interval_s = interval_s
        self.log = logger
        self._stop = threading.Event()
        self._reaped = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def reaped(self) -> bool:
        """True if the coordinator took our lease away while we worked."""
        return self._reaped.is_set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                if not self.coord.heartbeat(self.task_id, self.epoch):
                    self._reaped.set()
                    if self.log:
                        self.log.warning(
                            "[worker] lease LOST for %s; our result will be "
                            "rejected on commit", self.task_id)
                    return
            except Exception:  # noqa: BLE001 - a transient DB blip must not kill the run
                if self.log:
                    self.log.warning("[worker] heartbeat failed (transient?)")


def run_one(task: dict, device: str, hf_token: str | None = None,
            repeats: int = 5, warmup: int = 1, out_root: str = "/content/stage",
            logger=None) -> dict:
    """Execute one claimed task and return its serialisable result."""

    model_key = task["model_key"]
    experiment = task["experiment"]

    fetched = fetch(model_key, token=hf_token)
    model_dir = fetched.local_dir

    if experiment == "checkpoint_format":
        # Deriving a .bin doubles disk for large checkpoints; skip it there.
        from coldstart_lab.models import get_model
        include_bin = get_model(model_key).params_b <= 3.5
        exp = FormatExperiment(model_dir, device=device, include_bin=include_bin,
                               repeats=repeats, warmup=warmup)
    elif experiment == "storage_tier":
        tiers = [
            StorageTier(name="local-nvme", root=out_root),
            StorageTier(name="remote-emulated-200MiBs", root=out_root,
                        emulated_mib_s=200.0),
        ]
        exp = StorageExperiment(model_dir, tiers=tiers, device=device,
                                repeats=repeats, warmup=warmup)
    elif experiment == "engine_init":
        exp = EngineInitExperiment(model_dir, device=device,
                                   repeats=max(1, repeats - 2), warmup=0)
    else:
        raise ValueError(f"Unknown experiment {experiment!r}")

    result = exp.run().to_dict()
    result["pull_ms"] = round(fetched.pull_ms, 2)
    result["model_key"] = model_key
    return result


def _clear_staging(out_root: str, logger=None) -> None:
    """Delete staged checkpoint copies between tasks.

    The storage experiment copies the whole checkpoint once per tier. On a
    T4 Colab box (~78 GiB disk) two staged copies of a 7.6 GiB model plus the HF
    cache is already a third of the disk, and the next task would fail on ENOSPC
    with an error that looks nothing like "you ran out of room".
    """
    import glob
    import os

    for path in glob.glob(os.path.join(out_root, "coldstart_stage_*")):
        try:
            shutil.rmtree(path)
        except OSError:
            if logger:
                logger.warning("[worker] could not remove staging dir %s", path)


def work_loop(url: str, device: str, device_class: str,
              hf_token: str | None = None, repeats: int = 5, warmup: int = 1,
              max_tasks: int | None = None, out_root: str = "/content/stage",
              wait: bool = True, poll_interval_s: int = 30,
              logger=None) -> dict:
    """Claim-run-commit until every task is terminal. Returns a summary.

    `wait=True` (the default) is what makes a fleet behave sensibly. An empty
    *pending* queue does not mean the run is over: another worker may be holding
    the last task, and if that session is pre-empted the task comes back. A
    worker that exited on an empty pending queue would report "finished" while
    the fleet still had hours of work outstanding -- and worse, would not be
    there to pick the task up when it was released.

    So: exit only when `all_done()` (every task DONE or FAILED). Otherwise sleep
    and re-poll, with jitter so N sessions that started together do not stampede
    the same row on every tick.
    """

    coord = Coordinator(url, lease_timeout_s=DC.LEASE_TIMEOUT_S, logger=logger)
    fp = environment.probe()
    gpu_name = fp.gpus[0].name if fp.gpus else "cpu"

    done, failed, rejected, waited_s = 0, 0, 0, 0.0
    while max_tasks is None or (done + failed) < max_tasks:
        task = coord.claim()

        if task is None:
            if coord.all_done():
                if logger:
                    logger.info("[worker] all tasks terminal; fleet is finished")
                break
            if not wait:
                if logger:
                    left = coord.unfinished_counts()
                    logger.info("[worker] nothing claimable now (%s); exiting "
                                "because wait=False", left)
                break
            left = coord.unfinished_counts()
            # Jitter avoids N workers waking in lockstep and hammering the DB.
            nap = poll_interval_s * (0.5 + random.random())
            if logger:
                logger.info("[worker] no claimable task (pending=%d running=%d); "
                            "waiting %.0fs in case a lease is released",
                            left["pending"], left["running"], nap)
            time.sleep(nap)
            waited_s += nap
            continue

        tid, epoch = task["task_id"], task["epoch"]
        if logger:
            logger.info("[worker] running %s", tid)

        try:
            with _Heartbeat(coord, tid, epoch, DC.HEARTBEAT_INTERVAL_S, logger) as hb:
                result = run_one(task, device=device, hf_token=hf_token,
                                 repeats=repeats, warmup=warmup,
                                 out_root=out_root, logger=logger)
                if hb.reaped:
                    # Someone else owns this task now. Committing would be the
                    # zombie write the fencing token exists to stop -- so don't
                    # even try, and let the new owner finish it.
                    rejected += 1
                    continue

            if coord.complete(tid, epoch, result, gpu_name=gpu_name):
                done += 1
                if logger:
                    logger.info("[worker] committed %s", tid)
            else:
                rejected += 1
        except Exception as e:  # noqa: BLE001 - one bad task must not stop the fleet
            failed += 1
            permanent = is_non_retryable(e)
            coord.fail(tid, epoch, traceback.format_exc(), permanent=permanent)
            if logger:
                logger.error("[worker] FAILED %s (%s): %r", tid,
                             "permanent" if permanent else "will retry", e)
        finally:
            _clear_staging(out_root, logger)

    return {"worker_id": coord.worker_id, "gpu": gpu_name,
            "device_class": device_class, "completed": done,
            "failed": failed, "rejected_stale": rejected,
            "waited_s": round(waited_s, 1),
            "fleet_finished": coord.all_done()}
