"""
coordinator.py — shared work ledger for N independent Colab sessions.

Ported from a distributed OTFS/ISAC optimisation pipeline where the same
primitives were used to fan a parameter sweep across a Colab fleet. The
concurrency semantics are domain-agnostic; only the task identity changed.

DESIGN IN ONE PARAGRAPH
-----------------------
The unit of distributed work is ONE (model, experiment, device_class) triple —
e.g. `qwen2.5-3b__checkpoint_format__cuda`. A worker CLAIMS a task with a single
conditional UPDATE, runs that experiment exactly as the single-machine harness
would, and COMMITS the resulting report rows. No two workers can hold the same
task, because the claim is one atomic statement and we check the affected row
count.

WHY NOT A LOCK FILE ON GOOGLE DRIVE
-----------------------------------
Drive's FUSE mount does not implement POSIX advisory locking (flock/fcntl) and
its rename is not atomic. Both SQLite's locking protocol and the classic
"write .tmp then os.replace" trick silently degrade there -> torn reads, "database
is locked", and corrupted .db files. This module therefore talks to a real RDB
(Postgres) and never to a file on a synced drive.

WHY A CONDITIONAL UPDATE INSTEAD OF `SELECT ... FOR UPDATE SKIP LOCKED`
----------------------------------------------------------------------
SKIP LOCKED is the textbook queue primitive but is Postgres-only. The
conditional UPDATE below is atomic on *both* Postgres and SQLite, which lets the
identical code path be unit-tested locally (see tests/test_coordinator_race.py)
and then pointed at Postgres in production by changing one URL.

FENCING
-------
A crashed worker's lease expires and the task is reclaimed. But a worker that is
merely *slow* (or paused by Colab, or throttled mid-download) can wake up after
its lease was reclaimed and try to write a result — the classic zombie-writer
race. Every claim therefore bumps `epoch`, the worker carries that epoch as a
fencing token, and `complete()` only writes `WHERE epoch = :my_epoch`. A zombie's
write affects 0 rows and is rejected loudly instead of overwriting the live
worker's result.

WHY THIS MATTERS FOR A COLD-START BENCHMARK SPECIFICALLY
--------------------------------------------------------
Benchmark tasks are long (a 7B checkpoint pull alone can exceed 10 minutes) and
free-tier Colab sessions are pre-empted without warning. Lease reclamation is
therefore the normal case, not the exception, and a duplicated task would not
merely waste time — it would contend for the same disk and pollute the very
I/O measurement we are taking. Exactly-once execution is a correctness
requirement here, not an efficiency nicety.
"""
from __future__ import annotations

import json
import os
import random
import socket
import time
import uuid
from typing import Optional

from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table,
                        Text, create_engine, func, select, update)

# Task lifecycle: pending -> running -> done
#                            \-> pending (lease expired / explicit release)
PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"

# Terminal states: no worker will ever run these again. A fleet is finished when
# every task is terminal -- NOT when the pending queue is merely empty, since a
# task held by a live worker may still come back if that worker dies.
TERMINAL = (DONE, FAILED)

_META = MetaData()

TASKS = Table(
    "coldstart_tasks", _META,
    # "{model}__{experiment}__{device}"
    Column("task_id", String(160), primary_key=True),
    Column("model_key", String(64), nullable=False),
    Column("experiment", String(48), nullable=False),
    Column("device_class", String(16), nullable=False),
    Column("status", String(16), nullable=False, index=True),
    Column("owner", String(128)),          # worker id holding the lease
    Column("epoch", Integer, nullable=False, default=0),   # fencing token
    Column("heartbeat_at", DateTime),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("attempts", Integer, nullable=False, default=0),
    Column("cost_hint", Integer, nullable=False, default=0),  # ~GiB, LPT ordering
    Column("result_json", Text),           # the ExperimentResult payload
    Column("gpu_name", String(96)),        # what hardware produced the number
    Column("error", Text),
)


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise any datetime to naive-UTC so two of them can be subtracted.

    Belt and braces on top of the cast in reap_expired(): different drivers and
    server versions disagree about whether they hand back tz-aware values, and
    getting this wrong either reaps live workers or never reaps dead ones.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _age_s(now: datetime, hb: datetime) -> float:
    """Seconds since a heartbeat. Never negative: a small clock disagreement
    must not make a lease look infinitely fresh."""
    return max(0.0, (now - hb).total_seconds())


class Coordinator:
    def __init__(self, url: str, worker_id: Optional[str] = None,
                 lease_timeout_s: int = 900, logger=None):
        # pool_pre_ping: Colab sessions idle for minutes between trials and the
        # DB (or the NAT in front of it) will drop the TCP connection; without
        # this the next commit dies on a stale socket.
        #
        # timezone=utc: heartbeat_at is a naive TIMESTAMP, and Postgres converts
        # now() (timestamptz) into it using the SESSION's timezone. Pinning the
        # session to UTC means every worker writes the same wall clock no matter
        # what the VM or the connection defaults to -- otherwise two workers in
        # different timezones would compute lease ages hours apart.
        kw = {}
        if url.startswith("postgresql"):
            kw["connect_args"] = {"options": "-c timezone=utc"}
        self.engine = create_engine(url, pool_pre_ping=True, future=True, **kw)
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lease_timeout_s = lease_timeout_s
        self.log = logger

    # ------------------------------------------------------------------ setup
    def init_schema(self, attempts: int = 6) -> None:
        """Create the ledger table. Idempotent and safe to call concurrently.

        `create_all` already issues CREATE TABLE IF NOT EXISTS, but that is not
        enough when N fresh workers start against an EMPTY database at the same
        moment: Postgres checks existence and creates in separate steps, so two
        sessions can both pass the check and one then dies with "duplicate key
        value violates unique constraint pg_type_typname_nsp_index" or a
        deadlock. The operation is idempotent, so a bounded retry with jittered
        backoff is a complete fix -- whoever loses simply re-checks and finds
        the table already there.
        """
        last: Exception | None = None
        for i in range(attempts):
            try:
                _META.create_all(self.engine, checkfirst=True)
                if self.log and i:
                    self.log.info("[coord] schema ready after %d retry(ies)", i)
                return
            except Exception as e:  # noqa: BLE001 - see docstring
                last = e
                if self.schema_exists():
                    return          # someone else won the race; that is success
                time.sleep(min(2.0, 0.05 * (2 ** i)) * (0.5 + random.random()))
        raise RuntimeError(f"could not create the task ledger: {last!r}")

    def schema_exists(self) -> bool:
        """True if the task table is present. Used to give a human-readable
        error instead of letting a raw UndefinedTable traceback escape."""
        try:
            from sqlalchemy import inspect

            return inspect(self.engine).has_table(TASKS.name)
        except Exception:  # noqa: BLE001 - connectivity problems surface later
            return False

    def task_count(self) -> int:
        with self.engine.begin() as cx:
            return cx.execute(select(func.count()).select_from(TASKS)).scalar_one()

    def register(self, model_keys, experiments, device_class,
                 cost_hints=None) -> int:
        """Idempotently insert the task list. Safe to call from every worker.

        Device class is part of the identity, not just metadata: a cold-start
        number measured on a T4 is not interchangeable with one from an A100,
        so the same (model, experiment) on different hardware is a *different*
        task rather than a duplicate to be skipped.
        """
        cost_hints = cost_hints or {}
        added = 0
        with self.engine.begin() as cx:
            existing = {r[0] for r in cx.execute(select(TASKS.c.task_id))}
            rows = []
            for m in model_keys:
                for e in experiments:
                    tid = f"{m}__{e}__{device_class}"
                    if tid in existing:
                        continue
                    rows.append(dict(task_id=tid, model_key=m, experiment=e,
                                     device_class=device_class,
                                     status=PENDING, epoch=0, attempts=0,
                                     cost_hint=int(cost_hints.get(m, 0))))
            if rows:
                cx.execute(TASKS.insert(), rows)
                added = len(rows)
        return added

    def seed_done(self, task_id: str, result: dict) -> None:
        """Import an already-finished local result so nobody redoes it."""
        with self.engine.begin() as cx:
            cx.execute(update(TASKS).where(TASKS.c.task_id == task_id).values(
                status=DONE, result_json=json.dumps(result),
                finished_at=func.now(), owner=self.worker_id))

    # ------------------------------------------------------------------ claim
    def claim(self) -> Optional[dict]:
        """Atomically take one task. Returns dict(task_id, model_key,
        experiment, device_class, epoch) or None when nothing is left."""
        self.reap_expired()
        for tid, model_key, experiment, device_class in self._candidates():
            with self.engine.begin() as cx:
                # THE atomic step. Two workers racing on the same row: the DB
                # serialises the writes, the loser sees rowcount == 0 because
                # status is no longer 'pending', and moves to the next task.
                res = cx.execute(
                    update(TASKS)
                    .where(TASKS.c.task_id == tid, TASKS.c.status == PENDING)
                    .values(status=RUNNING, owner=self.worker_id,
                            epoch=TASKS.c.epoch + 1, attempts=TASKS.c.attempts + 1,
                            started_at=func.now(), heartbeat_at=func.now()))
                if res.rowcount != 1:
                    continue                      # lost the race; try next
                epoch = cx.execute(select(TASKS.c.epoch).where(
                    TASKS.c.task_id == tid)).scalar_one()
            if self.log:
                self.log.info("[coord] CLAIMED %s (epoch=%d, worker=%s)",
                              tid, epoch, self.worker_id)
            return {"task_id": tid, "model_key": model_key,
                    "experiment": experiment, "device_class": device_class,
                    "epoch": epoch}
        return None

    def _candidates(self):
        """Pending tasks, biggest checkpoint first.

        Longest-processing-time-first keeps the slowest job off the critical
        path at the end of the run: a 7B pull started last would leave the whole
        fleet idle waiting on one worker.
        """
        with self.engine.begin() as cx:
            return list(cx.execute(
                select(TASKS.c.task_id, TASKS.c.model_key,
                       TASKS.c.experiment, TASKS.c.device_class)
                .where(TASKS.c.status == PENDING)
                .order_by(TASKS.c.cost_hint.desc(), TASKS.c.task_id)).all())

    # -------------------------------------------------------------- heartbeat
    def heartbeat(self, task_id: str, epoch: int) -> bool:
        """Refresh the lease. False means we were reaped -> stop working."""
        with self.engine.begin() as cx:
            r = cx.execute(update(TASKS).where(
                TASKS.c.task_id == task_id, TASKS.c.epoch == epoch,
                TASKS.c.owner == self.worker_id, TASKS.c.status == RUNNING
            ).values(heartbeat_at=func.now()))
        return r.rowcount == 1

    def reap_expired(self) -> int:
        """Return leases whose owner stopped breathing to the pending pool.

        A reclaimed cold-start task restarts from scratch. That is acceptable
        and in fact correct here: a half-finished benchmark has no partial
        result worth keeping, and the HF cache means the re-run usually skips
        the download.
        """
        # Postgres now() is timestamptz (offset-AWARE) while heartbeat_at is
        # TIMESTAMP (offset-NAIVE); subtracting them raises TypeError. SQLite
        # returns naive for both, which is why this only ever broke on Postgres.
        # _naive_utc() collapses the difference, and the session is pinned to UTC
        # so the naive values stored by every worker mean the same thing.
        # (A CAST here would be wrong: SQLite gives "DATETIME" NUMERIC affinity
        # and would hand back the integer 2026.)
        with self.engine.begin() as cx:
            stale = cx.execute(select(TASKS.c.task_id, TASKS.c.heartbeat_at)
                               .where(TASKS.c.status == RUNNING)).all()
            now = _naive_utc(cx.execute(select(func.now())).scalar_one())
            dead = []
            for tid, hb in stale:
                hb = _naive_utc(hb)
                if hb is None:
                    continue
                if _age_s(now, hb) > self.lease_timeout_s:
                    dead.append(tid)
            for tid in dead:
                # epoch is NOT bumped here; the next claim bumps it, which is
                # what invalidates the zombie's fencing token.
                cx.execute(update(TASKS).where(
                    TASKS.c.task_id == tid, TASKS.c.status == RUNNING
                ).values(status=PENDING, owner=None))
                if self.log:
                    self.log.warning("[coord] RECLAIMED %s (lease expired, "
                                     "will resume from its Optuna trials)", tid)
        return len(dead)

    # ------------------------------------------------------------- completion
    def complete(self, task_id: str, epoch: int, result: dict,
                 gpu_name: Optional[str] = None) -> bool:
        """Commit a result. Fenced: only the current lease holder may write."""
        with self.engine.begin() as cx:
            r = cx.execute(update(TASKS).where(
                TASKS.c.task_id == task_id, TASKS.c.epoch == epoch,
                TASKS.c.status == RUNNING
            ).values(status=DONE, result_json=json.dumps(result, default=str),
                     gpu_name=gpu_name, finished_at=func.now(), error=None))
        if r.rowcount != 1 and self.log:
            self.log.error("[coord] REJECTED stale write for %s (epoch=%d). "
                           "Our lease was reclaimed; another worker owns it. "
                           "Discarding our result.", task_id, epoch)
        return r.rowcount == 1

    def fail(self, task_id: str, epoch: int, err: str,
             max_attempts: int = 3, permanent: bool = False) -> None:
        """Record an error.

        Retryable errors go back to PENDING until `max_attempts`, then park as
        FAILED so one poison task cannot spin the whole fleet forever.

        `permanent=True` skips retries entirely. Some failures are not transient
        -- a 401 on a gated repo will fail identically on every attempt, and
        because tasks are ordered longest-checkpoint-first a big gated model
        gets re-claimed ahead of real work every time. Burning three attempts on
        a certainty is worse than useless: it starves the queue.
        """
        with self.engine.begin() as cx:
            att = cx.execute(select(TASKS.c.attempts).where(
                TASKS.c.task_id == task_id)).scalar_one()
            new_status = FAILED if (permanent or att >= max_attempts) else PENDING
            cx.execute(update(TASKS).where(
                TASKS.c.task_id == task_id, TASKS.c.epoch == epoch
            ).values(status=new_status, owner=None, error=err[:4000]))
        if self.log and new_status == FAILED:
            self.log.error("[coord] %s parked as FAILED after %d attempt(s)%s",
                           task_id, att, " (non-retryable)" if permanent else "")

    # ------------------------------------------------------------------ query
    def status(self) -> list[dict]:
        with self.engine.begin() as cx:
            rows = cx.execute(select(
                TASKS.c.task_id, TASKS.c.status, TASKS.c.owner,
                TASKS.c.attempts, TASKS.c.heartbeat_at)).all()
        return [dict(task_id=r[0], status=r[1], owner=r[2], attempts=r[3],
                     heartbeat_at=r[4]) for r in rows]

    def dump_all(self) -> dict:
        """Every row in the ledger, for archival.

        `results()` returns only successful payloads, which is what the analysis
        needs. A published artifact should carry more than that: which tasks
        failed and why, how many attempts each took, which worker and which GPU
        produced each number, and when. That provenance is what lets someone
        else audit the run rather than take it on trust.
        """
        with self.engine.begin() as cx:
            rows = cx.execute(select(
                TASKS.c.task_id, TASKS.c.model_key, TASKS.c.experiment,
                TASKS.c.device_class, TASKS.c.status, TASKS.c.owner,
                TASKS.c.epoch, TASKS.c.attempts, TASKS.c.gpu_name,
                TASKS.c.error, TASKS.c.started_at, TASKS.c.finished_at,
                TASKS.c.result_json)).all()

        tasks = []
        for (tid, mk, exp, dc, status, owner, epoch, attempts, gpu, err,
             started, finished) in [r[:12] for r in rows]:
            tasks.append({
                "task_id": tid, "model_key": mk, "experiment": exp,
                "device_class": dc, "status": status, "owner": owner,
                "epoch": epoch, "attempts": attempts, "gpu_name": gpu,
                "error": (err or "").strip().splitlines()[-1][:300] if err else None,
                "started_at": str(started) if started else None,
                "finished_at": str(finished) if finished else None,
            })
        return {"tasks": sorted(tasks, key=lambda t: t["task_id"])}

    def failures(self) -> list[dict]:
        """Failed tasks with a one-line error, for triage."""
        with self.engine.begin() as cx:
            rows = cx.execute(select(TASKS.c.task_id, TASKS.c.error)
                              .where(TASKS.c.status == FAILED)).all()
        out = []
        for tid, err in rows:
            first = ""
            if err:
                # Tracebacks are stored whole; the last line is the useful one.
                lines = [ln for ln in err.strip().splitlines() if ln.strip()]
                first = lines[-1][:200] if lines else ""
            out.append({"task_id": tid, "error": first})
        return out

    def retry_failed(self) -> int:
        """Return FAILED tasks to the pool (e.g. after adding an HF token)."""
        with self.engine.begin() as cx:
            r = cx.execute(update(TASKS).where(TASKS.c.status == FAILED)
                           .values(status=PENDING, owner=None, attempts=0,
                                   error=None))
        return r.rowcount

    def all_done(self) -> bool:
        """True when no task can ever run again (every task is terminal).

        Deliberately NOT "the pending queue is empty": a task held by a live
        worker is still unfinished, and if that worker is pre-empted the task
        returns to the pool. A worker that exits on an empty pending queue is
        the bug that made idle sessions report 'finished' while one worker was
        still grinding through a 7 GiB checkpoint.
        """
        with self.engine.begin() as cx:
            n = cx.execute(select(func.count()).select_from(TASKS)
                           .where(TASKS.c.status.notin_(TERMINAL))).scalar_one()
        return n == 0

    def unfinished_counts(self) -> dict:
        """How much work is left, split by whether it is claimable right now."""
        with self.engine.begin() as cx:
            rows = cx.execute(
                select(TASKS.c.status, func.count())
                .where(TASKS.c.status.notin_(TERMINAL))
                .group_by(TASKS.c.status)).all()
        counts = {status: n for status, n in rows}
        return {"pending": counts.get(PENDING, 0),
                "running": counts.get(RUNNING, 0)}

    def results(self) -> dict:
        """{model_key: {experiment: record}} — the merge input."""
        out: dict = {}
        with self.engine.begin() as cx:
            rows = cx.execute(select(TASKS.c.model_key, TASKS.c.experiment,
                                     TASKS.c.device_class, TASKS.c.gpu_name,
                                     TASKS.c.result_json)
                              .where(TASKS.c.status == DONE)).all()
        for model_key, experiment, device_class, gpu, js in rows:
            if not js:
                continue
            rec = json.loads(js)
            rec["_device_class"] = device_class
            rec["_gpu_name"] = gpu
            out.setdefault(model_key, {})[experiment] = rec
        return out

    def progress(self) -> dict:
        """Counts by status — what the dashboard cell polls."""
        with self.engine.begin() as cx:
            rows = cx.execute(select(TASKS.c.status, func.count())
                              .group_by(TASKS.c.status)).all()
        return {status: n for status, n in rows}
