"""
test_coordinator_race.py — adversarial tests for the distributed work ledger.

These use REAL OS processes hammering the SAME database concurrently, because
the bugs we care about (lost updates, double-claim, zombie writes) do not
reproduce in a single-threaded mock.

Run:  python otfs_pipeline/tests/test_coordinator_race.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time

from coldstart_lab.distributed.coordinator import Coordinator, PENDING, RUNNING, DONE  # noqa


def _grab_everything(url, out_q):
    """A worker that claims as fast as it can and reports what it got."""
    c = Coordinator(url, lease_timeout_s=900)
    got = []
    while True:
        t = c.claim()
        if t is None:
            break
        got.append(t["task_id"])
        # simulate work; keeps several workers overlapping in the claim window
        time.sleep(0.01)
        c.complete(t["task_id"], t["epoch"], {"total_ms": 1.0})
    out_q.put(got)


def test_no_double_claim(url, n_workers=4, n_tasks=6):
    """THE property: every task is executed exactly once, by exactly one worker."""
    c = Coordinator(url)
    c.init_schema()
    c.register([f"model{i}" for i in range(n_tasks // 2)],
               ["checkpoint_format", "storage_tier"], "t4")

    q = mp.Queue()
    ps = [mp.Process(target=_grab_everything, args=(url, q)) for _ in range(n_workers)]
    [p.start() for p in ps]
    claimed = []
    for _ in ps:
        claimed += q.get()
    [p.join() for p in ps]

    dupes = {t for t in claimed if claimed.count(t) > 1}
    assert not dupes, f"DOUBLE-CLAIMED (work duplicated): {dupes}"
    assert len(claimed) == n_tasks, f"expected {n_tasks} claims, got {len(claimed)}"
    assert c.all_done(), "some task never reached DONE"
    print(f"  PASS  {n_workers} concurrent workers x {n_tasks} tasks: "
          f"{len(claimed)} claims, 0 duplicates, all DONE")


def test_zombie_write_is_fenced(url):
    """A worker whose lease expired must NOT be able to clobber the new owner."""
    slow = Coordinator(url, worker_id="slow-worker", lease_timeout_s=-1)
    slow.init_schema()
    slow.register(["zmodel"], ["checkpoint_format"], "t4")

    t = slow.claim()                       # slow worker takes the task
    assert t is not None

    # lease_timeout_s=-1 -> the next claim by anyone reaps it immediately
    # (a real deployment uses 900s; the DB clock has 1s granularity)
    fresh = Coordinator(url, worker_id="fresh-worker", lease_timeout_s=-1)
    t2 = fresh.claim()
    assert t2 is not None and t2["task_id"] == t["task_id"], "reclaim failed"
    assert t2["epoch"] > t["epoch"], "epoch must advance on re-claim (fencing)"

    # the zombie finally wakes up and tries to commit its stale result
    ok = slow.complete(t["task_id"], t["epoch"], {"total_ms": -999.0})
    assert ok is False, "ZOMBIE WRITE ACCEPTED -> results could be corrupted"

    # the legitimate owner still can
    ok2 = fresh.complete(t2["task_id"], t2["epoch"], {"total_ms": 30.0})
    assert ok2 is True, "live owner was blocked"
    assert fresh.results()["zmodel"]["checkpoint_format"]["total_ms"] == 30.0
    print("  PASS  zombie write rejected, live owner's result preserved")


def test_crash_recovery(url):
    """A dead worker's task returns to the pool; a healthy one keeps its lease."""
    dead = Coordinator(url, worker_id="dead", lease_timeout_s=1)
    dead.init_schema()
    dead.register(["cmodel"], ["checkpoint_format", "storage_tier"], "t4")
    t = dead.claim()
    assert t is not None

    other = Coordinator(url, worker_id="other", lease_timeout_s=1)
    assert other.reap_expired() == 0, "reaped a task that was still breathing"

    time.sleep(2.5)                        # "dead" stops heart-beating (DB clock ticks in whole seconds)
    assert other.reap_expired() == 1, "expired lease was not reclaimed"
    assert dead.heartbeat(t["task_id"], t["epoch"]) is False, \
        "reaped worker must learn it lost the lease"

    again = other.claim()
    assert again["task_id"] == t["task_id"], "reclaimed task not re-offered"
    print("  PASS  crash detected via lease expiry, task re-offered, "
          "old owner told to stop")


def test_timezone_mismatch(url):
    """REGRESSION: Postgres now() is tz-AWARE, heartbeat_at is tz-NAIVE.

    Subtracting them raises `TypeError: can't subtract offset-naive and
    offset-aware datetimes` and every worker dies the moment ANY task is
    running. SQLite hands back naive values for both, so this bug is invisible
    locally and only detonates on the real backend -- which is exactly what
    happened. This test pins the normalisation logic directly.
    """
    from datetime import datetime, timedelta, timezone
    from coldstart_lab.distributed.coordinator import _naive_utc, _age_s

    aware = datetime.now(timezone.utc)                    # what Postgres returns
    naive = aware.replace(tzinfo=None)                    # what the column holds

    # the original crash: this must not raise any more
    assert abs(_age_s(_naive_utc(aware), _naive_utc(naive))) < 1.0

    # a stale heartbeat is still detected across the aware/naive boundary
    old_naive = naive - timedelta(seconds=1200)
    assert _age_s(_naive_utc(aware), _naive_utc(old_naive)) > 900

    # a fresh heartbeat is never mistaken for stale
    assert _age_s(_naive_utc(aware), _naive_utc(naive)) < 900

    # tz-aware in a NON-UTC zone must normalise, not shift by hours
    ist = timezone(timedelta(hours=5, minutes=30))
    same_instant = aware.astimezone(ist)
    assert abs(_age_s(_naive_utc(aware), _naive_utc(same_instant))) < 1.0, \
        "a non-UTC session clock would fake a multi-hour lease age"

    # clock skew must not produce a negative age (an immortal lease)
    future = naive + timedelta(seconds=30)
    assert _age_s(_naive_utc(aware), _naive_utc(future)) == 0.0
    print("  PASS  aware/naive datetimes normalise; stale detected, fresh kept, "
          "non-UTC and skew handled")


def test_seed_and_idempotent_register(url):
    """Restarting a worker must not duplicate tasks or lose finished ones."""
    c = Coordinator(url)
    c.init_schema()
    c.register(["imodel"], ["checkpoint_format", "storage_tier"], "t4")
    n = c.register(["imodel"], ["checkpoint_format", "storage_tier"], "t4")       # every worker calls this
    assert n == 0, "register() is not idempotent -> duplicate tasks"

    c.seed_done("imodel__checkpoint_format__t4", {"total_ms": 29.35})
    assert c.claim()["task_id"] == "imodel__storage_tier__t4", "already-done task was re-offered"
    print("  PASS  register() idempotent; pre-seeded checkpoint never re-run")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    tmp = tempfile.mkdtemp()
    print("Coordinator concurrency suite (SQLite backend; Postgres in prod)")
    for name, fn in [("no_double_claim", test_no_double_claim),
                     ("zombie_write_is_fenced", test_zombie_write_is_fenced),
                     ("crash_recovery", test_crash_recovery),
                     ("timezone_mismatch", test_timezone_mismatch),
                     ("seed_and_idempotent_register",
                      test_seed_and_idempotent_register)]:
        url = f"sqlite:///{os.path.join(tmp, name)}.db"
        print(f"\n[{name}]")
        fn(url)
    print("\nALL COORDINATOR TESTS PASSED")
