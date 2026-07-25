"""Regression tests for fleet liveness and error classification.

Both cover bugs observed in a live 3-session Colab run:

  * Idle workers reported "finished" (completed=0, failed=0) and exited while
    another worker was still 13 minutes into a 7.6 GiB checkpoint. The pending
    queue was empty, but the run was nowhere near over.
  * A gated repo returned 401 on every attempt. Because tasks are ordered
    longest-checkpoint-first, the biggest gated model was re-claimed ahead of
    real work each time, producing 22 failures in one session.
"""

import pytest

from coldstart_lab.distributed.coordinator import (Coordinator, DONE, FAILED,
                                                   PENDING, RUNNING)
from coldstart_lab.distributed.worker import is_non_retryable


# --------------------------------------------------------------- liveness
def test_all_done_false_while_a_task_is_running(url):
    """THE bug: an empty pending queue is not a finished fleet."""
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")

    t = c.claim()
    assert t is not None
    # Nothing is claimable now...
    assert c.claim() is None
    # ...but the fleet is emphatically NOT done.
    assert c.all_done() is False
    assert c.unfinished_counts() == {"pending": 0, "running": 1}

    c.complete(t["task_id"], t["epoch"], {"total_ms": 1.0})
    assert c.all_done() is True


def test_all_done_counts_failed_as_terminal(url):
    """A permanently failed task must not keep the fleet waiting forever."""
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")

    t = c.claim()
    c.fail(t["task_id"], t["epoch"], "GatedRepoError: 401", permanent=True)
    assert c.all_done() is True, "FAILED must be terminal or workers spin forever"


def test_worker_waits_then_picks_up_released_task(url):
    """A waiting worker must claim a task released by a dead peer."""
    holder = Coordinator(url, worker_id="holder", lease_timeout_s=-1)
    holder.init_schema()
    holder.register(["m1"], ["checkpoint_format"], "t4")
    t = holder.claim()
    assert t is not None

    waiter = Coordinator(url, worker_id="waiter", lease_timeout_s=-1)
    # Nothing pending, but work remains -> a correct worker keeps waiting.
    assert waiter.all_done() is False
    # The holder's lease expires; the waiter's next claim reaps and takes it.
    got = waiter.claim()
    assert got is not None and got["task_id"] == t["task_id"]
    assert got["epoch"] > t["epoch"], "fencing token must advance"


# --------------------------------------------------- failure classification
def test_permanent_failure_is_not_retried(url):
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")

    t = c.claim()
    c.fail(t["task_id"], t["epoch"], "401 Client Error", permanent=True)
    # Must NOT come back around for another doomed attempt.
    assert c.claim() is None
    assert c.progress().get(FAILED) == 1


def test_retryable_failure_returns_to_pool(url):
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")

    t = c.claim()
    c.fail(t["task_id"], t["epoch"], "transient network blip", permanent=False)
    again = c.claim()
    assert again is not None, "a retryable failure should be re-offered"


def test_retryable_failure_parks_after_max_attempts(url):
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")

    for _ in range(5):
        t = c.claim()
        if t is None:
            break
        c.fail(t["task_id"], t["epoch"], "flaky", max_attempts=3)
    assert c.claim() is None, "poison task must stop being re-offered"
    assert c.all_done() is True


def test_retry_failed_returns_tasks_to_queue(url):
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")
    t = c.claim()
    c.fail(t["task_id"], t["epoch"], "401", permanent=True)
    assert c.retry_failed() == 1
    assert c.claim() is not None


def test_failures_reports_last_traceback_line(url):
    c = Coordinator(url)
    c.init_schema()
    c.register(["m1"], ["checkpoint_format"], "t4")
    t = c.claim()
    c.fail(t["task_id"], t["epoch"],
           "Traceback...\n  File x\nGatedRepoError: 401 Client Error",
           permanent=True)
    rows = c.failures()
    assert len(rows) == 1
    assert "GatedRepoError" in rows[0]["error"]


@pytest.mark.parametrize("exc", [
    Exception("401 Client Error. Cannot access gated repo"),
    Exception("403 Client Error"),
    Exception("404 Client Error"),
    KeyError("unknown-model"),
    FileNotFoundError("no safetensors"),
])
def test_non_retryable_detected(exc):
    assert is_non_retryable(exc) is True


@pytest.mark.parametrize("exc", [
    Exception("Connection reset by peer"),
    TimeoutError("read timed out"),
    RuntimeError("CUDA out of memory"),
])
def test_retryable_not_flagged(exc):
    assert is_non_retryable(exc) is False


def test_gated_repo_error_by_type_name():
    """huggingface_hub's error type is matched by name, without importing it."""

    class GatedRepoError(Exception):
        pass

    assert is_non_retryable(GatedRepoError("no message match here")) is True
