import time

import pytest

from coldstart_lab.timing import PhaseTimer, Stopwatch


def test_stopwatch_accumulates():
    sw = Stopwatch().start()
    time.sleep(0.02)
    first = sw.stop()
    assert first >= 15  # ~20ms, allow slack
    sw.start()
    time.sleep(0.02)
    total = sw.stop()
    assert total > first


def test_stopwatch_double_start_rejected():
    sw = Stopwatch().start()
    with pytest.raises(RuntimeError):
        sw.start()


def test_stopwatch_stop_without_start():
    with pytest.raises(RuntimeError):
        Stopwatch().stop()


def test_phase_timer_records_and_totals():
    t = PhaseTimer()
    with t.phase("a"):
        time.sleep(0.01)
    with t.phase("b"):
        time.sleep(0.01)
    assert set(t.phases_ms) == {"a", "b"}
    assert abs(t.total_ms - (t.phases_ms["a"] + t.phases_ms["b"])) < 1e-6
    d = t.as_dict()
    assert d["total_ms"] == pytest.approx(t.total_ms, rel=1e-3)


def test_phase_timer_rejects_overlap():
    t = PhaseTimer()
    with pytest.raises(RuntimeError):
        with t.phase("a"):
            with t.phase("b"):
                pass


def test_phase_timer_rejects_duplicate():
    t = PhaseTimer()
    with t.phase("a"):
        pass
    with pytest.raises(ValueError):
        with t.phase("a"):
            pass
