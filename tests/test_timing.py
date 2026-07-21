import time

from coldstart.timing import PhaseTimer, PhaseRecord, summarize, _percentile


def test_phase_accumulates_repeats():
    # a phase name used twice in a loop should sum, not overwrite — this is how
    # per-layer loads would show up if we ever time inside the load loop.
    t = PhaseTimer()
    for _ in range(3):
        with t.phase("chunk"):
            time.sleep(0.01)
    rec = t.record()
    assert rec.phases["chunk"] > 0.025


def test_total_is_sum_of_phases():
    r = PhaseRecord(phases={"a": 1.0, "b": 2.5})
    assert r.total == 3.5


def test_summarize_reports_tail():
    # p95 should sit at/above p50; that's the whole reason we quote it for SLAs.
    recs = [PhaseRecord(phases={"load": x}) for x in (1.0, 1.0, 1.0, 5.0)]
    s = summarize(recs)
    assert s["n"] == 4
    assert s["total"]["p95"] >= s["total"]["p50"]
    assert s["total"]["max"] == 5.0


def test_summarize_missing_phase_treated_as_zero():
    # not every run has every phase (e.g. to_device only on gpu). missing => 0,
    # so the arm still summarizes instead of KeyError-ing.
    recs = [PhaseRecord(phases={"load": 1.0, "to_device": 0.5}),
            PhaseRecord(phases={"load": 1.2})]
    s = summarize(recs)
    assert "to_device" in s["phases"]
    assert s["phases"]["to_device"]["p50"] >= 0


def test_percentile_edges():
    assert _percentile([], 50) == 0.0
    assert _percentile([4.2], 95) == 4.2
    # interpolated median of an even-length list
    assert _percentile([0.0, 10.0], 50) == 5.0
