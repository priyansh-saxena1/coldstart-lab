import math

from coldstart.timing import PhaseRecord
from coldstart import report


def test_fit_recovers_known_line():
    # load_s = 2*GB + 1, exactly. fit should recover slope 2, intercept 1, r2 1.
    gb = [1.0, 2.0, 4.0, 8.0]
    load = [2 * g + 1 for g in gb]
    curve = report.fit_load_curve(gb, load)
    assert curve is not None
    assert math.isclose(curve.slope_s_per_gb, 2.0, abs_tol=1e-6)
    assert math.isclose(curve.intercept_s, 1.0, abs_tol=1e-6)
    assert math.isclose(curve.r2, 1.0, abs_tol=1e-9)
    # slope 2 s/GB => 500 MB/s effective
    assert math.isclose(curve.eff_bandwidth_MBps, 500.0, rel_tol=1e-6)


def test_fit_needs_three_points():
    assert report.fit_load_curve([1.0, 2.0], [1.0, 2.0]) is None


def test_extrapolate_uses_the_line():
    curve = report.fit_load_curve([1.0, 2.0, 3.0], [3.0, 5.0, 7.0])  # 2*GB+1
    out = report.extrapolate(curve, {"27B": 54.0})
    assert out["27B"]["predicted_load_s"] == round(2 * 54.0 + 1, 1)


def test_curve_from_sweep_uses_load_phase_not_total():
    # total carries size-independent overhead (import etc). the fit must key off
    # the 'load' phase or the slope is garbage. build a sweep where total is
    # dominated by a constant import but load scales cleanly, and check the
    # recovered bandwidth reflects load, not total.
    class Spec:
        def __init__(self, gb): self.approx_gb = gb

    sweep = {
        "m1": [PhaseRecord(phases={"import": 8.0, "load": 1.0})],
        "m2": [PhaseRecord(phases={"import": 8.0, "load": 2.0})],
        "m3": [PhaseRecord(phases={"import": 8.0, "load": 4.0})],
    }
    specs = {"m1": Spec(1.0), "m2": Spec(2.0), "m3": Spec(4.0)}
    curve = report.curve_from_sweep(sweep, specs)
    assert curve is not None
    assert math.isclose(curve.slope_s_per_gb, 1.0, abs_tol=1e-6)  # 1 s/GB from load


def test_diff_total_speedup():
    results = {
        "baseline": [PhaseRecord(phases={"load": 4.0})],
        "fast": [PhaseRecord(phases={"load": 1.0})],
    }
    d = report.diff_total(results, baseline="baseline")
    assert d["fast"]["speedup_vs_baseline"] == 4.0


def test_summarize_arms_handles_skipped():
    results = {"fp16": [PhaseRecord(phases={}, meta={"skipped": "no cuda"})]}
    s = report.summarize_arms(results)
    assert "skipped" in s["fp16"]
