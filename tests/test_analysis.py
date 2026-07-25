"""Tests for the cross-model analysis layer.

The statistics here decide what the final report claims, so they are tested
against inputs whose right answer is known by construction.
"""

import math

import pytest

from coldstart_lab.analysis import (MIN_R2_FOR_PROJECTION, Analysis, Observation,
                                    build_report, choose_projection_fit,
                                    fit_linear, fit_power, fit_proportional,
                                    observations_from_merged, stable_regime)


# ------------------------------------------------------------------ regression
def test_fit_recovers_known_line():
    # ms = 100 + 500 * gib, exactly
    pts = [(g, 100 + 500 * g) for g in (1, 2, 4, 8, 16)]
    fit = fit_linear(pts)
    assert fit is not None
    assert fit.slope_ms_per_gib == pytest.approx(500.0, rel=1e-6)
    assert fit.intercept_ms == pytest.approx(100.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_implied_throughput_is_consistent():
    # 1024 ms per GiB == 1024 MiB per 1.024 s == 1000 MiB/s
    fit = fit_linear([(g, 1024 * g) for g in (1, 2, 3, 4)])
    assert fit.implied_mib_s == pytest.approx(1000.0, rel=1e-6)


def test_fit_needs_enough_points():
    assert fit_linear([(1.0, 10.0), (2.0, 20.0)]) is None


def test_fit_rejects_zero_variance_in_x():
    assert fit_linear([(1.0, 10.0), (1.0, 20.0), (1.0, 30.0)]) is None


def test_fit_on_noise_has_low_r2():
    pts = [(1, 900), (2, 100), (3, 850), (4, 120), (5, 800)]
    fit = fit_linear(pts)
    assert fit.r_squared < 0.5


# -------------------------------------------------------- projection selection
def test_projection_prefers_high_r2_not_shallow_slope():
    """REGRESSION: a fast slope on a meaningless fit must not win.

    Observed in a real run: `pytorch_bin` had the shallowest slope but R2=0.02,
    and was selected as the projection basis. Projecting from a fit that
    explains 2% of the variance is projecting from noise.
    """
    good = fit_linear([(g, 100 + 600 * g) for g in (1, 2, 4, 8, 16)])
    junk = fit_linear([(1, 900), (2, 100), (4, 850), (8, 120), (16, 800)])
    assert junk.slope_ms_per_gib < good.slope_ms_per_gib  # junk looks "faster"
    assert junk.r_squared < good.r_squared

    cond, fit, why = choose_projection_fit({"junk": junk, "good": good})
    assert cond == "good", "must select on fit quality, not on slope"
    assert "R2" in why


def test_projection_flags_when_nothing_clears_threshold():
    a = fit_linear([(1, 900), (2, 100), (4, 850), (8, 120)])
    b = fit_linear([(1, 800), (2, 200), (4, 700), (8, 300)])
    cond, fit, why = choose_projection_fit({"a": a, "b": b})
    assert fit.r_squared < MIN_R2_FOR_PROJECTION
    assert "indicative only" in why


def test_choose_handles_empty():
    cond, fit, why = choose_projection_fit({})
    assert cond is None and fit is None


# ------------------------------------------------------------------ parsing
def _merged_fixture():
    def summ(p50):
        return {"n": 3, "mean": p50, "p50": p50, "p95": p50 * 1.1,
                "min": p50 * 0.9, "max": p50 * 1.2, "stdev": p50 * 0.02}
    return {
        "qwen2.5-0.5b": {
            "checkpoint_format": {
                "_gpu_name": "Tesla T4", "_device_class": "t4",
                "summary": {"safetensors": summ(600.0),
                            "pytorch_bin": summ(1500.0)},
            },
            "storage_tier": {
                "_gpu_name": "Tesla T4", "_device_class": "t4",
                "summary": {"local-nvme": summ(400.0),
                            "remote-emulated-200MiBs": summ(4800.0)},
            },
        },
        "qwen2.5-3b": {
            "checkpoint_format": {
                "_gpu_name": "Tesla T4", "_device_class": "t4",
                "summary": {"safetensors": summ(3600.0),
                            "pytorch_bin": summ(9000.0)},
            },
        },
        "smollm2-135m": {
            "checkpoint_format": {
                "_gpu_name": "Tesla T4", "_device_class": "t4",
                "summary": {"safetensors": summ(180.0),
                            "pytorch_bin": summ(450.0)},
            },
        },
    }


def test_observations_flattened_correctly():
    obs = observations_from_merged(_merged_fixture())
    assert len(obs) == 8
    fmt = [o for o in obs if o.experiment == "checkpoint_format"]
    assert len(fmt) == 6
    assert all(o.gpu == "Tesla T4" for o in obs)


def test_observation_links_to_registry_size():
    obs = observations_from_merged(_merged_fixture())
    o = next(o for o in obs if o.model_key == "qwen2.5-3b")
    assert o.gib > 5.0, "should resolve real size from the registry"
    assert o.throughput_mib_s > 0


def test_unknown_model_key_does_not_crash():
    merged = {"not-a-real-model": {"checkpoint_format": {
        "summary": {"safetensors": {"n": 1, "p50": 10.0, "p95": 10.0, "stdev": 0.0}}}}}
    obs = observations_from_merged(merged)
    assert obs[0].gib == 0.0
    assert obs[0].throughput_mib_s == 0.0


# ------------------------------------------------------------------ analyses
def test_format_speedup_computed_per_model():
    a = Analysis(observations_from_merged(_merged_fixture()))
    rows = a.format_speedup()
    assert len(rows) == 3
    for r in rows:
        assert r["speedup"] == pytest.approx(2.5, rel=1e-6)


def test_storage_tiers_ranked_fastest_first():
    a = Analysis(observations_from_merged(_merged_fixture()))
    rows = a.storage_tiers()
    assert rows[0]["tier"] == "local-nvme"
    assert rows[0]["median_mib_s"] > rows[-1]["median_mib_s"]


def test_scaling_fit_per_condition():
    a = Analysis(observations_from_merged(_merged_fixture()))
    fits = a.scaling_by_condition("checkpoint_format")
    assert set(fits) == {"safetensors", "pytorch_bin"}
    # bin is uniformly 2.5x slower, so its slope must be steeper
    assert fits["pytorch_bin"].slope_ms_per_gib > fits["safetensors"].slope_ms_per_gib


def test_coverage_counts():
    a = Analysis(observations_from_merged(_merged_fixture()))
    cov = a.coverage()
    assert cov["models"] == 3
    assert cov["observations"] == 8
    assert "t4" not in cov["hardware"]  # hardware is the GPU name, not the class
    assert cov["total_weights_gib"] > 0


def test_noise_profile_reports_rsd():
    a = Analysis(observations_from_merged(_merged_fixture()))
    noise = a.noise_profile()
    assert noise["median_rsd"] == pytest.approx(0.02, abs=1e-6)


def test_projection_scales_monotonically():
    a = Analysis(observations_from_merged(_merged_fixture()))
    fits = a.scaling_by_condition("checkpoint_format")
    _, fit, _ = choose_projection_fit(fits)
    rows = a.project(fit)
    times = [r["predicted_load_s"] for r in rows]
    assert times == sorted(times), "bigger checkpoints must project to longer loads"


# ------------------------------------------------------------------ report
def test_report_renders_all_sections():
    md = build_report(observations_from_merged(_merged_fixture()))
    for heading in ["## 1. Coverage", "## 2.", "## 3.", "## 4.", "## 5.",
                    "## 6.", "## 7.", "## 8.", "## 9.", "## 10. Limitations"]:
        assert heading in md
    assert "lower bound" in md or "Not enough" in md or "stable regime" in md


def test_report_handles_empty_input():
    md = build_report([])
    assert "Models benchmarked: **0**" in md
    assert "Not enough models" in md


def test_report_flags_implausible_negative_intercept():
    """REGRESSION: a linear fit through convex data yields a negative fixed cost.

    Observed on real data: the report presented a -13,426 ms 'fixed cost' with
    a straight face. A checkpoint of size zero cannot take negative time, so
    that number is proof the model is the wrong shape and must be flagged.
    """
    def summ(p50):
        return {"n": 3, "p50": p50, "p95": p50, "stdev": 0.0}
    # Strongly convex: time grows ~quadratically with size.
    convex = {}
    for key, gib in [("bloomz-560m", 1.04), ("tinyllama-1.1b-v1.0", 2.05),
                     ("qwen2.5-1.5b", 2.88), ("phi-2", 5.18),
                     ("stablelm-2-1-6b", 6.13), ("qwen3-4b", 7.49)]:
        convex[key] = {"checkpoint_format": {
            "summary": {"safetensors": summ(500.0 * gib ** 2)}}}
    obs = observations_from_merged(convex)
    a = Analysis(obs)
    shapes = a.shape_by_condition()
    assert shapes["safetensors"]["intercept_implausible"] is True
    assert shapes["safetensors"]["power"].is_superlinear is True
    md = build_report(obs)
    assert "superlinear" in md.lower()


def test_power_fit_recovers_known_exponent():
    pts = [(g, 300.0 * g ** 1.8) for g in (1, 2, 3, 5, 8)]
    pf = fit_power(pts)
    assert pf.exponent == pytest.approx(1.8, rel=1e-6)
    assert pf.is_superlinear is True


def test_power_fit_linear_data_gives_exponent_one():
    pf = fit_power([(g, 700.0 * g) for g in (1, 2, 4, 8)])
    assert pf.exponent == pytest.approx(1.0, rel=1e-6)
    assert pf.is_superlinear is False


def test_proportional_fit_through_origin():
    pf = fit_proportional([(g, 800.0 * g) for g in (1, 2, 4, 8)])
    assert pf.ms_per_gib == pytest.approx(800.0, rel=1e-6)
    assert pf.mib_s == pytest.approx(1280.0, rel=1e-3)


def test_stable_regime_finds_collapse_boundary():
    """Healthy throughput up to 5 GiB, then it halves."""
    pts = []
    for g in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        pts.append((g, g * 1000.0))          # 1024 MiB/s
    for g in (5.0, 6.0, 7.0, 8.0):
        pts.append((g, g * 3000.0))          # ~341 MiB/s
    reg = stable_regime(pts)
    assert reg["boundary_gib"] == pytest.approx(5.0)
    assert reg["collapse_ratio"] > 2.0


def test_stable_regime_none_when_throughput_holds():
    reg = stable_regime([(g, g * 1000.0) for g in (1, 2, 3, 4, 5, 6, 7, 8)])
    assert reg["boundary_gib"] is None
