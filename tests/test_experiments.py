import pytest

from coldstart_lab.experiments.base import Experiment, _describe, _percentile
from coldstart_lab.experiments.format_experiment import FormatExperiment
from coldstart_lab.experiments.storage_experiment import StorageExperiment, StorageTier
from coldstart_lab.experiments.engine_experiment import EngineInitExperiment
from coldstart_lab.extrapolate import project_load_time
from coldstart_lab.models import models_in_tier


def test_percentile_endpoints():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 0.0) == 1.0
    assert _percentile(vals, 1.0) == 5.0
    assert _percentile(vals, 0.5) == 3.0


def test_describe_shape():
    d = _describe([10.0, 20.0, 30.0])
    assert d["n"] == 3
    assert d["min"] == 10.0
    assert d["max"] == 30.0
    assert d["p50"] == 20.0


class _CountingExperiment(Experiment):
    name = "counting"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = {"a": 0, "b": 0}

    def conditions(self):
        def mk(key):
            def run():
                self.calls[key] += 1
                return {"total_ms": float(self.calls[key])}
            return run
        return {"a": mk("a"), "b": mk("b")}


def test_warmup_runs_are_discarded():
    exp = _CountingExperiment(repeats=3, warmup=2)
    result = exp.run()
    # Each condition called warmup+repeats = 5 times, but only 3 recorded.
    assert exp.calls == {"a": 5, "b": 5}
    per_cond = {}
    for t in result.trials:
        per_cond.setdefault(t.condition, 0)
        per_cond[t.condition] += 1
    assert per_cond == {"a": 3, "b": 3}


def test_experiment_rejects_bad_params():
    with pytest.raises(ValueError):
        _CountingExperiment(repeats=0)
    with pytest.raises(ValueError):
        _CountingExperiment(warmup=-1)


def test_format_experiment_cpu(synthetic_checkpoint):
    exp = FormatExperiment(
        model_dir=synthetic_checkpoint, device="cpu",
        include_bin=True, repeats=2, warmup=0,
    )
    result = exp.run()
    summary = result.summary("total_ms")
    assert {"safetensors", "safetensors-nommap", "pytorch_bin"} <= set(summary)
    for stats in summary.values():
        assert stats["p50"] > 0


def test_storage_experiment_emulated_ceiling_is_monotonic(synthetic_checkpoint, tmp_path):
    tiers = [
        StorageTier(name="local", root=str(tmp_path)),
        StorageTier(name="remote", root=str(tmp_path), emulated_mib_s=1.0),
    ]
    exp = StorageExperiment(
        model_dir=synthetic_checkpoint, tiers=tiers, device="cpu",
        repeats=2, warmup=0,
    )
    result = exp.run()
    summary = result.summary("total_ms")
    # A 1 MiB/s emulated ceiling must not be faster than the real local read.
    assert summary["remote"]["p50"] >= summary["local"]["p50"]


def test_storage_experiment_requires_tiers(synthetic_checkpoint):
    with pytest.raises(ValueError):
        StorageExperiment(model_dir=synthetic_checkpoint, tiers=[])


def test_engine_experiment_transformers_cpu(synthetic_checkpoint):
    # The synthetic checkpoint has no config.json, so a full transformers load
    # would fail; we only assert backend selection logic here on CPU.
    exp = EngineInitExperiment(model_dir=synthetic_checkpoint, device="cpu")
    assert exp.backends == ["transformers"]


def test_projection_scales_with_size():
    targets = models_in_tier("reference")
    assert targets
    projections = project_load_time(1000.0, targets, basis="unit-test")
    # Bigger checkpoints must project to longer load times.
    by_key = {p.model_key: p for p in projections}
    ordered = sorted(by_key.values(), key=lambda p: p.approx_disk_gib)
    times = [p.predicted_load_s for p in ordered]
    assert times == sorted(times)


def test_projection_rejects_nonpositive_throughput():
    with pytest.raises(ValueError):
        project_load_time(0.0, models_in_tier("reference"), basis="x")
