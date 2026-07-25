import json
import os

import pytest

from coldstart_lab import environment
from coldstart_lab.experiments.base import ExperimentResult, Trial
from coldstart_lab.models import MODEL_REGISTRY, get_model, models_in_tier, downloadable_models
from coldstart_lab.report import Report


def test_registry_keys_unique_and_tiers_valid():
    tiers = {"ci", "micro", "small", "medium", "reference"}
    for key, spec in MODEL_REGISTRY.items():
        assert spec.key == key
        assert spec.tier in tiers
        assert spec.approx_disk_gib > 0


def test_reference_models_not_downloadable():
    for spec in models_in_tier("reference"):
        assert spec.downloadable is False
    assert all(s.downloadable for s in downloadable_models())


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("does-not-exist")


def test_environment_probe_populates_fields():
    fp = environment.probe()
    assert fp.cpu_count >= 1
    assert fp.python_version
    assert isinstance(fp.cuda_available, bool)


def test_report_roundtrip(tmp_path):
    fp = environment.probe()
    report = Report(fp)
    result = ExperimentResult(name="demo")
    for i in range(3):
        result.trials.append(Trial(condition="x", run_index=i, metrics={"total_ms": 10.0 + i}))
    report.add(result)
    report.add_extra("note", [{"model_key": "m", "predicted_load_s": 1.0}])

    json_path = report.write_json(str(tmp_path / "r.json"))
    md_path = report.write_markdown(str(tmp_path / "r.md"))

    assert os.path.isfile(json_path)
    assert os.path.isfile(md_path)
    data = json.loads(open(json_path).read())
    assert data["experiments"][0]["name"] == "demo"
    md = open(md_path).read()
    assert "Experiment: `demo`" in md
    assert "production" not in md.lower() or "note" in md.lower()


# ------------------------------------------------- registry integrity (61 models)
def test_registry_has_no_gated_repos():
    """A 401 is not transient and starves the queue; keep the registry ungated."""
    for spec in MODEL_REGISTRY.values():
        assert "gated" not in spec.tags, f"{spec.key} is gated"
        assert not spec.repo_id.startswith("meta-llama/"), spec.key
        assert not spec.repo_id.startswith("google/gemma"), spec.key


def test_registry_size_and_breadth():
    assert len(MODEL_REGISTRY) >= 40, "registry should cover a wide size range"
    families = {s.family for s in MODEL_REGISTRY.values() if s.family}
    assert len(families) >= 15, "need architectural diversity, not one family scaled"


def test_tiers_are_ordered_by_footprint():
    """Tier boundaries must not overlap, or 'small' stops meaning 'fits a T4'."""
    from coldstart_lab.models import models_in_tier

    micro = [s.approx_disk_gib for s in models_in_tier("micro")]
    small = [s.approx_disk_gib for s in models_in_tier("small")]
    medium = [s.approx_disk_gib for s in models_in_tier("medium")]
    reference = [s.approx_disk_gib for s in models_in_tier("reference")]
    assert max(micro) <= min(small)
    assert max(small) <= min(medium)
    assert max(medium) <= min(reference)


def test_small_tier_fits_a_t4_with_staging_headroom():
    from coldstart_lab.models import models_in_tier

    for spec in models_in_tier("small"):
        assert spec.approx_disk_gib <= 7.5, f"{spec.key} will not fit a T4 run"


def test_reference_models_are_not_downloadable():
    from coldstart_lab.models import models_in_tier

    for spec in models_in_tier("reference"):
        assert spec.downloadable is False


def test_bytes_per_param_detects_dtype():
    """~2 bytes/param is fp16; a 4-bit quant must be far below that."""
    fp16 = MODEL_REGISTRY["qwen2.5-7b"]
    awq = MODEL_REGISTRY["qwen2.5-7b-awq"]
    assert 1.8 <= fp16.bytes_per_param <= 2.3
    assert awq.bytes_per_param < 1.0
    assert awq.params_b == fp16.params_b, "pair must share a parameter count"


def test_quantized_pairs_are_matched():
    from coldstart_lab.models import quantized_pairs

    pairs = quantized_pairs()
    assert len(pairs) >= 3
    for base, quant in pairs:
        assert base.params_b == quant.params_b
        assert quant.approx_disk_gib < base.approx_disk_gib


def test_shard_counts_recorded():
    """Per-shard overhead analysis needs both single- and multi-shard models."""
    counts = {s.n_shards for s in MODEL_REGISTRY.values()}
    assert 1 in counts and any(c > 1 for c in counts)
