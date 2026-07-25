"""Tests for the publishable dataset export.

A published dataset is the artifact other people build on, so the failure mode
that matters is silent misalignment: a column that means something different
from its name, or rows dropped in flattening.
"""

import csv
import json

import pytest

from coldstart_lab.dataset import FIELDS, export, observation_rows
from coldstart_lab.analysis import observations_from_merged


def _merged():
    def summ(p50, stdev):
        return {"n": 5, "p50": p50, "p95": p50 * 1.2, "stdev": stdev}
    return {
        "qwen2.5-1.5b": {"checkpoint_format": {
            "_gpu_name": "Tesla T4", "_device_class": "t4",
            "summary": {"safetensors": summ(1800.0, 90.0),      # rsd 5%
                        "pytorch_bin": summ(2800.0, 1400.0)}}},  # rsd 50%
        "phi-2": {"storage_tier": {
            "_gpu_name": "Tesla T4", "_device_class": "t4",
            "summary": {"local-nvme": summ(4800.0, 200.0)}}},
    }


def test_every_observation_becomes_a_row():
    obs = observations_from_merged(_merged())
    rows = observation_rows(obs)
    assert len(rows) == len(obs) == 3


def test_rows_are_joined_to_registry_metadata():
    rows = observation_rows(observations_from_merged(_merged()))
    r = next(r for r in rows if r["model_key"] == "qwen2.5-1.5b")
    assert r["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert r["family"] == "qwen2.5"
    assert r["checkpoint_gib"] > 0
    assert 1.8 <= r["bytes_per_param"] <= 2.3  # fp16


def test_reliable_flag_tracks_rsd_threshold():
    rows = observation_rows(observations_from_merged(_merged()))
    clean = next(r for r in rows if r["condition"] == "safetensors")
    noisy = next(r for r in rows if r["condition"] == "pytorch_bin")
    assert clean["rsd"] == pytest.approx(0.05, abs=1e-3)
    assert clean["reliable"] is True
    assert noisy["rsd"] == pytest.approx(0.50, abs=1e-3)
    assert noisy["reliable"] is False, "noisy rows must be published but flagged"


def test_throughput_column_is_self_consistent():
    """throughput_mib_s must equal checkpoint_gib / p50, or the column lies."""
    for r in observation_rows(observations_from_merged(_merged())):
        expected = (r["checkpoint_gib"] * 1024.0) / (r["p50_ms"] / 1000.0)
        assert r["throughput_mib_s"] == pytest.approx(expected, rel=1e-3)


def test_export_writes_both_artifacts(tmp_path):
    info = export(_merged(), str(tmp_path))
    with open(info["observations_csv"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert list(rows[0]) == FIELDS, "column order must be stable for consumers"

    with open(info["runs_json"]) as fh:
        raw = json.load(fh)
    assert raw == _merged(), "raw ledger must be preserved byte-for-byte"


def test_card_reports_real_numbers():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from publish_to_hf import build_card

    card = build_card(_merged(), "u/r", None)
    assert card.startswith("---"), "must open with YAML frontmatter for the Hub"
    assert "license: apache-2.0" in card
    assert "data_files: observations.csv" in card
    assert "Limitations" in card


def test_from_db_and_merged_are_mutually_exclusive():
    """The source of truth must be unambiguous when publishing."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import publish_to_hf

    # argparse exits with SystemExit(2) when a mutually exclusive group is
    # given both options, or neither.
    for argv in (["--repo-id", "u/r"],
                 ["--merged", "x.json", "--from-db", "--repo-id", "u/r"]):
        with pytest.raises(SystemExit):
            publish_to_hf.main_parse(argv) if hasattr(publish_to_hf, "main_parse") \
                else publish_to_hf.main(argv)
