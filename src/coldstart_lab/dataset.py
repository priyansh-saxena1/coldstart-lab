"""Export a fleet's results as a publishable dataset.

The merged ledger is nested JSON shaped for the coordinator, which is awkward to
query. A published dataset should be flat, typed and self-describing so someone
can load it and immediately group by model or condition without writing a
parser first.

Two artifacts are produced:

  * ``observations.csv`` -- one row per (model, experiment, condition), joined
    against the registry so checkpoint size, parameter count, shard count and
    architecture family travel with the measurement. This is the table almost
    every question is asked of.
  * ``runs.json`` -- the raw merged ledger, unmodified, so nothing is lost to
    the flattening and the summary statistics can be recomputed from source.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from coldstart_lab.analysis import Observation, observations_from_merged
from coldstart_lab.models import MODEL_REGISTRY

FIELDS = [
    "model_key", "repo_id", "family", "tier",
    "params_b", "checkpoint_gib", "n_shards", "bytes_per_param",
    "experiment", "condition",
    "p50_ms", "p95_ms", "stdev_ms", "rsd", "n_trials",
    "throughput_mib_s", "gpu", "device_class", "reliable",
]

RSD_RELIABLE_THRESHOLD = 0.30


def observation_rows(observations: List[Observation]) -> List[Dict]:
    rows = []
    for o in observations:
        spec = MODEL_REGISTRY.get(o.model_key)
        rsd = (o.stdev_ms / o.p50_ms) if o.p50_ms > 0 else None
        rows.append({
            "model_key": o.model_key,
            "repo_id": spec.repo_id if spec else "",
            "family": spec.family if spec else "",
            "tier": spec.tier if spec else "",
            "params_b": spec.params_b if spec else None,
            "checkpoint_gib": spec.approx_disk_gib if spec else None,
            "n_shards": spec.n_shards if spec else None,
            "bytes_per_param": round(spec.bytes_per_param, 3) if spec else None,
            "experiment": o.experiment,
            "condition": o.condition,
            "p50_ms": round(o.p50_ms, 2),
            "p95_ms": round(o.p95_ms, 2),
            "stdev_ms": round(o.stdev_ms, 2),
            "rsd": round(rsd, 4) if rsd is not None else None,
            "n_trials": o.n,
            "throughput_mib_s": round(o.throughput_mib_s, 2),
            "gpu": o.gpu or "",
            "device_class": o.device_class or "",
            # Recorded rather than filtered: a reader may reasonably disagree
            # about the threshold, but should not have to guess which rows are
            # noisy enough that no conclusion should rest on them.
            "reliable": (rsd is not None and rsd < RSD_RELIABLE_THRESHOLD),
        })
    return sorted(rows, key=lambda r: (r["experiment"], r["condition"],
                                       r["checkpoint_gib"] or 0))


def export(merged: dict, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    obs = observations_from_merged(merged)
    rows = observation_rows(obs)

    csv_path = os.path.join(out_dir, "observations.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    raw_path = os.path.join(out_dir, "runs.json")
    with open(raw_path, "w") as fh:
        json.dump(merged, fh, indent=1, default=str)

    return {"observations_csv": csv_path, "runs_json": raw_path,
            "n_rows": len(rows)}
