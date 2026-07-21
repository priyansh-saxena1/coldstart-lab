"""Turn raw records into the numbers you'd actually put in front of someone.

Two jobs here:
  1. Summarize each experiment arm (p50/p95 per phase) and diff the arms.
  2. Fit load-time vs model-size across the curve and extrapolate to production
     sizes. This is the step that turns "I benchmarked some tiny models on a free
     GPU" into "here's what I'd predict for a 27B model on your storage setup" —
     which is the only version of this that's useful to an infra team.

The fit is deliberately a plain line: load_seconds ≈ slope * GB + intercept.
slope is 1/effective-bandwidth (how fast bytes actually reach the model), the
intercept is the size-independent overhead (imports, context init). It's a model,
not truth — we report R^2 so the reader can see how much to trust it, and it
breaks down once you cross into multi-GPU sharding, which we flag rather than
pretend to cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .timing import PhaseRecord, summarize


def summarize_arms(results: Dict[str, List[PhaseRecord]]) -> dict:
    out = {}
    for arm, recs in results.items():
        real = [r for r in recs if r.phases]  # drop skip-notes
        if not real:
            note = recs[0].meta.get("skipped") or recs[0].meta.get("todo") if recs else "no data"
            out[arm] = {"skipped": note}
        else:
            out[arm] = summarize(real)
    return out


def diff_total(results: Dict[str, List[PhaseRecord]], baseline: str) -> dict:
    """Speedup of each arm relative to a baseline arm, on p50 total."""
    summ = summarize_arms(results)
    if baseline not in summ or "skipped" in summ[baseline]:
        return {}
    base_p50 = summ[baseline]["total"]["p50"]
    out = {}
    for arm, s in summ.items():
        if "skipped" in s:
            continue
        p50 = s["total"]["p50"]
        out[arm] = {"p50_total_s": round(p50, 3),
                    "speedup_vs_%s" % baseline: round(base_p50 / p50, 2) if p50 else None}
    return out


@dataclass
class LoadCurve:
    slope_s_per_gb: float
    intercept_s: float
    r2: float
    eff_bandwidth_MBps: float
    n: int

    def predict(self, gb: float) -> float:
        return self.slope_s_per_gb * gb + self.intercept_s


def fit_load_curve(size_gb: List[float], load_s: List[float]) -> Optional[LoadCurve]:
    """Least-squares line through (GB, load seconds). Needs >= 3 points."""
    if len(size_gb) < 3:
        return None
    x = np.asarray(size_gb, float)
    y = np.asarray(load_s, float)
    slope, intercept = np.polyfit(x, y, 1)

    resid = y - (slope * x + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # slope is s/GB; invert to an effective bandwidth in MB/s.
    eff_bw = (1000.0 / slope) if slope > 0 else float("inf")
    return LoadCurve(float(slope), float(intercept), float(r2), float(eff_bw), len(x))


def extrapolate(curve: LoadCurve, targets_gb: Dict[str, float]) -> dict:
    """Predict load seconds for named production sizes."""
    return {
        name: {"gb": gb, "predicted_load_s": round(curve.predict(gb), 1)}
        for name, gb in targets_gb.items()
    }


def curve_from_sweep(sweep: Dict[str, List[PhaseRecord]], specs_by_id: dict) -> Optional[LoadCurve]:
    """Build the size/load points from a per-model sweep and fit them.

    `sweep` is {model_id: [records]}; specs_by_id maps model_id -> ModelSpec so we
    can get the on-disk GB. Uses p50 of the 'load' phase — the phase that scales
    with weight bytes — not the total, which includes size-independent overhead.
    """
    sizes, loads = [], []
    for mid, recs in sweep.items():
        real = [r for r in recs if r.phases]
        if not real or mid not in specs_by_id:
            continue
        s = summarize(real)
        load_p50 = s["phases"].get("load", {}).get("p50")
        if load_p50 is None:
            continue
        sizes.append(specs_by_id[mid].approx_gb)
        loads.append(load_p50)
    return fit_load_curve(sizes, loads)
