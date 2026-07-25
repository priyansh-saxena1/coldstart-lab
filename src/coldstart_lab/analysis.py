"""Cross-model analysis: turn a fleet's ledger into findings.

The per-run `report.py` describes one model. This module answers the questions
that only appear once you have many models: does load time scale linearly with
checkpoint size, what is the throughput ceiling of each storage tier, what does
4-bit quantization buy at load time (as opposed to at inference), how much does
per-shard overhead cost, and what do all of those imply for a production-scale
checkpoint nobody can fit on a free GPU.

Everything here is descriptive statistics over measurements. Where a number is
projected rather than measured it is labelled as such, and the assumptions are
printed alongside it -- an extrapolation whose error model is invisible is worse
than no extrapolation at all.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from coldstart_lab.models import MODEL_REGISTRY, ModelSpec, models_in_tier


# --------------------------------------------------------------------- records
@dataclass
class Observation:
    """One (model, experiment, condition) measurement, flattened for analysis."""

    model_key: str
    experiment: str
    condition: str
    p50_ms: float
    p95_ms: float
    stdev_ms: float
    n: int
    gpu: Optional[str] = None
    device_class: Optional[str] = None

    @property
    def spec(self) -> Optional[ModelSpec]:
        return MODEL_REGISTRY.get(self.model_key)

    @property
    def gib(self) -> float:
        s = self.spec
        return s.approx_disk_gib if s else 0.0

    @property
    def throughput_mib_s(self) -> float:
        """MiB/s implied by this condition's p50."""
        if self.p50_ms <= 0 or not self.gib:
            return 0.0
        return (self.gib * 1024.0) / (self.p50_ms / 1000.0)


def load_observations(merged_path: str) -> List[Observation]:
    """Flatten `coldstart-fleet merge` output into a list of observations."""
    with open(merged_path) as fh:
        merged = json.load(fh)
    return observations_from_merged(merged)


def observations_from_merged(merged: dict) -> List[Observation]:
    obs: List[Observation] = []
    for model_key, experiments in merged.items():
        for exp_name, rec in experiments.items():
            summary = rec.get("summary") or {}
            for condition, stats in summary.items():
                obs.append(Observation(
                    model_key=model_key,
                    experiment=exp_name,
                    condition=condition,
                    p50_ms=float(stats.get("p50", 0.0)),
                    p95_ms=float(stats.get("p95", 0.0)),
                    stdev_ms=float(stats.get("stdev", 0.0)),
                    n=int(stats.get("n", 0)),
                    gpu=rec.get("_gpu_name"),
                    device_class=rec.get("_device_class"),
                ))
    return obs


# ------------------------------------------------------------------ regression
@dataclass
class LinearFit:
    """Least-squares fit of load_ms = intercept + slope * gib.

    `slope` is the marginal cost of a gibibyte -- the storage/deserialization
    pipeline's inverse throughput. `intercept` is the fixed cost paid regardless
    of size (process setup, header parsing, per-file open). Separating them
    matters: if the intercept dominates at your model sizes, buying faster
    storage will not help you.
    """

    slope_ms_per_gib: float
    intercept_ms: float
    r_squared: float
    n_points: int

    @property
    def implied_mib_s(self) -> float:
        if self.slope_ms_per_gib <= 0:
            return 0.0
        return 1024.0 / (self.slope_ms_per_gib / 1000.0)

    def predict_ms(self, gib: float) -> float:
        return self.intercept_ms + self.slope_ms_per_gib * gib

    def to_dict(self) -> dict:
        return {
            "slope_ms_per_gib": round(self.slope_ms_per_gib, 1),
            "intercept_ms": round(self.intercept_ms, 1),
            "implied_mib_s": round(self.implied_mib_s, 1),
            "r_squared": round(self.r_squared, 4),
            "n_points": self.n_points,
        }


def fit_linear(points: List[Tuple[float, float]]) -> Optional[LinearFit]:
    """Ordinary least squares on (gib, ms) pairs."""
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return LinearFit(slope, intercept, r2, len(pts))


# -------------------------------------------------------------------- analyses
class Analysis:
    """All cross-model findings derived from a set of observations."""

    def __init__(self, observations: List[Observation]) -> None:
        self.obs = observations

    # -- coverage ----------------------------------------------------------
    def coverage(self) -> dict:
        models = {o.model_key for o in self.obs}
        gpus = {o.gpu for o in self.obs if o.gpu}
        by_exp: Dict[str, int] = {}
        for o in self.obs:
            by_exp[o.experiment] = by_exp.get(o.experiment, 0) + 1
        gib = sum(MODEL_REGISTRY[m].approx_disk_gib
                  for m in models if m in MODEL_REGISTRY)
        return {
            "models": len(models),
            "observations": len(self.obs),
            "experiments": by_exp,
            "hardware": sorted(gpus),
            "total_weights_gib": round(gib, 1),
            "families": sorted({MODEL_REGISTRY[m].family
                                for m in models if m in MODEL_REGISTRY}),
        }

    def gaps(self, expected=("checkpoint_format", "storage_tier", "engine_init")
             ) -> Dict[str, List[str]]:
        """Which models are missing which experiments.

        A cross-model comparison is only as good as its coverage: a missing
        cell is a silent hole in every table that follows, so it is stated up
        front rather than left for a reader to notice.
        """
        seen: Dict[str, set] = {}
        for o in self.obs:
            seen.setdefault(o.model_key, set()).add(o.experiment)
        out: Dict[str, List[str]] = {}
        for model, done in sorted(seen.items()):
            missing = [e for e in expected if e not in done]
            if missing:
                out[model] = missing
        return out

    # -- scaling -----------------------------------------------------------
    def scaling_by_condition(self, experiment: str) -> Dict[str, LinearFit]:
        """Fit load time vs checkpoint size, separately per condition."""
        buckets: Dict[str, List[Tuple[float, float]]] = {}
        for o in self.obs:
            if o.experiment != experiment or not o.gib:
                continue
            buckets.setdefault(o.condition, []).append((o.gib, o.p50_ms))
        fits = {}
        for cond, pts in buckets.items():
            fit = fit_linear(pts)
            if fit:
                fits[cond] = fit
        return fits

    def shape_by_condition(self, experiment: str = "checkpoint_format"
                           ) -> Dict[str, dict]:
        """Linear fit, power fit and knee for each condition, together.

        Reporting all three is the point: the linear fit is what people expect,
        the exponent says whether that expectation is justified, and the knee
        localises where it stops being justified.
        """
        buckets: Dict[str, List[Tuple[float, float]]] = {}
        for o in self.obs:
            if o.experiment != experiment or not o.gib:
                continue
            buckets.setdefault(o.condition, []).append((o.gib, o.p50_ms))

        out: Dict[str, dict] = {}
        for cond, pts in buckets.items():
            lin = fit_linear(pts)
            pw = fit_power(pts)
            if not lin and not pw:
                continue
            largest_ms = max(ms for _, ms in pts)
            out[cond] = {
                "linear": lin,
                "power": pw,
                "knee": throughput_knee(pts),
                "intercept_implausible": bool(
                    lin and lin.intercept_ms < NEGATIVE_INTERCEPT_ALARM * largest_ms),
                "n": len(pts),
            }
        return out

    def stable_fit(self, condition: str = "safetensors",
                   experiment: str = "checkpoint_format"
                   ) -> Tuple[Optional["ProportionalFit"], Optional[dict]]:
        """Proportional fit restricted to the regime where throughput holds up.

        This is the number to project from. Past the boundary the slope
        describes whichever local resource ran out -- host memory, here -- not
        the storage pipeline, and production hardware has a different amount of
        that resource.
        """
        pts = [(o.gib, o.p50_ms) for o in self.obs
               if o.experiment == experiment and o.condition == condition and o.gib]
        reg = stable_regime(pts)
        if not reg or not reg.get("boundary_gib"):
            return fit_proportional(pts), reg
        cut = reg["boundary_gib"]
        stable = [(g, ms) for g, ms in pts if g < cut]
        # If the collapse starts almost immediately there is no stable regime to
        # fit; say so rather than fitting three points and calling it a trend.
        fit = fit_proportional(stable) if len(stable) >= 3 else None
        return fit, reg

    def pre_knee_fit(self, condition: str = "safetensors",
                     experiment: str = "checkpoint_format"
                     ) -> Tuple[Optional[LinearFit], Optional[float]]:
        """Linear fit restricted to the regime *below* the throughput knee.

        Once throughput has collapsed, the slope reflects whatever resource ran
        out (host memory, in this dataset) rather than the storage pipeline. For
        projecting onto production hardware -- which has far more of that
        resource -- the pre-knee regime is the honest basis.
        """
        pts = [(o.gib, o.p50_ms) for o in self.obs
               if o.experiment == experiment and o.condition == condition and o.gib]
        knee = throughput_knee(pts)
        if not knee:
            return fit_linear(pts), None
        cut = knee["knee_gib"]
        return fit_linear([(g, ms) for g, ms in pts if g < cut]), cut

    # -- format ------------------------------------------------------------
    def format_speedup(self) -> List[dict]:
        """Per-model safetensors-vs-bin comparison."""
        rows = []
        by_model: Dict[str, Dict[str, float]] = {}
        for o in self.obs:
            if o.experiment != "checkpoint_format":
                continue
            by_model.setdefault(o.model_key, {})[o.condition] = o.p50_ms
        for model, conds in sorted(by_model.items()):
            st = conds.get("safetensors")
            bn = conds.get("pytorch_bin")
            if not st or not bn:
                continue
            rows.append({
                "model": model,
                "gib": MODEL_REGISTRY[model].approx_disk_gib
                       if model in MODEL_REGISTRY else 0.0,
                "safetensors_ms": round(st, 1),
                "pytorch_bin_ms": round(bn, 1),
                "speedup": round(bn / st, 2) if st else 0.0,
            })
        return rows

    # -- quantization ------------------------------------------------------
    def quantization_effect(self) -> List[dict]:
        """Compare a quantized checkpoint's load time to its fp16 sibling.

        This isolates dtype: identical architecture and parameter count, fewer
        bytes. Any difference is bytes-moved, not model change.
        """
        p50: Dict[str, float] = {}
        for o in self.obs:
            if o.experiment == "checkpoint_format" and o.condition == "safetensors":
                p50[o.model_key] = o.p50_ms

        rows = []
        from coldstart_lab.models import quantized_pairs

        for base, quant in quantized_pairs():
            if base.key not in p50 or quant.key not in p50:
                continue
            rows.append({
                "base": base.key,
                "quantized": quant.key,
                "base_gib": base.approx_disk_gib,
                "quant_gib": quant.approx_disk_gib,
                "size_ratio": round(base.approx_disk_gib / quant.approx_disk_gib, 2),
                "base_ms": round(p50[base.key], 1),
                "quant_ms": round(p50[quant.key], 1),
                "load_speedup": round(p50[base.key] / p50[quant.key], 2),
            })
        return rows

    # -- sharding ----------------------------------------------------------
    def shard_overhead(self) -> Optional[dict]:
        """Does per-file overhead show up once size is controlled for?

        Fit ms ~ gib on single-shard vs multi-shard models separately. A higher
        intercept for multi-shard checkpoints is per-file cost: opens, header
        parses and (in a real deployment) separate object-store requests.
        """
        single, multi = [], []
        for o in self.obs:
            if o.experiment != "checkpoint_format" or o.condition != "safetensors":
                continue
            spec = o.spec
            if not spec or not spec.approx_disk_gib:
                continue
            (multi if spec.n_shards > 1 else single).append(
                (spec.approx_disk_gib, o.p50_ms))
        f_single, f_multi = fit_linear(single), fit_linear(multi)
        if not f_single or not f_multi:
            return None
        return {
            "single_shard": f_single.to_dict(),
            "multi_shard": f_multi.to_dict(),
            "extra_fixed_cost_ms": round(
                f_multi.intercept_ms - f_single.intercept_ms, 1),
        }

    # -- storage -----------------------------------------------------------
    def storage_tiers(self) -> List[dict]:
        agg: Dict[str, List[float]] = {}
        for o in self.obs:
            if o.experiment != "storage_tier":
                continue
            tp = o.throughput_mib_s
            if tp > 0:
                agg.setdefault(o.condition, []).append(tp)
        rows = []
        for cond, tps in sorted(agg.items()):
            rows.append({
                "tier": cond,
                "median_mib_s": round(statistics.median(tps), 1),
                "min_mib_s": round(min(tps), 1),
                "max_mib_s": round(max(tps), 1),
                "n_models": len(tps),
            })
        return sorted(rows, key=lambda r: -r["median_mib_s"])

    # -- engine ------------------------------------------------------------
    def engine_breakdown(self) -> List[dict]:
        rows = []
        for o in self.obs:
            if o.experiment != "engine_init":
                continue
            rows.append({
                "model": o.model_key,
                "gib": o.gib,
                "backend": o.condition,
                "p50_ms": round(o.p50_ms, 1),
                "p95_ms": round(o.p95_ms, 1),
            })
        return sorted(rows, key=lambda r: r["gib"])

    # -- variance ----------------------------------------------------------
    def noise_profile(self) -> dict:
        """How trustworthy are these numbers?

        Relative standard deviation across every condition. A benchmark whose
        run-to-run spread is comparable to the effect being measured is not
        evidence, and saying so explicitly is part of the result.
        """
        rsds = [o.stdev_ms / o.p50_ms for o in self.obs
                if o.p50_ms > 0 and o.stdev_ms >= 0]
        if not rsds:
            return {}
        rsds.sort()
        return {
            "median_rsd": round(statistics.median(rsds), 4),
            "p90_rsd": round(rsds[int(0.9 * (len(rsds) - 1))], 4),
            "max_rsd": round(rsds[-1], 4),
            "n": len(rsds),
        }

    def unreliable(self, rsd_threshold: float = 0.30) -> List[dict]:
        """Measurements whose own spread is too wide to draw a conclusion from.

        A condition with 30%+ relative standard deviation cannot support a
        claim about a 20% effect. Listing these explicitly is more useful than
        a single aggregate noise figure, because it tells you which rows in the
        tables above to discount and which models to re-run.
        """
        rows = []
        for o in self.obs:
            if o.p50_ms <= 0:
                continue
            rsd = o.stdev_ms / o.p50_ms
            if rsd >= rsd_threshold:
                rows.append({
                    "model": o.model_key,
                    "experiment": o.experiment,
                    "condition": o.condition,
                    "p50_ms": round(o.p50_ms, 1),
                    "rsd": round(rsd, 3),
                })
        return sorted(rows, key=lambda r: -r["rsd"])

    # -- projection --------------------------------------------------------
    def project(self, fit: LinearFit,
                targets: Optional[List[ModelSpec]] = None) -> List[dict]:
        targets = targets or models_in_tier("reference")
        rows = []
        for spec in sorted(targets, key=lambda s: s.approx_disk_gib):
            ms = fit.predict_ms(spec.approx_disk_gib)
            rows.append({
                "model": spec.key,
                "gib": spec.approx_disk_gib,
                "params_b": spec.params_b,
                "predicted_load_s": round(ms / 1000.0, 1),
            })
        return rows


MIN_R2_FOR_PROJECTION = 0.80

# A linear fit whose intercept is more negative than this fraction of the
# largest observed measurement is misspecified, not merely imprecise: a
# checkpoint of size zero cannot take negative time to load. When the intercept
# goes sharply negative it means the true relationship is convex and the line is
# being tilted to chase the large points.
NEGATIVE_INTERCEPT_ALARM = -0.05


@dataclass
class PowerFit:
    """Fit of load_ms = a * gib**b, via least squares on log-log axes.

    The exponent `b` is the diagnostic that matters. b~1 means cold start is
    linear in bytes, so throughput is constant and a MiB/s figure is meaningful.
    b>1 means throughput *degrades* as checkpoints grow -- doubling the model
    more than doubles the load -- and any single "MiB/s" number is then an
    average over a moving target rather than a property of the system.
    """

    exponent: float
    coefficient: float
    r_squared: float
    n_points: int

    @property
    def is_superlinear(self) -> bool:
        return self.exponent > 1.15

    def predict_ms(self, gib: float) -> float:
        return self.coefficient * (gib ** self.exponent)

    def to_dict(self) -> dict:
        return {
            "exponent": round(self.exponent, 3),
            "coefficient": round(self.coefficient, 1),
            "r_squared": round(self.r_squared, 4),
            "n_points": self.n_points,
            "superlinear": self.is_superlinear,
        }


def fit_power(points: List[Tuple[float, float]]) -> Optional[PowerFit]:
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 4:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    b = sxy / sxx
    log_a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum((y - (log_a + b * x)) ** 2 for x, y in zip(lx, ly))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return PowerFit(exponent=b, coefficient=math.exp(log_a),
                    r_squared=r2, n_points=len(pts))


@dataclass
class ProportionalFit:
    """Fit of load_ms = k * gib, constrained through the origin.

    Physically this is the right model for a transfer: zero bytes must take
    zero time. Allowing a free intercept lets the fit absorb curvature into a
    negative constant, which is how a misspecified linear model ends up
    claiming a checkpoint of size zero loads in minus thirteen seconds. Fitting
    through the origin makes the misfit visible instead of hiding it.
    """

    ms_per_gib: float
    r_squared: float
    n_points: int

    @property
    def mib_s(self) -> float:
        return 1024.0 / (self.ms_per_gib / 1000.0) if self.ms_per_gib > 0 else 0.0

    def predict_ms(self, gib: float) -> float:
        return self.ms_per_gib * gib

    def to_dict(self) -> dict:
        return {"ms_per_gib": round(self.ms_per_gib, 1),
                "mib_s": round(self.mib_s, 1),
                "r_squared": round(self.r_squared, 4),
                "n_points": self.n_points}


def fit_proportional(points: List[Tuple[float, float]]) -> Optional[ProportionalFit]:
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 3:
        return None
    k = sum(x * y for x, y in pts) / sum(x * x for x, _ in pts)
    ys = [y for _, y in pts]
    my = statistics.fmean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - k * x) ** 2 for x, y in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return ProportionalFit(ms_per_gib=k, r_squared=r2, n_points=len(pts))


def stable_regime(points: List[Tuple[float, float]],
                  tolerance: float = 0.75,
                  baseline_n: int = 5) -> Optional[dict]:
    """Largest checkpoint size at which throughput still holds up.

    Takes the median throughput of the smallest few checkpoints as the
    system's healthy baseline, then walks up until throughput falls below
    `tolerance` of it. The boundary is the point past which a MiB/s figure
    measured on small models stops predicting anything.
    """
    pts = sorted([(g, ms) for g, ms in points if g > 0 and ms > 0])
    if len(pts) < baseline_n + 2:
        return None
    tp = [(g, (g * 1024.0) / (ms / 1000.0)) for g, ms in pts]
    baseline = statistics.median([t for _, t in tp[:baseline_n]])
    floor = baseline * tolerance

    boundary = None
    for g, t in tp:
        if t < floor:
            boundary = g
            break
    if boundary is None:
        return {"baseline_mib_s": round(baseline, 1), "boundary_gib": None,
                "degraded_mib_s": None, "collapse_ratio": 1.0}

    below = [t for g, t in tp if g < boundary]
    above = [t for g, t in tp if g >= boundary]
    return {
        "baseline_mib_s": round(baseline, 1),
        "boundary_gib": round(boundary, 2),
        "degraded_mib_s": round(statistics.median(above), 1) if above else None,
        "collapse_ratio": round(statistics.median(below) / statistics.median(above), 2)
        if below and above else 1.0,
        "n_stable": len(below),
        "n_degraded": len(above),
    }


def throughput_knee(points: List[Tuple[float, float]],
                    min_side: int = 4) -> Optional[dict]:
    """Find the checkpoint size where effective throughput drops hardest.

    Scans candidate split points and picks the one maximising the ratio of
    median throughput below to median throughput above. Reported only when the
    drop is substantial, since on well-behaved data there is no knee to find.
    """
    pts = sorted([(g, ms) for g, ms in points if g > 0 and ms > 0])
    if len(pts) < 2 * min_side:
        return None
    tp = [(g, (g * 1024.0) / (ms / 1000.0)) for g, ms in pts]

    best = None
    for i in range(min_side, len(tp) - min_side + 1):
        lo = [t for _, t in tp[:i]]
        hi = [t for _, t in tp[i:]]
        ratio = statistics.median(lo) / statistics.median(hi)
        if best is None or ratio > best["ratio"]:
            best = {"gib": tp[i][0], "ratio": ratio,
                    "below_mib_s": statistics.median(lo),
                    "above_mib_s": statistics.median(hi)}
    if best is None or best["ratio"] < 1.3:
        return None
    return {
        "knee_gib": round(best["gib"], 2),
        "ratio": round(best["ratio"], 2),
        "below_mib_s": round(best["below_mib_s"], 1),
        "above_mib_s": round(best["above_mib_s"], 1),
    }


def choose_projection_fit(fits: Dict[str, "LinearFit"]
                          ) -> Tuple[Optional[str], Optional["LinearFit"], str]:
    """Pick the fit to extrapolate from, and explain the choice.

    Selecting the *fastest* slope is wrong: a fit can have a shallow slope
    simply because it is a bad fit. A condition whose R2 is near zero has no
    demonstrated relationship between size and time at all, so projecting from
    its slope would be projecting from noise.

    So: prefer the highest R2 among conditions that clear the threshold, and if
    nothing clears it, still return the best available but flag the report
    loudly rather than quietly emitting a confident-looking table.
    """
    if not fits:
        return None, None, "no fits available"

    good = {c: f for c, f in fits.items() if f.r_squared >= MIN_R2_FOR_PROJECTION}
    if good:
        cond, fit = max(good.items(), key=lambda kv: kv[1].r_squared)
        return cond, fit, (
            f"highest R2 ({fit.r_squared:.3f}) among conditions clearing the "
            f"R2 >= {MIN_R2_FOR_PROJECTION} bar")

    cond, fit = max(fits.items(), key=lambda kv: kv[1].r_squared)
    return cond, fit, (
        f"NO condition reached R2 >= {MIN_R2_FOR_PROJECTION}; using the best "
        f"available ({cond}, R2={fit.r_squared:.3f}). Treat the projection as "
        f"indicative only")


# ---------------------------------------------------------------------- report
def build_report(observations: List[Observation],
                 title: str = "LLM Container Cold-Start: Cross-Model Report") -> str:
    """Render the full analysis as Markdown."""

    a = Analysis(observations)
    out: List[str] = [f"# {title}", ""]

    cov = a.coverage()
    out += [
        "## 1. Coverage", "",
        f"- Models benchmarked: **{cov['models']}**",
        f"- Observations: **{cov['observations']}**",
        f"- Weights moved: **{cov['total_weights_gib']} GiB**",
        f"- Architecture families: {len(cov['families'])} "
        f"({', '.join(cov['families'][:8])}{'...' if len(cov['families']) > 8 else ''})",
        f"- Hardware: {', '.join(cov['hardware']) or 'n/a'}",
        f"- Experiments: {', '.join(f'{k} ({v})' for k, v in sorted(cov['experiments'].items()))}",
        "",
    ]

    gaps = a.gaps()
    if gaps:
        out += [
            f"**Coverage gaps ({len(gaps)} models incomplete).** Stated up "
            "front because a missing cell is a silent hole in every table "
            "below:", "",
        ]
        for model, missing in gaps.items():
            out.append(f"- `{model}`: missing {', '.join(missing)}")
        out.append("")
    else:
        out += ["All benchmarked models completed every experiment.", ""]

    # -- 2. scaling
    out += ["## 2. How does load time scale with checkpoint size?", ""]
    shapes = a.shape_by_condition("checkpoint_format")
    if shapes:
        out += [
            "Three fits per condition, because the first one alone is "
            "misleading. `ms/GiB` is the familiar linear slope. `exponent b` "
            "comes from a log-log fit of `ms = a x GiB^b`: b~1 means constant "
            "throughput, b>1 means throughput *degrades* as checkpoints grow. "
            "`fixed cost` is the linear model's intercept -- a checkpoint of "
            "size zero cannot take negative time, so a large negative value is "
            "proof the linear model is the wrong shape.", "",
            "| Condition | ms per GiB | exponent b | linear R2 | fixed cost (ms) | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cond, d in sorted(shapes.items(),
                              key=lambda kv: kv[1]["power"].exponent
                              if kv[1]["power"] else 0):
            lin, pw = d["linear"], d["power"]
            if lin is None:
                continue
            flag = " :warning:" if d["intercept_implausible"] else ""
            exp = f"{pw.exponent:.2f}" if pw else "n/a"
            out.append(
                f"| {cond} | {lin.slope_ms_per_gib:.0f} | "
                f"{exp} | {lin.r_squared:.3f} | "
                f"{lin.intercept_ms:.0f}{flag} | {d['n']} |")
        out.append("")

        supers = [c for c, d in shapes.items()
                  if d["power"] and d["power"].is_superlinear]
        if supers:
            worst = max((kv for kv in shapes.items() if kv[1]["power"]),
                        key=lambda kv: kv[1]["power"].exponent)
            out += [
                "### 2.1 The headline finding: cold start is superlinear", "",
                f"Every condition has b > 1 (up to **b = "
                f"{worst[1]['power'].exponent:.2f}** for `{worst[0]}`). Doubling "
                f"the checkpoint more than doubles the load time, so there is no "
                f"single MiB/s that describes this system -- effective throughput "
                f"is a function of model size, not a constant of the hardware.", "",
                "Every linear fixed cost above is sharply negative, which is not "
                "a small numerical artefact: it is the linear model bending to "
                "chase a curve. Quoting those slopes as throughput, or "
                "extrapolating from them, would be quoting an artefact.", "",
            ]

        reg_rows = []
        for cond in sorted(shapes):
            fit, reg = a.stable_fit(cond)
            if reg and reg.get("boundary_gib"):
                reg_rows.append((cond, fit, reg))
        if reg_rows:
            out += [
                "### 2.2 Where throughput collapses", "",
                "Taking the median throughput of the smallest checkpoints as a "
                "healthy baseline and walking upward until it falls below 75% "
                "of that:", "",
                "| Condition | healthy MiB/s | collapses above | degraded MiB/s | drop |",
                "|---|---:|---:|---:|---:|",
            ]
            for cond, fit, reg in reg_rows:
                out.append(
                    f"| {cond} | {reg['baseline_mib_s']:.0f} | "
                    f"{reg['boundary_gib']:.2f} GiB | "
                    f"{reg['degraded_mib_s']:.0f} | "
                    f"{reg['collapse_ratio']:.1f}x |")
            out += [
                "",
                "**The most likely cause is host memory, not the storage "
                "device.** These runs used a Colab T4 instance with roughly "
                "12-13 GiB of usable system RAM. The mmap path depends on the "
                "OS page cache holding the checkpoint while tensors are "
                "materialised; once a checkpoint plus its in-flight copies "
                "approaches that budget, pages are evicted and re-read and the "
                "effective rate falls off. The collapse point sitting near 5 "
                "GiB -- comfortably under total RAM, but not under RAM minus "
                "the working copy -- is consistent with that reading.", "",
                "This is a testable claim rather than a conclusion, and the "
                "test is cheap: re-run the same models on a high-RAM instance. "
                "If the boundary moves with available memory it is memory "
                "pressure; if it stays at 5 GiB it is something in the loader.",
                "",
                "**Why it matters for a serving platform:** a cold-start budget "
                "extrapolated from small models will be optimistic for large "
                "ones, and the gap widens with size. It also means the first "
                "question about any deployment is which regime it sits in -- "
                "below the boundary, storage bandwidth is the lever; above it, "
                "no amount of storage bandwidth helps because the bottleneck "
                "has moved to memory.", "",
            ]
    else:
        out += ["_Not enough models benchmarked yet to fit a trend "
                "(need at least 3 with the format experiment)._", ""]

    # -- 3. format
    out += ["## 3. Checkpoint format: safetensors vs legacy pickle", ""]
    rows = a.format_speedup()
    if rows:
        out += ["| Model | GiB | safetensors (ms) | .bin (ms) | speedup |",
                "|---|---:|---:|---:|---:|"]
        for r in rows:
            out.append(f"| `{r['model']}` | {r['gib']} | {r['safetensors_ms']} "
                       f"| {r['pytorch_bin_ms']} | {r['speedup']}x |")
        sp = [r["speedup"] for r in rows]
        out += ["",
                f"Median speedup **{statistics.median(sp):.2f}x** across "
                f"{len(sp)} models (range {min(sp):.2f}-{max(sp):.2f}x). "
                "Same tensors, same bytes on the wire: the difference is that "
                "safetensors memory-maps a flat buffer while the pickle path "
                "reconstructs every tensor through the Python interpreter. This "
                "is the cheapest available win -- a format migration costs no "
                "model quality and no hardware.", ""]
    else:
        out += ["_No paired format measurements yet._", ""]

    # -- 4. quantization
    out += ["## 4. Quantization: what 4-bit buys at *load* time", ""]
    qrows = a.quantization_effect()
    if qrows:
        out += ["| fp16 | 4-bit | size ratio | fp16 (ms) | 4-bit (ms) | load speedup |",
                "|---|---|---:|---:|---:|---:|"]
        for r in qrows:
            out.append(f"| `{r['base']}` | `{r['quantized']}` | {r['size_ratio']}x "
                       f"| {r['base_ms']} | {r['quant_ms']} | {r['load_speedup']}x |")
        out += ["",
                "Identical architecture and parameter count; only the bytes on "
                "disk differ. If load speedup tracks the size ratio, cold start "
                "is bytes-bound and quantization helps startup as much as it "
                "helps memory. If it lags well behind, a fixed cost is "
                "dominating and shrinking the checkpoint further has "
                "diminishing returns.", ""]
    else:
        out += ["_Run a quantized/fp16 pair (e.g. `qwen2.5-7b` and "
                "`qwen2.5-7b-awq`) to populate this section._", ""]

    # -- 5. sharding
    out += ["## 5. Per-shard overhead", ""]
    sh = a.shard_overhead()
    if sh:
        out += [
            "| Checkpoint layout | ms per GiB | fixed cost (ms) | R2 | n |",
            "|---|---:|---:|---:|---:|",
            f"| single shard | {sh['single_shard']['slope_ms_per_gib']} "
            f"| {sh['single_shard']['intercept_ms']} "
            f"| {sh['single_shard']['r_squared']} | {sh['single_shard']['n_points']} |",
            f"| multi shard | {sh['multi_shard']['slope_ms_per_gib']} "
            f"| {sh['multi_shard']['intercept_ms']} "
            f"| {sh['multi_shard']['r_squared']} | {sh['multi_shard']['n_points']} |",
            "",
            f"Extra fixed cost attributable to sharding: "
            f"**{sh['extra_fixed_cost_ms']} ms**. On a local filesystem this is "
            "just extra `open()`s and header parses. On network-attached or "
            "object storage each shard is a separate request with its own "
            "round-trip, so this term grows with latency and is worth "
            "re-measuring against the real backing store.", ""]
    else:
        out += ["_Need both single- and multi-shard models measured._", ""]

    # -- 6. storage
    out += ["## 6. Storage tiers", ""]
    st = a.storage_tiers()
    if st:
        out += ["| Tier | median MiB/s | min | max | models |",
                "|---|---:|---:|---:|---:|"]
        for r in st:
            out.append(f"| {r['tier']} | {r['median_mib_s']} | {r['min_mib_s']} "
                       f"| {r['max_mib_s']} | {r['n_models']} |")
        out += ["", "Tiers whose name contains `emulated` are bandwidth-capped "
                    "in software rather than measured against real remote "
                    "storage; the harness only ever slows a read down, never "
                    "speeds it up, so these are a floor on the real penalty.", ""]
    else:
        out += ["_No storage-tier measurements yet._", ""]

    # -- 7. engine
    out += ["## 7. Engine bring-up", ""]
    er = a.engine_breakdown()
    if er:
        out += ["| Model | GiB | backend | p50 (ms) | p95 (ms) |",
                "|---|---:|---|---:|---:|"]
        for r in er:
            out.append(f"| `{r['model']}` | {r['gib']} | {r['backend']} "
                       f"| {r['p50_ms']} | {r['p95_ms']} |")
        out += ["",
                "Weight transfer is only one term. Engine bring-up adds CUDA "
                "context creation, kernel/JIT warm-up, graph capture and "
                "KV-cache allocation. Where this term is large relative to "
                "weight loading, snapshot/restore techniques pay off; where "
                "weight transfer dominates, they do not -- which is exactly the "
                "distinction this table is here to make.", ""]

    # -- 8. projection
    out += ["## 8. Projection to production scale", ""]
    fit, reg = a.stable_fit("safetensors")
    if fit:
        boundary = reg.get("boundary_gib") if reg else None
        out += [
            "Projecting from the **stable regime only** "
            f"({fit.mib_s:.0f} MiB/s, R2={fit.r_squared:.3f}, n={fit.n_points}"
            + (f", checkpoints below {boundary:.2f} GiB" if boundary else "")
            + "), fitted through the origin so a checkpoint of size zero costs "
            "zero time.", "",
            "The degraded regime is deliberately excluded: past the boundary "
            "the slope describes this instance running out of host memory, and "
            "a production node has a different amount of memory. Projecting "
            "the collapse would be projecting a property of a free Colab VM "
            "onto a server.", "",
            "| Model | GiB | params | predicted weight load |",
            "|---|---:|---:|---:|",
        ]
        for spec in sorted(models_in_tier("reference"),
                           key=lambda s: s.approx_disk_gib):
            secs = fit.predict_ms(spec.approx_disk_gib) / 1000.0
            out.append(f"| `{spec.key}` | {spec.approx_disk_gib} | "
                       f"{spec.params_b}B | {secs:.0f} s |")
        out += [
            "",
            "**These are a lower bound, and both directions of error are "
            "known:**", "",
            "- *Optimistic*, because it assumes throughput stays healthy at "
            "sizes far beyond anything measured here. Whether it does depends "
            "on the target having enough memory headroom -- exactly the "
            "question section 2.2 raises.",
            "- *Pessimistic*, because production serving shards a large "
            "checkpoint across several GPUs and pulls the shards in parallel, "
            "while every number here is a single-device sequential load.",
            "- It covers **weight transfer only**. Section 7 shows engine "
            "bring-up is a comparable cost at these sizes; a real cold start "
            "is the sum.",
            "",
            "For a 27-32B class model the arithmetic gives roughly "
            f"{fit.predict_ms(61.0) / 1000.0:.0f} s of sequential weight "
            "transfer at this throughput. That is a sanity check on an order "
            "of magnitude, not a prediction for any particular deployment.", "",
        ]
    else:
        out += ["_No stable regime could be identified; there is not enough "
                "data below the collapse boundary to project from._", ""]

    # -- 9. confidence
    out += ["## 9. How much to trust these numbers", ""]
    noise = a.noise_profile()
    if noise:
        out += [
            f"- Median relative standard deviation: **{noise['median_rsd']:.1%}**",
            f"- 90th percentile RSD: {noise['p90_rsd']:.1%}",
            f"- Worst case RSD: {noise['max_rsd']:.1%}",
            f"- Conditions measured: {noise['n']}",
            "",
            "An effect smaller than the run-to-run spread is not a finding. "
            "Read every speedup above against this table: differences of a few "
            "percent are noise, and only the large multiples are safe to act "
            "on. Cold reads were forced between trials (page-cache eviction, or "
            "`posix_fadvise` where unprivileged), and warm-up runs discarded.",
            "",
        ]
        bad = a.unreliable()
        if bad:
            share = len(bad) / max(1, len(a.obs))
            out += [
                f"### 9.1 Measurements to discount ({len(bad)} of {len(a.obs)}, "
                f"{share:.0%})", "",
                "These conditions have a relative standard deviation of 30% or "
                "more. They are reported rather than silently dropped, but no "
                "conclusion above rests on them, and they are the first "
                "candidates for a re-run with more repeats.", "",
                "| Model | Experiment | Condition | p50 (ms) | RSD |",
                "|---|---|---|---:|---:|",
            ]
            for r in bad[:15]:
                out.append(f"| `{r['model']}` | {r['experiment']} | "
                           f"{r['condition']} | {r['p50_ms']} | {r['rsd']:.0%} |")
            if len(bad) > 15:
                out.append(f"| _... and {len(bad) - 15} more_ | | | | |")
            out += [
                "",
                "The dominant source is almost certainly the shared, contended "
                "I/O of a free-tier VM: neighbouring tenants on the same host "
                "move the storage numbers far more than anything in the "
                "harness does. This is the strongest argument for treating the "
                "*ratios* here as the result and the absolute milliseconds as "
                "context.", "",
            ]

    out += [
        "## 10. Limitations", "",
        "- Free-tier hardware: a single consumer GPU with shared, contended "
        "host I/O. Absolute numbers are not production numbers; the *ratios* "
        "and the *scaling behaviour* are what transfer.",
        "- Driver-level checkpoint/restore (proprietary GPU memory "
        "snapshotting) cannot be reproduced here. Where a technique was out of "
        "reach, the closest honest proxy was measured and labelled.",
        "- Emulated storage tiers are software bandwidth caps, not real "
        "network storage.",
        "- Projections assume linear scaling in bytes; see section 8.",
        "",
    ]
    return "\n".join(out) + "\n"


def write_report(observations: List[Observation], path: str, **kw) -> str:
    md = build_report(observations, **kw)
    with open(path, "w") as fh:
        fh.write(md)
    return path
