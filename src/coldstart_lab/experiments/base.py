"""Experiment base class and statistics.

Every experiment produces a list of *trials*. A single cold-start measurement
is noisy -- background I/O, thermal throttling, allocator state -- so we repeat
each condition N times, drop the first run as a warm-up when asked, and report
percentiles rather than a single number. Reporting P50 and P95 (not just the
mean) is deliberate: tail latency is what a scale-to-zero platform's SLA is
actually written against.
"""

from __future__ import annotations

import abc
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Trial:
    condition: str
    run_index: int
    metrics: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    trials: List[Trial] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self, metric: str = "total_ms") -> Dict[str, Dict[str, float]]:
        """Percentile summary of ``metric`` grouped by condition."""

        grouped: Dict[str, List[float]] = {}
        for t in self.trials:
            if metric in t.metrics:
                grouped.setdefault(t.condition, []).append(t.metrics[metric])

        out: Dict[str, Dict[str, float]] = {}
        for cond, values in grouped.items():
            out[cond] = _describe(values)
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meta": self.meta,
            "trials": [
                {"condition": t.condition, "run_index": t.run_index, **t.metrics, **t.meta}
                for t in self.trials
            ],
            "summary": self.summary(),
        }


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _describe(values: List[float]) -> Dict[str, float]:
    s = sorted(values)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 3),
        "p50": round(_percentile(s, 0.50), 3),
        "p95": round(_percentile(s, 0.95), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "stdev": round(statistics.pstdev(s), 3) if len(s) > 1 else 0.0,
    }


class Experiment(abc.ABC):
    """Base class for repeatable, condition-based experiments."""

    name: str = "experiment"

    def __init__(self, repeats: int = 3, warmup: int = 0) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.repeats = repeats
        self.warmup = warmup

    @abc.abstractmethod
    def conditions(self) -> Dict[str, Callable[[], Dict[str, float]]]:
        """Map condition name -> a zero-arg callable returning that run's metrics."""

    def run(self) -> ExperimentResult:
        result = ExperimentResult(name=self.name, meta=self.meta())
        for cond_name, fn in self.conditions().items():
            total = self.warmup + self.repeats
            for i in range(total):
                metrics = fn()
                if i < self.warmup:
                    continue  # discard warm-up runs
                result.trials.append(
                    Trial(condition=cond_name, run_index=i - self.warmup, metrics=metrics)
                )
        return result

    def meta(self) -> Dict[str, Any]:
        return {"repeats": self.repeats, "warmup": self.warmup}
