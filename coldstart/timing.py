"""Phase timing primitives.

The whole point of this project is that "cold start" is not one number. It's a
sum of phases (stage the weights, import the framework, load weights into RAM,
move to device, run the first forward) and different engines/configs move the
bottleneck around. So the timer here is deliberately phase-oriented: you record
named spans and the record carries the breakdown, not just a total.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# perf_counter, not time.time — we want monotonic elapsed, not wall clock.
_clock = time.perf_counter


@dataclass
class PhaseRecord:
    """One measured run, broken into phases (seconds)."""
    phases: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.phases.values())

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


class PhaseTimer:
    """Accumulates named phase durations for a single run.

    Not thread-safe and not meant to be — one timer per process, and we run each
    cold load in its own subprocess anyway.
    """

    def __init__(self) -> None:
        self._phases: Dict[str, float] = {}
        self.meta: Dict[str, object] = {}

    @contextmanager
    def phase(self, name: str):
        start = _clock()
        try:
            yield
        finally:
            dt = _clock() - start
            # if a phase name repeats (e.g. per-layer load in a loop) we sum it
            self._phases[name] = self._phases.get(name, 0.0) + dt

    def mark(self, name: str, seconds: float) -> None:
        """Record a phase measured elsewhere (e.g. inside a child process)."""
        self._phases[name] = self._phases.get(name, 0.0) + seconds

    def record(self) -> PhaseRecord:
        return PhaseRecord(phases=dict(self._phases), meta=dict(self.meta))


def summarize(records: List[PhaseRecord]) -> dict:
    """Collapse repeated runs into p50/p95/mean per phase plus total.

    We report p95 alongside the median because cold starts are what users
    actually feel on a scale-from-zero event, and the tail is the honest number
    to quote for an SLA. Mean is kept mostly for sanity checking.
    """
    if not records:
        return {}

    phase_names = set()
    for r in records:
        phase_names.update(r.phases)

    out: dict = {"n": len(records), "phases": {}}
    for name in sorted(phase_names):
        xs = sorted(r.phases.get(name, 0.0) for r in records)
        out["phases"][name] = {
            "p50": _percentile(xs, 50),
            "p95": _percentile(xs, 95),
            "mean": sum(xs) / len(xs),
        }

    totals = sorted(r.total for r in records)
    out["total"] = {
        "p50": _percentile(totals, 50),
        "p95": _percentile(totals, 95),
        "mean": sum(totals) / len(totals),
        "min": totals[0],
        "max": totals[-1],
    }
    return out


def _percentile(sorted_xs: List[float], q: float) -> float:
    # linear interpolation between closest ranks. good enough for n=3..20 runs;
    # we're not doing statistics on thousands of samples here.
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    rank = (q / 100) * (len(sorted_xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = rank - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac
