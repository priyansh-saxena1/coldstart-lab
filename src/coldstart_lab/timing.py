"""Timing primitives.

We deliberately use ``time.perf_counter`` (monotonic, highest resolution the
platform offers) rather than ``time.time`` so that NTP adjustments or wall-clock
skew can never contaminate a measurement. Durations are reported in
milliseconds because most cold-start phases land in the 10 ms - 60 s range and
milliseconds keep the numbers readable without scientific notation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List


class Stopwatch:
    """A restartable monotonic stopwatch.

    Not thread-safe by design -- a cold-start measurement is inherently a
    single sequential timeline, and adding locks would only add jitter.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed_ms: float = 0.0

    def start(self) -> "Stopwatch":
        if self._start is not None:
            raise RuntimeError("Stopwatch already running; call stop() first.")
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Stopwatch is not running.")
        self._elapsed_ms += (time.perf_counter() - self._start) * 1_000.0
        self._start = None
        return self._elapsed_ms

    def reset(self) -> None:
        self._start = None
        self._elapsed_ms = 0.0

    @property
    def elapsed_ms(self) -> float:
        running = 0.0
        if self._start is not None:
            running = (time.perf_counter() - self._start) * 1_000.0
        return self._elapsed_ms + running


@dataclass
class PhaseTimer:
    """Accumulates named, non-overlapping phase durations for one run.

    Usage::

        timer = PhaseTimer()
        with timer.phase("weight_read"):
            blob = read_from_disk(path)
        with timer.phase("deserialize"):
            tensors = load(blob)
        print(timer.total_ms, timer.as_dict())

    Nested phases are rejected: a cold start is a sequence of stages, and
    silently double-counting overlapping regions is a classic way to produce
    numbers that look plausible but don't sum to the wall-clock total.
    """

    phases_ms: Dict[str, float] = field(default_factory=dict)
    _active: List[str] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self._active:
            raise RuntimeError(
                f"Cannot start phase {name!r} while {self._active[-1]!r} is "
                "active; phases must not overlap."
            )
        if name in self.phases_ms:
            raise ValueError(f"Phase {name!r} already recorded for this run.")
        self._active.append(name)
        sw = Stopwatch().start()
        try:
            yield
        finally:
            self.phases_ms[name] = sw.stop()
            self._active.pop()

    @property
    def total_ms(self) -> float:
        return sum(self.phases_ms.values())

    def as_dict(self) -> Dict[str, float]:
        out = {f"{k}_ms": round(v, 3) for k, v in self.phases_ms.items()}
        out["total_ms"] = round(self.total_ms, 3)
        return out
