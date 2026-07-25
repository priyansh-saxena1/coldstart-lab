"""Storage-tier experiment.

This is the most direct analogue to a real platform's cold-start pain: the same
checkpoint served from a network-attached shared filesystem versus a local NVMe
cache. On Colab the two tiers map naturally to a mounted Google Drive (network,
high latency) and the local runtime disk (fast, ephemeral).

Where a genuinely slower tier isn't available, an optional token-bucket
throttle emulates a bandwidth ceiling so the *shape* of the result (local beats
remote, and by how much) can still be demonstrated and clearly labelled as
emulated in the report.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from coldstart_lab.environment import drop_page_cache
from coldstart_lab.experiments.base import Experiment
from coldstart_lab.loaders.safetensors_loader import SafetensorsLoader


@dataclass
class StorageTier:
    name: str
    root: str                        # directory the checkpoint is copied under
    emulated_mib_s: Optional[float] = None  # None => real device speed

    @property
    def emulated(self) -> bool:
        return self.emulated_mib_s is not None


class StorageExperiment(Experiment):
    name = "storage_tier"

    def __init__(
        self,
        model_dir: str,
        tiers: List[StorageTier],
        device: str = "cpu",
        repeats: int = 3,
        warmup: int = 1,
    ) -> None:
        super().__init__(repeats=repeats, warmup=warmup)
        if not tiers:
            raise ValueError("At least one storage tier is required.")
        self.model_dir = model_dir
        self.tiers = tiers
        self.device = device
        self._staged: Dict[str, str] = {}

    def _stage(self, tier: StorageTier) -> str:
        """Copy the checkpoint under the tier root once, return its path."""
        if tier.name in self._staged:
            return self._staged[tier.name]
        dest = os.path.join(tier.root, f"coldstart_stage_{tier.name}")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(self.model_dir, dest, ignore=shutil.ignore_patterns("_derived_bin"))
        self._staged[tier.name] = dest
        return dest

    def conditions(self) -> Dict[str, Callable[[], Dict[str, float]]]:
        loader = SafetensorsLoader(use_mmap=True)
        conds: Dict[str, Callable[[], Dict[str, float]]] = {}

        def make(tier: StorageTier):
            def _run() -> Dict[str, float]:
                staged = self._stage(tier)
                drop_page_cache(staged)
                res = loader.load(staged, device=self.device)
                metrics = {**res.phases_ms, "total_ms": res.total_ms,
                           "throughput_mib_s": res.throughput_mib_s}
                if tier.emulated:
                    metrics = _apply_emulated_ceiling(
                        metrics, res.bytes_read, tier.emulated_mib_s
                    )
                return metrics

            return _run

        for tier in self.tiers:
            conds[tier.name] = make(tier)
        return conds

    def meta(self) -> Dict[str, object]:
        m = super().meta()
        m.update(
            {
                "model_dir": self.model_dir,
                "device": self.device,
                "tiers": [
                    {"name": t.name, "emulated_mib_s": t.emulated_mib_s} for t in self.tiers
                ],
            }
        )
        return m


def _apply_emulated_ceiling(
    metrics: Dict[str, float], nbytes: int, ceiling_mib_s: float
) -> Dict[str, float]:
    """Rewrite the read phase to reflect a synthetic bandwidth ceiling.

    We only ever *slow down* the read phase to the emulated rate; if the real
    device was already slower we leave it untouched, so emulation can't
    manufacture an artificially fast result.
    """

    emulated_read_ms = (nbytes / 2**20) / ceiling_mib_s * 1_000.0
    real_read_ms = metrics.get("open_and_read", 0.0)
    adjusted_read = max(real_read_ms, emulated_read_ms)
    delta = adjusted_read - real_read_ms

    out = dict(metrics)
    out["open_and_read"] = adjusted_read
    out["total_ms"] = metrics.get("total_ms", 0.0) + delta
    out["emulated"] = 1.0
    return out
