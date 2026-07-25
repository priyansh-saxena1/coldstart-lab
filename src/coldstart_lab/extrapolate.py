"""Extrapolation from measured throughput to production model sizes.

The whole point of running on a free GPU is to measure something that transfers
to hardware we can't touch. Weight-transfer cold start is, to first order,
linear in bytes moved: once you know the effective MiB/s of a
(storage tier x format) pipeline, the load time of an N-byte checkpoint is just
N / throughput. This module makes that projection explicit -- and honest about
its assumptions -- so a measured 3B-on-a-T4 number can be turned into a
predicted 32B or 70B load time with a stated error model.

Caveats deliberately surfaced in the output:
  * Linear model ignores fixed per-file overhead (amortised away at scale).
  * It assumes the production tier has the same per-byte characteristics as the
    measured tier; the report labels this as a projection, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from coldstart_lab.models import ModelSpec


@dataclass
class Projection:
    model_key: str
    approx_disk_gib: float
    throughput_mib_s: float
    predicted_load_s: float
    basis: str

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "approx_disk_gib": round(self.approx_disk_gib, 2),
            "throughput_mib_s": round(self.throughput_mib_s, 2),
            "predicted_load_s": round(self.predicted_load_s, 2),
            "basis": self.basis,
        }


def project_load_time(
    throughput_mib_s: float,
    targets: List[ModelSpec],
    basis: str,
) -> List[Projection]:
    """Predict weight-load time for each target model at a measured throughput."""

    if throughput_mib_s <= 0:
        raise ValueError("throughput_mib_s must be positive.")

    projections: List[Projection] = []
    for spec in targets:
        gib = spec.approx_disk_gib
        mib = gib * 1024.0
        predicted_s = mib / throughput_mib_s
        projections.append(
            Projection(
                model_key=spec.key,
                approx_disk_gib=gib,
                throughput_mib_s=throughput_mib_s,
                predicted_load_s=predicted_s,
                basis=basis,
            )
        )
    return projections
