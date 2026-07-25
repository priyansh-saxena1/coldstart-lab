"""coldstart_lab: a benchmarking harness for LLM container cold-start latency.

The package decomposes cold start into the phases that actually matter for a
serverless / scale-to-zero inference platform -- weight transfer from storage,
deserialization into host memory, and hand-off to the accelerator -- and lets
you measure each one in isolation across storage tiers, checkpoint formats and
quantization schemes.

The design goal is that every experiment runs on a free-tier CPU or a single
consumer GPU, while producing numbers that extrapolate cleanly to production
model sizes via `coldstart_lab.extrapolate`.
"""

from coldstart_lab.timing import PhaseTimer, Stopwatch
from coldstart_lab.models import MODEL_REGISTRY, ModelSpec, get_model

__version__ = "0.1.0"

__all__ = [
    "PhaseTimer",
    "Stopwatch",
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_model",
    "__version__",
]
