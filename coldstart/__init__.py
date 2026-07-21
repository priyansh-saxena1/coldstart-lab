"""coldstart-lab: phase-decomposed cold-start benchmarking for LLM inference.

See README for the why. Quick entry points:

    from coldstart import runner, experiments, report, models
"""

__version__ = "0.2.0"

from . import timing, models  # noqa: F401  (convenience re-exports)
