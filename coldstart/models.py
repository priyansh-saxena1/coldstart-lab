"""Model catalog for the cold-start sweep.

Picking models here is not arbitrary. Three things drive the list:

  * We need a clean *size-scaling curve* — the same architecture family at
    several sizes — so we can show load time as a function of parameter bytes
    and extrapolate to production sizes. Pythia and Qwen2.5 both give us that,
    ungated, in sizes that fit a free T4 (16 GB).
  * We need a couple of tiny models that run on a plain CPU so the harness is
    testable anywhere (CI, a laptop, this sandbox) without a GPU.
  * We need at least one 7B-ish model to make the storage/format/quant deltas
    big enough to be unambiguous — the effects are real at 0.5B but they're
    tens of milliseconds; at 7B they're seconds and nobody can argue with them.

`approx_gb` is the on-disk fp16 weight footprint, rounded. It's what actually
moves across the storage tier, so it's the number the extrapolation uses.
Gated models (Llama, Gemma, Mistral) need a HF token + license acceptance; they
are included because they're what people actually deploy, but the sweep runs
fine without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    id: str
    params: str          # human label, e.g. "0.5B"
    approx_gb: float     # fp16 on-disk footprint
    tier: str            # smoke | curve | large | gated
    gated: bool = False
    note: str = ""


CATALOG: List[ModelSpec] = [
    # --- tiny, CPU-runnable: used for the smoke test and for CI ---
    ModelSpec("sshleifer/tiny-gpt2", "0.1M", 0.002, "smoke", note="test fixture, not a real model"),
    ModelSpec("EleutherAI/pythia-70m", "70M", 0.15, "smoke"),
    ModelSpec("HuggingFaceTB/SmolLM2-135M-Instruct", "135M", 0.27, "smoke"),

    # --- Pythia size-scaling curve (ungated, one arch family) ---
    ModelSpec("EleutherAI/pythia-160m", "160M", 0.32, "curve"),
    ModelSpec("EleutherAI/pythia-410m", "410M", 0.82, "curve"),
    ModelSpec("EleutherAI/pythia-1b", "1B", 2.0, "curve"),
    ModelSpec("EleutherAI/pythia-1.4b", "1.4B", 2.8, "curve"),

    # --- Qwen2.5 curve: modern arch, the 7B is our "make the deltas obvious" model ---
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "0.5B", 1.0, "curve"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "1.5B", 3.1, "curve"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "3B", 6.2, "curve"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "7B", 15.2, "large",
              note="fp16 barely fits a T4; use it for the quant comparison"),

    # --- SmolLM2 mid, and TinyLlama as a sanity cross-check on a third arch ---
    ModelSpec("HuggingFaceTB/SmolLM2-360M-Instruct", "360M", 0.72, "curve"),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "1.7B", 3.4, "curve"),
    ModelSpec("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B", 2.2, "curve"),

    # --- gated but standard production models; opt-in via HF token ---
    ModelSpec("meta-llama/Llama-3.2-1B-Instruct", "1B", 2.5, "gated", gated=True),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "3B", 6.4, "gated", gated=True),
    ModelSpec("google/gemma-2-2b-it", "2B", 5.2, "gated", gated=True,
              note="gemma-2 keeps weights in bf16/fp32 mix; footprint runs high"),
    ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", "7B", 14.5, "gated", gated=True),
]


def by_tier(tier: str, include_gated: bool = False) -> List[ModelSpec]:
    return [m for m in CATALOG if m.tier == tier and (include_gated or not m.gated)]


def find(model_id: str) -> ModelSpec:
    for m in CATALOG:
        if m.id == model_id:
            return m
    raise KeyError(model_id)


# default sweep for a Colab T4 session that has no gated-model access.
# leaves out Qwen 7B fp16 from the plain curve because it's tight on 16 GB with
# everything else resident — it gets its own quant experiment instead.
DEFAULT_T4_SWEEP = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
