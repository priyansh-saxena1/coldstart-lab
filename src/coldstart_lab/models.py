"""The benchmark model registry.

Cold-start latency scales with the number of bytes that have to move from
storage to the accelerator, so the registry is organised by *weight footprint*
rather than by capability. Each tier answers a different question:

  * ``ci``        -- randomly-initialised, few MB. Runs in the test suite on a
                     CPU in seconds; never downloads a real checkpoint of size.
  * ``micro``     -- under 1 GiB. Real models that load on a free CPU runtime.
  * ``small``     -- up to 7.5 GiB. Fits a T4 (15 GiB) with room for the storage
                     experiment's staged copies. This is the core of the study.
  * ``medium``    -- 7.5-24 GiB. Needs an L4/A100, or a quantized variant.
  * ``reference`` -- above 24 GiB. **Never downloaded.** Metadata only, so
                     measured throughput can be projected onto production-scale
                     checkpoints.

EVERY ENTRY IS VERIFIED, NOT ESTIMATED
--------------------------------------
`approx_disk_gib`, `n_shards` and the absence of gating were read from the
Hugging Face API (see ``scripts/verify_registry.py``), not recalled from
memory. Sizes are the sum of the ``.safetensors`` shards a serving stack would
actually pull.

Two repos (Mistral) ship a ``consolidated.safetensors`` copy *alongside* the
shards -- the same tensors twice. A naive ``snapshot_download`` fetches both and
doubles the pull for no benefit, which is itself a cold-start finding; the size
here counts the shards only, and those repos are tagged ``dup-representation``.

No gated repos are included. A 401 is not a transient error, and because tasks
are ordered longest-checkpoint-first, a large gated model would be re-claimed
ahead of real work on every pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    tier: str
    params_b: float                 # parameter count in billions
    approx_disk_gib: float          # measured .safetensors footprint
    native_format: str
    family: str = ""                # architecture family, for grouping results
    n_shards: int = 1               # per-file overhead scales with this
    downloadable: bool = True       # False for reference-only models
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def bytes_per_param(self) -> float:
        """Effective bytes per parameter on disk.

        ~2.0 means fp16/bf16, ~4.0 fp32, ~0.7 a 4-bit quant. A cheap sanity
        check on a checkpoint, and the number that tells you how much of a
        cold start is dtype rather than model size.
        """
        if not self.params_b:
            return 0.0
        return (self.approx_disk_gib * 2**30) / (self.params_b * 1e9)


_CI = [
    ModelSpec(
        key="tiny-llama-random",
        repo_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        tier="ci",
        params_b=0.0,
        approx_disk_gib=0.004,
        native_format="safetensors",
        family="test",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
]

_MICRO = [
    ModelSpec(
        key="smollm2-135m",
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        tier="micro",
        params_b=0.135,
        approx_disk_gib=0.251,
        native_format="safetensors",
        family="smollm2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
    ModelSpec(
        key="pythia-160m",
        repo_id="EleutherAI/pythia-160m",
        tier="micro",
        params_b=0.16,
        approx_disk_gib=0.349,
        native_format="safetensors",
        family="pythia",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
    ModelSpec(
        key="gpt2",
        repo_id="openai-community/gpt2",
        tier="micro",
        params_b=0.124,
        approx_disk_gib=0.51,
        native_format="safetensors",
        family="gpt2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
    ModelSpec(
        key="smollm2-360m",
        repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        tier="micro",
        params_b=0.362,
        approx_disk_gib=0.674,
        native_format="safetensors",
        family="smollm2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
    ModelSpec(
        key="pythia-410m",
        repo_id="EleutherAI/pythia-410m",
        tier="micro",
        params_b=0.41,
        approx_disk_gib=0.849,
        native_format="safetensors",
        family="pythia",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
    ModelSpec(
        key="qwen2.5-0.5b",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        tier="micro",
        params_b=0.49,
        approx_disk_gib=0.92,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'cpu'],
    ),
]

_SMALL = [
    ModelSpec(
        key="bloomz-560m",
        repo_id="bigscience/bloomz-560m",
        tier="small",
        params_b=0.56,
        approx_disk_gib=1.042,
        native_format="safetensors",
        family="bloom",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-0.6b",
        repo_id="Qwen/Qwen3-0.6B",
        tier="small",
        params_b=0.6,
        approx_disk_gib=1.4,
        native_format="safetensors",
        family="qwen3",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="gpt2-medium",
        repo_id="openai-community/gpt2-medium",
        tier="small",
        params_b=0.355,
        approx_disk_gib=1.416,
        native_format="safetensors",
        family="gpt2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="tinyllama-1.1b-v1.0",
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tier="small",
        params_b=1.1,
        approx_disk_gib=2.049,
        native_format="safetensors",
        family="tinyllama",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="deepseek-coder-1.3b",
        repo_id="deepseek-ai/deepseek-coder-1.3b-instruct",
        tier="small",
        params_b=1.35,
        approx_disk_gib=2.508,
        native_format="safetensors",
        family="deepseek",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="pythia-1.4b",
        repo_id="EleutherAI/pythia-1.4b",
        tier="small",
        params_b=1.4,
        approx_disk_gib=2.729,
        native_format="safetensors",
        family="pythia",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        tier="small",
        params_b=1.54,
        approx_disk_gib=2.875,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-coder-1.5b",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tier="small",
        params_b=1.54,
        approx_disk_gib=2.875,
        native_format="safetensors",
        family="qwen2.5-coder",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="gpt2-large",
        repo_id="openai-community/gpt2-large",
        tier="small",
        params_b=0.774,
        approx_disk_gib=3.024,
        native_format="safetensors",
        family="gpt2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="falcon3-1b",
        repo_id="tiiuae/Falcon3-1B-Instruct",
        tier="small",
        params_b=1.0,
        approx_disk_gib=3.11,
        native_format="safetensors",
        family="falcon3",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="smollm2-1.7b",
        repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        tier="small",
        params_b=1.71,
        approx_disk_gib=3.188,
        native_format="safetensors",
        family="smollm2",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="bloom-1b7",
        repo_id="bigscience/bloom-1b7",
        tier="small",
        params_b=1.7,
        approx_disk_gib=3.208,
        native_format="safetensors",
        family="bloom",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="deepseek-r1-distill-qwen-1.5b",
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tier="small",
        params_b=1.78,
        approx_disk_gib=3.31,
        native_format="safetensors",
        family="deepseek-r1",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-1.7b",
        repo_id="Qwen/Qwen3-1.7B",
        tier="small",
        params_b=1.7,
        approx_disk_gib=3.784,
        native_format="safetensors",
        family="qwen3",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="granite-3.1-2b",
        repo_id="ibm-granite/granite-3.1-2b-instruct",
        tier="small",
        params_b=2.53,
        approx_disk_gib=4.719,
        native_format="safetensors",
        family="granite",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="phi-2",
        repo_id="microsoft/phi-2",
        tier="small",
        params_b=2.78,
        approx_disk_gib=5.178,
        native_format="safetensors",
        family="phi",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-7b-awq",
        repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
        tier="small",
        params_b=7.62,
        approx_disk_gib=5.188,
        native_format="safetensors",
        family="qwen2.5-awq",
        n_shards=2,
        downloadable=True,
        notes="2 shards; 4-bit AWQ; load-time A/B partner for the fp16 sibling",
        tags=['verified', 'quantized', 'awq', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-7b-gptq-int4",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        tier="small",
        params_b=7.62,
        approx_disk_gib=5.192,
        native_format="safetensors",
        family="qwen2.5-gptq",
        n_shards=2,
        downloadable=True,
        notes="2 shards; 4-bit GPTQ; load-time A/B partner for the fp16 sibling",
        tags=['verified', 'quantized', 'gptq', 'gpu'],
    ),
    ModelSpec(
        key="stablelm-zephyr-3b",
        repo_id="stabilityai/stablelm-zephyr-3b",
        tier="small",
        params_b=2.8,
        approx_disk_gib=5.207,
        native_format="safetensors",
        family="stablelm",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="pythia-2.8b",
        repo_id="EleutherAI/pythia-2.8b",
        tier="small",
        params_b=2.8,
        approx_disk_gib=5.294,
        native_format="safetensors",
        family="pythia",
        n_shards=1,
        downloadable=True,
        notes="",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="smollm3-3b",
        repo_id="HuggingFaceTB/SmolLM3-3B",
        tier="small",
        params_b=3.08,
        approx_disk_gib=5.728,
        native_format="safetensors",
        family="smollm3",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-3b",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        tier="small",
        params_b=3.09,
        approx_disk_gib=5.748,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="stablelm-2-1-6b",
        repo_id="stabilityai/stablelm-2-1_6b-chat",
        tier="small",
        params_b=1.64,
        approx_disk_gib=6.126,
        native_format="safetensors",
        family="stablelm",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="phi-3-mini-4k",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        tier="small",
        params_b=3.82,
        approx_disk_gib=7.117,
        native_format="safetensors",
        family="phi",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="phi-3.5-mini",
        repo_id="microsoft/Phi-3.5-mini-instruct",
        tier="small",
        params_b=3.82,
        approx_disk_gib=7.117,
        native_format="safetensors",
        family="phi",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-4b",
        repo_id="Qwen/Qwen3-4B",
        tier="small",
        params_b=4.0,
        approx_disk_gib=7.492,
        native_format="safetensors",
        family="qwen3",
        n_shards=3,
        downloadable=True,
        notes="3 shards",
        tags=['verified', 'gpu'],
    ),
]

_MEDIUM = [
    ModelSpec(
        key="qwen2.5-14b-awq",
        repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        tier="medium",
        params_b=14.8,
        approx_disk_gib=9.295,
        native_format="safetensors",
        family="qwen2.5-awq",
        n_shards=3,
        downloadable=True,
        notes="3 shards; 4-bit AWQ; load-time A/B partner for the fp16 sibling",
        tags=['verified', 'quantized', 'awq', 'gpu'],
    ),
    ModelSpec(
        key="yi-1.5-6b",
        repo_id="01-ai/Yi-1.5-6B-Chat",
        tier="medium",
        params_b=6.06,
        approx_disk_gib=11.29,
        native_format="safetensors",
        family="yi",
        n_shards=3,
        downloadable=True,
        notes="3 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="deepseek-coder-6.7b",
        repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        tier="medium",
        params_b=6.74,
        approx_disk_gib=12.555,
        native_format="safetensors",
        family="deepseek",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="falcon-7b",
        repo_id="tiiuae/falcon-7b-instruct",
        tier="medium",
        params_b=7.0,
        approx_disk_gib=13.443,
        native_format="safetensors",
        family="falcon",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="openhermes-2.5-mistral-7b",
        repo_id="teknium/OpenHermes-2.5-Mistral-7B",
        tier="medium",
        params_b=7.24,
        approx_disk_gib=13.489,
        native_format="safetensors",
        family="hermes",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="zephyr-7b-beta",
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        tier="medium",
        params_b=7.24,
        approx_disk_gib=13.489,
        native_format="safetensors",
        family="zephyr",
        n_shards=8,
        downloadable=True,
        notes="8 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="mistral-7b-v0.3",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        tier="medium",
        params_b=7.25,
        approx_disk_gib=13.501,
        native_format="safetensors",
        family="mistral",
        n_shards=3,
        downloadable=True,
        notes="ships a consolidated copy alongside shards; size counts shards only; 3 shards",
        tags=['verified', 'gpu', 'dup-representation'],
    ),
    ModelSpec(
        key="olmo-2-1124-7b",
        repo_id="allenai/OLMo-2-1124-7B-Instruct",
        tier="medium",
        params_b=7.3,
        approx_disk_gib=13.595,
        native_format="safetensors",
        family="olmo2",
        n_shards=3,
        downloadable=True,
        notes="3 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="falcon3-7b",
        repo_id="tiiuae/Falcon3-7B-Instruct",
        tier="medium",
        params_b=7.0,
        approx_disk_gib=13.887,
        native_format="safetensors",
        family="falcon3",
        n_shards=4,
        downloadable=True,
        notes="4 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="deepseek-r1-distill-qwen-7b",
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tier="medium",
        params_b=7.62,
        approx_disk_gib=14.185,
        native_format="safetensors",
        family="deepseek-r1",
        n_shards=2,
        downloadable=True,
        notes="2 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-7b",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        tier="medium",
        params_b=7.62,
        approx_disk_gib=14.185,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=4,
        downloadable=True,
        notes="4 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        tier="medium",
        params_b=7.62,
        approx_disk_gib=14.185,
        native_format="safetensors",
        family="qwen2.5-coder",
        n_shards=4,
        downloadable=True,
        notes="4 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="internlm2-5-7b",
        repo_id="internlm/internlm2_5-7b-chat",
        tier="medium",
        params_b=7.74,
        approx_disk_gib=14.413,
        native_format="safetensors",
        family="internlm",
        n_shards=8,
        downloadable=True,
        notes="8 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="granite-3.1-8b",
        repo_id="ibm-granite/granite-3.1-8b-instruct",
        tier="medium",
        params_b=8.17,
        approx_disk_gib=15.219,
        native_format="safetensors",
        family="granite",
        n_shards=4,
        downloadable=True,
        notes="4 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        tier="medium",
        params_b=8.2,
        approx_disk_gib=15.256,
        native_format="safetensors",
        family="qwen3",
        n_shards=5,
        downloadable=True,
        notes="5 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="yi-1.5-9b",
        repo_id="01-ai/Yi-1.5-9B-Chat",
        tier="medium",
        params_b=8.83,
        approx_disk_gib=16.446,
        native_format="safetensors",
        family="yi",
        n_shards=4,
        downloadable=True,
        notes="4 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="glm-4-9b",
        repo_id="THUDM/glm-4-9b-chat",
        tier="medium",
        params_b=9.4,
        approx_disk_gib=17.509,
        native_format="safetensors",
        family="glm4",
        n_shards=10,
        downloadable=True,
        notes="10 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-32b-awq",
        repo_id="Qwen/Qwen2.5-32B-Instruct-AWQ",
        tier="medium",
        params_b=32.5,
        approx_disk_gib=18.002,
        native_format="safetensors",
        family="qwen2.5-awq",
        n_shards=5,
        downloadable=True,
        notes="5 shards; 4-bit AWQ; load-time A/B partner for the fp16 sibling",
        tags=['verified', 'quantized', 'awq', 'gpu'],
    ),
    ModelSpec(
        key="solar-10.7b-v1.0",
        repo_id="upstage/SOLAR-10.7B-Instruct-v1.0",
        tier="medium",
        params_b=10.7,
        approx_disk_gib=19.989,
        native_format="safetensors",
        family="solar",
        n_shards=5,
        downloadable=True,
        notes="5 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="mistral-nemo-2407",
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        tier="medium",
        params_b=12.2,
        approx_disk_gib=22.813,
        native_format="safetensors",
        family="mistral",
        n_shards=5,
        downloadable=True,
        notes="ships a consolidated copy alongside shards; size counts shards only; 5 shards",
        tags=['verified', 'gpu', 'dup-representation'],
    ),
]

_REFERENCE = [
    ModelSpec(
        key="phi-3-medium-4k",
        repo_id="microsoft/Phi-3-medium-4k-instruct",
        tier="reference",
        params_b=14.0,
        approx_disk_gib=26.003,
        native_format="safetensors",
        family="phi",
        n_shards=6,
        downloadable=False,
        notes="6 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="phi-4",
        repo_id="microsoft/phi-4",
        tier="reference",
        params_b=14.7,
        approx_disk_gib=27.305,
        native_format="safetensors",
        family="phi",
        n_shards=6,
        downloadable=False,
        notes="6 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-14b",
        repo_id="Qwen/Qwen3-14B",
        tier="reference",
        params_b=14.8,
        approx_disk_gib=27.508,
        native_format="safetensors",
        family="qwen3",
        n_shards=8,
        downloadable=False,
        notes="8 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-14b",
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        tier="reference",
        params_b=14.8,
        approx_disk_gib=27.511,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=8,
        downloadable=False,
        notes="8 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-30b-a3b",
        repo_id="Qwen/Qwen3-30B-A3B",
        tier="reference",
        params_b=30.5,
        approx_disk_gib=56.873,
        native_format="safetensors",
        family="qwen3-moe",
        n_shards=16,
        downloadable=False,
        notes="16 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen3-32b",
        repo_id="Qwen/Qwen3-32B",
        tier="reference",
        params_b=32.8,
        approx_disk_gib=61.024,
        native_format="safetensors",
        family="qwen3",
        n_shards=17,
        downloadable=False,
        notes="17 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-32b",
        repo_id="Qwen/Qwen2.5-32B-Instruct",
        tier="reference",
        params_b=32.5,
        approx_disk_gib=61.028,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=17,
        downloadable=False,
        notes="17 shards",
        tags=['verified', 'gpu'],
    ),
    ModelSpec(
        key="qwen2.5-72b",
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        tier="reference",
        params_b=72.7,
        approx_disk_gib=135.426,
        native_format="safetensors",
        family="qwen2.5",
        n_shards=37,
        downloadable=False,
        notes="37 shards",
        tags=['verified', 'gpu'],
    ),
]

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (_CI + _MICRO + _SMALL + _MEDIUM + _REFERENCE)
}


# Stable aliases so scripts and docs survive registry key changes.
ALIASES: Dict[str, str] = {
    "tiny-random-llamaforcausallm": "tiny-llama-random",
    "qwen2.5-7b-gptq": "qwen2.5-7b-gptq-int4",
}


def get_model(key: str) -> ModelSpec:
    key = ALIASES.get(key, key)
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown model {key!r}. Known keys: {sorted(MODEL_REGISTRY)}"
        ) from None


def models_in_tier(tier: str) -> List[ModelSpec]:
    return [m for m in MODEL_REGISTRY.values() if m.tier == tier]


def models_in_family(family: str) -> List[ModelSpec]:
    return [m for m in MODEL_REGISTRY.values() if m.family == family]


def downloadable_models() -> List[ModelSpec]:
    return [m for m in MODEL_REGISTRY.values() if m.downloadable]


def quantized_pairs() -> List[tuple]:
    """(fp16, quantized) pairs of the same base model.

    These are the cleanest experiment in the registry: identical architecture
    and parameter count, different bytes on disk, so any load-time difference is
    attributable to dtype alone.
    """
    pairs = []
    for q in MODEL_REGISTRY.values():
        if "quantized" not in q.tags:
            continue
        base_key = q.key.replace("-awq", "").replace("-gptq-int4", "")
        base = MODEL_REGISTRY.get(base_key)
        if base is not None:
            pairs.append((base, q))
    return pairs


def tier_summary() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tier in ("ci", "micro", "small", "medium", "reference"):
        rows = models_in_tier(tier)
        if not rows:
            continue
        out[tier] = {
            "count": len(rows),
            "total_gib": round(sum(r.approx_disk_gib for r in rows), 1),
            "min_gib": round(min(r.approx_disk_gib for r in rows), 2),
            "max_gib": round(max(r.approx_disk_gib for r in rows), 2),
            "families": sorted({r.family for r in rows if r.family}),
        }
    return out
