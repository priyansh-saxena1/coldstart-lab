"""Inference-engine init experiment.

Weight loading is only one term in cold start. A serving engine also pays for
CUDA context creation, kernel/JIT warm-up, CUDA-graph capture and KV-cache
allocation before it can emit a first token. This experiment breaks the engine
bring-up into phases and, on vLLM, toggles the two knobs that most affect the
tail: ``enforce_eager`` (skip CUDA-graph capture) and a warm-up generation.

It degrades gracefully:

  * With vLLM + a GPU it times the real engine phases and a first generation.
  * With only ``transformers`` it still measures from-config init, weight load
    and a first forward pass -- enough to demonstrate the methodology on CPU.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from coldstart_lab.experiments.base import Experiment
from coldstart_lab.timing import PhaseTimer


class EngineInitExperiment(Experiment):
    name = "engine_init"

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        backends: List[str] | None = None,
        repeats: int = 3,
        warmup: int = 0,
    ) -> None:
        super().__init__(repeats=repeats, warmup=warmup)
        self.model_dir = model_dir
        self.device = device
        self.backends = backends or self._auto_backends()

    def _auto_backends(self) -> List[str]:
        backends = ["transformers"]
        if _vllm_available() and self.device.startswith("cuda"):
            backends = ["vllm-eager", "vllm-graph", "transformers"]
        return backends

    def conditions(self) -> Dict[str, Callable[[], Dict[str, float]]]:
        conds: Dict[str, Callable[[], Dict[str, float]]] = {}
        for backend in self.backends:
            conds[backend] = self._make_backend(backend)
        return conds

    def _make_backend(self, backend: str) -> Callable[[], Dict[str, float]]:
        if backend == "transformers":
            return self._run_transformers
        if backend.startswith("vllm"):
            enforce_eager = backend.endswith("eager")
            return lambda: self._run_vllm(enforce_eager=enforce_eager)
        raise ValueError(f"Unknown backend {backend!r}")

    def _run_transformers(self) -> Dict[str, float]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        timer = PhaseTimer()
        with timer.phase("tokenizer_init"):
            tok = AutoTokenizer.from_pretrained(self.model_dir)

        with timer.phase("weight_load"):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, torch_dtype="auto"
            )
            model = model.to(self.device)
            model.eval()

        with timer.phase("first_forward"):
            inputs = tok("Cold start benchmark prompt.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=8, do_sample=False)
            _sync(self.device)

        # Free between trials so the next run is a genuine re-init.
        del model
        _empty_cache(self.device)
        return {**timer.phases_ms, "total_ms": timer.total_ms}

    def _run_vllm(self, enforce_eager: bool) -> Dict[str, float]:  # pragma: no cover - GPU only
        from vllm import LLM, SamplingParams

        timer = PhaseTimer()
        with timer.phase("engine_init"):
            llm = LLM(
                model=self.model_dir,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.85,
                max_model_len=2048,
            )

        with timer.phase("first_generation"):
            llm.generate(
                ["Cold start benchmark prompt."],
                SamplingParams(max_tokens=8, temperature=0.0),
            )

        del llm
        _empty_cache(self.device)
        return {**timer.phases_ms, "total_ms": timer.total_ms}

    def meta(self) -> Dict[str, object]:
        m = super().meta()
        m.update({"model_dir": self.model_dir, "device": self.device, "backends": self.backends})
        return m


def _vllm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


def _sync(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _empty_cache(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
