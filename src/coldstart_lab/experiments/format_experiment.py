"""Checkpoint-format experiment.

Question: given identical weights, how much cold-start time do you spend purely
on the serialization format? Conditions:

  * ``safetensors``        -- memory-mapped, zero-copy where possible
  * ``safetensors-nommap`` -- eager read, no mapping
  * ``pytorch_bin``        -- legacy pickle container (derived locally)

Every condition drops the page cache for the checkpoint before loading, so each
trial is a genuine cold read rather than a RAM hit.
"""

from __future__ import annotations

import os
from typing import Callable, Dict

from coldstart_lab.environment import drop_page_cache
from coldstart_lab.experiments.base import Experiment
from coldstart_lab.loaders.converters import to_pytorch_bin
from coldstart_lab.loaders.pytorch_bin_loader import PyTorchBinLoader
from coldstart_lab.loaders.safetensors_loader import SafetensorsLoader


class FormatExperiment(Experiment):
    name = "checkpoint_format"

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        include_bin: bool = True,
        bin_dir: str | None = None,
        repeats: int = 3,
        warmup: int = 1,
    ) -> None:
        super().__init__(repeats=repeats, warmup=warmup)
        self.model_dir = model_dir
        self.device = device
        self.include_bin = include_bin
        self.bin_dir = bin_dir or os.path.join(model_dir, "_derived_bin")
        self._bin_path: str | None = None

    def _ensure_bin(self) -> str:
        if self._bin_path is None:
            self._bin_path = to_pytorch_bin(self.model_dir, self.bin_dir)
        return self._bin_path

    def conditions(self) -> Dict[str, Callable[[], Dict[str, float]]]:
        conds: Dict[str, Callable[[], Dict[str, float]]] = {}

        def make_st(use_mmap: bool):
            loader = SafetensorsLoader(use_mmap=use_mmap)

            def _run() -> Dict[str, float]:
                drop_page_cache(self.model_dir)
                res = loader.load(self.model_dir, device=self.device)
                return {**res.phases_ms, "total_ms": res.total_ms,
                        "throughput_mib_s": res.throughput_mib_s}

            return _run

        conds["safetensors"] = make_st(True)
        conds["safetensors-nommap"] = make_st(False)

        if self.include_bin:
            bin_loader = PyTorchBinLoader()

            def _run_bin() -> Dict[str, float]:
                bin_path = self._ensure_bin()
                drop_page_cache(bin_path)
                res = bin_loader.load(os.path.dirname(bin_path), device=self.device)
                return {**res.phases_ms, "total_ms": res.total_ms,
                        "throughput_mib_s": res.throughput_mib_s}

            conds["pytorch_bin"] = _run_bin

        return conds

    def meta(self) -> Dict[str, object]:
        m = super().meta()
        m.update({"model_dir": self.model_dir, "device": self.device})
        return m
