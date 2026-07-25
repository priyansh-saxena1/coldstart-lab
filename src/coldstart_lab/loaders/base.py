"""Loader abstraction.

A "loader" takes a checkpoint on disk and produces tensors in host (and
optionally device) memory, while recording where the time went. Splitting this
out from the experiments lets us A/B two loaders over the *identical* driver
loop, so any measured difference is attributable to the loading strategy and
not to differences in the harness around it.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Dict, List

from coldstart_lab.timing import PhaseTimer


@dataclass
class LoadResult:
    loader: str
    path: str
    bytes_read: int
    device: str
    phases_ms: Dict[str, float]
    tensor_count: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(self.phases_ms.values())

    @property
    def throughput_mib_s(self) -> float:
        """Effective end-to-end throughput in MiB/s.

        This is the single number that extrapolates: it captures the whole
        storage -> host -> (device) pipeline for this loader on this tier, and
        multiplying a production checkpoint's size by it predicts load time.
        """
        secs = self.total_ms / 1_000.0
        if secs <= 0:
            return 0.0
        return (self.bytes_read / 2**20) / secs

    def to_dict(self) -> dict:
        return {
            "loader": self.loader,
            "path": self.path,
            "bytes_read": self.bytes_read,
            "mib_read": round(self.bytes_read / 2**20, 2),
            "device": self.device,
            "tensor_count": self.tensor_count,
            "throughput_mib_s": round(self.throughput_mib_s, 2),
            **{f"{k}_ms": round(v, 3) for k, v in self.phases_ms.items()},
            "total_ms": round(self.total_ms, 3),
            "warnings": self.warnings,
        }


class Loader(abc.ABC):
    """Base class for checkpoint loaders."""

    name: str = "base"

    @abc.abstractmethod
    def load(self, path: str, device: str = "cpu") -> LoadResult:
        """Load every tensor file under ``path`` onto ``device``."""

    @staticmethod
    def _iter_weight_files(path: str, extension: str) -> List[str]:
        if os.path.isfile(path):
            return [path] if path.endswith(extension) else []
        matches = []
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                if name.endswith(extension):
                    matches.append(os.path.join(root, name))
        return matches

    @staticmethod
    def _dir_bytes(files: List[str]) -> int:
        return sum(os.path.getsize(f) for f in files)

    def _new_timer(self) -> PhaseTimer:
        return PhaseTimer()
