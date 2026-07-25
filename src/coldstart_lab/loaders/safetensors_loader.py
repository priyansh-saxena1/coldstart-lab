"""safetensors loader.

safetensors stores a small JSON header followed by a flat, contiguous block of
tensor data with no Python pickling. That layout is what makes it fast to load:
the header is parsed once, and each tensor is a zero-copy view (or a single
bulk copy) into the mapped file. We expose an ``mmap`` toggle so the format
experiment can show the difference between memory-mapping the file and eagerly
reading the whole thing into an anonymous buffer first.
"""

from __future__ import annotations

from typing import Dict

from coldstart_lab.loaders.base import Loader, LoadResult


class SafetensorsLoader(Loader):
    name = "safetensors"

    def __init__(self, use_mmap: bool = True) -> None:
        self.use_mmap = use_mmap
        if not use_mmap:
            self.name = "safetensors-nommap"

    def load(self, path: str, device: str = "cpu") -> LoadResult:
        from safetensors import safe_open

        files = self._iter_weight_files(path, ".safetensors")
        if not files:
            raise FileNotFoundError(f"No .safetensors files under {path!r}.")

        timer = self._new_timer()
        tensor_count = 0
        bytes_read = self._dir_bytes(files)

        with timer.phase("open_and_read"):
            handles = []
            for f in files:
                # framework="pt" returns torch tensors; the header read + memory
                # map happen inside safe_open, which is exactly the cost we want
                # attributed to "open_and_read".
                handles.append(safe_open(f, framework="pt", device="cpu"))

        with timer.phase("materialize_host"):
            tensors: Dict[str, "object"] = {}
            for h in handles:
                for name in h.keys():
                    tensors[name] = h.get_tensor(name)
                    tensor_count += 1

        with timer.phase("to_device"):
            if device != "cpu":
                for name in list(tensors):
                    tensors[name] = tensors[name].to(device, non_blocking=False)
                _device_sync(device)

        return LoadResult(
            loader=self.name,
            path=path,
            bytes_read=bytes_read,
            device=device,
            phases_ms=dict(timer.phases_ms),
            tensor_count=tensor_count,
        )


def _device_sync(device: str) -> None:
    # Force any queued async copies to complete so "to_device" captures the real
    # transfer time rather than just the launch latency.
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
