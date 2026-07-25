"""Legacy PyTorch ``.bin`` (pickle) loader.

The historical checkpoint format is a zip container of pickled tensors. It is
slower to load than safetensors for two structural reasons: the whole payload
is unpickled through Python, and there is no zero-copy path -- every tensor is
reconstructed and copied. Measuring it head-to-head against safetensors on the
same weights is the cleanest way to quantify the "just switch the format" win
that a platform can capture with zero model-quality cost.

We ``weights_only=True`` on load: it is both the safe choice (no arbitrary code
execution from a pickle) and representative of how modern serving stacks load
untrusted checkpoints.
"""

from __future__ import annotations

from typing import Dict

from coldstart_lab.loaders.base import Loader, LoadResult


class PyTorchBinLoader(Loader):
    name = "pytorch_bin"

    def load(self, path: str, device: str = "cpu") -> LoadResult:
        import torch

        files = self._iter_weight_files(path, ".bin")
        if not files:
            raise FileNotFoundError(
                f"No .bin files under {path!r}. Use converters.to_pytorch_bin "
                "to derive one from a safetensors checkpoint."
            )

        timer = self._new_timer()
        tensor_count = 0
        bytes_read = self._dir_bytes(files)

        with timer.phase("read_and_unpickle"):
            state_dicts = []
            for f in files:
                # map_location='cpu' keeps the unpickle cost separate from the
                # device transfer, which we time as its own phase below.
                sd = torch.load(f, map_location="cpu", weights_only=True)
                state_dicts.append(sd)

        with timer.phase("materialize_host"):
            merged: Dict[str, "object"] = {}
            for sd in state_dicts:
                for name, tensor in sd.items():
                    merged[name] = tensor
                    tensor_count += 1

        with timer.phase("to_device"):
            if device != "cpu":
                for name in list(merged):
                    merged[name] = merged[name].to(device, non_blocking=False)
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()

        return LoadResult(
            loader=self.name,
            path=path,
            bytes_read=bytes_read,
            device=device,
            phases_ms=dict(timer.phases_ms),
            tensor_count=tensor_count,
        )
