"""Checkpoint format converters.

Modern checkpoints ship as safetensors, so to run a safetensors-vs-bin A/B we
need to derive an equivalent ``.bin`` from the same weights. Doing the
conversion locally (rather than downloading a separate legacy repo) guarantees
the two formats hold byte-identical tensors, so the only variable in the
experiment is the serialization format itself.
"""

from __future__ import annotations

import os
from typing import Dict


def to_pytorch_bin(safetensors_dir: str, out_dir: str) -> str:
    """Write a single ``pytorch_model.bin`` equivalent of a safetensors dir.

    Returns the path to the written ``.bin`` file.
    """

    import torch
    from safetensors import safe_open

    os.makedirs(out_dir, exist_ok=True)
    merged: Dict[str, "torch.Tensor"] = {}

    for name in sorted(os.listdir(safetensors_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(safetensors_dir, name), framework="pt", device="cpu") as h:
            for key in h.keys():
                merged[key] = h.get_tensor(key)

    if not merged:
        raise FileNotFoundError(f"No .safetensors tensors found in {safetensors_dir!r}.")

    out_path = os.path.join(out_dir, "pytorch_model.bin")
    torch.save(merged, out_path)
    return out_path
