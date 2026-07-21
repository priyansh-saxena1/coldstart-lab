"""The actual experiments.

Each function returns a dict of {arm_label: [PhaseRecord, ...]} so report.py can
diff the arms. They're written to degrade gracefully off-GPU: the storage and
format experiments run fine on CPU (that's how the smoke test exercises them),
while quant and the vLLM engine-init experiment need CUDA and will skip with a
note if it isn't there.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List

from .timing import PhaseRecord
from .runner import run_config
from . import storage


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def storage_tier(model_id: str, tiers: Dict[str, str], repeats: int = 3,
                 device: str = "cpu") -> Dict[str, List[PhaseRecord]]:
    """Load the same model from each storage tier.

    tiers maps a label -> a root directory on that tier, e.g.
        {"local": "/content/staging", "drive": "/content/drive/MyDrive/staging"}
    We stage the weights into each root (timing the copy as the 'pull' phase),
    then measure loads from there with the page cache dropped between runs.
    """
    out: Dict[str, List[PhaseRecord]] = {}
    for label, root in tiers.items():
        staged_dir, copy_s, nbytes = storage.stage_model(model_id, root)
        cfg = {"backend": "transformers", "model_id": model_id,
               "load_path": staged_dir, "device": device, "dtype": "float32"}
        recs = run_config(cfg, repeats=repeats, drop_cache=True, warmup=True)
        # fold the staging copy in as its own phase and stash bandwidth
        for r in recs:
            r.phases["stage_copy"] = copy_s
            r.meta["staged_bytes"] = nbytes
            r.meta["stage_MBps"] = round(nbytes / 1e6 / copy_s, 1) if copy_s else None
        out[label] = recs
    return out


def checkpoint_format(model_id_safetensors: str, model_id_bin: str,
                      repeats: int = 3, device: str = "cpu") -> Dict[str, List[PhaseRecord]]:
    """safetensors (mmap) vs legacy pytorch .bin (pickle) load time.

    Ideally this is the *same* weights in two formats. In practice you often just
    pick one repo that ships safetensors and one that ships .bin. Pass whichever
    two you've got; the label is the format, not the model, so keep them the same
    size or the comparison is muddy.
    """
    arms = {"safetensors": model_id_safetensors, "pytorch_bin": model_id_bin}
    out: Dict[str, List[PhaseRecord]] = {}
    for label, mid in arms.items():
        cfg = {"backend": "transformers", "model_id": mid,
               "device": device, "dtype": "float32"}
        out[label] = run_config(cfg, repeats=repeats, drop_cache=True, warmup=True)
    return out


def quantization(model_id: str, repeats: int = 3) -> Dict[str, List[PhaseRecord]]:
    """fp16 vs bitsandbytes 4-bit: load time and footprint. GPU only."""
    if not _has_cuda():
        return {"_skipped": _skip_note("quantization needs CUDA")}

    out: Dict[str, List[PhaseRecord]] = {}
    fp16 = {"backend": "transformers", "model_id": model_id,
            "device": "cuda:0", "dtype": "float16"}
    out["fp16"] = run_config(fp16, repeats=repeats, drop_cache=True, warmup=True)

    q4 = {"backend": "transformers", "model_id": model_id,
          "device": "cuda:0", "dtype": "float16", "quantization": "bnb-4bit"}
    out["bnb-4bit"] = run_config(q4, repeats=repeats, drop_cache=True, warmup=True)
    return out


def engine_cuda_graphs(model_id: str, repeats: int = 3) -> Dict[str, List[PhaseRecord]]:
    """vLLM init with vs without CUDA-graph capture (enforce_eager).

    This isolates how much of vLLM's cold start is graph capture — the phase that
    memory-snapshot approaches (Modal) skip entirely by restoring a post-capture
    state. GPU only, and needs vllm installed.
    """
    if not _has_cuda():
        return {"_skipped": _skip_note("vLLM engine experiment needs CUDA")}

    out: Dict[str, List[PhaseRecord]] = {}
    for label, eager in (("graphs", False), ("eager", True)):
        cfg = {"backend": "vllm", "model_id": model_id, "enforce_eager": eager}
        try:
            out[label] = run_config(cfg, repeats=repeats, drop_cache=False, warmup=True)
        except Exception as e:  # vllm not installed / OOM — record and move on
            out[label] = _skip_note(f"{label}: {e}")
    return out


def sleep_wake(model_id: str, repeats: int = 3) -> Dict[str, List[PhaseRecord]]:
    """vLLM sleep/wake as an open-source proxy for snapshot/restore.

    vLLM's sleep mode offloads GPU state to host memory; waking restores it
    without re-running init. Timing wake vs a full cold init is a legitimate,
    if imperfect, stand-in for the driver-level GPU snapshot that Modal/InferX do
    with the proprietary CUDA checkpoint API. GPU only.

    Left as a documented stub — vLLM's sleep API surface has moved around between
    releases, so wire it against whatever version the Colab image ships rather
    than pinning to one call signature here.
    """
    if not _has_cuda():
        return {"_skipped": _skip_note("sleep/wake needs CUDA + vLLM")}
    return {"_todo": _skip_note("implement against installed vllm sleep API")}


def _skip_note(msg: str) -> List[PhaseRecord]:
    return [PhaseRecord(phases={}, meta={"skipped": msg})]
