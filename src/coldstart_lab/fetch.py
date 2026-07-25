"""Checkpoint fetching.

Thin wrapper over ``huggingface_hub.snapshot_download`` that (a) refuses to
download models flagged reference-only, (b) restricts the download to weight +
config + tokenizer files so we don't drag ONNX/GGUF siblings into the timing,
and (c) returns the local path plus the measured pull time -- which is itself a
cold-start phase worth recording.
"""

from __future__ import annotations

from dataclasses import dataclass

from coldstart_lab.models import ModelSpec, get_model
from coldstart_lab.timing import Stopwatch

_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.model", "*.txt", "tokenizer*"]


@dataclass
class FetchResult:
    model_key: str
    local_dir: str
    pull_ms: float


def fetch(model: "str | ModelSpec", token: str | None = None) -> FetchResult:
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    if not spec.downloadable:
        raise ValueError(
            f"Model {spec.key!r} is reference-only and must not be downloaded; "
            "use its metadata for extrapolation instead."
        )

    from huggingface_hub import snapshot_download

    sw = Stopwatch().start()
    local_dir = snapshot_download(
        spec.repo_id,
        allow_patterns=_ALLOW_PATTERNS,
        token=token,
    )
    pull_ms = sw.stop()
    return FetchResult(model_key=spec.key, local_dir=local_dir, pull_ms=pull_ms)
