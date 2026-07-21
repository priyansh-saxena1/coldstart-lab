"""One cold load, measured in isolation.

This module is meant to be run as a fresh process:

    python -m coldstart.worker /tmp/cfg.json

That freshness is the whole reason it exists. If you load a model twice in the
same interpreter, the second load is warm — imports are cached, the allocator is
primed, the CUDA context already exists, and the OS page cache is holding the
weight files. You'd be measuring a warm start and calling it cold. Spawning a
new process per measurement is the only way to get an honest number without
rebooting the box between every run.

The worker prints exactly one JSON object to stdout (the phase breakdown). The
parent parses that. Anything else the libraries scribble goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time

_clock = time.perf_counter


def _emit(obj: dict) -> None:
    # single line, stdout only. parent splits on this.
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def run(cfg: dict) -> dict:
    backend = cfg.get("backend", "transformers")
    if backend == "transformers":
        return _run_transformers(cfg)
    if backend in ("vllm", "sglang"):
        # these only make sense on a GPU box; the adapter raises a clear error
        # if the engine isn't importable so the parent can skip rather than crash.
        return _run_engine(cfg, backend)
    raise ValueError(f"unknown backend {backend!r}")


def _run_transformers(cfg: dict) -> dict:
    phases: dict = {}
    model_id = cfg["model_id"]
    load_path = cfg.get("load_path") or model_id  # staged dir, or fall back to hub/cache
    device = cfg.get("device", "cpu")
    dtype_name = cfg.get("dtype", "float32")
    quant = cfg.get("quantization")  # None | "bnb-4bit" | "bnb-8bit"

    t = _clock()
    import torch  # noqa: WPS433 (import inside fn is intentional — it's timed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    phases["import"] = _clock() - t

    dtype = getattr(torch, dtype_name)

    load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
    if quant:
        # bitsandbytes path is GPU-only. we build the config here but it will
        # blow up on CPU, which is fine — quant experiments are gated to GPU.
        from transformers import BitsAndBytesConfig
        if quant == "bnb-4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )
        elif quant == "bnb-8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"unknown quantization {quant!r}")
        # quantized loads place themselves on GPU via device_map
        load_kwargs["device_map"] = {"": 0}

    t = _clock()
    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    phases["load"] = _clock() - t

    # tokenizer load is cheap but we still count it — it's part of readiness.
    t = _clock()
    tok = AutoTokenizer.from_pretrained(load_path)
    phases["tokenizer"] = _clock() - t

    # move to device unless a device_map already placed it (quantized case)
    if not quant and device != "cpu":
        t = _clock()
        model = model.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        phases["to_device"] = _clock() - t

    # first forward = TTFT proxy. one token, greedy, fixed short prompt.
    prompt = cfg.get("prompt", "The quick brown fox")
    t = _clock()
    inputs = tok(prompt, return_tensors="pt")
    if device != "cpu" and not quant:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    phases["first_forward"] = _clock() - t

    meta = {
        "backend": "transformers",
        "model_id": model_id,
        "device": device,
        "dtype": dtype_name,
        "quantization": quant,
        "param_count": int(sum(p.numel() for p in model.parameters())),
    }
    return {"phases": phases, "meta": meta}


def _run_engine(cfg: dict, backend: str) -> dict:
    """vLLM / SGLang adapter. GPU-only; not exercised in CPU CI.

    We keep this thin on purpose. The engines do their own weight loading, JIT
    compile, CUDA-graph capture and KV-cache allocation inside one opaque call,
    so we can't decompose them the way we do transformers. What we *can* do is
    time engine-init vs first-request separately, and toggle the knobs that
    matter for cold start (eager mode skips CUDA-graph capture).
    """
    phases: dict = {}
    model_id = cfg["model_id"]

    t = _clock()
    try:
        if backend == "vllm":
            from vllm import LLM, SamplingParams
        else:  # sglang
            import sglang  # noqa: F401
            raise NotImplementedError("sglang adapter is a stub; wire up on GPU box")
    except ImportError as e:
        return {"error": f"{backend} not importable: {e}", "meta": {"backend": backend, "model_id": model_id}}
    phases["import"] = _clock() - t

    enforce_eager = cfg.get("enforce_eager", False)
    t = _clock()
    llm = LLM(
        model=cfg.get("load_path") or model_id,
        enforce_eager=enforce_eager,          # True => skip CUDA graph capture
        dtype=cfg.get("dtype", "auto"),
        gpu_memory_utilization=cfg.get("gpu_mem_util", 0.85),
    )
    phases["engine_init"] = _clock() - t

    t = _clock()
    llm.generate([cfg.get("prompt", "The quick brown fox")], SamplingParams(max_tokens=1))
    phases["first_request"] = _clock() - t

    return {
        "phases": phases,
        "meta": {
            "backend": backend,
            "model_id": model_id,
            "enforce_eager": enforce_eager,
        },
    }


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        _emit({"error": "no config path given"})
        return 2
    with open(argv[0]) as f:
        cfg = json.load(f)
    try:
        result = run(cfg)
    except Exception as e:  # noqa: BLE001 — worker must always report, never crash silently
        import traceback
        result = {"error": str(e), "trace": traceback.format_exc(), "meta": cfg}
    _emit(result)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
