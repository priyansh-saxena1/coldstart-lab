# coldstart-lab

Phase-decomposed cold-start benchmarking for open-source LLM inference.

The premise: "cold start" is not a single number. On a scale-from-zero event the
time-to-first-token is a sum of phases, and which phase dominates changes with the
model size, the storage tier the weights live on, the checkpoint format, and the
inference engine. Optimizing the wrong phase buys you nothing. This tool measures
the phases separately, sweeps them across model sizes, and extrapolates the load
curve to production sizes so the measurement is actually decision-useful.

Phases measured (transformers backend):

```
pull(stage_copy) -> import -> weight-load -> tokenizer -> to_device -> first_forward
```

For vLLM the internals are opaque (weight load, JIT, CUDA-graph capture and
KV-cache alloc happen inside one call), so there we time `import -> engine_init ->
first_request` and toggle the one knob that matters for cold start:
`enforce_eager`, which skips CUDA-graph capture.

## Why the numbers are honest

Two things sink most naive cold-start benchmarks, and both are handled here:

- **Warm page cache masquerading as cold.** Load a model twice in one process and
  the second load reads weights from RAM, not storage. Every measurement runs in a
  **fresh subprocess** (`coldstart/worker.py`) and the OS page cache is dropped
  between runs (`storage.drop_page_cache`, best-effort — needs root; it warns
  loudly if it can't so you don't quote warm numbers as cold).
- **First-run overhead attributed to the model.** The first cold start on a box
  also pays for the first-ever CUDA context, kernel autotune caches, and hub
  metadata. One discarded warmup run absorbs that.

## Quickstart

CPU smoke run (tiny models, no GPU, ~2 min) — verifies the whole pipeline:

```bash
pip install -e .
python scripts/smoke.py
```

On a Colab T4: open `notebooks/coldstart_bench.ipynb`, upload this repo as a zip,
run top to bottom. It does the size sweep, the storage-tier / quantization /
engine experiments, plots the phase breakdown, and saves `results.json`.

CLI sweep:

```bash
python -m coldstart.cli sweep --preset t4 --device cuda:0 --dtype float16 --out results.json
```

## The experiments (`coldstart/experiments.py`)

| experiment | what it isolates | needs GPU |
|---|---|---|
| `storage_tier` | network-attached vs local disk (Drive vs NVMe on Colab) | no |
| `checkpoint_format` | safetensors mmap vs legacy `.bin` pickle | no |
| `quantization` | fp16 vs bitsandbytes 4-bit: load time + footprint | yes |
| `engine_cuda_graphs` | vLLM init with vs without CUDA-graph capture | yes |
| `sleep_wake` | vLLM sleep/wake as an OSS proxy for snapshot/restore | yes (stub) |

The size sweep on its own fits `load_s ~ slope·GB + intercept`; `slope` is
1/effective-bandwidth and the intercept is size-independent overhead. That fit is
what powers `report.extrapolate(...)` to production sizes. R² is reported so you
can see how far to trust the line.

## Model catalog (`coldstart/models.py`)

Chosen to give a clean size-scaling curve in one arch family (Pythia 70M–1.4B,
Qwen2.5 0.5B–7B) that fits a 16 GB T4, plus tiny CPU-runnable models for CI, plus
one 7B to make the storage/format/quant deltas unambiguous (they're tens of ms at
0.5B, seconds at 7B). Gated production models (Llama 3.2, Gemma-2, Mistral-7B) are
included but opt-in behind a HF token.

## What this does *not* claim to measure

- It measures single-process transformers loads and single-node vLLM init. It does
  **not** model multi-GPU tensor-parallel sharding, where load parallelizes across
  ranks and the curve above stops being linear — flagged rather than faked.
- Colab's Drive mount is FUSE-backed, so the "network-attached" tier includes the
  FUSE layer's own caching. That's arguably close to what a shared FS does, but
  it's not a clean block-device page-cache drop; the storage-tier numbers carry
  that caveat.
- The driver-level GPU memory snapshot that Modal/InferX use (CUDA checkpoint API)
  is proprietary; `sleep_wake` approximates it with vLLM's sleep mode, which is a
  different mechanism and only a proxy.

## Layout

```
coldstart/
  timing.py       phase timer + p50/p95 summarization
  worker.py       one isolated cold load, prints phase JSON (run as a subprocess)
  runner.py       spawns workers, drops cache, repeats, collects
  storage.py      stage weights onto a tier, drop page cache
  models.py       catalog + T4 preset
  experiments.py  the five experiments
  report.py       curve fit + extrapolation + arm diffing
  cli.py          `coldstart sweep`
tests/            timing math, curve fit, worker-output parsing
notebooks/        Colab driver
scripts/smoke.py  CPU end-to-end check
```

## Tests

```bash
pytest tests/ -q
```

Coverage is concentrated on the parts that actually break: the percentile/summary
math, the load-curve fit and extrapolation, and the worker-output JSON parsing
(which has to find one JSON line amid whatever torch/transformers print to stdout).
