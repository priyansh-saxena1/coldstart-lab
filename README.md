# coldstart-lab

A benchmarking harness for **LLM container cold-start latency** — the time a
serverless / scale-to-zero inference platform pays to bring a model up from
nothing before it can serve a first token.

The harness decomposes cold start into the phases that actually move the needle
and measures each in isolation, on a **free-tier CPU or a single consumer GPU**,
then extrapolates the results to production model sizes with a stated error
model. It is built to answer a concrete platform question: *given our storage
tier, checkpoint format, and engine, where is cold-start time going, and which
lever is worth pulling first?*

## Why the phases matter

Cold start is a composite, and the dominant term shifts with the setup:

```
cold_start = pull            (weights leave storage)
           + deserialize     (bytes become tensors in host RAM)
           + to_device       (tensors cross PCIe to the GPU)
           + engine_bringup  (CUDA context, JIT/kernel warm-up,
                              CUDA-graph capture, KV-cache allocation)
```

Optimising the wrong term is a common failure mode — memory-snapshotting an
engine that is actually bottlenecked on weight transfer buys nothing. This
harness measures the terms separately so the optimisation follows the data.

## What it measures

| Experiment | Question | Runs on |
|---|---|---|
| `checkpoint_format` | safetensors (mmap / no-mmap) vs legacy pickled `.bin`, identical weights | CPU or GPU |
| `storage_tier` | local NVMe vs network-attached (real Drive mount or emulated bandwidth ceiling) | CPU or GPU |
| `engine_init` | transformers init→load→first-forward; on GPU, vLLM `enforce_eager` vs CUDA-graph capture | CPU (transformers) / GPU (vLLM) |
| extrapolation | project measured MiB/s onto 32B / 70B production checkpoints | anywhere |

Every experiment repeats each condition, discards warm-ups, drops the OS page
cache between trials (or falls back to `posix_fadvise` and says so), and reports
**p50 / p95**, not just a mean — because a scale-to-zero SLA is written against
the tail.

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[gpu]"     # vLLM engine experiment
pip install -e ".[plots]"   # matplotlib charts in the report
pip install -e ".[dev]"     # pytest
pip install -e ".[distributed]"  # multi-session Postgres fleet
```

Requires Python ≥ 3.10, and torch / transformers / safetensors (already present
on Colab).

## Quick start

```bash
# Fast CPU smoke run on random weights — no real checkpoint downloaded:
coldstart-lab --model tiny-llama-random --device cpu --skip-engine \
    --repeats 2 --warmup 1 --out-dir ./out

# Real micro study on a CPU:
coldstart-lab --model smollm2-135m --device cpu --out-dir ./out

# Core GPU study on Colab (T4/L4):
coldstart-lab --model qwen2.5-3b --device cuda --out-dir ./out

# Show that 4-bit cuts load time, not just memory:
coldstart-lab --model qwen2.5-7b     --device cuda --skip-bin --out-dir ./out
coldstart-lab --model qwen2.5-7b-awq --device cuda --skip-bin --out-dir ./out
```

Each run writes `<model>_report.json`, `<model>_report.md`, and PNG charts.

### Colab

Open `notebooks/coldstart_lab_colab.ipynb`. It unzips/clones the repo, installs,
prints the environment fingerprint and model registry, and walks through all
three experiments plus extrapolation. Set the runtime to a T4 for the GPU study.

## Model registry — 61 models, 28 architecture families

Grouped by **weight footprint**, since that is what cold start scales with
(`coldstart_lab/models.py`):

| Tier | Models | Size range | Runs on |
|---|---:|---|---|
| `ci` | 1 | ~4 MB | test suite, no network |
| `micro` | 6 | 0.25–0.92 GiB | free CPU runtime |
| `small` | 26 | 1.0–7.5 GiB | T4 (15 GiB) with staging headroom |
| `medium` | 20 | 9.3–22.8 GiB | L4 / A100, or quantized |
| `reference` | 8 | 26–135 GiB | **never downloaded** — extrapolation targets |

Families include Qwen2.5, Qwen3 (incl. a 30B MoE), Phi-2/3/3.5/4, SmolLM2/3,
Pythia, GPT-2, BLOOM, TinyLlama, StableLM, Falcon/Falcon3, OLMo-2, Granite,
InternLM, Yi, DeepSeek + R1-Distill, Zephyr, OpenHermes, SOLAR, GLM-4 and
Mistral — plus matched AWQ/GPTQ 4-bit pairs for `Qwen2.5-7B/14B/32B`, which are
the cleanest experiment available: identical architecture and parameter count,
different bytes on disk.

**Every entry is verified, not estimated.** Sizes, shard counts and gating status
were read from the Hugging Face API, not recalled from memory:

```bash
python scripts/verify_registry.py            # all 61
python scripts/verify_registry.py --tier small --fix
```

Two Mistral repos ship a `consolidated.safetensors` copy *alongside* the shards —
the same tensors twice. A naive `snapshot_download` fetches both and doubles the
pull for nothing; those are tagged `dup-representation` and sized by shards only.

No gated repos are included, by policy — see the fleet notes above.

## Cross-model analysis

`coldstart_lab.analysis` turns a fleet's ledger into findings that only appear
once many models are measured. `coldstart-fleet merge` writes it automatically:

```bash
coldstart-fleet merge --out ./merged   # -> merged_results.json + cross_model_report.md
```

The report covers: coverage; whether load time is linear in checkpoint size
(least-squares fit per condition, reporting slope as ms/GiB, the size-independent
floor as an intercept, and R²); safetensors-vs-pickle speedup across every model;
what 4-bit buys at *load* time versus its size ratio; per-shard fixed cost;
storage-tier throughput; engine bring-up; a projection to 32B/72B; a run-to-run
noise profile; and limitations.

Two deliberate design choices worth knowing:

- **The projection basis is chosen by R², not by the fastest slope.** A fit can
  look fast simply because it is a bad fit. If no condition clears R² ≥ 0.80 the
  report prints *"Do not quote these numbers"* above the table rather than
  emitting a confident-looking projection built on noise.
- **Every effect is reported against the measured noise floor.** An effect
  smaller than the run-to-run spread is not a finding, and the report says so.



## Design notes

- **Loaders** (`coldstart_lab/loaders/`) are swappable and share one driver loop,
  so any measured delta is attributable to the loading strategy, not the harness.
  A local converter derives a byte-identical `.bin` from safetensors so the
  format A/B holds the same weights.
- **Cold reads are real**: `environment.drop_page_cache` evicts cached pages
  system-wide when privileged, else `posix_fadvise(DONTNEED)` per file; the mode
  is recorded in the report.
- **Emulated storage tiers can only slow a read down**, never speed it up, so
  emulation cannot manufacture a favourable result. Emulated rows are labelled.
- **Extrapolation is linear in bytes** and explicitly surfaces its assumptions
  (ignores fixed per-file overhead; assumes matched per-byte characteristics).

## Distributed mode: N Colab sessions, one Postgres ledger

Free-tier sessions are slow and pre-emptible, so the practical way to cover a
whole model tier is to run several at once. `coldstart_lab.distributed` fans the
task list across as many sessions as you can open, coordinating through a single
Postgres database (Neon's free tier is enough).

```bash
# once, from any session:
coldstart-fleet init --tier small --device-class t4

# in every session:
coldstart-fleet work --device cuda --device-class t4

# from anywhere:
coldstart-fleet status --watch
coldstart-fleet merge --out ./merged
```

The unit of work is one `(model, experiment, device_class)` triple. Device class
is part of the *identity*, not metadata: a T4 number is not interchangeable with
an A100 number, so the same experiment on different hardware is a different task
rather than a duplicate to skip.

**Exactly-once execution is a correctness requirement here, not an efficiency
nicety** — two workers on the same task would contend for the same disk and
pollute the I/O measurement being taken. Guarantees:

- **Atomic claim.** One conditional `UPDATE ... WHERE status='pending'`; the
  loser of a race sees `rowcount == 0` and moves on. Portable across Postgres
  and SQLite, so the identical code path is unit-tested locally.
- **Epoch fencing.** A session paused past its lease can wake up and try to
  commit. Every claim bumps `epoch`, and `complete()` writes only
  `WHERE epoch = :mine`, so a zombie's write affects 0 rows and is rejected
  loudly instead of clobbering the live worker's result.
- **Lease reclamation.** A pre-empted worker's task returns to the pool after
  `COLDSTART_LEASE_TIMEOUT_S` (default 1800s, comfortably longer than a 7B pull).
  On Colab this is the normal case, not the exception.
- **LPT scheduling.** Biggest checkpoint claimed first, so the slowest job never
  lands at the end of the run with the fleet idle behind it.
- **Workers wait, they don't quit.** An empty *pending* queue is not a finished
  run — another worker may hold the last task, and if it is pre-empted the task
  comes back. Workers poll (with jitter) until every task is terminal, so an
  idle session is still there to pick up released work. `--no-wait` opts out.
- **Non-retryable failures fail once.** A 401, 404 or unknown model key will
  fail identically on every attempt; retrying it three times starves the queue,
  since the biggest checkpoint is claimed first. These park as `failed`
  immediately. `coldstart-fleet retry` re-queues them once you've fixed the cause.
- **Staging is cleared between tasks.** The storage experiment copies the whole
  checkpoint per tier; two copies of a 7.6 GiB model plus the HF cache is a third
  of a Colab disk, and the next task would die on an ENOSPC that looks like
  anything but "out of room".

Triage a stalled run with:

```bash
coldstart-fleet status          # counts, plus the error line for each failure
coldstart-fleet retry           # return failed tasks to the queue
```

### No gated models

Every model in the registry is ungated — nothing needs an HF token or an
accepted licence. `meta-llama/*` repos were removed deliberately: a 401 is not
transient, and because tasks are ordered longest-checkpoint-first, a large gated
model gets re-claimed ahead of real work on every pass. Ungated Qwen, SmolLM2
and Phi models cover the same size classes.

Credentials come from the environment and are never hardcoded:

```python
import os, getpass
os.environ["COLDSTART_DB_URL"] = getpass.getpass("DB URL: ")
```

`getpass` keeps the string out of saved notebook output. Paste the URL exactly as
Neon gives it — `normalise_db_url()` adds the psycopg2 driver, drops
`channel_binding` (libpq-version dependent, and psycopg2 rejects it) and keeps
`sslmode=require`.

> The coordinator is a port of one built for a distributed OTFS/ISAC optimisation
> pipeline. The concurrency semantics are domain-agnostic; only the task identity
> and payload changed.

## Publishing results

Turn a fleet's ledger into a Hugging Face dataset -- flattened observations, the
raw ledger, the analysis, and a dataset card whose headline numbers are filled
in from the data rather than typed by hand:

Results can come from a file or straight from the shared ledger, which matters
when a distributed run's output only ever lived in Postgres:

```bash
export HF_TOKEN=hf_...
python scripts/publish_to_hf.py \
    --from-db \
    --repo-id your-username/llm-cold-start-benchmark \
    --code-url https://github.com/your-username/coldstart-lab \
    --dry-run          # build locally and inspect first
```

`observations.csv` carries one row per (model, experiment, condition), joined to
checkpoint size, parameter count, shard count and family, plus a `reliable`
column that is `False` where relative standard deviation is 30% or more. Noisy
rows are published rather than dropped -- a reader may pick a different
threshold -- but nothing in the analysis rests on them.

## Tests

```bash
pytest             # 41 fast CPU tests, no network, no services
pytest -m network  # adds one real tiny-model download
```

The concurrency tests spawn **real OS processes hammering one database**, because
lost updates, double-claims and zombie writes do not reproduce single-threaded.
They run against SQLite by default and against Postgres when pointed at one:

```bash
COLDSTART_TEST_DB_URL=postgresql://... pytest tests/test_coordinator_race.py
```

Running both matters: Postgres returns tz-**aware** `now()` while `heartbeat_at`
is tz-**naive**, and subtracting them raises `TypeError`. SQLite returns naive for
both, so that bug is invisible locally and only detonates on the real backend.
`test_timezone_mismatch` pins the normalisation directly.

## Repository layout

```
src/coldstart_lab/
  timing.py          high-resolution phase timing
  environment.py     system fingerprint + page-cache control
  models.py          the model registry
  fetch.py           HF checkpoint fetch (weights/config/tokenizer only)
  loaders/           safetensors / bin loaders + converter
  experiments/       format / storage / engine + stats base
  extrapolate.py     MiB/s -> production load-time projection
  report.py          JSON + Markdown + optional charts
  cli.py             end-to-end orchestrator
  distributed/       Postgres work ledger, worker loop, fleet CLI
notebooks/           Colab drivers (single-machine + fleet worker)
scripts/             multi-model sweep
tests/               pytest suite
```

## Scope and honesty

This is a measurement and analysis tool, not a claim to have reimplemented a
production platform's cold-start path. It cannot replicate driver-level
checkpoint/restore (e.g. proprietary GPU memory snapshotting) without that
infrastructure; where a technique is out of reach on a free runtime, the harness
measures the closest honest proxy and says so in the report.

## License

Apache-2.0.
