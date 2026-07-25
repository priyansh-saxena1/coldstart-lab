---
license: apache-2.0
task_categories:
  - other
tags:
  - benchmark
  - inference
  - cold-start
  - llm-serving
  - mlops
  - systems
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: observations.csv
---

# LLM Container Cold-Start Benchmark

Measurements of how long it takes to bring a language model from cold storage to
a state where it can serve its first token, across **25 open-weight
models** spanning **17 architecture families** and
**100.9 GiB** of checkpoints, on a single NVIDIA T4.

Cold start is the latency a serverless or scale-to-zero inference platform pays
when it has no warm replica. It decomposes into weight transfer from storage,
deserialization into host memory, transfer to the accelerator, and engine
bring-up. This dataset measures each phase separately rather than reporting one
aggregate number, because the dominant term shifts with model size and setup --
and optimising the wrong term buys nothing.

## Headline finding: cold start is superlinear in checkpoint size

Fitting `load_ms = a x GiB^b` gives **b = 1.58 to 2.10** across conditions, where b = 1 would mean constant throughput. **Doubling the checkpoint more than doubles the load time**, so no single MiB/s figure describes the system -- effective throughput is a function of model size.

Concretely, throughput holds near **1535 MiB/s** below **5.18 GiB** and falls to **758 MiB/s** above it -- a **2.02x** collapse.

Throughput in the healthy regime: **1354 MiB/s (R2=0.868)**.

The most likely cause is **host memory, not the storage device**: the runtime
had roughly 12-13 GiB of usable system RAM, and the memory-mapped load path
depends on the OS page cache holding the checkpoint while tensors are
materialised. This is stated as a hypothesis with an obvious test attached
(re-run on a high-RAM instance and see whether the boundary moves), not as a
settled conclusion.

**Why it matters:** a cold-start budget extrapolated from small models will be
optimistic for large ones, and the gap widens with size. Below the boundary,
storage bandwidth is the lever; above it, more storage bandwidth does not help,
because the bottleneck has moved.

## Other results

- **safetensors vs legacy pickle:** median **1.71x** faster to
  load over 20 models with identical weights. The format costs no
  model quality and no hardware to change.
- **Storage tier:** a bandwidth-capped tier is roughly an order of magnitude
  slower than local NVMe on the same checkpoints.
- **Engine bring-up** is a comparable cost to weight transfer at these sizes,
  which is precisely the distinction that determines whether snapshot/restore
  techniques would pay off for a given deployment.

## Files

| File | Contents |
|---|---|
| `observations.csv` | One row per (model, experiment, condition), joined to checkpoint size, parameter count, shard count and family. |
| `raw/runs.json` | The raw merged ledger, unmodified, so all statistics can be recomputed from source. |
| `cross_model_report.md` | The generated analysis: scaling fits, per-model tables, projections, noise profile, limitations. |

## Columns

`model_key`, `repo_id`, `family`, `tier`, `params_b`, `checkpoint_gib`,
`n_shards`, `bytes_per_param`, `experiment`, `condition`, `p50_ms`, `p95_ms`,
`stdev_ms`, `rsd`, `n_trials`, `throughput_mib_s`, `gpu`, `device_class`,
`reliable`.

`reliable` is `False` where relative standard deviation is 30% or more. Those
rows are published rather than dropped -- a reader may reasonably pick a
different threshold -- but no conclusion above rests on them.

## Method

- Each condition is repeated, warm-up runs are discarded, and **p50/p95** are
  reported rather than a mean, because tail latency is what a scale-to-zero SLA
  is written against.
- The OS page cache is evicted between trials (`posix_fadvise(DONTNEED)` on an
  unprivileged runtime), so reads are genuinely cold rather than served from RAM.
- Runs were distributed across several Colab sessions coordinated through a
  shared Postgres ledger with atomic claims and epoch fencing, so no task was
  ever executed twice. Duplicate execution would not merely waste time -- two
  workers would contend for the same disk and corrupt the I/O measurement.
- Registry metadata (checkpoint sizes, shard counts, gating) was read from the
  Hugging Face API rather than estimated.

## Limitations

- **Free-tier hardware.** A single consumer GPU with shared, contended host I/O.
  The *ratios* and the *scaling behaviour* transfer; the absolute milliseconds
  are not production numbers.
- **Noise.** Median RSD is around 15%, with a long tail concentrated in engine
  bring-up where JIT compilation varies run to run. Small effects here are not
  findings.
- **Not measured:** driver-level checkpoint/restore (proprietary GPU memory
  snapshotting), multi-GPU tensor-parallel loading, and real network-attached
  storage -- the remote tier is a software bandwidth cap, which bounds the
  penalty from below rather than reproducing it.
- **Coverage gaps** are listed at the top of the report.

## Reproducing
- **Code:** https://github.com/priyansh-saxena1/coldstart-lab

```bash
pip install -e ".[distributed,plots]"
coldstart-fleet init --tier small --device-class t4
coldstart-fleet work --device cuda --device-class t4
coldstart-fleet merge --out ./merged
```

## Citation

```bibtex
@misc{coldstart_bench_2026,
  title  = {LLM Container Cold-Start Benchmark},
  year   = {2026},
  note   = {Dataset: https://huggingface.co/datasets/ArchCoder/llm-cold-start-benchmark}
}
```

## License

Apache-2.0. Measurements only; no model weights are redistributed.
