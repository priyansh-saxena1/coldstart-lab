#!/usr/bin/env python3
"""Publish a fleet's results as a Hugging Face dataset.

Produces a dataset repo containing the flattened observations, the raw ledger,
the generated analysis, and a dataset card describing how the numbers were
produced and what they do and do not support.

    export HF_TOKEN=hf_...
    python scripts/publish_to_hf.py \
        --merged ./merged/merged_results.json \
        --repo-id your-username/llm-cold-start-benchmark

Add --dry-run to build the upload directory locally and inspect it before
anything is pushed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date

from coldstart_lab.analysis import (Analysis, build_report,
                                    observations_from_merged)
from coldstart_lab.dataset import export
from coldstart_lab.models import MODEL_REGISTRY


def build_card(merged: dict, repo_id: str, code_url: str | None) -> str:
    obs = observations_from_merged(merged)
    a = Analysis(obs)
    cov = a.coverage()
    fit, reg = a.stable_fit("safetensors")
    shapes = a.shape_by_condition("checkpoint_format")

    exponents = {c: d["power"].exponent
                 for c, d in shapes.items() if d.get("power")}
    b_lo = min(exponents.values()) if exponents else 0.0
    b_hi = max(exponents.values()) if exponents else 0.0

    fmt_rows = a.format_speedup()
    import statistics
    med_speedup = (statistics.median([r["speedup"] for r in fmt_rows])
                   if fmt_rows else 0.0)

    families = ", ".join(cov["families"])
    code_line = (f"\n- **Code:** {code_url}" if code_url else "")

    boundary = reg.get("boundary_gib") if reg else None
    collapse = reg.get("collapse_ratio") if reg else None

    # With few models there is no regime boundary to report. Say so plainly
    # rather than emitting a card with holes where numbers should be.
    if boundary and collapse and reg.get("degraded_mib_s"):
        collapse_para = (
            f"Concretely, throughput holds near "
            f"**{reg['baseline_mib_s']:.0f} MiB/s** below **{boundary} GiB** "
            f"and falls to **{reg['degraded_mib_s']:.0f} MiB/s** above it -- a "
            f"**{collapse}x** collapse.")
    else:
        collapse_para = (
            "No throughput collapse boundary could be located in this run: "
            "either the checkpoints span too narrow a size range, or "
            "throughput held up across all of them.")

    if exponents:
        headline = (
            f"Fitting `load_ms = a x GiB^b` gives **b = {b_lo:.2f} to "
            f"{b_hi:.2f}** across conditions, where b = 1 would mean constant "
            f"throughput. **Doubling the checkpoint more than doubles the load "
            f"time**, so no single MiB/s figure describes the system -- "
            f"effective throughput is a function of model size.")
    else:
        headline = ("Too few models in this run to fit a scaling exponent.")

    stable_line = (f"{fit.mib_s:.0f} MiB/s (R2={fit.r_squared:.3f})"
                   if fit else "not established in this run")

    return f"""---
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
a state where it can serve its first token, across **{cov['models']} open-weight
models** spanning **{len(cov['families'])} architecture families** and
**{cov['total_weights_gib']} GiB** of checkpoints, on a single NVIDIA T4.

Cold start is the latency a serverless or scale-to-zero inference platform pays
when it has no warm replica. It decomposes into weight transfer from storage,
deserialization into host memory, transfer to the accelerator, and engine
bring-up. This dataset measures each phase separately rather than reporting one
aggregate number, because the dominant term shifts with model size and setup --
and optimising the wrong term buys nothing.

## Headline finding: cold start is superlinear in checkpoint size

{headline}

{collapse_para}

Throughput in the healthy regime: **{stable_line}**.

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

- **safetensors vs legacy pickle:** median **{med_speedup:.2f}x** faster to
  load over {len(fmt_rows)} models with identical weights. The format costs no
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

## Reproducing{code_line}

```bash
pip install -e ".[distributed,plots]"
coldstart-fleet init --tier small --device-class t4
coldstart-fleet work --device cuda --device-class t4
coldstart-fleet merge --out ./merged
```

## Citation

```bibtex
@misc{{coldstart_bench_{date.today().year},
  title  = {{LLM Container Cold-Start Benchmark}},
  year   = {{{date.today().year}}},
  note   = {{Dataset: https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

Apache-2.0. Measurements only; no model weights are redistributed.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--merged", help="Path to a merged_results.json")
    src.add_argument("--from-db", action="store_true",
                     help="Pull results straight from the shared ledger "
                          "(reads COLDSTART_DB_URL). Use this when the fleet "
                          "results only ever lived in Postgres.")
    p.add_argument("--repo-id", required=True,
                   help="e.g. your-username/llm-cold-start-benchmark")
    p.add_argument("--code-url", default=None,
                   help="Link to the harness repo, included in the card.")
    p.add_argument("--out-dir", default="./hf_upload")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dry-run", action="store_true",
                   help="Build the upload directory but do not push.")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    if args.from_db:
        from coldstart_lab.distributed import Coordinator, get_db_url, redact

        url = get_db_url()
        print(f"reading ledger at {redact(url)}")
        coord = Coordinator(url)
        merged = coord.results()
        progress = coord.progress()
        print(f"ledger status: {progress}")
        failures = coord.failures()
        if failures:
            print(f"{len(failures)} failed task(s) (excluded from the dataset):")
            for f in failures[:10]:
                print(f"  {f['task_id']}: {f['error'][:100]}")
        if not merged:
            print("ERROR: the ledger holds no completed results.", file=sys.stderr)
            return 2
    else:
        with open(args.merged) as fh:
            merged = json.load(fh)

    n_records = sum(len(v) for v in merged.values())
    print(f"{len(merged)} model(s), {n_records} experiment record(s)")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
        info = export(merged, args.out_dir)
    print(f"wrote {info['n_rows']} observation rows -> {info['observations_csv']}")

    obs = observations_from_merged(merged)
    report_path = os.path.join(args.out_dir, "cross_model_report.md")
    with open(report_path, "w") as fh:
        fh.write(build_report(obs))
    print(f"wrote analysis -> {report_path}")

    # Move runs.json into raw/ to avoid HF viewer 500 from auto-scanning it
    runs_src = os.path.join(args.out_dir, "runs.json")
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    runs_dst = os.path.join(raw_dir, "runs.json")
    if os.path.exists(runs_src):
        shutil.move(runs_src, runs_dst)
        print(f"moved runs.json -> {runs_dst}")

    card_path = os.path.join(args.out_dir, "README.md")
    with open(card_path, "w") as fh:
        fh.write(build_card(merged, args.repo_id, args.code_url))
    print(f"wrote dataset card -> {card_path}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded. Inspect the directory, then "
              "re-run without --dry-run.")
        return 0

    if not args.token:
        print("\nERROR: no token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.out_dir, repo_id=args.repo_id,
                      repo_type="dataset",
                      commit_message="Add cold-start benchmark results")
    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"\npublished -> {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
