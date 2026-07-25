"""Command-line entry point.

Runs the full pipeline for one model: fetch -> format experiment -> storage
experiment -> engine experiment -> extrapolation -> report. Every stage is
optional via flags so you can iterate on a single experiment without paying for
the others.

Examples
--------
    # Fast CPU smoke run on random weights (what CI does):
    coldstart-lab --model tiny-llama-random --device cpu \
        --skip-engine --repeats 2 --warmup 0 --out-dir /tmp/out

    # Real micro run on a CPU:
    coldstart-lab --model smollm2-135m --device cpu --out-dir ./out

    # GPU study on Colab:
    coldstart-lab --model qwen2.5-3b --device cuda --out-dir ./out
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from coldstart_lab import environment
from coldstart_lab.experiments.engine_experiment import EngineInitExperiment
from coldstart_lab.experiments.format_experiment import FormatExperiment
from coldstart_lab.experiments.storage_experiment import StorageExperiment, StorageTier
from coldstart_lab.extrapolate import project_load_time
from coldstart_lab.fetch import fetch
from coldstart_lab.models import get_model, models_in_tier
from coldstart_lab.report import Report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coldstart-lab", description=__doc__)
    p.add_argument("--model", required=True, help="Model key from the registry.")
    p.add_argument("--device", default="cpu", help="cpu or cuda[:N].")
    p.add_argument("--out-dir", default="./coldstart_out")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--skip-format", action="store_true")
    p.add_argument("--skip-storage", action="store_true")
    p.add_argument("--skip-engine", action="store_true")
    p.add_argument("--skip-bin", action="store_true",
                   help="Skip deriving a .bin (saves time/disk on large models).")
    p.add_argument("--emulate-remote-mib-s", type=float, default=None,
                   help="Add an emulated slow storage tier at this ceiling.")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    spec = get_model(args.model)
    fp = environment.probe()
    report = Report(fp)

    print(f"[coldstart-lab] fetching {spec.key} ({spec.repo_id}) ...", flush=True)
    fetched = fetch(spec, token=args.hf_token)
    report.add_extra("pull", {"model": spec.key, "pull_ms": round(fetched.pull_ms, 2)})
    model_dir = fetched.local_dir
    print(f"[coldstart-lab] local dir: {model_dir}", flush=True)

    if not args.skip_format:
        print("[coldstart-lab] format experiment ...", flush=True)
        fmt = FormatExperiment(
            model_dir=model_dir,
            device=args.device,
            include_bin=not args.skip_bin,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        report.add(fmt.run())

    if not args.skip_storage:
        print("[coldstart-lab] storage experiment ...", flush=True)
        tiers = [StorageTier(name="local", root=args.out_dir)]
        if args.emulate_remote_mib_s:
            tiers.append(
                StorageTier(
                    name="remote-emulated",
                    root=args.out_dir,
                    emulated_mib_s=args.emulate_remote_mib_s,
                )
            )
        storage = StorageExperiment(
            model_dir=model_dir,
            tiers=tiers,
            device=args.device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        report.add(storage.run())

    if not args.skip_engine:
        print("[coldstart-lab] engine experiment ...", flush=True)
        engine = EngineInitExperiment(
            model_dir=model_dir,
            device=args.device,
            repeats=max(1, args.repeats - 1),
            warmup=0,
        )
        report.add(engine.run())

    _add_projections(report)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{spec.key}_report.json")
    md_path = os.path.join(args.out_dir, f"{spec.key}_report.md")
    report.write_json(json_path)
    report.write_markdown(md_path)
    print(f"[coldstart-lab] wrote {json_path}", flush=True)
    print(f"[coldstart-lab] wrote {md_path}", flush=True)

    if not args.no_plots:
        _maybe_plot(report, args.out_dir, spec.key)

    return 0


def _add_projections(report: Report) -> None:
    """Use the fastest measured storage throughput to project production loads."""
    best_tp = 0.0
    for exp in report.experiments:
        for t in exp.trials:
            best_tp = max(best_tp, t.metrics.get("throughput_mib_s", 0.0))
    if best_tp <= 0:
        return
    targets = models_in_tier("reference")
    projections = project_load_time(
        best_tp, targets, basis=f"best measured throughput {best_tp:.1f} MiB/s"
    )
    report.add_extra("production_load_projection", [p.to_dict() for p in projections])


def _maybe_plot(report: Report, out_dir: str, key: str) -> None:
    from coldstart_lab.report import plot_experiment

    for exp in report.experiments:
        path = os.path.join(out_dir, f"{key}_{exp.name}.png")
        written = plot_experiment(exp, path)
        if written:
            print(f"[coldstart-lab] wrote {written}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
