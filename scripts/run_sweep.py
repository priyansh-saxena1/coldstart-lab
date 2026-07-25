#!/usr/bin/env python3
"""Run the benchmark across a list of models and collect their reports.

This is a thin wrapper over the CLI for when you want to sweep, e.g., the whole
`small` tier in one go on a GPU box. Each model gets its own report; failures on
one model (OOM, gated repo without a token) don't abort the sweep.

    python scripts/run_sweep.py --tier small --device cuda --out-dir ./sweep
"""

from __future__ import annotations

import argparse
import sys
import traceback

from coldstart_lab.cli import main as cli_main
from coldstart_lab.models import models_in_tier


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default="micro",
                   choices=["ci", "micro", "small", "medium"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="./sweep")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    models = [m for m in models_in_tier(args.tier) if m.downloadable]
    failures = {}
    for spec in models:
        print(f"\n{'='*70}\n {spec.key}\n{'='*70}")
        argv = [
            "--model", spec.key,
            "--device", args.device,
            "--out-dir", args.out_dir,
            "--repeats", str(args.repeats),
        ]
        # Deriving a .bin for large models is expensive; skip above ~3B.
        if spec.params_b > 3.5:
            argv.append("--skip-bin")
        try:
            cli_main(argv)
        except Exception:  # noqa: BLE001 - keep sweeping past a single failure
            failures[spec.key] = traceback.format_exc()
            print(f"[sweep] {spec.key} FAILED (continuing)")

    if failures:
        print(f"\n[sweep] {len(failures)} model(s) failed:")
        for key in failures:
            print(f"  - {key}")
        return 1
    print("\n[sweep] all models completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
