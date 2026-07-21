"""Command line for the sweep.

    python -m coldstart.cli sweep --models EleutherAI/pythia-160m EleutherAI/pythia-410m ...
    python -m coldstart.cli sweep --preset t4 --device cuda:0 --out results.json

The sweep is the bread-and-butter run: load each model cold N times, decompose the
phases, then fit the load curve and extrapolate. The individual experiments
(storage tier, quant, etc.) live in experiments.py and are driven from the
notebook where you can point them at real mounted storage.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import models, report
from .runner import run_config


def _sweep(args) -> int:
    if args.preset == "t4":
        model_ids = models.DEFAULT_T4_SWEEP
    elif args.models:
        model_ids = args.models
    else:
        # CPU-friendly default so `sweep` does something sane with no args
        model_ids = [m.id for m in models.by_tier("smoke") if m.id != "sshleifer/tiny-gpt2"]

    specs = {}
    results = {}
    for mid in model_ids:
        try:
            specs[mid] = models.find(mid)
        except KeyError:
            print(f"note: {mid} not in catalog, GB unknown, skipping from curve", file=sys.stderr)
        cfg = {"backend": "transformers", "model_id": mid,
               "device": args.device, "dtype": args.dtype}
        print(f"running {mid} ...", file=sys.stderr)
        recs = run_config(cfg, repeats=args.repeats, drop_cache=not args.no_drop_cache,
                          warmup=not args.no_warmup)
        results[mid] = recs

    summary = {mid: report.summarize(recs) for mid, recs in results.items()}

    curve = report.curve_from_sweep(results, specs)
    payload = {"summary": summary}
    if curve:
        payload["load_curve"] = {
            "slope_s_per_gb": round(curve.slope_s_per_gb, 3),
            "intercept_s": round(curve.intercept_s, 3),
            "r2": round(curve.r2, 4),
            "eff_bandwidth_MBps": round(curve.eff_bandwidth_MBps, 1),
        }
        payload["extrapolation"] = report.extrapolate(curve, {
            "13B": 26.0, "27B": 54.0, "70B": 140.0,
        })

    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="coldstart")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="run the size-scaling sweep")
    sw.add_argument("--models", nargs="+", help="explicit HF model ids")
    sw.add_argument("--preset", choices=["t4"], help="use a built-in model set")
    sw.add_argument("--device", default="cpu")
    sw.add_argument("--dtype", default="float32", help="float32 on cpu, float16 on gpu")
    sw.add_argument("--repeats", type=int, default=3)
    sw.add_argument("--no-warmup", action="store_true")
    sw.add_argument("--no-drop-cache", action="store_true")
    sw.add_argument("--out", help="write results json here instead of stdout")
    sw.set_defaults(func=_sweep)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
