"""CPU smoke run — no GPU, tiny models, finishes in a couple minutes.

Exercises the real paths (subprocess isolation, phase decomposition, storage
staging, curve fit) end to end so we know the plumbing works before burning GPU
minutes on Colab. Two "tiers" here are just two local dirs; on Colab you'd point
one at a mounted Drive.

    python scripts/smoke.py
"""

import sys
import tempfile

from coldstart import models, report
from coldstart.runner import run_config
from coldstart import experiments


def main() -> int:
    small = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
             "HuggingFaceTB/SmolLM2-135M-Instruct"]

    print("== sweep ==")
    specs, results = {}, {}
    for mid in small:
        specs[mid] = models.find(mid)
        cfg = {"backend": "transformers", "model_id": mid, "device": "cpu", "dtype": "float32"}
        recs = run_config(cfg, repeats=2, drop_cache=True, warmup=True)
        results[mid] = recs
        s = report.summarize(recs)
        ph = s["phases"]
        print(f"  {mid:42s} load p50={ph['load']['p50']:.3f}s  "
              f"total p50={s['total']['p50']:.3f}s")

    curve = report.curve_from_sweep(results, specs)
    if curve:
        print(f"  curve: {curve.slope_s_per_gb:.3f} s/GB + {curve.intercept_s:.3f}s  "
              f"(R^2={curve.r2:.3f}, eff BW ~{curve.eff_bandwidth_MBps:.0f} MB/s, n={curve.n})")
        pred = report.extrapolate(curve, {"7B": 15.2, "27B": 54.0})
        for name, d in pred.items():
            print(f"    extrapolated {name} ({d['gb']} GB): {d['predicted_load_s']}s")

    print("== storage tier (two local dirs as stand-in tiers) ==")
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        tier_res = experiments.storage_tier(
            "EleutherAI/pythia-70m",
            tiers={"tierA": a, "tierB": b},
            repeats=2, device="cpu",
        )
        summ = report.summarize_arms(tier_res)
        for tier, s in summ.items():
            bw = tier_res[tier][0].meta.get("stage_MBps")
            print(f"  {tier}: load p50={s['phases']['load']['p50']:.3f}s  stage ~{bw} MB/s")

    # cheap sanity gates so the smoke run fails loudly if plumbing breaks
    assert curve is not None, "curve fit returned None on 3 points"
    assert all(r.phases.get("load", 0) > 0 for r in results[small[0]]), "load phase not recorded"
    print("\nsmoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
