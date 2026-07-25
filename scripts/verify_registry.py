#!/usr/bin/env python3
"""Verify the registry against the Hugging Face API.

Checks every entry without downloading weights: the repo exists, it is not
gated, it ships safetensors, and the recorded size still matches what the hub
reports. Run it after editing the registry, and periodically -- repos get
re-uploaded, re-sharded and occasionally gated after the fact, and a registry
that has silently drifted produces a fleet run full of failures that look like
infrastructure bugs.

    python scripts/verify_registry.py                # all entries
    python scripts/verify_registry.py --tier small   # one tier
    python scripts/verify_registry.py --fix          # print corrected sizes
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys

from coldstart_lab.models import MODEL_REGISTRY, ModelSpec

TOLERANCE = 0.02  # 2% size drift is fine (re-uploads change padding slightly)


def check(spec: ModelSpec) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    out = {"key": spec.key, "repo": spec.repo_id, "problems": [], "actual_gib": None}
    try:
        info = api.model_info(spec.repo_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001 - report, don't abort the sweep
        out["problems"].append(f"unreachable: {type(e).__name__}")
        return out

    if info.gated:
        out["problems"].append("repo is GATED (needs a token/licence)")
    if info.private:
        out["problems"].append("repo is private")

    st = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    shards = [s for s in st if "consolidated" not in s.rfilename]
    if not shards:
        out["problems"].append("no .safetensors shards")
        return out

    actual = sum((s.size or 0) for s in shards) / 2**30
    out["actual_gib"] = round(actual, 3)
    if spec.approx_disk_gib > 0:
        drift = abs(actual - spec.approx_disk_gib) / spec.approx_disk_gib
        if drift > TOLERANCE:
            out["problems"].append(
                f"size drift {drift:.0%}: registry {spec.approx_disk_gib} GiB "
                f"vs hub {actual:.2f} GiB")
    if len(shards) != spec.n_shards:
        out["problems"].append(
            f"shard count {spec.n_shards} -> {len(shards)}")
    if len(st) != len(shards):
        out["problems"].append(
            "ships a duplicate consolidated copy (expected; size counts shards)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default=None)
    p.add_argument("--fix", action="store_true",
                   help="Print corrected approx_disk_gib values for drifted rows.")
    args = p.parse_args()

    specs = [s for s in MODEL_REGISTRY.values()
             if args.tier is None or s.tier == args.tier]
    print(f"verifying {len(specs)} registry entries against the HF API ...\n")

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(check, specs))

    bad = [r for r in results if r["problems"]]
    for r in sorted(bad, key=lambda r: r["key"]):
        print(f"  {r['key']:<28} {r['repo']}")
        for prob in r["problems"]:
            print(f"      - {prob}")

    if args.fix:
        print("\ncorrected sizes:")
        for r in bad:
            if r["actual_gib"]:
                print(f'  {r["key"]}: approx_disk_gib={r["actual_gib"]},')

    ok = len(results) - len(bad)
    print(f"\n{ok}/{len(results)} entries clean.")
    # "Duplicate consolidated copy" is expected and documented, not a failure.
    hard = [r for r in bad
            if any("consolidated" not in p for p in r["problems"])]
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
