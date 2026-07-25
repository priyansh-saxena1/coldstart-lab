#!/usr/bin/env python3
"""Fleet CLI: initialise the queue, run a worker, watch progress, merge results.

Typical use across three Colab sessions:

    # once, from any session:
    python -m coldstart_lab.distributed.cli init --tier small --device-class t4

    # in every session (they coordinate automatically):
    python -m coldstart_lab.distributed.cli work --device cuda --device-class t4

    # from anywhere, any time:
    python -m coldstart_lab.distributed.cli status
    python -m coldstart_lab.distributed.cli merge --out ./merged
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from coldstart_lab.distributed.config import (DISTRIBUTED_EXPERIMENTS,
                                              LEASE_TIMEOUT_S, get_db_url, redact)
from coldstart_lab.distributed.coordinator import Coordinator
from coldstart_lab.distributed.worker import work_loop
from coldstart_lab.models import models_in_tier


def _quiet_third_party(verbose: bool) -> None:
    """Silence the HF download bars and per-file HTTP chatter.

    A fleet writes every worker's stdout into one notebook cell; the hub's
    progress bars interleave into unreadable noise and bury the two lines that
    matter (CLAIMED / committed). The bars also cannot be trusted as timing
    output anyway -- the harness measures the pull itself.
    """
    if verbose:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for name in ("httpx", "urllib3", "huggingface_hub", "filelock",
                 "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("coldstart.fleet")


def cmd_init(args, log) -> int:
    url = get_db_url()
    log.info("connecting to %s", redact(url))
    coord = Coordinator(url, lease_timeout_s=LEASE_TIMEOUT_S, logger=log)
    coord.init_schema()

    models = [m for m in models_in_tier(args.tier) if m.downloadable]
    if args.max_gib is not None:
        models = [m for m in models if m.approx_disk_gib <= args.max_gib]
    if args.family:
        models = [m for m in models if m.family in set(args.family)]
    if not models:
        log.error("no downloadable models in tier %r", args.tier)
        return 1

    # Cost hint = checkpoint GiB, so the biggest pull is claimed first (LPT).
    hints = {m.key: int(m.approx_disk_gib) for m in models}
    added = coord.register([m.key for m in models], args.experiments,
                           args.device_class, cost_hints=hints)
    total_gib = sum(m.approx_disk_gib for m in models)
    log.info("registered %d new task(s) across %d model(s) x %d experiment(s)",
             added, len(models), len(args.experiments))
    log.info("total weights to pull: %.1f GiB (largest: %s at %.1f GiB)",
             total_gib, max(models, key=lambda m: m.approx_disk_gib).key,
             max(m.approx_disk_gib for m in models))
    log.info("queue: %s", coord.progress())
    return 0


def cmd_work(args, log) -> int:
    url = get_db_url()
    log.info("worker connecting to %s", redact(url))
    summary = work_loop(
        url,
        device=args.device,
        device_class=args.device_class,
        hf_token=args.hf_token,
        repeats=args.repeats,
        warmup=args.warmup,
        max_tasks=args.max_tasks,
        out_root=args.out_root,
        wait=not args.no_wait,
        poll_interval_s=args.poll_interval,
        logger=log,
    )
    log.info("worker finished: %s", json.dumps(summary))
    if not summary["fleet_finished"]:
        log.warning("NOTE: this worker stopped but the fleet is NOT finished. "
                    "Remaining: %s",
                    Coordinator(url).unfinished_counts())
    return 0


def cmd_status(args, log) -> int:
    coord = Coordinator(get_db_url(), lease_timeout_s=LEASE_TIMEOUT_S, logger=log)
    while True:
        prog = coord.progress()
        total = sum(prog.values())
        done = prog.get("done", 0)
        pct = (100.0 * done / total) if total else 0.0
        line = " | ".join(f"{k}={v}" for k, v in sorted(prog.items()))
        print(f"[{time.strftime('%H:%M:%S')}] {line}  ({pct:.0f}% done)", flush=True)
        if prog.get("failed") and not args.watch:
            print("\nFAILED tasks (first error line each):", flush=True)
            for row in coord.failures():
                print(f"  {row['task_id']}: {row['error']}", flush=True)
        if not args.watch or coord.all_done():
            break
        time.sleep(args.interval)
    return 0


def cmd_retry(args, log) -> int:
    coord = Coordinator(get_db_url(), lease_timeout_s=LEASE_TIMEOUT_S, logger=log)
    n = coord.retry_failed()
    log.info("returned %d failed task(s) to the queue", n)
    log.info("queue: %s", coord.progress())
    return 0


def cmd_export(args, log) -> int:
    """Build the complete publishable bundle straight from the database.

    Everything a reader needs and everything an auditor needs: the flattened
    observations, the raw payloads, the full task ledger including failures and
    attempt counts, the generated analysis, and the dataset card.
    """
    import sys

    from coldstart_lab.analysis import build_report, observations_from_merged
    from coldstart_lab.dataset import export as export_tables

    coord = Coordinator(get_db_url(), lease_timeout_s=LEASE_TIMEOUT_S, logger=log)

    results = coord.results()
    if not results:
        log.error("the ledger has no completed results; nothing to export")
        return 1

    info = export_tables(results, args.out)
    log.info("observations.csv: %d rows", info["n_rows"])

    ledger = coord.dump_all()
    ledger_path = os.path.join(args.out, "task_ledger.json")
    with open(ledger_path, "w") as fh:
        json.dump(ledger, fh, indent=1)
    done = sum(1 for t in ledger["tasks"] if t["status"] == "done")
    log.info("task_ledger.json: %d tasks (%d done, %d other)",
             len(ledger["tasks"]), done, len(ledger["tasks"]) - done)

    obs = observations_from_merged(results)
    report_path = os.path.join(args.out, "cross_model_report.md")
    with open(report_path, "w") as fh:
        fh.write(build_report(obs))
    log.info("cross_model_report.md: %d observations", len(obs))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "scripts"))
    try:
        from publish_to_hf import build_card

        card = build_card(results, args.repo_id, args.code_url)
        with open(os.path.join(args.out, "README.md"), "w") as fh:
            fh.write(card)
        log.info("README.md (dataset card) written")
    except ImportError:
        log.warning("could not import build_card; run scripts/publish_to_hf.py "
                    "to generate the dataset card")

    log.info("bundle ready in %s -- inspect it, then upload", args.out)
    for name in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, name))
        log.info("   %-28s %8.1f KiB", name, size / 1024)
    return 0


def cmd_merge(args, log) -> int:
    coord = Coordinator(get_db_url(), lease_timeout_s=LEASE_TIMEOUT_S, logger=log)
    results = coord.results()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "merged_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    n = sum(len(v) for v in results.values())
    log.info("merged %d result(s) across %d model(s) -> %s",
             n, len(results), path)

    from coldstart_lab.analysis import build_report, observations_from_merged

    obs = observations_from_merged(results)
    md_path = os.path.join(args.out, "cross_model_report.md")
    with open(md_path, "w") as fh:
        fh.write(build_report(obs))
    log.info("wrote cross-model analysis (%d observations) -> %s", len(obs), md_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coldstart-fleet", description=__doc__)
    p.add_argument("--verbose", action="store_true",
                   help="Keep HF progress bars and per-request HTTP logs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create schema and register tasks")
    pi.add_argument("--tier", default="micro",
                    choices=["ci", "micro", "small", "medium"])
    pi.add_argument("--max-gib", type=float, default=None,
                    help="Skip models whose checkpoint exceeds this size. Use it "
                         "to fit a tier onto a smaller GPU or a smaller disk.")
    pi.add_argument("--family", nargs="+", default=None,
                    help="Restrict to these architecture families.")
    pi.add_argument("--device-class", required=True,
                    help="Label for the hardware this batch targets, e.g. t4/a100/cpu.")
    pi.add_argument("--experiments", nargs="+", default=DISTRIBUTED_EXPERIMENTS,
                    choices=DISTRIBUTED_EXPERIMENTS)
    pi.set_defaults(func=cmd_init)

    pw = sub.add_parser("work", help="claim and run tasks until the queue drains")
    pw.add_argument("--device", default="cuda")
    pw.add_argument("--device-class", required=True)
    pw.add_argument("--repeats", type=int, default=5)
    pw.add_argument("--warmup", type=int, default=1)
    pw.add_argument("--max-tasks", type=int, default=None)
    pw.add_argument("--out-root", default="/content/stage")
    pw.add_argument("--no-wait", action="store_true",
                    help="Exit as soon as nothing is claimable instead of "
                         "waiting for in-flight tasks to finish or be released.")
    pw.add_argument("--poll-interval", type=int, default=30,
                    help="Seconds between polls while waiting for work.")
    pw.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    pw.set_defaults(func=cmd_work)

    ps = sub.add_parser("status", help="show queue progress")
    ps.add_argument("--watch", action="store_true")
    ps.add_argument("--interval", type=int, default=15)
    ps.set_defaults(func=cmd_status)

    pr = sub.add_parser("retry", help="return FAILED tasks to the pending pool")
    pr.set_defaults(func=cmd_retry)

    pe = sub.add_parser("export",
                        help="pull EVERYTHING from the ledger into an "
                             "upload-ready bundle")
    pe.add_argument("--out", default="./hf_upload")
    pe.add_argument("--repo-id", default="your-username/llm-cold-start-benchmark",
                    help="Used only to fill the citation in the dataset card.")
    pe.add_argument("--code-url", default=None)
    pe.set_defaults(func=cmd_export)

    pm = sub.add_parser("merge", help="pull all finished results into one JSON")
    pm.add_argument("--out", default="./merged")
    pm.set_defaults(func=cmd_merge)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _quiet_third_party(getattr(args, "verbose", False))
    log = _logger()
    try:
        return args.func(args, log)
    except RuntimeError as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
