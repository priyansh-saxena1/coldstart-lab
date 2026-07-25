"""Report generation.

Collects experiment results plus the system fingerprint into a single JSON
artifact and a human-readable Markdown summary. Plotting is optional and lazy:
if matplotlib is present we emit bar charts, otherwise the Markdown tables carry
the full story on their own. Keeping the report self-contained matters because
it is the actual deliverable that goes to a reviewer.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from coldstart_lab.environment import SystemFingerprint
from coldstart_lab.experiments.base import ExperimentResult


class Report:
    def __init__(self, fingerprint: SystemFingerprint) -> None:
        self.fingerprint = fingerprint
        self.experiments: List[ExperimentResult] = []
        self.extras: Dict[str, object] = {}

    def add(self, result: ExperimentResult) -> None:
        self.experiments.append(result)

    def add_extra(self, key: str, value: object) -> None:
        self.extras[key] = value

    def to_dict(self) -> dict:
        return {
            "system": self.fingerprint.to_dict(),
            "experiments": [e.to_dict() for e in self.experiments],
            "extras": self.extras,
        }

    def write_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    def write_markdown(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lines: List[str] = ["# Cold-start benchmark report", ""]
        lines += self._system_section()
        for exp in self.experiments:
            lines += self._experiment_section(exp)
        if self.extras:
            lines += self._extras_section()
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    # -- section builders ---------------------------------------------------

    def _system_section(self) -> List[str]:
        fp = self.fingerprint
        gpu = ", ".join(f"{g.name} ({g.memory_total_mib} MiB)" for g in fp.gpus) or "none"
        out = [
            "## Environment",
            "",
            f"- Python: {fp.python_version}",
            f"- Platform: {fp.platform}",
            f"- CPU count: {fp.cpu_count}",
            f"- RAM: {fp.total_ram_gib} GiB",
            f"- Torch: {fp.torch_version}",
            f"- CUDA available: {fp.cuda_available}",
            f"- GPU(s): {gpu}",
            f"- Page-cache control: {fp.page_cache_control}",
            "",
        ]
        if fp.warnings:
            out.append("**Warnings:**")
            out += [f"- {w}" for w in fp.warnings]
            out.append("")
        return out

    def _experiment_section(self, exp: ExperimentResult) -> List[str]:
        out = [f"## Experiment: `{exp.name}`", ""]
        summary = exp.summary("total_ms")
        if not summary:
            out += ["_No total_ms metric recorded._", ""]
            return out
        out += [
            "| Condition | n | mean (ms) | p50 (ms) | p95 (ms) | stdev (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cond, stats in sorted(summary.items(), key=lambda kv: kv[1]["p50"]):
            out.append(
                f"| {cond} | {int(stats['n'])} | {stats['mean']} | "
                f"{stats['p50']} | {stats['p95']} | {stats['stdev']} |"
            )
        out.append("")

        # Relative speedup vs the slowest condition, which is the headline a
        # reviewer scans for.
        by_p50 = {c: s["p50"] for c, s in summary.items()}
        slowest = max(by_p50, key=by_p50.get)
        base = by_p50[slowest]
        if base > 0:
            out += ["Relative to slowest condition (p50):", ""]
            for cond, p50 in sorted(by_p50.items(), key=lambda kv: kv[1]):
                speedup = base / p50 if p50 > 0 else float("inf")
                out.append(f"- `{cond}`: {speedup:.2f}x")
            out.append("")
        return out

    def _extras_section(self) -> List[str]:
        out = ["## Projections & notes", ""]
        for key, value in self.extras.items():
            out.append(f"### {key}")
            out.append("")
            if isinstance(value, list) and value and isinstance(value[0], dict):
                headers = list(value[0].keys())
                out.append("| " + " | ".join(headers) + " |")
                out.append("|" + "|".join("---" for _ in headers) + "|")
                for row in value:
                    out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
            else:
                out.append("```json")
                out.append(json.dumps(value, indent=2, default=str))
                out.append("```")
            out.append("")
        return out


def plot_experiment(
    exp: ExperimentResult, out_path: str, metric: str = "total_ms"
) -> Optional[str]:  # pragma: no cover - depends on matplotlib + display
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    summary = exp.summary(metric)
    if not summary:
        return None

    conds = list(summary.keys())
    p50 = [summary[c]["p50"] for c in conds]
    p95 = [summary[c]["p95"] for c in conds]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(conds))
    ax.bar(x, p50, label="p50")
    ax.bar(x, [b - a for a, b in zip(p50, p95)], bottom=p50, alpha=0.4, label="p50->p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(exp.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
