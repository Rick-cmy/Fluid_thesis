#!/usr/bin/env python3
"""
Analyze grid results: load all per-segment JSONL files, recompute metrics,
and output tables + plots.

Usage:
    cd first_experiment
    python scripts/analyze_grid.py [--results-dir results/grid] [--out-dir results/analysis]
                                    [--rag-levels no_rag vector rich]
                                    [--coord-levels single_agent pipeline debate]

Outputs:
    results/analysis/per_type_f1.csv       — full per-(cell, error_type) table
    results/analysis/summary.json          — aggregated metrics (averaged across trials)
    results/analysis/heatmap_f1.png        — heatmap: rag_level × error_type F1
    results/analysis/bar_per_type.png      — grouped bar: error_type × config F1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.injector import ErrorType
from coordination.base import DimensionResult, ReviewResult
from eval.metrics import CellMetrics, accumulate

# Canonical display order
_ERROR_TYPES = [e.value for e in ErrorType]
_RAG_ORDER = ["no_rag", "vector", "rich"]
_COORD_ORDER = ["single_agent", "pipeline", "debate"]

_FILENAME_RE = re.compile(r"^(.+)__(.+)__trial(\d+)\.jsonl$")


# ── Loading ──────────────────────────────────────────────────────────────────

def load_cell(path: Path) -> CellMetrics:
    """Replay a per-segment JSONL file into a CellMetrics object."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")
    rag_level, coord_level, trial = m.group(1), m.group(2), int(m.group(3))

    cell = CellMetrics(rag_level=rag_level, coord_level=coord_level, trial=trial)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                error_type = ErrorType(rec["error_type"])
                dim_results = [
                    DimensionResult(
                        agent=d["agent"],
                        has_issue=d["has_issue"],
                        severity=d["severity"],
                        issue_span=d.get("issue_span", ""),
                        suggested_revision=d.get("suggested_revision", ""),
                        explanation="",
                        confidence=0.0,
                    )
                    for d in rec.get("dimension_results", [])
                ]
                result = ReviewResult(
                    final_recommendation=rec.get("final_recommendation", ""),
                    dimension_results=dim_results,
                    prompt_tokens=rec.get("prompt_tokens", 0),
                    completion_tokens=rec.get("completion_tokens", 0),
                    latency_s=rec.get("latency_s", 0.0),
                )
                accumulate(cell, error_type, result, has_real_issue=True)
                n += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    cell._n_segments = n  # type: ignore[attr-defined]
    return cell


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_trials(cells: list[CellMetrics]) -> dict:
    """Average per-type metrics across trials for one (rag, coord) pair."""
    if not cells:
        return {}
    error_types = set()
    for c in cells:
        error_types.update(c.per_type.keys())

    result: dict = {
        "rag_level": cells[0].rag_level,
        "coord_level": cells[0].coord_level,
        "n_trials": len(cells),
        "macro_f1_mean": float(np.mean([c.macro_f1 for c in cells])),
        "macro_f1_std": float(np.std([c.macro_f1 for c in cells])),
        "per_type": {},
    }
    for et in sorted(error_types):
        f1s = [c.per_type[et].f1 if et in c.per_type else 0.0 for c in cells]
        precs = [c.per_type[et].precision if et in c.per_type else 0.0 for c in cells]
        recs = [c.per_type[et].recall if et in c.per_type else 0.0 for c in cells]
        result["per_type"][et] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "n_segments": sum(
                (c.per_type[et].tp + c.per_type[et].fn) for c in cells if et in c.per_type
            ),
        }
    return result


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(aggregated: list[dict], coord_level: str = "single_agent") -> None:
    rows = [a for a in aggregated if a["coord_level"] == coord_level]
    if not rows:
        return
    rows_by_rag = {r["rag_level"]: r for r in rows}

    rag_levels = [r for r in _RAG_ORDER if r in rows_by_rag]
    error_types = [et for et in _ERROR_TYPES if any(et in r["per_type"] for r in rows)]

    col_w = 10
    header = f"{'error_type':<18}" + "".join(f"{rl:>{col_w}}" for rl in rag_levels)
    print(f"\n=== F1 by error type × RAG level (coord={coord_level}) ===")
    print(header)
    print("-" * len(header))
    for et in error_types:
        row = f"{et:<18}"
        for rl in rag_levels:
            r = rows_by_rag.get(rl, {})
            f1 = r.get("per_type", {}).get(et, {}).get("f1_mean", float("nan"))
            n = r.get("per_type", {}).get(et, {}).get("n_segments", 0)
            row += f"  {f1:.3f}({'?' if n == 0 else n:>4})"
        print(row)
    print("-" * len(header))
    macro_row = f"{'MACRO F1':<18}"
    for rl in rag_levels:
        r = rows_by_rag.get(rl, {})
        f1 = r.get("macro_f1_mean", float("nan"))
        macro_row += f"  {f1:.3f}      "
    print(macro_row)


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(aggregated: list[dict], out_path: Path) -> None:
    import csv
    rows = []
    for agg in aggregated:
        rl, cl = agg["rag_level"], agg["coord_level"]
        for et, metrics in agg["per_type"].items():
            rows.append({
                "rag_level": rl,
                "coord_level": cl,
                "n_trials": agg["n_trials"],
                "error_type": et,
                "f1_mean": round(metrics["f1_mean"], 4),
                "f1_std": round(metrics["f1_std"], 4),
                "precision_mean": round(metrics["precision_mean"], 4),
                "recall_mean": round(metrics["recall_mean"], 4),
                "n_segments": metrics["n_segments"],
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {out_path}")


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(aggregated: list[dict], out_path: Path, coord_level: str = "single_agent") -> None:
    rows = [a for a in aggregated if a["coord_level"] == coord_level]
    if not rows:
        return
    rag_levels = [r for r in _RAG_ORDER if any(a["rag_level"] == r for a in rows)]
    error_types = [et for et in _ERROR_TYPES if any(et in r["per_type"] for r in rows)]

    matrix = np.full((len(rag_levels), len(error_types)), np.nan)
    rows_by_rag = {r["rag_level"]: r for r in rows}
    for i, rl in enumerate(rag_levels):
        for j, et in enumerate(error_types):
            f1 = rows_by_rag.get(rl, {}).get("per_type", {}).get(et, {}).get("f1_mean", np.nan)
            matrix[i, j] = f1

    fig, ax = plt.subplots(figsize=(10, max(3, len(rag_levels) * 1.2)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="F1")

    ax.set_xticks(range(len(error_types)))
    ax.set_xticklabels(error_types, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rag_levels)))
    ax.set_yticklabels(rag_levels, fontsize=9)

    for i in range(len(rag_levels)):
        for j in range(len(error_types)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < v < 0.8 else "white")

    ax.set_title(f"F1 — RAG level × error type  (coord={coord_level})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap written: {out_path}")


# ── Grouped bar ───────────────────────────────────────────────────────────────

def plot_bar(aggregated: list[dict], out_path: Path, coord_level: str = "single_agent") -> None:
    rows = [a for a in aggregated if a["coord_level"] == coord_level]
    if not rows:
        return
    rag_levels = [r for r in _RAG_ORDER if any(a["rag_level"] == r for a in rows)]
    error_types = [et for et in _ERROR_TYPES if any(et in r["per_type"] for r in rows)]
    rows_by_rag = {r["rag_level"]: r for r in rows}

    x = np.arange(len(error_types))
    width = 0.8 / len(rag_levels)
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for k, rl in enumerate(rag_levels):
        f1s = [rows_by_rag.get(rl, {}).get("per_type", {}).get(et, {}).get("f1_mean", 0.0)
               for et in error_types]
        errs = [rows_by_rag.get(rl, {}).get("per_type", {}).get(et, {}).get("f1_std", 0.0)
                for et in error_types]
        offset = (k - len(rag_levels) / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=rl, color=colors[k % len(colors)],
               yerr=errs, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(error_types, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-type F1 by RAG level  (coord={coord_level})", fontsize=11)
    ax.legend(title="RAG level", fontsize=8)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart written: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze grid results")
    parser.add_argument("--results-dir", default="results/grid")
    parser.add_argument("--out-dir", default="results/analysis")
    parser.add_argument("--rag-levels", nargs="+", default=None)
    parser.add_argument("--coord-levels", nargs="+", default=None)
    parser.add_argument("--coord-for-plots", default="single_agent",
                        help="Coord level to use for the RAG-axis heatmap/bar (default: single_agent)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    results_dir = base / args.results_dir
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all cells ──────────────────────────────────────────────────────
    paths = sorted(results_dir.glob("*__trial*.jsonl"))
    if not paths:
        print(f"No result files found in {results_dir}")
        sys.exit(1)

    cells: list[CellMetrics] = []
    for p in paths:
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        rag, coord = m.group(1), m.group(2)
        if args.rag_levels and rag not in args.rag_levels:
            continue
        if args.coord_levels and coord not in args.coord_levels:
            continue
        if p.stat().st_size == 0:
            print(f"  [skip] {p.name} (empty)")
            continue
        try:
            cell = load_cell(p)
            n = getattr(cell, "_n_segments", "?")
            print(f"  loaded {p.name}  ({n} segs, macro_F1={cell.macro_f1:.3f})")
            cells.append(cell)
        except Exception as e:
            print(f"  [error] {p.name}: {e}")

    if not cells:
        print("No cells loaded — nothing to analyze.")
        sys.exit(1)

    # ── Aggregate across trials ─────────────────────────────────────────────
    grouped: dict[tuple, list[CellMetrics]] = defaultdict(list)
    for c in cells:
        grouped[(c.rag_level, c.coord_level)].append(c)

    aggregated = [aggregate_trials(trial_cells) for trial_cells in grouped.values()]
    aggregated.sort(key=lambda a: (
        _COORD_ORDER.index(a["coord_level"]) if a["coord_level"] in _COORD_ORDER else 99,
        _RAG_ORDER.index(a["rag_level"]) if a["rag_level"] in _RAG_ORDER else 99,
    ))

    # ── Console output ──────────────────────────────────────────────────────
    coord_levels_present = list({a["coord_level"] for a in aggregated})
    for cl in _COORD_ORDER:
        if cl in coord_levels_present:
            print_table(aggregated, coord_level=cl)

    # ── Write outputs ───────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON written: {summary_path}")

    if aggregated and any(a["per_type"] for a in aggregated):
        write_csv(aggregated, out_dir / "per_type_f1.csv")
        plot_heatmap(aggregated, out_dir / "heatmap_f1.png", coord_level=args.coord_for_plots)
        plot_bar(aggregated, out_dir / "bar_per_type.png", coord_level=args.coord_for_plots)

    print("\nDone.")


if __name__ == "__main__":
    main()
