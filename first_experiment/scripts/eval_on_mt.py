#!/usr/bin/env python3
"""
eval_on_mt.py — Evaluate QA system on real MT error pairs (silver-standard labels).

Validates how well the QA system performs on real MT errors vs injected errors,
enabling the rich_rag circularity check: compare style_guide F1 here vs in grid.

Reads:  data/mt_eval/labelled_*{label_suffix}.jsonl
Writes: results/mt_eval/{rag}__{coord}{label_suffix}.jsonl  (per-segment)
        results/mt_eval/{rag}__{coord}{label_suffix}_summary.json

Usage:
    cd first_experiment
    python scripts/eval_on_mt.py \\
        --rag no_rag --coord single_agent --model qwen3:8b \\
        [--label-suffix _deepseek] [--provider ollama] [--limit 200]

    # Rich-RAG circularity check (key experiment):
    python scripts/eval_on_mt.py \\
        --rag rich --coord single_agent --model deepseek-chat --provider deepseek \\
        --label-suffix _deepseek --error-types style_guide
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from coordination.base import DimensionResult, ReviewResult
from data.injector import ErrorType
from eval.metrics import CellMetrics, ErrorTypeMetrics, accumulate
from eval.cost import CostRecord
from rag.no_rag import NoRAG
from rag.term_rag import TermRAG
from rag.rich_rag import RichRAG
from coordination.single_agent import SingleAgent
from coordination.pipeline import Pipeline
from coordination.debate import Debate

THESIS_ROOT = Path(__file__).resolve().parents[2]
MT_DIR = THESIS_ROOT / "data" / "mt_eval"

_LABEL_FILES = [
    "labelled_q3_2025{suffix}.jsonl",
    "labelled_q1_2026{suffix}.jsonl",
]

_VALID_ERROR_TYPES = {e.value for e in ErrorType} | {"none"}


# ── Noise filter (same thresholds as filter_noise.py) ────────────────────────

def _passes_noise_filter(rec: dict, min_jac: float, min_char: float) -> bool:
    jac = rec.get("word_jac", 1.0)
    ratio = rec.get("char_ratio", 1.0)
    return jac >= min_jac or ratio >= min_char


# ── Load silver labels ────────────────────────────────────────────────────────

def load_silver_labels(
    label_suffix: str,
    error_types: list[str] | None,
    min_jac: float,
    min_char: float,
    limit: int,
) -> list[dict]:
    records: list[dict] = []
    for tmpl in _LABEL_FILES:
        path = MT_DIR / tmpl.format(suffix=label_suffix)
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        n_loaded, n_skipped = 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                et = rec.get("error_type", "none")
                if et not in _VALID_ERROR_TYPES:
                    n_skipped += 1
                    continue
                if not _passes_noise_filter(rec, min_jac, min_char):
                    n_skipped += 1
                    continue
                if error_types and et not in error_types and et != "none":
                    continue
                records.append(rec)
                n_loaded += 1
        print(f"  {path.name}: {n_loaded} loaded, {n_skipped} skipped (noise/unknown type)")

    if limit:
        records = records[:limit]
    print(f"  Total: {len(records)} records")
    return records


# ── Wire retriever ────────────────────────────────────────────────────────────

def build_retriever(rag: str, base: Path, cfg: dict):
    termbase_path = base / cfg["rag"]["termbase_path"]
    from data.loader import load_termbase
    termbase = load_termbase(termbase_path)

    if rag == "no_rag":
        return NoRAG()
    if rag == "vector":
        return TermRAG(termbase)
    if rag == "rich":
        style_rules_path = base / cfg["rag"]["style_rules_path"]
        k_rules = cfg["rag"].get("k_rules", 3)
        return RichRAG(termbase, style_rules_path, k_rules=k_rules)
    raise ValueError(f"Unknown RAG level: {rag}")


def build_coordinator(coord: str, provider: str, base_url: str):
    if coord == "single_agent":
        return SingleAgent(provider=provider, base_url=base_url)
    if coord == "pipeline":
        return Pipeline(provider=provider, base_url=base_url)
    if coord == "debate":
        return Debate(provider=provider, base_url=base_url)
    raise ValueError(f"Unknown coord level: {coord}")


# ── Metrics tracking for "none" segments (true negatives) ────────────────────

class NoneMetrics:
    """Tracks FP/TN for segments with no real error (silver label = 'none')."""
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0

    def update(self, any_flagged: bool) -> None:
        if any_flagged:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def fpr(self) -> float:
        total = self.fp + self.tn
        return self.fp / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {"fp": self.fp, "tn": self.tn, "false_positive_rate": round(self.fpr, 4)}


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval(
    records: list[dict],
    retriever,
    coordinator,
    model: str,
    rag: str,
    coord: str,
    label_suffix: str,
    out_dir: Path,
    rag_k: int = 5,
) -> tuple[CellMetrics, NoneMetrics, CostRecord]:
    cell = CellMetrics(rag_level=rag, coord_level=coord, trial=0)
    none_metrics = NoneMetrics()
    cost = CostRecord(rag_level=rag, coord_level=coord, trial=0)

    out_path = out_dir / f"{rag}__{coord}{label_suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_uuids: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    uuid = rec.get("uuid", "")
                    if not uuid:
                        continue
                    et_str = rec.get("error_type", "none")
                    if et_str != "none":
                        et = ErrorType(et_str)
                        dim_results = [
                            DimensionResult(
                                agent=d["agent"], has_issue=d["has_issue"],
                                severity=d["severity"], issue_span=d.get("issue_span", ""),
                                suggested_revision=d.get("suggested_revision", ""),
                                explanation="", confidence=0.0,
                            )
                            for d in rec.get("dimension_results", [])
                        ]
                        fake = ReviewResult(
                            final_recommendation=rec.get("final_recommendation", ""),
                            dimension_results=dim_results,
                            prompt_tokens=rec.get("prompt_tokens", 0),
                            completion_tokens=rec.get("completion_tokens", 0),
                            latency_s=rec.get("latency_s", 0.0),
                        )
                        accumulate(cell, et, fake, has_real_issue=True)
                    else:
                        any_flagged = rec.get("any_flagged", False)
                        none_metrics.update(any_flagged)
                    cost.add(rec.get("prompt_tokens", 0), rec.get("completion_tokens", 0),
                             rec.get("latency_s", 0.0))
                    done_uuids.add(uuid)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        print(f"  [resume] {len(done_uuids)}/{len(records)} already done")

    file_mode = "a" if done_uuids else "w"
    n_done = len(done_uuids)
    t0 = time.monotonic()

    with open(out_path, file_mode, encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            uuid = rec.get("uuid", f"rec_{i}")
            if uuid in done_uuids:
                continue

            et_str = rec.get("error_type", "none")
            source = rec.get("source_sv", "")
            draft = rec.get("mt_en", "")

            hits = retriever.retrieve(source, draft, k=rag_k)
            rag_context = retriever.format_context(hits)

            result: ReviewResult = coordinator.run(
                source=source, draft=draft,
                source_lang="Swedish", target_lang="English",
                domain="IFRS financial reporting",
                rag_context=rag_context, model=model,
            )

            any_flagged = any(
                d.has_issue and d.severity != "none"
                for d in result.dimension_results
            )

            if et_str != "none":
                et = ErrorType(et_str)
                accumulate(cell, et, result, has_real_issue=True)
            else:
                none_metrics.update(any_flagged)

            cost.add(result.prompt_tokens, result.completion_tokens, result.latency_s)

            record: dict[str, Any] = {
                "uuid": uuid,
                "error_type": et_str,
                "source_sv": source,
                "mt_en": draft,
                "silver_label": et_str,
                "any_flagged": any_flagged,
                "final_recommendation": result.final_recommendation,
                "dimension_results": [
                    {"agent": d.agent, "has_issue": d.has_issue, "severity": d.severity,
                     "issue_span": d.issue_span, "suggested_revision": d.suggested_revision}
                    for d in result.dimension_results
                ],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": round(result.latency_s, 3),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1

            if n_done % 50 == 0 or n_done == len(records):
                elapsed = time.monotonic() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (len(records) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{len(records)}  "
                      f"macro_F1={cell.macro_f1:.3f}  "
                      f"FPR={none_metrics.fpr:.3f}  "
                      f"({rate:.2f}/s, ~{remaining/60:.0f}m left)")

    return cell, none_metrics, cost


# ── Console table ─────────────────────────────────────────────────────────────

def print_results(cell: CellMetrics, none_m: NoneMetrics, cost: CostRecord,
                  rag: str, coord: str, label_suffix: str) -> None:
    print(f"\n{'='*60}")
    print(f"MT Eval Results — {rag} × {coord}  (labels: {label_suffix or 'qwen3'})")
    print(f"{'='*60}")
    print(f"{'error_type':<18} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FN':>5}")
    print("-" * 50)
    for et in sorted(cell.per_type):
        m = cell.per_type[et]
        print(f"{et:<18} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} "
              f"{m.tp:>5} {m.fn:>5}")
    print("-" * 50)
    print(f"{'MACRO F1':<18} {'':>6} {'':>6} {cell.macro_f1:>6.3f}")
    print(f"\nTrue negatives (none-type): FP={none_m.fp}, TN={none_m.tn}, "
          f"FPR={none_m.fpr:.3f}")
    print(f"Cost: {cost.total_tokens} tokens, mean latency {cost.latency_mean_s:.1f}s/seg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA on real MT errors")
    parser.add_argument("--rag", default="no_rag", choices=["no_rag", "vector", "rich"])
    parser.add_argument("--coord", default="single_agent",
                        choices=["single_agent", "pipeline", "debate"])
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "deepseek"])
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--label-suffix", default="_deepseek",
                        help="Suffix on label files, e.g. '_deepseek' or '' for qwen3:8b")
    parser.add_argument("--error-types", nargs="+", default=None,
                        help="Filter to specific error types, e.g. --error-types style_guide")
    parser.add_argument("--limit", type=int, default=0, help="Max records (0=all)")
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--min-jaccard", type=float, default=0.05)
    parser.add_argument("--min-char-ratio", type=float, default=0.15)
    parser.add_argument("--out-dir", default="results/mt_eval")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    with open(base / "config" / "grid.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"Config: rag={args.rag}  coord={args.coord}  model={args.model} "
          f"({args.provider})  labels={args.label_suffix or 'qwen3'}")

    print("\nLoading silver labels...")
    records = load_silver_labels(
        label_suffix=args.label_suffix,
        error_types=args.error_types,
        min_jac=args.min_jaccard,
        min_char=args.min_char_ratio,
        limit=args.limit,
    )
    if not records:
        print("No records to evaluate.")
        sys.exit(1)

    dist: dict[str, int] = {}
    for r in records:
        et = r.get("error_type", "none")
        dist[et] = dist.get(et, 0) + 1
    print(f"  Distribution: {dict(sorted(dist.items()))}")

    print("\nBuilding retriever...")
    retriever = build_retriever(args.rag, base, cfg)

    print("Building coordinator...")
    coordinator = build_coordinator(args.coord, args.provider, args.ollama_url)

    out_dir = base / args.out_dir
    print(f"\nRunning evaluation → {out_dir}/")

    cell, none_metrics, cost = run_eval(
        records=records,
        retriever=retriever,
        coordinator=coordinator,
        model=args.model,
        rag=args.rag,
        coord=args.coord,
        label_suffix=args.label_suffix,
        out_dir=out_dir,
        rag_k=args.rag_k,
    )

    print_results(cell, none_metrics, cost, args.rag, args.coord, args.label_suffix)

    summary_path = out_dir / f"{args.rag}__{args.coord}{args.label_suffix}_summary.json"
    summary = {
        **cell.to_dict(),
        "none_metrics": none_metrics.to_dict(),
        "cost": cost.to_dict(),
        "config": {
            "label_suffix": args.label_suffix,
            "model": args.model,
            "provider": args.provider,
            "n_records": len(records),
            "error_types_filter": args.error_types,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    main()
