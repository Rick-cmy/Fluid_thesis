#!/usr/bin/env python3
"""
Run grid cell: term_rag × single_agent (trial 1).

Usage:
    cd first_experiment
    python scripts/run_term_rag_single_agent.py [--model qwen3:8b] [--n-per-type 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from data.loader import load_segments, load_termbase
from data.splitter import split_novel_vs_tm, dev_test_split
from data.injector import build_eval_set
from rag.term_rag import TermRAG
from coordination.single_agent import SingleAgent
from runner import run_cell


def main():
    parser = argparse.ArgumentParser(description="Run term_rag × single_agent")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--n-per-type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    with open(base / "config" / "grid.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    master_segments = load_segments(base / cfg["data"]["sentence_pairs_path"])
    q3_segments     = load_segments(base / cfg["data"]["q3_segments_path"])
    q1_segments     = load_segments(base / cfg["data"]["q1_segments_path"])
    termbase        = load_termbase(base / cfg["rag"]["termbase_path"])
    print(f"  {len(master_segments)} master segments, {len(termbase)} terms")

    tm_split = split_novel_vs_tm(master_segments, q3_segments + q1_segments)
    print(f"  {tm_split.summary()}")

    split_cfg = cfg.get("split", {})
    dt_split = dev_test_split(
        tm_split.novel,
        dev_ratio=split_cfg.get("dev_ratio", 0.2),
        seed=split_cfg.get("seed", 42),
    )
    print(f"  Dev/Test: {dt_split.summary()} — using TEST only")

    print(f"Building eval set ({args.n_per_type}/type)...")
    eval_segments = build_eval_set(dt_split.test, termbase, n_per_type=args.n_per_type, seed=args.seed)
    print(f"  {len(eval_segments)} segments")

    print("Building TermRAG index...")
    retriever = TermRAG(termbase)
    coordinator = SingleAgent(base_url=args.ollama_url)

    results_dir = base / "results" / "grid"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning: term_rag × single_agent — trial {args.trial}, model={args.model}")
    metrics, cost = run_cell(
        rag_level="term_rag",
        coord_level="single_agent",
        trial=args.trial,
        retriever=retriever,
        coordinator=coordinator,
        eval_segments=eval_segments,
        model=args.model,
        results_dir=results_dir,
    )

    print("\n=== RESULTS: term_rag × single_agent ===")
    print(f"Macro F1:     {metrics.macro_f1:.4f}")
    print(f"Total tokens: {cost.total_tokens:,}")
    print(f"Mean latency: {cost.latency_mean_s:.2f}s/segment")
    print("\nPer error type:")
    for et, m in metrics.per_type.items():
        print(f"  {et:20s}  P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}")

    summary_path = results_dir / "term_rag__single_agent__summary.json"
    with open(summary_path, "w") as f:
        json.dump({"metrics": metrics.to_dict(), "cost": cost.to_dict()}, f, indent=2)
    print(f"\nResults → {summary_path}")


if __name__ == "__main__":
    main()
