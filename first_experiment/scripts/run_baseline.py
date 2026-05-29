#!/usr/bin/env python3
"""
June milestone — Task 1: single_agent × no_rag baseline (grid cell C1).

Runs the no-RAG single-agent reviewer on the evaluation set,
computes per-error-type F1, and writes results to results/baseline/.

Usage:
    cd first_experiment
    python scripts/run_baseline.py [--model qwen3:8b] [--n-per-type 200] [--seed 42]
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
from rag.no_rag import NoRAG
from coordination.single_agent import SingleAgent
from runner import run_cell


def main():
    parser = argparse.ArgumentParser(description="Run C1 baseline: no_rag × single_agent")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--n-per-type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "grid.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(__file__).parent.parent
    data_root = base / cfg["data"]["sentence_pairs_path"]

    print("Loading data...")
    master_segments = load_segments(base / cfg["data"]["sentence_pairs_path"])
    q3_segments = load_segments(base / cfg["data"]["q3_segments_path"])
    q1_segments = load_segments(base / cfg["data"]["q1_segments_path"])
    termbase = load_termbase(base / cfg["rag"]["termbase_path"])

    print(f"  Loaded {len(master_segments)} master segments, {len(termbase)} terms")

    print("Splitting novel vs TM-matched segments...")
    quarterly = q3_segments + q1_segments
    tm_split = split_novel_vs_tm(master_segments, quarterly)
    print(f"  {tm_split.summary()}")

    split_cfg = cfg.get("split", {})
    dev_ratio = split_cfg.get("dev_ratio", 0.2)
    split_seed = split_cfg.get("seed", 42)
    dt_split = dev_test_split(tm_split.novel, dev_ratio=dev_ratio, seed=split_seed)
    print(f"  Dev/Test split: {dt_split.summary()}")
    print(f"  *** Using TEST split only for evaluation ***")

    print(f"Building evaluation set ({args.n_per_type} per error type, from test split)...")
    eval_segments = build_eval_set(
        segments=dt_split.test,
        termbase=termbase,
        n_per_type=args.n_per_type,
        seed=args.seed,
    )
    print(f"  Eval set: {len(eval_segments)} total segments")

    results_dir = base / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning baseline: no_rag × single_agent, model={args.model}")
    retriever = NoRAG()
    coordinator = SingleAgent(base_url=args.ollama_url)

    metrics, cost = run_cell(
        rag_level="no_rag",
        coord_level="single_agent",
        trial=1,
        retriever=retriever,
        coordinator=coordinator,
        eval_segments=eval_segments,
        model=args.model,
        results_dir=results_dir,
    )

    print("\n=== BASELINE RESULTS ===")
    print(f"Macro F1: {metrics.macro_f1:.4f}")
    print(f"Total tokens: {cost.total_tokens:,}")
    print(f"Mean latency: {cost.latency_mean_s:.2f}s/segment")
    print(f"P95 latency: {cost.latency_p95_s:.2f}s/segment")
    print("\nPer error type:")
    for et, m in metrics.per_type.items():
        print(f"  {et:20s}  P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}  "
              f"critical_catch={m.critical_catch_rate:.3f}")

    summary_path = results_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": metrics.to_dict(),
            "cost": cost.to_dict(),
        }, f, indent=2)
    print(f"\nResults written to {summary_path}")


if __name__ == "__main__":
    main()
