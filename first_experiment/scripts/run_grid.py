#!/usr/bin/env python3
"""
Full 3×3 grid runner — July+ task.

Runs all implemented (rag_level × coord_level) combinations.
Skips cells whose coordinator/retriever raises NotImplementedError.

Usage:
    cd first_experiment
    python scripts/run_grid.py [--model qwen3:8b] [--n-per-type 200] [--trials 3]
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
from rag.vector_rag import VectorRAG
from rag.hybrid_rag import HybridRAG
from coordination.single_agent import SingleAgent
from coordination.pipeline import Pipeline
from coordination.debate import Debate
from runner import run_grid


def main():
    parser = argparse.ArgumentParser(description="Run full 3×3 experiment grid")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--n-per-type", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    with open(base / "config" / "grid.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    master_segments = load_segments(base / cfg["data"]["sentence_pairs_path"])
    q3_segments = load_segments(base / cfg["data"]["q3_segments_path"])
    q1_segments = load_segments(base / cfg["data"]["q1_segments_path"])
    termbase = load_termbase(base / cfg["rag"]["termbase_path"])

    tm_split = split_novel_vs_tm(master_segments, q3_segments + q1_segments)
    print(tm_split.summary())

    split_cfg = cfg.get("split", {})
    dt_split = dev_test_split(
        tm_split.novel,
        dev_ratio=split_cfg.get("dev_ratio", 0.2),
        seed=split_cfg.get("seed", 42),
    )
    print(f"Dev/Test split: {dt_split.summary()}")
    print("*** Using TEST split only for grid evaluation ***")

    eval_segments = build_eval_set(dt_split.test, termbase, n_per_type=args.n_per_type, seed=args.seed)
    print(f"Eval set: {len(eval_segments)} segments")

    # Wire up retrievers — skip NotImplementedError ones
    retrievers = {"no_rag": NoRAG()}
    try:
        retrievers["vector"] = VectorRAG(termbase)
    except NotImplementedError:
        print("VectorRAG not implemented — skipping vector cells.")
    try:
        retrievers["hybrid"] = HybridRAG(termbase)
    except NotImplementedError:
        print("HybridRAG not implemented — skipping hybrid cells.")

    coordinators = {"single_agent": SingleAgent(base_url=args.ollama_url)}
    for name, cls in [("pipeline", Pipeline), ("debate", Debate)]:
        try:
            coordinators[name] = cls()
        except NotImplementedError:
            print(f"{name} coordinator not implemented — skipping.")

    results_dir = base / "results" / "grid"
    run_grid(
        grid_config=cfg,
        eval_segments=eval_segments,
        retrievers=retrievers,
        coordinators=coordinators,
        model=args.model,
        results_dir=results_dir,
        n_trials=args.trials,
    )


if __name__ == "__main__":
    main()
