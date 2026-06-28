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
from data.injector import build_eval_set, build_clean_set
from rag.no_rag import NoRAG
from rag.term_rag import TermRAG
from rag.rich_rag import RichRAG
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
    parser.add_argument("--provider", default="ollama", choices=["ollama", "deepseek"])
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument(
        "--coord-levels",
        nargs="+",
        default=None,
        help="Override coordination levels from grid.yaml (e.g. --coord-levels single_agent)",
    )
    parser.add_argument(
        "--rag-levels",
        nargs="+",
        default=None,
        help="Override RAG levels from grid.yaml (e.g. --rag-levels no_rag vector rich)",
    )
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

    eval_segments, used_ids = build_eval_set(dt_split.test, termbase, n_per_type=args.n_per_type, seed=args.seed)
    print(f"Eval set: {len(eval_segments)} injected segments")
    clean_segments = build_clean_set(dt_split.test, n_clean=args.n_per_type, seed=args.seed, exclude_ids=used_ids)
    print(f"Clean set: {len(clean_segments)} segments")

    # Wire up retrievers
    retrievers = {"no_rag": NoRAG()}
    try:
        retrievers["term_rag"] = TermRAG(termbase)
    except NotImplementedError:
        print("TermRAG not implemented — skipping term_rag cells.")
    try:
        style_rules_path = base / cfg["rag"]["style_rules_path"]
        k_rules = cfg["rag"].get("k_rules", 3)
        retrievers["rich_rag"] = RichRAG(termbase, style_rules_path, k_rules=k_rules)
    except NotImplementedError:
        print("RichRAG not implemented — skipping rich_rag cells.")

    provider = args.provider
    coordinators = {"single_agent": SingleAgent(provider=provider, base_url=args.ollama_url)}
    for name, cls in [("pipeline", Pipeline), ("debate", Debate)]:
        try:
            coordinators[name] = cls(provider=provider)
        except NotImplementedError:
            print(f"{name} coordinator not implemented — skipping.")

    if args.coord_levels:
        cfg["coordination_levels"] = args.coord_levels
    if args.rag_levels:
        cfg["rag_levels"] = args.rag_levels

    effective_model = args.deepseek_model if provider == "deepseek" else args.model

    results_dir = base / "results" / "grid"
    run_grid(
        grid_config=cfg,
        eval_segments=eval_segments,
        retrievers=retrievers,
        coordinators=coordinators,
        model=effective_model,
        results_dir=results_dir,
        n_trials=args.trials,
        clean_segments=clean_segments,
    )


if __name__ == "__main__":
    main()
