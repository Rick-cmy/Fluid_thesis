#!/usr/bin/env python3
"""
Smoke test for parallel Pipeline and Debate coordinators.
Runs 3 segments through each coordinator and reports latency + token counts.

Usage:
    cd first_experiment
    python scripts/smoke_test_parallel.py [--model qwen3:8b] [--provider ollama]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from data.loader import load_segments, load_termbase
from data.splitter import split_novel_vs_tm, dev_test_split
from data.injector import build_eval_set
from rag.no_rag import NoRAG
from coordination.single_agent import SingleAgent
from coordination.pipeline import Pipeline
from coordination.debate import Debate


def run_smoke(coordinator, name, segments, model, provider):
    print(f"\n{'='*50}")
    print(f"Coordinator: {name}  |  model={model}  |  provider={provider}")
    print(f"{'='*50}")
    retriever = NoRAG()
    for i, seg in enumerate(segments, 1):
        rag_context = retriever.format_context(retriever.retrieve(seg.source, seg.draft))
        t0 = time.monotonic()
        result = coordinator.run(
            source=seg.source,
            draft=seg.draft,
            rag_context=rag_context,
            model=model,
        )
        elapsed = time.monotonic() - t0
        issues = [d for d in result.dimension_results if d.has_issue]
        print(
            f"  [{i}] latency={elapsed:.1f}s | "
            f"tokens={result.prompt_tokens+result.completion_tokens} | "
            f"issues_found={len(issues)} | "
            f"error_type={seg.error_type.value}"
        )
        if result.final_recommendation != seg.draft:
            print(f"       draft   : {seg.draft[:80]}")
            print(f"       revised : {result.final_recommendation[:80]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "deepseek"])
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    with open(base / "config" / "grid.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    master = load_segments(base / cfg["data"]["sentence_pairs_path"])
    q3 = load_segments(base / cfg["data"]["q3_segments_path"])
    q1 = load_segments(base / cfg["data"]["q1_segments_path"])
    termbase = load_termbase(base / cfg["rag"]["termbase_path"])

    tm_split = split_novel_vs_tm(master, q3 + q1)
    dt = dev_test_split(tm_split.novel, dev_ratio=0.2, seed=42)

    # Use dev set for smoke testing (not test set)
    eval_segs = build_eval_set(dt.dev, termbase, n_per_type=1, seed=0)
    sample = eval_segs[:3]
    print(f"Smoke test: {len(sample)} segments from DEV split\n")

    kwargs = dict(provider=args.provider, base_url=args.ollama_url)

    coordinators = [
        (SingleAgent(**kwargs), "SingleAgent (1 call)"),
        (Pipeline(**kwargs),    "Pipeline   (7 calls, parallel)"),
        (Debate(**kwargs),      "Debate     (8 calls, parallel Round1 + coordinator)"),
    ]

    for coord, name in coordinators:
        run_smoke(coord, name, sample, args.model, args.provider)

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
