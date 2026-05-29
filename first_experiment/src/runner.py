from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from data.injector import EvalSegment, ErrorType
from coordination.base import Coordinator, ReviewResult
from rag.base import RAGRetriever
from eval.metrics import CellMetrics, accumulate
from eval.cost import CostRecord


def run_cell(
    rag_level: str,
    coord_level: str,
    trial: int,
    retriever: RAGRetriever,
    coordinator: Coordinator,
    eval_segments: list[EvalSegment],
    model: str,
    results_dir: str | Path,
    rag_k: int = 5,
    domain: str = "IFRS financial reporting",
    source_lang: str = "Swedish",
    target_lang: str = "English",
) -> tuple[CellMetrics, CostRecord]:
    """
    Runs one cell of the 3×3 grid: (rag_level × coord_level), one trial.
    Writes per-segment results to results_dir/{rag_level}_{coord_level}_trial{n}.jsonl
    Returns CellMetrics and CostRecord for this cell.
    """
    cell_metrics = CellMetrics(rag_level=rag_level, coord_level=coord_level, trial=trial)
    cost_record = CostRecord(rag_level=rag_level, coord_level=coord_level, trial=trial)

    out_path = Path(results_dir) / f"{rag_level}__{coord_level}__trial{trial:02d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for seg in eval_segments:
            # Retrieve RAG context
            hits = retriever.retrieve(seg.source, seg.draft, k=rag_k)
            rag_context = retriever.format_context(hits)

            # Run coordinator
            result: ReviewResult = coordinator.run(
                source=seg.source,
                draft=seg.draft,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                rag_context=rag_context,
                model=model,
            )

            # Accumulate metrics
            accumulate(cell_metrics, seg.error_type, result, has_real_issue=True)
            cost_record.add(result.prompt_tokens, result.completion_tokens, result.latency_s)

            # Write per-segment record
            record: dict[str, Any] = {
                "segment_id": seg.segment_id,
                "error_type": seg.error_type.value,
                "injected_span": seg.injected_span,
                "error_description": seg.error_description,
                "final_recommendation": result.final_recommendation,
                "dimension_results": [
                    {
                        "agent": d.agent,
                        "has_issue": d.has_issue,
                        "severity": d.severity,
                        "issue_span": d.issue_span,
                        "suggested_revision": d.suggested_revision,
                    }
                    for d in result.dimension_results
                ],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": round(result.latency_s, 3),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return cell_metrics, cost_record


def run_grid(
    grid_config: dict,
    eval_segments: list[EvalSegment],
    retrievers: dict[str, RAGRetriever],
    coordinators: dict[str, Coordinator],
    model: str,
    results_dir: str | Path,
    n_trials: int = 3,
) -> list[dict]:
    """
    Iterates the full 3×3 × n_trials grid. Returns summary list.
    """
    summary = []
    rag_levels = grid_config.get("rag_levels", ["no_rag"])
    coord_levels = grid_config.get("coordination_levels", ["single_agent"])

    for rag_level in rag_levels:
        if rag_level not in retrievers:
            print(f"Skipping RAG level '{rag_level}' — not implemented yet.")
            continue
        for coord_level in coord_levels:
            if coord_level not in coordinators:
                print(f"Skipping coordination level '{coord_level}' — not implemented yet.")
                continue
            for trial in range(1, n_trials + 1):
                print(f"Running: {rag_level} × {coord_level} — trial {trial}/{n_trials}")
                metrics, cost = run_cell(
                    rag_level=rag_level,
                    coord_level=coord_level,
                    trial=trial,
                    retriever=retrievers[rag_level],
                    coordinator=coordinators[coord_level],
                    eval_segments=eval_segments,
                    model=model,
                    results_dir=results_dir,
                )
                row = {**metrics.to_dict(), **cost.to_dict()}
                summary.append(row)
                print(f"  macro_F1={metrics.macro_f1:.3f} | tokens={cost.total_tokens} | "
                      f"latency_mean={cost.latency_mean_s:.2f}s")

    # Write summary
    summary_path = Path(results_dir) / "grid_summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nGrid complete. Summary written to {summary_path}")
    return summary
