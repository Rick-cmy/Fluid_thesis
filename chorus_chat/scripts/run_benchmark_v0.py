#!/usr/bin/env python3
"""
Run CHORUS-style financial translation QA benchmark.

Usage from project root:
    python3 scripts/run_benchmark_v0.py \
      --benchmark data/benchmark/financial_qa_benchmark_v0.jsonl \
      --models qwen3:8b,llama3.1:8b

Assumption:
    src/chorus_mvp/run_agents.py accepts --output and writes structured JSON
    directly to the per-case benchmark result path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def recursively_find_agent_items(obj: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of agent result dicts from unknown JSON shape."""
    found: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj.keys()}
        if "agent" in keys and (
            "has_issue" in keys
            or "issue" in keys
            or "severity" in keys
            or "suggested_revision" in keys
            or "suggestion" in keys
        ):
            found.append(obj)
        for v in obj.values():
            found.extend(recursively_find_agent_items(v))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursively_find_agent_items(item))

    return found


def get_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return None


def summarize_result(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        return {
            "json_loaded": False,
            "agent_count": "",
            "issue_count": "",
            "major_or_critical_count": "",
            "formatting_or_markup_issue": "",
        }

    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "json_loaded": False,
            "agent_count": "",
            "issue_count": "",
            "major_or_critical_count": "",
            "formatting_or_markup_issue": "",
        }

    agent_items = recursively_find_agent_items(obj)

    issue_count = 0
    major_or_critical = 0
    markup_issue = False

    for item in agent_items:
        item_lower = {str(k).lower(): v for k, v in item.items()}
        issue = get_boolish(item_lower.get("has_issue", item_lower.get("issue")))
        severity = str(item_lower.get("severity", "")).lower()
        agent_name = str(item_lower.get("agent", "")).lower()

        if issue is True:
            issue_count += 1
            if "markup" in agent_name or "format" in agent_name or "design" in agent_name:
                markup_issue = True

        if severity in {"major", "critical"}:
            major_or_critical += 1

    return {
        "json_loaded": True,
        "agent_count": len(agent_items),
        "issue_count": issue_count,
        "major_or_critical_count": major_or_critical,
        "formatting_or_markup_issue": markup_issue,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/financial_qa_benchmark_v0.jsonl")
    parser.add_argument("--models", default="qwen3:8b,llama3.1:8b")
    parser.add_argument("--run-agents", default="src/chorus_mvp/run_agents.py")
    parser.add_argument("--output-dir", default="outputs/benchmark_v0")
    parser.add_argument("--include-domain", action="store_true", help="Pass --domain to run_agents.py if supported.")
    args = parser.parse_args()

    project_root = Path.cwd()
    benchmark_path = project_root / args.benchmark
    run_agents_path = project_root / args.run_agents
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(benchmark_path)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary_rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["case_id"]

        for model in models:
            model_safe = safe_model_name(model)
            result_prefix = f"{case_id}__{model_safe}"

            stdout_path = output_dir / f"{result_prefix}.stdout.txt"
            stderr_path = output_dir / f"{result_prefix}.stderr.txt"
            copied_json_path = output_dir / f"{result_prefix}.json"

            cmd = [
                "python3",
                str(run_agents_path),
                "--model", model,
                "--source", case["source"],
                "--draft", case["draft"],
                "--output", str(copied_json_path),
            ]

            if args.include_domain:
                cmd.extend(["--domain", case.get("domain", "IFRS financial reporting")])

            env = os.environ.copy()
            env["PYTHONPATH"] = "src"

            print(f"\n=== Running {case_id} with {model} ===")
            completed = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                text=True,
                capture_output=True,
            )

            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")

            auto_summary = summarize_result(copied_json_path)

            summary_rows.append({
                "case_id": case_id,
                "model": model,
                "exit_code": completed.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "json_path": str(copied_json_path) if copied_json_path.exists() else "",
                "no_error_case": case.get("no_error_case", False),
                "expected_issue_dimensions": "|".join(case.get("expected_issue_dimensions", [])),
                "ideal_translation": case.get("ideal_translation", ""),
                **auto_summary,
            })

            if completed.returncode != 0:
                print(f"FAILED: {case_id} / {model}. See {stderr_path}")
            else:
                print(f"OK: saved to {copied_json_path}")

    summary_csv = output_dir / "benchmark_run_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
