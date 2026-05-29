from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chorus_mvp.agents import run_all_agents
from chorus_mvp.debate import run_debate_coordinator
from chorus_mvp.llm import DEFAULT_OLLAMA_MODEL


def model_output_path(model: str, output_dir: str = "outputs") -> Path:
    safe_model = (
        model.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
    return Path(output_dir) / f"agent_results_{safe_model}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CHORUS-style MQM agents on one translation segment.")

    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--source-lang", default="Swedish")
    parser.add_argument("--target-lang", default="English")
    parser.add_argument("--domain", default="financial translation")
    parser.add_argument(
        "--goal",
        default="Improve the translation for professional use while preserving meaning.",
    )
    parser.add_argument(
        "--source",
        default="Bolaget redovisade intäkter när kontrollen över varorna överfördes.",
    )
    parser.add_argument(
        "--draft",
        default="The company recognized revenue when control of the goods was transferred.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the JSON result. If omitted, the output is grouped by model, "
            "for example outputs/agent_results_qwen3_8b.json."
        ),
    )

    args = parser.parse_args()

    results = run_all_agents(
        source_text=args.source,
        draft_translation=args.draft,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        domain=args.domain,
        user_goal=args.goal,
        model=args.model,
    )
    
    debate_result = run_debate_coordinator(
        source_text=args.source,
        draft_translation=args.draft,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        domain=args.domain,
        user_goal=args.goal,
        agent_results=results,
        model=args.model,
    )

    output_path = Path(args.output) if args.output else model_output_path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "source_text": args.source,
        "draft_translation": args.draft,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "domain": args.domain,
        "user_goal": args.goal,
        "agent_results": results,
        "debate_result": debate_result,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console = Console()

    table = Table(title="CHORUS-style Agent Results")
    table.add_column("Agent")
    table.add_column("Issue?")
    table.add_column("Severity")
    table.add_column("Suggestion")
    table.add_column("Explanation")

    for item in results:
        table.add_row(
            str(item["agent"]),
            str(item["has_issue"]),
            str(item["severity"]),
            str(item["suggested_revision"]),
            str(item["explanation"]),
        )

    console.print(table)
    console.print(f"\nSaved JSON to: [bold]{output_path}[/bold]")
    
    console.print("\n[bold]Round 2 Debate Coordinator[/bold]")
    console.print(f"Final recommendation: {debate_result['final_recommendation']}")
    console.print(f"Reasoning: {debate_result['reasoning_summary']}")
    console.print(f"Confidence: {debate_result['confidence']}")

if __name__ == "__main__":
    main()
