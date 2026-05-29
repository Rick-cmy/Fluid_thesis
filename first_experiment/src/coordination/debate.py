from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from coordination.base import Coordinator, DimensionResult, ReviewResult
from coordination.pipeline import _AGENTS, _run_one_agent
from llm import call_llm

_COORDINATOR_SYSTEM = """You are the Round 2 coordinator in CHORUS, a multi-agent translation revision workflow.

CHORUS structure:
- Round 1: specialist agents independently inspect one translation dimension each.
- Round 2 (you): compare their evidence, filter invalid claims, resolve conflicts,
  and produce one final consolidated recommendation for the human translator.

Your job:
- Accept only suggestions supported by the source text and the agent's assigned role.
- Reject out-of-scope, duplicated, unsupported, or hallucinated suggestions.
- Resolve conflicts explicitly.
- Produce one consolidated target-language revision.
- If no suggestion is valid, keep the original draft.

Return this exact JSON schema:
{
  "final_recommendation": "one complete revised target translation",
  "accepted_points": [{"agent": "name", "point": "suggestion", "reason": "why"}],
  "rejected_points": [{"agent": "name", "point": "suggestion", "reason": "why"}],
  "conflicts": [{"description": "conflict", "resolution": "how resolved"}],
  "reasoning_summary": "short explanation for the translator",
  "confidence": 0.0
}"""


def _run_coordinator(
    source: str,
    draft: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    agent_results: list[DimensionResult],
    model: str,
    provider: str,
    base_url: str,
) -> tuple[dict, int, int]:
    agent_json = json.dumps(
        [
            {
                "agent": d.agent,
                "has_issue": d.has_issue,
                "severity": d.severity,
                "issue_span": d.issue_span,
                "suggested_revision": d.suggested_revision,
                "explanation": d.explanation,
            }
            for d in agent_results
        ],
        ensure_ascii=False,
        indent=2,
    )
    user = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Domain: {domain}\n\n"
        f"Source text:\n<SOURCE>\n{source}\n</SOURCE>\n\n"
        f"Current draft:\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
        f"Round 1 agent outputs:\n<AGENT_RESULTS>\n{agent_json}\n</AGENT_RESULTS>\n\n"
        "Produce the CHORUS Round 2 consolidated recommendation. Return JSON only."
    )
    messages = [
        {"role": "system", "content": _COORDINATOR_SYSTEM},
        {"role": "user", "content": user},
    ]
    return call_llm(messages, model=model, provider=provider, base_url=base_url)


class Debate(Coordinator):
    """
    Full CHORUS protocol:
      Round 1 — 7 specialist agents run in parallel (ThreadPoolExecutor).
      Round 2 — coordinator synthesises, filters, resolves conflicts.

    Token cost per segment: 7 (Round 1, concurrent) + 1 (Round 2) = 8 calls.
    Wall-clock latency ≈ max(Round 1 agent latency) + Round 2 latency.
    """

    def __init__(
        self,
        provider: str = "ollama",
        base_url: str = "http://127.0.0.1:11434",
        max_workers: int = 7,
    ):
        self.provider = provider
        self.base_url = base_url
        self.max_workers = max_workers

    def run(
        self,
        source: str,
        draft: str,
        source_lang: str = "Swedish",
        target_lang: str = "English",
        domain: str = "IFRS financial reporting",
        rag_context: str = "",
        model: str = "qwen3:8b",
    ) -> ReviewResult:
        t0 = time.monotonic()
        total_prompt, total_completion = 0, 0

        # Round 1: parallel specialists
        round1_results: list[DimensionResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    _run_one_agent,
                    name, focus, source, draft,
                    source_lang, target_lang, domain, rag_context,
                    model, self.provider, self.base_url,
                ): name
                for name, focus in _AGENTS.items()
            }
            for future in as_completed(futures):
                dim, pt, ct = future.result()
                round1_results.append(dim)
                total_prompt += pt
                total_completion += ct

        # Round 2: coordinator
        coord_json, pt2, ct2 = _run_coordinator(
            source, draft, source_lang, target_lang, domain,
            round1_results, model, self.provider, self.base_url,
        )
        total_prompt += pt2
        total_completion += ct2

        return ReviewResult(
            final_recommendation=coord_json.get("final_recommendation", draft),
            dimension_results=round1_results,
            accepted_points=coord_json.get("accepted_points", []),
            rejected_points=coord_json.get("rejected_points", []),
            conflicts=coord_json.get("conflicts", []),
            reasoning_summary=coord_json.get("reasoning_summary", ""),
            confidence=float(coord_json.get("confidence", 0.0)),
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_s=time.monotonic() - t0,
        )
