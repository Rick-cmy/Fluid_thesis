from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from coordination.base import Coordinator, DimensionResult, ReviewResult
from coordination.pipeline import _AGENTS, _run_one_agent
from llm import call_llm

# ---------------------------------------------------------------------------
# CHORUS-v2: Phase 2 meta-coordinator
#
# Design references:
#   ChatEval (Chan et al., 2024)   — Simultaneous-Talk-with-Summarizer:
#     parallel Phase 1 + single summariser outperforms sequential debate
#   ReConcile (Chen et al., 2024) — grouped presentation + confidence scoring:
#     present findings grouped by has_issue; confidence-weighted acceptance
#   ManyMinds (Ma et al., 2025)   — meta-judge approach:
#     iterative per-agent debate amplifies biases; single meta-judge is
#     more robust and bias-resistant
# ---------------------------------------------------------------------------

_COORDINATOR_SYSTEM = """\
You are a meta-coordinator synthesising the findings of six specialist \
translation quality reviewers (CHORUS-v2 protocol, Phase 2).

Each specialist independently reviewed one quality dimension of a \
Swedish→English financial translation and reported a confidence score [0, 1].

Your responsibilities:
1. ACCEPT findings where has_issue=true AND confidence ≥ 0.65,
   OR severity="critical" regardless of confidence.
2. REJECT findings where confidence < 0.40 AND severity is not "critical".
3. CROSS-VALIDATE: if two or more specialists flag overlapping spans,
   treat this as corroborating evidence and prefer these corrections.
4. RESOLVE CONFLICTS: if specialists disagree on the same span, state
   which correction you apply and why.
5. PRODUCE one complete revised English translation that incorporates
   all accepted corrections. If nothing is accepted, return the original
   draft unchanged.

Return this exact JSON schema — no other text:
{
  "final_recommendation": "complete revised translation or original draft",
  "accepted_points": [
    {"agent": "name", "correction": "what was fixed", "confidence": 0.0, "reason": "why accepted"}
  ],
  "rejected_points": [
    {"agent": "name", "finding": "what was rejected", "reason": "why rejected"}
  ],
  "conflicts_resolved": [
    {"span": "text span", "agents": ["a", "b"], "decision": "resolution applied"}
  ],
  "reasoning_summary": "one concise paragraph for the human translator",
  "confidence": 0.0
}"""


def _run_coordinator(
    source: str,
    draft: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    rag_context: str,
    specialist_results: list[DimensionResult],
    model: str,
    provider: str,
    base_url: str,
) -> tuple[dict, int, int]:
    # Grouped presentation (ReConcile): specialists with findings first,
    # then those that found nothing — keeps the coordinator focused.
    with_issues = [d for d in specialist_results if d.has_issue]
    without_issues = [d for d in specialist_results if not d.has_issue]

    findings_payload = json.dumps(
        {
            "specialists_with_findings": [
                {
                    "agent": d.agent,
                    "severity": d.severity,
                    "confidence": round(d.confidence, 3),
                    "issue_span": d.issue_span,
                    "suggested_revision": d.suggested_revision,
                    "explanation": d.explanation,
                }
                for d in with_issues
            ],
            "specialists_no_findings": [d.agent for d in without_issues],
        },
        ensure_ascii=False,
        indent=2,
    )

    rag_block = f"\n\n{rag_context}" if rag_context else ""
    user = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Domain: {domain}{rag_block}\n\n"
        f"Source text:\n<SOURCE>\n{source}\n</SOURCE>\n\n"
        f"Current draft:\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
        f"Phase 1 specialist findings:\n"
        f"<SPECIALIST_FINDINGS>\n{findings_payload}\n</SPECIALIST_FINDINGS>\n\n"
        "Synthesise and produce the Phase 2 consolidated recommendation. "
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": _COORDINATOR_SYSTEM},
        {"role": "user", "content": user},
    ]
    return call_llm(messages, model=model, provider=provider, base_url=base_url)


class Debate(Coordinator):
    """
    CHORUS-v2: 2-phase meta-coordination protocol.

    Phase 1 — 6 specialist agents review independently in parallel.
               Identical to Pipeline Phase 1; reuses _AGENTS + _run_one_agent.
    Phase 2 — 1 meta-coordinator synthesises all Phase 1 findings using
               confidence thresholding and cross-agent corroboration.

    Token cost: 6 (Phase 1) + 1 (Phase 2) = 7 calls/segment.
    Compare: old 3-round design = 13 calls/segment.

    Metrics contract: dimension_results carries Phase 1 outputs so that
    metrics.py per-error-type F1 evaluation is unchanged (looks up d.agent).
    """

    def __init__(
        self,
        provider: str = "ollama",
        base_url: str = "http://127.0.0.1:11434",
        max_workers: int = 6,
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

        # ── Phase 1: parallel independent specialist review ──────────────────
        phase1_results: list[DimensionResult] = []
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
                phase1_results.append(dim)
                total_prompt += pt
                total_completion += ct

        # ── Phase 2: meta-coordinator synthesises Phase 1 findings ──────────
        coord_json, pt2, ct2 = _run_coordinator(
            source, draft, source_lang, target_lang, domain, rag_context,
            phase1_results, model, self.provider, self.base_url,
        )
        total_prompt += pt2
        total_completion += ct2

        return ReviewResult(
            final_recommendation=coord_json.get("final_recommendation", draft),
            dimension_results=phase1_results,      # Phase 1 used for per-dim F1 in metrics.py
            accepted_points=coord_json.get("accepted_points", []),
            rejected_points=coord_json.get("rejected_points", []),
            conflicts=coord_json.get("conflicts_resolved", []),
            reasoning_summary=coord_json.get("reasoning_summary", ""),
            confidence=float(coord_json.get("confidence", 0.0)),
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_s=time.monotonic() - t0,
        )
