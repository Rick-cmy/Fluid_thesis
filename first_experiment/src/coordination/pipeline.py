from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from coordination.base import Coordinator, DimensionResult, ReviewResult
from llm import call_llm

_AGENTS = {
    "terminology":  "IFRS/IAS financial terms and glossary consistency: flag wrong or non-standard terms (e.g. 'income' vs 'revenue', 'cost' vs 'expense', 'fair value' vs 'market value')",
    "numeracy":     "numbers, amounts, percentages, dates, currencies, and units: verify every figure in the draft exactly matches the source",
    "named_entity": "company names, property names, accounting standards (IFRS/IAS/GAAP), index names, and geographic names",
    "fluency":      "grammar, naturalness, sentence structure, and readability in English",
    "consistency":  "meaning consistency with the source: flag contradictions such as increased/decreased, profit/loss, positive/negative directional or polarity errors",
    "style_guide":  "Client Financial Style Guide: decimal separator must be period not comma, use 'percent' not 'per cent', date format DD Month YYYY, British English spelling",
}

_SEVERITY_ORDER = {"critical": 3, "major": 2, "minor": 1, "none": 0}


def _build_specialist_messages(
    agent_name: str,
    focus: str,
    source: str,
    draft: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    rag_context: str,
) -> list[dict]:
    rag_block = f"\n\n{rag_context}" if rag_context else ""
    system = f"""You are a specialist translation reviewer for the {agent_name.upper()} dimension only.

Focus: {focus}

Severity calibration:
- none: no issue in your dimension.
- minor: small improvement needed.
- major: clear problem requiring correction.
- critical: severe meaning loss or unusable translation.

Rules:
- Judge ONLY your assigned dimension.
- Do NOT comment on other dimensions.
- If has_issue=false, use severity="none" and leave issue_span and suggested_revision empty.
- Return valid JSON only matching this exact schema:
{{
  "agent": "{agent_name}",
  "has_issue": true,
  "severity": "none|minor|major|critical",
  "issue_span": "problematic part of draft or empty string",
  "suggested_revision": "corrected phrase or empty string",
  "explanation": "brief explanation",
  "confidence": 0.0
}}"""

    user = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Domain: {domain}{rag_block}\n\n"
        f"Source text:\n<SOURCE>\n{source}\n</SOURCE>\n\n"
        f"Draft translation:\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
        f"Review the draft from the {agent_name} perspective only. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_one_agent(
    agent_name: str,
    focus: str,
    source: str,
    draft: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    rag_context: str,
    model: str,
    provider: str,
    base_url: str,
) -> tuple[DimensionResult, int, int]:
    messages = _build_specialist_messages(
        agent_name, focus, source, draft, source_lang, target_lang, domain, rag_context
    )
    result_json, pt, ct = call_llm(messages, model=model, provider=provider, base_url=base_url)
    dim = DimensionResult(
        agent=agent_name,
        has_issue=bool(result_json.get("has_issue", False)),
        severity=result_json.get("severity", "none"),
        issue_span=result_json.get("issue_span", ""),
        suggested_revision=result_json.get("suggested_revision", ""),
        explanation=result_json.get("explanation", ""),
        confidence=float(result_json.get("confidence", 0.0)),
    )
    return dim, pt, ct


class Pipeline(Coordinator):
    """
    6 specialist agents run in parallel (ThreadPoolExecutor).
    Results merged by highest-severity suggestion per dimension — no cross-review.

    Token cost per segment: 6 calls (concurrent, wall-clock ≈ slowest single call).
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
        dimension_results: list[DimensionResult] = []
        total_prompt, total_completion = 0, 0

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
                dimension_results.append(dim)
                total_prompt += pt
                total_completion += ct

        # Merge: pick highest-severity suggestion as final recommendation
        best = max(
            (d for d in dimension_results if d.has_issue and d.suggested_revision),
            key=lambda d: _SEVERITY_ORDER.get(d.severity, 0),
            default=None,
        )
        final = best.suggested_revision if best else draft

        return ReviewResult(
            final_recommendation=final,
            dimension_results=dimension_results,
            reasoning_summary="Pipeline: 7 parallel specialists, rule-based merge.",
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_s=time.monotonic() - t0,
        )
