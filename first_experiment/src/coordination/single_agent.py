from __future__ import annotations

import time

from coordination.base import Coordinator, DimensionResult, ReviewResult
from llm import call_llm

_MQM_DIMENSIONS = [
    ("terminology", "IFRS/IAS domain terms, glossary consistency, near-miss IFRS terms"),
    ("numeracy", "numbers, magnitudes, dates, currencies, percentage formats"),
    ("named_entity", "company names, geographic locations, standards identifiers (ISO, IAS, IFRS)"),
    ("fluency", "grammar, naturalness, readability, syntactic well-formedness"),
    ("style_guide", "Client Financial Style Guide rules: tone, register, spelling conventions"),
    ("consistency", "internal coherence, directional claims (profit/loss, increased/decreased)"),
]

_SYSTEM_PROMPT = """\
You are a professional translation quality reviewer for Swedish–English financial documents.
Review the draft translation across all six quality dimensions simultaneously.
Return a JSON object with one key per dimension. Each value must follow this schema:
{
  "has_issue": true/false,
  "severity": "none|minor|major|critical",
  "issue_span": "problematic part of draft, or empty string",
  "suggested_revision": "corrected phrase, or empty string",
  "explanation": "brief explanation"
}

Dimensions to assess: terminology, numeracy, named_entity, fluency, style_guide, consistency.

If RAG context is provided, use it to verify terminology and style-guide correctness.
Return valid JSON only."""


class SingleAgent(Coordinator):
    """
    One LLM call with a full-review prompt covering all 6 dimensions.
    Coordination intensity: 1 LLM call per segment.
    """

    def __init__(self, provider: str = "ollama", base_url: str = "http://127.0.0.1:11434"):
        self.provider = provider
        self.base_url = base_url

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
        rag_block = f"\n\n{rag_context}" if rag_context else ""
        user_prompt = (
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Domain: {domain}{rag_block}\n\n"
            f"Source text:\n<SOURCE>\n{source}\n</SOURCE>\n\n"
            f"Draft translation:\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
            "Review the draft across all MQM dimensions. Return JSON only."
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.monotonic()
        result_json, prompt_tokens, completion_tokens = call_llm(
            messages, model=model, provider=self.provider, base_url=self.base_url
        )
        latency_s = time.monotonic() - t0

        dimension_results = []
        for dim_name, _ in _MQM_DIMENSIONS:
            dim_data = result_json.get(dim_name, {})
            dimension_results.append(DimensionResult(
                agent=dim_name,
                has_issue=bool(dim_data.get("has_issue", False)),
                severity=dim_data.get("severity", "none"),
                issue_span=dim_data.get("issue_span", ""),
                suggested_revision=dim_data.get("suggested_revision", ""),
                explanation=dim_data.get("explanation", ""),
                confidence=float(dim_data.get("confidence", 0.0)),
            ))

        # Build final recommendation: apply the highest-severity suggestion
        final = draft
        critical = [d for d in dimension_results if d.severity == "critical" and d.suggested_revision]
        major = [d for d in dimension_results if d.severity == "major" and d.suggested_revision]
        if critical:
            final = critical[0].suggested_revision
        elif major:
            final = major[0].suggested_revision

        return ReviewResult(
            final_recommendation=final,
            dimension_results=dimension_results,
            reasoning_summary="Single-agent full-review pass.",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
        )
