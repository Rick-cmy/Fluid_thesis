from __future__ import annotations

import json
from typing import Any

from chorus_mvp.llm import call_ollama_chat_json


def build_debate_messages(
    source_text: str,
    draft_translation: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    user_goal: str,
    agent_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system_prompt = """
You are the Round 2 coordinator in CHORUS, a multi-agent translation revision workflow.

CHORUS structure:
- Round 1: specialist agents independently inspect one translation dimension each.
- Round 2: the coordinator compares their evidence, filters invalid claims, resolves conflicts,
  and produces one final recommendation for a human translator.

Your job:
- Compare outputs from multiple MQM-style specialist agents.
- Accept only suggestions supported by the source text, draft translation, and agent role.
- Reject out-of-scope, duplicated, unsupported, or hallucinated suggestions.
- Resolve conflicts explicitly.
- Produce one consolidated target-language revision.
- Explain the decision briefly so the human translator can accept, edit, or reject it.

Important rules:
- Do NOT blindly accept every agent suggestion.
- Reject suggestions that are outside an agent's assigned role.
- Reject terminology claims that are not supported by the source text or domain.
- Prefer minimal edits that improve professional translation quality.
- Preserve valid parts of the draft.
- Preserve the full meaning of the source.
- Do not add information not present in the source.
- If no agent suggestion is valid, keep the draft translation as the final recommendation.
- Return valid JSON only.

Return this exact JSON schema:

{
  "final_recommendation": "one complete revised target translation",
  "accepted_points": [
    {
      "agent": "agent name",
      "point": "accepted suggestion",
      "reason": "why accepted"
    }
  ],
  "rejected_points": [
    {
      "agent": "agent name",
      "point": "rejected suggestion",
      "reason": "why rejected"
    }
  ],
  "conflicts": [
    {
      "description": "conflict between suggestions",
      "resolution": "how you resolved it"
    }
  ],
  "reasoning_summary": "short explanation for the human translator",
  "confidence": 0.0
}
""".strip()

    user_prompt = f"""
Source language: {source_lang}
Target language: {target_lang}
Domain: {domain}
User goal: {user_goal}

Source text:
<SOURCE>
{source_text}
</SOURCE>

Current draft translation:
<DRAFT>
{draft_translation}
</DRAFT>

Round 1 agent outputs:
<AGENT_RESULTS>
{json.dumps(agent_results, ensure_ascii=False, indent=2)}
</AGENT_RESULTS>

Now produce the CHORUS Round 2 consolidated recommendation.
Return JSON only.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_debate_coordinator(
    source_text: str,
    draft_translation: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    user_goal: str,
    agent_results: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    messages = build_debate_messages(
        source_text=source_text,
        draft_translation=draft_translation,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        user_goal=user_goal,
        agent_results=agent_results,
    )

    result = call_ollama_chat_json(
        messages=messages,
        model=model,
        temperature=0.1,
    )

    return {
        "final_recommendation": result.get("final_recommendation", ""),
        "accepted_points": result.get("accepted_points", []),
        "rejected_points": result.get("rejected_points", []),
        "conflicts": result.get("conflicts", []),
        "reasoning_summary": result.get("reasoning_summary", ""),
        "confidence": result.get("confidence", 0.0),
    }
