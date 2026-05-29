from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chorus_mvp.llm import DEFAULT_OLLAMA_MODEL, call_ollama_chat_json


@dataclass(frozen=True)
class AgentSpec:
    name: str
    dimension: str
    focus: str


AGENTS: dict[str, dict[str, str]] = {
    "accuracy": {
        "dimension": "Accuracy",
        "focus": "meaning preservation, omissions, additions, distortions",
    },
    "terminology": {
        "dimension": "Terminology",
        "focus": "domain-specific terms, glossary consistency, IFRS terminology",
    },
    "fluency": {
        "dimension": "Fluency",
        "focus": "grammar, naturalness, readability in the target language",
    },
    "style": {
        "dimension": "Style",
        "focus": "tone, register, formality, financial reporting style",
    },
    "audience_appropriateness": {
        "dimension": "Audience Appropriateness",
        "focus": "fit for professional financial translators, auditors, investors, or report readers",
    },
    "locale_convention": {
        "dimension": "Locale Convention",
        "focus": "regional conventions, date/number/currency format, British vs American spelling",
    },
    "design_markup": {
        "dimension": "Design and Markup",
        "focus": "formatting, tags, placeholders, line breaks, tables, symbols, structural integrity",
    },
}

SEVERITY_GUIDANCE = """
Severity calibration:
- none: no actionable issue in your assigned dimension.
- minor: small improvement; meaning and professional acceptability mostly intact.
- major: clear problem that could mislead, sound unprofessional, or require human correction.
- critical: severe meaning loss, contradiction, or unusable translation in your dimension.
""".strip()


def get_agent_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            name=name,
            dimension=config["dimension"],
            focus=config["focus"],
        )
        for name, config in AGENTS.items()
    ]


def build_agent_messages(
    agent: AgentSpec,
    source_text: str,
    draft_translation: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    user_goal: str,
    memory_context: str = "",
) -> list[dict[str, str]]:
    system_prompt = f"""
You are a CHORUS specialist reviewer for the {agent.dimension} dimension.

You are part of CHORUS, a multi-agent translation revision workflow.
CHORUS uses independent specialist reviewers first, then a second-round coordinator
to compare evidence, resolve conflicts, and keep the human translator in control.

Round 1 task:
- Review the draft independently.
- Stay inside your assigned dimension.
- Ground every issue in the source text and draft translation.
- Propose the smallest useful correction, not a full rewrite.
- Leave final cross-agent decisions to the Round 2 coordinator.

Assigned dimension:
{agent.dimension} ({agent.name})

Dimension focus:
{agent.focus}

{SEVERITY_GUIDANCE}

Strict rules:
- Judge ONLY your assigned dimension.
- Treat your focus statement as the boundary of what you are allowed to assess.
- Do NOT rewrite the whole translation unless the whole translation is defective in your dimension.
- Do NOT give general translation advice outside your dimension.
- Do NOT repeat another agent's likely responsibility.
- Do NOT invent external facts.
- If there is no issue, say has_issue=false.
- If has_issue=false, use severity="none" and leave issue_span and suggested_revision empty.
- If has_issue=true, issue_span must quote or precisely identify the problematic draft span.
- Return valid JSON only.
- The JSON must follow this exact schema:

{{
  "agent": "{agent.name}",
  "has_issue": true,
  "severity": "none | minor | major | critical",
  "issue_span": "problematic part of the draft translation, or empty string",
  "suggested_revision": "suggested corrected phrase or sentence, or empty string",
  "explanation": "short explanation",
  "confidence": 0.0
}}
""".strip()

    user_prompt = f"""
Source language: {source_lang}
Target language: {target_lang}
Domain: {domain}
User goal: {user_goal}

Memory context from previous translator behavior:
{memory_context if memory_context else "No memory yet."}

Source text:
<SOURCE>
{source_text}
</SOURCE>

Current draft translation:
<DRAFT>
{draft_translation}
</DRAFT>

Review the draft translation from the perspective of your assigned dimension only.
Use the source text as evidence, and ignore problems outside your dimension.
Return JSON only.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_agent(
    agent: AgentSpec,
    source_text: str,
    draft_translation: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    user_goal: str,
    model: str,
    memory_context: str = "",
) -> dict[str, Any]:
    messages = build_agent_messages(
        agent=agent,
        source_text=source_text,
        draft_translation=draft_translation,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        user_goal=user_goal,
        memory_context=memory_context,
    )

    result = call_ollama_chat_json(messages=messages, model=model)

    # Normalize missing fields so frontend/backend will not crash later.
    return {
        "agent": result.get("agent", agent.name),
        "has_issue": bool(result.get("has_issue", False)),
        "severity": result.get("severity", "none"),
        "issue_span": result.get("issue_span", ""),
        "suggested_revision": result.get("suggested_revision", ""),
        "explanation": result.get("explanation", ""),
        "confidence": result.get("confidence", 0.0),
    }


def run_all_agents(
    source_text: str,
    draft_translation: str,
    source_lang: str = "Swedish",
    target_lang: str = "English",
    domain: str = "financial translation",
    user_goal: str = "Improve the translation for professional use while preserving meaning.",
    model: str = DEFAULT_OLLAMA_MODEL,
    memory_context: str = "",
) -> list[dict[str, Any]]:
    results = []

    for agent in get_agent_specs():
        print(f"Running agent: {agent.name}")
        result = run_agent(
            agent=agent,
            source_text=source_text,
            draft_translation=draft_translation,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            user_goal=user_goal,
            model=model,
            memory_context=memory_context,
        )
        results.append(result)

    return results
