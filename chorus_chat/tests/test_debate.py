from unittest import TestCase
from unittest.mock import patch

from chorus_mvp.debate import build_debate_messages, run_debate_coordinator


class DebatePromptTests(TestCase):
    def test_debate_prompt_uses_chorus_round_two_framing(self) -> None:
        messages = build_debate_messages(
            source_text="Bolaget redovisade intäkter.",
            draft_translation="The company recognized revenue.",
            source_lang="Swedish",
            target_lang="English",
            domain="financial translation",
            user_goal="Improve the translation.",
            agent_results=[
                {
                    "agent": "accuracy",
                    "has_issue": False,
                    "severity": "none",
                    "issue_span": "",
                    "suggested_revision": "",
                    "explanation": "",
                    "confidence": 0.9,
                }
            ],
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Round 2 coordinator in CHORUS", system_prompt)
        self.assertIn("specialist agents independently", system_prompt)
        self.assertIn("Reject out-of-scope", system_prompt)
        self.assertIn("If no agent suggestion is valid", system_prompt)

        user_prompt = messages[1]["content"]
        self.assertIn("CHORUS Round 2 consolidated recommendation", user_prompt)
        self.assertIn('"agent": "accuracy"', user_prompt)


class DebateRuntimeTests(TestCase):
    def test_run_debate_coordinator_normalizes_missing_fields(self) -> None:
        with patch(
            "chorus_mvp.debate.call_ollama_chat_json",
            return_value={
                "final_recommendation": "The company recognized revenue.",
                "confidence": 0.8,
            },
        ):
            result = run_debate_coordinator(
                source_text="Bolaget redovisade intäkter.",
                draft_translation="The company recognized revenue.",
                source_lang="Swedish",
                target_lang="English",
                domain="financial translation",
                user_goal="Improve the translation.",
                agent_results=[],
                model="qwen3:8b",
            )

        self.assertEqual(result["final_recommendation"], "The company recognized revenue.")
        self.assertEqual(result["accepted_points"], [])
        self.assertEqual(result["rejected_points"], [])
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(result["reasoning_summary"], "")
        self.assertEqual(result["confidence"], 0.8)
