from unittest import TestCase
from unittest.mock import patch

from chorus_mvp.agents import AGENTS, build_agent_messages, get_agent_specs, run_agent


class AgentPromptTests(TestCase):
    def test_agent_prompt_uses_chorus_round_one_framing(self) -> None:
        agent = next(item for item in get_agent_specs() if item.name == "accuracy")

        messages = build_agent_messages(
            agent=agent,
            source_text="Bolaget redovisade intäkter.",
            draft_translation="The company recognized revenue.",
            source_lang="Swedish",
            target_lang="English",
            domain="financial translation",
            user_goal="Improve the translation.",
            memory_context="The translator often omits IFRS qualifiers.",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        system_prompt = messages[0]["content"]
        self.assertIn("CHORUS", system_prompt)
        self.assertIn("Round 1 task", system_prompt)
        self.assertIn("independently", system_prompt)
        self.assertIn("Assigned dimension:\nAccuracy (accuracy)", system_prompt)
        self.assertIn("meaning preservation", system_prompt)
        self.assertIn("Severity calibration", system_prompt)
        self.assertIn("has_issue=false", system_prompt)

        user_prompt = messages[1]["content"]
        self.assertIn("Source language: Swedish", user_prompt)
        self.assertIn("Target language: English", user_prompt)
        self.assertIn("The translator often omits IFRS qualifiers.", user_prompt)
        self.assertIn("<SOURCE>\nBolaget redovisade intäkter.\n</SOURCE>", user_prompt)
        self.assertIn(
            "<DRAFT>\nThe company recognized revenue.\n</DRAFT>",
            user_prompt,
        )


class AgentRuntimeTests(TestCase):
    def test_agent_registry_matches_chorus_dimensions(self) -> None:
        self.assertEqual(
            list(AGENTS),
            [
                "accuracy",
                "terminology",
                "fluency",
                "style",
                "audience_appropriateness",
                "locale_convention",
                "design_markup",
            ],
        )

        specs = get_agent_specs()
        self.assertEqual(len(specs), 7)
        self.assertEqual(specs[0].dimension, "Accuracy")
        self.assertEqual(specs[-1].dimension, "Design and Markup")

    def test_run_agent_normalizes_missing_fields_without_calling_ollama(self) -> None:
        agent = next(item for item in get_agent_specs() if item.name == "fluency")

        with patch(
            "chorus_mvp.agents.call_ollama_chat_json",
            return_value={
                "has_issue": True,
                "severity": "minor",
                "suggested_revision": "The company recognised revenue.",
            },
        ):
            result = run_agent(
                agent=agent,
                source_text="Bolaget redovisade intäkter.",
                draft_translation="The company recognized revenue.",
                source_lang="Swedish",
                target_lang="English",
                domain="financial translation",
                user_goal="Improve the translation.",
                model="qwen3:8b",
            )

        self.assertEqual(result["agent"], "fluency")
        self.assertTrue(result["has_issue"])
        self.assertEqual(result["severity"], "minor")
        self.assertEqual(result["issue_span"], "")
        self.assertEqual(result["suggested_revision"], "The company recognised revenue.")
        self.assertEqual(result["explanation"], "")
        self.assertEqual(result["confidence"], 0.0)
