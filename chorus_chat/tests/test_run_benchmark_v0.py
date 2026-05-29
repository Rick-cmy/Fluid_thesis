import json
import tempfile
from pathlib import Path
from unittest import TestCase

from scripts.run_benchmark_v0 import safe_model_name, summarize_result


class BenchmarkScriptTests(TestCase):
    def test_safe_model_name(self) -> None:
        self.assertEqual(safe_model_name("qwen3:8b"), "qwen3_8b")
        self.assertEqual(safe_model_name("llama3.1:8b"), "llama3_1_8b")
        self.assertEqual(safe_model_name("org/model:latest"), "org_model_latest")

    def test_summarize_result_reads_current_has_issue_schema(self) -> None:
        payload = {
            "agent_results": [
                {
                    "agent": "accuracy",
                    "has_issue": True,
                    "severity": "major",
                },
                {
                    "agent": "design_markup",
                    "has_issue": True,
                    "severity": "minor",
                },
                {
                    "agent": "fluency",
                    "has_issue": False,
                    "severity": "none",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            summary = summarize_result(path)

        self.assertTrue(summary["json_loaded"])
        self.assertEqual(summary["agent_count"], 3)
        self.assertEqual(summary["issue_count"], 2)
        self.assertEqual(summary["major_or_critical_count"], 1)
        self.assertTrue(summary["formatting_or_markup_issue"])

    def test_summarize_result_missing_json(self) -> None:
        summary = summarize_result(Path("does-not-exist.json"))

        self.assertFalse(summary["json_loaded"])
        self.assertEqual(summary["agent_count"], "")
