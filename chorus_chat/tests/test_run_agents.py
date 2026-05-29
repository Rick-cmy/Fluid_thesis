from pathlib import Path
from unittest import TestCase

from chorus_mvp.run_agents import model_output_path


class RunAgentsOutputPathTests(TestCase):
    def test_model_output_path_sanitizes_ollama_model_name(self) -> None:
        self.assertEqual(
            model_output_path("qwen3:8b"),
            Path("outputs/agent_results_qwen3_8b.json"),
        )
        self.assertEqual(
            model_output_path("llama3.1:8b"),
            Path("outputs/agent_results_llama3.1_8b.json"),
        )

    def test_model_output_path_sanitizes_slashes_and_spaces(self) -> None:
        self.assertEqual(
            model_output_path("my org/qwen test:latest", output_dir="tmp"),
            Path("tmp/agent_results_my_org_qwen_test_latest.json"),
        )
