from unittest import TestCase

from chorus_mvp.llm import LLMError, extract_json


class ExtractJsonTests(TestCase):
    def test_extract_json_accepts_strict_json(self) -> None:
        self.assertEqual(extract_json('{"has_issue": false}'), {"has_issue": False})

    def test_extract_json_extracts_object_from_extra_text(self) -> None:
        self.assertEqual(
            extract_json('Here is the result: {"confidence": 0.7} done.'),
            {"confidence": 0.7},
        )

    def test_extract_json_raises_when_no_json_object_exists(self) -> None:
        with self.assertRaises(LLMError):
            extract_json("No structured result.")
