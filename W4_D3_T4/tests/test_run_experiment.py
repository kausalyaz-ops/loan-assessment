import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_experiment import (
    BAD_PROMPTS,
    EXAMPLE_PROMPT,
    REWRITTEN_PROMPTS,
    STRICT_PROMPT,
    ask_model,
    confidence_brier_score,
    extraction_is_correct,
    get_api_key,
    summarize,
    validate_output,
)


class ValidationTests(unittest.TestCase):
    def test_api_key_uses_environment_when_available(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("run_experiment.getpass.getpass") as prompt:
                self.assertEqual(get_api_key(), "env-key")
                prompt.assert_not_called()

    def test_api_key_is_prompted_for_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("run_experiment.getpass.getpass", return_value="typed-key") as prompt:
                self.assertEqual(get_api_key(), "typed-key")
                prompt.assert_called_once()

    def test_empty_api_key_has_actionable_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("run_experiment.getpass.getpass", return_value=" "):
                with self.assertRaisesRegex(RuntimeError, "API key is required"):
                    get_api_key()

    def test_valid_json_is_pydantic_validated(self):
        result = validate_output(
            '{"name":"Sam","skills":["Python"],"years_experience":3,"confidence":0.8}'
        )
        self.assertTrue(result.valid_json)
        self.assertEqual(result.parsed.name, "Sam")

    def test_integer_confidence_is_valid_json_number(self):
        result = validate_output(
            '{"name":"Sam","skills":[],"years_experience":3,"confidence":1}'
        )
        self.assertIsNotNone(result.parsed)

    def test_invalid_json_is_reported(self):
        result = validate_output('{"name":')
        self.assertFalse(result.valid_json)
        self.assertIsNone(result.parsed)

    def test_valid_json_with_invalid_schema_is_reported(self):
        result = validate_output(
            '{"name":"Sam","skills":[],"years_experience":"three","confidence":0.8}'
        )
        self.assertTrue(result.valid_json)
        self.assertIsNone(result.parsed)
        self.assertTrue(result.error)

    def test_prompt_catalog_has_four_rewritten_weak_prompts(self):
        self.assertEqual(set(BAD_PROMPTS), set(REWRITTEN_PROMPTS))
        self.assertEqual(len(BAD_PROMPTS), 4)

    def test_prompt_templates_render_resume_text(self):
        for template in (STRICT_PROMPT, EXAMPLE_PROMPT, *BAD_PROMPTS.values()):
            self.assertIn("Sample resume", template.format(resume="Sample resume"))

    def test_json_mode_sets_response_format_and_plain_mode_does_not(self):
        class Completions:
            def __init__(self):
                self.options = []

            def create(self, **options):
                self.options.append(options)
                message = SimpleNamespace(content='{"name":"Sam"}')
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)]
                )

        completions = Completions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        self.assertEqual(ask_model(client, "prompt", json_mode=False), '{"name":"Sam"}')
        self.assertEqual(ask_model(client, "prompt", json_mode=True), '{"name":"Sam"}')
        self.assertNotIn("response_format", completions.options[0])
        self.assertEqual(
            completions.options[1]["response_format"],
            {"type": "json_object"},
        )

    def test_exact_extraction_and_confidence_score(self):
        record = {
            "parsed": {
                "name": "Sam",
                "skills": ["Python"],
                "years_experience": 3,
                "confidence": 0.8,
            },
            "expected": {
                "name": "Sam",
                "skills": ["python"],
                "years_experience": 3,
            },
            "valid_json": True,
            "schema_valid": True,
        }
        self.assertTrue(extraction_is_correct(record))
        self.assertAlmostEqual(confidence_brier_score([record]), 0.04)
        self.assertEqual(summarize([record])["correct_extractions"], 1)


if __name__ == "__main__":
    unittest.main()
