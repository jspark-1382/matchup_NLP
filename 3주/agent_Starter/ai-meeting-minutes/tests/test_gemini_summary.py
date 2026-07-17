import unittest

from core.gemini_summary import REQUIRED_KEYS, generate_gemini_summary, validate_summary
from core.transcribe import load_demo_transcript


class GeminiSummaryTest(unittest.TestCase):
    def test_mock_does_not_call_api_and_has_required_keys(self):
        result = generate_gemini_summary(load_demo_transcript(), use_mock=True)
        self.assertEqual(set(result), REQUIRED_KEYS)
        self.assertEqual(result["action_items"][0]["owner"], "SPEAKER_01")

    def test_invalid_response_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_summary({"summary": "불완전"})


if __name__ == "__main__":
    unittest.main()

