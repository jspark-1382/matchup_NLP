import unittest

from core.formatters import format_seconds, transcript_to_text


class FormatterTest(unittest.TestCase):
    def test_format_seconds(self):
        self.assertEqual(format_seconds(65), "01:05")

    def test_transcript_text_keeps_speaker(self):
        text = transcript_to_text([
            {"speaker": "SPEAKER_00", "start": 0, "end": 3, "text": "안녕하세요."}
        ])
        self.assertIn("SPEAKER_00", text)
        self.assertIn("안녕하세요.", text)


if __name__ == "__main__":
    unittest.main()

