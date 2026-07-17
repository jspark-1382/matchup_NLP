import unittest

from core.traditional_summary import summarize_traditional
from core.transcribe import load_demo_transcript


class TraditionalSummaryTest(unittest.TestCase):
    def test_selects_three_original_sentences(self):
        transcript = load_demo_transcript()
        result = summarize_traditional(transcript, sentence_count=3, keyword_count=5)
        self.assertEqual(result["method"], "textrank")
        self.assertEqual(len(result["summary_sentences"]), 3)
        self.assertEqual(len(result["keywords"]), 5)
        original = {item["text"] for item in transcript}
        self.assertTrue(all(item["text"] in original for item in result["summary_sentences"]))
        starts = [item["start"] for item in result["summary_sentences"]]
        self.assertEqual(starts, sorted(starts))

    def test_is_reproducible(self):
        transcript = load_demo_transcript()
        self.assertEqual(summarize_traditional(transcript), summarize_traditional(transcript))


if __name__ == "__main__":
    unittest.main()
