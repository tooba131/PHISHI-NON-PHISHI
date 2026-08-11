import unittest

from phishing_detector import clean_url, predict_url, to_char_stream


class PredictorTests(unittest.TestCase):
    def test_clean_url_normalizes_common_noise(self):
        self.assertEqual(clean_url("https://WWW.Example.com/Path?A=1"), "example.com/path?a=1")

    def test_to_char_stream_produces_character_stream(self):
        stream = to_char_stream("https://example.com/path")
        self.assertIsInstance(stream, str)
        self.assertTrue(stream)

    def test_predict_url_cleans_full_url_before_inference(self):
        result = predict_url("https://www.linkedin.com/")
        self.assertNotIn("https://", result["char_stream"])
        self.assertNotIn("www.", result["char_stream"])
        self.assertEqual(result["url"], "https://www.linkedin.com/")

    def test_predict_url_returns_probabilities_and_label(self):
        result = predict_url("https://example.com")
        self.assertIn("probability", result)
        self.assertIn("label", result)
        self.assertIn("is_phishing", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
