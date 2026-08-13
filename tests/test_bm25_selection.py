import unittest

import bm25_selection


PASSAGE = (
    "The Coast Guard says the caller made 28 false distress alerts from Annapolis. "
    "Each call involved the same male voice on an emergency radio channel. "
    "Officials estimate the hoaxes cost about $500,000 over two years. "
    "A separate story reports that a stolen Picasso was seized at the Port of Newark. "
    "Weather today is cloudy with snow showers and winds from the south."
)


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self):
        self.assertEqual(len(bm25_selection.split_sentences(PASSAGE)), 5)

    def test_drops_fragments_too_short_to_carry_a_fact(self):
        self.assertEqual(
            bm25_selection.split_sentences("Ok. The fence was electrified by a live wire."),
            ["The fence was electrified by a live wire."],
        )


class TestSelectSentences(unittest.TestCase):
    def test_keeps_only_query_relevant_sentences(self):
        result = bm25_selection.select_sentences(
            "How many false distress calls were made from Annapolis?", PASSAGE, 2
        )
        self.assertIn("28 false distress alerts from Annapolis", result)
        self.assertNotIn("Picasso", result)
        self.assertNotIn("snow showers", result)

    def test_rare_proper_noun_outranks_common_words(self):
        result = bm25_selection.select_sentences(
            "Where was the Picasso seized?", PASSAGE, 1
        )
        self.assertIn("Port of Newark", result)

    def test_preserves_original_sentence_order(self):
        result = bm25_selection.select_sentences(
            "distress alerts radio channel Annapolis", PASSAGE, 2
        )
        self.assertLess(result.index("28 false"), result.index("male voice"))

    def test_returns_text_unchanged_when_already_short(self):
        short = "Only one sentence here about Newark."
        self.assertEqual(bm25_selection.select_sentences("Newark", short, 3), short)

    def test_no_query_overlap_falls_back_to_leading_sentences(self):
        # Dropping everything would discard evidence the retriever chose on purpose.
        result = bm25_selection.select_sentences("zzzz qqqq", PASSAGE, 2)
        self.assertTrue(result.startswith("The Coast Guard says"))

    def test_top_n_zero_disables_trimming(self):
        self.assertEqual(bm25_selection.select_sentences("Newark", PASSAGE, 0), PASSAGE)


if __name__ == "__main__":
    unittest.main()
