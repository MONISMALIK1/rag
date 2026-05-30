"""Tests for the Retriever — BM25 over real corpus chunks.

These check the end-to-end retrieval behavior on the bundled fictional corpus:
the right passage surfaces for a factual question.
"""

import unittest

from rag.corpus import Chunk, load_sample
from rag.retriever import Retriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(load_sample())

    def test_finds_founding_passage(self):
        hits = self.retriever.retrieve("When was Helix Dynamics founded?", k=3)
        self.assertTrue(hits)
        self.assertIn("2019", hits[0].text)

    def test_finds_battery_passage(self):
        hits = self.retriever.retrieve("which battery powers the drone", k=3)
        self.assertTrue(any("AuroraCell" in h.text for h in hits))

    def test_finds_platform_passage(self):
        hits = self.retriever.retrieve("fleet coordination software platform", k=3)
        self.assertTrue(any("Loomweave" in h.text for h in hits))

    def test_respects_k(self):
        hits = self.retriever.retrieve("Helix Dynamics drone", k=2)
        self.assertLessEqual(len(hits), 2)

    def test_results_carry_score_and_source(self):
        hit = self.retriever.retrieve("founded", k=1)[0]
        self.assertGreater(hit.score, 0.0)
        self.assertTrue(hit.source.startswith("handbook.md#"))

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            Retriever([])

    def test_custom_chunks(self):
        chunks = [
            Chunk("python is a programming language", "x#1"),
            Chunk("the eiffel tower is in paris", "x#2"),
        ]
        r = Retriever(chunks)
        hits = r.retrieve("where is the eiffel tower", k=1)
        self.assertIn("paris", hits[0].text)


if __name__ == "__main__":
    unittest.main()
