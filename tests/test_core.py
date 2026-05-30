"""Tests for the RAG loop — retriever is real, the LLM is mocked.

These prove the paper's mechanic: the retrieved evidence is actually placed in
the prompt the model sees (so generation is grounded), citations are parsed back
out, and the no-retrieval baseline genuinely sends no context.
"""

import unittest
from unittest.mock import patch

from rag import core
from rag.core import RAGResult, answer, answer_no_retrieval
from rag.retriever import Retriever
from rag.corpus import load_sample


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(load_sample())

    def test_retrieved_evidence_reaches_the_prompt(self):
        # The crux of RAG: the founding passage must be in the prompt the model
        # is given. We capture the prompt the mock receives and assert on it.
        captured = {}

        def fake_chat(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Helix Dynamics was founded in 2019 [1]."

        with patch.object(core, "chat", side_effect=fake_chat):
            res = answer("When was Helix Dynamics founded?", self.retriever, k=3)

        self.assertIsInstance(res, RAGResult)
        self.assertTrue(res.used_retrieval)
        self.assertIn("2019", captured["prompt"])          # evidence was injected
        self.assertIn("2019", res.answer)

    def test_citations_parsed_and_mapped_to_sources(self):
        with patch.object(core, "chat", return_value="Founded in 2019 [1]."):
            res = answer("founding year", self.retriever, k=3)
        self.assertEqual(res.citations, [1])
        self.assertEqual(len(res.cited_sources), 1)
        self.assertTrue(res.cited_sources[0].startswith("handbook.md#"))

    def test_out_of_range_citation_ignored(self):
        # Model cites [9] but only 3 passages were shown — drop it silently.
        with patch.object(core, "chat", return_value="answer [9]"):
            res = answer("anything about Helix", self.retriever, k=3)
        self.assertEqual(res.citations, [9])
        self.assertEqual(res.cited_sources, [])

    def test_k_controls_passage_count(self):
        with patch.object(core, "chat", return_value="ok"):
            res = answer("Helix Dynamics drone battery", self.retriever, k=2)
        self.assertLessEqual(len(res.retrieved), 2)

    def test_answer_records_prompt(self):
        with patch.object(core, "chat", return_value="ok"):
            res = answer("founded", self.retriever, k=1)
        self.assertIn("Question:", res.prompt)


class NoRetrievalTests(unittest.TestCase):
    def test_baseline_sends_no_context(self):
        captured = {}

        def fake_chat(prompt, **kwargs):
            captured["prompt"] = prompt
            return "I think 2018?"

        with patch.object(core, "chat", side_effect=fake_chat):
            res = answer_no_retrieval("When was Helix Dynamics founded?")

        self.assertFalse(res.used_retrieval)
        self.assertEqual(res.retrieved, [])
        self.assertNotIn("Context:", captured["prompt"])
        self.assertNotIn("AuroraCell", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
