"""Tests for prompt assembly and citation extraction."""

import unittest

from rag.corpus import Chunk
from rag.prompts import (
    NO_CONTEXT_PROMPT,
    RAG_PROMPT,
    extract_citations,
    format_context,
)
from rag.retriever import Retrieved


def _r(text, source, score=1.0):
    return Retrieved(chunk=Chunk(text=text, source=source), score=score)


class FormatContextTests(unittest.TestCase):
    def test_numbers_and_labels_passages(self):
        block = format_context([
            _r("founded in 2019", "handbook.md#1"),
            _r("flagship is the Kestrel-7", "handbook.md#2"),
        ])
        self.assertIn("[1] (handbook.md#1) founded in 2019", block)
        self.assertIn("[2] (handbook.md#2) flagship is the Kestrel-7", block)

    def test_empty(self):
        self.assertEqual(format_context([]), "")


class ExtractCitationsTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(extract_citations("The answer is 2019 [1]."), [1])

    def test_multiple_in_order(self):
        self.assertEqual(extract_citations("see [2] and also [1]"), [2, 1])

    def test_deduplicates(self):
        self.assertEqual(extract_citations("[1] foo [1] bar [3]"), [1, 3])

    def test_none(self):
        self.assertEqual(extract_citations("no citations here"), [])

    def test_adjacent_markers(self):
        self.assertEqual(extract_citations("grounded [1][2][3]"), [1, 2, 3])


class PromptShapeTests(unittest.TestCase):
    def test_rag_prompt_has_slots(self):
        out = RAG_PROMPT.format(context="CTX", question="Q?")
        self.assertIn("CTX", out)
        self.assertIn("Q?", out)
        self.assertIn("ONLY", out)  # the grounding instruction

    def test_no_context_prompt_has_question_slot(self):
        out = NO_CONTEXT_PROMPT.format(question="Q?")
        self.assertIn("Q?", out)
        self.assertNotIn("{question}", out)


if __name__ == "__main__":
    unittest.main()
