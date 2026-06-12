"""Pseudo-relevance-feedback query expansion (pure, offline)."""

import unittest

from rag.corpus import Chunk
from rag.expand import expand_query
from rag.retriever import Retriever


def _retriever() -> Retriever:
    return Retriever([
        Chunk("the kestrel drone has a lithium battery", "a#1"),
        Chunk("the kestrel drone flies for an hour on lithium", "a#2"),
        Chunk("lithium cells are lightweight and rechargeable", "b#1"),
        Chunk("the great wall of china is very long", "c#1"),
    ])


def _score_of(retriever: Retriever, query: str, source: str) -> float:
    for hit in retriever.retrieve(query, k=10):
        if hit.source == source:
            return hit.score
    return 0.0


class ExpandTests(unittest.TestCase):
    def test_keeps_original_query_and_appends(self):
        out = expand_query("kestrel drone", _retriever(), top_docs=2, n_terms=3)
        self.assertTrue(out.startswith("kestrel drone"))
        self.assertGreater(len(out.split()), 2)

    def test_harvests_term_common_to_top_hits(self):
        # 'lithium' appears in both top hits -> it should be added
        out = expand_query("kestrel drone", _retriever(), top_docs=2, n_terms=3)
        self.assertIn("lithium", out.split())

    def test_does_not_repeat_query_terms(self):
        out = expand_query("kestrel drone", _retriever(), top_docs=2, n_terms=5)
        added = out.split()[2:]  # everything after the original two query words
        self.assertNotIn("kestrel", added)
        self.assertNotIn("drone", added)

    def test_improves_recall_of_a_vocabulary_bridged_doc(self):
        r = _retriever()
        # bare 'kestrel drone' doesn't mention lithium, so doc b#1 scores ~0
        bare = _score_of(r, "kestrel drone", "b#1")
        expanded_q = expand_query("kestrel drone", r, top_docs=2, n_terms=3)
        better = _score_of(r, expanded_q, "b#1")
        self.assertGreater(better, bare)  # expansion surfaces the lithium-only doc

    def test_no_hits_returns_query_unchanged(self):
        # a query of only stopwords tokenizes to nothing -> no hits -> unchanged
        self.assertEqual(expand_query("the a an of", _retriever()), "the a an of")

    def test_zero_terms_returns_query(self):
        self.assertEqual(expand_query("kestrel", _retriever(), n_terms=0), "kestrel")


if __name__ == "__main__":
    unittest.main()
