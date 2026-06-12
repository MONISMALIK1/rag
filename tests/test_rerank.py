"""MMR reranking — relevance vs. diversity (pure, offline)."""

import unittest

from rag.corpus import Chunk
from rag.rerank import mmr_rerank
from rag.retriever import Retrieved


def R(text: str, source: str, score: float) -> Retrieved:
    return Retrieved(chunk=Chunk(text=text, source=source), score=score)


# A and B are near-duplicates (high token overlap); C is a distinct topic.
A = R("the cat sat on the mat", "a", 3.0)
B = R("the cat sat on the mat today", "b", 2.9)
C = R("python is a programming language", "c", 2.0)


class MMRTests(unittest.TestCase):
    def test_returns_k_capped_at_pool(self):
        self.assertEqual(len(mmr_rerank([A, B, C], k=2)), 2)
        self.assertEqual(len(mmr_rerank([A, B, C], k=5)), 3)

    def test_empty_and_zero(self):
        self.assertEqual(mmr_rerank([], k=3), [])
        self.assertEqual(mmr_rerank([A, B, C], k=0), [])

    def test_first_pick_is_most_relevant(self):
        self.assertIs(mmr_rerank([A, B, C], k=1)[0], A)

    def test_diversity_beats_near_duplicate(self):
        # with diversity weight, the distinct C is chosen over the near-duplicate B
        out = mmr_rerank([A, B, C], k=2, lambda_=0.5)
        self.assertIs(out[0], A)
        self.assertIs(out[1], C)

    def test_lambda_one_is_pure_relevance(self):
        # lambda=1 -> ignore diversity -> top-k strictly by BM25 score
        out = mmr_rerank([A, B, C], k=2, lambda_=1.0)
        self.assertEqual([r.source for r in out], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
