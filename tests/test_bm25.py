"""Tests for the BM25 retriever — the from-scratch centerpiece.

The ranker is fully deterministic, so these assert exact behavior: the right
document ranks first, rare terms outweigh common ones, length normalization
works, and degenerate inputs are handled.
"""

import math
import unittest

from rag.bm25 import BM25, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello, World! 42"), ["hello", "world", "42"])

    def test_drops_stopwords(self):
        # "the", "of", "a" are stopwords; "founding", "year" survive.
        self.assertEqual(tokenize("the founding year of a company"),
                         ["founding", "year", "company"])

    def test_empty(self):
        self.assertEqual(tokenize("   ,. ;  "), [])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "the cat sat on the mat",                 # 0
            "the dog chased the cat around the yard",  # 1
            "stock prices rose sharply on tuesday",    # 2
        ]
        self.bm25 = BM25().index(self.docs)

    def test_relevant_doc_ranks_first(self):
        hits = self.bm25.search("dog yard", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 1)  # the dog/yard document

    def test_unrelated_docs_excluded(self):
        # Only doc 2 mentions stocks; nothing else should score.
        hits = self.bm25.search("stock prices", k=3)
        self.assertEqual([i for i, _ in hits], [2])

    def test_scores_descending(self):
        hits = self.bm25.search("cat dog", k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_query_terms_returns_empty(self):
        # All-stopword query reduces to nothing.
        self.assertEqual(self.bm25.search("the of a", k=3), [])

    def test_k_limits_results(self):
        hits = self.bm25.search("cat", k=1)
        self.assertLessEqual(len(hits), 1)

    def test_deterministic_tie_break(self):
        # Two identical docs tie on score; the lower index must come first.
        bm = BM25().index(["alpha beta", "alpha beta"])
        hits = bm.search("alpha", k=2)
        self.assertEqual([i for i, _ in hits], [0, 1])


class IdfTests(unittest.TestCase):
    def test_rare_term_has_higher_idf(self):
        docs = [
            "common common common rare",
            "common common common",
            "common common common",
        ]
        bm = BM25().index(docs)
        self.assertGreater(bm.idf("rare"), bm.idf("common"))

    def test_idf_non_negative_for_ubiquitous_term(self):
        # A term in every document would have negative idf under the raw BM25
        # formula; the smoothed 1+... form keeps it >= 0.
        bm = BM25().index(["shared word", "shared thing", "shared item"])
        self.assertGreaterEqual(bm.idf("shared"), 0.0)

    def test_idf_matches_formula(self):
        docs = ["apple banana", "apple cherry", "date fig"]
        bm = BM25().index(docs)
        n, df_apple = 3, 2
        expected = math.log(1.0 + (n - df_apple + 0.5) / (df_apple + 0.5))
        self.assertAlmostEqual(bm.idf("apple"), expected, places=9)


class LengthNormalizationTests(unittest.TestCase):
    def test_shorter_doc_scores_higher_for_same_term_count(self):
        # Both docs contain "signal" once; the longer one is padded with filler.
        short = "signal here"
        long = "signal " + " ".join(f"filler{i}" for i in range(40))
        bm = BM25().index([short, long])
        s_short = bm.score("signal", 0)
        s_long = bm.score("signal", 1)
        self.assertGreater(s_short, s_long)


class DegenerateTests(unittest.TestCase):
    def test_empty_index_search(self):
        self.assertEqual(BM25().index([]).search("anything"), [])

    def test_num_docs_and_avg_len(self):
        # Avoid stopwords so token counts are predictable: lengths 3 and 2.
        bm = BM25().index(["alpha beta gamma", "delta epsilon"])
        self.assertEqual(bm.num_docs, 2)
        self.assertAlmostEqual(bm.avg_doc_len, 2.5)


if __name__ == "__main__":
    unittest.main()
