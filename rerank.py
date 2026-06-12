"""MMR reranking — diversify the retrieved passages.

Reference: Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based Reranking
for Reordering Documents and Producing Summaries."

BM25 ranks passages purely by relevance, so the top-k can be near-duplicates — three
passages all saying the same thing waste the context window and crowd out a passage
that holds the *other* fact the question needs. Maximal Marginal Relevance fixes that:
from a larger candidate pool it greedily picks the passage that is relevant **and**
dissimilar to what's already chosen::

    MMR = argmax_{d in pool}  [ λ · rel(d) − (1 − λ) · max_{s in selected} sim(d, s) ]

Relevance is the (normalized) BM25 score; similarity is token Jaccard — both pure and
deterministic, so reranking is unit-tested offline. ``λ=1`` reduces to plain BM25;
lower ``λ`` trades a little relevance for more diverse, less redundant context.
"""

from __future__ import annotations

from .bm25 import tokenize
from .retriever import Retrieved


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_rerank(candidates: list[Retrieved], k: int, lambda_: float = 0.7) -> list[Retrieved]:
    """Reorder ``candidates`` by Maximal Marginal Relevance and return the top ``k``.

    ``candidates`` is a relevance-ranked pool (e.g. the BM25 top-N); ``lambda_`` in
    [0, 1] balances relevance (high) against diversity (low).
    """
    if k <= 0 or not candidates:
        return []
    pool = list(candidates)

    top = max(c.score for c in pool) or 1.0
    rel = {id(c): (c.score / top if top else 0.0) for c in pool}
    toks = {id(c): set(tokenize(c.text)) for c in pool}

    selected: list[Retrieved] = []
    while pool and len(selected) < k:
        best, best_val = None, None
        for c in pool:
            max_sim = max((_jaccard(toks[id(c)], toks[id(s)]) for s in selected), default=0.0)
            val = lambda_ * rel[id(c)] - (1.0 - lambda_) * max_sim
            if best_val is None or val > best_val:
                best, best_val = c, val
        selected.append(best)
        pool = [c for c in pool if c is not best]  # remove by identity, not equality
    return selected


__all__ = ["mmr_rerank"]
