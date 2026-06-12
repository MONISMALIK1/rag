"""Pseudo-relevance-feedback query expansion (RM3-lite).

A short query often misses relevant passages that use *different words* for the same
thing. Pseudo-relevance feedback assumes the top BM25 hits are relevant, harvests the
terms that are prominent across them, and appends those to the query — broadening
recall without any thesaurus, embedding model, or extra LLM call.

Reference idea: Lavrenko & Croft (2001) relevance models (RM3); here we use a simple,
deterministic variant — rank candidate terms by how many feedback documents contain
them (then by total frequency), which is fully unit-testable offline.
"""

from __future__ import annotations

from collections import Counter

from .bm25 import tokenize
from .retriever import Retriever


def expand_query(query: str, retriever: Retriever, top_docs: int = 3,
                 n_terms: int = 5) -> str:
    """Return ``query`` broadened with up to ``n_terms`` terms drawn from the top hits.

    The original query is preserved (and still weighted, since its terms repeat);
    the harvested terms are appended. Returns the query unchanged if there are no
    hits or ``n_terms <= 0``.
    """
    if n_terms <= 0:
        return query
    hits = retriever.retrieve(query, k=top_docs)
    if not hits:
        return query

    q_terms = set(tokenize(query))
    doc_freq: Counter[str] = Counter()  # # of feedback docs containing the term
    term_freq: Counter[str] = Counter()  # total occurrences across feedback docs
    for hit in hits:
        toks = tokenize(hit.text)
        term_freq.update(toks)
        doc_freq.update(set(toks))

    candidates = [(doc_freq[t], term_freq[t], t) for t in doc_freq if t not in q_terms]
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    added = [t for _, _, t in candidates[:n_terms]]
    return (query + " " + " ".join(added)) if added else query


__all__ = ["expand_query"]
