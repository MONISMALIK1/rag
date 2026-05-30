"""BM25 Okapi ranking over an inverted index — the retriever, from scratch.

This is the lexical ranking function behind Lucene / Elasticsearch, implemented
with nothing but ``collections.Counter`` and ``math.log``. Given a corpus of
documents and a query, it ranks documents by relevance using three classic
ingredients:

* **term frequency (tf)** — how often a query term appears in a document,
  *saturated* so the 10th occurrence counts far less than the 1st (the ``k1``
  knob controls saturation);
* **inverse document frequency (idf)** — rare terms are discriminating and
  weigh more, common terms weigh less;
* **length normalization** — long documents are penalized so they don't win on
  term count alone (the ``b`` knob controls how much).

The whole thing is deterministic: the same corpus and query always produce the
same ranking, which is exactly what makes it pleasant to unit-test.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework:
BM25 and Beyond".
"""

from __future__ import annotations

import math
import re
from collections import Counter

# A token is a run of letters/digits; everything else is a separator. Lowercase
# so matching is case-insensitive.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, conventional English stoplist. Dropping these keeps the index focused
# on content words — "the founding year of helix" should match on "founding",
# "year", "helix", not on "the" and "of".
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as is are
    was were be been being do does did has have had this that these those it its
    i you he she they we me him her them my your his our their what which who whom
    when where why how not no nor so than too very can will just about into over
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens, and drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """An in-memory BM25 index.

    Build it once with :meth:`index`, then call :meth:`search` as many times as
    you like. ``k1`` and ``b`` are the standard BM25 hyperparameters; the
    defaults (1.5, 0.75) are the widely used Lucene-style settings.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = []   # per-document term-frequency tables
        self._doc_len: list[int] = []        # token count per document
        self._df: Counter[str] = Counter()   # document frequency per term
        self._idf: dict[str, float] = {}     # precomputed idf per term
        self._avgdl: float = 0.0             # average document length
        self._n: int = 0                     # number of documents

    # -- building -----------------------------------------------------------

    def index(self, documents: list[str]) -> "BM25":
        """Tokenize and index ``documents``. Returns ``self`` for chaining."""
        tokenized = [tokenize(d) for d in documents]
        self._tf = [Counter(toks) for toks in tokenized]
        self._doc_len = [len(toks) for toks in tokenized]
        self._n = len(tokenized)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0

        # Document frequency: in how many documents does each term appear?
        df: Counter[str] = Counter()
        for toks in tokenized:
            df.update(set(toks))
        self._df = df

        # BM25's idf, with the +0.5 smoothing of the probabilistic model. The
        # leading ``1 +`` keeps idf non-negative even for terms that appear in
        # more than half the corpus (the raw form can go negative there).
        self._idf = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    # -- querying -----------------------------------------------------------

    def score(self, query: str | list[str], doc_idx: int) -> float:
        """BM25 score of a single document against ``query``."""
        terms = tokenize(query) if isinstance(query, str) else query
        tf = self._tf[doc_idx]
        dl = self._doc_len[doc_idx]
        # Guard against an empty index / empty document.
        norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl)) if self._avgdl else self.k1

        total = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
        return total

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(doc_index, score)`` pairs, best first.

        Documents that no query term touches (score 0) are omitted. Ties break
        toward the lower document index, so results are fully deterministic.
        """
        terms = tokenize(query)
        if not terms or self._n == 0:
            return []
        scored = [(i, self.score(terms, i)) for i in range(self._n)]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    # -- introspection ------------------------------------------------------

    def idf(self, term: str) -> float:
        """Inverse document frequency of a single (already-lowercased) term."""
        return self._idf.get(term, 0.0)

    @property
    def num_docs(self) -> int:
        return self._n

    @property
    def avg_doc_len(self) -> float:
        return self._avgdl
