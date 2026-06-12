"""The RAG loop: retrieve, then generate grounded in what was retrieved.

Reference: Lewis et al. 2020, "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks", https://arxiv.org/abs/2005.11401.

    passages = retrieve(question)         # find the relevant evidence
    prompt   = ground(question, passages) # paste evidence above the question
    answer   = LLM(prompt)                # generate, citing the evidence

The original paper retrieves with a dense neural encoder (DPR). This
implementation swaps in a from-scratch **BM25** lexical retriever (see
``bm25.py``) so the whole system stays dependency-free and deterministic — the
retrieval half can be tested without a single network call. ``answer_no_retrieval``
is the matched baseline: the same question, no evidence, model memory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expand import expand_query
from .llm import chat
from .prompts import NO_CONTEXT_PROMPT, RAG_PROMPT, extract_citations, format_context
from .rerank import mmr_rerank
from .retriever import Retrieved, Retriever


@dataclass
class RAGResult:
    answer: str                              # the model's text answer
    retrieved: list[Retrieved] = field(default_factory=list)  # evidence used
    citations: list[int] = field(default_factory=list)         # [n] markers cited
    used_retrieval: bool = True              # False for the baseline path
    prompt: str = ""                         # the exact prompt sent to the model

    @property
    def cited_sources(self) -> list[str]:
        """Source labels (e.g. 'handbook.md#1') for the passages the model cited.

        Citations are 1-based and index into ``retrieved``; out-of-range markers
        (a model citing a passage that wasn't shown) are ignored.
        """
        out = []
        for n in self.citations:
            if 1 <= n <= len(self.retrieved):
                out.append(self.retrieved[n - 1].source)
        return out


def answer(
    question: str,
    retriever: Retriever,
    k: int = 3,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    mmr: bool = False,
    fetch_k: int | None = None,
    mmr_lambda: float = 0.7,
    expand: bool = False,
    expand_terms: int = 5,
) -> RAGResult:
    """Retrieve the top-``k`` passages for ``question`` and answer grounded in them.

    With ``expand=True`` the *search* query is first broadened by pseudo-relevance
    feedback (the model still sees the original question). With ``mmr=True``, a larger
    pool of ``fetch_k`` BM25 candidates is reranked by Maximal Marginal Relevance down
    to ``k``, trading a little relevance for diversity so the context is less redundant.
    """
    search_q = expand_query(question, retriever, n_terms=expand_terms) if expand else question
    if mmr:
        pool = retriever.retrieve(search_q, k=fetch_k or max(k * 4, 10))
        retrieved = mmr_rerank(pool, k=k, lambda_=mmr_lambda)
    else:
        retrieved = retriever.retrieve(search_q, k=k)
    context = format_context(retrieved)
    prompt = RAG_PROMPT.format(context=context, question=question)
    text = chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    return RAGResult(
        answer=text,
        retrieved=retrieved,
        citations=extract_citations(text),
        used_retrieval=True,
        prompt=prompt,
    )


def answer_no_retrieval(
    question: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> RAGResult:
    """Baseline: answer from the model's parametric memory, no retrieval."""
    prompt = NO_CONTEXT_PROMPT.format(question=question)
    text = chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    return RAGResult(answer=text, retrieved=[], citations=[], used_retrieval=False, prompt=prompt)
