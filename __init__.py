"""RAG: Retrieval-Augmented Generation (Lewis et al., 2020).

Retrieve relevant passages from a corpus, paste them above the question, and let
the model answer grounded in that evidence — with citations, and an honest
"I don't know" when the evidence isn't there.

The retriever is a from-scratch **BM25** index (pure stdlib), so the retrieval
half is fully deterministic and testable offline; only generation needs the LLM.

Public API:
    Retriever(chunks)                       # BM25 index over a corpus
    answer(question, retriever, k=3, ...)   # retrieve -> ground -> generate
    answer_no_retrieval(question, ...)      # baseline: model memory only
    load_sample() / load_dir(path)          # build a corpus of chunks
    BM25                                    # the ranking function itself
"""

from .bm25 import BM25, tokenize
from .core import RAGResult, answer, answer_no_retrieval
from .corpus import Chunk, chunk_text, load_dir, load_sample
from .evalset import QA, SAMPLE_QA
from .llm import DEFAULT_MODEL, chat
from .prompts import NO_CONTEXT_PROMPT, RAG_PROMPT, extract_citations, format_context
from .retriever import Retrieved, Retriever

__all__ = [
    "BM25",
    "Chunk",
    "DEFAULT_MODEL",
    "NO_CONTEXT_PROMPT",
    "QA",
    "RAGResult",
    "RAG_PROMPT",
    "Retrieved",
    "Retriever",
    "SAMPLE_QA",
    "answer",
    "answer_no_retrieval",
    "chat",
    "chunk_text",
    "extract_citations",
    "format_context",
    "load_dir",
    "load_sample",
    "tokenize",
]
