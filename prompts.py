"""Prompts and the small bits of text-wrangling around them.

The RAG prompt is the heart of "augmented generation": retrieved passages are
numbered and pasted above the question, and the model is told — firmly — to
answer *only* from them, to cite the passages it used, and to admit when the
answer isn't there. That instruction is what turns a confabulating model into a
grounded one.

``NO_CONTEXT_PROMPT`` is the matched baseline: the same question with no
passages, so the model must answer from parametric memory alone. The benchmark
runs both to show the gap retrieval closes.
"""

from __future__ import annotations

import re

from .retriever import Retrieved

RAG_PROMPT = """\
You are a careful assistant. Answer the question using ONLY the numbered context
passages below. Cite the passage number(s) you relied on in square brackets,
like [1] or [2][3]. If the passages do not contain the answer, reply exactly:
"I don't know based on the provided context." Do not use outside knowledge.

Context:
{context}

Question: {question}
Answer:"""

NO_CONTEXT_PROMPT = """\
Answer the question as accurately as you can. If you are not sure, say so rather
than guessing.

Question: {question}
Answer:"""

# Matches a citation marker like [1] or [12].
_CITE_RE = re.compile(r"\[(\d+)\]")


def format_context(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered, source-tagged block.

        [1] (handbook.md#1) Helix Dynamics was founded in 2019 ...
        [2] (handbook.md#3) The Kestrel-7 is powered by ...

    The numbering here is what the model cites, and it lines up with the order
    of ``retrieved`` so callers can map a citation back to its chunk.
    """
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(f"[{i}] ({r.source}) {r.text}")
    return "\n".join(lines)


def extract_citations(answer: str) -> list[int]:
    """Pull the distinct ``[n]`` citation numbers out of a model answer.

    Returns them in first-appearance order, de-duplicated.
    """
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
