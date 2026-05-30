"""CLI for RAG.

Usage:
    # Ask a question against the bundled 'Helix Dynamics' knowledge base
    python -m rag "When was Helix Dynamics founded?"

    # Show the passages the retriever pulled (and their BM25 scores)
    python -m rag "..." --show-context

    # Use your own documents instead of the sample corpus
    python -m rag "..." --corpus ./my_docs --k 5

    # Baseline: answer with no retrieval (model memory only)
    python -m rag "..." --no-retrieval

    # Benchmark RAG vs no-retrieval on the bundled fictional QA set
    python -m rag --bench
"""

from __future__ import annotations

import argparse
import sys
import time

from .core import answer, answer_no_retrieval
from .corpus import load_dir, load_sample
from .evalset import SAMPLE_QA
from .llm import DEFAULT_MODEL
from .retriever import Retriever


def _build_retriever(corpus_path: str | None) -> Retriever:
    if corpus_path:
        chunks = load_dir(corpus_path)
        if not chunks:
            print(f"No .txt/.md files found under {corpus_path}", file=sys.stderr)
            raise SystemExit(2)
    else:
        chunks = load_sample()
    return Retriever(chunks)


def _bench(args, model: str) -> int:
    retriever = _build_retriever(args.corpus)
    qa = SAMPLE_QA[: args.num] if args.num else SAMPLE_QA

    rag_correct = 0
    base_correct = 0
    total_secs = 0.0

    for i, item in enumerate(qa, 1):
        t0 = time.monotonic()
        rag = answer(item.question, retriever, k=args.k, model=model)
        base = answer_no_retrieval(item.question, model=model)
        secs = time.monotonic() - t0
        total_secs += secs

        rag_ok = item.is_correct(rag.answer)
        base_ok = item.is_correct(base.answer)
        rag_correct += int(rag_ok)
        base_correct += int(base_ok)

        cites = ",".join(f"[{n}]" for n in rag.citations) or "—"
        print(
            f"[{i:2d}/{len(qa)}] RAG {'OK' if rag_ok else '--'} cites {cites:<10} "
            f"| base {'OK' if base_ok else '--'} | {secs:4.1f}s  {item.question}",
            flush=True,
        )

    n = len(qa)
    print("\n" + "=" * 72)
    print(f"Helix Dynamics QA — {n} questions, model={model}")
    print("=" * 72)
    print(f"  RAG  (BM25 retrieval): {rag_correct}/{n} = {rag_correct / n * 100:5.1f}%")
    print(f"  Baseline (no context): {base_correct}/{n} = {base_correct / n * 100:5.1f}%")
    print(f"  RAG advantage:         {rag_correct - base_correct:+d} questions")
    print(f"  Avg time: {total_secs / n:.1f}s / question")
    print("\nThe corpus is fictional, so the baseline can only guess — the gap is"
          "\nexactly the knowledge retrieval supplies.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="rag",
        description="RAG: Retrieval-Augmented Generation (Lewis et al., 2020), "
                    "with a from-scratch BM25 retriever.",
    )
    p.add_argument("question", nargs="?", help="The question to answer.")
    p.add_argument("--corpus", default=None,
                   help="Directory of .txt/.md files to use (default: bundled sample).")
    p.add_argument("--k", type=int, default=3, help="Passages to retrieve (default: 3).")
    p.add_argument("--no-retrieval", action="store_true",
                   help="Baseline: answer without retrieval (model memory only).")
    p.add_argument("--show-context", action="store_true",
                   help="Print the retrieved passages and their BM25 scores.")
    p.add_argument("--model", default=None, help=f"Model slug (default: {DEFAULT_MODEL}).")

    p.add_argument("--bench", action="store_true",
                   help="Benchmark RAG vs no-retrieval on the bundled QA set.")
    p.add_argument("--num", type=int, default=0, help="Bench: limit number of questions.")
    args = p.parse_args()

    model = args.model or DEFAULT_MODEL

    if args.bench:
        return _bench(args, model)

    if not args.question:
        p.error("provide a question, or use --bench")

    print(f"\nQuestion: {args.question}", file=sys.stderr)
    print(f"Model: {model}\n", file=sys.stderr, flush=True)

    if args.no_retrieval:
        res = answer_no_retrieval(args.question, model=model)
        print("=" * 60)
        print(f"Answer (no retrieval): {res.answer}")
        return 0

    retriever = _build_retriever(args.corpus)
    res = answer(args.question, retriever, k=args.k, model=model)

    if args.show_context:
        print("--- retrieved passages ---")
        for i, r in enumerate(res.retrieved, 1):
            print(f"[{i}] ({r.source}) score={r.score:.3f}")
            print(f"    {r.text}")
        print("--------------------------")

    print("=" * 60)
    print(f"Answer (RAG): {res.answer}")
    if res.cited_sources:
        print(f"Cited: {', '.join(res.cited_sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
