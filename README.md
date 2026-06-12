# RAG — Retrieval-Augmented Generation

[![tests](https://github.com/MONISMALIK1/rag/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/rag/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **Retrieval-Augmented
Generation** (Lewis et al., NeurIPS 2020 —
[arXiv:2005.11401](https://arxiv.org/abs/2005.11401)).

A language model only knows what was baked into its weights at training time.
Ask it about your internal docs, a paper from last week, or a specific PDF on
your disk and it either doesn't know or — worse — confidently makes something
up. RAG hands the model an open book: **retrieve the relevant passages first,
paste them above the question, and let the model answer grounded in that
evidence** — with citations, and an honest "I don't know" when the answer isn't
there.

```
   question ──► retriever ──► top-k passages ──► prompt ──► LLM ──► grounded answer
                    │                                                   │
            (searches YOUR corpus,                              (cites the passages
             finds what's relevant)                              it actually used)
```

The retriever here is the real star: a **BM25 index built from scratch** over an
inverted index — pure `collections.Counter` + `math.log`, no embedding model, no
vector database, no third-party anything. Because it's deterministic, the entire
*retrieval* half of the system is unit-tested offline without a single network
call.

## The mechanic

```text
Q: When was Helix Dynamics founded?

BM25 retrieves from the corpus:
    [1] (handbook.md#1) Helix Dynamics was founded in 2019 in Trondheim, Norway,
        by Dr. Mira Solberg and Anton Vex...

The model is told "answer ONLY from the passages, and cite them", and replies:
    Helix Dynamics was founded in 2019 in Trondheim, Norway. [1]
```

`core.py` (`answer`) is the whole loop: retrieve the top-*k* chunks, format them
into a grounded prompt, call the model, parse the `[n]` citations back out. The
matched baseline (`answer_no_retrieval`) asks the same question with **no**
context, so `--bench` can measure exactly how much the retrieval adds.

## How BM25 works (the part worth reading)

Three classic ingredients, all in [`bm25.py`](bm25.py):

- **Term frequency**, *saturated* — a word appearing 10× in a document isn't 10×
  as relevant. The `k1` knob controls how fast the payoff flattens.
- **Inverse document frequency** — a term in every document (`the`, `company`)
  discriminates nothing; a rare term (`AuroraCell`) is highly diagnostic. Rare
  terms get more weight, via `math.log`.
- **Length normalization** — a long document shouldn't win just by having more
  words to match. The `b` knob controls how hard long documents are penalized.

```python
idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl/avgdl))
```

That single line, summed over the query terms, *is* BM25. It's the ranking
function inside Lucene and Elasticsearch, and it remains a hard-to-beat baseline
today.

## Install

Zero third-party dependencies — pure Python standard library plus an OpenRouter
HTTP call. Requires Python ≥ 3.11.

```bash
git clone https://github.com/MONISMALIK1/rag.git
cd rag
export OPENROUTER_API_KEY="sk-or-..."   # get one at https://openrouter.ai/keys
```

> **Never hard-code your key.** It is read only from the `OPENROUTER_API_KEY`
> environment variable. The default model is the free `openai/gpt-oss-120b:free`
> (override with `RAG_MODEL` or `--model`).

## Usage

The repo ships with a small, deliberately **fictional** knowledge base (the
"Helix Dynamics" company handbook) so you can see RAG work immediately — and so
the demo is honest: the model can't have memorized facts that were invented for
this repo, so a correct answer can *only* come from retrieval.

```bash
# Ask a question against the bundled knowledge base
python -m rag "When was Helix Dynamics founded?"

# See the passages BM25 retrieved, with their scores
python -m rag "Which battery powers the Kestrel-7?" --show-context

# Point it at your own documents (.txt / .md)
python -m rag "What's our refund policy?" --corpus ./my_docs --k 5

# Diversify the retrieved passages with MMR (less redundant context)
python -m rag "Which battery powers the Kestrel-7?" --mmr --show-context

# Broaden a short query with pseudo-relevance feedback before retrieving
python -m rag "Kestrel power?" --expand --show-context

# Baseline: answer with no retrieval (model memory only) — watch it guess
python -m rag "When was Helix Dynamics founded?" --no-retrieval

# Benchmark RAG vs no-retrieval, head to head
python -m rag --bench
```

As a library:

```python
from rag import Retriever, load_sample, answer

retriever = Retriever(load_sample())
res = answer("Which battery powers the Kestrel-7?", retriever, k=3)

print(res.answer)         # "The Kestrel-7 is powered by the AuroraCell. [2]"
print(res.citations)      # [2]
print(res.cited_sources)  # ["handbook.md#3"]
```

The retriever stands alone, too:

```python
from rag import BM25

bm25 = BM25().index(["the cat sat", "stock prices rose on tuesday"])
print(bm25.search("tuesday stocks", k=1))   # [(1, 0.60...)]  -> doc 1 wins on "tuesday"
```

## Diversity reranking (MMR)

Plain BM25 ranks by relevance alone, so the top-`k` passages can be near-duplicates —
three chunks all saying the same thing, wasting the context window and crowding out the
*other* fact the question needs. **Maximal Marginal Relevance** (Carbonell & Goldstein,
1998) reranks a larger candidate pool to be relevant **and** diverse:

```
MMR = argmax_d  [ λ · relevance(d) − (1 − λ) · max similarity(d, already-chosen) ]
```

Here relevance is the (normalized) BM25 score and similarity is token Jaccard — pure and
deterministic, so it's unit-tested offline. Turn it on with `--mmr` (or `answer(...,
mmr=True)`); `--mmr-lambda` tunes the balance (`1.0` = plain BM25, lower = more diverse).

```python
from rag import answer, Retriever, load_sample
res = answer("...", Retriever(load_sample()), k=3, mmr=True, mmr_lambda=0.6)
```

## Query expansion (pseudo-relevance feedback)

A short query can miss passages that use *different words* for the same idea — ask
"Kestrel power?" and BM25 may never reach the chunk that says "lithium battery."
**Pseudo-relevance feedback** assumes the top hits are relevant, harvests the terms most
prominent across them, and appends those to the query to broaden recall — no thesaurus,
embeddings, or extra LLM call:

```
"Kestrel power?"  ──retrieve──►  top hits mention "lithium", "battery", "drone"
        └──────────────► "Kestrel power? lithium battery drone"  ──retrieve again──► more, better passages
```

Turn it on with `--expand` (or `answer(..., expand=True)`); `--expand-terms` caps how many
terms are added. It composes with `--mmr` (expand first, then diversify). Pure and
deterministic — unit-tested offline.

## ⚠️ Honest caveat: lexical vs. dense retrieval

The original RAG paper retrieves with a **dense neural encoder** (DPR) — it
matches on *meaning*, so "how long can the drone fly" finds a passage about
"nine hours of **flight**" even though the words differ. This implementation uses
**BM25**, a **lexical** retriever: it matches on *words*, so `fly` and `flight`
are different tokens and won't match. That's the deliberate trade-off that keeps
this repo dependency-free and deterministic.

BM25 is genuinely strong — it's a standard, still-deployed baseline and the
sparse half of most modern *hybrid* search systems. But for production semantic
search you'd typically add or swap in dense embeddings. The retrieval interface
here (`Retriever.retrieve`) is small on purpose, so a dense retriever could drop
into the same seam.

## Tests

Fully offline — the LLM is mocked and the BM25 retriever runs for real (it needs
no network), so the suite needs no API key. The retriever tests are the heart of
the project: ranking order, idf behavior, and length normalization are all
asserted exactly.

```bash
# Run from the parent directory so `rag/` is importable (mirrors CI)
cd ..
python -m unittest discover -s rag/tests -t . -v
```

Coverage highlights:

- **`test_bm25.py`** — the relevant document ranks first, rare terms outweigh
  common ones, length normalization favors the shorter match, idf matches the
  closed-form formula, ties break deterministically.
- **`test_retriever.py`** — over the real corpus, the right passage surfaces for
  factual questions (founding year, battery, platform).
- **`test_core.py`** — the retrieved evidence actually reaches the prompt the
  model sees (grounding), citations are parsed and mapped to sources, and the
  baseline path genuinely sends no context.
- **`test_corpus.py` / `test_prompts.py`** — paragraph chunking with provenance
  labels, and `[n]` citation parsing.

## Layout

```
rag/
├── bm25.py        # inverted index + BM25 scoring — the from-scratch retriever
├── corpus.py      # load/chunk documents + bundled fictional knowledge base
├── retriever.py   # BM25 over corpus chunks (retrieve top-k)
├── rerank.py      # MMR reranking — diversify the retrieved passages
├── expand.py      # pseudo-relevance-feedback query expansion
├── prompts.py     # grounded RAG prompt + context formatting + citation parsing
├── core.py        # answer() and answer_no_retrieval() (with optional mmr=True)
├── evalset.py     # fictional QA set for the benchmark
├── llm.py         # OpenRouter HTTP wrapper (stdlib only)
├── __main__.py    # CLI (ask, --show-context, --corpus, --no-retrieval, --bench)
└── tests/         # offline unit tests (LLM mocked, retriever real)
```

## Citation

```bibtex
@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and
               Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and
               K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and
               Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

## Related — RAG variants

This implements the original RAG (retrieve with the raw query). Three from-scratch
follow-ups build on it:

- **[hyde](https://github.com/MONISMALIK1/hyde)** — retrieve with an LLM-written
  *hypothetical answer document* instead of the raw query (Gao et al., 2022).
- **[self_rag](https://github.com/MONISMALIK1/self_rag)** — grade the retrieved
  passages and **abstain** when they're weak (Asai et al., 2023).
- **[corrective_rag](https://github.com/MONISMALIK1/corrective_rag)** — grade
  retrieval and **correct** it: refine, fall back to external knowledge, or combine
  (Yan et al., 2024).

## License

MIT — see [LICENSE](LICENSE).
