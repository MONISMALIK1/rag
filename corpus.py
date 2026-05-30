"""Turn a pile of text into retrievable, citable chunks.

A *chunk* is one passage — small enough that several fit in a prompt, large
enough to carry a self-contained fact. We split documents on blank lines
(paragraph boundaries) and remember where each chunk came from so the answer
can cite it.

Two sources are supported:

* :func:`load_dir` — read ``.txt`` / ``.md`` files from a directory (your own
  knowledge base);
* :func:`load_sample` — a small, deliberately *fictional* knowledge base bundled
  with the package. It's fictional on purpose: a model can't have memorized
  facts that were invented for this repo, so retrieval is the *only* way to
  answer questions about it. That makes the RAG-vs-no-retrieval contrast honest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage plus a human-readable provenance label."""

    text: str
    source: str  # e.g. "handbook.md#2" — file (or doc) name and chunk number


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split one document into paragraph chunks on blank lines.

    Consecutive non-blank lines form a paragraph; runs of blank lines separate
    them. Each chunk is tagged ``{source}#{n}`` (1-based) for citation.
    """
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    return [Chunk(text=p, source=f"{source}#{i}") for i, p in enumerate(paragraphs, 1)]


def load_dir(path: str) -> list[Chunk]:
    """Load and chunk every ``.txt`` / ``.md`` file under ``path``.

    Files are visited in sorted order so the resulting corpus — and therefore
    every retrieval over it — is deterministic.
    """
    chunks: list[Chunk] = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not name.lower().endswith((".txt", ".md")):
                continue
            full = os.path.join(root, name)
            with open(full, encoding="utf-8") as fh:
                chunks.extend(chunk_text(fh.read(), source=name))
    return chunks


# --------------------------------------------------------------------------- #
# Bundled sample knowledge base.
#
# Helix Dynamics is an invented company. Every fact below is fictional, which is
# the whole point: the language model has no way to know any of it from training,
# so a correct answer can only come from retrieval.
# --------------------------------------------------------------------------- #

_SAMPLE_HANDBOOK = """\
Helix Dynamics was founded in 2019 in Trondheim, Norway, by Dr. Mira Solberg and
Anton Vex. The company builds autonomous logistics robots for indoor warehouses.

The company's flagship product is the Kestrel-7, a four-rotor autonomous
warehouse drone that picks and ferries small parcels between shelving and packing
stations. It was first shipped to customers in March 2022.

The Kestrel-7 is powered by a graphene-hybrid battery the company calls the
AuroraCell. A single AuroraCell charge gives the Kestrel-7 about nine hours of
continuous flight, and it recharges to eighty percent in twenty-five minutes.

Helix Dynamics runs all of its drones on an in-house fleet-coordination platform
named Loomweave. Loomweave assigns routes, prevents mid-air collisions, and
rebalances the fleet when a drone drops below fifteen percent battery.

As of 2024 the company employs 240 people across three offices: the Trondheim
headquarters, an engineering office in Tallinn, Estonia, and a sales office in
Austin, Texas.

Dr. Mira Solberg served as chief executive until 2023, when she moved to the role
of chief technology officer. Anton Vex has been the chief executive officer since
then.

The Kestrel-7 carries a maximum payload of 4.5 kilograms and is certified for
indoor operation under the fictional EN-Aero 19 safety standard. It is not
approved for outdoor flight.

Helix Dynamics sells the Kestrel-7 only as part of an annual subscription that
bundles hardware, the Loomweave platform, and maintenance. The standard plan
costs 18,000 euros per drone per year.

The company's customer support line operates from 07:00 to 19:00 Central European
Time on weekdays. Critical fleet outages are covered by a separate 24-hour
emergency hotline included in every subscription.

Helix Dynamics offers a 30-day return policy on new subscriptions. After thirty
days the annual subscription is non-refundable, though it can be cancelled before
the next yearly renewal.
"""


def load_sample() -> list[Chunk]:
    """The bundled fictional 'Helix Dynamics' knowledge base, as chunks."""
    return chunk_text(_SAMPLE_HANDBOOK, source="handbook.md")
