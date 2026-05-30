"""A tiny QA set over the bundled 'Helix Dynamics' knowledge base.

Each item pairs a question with one or more acceptable answer substrings. Because
every fact is fictional (invented for this repo), a model answering from memory
alone should miss them — while a model handed the retrieved passage should get
them right. That's the contrast ``python -m rag --bench`` reports.

Grading is deliberately simple and deterministic: an answer counts as correct if
it contains any of the accepted substrings, case-insensitively.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QA:
    question: str
    accept: tuple[str, ...]  # answer is correct if it contains any of these

    def is_correct(self, answer: str) -> bool:
        low = answer.lower()
        return any(a.lower() in low for a in self.accept)


SAMPLE_QA: list[QA] = [
    QA("In what year was Helix Dynamics founded?", ("2019",)),
    QA("Who founded Helix Dynamics?", ("Solberg", "Vex")),
    QA("What is the flagship product of Helix Dynamics?", ("Kestrel-7", "Kestrel")),
    QA("Which battery powers the Kestrel-7?", ("AuroraCell",)),
    QA("How many hours can the Kestrel-7 fly on a single charge?", ("nine", "9")),
    QA("What is the name of the fleet-coordination platform?", ("Loomweave",)),
    QA("How many people does Helix Dynamics employ?", ("240",)),
    QA("Who is the current chief executive officer of Helix Dynamics?", ("Vex",)),
    QA("What is the maximum payload of the Kestrel-7?", ("4.5",)),
    QA("How much does the standard Kestrel-7 subscription cost per year?", ("18,000", "18000")),
    QA("How many days is the return window on a new subscription?", ("30", "thirty")),
    QA("In which city is Helix Dynamics headquartered?", ("Trondheim",)),
]
