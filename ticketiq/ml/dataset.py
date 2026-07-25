"""Loading and splitting of the labelled ticket dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ticketiq.ml.text import join_ticket


@dataclass(frozen=True)
class LabelledTicket:
    subject: str
    body: str
    category: str
    tier: str = "free"

    @property
    def text(self) -> str:
        return join_ticket(self.subject, self.body)


def load_dataset(path: Path) -> list[LabelledTicket]:
    """Read a JSONL dataset of labelled tickets."""
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset not found at {path}; run `python scripts/generate_dataset.py`"
        )
    tickets: list[LabelledTicket] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tickets.append(
            LabelledTicket(
                subject=row["subject"],
                body=row["body"],
                category=row["category"],
                tier=row.get("tier", "free"),
            )
        )
    return tickets


def stratified_split(
    tickets: Sequence[LabelledTicket],
    test_size: float = 0.25,
    seed: int = 1,
) -> tuple[list[LabelledTicket], list[LabelledTicket]]:
    """Split preserving the class balance, deterministically for a given seed."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    by_class: dict[str, list[LabelledTicket]] = defaultdict(list)
    for t in tickets:
        by_class[t.category].append(t)

    rng = random.Random(seed)
    train: list[LabelledTicket] = []
    test: list[LabelledTicket] = []
    for category in sorted(by_class):
        rows = by_class[category][:]
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * test_size))
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
