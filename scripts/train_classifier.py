"""Train the ticket classifier and print a held-out evaluation report.

Also reports accuracy across several split seeds, because a single 25% split of
a 180-row dataset has enough variance that one number would be misleading.
"""

from __future__ import annotations

import argparse
import statistics

# Run as `PYTHONPATH=. python scripts/train_classifier.py` (or after `poetry
# install`), so the `ticketiq` package at the repo root is importable.
from ticketiq.config import get_settings
from ticketiq.ml.classifier import TicketClassifier, train_and_evaluate
from ticketiq.ml.dataset import load_dataset, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10, help="seeds for the variance check")
    parser.add_argument("--save", action="store_true", help="persist the model to disk")
    args = parser.parse_args()

    settings = get_settings()
    clf, report = train_and_evaluate(test_size=args.test_size, seed=args.seed)

    print(f"dataset: {settings.dataset_path}")
    print(f"vocabulary: {clf.vectorizer.n_features} features "
          f"(TF-IDF, unigrams + bigrams)\n")
    print(f"--- held-out report (seed={args.seed}) ---")
    print(report.render())

    print("\nconfusion matrix (rows = true, cols = predicted)")
    classes = sorted(report.confusion_matrix)
    print(f"{'':<16}" + "".join(f"{c[:12]:>14}" for c in classes))
    for truth in classes:
        row = report.confusion_matrix[truth]
        print(f"{truth:<16}" + "".join(f"{row.get(c, 0):>14}" for c in classes))

    tickets = load_dataset(settings.dataset_path)
    scores = []
    for seed in range(1, args.seeds + 1):
        train, test = stratified_split(tickets, args.test_size, seed)
        scores.append(TicketClassifier().fit(train).evaluate(test).accuracy)
    print(
        f"\naccuracy across {args.seeds} splits: "
        f"mean={statistics.mean(scores):.3f} "
        f"min={min(scores):.3f} max={max(scores):.3f} "
        f"stdev={statistics.pstdev(scores):.3f}"
    )

    if args.save:
        clf.save(settings.model_path)
        print(f"\nsaved model to {settings.model_path}")


if __name__ == "__main__":
    main()
