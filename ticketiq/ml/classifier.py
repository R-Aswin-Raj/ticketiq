"""Ticket classifier: TF-IDF + a from-scratch softmax logistic regression.

Both the vectoriser and the model are hand-rolled (no scikit-learn), and the
model is isolated behind ``SoftmaxRegression`` so this layer only orchestrates
fit / evaluate / predict / persistence.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticketiq.config import get_settings
from ticketiq.ml.dataset import LabelledTicket, load_dataset, stratified_split
from ticketiq.ml.logistic_regression import SoftmaxRegression
from ticketiq.ml.metrics import EvaluationReport, evaluate
from ticketiq.ml.text import join_ticket
from ticketiq.ml.vectorizer import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    category: str
    confidence: float
    scores: dict[str, float]


class TicketClassifier:
    """Thin orchestration layer over the hand-rolled vectoriser and model."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        model: SoftmaxRegression | None = None,
    ) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer()
        self.model = model or SoftmaxRegression()
        self.report: EvaluationReport | None = None

    # -- training -------------------------------------------------------
    def fit(self, tickets: Sequence[LabelledTicket]) -> TicketClassifier:
        texts = [t.text for t in tickets]
        labels = [t.category for t in tickets]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels, n_features=self.vectorizer.n_features)
        return self

    def evaluate(self, tickets: Sequence[LabelledTicket]) -> EvaluationReport:
        X = self.vectorizer.transform([t.text for t in tickets])
        y_true = [t.category for t in tickets]
        y_pred = self.model.predict(X)
        self.report = evaluate(y_true, y_pred, labels=self.model.classes_)
        return self.report

    # -- inference ------------------------------------------------------
    def predict(self, subject: str, body: str) -> Prediction:
        if not self.model.classes_:
            raise RuntimeError("classifier is not trained")
        vec = self.vectorizer.transform_one(join_ticket(subject, body))
        label, confidence, scores = self.model.predict_one(vec)
        return Prediction(category=label, confidence=confidence, scores=scores)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "vectorizer": self.vectorizer.to_dict(),
            "model": self.model.to_dict(),
            "report": self.report.to_dict() if self.report else None,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        logger.info("saved classifier to %s", path)

    @classmethod
    def load(cls, path: Path) -> TicketClassifier:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vectorizer=TfidfVectorizer.from_dict(payload["vectorizer"]),
            model=SoftmaxRegression.from_dict(payload["model"]),
        )


_lock = threading.Lock()
_cached: TicketClassifier | None = None


def train_and_evaluate(
    dataset_path: Path | None = None,
    test_size: float = 0.25,
    seed: int = 1,
) -> tuple[TicketClassifier, EvaluationReport]:
    """Train on a stratified split and report held-out metrics."""
    settings = get_settings()
    tickets = load_dataset(dataset_path or settings.dataset_path)
    train, test = stratified_split(tickets, test_size=test_size, seed=seed)
    clf = TicketClassifier().fit(train)
    report = clf.evaluate(test)
    # Refit on all data for serving; metrics above stay held-out and honest.
    serving = TicketClassifier().fit(tickets)
    serving.report = report
    return serving, report


def get_classifier() -> TicketClassifier:
    """Process-wide singleton, trained lazily on first use."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        settings = get_settings()
        if settings.model_path.is_file():
            try:
                _cached = TicketClassifier.load(settings.model_path)
                return _cached
            except (OSError, KeyError, ValueError):
                # Incompatible or corrupt file (e.g. an older model format): retrain.
                logger.warning("could not load %s, retraining", settings.model_path)
        clf, report = train_and_evaluate()
        logger.info("trained classifier: macro_f1=%.3f", report.macro_f1)
        try:
            clf.save(settings.model_path)
        except OSError:
            logger.warning("could not persist classifier to %s", settings.model_path)
        _cached = clf
        return _cached


def reset_classifier_cache() -> None:
    """Test hook."""
    global _cached
    with _lock:
        _cached = None
