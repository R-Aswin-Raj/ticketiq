"""Multinomial logistic regression (softmax), implemented from scratch.

A discriminative linear classifier. Rather than modelling ``P(features | class)``
under a feature-independence assumption, it optimises ``P(class | features)``
directly, by gradient descent on the cross-entropy loss with L2 regularisation.
On TF-IDF features that lets it account for correlated terms (a bigram and its
own unigrams) instead of double-counting them, at the cost of an iterative fit.

    z_c(x)     = w_c · x + b_c
    P(c | x)   = softmax(z)_c = exp(z_c) / Σ_k exp(z_k)
    loss       = -(1/N) Σ_x log P(y_x | x) + (λ/2) Σ_c ||w_c||²
    ∂/∂w_c     = (1/N) Σ_x (P(c|x) − 1[y_x = c]) x  +  λ w_c

The loss is convex, so gradient descent from a zero start reaches the global
optimum; that also makes training deterministic without a random seed. Weights
are stored per class as dense lists because the vocabulary (~1200 features) is
small enough that a dense weight matrix is simpler than a sparse one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ticketiq.ml.vectorizer import SparseVector


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax (shift by the max logit before exponentiating)."""
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class SoftmaxRegression:
    """Multinomial logistic regression trained with full-batch gradient descent."""

    learning_rate: float = 5.0
    l2: float = 1e-4
    epochs: int = 120
    classes_: list[str] = field(default_factory=list)
    # weights_[class_index][feature_index]
    weights_: list[list[float]] = field(default_factory=list)
    bias_: list[float] = field(default_factory=list)
    n_features_: int = 0

    # -- training -------------------------------------------------------
    def fit(
        self,
        X: Sequence[SparseVector],
        y: Sequence[str],
        n_features: int,
    ) -> SoftmaxRegression:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("cannot fit on an empty dataset")

        self.classes_ = sorted(set(y))
        self.n_features_ = n_features
        k = len(self.classes_)
        index_of = {c: i for i, c in enumerate(self.classes_)}
        targets = [index_of[label] for label in y]
        n = len(X)

        self.weights_ = [[0.0] * n_features for _ in range(k)]
        self.bias_ = [0.0] * k

        # Only features that actually occur can ever get a non-zero weight (an
        # unseen feature has zero gradient and zero L2 pull), so the weight-decay
        # pass iterates this set rather than the whole vocabulary.
        active = sorted({idx for vec in X for idx in vec if 0 <= idx < n_features})

        for _ in range(self.epochs):
            grad_w = [[0.0] * n_features for _ in range(k)]
            grad_b = [0.0] * k
            for vec, target in zip(X, targets, strict=True):
                probs = _softmax(self._logits(vec))
                for c in range(k):
                    err = probs[c] - (1.0 if c == target else 0.0)
                    if err == 0.0:
                        continue
                    grad_b[c] += err
                    row = grad_w[c]
                    for idx, value in vec.items():
                        if 0 <= idx < n_features:
                            row[idx] += err * value

            for c in range(k):
                w = self.weights_[c]
                gw = grad_w[c]
                for idx in active:
                    grad = gw[idx] / n + self.l2 * w[idx]
                    w[idx] -= self.learning_rate * grad
                self.bias_[c] -= self.learning_rate * (grad_b[c] / n)
        return self

    # -- inference ------------------------------------------------------
    def _logits(self, vec: SparseVector) -> list[float]:
        out: list[float] = []
        for ci in range(len(self.classes_)):
            w = self.weights_[ci]
            z = self.bias_[ci]
            for idx, value in vec.items():
                if 0 <= idx < self.n_features_:
                    z += w[idx] * value
            out.append(z)
        return out

    def predict_proba_one(self, vec: SparseVector) -> dict[str, float]:
        if not self.classes_:
            raise RuntimeError("model is not trained")
        probs = _softmax(self._logits(vec))
        return dict(zip(self.classes_, probs, strict=True))

    def predict_one(self, vec: SparseVector) -> tuple[str, float, dict[str, float]]:
        proba = self.predict_proba_one(vec)
        label = max(proba, key=lambda k: proba[k])
        return label, proba[label], proba

    def predict(self, X: Sequence[SparseVector]) -> list[str]:
        return [self.predict_one(vec)[0] for vec in X]

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
            "classes": self.classes_,
            "weights": self.weights_,
            "bias": self.bias_,
            "n_features": self.n_features_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoftmaxRegression:
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
            epochs=int(payload["epochs"]),
        )
        model.classes_ = list(payload["classes"])
        model.weights_ = [[float(v) for v in row] for row in payload["weights"]]
        model.bias_ = [float(v) for v in payload["bias"]]
        model.n_features_ = int(payload["n_features"])
        return model
