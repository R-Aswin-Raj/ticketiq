"""A hand-rolled TF-IDF vectoriser.

Requirement 2 forbids using scikit-learn's estimators for the classifier, so
the feature pipeline is implemented from first principles too. The maths:

    tf(t, d)  = count(t, d) / max(1, len(d))
    idf(t)    = ln((1 + N) / (1 + df(t))) + 1        # smoothed, always > 0
    x(t, d)   = tf(t, d) * idf(t),  then L2-normalised per document

Documents are represented as sparse dictionaries ``{feature_index: weight}``
because the vocabulary is large relative to the ~200 document corpus.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ticketiq.ml.text import featurize

SparseVector = dict[int, float]


@dataclass
class TfidfVectorizer:
    """Fit a vocabulary and transform documents into L2-normalised TF-IDF."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_features: int | None = 20_000
    vocabulary_: dict[str, int] = field(default_factory=dict)
    idf_: list[float] = field(default_factory=list)
    n_docs_: int = 0

    # -- fitting --------------------------------------------------------
    def fit(self, documents: Sequence[str]) -> TfidfVectorizer:
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            doc_freq.update(set(featurize(doc, ngram_range=self.ngram_range)))

        kept = [(term, df) for term, df in doc_freq.items() if df >= self.min_df]
        # Deterministic ordering: most frequent first, ties broken alphabetically.
        kept.sort(key=lambda item: (-item[1], item[0]))
        if self.max_features is not None:
            kept = kept[: self.max_features]

        n = len(documents)
        self.n_docs_ = n
        self.vocabulary_ = {term: i for i, (term, _) in enumerate(kept)}
        self.idf_ = [math.log((1 + n) / (1 + df)) + 1.0 for _, df in kept]
        return self

    # -- transforming ---------------------------------------------------
    def transform_one(self, document: str) -> SparseVector:
        features = featurize(document, ngram_range=self.ngram_range)
        if not features:
            return {}
        counts = Counter(f for f in features if f in self.vocabulary_)
        if not counts:
            return {}
        length = max(1, len(features))
        vec: SparseVector = {}
        for term, count in counts.items():
            idx = self.vocabulary_[term]
            vec[idx] = (count / length) * self.idf_[idx]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    def transform(self, documents: Sequence[str]) -> list[SparseVector]:
        return [self.transform_one(doc) for doc in documents]

    def fit_transform(self, documents: Sequence[str]) -> list[SparseVector]:
        return self.fit(documents).transform(documents)

    @property
    def n_features(self) -> int:
        return len(self.vocabulary_)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "vocabulary": self.vocabulary_,
            "idf": self.idf_,
            "n_docs": self.n_docs_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TfidfVectorizer:
        lo, hi = payload["ngram_range"]
        vec = cls(
            ngram_range=(int(lo), int(hi)),
            min_df=int(payload["min_df"]),
            max_features=payload["max_features"],
        )
        vec.vocabulary_ = {k: int(v) for k, v in payload["vocabulary"].items()}
        vec.idf_ = [float(v) for v in payload["idf"]]
        vec.n_docs_ = int(payload["n_docs"])
        return vec
