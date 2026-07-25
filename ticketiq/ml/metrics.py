"""Classification metrics (accuracy / precision / recall / F1, macro & per class)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field


@dataclass
class ClassMetrics:
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"samples={self.n_samples}  accuracy={self.accuracy:.3f}  "
            f"macro_f1={self.macro_f1:.3f}",
            f"{'class':<16}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>6}",
        ]
        for name, m in sorted(self.per_class.items()):
            lines.append(
                f"{name:<16}{m.precision:>8.3f}{m.recall:>8.3f}{m.f1:>8.3f}{m.support:>6}"
            )
        return "\n".join(lines)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> EvaluationReport:
    """Compute a full classification report without scikit-learn."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    classes = list(labels) if labels else sorted(set(y_true) | set(y_pred))

    matrix = {t: dict.fromkeys(classes, 0) for t in classes}
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix.setdefault(truth, dict.fromkeys(classes, 0))
        matrix[truth][pred] = matrix[truth].get(pred, 0) + 1

    per_class: dict[str, ClassMetrics] = {}
    for c in classes:
        tp = matrix[c].get(c, 0)
        fp = sum(matrix[o].get(c, 0) for o in classes if o != c)
        fn = sum(v for k, v in matrix[c].items() if k != c)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[c] = ClassMetrics(support=tp + fn, precision=precision, recall=recall, f1=f1)

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    n = len(y_true)
    k = max(1, len(classes))
    return EvaluationReport(
        accuracy=_safe_div(correct, n),
        macro_precision=sum(m.precision for m in per_class.values()) / k,
        macro_recall=sum(m.recall for m in per_class.values()) / k,
        macro_f1=sum(m.f1 for m in per_class.values()) / k,
        per_class=per_class,
        confusion_matrix=matrix,
        n_samples=n,
    )
