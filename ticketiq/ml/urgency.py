"""Urgency scoring.

A deliberately transparent linear blend, because the RL state space needs a
stable, interpretable bucketing — a learned urgency model would drift under
the bandit and make the reward signal hard to attribute.

    urgency = w_tier * tier_weight
            + w_cat  * category_weight
            + w_sent * negativity
            + w_kw   * escalation_keyword_hits

clipped into [0, 1]. Weights sum to 1 so the score is directly interpretable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ticketiq.ml.aspects import Aspect

TIER_WEIGHT: dict[str, float] = {"free": 0.15, "pro": 0.55, "enterprise": 1.0}
CATEGORY_WEIGHT: dict[str, float] = {
    "technical": 0.85,
    "billing": 0.70,
    "account": 0.60,
    "feature_request": 0.20,
}

ESCALATION_PATTERNS: tuple[str, ...] = (
    r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\bcritical\b", r"\bp1\b",
    r"\bproduction (is )?down\b", r"\boutage\b", r"\bblocked\b", r"\bcancel\b",
    r"\bchurn\b", r"\blegal\b", r"\blawyer\b", r"\bescalat", r"\brefund\b",
    r"\bentire team\b", r"\ball users\b", r"\bdata loss\b", r"\bsecurity\b",
)

W_TIER, W_CATEGORY, W_SENTIMENT, W_KEYWORDS = 0.30, 0.25, 0.30, 0.15


def keyword_pressure(text: str) -> float:
    lc = text.lower()
    hits = sum(1 for pattern in ESCALATION_PATTERNS if re.search(pattern, lc))
    return min(1.0, hits / 3.0)


def negativity(aspects: Sequence[Aspect]) -> float:
    """Worst-aspect negativity, softened by the mean, mapped to [0, 1]."""
    if not aspects:
        return 0.5
    scores = [a.sentiment for a in aspects]
    worst = min(scores)
    mean = sum(scores) / len(scores)
    blended = 0.7 * worst + 0.3 * mean
    return max(0.0, min(1.0, (1.0 - blended) / 2.0))


def urgency_score(
    category: str,
    aspects: Sequence[Aspect],
    tier: str,
    text: str = "",
) -> float:
    score = (
        W_TIER * TIER_WEIGHT.get(tier, 0.15)
        + W_CATEGORY * CATEGORY_WEIGHT.get(category, 0.5)
        + W_SENTIMENT * negativity(aspects)
        + W_KEYWORDS * keyword_pressure(text)
    )
    return round(max(0.0, min(1.0, score)), 4)


def urgency_bucket(score: float) -> str:
    """Discretise for the tabular bandit state."""
    if score < 0.40:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"
