"""Online contextual bandit for pipeline configuration selection.

Formulation
-----------
* **Context / state** ``s = (category, urgency_bucket, tier)`` — 4 x 3 x 3 = 36
  discrete states.
* **Action** ``a`` — one of the four arms in
  :data:`ticketiq.llm.configs.ARMS` (prompt variant x RAG top-K).
* **Reward** ``r = 10 * feedback - latency_seconds``.

Estimates are per (state, action) sample means maintained incrementally:

    N(s,a) <- N(s,a) + 1
    Q(s,a) <- Q(s,a) + (r - Q(s,a)) / N(s,a)

which is the standard stationary-average update and is exactly equivalent to
recomputing the mean, without storing history.

Why a tabular contextual bandit rather than Q-learning or a neural policy:
triage is a **one-step** decision — the configuration choice does not change
the next ticket's state — so there is no temporal credit assignment to do and
the discount factor of full RL would be modelling a dependency that does not
exist. With 36 states x 4 arms the table is 144 cells, small enough to learn
from realistic feedback volumes, fully inspectable, and cheap to persist.

Two exploration strategies are provided:

* ``epsilon_greedy`` (default) — explore uniformly with probability epsilon.
* ``ucb1`` — deterministic optimism, ``Q + c * sqrt(2 ln N_s / N(s,a))``, which
  needs no tuning of epsilon but is more aggressive early.

Unseen (state, action) pairs are pulled first in both strategies, so every arm
is tried at least once per state before exploitation begins.
"""

from __future__ import annotations

import json
import math
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Strategy = Literal["epsilon_greedy", "ucb1"]

FEEDBACK_WEIGHT = 10.0


def compute_reward(feedback: int, latency_s: float) -> float:
    """``r = 10 * feedback - latency`` as specified by the brief."""
    if feedback not in (0, 1):
        raise ValueError("feedback must be 0 or 1")
    if latency_s < 0:
        raise ValueError("latency must be non-negative")
    return round(FEEDBACK_WEIGHT * feedback - latency_s, 6)


def state_key(category: str, urgency_bucket: str, tier: str) -> str:
    return f"{category}|{urgency_bucket}|{tier}"


@dataclass
class ArmStats:
    pulls: int = 0
    mean_reward: float = 0.0
    total_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.total_reward += reward
        self.mean_reward += (reward - self.mean_reward) / self.pulls


@dataclass
class Selection:
    state: str
    arm_id: str
    explored: bool
    reason: str


class ContextualBandit:
    """Tabular contextual bandit over a fixed, small arm set."""

    def __init__(
        self,
        arms: Sequence[str],
        *,
        epsilon: float = 0.15,
        strategy: Strategy = "epsilon_greedy",
        optimistic_init: float = 0.0,
        ucb_c: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not arms:
            raise ValueError("bandit needs at least one arm")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.arms = list(arms)
        self.epsilon = epsilon
        self.strategy: Strategy = strategy
        self.optimistic_init = optimistic_init
        self.ucb_c = ucb_c
        self._rng = random.Random(seed)
        self._table: dict[str, dict[str, ArmStats]] = {}
        self._lock = threading.RLock()

    # -- table access ---------------------------------------------------
    def _row(self, state: str) -> dict[str, ArmStats]:
        row = self._table.get(state)
        if row is None:
            row = {arm: ArmStats(mean_reward=self.optimistic_init) for arm in self.arms}
            self._table[state] = row
        else:
            for arm in self.arms:
                row.setdefault(arm, ArmStats(mean_reward=self.optimistic_init))
        return row

    def stats(self, state: str, arm_id: str) -> ArmStats:
        with self._lock:
            return self._row(state)[arm_id]

    def state_pulls(self, state: str) -> int:
        with self._lock:
            return sum(s.pulls for s in self._row(state).values())

    # -- policy ---------------------------------------------------------
    def select(self, state: str) -> Selection:
        """Choose an arm for ``state`` according to the configured strategy."""
        with self._lock:
            row = self._row(state)
            untried = [arm for arm in self.arms if row[arm].pulls == 0]
            if untried:
                arm = untried[0] if self.strategy == "ucb1" else self._rng.choice(untried)
                return Selection(state, arm, True, "cold start: arm never pulled in this state")

            if self.strategy == "ucb1":
                total = sum(s.pulls for s in row.values())
                scores = {
                    arm: stats.mean_reward
                    + self.ucb_c * math.sqrt(2.0 * math.log(total) / stats.pulls)
                    for arm, stats in row.items()
                }
                arm = max(scores, key=lambda a: (scores[a], a))
                return Selection(state, arm, False, "ucb1 upper confidence bound")

            if self._rng.random() < self.epsilon:
                arm = self._rng.choice(self.arms)
                return Selection(
                    state, arm, True, f"epsilon-greedy exploration (eps={self.epsilon})"
                )

            arm = max(row, key=lambda a: (row[a].mean_reward, a))
            return Selection(state, arm, False, "greedy: highest mean reward")

    def update(self, state: str, arm_id: str, reward: float) -> ArmStats:
        """Incremental sample-mean update for one (state, action) pair."""
        if arm_id not in self.arms:
            raise KeyError(f"unknown arm {arm_id!r}")
        with self._lock:
            stats = self._row(state)[arm_id]
            stats.update(reward)
            return stats

    def best_arm(self, state: str) -> str:
        with self._lock:
            row = self._row(state)
            return max(row, key=lambda a: (row[a].mean_reward, a))

    # -- introspection / persistence ------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "strategy": self.strategy,
                "epsilon": self.epsilon,
                "arms": self.arms,
                "total_pulls": sum(s.pulls for row in self._table.values() for s in row.values()),
                "states": {
                    state: {
                        arm: {
                            "pulls": s.pulls,
                            "mean_reward": round(s.mean_reward, 4),
                            "total_reward": round(s.total_reward, 4),
                        }
                        for arm, s in row.items()
                    }
                    for state, row in sorted(self._table.items())
                },
            }

    def action_distribution(self) -> dict[str, float]:
        """Share of all pulls each arm has received, across every state."""
        with self._lock:
            counts = {arm: 0 for arm in self.arms}
            for row in self._table.values():
                for arm, s in row.items():
                    counts[arm] = counts.get(arm, 0) + s.pulls
            total = sum(counts.values())
            if not total:
                return {arm: 0.0 for arm in self.arms}
            return {arm: round(c / total, 4) for arm, c in counts.items()}

    def save(self, path: Path) -> None:
        with self._lock:
            payload = {
                "epsilon": self.epsilon,
                "strategy": self.strategy,
                "optimistic_init": self.optimistic_init,
                "ucb_c": self.ucb_c,
                "arms": self.arms,
                "table": {
                    state: {arm: [s.pulls, s.mean_reward, s.total_reward] for arm, s in row.items()}
                    for state, row in self._table.items()
                },
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX; avoids a torn file under concurrency

    @classmethod
    def load(cls, path: Path, *, seed: int | None = None) -> ContextualBandit:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bandit = cls(
            arms=payload["arms"],
            epsilon=float(payload.get("epsilon", 0.15)),
            strategy=payload.get("strategy", "epsilon_greedy"),
            optimistic_init=float(payload.get("optimistic_init", 0.0)),
            ucb_c=float(payload.get("ucb_c", 1.0)),
            seed=seed,
        )
        for state, row in payload.get("table", {}).items():
            target = bandit._row(state)
            for arm, (pulls, mean, total) in row.items():
                if arm in target:
                    target[arm] = ArmStats(
                        pulls=int(pulls), mean_reward=float(mean), total_reward=float(total)
                    )
        return bandit


@dataclass
class PendingDecision:
    """A served ticket awaiting feedback."""

    transaction_id: str
    state: str
    arm_id: str
    latency_s: float
    created_at: float = field(default=0.0)
