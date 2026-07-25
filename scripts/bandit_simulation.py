"""Simulate the online learning loop and show that the bandit is learning.

The environment has a **context-dependent** optimum, which is the point: a
non-contextual bandit would converge on one globally decent arm, whereas the
contextual one should learn a different best arm per state.

Ground truth (hidden from the bandit):

* ``concise`` replies satisfy most customers and are fast.
* ``empathetic`` replies are slower, and only pay off when urgency is high —
  an angry enterprise customer wants acknowledgement, a low-urgency question
  does not.
* ``top_k=5`` adds a little accuracy on policy-heavy categories and always
  costs latency.

So the optimal policy is roughly *concise-k2 when urgency is low, empathetic-k5
when urgency is high*, and the reward ``10 * feedback - latency`` is what has
to discover that.

Outputs: action distribution in the first vs last quarter of the run, mean
reward per quarter, cumulative regret against the oracle, and the per-state
learned argmax compared with the true argmax.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Run as `PYTHONPATH=. python scripts/bandit_simulation.py` (or after `poetry
# install`), so the `ticketiq` package at the repo root is importable.
from ticketiq.llm.configs import ARMS, get_arm
from ticketiq.ml.urgency import urgency_bucket
from ticketiq.rl.bandit import ContextualBandit, compute_reward, state_key

ROOT = Path(__file__).resolve().parents[1]  # for the default --plot output path

CATEGORIES = ("billing", "technical", "account", "feature_request")
TIERS = ("free", "pro", "enterprise")
BUCKETS = ("low", "medium", "high")
POLICY_HEAVY = frozenset({"billing", "technical"})


@dataclass(frozen=True)
class Environment:
    """Hidden reward model the bandit has to discover."""

    noise_s: float = 0.12

    def satisfaction_prob(self, category: str, bucket: str, tier: str, arm_id: str) -> float:
        arm = get_arm(arm_id)
        p = 0.70 if arm.variant_id == "concise" else 0.66
        if bucket == "high":
            p += 0.20 if arm.variant_id == "empathetic" else -0.05
        elif bucket == "medium":
            p += 0.08 if arm.variant_id == "empathetic" else 0.0
        if arm.top_k == 5:
            p += 0.07 if category in POLICY_HEAVY else 0.02
        if tier == "enterprise" and arm.variant_id == "empathetic":
            p += 0.03
        return max(0.02, min(0.98, p))

    def latency(self, arm_id: str, rng: random.Random) -> float:
        arm = get_arm(arm_id)
        base = 0.50 if arm.variant_id == "concise" else 1.10
        return max(0.05, base + 0.10 * arm.top_k + rng.gauss(0.0, self.noise_s))

    def expected_reward(self, category: str, bucket: str, tier: str, arm_id: str) -> float:
        arm = get_arm(arm_id)
        base = 0.50 if arm.variant_id == "concise" else 1.10
        mean_latency = base + 0.10 * arm.top_k
        return 10.0 * self.satisfaction_prob(category, bucket, tier, arm_id) - mean_latency

    def best_arm(self, category: str, bucket: str, tier: str) -> tuple[str, float]:
        scores = {arm.id: self.expected_reward(category, bucket, tier, arm.id) for arm in ARMS}
        best = max(scores, key=lambda a: scores[a])
        return best, scores[best]


def run_simulation(
    rounds: int = 4000,
    epsilon: float = 0.15,
    strategy: str = "epsilon_greedy",
    seed: int = 42,
) -> dict[str, object]:
    rng = random.Random(seed)
    env = Environment()
    bandit = ContextualBandit(
        arms=[arm.id for arm in ARMS], epsilon=epsilon, strategy=strategy, seed=seed
    )

    quarter = max(1, rounds // 4)
    quarter_pulls: list[dict[str, int]] = [{arm.id: 0 for arm in ARMS} for _ in range(4)]
    quarter_reward: list[list[float]] = [[] for _ in range(4)]
    cumulative_regret = 0.0
    regret_curve: list[float] = []

    for t in range(rounds):
        category = rng.choice(CATEGORIES)
        tier = rng.choice(TIERS)
        # Urgency is correlated with tier and category, as in the real pipeline.
        raw = rng.betavariate(2, 3) + (0.15 if tier == "enterprise" else 0.0)
        raw += 0.10 if category in ("technical", "billing") else -0.10
        bucket = urgency_bucket(max(0.0, min(1.0, raw)))

        state = state_key(category, bucket, tier)
        selection = bandit.select(state)
        arm_id = selection.arm_id

        latency = env.latency(arm_id, rng)
        p = env.satisfaction_prob(category, bucket, tier, arm_id)
        feedback = 1 if rng.random() < p else 0
        reward = compute_reward(feedback, latency)
        bandit.update(state, arm_id, reward)

        _, best_expected = env.best_arm(category, bucket, tier)
        cumulative_regret += best_expected - env.expected_reward(category, bucket, tier, arm_id)
        regret_curve.append(round(cumulative_regret, 3))

        q = min(3, t // quarter)
        quarter_pulls[q][arm_id] += 1
        quarter_reward[q].append(reward)

    # How often did the bandit's learned argmax match the oracle's?
    # Variant choice (concise vs empathetic) is reported separately: the gaps
    # between variants are large, while k=2 vs k=5 within a variant is often
    # within noise at realistic sample sizes. Conflating the two would hide
    # what the bandit has genuinely learned.
    matches, variant_matches, total = 0, 0, 0
    per_state: list[dict[str, object]] = []
    for state in sorted(bandit.snapshot()["states"]):  # type: ignore[index]
        category, bucket, tier = state.split("|")
        true_best, _ = env.best_arm(category, bucket, tier)
        learned = bandit.best_arm(state)
        pulls = bandit.state_pulls(state)
        total += 1
        matches += int(learned == true_best)
        variant_matches += int(get_arm(learned).variant_id == get_arm(true_best).variant_id)
        per_state.append(
            {
                "state": state,
                "pulls": pulls,
                "learned": learned,
                "oracle": true_best,
                "match": learned == true_best,
            }
        )

    # Reference points: what a uniform-random router and an oracle would score.
    weights: dict[str, int] = {}
    for row in per_state:
        weights[str(row["state"])] = int(row["pulls"])
    total_pulls = sum(weights.values()) or 1
    random_baseline = 0.0
    oracle_baseline = 0.0
    for state, w in weights.items():
        category, bucket, tier = state.split("|")
        arm_scores = [env.expected_reward(category, bucket, tier, a.id) for a in ARMS]
        random_baseline += (w / total_pulls) * (sum(arm_scores) / len(arm_scores))
        oracle_baseline += (w / total_pulls) * max(arm_scores)

    def share(counts: dict[str, int]) -> dict[str, float]:
        n = sum(counts.values()) or 1
        return {arm: round(c / n, 4) for arm, c in counts.items()}

    return {
        "rounds": rounds,
        "strategy": strategy,
        "epsilon": epsilon,
        "quarters": [
            {
                "quarter": i + 1,
                "action_distribution": share(quarter_pulls[i]),
                "mean_reward": round(sum(quarter_reward[i]) / max(1, len(quarter_reward[i])), 4),
            }
            for i in range(4)
        ],
        "final_action_distribution": bandit.action_distribution(),
        "cumulative_regret": round(cumulative_regret, 2),
        "mean_regret_per_round": round(cumulative_regret / rounds, 5),
        "regret_curve": regret_curve,
        "state_argmax_accuracy": round(matches / max(1, total), 4),
        "state_variant_accuracy": round(variant_matches / max(1, total), 4),
        "random_policy_expected_reward": round(random_baseline, 4),
        "oracle_expected_reward": round(oracle_baseline, 4),
        "per_state": per_state,
    }


def plot(result: dict[str, object], out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    quarters = result["quarters"]  # type: ignore[index]
    arms = [arm.id for arm in ARMS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    width = 0.2
    for i, arm in enumerate(arms):
        ax1.bar(
            [q + i * width for q in range(4)],
            [quarters[q]["action_distribution"][arm] for q in range(4)],  # type: ignore[index]
            width=width,
            label=arm,
        )
    ax1.set_xticks([q + 1.5 * width for q in range(4)])
    ax1.set_xticklabels([f"Q{q + 1}" for q in range(4)])
    ax1.set_ylabel("share of pulls")
    ax1.set_title("Action distribution shifts toward better arms")
    ax1.legend(fontsize=8)

    ax2.plot(result["regret_curve"])  # type: ignore[index]
    ax2.set_xlabel("tickets")
    ax2.set_ylabel("cumulative regret")
    ax2.set_title("Cumulative regret (flattening = converged)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--strategy", choices=("epsilon_greedy", "ucb1"), default="epsilon_greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="write data/bandit_learning.png")
    parser.add_argument("--json", type=Path, default=None, help="dump the full result")
    args = parser.parse_args()

    result = run_simulation(args.rounds, args.epsilon, args.strategy, args.seed)

    print(f"strategy={result['strategy']} rounds={result['rounds']} eps={result['epsilon']}\n")
    header = f"{'quarter':<9}" + "".join(f"{arm.id:>15}" for arm in ARMS) + f"{'mean reward':>14}"
    print(header)
    for q in result["quarters"]:  # type: ignore[index]
        row = f"Q{q['quarter']:<8}" + "".join(
            f"{q['action_distribution'][arm.id]:>15.3f}" for arm in ARMS
        )
        print(row + f"{q['mean_reward']:>14.3f}")

    print(f"\ncumulative regret      : {result['cumulative_regret']}")
    print(f"mean regret per ticket : {result['mean_regret_per_round']}")
    print(f"per-state argmax match : {result['state_argmax_accuracy']:.1%} of states")
    print(f"per-state variant match: {result['state_variant_accuracy']:.1%} of states")
    print(
        f"\nmean reward  random={result['random_policy_expected_reward']:.3f}  "
        f"bandit(Q4)={result['quarters'][3]['mean_reward']:.3f}  "
        f"oracle={result['oracle_expected_reward']:.3f}"
    )

    print("\nsample of learned vs oracle best arm:")
    for row in result["per_state"][:8]:  # type: ignore[index]
        flag = "ok " if row["match"] else "MISS"
        print(f"  {flag} {row['state']:<32} learned={row['learned']:<14} oracle={row['oracle']}")

    if args.plot and plot(result, ROOT / "data" / "bandit_learning.png"):
        print("\nwrote data/bandit_learning.png")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
