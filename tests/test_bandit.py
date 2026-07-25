"""Unit tests for the reinforcement learning layer."""

from __future__ import annotations

import pytest
from ticketiq.rl.bandit import ArmStats, ContextualBandit, compute_reward, state_key

ARMS = ["a", "b", "c"]


# --------------------------------------------------------------------------
# Reward function
# --------------------------------------------------------------------------
def test_reward_matches_the_specified_formula() -> None:
    assert compute_reward(1, 2.5) == pytest.approx(7.5)
    assert compute_reward(0, 2.5) == pytest.approx(-2.5)
    assert compute_reward(1, 0.0) == pytest.approx(10.0)


def test_reward_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        compute_reward(2, 1.0)
    with pytest.raises(ValueError):
        compute_reward(1, -1.0)


def test_state_key_is_stable_and_ordered() -> None:
    assert state_key("billing", "high", "pro") == "billing|high|pro"


# --------------------------------------------------------------------------
# Update rule
# --------------------------------------------------------------------------
def test_incremental_mean_equals_the_arithmetic_mean() -> None:
    stats = ArmStats()
    rewards = [10.0, 0.0, 5.0, 7.5, -2.5]
    for r in rewards:
        stats.update(r)
    assert stats.pulls == len(rewards)
    assert stats.mean_reward == pytest.approx(sum(rewards) / len(rewards))
    assert stats.total_reward == pytest.approx(sum(rewards))


def test_first_update_sets_the_mean_exactly() -> None:
    bandit = ContextualBandit(ARMS, seed=0)
    stats = bandit.update("s", "a", 7.5)
    assert stats.pulls == 1
    assert stats.mean_reward == pytest.approx(7.5)


def test_update_is_isolated_per_state_and_arm() -> None:
    bandit = ContextualBandit(ARMS, seed=0)
    bandit.update("s1", "a", 10.0)
    bandit.update("s2", "a", 0.0)
    assert bandit.stats("s1", "a").mean_reward == pytest.approx(10.0)
    assert bandit.stats("s2", "a").mean_reward == pytest.approx(0.0)
    assert bandit.stats("s1", "b").pulls == 0


def test_update_rejects_unknown_arm() -> None:
    with pytest.raises(KeyError):
        ContextualBandit(ARMS).update("s", "nope", 1.0)


def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError):
        ContextualBandit([])
    with pytest.raises(ValueError):
        ContextualBandit(ARMS, epsilon=1.5)


# --------------------------------------------------------------------------
# Policy behaviour
# --------------------------------------------------------------------------
def test_cold_start_tries_every_arm_before_exploiting() -> None:
    bandit = ContextualBandit(ARMS, epsilon=0.0, seed=1)
    chosen = set()
    for _ in range(len(ARMS)):
        selection = bandit.select("s")
        assert selection.explored is True
        chosen.add(selection.arm_id)
        bandit.update("s", selection.arm_id, 1.0)
    assert chosen == set(ARMS)


def test_greedy_selects_the_highest_mean_when_epsilon_is_zero() -> None:
    bandit = ContextualBandit(ARMS, epsilon=0.0, seed=1)
    for arm, reward in (("a", 1.0), ("b", 9.0), ("c", 2.0)):
        bandit.update("s", arm, reward)
    selection = bandit.select("s")
    assert selection.arm_id == "b"
    assert selection.explored is False


def test_epsilon_one_always_explores() -> None:
    bandit = ContextualBandit(ARMS, epsilon=1.0, seed=1)
    for arm in ARMS:
        bandit.update("s", arm, 1.0)
    assert all(bandit.select("s").explored for _ in range(20))


def test_exploration_rate_is_approximately_epsilon() -> None:
    bandit = ContextualBandit(ARMS, epsilon=0.3, seed=7)
    for arm, reward in (("a", 1.0), ("b", 9.0), ("c", 2.0)):
        bandit.update("s", arm, reward)
    trials = 4000
    explored = sum(bandit.select("s").explored for _ in range(trials))
    assert 0.24 < explored / trials < 0.36


def test_bandit_converges_on_the_better_arm() -> None:
    """The whole point: with feedback, pulls concentrate on the best arm."""
    import random

    rng = random.Random(3)
    bandit = ContextualBandit(ARMS, epsilon=0.1, seed=3)
    # 'b' is genuinely better: higher satisfaction, same latency.
    true_p = {"a": 0.3, "b": 0.9, "c": 0.4}
    counts = {arm: 0 for arm in ARMS}
    for i in range(2000):
        selection = bandit.select("billing|high|pro")
        feedback = 1 if rng.random() < true_p[selection.arm_id] else 0
        bandit.update("billing|high|pro", selection.arm_id, compute_reward(feedback, 1.0))
        if i >= 1500:  # measure only after learning
            counts[selection.arm_id] += 1
    assert bandit.best_arm("billing|high|pro") == "b"
    assert counts["b"] / sum(counts.values()) > 0.8


def test_contextual_bandit_learns_different_arms_per_state() -> None:
    bandit = ContextualBandit(ARMS, epsilon=0.0, seed=5)
    for _ in range(5):
        bandit.update("low", "a", 9.0)
        bandit.update("low", "b", 1.0)
        bandit.update("low", "c", 1.0)
        bandit.update("high", "a", 1.0)
        bandit.update("high", "b", 9.0)
        bandit.update("high", "c", 1.0)
    assert bandit.best_arm("low") == "a"
    assert bandit.best_arm("high") == "b"


def test_ucb1_prefers_the_higher_mean_when_pulls_are_equal() -> None:
    bandit = ContextualBandit(ARMS, strategy="ucb1", seed=0)
    for _ in range(10):
        bandit.update("s", "a", 1.0)
        bandit.update("s", "b", 8.0)
        bandit.update("s", "c", 2.0)
    assert bandit.select("s").arm_id == "b"


def test_ucb1_revisits_an_under_sampled_arm() -> None:
    """Optimism: a rarely pulled arm keeps a wide confidence bonus."""
    bandit = ContextualBandit(ARMS, strategy="ucb1", seed=0)
    for _ in range(200):
        bandit.update("s", "a", 5.0)
        bandit.update("s", "b", 5.2)
    bandit.update("s", "c", 4.9)  # only one pull
    assert bandit.select("s").arm_id == "c"


# --------------------------------------------------------------------------
# Introspection and persistence
# --------------------------------------------------------------------------
def test_action_distribution_sums_to_one_after_pulls() -> None:
    bandit = ContextualBandit(ARMS, seed=0)
    bandit.update("s", "a", 1.0)
    bandit.update("s", "b", 1.0)
    bandit.update("t", "a", 1.0)
    distribution = bandit.action_distribution()
    assert distribution["a"] == pytest.approx(2 / 3, abs=1e-3)
    assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-3)


def test_action_distribution_is_zero_before_any_pull() -> None:
    assert set(ContextualBandit(ARMS).action_distribution().values()) == {0.0}


def test_snapshot_reports_totals() -> None:
    bandit = ContextualBandit(ARMS, seed=0)
    bandit.update("s", "a", 4.0)
    snapshot = bandit.snapshot()
    assert snapshot["total_pulls"] == 1
    assert snapshot["states"]["s"]["a"]["mean_reward"] == pytest.approx(4.0)


def test_save_and_load_preserves_the_learned_table(tmp_path) -> None:
    bandit = ContextualBandit(ARMS, epsilon=0.2, seed=0)
    for arm, reward in (("a", 3.0), ("b", 6.0), ("c", 1.0)):
        bandit.update("billing|high|pro", arm, reward)
    path = tmp_path / "bandit.json"
    bandit.save(path)

    restored = ContextualBandit.load(path)
    assert restored.epsilon == pytest.approx(0.2)
    assert restored.best_arm("billing|high|pro") == "b"
    assert restored.stats("billing|high|pro", "b").pulls == 1


def test_optimistic_initialisation_is_applied_to_new_states() -> None:
    bandit = ContextualBandit(ARMS, optimistic_init=5.0)
    assert bandit.stats("fresh", "a").mean_reward == pytest.approx(5.0)
    assert bandit.stats("fresh", "a").pulls == 0
