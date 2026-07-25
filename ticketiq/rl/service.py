"""Process-wide bandit instance plus persistence wiring."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from ticketiq.config import get_settings
from ticketiq.llm.configs import ARMS
from ticketiq.rl.bandit import ContextualBandit, Strategy

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_bandit: ContextualBandit | None = None


def bandit_path() -> Path:
    return get_settings().data_dir / "bandit.json"


def get_bandit() -> ContextualBandit:
    """Load the persisted bandit if present, otherwise start a fresh one."""
    global _bandit
    with _lock:
        if _bandit is not None:
            return _bandit
        settings = get_settings()
        strategy: Strategy = os.environ.get("BANDIT_STRATEGY", "epsilon_greedy")  # type: ignore[assignment]
        path = bandit_path()
        if path.is_file():
            try:
                _bandit = ContextualBandit.load(path)
                logger.info("loaded bandit state from %s", path)
                return _bandit
            except (OSError, ValueError, KeyError):
                logger.warning("could not load bandit state at %s; starting fresh", path)
        _bandit = ContextualBandit(
            arms=[arm.id for arm in ARMS],
            epsilon=settings.bandit_epsilon,
            strategy=strategy,
            optimistic_init=settings.bandit_optimistic_init,
        )
        return _bandit


def persist_bandit() -> None:
    """Best-effort save; a failure here must never break a request."""
    try:
        get_bandit().save(bandit_path())
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.warning("could not persist bandit state: %s", exc)


def reset_bandit() -> None:
    """Test hook."""
    global _bandit
    with _lock:
        _bandit = None
