"""SQLite-backed persistence for pipeline state.

One row per (transaction, stage) is what makes ``GET /ticket/{id}/status``
report real state rather than a guess, and what lets a failed stage be re-run
without repeating completed upstream work.

SQLite is the right size of tool here: it is in the standard library, gives
atomic writes and crash safety, and survives a process restart — which a
dictionary would not. WAL mode is enabled so a status read never blocks a
pipeline write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    transaction_id TEXT PRIMARY KEY,
    status         TEXT NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    request        TEXT NOT NULL,
    error          TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    transaction_id TEXT NOT NULL,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    started_at     REAL,
    finished_at    REAL,
    error          TEXT,
    output         TEXT,
    PRIMARY KEY (transaction_id, stage)
);
CREATE TABLE IF NOT EXISTS decisions (
    transaction_id TEXT PRIMARY KEY,
    state          TEXT NOT NULL,
    arm_id         TEXT NOT NULL,
    latency_s      REAL NOT NULL,
    feedback       INTEGER,
    reward         REAL,
    created_at     REAL NOT NULL,
    updated_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_stages_txn ON stages (transaction_id);
"""


@dataclass
class StageRecord:
    stage: str
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output: dict[str, Any] | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 4)


@dataclass
class DecisionRecord:
    transaction_id: str
    state: str
    arm_id: str
    latency_s: float
    feedback: int | None = None
    reward: float | None = None


class StateStore:
    """Thread-safe SQLite wrapper. One instance per process is enough."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- runs -----------------------------------------------------------
    def create_run(self, transaction_id: str, request: dict[str, Any]) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(transaction_id, status, created_at, updated_at, request, error) "
                "VALUES (?, 'running', ?, ?, ?, NULL)",
                (transaction_id, now, now, json.dumps(request)),
            )

    def set_run_status(
        self, transaction_id: str, status: str, error: str | None = None
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ?, error = ? WHERE transaction_id = ?",
                (status, time.time(), error, transaction_id),
            )

    def get_run(self, transaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE transaction_id = ?", (transaction_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "transaction_id": row["transaction_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request": json.loads(row["request"]),
            "error": row["error"],
        }

    # -- stages ---------------------------------------------------------
    def start_stage(self, transaction_id: str, stage: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO stages (transaction_id, stage, status, started_at) "
                "VALUES (?, ?, 'running', ?) "
                "ON CONFLICT(transaction_id, stage) DO UPDATE SET "
                "status = 'running', started_at = excluded.started_at, "
                "finished_at = NULL, error = NULL",
                (transaction_id, stage, time.time()),
            )

    def complete_stage(
        self, transaction_id: str, stage: str, output: dict[str, Any]
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE stages SET status = 'completed', finished_at = ?, output = ?, "
                "error = NULL WHERE transaction_id = ? AND stage = ?",
                (time.time(), json.dumps(output, default=str), transaction_id, stage),
            )

    def fail_stage(self, transaction_id: str, stage: str, error: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE stages SET status = 'failed', finished_at = ?, error = ? "
                "WHERE transaction_id = ? AND stage = ?",
                (time.time(), error[:2000], transaction_id, stage),
            )

    def get_stage(self, transaction_id: str, stage: str) -> StageRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM stages WHERE transaction_id = ? AND stage = ?",
                (transaction_id, stage),
            ).fetchone()
        return self._to_stage(row) if row else None

    def get_stages(self, transaction_id: str) -> list[StageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM stages WHERE transaction_id = ? ORDER BY started_at",
                (transaction_id,),
            ).fetchall()
        return [self._to_stage(row) for row in rows]

    @staticmethod
    def _to_stage(row: sqlite3.Row) -> StageRecord:
        return StageRecord(
            stage=row["stage"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=row["error"],
            output=json.loads(row["output"]) if row["output"] else None,
        )

    # -- bandit decisions ----------------------------------------------
    def record_decision(
        self, transaction_id: str, state: str, arm_id: str, latency_s: float
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions "
                "(transaction_id, state, arm_id, latency_s, feedback, reward, created_at) "
                "VALUES (?, ?, ?, ?, NULL, NULL, ?)",
                (transaction_id, state, arm_id, latency_s, time.time()),
            )

    def get_decision(self, transaction_id: str) -> DecisionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM decisions WHERE transaction_id = ?", (transaction_id,)
            ).fetchone()
        if row is None:
            return None
        return DecisionRecord(
            transaction_id=row["transaction_id"],
            state=row["state"],
            arm_id=row["arm_id"],
            latency_s=row["latency_s"],
            feedback=row["feedback"],
            reward=row["reward"],
        )

    def record_feedback(self, transaction_id: str, feedback: int, reward: float) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE decisions SET feedback = ?, reward = ?, updated_at = ? "
                "WHERE transaction_id = ?",
                (feedback, reward, time.time(), transaction_id),
            )


_store: StateStore | None = None
_store_lock = threading.Lock()


def get_state_store() -> StateStore:
    global _store
    with _store_lock:
        if _store is None:
            from ticketiq.config import get_settings

            _store = StateStore(get_settings().state_db_path)
        return _store


def reset_state_store() -> None:
    """Test hook."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
