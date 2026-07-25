"""A small dependency-aware workflow engine.

Design goals, in priority order:

1. **Explicit dependencies.** A stage declares what it needs by name; the
   engine derives the execution order. Nothing is implied by call order.
2. **Resumability.** Every stage's output is persisted as it completes. Re-running
   a transaction replays completed stages from the store instead of recomputing
   them, so a failure in stage 5 costs one stage of work, not five.
3. **Concurrency where the graph allows it.** Stages are grouped into
   topological *levels*; everything in a level is independent by construction
   and runs under a single ``asyncio.gather``. Classification and aspect
   sentiment sit in the same level and genuinely overlap.

Not used: Airflow/Prefect/Celery. At this size they would add an operational
dependency and hide the very logic being assessed. The trade-off is that this
engine is single-process and has no scheduler or retry backoff — noted in the
README as deliberate scope.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ticketiq.workflow.state import StateStore

logger = logging.getLogger(__name__)

StageOutput = dict[str, Any]
StageFn = Callable[["StageContext"], StageOutput | Awaitable[StageOutput]]


class WorkflowError(RuntimeError):
    """Raised when the graph is invalid or a stage fails terminally."""


@dataclass
class StageContext:
    """What a stage sees: the original request plus all upstream outputs."""

    transaction_id: str
    request: dict[str, Any]
    outputs: dict[str, StageOutput] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)

    def output_of(self, stage: str) -> StageOutput:
        try:
            return self.outputs[stage]
        except KeyError as exc:
            raise WorkflowError(
                f"stage {stage!r} output requested before it completed"
            ) from exc


@dataclass(frozen=True)
class Stage:
    name: str
    fn: StageFn
    depends_on: tuple[str, ...] = ()
    description: str = ""


class Workflow:
    """An immutable DAG of named stages."""

    def __init__(self, name: str, stages: Sequence[Stage]) -> None:
        self.name = name
        self.stages: dict[str, Stage] = {}
        for stage in stages:
            if stage.name in self.stages:
                raise WorkflowError(f"duplicate stage name {stage.name!r}")
            self.stages[stage.name] = stage
        self._validate()
        self.levels: list[list[str]] = self._topological_levels()

    # -- graph ----------------------------------------------------------
    def _validate(self) -> None:
        for stage in self.stages.values():
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise WorkflowError(
                        f"stage {stage.name!r} depends on unknown stage {dep!r}"
                    )
                if dep == stage.name:
                    raise WorkflowError(f"stage {stage.name!r} depends on itself")

    def _topological_levels(self) -> list[list[str]]:
        """Kahn's algorithm, grouped by level so each level can run in parallel.

        Within a level, stages keep their *declaration* order instead of being
        sorted alphabetically. The order inside a level is a free choice (its
        stages are independent by construction), and declaration order is what
        the DAG diagram, ``/status`` and ``/pipeline/graph`` read back, e.g.
        ``classify`` before ``aspects``. Sorting alphabetically would silently
        reorder that to ``aspects, classify``.
        """
        declared = {name: i for i, name in enumerate(self.stages)}
        remaining = {name: set(stage.depends_on) for name, stage in self.stages.items()}
        levels: list[list[str]] = []
        resolved: set[str] = set()

        while remaining:
            ready = sorted(
                (name for name, deps in remaining.items() if deps <= resolved),
                key=lambda name: declared[name],
            )
            if not ready:
                raise WorkflowError(f"cycle detected among stages: {sorted(remaining)}")
            levels.append(ready)
            resolved.update(ready)
            for name in ready:
                del remaining[name]
        return levels

    def order(self) -> list[str]:
        return [name for level in self.levels for name in level]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "level": i,
                "stages": [
                    {
                        "name": name,
                        "depends_on": list(self.stages[name].depends_on),
                        "description": self.stages[name].description,
                    }
                    for name in level
                ],
                "parallel": len(level) > 1,
            }
            for i, level in enumerate(self.levels)
        ]

    # -- execution ------------------------------------------------------
    async def run(
        self,
        transaction_id: str,
        request: Mapping[str, Any],
        store: StateStore,
        *,
        resume: bool = True,
        only: Sequence[str] | None = None,
    ) -> StageContext:
        """Execute the DAG, skipping stages already completed for this id.

        ``only`` restricts execution to the named stages and their already
        completed upstream outputs, which is how a single failed stage is
        re-run from the API without repeating the whole pipeline.
        """
        ctx = StageContext(transaction_id=transaction_id, request=dict(request))
        if store.get_run(transaction_id) is None:
            store.create_run(transaction_id, dict(request))
        else:
            store.set_run_status(transaction_id, "running")

        try:
            for level in self.levels:
                pending: list[str] = []
                for name in level:
                    record = store.get_stage(transaction_id, name) if resume else None
                    replayable = (
                        record is not None
                        and record.status == "completed"
                        and (only is None or name not in only)
                    )
                    if replayable and record is not None:
                        ctx.outputs[name] = record.output or {}
                        logger.debug("replaying completed stage %s/%s", transaction_id, name)
                        continue
                    if only is not None and name not in only:
                        # Not requested and not completed: nothing to replay.
                        continue
                    pending.append(name)

                if not pending:
                    continue
                if len(pending) == 1:
                    await self._run_stage(pending[0], ctx, store)
                else:
                    await asyncio.gather(
                        *(self._run_stage(name, ctx, store) for name in pending)
                    )
        except Exception as exc:
            store.set_run_status(transaction_id, "failed", error=str(exc))
            raise

        completed = {
            record.stage
            for record in store.get_stages(transaction_id)
            if record.status == "completed"
        }
        status = "completed" if completed >= set(self.stages) else "running"
        store.set_run_status(transaction_id, status)
        return ctx

    async def _run_stage(self, name: str, ctx: StageContext, store: StateStore) -> None:
        stage = self.stages[name]
        store.start_stage(ctx.transaction_id, name)
        started = time.perf_counter()
        try:
            result = stage.fn(ctx)
            if inspect.isawaitable(result):
                result = await result
            output: StageOutput = dict(result or {})
        except Exception as exc:
            logger.exception("stage %s failed for %s", name, ctx.transaction_id)
            store.fail_stage(ctx.transaction_id, name, f"{type(exc).__name__}: {exc}")
            raise WorkflowError(f"stage {name!r} failed: {exc}") from exc

        output.setdefault("_duration_s", round(time.perf_counter() - started, 4))
        ctx.outputs[name] = output
        store.complete_stage(ctx.transaction_id, name, output)
