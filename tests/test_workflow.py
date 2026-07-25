"""Unit tests for the workflow engine."""

from __future__ import annotations

import asyncio
import time

import pytest
from ticketiq.workflow.engine import (
    Stage,
    StageContext,
    Workflow,
    WorkflowError,
)


def noop(_: StageContext) -> dict[str, str]:
    return {"ok": "yes"}


# --------------------------------------------------------------------------
# Dependency resolution
# --------------------------------------------------------------------------
def test_levels_group_independent_stages_together() -> None:
    wf = Workflow(
        "w",
        [
            Stage("a", noop),
            Stage("b", noop),
            Stage("c", noop, ("a", "b")),
            Stage("d", noop, ("c",)),
        ],
    )
    assert wf.levels == [["a", "b"], ["c"], ["d"]]
    assert wf.order() == ["a", "b", "c", "d"]


def test_a_linear_chain_produces_one_stage_per_level() -> None:
    wf = Workflow("w", [Stage("a", noop), Stage("b", noop, ("a",)), Stage("c", noop, ("b",))])
    assert wf.levels == [["a"], ["b"], ["c"]]


def test_diamond_dependencies_resolve_correctly() -> None:
    wf = Workflow(
        "w",
        [
            Stage("root", noop),
            Stage("left", noop, ("root",)),
            Stage("right", noop, ("root",)),
            Stage("join", noop, ("left", "right")),
        ],
    )
    assert wf.levels == [["root"], ["left", "right"], ["join"]]


def test_declaration_order_does_not_affect_execution_order() -> None:
    wf = Workflow("w", [Stage("last", noop, ("first",)), Stage("first", noop)])
    assert wf.order() == ["first", "last"]


def test_cycle_is_rejected() -> None:
    with pytest.raises(WorkflowError, match="cycle"):
        Workflow("w", [Stage("a", noop, ("b",)), Stage("b", noop, ("a",))])


def test_unknown_dependency_is_rejected() -> None:
    with pytest.raises(WorkflowError, match="unknown stage"):
        Workflow("w", [Stage("a", noop, ("ghost",))])


def test_self_dependency_is_rejected() -> None:
    with pytest.raises(WorkflowError, match="itself"):
        Workflow("w", [Stage("a", noop, ("a",))])


def test_duplicate_stage_names_are_rejected() -> None:
    with pytest.raises(WorkflowError, match="duplicate"):
        Workflow("w", [Stage("a", noop), Stage("a", noop)])


def test_describe_flags_parallel_levels() -> None:
    wf = Workflow("w", [Stage("a", noop), Stage("b", noop), Stage("c", noop, ("a", "b"))])
    described = wf.describe()
    assert described[0]["parallel"] is True
    assert described[1]["parallel"] is False


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def test_stage_outputs_are_available_downstream(state_store) -> None:
    def produce(_: StageContext) -> dict[str, int]:
        return {"value": 21}

    def consume(ctx: StageContext) -> dict[str, int]:
        return {"doubled": ctx.output_of("produce")["value"] * 2}

    wf = Workflow("w", [Stage("produce", produce), Stage("consume", consume, ("produce",))])
    ctx = asyncio.run(wf.run("t1", {}, state_store))
    assert ctx.outputs["consume"]["doubled"] == 42


def test_requesting_an_incomplete_output_raises() -> None:
    ctx = StageContext(transaction_id="t", request={})
    with pytest.raises(WorkflowError):
        ctx.output_of("missing")


def test_independent_stages_actually_run_concurrently(state_store) -> None:
    """Wall-clock proof, not just a claim about the graph shape."""

    async def sleeper(_: StageContext) -> dict[str, str]:
        await asyncio.sleep(0.25)
        return {"slept": "yes"}

    wf = Workflow("w", [Stage("a", sleeper), Stage("b", sleeper)])
    started = time.perf_counter()
    asyncio.run(wf.run("t2", {}, state_store))
    elapsed = time.perf_counter() - started
    # Sequential execution would take >= 0.5s; concurrent takes ~0.25s.
    assert elapsed < 0.4


def test_dependent_stages_do_not_overlap(state_store) -> None:
    order: list[str] = []

    async def first(_: StageContext) -> dict[str, str]:
        order.append("first_start")
        await asyncio.sleep(0.05)
        order.append("first_end")
        return {}

    async def second(_: StageContext) -> dict[str, str]:
        order.append("second_start")
        return {}

    wf = Workflow("w", [Stage("first", first), Stage("second", second, ("first",))])
    asyncio.run(wf.run("t3", {}, state_store))
    assert order == ["first_start", "first_end", "second_start"]


def test_sync_and_async_stage_functions_both_work(state_store) -> None:
    async def async_stage(_: StageContext) -> dict[str, str]:
        return {"kind": "async"}

    def sync_stage(_: StageContext) -> dict[str, str]:
        return {"kind": "sync"}

    wf = Workflow("w", [Stage("a", async_stage), Stage("b", sync_stage)])
    ctx = asyncio.run(wf.run("t4", {}, state_store))
    assert ctx.outputs["a"]["kind"] == "async"
    assert ctx.outputs["b"]["kind"] == "sync"


def test_every_stage_records_a_duration(state_store) -> None:
    wf = Workflow("w", [Stage("a", noop)])
    ctx = asyncio.run(wf.run("t5", {}, state_store))
    assert ctx.outputs["a"]["_duration_s"] >= 0.0


# --------------------------------------------------------------------------
# State, failure and resumption
# --------------------------------------------------------------------------
def test_state_is_persisted_per_stage(state_store) -> None:
    wf = Workflow("w", [Stage("a", noop), Stage("b", noop, ("a",))])
    asyncio.run(wf.run("t6", {}, state_store))
    records = {r.stage: r.status for r in state_store.get_stages("t6")}
    assert records == {"a": "completed", "b": "completed"}
    assert state_store.get_run("t6")["status"] == "completed"


def test_a_failing_stage_is_recorded_and_stops_the_run(state_store) -> None:
    def boom(_: StageContext) -> dict[str, str]:
        raise RuntimeError("kaboom")

    wf = Workflow("w", [Stage("a", noop), Stage("b", boom, ("a",)), Stage("c", noop, ("b",))])
    with pytest.raises(WorkflowError, match="kaboom"):
        asyncio.run(wf.run("t7", {}, state_store))

    records = {r.stage: r.status for r in state_store.get_stages("t7")}
    assert records["a"] == "completed"
    assert records["b"] == "failed"
    assert "c" not in records  # downstream never started
    assert state_store.get_run("t7")["status"] == "failed"
    assert "kaboom" in (state_store.get_stage("t7", "b").error or "")


def test_completed_stages_are_replayed_not_recomputed(state_store) -> None:
    calls = {"a": 0, "b": 0}

    def count_a(_: StageContext) -> dict[str, int]:
        calls["a"] += 1
        return {"n": calls["a"]}

    def count_b(_: StageContext) -> dict[str, int]:
        calls["b"] += 1
        return {"n": calls["b"]}

    wf = Workflow("w", [Stage("a", count_a), Stage("b", count_b, ("a",))])
    asyncio.run(wf.run("t8", {}, state_store))
    asyncio.run(wf.run("t8", {}, state_store))
    assert calls == {"a": 1, "b": 1}


def test_resume_false_recomputes_everything(state_store) -> None:
    calls = {"n": 0}

    def counter(_: StageContext) -> dict[str, int]:
        calls["n"] += 1
        return {"n": calls["n"]}

    wf = Workflow("w", [Stage("a", counter)])
    asyncio.run(wf.run("t9", {}, state_store))
    asyncio.run(wf.run("t9", {}, state_store, resume=False))
    assert calls["n"] == 2


def test_a_failed_stage_reruns_without_repeating_upstream_work(state_store) -> None:
    """The core resumability guarantee."""
    calls = {"upstream": 0, "flaky": 0}
    should_fail = {"value": True}

    def upstream(_: StageContext) -> dict[str, int]:
        calls["upstream"] += 1
        return {"value": 7}

    def flaky(ctx: StageContext) -> dict[str, int]:
        calls["flaky"] += 1
        if should_fail["value"]:
            raise RuntimeError("transient")
        return {"value": ctx.output_of("upstream")["value"] * 2}

    wf = Workflow("w", [Stage("upstream", upstream), Stage("flaky", flaky, ("upstream",))])

    with pytest.raises(WorkflowError):
        asyncio.run(wf.run("t10", {}, state_store))
    assert calls == {"upstream": 1, "flaky": 1}

    should_fail["value"] = False
    ctx = asyncio.run(wf.run("t10", {}, state_store))
    assert calls == {"upstream": 1, "flaky": 2}  # upstream was NOT recomputed
    assert ctx.outputs["flaky"]["value"] == 14
    assert state_store.get_run("t10")["status"] == "completed"


def test_only_restricts_execution_to_the_named_stages(state_store) -> None:
    calls = {"a": 0, "b": 0}

    def make(name: str):
        def fn(_: StageContext) -> dict[str, int]:
            calls[name] += 1
            return {"n": calls[name]}

        return fn

    wf = Workflow("w", [Stage("a", make("a")), Stage("b", make("b"), ("a",))])
    asyncio.run(wf.run("t11", {}, state_store))
    asyncio.run(wf.run("t11", {}, state_store, only=["b"]))
    assert calls == {"a": 1, "b": 2}


def test_state_survives_a_new_store_instance(state_store) -> None:
    from ticketiq.workflow.state import StateStore

    wf = Workflow("w", [Stage("a", noop)])
    asyncio.run(wf.run("t12", {}, state_store))
    state_store.close()

    reopened = StateStore(state_store.path)
    try:
        assert reopened.get_stage("t12", "a").status == "completed"
    finally:
        reopened.close()


def test_run_request_payload_is_stored(state_store) -> None:
    wf = Workflow("w", [Stage("a", noop)])
    asyncio.run(wf.run("t13", {"subject": "hello"}, state_store))
    assert state_store.get_run("t13")["request"]["subject"] == "hello"


def test_unknown_transaction_returns_none(state_store) -> None:
    assert state_store.get_run("nope") is None
    assert state_store.get_stages("nope") == []
    assert state_store.get_decision("nope") is None
