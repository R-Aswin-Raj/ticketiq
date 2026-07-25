"""FastAPI application.

Endpoints
---------
``POST /ticket``                        run the full triage pipeline
``POST /feedback``                      1/0 feedback, updates the bandit
``GET  /ticket/{id}/status``            real per-stage state from the DAG store
``POST /ticket/{id}/rerun/{stage}``     re-run one stage, replaying upstream
``GET  /rl/stats``                      bandit table and action distribution
``GET  /pipeline/graph``                the DAG, including which levels are parallel
``GET  /healthz``                       liveness
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ticketiq import __version__
from ticketiq.config import get_settings
from ticketiq.llm.factory import describe_model
from ticketiq.pipeline.service import (
    UnknownTransaction,
    apply_feedback,
    get_status,
    process_ticket,
    rerun_stage,
)
from ticketiq.pipeline.stages import get_workflow
from ticketiq.rl.service import get_bandit, persist_bandit
from ticketiq.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    TicketRequest,
    TicketResponse,
    TicketStatusResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ticketiq.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the expensive singletons so the first request is not an outlier."""
    settings = get_settings()
    logger.info("starting TicketIQ %s in LLM_MODE=%s", __version__, settings.llm_mode)
    from ticketiq.ml.classifier import get_classifier
    from ticketiq.rag.store import get_vector_store

    get_classifier()
    store = get_vector_store()
    logger.info("knowledge base indexed: %d chunks (%s)", len(store), store.backend)
    get_workflow()
    yield
    persist_bandit()
    logger.info("shutdown complete")


app = FastAPI(
    title="TicketIQ",
    version=__version__,
    description="Self-optimizing support triage agent.",
    lifespan=lifespan,
)


@app.exception_handler(UnknownTransaction)
async def _unknown_transaction_handler(_: Any, exc: UnknownTransaction) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"unknown transaction {exc.args[0]}"})


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_mode": settings.llm_mode,
        "model": describe_model(),
    }


@app.post("/ticket", response_model=TicketResponse)
async def post_ticket(request: TicketRequest) -> TicketResponse:
    """Run classification, aspect sentiment, retrieval, the agent and the reply."""
    try:
        return await process_ticket(request)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 500 with context
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc


@app.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record 1/0 feedback and update the bandit for the arm that was used."""
    try:
        return apply_feedback(request.transaction_id, request.score)
    except UnknownTransaction as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown transaction {request.transaction_id}"
        ) from exc


@app.get("/ticket/{transaction_id}/status", response_model=TicketStatusResponse)
async def get_ticket_status(transaction_id: str) -> TicketStatusResponse:
    """Per-stage completion state, read from the workflow engine's store."""
    try:
        return get_status(transaction_id)
    except UnknownTransaction as exc:
        raise HTTPException(status_code=404, detail=f"unknown transaction {transaction_id}") from exc


@app.post("/ticket/{transaction_id}/rerun/{stage}", response_model=TicketStatusResponse)
async def post_rerun_stage(transaction_id: str, stage: str) -> TicketStatusResponse:
    """Re-run a failed stage without repeating completed upstream stages."""
    try:
        return await rerun_stage(transaction_id, stage)
    except UnknownTransaction as exc:
        raise HTTPException(status_code=404, detail=f"unknown transaction {transaction_id}") from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/rl/stats")
async def get_rl_stats() -> dict[str, Any]:
    """Inspect the bandit: per-state Q values, pull counts, action distribution."""
    bandit = get_bandit()
    return {
        "action_distribution": bandit.action_distribution(),
        "table": bandit.snapshot(),
    }


@app.get("/pipeline/graph")
async def get_pipeline_graph() -> dict[str, Any]:
    """Expose the DAG so the parallelism and dependencies are inspectable."""
    workflow = get_workflow()
    return {"name": workflow.name, "levels": workflow.describe(), "order": workflow.order()}
