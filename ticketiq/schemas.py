"""Pydantic models for the public API and shared domain types."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

CATEGORIES: tuple[str, ...] = ("billing", "technical", "account", "feature_request")
TIERS: tuple[str, ...] = ("free", "pro", "enterprise")


class Tier(str, Enum):
    free = "free"
    pro = "pro"
    enterprise = "enterprise"


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    feature_request = "feature_request"


class AgentAction(str, Enum):
    """What the agentic layer decided to do with the ticket."""

    answer_directly = "answer_directly"
    tool_call = "tool_call"
    escalate = "escalate"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------
class TicketRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=8000)
    tier: Tier = Tier.free
    customer_id: str | None = Field(default=None, max_length=64)
    order_id: str | None = Field(default=None, max_length=64)


class FeedbackRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=64)
    score: Literal[0, 1] = Field(description="1 = helpful, 0 = unhelpful")


# --------------------------------------------------------------------------
# Response fragments
# --------------------------------------------------------------------------
class AspectSentiment(BaseModel):
    aspect: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    label: Literal["negative", "neutral", "positive"]
    evidence: str


class ClassificationResult(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float]


class KnowledgeSnippet(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class PipelineConfig(BaseModel):
    """The arm the bandit pulled, echoed back for observability."""

    arm_id: str
    llm_variant: str
    model: str
    top_k: int


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class AgentTrace(BaseModel):
    action: AgentAction
    reasoning: list[str]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    escalation_reason: str | None = None


class TicketResponse(BaseModel):
    transaction_id: str
    classification: ClassificationResult
    aspects: list[AspectSentiment]
    urgency: float = Field(ge=0.0, le=1.0)
    snippets: list[KnowledgeSnippet]
    agent: AgentTrace
    response_text: str
    config: PipelineConfig
    latency_s: float


class StageStatus(BaseModel):
    name: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    started_at: float | None = None
    finished_at: float | None = None
    duration_s: float | None = None
    error: str | None = None
    output: dict[str, Any] | None = None


class TicketStatusResponse(BaseModel):
    transaction_id: str
    pipeline_status: Literal["pending", "running", "completed", "failed"]
    stages: list[StageStatus]


class FeedbackResponse(BaseModel):
    transaction_id: str
    arm_id: str
    reward: float
    pulls: int
    mean_reward: float
