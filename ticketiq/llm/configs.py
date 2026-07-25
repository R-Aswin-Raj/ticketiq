"""The configuration space the bandit chooses from.

This module is where requirement 3's "two distinct LLM configurations" lives:
the two prompt variants below (``concise`` / ``empathetic``) are the genuine
choice the RL layer learns between. Note that ticket *classification* is not an
LLM at all -- that is the hand-rolled logistic regression in ``ml/`` -- and
``LLM_MODE``
(mock / local / cloud) swaps the backend under all arms at once rather than
being one of the choices the bandit makes.

Two dimensions:

1. **Prompt variant** — ``concise`` (short, policy-first, low temperature) vs
   ``empathetic`` (longer, apologetic, explicit next steps). Same model, two
   genuinely different system instructions, so the choice is real without
   requiring two models to be pulled.
2. **RAG top-K** — 2 vs 5. More context usually helps quality and always costs
   latency.

The cross product gives four arms. That is small enough for a tabular
contextual bandit to learn quickly with realistic feedback volumes, which
matters more here than a large action space.
"""

from __future__ import annotations

from dataclasses import dataclass

CONCISE_SYSTEM = """You are TicketIQ, a B2B SaaS support agent.
Answer in at most four sentences. Lead with the policy or fix that applies.
Do not apologise more than once. Do not invent policy: if the provided context
does not cover the question, say so and state that it is being escalated.
Never promise a delivery date for a feature."""

EMPATHETIC_SYSTEM = """You are TicketIQ, a senior B2B SaaS support agent.
Acknowledge the customer's impact first, then explain step-by-step what applies
and what happens next, using a short numbered list. Be warm but precise. Ground
every claim in the provided context; if the context does not cover it, say so
and state that it is being escalated. Never promise a delivery date."""


@dataclass(frozen=True)
class PromptVariant:
    id: str
    system: str
    temperature: float
    max_tokens: int
    # Only used by the offline mock, to simulate the quality/latency trade-off.
    mock_base_latency_s: float


VARIANTS: dict[str, PromptVariant] = {
    "concise": PromptVariant(
        id="concise",
        system=CONCISE_SYSTEM,
        temperature=0.1,
        max_tokens=220,
        mock_base_latency_s=0.05,
    ),
    "empathetic": PromptVariant(
        id="empathetic",
        system=EMPATHETIC_SYSTEM,
        temperature=0.35,
        max_tokens=520,
        mock_base_latency_s=0.16,
    ),
}

TOP_K_CHOICES: tuple[int, ...] = (2, 5)


@dataclass(frozen=True)
class Arm:
    """One selectable pipeline configuration."""

    id: str
    variant_id: str
    top_k: int

    @property
    def variant(self) -> PromptVariant:
        return VARIANTS[self.variant_id]


ARMS: tuple[Arm, ...] = tuple(
    Arm(id=f"{variant_id}-k{top_k}", variant_id=variant_id, top_k=top_k)
    for variant_id in VARIANTS
    for top_k in TOP_K_CHOICES
)

ARMS_BY_ID: dict[str, Arm] = {arm.id: arm for arm in ARMS}
DEFAULT_ARM_ID = ARMS[0].id


def get_arm(arm_id: str) -> Arm:
    try:
        return ARMS_BY_ID[arm_id]
    except KeyError as exc:
        raise KeyError(f"unknown arm {arm_id!r}; known: {sorted(ARMS_BY_ID)}") from exc
