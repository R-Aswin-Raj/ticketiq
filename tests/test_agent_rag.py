"""Tests for retrieval, aspect sentiment, urgency, tools and the agent loop."""

from __future__ import annotations

import asyncio

import pytest
from ticketiq.agent.react import AgentContext, ReActAgent, hard_escalation_reason
from ticketiq.agent.tools import check_account_status, check_refund_eligibility, run_tool
from ticketiq.llm.base import extract_json
from ticketiq.llm.configs import ARMS, ARMS_BY_ID, get_arm
from ticketiq.llm.mock import MockLLM
from ticketiq.ml.aspects import extract_aspects, split_sentences
from ticketiq.ml.sentiment import LexiconSentiment, label_for
from ticketiq.ml.urgency import urgency_bucket, urgency_score
from ticketiq.rag.embeddings import HashingEmbedder, cosine
from ticketiq.rag.kb import chunk_markdown
from ticketiq.rag.store import VectorStore, get_vector_store, sparse_cosine


# --------------------------------------------------------------------------
# Sentiment and aspects
# --------------------------------------------------------------------------
def test_lexicon_scores_polarity_in_the_right_direction() -> None:
    scorer = LexiconSentiment()
    assert scorer.score("this is terrible and broken") < -0.3
    assert scorer.score("the support was excellent and fast") > 0.3
    assert scorer.score("the invoice arrived on tuesday") == pytest.approx(0.0)


def test_negation_flips_polarity() -> None:
    scorer = LexiconSentiment()
    assert scorer.score("this is not great") < scorer.score("this is great")


def test_label_for_thresholds() -> None:
    assert label_for(-0.9) == "negative"
    assert label_for(0.0) == "neutral"
    assert label_for(0.9) == "positive"


def test_contrastive_clauses_are_split() -> None:
    parts = split_sentences("Support was great but the app crashes.")
    assert len(parts) == 2


def test_aspects_are_scored_independently() -> None:
    """The defining property of aspect-level sentiment."""
    aspects = extract_aspects(
        "Mixed experience",
        "Support was great but the dashboard crashes constantly and exports time out.",
    )
    by_name = {a.aspect: a for a in aspects}
    assert "performance" in by_name
    assert by_name["performance"].sentiment < 0
    if "support response time" in by_name:
        assert by_name["support response time"].sentiment > 0


def test_at_most_three_aspects_are_returned() -> None:
    aspects = extract_aspects(
        "Everything",
        "Billing is wrong, login fails, performance is slow, support never replies, "
        "reports are broken and permissions are confusing.",
    )
    assert 1 <= len(aspects) <= 3


def test_a_ticket_with_no_keyword_still_returns_one_aspect() -> None:
    aspects = extract_aspects("Hello", "Just saying hi.")
    assert len(aspects) == 1


def test_evidence_is_quoted_from_the_ticket() -> None:
    subject, body = "Billing problem", "I was charged twice this month."
    haystack = f"{subject}. {body}".lower()
    aspects = extract_aspects(subject, body)
    assert all(a.evidence for a in aspects)
    assert all(a.evidence.rstrip(".").lower() in haystack for a in aspects)


# --------------------------------------------------------------------------
# Urgency
# --------------------------------------------------------------------------
def test_urgency_is_bounded() -> None:
    aspects = extract_aspects("Outage", "Everything is broken and unusable. Urgent!")
    score = urgency_score("technical", aspects, "enterprise", "urgent outage critical")
    assert 0.0 <= score <= 1.0


def test_enterprise_outranks_free_all_else_equal() -> None:
    aspects = extract_aspects("Slow", "The dashboard is slow.")
    text = "The dashboard is slow."
    assert urgency_score("technical", aspects, "enterprise", text) > urgency_score(
        "technical", aspects, "free", text
    )


def test_feature_requests_are_less_urgent_than_outages() -> None:
    calm = extract_aspects("Idea", "It would be great to have dark mode.")
    angry = extract_aspects("Down", "Production is down and we are blocked. Urgent.")
    assert urgency_score("feature_request", calm, "pro", "dark mode") < urgency_score(
        "technical", angry, "pro", "production is down urgent blocked"
    )


def test_urgency_buckets_partition_the_range() -> None:
    assert urgency_bucket(0.1) == "low"
    assert urgency_bucket(0.5) == "medium"
    assert urgency_bucket(0.9) == "high"


# --------------------------------------------------------------------------
# Chunking and retrieval
# --------------------------------------------------------------------------
def test_frontmatter_is_stripped_and_title_extracted() -> None:
    raw = "---\nid: doc\ntitle: My Policy\n---\n# Heading\nBody text here.\n"
    chunks = chunk_markdown(raw, "doc")
    assert chunks[0].title == "My Policy"
    assert "title:" not in chunks[0].text


def test_chunks_split_on_headings() -> None:
    raw = "# One\nAlpha content.\n\n# Two\nBeta content.\n"
    chunks = chunk_markdown(raw, "doc")
    assert {c.heading for c in chunks} == {"One", "Two"}


def test_oversized_sections_are_split_with_overlap() -> None:
    raw = "# Big\n" + ("word " * 400)
    chunks = chunk_markdown(raw, "doc", chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_ids_are_unique() -> None:
    raw = "# A\nalpha\n# B\nbeta\n# C\ngamma\n"
    ids = [c.chunk_id for c in chunk_markdown(raw, "doc")]
    assert len(ids) == len(set(ids))


def test_cosine_bounds() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0


def test_sparse_cosine_matches_dense_intuition() -> None:
    assert sparse_cosine({0: 1.0}, {0: 1.0}) == pytest.approx(1.0)
    assert sparse_cosine({0: 1.0}, {1: 1.0}) == pytest.approx(0.0)
    assert sparse_cosine({}, {0: 1.0}) == 0.0


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed("refund my invoice")
    b = embedder.embed("refund my invoice")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0, abs=1e-6)


def test_retrieval_returns_the_expected_document(isolated_env) -> None:
    store = get_vector_store()
    hits = store.search("SAML single sign on fails for all users", top_k=3)
    assert hits[0].chunk.doc_id == "account_access_troubleshooting"


def test_retrieval_respects_top_k(isolated_env) -> None:
    store = get_vector_store()
    assert len(store.search("refund policy", top_k=2)) == 2
    assert len(store.search("refund policy", top_k=5)) == 5


def test_retrieval_scores_are_ordered(isolated_env) -> None:
    hits = get_vector_store().search("password reset link", top_k=5)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_empty_store_returns_nothing() -> None:
    assert VectorStore().search("anything", top_k=3) == []


def test_top_k_zero_returns_nothing(isolated_env) -> None:
    assert get_vector_store().search("refund", top_k=0) == []


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def test_tools_are_deterministic() -> None:
    assert check_account_status("cust-1") == check_account_status("cust-1")
    assert check_refund_eligibility("ord-1") == check_refund_eligibility("ord-1")


def test_tools_validate_missing_identifiers() -> None:
    assert check_account_status("")["found"] is False
    assert check_refund_eligibility("")["found"] is False


def test_run_tool_rejects_unknown_names() -> None:
    assert "error" in run_tool("nonexistent", {})


def test_run_tool_tolerates_missing_arguments() -> None:
    assert run_tool("check_account_status", {})["found"] is False


def test_refund_eligibility_flags_human_approval() -> None:
    outcomes = [check_refund_eligibility(f"ord-{i}") for i in range(40)]
    assert any(o["requires_human_approval"] for o in outcomes)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def test_arm_space_is_the_full_cross_product() -> None:
    assert len(ARMS) == 4
    assert {a.top_k for a in ARMS} == {2, 5}
    assert {a.variant_id for a in ARMS} == {"concise", "empathetic"}


def test_arms_have_distinct_system_prompts() -> None:
    prompts = {get_arm(a.id).variant.system for a in ARMS}
    assert len(prompts) == 2


def test_unknown_arm_raises() -> None:
    with pytest.raises(KeyError):
        get_arm("does-not-exist")


# --------------------------------------------------------------------------
# JSON extraction and the agent loop
# --------------------------------------------------------------------------
def test_extract_json_handles_bare_json() -> None:
    assert extract_json('{"action": "escalate"}')["action"] == "escalate"


def test_extract_json_handles_code_fences() -> None:
    assert extract_json('```json\n{"action": "escalate"}\n```')["action"] == "escalate"


def test_extract_json_handles_surrounding_prose() -> None:
    text = 'Sure! Here is my decision:\n{"action": "answer_directly"}\nHope that helps.'
    assert extract_json(text)["action"] == "answer_directly"


def test_extract_json_repairs_trailing_comma() -> None:
    assert extract_json('{"action": "escalate", "tool": null,}')["action"] == "escalate"


def test_extract_json_repairs_single_quotes() -> None:
    assert extract_json("{'action': 'escalate', 'tool': null,}")["action"] == "escalate"


def test_extract_json_repair_never_corrupts_valid_json() -> None:
    # An apostrophe inside a value must not be mangled by the single-quote repair.
    assert (
        extract_json('{"thought": "it\'s locked", "action": "escalate"}')["thought"]
        == "it's locked"
    )


def test_extract_json_returns_empty_on_garbage() -> None:
    assert extract_json("no json here at all") == {}
    assert extract_json("{not: valid json}") == {}


@pytest.mark.parametrize(
    "text",
    [
        "We have suffered data loss across the workspace",
        "Our lawyer will be in touch about this",
        "There has been a security breach",
        "Production is down for everyone",
    ],
)
def test_hard_escalation_triggers_fire(text: str) -> None:
    assert hard_escalation_reason(text) is not None


def test_hard_escalation_does_not_fire_on_ordinary_tickets() -> None:
    assert hard_escalation_reason("Could you add dark mode please?") is None


def _context(**overrides) -> AgentContext:
    base = {
        "subject": "Double charged",
        "body": "We were billed twice and would like a refund.",
        "tier": "pro",
        "category": "billing",
        "urgency": 0.5,
        "urgency_bucket": "medium",
        "aspects": [{"aspect": "billing", "sentiment": -0.6, "label": "negative", "evidence": "x"}],
        "snippets": [],
        "customer_id": "cust-1",
        "order_id": "ord-1",
    }
    base.update(overrides)
    return AgentContext(**base)  # type: ignore[arg-type]


def test_agent_escalates_immediately_on_a_guardrail_hit() -> None:
    agent = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["concise-k2"])
    result = asyncio.run(agent.run(_context(body="We have suffered data loss.")))
    assert result.action == "escalate"
    assert result.tool_calls == []
    assert "Guardrail" in result.reasoning[0]
    assert result.response_text


def test_agent_calls_a_tool_for_a_refund_request() -> None:
    agent = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["concise-k2"])
    result = asyncio.run(agent.run(_context()))
    assert result.action == "tool_call"
    assert result.tool_calls[0]["tool"] == "check_refund_eligibility"
    assert result.tool_calls[0]["arguments"]["order_id"] == "ord-1"


def test_agent_records_a_reasoning_trace() -> None:
    agent = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["concise-k2"])
    result = asyncio.run(agent.run(_context()))
    assert any(step.startswith("Thought:") for step in result.reasoning)
    assert any(step.startswith("Observation:") for step in result.reasoning)


def test_agent_answers_directly_when_no_tool_is_needed() -> None:
    agent = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["concise-k2"])
    result = asyncio.run(
        agent.run(
            _context(
                subject="Roadmap question",
                body="Is dark mode on the roadmap?",
                category="feature_request",
            )
        )
    )
    assert result.action == "answer_directly"
    assert result.tool_calls == []


def test_agent_falls_back_to_escalation_when_the_model_fails() -> None:
    from ticketiq.llm.base import LLMError

    class BrokenLLM:
        name = "broken"

        async def complete(self, prompt: str, **kwargs) -> object:
            raise LLMError("backend down")

    agent = ReActAgent(client=BrokenLLM(), arm=ARMS_BY_ID["concise-k2"])  # type: ignore[arg-type]
    result = asyncio.run(agent.run(_context()))
    assert result.action == "escalate"
    assert result.response_text  # the customer still gets a reply


def test_agent_tolerates_an_unparseable_decision() -> None:
    class BabblingLLM:
        name = "babble"

        async def complete(self, prompt: str, **kwargs):
            from ticketiq.llm.base import LLMResult

            return LLMResult(text="I have no idea what to do", model="babble", latency_s=0.0)

    agent = ReActAgent(client=BabblingLLM(), arm=ARMS_BY_ID["concise-k2"])  # type: ignore[arg-type]
    result = asyncio.run(agent.run(_context()))
    assert result.action == "answer_directly"


def test_the_two_prompt_variants_produce_different_replies() -> None:
    """If the arms were interchangeable the bandit would have nothing to learn."""
    concise = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["concise-k2"])
    empathetic = ReActAgent(client=MockLLM(), arm=ARMS_BY_ID["empathetic-k2"])
    a = asyncio.run(concise.run(_context())).response_text
    b = asyncio.run(empathetic.run(_context())).response_text
    assert a != b
    assert len(b) > len(a)
