"""
Tests for agent.py — the real tool-calling agent loop.

Fully offline: uses ScriptedMockAgentClient (no API key/network) and the
same Confluence fakes as the rest of the test suite.

Run with: pytest test_agent.py -v
"""

import pytest

from agent import (
    AgentContext,
    AgentError,
    PrivacyAssessmentAgent,
    ScriptedMockAgentClient,
    TOOL_SCHEMAS,
    build_draft_from_submission,
    dispatch_tool,
)
from confluence_client import ConfluenceClient
from draft_generator import classify_risk_level
from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow
from vector_store import HashingEmbedder, PriorAssessmentStore


def make_context(assessment_store=None, run_attacker_model=None):
    confluence_client = ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(make_confluence_page_flow(
            "Architecture Overview", SAMPLE_STORAGE_HTML, "/spaces/CUST014/pages/page-1"
        )),
    )
    return AgentContext(
        confluence_client=confluence_client,
        assessment_store=assessment_store or PriorAssessmentStore(embedder=HashingEmbedder(dim=128)),
        run_attacker_model=run_attacker_model or (lambda use_case: 0.5),
        space_key="CUST014",
    )


# --------------------------------------------------------------------------
# Schema-level safety guarantee
# --------------------------------------------------------------------------

def test_submit_tool_schema_has_no_risk_level_field():
    """The strongest version of the 'don't let the LLM set risk_level' guarantee:
    it isn't in the schema at all, so the model has no way to even attempt it."""
    submit_schema = next(t for t in TOOL_SCHEMAS if t["name"] == "submit_privacy_assessment")
    properties = submit_schema["input_schema"]["properties"]
    assert "risk_level" not in properties
    assert "risk_score" in properties


def test_build_draft_from_submission_computes_risk_level_deterministically():
    submission = {
        "customer_ref": "cust_x",
        "risk_score": 0.62,
        "threat_model": "text",
        "config_recommendation": "text",
        "gdpr_considerations": [{"article": "Art. 5", "relevance": "text"}],
    }
    draft = build_draft_from_submission(submission)
    assert draft.risk_level == classify_risk_level(0.62) == "high"


# --------------------------------------------------------------------------
# Agent loop with the scripted mock — proves the branching actually happens
# --------------------------------------------------------------------------

def test_agent_runs_attacker_model_when_no_prior_match():
    call_log = []
    context = make_context(run_attacker_model=lambda use_case: call_log.append(use_case) or 0.35)

    agent = PrivacyAssessmentAgent(llm_client=ScriptedMockAgentClient())
    draft, trace = agent.run(customer_ref="cust_014", context=context)

    assert trace == [
        "search_confluence_docs",
        "search_prior_assessments",
        "run_attacker_model",
        "submit_privacy_assessment",
    ]
    assert len(call_log) == 1
    assert draft.risk_score == 0.35
    assert draft.risk_level == "medium"


def test_agent_skips_attacker_model_when_prior_match_exists():
    assessment_store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))
    assessment_store.add_assessment(
        customer_ref="cust_prior",
        use_case_summary=(
            "Data Sharing Overview This customer requests anonymized vehicle "
            "trajectory data Format CSV Frequency daily batch"
        ),
        risk_level="medium",
        risk_score=0.35,
    )
    call_log = []
    context = make_context(
        assessment_store=assessment_store,
        run_attacker_model=lambda use_case: call_log.append(use_case) or 0.9,  # would be VERY wrong if called
    )

    agent = PrivacyAssessmentAgent(llm_client=ScriptedMockAgentClient())
    draft, trace = agent.run(customer_ref="cust_014", context=context)

    assert "run_attacker_model" not in trace
    assert len(call_log) == 0
    assert draft.risk_score == 0.35  # from the prior assessment, not the (unused) attacker model
    assert draft.risk_level == "medium"


def test_agent_trace_always_ends_with_submission():
    context = make_context()
    agent = PrivacyAssessmentAgent(llm_client=ScriptedMockAgentClient())
    _draft, trace = agent.run(customer_ref="cust_014", context=context)
    assert trace[-1] == "submit_privacy_assessment"


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------

class NeverSubmitsClient:
    """A 'model' that keeps calling a tool forever and never submits — must
    trip the max-iterations guard rather than looping forever."""

    def send(self, system, messages, tools):
        return [{"type": "tool_use", "id": "x", "name": "search_confluence_docs", "input": {"space_key": "CUST014"}}]


class UnlimitedConfluenceStub:
    """Returns the same doc every call, unlike the finite FakeSession-backed
    client — needed here because this test calls the tool repeatedly on
    purpose, to prove the iteration limit (not the fake HTTP queue) is what
    stops it."""

    def get_customer_docs(self, space_key: str):
        from confluence_client import CustomerDoc
        return [CustomerDoc(page_id="p1", title="Doc", content="content", version=1, url="https://example.com")]


def test_agent_raises_after_max_iterations_if_never_submitted():
    context = make_context()
    context.confluence_client = UnlimitedConfluenceStub()
    agent = PrivacyAssessmentAgent(llm_client=NeverSubmitsClient(), max_iterations=3)

    with pytest.raises(AgentError):
        agent.run(customer_ref="cust_014", context=context)


class NoToolCallClient:
    """A 'model' that returns plain text instead of a tool call — should be
    treated as an error, not silently accepted as a final answer."""

    def send(self, system, messages, tools):
        return [{"type": "text", "text": "I think the risk is probably fine."}]


def test_agent_raises_if_model_returns_no_tool_calls():
    context = make_context()
    agent = PrivacyAssessmentAgent(llm_client=NoToolCallClient())

    with pytest.raises(AgentError):
        agent.run(customer_ref="cust_014", context=context)


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

def test_dispatch_search_confluence_docs():
    context = make_context()
    result = dispatch_tool("search_confluence_docs", {"space_key": "CUST014"}, context)
    assert len(result["docs"]) == 1
    assert "anonymized vehicle trajectory data" in result["docs"][0]["content"]


def test_dispatch_search_prior_assessments_empty_store():
    context = make_context()
    result = dispatch_tool("search_prior_assessments", {"use_case_summary": "anything"}, context)
    assert result["matches"] == []


def test_dispatch_unknown_tool_raises():
    from agent import ToolExecutionError
    context = make_context()
    with pytest.raises(ToolExecutionError):
        dispatch_tool("not_a_real_tool", {}, context)
