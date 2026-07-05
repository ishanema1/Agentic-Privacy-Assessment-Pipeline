"""
Tests for draft_generator.py. Fully offline — uses a fake LLM client, so
no API key or network access is required.

Run with: pytest test_draft_generator.py -v
"""

import json

import pytest

from draft_generator import (
    DraftGenerationError,
    DraftGenerator,
    MockLLMClient,
    classify_risk_level,
)


class FakeLLMClient:
    """Returns a canned response so we control exactly what the 'model' says."""

    def __init__(self, response: str):
        self.response = response
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.response


VALID_RESPONSE = json.dumps({
    "threat_model": "Example threat model text.",
    "config_recommendation": "Example recommendation text.",
    "gdpr_considerations": [
        {"article": "Article 5", "relevance": "Data minimization applies here."}
    ],
})


# --------------------------------------------------------------------------
# Deterministic risk classification — the core design guarantee
# --------------------------------------------------------------------------

def test_classify_risk_level_boundaries():
    assert classify_risk_level(0.0) == "low"
    assert classify_risk_level(0.14) == "low"
    assert classify_risk_level(0.15) == "medium"
    assert classify_risk_level(0.49) == "medium"
    assert classify_risk_level(0.50) == "high"
    assert classify_risk_level(1.0) == "high"


def test_classify_risk_level_rejects_out_of_range():
    with pytest.raises(ValueError):
        classify_risk_level(1.5)
    with pytest.raises(ValueError):
        classify_risk_level(-0.1)


def test_risk_level_is_not_taken_from_llm_even_if_it_tries_to_supply_one():
    """
    If the LLM's response smuggles in a risk_level/risk_score key, the
    generator must ignore it — those fields are only ever set from the
    deterministic classifier, never from parsed LLM output.
    """
    response_with_smuggled_risk = json.dumps({
        "threat_model": "text",
        "config_recommendation": "text",
        "gdpr_considerations": [],
        "risk_level": "low",       # LLM should not be able to override this
        "risk_score": 0.01,        # or this
    })
    fake_client = FakeLLMClient(response_with_smuggled_risk)
    generator = DraftGenerator(llm_client=fake_client)

    draft = generator.generate(customer_ref="cust_x", customer_docs=["doc"], risk_score=0.80)

    assert draft.risk_score == 0.80          # from the real input, not the LLM
    assert draft.risk_level == "high"        # classify_risk_level(0.80), not the LLM's "low"


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

def test_generate_happy_path():
    fake_client = FakeLLMClient(VALID_RESPONSE)
    generator = DraftGenerator(llm_client=fake_client)

    draft = generator.generate(customer_ref="cust_014", customer_docs=["some doc"], risk_score=0.30)

    assert draft.customer_ref == "cust_014"
    assert draft.risk_level == "medium"
    assert draft.threat_model == "Example threat model text."
    assert len(draft.gdpr_considerations) == 1
    assert draft.gdpr_considerations[0].article == "Article 5"


def test_parses_response_wrapped_in_markdown_code_fence():
    fenced = f"```json\n{VALID_RESPONSE}\n```"
    fake_client = FakeLLMClient(fenced)
    generator = DraftGenerator(llm_client=fake_client)

    draft = generator.generate(customer_ref="cust_014", customer_docs=["doc"], risk_score=0.05)
    assert draft.threat_model == "Example threat model text."


def test_invalid_json_raises_draft_generation_error():
    fake_client = FakeLLMClient("this is not json at all")
    generator = DraftGenerator(llm_client=fake_client)

    with pytest.raises(DraftGenerationError):
        generator.generate(customer_ref="cust_014", customer_docs=["doc"], risk_score=0.1)


def test_missing_required_key_raises_draft_generation_error():
    incomplete = json.dumps({"threat_model": "text"})  # missing other required keys
    fake_client = FakeLLMClient(incomplete)
    generator = DraftGenerator(llm_client=fake_client)

    with pytest.raises(DraftGenerationError):
        generator.generate(customer_ref="cust_014", customer_docs=["doc"], risk_score=0.1)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def test_prompt_includes_customer_docs_and_risk_score():
    fake_client = FakeLLMClient(VALID_RESPONSE)
    generator = DraftGenerator(llm_client=fake_client)

    generator.generate(
        customer_ref="cust_014",
        customer_docs=["Architecture doc content goes here."],
        risk_score=0.33,
    )

    assert "Architecture doc content goes here." in fake_client.last_user
    assert "0.330" in fake_client.last_user


def test_prompt_includes_prior_assessment_when_provided():
    fake_client = FakeLLMClient(VALID_RESPONSE)
    generator = DraftGenerator(llm_client=fake_client)

    generator.generate(
        customer_ref="cust_014",
        customer_docs=["doc"],
        risk_score=0.2,
        prior_assessment_summary="Prior case cust_009 was approved at 50-unit cells.",
    )

    assert "cust_009" in fake_client.last_user


def test_system_prompt_instructs_llm_not_to_set_risk_fields():
    fake_client = FakeLLMClient(VALID_RESPONSE)
    generator = DraftGenerator(llm_client=fake_client)
    generator.generate(customer_ref="cust_014", customer_docs=["doc"], risk_score=0.2)

    assert "not yours to determine" in fake_client.last_system.lower() or \
           "risk level or risk score" in fake_client.last_system.lower()


# --------------------------------------------------------------------------
# Mock client + markdown rendering
# --------------------------------------------------------------------------

def test_mock_llm_client_produces_a_valid_parseable_draft():
    generator = DraftGenerator(llm_client=MockLLMClient())
    draft = generator.generate(customer_ref="cust_demo", customer_docs=["doc"], risk_score=0.42)
    assert draft.risk_level == "medium"
    assert len(draft.gdpr_considerations) >= 1


def test_as_markdown_includes_key_sections():
    fake_client = FakeLLMClient(VALID_RESPONSE)
    generator = DraftGenerator(llm_client=fake_client)
    draft = generator.generate(customer_ref="cust_014", customer_docs=["doc"], risk_score=0.60)

    markdown = draft.as_markdown()
    assert "cust_014" in markdown
    assert "HIGH" in markdown
    assert "Example threat model text." in markdown
    assert "Article 5" in markdown
