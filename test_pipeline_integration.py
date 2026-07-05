"""
test_pipeline_integration.py

End-to-end test of pipeline.run_pipeline() wiring together all four
implemented modules:

    confluence_client -> vector_store -> attacker_model -> draft_generator

Fully offline: Confluence is mocked (fakes.py), the embedder falls back to
the dependency-free hashing embedder, the attacker model runs against a
small synthetic population, and the LLM is the deterministic MockLLMClient.
No credentials, API keys, or network access are required.

Run with: pytest test_pipeline_integration.py -v
"""

from confluence_client import ConfluenceClient
from draft_generator import DraftGenerator, MockLLMClient, classify_risk_level
from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow
from pipeline import run_pipeline
from vector_store import HashingEmbedder, PriorAssessmentStore
from attacker_model import evaluate_reidentification_risk


def make_confluence_client(storage_html: str = SAMPLE_STORAGE_HTML) -> ConfluenceClient:
    responses = make_confluence_page_flow(
        title="Architecture Overview",
        storage_html=storage_html,
        webui_path="/spaces/CUST014/pages/page-1",
    )
    return ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(responses),
    )


def counting_attacker_model(call_log: list[str]):
    """Wraps the real attacker model but records whether it was invoked,
    so tests can assert the reuse path skips it entirely."""
    def _run(use_case_summary: str) -> float:
        call_log.append(use_case_summary)
        # small population keeps this fast; we only care that it *runs*, not
        # about a specific number here
        return evaluate_reidentification_risk(cell_size=30, n_individuals=15, n_days=3, seed=0)
    return _run


# --------------------------------------------------------------------------
# No prior assessment exists -> attacker model runs -> fresh draft
# --------------------------------------------------------------------------

def test_pipeline_runs_attacker_model_when_no_prior_assessment_exists():
    confluence_client = make_confluence_client()
    assessment_store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))  # empty store
    draft_generator = DraftGenerator(llm_client=MockLLMClient())
    call_log: list[str] = []

    result = run_pipeline(
        customer_ref="cust_new",
        space_key="CUST_NEW",
        confluence_client=confluence_client,
        assessment_store=assessment_store,
        draft_generator=draft_generator,
        run_attacker_model=counting_attacker_model(call_log),
    )

    assert result.reused_prior_assessment is False
    assert len(call_log) == 1                       # attacker model WAS invoked
    assert 0.0 <= result.risk_score <= 1.0
    assert result.draft.risk_level == classify_risk_level(result.risk_score)
    assert result.draft.customer_ref == "cust_new"
    assert len(result.customer_docs) == 1
    assert "anonymized vehicle trajectory data" in result.customer_docs[0].content


# --------------------------------------------------------------------------
# A closely matching prior assessment exists -> reused, attacker model skipped
# --------------------------------------------------------------------------

def test_pipeline_reuses_prior_assessment_and_skips_attacker_model():
    # The Confluence doc content becomes the retrieval query, so seed the
    # store with a prior assessment whose text closely matches it.
    confluence_client = make_confluence_client()
    assessment_store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))

    # Seed with text engineered to overlap heavily with SAMPLE_STORAGE_HTML's
    # extracted content, so the hashing embedder scores it above threshold.
    assessment_store.add_assessment(
        customer_ref="cust_prior",
        use_case_summary=(
            "Data Sharing Overview This customer requests anonymized vehicle "
            "trajectory data Format CSV Frequency daily batch"
        ),
        risk_level="medium",
        risk_score=0.35,
    )

    draft_generator = DraftGenerator(llm_client=MockLLMClient())
    call_log: list[str] = []

    result = run_pipeline(
        customer_ref="cust_new",
        space_key="CUST_NEW",
        confluence_client=confluence_client,
        assessment_store=assessment_store,
        draft_generator=draft_generator,
        run_attacker_model=counting_attacker_model(call_log),
    )

    assert result.reused_prior_assessment is True
    assert len(call_log) == 0                        # attacker model was SKIPPED
    assert result.risk_score == 0.35                  # taken from the prior assessment
    assert result.draft.risk_level == "medium"         # classify_risk_level(0.35)


# --------------------------------------------------------------------------
# End-to-end sanity: everything downstream is consistent
# --------------------------------------------------------------------------

def test_pipeline_draft_is_internally_consistent():
    confluence_client = make_confluence_client()
    assessment_store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))
    draft_generator = DraftGenerator(llm_client=MockLLMClient())

    result = run_pipeline(
        customer_ref="cust_014",
        space_key="CUST014",
        confluence_client=confluence_client,
        assessment_store=assessment_store,
        draft_generator=draft_generator,
        run_attacker_model=lambda _use_case: 0.62,
    )

    # The deterministic risk-level guarantee (from draft_generator) must
    # hold end-to-end, not just in draft_generator's own unit tests.
    assert result.draft.risk_level == "high"
    assert result.draft.risk_score == 0.62
    assert len(result.draft.gdpr_considerations) >= 1
    assert "cust_014" in result.draft.as_markdown()
