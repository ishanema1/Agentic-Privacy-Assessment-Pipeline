"""
pipeline.py

Orchestrates the four implemented pipeline steps into a single callable:

  1. Confluence context gathering   (confluence_client.ConfluenceClient)
  2. Prior assessment search        (vector_store.PriorAssessmentStore)
  3. Attacker-model risk scoring    (attacker_model, only when no reusable
                                      prior assessment is found)
  4. Draft generation (LLM)         (draft_generator.DraftGenerator)

In production this sequence runs as a LangGraph state machine (see the
architecture diagram in README.md) with a fact-check pass and a human
review checkpoint after step 4 — neither is in this repo yet. This module
expresses the same call sequence as plain, testable Python: a four-step
linear pipeline with one conditional branch doesn't need a graph engine to
be understood, and keeping it framework-free here makes the domain logic
(the actual point of this repo) easier to read and test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from confluence_client import ConfluenceClient, CustomerDoc
from draft_generator import DraftGenerator, PrivacyAssessmentDraft
from vector_store import PriorAssessmentStore

# Bounds how much gathered context goes into the similarity-search query,
# consistent with the "keep it retrieval, not a wall of text" intent of
# the vector_store design.
MAX_QUERY_CHARS = 2000


@dataclass
class PipelineResult:
    customer_ref: str
    customer_docs: list[CustomerDoc]
    reused_prior_assessment: bool
    risk_score: float
    draft: PrivacyAssessmentDraft


def run_pipeline(
    customer_ref: str,
    space_key: str,
    confluence_client: ConfluenceClient,
    assessment_store: PriorAssessmentStore,
    draft_generator: DraftGenerator,
    run_attacker_model: Callable[[str], float],
) -> PipelineResult:
    """
    Run the full data-sharing assessment pipeline for one customer.

    `run_attacker_model` is injected rather than called directly from this
    module because a fresh attacker-model evaluation is comparatively
    expensive (it trains a classifier — see attacker_model.py) and this
    pipeline should only pay that cost when step 2 doesn't find a
    sufficiently similar prior assessment to reuse instead. Callers wire
    in whatever attacker-model invocation makes sense for their context
    (e.g. `lambda _use_case: evaluate_reidentification_risk(cell_size=...)`).
    """
    # Step 1: context gathering
    docs = confluence_client.get_customer_docs(space_key=space_key)
    doc_texts = [d.as_context_block() for d in docs]
    use_case_summary = "\n".join(doc_texts)[:MAX_QUERY_CHARS]

    # Step 2/3: reuse a prior assessment if one is similar enough, otherwise
    # run a fresh attacker-model evaluation.
    match = assessment_store.find_reusable_assessment(use_case_summary)
    if match and match[0].risk_score is not None:
        prior_assessment, _similarity = match
        risk_score = prior_assessment.risk_score
        prior_summary = (
            f"{prior_assessment.customer_ref}: {prior_assessment.use_case_summary} "
            f"(risk_level: {prior_assessment.risk_level})"
        )
        reused = True
    else:
        risk_score = run_attacker_model(use_case_summary)
        prior_summary = None
        reused = False

    # Step 4: draft generation
    draft = draft_generator.generate(
        customer_ref=customer_ref,
        customer_docs=doc_texts,
        risk_score=risk_score,
        prior_assessment_summary=prior_summary,
    )

    return PipelineResult(
        customer_ref=customer_ref,
        customer_docs=docs,
        reused_prior_assessment=reused,
        risk_score=risk_score,
        draft=draft,
    )
