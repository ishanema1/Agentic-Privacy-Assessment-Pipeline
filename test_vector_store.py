"""
Tests for vector_store.py.

Run with: pytest test_vector_store.py -v
"""

import numpy as np

from vector_store import (
    HashingEmbedder,
    PriorAssessmentStore,
    REUSE_SIMILARITY_THRESHOLD,
)


def test_hashing_embedder_is_normalized():
    embedder = HashingEmbedder(dim=64)
    vec = embedder.embed("anonymized vehicle trajectory data")
    assert vec.shape == (64,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_hashing_embedder_empty_string_does_not_crash():
    embedder = HashingEmbedder(dim=64)
    vec = embedder.embed("")
    assert vec.shape == (64,)
    assert np.linalg.norm(vec) == 0.0


def test_identical_text_has_similarity_one():
    store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))
    store.add_assessment("cust_001", "anonymized fleet trajectory data for routing", "medium")

    results = store.search("anonymized fleet trajectory data for routing", top_k=1)
    assert len(results) == 1
    _, score = results[0]
    assert np.isclose(score, 1.0, atol=1e-5)


def test_search_ranks_more_similar_case_first():
    store = PriorAssessmentStore(embedder=HashingEmbedder(dim=256))
    store.add_assessment(
        "cust_001",
        "automotive oem anonymized vehicle trajectory data for fleet routing analytics",
        "medium",
    )
    store.add_assessment(
        "cust_002",
        "insurance provider aggregated driving behavior scores for pricing",
        "low",
    )

    results = store.search(
        "automotive manufacturer anonymized trajectory data for vehicle fleet routing",
        top_k=2,
    )

    assert results[0][0].customer_ref == "cust_001"
    assert results[0][1] > results[1][1]


def test_find_reusable_assessment_returns_none_when_below_threshold():
    store = PriorAssessmentStore(embedder=HashingEmbedder(dim=256))
    store.add_assessment("cust_001", "completely unrelated topic about weather forecasting", "low")

    match = store.find_reusable_assessment("anonymized vehicle trajectory data for routing")
    assert match is None


def test_find_reusable_assessment_returns_match_above_threshold():
    store = PriorAssessmentStore(embedder=HashingEmbedder(dim=256))
    text = "anonymized vehicle trajectory data for fleet routing analytics"
    store.add_assessment("cust_001", text, "medium")

    match = store.find_reusable_assessment(text)  # identical text -> similarity 1.0
    assert match is not None
    assessment, score = match
    assert assessment.customer_ref == "cust_001"
    assert score >= REUSE_SIMILARITY_THRESHOLD


def test_search_on_empty_store_returns_empty_list():
    store = PriorAssessmentStore(embedder=HashingEmbedder(dim=64))
    assert store.search("anything") == []
    assert store.find_reusable_assessment("anything") is None
