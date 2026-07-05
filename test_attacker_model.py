"""
Tests for attacker_model.py. Everything here operates on synthetic data
with fixed random seeds for determinism.

Run with: pytest test_attacker_model.py -v
"""

import numpy as np

from attacker_model import (
    AttackerModel,
    Ping,
    SpatialAnonymizer,
    TrajectoryReconstructor,
    evaluate_reidentification_risk,
    extract_features,
    generate_population,
)


def test_generate_population_shapes():
    individuals, trajectories = generate_population(n_individuals=5, n_days=3, seed=1)
    assert len(individuals) == 5
    assert len(trajectories) == 5
    for daily_trajectories in trajectories.values():
        assert len(daily_trajectories) == 3           # 3 days
        assert all(len(day) == 24 for day in daily_trajectories)  # 24 hourly pings/day


def test_generate_population_is_deterministic_given_seed():
    _, traj_a = generate_population(n_individuals=3, n_days=1, seed=42)
    _, traj_b = generate_population(n_individuals=3, n_days=1, seed=42)
    a = traj_a[0][0][0]
    b = traj_b[0][0][0]
    assert (a.x, a.y) == (b.x, b.y)


def test_spatial_anonymizer_snaps_to_grid():
    anonymizer = SpatialAnonymizer(cell_size=10)
    pings = [Ping(hour=0, x=13.2, y=27.9), Ping(hour=1, x=15.8, y=22.1)]

    anonymized = anonymizer.anonymize(pings)

    # Both points fall in the same 10-unit cell -> identical anonymized coords
    assert anonymized[0].x == anonymized[1].x
    assert anonymized[0].y == anonymized[1].y


def test_larger_cell_size_loses_more_precision():
    true_point = Ping(hour=0, x=13.2, y=27.9)
    fine = SpatialAnonymizer(cell_size=2).anonymize([true_point])[0]
    coarse = SpatialAnonymizer(cell_size=50).anonymize([true_point])[0]

    fine_error = abs(fine.x - true_point.x) + abs(fine.y - true_point.y)
    coarse_error = abs(coarse.x - true_point.x) + abs(coarse.y - true_point.y)
    assert coarse_error >= fine_error


def test_reconstructor_respects_speed_constraint():
    """Reconstructed consecutive points shouldn't imply impossible speeds."""
    _, trajectories = generate_population(n_individuals=1, n_days=1, seed=3)
    pings = trajectories[0][0]
    anonymized = SpatialAnonymizer(cell_size=20).anonymize(pings)

    reconstructor = TrajectoryReconstructor(cell_size=20, seed=3)
    path = reconstructor.reconstruct(anonymized)

    assert len(path) == len(pings)
    for i in range(1, len(path)):
        dt = max(anonymized[i].hour - anonymized[i - 1].hour, 1)
        dist = np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        # allow the same soft penalty margin used internally, not a hard wall
        assert dist / dt < 60


def test_extract_features_shape_is_fixed_length():
    _, trajectories = generate_population(n_individuals=1, n_days=1, seed=5)
    pings = trajectories[0][0]
    features = extract_features(pings)
    # 2 (home) + 2 (work) + 25 (5x5 histogram) = 29
    assert features.shape == (29,)


def test_extract_features_histogram_sums_to_one():
    _, trajectories = generate_population(n_individuals=1, n_days=1, seed=5)
    pings = trajectories[0][0]
    features = extract_features(pings)
    histogram = features[4:]
    assert np.isclose(histogram.sum(), 1.0)


def test_attacker_model_trains_and_predicts():
    _, trajectories = generate_population(n_individuals=4, n_days=4, seed=7)
    X, y = [], []
    for identity, days in trajectories.items():
        for pings in days:
            X.append(extract_features(pings))
            y.append(identity)
    X, y = np.array(X), np.array(y)

    model = AttackerModel(seed=7).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

    accuracy = model.reidentification_accuracy(X, y)
    assert 0.0 <= accuracy <= 1.0


def test_reidentification_risk_decreases_with_stronger_anonymization():
    """The core claim of the whole module: more anonymization -> lower risk."""
    weak_anonymization_risk = evaluate_reidentification_risk(cell_size=2, n_individuals=30, n_days=6, seed=0)
    strong_anonymization_risk = evaluate_reidentification_risk(cell_size=80, n_individuals=30, n_days=6, seed=0)

    assert strong_anonymization_risk < weak_anonymization_risk


def test_reidentification_risk_is_a_valid_probability():
    risk = evaluate_reidentification_risk(cell_size=15, n_individuals=20, n_days=4, seed=2)
    assert 0.0 <= risk <= 1.0
