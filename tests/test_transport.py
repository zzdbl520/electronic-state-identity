from __future__ import annotations

import numpy as np

from electronic_state_identity import (
    assignment,
    back_transport,
    edge_fidelity,
    plaquette,
    polar_transport,
    route_nonclosure,
)


def random_unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    trial = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(trial)
    phase = np.diag(r)
    phase = np.where(np.abs(phase) > 0.0, phase / np.abs(phase), 1.0)
    return q @ np.diag(phase.conj())


def test_polar_transport_is_unitary_and_reports_fidelity() -> None:
    overlap = np.diag([0.99, 0.95]).astype(complex)
    transport, singular_values = polar_transport(overlap)
    np.testing.assert_allclose(transport.conj().T @ transport, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(singular_values, [0.99, 0.95])
    metrics = edge_fidelity(overlap)
    assert metrics["sigma_min"] == 0.95
    assert np.isclose(metrics["leakage"], 1.0 - np.mean(np.array([0.99, 0.95]) ** 2))


def test_local_assignment_can_miss_continuous_route_difference() -> None:
    angle = 0.2
    first = np.eye(2)
    second = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    assert assignment(first)["permutation"] == assignment(second)["permutation"]
    assert route_nonclosure(first, second) > 0.0

    endpoint_operator = np.diag([1.0, -1.0])
    difference = np.linalg.norm(
        back_transport(endpoint_operator, first)
        - back_transport(endpoint_operator, second),
        ord="fro",
    )
    assert difference > 0.0


def test_plaquette_scalars_and_back_transport_are_gauge_invariant() -> None:
    rng = np.random.default_rng(20260921)
    overlaps = [
        np.eye(3) + 0.03 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        for _ in range(4)
    ]
    reference = plaquette(*overlaps)
    gauges = [random_unitary(rng, 3) for _ in range(4)]
    a, b, c, d = gauges
    transformed = (
        a.conj().T @ overlaps[0] @ b,
        b.conj().T @ overlaps[1] @ d,
        a.conj().T @ overlaps[2] @ c,
        c.conj().T @ overlaps[3] @ d,
    )
    checked = plaquette(*transformed)
    assert abs(reference["C_qp"] - checked["C_qp"]) < 1e-13
    np.testing.assert_allclose(
        reference["edge_singular_values"], checked["edge_singular_values"], atol=1e-13
    )

    operators = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
    operators = operators + operators.conj().transpose(0, 2, 1)
    ref_difference = np.linalg.norm(
        back_transport(operators, reference["route_qp"])
        - back_transport(operators, reference["route_pq"])
    )
    gauged_operators = np.asarray([d.conj().T @ item @ d for item in operators])
    checked_difference = np.linalg.norm(
        back_transport(gauged_operators, checked["route_qp"])
        - back_transport(gauged_operators, checked["route_pq"])
    )
    assert abs(ref_difference - checked_difference) < 1e-12


def test_normalized_nonclosure_is_bounded() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        first = rng.normal(size=(4, 4))
        second = rng.normal(size=(4, 4))
        assert 0.0 <= route_nonclosure(first, second) <= 1.0
