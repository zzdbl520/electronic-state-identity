#!/usr/bin/env python3
r"""Numerical validation of mixed variational curvature.

The script tests the plaquette construction

    K_qp = T_AB T_BD,     K_pq = T_AC T_CD,
    W_qp = K_qp K_pq^\dagger,
    C_qp = ||W_qp-I||_F/(2 sqrt(m)),

where T_XY is the unitary polar factor of the cross-overlap M_XY.
It uses only NumPy and synthetic overlap matrices, so the algebra can be
validated independently of an electronic-structure implementation.

This is a mathematical unit-test layer.  It does not replace validation of
the many-electron overlap engine or molecule-level negative/positive controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray


@dataclass
class PlaquetteMetrics:
    """Gauge-invariant scalar diagnostics and matrix-valued transports."""

    curvature: float
    curvature_from_paths: float
    eta_edge: float
    lambda_edge: float
    holonomy_phases: Array
    k_q_to_p: Array
    k_p_to_q: Array
    holonomy: Array
    polar_unitarity_defect: float


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def dagger(matrix: Array) -> Array:
    return matrix.conj().T


def frobenius(matrix: Array) -> float:
    array = np.asarray(matrix)
    if array.ndim == 0:
        return float(np.abs(array))
    return float(np.linalg.norm(array.ravel()))


def polar_transport(overlap: Array) -> tuple[Array, Array, Array]:
    """Return unitary polar factor T, positive factor H, and singular values.

    For M = U diag(s) V^dagger, this uses

        T = U V^dagger,    H = V diag(s) V^dagger,

    so M = T H.  The SVD gives a well-defined numerical convention even when
    M is nearly singular, while eta_edge reports when the transport should not
    be interpreted as a well-conditioned common state manifold.
    """

    overlap = np.asarray(overlap, dtype=np.complex128)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("Each cross-overlap must be a square m x m matrix.")
    u, singular_values, vh = np.linalg.svd(overlap, full_matrices=False)
    transport = u @ vh
    v = dagger(vh)
    positive = (v * singular_values) @ vh
    return transport, positive, singular_values


def curvature_from_paths(k_q_to_p: Array, k_p_to_q: Array) -> tuple[float, float, Array]:
    """Compute both equivalent curvature formulas and the holonomy."""

    if k_q_to_p.shape != k_p_to_q.shape or k_q_to_p.shape[0] != k_q_to_p.shape[1]:
        raise ValueError("The two path transports must have the same square shape.")
    m = k_q_to_p.shape[0]
    holonomy = k_q_to_p @ dagger(k_p_to_q)
    c_holonomy = frobenius(holonomy - np.eye(m)) / (2.0 * math.sqrt(m))
    c_paths = frobenius(k_q_to_p - k_p_to_q) / (2.0 * math.sqrt(m))
    return c_holonomy, c_paths, holonomy


def plaquette_metrics(
    m_ab: Array,
    m_bd: Array,
    m_ac: Array,
    m_cd: Array,
) -> PlaquetteMetrics:
    """Evaluate the complete mixed-curvature diagnostic on one plaquette."""

    overlaps = [
        np.asarray(x, dtype=np.complex128) for x in (m_ab, m_bd, m_ac, m_cd)
    ]
    shapes = {x.shape for x in overlaps}
    if len(shapes) != 1 or overlaps[0].ndim != 2 or overlaps[0].shape[0] != overlaps[0].shape[1]:
        raise ValueError("All four cross-overlaps must have the same square shape.")

    decompositions = [polar_transport(x) for x in overlaps]
    transports = [x[0] for x in decompositions]
    positives = [x[1] for x in decompositions]
    singular_values = [x[2] for x in decompositions]
    m = overlaps[0].shape[0]

    k_q_to_p = transports[0] @ transports[1]
    k_p_to_q = transports[2] @ transports[3]
    curvature, curvature_from_path_difference, holonomy = curvature_from_paths(
        k_q_to_p, k_p_to_q
    )

    eta_edge = min(float(np.min(s)) for s in singular_values)
    lambda_edge = max(
        frobenius(h - np.eye(m)) / math.sqrt(m) for h in positives
    )
    unitarity_defect = max(
        frobenius(dagger(t) @ t - np.eye(m)) for t in transports
    )
    phases = np.sort(np.angle(np.linalg.eigvals(holonomy)))

    return PlaquetteMetrics(
        curvature=curvature,
        curvature_from_paths=curvature_from_path_difference,
        eta_edge=eta_edge,
        lambda_edge=lambda_edge,
        holonomy_phases=phases,
        k_q_to_p=k_q_to_p,
        k_p_to_q=k_p_to_q,
        holonomy=holonomy,
        polar_unitarity_defect=unitarity_defect,
    )


def random_unitary(m: int, rng: np.random.Generator) -> Array:
    raw = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(phases.conj())


def random_isometry(rows: int, columns: int, rng: np.random.Generator) -> Array:
    """Return a complex rows x columns matrix with orthonormal columns."""

    if rows < columns:
        raise ValueError("An isometry requires rows >= columns.")
    raw = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    q, _r = np.linalg.qr(raw)
    return q[:, :columns]


def unitary_from_hermitian(generator: Array, step: float) -> Array:
    """Compute exp(i*step*generator) for a Hermitian generator."""

    generator = np.asarray(generator, dtype=np.complex128)
    if not np.allclose(generator, dagger(generator), atol=1.0e-13, rtol=0.0):
        raise ValueError("Generator must be Hermitian.")
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    return (eigenvectors * np.exp(1j * step * eigenvalues)) @ dagger(eigenvectors)


def positive_factor(
    singular_values: Iterable[float], rng: np.random.Generator | None = None
) -> Array:
    values = np.asarray(tuple(singular_values), dtype=float)
    if np.any(values < 0.0):
        raise ValueError("Polar singular values must be nonnegative.")
    if rng is None:
        return np.diag(values).astype(np.complex128)
    v = random_unitary(values.size, rng)
    return (v * values) @ dagger(v)


def determinant_configurations(
    number_of_orbitals: int, number_alpha: int, number_beta: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    alpha = list(combinations(range(number_of_orbitals), number_alpha))
    beta = list(combinations(range(number_of_orbitals), number_beta))
    return [(a, b) for a in alpha for b in beta]


def determinant_cross_overlap_matrix(
    orbital_cross_overlap: Array,
    configurations_x: list[tuple[tuple[int, ...], tuple[int, ...]]],
    configurations_y: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> Array:
    """Build determinant overlaps det(S_alpha) det(S_beta)."""

    result = np.empty((len(configurations_x), len(configurations_y)), dtype=np.complex128)
    for i, (alpha_x, beta_x) in enumerate(configurations_x):
        for j, (alpha_y, beta_y) in enumerate(configurations_y):
            alpha_block = orbital_cross_overlap[np.ix_(alpha_x, alpha_y)]
            beta_block = orbital_cross_overlap[np.ix_(beta_x, beta_y)]
            result[i, j] = np.linalg.det(alpha_block) * np.linalg.det(beta_block)
    return result


def fock_embedding(
    orbitals_in_ambient_basis: Array,
    local_configurations: list[tuple[tuple[int, ...], tuple[int, ...]]],
    number_alpha: int,
    number_beta: int,
) -> Array:
    """Expand local Slater determinants in a common orthonormal Fock basis."""

    ambient_dimension = orbitals_in_ambient_basis.shape[0]
    ambient_configurations = determinant_configurations(
        ambient_dimension, number_alpha, number_beta
    )
    embedding = np.empty(
        (len(ambient_configurations), len(local_configurations)), dtype=np.complex128
    )
    for row, (ambient_alpha, ambient_beta) in enumerate(ambient_configurations):
        for column, (local_alpha, local_beta) in enumerate(local_configurations):
            alpha_minor = orbitals_in_ambient_basis[np.ix_(ambient_alpha, local_alpha)]
            beta_minor = orbitals_in_ambient_basis[np.ix_(ambient_beta, local_beta)]
            embedding[row, column] = (
                np.linalg.det(alpha_minor) * np.linalg.det(beta_minor)
            )
    return embedding


def gauge_transform_edges(
    overlaps: tuple[Array, Array, Array, Array],
    gauges: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array, Array, Array]:
    """Apply independent corner gauges G_A, G_B, G_C, G_D."""

    m_ab, m_bd, m_ac, m_cd = overlaps
    g_a, g_b, g_c, g_d = gauges
    return (
        dagger(g_a) @ m_ab @ g_b,
        dagger(g_b) @ m_bd @ g_d,
        dagger(g_a) @ m_ac @ g_c,
        dagger(g_c) @ m_cd @ g_d,
    )


def noncommuting_edges(delta_q: float, delta_p: float) -> tuple[Array, Array, Array, Array]:
    """Constant non-Abelian connection sampled around one plaquette."""

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    h_q = 0.5 * sigma_x
    h_p = 0.5 * sigma_z
    t_q = unitary_from_hermitian(h_q, delta_q)
    t_p = unitary_from_hermitian(h_p, delta_p)
    return t_q, t_p, t_p, t_q


def assert_close(actual: float | Array, expected: float | Array, tolerance: float, label: str) -> None:
    difference = frobenius(np.asarray(actual) - np.asarray(expected))
    if difference > tolerance:
        raise AssertionError(
            f"{label}: difference {difference:.3e} exceeds tolerance {tolerance:.3e}"
        )


def check_metric_identities(metrics: PlaquetteMetrics, tolerance: float) -> None:
    assert_close(
        metrics.curvature,
        metrics.curvature_from_paths,
        tolerance,
        "two curvature formulas",
    )
    if not (-tolerance <= metrics.curvature <= 1.0 + tolerance):
        raise AssertionError(f"curvature bound violated: C={metrics.curvature:.16g}")
    if metrics.polar_unitarity_defect > 20.0 * tolerance:
        raise AssertionError(
            "polar transport is not unitary: "
            f"defect={metrics.polar_unitarity_defect:.3e}"
        )


def test_identity(tolerance: float, _: np.random.Generator) -> str:
    identity = np.eye(3, dtype=np.complex128)
    metrics = plaquette_metrics(identity, identity, identity, identity)
    check_metric_identities(metrics, tolerance)
    assert_close(metrics.curvature, 0.0, tolerance, "identity curvature")
    assert_close(metrics.eta_edge, 1.0, tolerance, "identity eta")
    assert_close(metrics.lambda_edge, 0.0, tolerance, "identity leakage")
    return "C=0, eta_edge=1, lambda_edge=0"


def test_phase_and_root_permutation(tolerance: float, _: np.random.Generator) -> str:
    permutation = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    phases = [0.17, -1.3, 2.2]
    corner_frames = (
        np.eye(3, dtype=np.complex128),
        permutation @ np.diag(np.exp(1j * np.asarray(phases))),
        np.diag(np.exp(-1j * np.asarray(phases))) @ dagger(permutation),
        permutation @ np.diag(np.exp(0.5j * np.asarray(phases))),
    )
    g_a, g_b, g_c, g_d = corner_frames
    overlaps = (
        dagger(g_a) @ g_b,
        dagger(g_b) @ g_d,
        dagger(g_a) @ g_c,
        dagger(g_c) @ g_d,
    )
    metrics = plaquette_metrics(*overlaps)
    check_metric_identities(metrics, tolerance)
    assert_close(metrics.curvature, 0.0, tolerance, "phase/permutation curvature")
    return "independent phases and root relabeling leave C=0"


def test_general_pure_gauge(tolerance: float, rng: np.random.Generator) -> str:
    frames = tuple(random_unitary(4, rng) for _ in range(4))
    g_a, g_b, g_c, g_d = frames
    overlaps = (
        dagger(g_a) @ g_b,
        dagger(g_b) @ g_d,
        dagger(g_a) @ g_c,
        dagger(g_c) @ g_d,
    )
    metrics = plaquette_metrics(*overlaps)
    check_metric_identities(metrics, tolerance)
    assert_close(metrics.curvature, 0.0, 20.0 * tolerance, "pure-gauge curvature")
    return "arbitrary U(4) corner frames close exactly"


def make_nontrivial_overlaps(rng: np.random.Generator) -> tuple[Array, Array, Array, Array]:
    transports = noncommuting_edges(0.73, -0.51)
    spectra = (
        (0.98, 0.83),
        (0.91, 0.76),
        (0.95, 0.88),
        (0.89, 0.79),
    )
    return tuple(
        transport @ positive_factor(values, rng)
        for transport, values in zip(transports, spectra, strict=True)
    )


def test_gauge_invariance_nonzero_curvature(tolerance: float, rng: np.random.Generator) -> str:
    overlaps = make_nontrivial_overlaps(rng)
    reference = plaquette_metrics(*overlaps)
    gauges = tuple(random_unitary(2, rng) for _ in range(4))
    transformed = plaquette_metrics(*gauge_transform_edges(overlaps, gauges))
    check_metric_identities(reference, tolerance)
    check_metric_identities(transformed, tolerance)
    assert_close(transformed.curvature, reference.curvature, 30.0 * tolerance, "gauge C")
    assert_close(transformed.eta_edge, reference.eta_edge, 30.0 * tolerance, "gauge eta")
    assert_close(
        transformed.lambda_edge,
        reference.lambda_edge,
        30.0 * tolerance,
        "gauge leakage",
    )
    assert_close(
        transformed.holonomy_phases,
        reference.holonomy_phases,
        100.0 * tolerance,
        "gauge holonomy phases",
    )
    g_a, _g_b, _g_c, g_d = gauges
    assert_close(
        transformed.k_q_to_p,
        dagger(g_a) @ reference.k_q_to_p @ g_d,
        100.0 * tolerance,
        "K_q_to_p gauge covariance",
    )
    assert_close(
        transformed.k_p_to_q,
        dagger(g_a) @ reference.k_p_to_q @ g_d,
        100.0 * tolerance,
        "K_p_to_q gauge covariance",
    )
    assert_close(
        transformed.holonomy,
        dagger(g_a) @ reference.holonomy @ g_a,
        100.0 * tolerance,
        "holonomy gauge covariance",
    )
    return (
        f"C={reference.curvature:.8f}; K paths transform endpoint-covariantly "
        "and W by similarity"
    )


def test_leakage_without_curvature(tolerance: float, _: np.random.Generator) -> str:
    scales = (0.94, 0.81, 0.88, 0.72)
    overlaps = tuple(scale * np.eye(2, dtype=np.complex128) for scale in scales)
    metrics = plaquette_metrics(*overlaps)
    check_metric_identities(metrics, tolerance)
    assert_close(metrics.curvature, 0.0, tolerance, "leakage-only curvature")
    assert_close(metrics.eta_edge, min(scales), tolerance, "leakage-only eta")
    assert_close(metrics.lambda_edge, 1.0 - min(scales), tolerance, "leakage-only lambda")
    return f"C=0 despite lambda_edge={metrics.lambda_edge:.2f}; leakage is a separate diagnostic"


def test_rank_loss_flag(tolerance: float, _: np.random.Generator) -> str:
    identity = np.eye(2, dtype=np.complex128)
    rank_lost = np.diag([1.0, 1.0e-14]).astype(np.complex128)
    metrics = plaquette_metrics(rank_lost, identity, identity, identity)
    check_metric_identities(metrics, tolerance)
    if not metrics.eta_edge < 1.0e-12:
        raise AssertionError(f"rank loss was not detected: eta_edge={metrics.eta_edge:.3e}")
    return f"eta_edge={metrics.eta_edge:.1e} flags ill-conditioned transport"


def test_orientation_reversal(tolerance: float, _: np.random.Generator) -> str:
    metrics = plaquette_metrics(*noncommuting_edges(0.63, 0.47))
    reversed_c, reversed_path_c, reversed_w = curvature_from_paths(
        metrics.k_p_to_q, metrics.k_q_to_p
    )
    check_metric_identities(metrics, tolerance)
    assert_close(reversed_c, metrics.curvature, 20.0 * tolerance, "orientation-reversed C")
    assert_close(reversed_path_c, metrics.curvature, 20.0 * tolerance, "reversed path formula")
    assert_close(reversed_w, dagger(metrics.holonomy), 20.0 * tolerance, "reversed holonomy")
    return "orientation reversal maps W to W^dagger and leaves scalar C unchanged"


def test_many_electron_overlap_contraction(tolerance: float, rng: np.random.Generator) -> str:
    """Cross-check determinant/CI contraction against a direct Fock expansion."""

    ambient_dimension = 6
    number_ao = 4
    number_alpha = 2
    number_beta = 1
    number_states = 2

    ao_frame_x = random_isometry(ambient_dimension, number_ao, rng)
    ao_frame_y = random_isometry(ambient_dimension, number_ao, rng)
    ao_cross_overlap = dagger(ao_frame_x) @ ao_frame_y
    orbital_coefficients_x = random_unitary(number_ao, rng)
    orbital_coefficients_y = random_unitary(number_ao, rng)
    orbitals_x = ao_frame_x @ orbital_coefficients_x
    orbitals_y = ao_frame_y @ orbital_coefficients_y

    active_orbital_overlap = (
        dagger(orbital_coefficients_x)
        @ ao_cross_overlap
        @ orbital_coefficients_y
    )
    configurations = determinant_configurations(number_ao, number_alpha, number_beta)
    determinant_overlap = determinant_cross_overlap_matrix(
        active_orbital_overlap, configurations, configurations
    )

    ci_x = random_isometry(len(configurations), number_states, rng)
    ci_y = random_isometry(len(configurations), number_states, rng)
    contracted_overlap = dagger(ci_x) @ determinant_overlap @ ci_y

    embedding_x = fock_embedding(
        orbitals_x, configurations, number_alpha, number_beta
    )
    embedding_y = fock_embedding(
        orbitals_y, configurations, number_alpha, number_beta
    )
    direct_states_x = embedding_x @ ci_x
    direct_states_y = embedding_y @ ci_y
    direct_overlap = dagger(direct_states_x) @ direct_states_y

    assert_close(
        contracted_overlap,
        direct_overlap,
        200.0 * tolerance,
        "many-electron cross-overlap",
    )
    assert_close(
        dagger(embedding_x) @ embedding_x,
        np.eye(len(configurations)),
        200.0 * tolerance,
        "X determinant embedding orthonormality",
    )
    assert_close(
        dagger(embedding_y) @ embedding_y,
        np.eye(len(configurations)),
        200.0 * tolerance,
        "Y determinant embedding orthonormality",
    )

    naive_ci_dot_product = dagger(ci_x) @ ci_y
    naive_error = frobenius(naive_ci_dot_product - direct_overlap)
    if naive_error < 1.0e-2:
        raise AssertionError(
            "random example did not clearly distinguish the physical overlap "
            "from a CI-coefficient dot product"
        )
    contraction_error = frobenius(contracted_overlap - direct_overlap)
    return (
        f"direct Fock-space error={contraction_error:.2e}; "
        f"naive CI-dot-product error={naive_error:.3f}"
    )


def scaling_data() -> tuple[list[dict[str, float]], float, float, float]:
    equal_steps = np.asarray([0.20, 0.10, 0.05, 0.025, 0.0125])
    equal_curvatures = np.asarray(
        [plaquette_metrics(*noncommuting_edges(h, h)).curvature for h in equal_steps]
    )
    equal_slope = float(np.polyfit(np.log(equal_steps), np.log(equal_curvatures), 1)[0])

    fixed_p = 0.15
    q_steps = np.asarray([0.08, 0.04, 0.02, 0.01, 0.005])
    fixed_curvatures = np.asarray(
        [
            plaquette_metrics(*noncommuting_edges(delta_q, fixed_p)).curvature
            for delta_q in q_steps
        ]
    )
    fixed_slope = float(np.polyfit(np.log(q_steps), np.log(fixed_curvatures), 1)[0])

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    a_q = 0.5j * sigma_x
    a_p = 0.5j * sigma_z
    curvature_tensor = a_q @ a_p - a_p @ a_q
    predicted_coefficient = frobenius(curvature_tensor) / (2.0 * math.sqrt(2.0))
    observed_coefficient = float(equal_curvatures[-1] / equal_steps[-1] ** 2)

    rows = [
        {"delta_q": float(h), "delta_p": float(h), "curvature": float(c)}
        for h, c in zip(equal_steps, equal_curvatures, strict=True)
    ]
    rows.extend(
        {
            "delta_q": float(delta_q),
            "delta_p": fixed_p,
            "curvature": float(c),
        }
        for delta_q, c in zip(q_steps, fixed_curvatures, strict=True)
    )
    return rows, equal_slope, fixed_slope, observed_coefficient / predicted_coefficient


def test_continuous_limit(tolerance: float, _: np.random.Generator) -> str:
    _rows, equal_slope, fixed_slope, coefficient_ratio = scaling_data()
    if not 1.98 < equal_slope < 2.02:
        raise AssertionError(f"equal-step log-log slope is {equal_slope:.8f}, expected 2")
    if not 0.98 < fixed_slope < 1.02:
        raise AssertionError(f"fixed-p log-log slope is {fixed_slope:.8f}, expected 1")
    if not abs(coefficient_ratio - 1.0) < 2.0e-4:
        raise AssertionError(
            f"finite-difference/commutator coefficient ratio is {coefficient_ratio:.8f}"
        )
    return (
        f"slope(h,h)={equal_slope:.6f}, slope(h,p0)={fixed_slope:.6f}, "
        f"coefficient ratio={coefficient_ratio:.8f}"
    )


def test_random_bounds_and_equivalence(tolerance: float, rng: np.random.Generator) -> str:
    largest_c = 0.0
    largest_formula_error = 0.0
    for _ in range(200):
        transports = tuple(random_unitary(3, rng) for _ in range(4))
        overlaps = tuple(
            transport @ positive_factor(rng.uniform(0.2, 1.0, size=3), rng)
            for transport in transports
        )
        metrics = plaquette_metrics(*overlaps)
        largest_c = max(largest_c, metrics.curvature)
        largest_formula_error = max(
            largest_formula_error,
            abs(metrics.curvature - metrics.curvature_from_paths),
        )
        check_metric_identities(metrics, 100.0 * tolerance)
    return (
        f"200 random plaquettes: max C={largest_c:.8f}, "
        f"max formula error={largest_formula_error:.2e}"
    )


TESTS: tuple[tuple[str, Callable[[float, np.random.Generator], str]], ...] = (
    ("identity plaquette", test_identity),
    ("phase and root permutation", test_phase_and_root_permutation),
    ("general pure gauge", test_general_pure_gauge),
    ("nonzero-curvature gauge invariance", test_gauge_invariance_nonzero_curvature),
    ("leakage without curvature", test_leakage_without_curvature),
    ("rank-loss warning", test_rank_loss_flag),
    ("orientation reversal", test_orientation_reversal),
    ("many-electron overlap contraction", test_many_electron_overlap_contraction),
    ("continuous-limit scaling", test_continuous_limit),
    ("random bounds and formula equivalence", test_random_bounds_and_equivalence),
)


def run_suite(tolerance: float, seed: int) -> tuple[list[TestResult], dict[str, object]]:
    rng = np.random.default_rng(seed)
    results: list[TestResult] = []
    for name, test in TESTS:
        try:
            detail = test(tolerance, rng)
            results.append(TestResult(name=name, passed=True, detail=detail))
        except Exception as exc:  # keep the report complete if one test fails
            results.append(TestResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"))

    rows, equal_slope, fixed_slope, coefficient_ratio = scaling_data()
    diagnostics: dict[str, object] = {
        "tolerance": tolerance,
        "seed": seed,
        "numpy_version": np.__version__,
        "scaling_rows": rows,
        "equal_step_slope": equal_slope,
        "fixed_delta_p_slope": fixed_slope,
        "coefficient_ratio": coefficient_ratio,
    }
    return results, diagnostics


def json_safe_payload(results: list[TestResult], diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "passed": all(result.passed for result in results),
        "tests": [asdict(result) for result in results],
        "diagnostics": diagnostics,
    }


def print_report(results: list[TestResult], diagnostics: dict[str, object]) -> None:
    print("Mixed variational curvature: numerical validation")
    print("=" * 57)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    passed = sum(result.passed for result in results)
    print()
    print(f"Summary: {passed}/{len(results)} tests passed")
    print(
        "Continuous limit: "
        f"slope(h,h)={diagnostics['equal_step_slope']:.8f}; "
        f"slope(h,p0)={diagnostics['fixed_delta_p_slope']:.8f}; "
        f"coefficient ratio={diagnostics['coefficient_ratio']:.8f}"
    )


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optionally save the complete machine-readable test report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    results, diagnostics = run_suite(args.tolerance, args.seed)
    print_report(results, diagnostics)
    payload = json_safe_payload(results, diagnostics)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Machine-readable report: {args.json}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
