#!/usr/bin/env python3
"""Exact determinant-overlap utilities for the CPDC multiring pilot.

The routines are deliberately molecule agnostic.  A saved solution is a
mapping containing ``metadata``, ``mo_coeff``, and ``ci_roots``; metadata must
provide ``ncore``, ``ncas``, ``nroots``, and the active alpha/beta counts.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


def active_strings(norb: int, nelec: int) -> Tuple[Tuple[int, ...], ...]:
    """Return occupied-orbital tuples in PySCF's increasing-bitstring order."""
    combinations = list(itertools.combinations(range(int(norb)), int(nelec)))
    combinations.sort(key=lambda item: sum(1 << orbital for orbital in item))
    return tuple(combinations)


def determinant_kernel_rectangular(
    orbital_overlap: np.ndarray,
    ncore_left: int,
    configurations_left: Sequence[Tuple[int, ...]],
    ncore_right: int,
    configurations_right: Sequence[Tuple[int, ...]],
) -> np.ndarray:
    """Overlap kernel between two Slater-determinant lists."""
    occupied_left = [
        tuple(range(int(ncore_left)))
        + tuple(int(ncore_left) + orbital for orbital in configuration)
        for configuration in configurations_left
    ]
    occupied_right = [
        tuple(range(int(ncore_right)))
        + tuple(int(ncore_right) + orbital for orbital in configuration)
        for configuration in configurations_right
    ]
    if occupied_left and occupied_right:
        if len(occupied_left[0]) != len(occupied_right[0]):
            raise RuntimeError("determinant overlap has unequal electron counts")
    kernel = np.empty(
        (len(occupied_left), len(occupied_right)), dtype=np.complex128
    )
    for left, occ_left in enumerate(occupied_left):
        for right, occ_right in enumerate(occupied_right):
            kernel[left, right] = np.linalg.det(
                orbital_overlap[np.ix_(occ_left, occ_right)]
            )
    return kernel


def active_electron_tuple(metadata: Mapping[str, Any]) -> Tuple[int, int]:
    if "nact_alpha" in metadata and "nact_beta" in metadata:
        return int(metadata["nact_alpha"]), int(metadata["nact_beta"])
    nelecas = int(metadata["nelecas"])
    spin = int(metadata.get("spin_2S", 0))
    return (nelecas + spin) // 2, (nelecas - spin) // 2


def contract_ci_overlap(
    ci_left: np.ndarray,
    ci_right: np.ndarray,
    alpha_kernel: np.ndarray,
    beta_kernel: np.ndarray,
) -> complex:
    return np.einsum(
        "ab,ac,bd,cd->",
        ci_left.conj(),
        alpha_kernel,
        beta_kernel,
        ci_right,
        optimize=True,
    )


def exact_overlap_matrix(sol_a, sol_b, mol_a, mol_b) -> np.ndarray:
    """Exact many-electron overlaps between two CASSCF root sets."""
    try:
        from pyscf import gto
    except ImportError as exc:
        raise RuntimeError("PySCF is required for molecular overlaps") from exc

    metadata_a = sol_a["metadata"]
    metadata_b = sol_b["metadata"]
    ncore_a = int(metadata_a["ncore"])
    ncore_b = int(metadata_b["ncore"])
    ncas_a = int(metadata_a["ncas"])
    ncas_b = int(metadata_b["ncas"])
    nalpha_a, nbeta_a = active_electron_tuple(metadata_a)
    nalpha_b, nbeta_b = active_electron_tuple(metadata_b)
    nroots_a = int(metadata_a["nroots"])
    nroots_b = int(metadata_b["nroots"])

    s_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    coeff_a = sol_a["mo_coeff"][:, : ncore_a + ncas_a]
    coeff_b = sol_b["mo_coeff"][:, : ncore_b + ncas_b]
    orbital_overlap = coeff_a.conj().T @ s_ab @ coeff_b

    alpha_a = active_strings(ncas_a, nalpha_a)
    beta_a = active_strings(ncas_a, nbeta_a)
    alpha_b = active_strings(ncas_b, nalpha_b)
    beta_b = active_strings(ncas_b, nbeta_b)
    alpha_kernel = determinant_kernel_rectangular(
        orbital_overlap, ncore_a, alpha_a, ncore_b, alpha_b
    )
    beta_kernel = determinant_kernel_rectangular(
        orbital_overlap, ncore_a, beta_a, ncore_b, beta_b
    )

    result = np.empty((nroots_a, nroots_b), dtype=np.complex128)
    for left in range(nroots_a):
        ci_left = np.asarray(sol_a["ci_roots"][left]).reshape(
            len(alpha_a), len(beta_a)
        )
        for right in range(nroots_b):
            ci_right = np.asarray(sol_b["ci_roots"][right]).reshape(
                len(alpha_b), len(beta_b)
            )
            result[left, right] = contract_ci_overlap(
                ci_left, ci_right, alpha_kernel, beta_kernel
            )
    return result


_PERMUTATIONS: Dict[int, np.ndarray] = {}


def assignment(matrix: np.ndarray) -> Dict[str, Any]:
    """Maximum squared-overlap assignment and its runner-up margin."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment requires a square matrix")
    size = matrix.shape[0]
    if size not in _PERMUTATIONS:
        _PERMUTATIONS[size] = np.asarray(
            list(itertools.permutations(range(size))), dtype=int
        )
    permutations = _PERMUTATIONS[size]
    values = np.abs(matrix) ** 2
    scores = np.sum(values[np.arange(size)[:, None], permutations.T], axis=0)
    order = np.argsort(-scores, kind="stable")
    best = int(order[0])
    runner = int(order[1]) if order.size > 1 else int(order[0])
    best_score = float(scores[best])
    runner_score = float(scores[runner]) if order.size > 1 else 0.0
    return {
        "permutation": permutations[best].tolist(),
        "score": best_score,
        "runner_up_score": runner_score,
        "margin": best_score - runner_score,
        "normalized_margin": (best_score - runner_score) / float(size),
    }


def polar_metrics(matrix: np.ndarray) -> Dict[str, Any]:
    left, singular_values, right_h = np.linalg.svd(matrix)
    return {
        "transport": left @ right_h,
        "singular_values": singular_values,
        "sigma_min": float(np.min(singular_values)),
        "leakage": float(1.0 - np.mean(singular_values**2)),
        **assignment(matrix),
    }


def normalized_phase_product(overlaps: Sequence[complex]) -> complex:
    result = 1.0 + 0.0j
    for overlap in overlaps:
        magnitude = abs(overlap)
        if magnitude < 1.0e-14:
            raise RuntimeError("zero overlap in scalar Berry product")
        result *= overlap / magnitude
    return complex(result)


def scalar_phase_audit(
    overlaps: Sequence[complex], seed: int = 20260918
) -> Dict[str, Any]:
    """Gauge-invariant scalar Berry product with a random U(1) audit."""
    product = normalized_phase_product(overlaps)
    rng = np.random.default_rng(int(seed))
    phases = np.exp(1j * rng.uniform(-math.pi, math.pi, len(overlaps)))
    gauged = [
        np.conj(phases[index])
        * overlaps[index]
        * phases[(index + 1) % len(overlaps)]
        for index in range(len(overlaps))
    ]
    gauged_product = normalized_phase_product(gauged)
    return {
        "phase_product": product,
        "phase_angle_rad": float(np.angle(product)),
        "phase_real": float(np.real(product)),
        "phase_imag": float(np.imag(product)),
        "minimum_edge_overlap_abs": float(min(abs(value) for value in overlaps)),
        "gauge_phase_product": gauged_product,
        "gauge_error": float(abs(product - gauged_product)),
    }


def selftest() -> None:
    identity = np.eye(3)
    kernel = determinant_kernel_rectangular(
        identity, 1, ((0, 1),), 0, ((0, 1, 2),)
    )
    assert kernel.shape == (1, 1)
    assert abs(kernel[0, 0] - 1.0) < 1.0e-14

    ci = np.asarray([[1.0, 2.0j], [-0.5, 0.25]], dtype=np.complex128)
    ci /= np.linalg.norm(ci)
    value = contract_ci_overlap(ci, ci, np.eye(2), np.eye(2))
    assert abs(value - 1.0) < 1.0e-14

    metrics = polar_metrics(np.diag([0.99, 0.98]))
    assert metrics["permutation"] == [0, 1]
    assert metrics["sigma_min"] > 0.97

    negative = scalar_phase_audit([1.0, 1.0, -1.0, 1.0])
    positive = scalar_phase_audit([1.0, -1.0, -1.0, 1.0])
    assert negative["phase_real"] < -0.999999999999
    assert positive["phase_real"] > 0.999999999999
    assert negative["gauge_error"] < 1.0e-12
    assert positive["gauge_error"] < 1.0e-12


if __name__ == "__main__":
    selftest()
    print("CPDC_OVERLAP_SELFTEST=PASS")
