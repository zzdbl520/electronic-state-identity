"""Small, dependency-light transport routines used across the workflows.

An edge overlap ``M_ab`` transforms as ``G_a^dagger M_ab G_b`` under independent
unitary changes of frame at its endpoints. Its unitary polar factor has the same
covariance. Scalar quantities returned here are invariant under those frame
changes; transport matrices themselves are covariant, not invariant.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def _square(matrix: ArrayLike, name: str) -> ComplexArray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix; got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains a non-finite value.")
    return value


def polar_transport(overlap: ArrayLike) -> tuple[ComplexArray, NDArray[np.float64]]:
    """Return the unitary polar factor and singular values of an edge overlap."""

    matrix = _square(overlap, "overlap")
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_h, singular_values


def edge_fidelity(overlap: ArrayLike) -> dict[str, Any]:
    """Return retained-subspace conditioning diagnostics for one edge."""

    transport, singular_values = polar_transport(overlap)
    return {
        "transport": transport,
        "singular_values": singular_values,
        "sigma_min": float(np.min(singular_values)),
        "leakage": float(1.0 - np.mean(singular_values**2)),
    }


def compose(transports: Iterable[ArrayLike]) -> ComplexArray:
    """Compose edge transports in traversal order."""

    values = [_square(value, "transport") for value in transports]
    if not values:
        raise ValueError("At least one transport is required.")
    dimension = values[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in values):
        raise ValueError("All transports must have the same dimension.")
    result = np.eye(dimension, dtype=np.complex128)
    for value in values:
        result = result @ value
    return result


def route_nonclosure(first: ArrayLike, second: ArrayLike) -> float:
    """Return the bounded normalized Frobenius difference between two routes.

    For two ``m``-dimensional unitary routes this is
    ``||K1-K2||_F / (2 sqrt(m))``, the ``C_qp`` convention used in the LiH and
    BeH2 analyses. The norm-sum form also remains bounded for nonunitary inputs.
    """

    left = _square(first, "first route")
    right = _square(second, "second route")
    if left.shape != right.shape:
        raise ValueError("Route matrices must have the same dimension.")
    denominator = np.linalg.norm(left, ord="fro") + np.linalg.norm(right, ord="fro")
    if denominator == 0.0:
        raise ZeroDivisionError("Both route matrices have zero Frobenius norm.")
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def back_transport(operator: ArrayLike, transport: ArrayLike) -> ComplexArray:
    """Back-transport one operator, or a stack of operators, from the endpoint."""

    route = _square(transport, "transport")
    value = np.asarray(operator, dtype=np.complex128)
    if value.ndim == 2:
        if value.shape != route.shape:
            raise ValueError("Operator and transport dimensions differ.")
        return route @ value @ route.conj().T
    if value.ndim == 3 and value.shape[1:] == route.shape:
        return np.asarray([route @ item @ route.conj().T for item in value])
    raise ValueError("operator must be square or a stack of equally sized square matrices.")


def assignment(matrix: ArrayLike) -> dict[str, Any]:
    """Maximum-absolute-overlap assignment and its runner-up margin.

    This exhaustive implementation is intended for the small retained manifolds
    used here, not as a general large assignment solver.
    """

    value = _square(matrix, "matrix")
    dimension = value.shape[0]
    scored = [
        (float(sum(abs(value[i, p[i]]) for i in range(dimension))), p)
        for p in itertools.permutations(range(dimension))
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    runner_score = scored[1][0] if len(scored) > 1 else 0.0
    return {
        "permutation": list(best),
        "score": best_score,
        "runner_up_score": runner_score,
        "margin": best_score - runner_score,
        "normalized_margin": (best_score - runner_score) / dimension,
    }


def plaquette(ab: ArrayLike, bd: ArrayLike, ac: ArrayLike, cd: ArrayLike) -> dict[str, Any]:
    """Evaluate the two A-to-D transports around a four-vertex plaquette."""

    overlaps = [ab, bd, ac, cd]
    edges = [polar_transport(value) for value in overlaps]
    route_qp = compose((edges[0][0], edges[1][0]))
    route_pq = compose((edges[2][0], edges[3][0]))
    holonomy = route_qp.conj().T @ route_pq
    return {
        "route_qp": route_qp,
        "route_pq": route_pq,
        "C_qp": route_nonclosure(route_qp, route_pq),
        "holonomy": holonomy,
        "holonomy_phases_rad": np.sort(np.angle(np.linalg.eigvals(holonomy))),
        "edge_singular_values": tuple(item[1] for item in edges),
        "eta_edge": min(float(np.min(item[1])) for item in edges),
    }
