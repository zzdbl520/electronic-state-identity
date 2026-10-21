#!/usr/bin/env python3
r"""Gate-0 molecular validation of mixed nuclear--variational transport.

The four corners are

    A = (q0, equal-weight SA2-CASSCF)
    B = (q1, equal-weight SA2-CASSCF)
    C = (q0, equal-weight SA3-CASSCF; lowest two-state subspace retained)
    D = (q1, equal-weight SA3-CASSCF; lowest two-state subspace retained)

where q0 and q1 are adjacent points on a small two-dimensional distortion
circle around equilateral H3.  For every edge, the program evaluates a genuine
many-electron overlap with PySCF's nonorthogonal CI contraction and verifies it
against an independent expansion in a common orthonormal Fock space.

Default production run
----------------------

    python H3_mixed_curvature_gate0.py \
        --threads 16 --memory-mb 32000 \
        --workdir H3_gate0

Quick checks that do not require PySCF
--------------------------------------

    python H3_mixed_curvature_gate0.py --self-test
    python H3_mixed_curvature_gate0.py --geometry-only

Outputs
-------

    H3_gate0/gate0_summary.json
    H3_gate0/gate0_matrices.npz
    H3_gate0/corners/*.npz
    H3_gate0/logs/*.log

The script intentionally uses restricted open-shell orbitals (ROHF followed by
restricted CASSCF) so that one common spatial-orbital overlap enters the alpha
and beta determinant contractions.  The target active-electron sector is
(N_alpha,N_beta)=(2,1), i.e. a neutral-doublet CAS(3,3).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


Array = np.ndarray
N_CAS = 3
NELE_CAS = (2, 1)
SELECTED_M = 2
TARGET_S2 = 0.75


@dataclass
class Corner:
    label: str
    protocol: str
    phi_deg: float
    atoms: list[tuple[str, tuple[float, float, float]]]
    mol: Any
    mo_coeff: Array
    ci_states: list[Array]
    selected_ci: list[Array]
    energies: list[float]
    spin_squares: list[float]
    ncore: int
    frame_orthogonality_error: float


@dataclass
class Edge:
    label: str
    overlap: Array
    direct_overlap: Array
    active_overlap: Array
    transport: Array
    positive: Array
    state_singular_values: Array
    active_singular_values: Array
    direct_overlap_error: float
    reverse_overlap_error: float
    reverse_transport_error: float
    unitarity_error: float
    lambda_state: float
    ambient_gram_error: float
    direct_state_norm_error: float


def dagger(matrix: Array) -> Array:
    return np.asarray(matrix).conj().T


def frobenius(matrix: Array) -> float:
    return float(np.linalg.norm(np.asarray(matrix).ravel()))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.asarray(value.real).tolist(),
                "imag": np.asarray(value.imag).tolist(),
            }
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def polar_transport(overlap: Array) -> tuple[Array, Array, Array]:
    u, singular_values, vh = np.linalg.svd(
        np.asarray(overlap, dtype=np.complex128), full_matrices=False
    )
    transport = u @ vh
    v = dagger(vh)
    positive = (v * singular_values) @ vh
    return transport, positive, singular_values


def plaquette_curvature(
    m_ab: Array,
    m_bd: Array,
    m_ac: Array,
    m_cd: Array,
) -> dict[str, Any]:
    matrices = [m_ab, m_bd, m_ac, m_cd]
    decompositions = [polar_transport(matrix) for matrix in matrices]
    transports = [item[0] for item in decompositions]
    singular_values = [item[2] for item in decompositions]
    k_q_to_p = transports[0] @ transports[1]
    k_p_to_q = transports[2] @ transports[3]
    holonomy = k_q_to_p @ dagger(k_p_to_q)
    identity = np.eye(m_ab.shape[0], dtype=np.complex128)
    denominator = 2.0 * math.sqrt(m_ab.shape[0])
    c_wilson = frobenius(holonomy - identity) / denominator
    c_paths = frobenius(k_q_to_p - k_p_to_q) / denominator
    eta_edge = min(float(np.min(values)) for values in singular_values)
    lambdas = [
        1.0 - float(np.sum(values**2)) / m_ab.shape[0]
        for values in singular_values
    ]
    phases = np.sort(np.angle(np.linalg.eigvals(holonomy)))
    return {
        "curvature": c_wilson,
        "curvature_from_paths": c_paths,
        "formula_error": abs(c_wilson - c_paths),
        "eta_edge": eta_edge,
        "lambda_edges": lambdas,
        "lambda_edge_max": max(lambdas),
        "k_q_to_p": k_q_to_p,
        "k_p_to_q": k_p_to_q,
        "holonomy": holonomy,
        "holonomy_phases": phases,
        "transports": transports,
        "state_singular_values": singular_values,
    }


def random_unitary(dimension: int, rng: np.random.Generator) -> Array:
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
    return q @ np.diag(phases.conj())


def gauge_test(overlaps: Sequence[Array], reference_c: float) -> float:
    rng = np.random.default_rng(20260811)
    g_a, g_b, g_c, g_d = [random_unitary(SELECTED_M, rng) for _ in range(4)]
    m_ab, m_bd, m_ac, m_cd = overlaps
    transformed = (
        dagger(g_a) @ m_ab @ g_b,
        dagger(g_b) @ m_bd @ g_d,
        dagger(g_a) @ m_ac @ g_c,
        dagger(g_c) @ m_cd @ g_d,
    )
    transformed_c = plaquette_curvature(*transformed)["curvature"]
    return abs(float(transformed_c) - float(reference_c))


def h3_distortion_geometry(
    phi_deg: float,
    rho_angstrom: float,
    circumradius_angstrom: float,
) -> tuple[list[tuple[str, tuple[float, float, float]]], dict[str, float]]:
    """Construct H3 from three distorted H--H distances.

    Qx and Qy span the two symmetry-breaking components at fixed mean bond
    length.  The equilateral point is Qx=Qy=0.
    """

    phi = math.radians(phi_deg)
    q_x = rho_angstrom * math.cos(phi)
    q_y = rho_angstrom * math.sin(phi)
    r0 = math.sqrt(3.0) * circumradius_angstrom
    r12 = r0 + q_x
    r23 = r0 - 0.5 * q_x + 0.5 * math.sqrt(3.0) * q_y
    r31 = r0 - 0.5 * q_x - 0.5 * math.sqrt(3.0) * q_y
    if min(r12, r23, r31) <= 0.0:
        raise ValueError("The requested distortion produced a nonpositive bond length.")
    if not (
        r12 + r23 > r31
        and r12 + r31 > r23
        and r23 + r31 > r12
    ):
        raise ValueError("The requested H--H distances violate the triangle inequality.")
    x3 = (r12 * r12 + r31 * r31 - r23 * r23) / (2.0 * r12)
    y3_square = r31 * r31 - x3 * x3
    if y3_square < -1.0e-12:
        raise ValueError(f"Negative y3^2={y3_square:.6e} in geometry reconstruction.")
    y3 = math.sqrt(max(0.0, y3_square))
    atoms = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (r12, 0.0, 0.0)),
        ("H", (x3, y3, 0.0)),
    ]
    metadata = {
        "phi_deg": phi_deg,
        "rho_angstrom": rho_angstrom,
        "Qx_angstrom": q_x,
        "Qy_angstrom": q_y,
        "r0_angstrom": r0,
        "r12_angstrom": r12,
        "r23_angstrom": r23,
        "r31_angstrom": r31,
    }
    return atoms, metadata


def distances_from_atoms(
    atoms: Sequence[tuple[str, tuple[float, float, float]]]
) -> tuple[float, float, float]:
    coordinates = [np.asarray(item[1], dtype=float) for item in atoms]
    return (
        float(np.linalg.norm(coordinates[0] - coordinates[1])),
        float(np.linalg.norm(coordinates[1] - coordinates[2])),
        float(np.linalg.norm(coordinates[2] - coordinates[0])),
    )


def set_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def import_pyscf():
    try:
        import pyscf
        from pyscf import fci, gto, lib, mcscf, scf
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is not importable. Activate the intended environment, for "
            "example `conda activate pyscf`, and rerun this input."
        ) from exc
    return pyscf, fci, gto, lib, mcscf, scf


def ci_state_list(ci: Any) -> list[Array]:
    if isinstance(ci, (list, tuple)):
        return [np.asarray(item) for item in ci]
    array = np.asarray(ci)
    if array.ndim >= 3:
        return [np.asarray(item) for item in array]
    return [array]


def extract_state_energies(mc: Any, nroots: int) -> list[float]:
    values = getattr(mc, "e_states", None)
    if values is None:
        candidate = np.asarray(mc.e_tot)
        if candidate.ndim == 1:
            values = candidate
    if values is None:
        raise RuntimeError("PySCF did not expose state-resolved CASSCF energies.")
    result = [float(item) for item in np.asarray(values).reshape(-1)]
    if len(result) != nroots:
        raise RuntimeError(f"Expected {nroots} state energies, received {result}.")
    return result


def selected_frame_orthogonality(ci_states: Sequence[Array]) -> float:
    count = len(ci_states)
    gram = np.empty((count, count), dtype=np.complex128)
    for i, bra in enumerate(ci_states):
        for j, ket in enumerate(ci_states):
            gram[i, j] = np.vdot(np.asarray(bra).ravel(), np.asarray(ket).ravel())
    return frobenius(gram - np.eye(count))


def spin_squares(ci_states: Sequence[Array], fci: Any) -> list[float]:
    values: list[float] = []
    for ci_vector in ci_states:
        try:
            s2, _multiplicity = fci.spin_op.spin_square0(
                np.asarray(ci_vector), N_CAS, NELE_CAS
            )
            values.append(float(s2))
        except Exception:
            values.append(float("nan"))
    return values


def close_mol_log(mol: Any) -> None:
    stream = getattr(mol, "stdout", None)
    if stream not in (None, sys.stdout, sys.stderr):
        try:
            stream.flush()
            stream.close()
        except Exception:
            pass


def run_corner(
    label: str,
    phi_deg: float,
    nroots: int,
    args: argparse.Namespace,
    fci: Any,
    gto: Any,
    mcscf: Any,
    scf: Any,
) -> Corner:
    atoms, _metadata = h3_distortion_geometry(
        phi_deg, args.rho_angstrom, args.circumradius_angstrom
    )
    log_path = args.workdir / "logs" / f"{label}.log"
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Angstrom"
    mol.basis = args.basis
    mol.charge = 0
    mol.spin = 1
    mol.symmetry = False
    mol.max_memory = args.memory_mb
    mol.verbose = args.verbose
    mol.output = str(log_path)
    mol.build()

    mf = scf.ROHF(mol)
    mf.conv_tol = args.hf_conv_tol
    mf.max_cycle = args.hf_max_cycle
    mf.chkfile = str(args.workdir / "corners" / f"{label}_rohf.chk")
    mf.kernel()
    if not mf.converged and args.newton_fallback:
        mf_newton = mf.newton()
        mf_newton.conv_tol = args.hf_conv_tol
        mf_newton.max_cycle = args.hf_max_cycle
        mf_newton.kernel(mf.mo_coeff, mf.mo_occ)
        mf = mf_newton
    if not mf.converged:
        close_mol_log(mol)
        raise RuntimeError(f"{label}: ROHF did not converge.")

    mc = mcscf.CASSCF(mf, N_CAS, NELE_CAS)
    mc.max_memory = args.memory_mb
    mc.conv_tol = args.mc_conv_tol
    mc.conv_tol_grad = args.mc_conv_tol_grad
    mc.max_cycle_macro = args.mc_max_cycle_macro
    mc.max_cycle_micro = args.mc_max_cycle_micro
    mc.canonicalization = True

    solver = fci.direct_spin1.FCI(mol)
    solver.spin = 1
    solver.nroots = nroots
    solver.conv_tol = args.fci_conv_tol
    solver.max_cycle = args.fci_max_cycle
    solver = fci.addons.fix_spin(
        solver, shift=args.spin_penalty_shift, ss=TARGET_S2
    )
    mc.fcisolver = solver
    mc.state_average_([1.0 / nroots] * nroots)
    mc.kernel(mf.mo_coeff)
    if not mc.converged and args.newton_fallback:
        mc_newton = mc.newton()
        mc_newton.max_cycle_macro = args.mc_newton_max_cycle
        mc_newton.conv_tol = args.mc_conv_tol
        mc_newton.conv_tol_grad = args.mc_conv_tol_grad
        mc_newton.kernel(mc.mo_coeff, mc.ci)
        mc = mc_newton
    if not mc.converged:
        close_mol_log(mol)
        raise RuntimeError(f"{label}: SA{nroots}-CASSCF did not converge.")

    ci_states = ci_state_list(mc.ci)
    if len(ci_states) != nroots:
        close_mol_log(mol)
        raise RuntimeError(
            f"{label}: expected {nroots} CI vectors, received {len(ci_states)}."
        )
    energies = extract_state_energies(mc, nroots)
    selected_ci = ci_states[:SELECTED_M]
    frame_error = selected_frame_orthogonality(selected_ci)
    s2_values = spin_squares(ci_states, fci)
    ncore = int(mc.ncore)
    if ncore != 0:
        close_mol_log(mol)
        raise RuntimeError(
            f"{label}: Gate-0 direct Fock check assumes ncore=0, got {ncore}."
        )

    checkpoint = args.workdir / "corners" / f"{label}.npz"
    np.savez_compressed(
        checkpoint,
        mo_coeff=np.asarray(mc.mo_coeff),
        ci_states=np.stack([np.asarray(item) for item in ci_states]),
        energies=np.asarray(energies),
        spin_squares=np.asarray(s2_values),
        ncore=ncore,
        phi_deg=phi_deg,
        nroots=nroots,
        atoms=np.asarray([item[1] for item in atoms]),
    )
    mol.stdout.flush()
    return Corner(
        label=label,
        protocol=f"SA{nroots}",
        phi_deg=phi_deg,
        atoms=atoms,
        mol=mol,
        mo_coeff=np.asarray(mc.mo_coeff),
        ci_states=ci_states,
        selected_ci=selected_ci,
        energies=energies,
        spin_squares=s2_values,
        ncore=ncore,
        frame_orthogonality_error=frame_error,
    )


def occupation_lists(fci: Any, norb: int, nelec: int) -> list[tuple[int, ...]]:
    strings = fci.cistring.make_strings(range(norb), nelec)
    return [
        tuple(index for index in range(norb) if int(bitstring) & (1 << index))
        for bitstring in np.asarray(strings).reshape(-1)
    ]


def union_orbital_embeddings(active_overlap: Array) -> tuple[Array, Array, float]:
    norb = active_overlap.shape[0]
    identity = np.eye(norb, dtype=np.complex128)
    gram = np.block(
        [[identity, active_overlap], [dagger(active_overlap), identity]]
    )
    gram = 0.5 * (gram + dagger(gram))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues.min()) < -1.0e-9:
        raise RuntimeError(
            "Union active-orbital Gram matrix is not positive semidefinite: "
            f"minimum eigenvalue={float(eigenvalues.min()):.3e}."
        )
    threshold = max(1.0e-12, 1.0e-11 * float(eigenvalues.max()))
    keep = eigenvalues > threshold
    coordinates = (
        np.sqrt(eigenvalues[keep])[:, None]
        * eigenvectors[:, keep].conj().T
    )
    embedding_x = coordinates[:, :norb]
    embedding_y = coordinates[:, norb:]
    reconstructed = dagger(coordinates) @ coordinates
    reconstruction_error = frobenius(reconstructed - gram)
    return embedding_x, embedding_y, reconstruction_error


def determinant_embedding(
    orbital_embedding: Array,
    ambient_occupations: Sequence[tuple[int, ...]],
    local_occupations: Sequence[tuple[int, ...]],
) -> Array:
    matrix = np.empty(
        (len(ambient_occupations), len(local_occupations)),
        dtype=np.complex128,
    )
    for ambient_index, ambient_occ in enumerate(ambient_occupations):
        for local_index, local_occ in enumerate(local_occupations):
            block = orbital_embedding[np.ix_(ambient_occ, local_occ)]
            matrix[ambient_index, local_index] = np.linalg.det(block)
    return matrix


def embed_ci_states_in_common_fock_space(
    ci_states: Sequence[Array],
    orbital_embedding: Array,
    fci: Any,
) -> list[Array]:
    ambient_norb = orbital_embedding.shape[0]
    local_alpha = occupation_lists(fci, N_CAS, NELE_CAS[0])
    local_beta = occupation_lists(fci, N_CAS, NELE_CAS[1])
    ambient_alpha = occupation_lists(fci, ambient_norb, NELE_CAS[0])
    ambient_beta = occupation_lists(fci, ambient_norb, NELE_CAS[1])
    alpha_embedding = determinant_embedding(
        orbital_embedding, ambient_alpha, local_alpha
    )
    beta_embedding = determinant_embedding(
        orbital_embedding, ambient_beta, local_beta
    )
    expected_shape = (len(local_alpha), len(local_beta))
    embedded: list[Array] = []
    for ci_vector in ci_states:
        ci_matrix = np.asarray(ci_vector).reshape(expected_shape)
        embedded.append(alpha_embedding @ ci_matrix @ beta_embedding.T)
    return embedded


def direct_fock_overlap_matrix(
    ci_x: Sequence[Array],
    ci_y: Sequence[Array],
    active_overlap: Array,
    fci: Any,
) -> tuple[Array, float, float]:
    embedding_x, embedding_y, gram_error = union_orbital_embeddings(active_overlap)
    states_x = embed_ci_states_in_common_fock_space(ci_x, embedding_x, fci)
    states_y = embed_ci_states_in_common_fock_space(ci_y, embedding_y, fci)
    overlap = np.empty((len(states_x), len(states_y)), dtype=np.complex128)
    for i, bra in enumerate(states_x):
        for j, ket in enumerate(states_y):
            overlap[i, j] = np.vdot(bra.ravel(), ket.ravel())
    norm_errors = [
        abs(float(np.vdot(state.ravel(), state.ravel()).real) - 1.0)
        for state in [*states_x, *states_y]
    ]
    return overlap, gram_error, max(norm_errors)


def many_electron_overlap(
    corner_x: Corner,
    corner_y: Corner,
    fci: Any,
    gto: Any,
) -> tuple[Array, Array, Array, float, float]:
    cross_ao = gto.intor_cross("int1e_ovlp", corner_x.mol, corner_y.mol)
    active_x = corner_x.mo_coeff[
        :, corner_x.ncore : corner_x.ncore + N_CAS
    ]
    active_y = corner_y.mo_coeff[
        :, corner_y.ncore : corner_y.ncore + N_CAS
    ]
    active_overlap = dagger(active_x) @ cross_ao @ active_y
    overlap = np.empty((SELECTED_M, SELECTED_M), dtype=np.complex128)
    for i in range(SELECTED_M):
        for j in range(SELECTED_M):
            overlap[i, j] = fci.addons.overlap(
                corner_x.selected_ci[i],
                corner_y.selected_ci[j],
                N_CAS,
                NELE_CAS,
                s=active_overlap,
            )
    direct, gram_error, norm_error = direct_fock_overlap_matrix(
        corner_x.selected_ci,
        corner_y.selected_ci,
        active_overlap,
        fci,
    )
    return overlap, direct, active_overlap, gram_error, norm_error


def build_edge(
    label: str,
    corner_x: Corner,
    corner_y: Corner,
    fci: Any,
    gto: Any,
) -> Edge:
    overlap, direct, active_overlap, gram_error, norm_error = many_electron_overlap(
        corner_x, corner_y, fci, gto
    )
    reverse, _reverse_direct, _reverse_active, _reverse_gram, _reverse_norm = (
        many_electron_overlap(corner_y, corner_x, fci, gto)
    )
    transport, positive, state_singular_values = polar_transport(overlap)
    reverse_transport = polar_transport(reverse)[0]
    active_singular_values = np.linalg.svd(active_overlap, compute_uv=False)
    direct_error = frobenius(overlap - direct)
    reverse_overlap_error = frobenius(reverse - dagger(overlap))
    reverse_transport_error = frobenius(reverse_transport - dagger(transport))
    unitarity_error = frobenius(
        dagger(transport) @ transport - np.eye(SELECTED_M)
    )
    lambda_state = 1.0 - float(np.sum(state_singular_values**2)) / SELECTED_M
    return Edge(
        label=label,
        overlap=overlap,
        direct_overlap=direct,
        active_overlap=active_overlap,
        transport=transport,
        positive=positive,
        state_singular_values=state_singular_values,
        active_singular_values=active_singular_values,
        direct_overlap_error=direct_error,
        reverse_overlap_error=reverse_overlap_error,
        reverse_transport_error=reverse_transport_error,
        unitarity_error=unitarity_error,
        lambda_state=lambda_state,
        ambient_gram_error=gram_error,
        direct_state_norm_error=norm_error,
    )


def edge_summary(edge: Edge) -> dict[str, Any]:
    return {
        "state_singular_values": edge.state_singular_values,
        "active_orbital_singular_values": edge.active_singular_values,
        "sigma_min_state": float(np.min(edge.state_singular_values)),
        "sigma_min_active": float(np.min(edge.active_singular_values)),
        "lambda_state": edge.lambda_state,
        "production_vs_direct_fock_error": edge.direct_overlap_error,
        "reverse_overlap_error": edge.reverse_overlap_error,
        "reverse_transport_error": edge.reverse_transport_error,
        "polar_unitarity_error": edge.unitarity_error,
        "ambient_gram_reconstruction_error": edge.ambient_gram_error,
        "direct_state_norm_error": edge.direct_state_norm_error,
    }


def run_gate0(args: argparse.Namespace) -> int:
    set_thread_environment(args.threads)
    pyscf, fci, gto, lib, mcscf, scf = import_pyscf()
    lib.num_threads(args.threads)
    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError(
            "This PySCF version does not provide fci.addons.overlap."
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "logs").mkdir(exist_ok=True)
    (args.workdir / "corners").mkdir(exist_ok=True)

    corner_specs = (
        ("A_q0_SA2", args.phi0_deg, 2),
        ("B_q1_SA2", args.phi1_deg, 2),
        ("C_q0_SA3", args.phi0_deg, 3),
        ("D_q1_SA3", args.phi1_deg, 3),
    )
    corners: dict[str, Corner] = {}
    print("H3 mixed-curvature Gate 0", flush=True)
    print(
        f"rho={args.rho_angstrom:.6f} Angstrom; "
        f"phi0={args.phi0_deg:.6f} deg; phi1={args.phi1_deg:.6f} deg",
        flush=True,
    )
    for label, phi_deg, nroots in corner_specs:
        print(f"Running {label} ...", flush=True)
        corners[label] = run_corner(
            label, phi_deg, nroots, args, fci, gto, mcscf, scf
        )
        corner = corners[label]
        print(
            f"  converged energies="
            + ", ".join(f"{energy:.12f}" for energy in corner.energies),
            flush=True,
        )

    edge_specs = (
        ("AB", "A_q0_SA2", "B_q1_SA2"),
        ("BD", "B_q1_SA2", "D_q1_SA3"),
        ("AC", "A_q0_SA2", "C_q0_SA3"),
        ("CD", "C_q0_SA3", "D_q1_SA3"),
    )
    edges: dict[str, Edge] = {}
    for label, x_label, y_label in edge_specs:
        print(f"Building edge {label}: {x_label} -> {y_label}", flush=True)
        edges[label] = build_edge(
            label, corners[x_label], corners[y_label], fci, gto
        )
        print(
            f"  direct error={edges[label].direct_overlap_error:.3e}; "
            f"sigma_min={float(edges[label].state_singular_values.min()):.8f}",
            flush=True,
        )

    metrics = plaquette_curvature(
        edges["AB"].overlap,
        edges["BD"].overlap,
        edges["AC"].overlap,
        edges["CD"].overlap,
    )
    gauge_error = gauge_test(
        [
            edges["AB"].overlap,
            edges["BD"].overlap,
            edges["AC"].overlap,
            edges["CD"].overlap,
        ],
        metrics["curvature"],
    )

    tolerances = {
        "frame_orthogonality": args.validation_tolerance,
        "production_vs_direct_fock": args.validation_tolerance,
        "polar_unitarity": args.validation_tolerance,
        "reverse_overlap": args.validation_tolerance,
        "reverse_transport": args.validation_tolerance,
        "curvature_formula": args.validation_tolerance,
        "gauge_invariance": args.gauge_tolerance,
    }
    checks: dict[str, dict[str, Any]] = {}

    def record_check(name: str, value: float, tolerance: float) -> None:
        checks[name] = {
            "value": float(value),
            "tolerance": float(tolerance),
            "passed": bool(value < tolerance),
        }

    record_check(
        "maximum_frame_orthogonality_error",
        max(corner.frame_orthogonality_error for corner in corners.values()),
        tolerances["frame_orthogonality"],
    )
    record_check(
        "maximum_production_vs_direct_fock_error",
        max(edge.direct_overlap_error for edge in edges.values()),
        tolerances["production_vs_direct_fock"],
    )
    record_check(
        "maximum_polar_unitarity_error",
        max(edge.unitarity_error for edge in edges.values()),
        tolerances["polar_unitarity"],
    )
    record_check(
        "maximum_reverse_overlap_error",
        max(edge.reverse_overlap_error for edge in edges.values()),
        tolerances["reverse_overlap"],
    )
    record_check(
        "maximum_reverse_transport_error",
        max(edge.reverse_transport_error for edge in edges.values()),
        tolerances["reverse_transport"],
    )
    record_check(
        "curvature_formula_error",
        metrics["formula_error"],
        tolerances["curvature_formula"],
    )
    record_check(
        "random_corner_gauge_error",
        gauge_error,
        tolerances["gauge_invariance"],
    )
    all_passed = all(item["passed"] for item in checks.values())

    geometry_metadata: dict[str, Any] = {}
    for phi_deg in (args.phi0_deg, args.phi1_deg):
        atoms, metadata = h3_distortion_geometry(
            phi_deg, args.rho_angstrom, args.circumradius_angstrom
        )
        reconstructed = distances_from_atoms(atoms)
        metadata["reconstructed_distances_angstrom"] = reconstructed
        geometry_metadata[f"phi_{phi_deg:.8f}"] = metadata

    summary = {
        "gate": "H3 production many-electron overlap and four-corner plaquette",
        "status": "PASS" if all_passed else "FAIL",
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyscf": getattr(pyscf, "__version__", "unknown"),
        },
        "parameters": {
            "basis": args.basis,
            "active_space": "CAS(3,3)",
            "active_electrons": NELE_CAS,
            "selected_m": SELECTED_M,
            "protocols": ["equal-weight SA2", "equal-weight SA3"],
            "rho_angstrom": args.rho_angstrom,
            "phi0_deg": args.phi0_deg,
            "phi1_deg": args.phi1_deg,
            "threads": args.threads,
            "memory_mb": args.memory_mb,
        },
        "geometries": geometry_metadata,
        "corners": {
            label: {
                "protocol": corner.protocol,
                "phi_deg": corner.phi_deg,
                "energies_hartree": corner.energies,
                "spin_squares": corner.spin_squares,
                "selected_frame_orthogonality_error": (
                    corner.frame_orthogonality_error
                ),
                "checkpoint": args.workdir / "corners" / f"{label}.npz",
                "log": args.workdir / "logs" / f"{label}.log",
            }
            for label, corner in corners.items()
        },
        "edges": {label: edge_summary(edge) for label, edge in edges.items()},
        "plaquette": {
            "curvature": metrics["curvature"],
            "curvature_from_paths": metrics["curvature_from_paths"],
            "formula_error": metrics["formula_error"],
            "eta_edge": metrics["eta_edge"],
            "lambda_edges_order_AB_BD_AC_CD": metrics["lambda_edges"],
            "lambda_edge_max": metrics["lambda_edge_max"],
            "holonomy_phases_radian": metrics["holonomy_phases"],
            "random_corner_gauge_error": gauge_error,
        },
        "acceptance_checks": checks,
    }

    matrix_archive: dict[str, Array] = {
        "K_q_to_p": metrics["k_q_to_p"],
        "K_p_to_q": metrics["k_p_to_q"],
        "W_qp": metrics["holonomy"],
        "holonomy_phases": metrics["holonomy_phases"],
    }
    for label, edge in edges.items():
        matrix_archive[f"M_{label}"] = edge.overlap
        matrix_archive[f"M_direct_{label}"] = edge.direct_overlap
        matrix_archive[f"S_active_{label}"] = edge.active_overlap
        matrix_archive[f"T_{label}"] = edge.transport
        matrix_archive[f"H_{label}"] = edge.positive
        matrix_archive[f"sigma_state_{label}"] = edge.state_singular_values
        matrix_archive[f"sigma_active_{label}"] = edge.active_singular_values
    np.savez_compressed(args.workdir / "gate0_matrices.npz", **matrix_archive)
    with (args.workdir / "gate0_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    for corner in corners.values():
        close_mol_log(corner.mol)

    print("", flush=True)
    print(f"Gate-0 status: {'PASS' if all_passed else 'FAIL'}", flush=True)
    print(f"C_qp       = {metrics['curvature']:.12e}", flush=True)
    print(f"eta_edge   = {metrics['eta_edge']:.12e}", flush=True)
    print(f"lambda_max = {metrics['lambda_edge_max']:.12e}", flush=True)
    print(
        "max direct Fock error = "
        f"{max(edge.direct_overlap_error for edge in edges.values()):.3e}",
        flush=True,
    )
    print(f"Wrote {args.workdir / 'gate0_summary.json'}", flush=True)
    print(f"Wrote {args.workdir / 'gate0_matrices.npz'}", flush=True)
    return 0 if all_passed else 2


def run_self_test(args: argparse.Namespace) -> int:
    for phi_deg in (args.phi0_deg, args.phi1_deg):
        atoms, metadata = h3_distortion_geometry(
            phi_deg, args.rho_angstrom, args.circumradius_angstrom
        )
        reconstructed = distances_from_atoms(atoms)
        expected = (
            metadata["r12_angstrom"],
            metadata["r23_angstrom"],
            metadata["r31_angstrom"],
        )
        if max(abs(a - b) for a, b in zip(reconstructed, expected)) > 1.0e-12:
            raise AssertionError("H3 Cartesian reconstruction failed.")
    identity = np.eye(2)
    metrics = plaquette_curvature(identity, identity, identity, identity)
    if abs(metrics["curvature"]) > 1.0e-14:
        raise AssertionError("Identity plaquette has nonzero curvature.")
    if abs(metrics["eta_edge"] - 1.0) > 1.0e-14:
        raise AssertionError("Identity plaquette has incorrect eta_edge.")
    print("Self-test PASS: geometry reconstruction and plaquette algebra.")
    return 0


def print_geometries(args: argparse.Namespace) -> int:
    for label, phi_deg in (("q0", args.phi0_deg), ("q1", args.phi1_deg)):
        atoms, metadata = h3_distortion_geometry(
            phi_deg, args.rho_angstrom, args.circumradius_angstrom
        )
        print(f"{label}: phi={phi_deg:.8f} deg")
        for symbol, (x, y, z) in atoms:
            print(f"  {symbol:2s} {x: .12f} {y: .12f} {z: .12f}")
        print(
            "  distances: "
            f"r12={metadata['r12_angstrom']:.12f}, "
            f"r23={metadata['r23_angstrom']:.12f}, "
            f"r31={metadata['r31_angstrom']:.12f} Angstrom"
        )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="H3 four-corner production-overlap validation for mixed curvature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workdir", type=Path, default=Path("H3_gate0"))
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=32000)
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--rho-angstrom", type=float, default=0.010)
    parser.add_argument("--circumradius-angstrom", type=float, default=0.90)
    parser.add_argument("--phi0-deg", type=float, default=0.0)
    parser.add_argument("--phi1-deg", type=float, default=10.0)
    parser.add_argument("--hf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--hf-max-cycle", type=int, default=100)
    parser.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    parser.add_argument("--mc-max-cycle-macro", type=int, default=100)
    parser.add_argument("--mc-max-cycle-micro", type=int, default=10)
    parser.add_argument("--mc-newton-max-cycle", type=int, default=60)
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    parser.add_argument("--fci-max-cycle", type=int, default=200)
    parser.add_argument("--spin-penalty-shift", type=float, default=0.2)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--gauge-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--newton-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.memory_mb < 1000:
        raise ValueError("--memory-mb is implausibly small for this input.")
    if args.rho_angstrom <= 0.0:
        raise ValueError("--rho-angstrom must be positive.")
    if args.self_test:
        return run_self_test(args)
    if args.geometry_only:
        return print_geometries(args)
    return run_gate0(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

