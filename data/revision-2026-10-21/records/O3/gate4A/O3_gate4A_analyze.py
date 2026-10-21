#!/usr/bin/env python3
"""Analyze O3 Gate 4A checkpoints with genuine many-electron overlaps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass
class Wavefunction:
    tag: str
    protocol: str
    basis: str
    r1: float
    r2: float
    theta: float
    ncas: int
    nelecas: int
    ncore: int
    mo: object
    ci: list
    energies: object


def import_modules():
    try:
        import numpy as np
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for AO cross-overlaps. Activate the pyscf "
            "environment before running this analyzer."
        ) from exc
    return np, fci, gto


def atoms(wfn: Wavefunction):
    half = math.radians(wfn.theta) / 2.0
    s, c = math.sin(half), math.cos(half)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("O", (wfn.r1 * s, wfn.r1 * c, 0.0)),
        ("O", (-wfn.r2 * s, wfn.r2 * c, 0.0)),
    ]


def molecule(wfn: Wavefunction, gto):
    mol = gto.Mole()
    mol.atom = atoms(wfn)
    mol.unit = "Angstrom"
    mol.basis = wfn.basis
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = "Cs"
    mol.verbose = 0
    mol.build()
    if mol.groupname != "Cs":
        raise RuntimeError(f"Expected forced Cs, got {mol.groupname} for {wfn.tag}.")
    return mol


def load(path: Path, np) -> Wavefunction:
    with np.load(path, allow_pickle=False) as data:
        return Wavefunction(
            tag=str(data["tag"].item()),
            protocol=str(data["protocol"].item()),
            basis=str(data["basis"].item()),
            r1=float(data["r1"].item()),
            r2=float(data["r2"].item()),
            theta=float(data["theta"].item()),
            ncas=int(data["ncas"].item()),
            nelecas=int(data["nelecas"].item()),
            ncore=int(data["ncore"].item()),
            mo=data["mo_coeff"].copy(),
            ci=[x.copy() for x in data["ci_vectors"]],
            energies=data["state_energies"].copy(),
        )


def full_overlap(x: Wavefunction, y: Wavefunction, mol_x, mol_y, fci, gto, np):
    """Exact overlap for a common inactive-core/CAS partition.

    For the orbital cross-overlap partitioned into core and active blocks,
    determinant block elimination gives

      <Psi_X|Psi_Y> = det(S_cc)^2 <CI_X|CI_Y>_{S_eff},
      S_eff = S_aa - S_ac S_cc^{-1} S_ca.

    This retains core-active cross terms and is therefore not merely a CI
    coefficient dot product or an active-only overlap.
    """
    if (x.ncas, x.nelecas, x.ncore) != (y.ncas, y.nelecas, y.ncore):
        raise RuntimeError("The two checkpoints do not share one CAS/core partition.")
    if x.basis.lower() != y.basis.lower():
        raise RuntimeError("The two checkpoints do not use the same AO basis.")
    overlap_fn = getattr(fci.addons, "overlap", None)
    if overlap_fn is None:
        raise RuntimeError("This PySCF version lacks fci.addons.overlap.")

    sao = gto.intor_cross("int1e_ovlp", mol_x, mol_y)
    core_x = x.mo[:, : x.ncore]
    core_y = y.mo[:, : y.ncore]
    active_x = x.mo[:, x.ncore : x.ncore + x.ncas]
    active_y = y.mo[:, y.ncore : y.ncore + y.ncas]
    scc = core_x.T.conj() @ sao @ core_y
    sca = core_x.T.conj() @ sao @ active_y
    sac = active_x.T.conj() @ sao @ core_y
    saa = active_x.T.conj() @ sao @ active_y
    core_singular = np.linalg.svd(scc, compute_uv=False)
    if float(core_singular.min()) < 1.0e-10:
        raise RuntimeError(
            f"Core cross-overlap is singular for {x.tag}/{x.protocol} -> "
            f"{y.tag}/{y.protocol}."
        )
    effective_active = saa - sac @ np.linalg.solve(scc, sca)
    core_factor = np.linalg.det(scc) ** 2
    matrix = np.empty((len(x.ci), len(y.ci)), dtype=complex)
    for i, ci_x in enumerate(x.ci):
        for j, ci_y in enumerate(y.ci):
            matrix[i, j] = core_factor * overlap_fn(
                ci_x, ci_y, x.ncas, x.nelecas, s=effective_active
            )
    metadata = {
        "abs_core_factor": float(abs(core_factor)),
        "core_sigma_min": float(core_singular.min()),
        "effective_active_sigma_min": float(
            np.linalg.svd(effective_active, compute_uv=False).min()
        ),
    }
    return matrix, metadata


def polar(matrix, np):
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh, singular


def leakage(singular, m: int) -> float:
    value = 1.0 - float(sum(abs(x) ** 2 for x in singular)) / m
    return 0.0 if -1.0e-12 < value < 0.0 else value


def random_unitary(m: int, rng, np):
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(phases.conj())


def plaquette_metrics(edges: dict, np, gauge_test: bool = True):
    transports = {}
    singular = {}
    for name, matrix in edges.items():
        transports[name], singular[name] = polar(matrix, np)
    k_qp = transports["AB"] @ transports["BD"]
    k_pq = transports["AC"] @ transports["CD"]
    numerator = float(np.linalg.norm(k_qp - k_pq, ord="fro"))
    denominator = float(
        np.linalg.norm(k_qp, ord="fro") + np.linalg.norm(k_pq, ord="fro")
    )
    result = {
        "C_qp": numerator / denominator,
        "path_difference_F": numerator,
        "eta_edge": min(float(values.min()) for values in singular.values()),
    }
    for name in ("AB", "BD", "AC", "CD"):
        result[f"sigma_min_{name}"] = float(singular[name].min())
        result[f"lambda_{name}"] = leakage(singular[name], 2)

    if gauge_test:
        rng = np.random.default_rng(20260811)
        ua, ub, uc, ud = [random_unitary(2, rng, np) for _ in range(4)]
        transformed = {
            "AB": ua.T.conj() @ edges["AB"] @ ub,
            "BD": ub.T.conj() @ edges["BD"] @ ud,
            "AC": ua.T.conj() @ edges["AC"] @ uc,
            "CD": uc.T.conj() @ edges["CD"] @ ud,
        }
        tested = plaquette_metrics(transformed, np, gauge_test=False)
        result["gauge_test_abs_error"] = abs(result["C_qp"] - tested["C_qp"])
    return result


def clean_complex_matrix(matrix):
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in matrix
    ]


def analyze(args: argparse.Namespace) -> None:
    np, fci, gto = import_modules()
    directory = args.workdir / "wavefunctions"
    tags = ["center", "dr_minus", "dr_plus", "theta_minus", "theta_plus"]
    wfns: dict[tuple[str, str], Wavefunction] = {}
    mols = {}
    missing = []
    for tag in tags:
        for protocol in ("SA2", "SA3"):
            path = directory / f"{tag}_{protocol}.npz"
            if not path.exists():
                missing.append(str(path))
                continue
            wfn = load(path, np)
            expected = 2 if protocol == "SA2" else 3
            if len(wfn.ci) != expected:
                raise RuntimeError(f"{path} contains {len(wfn.ci)} roots, expected {expected}.")
            wfns[(tag, protocol)] = wfn
            mols[(tag, protocol)] = molecule(wfn, gto)
    if missing:
        raise RuntimeError("Missing checkpoints:\n  " + "\n  ".join(missing))

    # At each geometry, choose the optimal two-dimensional frame inside the
    # three-state SA3 manifold by the right singular vectors of M(SA2,SA3).
    selection = {}
    same_geometry = {}
    subspace_rows = []
    for tag in tags:
        x, y = wfns[(tag, "SA2")], wfns[(tag, "SA3")]
        m23, overlap_meta = full_overlap(
            x, y, mols[(tag, "SA2")], mols[(tag, "SA3")], fci, gto, np
        )
        _u, singular, vh = np.linalg.svd(m23, full_matrices=True)
        frame = vh.T.conj()[:, :2]
        selection[tag] = frame
        same_geometry[tag] = m23
        subspace_rows.append({
            "tag": tag,
            "delta_r_A": f"{x.r1-x.r2:.8f}",
            "theta_deg": f"{x.theta:.8f}",
            "SA2_gap01_Eh": f"{float(x.energies[1]-x.energies[0]):.12e}",
            "SA3_gap01_Eh": f"{float(y.energies[1]-y.energies[0]):.12e}",
            "SA3_gap12_Eh": f"{float(y.energies[2]-y.energies[1]):.12e}",
            "sigma_match_1": f"{singular[0]:.12e}",
            "sigma_match_2": f"{singular[1]:.12e}",
            "lambda_protocol": f"{leakage(singular, 2):.12e}",
            "SA3_root3_weight_in_selected_subspace": f"{float(np.sum(abs(frame[2,:])**2)):.12e}",
            **{name: f"{value:.12e}" for name, value in overlap_meta.items()},
        })

    definitions = [
        ("dr_minus_to_center", "delta_r", "dr_minus", "center"),
        ("center_to_dr_plus", "delta_r", "center", "dr_plus"),
        ("theta_minus_to_center", "theta", "theta_minus", "center"),
        ("center_to_theta_plus", "theta", "center", "theta_plus"),
    ]
    results = []
    detailed = {}
    for label, direction, q0, q1 in definitions:
        sa2_0, sa2_1 = wfns[(q0, "SA2")], wfns[(q1, "SA2")]
        sa3_0, sa3_1 = wfns[(q0, "SA3")], wfns[(q1, "SA3")]
        m22, meta22 = full_overlap(
            sa2_0, sa2_1, mols[(q0, "SA2")], mols[(q1, "SA2")], fci, gto, np
        )
        m33, meta33 = full_overlap(
            sa3_0, sa3_1, mols[(q0, "SA3")], mols[(q1, "SA3")], fci, gto, np
        )
        v0, v1 = selection[q0], selection[q1]
        edges = {
            "AB": m22,
            "BD": same_geometry[q1] @ v1,
            "AC": same_geometry[q0] @ v0,
            "CD": v0.T.conj() @ m33 @ v1,
        }
        metrics = plaquette_metrics(edges, np)
        delta_q = (
            abs((sa2_1.r1-sa2_1.r2) - (sa2_0.r1-sa2_0.r2))
            if direction == "delta_r" else abs(sa2_1.theta-sa2_0.theta)
        )
        row = {
            "plaquette": label,
            "direction": direction,
            "q0": q0,
            "q1": q1,
            "delta_q": f"{delta_q:.12e}",
            **{name: f"{value:.12e}" for name, value in metrics.items()},
        }
        results.append(row)
        detailed[label] = {
            "metrics": metrics,
            "edges": {name: clean_complex_matrix(value) for name, value in edges.items()},
            "SA2_nuclear_overlap_metadata": meta22,
            "SA3_nuclear_overlap_metadata": meta33,
        }

    subspace_path = args.workdir / "O3_gate4A_subspaces.csv"
    with subspace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subspace_rows[0]))
        writer.writeheader()
        writer.writerows(subspace_rows)
    result_path = args.workdir / "O3_gate4A_plaquettes.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "schema_version": 1,
        "definition": (
            "polar-unitary mixed nuclear-variational nonclosure; SA3 m=2 frame "
            "selected by the right singular subspace of the same-geometry "
            "genuine many-electron SA2-to-SA3 overlap"
        ),
        "many_electron_overlap": (
            "det(Scc)^2 times the CAS-CI overlap in the Schur-complement "
            "active metric; core-active cross terms retained"
        ),
        "subspaces": subspace_rows,
        "plaquettes": detailed,
    }
    (args.workdir / "O3_gate4A_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nO3 Gate 4A results")
    print("plaquette                         C_qp        eta_edge    gauge_error")
    for row in results:
        print(
            f"{row['plaquette']:32s} {float(row['C_qp']):.6e}  "
            f"{float(row['eta_edge']):.6e}  {float(row['gauge_test_abs_error']):.3e}"
        )
    print(f"\nWrote {result_path}")
    print(f"Wrote {subspace_path}")
    print(f"Wrote {args.workdir / 'O3_gate4A_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("O3_gate4A_work"))
    args = parser.parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    analyze(args)


if __name__ == "__main__":
    main()
