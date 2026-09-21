#!/usr/bin/env python3
"""Evaluate refined BeH2 mixed nuclear--variational plaquettes.

The production overlap includes AO cross-overlaps and the complete correlated
wavefunction.  If inactive core orbitals are present, active-space CI vectors
are embedded into the full core-plus-active determinant space before calling
PySCF's nonorthogonal FCI overlap.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from BeH2_B0_analyze import Solution, load_solution, pair_metrics, vector_text


Array = np.ndarray
SELECTED_M = 3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze BeH2 B2 mixed plaquettes.")
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B2_work"))
    p.add_argument("--output-prefix", default="B2")
    p.add_argument("--self-test", action="store_true")
    return p


def dagger(matrix: Array) -> Array:
    return np.asarray(matrix).conj().T


def frobenius(matrix: Array) -> float:
    return float(np.linalg.norm(np.asarray(matrix), ord="fro"))


def polar_transport(overlap: Array) -> tuple[Array, Array]:
    u, singular, vh = np.linalg.svd(overlap, full_matrices=False)
    return u @ vh, singular


def plaquette_metrics(ab: Array, bd: Array, ac: Array, cd: Array) -> dict[str, Any]:
    t_ab, s_ab = polar_transport(ab)
    t_bd, s_bd = polar_transport(bd)
    t_ac, s_ac = polar_transport(ac)
    t_cd, s_cd = polar_transport(cd)
    k_qp = t_ab @ t_bd
    k_pq = t_ac @ t_cd
    denominator = frobenius(k_qp) + frobenius(k_pq)
    curvature = frobenius(k_qp - k_pq) / denominator
    holonomy = k_qp @ dagger(k_pq)
    curvature_holonomy = frobenius(holonomy - np.eye(ab.shape[0])) / denominator
    unitarity = max(
        frobenius(dagger(t) @ t - np.eye(t.shape[0]))
        for t in (t_ab, t_bd, t_ac, t_cd)
    )
    return {
        "curvature": curvature,
        "curvature_holonomy": curvature_holonomy,
        "formula_error": abs(curvature - curvature_holonomy),
        "eta_edge": min(float(np.min(s)) for s in (s_ab, s_bd, s_ac, s_cd)),
        "edge_singular_values": (s_ab, s_bd, s_ac, s_cd),
        "holonomy_phases": np.sort(np.angle(np.linalg.eigvals(holonomy))),
        "unitarity_error": unitarity,
    }


def random_unitary(rng: np.random.Generator, dimension: int) -> Array:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(phases.conj())


def gauge_error(overlaps: Sequence[Array], reference: float) -> float:
    rng = np.random.default_rng(20260811)
    ua, ub, uc, ud = (random_unitary(rng, SELECTED_M) for _ in range(4))
    ab, bd, ac, cd = overlaps
    transformed = (
        dagger(ua) @ ab @ ub,
        dagger(ub) @ bd @ ud,
        dagger(ua) @ ac @ uc,
        dagger(uc) @ cd @ ud,
    )
    value = plaquette_metrics(*transformed)["curvature"]
    return abs(float(value) - reference)


def import_modules() -> tuple[Any, Any]:
    try:
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError("PySCF is required for production many-electron overlaps.") from exc
    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError("This PySCF version does not provide fci.addons.overlap.")
    return fci, gto


def molecule(lam: float, basis: str, gto: Any) -> Any:
    y = 2.540 - 0.46 * lam
    return gto.M(
        atom=f"""
        Be  0.000000000000  0.000000000000  0.000000000000
        H  {lam: .12f}     {y: .12f}        0.000000000000
        H  {lam: .12f}    {-y: .12f}        0.000000000000
        """,
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=0,
        symmetry="C2v",
        verbose=0,
    )


def string_index_map(fci: Any, norb: int, nelec: int) -> dict[int, int]:
    strings = np.asarray(fci.cistring.make_strings(range(norb), nelec)).reshape(-1)
    return {int(value): index for index, value in enumerate(strings)}


def embed_core(ci_vector: Array, solution: Solution, fci: Any) -> tuple[Array, int, tuple[int, int]]:
    if solution.ncore == 0:
        return np.asarray(ci_vector), solution.ncas, solution.nelecas
    nfull = solution.ncore + solution.ncas
    nelec_full = (
        solution.ncore + solution.nelecas[0],
        solution.ncore + solution.nelecas[1],
    )
    active_alpha = np.asarray(
        fci.cistring.make_strings(range(solution.ncas), solution.nelecas[0])
    ).reshape(-1)
    active_beta = np.asarray(
        fci.cistring.make_strings(range(solution.ncas), solution.nelecas[1])
    ).reshape(-1)
    full_alpha = string_index_map(fci, nfull, nelec_full[0])
    full_beta = string_index_map(fci, nfull, nelec_full[1])
    core_bits = (1 << solution.ncore) - 1
    alpha_indices = [full_alpha[core_bits | (int(value) << solution.ncore)] for value in active_alpha]
    beta_indices = [full_beta[core_bits | (int(value) << solution.ncore)] for value in active_beta]
    result = np.zeros((len(full_alpha), len(full_beta)), dtype=np.asarray(ci_vector).dtype)
    active_matrix = np.asarray(ci_vector).reshape(len(active_alpha), len(active_beta))
    result[np.ix_(alpha_indices, beta_indices)] = active_matrix
    return result, nfull, nelec_full


def state_overlap_matrix(
    x: Solution,
    y: Solution,
    indices_x: Sequence[int],
    indices_y: Sequence[int],
    fci: Any,
    gto: Any,
    basis: str,
) -> tuple[Array, Array]:
    if x.ncas != y.ncas or x.nelecas != y.nelecas or x.ncore != y.ncore:
        raise ValueError("Every edge must use the same core and active-space dimensions.")
    mol_x = molecule(x.lam, basis, gto)
    mol_y = molecule(y.lam, basis, gto)
    cross_ao = gto.intor_cross("int1e_ovlp", mol_x, mol_y)
    occupied_x = x.mo_coeff[:, : x.ncore + x.ncas]
    occupied_y = y.mo_coeff[:, : y.ncore + y.ncas]
    orbital_overlap = dagger(occupied_x) @ cross_ao @ occupied_y
    result = np.empty((len(indices_x), len(indices_y)), dtype=np.complex128)
    for row, i in enumerate(indices_x):
        ci_x, norb_x, nelec_x = embed_core(x.ci_states[i], x, fci)
        for col, j in enumerate(indices_y):
            ci_y, norb_y, nelec_y = embed_core(y.ci_states[j], y, fci)
            if norb_x != norb_y or nelec_x != nelec_y:
                raise ValueError("Embedded determinant spaces differ.")
            result[row, col] = fci.addons.overlap(
                ci_x, ci_y, norb_x, nelec_x, s=orbital_overlap
            )
    return result, orbital_overlap


def best_sa4_subset(
    sa3: Solution,
    sa4: Solution,
    fci: Any,
    gto: Any,
    basis: str,
) -> dict[str, Any]:
    full, _orbitals = state_overlap_matrix(
        sa3, sa4, range(sa3.nroots), range(sa4.nroots), fci, gto, basis
    )
    best = None
    for subset in itertools.combinations(range(sa4.nroots), sa3.nroots):
        matrix = full[:, subset]
        singular = np.linalg.svd(matrix, compute_uv=False)
        score = float(np.sum(singular**2))
        candidate = (score, subset, singular, matrix)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No SA4 subset could be selected.")
    score, subset, singular, matrix = best
    return {
        "subset": tuple(int(value) for value in subset),
        "singular_values": singular,
        "score": score,
        "matrix": matrix,
    }


def leakage(singular: Array, dimension: int = SELECTED_M) -> float:
    return 1.0 - float(np.sum(np.asarray(singular) ** 2)) / dimension


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def branch_map(
    solutions: Sequence[Solution], fci: Any
) -> list[dict[str, Any]]:
    by_key = {
        (round(item.lam, 10), item.protocol, item.source): item for item in solutions
    }
    rows: list[dict[str, Any]] = []
    lambdas = sorted({round(item.lam, 10) for item in solutions})
    comparisons = {
        "SA3": (("upper_down", "lower_down"), ("upper_down", "unique_up"), ("lower_down", "unique_up")),
        "SA4": (("upper_down", "upper_up"),),
    }
    for lam in lambdas:
        for protocol, pairs in comparisons.items():
            for source_x, source_y in pairs:
                x = by_key.get((lam, protocol, source_x))
                y = by_key.get((lam, protocol, source_y))
                if x is None or y is None:
                    continue
                values = pair_metrics(x, y, fci)
                rows.append(
                    {
                        "lambda_bohr": lam,
                        "protocol": protocol,
                        "source_x": source_x,
                        "source_y": source_y,
                        "delta_e_x_minus_y": x.e_average - y.e_average,
                        "delta_e_absolute": values["delta_e_average"],
                        "active_angle_max_rad": values["active_angle_max_rad"],
                        "state_sigma_min": values["state_sigma_min"],
                        "state_leakage": values["state_leakage"],
                        "rdm1_procrustes_fro": values["rdm1_procrustes_fro"],
                        "noon_rms": values["noon_rms"],
                    }
                )
    return rows


@dataclass(frozen=True)
class Route:
    name: str
    sa3_source: str
    sa4_source: str


def analyze_route(
    route: Route,
    solutions: Sequence[Solution],
    fci: Any,
    gto: Any,
    basis: str,
) -> list[dict[str, Any]]:
    by_key = {
        (round(item.lam, 10), item.protocol, item.source): item for item in solutions
    }
    lambdas = sorted(
        set(lam for lam, protocol, source in by_key if protocol == "SA3" and source == route.sa3_source)
        & set(lam for lam, protocol, source in by_key if protocol == "SA4" and source == route.sa4_source)
    )
    selected: dict[float, tuple[int, ...]] = {}
    for lam in lambdas:
        sa3 = by_key[(lam, "SA3", route.sa3_source)]
        sa4 = by_key[(lam, "SA4", route.sa4_source)]
        selected[lam] = best_sa4_subset(sa3, sa4, fci, gto, basis)["subset"]

    rows: list[dict[str, Any]] = []
    for q0, q1 in zip(lambdas[:-1], lambdas[1:]):
        a = by_key[(q0, "SA3", route.sa3_source)]
        b = by_key[(q1, "SA3", route.sa3_source)]
        c = by_key[(q0, "SA4", route.sa4_source)]
        d = by_key[(q1, "SA4", route.sa4_source)]
        subset_c = selected[q0]
        subset_d = selected[q1]
        ab, active_ab = state_overlap_matrix(a, b, range(3), range(3), fci, gto, basis)
        bd, active_bd = state_overlap_matrix(b, d, range(3), subset_d, fci, gto, basis)
        ac, active_ac = state_overlap_matrix(a, c, range(3), subset_c, fci, gto, basis)
        cd, active_cd = state_overlap_matrix(c, d, subset_c, subset_d, fci, gto, basis)
        overlaps = (ab, bd, ac, cd)
        values = plaquette_metrics(*overlaps)
        singular = values["edge_singular_values"]
        reverse_errors = []
        for x, y, ix, iy, forward in (
            (a, b, range(3), range(3), ab),
            (b, d, range(3), subset_d, bd),
            (a, c, range(3), subset_c, ac),
            (c, d, subset_c, subset_d, cd),
        ):
            reverse, _active = state_overlap_matrix(y, x, iy, ix, fci, gto, basis)
            reverse_errors.append(frobenius(reverse - dagger(forward)))
        active_singular = tuple(
            np.linalg.svd(matrix, compute_uv=False)
            for matrix in (active_ab, active_bd, active_ac, active_cd)
        )
        active_angles = tuple(
            float(np.max(np.arccos(np.clip(values_s, 0.0, 1.0))))
            for values_s in active_singular
        )
        rows.append(
            {
                "route": route.name,
                "q0_bohr": q0,
                "q1_bohr": q1,
                "delta_q_bohr": q1 - q0,
                "curvature": values["curvature"],
                "eta_edge": values["eta_edge"],
                "sigma_AB": float(np.min(singular[0])),
                "sigma_BD": float(np.min(singular[1])),
                "sigma_AC": float(np.min(singular[2])),
                "sigma_CD": float(np.min(singular[3])),
                "leakage_AB": leakage(singular[0]),
                "leakage_BD": leakage(singular[1]),
                "leakage_AC": leakage(singular[2]),
                "leakage_CD": leakage(singular[3]),
                "active_angle_AB_rad": active_angles[0],
                "active_angle_BD_rad": active_angles[1],
                "active_angle_AC_rad": active_angles[2],
                "active_angle_CD_rad": active_angles[3],
                "selected_SA4_q0": ";".join(str(i) for i in subset_c),
                "selected_SA4_q1": ";".join(str(i) for i in subset_d),
                "holonomy_phases_rad": vector_text(values["holonomy_phases"]),
                "curvature_formula_error": values["formula_error"],
                "random_gauge_error": gauge_error(overlaps, values["curvature"]),
                "polar_unitarity_error": values["unitarity_error"],
                "reverse_overlap_error": max(reverse_errors),
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    fci, gto = import_modules()
    config_path = args.workdir / "B2_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    basis = str(config["basis"])
    solutions = [
        load_solution(path)
        for path in sorted((args.workdir / "solutions").glob("A1/SA*/*/lambda_*.npz"))
    ]
    if not solutions:
        raise RuntimeError("No B2 solutions found.")
    prefix = args.workdir / args.output_prefix

    branch_rows = branch_map(solutions, fci)
    branch_path = prefix.with_name(prefix.name + "_branch_map.csv")
    rows_to_csv(
        branch_path,
        branch_rows,
        [
            "lambda_bohr", "protocol", "source_x", "source_y",
            "delta_e_x_minus_y", "delta_e_absolute", "active_angle_max_rad",
            "state_sigma_min", "state_leakage", "rdm1_procrustes_fro", "noon_rms",
        ],
    )

    routes = (
        Route("upper_down", "upper_down", "upper_down"),
        Route("unique_up", "unique_up", "upper_up"),
    )
    plaquette_rows: list[dict[str, Any]] = []
    for route in routes:
        plaquette_rows.extend(analyze_route(route, solutions, fci, gto, basis))
    plaquette_path = prefix.with_name(prefix.name + "_plaquettes.csv")
    fields = [
        "route", "q0_bohr", "q1_bohr", "delta_q_bohr", "curvature", "eta_edge",
        "sigma_AB", "sigma_BD", "sigma_AC", "sigma_CD", "leakage_AB",
        "leakage_BD", "leakage_AC", "leakage_CD", "active_angle_AB_rad",
        "active_angle_BD_rad", "active_angle_AC_rad", "active_angle_CD_rad",
        "selected_SA4_q0", "selected_SA4_q1", "holonomy_phases_rad",
        "curvature_formula_error", "random_gauge_error", "polar_unitarity_error",
        "reverse_overlap_error",
    ]
    rows_to_csv(plaquette_path, plaquette_rows, fields)

    summary: dict[str, Any] = {
        "gate": "BeH2 B2 fine SA3-fold and mixed-plaquette refinement",
        "status": "ANALYZED",
        "solution_count": len(solutions),
        "branch_comparison_count": len(branch_rows),
        "plaquette_count": len(plaquette_rows),
        "routes": {},
        "validation": {
            "maximum_formula_error": max(row["curvature_formula_error"] for row in plaquette_rows),
            "maximum_random_gauge_error": max(row["random_gauge_error"] for row in plaquette_rows),
            "maximum_polar_unitarity_error": max(row["polar_unitarity_error"] for row in plaquette_rows),
            "maximum_reverse_overlap_error": max(row["reverse_overlap_error"] for row in plaquette_rows),
        },
    }
    for route in routes:
        group = [row for row in plaquette_rows if row["route"] == route.name]
        peak = max(group, key=lambda row: row["curvature"])
        conditioned = [row for row in group if row["eta_edge"] >= 0.85]
        peak_conditioned = max(conditioned, key=lambda row: row["curvature"]) if conditioned else None
        summary["routes"][route.name] = {
            "maximum_curvature": peak["curvature"],
            "maximum_curvature_interval_bohr": [peak["q0_bohr"], peak["q1_bohr"]],
            "eta_at_maximum": peak["eta_edge"],
            "minimum_eta": min(row["eta_edge"] for row in group),
            "maximum_curvature_with_eta_ge_0p85": None if peak_conditioned is None else peak_conditioned["curvature"],
            "conditioned_peak_interval_bohr": None if peak_conditioned is None else [peak_conditioned["q0_bohr"], peak_conditioned["q1_bohr"]],
        }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"branch": branch_path, "plaquettes": plaquette_path, "summary": summary_path}


def self_test() -> int:
    identity = np.eye(SELECTED_M)
    values = plaquette_metrics(identity, identity, identity, identity)
    if abs(values["curvature"]) > 1.0e-14:
        raise AssertionError("Identity plaquette has nonzero curvature.")
    if gauge_error((identity, identity, identity, identity), 0.0) > 1.0e-12:
        raise AssertionError("Identity gauge test failed.")
    print("BeH2 B2 plaquette analyzer self-test: PASS")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    outputs = analyze(args)
    for label, path in outputs.items():
        print(f"{label:12s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
