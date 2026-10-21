#!/usr/bin/env python3
"""Analyze projection-safe BeH2 B3R plaquettes at several step sizes."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from BeH2_B0_analyze import Solution, load_solution, pair_metrics, vector_text


Array = np.ndarray
SELECTED_M = 3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze BeH2 B3R multiscale plaquettes.")
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B3R_work"))
    p.add_argument("--output-prefix", default="B3R")
    p.add_argument("--strides", default="1,2,4,8")
    p.add_argument("--self-test", action="store_true")
    return p


def parse_strides(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in text.split(",") if item.strip())))
    if not values or values[0] <= 0:
        raise ValueError("Strides must be positive integers.")
    return values


def dagger(matrix: Array) -> Array:
    return np.asarray(matrix).conj().T


def frobenius(matrix: Array) -> float:
    return float(np.linalg.norm(np.asarray(matrix), ord="fro"))


def polar_transport(overlap: Array) -> tuple[Array, Array]:
    u, singular, vh = np.linalg.svd(overlap, full_matrices=False)
    return u @ vh, singular


def normalized_difference(x: Array, y: Array) -> float:
    denominator = frobenius(x) + frobenius(y)
    if denominator == 0.0:
        raise ZeroDivisionError("Both path matrices have zero Frobenius norm.")
    return frobenius(x - y) / denominator


def plaquette_metrics(ab: Array, bd: Array, ac: Array, cd: Array) -> dict[str, Any]:
    t_ab, s_ab = polar_transport(ab)
    t_bd, s_bd = polar_transport(bd)
    t_ac, s_ac = polar_transport(ac)
    t_cd, s_cd = polar_transport(cd)
    k_qp = t_ab @ t_bd
    k_pq = t_ac @ t_cd
    curvature = normalized_difference(k_qp, k_pq)
    holonomy = k_qp @ dagger(k_pq)
    curvature_holonomy = normalized_difference(holonomy, np.eye(ab.shape[0]))

    raw_qp = ab @ bd
    raw_pq = ac @ cd
    raw_nonclosure = normalized_difference(raw_qp, raw_pq)
    unitarity = max(
        frobenius(dagger(t) @ t - np.eye(t.shape[0]))
        for t in (t_ab, t_bd, t_ac, t_cd)
    )
    return {
        "curvature": curvature,
        "curvature_holonomy": curvature_holonomy,
        "formula_error": abs(curvature - curvature_holonomy),
        "raw_path_nonclosure": raw_nonclosure,
        "raw_path_norm_qp": frobenius(raw_qp),
        "raw_path_norm_pq": frobenius(raw_pq),
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


def gauge_errors(overlaps: Sequence[Array], reference: dict[str, Any]) -> tuple[float, float]:
    rng = np.random.default_rng(20260811)
    ua, ub, uc, ud = (random_unitary(rng, SELECTED_M) for _ in range(4))
    ab, bd, ac, cd = overlaps
    transformed = (
        dagger(ua) @ ab @ ub,
        dagger(ub) @ bd @ ud,
        dagger(ua) @ ac @ uc,
        dagger(uc) @ cd @ ud,
    )
    value = plaquette_metrics(*transformed)
    return (
        abs(float(value["curvature"]) - float(reference["curvature"])),
        abs(
            float(value["raw_path_nonclosure"])
            - float(reference["raw_path_nonclosure"])
        ),
    )


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
    alpha_indices = [
        full_alpha[core_bits | (int(value) << solution.ncore)]
        for value in active_alpha
    ]
    beta_indices = [
        full_beta[core_bits | (int(value) << solution.ncore)]
        for value in active_beta
    ]
    result = np.zeros(
        (len(full_alpha), len(full_beta)), dtype=np.asarray(ci_vector).dtype
    )
    active_matrix = np.asarray(ci_vector).reshape(
        len(active_alpha), len(active_beta)
    )
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
    overlap: Callable[[Solution, Solution, Sequence[int], Sequence[int]], tuple[Array, Array]],
) -> dict[str, Any]:
    full, _orbitals = overlap(
        sa3, sa4, range(sa3.nroots), range(sa4.nroots)
    )
    best = None
    for subset in itertools.combinations(range(sa4.nroots), sa3.nroots):
        matrix = full[:, subset]
        singular = np.linalg.svd(matrix, compute_uv=False)
        score = float(np.sum(singular**2))
        candidate = (score, subset, singular)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No SA4 subset could be selected.")
    score, subset, singular = best
    return {
        "subset": tuple(int(value) for value in subset),
        "singular_values": singular,
        "score": score,
    }


def leakage(singular: Array, dimension: int = SELECTED_M) -> float:
    return 1.0 - float(np.sum(np.asarray(singular) ** 2)) / dimension


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def branch_map(solutions: Sequence[Solution], fci: Any) -> list[dict[str, Any]]:
    by_key = {
        (round(item.lam, 10), item.protocol, item.source): item for item in solutions
    }
    rows: list[dict[str, Any]] = []
    lambdas = sorted({round(item.lam, 10) for item in solutions})
    comparisons = {
        "SA3": (("upper_down", "unique_up"),),
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
                        "macroiterations_x": x.macroiterations,
                        "macroiterations_y": y.macroiterations,
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
    strides: Sequence[int],
    fci: Any,
    gto: Any,
    basis: str,
) -> list[dict[str, Any]]:
    by_key = {
        (round(item.lam, 10), item.protocol, item.source): item for item in solutions
    }
    lambdas = sorted(
        set(
            lam
            for lam, protocol, source in by_key
            if protocol == "SA3" and source == route.sa3_source
        )
        & set(
            lam
            for lam, protocol, source in by_key
            if protocol == "SA4" and source == route.sa4_source
        )
    )
    if len(lambdas) < 2:
        raise RuntimeError(f"Route {route.name} has fewer than two common geometries.")

    cache: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], tuple[Array, Array]] = {}

    def overlap(
        x: Solution, y: Solution, ix: Sequence[int], iy: Sequence[int]
    ) -> tuple[Array, Array]:
        key = (str(x.path), str(y.path), tuple(ix), tuple(iy))
        if key not in cache:
            cache[key] = state_overlap_matrix(x, y, ix, iy, fci, gto, basis)
        return cache[key]

    selected: dict[float, tuple[int, ...]] = {}
    for lam in lambdas:
        sa3 = by_key[(lam, "SA3", route.sa3_source)]
        sa4 = by_key[(lam, "SA4", route.sa4_source)]
        selected[lam] = best_sa4_subset(sa3, sa4, overlap)["subset"]

    rows: list[dict[str, Any]] = []
    for stride in strides:
        if stride >= len(lambdas):
            continue
        for index in range(len(lambdas) - stride):
            q0 = lambdas[index]
            q1 = lambdas[index + stride]
            a = by_key[(q0, "SA3", route.sa3_source)]
            b = by_key[(q1, "SA3", route.sa3_source)]
            c = by_key[(q0, "SA4", route.sa4_source)]
            d = by_key[(q1, "SA4", route.sa4_source)]
            subset_c = selected[q0]
            subset_d = selected[q1]
            ab, active_ab = overlap(a, b, range(3), range(3))
            bd, active_bd = overlap(b, d, range(3), subset_d)
            ac, active_ac = overlap(a, c, range(3), subset_c)
            cd, active_cd = overlap(c, d, subset_c, subset_d)
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
                reverse, _active = overlap(y, x, iy, ix)
                reverse_errors.append(frobenius(reverse - dagger(forward)))
            active_singular = tuple(
                np.linalg.svd(matrix, compute_uv=False)
                for matrix in (active_ab, active_bd, active_ac, active_cd)
            )
            active_angles = tuple(
                float(np.max(np.arccos(np.clip(items, 0.0, 1.0))))
                for items in active_singular
            )
            edge_leakage = tuple(leakage(items) for items in singular)
            gauge_c, gauge_raw = gauge_errors(overlaps, values)
            rows.append(
                {
                    "route": route.name,
                    "stride": stride,
                    "q0_bohr": q0,
                    "q1_bohr": q1,
                    "q_mid_bohr": 0.5 * (q0 + q1),
                    "delta_q_bohr": q1 - q0,
                    "curvature": values["curvature"],
                    "raw_path_nonclosure": values["raw_path_nonclosure"],
                    "raw_path_norm_qp": values["raw_path_norm_qp"],
                    "raw_path_norm_pq": values["raw_path_norm_pq"],
                    "eta_edge": values["eta_edge"],
                    "sigma_AB": float(np.min(singular[0])),
                    "sigma_BD": float(np.min(singular[1])),
                    "sigma_AC": float(np.min(singular[2])),
                    "sigma_CD": float(np.min(singular[3])),
                    "leakage_AB": edge_leakage[0],
                    "leakage_BD": edge_leakage[1],
                    "leakage_AC": edge_leakage[2],
                    "leakage_CD": edge_leakage[3],
                    "nuclear_leakage_imbalance": abs(edge_leakage[0] - edge_leakage[3]),
                    "protocol_leakage_imbalance": abs(edge_leakage[2] - edge_leakage[1]),
                    "active_angle_AB_rad": active_angles[0],
                    "active_angle_BD_rad": active_angles[1],
                    "active_angle_AC_rad": active_angles[2],
                    "active_angle_CD_rad": active_angles[3],
                    "selected_SA4_q0": ";".join(str(i) for i in subset_c),
                    "selected_SA4_q1": ";".join(str(i) for i in subset_d),
                    "holonomy_phases_rad": vector_text(values["holonomy_phases"]),
                    "curvature_formula_error": values["formula_error"],
                    "curvature_gauge_error": gauge_c,
                    "raw_nonclosure_gauge_error": gauge_raw,
                    "polar_unitarity_error": values["unitarity_error"],
                    "reverse_overlap_error": max(reverse_errors),
                }
            )
    return rows


def scaling_rows(
    rows: Sequence[dict[str, Any]], routes: Sequence[Route]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for route in routes:
        route_rows = [row for row in rows if row["route"] == route.name]
        for stride in sorted({int(row["stride"]) for row in route_rows}):
            group = [row for row in route_rows if int(row["stride"]) == stride]
            peak_c = max(group, key=lambda row: row["curvature"])
            peak_raw = max(group, key=lambda row: row["raw_path_nonclosure"])
            peak_leakage = max(group, key=lambda row: row["leakage_AB"])
            median_c = float(np.median([row["curvature"] for row in group]))
            output.append(
                {
                    "route": route.name,
                    "stride": stride,
                    "delta_q_bohr": peak_c["delta_q_bohr"],
                    "curvature_max": peak_c["curvature"],
                    "curvature_median": median_c,
                    "curvature_peak_to_median": peak_c["curvature"] / max(median_c, 1.0e-300),
                    "curvature_peak_q0_bohr": peak_c["q0_bohr"],
                    "curvature_peak_q1_bohr": peak_c["q1_bohr"],
                    "eta_at_curvature_peak": peak_c["eta_edge"],
                    "raw_path_nonclosure_max": peak_raw["raw_path_nonclosure"],
                    "raw_peak_q0_bohr": peak_raw["q0_bohr"],
                    "raw_peak_q1_bohr": peak_raw["q1_bohr"],
                    "leakage_AB_max": peak_leakage["leakage_AB"],
                    "leakage_peak_q0_bohr": peak_leakage["q0_bohr"],
                    "leakage_peak_q1_bohr": peak_leakage["q1_bohr"],
                    "minimum_eta": min(row["eta_edge"] for row in group),
                }
            )
    return output


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    fci, gto = import_modules()
    strides = parse_strides(args.strides)
    config_path = args.workdir / "B3R_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    basis = str(config["basis"])
    all_solutions = [
        load_solution(path)
        for path in sorted((args.workdir / "solutions").glob("A1/SA*/*/lambda_*.npz"))
    ]
    analysis_grid = {round(float(value), 10) for value in config["grid_bohr"]}
    solutions = [
        item for item in all_solutions if round(float(item.lam), 10) in analysis_grid
    ]
    if not solutions:
        raise RuntimeError("No B3R solutions found on the configured analysis grid.")
    expected = 4 * len(analysis_grid)
    if len(solutions) != expected:
        raise RuntimeError(
            f"Expected {expected} analysis-grid solutions, found {len(solutions)}."
        )
    prefix = args.workdir / args.output_prefix

    branch_rows = branch_map(solutions, fci)
    branch_path = prefix.with_name(prefix.name + "_branch_map.csv")
    branch_fields = [
        "lambda_bohr", "protocol", "source_x", "source_y",
        "delta_e_x_minus_y", "delta_e_absolute", "active_angle_max_rad",
        "state_sigma_min", "state_leakage", "rdm1_procrustes_fro", "noon_rms",
        "macroiterations_x", "macroiterations_y",
    ]
    rows_to_csv(branch_path, branch_rows, branch_fields)

    routes = (
        Route("upper_down", "upper_down", "upper_down"),
        Route("unique_up", "unique_up", "upper_up"),
    )
    plaquette_rows: list[dict[str, Any]] = []
    for route in routes:
        plaquette_rows.extend(
            analyze_route(route, solutions, strides, fci, gto, basis)
        )
    plaquette_path = prefix.with_name(prefix.name + "_multiscale_plaquettes.csv")
    plaquette_fields = [
        "route", "stride", "q0_bohr", "q1_bohr", "q_mid_bohr", "delta_q_bohr",
        "curvature", "raw_path_nonclosure", "raw_path_norm_qp", "raw_path_norm_pq",
        "eta_edge", "sigma_AB", "sigma_BD", "sigma_AC", "sigma_CD",
        "leakage_AB", "leakage_BD", "leakage_AC", "leakage_CD",
        "nuclear_leakage_imbalance", "protocol_leakage_imbalance",
        "active_angle_AB_rad", "active_angle_BD_rad", "active_angle_AC_rad",
        "active_angle_CD_rad", "selected_SA4_q0", "selected_SA4_q1",
        "holonomy_phases_rad", "curvature_formula_error", "curvature_gauge_error",
        "raw_nonclosure_gauge_error", "polar_unitarity_error", "reverse_overlap_error",
    ]
    rows_to_csv(plaquette_path, plaquette_rows, plaquette_fields)

    scale_rows = scaling_rows(plaquette_rows, routes)
    scaling_path = prefix.with_name(prefix.name + "_scaling.csv")
    scaling_fields = [
        "route", "stride", "delta_q_bohr", "curvature_max", "curvature_median",
        "curvature_peak_to_median", "curvature_peak_q0_bohr",
        "curvature_peak_q1_bohr", "eta_at_curvature_peak",
        "raw_path_nonclosure_max", "raw_peak_q0_bohr", "raw_peak_q1_bohr",
        "leakage_AB_max", "leakage_peak_q0_bohr", "leakage_peak_q1_bohr",
        "minimum_eta",
    ]
    rows_to_csv(scaling_path, scale_rows, scaling_fields)

    upper_branch = [
        row
        for row in branch_rows
        if row["protocol"] == "SA3"
        and row["source_x"] == "upper_down"
        and row["source_y"] == "unique_up"
    ]
    branch_jump = max(upper_branch, key=lambda row: row["state_leakage"])
    qmax = max(analysis_grid)
    upper_at_qmax = next(
        row for row in upper_branch if round(float(row["lambda_bohr"]), 10) == qmax
    )
    upper_branch_retained = bool(
        upper_at_qmax["delta_e_absolute"] >= 1.0e-3
        and upper_at_qmax["state_leakage"] >= 1.0e-2
    )
    selected_subsets = sorted(
        {
            (row["selected_SA4_q0"], row["selected_SA4_q1"])
            for row in plaquette_rows
        }
    )
    summary: dict[str, Any] = {
        "gate": "BeH2 B3R projection-safe spinodal localization and scaling",
        "status": (
            "PASS_BRANCH_RETAINED"
            if upper_branch_retained
            else "FAIL_BRANCH_LOST"
        ),
        "solution_count": len(solutions),
        "branch_comparison_count": len(branch_rows),
        "plaquette_count": len(plaquette_rows),
        "strides": list(strides),
        "selected_SA4_subsets_q0_q1": [list(items) for items in selected_subsets],
        "upper_branch_retained_at_qmax": {
            "passed": upper_branch_retained,
            "lambda_bohr": upper_at_qmax["lambda_bohr"],
            "delta_e_absolute": upper_at_qmax["delta_e_absolute"],
            "state_leakage": upper_at_qmax["state_leakage"],
            "state_sigma_min": upper_at_qmax["state_sigma_min"],
        },
        "largest_SA3_history_separation": {
            "lambda_bohr": branch_jump["lambda_bohr"],
            "state_leakage": branch_jump["state_leakage"],
            "state_sigma_min": branch_jump["state_sigma_min"],
            "delta_e_absolute": branch_jump["delta_e_absolute"],
            "active_angle_max_rad": branch_jump["active_angle_max_rad"],
        },
        "scaling": scale_rows,
        "validation": {
            "maximum_formula_error": max(row["curvature_formula_error"] for row in plaquette_rows),
            "maximum_curvature_gauge_error": max(row["curvature_gauge_error"] for row in plaquette_rows),
            "maximum_raw_nonclosure_gauge_error": max(row["raw_nonclosure_gauge_error"] for row in plaquette_rows),
            "maximum_polar_unitarity_error": max(row["polar_unitarity_error"] for row in plaquette_rows),
            "maximum_reverse_overlap_error": max(row["reverse_overlap_error"] for row in plaquette_rows),
            "maximum_curvature": max(row["curvature"] for row in plaquette_rows),
            "maximum_raw_path_nonclosure": max(row["raw_path_nonclosure"] for row in plaquette_rows),
        },
    }
    if summary["validation"]["maximum_curvature"] > 1.0 + 1.0e-12:
        raise AssertionError("Polar-unitary nonclosure exceeds its upper bound.")
    if summary["validation"]["maximum_raw_path_nonclosure"] > 1.0 + 1.0e-12:
        raise AssertionError("Raw path nonclosure exceeds its upper bound.")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "branch": branch_path,
        "plaquettes": plaquette_path,
        "scaling": scaling_path,
        "summary": summary_path,
    }


def self_test() -> int:
    identity = np.eye(SELECTED_M)
    values = plaquette_metrics(identity, identity, identity, identity)
    if abs(values["curvature"]) > 1.0e-14:
        raise AssertionError("Identity plaquette has nonzero polar nonclosure.")
    if abs(values["raw_path_nonclosure"]) > 1.0e-14:
        raise AssertionError("Identity plaquette has nonzero raw nonclosure.")
    ab = np.diag([1.0, 0.9, 0.8])
    bd = np.eye(SELECTED_M)
    ac = np.eye(SELECTED_M)
    cd = np.diag([0.95, 0.85, 0.75])
    values = plaquette_metrics(ab, bd, ac, cd)
    gauge_c, gauge_raw = gauge_errors((ab, bd, ac, cd), values)
    if gauge_c > 1.0e-12 or gauge_raw > 1.0e-12:
        raise AssertionError("Gauge-invariance self-test failed.")
    if not (0.0 <= values["curvature"] <= 1.0):
        raise AssertionError("Polar nonclosure bound failed.")
    if not (0.0 <= values["raw_path_nonclosure"] <= 1.0):
        raise AssertionError("Raw nonclosure bound failed.")
    if parse_strides("8,2,1,2,4") != (1, 2, 4, 8):
        raise AssertionError("Stride parsing failed.")
    print("BeH2 B3R analyzer self-test: PASS")
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
