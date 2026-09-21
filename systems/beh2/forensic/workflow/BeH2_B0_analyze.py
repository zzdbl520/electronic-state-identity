#!/usr/bin/env python3
"""Analyze BeH2 B0 stationary solutions and identify the switching block.

The script compares all initialization sources at the same geometry, irrep,
and state-averaging protocol using state-averaged energy, active-space
principal angles, genuine many-electron subspace overlaps, Procrustes-aligned
1-RDM distances, and natural-orbital occupation differences.  It also compares
the SA3 manifold with the best three-state subset extracted from SA4.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TARGET_SOURCES = ("left", "right", "common_sa2", "rhf")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Analyze the BeH2 B0 multi-initialization solution set."
    )
    result.add_argument("--workdir", type=Path, default=Path("BeH2_B0_work"))
    result.add_argument("--output-prefix", default="B0")
    result.add_argument("--energy-tol", type=float, default=1.0e-7)
    result.add_argument("--angle-tol", type=float, default=1.0e-3)
    result.add_argument("--state-leakage-tol", type=float, default=1.0e-6)
    result.add_argument("--rdm-tol", type=float, default=1.0e-4)
    result.add_argument("--noon-tol", type=float, default=1.0e-4)
    result.add_argument("--self-test", action="store_true")
    return result


def import_fci() -> Any:
    try:
        from pyscf import fci
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required because the analysis evaluates genuine "
            "many-electron overlaps with nonorthogonal active orbitals."
        ) from exc
    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError("This PySCF version does not provide fci.addons.overlap.")
    return fci


def text_scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


@dataclass
class Solution:
    path: Path
    lam: float
    irrep: str
    protocol: str
    nroots: int
    source: str
    ncas: int
    nelecas: tuple[int, int]
    ncore: int
    ao_overlap: np.ndarray
    mo_coeff: np.ndarray
    ci_states: np.ndarray
    rdm1_states: np.ndarray
    energies: np.ndarray
    e_average: float
    macroiterations: int
    norm_gorb: float
    norm_ddm: float

    @property
    def active_orbitals(self) -> np.ndarray:
        return self.mo_coeff[:, self.ncore : self.ncore + self.ncas]

    @property
    def average_rdm1(self) -> np.ndarray:
        return np.mean(self.rdm1_states, axis=0)

    @property
    def average_noons(self) -> np.ndarray:
        values = np.linalg.eigvalsh(self.average_rdm1)
        return np.sort(np.real_if_close(values))[::-1]


def load_solution(path: Path) -> Solution:
    with np.load(path, allow_pickle=False) as archive:
        nelecas_array = np.asarray(archive["nelecas"], dtype=int).reshape(-1)
        if nelecas_array.size != 2:
            raise ValueError(f"Expected alpha/beta active electrons in {path}.")
        return Solution(
            path=path,
            lam=float(archive["lambda_bohr"]),
            irrep=text_scalar(archive["irrep"]),
            protocol=text_scalar(archive["protocol"]),
            nroots=int(archive["nroots"]),
            source=text_scalar(archive["source"]),
            ncas=int(archive["ncas"]),
            nelecas=(int(nelecas_array[0]), int(nelecas_array[1])),
            ncore=int(archive["ncore"]),
            ao_overlap=np.asarray(archive["ao_overlap"]),
            mo_coeff=np.asarray(archive["mo_coeff"]),
            ci_states=np.asarray(archive["ci_states"]),
            rdm1_states=np.asarray(archive["rdm1_states"]),
            energies=np.asarray(archive["energies"], dtype=float),
            e_average=float(archive["e_average"]),
            macroiterations=int(archive["macroiterations"]),
            norm_gorb=float(archive["norm_gorb"]),
            norm_ddm=float(archive["norm_ddm"]),
        )


def active_overlap(x: Solution, y: Solution) -> np.ndarray:
    if abs(x.lam - y.lam) > 1.0e-9:
        raise ValueError("B0 pairwise comparisons require the same geometry.")
    if x.ncas != y.ncas or x.nelecas != y.nelecas:
        raise ValueError("Active spaces differ.")
    return x.active_orbitals.conj().T @ x.ao_overlap @ y.active_orbitals


def procrustes_rotation(overlap: np.ndarray) -> np.ndarray:
    u, _singular_values, vh = np.linalg.svd(overlap, full_matrices=False)
    v = vh.conj().T
    return v @ u.conj().T


def many_electron_overlap(x: Solution, y: Solution, fci: Any) -> np.ndarray:
    overlap_active = active_overlap(x, y)
    result = np.empty((x.nroots, y.nroots), dtype=np.complex128)
    for i in range(x.nroots):
        for j in range(y.nroots):
            result[i, j] = fci.addons.overlap(
                x.ci_states[i],
                y.ci_states[j],
                x.ncas,
                x.nelecas,
                s=overlap_active,
            )
    return result


def leakage_from_singular_values(values: np.ndarray, dimension: int) -> float:
    return 1.0 - float(np.sum(np.asarray(values, dtype=float) ** 2)) / dimension


def pair_metrics(x: Solution, y: Solution, fci: Any) -> dict[str, Any]:
    overlap_active = active_overlap(x, y)
    active_singular = np.linalg.svd(overlap_active, compute_uv=False)
    clipped = np.clip(active_singular, 0.0, 1.0)
    angles = np.arccos(clipped)
    state_overlap = many_electron_overlap(x, y, fci)
    state_singular = np.linalg.svd(state_overlap, compute_uv=False)
    rotation = procrustes_rotation(overlap_active)
    rdm_y_aligned = rotation.conj().T @ y.average_rdm1 @ rotation
    rdm_distance = float(np.linalg.norm(x.average_rdm1 - rdm_y_aligned))
    noon_distance = float(
        np.sqrt(np.mean((x.average_noons - y.average_noons) ** 2))
    )
    return {
        "delta_e_average": abs(x.e_average - y.e_average),
        "active_sigma_min": float(np.min(active_singular)),
        "active_angle_max_rad": float(np.max(angles)),
        "state_sigma_min": float(np.min(state_singular)),
        "state_leakage": leakage_from_singular_values(state_singular, x.nroots),
        "rdm1_procrustes_fro": rdm_distance,
        "noon_rms": noon_distance,
        "state_overlap": state_overlap,
        "active_singular_values": active_singular,
        "state_singular_values": state_singular,
    }


def same_branch(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metrics["delta_e_average"] <= args.energy_tol
        and metrics["active_angle_max_rad"] <= args.angle_tol
        and metrics["state_leakage"] <= args.state_leakage_tol
        and metrics["rdm1_procrustes_fro"] <= args.rdm_tol
        and metrics["noon_rms"] <= args.noon_tol
    )


class UnionFind:
    def __init__(self, labels: Iterable[str]):
        self.parent = {label: label for label in labels}

    def find(self, label: str) -> str:
        parent = self.parent[label]
        if parent != label:
            self.parent[label] = self.find(parent)
        return self.parent[label]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def vector_text(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.12e}" for value in np.asarray(values).reshape(-1))


def best_sa4_subset(sa3: Solution, sa4: Solution, fci: Any) -> dict[str, Any]:
    full_overlap = many_electron_overlap(sa3, sa4, fci)
    best = None
    for subset in itertools.combinations(range(sa4.nroots), sa3.nroots):
        matrix = full_overlap[:, subset]
        singular = np.linalg.svd(matrix, compute_uv=False)
        score = float(np.sum(singular**2))
        candidate = (score, subset, singular, matrix)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No SA4 subset could be selected.")
    score, subset, singular, matrix = best
    return {
        "selected_sa4_roots": subset,
        "singular_values": singular,
        "sigma_min": float(np.min(singular)),
        "leakage": leakage_from_singular_values(singular, sa3.nroots),
        "score": score,
        "overlap": matrix,
    }


def solution_row(solution: Solution) -> dict[str, Any]:
    row = {
        "lambda_bohr": solution.lam,
        "irrep": solution.irrep,
        "protocol": solution.protocol,
        "source": solution.source,
        "e_average": solution.e_average,
        "macroiterations": solution.macroiterations,
        "norm_gorb": solution.norm_gorb,
        "norm_ddm": solution.norm_ddm,
        "average_noons": vector_text(solution.average_noons),
        "solution_file": str(solution.path.resolve()),
    }
    for index, energy in enumerate(solution.energies):
        row[f"E{index}"] = float(energy)
    return row


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    fci = import_fci()
    paths = sorted((args.workdir / "solutions").glob("*/*/*/lambda_*.npz"))
    solutions = [load_solution(path) for path in paths]
    if not solutions:
        raise RuntimeError(f"No solution NPZ files found below {args.workdir / 'solutions'}.")
    targets = set()
    config_path = args.workdir / "B0_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        targets = {round(float(value), 8) for value in config.get("targets_bohr", [])}
    if targets:
        solutions = [solution for solution in solutions if round(solution.lam, 8) in targets]

    prefix = args.workdir / args.output_prefix
    solution_rows = [solution_row(solution) for solution in solutions]
    energy_keys = sorted(
        {key for row in solution_rows for key in row if key.startswith("E") and key[1:].isdigit()},
        key=lambda item: int(item[1:]),
    )
    solutions_path = prefix.with_name(prefix.name + "_solutions.csv")
    rows_to_csv(
        solutions_path,
        solution_rows,
        [
            "lambda_bohr",
            "irrep",
            "protocol",
            "source",
            "e_average",
            *energy_keys,
            "macroiterations",
            "norm_gorb",
            "norm_ddm",
            "average_noons",
            "solution_file",
        ],
    )

    groups: dict[tuple[float, str, str], list[Solution]] = {}
    for solution in solutions:
        groups.setdefault((solution.lam, solution.irrep, solution.protocol), []).append(solution)

    pair_rows = []
    cluster_rows = []
    switching_candidates = []
    for (lam, irrep, protocol), group in sorted(groups.items()):
        uf = UnionFind(solution.source for solution in group)
        for x, y in itertools.combinations(group, 2):
            metrics = pair_metrics(x, y, fci)
            is_same = same_branch(metrics, args)
            if is_same:
                uf.union(x.source, y.source)
            pair_rows.append(
                {
                    "lambda_bohr": lam,
                    "irrep": irrep,
                    "protocol": protocol,
                    "source_x": x.source,
                    "source_y": y.source,
                    "delta_e_average": metrics["delta_e_average"],
                    "active_sigma_min": metrics["active_sigma_min"],
                    "active_angle_max_rad": metrics["active_angle_max_rad"],
                    "state_sigma_min": metrics["state_sigma_min"],
                    "state_leakage": metrics["state_leakage"],
                    "rdm1_procrustes_fro": metrics["rdm1_procrustes_fro"],
                    "noon_rms": metrics["noon_rms"],
                    "same_branch": is_same,
                    "active_singular_values": vector_text(metrics["active_singular_values"]),
                    "state_singular_values": vector_text(metrics["state_singular_values"]),
                }
            )
        roots = sorted({uf.find(solution.source) for solution in group})
        cluster_id = {root: index + 1 for index, root in enumerate(roots)}
        source_clusters = {}
        for solution in group:
            value = cluster_id[uf.find(solution.source)]
            source_clusters[solution.source] = value
            cluster_rows.append(
                {
                    "lambda_bohr": lam,
                    "irrep": irrep,
                    "protocol": protocol,
                    "source": solution.source,
                    "cluster": value,
                }
            )
        left_right_distinct = (
            "left" in source_clusters
            and "right" in source_clusters
            and source_clusters["left"] != source_clusters["right"]
        )
        if len(set(source_clusters.values())) > 1:
            switching_candidates.append(
                {
                    "lambda_bohr": lam,
                    "irrep": irrep,
                    "protocol": protocol,
                    "number_of_clusters": len(set(source_clusters.values())),
                    "left_right_distinct": left_right_distinct,
                    "source_clusters": source_clusters,
                }
            )

    pairwise_path = prefix.with_name(prefix.name + "_pairwise.csv")
    rows_to_csv(
        pairwise_path,
        pair_rows,
        [
            "lambda_bohr",
            "irrep",
            "protocol",
            "source_x",
            "source_y",
            "delta_e_average",
            "active_sigma_min",
            "active_angle_max_rad",
            "state_sigma_min",
            "state_leakage",
            "rdm1_procrustes_fro",
            "noon_rms",
            "same_branch",
            "active_singular_values",
            "state_singular_values",
        ],
    )
    clusters_path = prefix.with_name(prefix.name + "_clusters.csv")
    rows_to_csv(
        clusters_path,
        cluster_rows,
        ["lambda_bohr", "irrep", "protocol", "source", "cluster"],
    )

    by_key = {(item.lam, item.irrep, item.protocol, item.source): item for item in solutions}
    protocol_rows = []
    lambda_irrep_sources = sorted(
        {
            (item.lam, item.irrep, item.source)
            for item in solutions
            if item.protocol in {"SA3", "SA4"}
        }
    )
    for lam, irrep, source in lambda_irrep_sources:
        sa3 = by_key.get((lam, irrep, "SA3", source))
        sa4 = by_key.get((lam, irrep, "SA4", source))
        if sa3 is None or sa4 is None:
            continue
        result = best_sa4_subset(sa3, sa4, fci)
        protocol_rows.append(
            {
                "lambda_bohr": lam,
                "irrep": irrep,
                "source": source,
                "selected_sa4_roots": ";".join(str(value) for value in result["selected_sa4_roots"]),
                "sigma_min": result["sigma_min"],
                "leakage": result["leakage"],
                "overlap_score": result["score"],
                "singular_values": vector_text(result["singular_values"]),
            }
        )
    protocol_path = prefix.with_name(prefix.name + "_SA3_SA4.csv")
    rows_to_csv(
        protocol_path,
        protocol_rows,
        [
            "lambda_bohr",
            "irrep",
            "source",
            "selected_sa4_roots",
            "sigma_min",
            "leakage",
            "overlap_score",
            "singular_values",
        ],
    )

    candidate_counts: dict[str, int] = {}
    left_right_counts: dict[str, int] = {}
    for item in switching_candidates:
        key = item["irrep"]
        candidate_counts[key] = candidate_counts.get(key, 0) + 1
        if item["left_right_distinct"]:
            left_right_counts[key] = left_right_counts.get(key, 0) + 1
    ranked_irreps = sorted(
        set(candidate_counts) | set(left_right_counts),
        key=lambda irrep: (left_right_counts.get(irrep, 0), candidate_counts.get(irrep, 0)),
        reverse=True,
    )
    summary = {
        "thresholds": {
            "energy_tol_Eh": args.energy_tol,
            "active_angle_tol_rad": args.angle_tol,
            "state_leakage_tol": args.state_leakage_tol,
            "rdm1_fro_tol": args.rdm_tol,
            "noon_rms_tol": args.noon_tol,
        },
        "solution_count": len(solutions),
        "pair_count": len(pair_rows),
        "protocol_comparison_count": len(protocol_rows),
        "switching_candidates": switching_candidates,
        "candidate_counts_by_irrep": candidate_counts,
        "left_right_distinct_counts_by_irrep": left_right_counts,
        "provisional_switching_irrep": ranked_irreps[0] if ranked_irreps else None,
        "interpretation": (
            "Provisional only: inspect pairwise metrics and convergence logs before "
            "locking the switching block. No threshold was tuned to the result."
        ),
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "solutions": solutions_path,
        "pairwise": pairwise_path,
        "clusters": clusters_path,
        "protocol": protocol_path,
        "summary": summary_path,
    }


def self_test() -> int:
    overlap = np.array([[0.0, 1.0], [1.0, 0.0]])
    rotation = procrustes_rotation(overlap)
    if np.linalg.norm(overlap @ rotation - np.eye(2)) > 1.0e-12:
        raise AssertionError("Procrustes rotation failed.")
    values = np.array([1.0, 0.8])
    expected = 1.0 - (1.0 + 0.64) / 2.0
    if abs(leakage_from_singular_values(values, 2) - expected) > 1.0e-12:
        raise AssertionError("Leakage normalization failed.")
    print("BeH2 B0 analyzer self-test: PASS")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    outputs = analyze(args)
    for label, path in outputs.items():
        print(f"{label:10s}: {path}")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    print(
        "provisional switching irrep:",
        summary.get("provisional_switching_irrep"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
