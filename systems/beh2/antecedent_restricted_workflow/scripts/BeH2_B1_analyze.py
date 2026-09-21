#!/usr/bin/env python3
"""Analyze the BeH2 B1 A1 branch-support map.

Outputs retain continuous diagnostics.  The script reports branch merger and
energy-crossing candidates but does not convert microscopic numerical scatter
into extra stationary-solution clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from BeH2_B0_analyze import (
    Solution,
    best_sa4_subset,
    load_solution,
    pair_metrics,
    vector_text,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze BeH2 B1 A1 branch support.")
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B1_work"))
    p.add_argument("--output-prefix", default="B1")
    p.add_argument("--self-test", action="store_true")
    return p


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_overlap_modules() -> tuple[Any, Any]:
    try:
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError("PySCF is required for genuine cross-geometry overlaps.") from exc
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


def cross_metrics(x: Solution, y: Solution, fci: Any, gto: Any, basis: str) -> dict[str, Any]:
    if x.ncas != y.ncas or x.nelecas != y.nelecas or x.nroots != y.nroots:
        raise ValueError("Continuity edges require identical active spaces and dimensions.")
    mol_x = molecule(x.lam, basis, gto)
    mol_y = molecule(y.lam, basis, gto)
    cross_ao = gto.intor_cross("int1e_ovlp", mol_x, mol_y)
    active_x = x.mo_coeff[:, x.ncore : x.ncore + x.ncas]
    active_y = y.mo_coeff[:, y.ncore : y.ncore + y.ncas]
    active = active_x.conj().T @ cross_ao @ active_y
    active_singular = np.linalg.svd(active, compute_uv=False)
    state = np.empty((x.nroots, y.nroots), dtype=np.complex128)
    for i in range(x.nroots):
        for j in range(y.nroots):
            state[i, j] = fci.addons.overlap(
                x.ci_states[i], y.ci_states[j], x.ncas, x.nelecas, s=active
            )
    state_singular = np.linalg.svd(state, compute_uv=False)
    active_angles = np.arccos(np.clip(active_singular, 0.0, 1.0))
    leakage = 1.0 - float(np.sum(state_singular**2)) / x.nroots
    return {
        "delta_e_average_signed": float(y.e_average - x.e_average),
        "active_sigma_min": float(np.min(active_singular)),
        "active_angle_max_rad": float(np.max(active_angles)),
        "state_sigma_min": float(np.min(state_singular)),
        "state_leakage": leakage,
        "active_singular_values": active_singular,
        "state_singular_values": state_singular,
    }


def structurally_same(metrics: dict[str, Any]) -> bool:
    """Loose, scale-separated merger test; energy is intentionally excluded."""
    return bool(
        metrics["active_angle_max_rad"] < 1.0e-2
        and metrics["state_leakage"] < 1.0e-4
        and metrics["rdm1_procrustes_fro"] < 1.0e-2
        and metrics["noon_rms"] < 5.0e-3
    )


def sign_change_brackets(rows: Sequence[dict[str, Any]]) -> list[list[float]]:
    result: list[list[float]] = []
    ordered = sorted(rows, key=lambda row: row["lambda_bohr"])
    for left, right in zip(ordered[:-1], ordered[1:]):
        a = float(left["delta_e_upper_minus_lower"])
        b = float(right["delta_e_upper_minus_lower"])
        if a == 0.0 or b == 0.0 or a * b < 0.0:
            result.append([float(left["lambda_bohr"]), float(right["lambda_bohr"])])
    return result


def transition_brackets(rows: Sequence[dict[str, Any]]) -> list[list[float]]:
    result: list[list[float]] = []
    ordered = sorted(rows, key=lambda row: row["lambda_bohr"])
    for left, right in zip(ordered[:-1], ordered[1:]):
        if bool(left["same_stationary_solution"]) != bool(right["same_stationary_solution"]):
            result.append([float(left["lambda_bohr"]), float(right["lambda_bohr"])])
    return result


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    fci, gto = import_overlap_modules()
    config_path = args.workdir / "B1_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    basis = str(config["basis"])
    solutions = [
        load_solution(path)
        for path in sorted((args.workdir / "solutions").glob("A1/SA*/*/lambda_*.npz"))
    ]
    if not solutions:
        raise RuntimeError("No B1 solution files found.")
    by_key = {
        (round(item.lam, 10), item.protocol, item.source): item for item in solutions
    }
    prefix = args.workdir / args.output_prefix

    branch_rows: list[dict[str, Any]] = []
    for lam in sorted({round(item.lam, 10) for item in solutions}):
        for protocol in ("SA3", "SA4"):
            upper = by_key.get((lam, protocol, "upper"))
            lower = by_key.get((lam, protocol, "lower"))
            if upper is None or lower is None:
                continue
            values = pair_metrics(upper, lower, fci)
            branch_rows.append(
                {
                    "lambda_bohr": lam,
                    "protocol": protocol,
                    "delta_e_upper_minus_lower": upper.e_average - lower.e_average,
                    "delta_e_absolute": values["delta_e_average"],
                    "active_sigma_min": values["active_sigma_min"],
                    "active_angle_max_rad": values["active_angle_max_rad"],
                    "state_sigma_min": values["state_sigma_min"],
                    "state_leakage": values["state_leakage"],
                    "rdm1_procrustes_fro": values["rdm1_procrustes_fro"],
                    "noon_rms": values["noon_rms"],
                    "same_stationary_solution": structurally_same(values),
                    "active_singular_values": vector_text(values["active_singular_values"]),
                    "state_singular_values": vector_text(values["state_singular_values"]),
                }
            )
    branch_path = prefix.with_name(prefix.name + "_branch_separation.csv")
    rows_to_csv(
        branch_path,
        branch_rows,
        [
            "lambda_bohr", "protocol", "delta_e_upper_minus_lower",
            "delta_e_absolute", "active_sigma_min", "active_angle_max_rad",
            "state_sigma_min", "state_leakage", "rdm1_procrustes_fro",
            "noon_rms", "same_stationary_solution", "active_singular_values",
            "state_singular_values",
        ],
    )

    continuity_rows: list[dict[str, Any]] = []
    for protocol in ("SA3", "SA4"):
        for source in ("upper", "lower"):
            group = sorted(
                [x for x in solutions if x.protocol == protocol and x.source == source],
                key=lambda x: x.lam,
            )
            for x, y in zip(group[:-1], group[1:]):
                values = cross_metrics(x, y, fci, gto, basis)
                continuity_rows.append(
                    {
                        "lambda_x_bohr": x.lam,
                        "lambda_y_bohr": y.lam,
                        "protocol": protocol,
                        "source": source,
                        **{key: value for key, value in values.items() if not isinstance(value, np.ndarray)},
                        "active_singular_values": vector_text(values["active_singular_values"]),
                        "state_singular_values": vector_text(values["state_singular_values"]),
                    }
                )
    continuity_path = prefix.with_name(prefix.name + "_continuity.csv")
    rows_to_csv(
        continuity_path,
        continuity_rows,
        [
            "lambda_x_bohr", "lambda_y_bohr", "protocol", "source",
            "delta_e_average_signed", "active_sigma_min", "active_angle_max_rad",
            "state_sigma_min", "state_leakage", "active_singular_values",
            "state_singular_values",
        ],
    )

    protocol_rows: list[dict[str, Any]] = []
    for lam in sorted({round(item.lam, 10) for item in solutions}):
        for source in ("upper", "lower"):
            sa3 = by_key.get((lam, "SA3", source))
            sa4 = by_key.get((lam, "SA4", source))
            if sa3 is None or sa4 is None:
                continue
            result = best_sa4_subset(sa3, sa4, fci)
            protocol_rows.append(
                {
                    "lambda_bohr": lam,
                    "source": source,
                    "selected_sa4_roots": ";".join(str(i) for i in result["selected_sa4_roots"]),
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
        ["lambda_bohr", "source", "selected_sa4_roots", "sigma_min", "leakage", "overlap_score", "singular_values"],
    )

    summary: dict[str, Any] = {
        "gate": "BeH2 B1 A1 stationary-branch support map",
        "status": "ANALYZED",
        "solution_count": len(solutions),
        "branch_comparison_count": len(branch_rows),
        "continuity_edge_count": len(continuity_rows),
        "protocol_comparison_count": len(protocol_rows),
        "merger_test": {
            "active_angle_max_rad": 1.0e-2,
            "state_leakage": 1.0e-4,
            "rdm1_procrustes_fro": 1.0e-2,
            "noon_rms": 5.0e-3,
            "energy_used": False,
        },
        "by_protocol": {},
    }
    for protocol in ("SA3", "SA4"):
        group = [row for row in branch_rows if row["protocol"] == protocol]
        summary["by_protocol"][protocol] = {
            "lambda_points_compared": len(group),
            "distinct_solution_points": sum(not bool(row["same_stationary_solution"]) for row in group),
            "same_solution_points": sum(bool(row["same_stationary_solution"]) for row in group),
            "energy_crossing_brackets_bohr": sign_change_brackets(group),
            "merger_transition_brackets_bohr": transition_brackets(group),
            "signed_energy_difference_range_Eh": [
                min(float(row["delta_e_upper_minus_lower"]) for row in group) if group else None,
                max(float(row["delta_e_upper_minus_lower"]) for row in group) if group else None,
            ],
        }
    if continuity_rows:
        summary["continuity_quality"] = {
            "minimum_state_sigma": min(float(row["state_sigma_min"]) for row in continuity_rows),
            "maximum_state_leakage": max(float(row["state_leakage"]) for row in continuity_rows),
        }
    if protocol_rows:
        summary["protocol_quality"] = {
            source: {
                "minimum_sigma": min(float(row["sigma_min"]) for row in protocol_rows if row["source"] == source),
                "maximum_leakage": max(float(row["leakage"]) for row in protocol_rows if row["source"] == source),
            }
            for source in ("upper", "lower")
        }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "branch": branch_path,
        "continuity": continuity_path,
        "protocol": protocol_path,
        "summary": summary_path,
    }


def self_test() -> int:
    rows = [
        {"lambda_bohr": 1.0, "delta_e_upper_minus_lower": 1.0, "same_stationary_solution": False},
        {"lambda_bohr": 2.0, "delta_e_upper_minus_lower": -1.0, "same_stationary_solution": True},
    ]
    if sign_change_brackets(rows) != [[1.0, 2.0]]:
        raise AssertionError("Energy-bracket test failed.")
    if transition_brackets(rows) != [[1.0, 2.0]]:
        raise AssertionError("Merger-bracket test failed.")
    print("BeH2 B1 analyzer self-test: PASS")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    outputs = analyze(args)
    for label, path in outputs.items():
        print(f"{label:10s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
