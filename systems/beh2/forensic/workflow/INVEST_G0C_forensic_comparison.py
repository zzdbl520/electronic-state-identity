#!/usr/bin/env python3
"""Post-process the INVEST BeH2 G0/G0B archives into a forensic comparison.

This script performs no SCF or CASSCF optimization.  It reads the converged
wavefunctions produced by G0 and G0B and compares the mixed nuclear--protocol
plaquettes for three precisely defined cc-pVTZ models:

1. legacy CAS(6e,6o) with active MO 0 excluded from orbital rotations;
2. unconstrained all-electron CAS(6e,6o);
3. standard valence CAS(4e,6o) with an optimized inactive doubly occupied core.

The output is intended to support the constraint-induced variational-branch
interpretation without reviving the rejected physical-spinodal claim.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from BeH2_B0_analyze import Solution, load_solution, vector_text
from BeH2_B3R_analyze import (
    best_sa4_subset,
    dagger,
    frobenius,
    gauge_errors,
    leakage,
    plaquette_metrics,
    state_overlap_matrix,
)


DIMENSION = 3
BASIS = "cc-pVTZ"
ROUND_DIGITS = 10


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    color: str
    root_kind: str
    relative_dir: str
    expected_nelecas: int
    expected_ncore: int
    expected_frozen_flag: bool


@dataclass(frozen=True)
class Route:
    key: str
    label: str
    sa3_source: str
    sa4_source: str
    linestyle: str


MODELS = (
    Model(
        "legacy_frozen_active_mo0",
        "legacy: frozen active MO 0",
        "#C44E52",
        "g0",
        "legacy_frozen_mo0",
        6,
        0,
        True,
    ),
    Model(
        "all_electron_cas66",
        "all-electron CAS(6e,6o)",
        "#4C72B0",
        "g0",
        "all_electron_cas66",
        6,
        0,
        False,
    ),
    Model(
        "standard_valence_cas46",
        "standard valence CAS(4e,6o)",
        "#55A868",
        "g0b",
        "valence_cas46_optimized_core",
        4,
        1,
        False,
    ),
)


ROUTES = (
    Route("upper_history", "upper-history route", "upper_down", "upper_down", "-"),
    Route("reference_history", "reference route", "unique_up", "upper_up", "--"),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Zero-new-QC BeH2 constraint-on/constraint-released comparison."
    )
    result.add_argument("--g0-workdir", type=Path)
    result.add_argument("--g0b-workdir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--strides", default="1,2,4")
    result.add_argument("--forensic-eta-threshold", type=float, default=0.85)
    result.add_argument("--strict-eta-threshold", type=float, default=0.98)
    result.add_argument("--strict-edge-leakage-threshold", type=float, default=0.01)
    result.add_argument("--assignment-margin-threshold", type=float, default=0.10)
    result.add_argument("--contrast-threshold", type=float, default=10.0)
    result.add_argument("--self-test", action="store_true")
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_strides(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(x.strip()) for x in text.split(",") if x.strip())))
    if not values or values[0] < 1:
        raise ValueError("Strides must be positive integers.")
    return values


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_assignment_margin(overlap: np.ndarray) -> tuple[str, float, float]:
    matrix = np.asarray(overlap)
    if matrix.shape != (DIMENSION, DIMENSION):
        raise ValueError(f"Expected a {DIMENSION}x{DIMENSION} overlap matrix.")
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(DIMENSION)):
        score = float(sum(abs(matrix[i, permutation[i]]) for i in range(DIMENSION)))
        candidates.append((score, permutation))
    candidates.sort(reverse=True)
    best_score, best_permutation = candidates[0]
    second_score = candidates[1][0]
    return (
        ";".join(str(x) for x in best_permutation),
        best_score,
        (best_score - second_score) / DIMENSION,
    )


def model_dir(model: Model, g0_root: Path, g0b_root: Path) -> Path:
    parent = g0_root if model.root_kind == "g0" else g0b_root
    return parent / model.relative_dir


def validate_decisions(g0_root: Path, g0b_root: Path) -> dict[str, Any]:
    g0 = read_json(g0_root / "G0_decision.json")
    g0b = read_json(g0b_root / "G0B_decision.json")
    if g0.get("status") != "STOP_LEGACY_CONSTRAINT_ONLY":
        raise RuntimeError(
            "G0 status is not STOP_LEGACY_CONSTRAINT_ONLY; forensic closure is not locked."
        )
    if g0b.get("status") != "CONFIRMED_CONSTRAINT_INDUCED":
        raise RuntimeError(
            "G0B status is not CONFIRMED_CONSTRAINT_INDUCED; forensic closure is not locked."
        )
    return {"g0_status": g0["status"], "g0b_status": g0b["status"]}


def validate_model_config(model: Model, directory: Path) -> dict[str, Any]:
    config = read_json(directory / "variant_config.json")
    if int(config.get("nelecas", -1)) != model.expected_nelecas:
        raise RuntimeError(f"{model.key}: active-electron count differs from the model lock.")
    if int(config.get("ncas", -1)) != 6:
        raise RuntimeError(f"{model.key}: active-orbital count differs from the model lock.")
    if bool(config.get("freeze_lowest_mo")) != model.expected_frozen_flag:
        raise RuntimeError(f"{model.key}: frozen-orbital flag differs from the model lock.")
    if str(config.get("basis", "")).lower() != BASIS.lower():
        raise RuntimeError(f"{model.key}: basis is not {BASIS}.")
    if int(config.get("ncore", -1)) != model.expected_ncore:
        raise RuntimeError(f"{model.key}: inactive-core count differs from the model lock.")
    return config


def load_model_solutions(directory: Path) -> tuple[dict[tuple[float, str, str], Solution], list[float]]:
    paths = sorted(directory.glob("solutions/A1/SA*/*/lambda_*.npz"))
    if not paths:
        raise RuntimeError(f"No solution archives found below {directory}.")
    required_sources = {
        ("SA3", "upper_down"),
        ("SA3", "unique_up"),
        ("SA4", "upper_down"),
        ("SA4", "upper_up"),
    }
    table: dict[tuple[float, str, str], Solution] = {}
    for path in paths:
        item = load_solution(path)
        pair = (item.protocol, item.source)
        if pair not in required_sources:
            continue
        key = (round(item.lam, ROUND_DIGITS), item.protocol, item.source)
        if key in table:
            raise RuntimeError(f"Duplicate solution key {key} in {directory}.")
        table[key] = item
    grids = []
    for protocol, source in sorted(required_sources):
        grids.append(
            {lam for lam, p, s in table if p == protocol and s == source}
        )
    common = sorted(set.intersection(*grids))
    if len(common) < 2:
        raise RuntimeError(f"Fewer than two complete common geometries in {directory}.")
    for lam in common:
        for protocol, source in required_sources:
            item = table[(lam, protocol, source)]
            if item.ncas != 6 or sum(item.nelecas) not in (4, 6):
                raise RuntimeError(f"Unexpected solution dimensions in {item.path}.")
    return table, common


class OverlapCache:
    def __init__(self, fci: Any, gto: Any):
        self.fci = fci
        self.gto = gto
        self.values: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}

    def get(
        self,
        x: Solution,
        y: Solution,
        indices_x: Sequence[int],
        indices_y: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (str(x.path), str(y.path), tuple(indices_x), tuple(indices_y))
        if key not in self.values:
            self.values[key] = state_overlap_matrix(
                x,
                y,
                indices_x,
                indices_y,
                self.fci,
                self.gto,
                BASIS,
            )
        return self.values[key]


def history_row(
    model: Model,
    protocol: str,
    lam: float,
    x: Solution,
    y: Solution,
    cache: OverlapCache,
) -> dict[str, Any]:
    indices = tuple(range(x.nroots))
    overlap, _ = cache.get(x, y, indices, indices)
    singular = np.linalg.svd(overlap, compute_uv=False)
    assignment, score, margin = normalized_assignment_margin(overlap) if x.nroots == 3 else ("", math.nan, math.nan)
    state_leakage = 1.0 - float(np.sum(singular**2)) / x.nroots
    delta_e = abs(x.e_average - y.e_average)
    same = bool(delta_e <= 1.0e-7 and state_leakage <= 1.0e-6)
    return {
        "model": model.key,
        "model_label": model.label,
        "lambda_bohr": lam,
        "protocol": protocol,
        "source_x": x.source,
        "source_y": y.source,
        "delta_e_average_Eh": delta_e,
        "state_sigma_min": float(np.min(singular)),
        "state_leakage": state_leakage,
        "maximum_overlap_assignment": assignment,
        "assignment_score": score,
        "normalized_assignment_margin": margin,
        "same_stationary_solution_reduced_test": same,
    }


def analyze_model(
    model: Model,
    table: dict[tuple[float, str, str], Solution],
    grid: Sequence[float],
    strides: Sequence[int],
    cache: OverlapCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branch_rows: list[dict[str, Any]] = []
    for lam in grid:
        branch_rows.append(
            history_row(
                model,
                "SA3",
                lam,
                table[(lam, "SA3", "upper_down")],
                table[(lam, "SA3", "unique_up")],
                cache,
            )
        )
        branch_rows.append(
            history_row(
                model,
                "SA4",
                lam,
                table[(lam, "SA4", "upper_down")],
                table[(lam, "SA4", "upper_up")],
                cache,
            )
        )

    plaquette_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        selected: dict[float, tuple[int, ...]] = {}
        for lam in grid:
            sa3 = table[(lam, "SA3", route.sa3_source)]
            sa4 = table[(lam, "SA4", route.sa4_source)]
            selected[lam] = best_sa4_subset(sa3, sa4, cache.get)["subset"]

        for stride in strides:
            if stride >= len(grid):
                continue
            for index in range(len(grid) - stride):
                q0 = grid[index]
                q1 = grid[index + stride]
                a = table[(q0, "SA3", route.sa3_source)]
                b = table[(q1, "SA3", route.sa3_source)]
                c = table[(q0, "SA4", route.sa4_source)]
                d = table[(q1, "SA4", route.sa4_source)]
                subset_c = selected[q0]
                subset_d = selected[q1]
                ab, _ = cache.get(a, b, range(3), range(3))
                bd, _ = cache.get(b, d, range(3), subset_d)
                ac, _ = cache.get(a, c, range(3), subset_c)
                cd, _ = cache.get(c, d, subset_c, subset_d)
                overlaps = (ab, bd, ac, cd)
                values = plaquette_metrics(*overlaps)
                singular_values = values["edge_singular_values"]
                edge_leakages = [leakage(values) for values in singular_values]
                assignments = [normalized_assignment_margin(values) for values in overlaps]
                gauge_curvature, gauge_raw = gauge_errors(overlaps, values)
                reverse_errors = []
                for x, y, ix, iy, forward in (
                    (a, b, range(3), range(3), ab),
                    (b, d, range(3), subset_d, bd),
                    (a, c, range(3), subset_c, ac),
                    (c, d, subset_c, subset_d, cd),
                ):
                    reverse, _ = cache.get(y, x, iy, ix)
                    reverse_errors.append(frobenius(reverse - dagger(forward)))
                plaquette_rows.append(
                    {
                        "model": model.key,
                        "model_label": model.label,
                        "route": route.key,
                        "route_label": route.label,
                        "stride": stride,
                        "q0_bohr": q0,
                        "q1_bohr": q1,
                        "q_mid_bohr": 0.5 * (q0 + q1),
                        "delta_q_bohr": q1 - q0,
                        "C_qp": values["curvature"],
                        "raw_path_nonclosure": values["raw_path_nonclosure"],
                        "eta_edge": values["eta_edge"],
                        "maximum_edge_leakage": max(edge_leakages),
                        "minimum_assignment_margin": min(item[2] for item in assignments),
                        "edge_assignments": "|".join(item[0] for item in assignments),
                        "selected_SA4_q0": ";".join(str(x) for x in subset_c),
                        "selected_SA4_q1": ";".join(str(x) for x in subset_d),
                        "sigma_AB": float(np.min(singular_values[0])),
                        "sigma_BD": float(np.min(singular_values[1])),
                        "sigma_AC": float(np.min(singular_values[2])),
                        "sigma_CD": float(np.min(singular_values[3])),
                        "leakage_AB": edge_leakages[0],
                        "leakage_BD": edge_leakages[1],
                        "leakage_AC": edge_leakages[2],
                        "leakage_CD": edge_leakages[3],
                        "holonomy_phases_rad": vector_text(values["holonomy_phases"]),
                        "curvature_formula_error": values["formula_error"],
                        "curvature_gauge_error": gauge_curvature,
                        "raw_nonclosure_gauge_error": gauge_raw,
                        "polar_unitarity_error": values["unitarity_error"],
                        "reverse_overlap_error": max(reverse_errors),
                    }
                )
    return branch_rows, plaquette_rows


def safe_ratio(numerator: float, denominator: float, floor: float = 1.0e-15) -> float:
    return numerator / max(abs(denominator), floor)


def summarize(
    decisions: dict[str, Any],
    branch_rows: Sequence[dict[str, Any]],
    plaquette_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in MODELS:
        sa3 = [
            row
            for row in branch_rows
            if row["model"] == model.key and row["protocol"] == "SA3"
        ]
        upper_stride1 = [
            row
            for row in plaquette_rows
            if row["model"] == model.key
            and row["route"] == "upper_history"
            and int(row["stride"]) == 1
        ]
        reference_stride1 = [
            row
            for row in plaquette_rows
            if row["model"] == model.key
            and row["route"] == "reference_history"
            and int(row["stride"]) == 1
        ]
        high = max(sa3, key=lambda row: row["lambda_bohr"])
        by_model[model.key] = {
            "label": model.label,
            "SA3_high_end": {
                "lambda_bohr": high["lambda_bohr"],
                "delta_e_average_Eh": high["delta_e_average_Eh"],
                "state_leakage": high["state_leakage"],
                "same_stationary_solution_reduced_test": high[
                    "same_stationary_solution_reduced_test"
                ],
            },
            "upper_history_stride1": {
                "maximum_C_qp": max(row["C_qp"] for row in upper_stride1),
                "minimum_eta_edge": min(row["eta_edge"] for row in upper_stride1),
                "maximum_edge_leakage": max(
                    row["maximum_edge_leakage"] for row in upper_stride1
                ),
                "minimum_assignment_margin": min(
                    row["minimum_assignment_margin"] for row in upper_stride1
                ),
            },
            "reference_history_stride1": {
                "maximum_C_qp": max(row["C_qp"] for row in reference_stride1)
            },
        }

    legacy = by_model["legacy_frozen_active_mo0"]
    clean_keys = ("all_electron_cas66", "standard_valence_cas46")
    clean_c_max = max(
        by_model[key]["upper_history_stride1"]["maximum_C_qp"]
        for key in clean_keys
    )
    legacy_c_max = legacy["upper_history_stride1"]["maximum_C_qp"]
    contrast = safe_ratio(legacy_c_max, clean_c_max)
    legacy_event = bool(
        legacy["SA3_high_end"]["delta_e_average_Eh"] >= 1.0e-3
        and legacy["SA3_high_end"]["state_leakage"] >= 1.0e-2
    )
    clean_unique = bool(
        all(
            by_model[key]["SA3_high_end"]["same_stationary_solution_reduced_test"]
            for key in clean_keys
        )
    )
    legacy_upper_rows = [
        row
        for row in plaquette_rows
        if row["model"] == "legacy_frozen_active_mo0"
        and row["route"] == "upper_history"
        and int(row["stride"]) == 1
    ]
    legacy_peak = max(legacy_upper_rows, key=lambda row: row["C_qp"])
    forensic_local_gate = bool(
        legacy_peak["eta_edge"] >= args.forensic_eta_threshold
        and legacy_peak["minimum_assignment_margin"]
        >= args.assignment_margin_threshold
    )
    all_assignments_unambiguous = bool(
        all(
            values["upper_history_stride1"]["minimum_assignment_margin"]
            >= args.assignment_margin_threshold
            for values in by_model.values()
        )
    )
    strict_edge_gate = bool(
        all(
            values["upper_history_stride1"]["minimum_eta_edge"]
            >= args.strict_eta_threshold
            and values["upper_history_stride1"]["maximum_edge_leakage"]
            <= args.strict_edge_leakage_threshold
            and values["upper_history_stride1"]["minimum_assignment_margin"]
            >= args.assignment_margin_threshold
            for values in by_model.values()
        )
    )
    forensic_ready = bool(
        legacy_event
        and clean_unique
        and forensic_local_gate
        and all_assignments_unambiguous
        and contrast >= args.contrast_threshold
    )
    strict_false_negative_candidate = bool(forensic_ready and strict_edge_gate)
    return {
        "gate": "G0C zero-new-QC BeH2 forensic comparison",
        "status": "FORENSIC_BEH2_READY" if forensic_ready else "REVIEW_REQUIRED",
        "input_decisions": decisions,
        "basis": BASIS,
        "model_semantics": {
            "legacy": "CAS(6e,6o); active MO 0 excluded from orbital rotations",
            "all_electron": "unconstrained all-electron CAS(6e,6o)",
            "standard_valence": (
                "CAS(4e,6o) with one optimized inactive doubly occupied core; "
                "no orbital removed from optimization"
            ),
        },
        "decision_checks": {
            "legacy_event_present": legacy_event,
            "both_clean_models_single_solution": clean_unique,
            "legacy_peak_conditioned_for_forensic_use": forensic_local_gate,
            "all_stride1_assignments_unambiguous": all_assignments_unambiguous,
            "legacy_to_clean_max_C_qp_ratio": contrast,
            "C_qp_contrast_threshold": args.contrast_threshold,
            "strict_false_negative_edge_gate_passed": strict_edge_gate,
            "strict_molecular_false_negative_candidate": strict_false_negative_candidate,
            "BeH2_forensic_figure_ready": forensic_ready,
        },
        "legacy_peak_stride1": {
            "q0_bohr": legacy_peak["q0_bohr"],
            "q1_bohr": legacy_peak["q1_bohr"],
            "C_qp": legacy_peak["C_qp"],
            "eta_edge": legacy_peak["eta_edge"],
            "maximum_edge_leakage": legacy_peak["maximum_edge_leakage"],
            "minimum_assignment_margin": legacy_peak["minimum_assignment_margin"],
        },
        "thresholds": {
            "forensic_BeH2": {
                "eta_at_peak_minimum": args.forensic_eta_threshold,
                "minimum_normalized_assignment_margin": args.assignment_margin_threshold,
                "minimum_C_qp_contrast": args.contrast_threshold,
            },
            "strict_molecular_false_negative": {
                "eta_edge_minimum": args.strict_eta_threshold,
                "maximum_edge_leakage": args.strict_edge_leakage_threshold,
                "minimum_normalized_assignment_margin": args.assignment_margin_threshold,
            },
        },
        "models": by_model,
        "authorized_interpretation": (
            "The archived BeH2 event is a constraint-induced variational branch "
            "termination within a deliberately restricted orbital-optimization "
            "manifold, not a physical spinodal of unconstrained CASSCF."
        ),
        "prohibited_interpretations": [
            "physical spinodal",
            "intrinsic BeH2 branch collapse",
            "frozen-core CAS(6e,6o)",
        ],
        "scope_note": (
            "BeH2 passes the forensic model-audit standard. It does not pass the "
            "stricter edge-fidelity standard reserved for a flagship molecular "
            "false-negative demonstration."
        ),
    }


def make_figure(
    path_png: Path,
    path_pdf: Path,
    branch_rows: Sequence[dict[str, Any]],
    plaquette_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create the forensic figure.") from exc

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.45,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(6.6, 7.5), sharex=True)
    ax_c, ax_branch, ax_eta = axes
    floor = 1.0e-16
    coordinate_origin = 2.5218
    coordinate_scale = 1.0e4

    for model in MODELS:
        for route in ROUTES:
            rows = sorted(
                (
                    row
                    for row in plaquette_rows
                    if row["model"] == model.key
                    and row["route"] == route.key
                    and int(row["stride"]) == 1
                ),
                key=lambda row: row["q_mid_bohr"],
            )
            x = [
                coordinate_scale * (row["q_mid_bohr"] - coordinate_origin)
                for row in rows
            ]
            y = [max(float(row["C_qp"]), floor) for row in rows]
            alpha = 1.0 if route.key == "upper_history" else 0.58
            ax_c.plot(
                x,
                y,
                color=model.color,
                linestyle=route.linestyle,
                marker="o" if route.key == "upper_history" else None,
                markersize=3.1,
                alpha=alpha,
            )

        sa3 = sorted(
            (
                row
                for row in branch_rows
                if row["model"] == model.key and row["protocol"] == "SA3"
            ),
            key=lambda row: row["lambda_bohr"],
        )
        ax_branch.plot(
            [
                coordinate_scale * (row["lambda_bohr"] - coordinate_origin)
                for row in sa3
            ],
            [max(float(row["state_leakage"]), floor) for row in sa3],
            color=model.color,
            marker="o",
            markersize=3.1,
        )

        upper = sorted(
            (
                row
                for row in plaquette_rows
                if row["model"] == model.key
                and row["route"] == "upper_history"
                and int(row["stride"]) == 1
            ),
            key=lambda row: row["q_mid_bohr"],
        )
        ax_eta.plot(
            [
                coordinate_scale * (row["q_mid_bohr"] - coordinate_origin)
                for row in upper
            ],
            [row["eta_edge"] for row in upper],
            color=model.color,
            marker="o",
            markersize=3.1,
        )

    ax_c.set_yscale("log")
    ax_branch.set_yscale("log")
    ax_c.set_ylabel(r"mixed nonclosure $C_{qp}$")
    ax_branch.set_ylabel("SA3 history leakage")
    ax_eta.set_ylabel(r"edge fidelity $\eta_{\rm edge}$")
    ax_eta.set_xlabel(
        r"$(\lambda-2.5218\,\mathrm{bohr})/(10^{-4}\,\mathrm{bohr})$"
    )
    ax_eta.set_ylim(0.82, 1.01)
    ax_eta.axhline(0.85, color="0.45", linestyle=":", linewidth=0.9)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(0.012, 0.94, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(False)

    model_handles = [
        Line2D([0], [0], color=model.color, marker="o", markersize=3.1, label=model.label)
        for model in MODELS
    ]
    route_handles = [
        Line2D([0], [0], color="0.2", linestyle=route.linestyle, label=route.label)
        for route in ROUTES
    ]
    first = ax_c.legend(handles=model_handles, loc="upper left", frameon=False, fontsize=7.4)
    ax_c.add_artist(first)
    ax_c.legend(handles=route_handles, loc="lower right", frameon=False, fontsize=7.4)

    figure.align_ylabels(axes)
    figure.savefig(path_png, dpi=600)
    figure.savefig(path_pdf)
    plt.close(figure)


def import_pyscf_modules() -> tuple[Any, Any]:
    try:
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError("PySCF is required only to evaluate archived wavefunction overlaps.") from exc
    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError("This PySCF version does not provide fci.addons.overlap.")
    return fci, gto


def self_test() -> int:
    identity = np.eye(DIMENSION)
    values = plaquette_metrics(identity, identity, identity, identity)
    if abs(values["curvature"]) > 1.0e-14:
        raise AssertionError("Identity plaquette has nonzero C_qp.")
    assignment, score, margin = normalized_assignment_margin(identity)
    if assignment != "0;1;2" or abs(score - 3.0) > 1.0e-14 or margin < 0.6:
        raise AssertionError("Assignment-margin self-test failed.")
    if parse_strides("4,1,2,2") != (1, 2, 4):
        raise AssertionError("Stride parser self-test failed.")
    if abs(safe_ratio(2.0, 0.5) - 4.0) > 1.0e-14:
        raise AssertionError("Contrast-ratio self-test failed.")
    print("INVEST BeH2 G0C forensic-comparison self-test: PASS")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    if args.g0_workdir is None or args.g0b_workdir is None or args.output_dir is None:
        raise ValueError("Production mode requires --g0-workdir, --g0b-workdir, and --output-dir.")

    decisions = validate_decisions(args.g0_workdir, args.g0b_workdir)
    strides = parse_strides(args.strides)
    fci, gto = import_pyscf_modules()
    cache = OverlapCache(fci, gto)

    tables: dict[str, dict[tuple[float, str, str], Solution]] = {}
    grids: dict[str, list[float]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        directory = model_dir(model, args.g0_workdir, args.g0b_workdir)
        configs[model.key] = validate_model_config(model, directory)
        tables[model.key], grids[model.key] = load_model_solutions(directory)

    common_grid = sorted(set.intersection(*(set(values) for values in grids.values())))
    if len(common_grid) < 2:
        raise RuntimeError("The three models do not share a usable analysis grid.")

    branch_rows: list[dict[str, Any]] = []
    plaquette_rows: list[dict[str, Any]] = []
    for model in MODELS:
        b_rows, p_rows = analyze_model(
            model,
            tables[model.key],
            common_grid,
            strides,
            cache,
        )
        branch_rows.extend(b_rows)
        plaquette_rows.extend(p_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    branch_path = args.output_dir / "G0C_history_separation.csv"
    plaquette_path = args.output_dir / "G0C_mixed_plaquettes.csv"
    summary_path = args.output_dir / "G0C_forensic_summary.json"
    figure_png = args.output_dir / "G0C_constraint_forensic_comparison.png"
    figure_pdf = args.output_dir / "G0C_constraint_forensic_comparison.pdf"

    rows_to_csv(branch_path, branch_rows)
    rows_to_csv(plaquette_path, plaquette_rows)
    summary = summarize(decisions, branch_rows, plaquette_rows, args)
    summary["common_grid_bohr"] = common_grid
    summary["strides"] = list(strides)
    summary["model_configs"] = configs
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(figure_png, figure_pdf, branch_rows, plaquette_rows)

    print(json.dumps(summary, indent=2))
    print(f"history     : {branch_path}")
    print(f"plaquettes  : {plaquette_path}")
    print(f"summary     : {summary_path}")
    print(f"figure_png  : {figure_png}")
    print(f"figure_pdf  : {figure_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
