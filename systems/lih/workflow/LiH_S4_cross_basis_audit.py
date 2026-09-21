#!/usr/bin/env python3
"""LiH S4: zero-new-QC cross-basis R x basis-protocol audit.

S4 consumes the completed aug-cc-pVDZ S1/S2 and fixed-window aug-cc-pVTZ S3
wavefunctions.  It performs no SCF or CASSCF calculation.  At each fixed field
slice it constructs R x basis plaquettes from exact active-only CAS-wavefunction
overlaps, including true cross-basis AO overlaps from ``gto.intor_cross``.

The purpose is diagnostic, not selective: every R interval in the already
frozen window is analyzed, no coordinate can be moved, and no signal-size gate
is imposed on the basis-protocol nonclosure.  S4 asks whether the pVDZ and pVTZ
retained manifolds can be compared faithfully and reports how large the basis
path effect is relative to the independently reproduced physical R x Fz signal.

Typical use
-----------
python LiH_S4_cross_basis_audit.py selftest
python LiH_S4_cross_basis_audit.py analyze \
  --pvdz-root /path/to/LiH_S1_field_locator_work \
  --pvtz-root /path/to/LiH_S3_fixed_window_augccpvtz_work \
  --output-root /path/to/LiH_S4_cross_basis_audit_work
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import LiH_S0_SA4_qualification as s0
import LiH_S1_physical_field_locator as s1
import LiH_S3_fixed_window_basis_robustness as s3


SCRIPT_VERSION = "LiH-S4-cross-basis-audit-v1.0"
R_GRID_ANGSTROM = s3.R_GRID_ANGSTROM
FIELD_GRID_AU = s3.FIELD_GRID_AU
BASIS_PVDZ = "aug-cc-pvdz"
BASIS_PVTZ = "aug-cc-pvtz"


@dataclass(frozen=True)
class Thresholds:
    eta_edge_minimum: float = 0.98
    maximum_edge_leakage: float = 0.01
    minimum_normalized_assignment_margin: float = 0.10
    active_orbital_subspace_sigma_minimum: float = 0.90
    active_orbital_subspace_leakage_maximum: float = 0.10
    maximum_numerical_check_error: float = 1.0e-9


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return obj


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close_scalar(a: Any, b: float, atol: float = 5.0e-12) -> bool:
    return bool(abs(float(a) - float(b)) <= atol)


def load_references(pvdz_root: Path, pvtz_root: Path) -> Dict[str, Any]:
    s1_path = pvdz_root / "LiH_S1_decision.json"
    s2_path = pvdz_root / "LiH_S2_decision.json"
    s3_path = pvtz_root / "LiH_S3_decision.json"
    d1 = read_json(s1_path)
    d2 = read_json(s2_path)
    d3 = read_json(s3_path)

    if d1.get("status") != "CANDIDATE_WINDOW_FOUND":
        raise RuntimeError(f"invalid S1 status: {d1.get('status')!r}")
    if d2.get("status") != "PUBLICATION_DEMONSTRATION_READY":
        raise RuntimeError(f"invalid S2 status: {d2.get('status')!r}")
    if d3.get("status") != "BASIS_ROBUSTNESS_CONFIRMED":
        raise RuntimeError(f"invalid S3 status: {d3.get('status')!r}")
    if not bool(d2.get("manuscript_claim_authorized")):
        raise RuntimeError("S2 manuscript claim was not authorized")
    if not bool(d3.get("manuscript_claim_authorized")):
        raise RuntimeError("S3 basis robustness was not authorized")
    if str(d1.get("model", {}).get("basis", "")).lower() != BASIS_PVDZ:
        raise RuntimeError("S1 basis is not aug-cc-pVDZ")
    if str(d3.get("model", {}).get("basis", "")).lower() != BASIS_PVTZ:
        raise RuntimeError("S3 basis is not aug-cc-pVTZ")

    fixed2 = d2.get("fixed_selected_window", {})
    fixed3 = d3.get("fixed_grid", {})
    expected = {
        "R0_angstrom": R_GRID_ANGSTROM[0],
        "R1_angstrom": R_GRID_ANGSTROM[-1],
        "F0_au": FIELD_GRID_AU[0],
        "F1_au": FIELD_GRID_AU[-1],
    }
    for key, value in expected.items():
        if key not in fixed2 or not close_scalar(fixed2[key], value):
            raise RuntimeError(
                f"S2 fixed-window mismatch for {key}: {fixed2.get(key)!r}"
            )
    if tuple(float(x) for x in fixed3.get("R_angstrom", [])) != R_GRID_ANGSTROM:
        raise RuntimeError("S3 R grid does not match the frozen S2 window")
    if tuple(float(x) for x in fixed3.get("field_z_au", [])) != FIELD_GRID_AU:
        raise RuntimeError("S3 field grid does not match the frozen S2 window")
    if bool(fixed3.get("selection_allowed", True)):
        raise RuntimeError("S3 did not forbid coordinate selection")

    physical_pvdz = d2.get("selected_window_results", {}).get("direct", {})
    physical_pvtz = d3.get("fixed_window_result", {})
    if not bool(physical_pvdz.get("candidate_window")):
        raise RuntimeError("S2 physical reference is not a passed candidate")
    if not bool(physical_pvtz.get("all_four_direct_edges_strict")):
        raise RuntimeError("S3 physical reference lacks four strict edges")

    return {
        "paths": {
            "S1": str(s1_path.resolve()),
            "S2": str(s2_path.resolve()),
            "S3": str(s3_path.resolve()),
        },
        "sha256": {
            "S1": file_sha256(s1_path),
            "S2": file_sha256(s2_path),
            "S3": file_sha256(s3_path),
        },
        "statuses": {
            "S1": d1["status"],
            "S2": d2["status"],
            "S3": d3["status"],
        },
        "physical_pvdz": physical_pvdz,
        "physical_pvtz": physical_pvtz,
    }


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conj(phases))


def solution_path(root: Path, r_angstrom: float, field_au: float) -> Path:
    return s1.solution_path(root, r_angstrom, field_au)


def analyze(
    pvdz_root: Path,
    pvtz_root: Path,
    output_root: Path,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    references = load_references(pvdz_root, pvtz_root)
    output_root.mkdir(parents=True, exist_ok=True)

    solutions: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    # Vertex key: (basis_index, R_index, field_index), where 0=pVDZ, 1=pVTZ.
    for basis_index, root in ((0, pvdz_root), (1, pvtz_root)):
        for i, r in enumerate(R_GRID_ANGSTROM):
            for j, field in enumerate(FIELD_GRID_AU):
                solutions[(basis_index, i, j)] = s1.load_solution(
                    solution_path(root, r, field)
                )

    mols = {
        (basis_index, i): s0.make_mol(r, basis, verbose=0)
        for basis_index, basis in ((0, BASIS_PVDZ), (1, BASIS_PVTZ))
        for i, r in enumerate(R_GRID_ANGSTROM)
    }
    terms_cache: MutableMapping[Tuple[str, int], Any] = {}
    edge_objects: Dict[
        Tuple[Tuple[int, int, int], Tuple[int, int, int]], Dict[str, Any]
    ] = {}
    edge_records: Dict[
        Tuple[Tuple[int, int, int], Tuple[int, int, int]], Dict[str, Any]
    ] = {}

    def vertex_record(v: Tuple[int, int, int]) -> Dict[str, Any]:
        basis_index, i, j = v
        return {
            "basis": BASIS_PVDZ if basis_index == 0 else BASIS_PVTZ,
            "R_angstrom": R_GRID_ANGSTROM[i],
            "field_z_au": FIELD_GRID_AU[j],
        }

    def edge(
        a: Tuple[int, int, int],
        b: Tuple[int, int, int],
        kind: str,
    ) -> Dict[str, Any]:
        key = (a, b)
        if key not in edge_objects:
            sa = solutions[a]
            sb = solutions[b]
            mol_a = mols[(a[0], a[1])]
            mol_b = mols[(b[0], b[1])]
            matrix = s0.active_overlap_matrix(
                sa, sb, mol_a, mol_b, terms_cache
            )
            obj = s0.edge_metrics(matrix)
            active = s0.active_orbital_metrics(sa, sb, mol_a, mol_b)
            edge_objects[key] = obj
            edge_records[key] = {
                "edge": (
                    f"{kind}:b{a[0]}R{a[1]}F{a[2]}->"
                    f"b{b[0]}R{b[1]}F{b[2]}"
                ),
                "kind": kind,
                "R_stride_points": abs(b[1] - a[1]),
                "basis_stride": abs(b[0] - a[0]),
                "from": vertex_record(a),
                "to": vertex_record(b),
                "state_subspace": obj["metrics"],
                "active_orbital_subspace": active,
            }
        return edge_objects[key]

    def edge_pass(record: Mapping[str, Any]) -> bool:
        state = record["state_subspace"]
        active = record["active_orbital_subspace"]
        return bool(
            state["sigma_min"] >= thresholds.eta_edge_minimum
            and state["leakage"] <= thresholds.maximum_edge_leakage
            and state["normalized_margin"]
            >= thresholds.minimum_normalized_assignment_margin
            and active["sigma_min"]
            >= thresholds.active_orbital_subspace_sigma_minimum
            and active["leakage"]
            <= thresholds.active_orbital_subspace_leakage_maximum
        )

    # Cache every elementary nuclear edge and every cross-basis edge.
    for j in range(len(FIELD_GRID_AU)):
        for basis_index in (0, 1):
            basis_label = "nuclear_pvdz" if basis_index == 0 else "nuclear_pvtz"
            for i in range(len(R_GRID_ANGSTROM) - 1):
                edge(
                    (basis_index, i, j),
                    (basis_index, i + 1, j),
                    basis_label,
                )
        for i in range(len(R_GRID_ANGSTROM)):
            edge((0, i, j), (1, i, j), "basis_pvdz_to_pvtz")

    rng = np.random.default_rng(20260918)
    plaquettes: List[Dict[str, Any]] = []
    maximum_gauge_c_error = 0.0
    maximum_gauge_dipole_error = 0.0

    for j, field in enumerate(FIELD_GRID_AU):
        for i0 in range(len(R_GRID_ANGSTROM) - 1):
            for i1 in range(i0 + 1, len(R_GRID_ANGSTROM)):
                a = (0, i0, j)
                b = (0, i1, j)
                c = (1, i0, j)
                d = (1, i1, j)
                ab = edge(a, b, "nuclear_pvdz")
                bd = edge(b, d, "basis_pvdz_to_pvtz")
                ac = edge(a, c, "basis_pvdz_to_pvtz")
                cd = edge(c, d, "nuclear_pvtz")
                boundary_keys = ((a, b), (b, d), (a, c), (c, d))
                boundary_records = [edge_records[key] for key in boundary_keys]

                route_r_then_basis = ab["Q"] @ bd["Q"]
                route_basis_then_r = ac["Q"] @ cd["Q"]
                dipole_d = solutions[d]["dipole_au"]
                result = s1.route_result(
                    route_r_then_basis, route_basis_then_r, dipole_d
                )

                gauges = {v: random_unitary(s0.NROOTS, rng) for v in (a, b, c, d)}

                def gauged_q(
                    x: Tuple[int, int, int], y: Tuple[int, int, int]
                ) -> np.ndarray:
                    transformed = (
                        gauges[x].conj().T
                        @ edge_objects[(x, y)]["matrix"]
                        @ gauges[y]
                    )
                    return s0.polar_unitary(transformed)[0]

                gauge_route_1 = gauged_q(a, b) @ gauged_q(b, d)
                gauge_route_2 = gauged_q(a, c) @ gauged_q(c, d)
                gd = gauges[d]
                gauge_dipole = np.asarray(
                    [gd.conj().T @ dipole_d[k] @ gd for k in range(3)]
                )
                gauge_result = s1.route_result(
                    gauge_route_1, gauge_route_2, gauge_dipole
                )
                gauge_c_error = abs(gauge_result["C_qp"] - result["C_qp"])
                gauge_dipole_error = abs(
                    gauge_result["dipole_path_difference_au"]
                    - result["dipole_path_difference_au"]
                )
                maximum_gauge_c_error = max(
                    maximum_gauge_c_error, gauge_c_error
                )
                maximum_gauge_dipole_error = max(
                    maximum_gauge_dipole_error, gauge_dipole_error
                )

                plaquettes.append(
                    {
                        "plaquette": f"F{j}:R{i0}->{i1}:pVDZ->pVTZ",
                        "field_z_au": field,
                        "R0_angstrom": R_GRID_ANGSTROM[i0],
                        "R1_angstrom": R_GRID_ANGSTROM[i1],
                        "R_stride_points": i1 - i0,
                        "full_R_span": i0 == 0 and i1 == len(R_GRID_ANGSTROM) - 1,
                        "all_four_direct_edges_strict": all(
                            edge_pass(record) for record in boundary_records
                        ),
                        **result,
                        "gauge_test_C_qp_abs_error": gauge_c_error,
                        "gauge_test_dipole_abs_error": gauge_dipole_error,
                        "edges": {
                            "AB_nuclear_pvdz": boundary_records[0],
                            "BD_basis_at_R1": boundary_records[1],
                            "AC_basis_at_R0": boundary_records[2],
                            "CD_nuclear_pvtz": boundary_records[3],
                        },
                    }
                )

    all_records = list(edge_records.values())
    cross_basis_records = [
        record for record in all_records if record["kind"] == "basis_pvdz_to_pvtz"
    ]
    elementary_nuclear_records = [
        record
        for record in all_records
        if record["kind"].startswith("nuclear_")
        and record["R_stride_points"] == 1
    ]
    full_span = [p for p in plaquettes if p["full_R_span"]]

    semantics = True
    for (basis_index, _, _), sol in solutions.items():
        md = sol["metadata"]
        expected_basis = BASIS_PVDZ if basis_index == 0 else BASIS_PVTZ
        semantics = semantics and bool(
            str(md.get("basis", "")).lower() == expected_basis
            and int(md.get("nelecas", -1)) == s0.NELECAS
            and int(md.get("ncas", -1)) == s0.NCAS
            and int(md.get("nroots", -1)) == s0.NROOTS
            and int(md.get("ncore", -1)) == 1
            and int(md.get("frozen_orbitals", -1)) == 0
            and md.get("active_irrep_counts") == s0.ACTIVE_IRREP_COUNTS
        )

    all_cross_basis_strict = bool(cross_basis_records) and all(
        edge_pass(record) for record in cross_basis_records
    )
    all_elementary_nuclear_strict = bool(elementary_nuclear_records) and all(
        edge_pass(record) for record in elementary_nuclear_records
    )
    all_plaquette_edges_strict = all(
        p["all_four_direct_edges_strict"] for p in plaquettes
    )
    all_route_assignments_same = all(
        p["same_composed_assignment"] for p in plaquettes
    )
    numerical_validation = bool(
        maximum_gauge_c_error <= thresholds.maximum_numerical_check_error
        and maximum_gauge_dipole_error <= thresholds.maximum_numerical_check_error
    )

    if not semantics:
        status = "STOP_MODEL_SEMANTICS_FAILURE"
    elif not numerical_validation:
        status = "STOP_NUMERICAL_VALIDATION_FAILURE"
    elif not all_cross_basis_strict:
        status = "STOP_CROSS_BASIS_EDGE_FIDELITY_FAILURE"
    elif not all_elementary_nuclear_strict or not all_plaquette_edges_strict:
        status = "STOP_NUCLEAR_EDGE_FIDELITY_FAILURE"
    elif not all_route_assignments_same:
        status = "CROSS_BASIS_DISCRETE_ASSIGNMENT_CHANGE_DETECTED"
    else:
        status = "CROSS_BASIS_AUDIT_READY"

    physical_c_pvdz = float(references["physical_pvdz"]["C_qp"])
    physical_c_pvtz = float(references["physical_pvtz"]["C_qp"])
    max_c = max(float(p["C_qp"]) for p in plaquettes)
    max_full_c = max(float(p["C_qp"]) for p in full_span)
    full_by_field = sorted(full_span, key=lambda p: p["field_z_au"])

    decision = {
        "gate": "LiH S4 zero-new-QC R x basis-protocol audit",
        "status": status,
        "script_version": SCRIPT_VERSION,
        "new_quantum_chemistry_calculations": 0,
        "input_references": {
            "paths": references["paths"],
            "sha256": references["sha256"],
            "statuses": references["statuses"],
        },
        "model": {
            "molecule": "LiH",
            "basis_protocol_axis": [BASIS_PVDZ, BASIS_PVTZ],
            "R_grid_angstrom": R_GRID_ANGSTROM,
            "field_slices_au": FIELD_GRID_AU,
            "state_symmetry": s0.WFNSYM,
            "active_space": [s0.NELECAS, s0.NCAS],
            "active_irrep_counts": s0.ACTIVE_IRREP_COUNTS,
            "ncore": 1,
            "frozen_orbitals": 0,
            "retained_roots": s0.NROOTS,
            "edge_metric": "exact active-only cross-basis CAS-wavefunction overlap",
        },
        "thresholds": asdict(thresholds),
        "decision_checks": {
            "input_S1_S2_S3_decisions_valid": True,
            "model_semantics_exact": semantics,
            "numerical_gauge_invariance": numerical_validation,
            "all_cross_basis_edges_strict": all_cross_basis_strict,
            "all_elementary_nuclear_edges_strict": all_elementary_nuclear_strict,
            "all_direct_plaquette_edges_strict": all_plaquette_edges_strict,
            "all_conventional_composed_assignments_same": all_route_assignments_same,
        },
        "counts": {
            "input_wavefunctions": len(solutions),
            "new_wavefunctions": 0,
            "cross_basis_edges": len(cross_basis_records),
            "elementary_nuclear_edges": len(elementary_nuclear_records),
            "R_x_basis_plaquettes": len(plaquettes),
            "full_R_span_plaquettes": len(full_span),
        },
        "cross_basis_edge_summary": {
            "minimum_eta": min(
                record["state_subspace"]["sigma_min"]
                for record in cross_basis_records
            ),
            "maximum_leakage": max(
                record["state_subspace"]["leakage"]
                for record in cross_basis_records
            ),
            "minimum_normalized_assignment_margin": min(
                record["state_subspace"]["normalized_margin"]
                for record in cross_basis_records
            ),
            "minimum_active_orbital_sigma": min(
                record["active_orbital_subspace"]["sigma_min"]
                for record in cross_basis_records
            ),
            "maximum_active_orbital_leakage": max(
                record["active_orbital_subspace"]["leakage"]
                for record in cross_basis_records
            ),
        },
        "basis_protocol_nonclosure": {
            "maximum_C_Rbasis_all_spans": max_c,
            "maximum_C_Rbasis_full_R_span": max_full_c,
            "mean_C_Rbasis_full_R_span": float(
                np.mean([p["C_qp"] for p in full_span])
            ),
            "rms_C_Rbasis_full_R_span": float(
                np.sqrt(np.mean([float(p["C_qp"]) ** 2 for p in full_span]))
            ),
            "full_R_span_by_field": [
                {
                    "field_z_au": p["field_z_au"],
                    "C_Rbasis": p["C_qp"],
                    "dipole_path_difference_au": p["dipole_path_difference_au"],
                    "dipole_path_difference_normalized": p[
                        "dipole_path_difference_normalized"
                    ],
                    "route_R_then_basis_permutation": p[
                        "route_R_then_F_permutation"
                    ],
                    "route_basis_then_R_permutation": p[
                        "route_F_then_R_permutation"
                    ],
                    "same_composed_assignment": p["same_composed_assignment"],
                    "all_four_direct_edges_strict": p[
                        "all_four_direct_edges_strict"
                    ],
                }
                for p in full_by_field
            ],
        },
        "contrast_to_physical_R_x_Fz_signal": {
            "physical_C_RF_aug_cc_pvdz": physical_c_pvdz,
            "physical_C_RF_aug_cc_pvtz": physical_c_pvtz,
            "maximum_full_span_C_Rbasis_over_physical_C_RF_pvdz": (
                max_full_c / physical_c_pvdz
            ),
            "maximum_full_span_C_Rbasis_over_physical_C_RF_pvtz": (
                max_full_c / physical_c_pvtz
            ),
            "note": (
                "Ratios are reported descriptively; no post hoc signal-contrast "
                "threshold is imposed. Independent within-basis S2/S3 reproduction "
                "is the primary basis-robustness evidence."
            ),
        },
        "maximum_gauge_C_qp_error": maximum_gauge_c_error,
        "maximum_gauge_dipole_error": maximum_gauge_dipole_error,
        "worst_cross_basis_edge": min(
            cross_basis_records,
            key=lambda record: record["state_subspace"]["sigma_min"],
        ),
        "highest_C_Rbasis_plaquette": max(
            plaquettes, key=lambda p: p["C_qp"]
        ),
        "failed_edges": [record for record in all_records if not edge_pass(record)],
        "authorized_interpretation": (
            "The existing pVDZ and pVTZ LiH manifolds admit a faithful direct "
            "cross-basis audit over the frozen window. The measured R x basis "
            "nonclosure is a computational-protocol diagnostic and must be reported "
            "relative to, not substituted for, the independently reproduced physical "
            "R x Fz signal."
            if status == "CROSS_BASIS_AUDIT_READY"
            else "The cross-basis audit did not pass every comparison gate."
        ),
        "prohibited_interpretations": [
            "new independent quantum-chemistry reproduction",
            "complete-basis-set convergence",
            "physical electric-field effect inferred from the basis-protocol loop",
            "post hoc thresholding of the basis-protocol signal",
            "replacement of active-only overlap by scalar inactive-core division",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "timestamp_unix": time.time(),
        },
    }

    diagnostics = {
        "edges": all_records,
        "plaquettes": sorted(
            plaquettes,
            key=lambda p: (p["field_z_au"], p["R0_angstrom"], p["R1_angstrom"]),
        ),
    }
    manifest = {
        "script_version": SCRIPT_VERSION,
        "gate": decision["gate"],
        "input_references": decision["input_references"],
        "model": decision["model"],
        "thresholds": decision["thresholds"],
        "selection_allowed": False,
        "new_quantum_chemistry_calculations": 0,
    }
    s0.write_json(output_root / "LiH_S4_manifest.json", manifest)
    s0.write_json(output_root / "LiH_S4_decision.json", decision)
    s0.write_json(output_root / "LiH_S4_diagnostics.json", diagnostics)

    summary = decision["cross_basis_edge_summary"]
    print("\nLiH S4 RESULT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"Cross-basis edges: eta_min={summary['minimum_eta']:.12f}; "
        f"leakage_max={summary['maximum_leakage']:.12f}; "
        f"margin_min={summary['minimum_normalized_assignment_margin']:.12f}",
        flush=True,
    )
    print(
        f"Full-span C_Rbasis: max={max_full_c:.12e}; "
        f"mean={decision['basis_protocol_nonclosure']['mean_C_Rbasis_full_R_span']:.12e}; "
        f"rms={decision['basis_protocol_nonclosure']['rms_C_Rbasis_full_R_span']:.12e}",
        flush=True,
    )
    print(
        f"Contrast max(C_Rbasis)/C_RF: "
        f"pVDZ={max_full_c/physical_c_pvdz:.6f}; "
        f"pVTZ={max_full_c/physical_c_pvtz:.6f}",
        flush=True,
    )
    print(f"Decision file: {output_root / 'LiH_S4_decision.json'}", flush=True)
    return decision


def selftest() -> None:
    assert R_GRID_ANGSTROM == (5.80, 5.85, 5.90, 5.95)
    assert FIELD_GRID_AU == (0.01050, 0.01075, 0.01100, 0.01125, 0.01150)
    assert BASIS_PVDZ == "aug-cc-pvdz"
    assert BASIS_PVTZ == "aug-cc-pvtz"

    rng = np.random.default_rng(59)
    matrices = [
        np.eye(s0.NROOTS) + 0.01 * rng.normal(size=(s0.NROOTS, s0.NROOTS))
        for _ in range(4)
    ]
    q = [s0.polar_unitary(matrix)[0] for matrix in matrices]
    dipole = rng.normal(size=(3, s0.NROOTS, s0.NROOTS))
    dipole = (dipole + dipole.transpose(0, 2, 1)) / 2.0
    reference = s1.route_result(q[0] @ q[1], q[2] @ q[3], dipole)
    gauges = [random_unitary(s0.NROOTS, rng) for _ in range(4)]
    transformed = [
        gauges[0].conj().T @ matrices[0] @ gauges[1],
        gauges[1].conj().T @ matrices[1] @ gauges[3],
        gauges[0].conj().T @ matrices[2] @ gauges[2],
        gauges[2].conj().T @ matrices[3] @ gauges[3],
    ]
    tq = [s0.polar_unitary(matrix)[0] for matrix in transformed]
    gd = gauges[3]
    gauge_dipole = np.asarray([gd.conj().T @ dipole[k] @ gd for k in range(3)])
    checked = s1.route_result(tq[0] @ tq[1], tq[2] @ tq[3], gauge_dipole)
    assert abs(reference["C_qp"] - checked["C_qp"]) < 1.0e-12
    assert (
        abs(
            reference["dipole_path_difference_au"]
            - checked["dipole_path_difference_au"]
        )
        < 1.0e-12
    )
    print("SELFTEST=PASS")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run NumPy-only algebra and locked-grid tests")
    p = sub.add_parser("analyze")
    p.add_argument("--pvdz-root", type=Path, required=True)
    p.add_argument("--pvtz-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    if args.threads < 1:
        raise ValueError("--threads must be a positive integer")
    s0.require_pyscf()
    import pyscf

    pyscf.lib.num_threads(args.threads)
    analyze(
        args.pvdz_root.resolve(),
        args.pvtz_root.resolve(),
        args.output_root.resolve(),
        Thresholds(),
    )


if __name__ == "__main__":
    main()
