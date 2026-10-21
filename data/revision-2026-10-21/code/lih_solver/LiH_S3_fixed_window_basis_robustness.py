#!/usr/bin/env python3
"""LiH S3: locked-window aug-cc-pVTZ robustness test.

This is the final, deliberately narrow robustness calculation following the
aug-cc-pVDZ LiH S1 locator and S2 opposed-history confirmation.  S3 does not
search for a new hotspot.  It reads the completed S1/S2 decision files,
verifies their statuses and frozen coordinates, and repeats the same physical
R x static-field window with aug-cc-pVTZ.

Fixed electronic model
----------------------
Equal-weight singlet-A1 SA4-CASSCF(2e,5o), one optimized inactive core, no
frozen orbital, active irreps A1^3 B1^1 B2^1, and a fifth A1 CASCI root as an
external-root-isolation probe.  State overlaps are exact active-only
CAS-wavefunction overlaps.  Every S3 vertex starts independently from its own
field-specific RHF solution.

Fixed physical window
---------------------
R = 5.80, 5.85, 5.90, 5.95 Angstrom
Fz = 0.01050, 0.01075, 0.01100, 0.01125, 0.01150 atomic units

The Hamiltonian convention is H(Fz)=H(0)-mu_z Fz, with both electronic and
nuclear field terms.  LiH is neutral, so the total energy is origin
independent.  Passing S3 establishes fixed-window basis robustness, not a
complete-basis limit or experimental field accessibility.

Typical use
-----------
python LiH_S3_fixed_window_basis_robustness.py selftest
python LiH_S3_fixed_window_basis_robustness.py all \
  --reference-root /path/to/LiH_S1_field_locator_work \
  --root /path/to/LiH_S3_fixed_window_augccpvtz_work \
  --threads 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import LiH_S0_SA4_qualification as s0
import LiH_S1_physical_field_locator as s1


SCRIPT_VERSION = "LiH-S3-fixed-window-basis-robustness-v1.1"
SEED_SELECTOR_VERSION = "occupied-valence-locked-projector-v1"
BASIS = "aug-cc-pvtz"
R_GRID_ANGSTROM = (5.80, 5.85, 5.90, 5.95)
FIELD_GRID_AU = (0.01050, 0.01075, 0.01100, 0.01125, 0.01150)
TARGET_PAIR = (2, 3)
EXPECTED_REFERENCE_S1_STATUS = "CANDIDATE_WINDOW_FOUND"
EXPECTED_REFERENCE_S2_STATUS = "PUBLICATION_DEMONSTRATION_READY"


@dataclass(frozen=True)
class Thresholds:
    eta_edge_minimum: float = 0.98
    maximum_edge_leakage: float = 0.01
    minimum_normalized_assignment_margin: float = 0.10
    active_orbital_subspace_sigma_minimum: float = 0.90
    active_orbital_subspace_leakage_maximum: float = 0.10
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square: float = 1.0e-7
    maximum_mo_orthonormality_error: float = 1.0e-9
    excluded_root_minimum_gap_Eh: float = 1.0e-2
    probe_retained_energy_maximum_error_Eh: float = 1.0e-8
    target_pair_maximum_gap_Eh: float = 1.0e-2
    minimum_C_qp: float = 1.0e-4
    minimum_normalized_dipole_path_difference: float = 1.0e-5
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


def load_and_validate_reference(reference_root: Path) -> Dict[str, Any]:
    s1_path = reference_root / "LiH_S1_decision.json"
    s2_path = reference_root / "LiH_S2_decision.json"
    s1_decision = read_json(s1_path)
    s2_decision = read_json(s2_path)

    if s1_decision.get("status") != EXPECTED_REFERENCE_S1_STATUS:
        raise RuntimeError(
            f"reference S1 status must be {EXPECTED_REFERENCE_S1_STATUS}, got "
            f"{s1_decision.get('status')!r}"
        )
    if s2_decision.get("status") != EXPECTED_REFERENCE_S2_STATUS:
        raise RuntimeError(
            f"reference S2 status must be {EXPECTED_REFERENCE_S2_STATUS}, got "
            f"{s2_decision.get('status')!r}"
        )
    if not bool(s2_decision.get("manuscript_claim_authorized")):
        raise RuntimeError("reference S2 did not authorize the molecular claim")
    if str(s1_decision.get("model", {}).get("basis", "")).lower() != "aug-cc-pvdz":
        raise RuntimeError("reference S1 basis must be aug-cc-pVDZ")

    fixed = s2_decision.get("fixed_selected_window", {})
    expected = {
        "R0_angstrom": R_GRID_ANGSTROM[0],
        "R1_angstrom": R_GRID_ANGSTROM[-1],
        "F0_au": FIELD_GRID_AU[0],
        "F1_au": FIELD_GRID_AU[-1],
    }
    for key, value in expected.items():
        if key not in fixed or not close_scalar(fixed[key], value):
            raise RuntimeError(
                f"reference fixed window mismatch for {key}: "
                f"expected {value}, got {fixed.get(key)!r}"
            )

    direct = s2_decision.get("selected_window_results", {}).get("direct", {})
    if not bool(direct.get("candidate_window")):
        raise RuntimeError("reference S2 direct window is not a passed candidate")
    if not bool(direct.get("all_four_direct_edges_strict")):
        raise RuntimeError("reference S2 direct window does not have four strict edges")
    if not bool(direct.get("same_composed_assignment")):
        raise RuntimeError("reference S2 routes do not have the same assignment")

    return {
        "S1_path": str(s1_path.resolve()),
        "S2_path": str(s2_path.resolve()),
        "S1_sha256": file_sha256(s1_path),
        "S2_sha256": file_sha256(s2_path),
        "S1_status": s1_decision["status"],
        "S2_status": s2_decision["status"],
        "basis": s1_decision["model"]["basis"],
        "fixed_window": fixed,
        "direct_result": direct,
    }


def protocol() -> s1.Protocol:
    return s1.Protocol(BASIS, R_GRID_ANGSTROM, FIELD_GRID_AU)


def solution_path(root: Path, r_angstrom: float, field_au: float) -> Path:
    return s1.solution_path(root, r_angstrom, field_au)


def build_initial_mos_pvtz(mf) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the same CAS(2e,5o) seed without losing the occupied valence MO.

    The original S0/S1 selector ranks every non-core orbital only by its total
    target-AO population.  That ranking is adequate for aug-cc-pVDZ, but in
    aug-cc-pVTZ several diffuse virtual A1 orbitals can outrank the occupied
    Li--H valence orbital.  CAS(2e,5o) has exactly one non-core occupied RHF
    orbital by construction, so its inclusion is a model semantic, not an
    adjustable selection threshold.  Lock it into the A1 quota and use the
    unchanged projector ranking only for the remaining virtual slots.
    """
    mol = mf.mol
    mo = np.asarray(mf.mo_coeff)
    occ = np.where(np.asarray(mf.mo_occ) > 0.0)[0].tolist()
    expected_ncore = (mol.nelectron - s0.NELECAS) // 2
    if expected_ncore != 1 or len(occ) != expected_ncore + 1:
        raise RuntimeError(
            f"unexpected LiH occupation: ncore={expected_ncore}, occupied MOs={len(occ)}"
        )
    core = [min(occ, key=lambda i: float(mf.mo_energy[i]))]
    occupied_valence = [i for i in occ if i not in core]
    if len(occupied_valence) != 1:
        raise RuntimeError(
            "CAS(2e,5o) requires exactly one non-core occupied RHF orbital; "
            f"got {occupied_valence}"
        )
    candidates = [i for i in range(mo.shape[1]) if i not in core]
    s_ao = mf.get_ovlp()

    targets = {
        "Li_2s": s0.ao_indices(mol, "li", "2s"),
        "Li_2pz": s0.ao_indices(mol, "li", "2p", "z"),
        "H_1s": s0.ao_indices(mol, "h", "1s"),
        "H_2px": s0.ao_indices(mol, "h", "2p", "x"),
        "H_2py": s0.ao_indices(mol, "h", "2p", "y"),
    }
    missing = [name for name, indices in targets.items() if not indices]
    if missing:
        raise RuntimeError(
            f"could not identify target AO labels: {missing}; labels={mol.ao_labels()}"
        )
    scores_by_target = {
        name: s0.subspace_population(mo, s_ao, indices)
        for name, indices in targets.items()
    }
    total_score = sum(scores_by_target.values())
    irreps = s0.orbital_irrep_names(mol, mo)
    occupied_valence_irrep = irreps[occupied_valence[0]]
    if occupied_valence_irrep != "A1":
        raise RuntimeError(
            "the unique occupied LiH valence MO must be A1 for the locked model; "
            f"got {occupied_valence_irrep}"
        )

    active: List[int] = []
    selected_by_irrep: Dict[str, List[int]] = {}
    for irrep, count in s0.ACTIVE_IRREP_COUNTS.items():
        required = occupied_valence if irrep == occupied_valence_irrep else []
        if len(required) > count:
            raise RuntimeError(
                f"required occupied orbitals exceed the {irrep} active quota"
            )
        eligible = [
            i
            for i in candidates
            if irreps[i] == irrep and i not in required
        ]
        eligible.sort(
            key=lambda i: (float(total_score[i]), -float(mf.mo_energy[i])),
            reverse=True,
        )
        needed = count - len(required)
        if len(eligible) < needed:
            raise RuntimeError(
                f"not enough {irrep} orbitals for active-space quota {count}"
            )
        chosen = list(required) + eligible[:needed]
        selected_by_irrep[irrep] = chosen
        active.extend(chosen)

    active.sort()
    if len(active) != s0.NCAS or len(set(active)) != s0.NCAS:
        raise RuntimeError(f"invalid active selection: {active}")
    active_occupied = [i for i in active if i in occ]
    if active_occupied != occupied_valence:
        raise RuntimeError(
            "occupied-valence lock failed: "
            f"expected {occupied_valence}, selected {active_occupied}"
        )
    if Counter(irreps[i] for i in active) != Counter(s0.ACTIVE_IRREP_COUNTS):
        raise RuntimeError(
            "active irrep quota changed during occupied-valence locking: "
            f"{[irreps[i] for i in active]}"
        )

    external = [
        i for i in range(mo.shape[1]) if i not in core and i not in active
    ]
    order = core + active + external
    if len(order) != mo.shape[1] or len(set(order)) != mo.shape[1]:
        raise RuntimeError("MO reordering is not a permutation")

    diagnostics = {
        "selector_version": SEED_SELECTOR_VERSION,
        "selector_semantics": (
            "lock the unique non-core occupied RHF orbital into the A1 active "
            "quota; fill all remaining symmetry quotas by the unchanged target-AO "
            "projector ranking"
        ),
        "ncore": expected_ncore,
        "core_canonical_index": core[0],
        "occupied_valence_canonical_index": occupied_valence[0],
        "occupied_valence_irrep": occupied_valence_irrep,
        "active_canonical_indices": active,
        "active_irrep_names": [irreps[i] for i in active],
        "selected_by_irrep": selected_by_irrep,
        "active_target_population_total": [float(total_score[i]) for i in active],
        "target_populations_on_active": {
            name: [float(scores[i]) for i in active]
            for name, scores in scores_by_target.items()
        },
        "target_ao_indices": targets,
        "mo_reordering": order,
    }
    return mo[:, order], diagnostics


def run_protocol(
    root: Path,
    reference: Mapping[str, Any],
    overwrite: bool,
) -> None:
    cfg = protocol()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "engine_script_version": s1.SCRIPT_VERSION,
        "gate": "LiH S3 locked-window aug-cc-pVTZ robustness",
        "scientific_semantics": (
            "Repeat the already selected aug-cc-pVDZ LiH R x Fz window at "
            "aug-cc-pVTZ without coordinate relocation, threshold relaxation, or "
            "alternative-window selection."
        ),
        "reference": {
            key: reference[key]
            for key in (
                "S1_path",
                "S2_path",
                "S1_sha256",
                "S2_sha256",
                "S1_status",
                "S2_status",
                "basis",
                "fixed_window",
            )
        },
        "protocol": asdict(cfg),
        "thresholds": asdict(Thresholds()),
        "model_guards": {
            "molecule": "LiH",
            "point_group": "C2v",
            "basis": BASIS,
            "state_symmetry": s0.WFNSYM,
            "active_space": [s0.NELECAS, s0.NCAS],
            "active_irrep_counts": s0.ACTIVE_IRREP_COUNTS,
            "optimized_and_retained_roots": s0.NROOTS,
            "target_pair": list(TARGET_PAIR),
            "state_average_weights": [1.0 / s0.NROOTS] * s0.NROOTS,
            "probe_roots": s0.NPROBE_ROOTS,
            "ncore": 1,
            "frozen_orbitals": 0,
            "edge_metric": "exact active-only CAS-wavefunction overlap",
            "field_hamiltonian": (
                "H(Fz)=H(0)-mu_z*Fz including electronic and nuclear terms"
            ),
            "vertex_initialization": "independent field-specific RHF starts",
            "seed_selector": SEED_SELECTOR_VERSION,
            "seed_selector_guard": (
                "the unique non-core occupied RHF orbital is retained; the remaining "
                "four active orbitals obey the original fixed-irrep projector ranking"
            ),
            "window_selection": "forbidden; coordinates locked to reference S2",
        },
    }
    s0.write_json(root / "LiH_S3_manifest.json", manifest)

    for field in cfg.field_grid_au:
        for r in cfg.r_grid_angstrom:
            target = solution_path(root, r, field)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                print(f"SKIP existing R={r:.8f} Fz={field:+.8f}: {target}", flush=True)
                continue
            print(f"RUN S3 R={r:.8f} Fz={field:+.8f}", flush=True)
            original_builder = s0.build_initial_mos
            try:
                # s1.casscf_job resolves the selector through its shared s0 module.
                # Replace it only for this one call and restore it unconditionally.
                s0.build_initial_mos = build_initial_mos_pvtz
                md = s1.casscf_job(
                    r,
                    field,
                    cfg,
                    target,
                    initial_from=None,
                    history="S3_augccpvtz_independent",
                )
            except Exception as exc:
                s0.write_json(
                    target.with_suffix(".failed.json"),
                    {
                        "script_version": SCRIPT_VERSION,
                        "engine_script_version": s1.SCRIPT_VERSION,
                        "R_angstrom": r,
                        "field_z_au": field,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                s0.build_initial_mos = original_builder
            target.with_suffix(".failed.json").unlink(missing_ok=True)
            print(
                f"DONE S3 R={r:.8f} Fz={field:+.8f} "
                f"gap23={md['target_pair_gap_Eh']:.6e} "
                f"gap5={md['excluded_root_gap_Eh']:.6e} "
                f"grad={md['maximum_orbital_gradient']:.3e}",
                flush=True,
            )
    print("LiH S3 wavefunctions complete. Run the analyzer or use 'all'.", flush=True)


def model_semantics_pass(sol: Mapping[str, Any]) -> bool:
    md = sol["metadata"]
    return bool(
        md.get("script_version") == s1.SCRIPT_VERSION
        and md.get("history") == "S3_augccpvtz_independent"
        and str(md.get("basis", "")).lower() == BASIS
        and md.get("point_group") == "C2v"
        and md.get("target_state_symmetry") == s0.WFNSYM
        and int(md.get("nelecas", -1)) == s0.NELECAS
        and int(md.get("ncas", -1)) == s0.NCAS
        and int(md.get("nroots", -1)) == s0.NROOTS
        and int(md.get("ncore", -1)) == 1
        and int(md.get("frozen_orbitals", -1)) == 0
        and md.get("active_irrep_counts") == s0.ACTIVE_IRREP_COUNTS
        and md.get("active_orbital_selection", {}).get("selector_version")
        == SEED_SELECTOR_VERSION
        and md.get("field_hamiltonian", {}).get("hamiltonian_convention")
        == "H(Fz)=H(0)-mu_z*Fz"
    )


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conj(phases))


def analyze_protocol(
    root: Path,
    reference: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    cfg = protocol()
    nr = len(cfg.r_grid_angstrom)
    nf = len(cfg.field_grid_au)
    solutions: Dict[Tuple[int, int], Dict[str, Any]] = {}
    mols = {
        i: s0.make_mol(r, cfg.basis, verbose=0)
        for i, r in enumerate(cfg.r_grid_angstrom)
    }
    for i, r in enumerate(cfg.r_grid_angstrom):
        for j, field in enumerate(cfg.field_grid_au):
            solutions[(i, j)] = s1.load_solution(solution_path(root, r, field))

    terms_cache: MutableMapping[Tuple[str, int], Any] = {}
    edge_objects: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]] = {}
    edge_records: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]] = {}

    def edge(a: Tuple[int, int], b: Tuple[int, int], kind: str) -> Dict[str, Any]:
        key = (a, b)
        if key not in edge_objects:
            sa = solutions[a]
            sb = solutions[b]
            matrix = s0.active_overlap_matrix(
                sa, sb, mols[a[0]], mols[b[0]], terms_cache
            )
            obj = s0.edge_metrics(matrix)
            orb = s0.active_orbital_metrics(sa, sb, mols[a[0]], mols[b[0]])
            edge_objects[key] = obj
            edge_records[key] = {
                "edge": f"{kind}:R{a[0]}F{a[1]}->R{b[0]}F{b[1]}",
                "kind": kind,
                "R_stride_points": abs(b[0] - a[0]),
                "F_stride_points": abs(b[1] - a[1]),
                "elementary": abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1,
                "from": {
                    "R_angstrom": cfg.r_grid_angstrom[a[0]],
                    "field_z_au": cfg.field_grid_au[a[1]],
                },
                "to": {
                    "R_angstrom": cfg.r_grid_angstrom[b[0]],
                    "field_z_au": cfg.field_grid_au[b[1]],
                },
                "state_subspace": obj["metrics"],
                "active_orbital_subspace": orb,
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

    # Evaluate all elementary edges.  These are local-fidelity diagnostics only;
    # the publication loop uses the four directly evaluated boundary edges below.
    for j in range(nf):
        for i in range(nr - 1):
            edge((i, j), (i + 1, j), "nuclear")
    for i in range(nr):
        for j in range(nf - 1):
            edge((i, j), (i, j + 1), "physical_field")

    a = (0, 0)
    b = (nr - 1, 0)
    c = (0, nf - 1)
    d = (nr - 1, nf - 1)
    ab = edge(a, b, "nuclear")
    bd = edge(b, d, "physical_field")
    ac = edge(a, c, "physical_field")
    cd = edge(c, d, "nuclear")
    boundary_keys = ((a, b), (b, d), (a, c), (c, d))
    route_r_then_f = ab["Q"] @ bd["Q"]
    route_f_then_r = ac["Q"] @ cd["Q"]
    direct_result = s1.route_result(
        route_r_then_f, route_f_then_r, solutions[d]["dipole_au"]
    )

    rng = np.random.default_rng(20260917)
    gauges = {v: random_unitary(s0.NROOTS, rng) for v in (a, b, c, d)}

    def gauged_q(x: Tuple[int, int], y: Tuple[int, int]) -> np.ndarray:
        transformed = (
            gauges[x].conj().T @ edge_objects[(x, y)]["matrix"] @ gauges[y]
        )
        return s0.polar_unitary(transformed)[0]

    gauge_route_1 = gauged_q(a, b) @ gauged_q(b, d)
    gauge_route_2 = gauged_q(a, c) @ gauged_q(c, d)
    gauge_d = gauges[d]
    dipole_d = solutions[d]["dipole_au"]
    gauge_dipole = np.asarray(
        [gauge_d.conj().T @ dipole_d[k] @ gauge_d for k in range(3)]
    )
    gauge_result = s1.route_result(gauge_route_1, gauge_route_2, gauge_dipole)
    gauge_c_error = abs(gauge_result["C_qp"] - direct_result["C_qp"])
    gauge_dipole_error = abs(
        gauge_result["dipole_path_difference_au"]
        - direct_result["dipole_path_difference_au"]
    )

    all_solutions = list(solutions.values())
    gradients = [float(x["metadata"]["maximum_orbital_gradient"]) for x in all_solutions]
    maximum_gradient = max(gradients)
    maximum_s2 = max(
        abs(float(value))
        for sol in all_solutions
        for value in sol["metadata"]["state_S2"]
    )
    maximum_mo_error = max(
        float(sol["metadata"]["mo_orthonormality_max_error"])
        for sol in all_solutions
    )
    maximum_probe_error = max(
        float(sol["metadata"]["probe_retained_energy_max_error_Eh"])
        for sol in all_solutions
    )
    minimum_excluded = min(
        (
            {
                "R_angstrom": float(sol["metadata"]["R_angstrom"]),
                "field_z_au": float(sol["metadata"]["field_z_au"]),
                "gap_Eh": float(sol["metadata"]["excluded_root_gap_Eh"]),
            }
            for sol in all_solutions
        ),
        key=lambda item: item["gap_Eh"],
    )
    minimum_target = min(
        (
            {
                "R_angstrom": float(sol["metadata"]["R_angstrom"]),
                "field_z_au": float(sol["metadata"]["field_z_au"]),
                "gap_Eh": float(sol["metadata"]["target_pair_gap_Eh"]),
            }
            for sol in all_solutions
        ),
        key=lambda item: item["gap_Eh"],
    )

    stationary = bool(
        all(bool(sol["metadata"]["converged"]) for sol in all_solutions)
        and all(np.isfinite(gradients))
        and maximum_gradient <= thresholds.maximum_final_orbital_gradient
        and maximum_s2 <= thresholds.maximum_spin_square
        and maximum_mo_error <= thresholds.maximum_mo_orthonormality_error
    )
    external_isolated = bool(
        minimum_excluded["gap_Eh"] >= thresholds.excluded_root_minimum_gap_Eh
        and maximum_probe_error <= thresholds.probe_retained_energy_maximum_error_Eh
    )
    semantics = all(model_semantics_pass(sol) for sol in all_solutions)
    numerical_validation = bool(
        gauge_c_error <= thresholds.maximum_numerical_check_error
        and gauge_dipole_error <= thresholds.maximum_numerical_check_error
    )

    elementary_records = [rec for rec in edge_records.values() if rec["elementary"]]
    boundary_records = [edge_records[key] for key in boundary_keys]
    all_elementary_strict = bool(elementary_records) and all(
        edge_pass(rec) for rec in elementary_records
    )
    all_boundary_strict = all(edge_pass(rec) for rec in boundary_records)
    reference_direct = reference["direct_result"]
    reference_permutations = (
        reference_direct["route_R_then_F_permutation"],
        reference_direct["route_F_then_R_permutation"],
    )
    pvtz_permutations = (
        direct_result["route_R_then_F_permutation"],
        direct_result["route_F_then_R_permutation"],
    )
    assignment_conclusion_reproduced = bool(
        direct_result["same_composed_assignment"]
        and pvtz_permutations == reference_permutations
    )
    near_degeneracy_reproduced = bool(
        minimum_target["gap_Eh"] <= thresholds.target_pair_maximum_gap_Eh
    )
    holonomy_reproduced = bool(direct_result["C_qp"] >= thresholds.minimum_C_qp)
    dipole_reproduced = bool(
        direct_result["dipole_path_difference_normalized"]
        >= thresholds.minimum_normalized_dipole_path_difference
    )

    checks = {
        "reference_decisions_locked_and_valid": True,
        "model_semantics_exact": semantics,
        "all_wavefunctions_stationary": stationary,
        "first_excluded_A1_root_isolated": external_isolated,
        "numerical_gauge_invariance": numerical_validation,
        "all_elementary_grid_edges_strict": all_elementary_strict,
        "all_four_fixed_boundary_edges_strict": all_boundary_strict,
        "fixed_window_near_degeneracy_reproduced": near_degeneracy_reproduced,
        "conventional_composed_assignment_reproduced": assignment_conclusion_reproduced,
        "nonzero_continuous_holonomy_reproduced": holonomy_reproduced,
        "observable_dipole_consequence_reproduced": dipole_reproduced,
    }

    if not semantics:
        status = "STOP_MODEL_SEMANTICS_FAILURE"
    elif not stationary:
        status = "STOP_WAVEFUNCTION_FAILURE"
    elif not external_isolated:
        status = "STOP_EXTERNAL_ROOT_ISOLATION_FAILURE"
    elif not numerical_validation:
        status = "STOP_NUMERICAL_VALIDATION_FAILURE"
    elif not all_elementary_strict or not all_boundary_strict:
        status = "STOP_EDGE_FIDELITY_FAILURE"
    elif not near_degeneracy_reproduced:
        status = "STOP_FIXED_WINDOW_AVOIDED_CROSSING_NOT_REPRODUCED"
    elif not assignment_conclusion_reproduced:
        status = "STOP_ASSIGNMENT_CONCLUSION_NOT_REPRODUCED"
    elif not holonomy_reproduced:
        status = "STOP_HOLONOMY_NOT_REPRODUCED"
    elif not dipole_reproduced:
        status = "STOP_DIPOLE_CONSEQUENCE_NOT_REPRODUCED"
    else:
        status = "BASIS_ROBUSTNESS_CONFIRMED"

    baseline_c = float(reference_direct["C_qp"])
    baseline_dipole = float(reference_direct["dipole_path_difference_normalized"])
    fixed_result = {
        "R0_angstrom": cfg.r_grid_angstrom[0],
        "R1_angstrom": cfg.r_grid_angstrom[-1],
        "F0_au": cfg.field_grid_au[0],
        "F1_au": cfg.field_grid_au[-1],
        "target_pair": list(TARGET_PAIR),
        "minimum_target_pair_gap_inside_window_Eh": minimum_target["gap_Eh"],
        "minimum_excluded_root_gap_inside_window_Eh": minimum_excluded["gap_Eh"],
        "all_four_direct_edges_strict": all_boundary_strict,
        **direct_result,
        "gauge_test_C_qp_abs_error": gauge_c_error,
        "gauge_test_dipole_abs_error": gauge_dipole_error,
        "edges": {
            "AB": boundary_records[0],
            "BD": boundary_records[1],
            "AC": boundary_records[2],
            "CD": boundary_records[3],
        },
    }

    nonstationary = [
        {
            "path": sol["path"],
            "R_angstrom": sol["metadata"]["R_angstrom"],
            "field_z_au": sol["metadata"]["field_z_au"],
            "converged": sol["metadata"]["converged"],
            "maximum_orbital_gradient": sol["metadata"]["maximum_orbital_gradient"],
            "state_S2": sol["metadata"]["state_S2"],
            "mo_orthonormality_max_error": sol["metadata"][
                "mo_orthonormality_max_error"
            ],
        }
        for sol in all_solutions
        if not bool(sol["metadata"]["converged"])
        or not np.isfinite(float(sol["metadata"]["maximum_orbital_gradient"]))
        or float(sol["metadata"]["maximum_orbital_gradient"])
        > thresholds.maximum_final_orbital_gradient
        or max(abs(float(v)) for v in sol["metadata"]["state_S2"])
        > thresholds.maximum_spin_square
        or float(sol["metadata"]["mo_orthonormality_max_error"])
        > thresholds.maximum_mo_orthonormality_error
    ]

    decision = {
        "gate": "LiH S3 fixed-window aug-cc-pVTZ basis robustness",
        "status": status,
        "manuscript_claim_authorized": status == "BASIS_ROBUSTNESS_CONFIRMED",
        "script_version": SCRIPT_VERSION,
        "engine_script_version": s1.SCRIPT_VERSION,
        "reference": {
            key: reference[key]
            for key in (
                "S1_path",
                "S2_path",
                "S1_sha256",
                "S2_sha256",
                "S1_status",
                "S2_status",
                "basis",
                "fixed_window",
            )
        },
        "model": {
            "molecule": "LiH",
            "basis": BASIS,
            "point_group": "C2v",
            "state_symmetry": s0.WFNSYM,
            "active_space": [s0.NELECAS, s0.NCAS],
            "active_irrep_counts": s0.ACTIVE_IRREP_COUNTS,
            "ncore": 1,
            "frozen_orbitals": 0,
            "optimized_and_retained_roots": s0.NROOTS,
            "target_pair": list(TARGET_PAIR),
            "state_average_weights": [1.0 / s0.NROOTS] * s0.NROOTS,
            "excluded_root_probe": (
                "five-root A1 CASCI on each optimized SA4 orbital set"
            ),
            "edge_metric": "exact active-only CAS-wavefunction overlap",
            "physical_axis": "uniform static Fz with H(Fz)=H(0)-mu_z*Fz",
            "vertex_initialization": "independent field-specific RHF starts",
            "seed_selector": SEED_SELECTOR_VERSION,
        },
        "fixed_grid": {
            "R_angstrom": cfg.r_grid_angstrom,
            "field_z_au": cfg.field_grid_au,
            "selection_allowed": False,
        },
        "thresholds": asdict(thresholds),
        "decision_checks": checks,
        "counts": {
            "wavefunctions": len(all_solutions),
            "elementary_nuclear_edges": (nr - 1) * nf,
            "elementary_physical_field_edges": nr * (nf - 1),
            "direct_fixed_boundary_edges": 4,
            "deterministic_fallback_uses": sum(
                bool(sol["metadata"]["optimizer_protocol"]["fallback_used"])
                for sol in all_solutions
            ),
        },
        "maximum_final_orbital_gradient": maximum_gradient,
        "maximum_state_S2": maximum_s2,
        "maximum_mo_orthonormality_error": maximum_mo_error,
        "maximum_probe_retained_energy_error_Eh": maximum_probe_error,
        "minimum_target_pair_gap": minimum_target,
        "minimum_excluded_root_gap": minimum_excluded,
        "worst_elementary_state_edge": min(
            elementary_records, key=lambda rec: rec["state_subspace"]["sigma_min"]
        ),
        "fixed_window_result": fixed_result,
        "basis_comparison": {
            "aug_cc_pvdz_C_qp": baseline_c,
            "aug_cc_pvtz_C_qp": direct_result["C_qp"],
            "C_qp_ratio_pvtz_over_pvdz": direct_result["C_qp"] / baseline_c,
            "aug_cc_pvdz_normalized_dipole_path_difference": baseline_dipole,
            "aug_cc_pvtz_normalized_dipole_path_difference": direct_result[
                "dipole_path_difference_normalized"
            ],
            "dipole_ratio_pvtz_over_pvdz": direct_result[
                "dipole_path_difference_normalized"
            ]
            / baseline_dipole,
            "reference_route_permutations": reference_permutations,
            "pvtz_route_permutations": pvtz_permutations,
        },
        "failed_elementary_edges": [
            rec for rec in elementary_records if not edge_pass(rec)
        ],
        "failed_boundary_edges": [
            rec for rec in boundary_records if not edge_pass(rec)
        ],
        "nonstationary_wavefunctions": nonstationary,
        "authorized_interpretation": (
            "The fixed LiH R x static-field false-negative demonstration is "
            "reproduced under basis enlargement from aug-cc-pVDZ to aug-cc-pVTZ "
            "without moving the window or changing thresholds."
            if status == "BASIS_ROBUSTNESS_CONFIRMED"
            else "The aug-cc-pVTZ calculation has not established fixed-window basis robustness."
        ),
        "authorized_next_step": (
            "Stop molecular searching and production quantum chemistry; prepare the "
            "main-text LiH figure, revised manuscript, Supporting Information, and "
            "point-by-point response."
            if status == "BASIS_ROBUSTNESS_CONFIRMED"
            else "Stop and diagnose only the first failed S3 gate; do not relocate the window or relax thresholds."
        ),
        "prohibited_interpretations": [
            "complete-basis-set convergence",
            "experimental accessibility of the applied field",
            "a conical intersection proved solely by the avoided crossing",
            "pVTZ history independence (not tested in S3)",
            "coordinate relocation or favorable-window reselection",
            "replacement of active-only overlap by scalar inactive-core division",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "timestamp_unix": time.time(),
        },
    }

    diagnostics = {
        "elementary_edges": elementary_records,
        "fixed_boundary_edges": boundary_records,
        "fixed_window_result": fixed_result,
        "reference_direct_result": reference_direct,
    }
    s0.write_json(root / "LiH_S3_decision.json", decision)
    s0.write_json(root / "LiH_S3_diagnostics.json", diagnostics)

    print("\nLiH S3 RESULT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"Fixed window: R={cfg.r_grid_angstrom[0]:.4f}->{cfg.r_grid_angstrom[-1]:.4f} A; "
        f"Fz={cfg.field_grid_au[0]:+.6f}->{cfg.field_grid_au[-1]:+.6f} au",
        flush=True,
    )
    print(
        f"C_qp={direct_result['C_qp']:.12e}; "
        f"dipole_norm={direct_result['dipole_path_difference_normalized']:.12e}; "
        f"minimum gap23={minimum_target['gap_Eh']:.12e} Eh",
        flush=True,
    )
    print(
        f"Worst elementary edge: "
        f"eta={min(rec['state_subspace']['sigma_min'] for rec in elementary_records):.12f}; "
        f"leakage={max(rec['state_subspace']['leakage'] for rec in elementary_records):.12f}; "
        f"margin={min(rec['state_subspace']['normalized_margin'] for rec in elementary_records):.12f}",
        flush=True,
    )
    print(
        f"Basis ratios pVTZ/pVDZ: C_qp={direct_result['C_qp']/baseline_c:.6f}; "
        f"dipole={direct_result['dipole_path_difference_normalized']/baseline_dipole:.6f}",
        flush=True,
    )
    print(f"Decision file: {root / 'LiH_S3_decision.json'}", flush=True)
    return decision


def selftest() -> None:
    assert BASIS == "aug-cc-pvtz"
    assert SEED_SELECTOR_VERSION == "occupied-valence-locked-projector-v1"
    assert R_GRID_ANGSTROM == (5.80, 5.85, 5.90, 5.95)
    assert FIELD_GRID_AU[0] == 0.01050 and FIELD_GRID_AU[-1] == 0.01150
    assert tuple(sorted(R_GRID_ANGSTROM)) == R_GRID_ANGSTROM
    assert tuple(sorted(FIELD_GRID_AU)) == FIELD_GRID_AU
    assert len(set(R_GRID_ANGSTROM)) == len(R_GRID_ANGSTROM)
    assert len(set(FIELD_GRID_AU)) == len(FIELD_GRID_AU)

    # Regression: diffuse virtual A1 orbitals deliberately outrank the occupied
    # valence orbital.  The locked selector must still retain MO 1 and fill the
    # remaining A1 quota from the two highest-scoring virtuals.
    class FakeMol:
        nelectron = 4

        @staticmethod
        def ao_labels() -> List[str]:
            return [f"AO{i}" for i in range(8)]

    class FakeMF:
        mol = FakeMol()
        mo_coeff = np.eye(8)
        mo_occ = np.asarray([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mo_energy = np.asarray([-2.0, -0.3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

        @staticmethod
        def get_ovlp() -> np.ndarray:
            return np.eye(8)

    fake_scores = np.asarray([0.0, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    fake_irreps = ["A1", "A1", "A1", "A1", "A1", "B1", "B2", "B1"]
    old_ao_indices = s0.ao_indices
    old_population = s0.subspace_population
    old_irreps = s0.orbital_irrep_names
    try:
        s0.ao_indices = lambda *args, **kwargs: [0]
        s0.subspace_population = lambda *args, **kwargs: fake_scores.copy()
        s0.orbital_irrep_names = lambda *args, **kwargs: list(fake_irreps)
        ordered, selector = build_initial_mos_pvtz(FakeMF())
    finally:
        s0.ao_indices = old_ao_indices
        s0.subspace_population = old_population
        s0.orbital_irrep_names = old_irreps
    assert ordered.shape == (8, 8)
    assert selector["occupied_valence_canonical_index"] == 1
    assert selector["active_canonical_indices"] == [1, 2, 3, 5, 6]
    assert selector["active_irrep_names"] == ["A1", "A1", "A1", "B1", "B2"]

    rng = np.random.default_rng(41)
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
    for name in ("run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--reference-root", type=Path, required=True)
        p.add_argument(
            "--root",
            type=Path,
            default=Path("/mnt/LiH/LiH_S3_fixed_window_augccpvtz_work"),
        )
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    if args.threads < 1:
        raise ValueError("--threads must be a positive integer")
    reference = load_and_validate_reference(args.reference_root.resolve())
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    s0.require_pyscf()
    import pyscf

    pyscf.lib.num_threads(args.threads)
    if args.command in ("run", "all"):
        run_protocol(args.root.resolve(), reference, args.overwrite)
    if args.command in ("analyze", "all"):
        analyze_protocol(args.root.resolve(), reference, Thresholds())


if __name__ == "__main__":
    main()
