#!/usr/bin/env python3
"""Audit the decision records used by the manuscript-level claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    path = REPOSITORY / relative
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_lih() -> None:
    pvdz = load("systems/lih/reference/aug-cc-pVDZ/LiH_S2_decision.json")
    pvtz = load("systems/lih/reference/cc-pVTZ/LiH_S3_decision.json")
    require(pvdz["status"] == "PUBLICATION_DEMONSTRATION_READY", "LiH S2 status changed")
    checks = pvdz["decision_checks"]
    for key in (
        "all_history_wavefunctions_stationary",
        "direct_vs_two_opposed_histories_same_solution",
        "conventional_composed_assignment_same",
        "nonzero_continuous_holonomy",
        "observable_dipole_consequence",
    ):
        require(checks[key] is True, f"LiH check failed: {key}")
    direct = pvdz["selected_window_results"]["direct"]
    require(direct["same_composed_assignment"] is True, "LiH assignments no longer agree")
    require(direct["C_qp"] > 0.0, "LiH route nonclosure vanished")
    require(direct["dipole_path_difference_au"] > 0.0, "LiH dipole consequence vanished")
    require(pvtz["status"] == "BASIS_ROBUSTNESS_CONFIRMED", "LiH cc-pVTZ status changed")
    pvtz_checks = pvtz["decision_checks"]
    for key in (
        "all_wavefunctions_stationary",
        "numerical_gauge_invariance",
        "all_elementary_grid_edges_strict",
        "conventional_composed_assignment_reproduced",
        "nonzero_continuous_holonomy_reproduced",
        "observable_dipole_consequence_reproduced",
    ):
        require(pvtz_checks[key] is True, f"LiH cc-pVTZ check failed: {key}")
    pvtz_window = pvtz["fixed_window_result"]
    require(pvtz_window["same_composed_assignment"] is True, "LiH cc-pVTZ assignments differ")
    require(pvtz_window["C_qp"] > 0.0, "LiH cc-pVTZ route nonclosure vanished")


def verify_beh2() -> None:
    result = load("systems/beh2/reference/G0C_forensic_summary.json")
    require(result["status"] == "REVIEW_REQUIRED", "BeH2 review status must be preserved")
    require(
        result["input_decisions"]["g0b_status"] == "CONFIRMED_CONSTRAINT_INDUCED",
        "BeH2 unconstrained comparison changed",
    )
    checks = result["decision_checks"]
    require(checks["legacy_event_present"] is True, "BeH2 legacy event missing")
    require(checks["both_clean_models_single_solution"] is True, "BeH2 clean controls failed")
    require(checks["all_stride1_upper_edges_locally_valid"] is False, "BeH2 limitation disappeared")
    require(checks["main_figure_candidate"] is False, "BeH2 record was over-promoted")


def verify_cyclopentadienyl_cation() -> None:
    base = "systems/cyclopentadienyl-cation/reference"
    projector = load(f"{base}/CPDC_ccpVTZ_rank2_projector_v1_decision.json")
    opposed = load(f"{base}/CPDC_ccpVTZ_rank2_opposed_history_v1_decision.json")
    require(projector["status"] == "PASS_RANK2_PROJECTOR_QUALIFICATION", "CP+ projector status changed")
    require(projector["individual_root_assignment_used"] is False, "CP+ used root assignment")
    require(
        projector["SA3_wedge_arbitration"]["status"] == "PASS_SA3_WEDGE_ARBITRATION",
        "CP+ SA3 external-isolation gate changed",
    )
    for mesh in projector["projector_meshes"]:
        require(mesh["all_points_pass"] is True, "CP+ point-quality gate failed")
        require(mesh["all_projector_edges_pass"] is True, "CP+ edge-fidelity gate failed")
        require(mesh["gauge_pass"] is True, "CP+ U(2)-gauge audit failed")
        require(mesh["individual_root_assignment_used"] is False, "CP+ mesh used root assignment")
    convergence = projector["mesh_40_80_convergence"]["eigenphase_set_difference_rad"]
    require(convergence < 1.0e-5, "CP+ 40/80 mesh convergence changed")
    require(
        opposed["status"] == "PASS_RANK2_OPPOSED_HISTORY_REPLICATION",
        "CP+ opposed-history status changed",
    )
    for key in (
        "quality_pass",
        "cross_history_reproduction_pass",
        "reference_reproduction_pass",
        "seed_rank_pass",
        "result_authorizes_inclusion_as_rank2_manifold_evidence",
    ):
        require(opposed[key] is True, f"CP+ check failed: {key}")
    require(opposed["individual_root_assignment_used"] is False, "CP+ used root assignment")


def main() -> None:
    verifiers = {
        "LiH": verify_lih,
        "BeH2": verify_beh2,
        "cyclopentadienyl cation": verify_cyclopentadienyl_cation,
    }
    for label, verifier in verifiers.items():
        verifier()
        print(f"PASS  {label}")
    print("REFERENCE_AUDIT=PASS")


if __name__ == "__main__":
    main()
