#!/usr/bin/env python3
"""Run every NumPy-only self-test shipped with the archived workflows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]

COMMANDS = {
    "LiH S0": ("systems/lih/workflow", "LiH_S0_SA4_qualification.py", "selftest"),
    "LiH S1": ("systems/lih/workflow", "LiH_S1_physical_field_locator.py", "selftest"),
    "LiH S2": ("systems/lih/workflow", "LiH_S2_physical_field_confirmation.py", "selftest"),
    "LiH S3": ("systems/lih/workflow", "LiH_S3_fixed_window_basis_robustness.py", "selftest"),
    "LiH S4": ("systems/lih/workflow", "LiH_S4_cross_basis_audit.py", "selftest"),
    "CP+ overlap": ("systems/cyclopentadienyl-cation/workflow", "cpdc_overlap.py"),
    "CP+ lift": ("systems/cyclopentadienyl-cation/workflow", "cpdc_lift_engine.py", "selftest"),
    "CP+ landscape": ("systems/cyclopentadienyl-cation/workflow", "CPDC_ccpVTZ_central_gap_landscape_v1.py", "selftest"),
    "CP+ multiring": ("systems/cyclopentadienyl-cation/workflow", "CPDC_ccpVTZ_relaxed_multiring_v1.py", "selftest"),
    "CP+ projector": ("systems/cyclopentadienyl-cation/workflow", "CPDC_ccpVTZ_rank2_projector_v1.py", "selftest"),
    "CP+ opposed history": ("systems/cyclopentadienyl-cation/workflow", "CPDC_ccpVTZ_rank2_opposed_history_v1.py", "selftest"),
    "BeH2 forensic B0": ("systems/beh2/forensic/workflow", "BeH2_B0_analyze.py", "--self-test"),
    "BeH2 forensic B3R": ("systems/beh2/forensic/workflow", "BeH2_B3R_analyze.py", "--self-test"),
    "BeH2 forensic comparison": ("systems/beh2/forensic/workflow", "INVEST_G0C_forensic_comparison.py", "--self-test"),
    "BeH2 antecedent B0 runner": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B0_stationary_solutions.py", "--self-test"),
    "BeH2 antecedent B0 analysis": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B0_analyze.py", "--self-test"),
    "BeH2 antecedent B1 runner": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B1_branch_support.py", "--self-test"),
    "BeH2 antecedent B1 analysis": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B1_analyze.py", "--self-test"),
    "BeH2 antecedent B2 analysis": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B2_plaquettes.py", "--self-test"),
    "BeH2 antecedent B2 runner": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B2_refine.py", "--self-test"),
    "BeH2 antecedent B3R runner": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B3R_scaling.py", "--self-test"),
    "BeH2 antecedent B3R analysis": ("systems/beh2/antecedent_restricted_workflow/scripts", "BeH2_B3R_analyze.py", "--self-test"),
}


def main() -> None:
    for label, command in COMMANDS.items():
        directory, script, *arguments = command
        completed = subprocess.run(
            [sys.executable, script, *arguments],
            cwd=REPOSITORY / directory,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            raise SystemExit(f"FAIL  {label} (exit {completed.returncode})")
        print(f"PASS  {label}")
    print("WORKFLOW_SELFTESTS=PASS")


if __name__ == "__main__":
    main()
