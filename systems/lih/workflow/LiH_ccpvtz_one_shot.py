#!/usr/bin/env python3
"""One-shot, within-basis cc-pVTZ reproduction of the LiH fixed-window audit.

This wrapper deliberately does not transport states between basis sets.  It makes
a private cc-pVTZ copy of the already validated S3 aug-cc-pVTZ driver, compiles
that copy, runs its complete ``all`` workflow, and independently audits the
resulting decision JSON.  Equality of route-permutation arrays across different
bases is recorded but is not a gate; the physical assignment gate is whether the
two routes agree within cc-pVTZ.  The original driver is never modified.

Default invocation:

    python LiH_ccpvtz_one_shot.py --threads 96

Use ``--overwrite`` only when an existing cc-pVTZ work directory is meant to be
recomputed.  ``--prepare-only`` writes and syntax-checks the generated driver
without starting quantum-chemistry calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKFLOW_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = WORKFLOW_DIR.parent
DEFAULT_DRIVER = WORKFLOW_DIR / "LiH_S3_fixed_window_basis_robustness.py"
DEFAULT_HISTORY_DRIVER = WORKFLOW_DIR / "LiH_S2_physical_field_confirmation.py"
DEFAULT_REFERENCE = SYSTEM_DIR / "reference" / "aug-cc-pVDZ"
DEFAULT_ROOT = Path.cwd() / "work" / "LiH_S5_fixed_window_ccpvtz_work"
WRAPPER_VERSION = "LiH-cc-pVTZ-one-shot-v2.0"

EXPECTED_WINDOW = {
    "R_min_angstrom": 5.8000,
    "R_max_angstrom": 5.9500,
    "Fz_min_au": 0.010500,
    "Fz_max_au": 0.011500,
}

THRESHOLDS = {
    "maximum_final_orbital_gradient": 1.0e-4,
    "eta_edge_minimum": 0.98,
    "maximum_edge_leakage": 0.01,
    "minimum_normalized_assignment_margin": 0.10,
    "minimum_C_qp": 1.0e-4,
    "minimum_normalized_dipole_path_difference": 1.0e-5,
    "minimum_excluded_root_gap_Eh": 1.0e-2,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patch_driver(source: Path, destination: Path) -> dict[str, Any]:
    """Create a narrowly patched cc-pVTZ driver and return provenance."""
    text = source.read_text(encoding="utf-8")
    original = text

    # Only the target triple-zeta basis is changed.  References to the original
    # aug-cc-pVDZ calculation are intentionally retained for contextual ratios.
    substitutions = (
        (r"aug-cc-pVTZ", "cc-pVTZ"),
        (r"aug-cc-pvtz", "cc-pvtz"),
        (r"AUG-CC-PVTZ", "CC-PVTZ"),
        (r"augccpvtz", "ccpvtz"),
        (r"AUGCCPVTZ", "CCPVTZ"),
        (r"aug_cc_pvtz", "cc_pvtz"),
        (r"AUG_CC_PVTZ", "CC_PVTZ"),
        (r"LiH_S3_fixed_window_augccpvtz_work", "LiH_S5_fixed_window_ccpvtz_work"),
        (r"basis enlargement from aug-cc-pVDZ to cc-pVTZ", "independent basis change from aug-cc-pVDZ to cc-pVTZ"),
        (r"Basis ratios pVTZ/pVDZ", "Basis ratios cc-pVTZ/aug-cc-pVDZ"),
        (r"C_qp_ratio_pvtz_over_pvdz", "C_qp_ratio_ccpvtz_over_augccpvdz"),
        (r"dipole_ratio_pvtz_over_pvdz", "dipole_ratio_ccpvtz_over_augccpvdz"),
        (r'"pvtz_route_permutations"', '"ccpvtz_route_permutations"'),
        (r"pVTZ history independence \(not tested in S3\)", "cc-pVTZ history independence unless the appended opposed-history gate passes"),
    )
    counts: dict[str, int] = {}
    for old, new in substitutions:
        text, count = re.subn(old, new, text)
        if count:
            counts[f"{old} -> {new}"] = count

    # The original S3 robustness script required the literal route-permutation
    # arrays to equal those of aug-cc-pVDZ.  Root numbering and endpoint frames
    # are basis-specific gauge choices, so that equality is useful metadata but
    # not a valid cross-basis acceptance gate.  We retain the demanding physical
    # condition: both routes must give the same composed assignment *within*
    # cc-pVTZ.  This is a semantic correction, not a threshold relaxation.
    old_assignment_gate = '''    assignment_conclusion_reproduced = bool(
        direct_result["same_composed_assignment"]
        and pvtz_permutations == reference_permutations
    )'''
    new_assignment_gate = '''    cross_basis_permutation_arrays_same = bool(
        pvtz_permutations == reference_permutations
    )
    assignment_conclusion_reproduced = bool(
        direct_result["same_composed_assignment"]
    )'''
    if old_assignment_gate not in text:
        raise RuntimeError(
            "The expected S3 cross-basis assignment gate was not found. "
            "Refusing to alter unfamiliar source code."
        )
    text = text.replace(old_assignment_gate, new_assignment_gate, 1)
    counts[
        "cross-basis permutation equality: gating -> diagnostic"
    ] = 1

    old_check = (
        '        "conventional_composed_assignment_reproduced": '
        'assignment_conclusion_reproduced,\n'
    )
    new_check = old_check + (
        '        "cross_basis_permutation_arrays_same_non_gating": '
        'cross_basis_permutation_arrays_same,\n'
    )
    if old_check not in text:
        raise RuntimeError("The expected S3 assignment decision-check field was not found.")
    text = text.replace(old_check, new_check, 1)

    basis_changes = sum(
        count
        for label, count in counts.items()
        if "pVTZ" in label or "pvtz" in label or "pvtz" in label.lower()
    )
    if basis_changes == 0:
        raise RuntimeError(
            "No aug-cc-pVTZ token was found in the S3 driver. Refusing a blind patch; "
            "inspect the driver and pass the correct file with --driver."
        )
    if re.search(r"aug-cc-pvtz|augccpvtz", text, flags=re.IGNORECASE):
        raise RuntimeError("An aug-cc-pVTZ token remains after patching; refusing to run.")

    banner = (
        "# AUTO-GENERATED by LiH_ccpvtz_one_shot.py; do not edit.\n"
        f"# Source: {source}\n"
        f"# Source SHA256: {hashlib.sha256(original.encode('utf-8')).hexdigest()}\n"
        "# Scientific changes: aug-cc-pVTZ -> cc-pVTZ; cross-basis root-permutation\n"
        "# equality is diagnostic, while within-cc-pVTZ route agreement remains gating.\n\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(banner + text, encoding="utf-8")
    py_compile.compile(str(destination), doraise=True)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "generated_driver": str(destination),
        "generated_sha256": sha256(destination),
        "substitutions": counts,
    }


def walk_items(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from walk_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_items(value)


def values_for_keys(obj: Any, names: Iterable[str]) -> list[Any]:
    wanted = {name.lower() for name in names}
    return [value for key, value in walk_items(obj) if key.lower() in wanted]


def finite_numbers(values: Iterable[Any]) -> list[float]:
    answer: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            answer.append(number)
    return answer


def first_text(obj: Any, names: Iterable[str]) -> str | None:
    for value in values_for_keys(obj, names):
        if isinstance(value, str):
            return value
    return None


def extrema(obj: Any, names: Iterable[str], mode: str) -> float | None:
    numbers = finite_numbers(values_for_keys(obj, names))
    if not numbers:
        return None
    return min(numbers) if mode == "min" else max(numbers)


def find_decision(root: Path) -> Path:
    preferred = [
        root / "LiH_S5_decision.json",
        root / "LiH_S3_decision.json",
        root / "decision.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(
        root.rglob("*decision*.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No decision JSON was produced below {root}")
    return candidates[0]


def audit_decisions(s3: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    """Audit the exact S3/S2 schemas; missing fields are hard failures."""

    model = s3.get("model", {})
    fixed = s3.get("fixed_window_result", {})
    s3_checks = s3.get("decision_checks", {})
    h_checks = history.get("decision_checks", {})
    worst = s3.get("worst_elementary_state_edge", {}).get("state_subspace", {})

    basis = str(model.get("basis", "")).strip().lower()
    gradient = s3.get("maximum_final_orbital_gradient")
    eta = worst.get("sigma_min")
    leakage = worst.get("leakage")
    margin = worst.get("normalized_margin")
    c_qp = fixed.get("C_qp")
    dipole = fixed.get("dipole_path_difference_normalized")
    excluded_gap = fixed.get("minimum_excluded_root_gap_inside_window_Eh")
    target_gap = fixed.get("minimum_target_pair_gap_inside_window_Eh")

    window_matches = all(
        abs(float(fixed.get(key, float("nan"))) - expected) <= 5.0e-12
        for key, expected in {
            "R0_angstrom": EXPECTED_WINDOW["R_min_angstrom"],
            "R1_angstrom": EXPECTED_WINDOW["R_max_angstrom"],
            "F0_au": EXPECTED_WINDOW["Fz_min_au"],
            "F1_au": EXPECTED_WINDOW["Fz_max_au"],
        }.items()
    )

    required_s3_flags = (
        "reference_decisions_locked_and_valid",
        "model_semantics_exact",
        "all_wavefunctions_stationary",
        "first_excluded_A1_root_isolated",
        "numerical_gauge_invariance",
        "all_elementary_grid_edges_strict",
        "all_four_fixed_boundary_edges_strict",
        "fixed_window_near_degeneracy_reproduced",
        "conventional_composed_assignment_reproduced",
        "nonzero_continuous_holonomy_reproduced",
        "observable_dipole_consequence_reproduced",
    )
    required_history_flags = (
        "S1_candidate_window_found",
        "all_history_wavefunctions_stationary",
        "direct_vs_two_opposed_histories_same_solution",
        "selected_window_candidate_under_all_histories",
        "selected_window_C_qp_history_invariant",
        "selected_window_dipole_consequence_history_invariant",
        "conventional_composed_assignment_same",
        "nonzero_continuous_holonomy",
        "observable_dipole_consequence",
    )

    checks: dict[str, dict[str, Any]] = {
        "basis_is_cc_pVTZ": {
            "passed": basis == "cc-pvtz",
            "value": basis,
            "required": "cc-pVTZ (not aug-cc-pVTZ)",
        },
        "fixed_window_unchanged": {
            "passed": window_matches,
            "value": {
                key: fixed.get(key)
                for key in ("R0_angstrom", "R1_angstrom", "F0_au", "F1_au")
            },
            "required": EXPECTED_WINDOW,
        },
        "S3_driver_gate": {
            "passed": s3.get("status") == "BASIS_ROBUSTNESS_CONFIRMED"
            and bool(s3.get("manuscript_claim_authorized")),
            "value": s3.get("status"),
            "required": "BASIS_ROBUSTNESS_CONFIRMED",
        },
        "all_S3_decision_checks": {
            "passed": all(s3_checks.get(key) is True for key in required_s3_flags),
            "value": {key: s3_checks.get(key) for key in required_s3_flags},
            "required": "all True",
        },
        "stationarity": {
            "passed": isinstance(gradient, (int, float))
            and gradient <= THRESHOLDS["maximum_final_orbital_gradient"],
            "value": gradient,
            "required": f"gradient <= {THRESHOLDS['maximum_final_orbital_gradient']:.1e}",
        },
        "edge_conditioning": {
            "passed": s3_checks.get("all_elementary_grid_edges_strict") is True
            and s3_checks.get("all_four_fixed_boundary_edges_strict") is True,
            "value": eta,
            "required": f"all edges eta >= {THRESHOLDS['eta_edge_minimum']}",
        },
        "edge_leakage": {
            "passed": s3_checks.get("all_elementary_grid_edges_strict") is True
            and s3_checks.get("all_four_fixed_boundary_edges_strict") is True,
            "value": leakage,
            "required": f"all edges leakage <= {THRESHOLDS['maximum_edge_leakage']}",
        },
        "assignment_margin": {
            "passed": s3_checks.get("all_elementary_grid_edges_strict") is True
            and s3_checks.get("all_four_fixed_boundary_edges_strict") is True,
            "value": margin,
            "required": f"all edges margin >= {THRESHOLDS['minimum_normalized_assignment_margin']}",
        },
        "same_conventional_assignment": {
            "passed": s3_checks.get("conventional_composed_assignment_reproduced") is True
            and fixed.get("same_composed_assignment") is True,
            "value": fixed.get("same_composed_assignment"),
            "required": True,
        },
        "nonzero_nonclosure": {
            "passed": isinstance(c_qp, (int, float))
            and c_qp >= THRESHOLDS["minimum_C_qp"],
            "value": c_qp,
            "required": f">= {THRESHOLDS['minimum_C_qp']:.1e}",
        },
        "observable_consequence": {
            "passed": isinstance(dipole, (int, float))
            and dipole >= THRESHOLDS["minimum_normalized_dipole_path_difference"],
            "value": dipole,
            "required": f">= {THRESHOLDS['minimum_normalized_dipole_path_difference']:.1e}",
        },
        "excluded_root_gap": {
            "passed": isinstance(excluded_gap, (int, float))
            and excluded_gap >= THRESHOLDS["minimum_excluded_root_gap_Eh"],
            "value": excluded_gap,
            "required": f">= {THRESHOLDS['minimum_excluded_root_gap_Eh']:.2e} Eh",
        },
        "target_pair_near_degeneracy": {
            "passed": isinstance(target_gap, (int, float)) and target_gap <= 1.0e-2,
            "value": target_gap,
            "required": "<= 1.00e-2 Eh",
        },
        "opposed_history_driver_gate": {
            "passed": history.get("status") == "PUBLICATION_DEMONSTRATION_READY"
            and bool(history.get("manuscript_claim_authorized")),
            "value": history.get("status"),
            "required": "PUBLICATION_DEMONSTRATION_READY",
        },
        "all_history_decision_checks": {
            "passed": all(h_checks.get(key) is True for key in required_history_flags),
            "value": {key: h_checks.get(key) for key in required_history_flags},
            "required": "all True",
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "gate": "LiH independent within-basis cc-pVTZ fixed-window reproduction",
        "status": "PASS_CC_PVTZ_REPRODUCTION" if passed else "STOP_CC_PVTZ_REPRODUCTION",
        "passed": passed,
        "thresholds": THRESHOLDS,
        "expected_window": EXPECTED_WINDOW,
        "checks": checks,
        "reported_S3_status": s3.get("status"),
        "reported_history_status": history.get("status"),
        "interpretation_if_passed": (
            "The LiH closed-loop signal is reproduced independently with cc-pVTZ on the "
            "same physical R-Fz window; the conclusion does not rely on a single basis."
        ),
        "scope": (
            "This is a within-basis reproduction. Cross-basis wave-function overlaps are "
            "neither required nor interpreted."
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_and_tee(command: list[str], log_path: Path, pythonpath: Path) -> int:
    print("RUN:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(pythonpath)
            if not old_pythonpath
            else str(pythonpath) + os.pathsep + old_pythonpath
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and audit the LiH fixed-window calculation independently with cc-pVTZ."
    )
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--history-driver", type=Path, default=DEFAULT_HISTORY_DRIVER)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    driver = args.driver.expanduser().resolve()
    history_driver = args.history_driver.expanduser().resolve()
    reference = args.reference_root.expanduser().resolve()
    root = args.root.expanduser().resolve()

    if not driver.is_file():
        raise FileNotFoundError(f"S3 driver not found: {driver}")
    if not history_driver.is_file():
        raise FileNotFoundError(f"S2 history driver not found: {history_driver}")
    if not reference.is_dir():
        raise FileNotFoundError(f"S1 reference root not found: {reference}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    root.mkdir(parents=True, exist_ok=True)
    generated = root / "LiH_S5_fixed_window_ccpvtz_generated.py"
    provenance = patch_driver(driver, generated)
    provenance.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "wrapper_version": WRAPPER_VERSION,
            "reference_root": str(reference),
            "work_root": str(root),
            "threads": args.threads,
        }
    )
    write_json(root / "LiH_S5_generation_provenance.json", provenance)

    print(f"Generated and syntax-checked: {generated}")
    print(f"Original driver unchanged: {driver}")
    print("Substitutions:")
    for label, count in provenance["substitutions"].items():
        print(f"  {count:3d}  {label}")

    if args.prepare_only:
        print("PREPARE_ONLY=COMPLETE")
        return

    command = [
        sys.executable,
        str(generated),
        "all",
        "--reference-root",
        str(reference),
        "--root",
        str(root),
        "--threads",
        str(args.threads),
    ]
    if args.overwrite:
        command.append("--overwrite")

    return_code = run_and_tee(command, root / "LiH_S5_ccpvtz_run.log", driver.parent)
    if return_code != 0:
        print(f"\nLIH_CC_PVTZ_ONE_SHOT=DRIVER_FAILURE (exit {return_code})", file=sys.stderr)
        raise SystemExit(return_code)

    decision_path = root / "LiH_S3_decision.json"
    if not decision_path.is_file():
        decision_path = find_decision(root)
    s3_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if s3_decision.get("status") != "BASIS_ROBUSTNESS_CONFIRMED":
        print("\nS3 did not pass; opposed-history calculations will not be started.", file=sys.stderr)
        empty_history = {"status": "NOT_RUN_BECAUSE_S3_FAILED", "decision_checks": {}}
        audit = audit_decisions(s3_decision, empty_history)
        audit_path = root / "LiH_S5_ccpvtz_independent_audit.json"
        audit.update({"source_decision_file": str(decision_path)})
        write_json(audit_path, audit)
        print(f"Audit file   : {audit_path}")
        print("LIH_CC_PVTZ_ONE_SHOT=STOP")
        raise SystemExit(2)

    # S2 expects an S1-shaped locked-window record in the same directory as
    # the direct solutions.  This bridge does not select a new window: its four
    # coordinates are copied verbatim from the passed S3 fixed result.
    fixed = s3_decision["fixed_window_result"]
    bridge = {
        "gate": "Adapter record for cc-pVTZ opposed-history confirmation",
        "status": "CANDIDATE_WINDOW_FOUND",
        "synthetic_adapter": True,
        "selection_prohibited": True,
        "source_decision": str(decision_path),
        "selected_candidate_window": {
            "R0_angstrom": fixed["R0_angstrom"],
            "R1_angstrom": fixed["R1_angstrom"],
            "F0_au": fixed["F0_au"],
            "F1_au": fixed["F1_au"],
        },
    }
    bridge_path = root / "LiH_S1_decision.json"
    if bridge_path.exists() and not args.overwrite:
        existing_bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        if existing_bridge != bridge:
            raise RuntimeError(
                f"Refusing to replace a nonmatching bridge at {bridge_path}; "
                "use a fresh --root or inspect the file."
            )
    else:
        write_json(bridge_path, bridge)

    adapter = root / "LiH_S5_ccpvtz_opposed_history_adapter.py"
    adapter_source = f'''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

SOURCE = Path({str(driver.parent)!r})
ROOT = Path({str(root)!r})
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT))

import LiH_S0_SA4_qualification as s0
import LiH_S1_physical_field_locator as s1
import LiH_S2_physical_field_confirmation as s2
import LiH_S5_fixed_window_ccpvtz_generated as s5

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
s0.require_pyscf()
import pyscf
pyscf.lib.num_threads(args.threads)

protocol = s1.Protocol(
    basis="cc-pvtz",
    r_grid_angstrom=(5.80, 5.85, 5.90, 5.95),
    field_grid_au=(0.01050, 0.01075, 0.01100, 0.01125, 0.01150),
)
original_builder = s0.build_initial_mos
try:
    s0.build_initial_mos = s5.build_initial_mos_pvtz
    s2.run_protocol(ROOT, protocol, args.overwrite)
finally:
    s0.build_initial_mos = original_builder
s2.analyze_protocol(ROOT, protocol, s2.Thresholds())
'''
    adapter.write_text(adapter_source, encoding="utf-8")
    py_compile.compile(str(adapter), doraise=True)
    history_command = [sys.executable, str(adapter), "--threads", str(args.threads)]
    if args.overwrite:
        history_command.append("--overwrite")
    return_code = run_and_tee(
        history_command, root / "LiH_S5_ccpvtz_history_run.log", driver.parent
    )
    if return_code != 0:
        print(
            f"\nLIH_CC_PVTZ_ONE_SHOT=HISTORY_DRIVER_FAILURE (exit {return_code})",
            file=sys.stderr,
        )
        raise SystemExit(return_code)

    history_path = root / "LiH_S2_decision.json"
    history_decision = json.loads(history_path.read_text(encoding="utf-8"))
    audit = audit_decisions(s3_decision, history_decision)
    audit.update(
        {
            "source_S3_decision_file": str(decision_path),
            "source_history_decision_file": str(history_path),
            "generated_driver": str(generated),
            "generation_provenance": str(root / "LiH_S5_generation_provenance.json"),
        }
    )
    audit_path = root / "LiH_S5_ccpvtz_independent_audit.json"
    write_json(audit_path, audit)

    print("\n=== INDEPENDENT CC-PVTZ ACCEPTANCE ===")
    for name, item in audit["checks"].items():
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"{mark:4s}  {name:31s} value={item['value']!r}  required={item['required']}")
    print(f"\nStatus       : {audit['status']}")
    print(f"Decision file: {decision_path}")
    print(f"History file : {history_path}")
    print(f"Audit file   : {audit_path}")
    print(f"LIH_CC_PVTZ_ONE_SHOT={'PASS' if audit['passed'] else 'STOP'}")
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
