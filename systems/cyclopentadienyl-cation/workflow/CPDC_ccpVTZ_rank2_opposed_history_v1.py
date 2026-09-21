#!/usr/bin/env python3
"""Opposed-history replication of the qualified C5H5+ rank-2 manifold.

The immutable antecedent is a PASS_RANK2_PROJECTOR_QUALIFICATION decision for
the central 80-point cc-pVTZ loop.  This executable performs the preregistered
last robustness test in two separately gated stages.

``history`` launches two sequential equal-weight SA2-CASSCF chains from the
same frozen p000 solution.  One chain reaches every remaining vertex in the
clockwise order and the other in the counterclockwise order.  Both return to
p000 with a fresh closure optimization.  Their point projectors, closure,
edge transports, and Wilson invariants are compared with each other and with
the independently seeded reference mesh.

``sa3`` is code-blocked until the history stage passes.  It uses the clockwise
and counterclockwise SA2 histories as distinct MO seeds for ten equal-weight
SA3 calculations at the five angles already covered by the independent SA3
reference wedge.  The resulting lower-two-state projectors are compared with
the independent SA3 reference and with the opposite history.

No signed one-state overlap, scalar Berry product, individual-root assignment,
or CI count is evaluated.  Only the rank-2 manifold is under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import CPDC_ccpVTZ_rank2_projector_v1 as rank2
import CPDC_ccpVTZ_relaxed_multiring_v1 as antecedent
import cpdc_lift_engine as eng
import cpdc_overlap as ov


SCRIPT_VERSION = "CPDC-CCPVTZ-RANK2-OPPOSED-HISTORY-v1.0"
PROTOCOL_ID = "CPDC-CCPVTZ-RANK2-OPPOSED-HISTORY-20260920"
PROTOCOL_FILENAME = "CPDC_ccpVTZ_rank2_opposed_history_protocol_v1.json"
AUTHORIZATION_FILENAME = "CPDC_ccpVTZ_rank2_opposed_history_authorization_v1.json"
MANIFEST_FILENAME = "CPDC_ccpVTZ_rank2_opposed_history_v1_manifest.json"
HISTORY_DECISION_FILENAME = "CPDC_ccpVTZ_rank2_history_v1_decision.json"
FINAL_DECISION_FILENAME = "CPDC_ccpVTZ_rank2_opposed_history_v1_decision.json"

EXPECTED_RANK2_DRIVER_SHA256 = (
    "710b1628f8cc78bf80f2ed5b721e980d5c560cc0d7e9709ec84e24f610f9e01a"
)
EXPECTED_RANK2_PROTOCOL_SHA256 = (
    "3eb0a624f7264b5229f1ba592f148f801c4ccc7a04476031d7846aaf6082bfec"
)
EXPECTED_ANTECEDENT_DRIVER_SHA256 = (
    "ca724ea4f56428f84043b6fd28f442811f9dfb5107039bad5b325c9264cb2942"
)
EXPECTED_ENGINE_SHA256 = (
    "7ff5d5dab945128e6031e61e2510d836ffd42d0a5e1f86078ef72e9b0ddcd325"
)
EXPECTED_OVERLAP_SHA256 = (
    "f3800d28751ac623bc7c75b8a5ea50d1786754b42ece9e5ca14be6c18b91a429"
)

PYSCF_VERSION_EXACT = "2.13.1"
BASIS_KEY = "cc-pvtz"
NPOINTS = 80
DIRECTIONS = ("clockwise", "counterclockwise")
HISTORY_SA2_ROLE = "rank2_opposed_history_SA2_loop_vertex"
HISTORY_SA3_ROLE = "rank2_opposed_history_SA3_wedge"
# The independent SA3 references were generated on the prior 40-point wedge at
# 0, 9, 18, 27, and 36 degrees.  The same angles are the even vertices below on
# the 80-point history mesh.  Keeping these index spaces explicit prevents the
# 40-point reference labels from being mistaken for the first five 80-point
# vertices (0, 4.5, 9, 13.5, and 18 degrees).
REFERENCE_SPOT_INDICES = tuple(int(index) for index in rank2.SPOT_INDICES)
SPOT_INDICES = tuple(2 * index for index in REFERENCE_SPOT_INDICES)
GAUGE_SEEDS = rank2.GAUGE_SEEDS


@dataclass(frozen=True)
class Thresholds:
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square_error: float = 1.0e-6
    maximum_mo_orthonormality_error: float = 1.0e-8
    maximum_casci_reproduction_error_Eh: float = 1.0e-7
    minimum_active_reference_sigma: float = 0.90
    minimum_third_singlet_gap_Eh: float = 2.0e-3
    history_pair_sigma_minimum: float = 0.98
    history_pair_leakage_maximum: float = 0.01
    history_active_sigma_minimum: float = 0.98
    history_active_leakage_maximum: float = 0.01
    maximum_history_average_energy_difference_Eh: float = 1.0e-7
    maximum_history_gauge_error: float = 1.0e-10
    maximum_history_cyclic_start_error: float = 1.0e-10
    maximum_history_Wilson_eigenphase_difference_rad: float = 1.0e-3
    maximum_history_Wilson_trace_difference: float = 1.0e-3
    maximum_history_Wilson_determinant_difference: float = 1.0e-3
    SA2_SA3_projector_sigma_minimum: float = 0.95
    SA2_SA3_projector_leakage_maximum: float = 0.03
    SA3_projector_sigma_minimum: float = 0.95
    SA3_projector_leakage_maximum: float = 0.03
    SA3_cross_third_weight_maximum: float = 0.05
    maximum_SA3_average_energy_difference_Eh: float = 1.0e-6


@dataclass(frozen=True)
class HistoryPoint:
    direction: str
    index: int
    geometry_hash: str
    reference_path: Path
    target_path: Path
    seed_path: Path
    closure: bool


@dataclass(frozen=True)
class HistorySA3Spec:
    direction: str
    index: int
    angle_degrees: float
    geometry_hash: str
    seed_sa2_path: Path
    reference_sa3_path: Path
    target_sa3_path: Path


def jsonable(value: Any) -> Any:
    return rank2.jsonable(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    rank2.write_json(path, payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_path() -> Path:
    return Path(__file__).resolve().with_name(PROTOCOL_FILENAME)


def authorization_path(root: Path) -> Path:
    return root / AUTHORIZATION_FILENAME


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_FILENAME


def history_decision_path(root: Path) -> Path:
    return root / HISTORY_DECISION_FILENAME


def final_decision_path(root: Path) -> Path:
    return root / FINAL_DECISION_FILENAME


def configure_threads(threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(int(threads))


def require_pyscf() -> None:
    eng.require_pyscf()
    version = __import__("pyscf").__version__
    if version != PYSCF_VERSION_EXACT:
        raise RuntimeError(
            f"opposed-history protocol requires PySCF {PYSCF_VERSION_EXACT}; "
            f"found {version}"
        )


def validate_protocol() -> Dict[str, Any]:
    path = protocol_path()
    if not path.exists():
        raise RuntimeError(f"missing protocol declaration: {path}")
    declaration = json.loads(path.read_text())
    expected = {
        "protocol_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "basis": BASIS_KEY,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "history_mesh_npoints": NPOINTS,
        "directions": list(DIRECTIONS),
        "spot_indices": list(SPOT_INDICES),
        "scalar_phase_allowed": False,
        "individual_root_assignment_allowed": False,
    }
    actual = {
        "protocol_version": declaration.get("protocol_version"),
        "protocol_id": declaration.get("protocol_id"),
        "basis": str(declaration.get("locked_model", {}).get("basis", "")).lower(),
        "pyscf_version_exact": declaration.get("locked_model", {}).get(
            "pyscf_version_exact"
        ),
        "history_mesh_npoints": declaration.get("history_stage", {}).get(
            "mesh_npoints"
        ),
        "directions": declaration.get("history_stage", {}).get("directions"),
        "spot_indices": declaration.get("SA3_stage", {}).get("point_indices"),
        "scalar_phase_allowed": declaration.get("scalar_phase_allowed"),
        "individual_root_assignment_allowed": declaration.get(
            "individual_root_assignment_allowed"
        ),
    }
    if actual != expected:
        raise RuntimeError(f"protocol declaration mismatch: {actual} != {expected}")
    if declaration.get("thresholds") != jsonable(asdict(Thresholds())):
        raise RuntimeError("protocol thresholds differ from executable")
    return declaration


def _mesh80_from_rank2(decision: Mapping[str, Any]) -> Dict[str, Any]:
    meshes = decision.get("projector_meshes", [])
    matches = [item for item in meshes if int(item.get("mesh_npoints", -1)) == NPOINTS]
    if len(matches) != 1:
        raise RuntimeError("rank-2 decision does not contain exactly one 80-point mesh")
    return matches[0]


def build_authorization(
    root: Path,
    source_root: Path,
    loop_root: Path,
    screen_root: Path,
    rank2_root: Path,
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    loop_root = loop_root.resolve()
    screen_root = screen_root.resolve()
    rank2_root = rank2_root.resolve()
    rank2_authorization = rank2.validate_authorization(
        rank2_root, source_root, loop_root, screen_root
    )
    decision = rank2.validate_existing_decision(rank2_root)
    decision_file = rank2.decision_path(rank2_root)
    mesh80 = _mesh80_from_rank2(decision)
    point_records = mesh80.get("point_records", [])
    sa3_records = decision.get("SA3_wedge_arbitration", {}).get("records", [])
    thresholds = rank2.Thresholds()
    convergence = decision.get("mesh_40_80_convergence", {})
    checks: Dict[str, bool] = {
        "rank2_authorization_passed": rank2_authorization.get("status")
        == "PASS_RANK2_PROJECTOR_AUTHORIZATION",
        "rank2_decision_passed": decision.get("status")
        == "PASS_RANK2_PROJECTOR_QUALIFICATION",
        "rank2_phase_not_evaluated": decision.get("scalar_phase_evaluated") is False,
        "rank2_signed_single_state_overlap_absent": decision.get(
            "signed_single_state_overlap_recorded"
        )
        is False,
        "rank2_individual_assignment_absent": decision.get(
            "individual_root_assignment_used"
        )
        is False,
        "rank2_CI_count_not_claimed": decision.get(
            "CI_count_from_rank2_Wilson_claimed"
        )
        is False,
        "rank2_80_mesh_complete": isinstance(point_records, list)
        and len(point_records) == NPOINTS,
        "rank2_80_points_pass": bool(
            point_records
            and all(item.get("all_checks_pass") is True for item in point_records)
        ),
        "rank2_80_edges_pass": mesh80.get("all_projector_edges_pass") is True,
        "rank2_80_invariance_pass": bool(
            mesh80.get("gauge_pass") is True
            and mesh80.get("orientation_pass") is True
            and mesh80.get("cyclic_start_pass") is True
        ),
        "rank2_mesh_convergence_pass": bool(
            float(convergence.get("eigenphase_set_difference_rad", float("inf")))
            <= thresholds.maximum_40_80_eigenphase_difference_rad
            and float(convergence.get("trace_difference", float("inf")))
            <= thresholds.maximum_40_80_trace_difference
            and float(convergence.get("determinant_difference", float("inf")))
            <= thresholds.maximum_40_80_determinant_difference
        ),
        "five_independent_SA3_references_pass": bool(
            isinstance(sa3_records, list)
            and len(sa3_records) == len(SPOT_INDICES)
            and all(item.get("pass") is True for item in sa3_records)
        ),
        "scalar_topology_decision_absent": not antecedent.topology_decision_path(
            loop_root
        ).exists(),
    }

    point_integrity: List[Dict[str, Any]] = []
    try:
        for index, item in enumerate(point_records):
            path = Path(str(item.get("solution_file", ""))).resolve()
            if (
                int(item.get("index", -1)) != index
                or not path.exists()
                or not path.is_relative_to(loop_root)
                or sha256_file(path) != item.get("solution_sha256")
            ):
                raise RuntimeError(f"rank-2 reference point changed: {index}")
            point_integrity.append(
                {
                    "index": index,
                    "angle_degrees": float(item["angle_degrees"]),
                    "geometry_hash": item["geometry_hash"],
                    "path": str(path),
                    "sha256": item["solution_sha256"],
                }
            )
        checks["reference_point_artifacts_intact"] = True
    except Exception:
        checks["reference_point_artifacts_intact"] = False

    sa3_integrity: List[Dict[str, Any]] = []
    try:
        for reference_index, history_index, item in zip(
            REFERENCE_SPOT_INDICES, SPOT_INDICES, sa3_records
        ):
            path = Path(str(item.get("SA3_file", ""))).resolve()
            expected_angle = 360.0 * history_index / NPOINTS
            if (
                int(item.get("index", -1)) != reference_index
                or not math.isclose(
                    float(item.get("angle_degrees", float("nan"))),
                    expected_angle,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or item.get("geometry_hash")
                != point_records[history_index].get("geometry_hash")
                or not path.exists()
                or not path.is_relative_to(rank2_root)
                or sha256_file(path) != item.get("SA3_sha256")
            ):
                raise RuntimeError(
                    f"rank-2 reference SA3 changed at history index "
                    f"{history_index}"
                )
            sa3_integrity.append(
                {
                    "index": history_index,
                    "reference_index": reference_index,
                    "angle_degrees": float(item["angle_degrees"]),
                    "geometry_hash": item["geometry_hash"],
                    "path": str(path),
                    "sha256": item["SA3_sha256"],
                }
            )
        checks["reference_SA3_artifacts_intact"] = True
    except Exception:
        checks["reference_SA3_artifacts_intact"] = False

    status = (
        "PASS_OPPOSED_HISTORY_AUTHORIZATION"
        if all(checks.values())
        else "STOP_OPPOSED_HISTORY_AUTHORIZATION_FAILURE"
    )
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "loop_root": str(loop_root),
        "screen_root": str(screen_root),
        "rank2_root": str(rank2_root),
        "status": status,
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "rank2_authorization_file": str(rank2.authorization_path(rank2_root)),
        "rank2_authorization_sha256": sha256_file(
            rank2.authorization_path(rank2_root)
        ),
        "rank2_decision_file": str(decision_file.resolve()),
        "rank2_decision_sha256": sha256_file(decision_file),
        "reference_point_integrity": point_integrity,
        "reference_SA3_integrity": sa3_integrity,
        "reference_Wilson_invariants": mesh80.get("wilson_invariants"),
        "reference_mesh_convergence": convergence,
    }


def validate_authorization(
    root: Path,
    source_root: Path,
    loop_root: Path,
    screen_root: Path,
    rank2_root: Path,
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    loop_root = loop_root.resolve()
    screen_root = screen_root.resolve()
    rank2_root = rank2_root.resolve()
    path = authorization_path(root)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run qualify first")
    record = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "loop_root": str(loop_root),
        "screen_root": str(screen_root),
        "rank2_root": str(rank2_root),
        "status": "PASS_OPPOSED_HISTORY_AUTHORIZATION",
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "manuscript_claim_authorized": False,
    }
    if {key: record.get(key) for key in expected} != expected:
        raise RuntimeError("opposed-history authorization immutable fields changed")
    checks = record.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("opposed-history authorization checks are incomplete")
    point_integrity = record.get("reference_point_integrity")
    if (
        not isinstance(point_integrity, list)
        or len(point_integrity) != NPOINTS
        or [int(item.get("index", -1)) for item in point_integrity]
        != list(range(NPOINTS))
    ):
        raise RuntimeError("opposed-history authorization lacks the 80 references")
    sa3_integrity = record.get("reference_SA3_integrity")
    if (
        not isinstance(sa3_integrity, list)
        or len(sa3_integrity) != len(SPOT_INDICES)
        or [int(item.get("index", -1)) for item in sa3_integrity]
        != list(SPOT_INDICES)
        or [int(item.get("reference_index", -1)) for item in sa3_integrity]
        != list(REFERENCE_SPOT_INDICES)
    ):
        raise RuntimeError("opposed-history authorization lacks the five SA3 references")
    frozen_wilson = record.get("reference_Wilson_invariants")
    if not isinstance(frozen_wilson, Mapping) or not {
        "trace",
        "determinant",
        "eigenphases_rad",
    }.issubset(frozen_wilson):
        raise RuntimeError("opposed-history authorization lacks Wilson invariants")
    rank2.validate_authorization(rank2_root, source_root, loop_root, screen_root)
    rank2_decision = rank2.validate_existing_decision(rank2_root)
    if rank2_decision.get("status") != "PASS_RANK2_PROJECTOR_QUALIFICATION":
        raise RuntimeError("rank-2 antecedent no longer passes")
    artifacts = [
        (
            record["rank2_authorization_file"],
            record["rank2_authorization_sha256"],
            rank2_root,
        ),
        (record["rank2_decision_file"], record["rank2_decision_sha256"], rank2_root),
    ]
    for item in point_integrity:
        artifacts.append((item["path"], item["sha256"], loop_root))
    for item in sa3_integrity:
        artifacts.append((item["path"], item["sha256"], rank2_root))
    for raw_path, expected_hash, owner in artifacts:
        artifact = Path(str(raw_path)).resolve()
        if (
            not artifact.exists()
            or not artifact.is_relative_to(owner)
            or sha256_file(artifact) != expected_hash
        ):
            raise RuntimeError(f"authorized artifact changed: {artifact}")
    if antecedent.topology_decision_path(loop_root).exists():
        raise RuntimeError("forbidden scalar topology decision appeared")
    return record


def run_qualify(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = authorization_path(root)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("status") == "PASS_OPPOSED_HISTORY_AUTHORIZATION":
            result = validate_authorization(
                root,
                args.source_root,
                args.loop_root,
                args.screen_root,
                args.rank2_root,
            )
        elif existing.get("status") == "STOP_OPPOSED_HISTORY_AUTHORIZATION_FAILURE":
            result = existing
        else:
            raise RuntimeError("existing authorization has an unknown status")
    else:
        try:
            result = build_authorization(
                root,
                args.source_root,
                args.loop_root,
                args.screen_root,
                args.rank2_root,
            )
        except (Exception, SystemExit) as exc:
            result = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "root": str(root),
                "source_root": str(args.source_root.resolve()),
                "loop_root": str(args.loop_root.resolve()),
                "screen_root": str(args.screen_root.resolve()),
                "rank2_root": str(args.rank2_root.resolve()),
                "status": "STOP_OPPOSED_HISTORY_AUTHORIZATION_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}",
                "scalar_phase_evaluated": False,
                "signed_single_state_overlap_recorded": False,
                "individual_root_assignment_used": False,
                "manuscript_claim_authorized": False,
            }
        write_json(path, result)
    print("\nCPDC RANK-2 OPPOSED-HISTORY AUTHORIZATION", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Decision file: {path}", flush=True)
    print(f"CPDC_OPPOSED_HISTORY_SOURCE={result['status']}", flush=True)
    return result


def build_manifest(root: Path) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyscf_version": __import__("pyscf").__version__,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "component_hashes": {
            Path(__file__).name: sha256_file(Path(__file__).resolve()),
            PROTOCOL_FILENAME: sha256_file(protocol_path()),
            Path(rank2.__file__).name: sha256_file(Path(rank2.__file__)),
            Path(antecedent.__file__).name: sha256_file(Path(antecedent.__file__)),
            Path(eng.__file__).name: sha256_file(Path(eng.__file__)),
            Path(ov.__file__).name: sha256_file(Path(ov.__file__)),
        },
        "scalar_phase_evaluated": False,
        "individual_root_assignment_used": False,
    }


def create_or_validate_manifest(root: Path) -> Dict[str, Any]:
    path = manifest_path(root)
    expected_dependencies = {
        Path(rank2.__file__).name: EXPECTED_RANK2_DRIVER_SHA256,
        Path(antecedent.__file__).name: EXPECTED_ANTECEDENT_DRIVER_SHA256,
        Path(eng.__file__).name: EXPECTED_ENGINE_SHA256,
        Path(ov.__file__).name: EXPECTED_OVERLAP_SHA256,
    }
    actual_dependencies = {
        key: sha256_file(Path(module_path))
        for key, module_path in (
            (Path(rank2.__file__).name, rank2.__file__),
            (Path(antecedent.__file__).name, antecedent.__file__),
            (Path(eng.__file__).name, eng.__file__),
            (Path(ov.__file__).name, ov.__file__),
        )
    }
    if actual_dependencies != expected_dependencies:
        raise RuntimeError("frozen dependency hash mismatch")
    if sha256_file(Path(__file__).resolve().with_name(rank2.PROTOCOL_FILENAME)) != EXPECTED_RANK2_PROTOCOL_SHA256:
        raise RuntimeError("frozen rank-2 protocol hash mismatch")
    if path.exists():
        record = json.loads(path.read_text())
        expected_hashes = {
            Path(__file__).name: sha256_file(Path(__file__).resolve()),
            PROTOCOL_FILENAME: sha256_file(protocol_path()),
            **expected_dependencies,
        }
        if (
            record.get("script_version") != SCRIPT_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("pyscf_version") != PYSCF_VERSION_EXACT
            or record.get("authorization_sha256")
            != sha256_file(authorization_path(root))
            or record.get("component_hashes") != expected_hashes
        ):
            raise RuntimeError("existing manifest failed integrity validation")
        return record
    record = build_manifest(root)
    write_json(path, record)
    return record


def reference_points(args: argparse.Namespace) -> Tuple[Mapping[str, Any], List[Any]]:
    surface = antecedent.load_surface(
        antecedent.surface_npz_path(args.loop_root.resolve())
    )
    loop = antecedent.build_loop_specs()[0]
    points = antecedent.loop_points(
        args.loop_root.resolve(), surface, loop, 2, NPOINTS
    )
    return surface, points


def history_order(direction: str) -> List[int]:
    if direction == "clockwise":
        return list(range(1, NPOINTS))
    if direction == "counterclockwise":
        return list(range(NPOINTS - 1, 0, -1))
    raise ValueError(f"unknown direction: {direction}")


def history_target(root: Path, direction: str, index: int, geometry_hash: str) -> Path:
    return (
        root
        / "histories"
        / direction
        / "points"
        / f"p{index:03d}_{geometry_hash}"
        / "solution.npz"
    )


def closure_target(root: Path, direction: str) -> Path:
    return root / "histories" / direction / "closure_p000" / "solution.npz"


def validate_history_solution(
    target: Path,
    geometry: np.ndarray,
    direction: str,
    index: int,
    seed_path: Path,
    closure: bool,
) -> Dict[str, Any]:
    metadata = eng.validate_existing_solution(
        target,
        geometry,
        BASIS_KEY,
        HISTORY_SA2_ROLE,
        gradients_required=False,
    )
    descriptor = metadata.get("point_descriptor", {})
    expected = {
        "kind": "rank2_opposed_history_loop_closure"
        if closure
        else "rank2_opposed_history_loop_vertex",
        "opposed_history_protocol_id": PROTOCOL_ID,
        "history_direction": direction,
        "point_index": int(index),
        "closure": bool(closure),
        "geometry_hash": eng.array_hash(geometry),
        "seed_solution_file": str(seed_path.resolve()),
        "seed_solution_sha256": sha256_file(seed_path),
    }
    if {key: descriptor.get(key) for key in expected} != expected:
        raise RuntimeError(f"history descriptor mismatch for {target}")
    return metadata


def run_history_solution(
    target: Path,
    geometry: np.ndarray,
    direction: str,
    index: int,
    seed_path: Path,
    closure: bool,
    memory_mb: int,
) -> Dict[str, Any]:
    if target.exists():
        return validate_history_solution(
            target, geometry, direction, index, seed_path, closure
        )
    descriptor = {
        "kind": "rank2_opposed_history_loop_closure"
        if closure
        else "rank2_opposed_history_loop_vertex",
        "opposed_history_protocol_id": PROTOCOL_ID,
        "history_direction": direction,
        "point_index": int(index),
        "closure": bool(closure),
        "geometry_hash": eng.array_hash(geometry),
        "seed_solution_file": str(seed_path.resolve()),
        "seed_solution_sha256": sha256_file(seed_path),
    }
    eng.run_sa_solution(
        target=target,
        geometry=geometry,
        basis=BASIS_KEY,
        memory_mb=memory_mb,
        role=HISTORY_SA2_ROLE,
        center_seed=seed_path,
        point_descriptor=descriptor,
        compute_state_gradients=False,
    )
    return validate_history_solution(
        target, geometry, direction, index, seed_path, closure
    )


def history_worker(args: argparse.Namespace) -> None:
    configure_threads(args.threads)
    require_pyscf()
    surface, points = reference_points(args)
    direction = args.direction
    previous = points[0].target.resolve()
    for step, index in enumerate(history_order(direction), start=1):
        point = points[index]
        geometry = antecedent.surface_geometry(surface, point.ux, point.uy)
        target = history_target(
            args.root.resolve(), direction, index, point.geometry_hash
        )
        metadata = run_history_solution(
            target,
            geometry,
            direction,
            index,
            previous,
            False,
            args.memory_mb,
        )
        print(
            f"HISTORY {direction} step={step:03d} p={index:03d} "
            f"gap12={float(metadata['adiabatic_gap_Eh']):.9e} Eh "
            f"gap23={float(metadata['third_singlet_gap_Eh']):.9e} Eh "
            f"grad={float(metadata['maximum_orbital_gradient']):.3e}",
            flush=True,
        )
        previous = target.resolve()
    p0 = points[0]
    geometry0 = antecedent.surface_geometry(surface, p0.ux, p0.uy)
    closure = closure_target(args.root.resolve(), direction)
    metadata = run_history_solution(
        closure,
        geometry0,
        direction,
        0,
        previous,
        True,
        args.memory_mb,
    )
    print(
        f"HISTORY {direction} closure p=000 "
        f"gap12={float(metadata['adiabatic_gap_Eh']):.9e} Eh "
        f"gap23={float(metadata['third_singlet_gap_Eh']):.9e} Eh",
        flush=True,
    )


def worker_environment(threads: int) -> Dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(int(threads))
    return environment


def run_subprocess(command: Sequence[str], log_path: Path, threads: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n===== worker launch {time.time():.6f} =====\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=worker_environment(threads),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed with code {completed.returncode}; see {log_path}")


def history_worker_command(args: argparse.Namespace, direction: str) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "history-worker",
        "--root",
        str(args.root.resolve()),
        "--source-root",
        str(args.source_root.resolve()),
        "--loop-root",
        str(args.loop_root.resolve()),
        "--direction",
        direction,
        "--threads",
        str(args.threads),
        "--memory-mb",
        str(args.memory_mb),
    ]


def run_history_workers(args: argparse.Namespace) -> None:
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for direction in DIRECTIONS:
            log_path = args.root.resolve() / "histories" / direction / "worker.stdout"
            future = executor.submit(
                run_subprocess,
                history_worker_command(args, direction),
                log_path,
                args.threads,
            )
            futures[future] = direction
        for future in as_completed(futures):
            direction = futures[future]
            try:
                future.result()
                print(f"DONE history chain: {direction}", flush=True)
            except Exception as exc:
                failures.append(f"{direction}: {type(exc).__name__}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def load_history_sequences(
    args: argparse.Namespace,
) -> Tuple[Mapping[str, Any], List[Any], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    surface, points = reference_points(args)
    sequences: Dict[str, List[Dict[str, Any]]] = {}
    closures: Dict[str, Dict[str, Any]] = {}
    reference0 = eng.load_solution(points[0].target)
    for direction in DIRECTIONS:
        values: List[Dict[str, Any]] = [reference0]
        order = history_order(direction)
        previous = points[0].target.resolve()
        by_index: Dict[int, Dict[str, Any]] = {}
        for index in order:
            point = points[index]
            geometry = antecedent.surface_geometry(surface, point.ux, point.uy)
            target = history_target(
                args.root.resolve(), direction, index, point.geometry_hash
            )
            validate_history_solution(
                target, geometry, direction, index, previous, False
            )
            by_index[index] = eng.load_solution(target)
            previous = target.resolve()
        values.extend(by_index[index] for index in range(1, NPOINTS))
        sequences[direction] = values
        p0 = points[0]
        geometry0 = antecedent.surface_geometry(surface, p0.ux, p0.uy)
        closure = closure_target(args.root.resolve(), direction)
        validate_history_solution(
            closure, geometry0, direction, 0, previous, True
        )
        closures[direction] = eng.load_solution(closure)
    return surface, points, sequences, closures


def projector_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    mol: Any,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    matrix = ov.exact_overlap_matrix(left, right, mol, mol)
    if matrix.shape != (2, 2):
        raise RuntimeError(f"unexpected SA2 comparison shape: {matrix.shape}")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    leakage = float(1.0 - np.mean(singular_values**2))
    active = antecedent.active_edge_metrics(left, right, mol, mol)
    energy_difference = abs(
        float(left["metadata"]["average_energy_Eh"])
        - float(right["metadata"]["average_energy_Eh"])
    )
    return {
        "pair_singular_values": singular_values,
        "pair_sigma_min": sigma_min,
        "pair_leakage": leakage,
        "active_orbital_subspace": active,
        "average_energy_difference_Eh": energy_difference,
        "projector_pass": bool(
            sigma_min >= thresholds.history_pair_sigma_minimum
            and leakage <= thresholds.history_pair_leakage_maximum
            and active["sigma_min"] >= thresholds.history_active_sigma_minimum
            and active["leakage"] <= thresholds.history_active_leakage_maximum
        ),
        "energy_pass": bool(
            energy_difference
            <= thresholds.maximum_history_average_energy_difference_Eh
        ),
        "individual_root_assignment_used": False,
    }


def sequence_wilson(
    sequence: Sequence[Mapping[str, Any]],
    molecules: Sequence[Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    overlaps: List[np.ndarray] = []
    edges: List[Dict[str, Any]] = []
    npoints = len(sequence)
    for index in range(npoints):
        neighbor = (index + 1) % npoints
        matrix = ov.exact_overlap_matrix(
            sequence[index],
            sequence[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        sigma_min = float(np.min(singular_values))
        leakage = float(1.0 - np.mean(singular_values**2))
        active = antecedent.active_edge_metrics(
            sequence[index],
            sequence[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        strict = bool(
            sigma_min >= thresholds.history_pair_sigma_minimum
            and leakage <= thresholds.history_pair_leakage_maximum
            and active["sigma_min"] >= thresholds.history_active_sigma_minimum
            and active["leakage"] <= thresholds.history_active_leakage_maximum
        )
        overlaps.append(matrix)
        edges.append(
            {
                "from_index": index,
                "to_index": neighbor,
                "pair_singular_values": singular_values,
                "pair_sigma_min": sigma_min,
                "pair_leakage": leakage,
                "active_orbital_subspace": active,
                "strict": strict,
                "signed_single_state_overlap_recorded": False,
            }
        )
    invariants = rank2.wilson_invariants(rank2.wilson_product(overlaps))
    gauge = rank2.gauge_audit(overlaps, invariants)
    cyclic = rank2.cyclic_start_audit(overlaps, invariants)
    return {
        "edge_records": edges,
        "all_edges_pass": all(item["strict"] for item in edges),
        "minimum_pair_sigma": min(item["pair_sigma_min"] for item in edges),
        "maximum_pair_leakage": max(item["pair_leakage"] for item in edges),
        "minimum_active_sigma": min(
            item["active_orbital_subspace"]["sigma_min"] for item in edges
        ),
        "maximum_active_leakage": max(
            item["active_orbital_subspace"]["leakage"] for item in edges
        ),
        "wilson_invariants": invariants,
        "random_U2_gauge_audit": gauge,
        "cyclic_start_audit": cyclic,
        "gauge_pass": gauge["maximum_error"]
        <= thresholds.maximum_history_gauge_error,
        "cyclic_start_pass": cyclic["maximum_error"]
        <= thresholds.maximum_history_cyclic_start_error,
    }


def wilson_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    phase = rank2.phase_set_error(
        left["eigenphases_rad"], right["eigenphases_rad"]
    )
    trace = float(abs(complex(left["trace"]) - complex(right["trace"])))
    determinant = float(
        abs(complex(left["determinant"]) - complex(right["determinant"]))
    )
    return {
        "eigenphase_set_difference_rad": phase,
        "trace_difference": trace,
        "determinant_difference": determinant,
        "pass": bool(
            phase
            <= thresholds.maximum_history_Wilson_eigenphase_difference_rad
            and trace <= thresholds.maximum_history_Wilson_trace_difference
            and determinant
            <= thresholds.maximum_history_Wilson_determinant_difference
        ),
    }


def analyze_history(args: argparse.Namespace, authorization: Mapping[str, Any]) -> Dict[str, Any]:
    thresholds = Thresholds()
    surface, points, histories, closures = load_history_sequences(args)
    reference = [eng.load_solution(point.target) for point in points]
    molecules = [
        eng.make_mol(
            antecedent.surface_geometry(surface, point.ux, point.uy),
            BASIS_KEY,
            args.memory_mb,
            verbose=0,
        )
        for point in points
    ]
    source = antecedent.validate_source_authorization(
        args.loop_root.resolve(), args.source_root.resolve(), require_pass=True
    )
    center_geometry = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    center_energy = float(source["center_average_energy_Eh"])
    base_thresholds = antecedent.Thresholds()

    point_records: List[Dict[str, Any]] = []
    for index in range(1, NPOINTS):
        geometry = antecedent.surface_geometry(
            surface, points[index].ux, points[index].uy
        )
        direction_checks = {}
        for direction in DIRECTIONS:
            checks, metrics = antecedent.production_point_checks(
                histories[direction][index],
                geometry,
                center_geometry,
                center_energy,
                base_thresholds,
            )
            direction_checks[direction] = {
                "checks": checks,
                "metrics": metrics,
                "all_checks_pass": all(checks.values()),
                "solution_file": histories[direction][index]["path"],
                "solution_sha256": sha256_file(
                    Path(histories[direction][index]["path"])
                ),
            }
        comparisons = {
            "clockwise_vs_reference": projector_comparison(
                histories["clockwise"][index], reference[index], molecules[index], thresholds
            ),
            "counterclockwise_vs_reference": projector_comparison(
                histories["counterclockwise"][index], reference[index], molecules[index], thresholds
            ),
            "clockwise_vs_counterclockwise": projector_comparison(
                histories["clockwise"][index],
                histories["counterclockwise"][index],
                molecules[index],
                thresholds,
            ),
        }
        point_records.append(
            {
                "index": index,
                "angle_degrees": 360.0 * index / NPOINTS,
                "geometry_hash": points[index].geometry_hash,
                "direction_records": direction_checks,
                "comparisons": comparisons,
                "point_quality_pass": all(
                    item["all_checks_pass"] for item in direction_checks.values()
                ),
                "projector_pass": all(
                    item["projector_pass"] for item in comparisons.values()
                ),
                "energy_pass": all(item["energy_pass"] for item in comparisons.values()),
            }
        )

    closure_checks: Dict[str, Any] = {}
    for direction in DIRECTIONS:
        checks, metrics = antecedent.production_point_checks(
            closures[direction],
            np.asarray(reference[0]["geometry_angstrom"]),
            center_geometry,
            center_energy,
            base_thresholds,
        )
        closure_checks[direction] = {
            "checks": checks,
            "metrics": metrics,
            "all_checks_pass": all(checks.values()),
            "solution_file": closures[direction]["path"],
            "solution_sha256": sha256_file(Path(closures[direction]["path"])),
            "vs_reference": projector_comparison(
                closures[direction], reference[0], molecules[0], thresholds
            ),
        }
    closure_checks["clockwise_vs_counterclockwise"] = projector_comparison(
        closures["clockwise"],
        closures["counterclockwise"],
        molecules[0],
        thresholds,
    )

    reference_wilson = sequence_wilson(reference, molecules, thresholds)
    # Include each freshly optimized returned p000 solution in the product.
    # The counterclockwise optimization history is reversed below so that both
    # Wilson spectra use the same geometric orientation.  The final
    # duplicate-geometry edge closes the returned and immutable p000 frames
    # and preserves covariance to independent endpoint U(2) gauges.
    history_wilson_sequences = {
        "clockwise": [*histories["clockwise"], closures["clockwise"]],
        "counterclockwise": [
            closures["counterclockwise"],
            *histories["counterclockwise"][1:],
            reference[0],
        ],
    }
    history_wilson_molecules = {
        "clockwise": [*molecules, molecules[0]],
        "counterclockwise": [molecules[0], *molecules[1:], molecules[0]],
    }
    history_wilson = {
        direction: sequence_wilson(
            history_wilson_sequences[direction],
            history_wilson_molecules[direction],
            thresholds,
        )
        for direction in DIRECTIONS
    }
    stored_reference = authorization["reference_Wilson_invariants"]
    decoded_stored = {
        "trace": complex(
            float(stored_reference["trace"]["real"]),
            float(stored_reference["trace"]["imag"]),
        ),
        "determinant": complex(
            float(stored_reference["determinant"]["real"]),
            float(stored_reference["determinant"]["imag"]),
        ),
        "eigenphases_rad": np.asarray(stored_reference["eigenphases_rad"], dtype=float),
    }
    stored_reproduction = wilson_comparison(
        reference_wilson["wilson_invariants"], decoded_stored, thresholds
    )
    wilson_comparisons = {
        "recomputed_reference_vs_frozen_decision": stored_reproduction,
        "clockwise_vs_reference": wilson_comparison(
            history_wilson["clockwise"]["wilson_invariants"],
            reference_wilson["wilson_invariants"],
            thresholds,
        ),
        "counterclockwise_vs_reference": wilson_comparison(
            history_wilson["counterclockwise"]["wilson_invariants"],
            reference_wilson["wilson_invariants"],
            thresholds,
        ),
        "clockwise_vs_counterclockwise": wilson_comparison(
            history_wilson["clockwise"]["wilson_invariants"],
            history_wilson["counterclockwise"]["wilson_invariants"],
            thresholds,
        ),
    }

    point_quality_pass = all(item["point_quality_pass"] for item in point_records)
    projector_pass = all(item["projector_pass"] for item in point_records)
    energy_pass = all(item["energy_pass"] for item in point_records)
    closure_pass = bool(
        all(
            closure_checks[direction]["all_checks_pass"]
            and closure_checks[direction]["vs_reference"]["projector_pass"]
            and closure_checks[direction]["vs_reference"]["energy_pass"]
            for direction in DIRECTIONS
        )
        and closure_checks["clockwise_vs_counterclockwise"]["projector_pass"]
        and closure_checks["clockwise_vs_counterclockwise"]["energy_pass"]
    )
    edge_pass = all(
        item["all_edges_pass"] and item["gauge_pass"] and item["cyclic_start_pass"]
        for item in [reference_wilson, *history_wilson.values()]
    )
    wilson_pass = all(item["pass"] for item in wilson_comparisons.values())
    if not point_quality_pass:
        status = "STOP_HISTORY_WAVEFUNCTION_FAILURE"
        reason = "At least one opposed-history SA2 solution fails the frozen point checks."
    elif not projector_pass or not energy_pass:
        status = "STOP_HISTORY_PROJECTOR_REPRODUCTION_FAILURE"
        reason = "At least one vertex history fails pair/active projector or average-energy reproduction."
    elif not closure_pass:
        status = "STOP_HISTORY_CLOSURE_FAILURE"
        reason = "At least one returned p000 history fails projector, energy, or point closure."
    elif not edge_pass:
        status = "STOP_HISTORY_EDGE_INVARIANCE_FAILURE"
        reason = "A history edge, U(2) gauge audit, or cyclic-start audit fails."
    elif not wilson_pass:
        status = "STOP_HISTORY_WILSON_REPRODUCTION_FAILURE"
        reason = "Clockwise/counterclockwise Wilson invariants do not reproduce the frozen reference."
    else:
        status = "PASS_RANK2_OPPOSED_HISTORY"
        reason = "Clockwise and counterclockwise SA2 histories reproduce the same closed rank-2 manifold and Wilson spectrum."
    return {
        "status": status,
        "reason": reason,
        "thresholds": asdict(thresholds),
        "point_records": point_records,
        "closure_checks": closure_checks,
        "reference_Wilson": reference_wilson,
        "history_Wilson": history_wilson,
        "Wilson_comparisons": wilson_comparisons,
        "point_quality_pass": point_quality_pass,
        "projector_reproduction_pass": projector_pass,
        "energy_reproduction_pass": energy_pass,
        "closure_pass": closure_pass,
        "edge_invariance_pass": edge_pass,
        "Wilson_reproduction_pass": wilson_pass,
        "scalar_phase_evaluated": False,
        "individual_root_assignment_used": False,
    }


def validate_history_decision(root: Path) -> Dict[str, Any]:
    path = history_decision_path(root)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run history first")
    result = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "scalar_phase_evaluated": False,
        "individual_root_assignment_used": False,
        "manuscript_claim_authorized": False,
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError("history decision immutable fields changed")
    for item in result.get("point_records", []):
        for direction in DIRECTIONS:
            record = item["direction_records"][direction]
            path = Path(record["solution_file"]).resolve()
            if not path.exists() or sha256_file(path) != record["solution_sha256"]:
                raise RuntimeError(f"history point changed: {path}")
    for direction in DIRECTIONS:
        record = result["closure_checks"][direction]
        path = Path(record["solution_file"]).resolve()
        if not path.exists() or sha256_file(path) != record["solution_sha256"]:
            raise RuntimeError(f"history closure changed: {path}")
    return result


def run_history(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol()
    root = args.root.resolve()
    authorization = validate_authorization(
        root,
        args.source_root,
        args.loop_root,
        args.screen_root,
        args.rank2_root,
    )
    require_pyscf()
    create_or_validate_manifest(root)
    output = history_decision_path(root)
    if output.exists():
        result = validate_history_decision(root)
        print(f"CPDC_RANK2_HISTORY={result['status']}", flush=True)
        return result
    run_history_workers(args)
    analysis = analyze_history(args, authorization)
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "manuscript_claim_authorized": False,
        **analysis,
        "authorized_next_step": (
            "Run the separately gated history-seeded SA3 confirmation."
            if analysis["status"] == "PASS_RANK2_OPPOSED_HISTORY"
            else "Stop; do not run the SA3 stage or expose scalar phase."
        ),
    }
    write_json(output, result)
    print("\nCPDC RANK-2 OPPOSED-HISTORY SA2", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Reason: {result['reason']}", flush=True)
    comparisons = result["Wilson_comparisons"]
    for label in (
        "clockwise_vs_reference",
        "counterclockwise_vs_reference",
        "clockwise_vs_counterclockwise",
    ):
        item = comparisons[label]
        print(
            f"{label}: dphase={item['eigenphase_set_difference_rad']:.3e} "
            f"dtrace={item['trace_difference']:.3e} "
            f"ddet={item['determinant_difference']:.3e}",
            flush=True,
        )
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_RANK2_HISTORY={result['status']}", flush=True)
    return result


def history_sa3_target(root: Path, direction: str, index: int) -> Path:
    angle = 360.0 * index / NPOINTS
    return (
        root
        / "sa3_histories"
        / direction
        / f"theta_{angle:05.1f}_deg"
        / "solution.npz"
    )


def build_history_sa3_specs(
    args: argparse.Namespace,
    authorization: Mapping[str, Any],
) -> List[HistorySA3Spec]:
    surface, points = reference_points(args)
    references = {
        int(item["index"]): item for item in authorization["reference_SA3_integrity"]
    }
    specs: List[HistorySA3Spec] = []
    for direction in DIRECTIONS:
        for index in SPOT_INDICES:
            if index == 0:
                seed = closure_target(args.root.resolve(), direction)
            else:
                seed = history_target(
                    args.root.resolve(), direction, index, points[index].geometry_hash
                )
            specs.append(
                HistorySA3Spec(
                    direction=direction,
                    index=index,
                    angle_degrees=360.0 * index / NPOINTS,
                    geometry_hash=points[index].geometry_hash,
                    seed_sa2_path=seed.resolve(),
                    reference_sa3_path=Path(references[index]["path"]).resolve(),
                    target_sa3_path=history_sa3_target(
                        args.root.resolve(), direction, index
                    ),
                )
            )
    return specs


def validate_history_sa3(target: Path, spec: HistorySA3Spec) -> Dict[str, Any]:
    solution = eng.load_solution(target)
    metadata = solution["metadata"]
    weights = np.asarray(metadata.get("state_average_weights", []), dtype=float)
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": HISTORY_SA3_ROLE,
        "basis": BASIS_KEY,
        "pyscf_version": PYSCF_VERSION_EXACT,
        "basis_definition_sha256": eng.BASIS_DEFINITION_SHA256,
        "charge": 1,
        "spin_2S": 0,
        "ncas": antecedent.NCAS,
        "nelecas": antecedent.NELECAS,
        "nact_alpha": antecedent.NACT_ALPHA,
        "nact_beta": antecedent.NACT_BETA,
        "nroots": 3,
        "frozen_orbitals": 0,
        "absolute_ground_state_claimed": False,
        "history_direction": spec.direction,
        "spot_index": spec.index,
        "geometry_hash": spec.geometry_hash,
        "seed_SA2_file": str(spec.seed_sa2_path),
        "seed_SA2_sha256": sha256_file(spec.seed_sa2_path),
        "reference_SA3_file": str(spec.reference_sa3_path),
        "reference_SA3_sha256": sha256_file(spec.reference_sa3_path),
    }
    actual = {key: metadata.get(key) for key in expected}
    actual["basis"] = str(actual.get("basis", "")).lower()
    if actual != expected:
        raise RuntimeError(f"history SA3 metadata mismatch: {actual} != {expected}")
    if weights.shape != (3,) or not np.allclose(
        weights, [1.0 / 3.0] * 3, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("history SA3 state-average weights changed")
    seed = eng.load_solution(spec.seed_sa2_path)
    if np.max(np.abs(solution["geometry_angstrom"] - seed["geometry_angstrom"])) > 1.0e-11:
        raise RuntimeError("history SA3 geometry changed")
    return metadata


def run_history_sa3_solution(
    target: Path,
    spec: HistorySA3Spec,
    memory_mb: int,
) -> Dict[str, Any]:
    if target.exists():
        return validate_history_sa3(target, spec)
    fci, _, mcscf, _ = eng.require_pyscf()
    seed = eng.load_solution(spec.seed_sa2_path)
    geometry = np.asarray(seed["geometry_angstrom"], dtype=float)
    if eng.array_hash(geometry) != spec.geometry_hash:
        raise RuntimeError("history SA3 seed geometry hash changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    mol = eng.make_mol(
        geometry, BASIS_KEY, memory_mb, str(target.parent / "casscf.log"), verbose=4
    )
    mf = eng.run_rhf(mol)
    metric = mol.intor_symmetric("int1e_ovlp")
    mo_guess = eng.metric_orthonormalize(np.asarray(seed["mo_coeff"]), metric)
    ncore_expected = (mol.nelectron - antecedent.NELECAS) // 2
    reference_active = mo_guess[
        :, ncore_expected : ncore_expected + antecedent.NCAS
    ].copy()
    mc = mcscf.CASSCF(
        mf, antecedent.NCAS, (antecedent.NACT_ALPHA, antecedent.NACT_BETA)
    )
    mc.fcisolver = eng.make_singlet_solver(mol, 3)
    mc = mc.state_average_([1.0 / 3.0] * 3)
    mc.conv_tol = antecedent.CASSCF_CONV_TOL
    mc.conv_tol_grad = antecedent.CASSCF_GRAD_TOL
    mc.max_cycle_macro = 160
    mc.max_cycle_micro = 40
    mc.canonicalization = False
    mc.kernel(mo_guess)
    primary_converged = bool(mc.converged)
    fallback_attempts = 0
    while not bool(mc.converged) and fallback_attempts < 2:
        mc = mc.newton()
        mc.conv_tol = antecedent.CASSCF_CONV_TOL
        mc.conv_tol_grad = antecedent.CASSCF_GRAD_TOL
        mc.max_cycle_macro = 160
        mc.max_cycle_micro = 40
        mc.canonicalization = False
        mc.kernel(mc.mo_coeff)
        fallback_attempts += 1
    energies = np.asarray(mc.e_states, dtype=float).reshape(-1)
    ci_roots = [np.asarray(item) for item in mc.ci]
    if energies.size != 3 or len(ci_roots) != 3:
        raise RuntimeError("history SA3 returned an unexpected root count")
    try:
        orbital_gradient = np.asarray(mc.get_grad(mc.mo_coeff))
        maximum_orbital_gradient = (
            float(np.max(np.abs(orbital_gradient)))
            if orbital_gradient.size
            else 0.0
        )
    except Exception:
        maximum_orbital_gradient = float("nan")
    final_mo = np.asarray(mc.mo_coeff)
    ncore = int(mc.ncore)
    if ncore != ncore_expected:
        raise RuntimeError(f"unexpected history SA3 ncore={ncore}")
    probe_energies, _ = eng.casci_probe(mf, final_mo, 3)
    reproduction_error = float(np.max(np.abs(probe_energies - energies)))
    spin_square = [
        float(
            fci.spin_op.spin_square0(
                item,
                antecedent.NCAS,
                (antecedent.NACT_ALPHA, antecedent.NACT_BETA),
            )[0]
        )
        for item in ci_roots
    ]
    mo_error = float(
        np.max(
            np.abs(
                final_mo.conj().T @ metric @ final_mo
                - np.eye(final_mo.shape[1])
            )
        )
    )
    final_active = final_mo[:, ncore : ncore + antecedent.NCAS]
    active_sv = np.linalg.svd(
        reference_active.conj().T @ metric @ final_active, compute_uv=False
    )
    metadata: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": HISTORY_SA3_ROLE,
        "molecule": "cyclopentadienyl cation",
        "formula": "C5H5+",
        "target": "rank-2 lowest-singlet manifold only",
        "absolute_ground_state_claimed": False,
        "basis": BASIS_KEY,
        "basis_definition_sha256": eng.BASIS_DEFINITION_SHA256,
        "method": "history-seeded equal-weight SA3-CASSCF(4e,5o)",
        "charge": 1,
        "spin_2S": 0,
        "ncas": antecedent.NCAS,
        "nelecas": antecedent.NELECAS,
        "nact_alpha": antecedent.NACT_ALPHA,
        "nact_beta": antecedent.NACT_BETA,
        "nroots": 3,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "state_average_weights": [1.0 / 3.0] * 3,
        "history_direction": spec.direction,
        "spot_index": spec.index,
        "spot_angle_degrees": spec.angle_degrees,
        "geometry_hash": spec.geometry_hash,
        "seed_SA2_file": str(spec.seed_sa2_path),
        "seed_SA2_sha256": sha256_file(spec.seed_sa2_path),
        "reference_SA3_file": str(spec.reference_sa3_path),
        "reference_SA3_sha256": sha256_file(spec.reference_sa3_path),
        "converged": bool(mc.converged),
        "primary_converged": primary_converged,
        "fallback_attempts": fallback_attempts,
        "average_energy_Eh": float(np.mean(energies)),
        "state_energies_Eh": energies,
        "internal_pair_gap_Eh": float(energies[1] - energies[0]),
        "third_singlet_gap_Eh": float(energies[2] - energies[1]),
        "maximum_orbital_gradient": maximum_orbital_gradient,
        "state_S2": spin_square,
        "maximum_spin_square_error": max(abs(value) for value in spin_square),
        "mo_orthonormality_max_error": mo_error,
        "CASCI_probe_energies_Eh": probe_energies,
        "CASCI_reproduction_max_error_Eh": reproduction_error,
        "active_reference_singular_values": active_sv,
        "active_reference_sigma_min": float(np.min(active_sv)),
        "pyscf_version": __import__("pyscf").__version__,
        "python_version": platform.python_version(),
        "timestamp_unix": time.time(),
    }
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(jsonable(metadata))),
        "geometry_angstrom": geometry,
        "mo_coeff": final_mo,
        "state_energies_Eh": energies,
        "CASCI_probe_energies_Eh": probe_energies,
    }
    for index, ci in enumerate(ci_roots):
        arrays[f"ci_root_{index}"] = ci
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return validate_history_sa3(target, spec)


def sa3_worker(args: argparse.Namespace) -> None:
    configure_threads(args.threads)
    require_pyscf()
    spec = HistorySA3Spec(
        direction=args.direction,
        index=args.spot_index,
        angle_degrees=args.angle_degrees,
        geometry_hash=args.geometry_hash,
        seed_sa2_path=args.seed_sa2.resolve(),
        reference_sa3_path=args.reference_sa3.resolve(),
        target_sa3_path=args.target.resolve(),
    )
    metadata = run_history_sa3_solution(spec.target_sa3_path, spec, args.memory_mb)
    print(
        f"SA3_HISTORY {spec.direction} theta={spec.angle_degrees:.1f} "
        f"gap12={float(metadata['internal_pair_gap_Eh']):.9e} Eh "
        f"gap23={float(metadata['third_singlet_gap_Eh']):.9e} Eh",
        flush=True,
    )


def sa3_worker_command(args: argparse.Namespace, spec: HistorySA3Spec) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "sa3-worker",
        "--target",
        str(spec.target_sa3_path.resolve()),
        "--seed-sa2",
        str(spec.seed_sa2_path),
        "--reference-sa3",
        str(spec.reference_sa3_path),
        "--direction",
        spec.direction,
        "--spot-index",
        str(spec.index),
        "--angle-degrees",
        f"{spec.angle_degrees:.17g}",
        "--geometry-hash",
        spec.geometry_hash,
        "--threads",
        str(args.threads),
        "--memory-mb",
        str(args.memory_mb),
    ]


def run_sa3_workers(args: argparse.Namespace, specs: Sequence[HistorySA3Spec]) -> None:
    pending = []
    for spec in specs:
        if spec.target_sa3_path.exists():
            validate_history_sa3(spec.target_sa3_path, spec)
        else:
            pending.append(spec)
    if not pending:
        return
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {}
        for spec in pending:
            future = executor.submit(
                run_subprocess,
                sa3_worker_command(args, spec),
                spec.target_sa3_path.parent / "worker.stdout",
                args.threads,
            )
            futures[future] = spec
        for future in as_completed(futures):
            spec = futures[future]
            try:
                future.result()
                validate_history_sa3(spec.target_sa3_path, spec)
                print(
                    f"DONE SA3 history {spec.direction} theta={spec.angle_degrees:.1f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    f"{spec.direction}/{spec.index}: {type(exc).__name__}: {exc}"
                )
                print(f"FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def sa3_quality(metadata: Mapping[str, Any], thresholds: Thresholds) -> Dict[str, bool]:
    return {
        "converged_and_stationary": bool(
            metadata.get("converged")
            and np.isfinite(float(metadata.get("maximum_orbital_gradient", float("nan"))))
            and float(metadata["maximum_orbital_gradient"])
            <= thresholds.maximum_final_orbital_gradient
        ),
        "singlet_spin_pure": float(
            metadata.get("maximum_spin_square_error", float("inf"))
        )
        <= thresholds.maximum_spin_square_error,
        "MO_frame_orthonormal": float(
            metadata.get("mo_orthonormality_max_error", float("inf"))
        )
        <= thresholds.maximum_mo_orthonormality_error,
        "CASCI_reproduces_states": float(
            metadata.get("CASCI_reproduction_max_error_Eh", float("inf"))
        )
        <= thresholds.maximum_casci_reproduction_error_Eh,
        "active_reference_retained": float(
            metadata.get("active_reference_sigma_min", -float("inf"))
        )
        >= thresholds.minimum_active_reference_sigma,
        "third_singlet_isolated": float(
            metadata.get("third_singlet_gap_Eh", -float("inf"))
        )
        >= thresholds.minimum_third_singlet_gap_Eh,
    }


def rectangular_pair_metrics(
    matrix: np.ndarray,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    pair = np.asarray(matrix)[:2, :2]
    singular_values = np.linalg.svd(pair, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    leakage = float(1.0 - np.mean(singular_values**2))
    left_pair_to_right_third = float(np.sum(np.abs(np.asarray(matrix)[:2, 2]) ** 2))
    left_third_to_right_pair = float(np.sum(np.abs(np.asarray(matrix)[2, :2]) ** 2))
    return {
        "pair_singular_values": singular_values,
        "pair_sigma_min": sigma_min,
        "pair_leakage": leakage,
        "left_pair_to_right_third_weight": left_pair_to_right_third,
        "left_third_to_right_pair_weight": left_third_to_right_pair,
        "pass": bool(
            sigma_min >= thresholds.SA3_projector_sigma_minimum
            and leakage <= thresholds.SA3_projector_leakage_maximum
            and left_pair_to_right_third
            <= thresholds.SA3_cross_third_weight_maximum
            and left_third_to_right_pair
            <= thresholds.SA3_cross_third_weight_maximum
        ),
        "individual_root_assignment_used": False,
    }


def analyze_history_sa3(
    args: argparse.Namespace,
    specs: Sequence[HistorySA3Spec],
) -> Dict[str, Any]:
    thresholds = Thresholds()
    records: List[Dict[str, Any]] = []
    loaded: Dict[Tuple[str, int], Tuple[Dict[str, Any], Any]] = {}
    for spec in specs:
        history_sa3 = eng.load_solution(spec.target_sa3_path)
        seed_sa2 = eng.load_solution(spec.seed_sa2_path)
        reference_sa3 = eng.load_solution(spec.reference_sa3_path)
        geometry = np.asarray(history_sa3["geometry_angstrom"], dtype=float)
        mol = eng.make_mol(geometry, BASIS_KEY, args.memory_mb, verbose=0)
        loaded[(spec.direction, spec.index)] = (history_sa3, mol)
        seed_overlap = ov.exact_overlap_matrix(seed_sa2, history_sa3, mol, mol)
        seed_pair = np.asarray(seed_overlap)[:, :2]
        seed_sv = np.linalg.svd(seed_pair, compute_uv=False)
        seed_leak = float(1.0 - np.mean(seed_sv**2))
        seed_third = float(np.sum(np.abs(seed_overlap[:, 2]) ** 2))
        seed_pass = bool(
            float(np.min(seed_sv)) >= thresholds.SA2_SA3_projector_sigma_minimum
            and seed_leak <= thresholds.SA2_SA3_projector_leakage_maximum
            and seed_third <= thresholds.SA3_cross_third_weight_maximum
        )
        reference_overlap = ov.exact_overlap_matrix(
            reference_sa3, history_sa3, mol, mol
        )
        reference_metrics = rectangular_pair_metrics(reference_overlap, thresholds)
        energy_difference = abs(
            float(reference_sa3["metadata"]["average_energy_Eh"])
            - float(history_sa3["metadata"]["average_energy_Eh"])
        )
        quality = sa3_quality(history_sa3["metadata"], thresholds)
        records.append(
            {
                "direction": spec.direction,
                "index": spec.index,
                "angle_degrees": spec.angle_degrees,
                "geometry_hash": spec.geometry_hash,
                "seed_SA2_file": str(spec.seed_sa2_path),
                "seed_SA2_sha256": sha256_file(spec.seed_sa2_path),
                "reference_SA3_file": str(spec.reference_sa3_path),
                "reference_SA3_sha256": sha256_file(spec.reference_sa3_path),
                "history_SA3_file": str(spec.target_sa3_path.resolve()),
                "history_SA3_sha256": sha256_file(spec.target_sa3_path),
                "quality_checks": quality,
                "quality_pass": all(quality.values()),
                "seed_SA2_to_SA3": {
                    "pair_singular_values": seed_sv,
                    "pair_sigma_min": float(np.min(seed_sv)),
                    "pair_leakage": seed_leak,
                    "third_root_SA2_weight": seed_third,
                    "pass": seed_pass,
                },
                "reference_SA3_to_history_SA3": reference_metrics,
                "average_energy_difference_from_reference_Eh": energy_difference,
                "energy_pass": energy_difference
                <= thresholds.maximum_SA3_average_energy_difference_Eh,
            }
        )

    cross_history: List[Dict[str, Any]] = []
    for index in SPOT_INDICES:
        clockwise, mol = loaded[("clockwise", index)]
        counterclockwise, _ = loaded[("counterclockwise", index)]
        overlap = ov.exact_overlap_matrix(clockwise, counterclockwise, mol, mol)
        metrics = rectangular_pair_metrics(overlap, thresholds)
        energy_difference = abs(
            float(clockwise["metadata"]["average_energy_Eh"])
            - float(counterclockwise["metadata"]["average_energy_Eh"])
        )
        cross_history.append(
            {
                "index": index,
                "angle_degrees": 360.0 * index / NPOINTS,
                "projector_metrics": metrics,
                "average_energy_difference_Eh": energy_difference,
                "energy_pass": energy_difference
                <= thresholds.maximum_SA3_average_energy_difference_Eh,
            }
        )
    quality_pass = all(item["quality_pass"] for item in records)
    seed_pass = all(item["seed_SA2_to_SA3"]["pass"] for item in records)
    reference_pass = all(
        item["reference_SA3_to_history_SA3"]["pass"] and item["energy_pass"]
        for item in records
    )
    cross_pass = all(
        item["projector_metrics"]["pass"] and item["energy_pass"]
        for item in cross_history
    )
    status = (
        "PASS_RANK2_OPPOSED_HISTORY_REPLICATION"
        if quality_pass and seed_pass and reference_pass and cross_pass
        else "STOP_SA3_HISTORY_REPLICATION_FAILURE"
    )
    reason = (
        "Both SA2 histories generate the same independently referenced, externally isolated SA3 lower-two-state manifold."
        if status == "PASS_RANK2_OPPOSED_HISTORY_REPLICATION"
        else "At least one history-seeded SA3 point fails quality, rank, reference, cross-history, or energy reproduction."
    )
    return {
        "status": status,
        "reason": reason,
        "thresholds": asdict(thresholds),
        "history_SA3_records": records,
        "cross_history_records": cross_history,
        "quality_pass": quality_pass,
        "seed_rank_pass": seed_pass,
        "reference_reproduction_pass": reference_pass,
        "cross_history_reproduction_pass": cross_pass,
        "scalar_phase_evaluated": False,
        "individual_root_assignment_used": False,
    }


def validate_final_decision(root: Path) -> Dict[str, Any]:
    path = final_decision_path(root)
    result = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "history_decision_sha256": sha256_file(history_decision_path(root)),
        "scalar_phase_evaluated": False,
        "individual_root_assignment_used": False,
        "CI_count_from_rank2_Wilson_claimed": False,
        "manuscript_claim_authorized": False,
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError("final opposed-history decision immutable fields changed")
    for record in result.get("history_SA3_records", []):
        path = Path(record["history_SA3_file"]).resolve()
        if not path.exists() or sha256_file(path) != record["history_SA3_sha256"]:
            raise RuntimeError(f"history SA3 artifact changed: {path}")
    return result


def run_sa3(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol()
    root = args.root.resolve()
    authorization = validate_authorization(
        root,
        args.source_root,
        args.loop_root,
        args.screen_root,
        args.rank2_root,
    )
    require_pyscf()
    create_or_validate_manifest(root)
    history = validate_history_decision(root)
    if history.get("status") != "PASS_RANK2_OPPOSED_HISTORY":
        raise SystemExit(
            f"History decision is {history.get('status', 'UNKNOWN')}; SA3 is forbidden"
        )
    output = final_decision_path(root)
    if output.exists():
        result = validate_final_decision(root)
        print(f"CPDC_RANK2_OPPOSED_HISTORY={result['status']}", flush=True)
        return result
    specs = build_history_sa3_specs(args, authorization)
    run_sa3_workers(args, specs)
    analysis = analyze_history_sa3(args, specs)
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "history_decision_sha256": sha256_file(history_decision_path(root)),
        "manuscript_claim_authorized": False,
        "CI_count_from_rank2_Wilson_claimed": False,
        **analysis,
        "result_authorizes_inclusion_as_rank2_manifold_evidence": analysis["status"]
        == "PASS_RANK2_OPPOSED_HISTORY_REPLICATION",
        "scientific_scope": (
            "A pass establishes history-independent identity of the isolated rank-2 "
            "lowest-singlet manifold on the frozen central loop. It does not assign "
            "a scalar Berry phase or count internal conical intersections."
        ),
    }
    write_json(output, result)
    print("\nCPDC RANK-2 OPPOSED-HISTORY FINAL", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Reason: {result['reason']}", flush=True)
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_RANK2_OPPOSED_HISTORY={result['status']}", flush=True)
    return result


def classify_history_synthetic(
    point: bool, projector: bool, energy: bool, closure: bool, edge: bool, wilson: bool
) -> str:
    if not point:
        return "STOP_HISTORY_WAVEFUNCTION_FAILURE"
    if not projector or not energy:
        return "STOP_HISTORY_PROJECTOR_REPRODUCTION_FAILURE"
    if not closure:
        return "STOP_HISTORY_CLOSURE_FAILURE"
    if not edge:
        return "STOP_HISTORY_EDGE_INVARIANCE_FAILURE"
    if not wilson:
        return "STOP_HISTORY_WILSON_REPRODUCTION_FAILURE"
    return "PASS_RANK2_OPPOSED_HISTORY"


def selftest() -> None:
    validate_protocol()
    hashes = {
        "rank2": sha256_file(Path(rank2.__file__)),
        "rank2_protocol": sha256_file(
            Path(__file__).resolve().with_name(rank2.PROTOCOL_FILENAME)
        ),
        "antecedent": sha256_file(Path(antecedent.__file__)),
        "engine": sha256_file(Path(eng.__file__)),
        "overlap": sha256_file(Path(ov.__file__)),
    }
    expected = {
        "rank2": EXPECTED_RANK2_DRIVER_SHA256,
        "rank2_protocol": EXPECTED_RANK2_PROTOCOL_SHA256,
        "antecedent": EXPECTED_ANTECEDENT_DRIVER_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
        "overlap": EXPECTED_OVERLAP_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen component hash selftest failed: {hashes}")
    clockwise = history_order("clockwise")
    counterclockwise = history_order("counterclockwise")
    if (
        clockwise != list(range(1, NPOINTS))
        or counterclockwise != list(range(NPOINTS - 1, 0, -1))
        or set(clockwise) != set(counterclockwise)
        or len(clockwise) != NPOINTS - 1
    ):
        raise RuntimeError("opposed history ordering selftest failed")
    if (
        REFERENCE_SPOT_INDICES != (0, 1, 2, 3, 4)
        or SPOT_INDICES != (0, 2, 4, 6, 8)
        or [360.0 * index / NPOINTS for index in SPOT_INDICES]
        != [0.0, 9.0, 18.0, 27.0, 36.0]
    ):
        raise RuntimeError("40-to-80 SA3 reference mapping selftest failed")
    overlaps = rank2._synthetic_overlaps()
    invariants = rank2.wilson_invariants(rank2.wilson_product(overlaps))
    if rank2.gauge_audit(overlaps, invariants)["maximum_error"] > 1.0e-12:
        raise RuntimeError("synthetic U(2) gauge selftest failed")
    same = wilson_comparison(invariants, invariants, Thresholds())
    if not same["pass"] or max(
        same["eigenphase_set_difference_rad"],
        same["trace_difference"],
        same["determinant_difference"],
    ) > 1.0e-14:
        raise RuntimeError("synthetic Wilson reproduction selftest failed")
    statuses = {
        classify_history_synthetic(*flags)
        for flags in (
            (False, True, True, True, True, True),
            (True, False, True, True, True, True),
            (True, True, False, True, True, True),
            (True, True, True, False, True, True),
            (True, True, True, True, False, True),
            (True, True, True, True, True, False),
            (True, True, True, True, True, True),
        )
    }
    expected_statuses = {
        "STOP_HISTORY_WAVEFUNCTION_FAILURE",
        "STOP_HISTORY_PROJECTOR_REPRODUCTION_FAILURE",
        "STOP_HISTORY_CLOSURE_FAILURE",
        "STOP_HISTORY_EDGE_INVARIANCE_FAILURE",
        "STOP_HISTORY_WILSON_REPRODUCTION_FAILURE",
        "PASS_RANK2_OPPOSED_HISTORY",
    }
    if statuses != expected_statuses:
        raise RuntimeError(f"history decision branch selftest failed: {statuses}")
    print("CPDC_RANK2_OPPOSED_HISTORY_V1_SELFTEST=PASS")


def add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--loop-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--rank2-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest")
    qualify = subparsers.add_parser("qualify")
    add_roots(qualify)
    history = subparsers.add_parser("history")
    add_roots(history)
    history.add_argument("--basis", default=BASIS_KEY)
    history.add_argument("--threads", type=int, default=12)
    history.add_argument("--memory-mb", type=int, default=200000)
    sa3 = subparsers.add_parser("sa3")
    add_roots(sa3)
    sa3.add_argument("--basis", default=BASIS_KEY)
    sa3.add_argument("--threads", type=int, default=6)
    sa3.add_argument("--memory-mb", type=int, default=100000)
    sa3.add_argument("--max-jobs", type=int, default=4)

    history_worker_parser = subparsers.add_parser("history-worker")
    history_worker_parser.add_argument("--root", type=Path, required=True)
    history_worker_parser.add_argument("--source-root", type=Path, required=True)
    history_worker_parser.add_argument("--loop-root", type=Path, required=True)
    history_worker_parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    history_worker_parser.add_argument("--threads", type=int, default=12)
    history_worker_parser.add_argument("--memory-mb", type=int, default=200000)

    sa3_worker_parser = subparsers.add_parser("sa3-worker")
    sa3_worker_parser.add_argument("--target", type=Path, required=True)
    sa3_worker_parser.add_argument("--seed-sa2", type=Path, required=True)
    sa3_worker_parser.add_argument("--reference-sa3", type=Path, required=True)
    sa3_worker_parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    sa3_worker_parser.add_argument("--spot-index", type=int, required=True)
    sa3_worker_parser.add_argument("--angle-degrees", type=float, required=True)
    sa3_worker_parser.add_argument("--geometry-hash", required=True)
    sa3_worker_parser.add_argument("--threads", type=int, default=6)
    sa3_worker_parser.add_argument("--memory-mb", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "selftest":
        selftest()
        return
    if args.command == "qualify":
        run_qualify(args)
        return
    if args.threads < 1 or args.memory_mb < 2000:
        raise SystemExit("threads must be positive and memory-mb at least 2000")
    configure_threads(args.threads)
    if args.command == "history":
        if args.basis.lower() != BASIS_KEY:
            raise SystemExit("This protocol is locked to cc-pVTZ")
        run_history(args)
    elif args.command == "sa3":
        if args.basis.lower() != BASIS_KEY:
            raise SystemExit("This protocol is locked to cc-pVTZ")
        if args.max_jobs < 1:
            raise SystemExit("max-jobs must be positive")
        run_sa3(args)
    elif args.command == "history-worker":
        history_worker(args)
    elif args.command == "sa3-worker":
        sa3_worker(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
