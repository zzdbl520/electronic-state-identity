#!/usr/bin/env python3
"""Phase-blind 40-point central-gap landscape screen for singlet C5H5+.

This protocol follows a preregistered relaxed-multiring STOP that occurred
before phase exposure.  It compares the original conical radial extension of
the five frozen secondary-CI anchors with a smooth quadratic Fourier extension.
Only SA2 electronic quality and energy gaps are evaluated; no signed overlap,
Berry product, Wilson product, or topological phase is computed.
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

import CPDC_ccpVTZ_relaxed_multiring_v1 as antecedent
import cpdc_lift_engine as eng


SCRIPT_VERSION = "CPDC-CCPVTZ-CENTRAL-GAP-LANDSCAPE-v1.0"
PROTOCOL_ID = "CPDC-CCPVTZ-CENTRAL-GAP-LANDSCAPE-20260920"
PROTOCOL_FILENAME = "CPDC_ccpVTZ_central_gap_landscape_protocol_v1.json"
AUTHORIZATION_FILENAME = "CPDC_ccpVTZ_central_gap_landscape_authorization_v1.json"
MANIFEST_FILENAME = "CPDC_ccpVTZ_central_gap_landscape_manifest_v1.json"
SURFACE_FILENAME = "CPDC_ccpVTZ_central_gap_surfaces_v1.npz"
SURFACE_METADATA_FILENAME = "CPDC_ccpVTZ_central_gap_surfaces_v1.json"
DECISION_FILENAME = "CPDC_ccpVTZ_central_gap_landscape_v1_decision.json"

PYSCF_VERSION_EXACT = "2.13.1"
BASIS_KEY = "cc-pvtz"
N_RING = 5
N_ATOMS = 10
SURFACE_KINDS = ("cone_v1", "smooth_quadratic")
DIRECTIONS_DEGREES = (0.0, 36.0)
RADII = (0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)
TARGET_INTERNAL_GAP_EH = 2.0e-4
MAXIMUM_REPRODUCTION_ERROR_EH = 5.0e-7
POINT_ROLE = "central_gap_landscape_phase_blind_SA2_point"

EXPECTED_ANTECEDENT_DRIVER_SHA256 = (
    "ca724ea4f56428f84043b6fd28f442811f9dfb5107039bad5b325c9264cb2942"
)
EXPECTED_ENGINE_SHA256 = (
    "7ff5d5dab945128e6031e61e2510d836ffd42d0a5e1f86078ef72e9b0ddcd325"
)
EXPECTED_SOURCE_CANONICAL_SHA256 = (
    "21f4cb12adaed4f6fe8576b4982c2a881f9df12a9a4b341833fac6efa8f564b9"
)


@dataclass(frozen=True)
class ScreenPoint:
    surface_kind: str
    radius: float
    theta_degrees: float
    geometry_hash: str
    target: Path
    label: str


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.asarray(value.real).tolist(),
                "imag": np.asarray(value.imag).tolist(),
            }
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2) + "\n")
    os.replace(temporary, path)


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


def surface_path(root: Path) -> Path:
    return root / SURFACE_FILENAME


def surface_metadata_path(root: Path) -> Path:
    return root / SURFACE_METADATA_FILENAME


def decision_path(root: Path) -> Path:
    return root / DECISION_FILENAME


def require_pyscf() -> None:
    antecedent.require_pyscf()


def configure_threads(threads: int) -> None:
    antecedent.configure_threads(threads)


def validate_protocol() -> Dict[str, Any]:
    path = protocol_path()
    if not path.exists():
        raise RuntimeError(f"missing protocol: {path}")
    declaration = json.loads(path.read_text())
    grid = declaration.get("screen_grid", {})
    model = declaration.get("locked_model", {})
    thresholds = declaration.get("thresholds", {})
    production_thresholds = antecedent.Thresholds()
    expected_thresholds = {
        "maximum_final_orbital_gradient": production_thresholds.maximum_final_orbital_gradient,
        "maximum_spin_square_error": production_thresholds.maximum_spin_square_error,
        "maximum_mo_orthonormality_error": production_thresholds.maximum_mo_orthonormality_error,
        "maximum_casci_reproduction_error_Eh": production_thresholds.maximum_casci_reproduction_error_Eh,
        "minimum_active_reference_sigma": production_thresholds.minimum_active_reference_sigma,
        "minimum_third_singlet_gap_Eh": production_thresholds.minimum_third_singlet_gap_Eh,
        "maximum_average_energy_above_center_Eh": production_thresholds.maximum_average_energy_above_center_Eh,
        "minimum_CC_bond_angstrom": production_thresholds.minimum_CC_bond_angstrom,
        "maximum_CC_bond_angstrom": production_thresholds.maximum_CC_bond_angstrom,
        "minimum_CH_bond_angstrom": production_thresholds.minimum_CH_bond_angstrom,
        "maximum_CH_bond_angstrom": production_thresholds.maximum_CH_bond_angstrom,
        "maximum_planarity_error_angstrom": production_thresholds.maximum_planarity_error_angstrom,
        "target_internal_loop_gap_Eh": TARGET_INTERNAL_GAP_EH,
        "maximum_independent_gap_reproduction_error_Eh": MAXIMUM_REPRODUCTION_ERROR_EH,
    }
    expected = {
        "protocol_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "basis": BASIS_KEY,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "surface_kinds": list(SURFACE_KINDS),
        "directions_degrees": list(DIRECTIONS_DEGREES),
        "radii": list(RADII),
        "total_logical_points": 40,
        "target_internal_loop_gap_Eh": TARGET_INTERNAL_GAP_EH,
        "maximum_independent_gap_reproduction_error_Eh": (
            MAXIMUM_REPRODUCTION_ERROR_EH
        ),
    }
    actual = {
        "protocol_version": declaration.get("protocol_version"),
        "protocol_id": declaration.get("protocol_id"),
        "basis": str(model.get("basis", "")).lower(),
        "pyscf_version_exact": model.get("pyscf_version_exact"),
        "surface_kinds": grid.get("surface_kinds"),
        "directions_degrees": grid.get("directions_degrees"),
        "radii": grid.get("radii"),
        "total_logical_points": grid.get("total_logical_points"),
        "target_internal_loop_gap_Eh": thresholds.get(
            "target_internal_loop_gap_Eh"
        ),
        "maximum_independent_gap_reproduction_error_Eh": thresholds.get(
            "maximum_independent_gap_reproduction_error_Eh"
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"protocol disagrees with executable: expected={expected}, actual={actual}"
        )
    if thresholds != expected_thresholds:
        raise RuntimeError(
            "protocol quality thresholds disagree with the locked executable values"
        )
    if declaration.get("antecedent", {}).get("required_phase_inspected") is not False:
        raise RuntimeError("antecedent phase embargo changed")
    if declaration.get("screen_grid", {}).get("phase_or_signed_overlap_allowed") is not False:
        raise RuntimeError("screen may not inspect phase or signed overlaps")
    return declaration


def _all_point_checks_pass(attempt: Mapping[str, Any]) -> bool:
    checks = attempt.get("point_checks", {})
    return bool(checks) and all(
        isinstance(item, Mapping) and item and all(bool(value) for value in item.values())
        for item in checks.values()
    )


def _integrity_item(
    item: Mapping[str, Any], prior_root: Path, expected_index: int
) -> Dict[str, Any]:
    path = Path(str(item.get("path", ""))).resolve()
    if (
        int(item.get("index", -1)) != expected_index
        or not path.exists()
        or not path.is_relative_to(prior_root)
        or sha256_file(path) != item.get("sha256")
    ):
        raise RuntimeError(f"invalid antecedent point integrity at index {expected_index}")
    solution = eng.load_solution(path)
    metadata = solution["metadata"]
    return {
        "index": expected_index,
        "path": str(path),
        "sha256": sha256_file(path),
        "ux": float(item["ux"]),
        "uy": float(item["uy"]),
        "geometry_hash": item["geometry_hash"],
        "adiabatic_gap_Eh": float(metadata["adiabatic_gap_Eh"]),
        "average_energy_Eh": float(metadata["average_energy_Eh"]),
    }


def build_authorization(
    root: Path, source_root: Path, prior_root: Path
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    prior_root = prior_root.resolve()
    prior_source = antecedent.validate_source_authorization(
        prior_root, source_root, require_pass=True
    )
    prior_decision_file = antecedent.loop_geometry_decision_path(prior_root)
    if not prior_decision_file.exists():
        raise RuntimeError(f"missing antecedent decision: {prior_decision_file}")
    prior = json.loads(prior_decision_file.read_text())
    qualifications = prior.get("loop_qualifications", [])
    central = qualifications[0] if qualifications else {}
    attempts = central.get("attempts", [])
    level0 = attempts[0] if attempts else {}
    point_inventory = level0.get("point_integrity", [])
    checks: Dict[str, bool] = {
        "antecedent_script_version": prior.get("script_version")
        == antecedent.SCRIPT_VERSION,
        "antecedent_protocol_id": prior.get("protocol_id")
        == antecedent.PROTOCOL_ID,
        "antecedent_stop_status": prior.get("status")
        == "STOP_BOUNDED_LOOP_GEOMETRY_FAILURE",
        "antecedent_phase_not_inspected": prior.get("phase_inspected") is False,
        "antecedent_phase_product_not_evaluated": prior.get(
            "phase_product_evaluated"
        )
        is False,
        "antecedent_manuscript_claim_not_authorized": prior.get(
            "manuscript_claim_authorized"
        )
        is False,
        "topology_decision_absent": not antecedent.topology_decision_path(
            prior_root
        ).exists(),
        "only_central_loop_was_qualified": bool(
            len(qualifications) == 1
            and central.get("loop_name") == "central_inner"
            and central.get("status") == "STOP_LOOP_GEOMETRY_FAILURE"
        ),
        "all_three_meshes_attempted": bool(
            len(attempts) == 3
            and [int(item.get("npoints", -1)) for item in attempts]
            == [20, 40, 80]
        ),
        "all_antecedent_point_checks_passed": bool(
            attempts and all(_all_point_checks_pass(item) for item in attempts)
        ),
        "all_antecedent_meshes_phase_blind": bool(
            attempts
            and all(
                item.get("signed_edge_overlaps_recorded") is False
                and item.get("phase_product_evaluated") is False
                for item in attempts
            )
        ),
        "level0_inventory_complete": bool(
            isinstance(point_inventory, list) and len(point_inventory) == 20
        ),
        "prior_source_authorization_passed": prior_source.get("status")
        == "PASS_SOURCE_AUTHORIZATION",
    }

    controls: Dict[str, Any] = {}
    artifact_error = None
    try:
        if not checks["level0_inventory_complete"]:
            raise RuntimeError("antecedent level-0 inventory is incomplete")
        controls["cone_r0p45_theta0"] = _integrity_item(
            point_inventory[0], prior_root, 0
        )
        controls["cone_r0p45_theta36"] = _integrity_item(
            point_inventory[2], prior_root, 2
        )
        checks["antecedent_control_angles"] = bool(
            abs(controls["cone_r0p45_theta0"]["ux"] - 0.45) <= 1.0e-14
            and abs(controls["cone_r0p45_theta0"]["uy"]) <= 1.0e-14
            and abs(
                math.degrees(
                    math.atan2(
                        controls["cone_r0p45_theta36"]["uy"],
                        controls["cone_r0p45_theta36"]["ux"],
                    )
                )
                - 36.0
            )
            <= 1.0e-12
        )
        checks["antecedent_control_artifacts_intact"] = True
    except Exception as exc:
        checks["antecedent_control_artifacts_intact"] = False
        artifact_error = f"{type(exc).__name__}: {exc}"

    status = (
        "PASS_GAP_SCREEN_AUTHORIZATION"
        if all(checks.values())
        else "STOP_GAP_SCREEN_AUTHORIZATION_FAILURE"
    )
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "prior_root": str(prior_root),
        "status": status,
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "signed_overlap_evaluated": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "artifact_error": artifact_error,
        "antecedent_decision_file": str(prior_decision_file.resolve()),
        "antecedent_decision_sha256": sha256_file(prior_decision_file),
        "antecedent_source_authorization_file": str(
            antecedent.source_authorization_path(prior_root).resolve()
        ),
        "antecedent_source_authorization_sha256": sha256_file(
            antecedent.source_authorization_path(prior_root)
        ),
        "source_decision_canonical_sha256": prior_source.get(
            "source_decision_canonical_sha256"
        ),
        "center_solution_file": prior_source.get("center_solution_file"),
        "center_solution_sha256": prior_source.get("center_solution_sha256"),
        "center_average_energy_Eh": prior_source.get("center_average_energy_Eh"),
        "replica_solution_integrity": prior_source.get(
            "replica_solution_integrity", []
        ),
        "antecedent_controls": controls,
    }


def validate_authorization(
    root: Path, source_root: Path, prior_root: Path
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    prior_root = prior_root.resolve()
    path = authorization_path(root)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run qualify first")
    record = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "prior_root": str(prior_root),
        "status": "PASS_GAP_SCREEN_AUTHORIZATION",
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "signed_overlap_evaluated": False,
        "manuscript_claim_authorized": False,
        "source_decision_canonical_sha256": EXPECTED_SOURCE_CANONICAL_SHA256,
    }
    if {key: record.get(key) for key in expected} != expected:
        raise RuntimeError("gap-screen authorization failed immutable-field validation")
    artifacts = [
        (
            record.get("antecedent_decision_file"),
            record.get("antecedent_decision_sha256"),
            prior_root,
        ),
        (
            record.get("antecedent_source_authorization_file"),
            record.get("antecedent_source_authorization_sha256"),
            prior_root,
        ),
        (
            record.get("center_solution_file"),
            record.get("center_solution_sha256"),
            source_root,
        ),
    ]
    for replica in record.get("replica_solution_integrity", []):
        artifacts.append((replica.get("path"), replica.get("sha256"), source_root))
    for control in record.get("antecedent_controls", {}).values():
        artifacts.append((control.get("path"), control.get("sha256"), prior_root))
    for raw_path, expected_hash, owner in artifacts:
        artifact = Path(str(raw_path)).resolve()
        if (
            not artifact.exists()
            or not artifact.is_relative_to(owner)
            or sha256_file(artifact) != expected_hash
        ):
            raise RuntimeError(f"authorized artifact changed: {artifact}")
    if antecedent.topology_decision_path(prior_root).exists():
        raise RuntimeError("antecedent topology decision appeared after authorization")
    return record


def run_qualify(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = authorization_path(root)
    if path.exists():
        result = validate_authorization(root, args.source_root, args.prior_root)
    else:
        try:
            result = build_authorization(root, args.source_root, args.prior_root)
        except Exception as exc:
            result = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "root": str(root),
                "source_root": str(args.source_root.resolve()),
                "prior_root": str(args.prior_root.resolve()),
                "status": "STOP_GAP_SCREEN_AUTHORIZATION_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}",
                "phase_inspected": False,
                "phase_product_evaluated": False,
                "signed_overlap_evaluated": False,
                "manuscript_claim_authorized": False,
            }
        write_json(path, result)
    print("\nCPDC CENTRAL-GAP LANDSCAPE AUTHORIZATION", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Decision file: {path}", flush=True)
    print(f"CPDC_GAP_SCREEN_SOURCE={result['status']}", flush=True)
    return result


def build_manifest(root: Path) -> Dict[str, Any]:
    files = {
        "screen_driver": Path(__file__).resolve(),
        "screen_protocol": protocol_path(),
        "antecedent_driver": Path(antecedent.__file__).resolve(),
        "production_engine": Path(eng.__file__).resolve(),
        "overlap_module": Path(antecedent.ov.__file__).resolve(),
        "frozen_source_snapshot": antecedent.frozen_snapshot_path(),
    }
    auth = authorization_path(root)
    if not auth.exists():
        raise RuntimeError("authorization must precede manifest creation")
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root.resolve()),
        "basis": BASIS_KEY,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "immutable_files": {
            key: {"name": path.name, "sha256": sha256_file(path)}
            for key, path in files.items()
        },
        "authorization": {
            "name": auth.name,
            "sha256": sha256_file(auth),
        },
        "phase_embargo": "No phase or signed overlap is implemented by this executable.",
        "created_unix": time.time(),
    }


def create_or_validate_manifest(root: Path) -> Dict[str, Any]:
    path = manifest_path(root)
    current = build_manifest(root)
    if not path.exists():
        write_json(path, current)
        return current
    stored = json.loads(path.read_text())
    for key in (
        "script_version",
        "protocol_id",
        "root",
        "basis",
        "pyscf_version_exact",
        "immutable_files",
        "authorization",
        "phase_embargo",
    ):
        if stored.get(key) != current.get(key):
            raise RuntimeError(f"manifest mismatch for {key}; use a new root")
    return stored


def _surface_coefficients_from_authorization(
    authorization: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = eng.load_solution(
        Path(str(authorization["center_solution_file"]))
    )["geometry_angstrom"]
    anchors = []
    for item in authorization["replica_solution_integrity"]:
        anchors.append(
            eng.load_solution(Path(str(item["path"])))["geometry_angstrom"]
        )
    anchor_array = np.asarray(anchors, dtype=float)
    frequencies, coefficients = antecedent._fourier_coefficients(
        anchor_array - np.asarray(center, dtype=float)[None, :, :]
    )
    return (
        np.asarray(center, dtype=float),
        anchor_array,
        frequencies,
        coefficients,
    )


def surface_geometry(
    surface: Mapping[str, Any], surface_kind: str, radius: float, theta_degrees: float
) -> np.ndarray:
    if surface_kind not in SURFACE_KINDS:
        raise ValueError(f"unknown surface kind: {surface_kind}")
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    center = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    if radius == 0.0:
        return center.copy()
    theta = math.radians(float(theta_degrees))
    frequencies = np.asarray(surface["frequencies"], dtype=int)
    coefficients = np.asarray(surface["fourier_coefficients"])
    phase = np.exp(1j * frequencies.astype(float) * theta)
    if surface_kind == "cone_v1":
        radial_weights = np.full(frequencies.shape, radius, dtype=float)
    else:
        radial_weights = np.where(np.abs(frequencies) == 1, radius, radius**2)
    displacement = np.tensordot(
        radial_weights * phase, coefficients, axes=(0, 0)
    )
    imaginary_error = float(np.max(np.abs(np.imag(displacement))))
    if imaginary_error > 5.0e-11:
        raise RuntimeError(
            f"surface lost reality: kind={surface_kind} error={imaginary_error:.3e}"
        )
    return center + np.asarray(np.real(displacement), dtype=float)


def build_or_validate_surfaces(
    root: Path, authorization: Mapping[str, Any]
) -> Dict[str, Any]:
    npz = surface_path(root)
    metadata_file = surface_metadata_path(root)
    if npz.exists() or metadata_file.exists():
        if not (npz.exists() and metadata_file.exists()):
            raise RuntimeError("partial surface definition exists; use a new root")
        metadata = json.loads(metadata_file.read_text())
        if metadata.get("surface_npz_sha256") != sha256_file(npz):
            raise RuntimeError("surface NPZ changed")
        return metadata

    center, anchors, frequencies, coefficients = (
        _surface_coefficients_from_authorization(authorization)
    )
    surface = {
        "center_geometry_angstrom": center,
        "frequencies": frequencies,
        "fourier_coefficients": coefficients,
    }
    errors: Dict[str, List[float]] = {kind: [] for kind in SURFACE_KINDS}
    for kind in SURFACE_KINDS:
        for replica in range(N_RING):
            theta = 72.0 * replica
            geometry = surface_geometry(surface, kind, 1.0, theta)
            errors[kind].append(
                float(np.max(np.abs(geometry - anchors[replica])))
            )
    maximum_error = max(max(values) for values in errors.values())
    if maximum_error > 2.0e-10:
        raise RuntimeError(
            f"screen surfaces miss a frozen anchor by {maximum_error:.3e} Angstrom"
        )
    npz_metadata = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "surface_kinds": list(SURFACE_KINDS),
        "frequencies": frequencies.tolist(),
        "maximum_anchor_error_angstrom": maximum_error,
        "phase_inspected": False,
    }
    temporary = npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(jsonable(npz_metadata))),
        center_geometry_angstrom=center,
        anchor_geometries_angstrom=anchors,
        frequencies=frequencies,
        fourier_coefficients_real=np.real(coefficients),
        fourier_coefficients_imag=np.imag(coefficients),
    )
    os.replace(temporary, npz)
    metadata = {
        **npz_metadata,
        "surface_npz_file": str(npz.resolve()),
        "surface_npz_sha256": sha256_file(npz),
        "anchor_errors_angstrom": errors,
        "cone_formula": "r times all Fourier modes",
        "smooth_formula": "r times |m|=1; r^2 times m=0 and |m|=2",
    }
    write_json(metadata_file, metadata)
    return metadata


def load_surfaces(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "metadata": json.loads(str(data["metadata_json"].item())),
            "center_geometry_angstrom": np.asarray(
                data["center_geometry_angstrom"]
            ),
            "anchor_geometries_angstrom": np.asarray(
                data["anchor_geometries_angstrom"]
            ),
            "frequencies": np.asarray(data["frequencies"], dtype=int),
            "fourier_coefficients": np.asarray(
                data["fourier_coefficients_real"]
            )
            + 1j * np.asarray(data["fourier_coefficients_imag"]),
        }


def screen_points(
    root: Path, surface: Mapping[str, Any]
) -> List[ScreenPoint]:
    result: List[ScreenPoint] = []
    for surface_kind in SURFACE_KINDS:
        for theta in DIRECTIONS_DEGREES:
            for radius in RADII:
                geometry = surface_geometry(surface, surface_kind, radius, theta)
                geometry_hash = eng.array_hash(geometry)
                target = root / "points" / f"point_{geometry_hash}" / "solution.npz"
                result.append(
                    ScreenPoint(
                        surface_kind=surface_kind,
                        radius=float(radius),
                        theta_degrees=float(theta),
                        geometry_hash=geometry_hash,
                        target=target,
                        label=(
                            f"{surface_kind}_theta{theta:05.1f}_r{radius:05.2f}"
                        ),
                    )
                )
    if len(result) != 40:
        raise RuntimeError(f"expected 40 logical points; found {len(result)}")
    return result


def validate_existing_point(
    point: ScreenPoint, surface: Mapping[str, Any], basis: str
) -> Dict[str, Any]:
    geometry = surface_geometry(
        surface, point.surface_kind, point.radius, point.theta_degrees
    )
    if eng.array_hash(geometry) != point.geometry_hash:
        raise RuntimeError(f"geometry changed for {point.label}")
    eng.validate_existing_solution(
        point.target,
        geometry,
        basis,
        POINT_ROLE,
        gradients_required=False,
    )
    return eng.load_solution(point.target)


def point_worker(args: argparse.Namespace) -> None:
    configure_threads(args.threads)
    require_pyscf()
    surface = load_surfaces(args.surface_npz.resolve())
    geometry = surface_geometry(
        surface, args.surface_kind, args.radius, args.theta_degrees
    )
    geometry_hash = eng.array_hash(geometry)
    if geometry_hash != args.geometry_hash:
        raise RuntimeError(
            f"worker geometry hash mismatch: {geometry_hash} != {args.geometry_hash}"
        )
    metadata = eng.run_sa_solution(
        target=args.target.resolve(),
        geometry=geometry,
        basis=args.basis,
        memory_mb=args.memory_mb,
        role=POINT_ROLE,
        center_seed=args.center_seed.resolve(),
        point_descriptor={
            "kind": "central_gap_landscape_phase_blind_point",
            "surface_kind": args.surface_kind,
            "radius": float(args.radius),
            "theta_degrees": float(args.theta_degrees),
            "geometry_hash": geometry_hash,
        },
        compute_state_gradients=False,
    )
    print(
        f"POINT {args.surface_kind} r={args.radius:.6f} "
        f"theta={args.theta_degrees:.3f} gap={metadata['adiabatic_gap_Eh']:.9e} Eh "
        f"grad={metadata['maximum_orbital_gradient']:.3e}",
        flush=True,
    )


def worker_command(
    args: argparse.Namespace,
    point: ScreenPoint,
    center_seed: Path,
    surfaces_npz: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "point-worker",
        "--target",
        str(point.target.resolve()),
        "--surface-npz",
        str(surfaces_npz.resolve()),
        f"--surface-kind={point.surface_kind}",
        f"--radius={point.radius:.17g}",
        f"--theta-degrees={point.theta_degrees:.17g}",
        "--geometry-hash",
        point.geometry_hash,
        "--center-seed",
        str(center_seed.resolve()),
        "--basis",
        args.basis,
        "--threads",
        str(args.threads),
        "--memory-mb",
        str(args.memory_mb),
    ]


def subprocess_worker(command: Sequence[str], log_path: Path, threads: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(int(threads))
    with log_path.open("a") as log:
        log.write(f"\n===== worker launch {time.time():.6f} =====\n")
        log.write("command=" + json.dumps(list(command)) + "\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"point worker failed with code {completed.returncode}; see {log_path}"
        )


def run_point_batch(
    args: argparse.Namespace,
    points: Sequence[ScreenPoint],
    surface: Mapping[str, Any],
    center_seed: Path,
) -> None:
    unique: Dict[str, ScreenPoint] = {
        str(point.target.resolve()): point for point in points
    }
    pending: List[ScreenPoint] = []
    for point in unique.values():
        if point.target.exists():
            validate_existing_point(point, surface, args.basis)
        else:
            pending.append(point)
    if not pending:
        return
    print(
        f"Launching {len(pending)} new phase-blind screen points with "
        f"max_jobs={args.max_jobs}, threads_per_job={args.threads}",
        flush=True,
    )
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {}
        for point in pending:
            command = worker_command(
                args, point, center_seed, surface_path(args.root.resolve())
            )
            future = executor.submit(
                subprocess_worker,
                command,
                point.target.parent / "worker.stdout",
                args.threads,
            )
            futures[future] = point
        for future in as_completed(futures):
            point = futures[future]
            try:
                future.result()
                solution = validate_existing_point(point, surface, args.basis)
                metadata = solution["metadata"]
                print(
                    f"DONE {point.label} gap={metadata['adiabatic_gap_Eh']:.9e} Eh",
                    flush=True,
                )
            except Exception as exc:
                failure = f"{point.label}: {type(exc).__name__}: {exc}"
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def analyze_points(
    args: argparse.Namespace,
    points: Sequence[ScreenPoint],
    surface: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    center = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    center_energy = float(authorization["center_average_energy_Eh"])
    thresholds = antecedent.Thresholds()
    records: List[Dict[str, Any]] = []
    for point in points:
        solution = validate_existing_point(point, surface, args.basis)
        geometry = surface_geometry(
            surface, point.surface_kind, point.radius, point.theta_degrees
        )
        checks, geometry_metrics = antecedent.production_point_checks(
            solution, geometry, center, center_energy, thresholds
        )
        metadata = solution["metadata"]
        records.append(
            {
                "surface_kind": point.surface_kind,
                "radius": point.radius,
                "theta_degrees": point.theta_degrees,
                "label": point.label,
                "geometry_hash": point.geometry_hash,
                "solution_file": str(point.target.resolve()),
                "solution_sha256": sha256_file(point.target),
                "adiabatic_gap_Eh": float(metadata["adiabatic_gap_Eh"]),
                "third_singlet_gap_Eh": float(metadata["third_singlet_gap_Eh"]),
                "average_energy_Eh": float(metadata["average_energy_Eh"]),
                "maximum_orbital_gradient": float(
                    metadata["maximum_orbital_gradient"]
                ),
                "point_checks": checks,
                "all_point_checks_pass": bool(all(checks.values())),
                "geometry_metrics": geometry_metrics,
                "phase_inspected": False,
                "signed_overlap_evaluated": False,
            }
        )
    return records


def _find_record(
    records: Sequence[Mapping[str, Any]],
    surface_kind: str,
    radius: float,
    theta: float,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in records
        if item["surface_kind"] == surface_kind
        and abs(float(item["radius"]) - radius) <= 1.0e-14
        and abs(float(item["theta_degrees"]) - theta) <= 1.0e-14
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one record for {surface_kind}/r={radius}/theta={theta}"
        )
    return matches[0]


def summarize_surface(
    records: Sequence[Mapping[str, Any]], surface_kind: str
) -> Dict[str, Any]:
    selected = [item for item in records if item["surface_kind"] == surface_kind]
    directions: Dict[str, Any] = {}
    for theta in DIRECTIONS_DEGREES:
        ray = [
            item
            for item in selected
            if abs(float(item["theta_degrees"]) - theta) <= 1.0e-14
        ]
        ray.sort(key=lambda item: float(item["radius"]))
        gaps = [float(item["adiabatic_gap_Eh"]) for item in ray]
        maximum_index = int(np.argmax(gaps))
        minimum_index = int(np.argmin(gaps))
        directions[f"theta_{theta:.1f}_degrees"] = {
            "role": "secondary_CI_spoke" if theta == 0.0 else "spoke_bisector",
            "radii": [float(item["radius"]) for item in ray],
            "gaps_Eh": gaps,
            "minimum_sampled_gap_Eh": gaps[minimum_index],
            "minimum_sampled_gap_radius": float(ray[minimum_index]["radius"]),
            "maximum_sampled_gap_Eh": gaps[maximum_index],
            "maximum_sampled_gap_radius": float(ray[maximum_index]["radius"]),
            "sampled_target_crossed": bool(max(gaps) >= TARGET_INTERNAL_GAP_EH),
        }
    spoke = directions["theta_0.0_degrees"]
    bisector = directions["theta_36.0_degrees"]
    return {
        "surface_kind": surface_kind,
        "all_points_pass": bool(
            all(bool(item["all_point_checks_pass"]) for item in selected)
        ),
        "directions": directions,
        "spoke_sampled_maximum_gap_Eh": spoke["maximum_sampled_gap_Eh"],
        "spoke_sampled_maximum_radius": spoke["maximum_sampled_gap_radius"],
        "spoke_target_crossed": spoke["sampled_target_crossed"],
        "bisector_sampled_maximum_gap_Eh": bisector["maximum_sampled_gap_Eh"],
        "bisector_sampled_maximum_radius": bisector["maximum_sampled_gap_radius"],
        "sampled_values_are_not_continuous_upper_bounds": True,
    }


def classify(
    records: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    cone0 = _find_record(records, "cone_v1", 0.45, 0.0)
    cone36 = _find_record(records, "cone_v1", 0.45, 36.0)
    old0 = authorization["antecedent_controls"]["cone_r0p45_theta0"]
    old36 = authorization["antecedent_controls"]["cone_r0p45_theta36"]
    reproduction = {
        "theta0_geometry_hash_matches": cone0["geometry_hash"]
        == old0["geometry_hash"],
        "theta36_geometry_hash_matches": cone36["geometry_hash"]
        == old36["geometry_hash"],
        "theta0_gap_difference_Eh": abs(
            float(cone0["adiabatic_gap_Eh"])
            - float(old0["adiabatic_gap_Eh"])
        ),
        "theta36_gap_difference_Eh": abs(
            float(cone36["adiabatic_gap_Eh"])
            - float(old36["adiabatic_gap_Eh"])
        ),
    }
    reproduction["pass"] = bool(
        reproduction["theta0_geometry_hash_matches"]
        and reproduction["theta36_geometry_hash_matches"]
        and reproduction["theta0_gap_difference_Eh"]
        <= MAXIMUM_REPRODUCTION_ERROR_EH
        and reproduction["theta36_gap_difference_Eh"]
        <= MAXIMUM_REPRODUCTION_ERROR_EH
    )
    summaries = {
        kind: summarize_surface(records, kind) for kind in SURFACE_KINDS
    }
    all_points_pass = all(
        bool(item["all_point_checks_pass"]) for item in records
    )
    cone_pass = bool(summaries["cone_v1"]["spoke_target_crossed"])
    smooth_pass = bool(
        summaries["smooth_quadratic"]["spoke_target_crossed"]
    )
    if not all_points_pass:
        status = "STOP_SCREEN_WAVEFUNCTION_FAILURE"
        reason = "At least one screen point failed a locked quality gate."
    elif not reproduction["pass"]:
        status = "STOP_V1_REPRODUCTION_FAILURE"
        reason = "Independent cone-v1 controls did not reproduce the antecedent."
    elif cone_pass and smooth_pass:
        status = "PROMOTE_BOTH_SURFACES_TO_ANGULAR_ARBITRATION"
        reason = "Both sampled spokes reach the unchanged internal-gap target."
    elif smooth_pass:
        status = "PROMOTE_SMOOTH_SURFACE_TO_ANGULAR_QUALIFICATION"
        reason = "Only the smooth sampled spoke reaches the unchanged gap target."
    elif cone_pass:
        status = "PROMOTE_CONE_SURFACE_TO_ANGULAR_QUALIFICATION"
        reason = "Only the cone sampled spoke reaches the unchanged gap target."
    else:
        status = "REFINE_SPOKE_MAXIMUM_OR_PIVOT_PROJECTOR"
        reason = (
            "Neither sampled spoke reaches the unchanged 2e-4 Eh target; no scalar "
            "phase is authorized."
        )
    return status, reason, summaries, reproduction


def validate_existing_decision(root: Path) -> Dict[str, Any]:
    path = decision_path(root)
    result = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "signed_overlap_evaluated": False,
        "manuscript_claim_authorized": False,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "surface_npz_sha256": sha256_file(surface_path(root)),
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError("existing screen decision failed integrity validation")
    records = result.get("point_records", [])
    if not isinstance(records, list) or len(records) != 40:
        raise RuntimeError("existing screen decision has incomplete point inventory")
    for item in records:
        path = Path(str(item.get("solution_file", ""))).resolve()
        if (
            not path.exists()
            or not path.is_relative_to(root.resolve())
            or sha256_file(path) != item.get("solution_sha256")
        ):
            raise RuntimeError(f"screen point changed after decision: {path}")
    return result


def run_screen(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != BASIS_KEY:
        raise SystemExit("This protocol is locked to cc-pVTZ")
    validate_protocol()
    root = args.root.resolve()
    authorization = validate_authorization(
        root, args.source_root, args.prior_root
    )
    create_or_validate_manifest(root)
    require_pyscf()
    output = decision_path(root)
    if output.exists():
        result = validate_existing_decision(root)
        print(f"CPDC_CENTRAL_GAP_SCREEN={result['status']}", flush=True)
        return result

    surface_definition = build_or_validate_surfaces(root, authorization)
    surface = load_surfaces(surface_path(root))
    points = screen_points(root, surface)
    center_seed = Path(str(authorization["center_solution_file"]))
    run_point_batch(args, points, surface, center_seed)
    records = analyze_points(args, points, surface, authorization)
    status, reason, summaries, reproduction = classify(records, authorization)
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "reason": reason,
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "signed_overlap_evaluated": False,
        "manuscript_claim_authorized": False,
        "authorization_file": str(authorization_path(root)),
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "surface_definition": surface_definition,
        "surface_npz_sha256": sha256_file(surface_path(root)),
        "screen_grid": {
            "surface_kinds": list(SURFACE_KINDS),
            "directions_degrees": list(DIRECTIONS_DEGREES),
            "radii": list(RADII),
            "logical_point_count": len(points),
            "unique_geometry_count": len({item.geometry_hash for item in points}),
        },
        "target_internal_gap_Eh": TARGET_INTERNAL_GAP_EH,
        "surface_summaries": summaries,
        "antecedent_reproduction": reproduction,
        "point_records": records,
        "decision_scope": (
            "This screen selects only the next phase-blind protocol. Sampled maxima "
            "are not asserted to be continuous upper bounds."
        ),
        "authorized_next_step": {
            "PROMOTE_SMOOTH_SURFACE_TO_ANGULAR_QUALIFICATION": (
                "Run a separately preregistered full-angle maximin loop qualification "
                "on the smooth surface without phase."
            ),
            "PROMOTE_CONE_SURFACE_TO_ANGULAR_QUALIFICATION": (
                "Run a separately preregistered full-angle maximin loop qualification "
                "on the cone surface without phase."
            ),
            "PROMOTE_BOTH_SURFACES_TO_ANGULAR_ARBITRATION": (
                "Qualify full-angle maximin loops on both surfaces before selecting one."
            ),
            "REFINE_SPOKE_MAXIMUM_OR_PIVOT_PROJECTOR": (
                "Do not run scalar phase. Refine the phase-blind spoke maximum or "
                "preregister a rank-2 projector-topology calculation."
            ),
        }.get(status, "Stop and diagnose the failed quality or reproduction gate."),
    }
    write_json(output, result)
    print("\nCPDC CENTRAL-GAP LANDSCAPE SCREEN", flush=True)
    for kind in SURFACE_KINDS:
        summary = summaries[kind]
        print(
            f"{kind}: spoke_max={summary['spoke_sampled_maximum_gap_Eh']:.9e} Eh "
            f"at r={summary['spoke_sampled_maximum_radius']:.2f}; "
            f"target_crossed={summary['spoke_target_crossed']}",
            flush=True,
        )
    print(
        "antecedent reproduction: "
        f"theta0_dgap={reproduction['theta0_gap_difference_Eh']:.3e} Eh "
        f"theta36_dgap={reproduction['theta36_gap_difference_Eh']:.3e} Eh "
        f"pass={reproduction['pass']}",
        flush=True,
    )
    print(f"Status: {status}", flush=True)
    print(f"Reason: {reason}", flush=True)
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_CENTRAL_GAP_SCREEN={status}", flush=True)
    return result


def selftest() -> None:
    validate_protocol()
    if sha256_file(Path(antecedent.__file__).resolve()) != EXPECTED_ANTECEDENT_DRIVER_SHA256:
        raise RuntimeError("antecedent driver hash mismatch")
    if sha256_file(Path(eng.__file__).resolve()) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("production engine hash mismatch")
    snapshot = json.loads(antecedent.frozen_snapshot_path().read_text())
    if antecedent.canonical_json_sha256(snapshot) != EXPECTED_SOURCE_CANONICAL_SHA256:
        raise RuntimeError("frozen source snapshot mismatch")
    coefficients_q = np.asarray(
        snapshot["candidate_coefficients_angstrom"], dtype=float
    )
    center = np.asarray(eng.REFERENCE_GEOMETRY, dtype=float)
    candidate = eng.geometry_from_coefficients(coefficients_q)
    anchors = np.asarray(
        [eng.symmetry_replica(candidate, replica) for replica in range(N_RING)]
    )
    frequencies, coefficients = antecedent._fourier_coefficients(
        anchors - center[None, :, :]
    )
    surface = {
        "center_geometry_angstrom": center,
        "anchor_geometries_angstrom": anchors,
        "frequencies": frequencies,
        "fourier_coefficients": coefficients,
    }
    for kind in SURFACE_KINDS:
        if np.max(np.abs(surface_geometry(surface, kind, 0.0, 19.0) - center)) > 1.0e-14:
            raise RuntimeError(f"{kind} misses the common center")
        for replica in range(N_RING):
            reconstructed = surface_geometry(surface, kind, 1.0, 72.0 * replica)
            if np.max(np.abs(reconstructed - anchors[replica])) > 2.0e-12:
                raise RuntimeError(f"{kind} misses anchor {replica}")
    epsilon = 1.0e-7
    theta = 23.0
    phase = np.exp(1j * frequencies.astype(float) * math.radians(theta))
    linear = np.tensordot(
        phase * (np.abs(frequencies) == 1), coefficients, axes=(0, 0)
    ).real
    derivative = (
        surface_geometry(surface, "smooth_quadratic", epsilon, theta) - center
    ) / epsilon
    if np.max(np.abs(derivative - linear)) > 2.0e-8:
        raise RuntimeError("smooth surface lacks the declared linear center tangent")
    points = screen_points(Path("/tmp/cpdc_gap_screen_selftest"), surface)
    if len(points) != 40 or len({point.geometry_hash for point in points}) != 40:
        raise RuntimeError("screen inventory selftest failed")
    thresholds = antecedent.Thresholds()
    for point in points:
        geometry = surface_geometry(
            surface, point.surface_kind, point.radius, point.theta_degrees
        )
        metrics = antecedent.geometry_metrics(geometry, center)
        if not (
            metrics["finite_geometry"]
            and metrics["planarity_error_angstrom"]
            <= thresholds.maximum_planarity_error_angstrom
            and metrics["minimum_CC_bond_angstrom"]
            >= thresholds.minimum_CC_bond_angstrom
            and metrics["maximum_CC_bond_angstrom"]
            <= thresholds.maximum_CC_bond_angstrom
            and metrics["minimum_CH_bond_angstrom"]
            >= thresholds.minimum_CH_bond_angstrom
            and metrics["maximum_CH_bond_angstrom"]
            <= thresholds.maximum_CH_bond_angstrom
        ):
            raise RuntimeError(f"screen geometry selftest failed for {point.label}")

    def classification_probe(cone_max: float, smooth_max: float) -> str:
        records: List[Dict[str, Any]] = []
        for point in points:
            gap = 1.0e-5
            if point.theta_degrees == 0.0 and point.radius == 0.50:
                gap = cone_max if point.surface_kind == "cone_v1" else smooth_max
            records.append(
                {
                    "surface_kind": point.surface_kind,
                    "radius": point.radius,
                    "theta_degrees": point.theta_degrees,
                    "geometry_hash": point.geometry_hash,
                    "adiabatic_gap_Eh": gap,
                    "all_point_checks_pass": True,
                }
            )
        cone0 = _find_record(records, "cone_v1", 0.45, 0.0)
        cone36 = _find_record(records, "cone_v1", 0.45, 36.0)
        authorization = {
            "antecedent_controls": {
                "cone_r0p45_theta0": {
                    "geometry_hash": cone0["geometry_hash"],
                    "adiabatic_gap_Eh": cone0["adiabatic_gap_Eh"],
                },
                "cone_r0p45_theta36": {
                    "geometry_hash": cone36["geometry_hash"],
                    "adiabatic_gap_Eh": cone36["adiabatic_gap_Eh"],
                },
            }
        }
        return classify(records, authorization)[0]

    probes = {
        classification_probe(3.0e-4, 3.0e-4): (
            "PROMOTE_BOTH_SURFACES_TO_ANGULAR_ARBITRATION"
        ),
        classification_probe(1.0e-5, 3.0e-4): (
            "PROMOTE_SMOOTH_SURFACE_TO_ANGULAR_QUALIFICATION"
        ),
        classification_probe(3.0e-4, 1.0e-5): (
            "PROMOTE_CONE_SURFACE_TO_ANGULAR_QUALIFICATION"
        ),
        classification_probe(1.0e-5, 1.0e-5): (
            "REFINE_SPOKE_MAXIMUM_OR_PIVOT_PROJECTOR"
        ),
    }
    if any(observed != expected for observed, expected in probes.items()):
        raise RuntimeError(f"classification selftest failed: {probes}")
    print("CPDC_CENTRAL_GAP_LANDSCAPE_V1_SELFTEST=PASS")


def add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest")
    qualify = subparsers.add_parser("qualify")
    add_roots(qualify)
    run = subparsers.add_parser("run")
    add_roots(run)
    run.add_argument("--basis", default=BASIS_KEY)
    run.add_argument("--threads", type=int, default=6)
    run.add_argument("--memory-mb", type=int, default=100000)
    run.add_argument("--max-jobs", type=int, default=4)

    worker = subparsers.add_parser("point-worker")
    worker.add_argument("--target", type=Path, required=True)
    worker.add_argument("--surface-npz", type=Path, required=True)
    worker.add_argument("--surface-kind", choices=SURFACE_KINDS, required=True)
    worker.add_argument("--radius", type=float, required=True)
    worker.add_argument("--theta-degrees", type=float, required=True)
    worker.add_argument("--geometry-hash", required=True)
    worker.add_argument("--center-seed", type=Path, required=True)
    worker.add_argument("--basis", default=BASIS_KEY)
    worker.add_argument("--threads", type=int, default=6)
    worker.add_argument("--memory-mb", type=int, default=100000)
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
        raise SystemExit("--threads must be positive and --memory-mb at least 2000")
    configure_threads(args.threads)
    if args.command == "run":
        if args.max_jobs < 1:
            raise SystemExit("--max-jobs must be positive")
        run_screen(args)
    elif args.command == "point-worker":
        point_worker(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
