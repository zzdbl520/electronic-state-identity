#!/usr/bin/env python3
"""Preregistered relaxed-surface multiring topology pilot for singlet C5H5+.

The immutable source is a completed CPDC-SCI-CCPVTZ-LIFT-v1.0 calculation
with status PASS_CCPVTZ_SECONDARY_CI_LIFT.  Its D5h center and five relaxed
secondary-CI replicas define a two-dimensional Fourier-interpolated nuclear
surface.  The source points sit at u=0 and at the five unit-circle anchors.

The executable enforces three separate commands:

* ``qualify`` verifies and hash-freezes the source evidence;
* ``surface`` constructs the relaxed surface, performs independent SA3 rank
  arbitration, computes loop vertices, and freezes two fidelity-qualified
  meshes for every loop without evaluating a Berry product; and
* ``phase`` may run only after PASS_LOOP_GEOMETRY and exposes scalar Berry
  signs with mesh, orientation, cyclic-start, and random-gauge controls.

No command authorizes a manuscript claim.  A topology pass authorizes only a
separate opposed-orbital-history confirmation on the frozen polygons.
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

import cpdc_lift_engine as eng
import cpdc_overlap as ov


SCRIPT_VERSION = "CPDC-CCPVTZ-RELAXED-MULTIRING-v1.0"
PROTOCOL_ID = "CPDC-CCPVTZ-RELAXED-MULTIRING-20260920"
PROTOCOL_FILENAME = "CPDC_ccpVTZ_relaxed_multiring_protocol_v1.json"
MANIFEST_FILENAME = "CPDC_ccpVTZ_relaxed_multiring_v1_manifest.json"
SOURCE_AUTHORIZATION_FILENAME = "CPDC_ccpVTZ_multiring_source_authorization_v1.json"
SURFACE_NPZ_FILENAME = "CPDC_relaxed_Fourier_surface_v1.npz"
SURFACE_METADATA_FILENAME = "CPDC_relaxed_Fourier_surface_v1.json"
RANK_DECISION_FILENAME = "CPDC_ccpVTZ_SA3_rank_arbitration_v1.json"
LOOP_GEOMETRY_DECISION_FILENAME = "CPDC_ccpVTZ_loop_geometry_v1_decision.json"
TOPOLOGY_DECISION_FILENAME = "CPDC_ccpVTZ_relaxed_multiring_v1_decision.json"
FROZEN_SOURCE_SNAPSHOT_FILENAME = "CPDC_frozen_ccpVTZ_lift_decision_20260920.json"

SOURCE_DECISION_FILENAME = "CPDC_secondary_CI_ccpVTZ_lift_v1_decision.json"
SOURCE_SCRIPT_VERSION = "CPDC-SCI-CCPVTZ-LIFT-v1.0"
SOURCE_PROTOCOL_ID = "CPDC-SECONDARY-CI-CCPVTZ-LIFT-20260920"
SOURCE_REQUIRED_STATUS = "PASS_CCPVTZ_SECONDARY_CI_LIFT"
EXPECTED_SOURCE_CANONICAL_SHA256 = (
    "21f4cb12adaed4f6fe8576b4982c2a881f9df12a9a4b341833fac6efa8f564b9"
)
EXPECTED_ENGINE_SHA256 = (
    "7ff5d5dab945128e6031e61e2510d836ffd42d0a5e1f86078ef72e9b0ddcd325"
)

PYSCF_VERSION_EXACT = "2.13.1"
BASIS_KEY = "cc-pvtz"
N_RING = 5
N_ATOMS = 10
ELEMENTS = ("C",) * N_RING + ("H",) * N_RING
NCAS = 5
NELECAS = 4
NACT_ALPHA = 2
NACT_BETA = 2
NROOTS = 2
ARBITRATION_NROOTS = 3
PROBE_NROOTS = 3
CASSCF_CONV_TOL = 1.0e-10
CASSCF_GRAD_TOL = 1.0e-5

GLOBAL_VERTEX_LEVELS = (20, 40, 80)
LOCAL_VERTEX_LEVELS = (12, 24, 48)
INNER_RADIUS_U = 0.45
LOCAL_RADIUS_U = 0.08
OUTER_RADIUS_U = 1.45
GAUGE_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924)
POINT_ROLE = "relaxed_Fourier_surface_SA2_loop_vertex"


@dataclass(frozen=True)
class Thresholds:
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square_error: float = 1.0e-6
    maximum_mo_orthonormality_error: float = 1.0e-8
    maximum_casci_reproduction_error_Eh: float = 1.0e-7
    minimum_active_reference_sigma: float = 0.90
    minimum_third_singlet_gap_Eh: float = 2.0e-3
    minimum_center_third_singlet_gap_Eh: float = 3.0e-3
    maximum_center_pair_splitting_Eh: float = 2.0e-4
    arbitration_projector_sigma_minimum: float = 0.95
    arbitration_projector_leakage_maximum: float = 0.03
    arbitration_third_root_SA2_weight_maximum: float = 0.05
    maximum_average_energy_above_center_Eh: float = 0.030
    minimum_CC_bond_angstrom: float = 1.20
    maximum_CC_bond_angstrom: float = 1.70
    minimum_CH_bond_angstrom: float = 0.90
    maximum_CH_bond_angstrom: float = 1.25
    maximum_planarity_error_angstrom: float = 1.0e-10
    maximum_anchor_interpolation_error_angstrom: float = 2.0e-10
    pair_edge_sigma_minimum: float = 0.98
    pair_edge_leakage_maximum: float = 0.01
    active_edge_sigma_minimum: float = 0.98
    active_edge_leakage_maximum: float = 0.01
    minimum_assignment_margin: float = 0.10
    minimum_single_state_edge_overlap: float = 0.50
    minimum_loop_adiabatic_gap_Eh: float = 2.0e-4
    minimum_phase_real_magnitude: float = 0.90
    maximum_phase_imaginary_abs: float = 1.0e-8
    maximum_gauge_error: float = 1.0e-10
    maximum_orientation_error: float = 1.0e-10
    maximum_cyclic_start_error: float = 1.0e-10
    maximum_mesh_phase_difference: float = 1.0e-8


@dataclass(frozen=True)
class LoopSpec:
    name: str
    center_ux: float
    center_uy: float
    radius_u: float
    vertex_levels: Tuple[int, ...]
    expected_sign: int
    topological_role: str


@dataclass(frozen=True)
class PointSpec:
    target: Path
    ux: float
    uy: float
    geometry_hash: str
    label: str
    descriptor: Dict[str, Any]


def _jsonable(value: Any) -> Any:
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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_threads(threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(int(threads))
    try:
        from pyscf import lib

        lib.num_threads(int(threads))
    except ImportError:
        pass


def require_pyscf():
    modules = eng.require_pyscf()
    installed = str(__import__("pyscf").__version__)
    if installed != PYSCF_VERSION_EXACT:
        raise SystemExit(
            f"This protocol requires exact PySCF {PYSCF_VERSION_EXACT}; found {installed}"
        )
    return modules


def protocol_path() -> Path:
    return Path(__file__).resolve().with_name(PROTOCOL_FILENAME)


def frozen_snapshot_path() -> Path:
    return Path(__file__).resolve().with_name(FROZEN_SOURCE_SNAPSHOT_FILENAME)


def source_authorization_path(root: Path) -> Path:
    return root / SOURCE_AUTHORIZATION_FILENAME


def surface_npz_path(root: Path) -> Path:
    return root / SURFACE_NPZ_FILENAME


def surface_metadata_path(root: Path) -> Path:
    return root / SURFACE_METADATA_FILENAME


def rank_decision_path(root: Path) -> Path:
    return root / RANK_DECISION_FILENAME


def loop_geometry_decision_path(root: Path) -> Path:
    return root / LOOP_GEOMETRY_DECISION_FILENAME


def topology_decision_path(root: Path) -> Path:
    return root / TOPOLOGY_DECISION_FILENAME


def _all_true(mapping: Any) -> bool:
    return bool(
        isinstance(mapping, Mapping)
        and mapping
        and all(value is True for value in mapping.values())
    )


def _resolve_source_artifact(source_root: Path, recorded: Any) -> Path:
    source_root = source_root.resolve()
    raw = Path(str(recorded))
    candidates = [raw] if raw.is_absolute() else [source_root / raw]
    for marker in (
        "center_ccpvtz",
        "evaluations",
        "replicas",
    ):
        if marker in raw.parts:
            index = raw.parts.index(marker)
            candidates.append(source_root.joinpath(*raw.parts[index:]))
    if raw.name in {
        SOURCE_DECISION_FILENAME,
        "CPDC_secondary_CI_ccpVTZ_center_v1_decision.json",
        "CPDC_secondary_CI_ccpVTZ_source_authorization_v1.json",
        "CPDC_secondary_CI_ccpVTZ_search_record_v1.json",
    }:
        candidates.append(source_root / raw.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_relative_to(source_root):
            return resolved
    raise RuntimeError(
        f"source artifact is missing or outside --source-root: {recorded}"
    )


def _source_solution_checks(
    solution: Mapping[str, Any], role: str, expected_gradients: bool
) -> Dict[str, bool]:
    metadata = solution["metadata"]
    return {
        f"{role}_script_version": metadata.get("script_version")
        == SOURCE_SCRIPT_VERSION,
        f"{role}_protocol_id": metadata.get("protocol_id") == SOURCE_PROTOCOL_ID,
        f"{role}_basis": str(metadata.get("basis", "")).lower() == BASIS_KEY,
        f"{role}_basis_declaration": metadata.get("basis_definition_sha256")
        == eng.BASIS_DEFINITION_SHA256,
        f"{role}_pyscf_version": metadata.get("pyscf_version")
        == PYSCF_VERSION_EXACT,
        f"{role}_active_space": bool(
            int(metadata.get("ncas", -1)) == NCAS
            and int(metadata.get("nelecas", -1)) == NELECAS
            and int(metadata.get("nroots", -1)) == NROOTS
        ),
        f"{role}_gradient_semantics": bool(
            bool(metadata.get("state_gradients_computed"))
            == bool(expected_gradients)
        ),
    }


def build_source_authorization(root: Path, source_root: Path) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    decision_path = source_root / SOURCE_DECISION_FILENAME
    if not decision_path.exists():
        raise RuntimeError(f"missing source decision: {decision_path}")
    source_decision = json.loads(decision_path.read_text())
    snapshot = json.loads(frozen_snapshot_path().read_text())
    source_canonical_hash = canonical_json_sha256(source_decision)
    snapshot_canonical_hash = canonical_json_sha256(snapshot)
    checks: Dict[str, bool] = {
        "frozen_snapshot_fingerprint": snapshot_canonical_hash
        == EXPECTED_SOURCE_CANONICAL_SHA256,
        "source_matches_frozen_snapshot": source_canonical_hash
        == EXPECTED_SOURCE_CANONICAL_SHA256,
        "source_script_version": source_decision.get("script_version")
        == SOURCE_SCRIPT_VERSION,
        "source_protocol_id": source_decision.get("protocol_id")
        == SOURCE_PROTOCOL_ID,
        "source_status": source_decision.get("status") == SOURCE_REQUIRED_STATUS,
        "source_authorization_status": source_decision.get(
            "source_authorization_status"
        )
        == "PASS_SOURCE_AUTHORIZATION",
        "source_signal_blind": source_decision.get("signal_inspected") is False,
        "source_phase_or_loop_not_calculated": source_decision.get(
            "phase_or_loop_calculated"
        )
        is False,
        "source_manuscript_claim_not_authorized": source_decision.get(
            "manuscript_claim_authorized"
        )
        is False,
        "source_candidate_checks": _all_true(source_decision.get("candidate_checks")),
        "source_replica_checks": _all_true(
            source_decision.get("replicas", {}).get("checks")
        ),
    }

    artifact_error: str | None = None
    center_decision_path: Path | None = None
    center_solution_path: Path | None = None
    candidate_solution_path: Path | None = None
    source_authorization_file: Path | None = None
    replica_integrity: List[Dict[str, Any]] = []
    center_average_energy = float("nan")
    try:
        source_authorization_file = _resolve_source_artifact(
            source_root, source_decision.get("source_authorization_file", "")
        )
        checks["source_authorization_hash"] = bool(
            sha256_file(source_authorization_file)
            == source_decision.get("source_authorization_sha256")
        )

        center_decision_path = _resolve_source_artifact(
            source_root, source_decision.get("center_decision_file", "")
        )
        center_decision = json.loads(center_decision_path.read_text())
        checks["center_decision_passed"] = center_decision.get("status") == "PASS_CENTER"
        center_solution_path = _resolve_source_artifact(
            source_root, center_decision.get("solution_file", "")
        )
        checks["center_solution_hash"] = bool(
            sha256_file(center_solution_path)
            == center_decision.get("solution_sha256")
        )
        center_solution = eng.load_solution(center_solution_path)
        checks.update(
            _source_solution_checks(center_solution, "center", expected_gradients=False)
        )
        center_geometry = np.asarray(center_solution["geometry_angstrom"], dtype=float)
        checks["center_geometry_is_locked_D5h"] = bool(
            center_geometry.shape == (N_ATOMS, 3)
            and np.max(np.abs(center_geometry - eng.REFERENCE_GEOMETRY)) <= 2.0e-10
        )
        center_average_energy = float(
            center_solution["metadata"].get("average_energy_Eh", float("nan"))
        )

        coefficients = np.asarray(
            source_decision.get("candidate_coefficients_angstrom", []), dtype=float
        )
        checks["candidate_has_nine_finite_coordinates"] = bool(
            coefficients.shape == (eng.N_COORDS,) and np.all(np.isfinite(coefficients))
        )
        candidate_solution_path = _resolve_source_artifact(
            source_root, source_decision.get("candidate_solution_file", "")
        )
        candidate_solution = eng.load_solution(candidate_solution_path)
        checks.update(
            _source_solution_checks(
                candidate_solution, "candidate", expected_gradients=True
            )
        )
        candidate_geometry = np.asarray(
            candidate_solution["geometry_angstrom"], dtype=float
        )
        checks["candidate_geometry_reconstructed"] = bool(
            coefficients.shape == (eng.N_COORDS,)
            and np.max(
                np.abs(candidate_geometry - eng.geometry_from_coefficients(coefficients))
            )
            <= 2.0e-10
        )
        checks["candidate_gap_matches_decision"] = bool(
            abs(
                float(candidate_solution["metadata"]["adiabatic_gap_Eh"])
                - float(source_decision.get("candidate_gap_Eh", float("nan")))
            )
            <= 1.0e-10
        )

        records = source_decision.get("replicas", {}).get("records", [])
        checks["five_replica_records"] = bool(
            isinstance(records, list) and len(records) == N_RING
        )
        seen_hashes = set()
        gaps: List[float] = []
        energies: List[float] = []
        for replica, record in enumerate(records):
            path = _resolve_source_artifact(
                source_root, record.get("solution_file", "")
            )
            solution = eng.load_solution(path)
            checks.update(
                _source_solution_checks(
                    solution, f"replica_{replica}", expected_gradients=False
                )
            )
            geometry = np.asarray(solution["geometry_angstrom"], dtype=float)
            expected_geometry = eng.symmetry_replica(candidate_geometry, replica)
            checks[f"replica_{replica}_geometry"] = bool(
                np.max(np.abs(geometry - expected_geometry)) <= 2.0e-10
            )
            gap = float(solution["metadata"]["adiabatic_gap_Eh"])
            energy = float(solution["metadata"]["average_energy_Eh"])
            checks[f"replica_{replica}_gap"] = bool(
                abs(gap - float(record.get("adiabatic_gap_Eh", float("nan"))))
                <= 1.0e-10
            )
            geometry_hash = eng.array_hash(geometry)
            seen_hashes.add(geometry_hash)
            gaps.append(gap)
            energies.append(energy)
            replica_integrity.append(
                {
                    "replica": replica,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "geometry_hash": geometry_hash,
                    "gap_Eh": gap,
                    "average_energy_Eh": energy,
                }
            )
        checks["five_distinct_replica_geometries"] = len(seen_hashes) == N_RING
        checks["replica_gap_recomputed"] = bool(
            gaps
            and abs(
                max(gaps)
                - float(
                    source_decision.get("replicas", {}).get(
                        "maximum_gap_Eh", float("nan")
                    )
                )
            )
            <= 1.0e-10
        )
        checks["replica_energy_spread_recomputed"] = bool(
            energies
            and abs(
                (max(energies) - min(energies))
                - float(
                    source_decision.get("replicas", {}).get(
                        "average_energy_spread_Eh", float("nan")
                    )
                )
            )
            <= 1.0e-10
        )
    except Exception as exc:
        checks["source_artifact_integrity"] = False
        artifact_error = f"{type(exc).__name__}: {exc}"
    else:
        checks["source_artifact_integrity"] = True

    status = (
        "PASS_SOURCE_AUTHORIZATION"
        if all(checks.values())
        else "STOP_SOURCE_AUTHORIZATION_FAILURE"
    )
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "source_decision_file": str(decision_path.resolve()),
        "source_decision_sha256": sha256_file(decision_path),
        "source_decision_canonical_sha256": source_canonical_hash,
        "frozen_snapshot_sha256": sha256_file(frozen_snapshot_path()),
        "status": status,
        "surface_or_loop_constructed": False,
        "phase_inspected": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "artifact_error": artifact_error,
        "lift_source_authorization_file": (
            str(source_authorization_file) if source_authorization_file else None
        ),
        "lift_source_authorization_sha256": (
            sha256_file(source_authorization_file)
            if source_authorization_file
            else None
        ),
        "center_decision_file": str(center_decision_path) if center_decision_path else None,
        "center_decision_sha256": (
            sha256_file(center_decision_path) if center_decision_path else None
        ),
        "center_solution_file": str(center_solution_path) if center_solution_path else None,
        "center_solution_sha256": (
            sha256_file(center_solution_path) if center_solution_path else None
        ),
        "center_average_energy_Eh": center_average_energy,
        "candidate_solution_file": (
            str(candidate_solution_path) if candidate_solution_path else None
        ),
        "candidate_solution_sha256": (
            sha256_file(candidate_solution_path) if candidate_solution_path else None
        ),
        "candidate_coefficients_angstrom": source_decision.get(
            "candidate_coefficients_angstrom"
        ),
        "replica_solution_integrity": replica_integrity,
    }


def validate_source_authorization(
    root: Path, source_root: Path, require_pass: bool = True
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    path = source_authorization_path(root)
    if not path.exists():
        if require_pass:
            raise SystemExit(f"Missing {path}; run qualify first")
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "phase_inspected": False,
        "manuscript_claim_authorized": False,
    }
    if {key: record.get(key) for key in expected} != expected:
        raise RuntimeError("source authorization belongs to another immutable run")
    decision_path = source_root / SOURCE_DECISION_FILENAME
    if not decision_path.exists():
        raise RuntimeError("source decision disappeared after authorization")
    decision = json.loads(decision_path.read_text())
    if canonical_json_sha256(decision) != record.get(
        "source_decision_canonical_sha256"
    ):
        raise RuntimeError("source decision changed after authorization")
    integrity = [
        {
            "path": record.get("lift_source_authorization_file"),
            "sha256": record.get("lift_source_authorization_sha256"),
        },
        {
            "path": record.get("center_decision_file"),
            "sha256": record.get("center_decision_sha256"),
        },
        {
            "path": record.get("center_solution_file"),
            "sha256": record.get("center_solution_sha256"),
        },
        {
            "path": record.get("candidate_solution_file"),
            "sha256": record.get("candidate_solution_sha256"),
        },
    ] + list(record.get("replica_solution_integrity", []))
    for item in integrity:
        artifact = Path(str(item.get("path", ""))).resolve()
        if (
            not artifact.exists()
            or not artifact.is_relative_to(source_root)
            or sha256_file(artifact) != item.get("sha256")
        ):
            raise RuntimeError(f"source artifact changed: {artifact}")
    if require_pass and record.get("status") != "PASS_SOURCE_AUTHORIZATION":
        raise SystemExit(
            f"Source authorization is {record.get('status', 'UNKNOWN')}; stop"
        )
    return record


def run_qualify(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol_declaration()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = source_authorization_path(root)
    if path.exists():
        result = validate_source_authorization(
            root, args.source_root, require_pass=False
        )
    else:
        try:
            result = build_source_authorization(root, args.source_root)
        except Exception as exc:
            result = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "root": str(root),
                "source_root": str(args.source_root.resolve()),
                "status": "STOP_SOURCE_AUTHORIZATION_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}",
                "surface_or_loop_constructed": False,
                "phase_inspected": False,
                "manuscript_claim_authorized": False,
            }
        write_json(path, result)
    print("\nCPDC RELAXED-MULTIRING SOURCE AUTHORIZATION", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Decision file: {path}", flush=True)
    print(f"CPDC_MULTI_SOURCE={result['status']}", flush=True)
    return result


def validate_protocol_declaration() -> Dict[str, Any]:
    path = protocol_path()
    if not path.exists():
        raise RuntimeError(f"missing protocol declaration: {path}")
    declaration = json.loads(path.read_text())
    model = declaration.get("locked_model", {})
    surface = declaration.get("relaxed_surface", {})
    loops = declaration.get("loops", {})
    expected = {
        "protocol_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_canonical_sha256": EXPECTED_SOURCE_CANONICAL_SHA256,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "basis": BASIS_KEY,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "production_nroots": NROOTS,
        "arbitration_nroots": ARBITRATION_NROOTS,
        "anchor_count": N_RING,
        "interpolation": "five-point real discrete Fourier interpolation",
        "inner_radius_u": INNER_RADIUS_U,
        "local_radius_u": LOCAL_RADIUS_U,
        "outer_radius_u": OUTER_RADIUS_U,
        "global_vertex_levels": list(GLOBAL_VERTEX_LEVELS),
        "local_vertex_levels": list(LOCAL_VERTEX_LEVELS),
        "gauge_seeds": list(GAUGE_SEEDS),
    }
    actual = {
        "protocol_version": declaration.get("protocol_version"),
        "protocol_id": declaration.get("protocol_id"),
        "source_canonical_sha256": declaration.get("source", {}).get(
            "decision_canonical_sha256"
        ),
        "pyscf_version_exact": model.get("pyscf_version_exact"),
        "basis": str(model.get("basis", "")).lower(),
        "ncas": model.get("ncas"),
        "nelecas": model.get("nelecas"),
        "production_nroots": model.get("production_nroots"),
        "arbitration_nroots": model.get("arbitration_nroots"),
        "anchor_count": surface.get("anchor_count"),
        "interpolation": surface.get("interpolation"),
        "inner_radius_u": loops.get("inner_radius_u"),
        "local_radius_u": loops.get("local_radius_u"),
        "outer_radius_u": loops.get("outer_radius_u"),
        "global_vertex_levels": loops.get("global_vertex_levels"),
        "local_vertex_levels": loops.get("local_vertex_levels"),
        "gauge_seeds": declaration.get("phase_controls", {}).get(
            "random_gauge_seeds"
        ),
    }
    if actual != expected:
        raise RuntimeError(
            "protocol declaration disagrees with executable constants: "
            f"expected={expected}, actual={actual}"
        )
    if declaration.get("thresholds") != asdict(Thresholds()):
        raise RuntimeError("protocol thresholds disagree with executable")
    policy = declaration.get("publication_policy", {})
    if bool(policy.get("manuscript_claim_authorized")):
        raise RuntimeError("pilot protocol may not authorize a manuscript claim")
    return declaration


def build_manifest(root: Path, basis: str) -> Dict[str, Any]:
    source_auth = source_authorization_path(root)
    if not source_auth.exists():
        raise RuntimeError("source authorization must precede manifest creation")
    files = {
        "driver": Path(__file__).resolve(),
        "engine": Path(eng.__file__).resolve(),
        "overlap_module": Path(ov.__file__).resolve(),
        "protocol": protocol_path(),
        "frozen_source_snapshot": frozen_snapshot_path(),
    }
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root.resolve()),
        "basis": basis.lower(),
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "immutable_files": {
            label: {"name": path.name, "sha256": sha256_file(path)}
            for label, path in files.items()
        },
        "source_authorization": {
            "name": source_auth.name,
            "sha256": sha256_file(source_auth),
        },
        "phase_embargo": (
            "phase command is forbidden until PASS_LOOP_GEOMETRY exists on this root"
        ),
        "created_unix": time.time(),
    }


def create_or_validate_manifest(root: Path, basis: str) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILENAME
    current = build_manifest(root, basis)
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
        "source_authorization",
        "phase_embargo",
    ):
        if stored.get(key) != current.get(key):
            raise RuntimeError(f"manifest mismatch for {key}; use a new --root")
    return stored


def _fourier_coefficients(anchor_displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(anchor_displacements, dtype=float)
    if values.shape != (N_RING, N_ATOMS, 3):
        raise ValueError(f"unexpected anchor shape: {values.shape}")
    frequencies = np.asarray([0, 1, 2, -2, -1], dtype=int)
    coefficients = np.fft.fft(values, axis=0) / float(N_RING)
    return frequencies, coefficients


def _surface_displacement_from_arrays(
    theta: float, frequencies: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    phase = np.exp(1j * np.asarray(frequencies, dtype=float) * float(theta))
    displacement = np.tensordot(phase, coefficients, axes=(0, 0))
    imaginary_error = float(np.max(np.abs(np.imag(displacement))))
    if imaginary_error > 5.0e-11:
        raise RuntimeError(
            f"Fourier surface lost reality: imaginary error={imaginary_error:.3e}"
        )
    return np.asarray(np.real(displacement), dtype=float)


def load_surface(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            "metadata": json.loads(str(data["metadata_json"].item())),
            "center_geometry_angstrom": np.asarray(data["center_geometry_angstrom"]),
            "anchor_geometries_angstrom": np.asarray(data["anchor_geometries_angstrom"]),
            "frequencies": np.asarray(data["frequencies"], dtype=int),
            "fourier_coefficients": (
                np.asarray(data["fourier_coefficients_real"])
                + 1j * np.asarray(data["fourier_coefficients_imag"])
            ),
        }


def surface_geometry(surface: Mapping[str, Any], ux: float, uy: float) -> np.ndarray:
    radius = math.hypot(float(ux), float(uy))
    center = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    if radius < 1.0e-15:
        return center.copy()
    theta = math.atan2(float(uy), float(ux))
    displacement = _surface_displacement_from_arrays(
        theta,
        np.asarray(surface["frequencies"], dtype=int),
        np.asarray(surface["fourier_coefficients"]),
    )
    return center + radius * displacement


def geometry_metrics(
    geometry: np.ndarray, center_geometry: np.ndarray
) -> Dict[str, Any]:
    geometry = np.asarray(geometry, dtype=float)
    cc = [
        float(np.linalg.norm(geometry[index] - geometry[(index + 1) % N_RING]))
        for index in range(N_RING)
    ]
    ch = [
        float(np.linalg.norm(geometry[index] - geometry[N_RING + index]))
        for index in range(N_RING)
    ]
    return {
        "finite_geometry": bool(np.all(np.isfinite(geometry))),
        "planarity_error_angstrom": float(np.max(np.abs(geometry[:, 2]))),
        "CC_bonds_angstrom": cc,
        "minimum_CC_bond_angstrom": min(cc),
        "maximum_CC_bond_angstrom": max(cc),
        "CH_bonds_angstrom": ch,
        "minimum_CH_bond_angstrom": min(ch),
        "maximum_CH_bond_angstrom": max(ch),
        "maximum_atom_displacement_from_center_angstrom": float(
            np.max(np.linalg.norm(geometry - center_geometry, axis=1))
        ),
    }


def build_or_validate_surface(
    root: Path, source_authorization: Mapping[str, Any]
) -> Dict[str, Any]:
    npz_path = surface_npz_path(root)
    metadata_path = surface_metadata_path(root)
    if npz_path.exists() or metadata_path.exists():
        if not (npz_path.exists() and metadata_path.exists()):
            raise RuntimeError("partial surface definition exists; use a new root")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("surface_npz_sha256") != sha256_file(npz_path):
            raise RuntimeError("surface NPZ hash mismatch")
        surface = load_surface(npz_path)
        if surface["metadata"].get("definition_sha256") != metadata.get(
            "definition_sha256"
        ):
            raise RuntimeError("surface metadata mismatch")
        return metadata

    center_solution = eng.load_solution(
        Path(str(source_authorization["center_solution_file"]))
    )
    center_geometry = np.asarray(center_solution["geometry_angstrom"], dtype=float)
    anchors = []
    for item in source_authorization["replica_solution_integrity"]:
        solution = eng.load_solution(Path(str(item["path"])))
        anchors.append(np.asarray(solution["geometry_angstrom"], dtype=float))
    anchor_geometries = np.asarray(anchors, dtype=float)
    displacements = anchor_geometries - center_geometry[None, :, :]
    frequencies, coefficients = _fourier_coefficients(displacements)
    interpolation_errors = []
    for replica in range(N_RING):
        theta = 2.0 * math.pi * replica / N_RING
        reconstructed = center_geometry + _surface_displacement_from_arrays(
            theta, frequencies, coefficients
        )
        interpolation_errors.append(
            float(np.max(np.abs(reconstructed - anchor_geometries[replica])))
        )
    maximum_error = max(interpolation_errors)
    if maximum_error > Thresholds().maximum_anchor_interpolation_error_angstrom:
        raise RuntimeError(
            f"Fourier surface misses an anchor by {maximum_error:.3e} Angstrom"
        )

    definition_payload = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "coordinate_domain": "u=(ux,uy) in a dimensionless parameter plane",
        "map": "R(u)=R_center+|u|*D_Fourier(arg(u))",
        "frequencies": frequencies.tolist(),
        "anchor_parameter_points": [
            [
                math.cos(2.0 * math.pi * replica / N_RING),
                math.sin(2.0 * math.pi * replica / N_RING),
            ]
            for replica in range(N_RING)
        ],
        "center_geometry_hash": eng.array_hash(center_geometry),
        "anchor_geometry_hashes": [
            eng.array_hash(geometry) for geometry in anchor_geometries
        ],
        "maximum_anchor_interpolation_error_angstrom": maximum_error,
        "phase_inspected": False,
    }
    definition_sha256 = canonical_json_sha256(definition_payload)
    npz_metadata = {**definition_payload, "definition_sha256": definition_sha256}
    temporary = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(_jsonable(npz_metadata))),
        center_geometry_angstrom=center_geometry,
        anchor_geometries_angstrom=anchor_geometries,
        frequencies=frequencies,
        fourier_coefficients_real=np.real(coefficients),
        fourier_coefficients_imag=np.imag(coefficients),
    )
    os.replace(temporary, npz_path)
    metadata = {
        **npz_metadata,
        "surface_npz_file": str(npz_path.resolve()),
        "surface_npz_sha256": sha256_file(npz_path),
        "center_displacement_norms_angstrom": [
            float(np.linalg.norm(displacement)) for displacement in displacements
        ],
    }
    write_json(metadata_path, metadata)
    return metadata


def build_loop_specs() -> List[LoopSpec]:
    if not (0.0 < INNER_RADIUS_U < 1.0 - LOCAL_RADIUS_U):
        raise RuntimeError("inner loop is not separated from satellite loops")
    if not (OUTER_RADIUS_U > 1.0 + LOCAL_RADIUS_U):
        raise RuntimeError("outer loop does not enclose satellite loops")
    loops = [
        LoopSpec(
            name="central_inner",
            center_ux=0.0,
            center_uy=0.0,
            radius_u=INNER_RADIUS_U,
            vertex_levels=GLOBAL_VERTEX_LEVELS,
            expected_sign=-1,
            topological_role="local control around the central D5h intersection",
        )
    ]
    for replica in range(N_RING):
        theta = 2.0 * math.pi * replica / N_RING
        loops.append(
            LoopSpec(
                name=f"satellite_{replica:02d}",
                center_ux=math.cos(theta),
                center_uy=math.sin(theta),
                radius_u=LOCAL_RADIUS_U,
                vertex_levels=LOCAL_VERTEX_LEVELS,
                expected_sign=-1,
                topological_role=f"local control around relaxed secondary CI {replica}",
            )
        )
    loops.append(
        LoopSpec(
            name="outer_all_six",
            center_ux=0.0,
            center_uy=0.0,
            radius_u=OUTER_RADIUS_U,
            vertex_levels=GLOBAL_VERTEX_LEVELS,
            expected_sign=1,
            topological_role="single loop around the central plus five secondary CIs",
        )
    )
    return loops


def arbitration_solution_path(root: Path) -> Path:
    return root / "rank_arbitration" / "sa3_solution.npz"


def validate_existing_sa3(target: Path, geometry: np.ndarray, basis: str) -> Dict[str, Any]:
    solution = eng.load_solution(target)
    metadata = solution["metadata"]
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": "independent_D5h_SA3_rank_arbitration",
        "basis": basis.lower(),
        "pyscf_version": PYSCF_VERSION_EXACT,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": ARBITRATION_NROOTS,
    }
    actual = {
        "script_version": metadata.get("script_version"),
        "protocol_id": metadata.get("protocol_id"),
        "protocol_role": metadata.get("protocol_role"),
        "basis": str(metadata.get("basis", "")).lower(),
        "pyscf_version": metadata.get("pyscf_version"),
        "ncas": metadata.get("ncas"),
        "nelecas": metadata.get("nelecas"),
        "nroots": metadata.get("nroots"),
    }
    if actual != expected:
        raise RuntimeError(f"existing SA3 metadata mismatch: {actual} != {expected}")
    if np.max(np.abs(solution["geometry_angstrom"] - geometry)) > 1.0e-11:
        raise RuntimeError("existing SA3 geometry mismatch")
    return metadata


def run_sa3_solution(
    target: Path, geometry: np.ndarray, basis: str, memory_mb: int
) -> Dict[str, Any]:
    if target.exists():
        return validate_existing_sa3(target, geometry, basis)
    fci, _, mcscf, _ = require_pyscf()
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.parent / "casscf.log"
    mol = eng.make_mol(geometry, basis, memory_mb, str(log_path), verbose=4)
    mf = eng.run_rhf(mol)
    mo_guess, selection = eng.build_pi_initial_mos(mf)
    ncore_expected = (mol.nelectron - NELECAS) // 2
    reference_active = mo_guess[:, ncore_expected : ncore_expected + NCAS].copy()

    mc = mcscf.CASSCF(mf, NCAS, (NACT_ALPHA, NACT_BETA))
    mc.fcisolver = eng.make_singlet_solver(mol, ARBITRATION_NROOTS)
    mc = mc.state_average_([1.0 / ARBITRATION_NROOTS] * ARBITRATION_NROOTS)
    mc.conv_tol = CASSCF_CONV_TOL
    mc.conv_tol_grad = CASSCF_GRAD_TOL
    mc.max_cycle_macro = 160
    mc.max_cycle_micro = 40
    mc.canonicalization = False
    mc.kernel(mo_guess)
    primary_converged = bool(mc.converged)
    fallback_attempts = 0
    while not bool(mc.converged) and fallback_attempts < 2:
        mc = mc.newton()
        mc.conv_tol = CASSCF_CONV_TOL
        mc.conv_tol_grad = CASSCF_GRAD_TOL
        mc.max_cycle_macro = 160
        mc.max_cycle_micro = 40
        mc.canonicalization = False
        mc.kernel(mc.mo_coeff)
        fallback_attempts += 1

    energies = np.asarray(mc.e_states, dtype=float).reshape(-1)
    ci_roots = [np.asarray(root) for root in mc.ci]
    if energies.size != ARBITRATION_NROOTS or len(ci_roots) != ARBITRATION_NROOTS:
        raise RuntimeError("SA3-CASSCF returned an unexpected root count")
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
        raise RuntimeError(f"unexpected SA3 ncore={ncore}; expected {ncore_expected}")
    probe_energies, _ = eng.casci_probe(mf, final_mo, PROBE_NROOTS)
    reproduction_error = float(np.max(np.abs(probe_energies - energies)))
    spin_square = [
        float(
            fci.spin_op.spin_square0(
                root, NCAS, (NACT_ALPHA, NACT_BETA)
            )[0]
        )
        for root in ci_roots
    ]
    metric = mol.intor_symmetric("int1e_ovlp")
    mo_error = float(
        np.max(
            np.abs(
                final_mo.conj().T @ metric @ final_mo
                - np.eye(final_mo.shape[1])
            )
        )
    )
    final_active = final_mo[:, ncore : ncore + NCAS]
    active_sv = np.linalg.svd(
        reference_active.conj().T @ metric @ final_active, compute_uv=False
    )
    metadata: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": "independent_D5h_SA3_rank_arbitration",
        "molecule": "cyclopentadienyl cation",
        "formula": "C5H5+",
        "target": "lowest singlet manifold only",
        "absolute_ground_state_claimed": False,
        "basis": basis.lower(),
        "basis_definition_sha256": eng.BASIS_DEFINITION_SHA256,
        "method": "equal-weight SA3-CASSCF(4e,5o) full-valence-pi singlet",
        "charge": 1,
        "spin_2S": 0,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nact_alpha": NACT_ALPHA,
        "nact_beta": NACT_BETA,
        "nroots": ARBITRATION_NROOTS,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "state_average_weights": [1.0 / ARBITRATION_NROOTS] * ARBITRATION_NROOTS,
        "geometry_hash": eng.array_hash(geometry),
        "converged": bool(mc.converged),
        "primary_converged": primary_converged,
        "fallback_attempts": fallback_attempts,
        "average_energy_Eh": float(np.mean(energies)),
        "state_energies_Eh": energies,
        "adiabatic_gap_Eh": float(energies[1] - energies[0]),
        "third_singlet_gap_Eh": float(energies[2] - energies[1]),
        "maximum_orbital_gradient": maximum_orbital_gradient,
        "state_S2": spin_square,
        "maximum_spin_square_error": max(abs(value) for value in spin_square),
        "mo_orthonormality_max_error": mo_error,
        "CASCI_probe_energies_Eh": probe_energies,
        "CASCI_reproduction_max_error_Eh": reproduction_error,
        "active_reference_singular_values": active_sv,
        "active_reference_sigma_min": float(np.min(active_sv)),
        "active_orbital_selection": selection,
        "pyscf_version": __import__("pyscf").__version__,
        "python_version": platform.python_version(),
        "timestamp_unix": time.time(),
    }
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata))),
        "geometry_angstrom": np.asarray(geometry),
        "mo_coeff": final_mo,
        "state_energies_Eh": energies,
        "CASCI_probe_energies_Eh": probe_energies,
    }
    for index, root_ci in enumerate(ci_roots):
        arrays[f"ci_root_{index}"] = root_ci
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return metadata


def run_rank_arbitration(
    args: argparse.Namespace,
    root: Path,
    source_authorization: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    output = rank_decision_path(root)
    target = arbitration_solution_path(root)
    if output.exists():
        result = json.loads(output.read_text())
        if (
            result.get("script_version") != SCRIPT_VERSION
            or result.get("protocol_id") != PROTOCOL_ID
            or not target.exists()
            or result.get("SA3_solution_sha256") != sha256_file(target)
        ):
            raise RuntimeError("existing SA3 arbitration failed integrity validation")
        return result

    center_path = Path(str(source_authorization["center_solution_file"]))
    center = eng.load_solution(center_path)
    center_geometry = np.asarray(center["geometry_angstrom"], dtype=float)
    metadata = run_sa3_solution(target, center_geometry, args.basis, args.memory_mb)
    arbitration = eng.load_solution(target)
    mol_sa2 = eng.make_mol(center_geometry, args.basis, args.memory_mb, verbose=0)
    mol_sa3 = eng.make_mol(center_geometry, args.basis, args.memory_mb, verbose=0)
    overlap = ov.exact_overlap_matrix(center, arbitration, mol_sa2, mol_sa3)
    if overlap.shape != (NROOTS, ARBITRATION_NROOTS):
        raise RuntimeError(f"unexpected SA2/SA3 overlap shape: {overlap.shape}")
    pair_block = overlap[:, :NROOTS]
    singular_values = np.linalg.svd(pair_block, compute_uv=False)
    leakage = float(1.0 - np.mean(singular_values**2))
    third_weight = float(np.sum(np.abs(overlap[:, 2]) ** 2))
    checks = {
        "SA2_source_center_qualified": True,
        "SA3_converged_and_stationary": bool(
            metadata.get("converged")
            and float(metadata.get("maximum_orbital_gradient", float("inf")))
            <= thresholds.maximum_final_orbital_gradient
        ),
        "SA3_singlet_spin_pure": bool(
            float(metadata.get("maximum_spin_square_error", float("inf")))
            <= thresholds.maximum_spin_square_error
        ),
        "SA3_MO_frame_orthonormal": bool(
            float(metadata.get("mo_orthonormality_max_error", float("inf")))
            <= thresholds.maximum_mo_orthonormality_error
        ),
        "SA3_CASCI_reproduces_states": bool(
            float(metadata.get("CASCI_reproduction_max_error_Eh", float("inf")))
            <= thresholds.maximum_casci_reproduction_error_Eh
        ),
        "SA3_active_reference_retained": bool(
            float(metadata.get("active_reference_sigma_min", -float("inf")))
            >= thresholds.minimum_active_reference_sigma
        ),
        "SA2_subspace_retained_in_SA3": bool(
            float(np.min(singular_values))
            >= thresholds.arbitration_projector_sigma_minimum
            and leakage <= thresholds.arbitration_projector_leakage_maximum
        ),
        "SA3_third_root_excluded_from_SA2": bool(
            third_weight <= thresholds.arbitration_third_root_SA2_weight_maximum
        ),
        "SA3_third_singlet_isolated": bool(
            float(metadata.get("third_singlet_gap_Eh", -float("inf")))
            >= thresholds.minimum_center_third_singlet_gap_Eh
        ),
        "SA3_center_pair_preserved": bool(
            abs(float(metadata.get("adiabatic_gap_Eh", float("inf"))))
            <= thresholds.maximum_center_pair_splitting_Eh
        ),
    }
    status = "PASS_SA3_ARBITRATION" if all(checks.values()) else "STOP_SA3_ARBITRATION_FAILURE"
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "gate": "independent D5h SA3 singlet-rank arbitration",
        "status": status,
        "phase_inspected": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "SA2_source_solution": str(center_path),
        "SA2_source_solution_sha256": sha256_file(center_path),
        "SA3_solution_file": str(target.resolve()),
        "SA3_solution_sha256": sha256_file(target),
        "SA2_SA3_overlap_magnitude_squared": np.abs(overlap) ** 2,
        "projector_singular_values": singular_values,
        "projector_sigma_min": float(np.min(singular_values)),
        "projector_leakage": leakage,
        "third_root_SA2_weight": third_weight,
        "SA3_metadata": metadata,
    }
    write_json(output, result)
    print("\nCPDC cc-pVTZ SA3 RANK ARBITRATION", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"sigma={result['projector_sigma_min']:.9f}; "
        f"leakage={leakage:.3e}; third_weight={third_weight:.3e}; "
        f"gap23={metadata['third_singlet_gap_Eh']:.6e} Eh",
        flush=True,
    )
    print(f"CPDC_MULTI_SA3={status}", flush=True)
    return result


def point_for_parameter(
    root: Path,
    surface: Mapping[str, Any],
    ux: float,
    uy: float,
    label: str,
    descriptor: Mapping[str, Any],
) -> PointSpec:
    geometry = surface_geometry(surface, ux, uy)
    geometry_hash = eng.array_hash(geometry)
    target = root / "points" / f"point_{geometry_hash}" / "solution.npz"
    return PointSpec(
        target=target,
        ux=float(ux),
        uy=float(uy),
        geometry_hash=geometry_hash,
        label=label,
        descriptor=dict(descriptor),
    )


def loop_points(
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    level_index: int,
    npoints: int,
) -> List[PointSpec]:
    points: List[PointSpec] = []
    for index in range(npoints):
        angle = 2.0 * math.pi * index / npoints
        ux = loop.center_ux + loop.radius_u * math.cos(angle)
        uy = loop.center_uy + loop.radius_u * math.sin(angle)
        points.append(
            point_for_parameter(
                root,
                surface,
                ux,
                uy,
                f"{loop.name}_L{level_index:02d}_p{index:03d}",
                {
                    "kind": "relaxed_Fourier_surface_loop_vertex",
                    "loop_name": loop.name,
                    "level_index": level_index,
                    "npoints": npoints,
                    "point_index": index,
                    "ux": ux,
                    "uy": uy,
                },
            )
        )
    return points


def point_worker(args: argparse.Namespace) -> None:
    configure_threads(args.threads)
    require_pyscf()
    surface = load_surface(args.surface_npz.resolve())
    geometry = surface_geometry(surface, args.ux, args.uy)
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
            "kind": "relaxed_Fourier_surface_loop_vertex",
            "ux": float(args.ux),
            "uy": float(args.uy),
            "geometry_hash": geometry_hash,
        },
        compute_state_gradients=False,
    )
    print(
        f"POINT {geometry_hash[:12]} ux={args.ux:+.9f} uy={args.uy:+.9f} "
        f"gap={metadata['adiabatic_gap_Eh']:.9e} Eh "
        f"grad={metadata['maximum_orbital_gradient']:.3e}",
        flush=True,
    )


def worker_command(
    args: argparse.Namespace,
    point: PointSpec,
    center_seed: Path,
    surface_npz: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "point-worker",
        "--target",
        str(point.target.resolve()),
        "--surface-npz",
        str(surface_npz.resolve()),
        "--ux",
        f"{point.ux:.17g}",
        "--uy",
        f"{point.uy:.17g}",
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


def load_validated_point(
    point: PointSpec,
    surface: Mapping[str, Any],
    basis: str,
) -> Dict[str, Any]:
    geometry = surface_geometry(surface, point.ux, point.uy)
    if eng.array_hash(geometry) != point.geometry_hash:
        raise RuntimeError(f"point geometry changed for {point.label}")
    eng.validate_existing_solution(
        point.target,
        geometry,
        basis,
        POINT_ROLE,
        gradients_required=False,
    )
    return eng.load_solution(point.target)


def run_point_batch(
    args: argparse.Namespace,
    points: Sequence[PointSpec],
    surface: Mapping[str, Any],
    center_seed: Path,
    surface_npz: Path,
) -> None:
    unique: Dict[str, PointSpec] = {}
    for point in points:
        unique[str(point.target.resolve())] = point
    pending: List[PointSpec] = []
    for point in unique.values():
        if point.target.exists():
            load_validated_point(point, surface, args.basis)
        else:
            pending.append(point)
    if not pending:
        return
    print(
        f"Launching {len(pending)} new loop points with max_jobs={args.max_jobs}, "
        f"threads_per_job={args.threads}",
        flush=True,
    )
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {}
        for point in pending:
            command = worker_command(args, point, center_seed, surface_npz)
            log_path = point.target.parent / "worker.stdout"
            future = executor.submit(
                subprocess_worker, command, log_path, args.threads
            )
            futures[future] = point
        for future in as_completed(futures):
            point = futures[future]
            try:
                future.result()
                solution = load_validated_point(point, surface, args.basis)
                metadata = solution["metadata"]
                print(
                    f"DONE {point.label} gap={metadata['adiabatic_gap_Eh']:.9e} Eh "
                    f"grad={metadata['maximum_orbital_gradient']:.3e}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{point.label}: {type(exc).__name__}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def production_point_checks(
    solution: Mapping[str, Any],
    geometry: np.ndarray,
    center_geometry: np.ndarray,
    center_average_energy: float,
    thresholds: Thresholds,
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    metadata = solution["metadata"]
    weights = np.asarray(metadata.get("state_average_weights", []), dtype=float)
    metrics = geometry_metrics(geometry, center_geometry)
    energy_above = float(metadata["average_energy_Eh"] - center_average_energy)
    metrics["average_energy_above_center_Eh"] = energy_above
    checks = {
        "model_semantics_exact": bool(
            metadata.get("script_version") == SOURCE_SCRIPT_VERSION
            and metadata.get("protocol_id") == SOURCE_PROTOCOL_ID
            and str(metadata.get("basis", "")).lower() == BASIS_KEY
            and metadata.get("basis_definition_sha256")
            == eng.BASIS_DEFINITION_SHA256
            and metadata.get("pyscf_version") == PYSCF_VERSION_EXACT
            and int(metadata.get("charge", 999)) == 1
            and int(metadata.get("spin_2S", 999)) == 0
            and int(metadata.get("ncas", -1)) == NCAS
            and int(metadata.get("nelecas", -1)) == NELECAS
            and int(metadata.get("nroots", -1)) == NROOTS
            and int(metadata.get("frozen_orbitals", -1)) == 0
            and weights.shape == (NROOTS,)
            and np.allclose(weights, [0.5, 0.5], rtol=0.0, atol=1.0e-15)
            and not bool(metadata.get("absolute_ground_state_claimed", True))
        ),
        "wavefunction_stationary": bool(
            metadata.get("converged")
            and np.isfinite(
                float(metadata.get("maximum_orbital_gradient", float("nan")))
            )
            and float(metadata["maximum_orbital_gradient"])
            <= thresholds.maximum_final_orbital_gradient
        ),
        "singlet_spin_pure": bool(
            float(metadata.get("maximum_spin_square_error", float("inf")))
            <= thresholds.maximum_spin_square_error
        ),
        "MO_frame_orthonormal": bool(
            float(metadata.get("mo_orthonormality_max_error", float("inf")))
            <= thresholds.maximum_mo_orthonormality_error
        ),
        "CASCI_reproduces_SA2": bool(
            float(metadata.get("CASCI_reproduction_max_error_Eh", float("inf")))
            <= thresholds.maximum_casci_reproduction_error_Eh
        ),
        "active_reference_retained": bool(
            float(metadata.get("active_reference_sigma_min", -float("inf")))
            >= thresholds.minimum_active_reference_sigma
        ),
        "third_singlet_isolated": bool(
            float(metadata.get("third_singlet_gap_Eh", -float("inf")))
            >= thresholds.minimum_third_singlet_gap_Eh
        ),
        "geometry_finite": metrics["finite_geometry"],
        "planarity_preserved": bool(
            metrics["planarity_error_angstrom"]
            <= thresholds.maximum_planarity_error_angstrom
        ),
        "CC_bonds_admissible": bool(
            metrics["minimum_CC_bond_angstrom"]
            >= thresholds.minimum_CC_bond_angstrom
            and metrics["maximum_CC_bond_angstrom"]
            <= thresholds.maximum_CC_bond_angstrom
        ),
        "CH_bonds_admissible": bool(
            metrics["minimum_CH_bond_angstrom"]
            >= thresholds.minimum_CH_bond_angstrom
            and metrics["maximum_CH_bond_angstrom"]
            <= thresholds.maximum_CH_bond_angstrom
        ),
        "average_energy_admissible": bool(
            energy_above <= thresholds.maximum_average_energy_above_center_Eh
        ),
    }
    return checks, metrics


def active_edge_metrics(sol_a, sol_b, mol_a, mol_b) -> Dict[str, Any]:
    _, gto, _, _ = require_pyscf()
    ncore_a = int(sol_a["metadata"]["ncore"])
    ncore_b = int(sol_b["metadata"]["ncore"])
    s_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    active_a = sol_a["mo_coeff"][:, ncore_a : ncore_a + NCAS]
    active_b = sol_b["mo_coeff"][:, ncore_b : ncore_b + NCAS]
    singular_values = np.linalg.svd(
        active_a.conj().T @ s_ab @ active_b, compute_uv=False
    )
    return {
        "singular_values": singular_values,
        "sigma_min": float(np.min(singular_values)),
        "leakage": float(1.0 - np.mean(singular_values**2)),
    }


def load_loop_objects(
    args: argparse.Namespace,
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    level_index: int,
    npoints: int,
) -> Tuple[List[PointSpec], Dict[int, Dict[str, Any]], Dict[int, Any]]:
    points = loop_points(root, surface, loop, level_index, npoints)
    solutions: Dict[int, Dict[str, Any]] = {}
    molecules: Dict[int, Any] = {}
    for index, point in enumerate(points):
        solution = load_validated_point(point, surface, args.basis)
        geometry = surface_geometry(surface, point.ux, point.uy)
        solutions[index] = solution
        molecules[index] = eng.make_mol(
            geometry, args.basis, args.memory_mb, verbose=0
        )
    return points, solutions, molecules


def analyze_loop_fidelity(
    args: argparse.Namespace,
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    level_index: int,
    npoints: int,
    center_average_energy: float,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    points, solutions, molecules = load_loop_objects(
        args, root, surface, loop, level_index, npoints
    )
    center_geometry = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    point_checks: Dict[int, Dict[str, bool]] = {}
    point_metrics: Dict[int, Dict[str, Any]] = {}
    for index, point in enumerate(points):
        geometry = surface_geometry(surface, point.ux, point.uy)
        checks, metrics = production_point_checks(
            solutions[index],
            geometry,
            center_geometry,
            center_average_energy,
            thresholds,
        )
        point_checks[index] = checks
        point_metrics[index] = metrics

    edge_records: Dict[int, Dict[str, Any]] = {}
    for index in range(npoints):
        neighbor = (index + 1) % npoints
        matrix = ov.exact_overlap_matrix(
            solutions[index],
            solutions[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        if matrix.shape != (NROOTS, NROOTS):
            raise RuntimeError(f"unexpected overlap shape: {matrix.shape}")
        pair = ov.polar_metrics(matrix)
        active = active_edge_metrics(
            solutions[index],
            solutions[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        scalar_abs = float(abs(complex(matrix[0, 0])))
        strict = bool(
            pair["sigma_min"] >= thresholds.pair_edge_sigma_minimum
            and pair["leakage"] <= thresholds.pair_edge_leakage_maximum
            and pair["normalized_margin"] >= thresholds.minimum_assignment_margin
            and pair["permutation"] == [0, 1]
            and active["sigma_min"] >= thresholds.active_edge_sigma_minimum
            and active["leakage"] <= thresholds.active_edge_leakage_maximum
            and scalar_abs >= thresholds.minimum_single_state_edge_overlap
        )
        edge_records[index] = {
            "from_index": index,
            "to_index": neighbor,
            "pair_subspace": {
                key: value for key, value in pair.items() if key != "transport"
            },
            "active_orbital_subspace": active,
            "lowest_state_overlap_abs": scalar_abs,
            "strict": strict,
        }

    gaps = [
        float(solution["metadata"]["adiabatic_gap_Eh"])
        for solution in solutions.values()
    ]
    all_points_pass = all(
        all(checks.values()) for checks in point_checks.values()
    )
    all_edges_strict = all(record["strict"] for record in edge_records.values())
    loop_gapped = min(gaps) >= thresholds.minimum_loop_adiabatic_gap_Eh
    point_integrity = [
        {
            "index": index,
            "ux": point.ux,
            "uy": point.uy,
            "geometry_hash": point.geometry_hash,
            "path": str(point.target.resolve()),
            "sha256": sha256_file(point.target),
        }
        for index, point in enumerate(points)
    ]
    return {
        "loop_name": loop.name,
        "topological_role": loop.topological_role,
        "expected_sign_preregistered": loop.expected_sign,
        "level_index": level_index,
        "npoints": npoints,
        "angular_step_degrees": 360.0 / npoints,
        "point_integrity": point_integrity,
        "point_checks": point_checks,
        "point_metrics": point_metrics,
        "edge_records": edge_records,
        "minimum_adiabatic_gap_Eh": float(min(gaps)),
        "maximum_adiabatic_gap_Eh": float(max(gaps)),
        "all_points_pass": bool(all_points_pass),
        "all_edges_strict": bool(all_edges_strict),
        "loop_is_adiabatically_gapped": bool(loop_gapped),
        "strict": bool(all_points_pass and all_edges_strict and loop_gapped),
        "minimum_pair_sigma": min(
            record["pair_subspace"]["sigma_min"] for record in edge_records.values()
        ),
        "maximum_pair_leakage": max(
            record["pair_subspace"]["leakage"] for record in edge_records.values()
        ),
        "minimum_assignment_margin": min(
            record["pair_subspace"]["normalized_margin"]
            for record in edge_records.values()
        ),
        "minimum_active_sigma": min(
            record["active_orbital_subspace"]["sigma_min"]
            for record in edge_records.values()
        ),
        "maximum_active_leakage": max(
            record["active_orbital_subspace"]["leakage"]
            for record in edge_records.values()
        ),
        "minimum_lowest_state_overlap_abs": min(
            record["lowest_state_overlap_abs"] for record in edge_records.values()
        ),
        "signed_edge_overlaps_recorded": False,
        "phase_product_evaluated": False,
    }


def qualify_loop_geometry(
    args: argparse.Namespace,
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    center_seed: Path,
    center_average_energy: float,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    selected: Dict[str, Any] | None = None
    confirmation: Dict[str, Any] | None = None
    for level_index, npoints in enumerate(loop.vertex_levels):
        print(
            f"\nCPDC PHASE-BLIND LOOP GEOMETRY {loop.name}: "
            f"level={level_index} npoints={npoints}",
            flush=True,
        )
        points = loop_points(root, surface, loop, level_index, npoints)
        run_point_batch(
            args,
            points,
            surface,
            center_seed,
            surface_npz_path(root),
        )
        audit = analyze_loop_fidelity(
            args,
            root,
            surface,
            loop,
            level_index,
            npoints,
            center_average_energy,
            thresholds,
        )
        attempts.append(audit)
        print(
            f"fidelity: pair_sigma={audit['minimum_pair_sigma']:.9f} "
            f"pair_leak={audit['maximum_pair_leakage']:.3e} "
            f"active_sigma={audit['minimum_active_sigma']:.9f} "
            f"state_overlap={audit['minimum_lowest_state_overlap_abs']:.9f} "
            f"min_gap={audit['minimum_adiabatic_gap_Eh']:.6e} Eh "
            f"strict={audit['strict']}",
            flush=True,
        )
        if audit["strict"]:
            if selected is None:
                selected = audit
            else:
                confirmation = audit
                break
    status = (
        "PASS_LOOP_GEOMETRY"
        if selected is not None and confirmation is not None
        else "STOP_LOOP_GEOMETRY_FAILURE"
    )
    return {
        "loop_name": loop.name,
        "status": status,
        "selection_rule": (
            "first strict preregistered mesh and first higher strict mesh; "
            "only point, gap, assignment, subspace, active-space, and absolute "
            "edge-overlap quantities may enter selection"
        ),
        "phase_used_for_selection": False,
        "attempts": attempts,
        "selected": selected,
        "confirmation": confirmation,
    }


def surface_stop(
    root: Path,
    status: str,
    reason: str,
    source_authorization: Mapping[str, Any],
    rank_arbitration: Mapping[str, Any] | None,
    surface_definition: Mapping[str, Any] | None,
    loop_qualifications: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "reason": reason,
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "manuscript_claim_authorized": False,
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "source_status": source_authorization.get("status"),
        "rank_arbitration": rank_arbitration,
        "surface_definition": surface_definition,
        "loop_qualifications": list(loop_qualifications),
        "authorized_next_step": (
            "Stop this immutable protocol instance; do not change a radius, mesh, "
            "threshold, surface, or electronic model after seeing the result."
        ),
    }
    output = loop_geometry_decision_path(root)
    write_json(output, result)
    print("\nCPDC RELAXED MULTIRING LOOP GEOMETRY", flush=True)
    print(f"Status: {status}", flush=True)
    print(f"Reason: {reason}", flush=True)
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_LOOP_GEOMETRY={status}", flush=True)
    return result


def run_surface(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != BASIS_KEY:
        raise SystemExit("This protocol is locked to cc-pVTZ")
    validate_protocol_declaration()
    root = args.root.resolve()
    source_authorization = validate_source_authorization(
        root, args.source_root, require_pass=True
    )
    create_or_validate_manifest(root, args.basis)
    require_pyscf()
    decision_path = loop_geometry_decision_path(root)
    if decision_path.exists():
        result = json.loads(decision_path.read_text())
        immutable = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "phase_inspected": False,
            "phase_product_evaluated": False,
            "manuscript_claim_authorized": False,
            "source_authorization_sha256": sha256_file(
                source_authorization_path(root)
            ),
        }
        if {key: result.get(key) for key in immutable} != immutable:
            raise RuntimeError(
                "existing loop-geometry decision failed restart integrity validation"
            )
        if result.get("status") == "PASS_LOOP_GEOMETRY":
            result = validate_loop_geometry_decision(root, source_authorization)
        elif result.get("status") not in {
            "STOP_SA3_ARBITRATION_FAILURE",
            "STOP_BOUNDED_LOOP_GEOMETRY_FAILURE",
        }:
            raise RuntimeError(
                "existing loop-geometry decision has an unknown terminal status"
            )
        print(f"CPDC_LOOP_GEOMETRY={result.get('status', 'UNKNOWN')}", flush=True)
        return result
    if topology_decision_path(root).exists():
        raise RuntimeError("topology decision exists without geometry decision")
    thresholds = Thresholds()
    rank = run_rank_arbitration(
        args, root, source_authorization, thresholds
    )
    if rank.get("status") != "PASS_SA3_ARBITRATION":
        return surface_stop(
            root,
            "STOP_SA3_ARBITRATION_FAILURE",
            "Independent cc-pVTZ SA3 rank arbitration failed.",
            source_authorization,
            rank,
            None,
            [],
        )

    surface_definition = build_or_validate_surface(root, source_authorization)
    surface = load_surface(surface_npz_path(root))
    loops = build_loop_specs()
    center_seed = Path(str(source_authorization["center_solution_file"]))
    center_average_energy = float(source_authorization["center_average_energy_Eh"])
    qualifications: List[Dict[str, Any]] = []
    for loop in loops:
        qualification = qualify_loop_geometry(
            args,
            root,
            surface,
            loop,
            center_seed,
            center_average_energy,
            thresholds,
        )
        qualifications.append(qualification)
        if qualification["status"] != "PASS_LOOP_GEOMETRY":
            return surface_stop(
                root,
                "STOP_BOUNDED_LOOP_GEOMETRY_FAILURE",
                f"Loop {loop.name} lacks two phase-blind strict meshes.",
                source_authorization,
                rank,
                surface_definition,
                qualifications,
            )

    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_LOOP_GEOMETRY",
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "signed_edge_overlaps_recorded": False,
        "manuscript_claim_authorized": False,
        "source_authorization_file": str(source_authorization_path(root)),
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "rank_arbitration_decision_file": str(rank_decision_path(root)),
        "rank_arbitration_decision_sha256": sha256_file(rank_decision_path(root)),
        "surface_definition": surface_definition,
        "surface_npz_sha256": sha256_file(surface_npz_path(root)),
        "loop_specs": [asdict(loop) for loop in loops],
        "loop_qualifications": qualifications,
        "thresholds": asdict(thresholds),
        "geometry_frozen": True,
        "selection_discipline": {
            "surface_uses": "only the six frozen source CI geometries",
            "mesh_selection_uses": (
                "wavefunction stationarity, root isolation, gaps, assignment, "
                "subspace fidelity, active-space fidelity, and overlap magnitudes"
            ),
            "Berry_or_Wilson_product_used": False,
        },
        "authorized_next_step": (
            "Run the separate phase command on these exact frozen polygons."
        ),
    }
    write_json(decision_path, result)
    print("\nCPDC RELAXED MULTIRING LOOP GEOMETRY", flush=True)
    print("Status: PASS_LOOP_GEOMETRY", flush=True)
    for qualification in qualifications:
        print(
            f"{qualification['loop_name']}: "
            f"selected={qualification['selected']['npoints']} "
            f"confirmation={qualification['confirmation']['npoints']}",
            flush=True,
        )
    print(f"Decision file: {decision_path}", flush=True)
    print("CPDC_LOOP_GEOMETRY=PASS_LOOP_GEOMETRY", flush=True)
    return result


def validate_loop_geometry_decision(
    root: Path,
    source_authorization: Mapping[str, Any],
) -> Dict[str, Any]:
    path = loop_geometry_decision_path(root)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run surface first")
    decision = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase_inspected": False,
        "phase_product_evaluated": False,
        "manuscript_claim_authorized": False,
        "geometry_frozen": True,
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "surface_npz_sha256": sha256_file(surface_npz_path(root)),
    }
    if {key: decision.get(key) for key in expected} != expected:
        raise RuntimeError("loop-geometry decision failed immutable-field validation")
    if decision.get("status") != "PASS_LOOP_GEOMETRY":
        raise SystemExit(
            f"Loop geometry is {decision.get('status', 'UNKNOWN')}; phase is forbidden"
        )
    if decision.get("rank_arbitration_decision_sha256") != sha256_file(
        rank_decision_path(root)
    ):
        raise RuntimeError("rank-arbitration decision changed after surface freeze")
    if decision.get("source_authorization_sha256") != sha256_file(
        source_authorization_path(root)
    ):
        raise RuntimeError("source authorization changed after surface freeze")
    if decision.get("thresholds") != _jsonable(asdict(Thresholds())):
        raise RuntimeError("frozen loop thresholds disagree with executable")
    loops = build_loop_specs()
    current_specs = [asdict(loop) for loop in loops]
    if decision.get("loop_specs") != _jsonable(current_specs):
        raise RuntimeError("frozen loop specs disagree with executable")
    qualifications = decision.get("loop_qualifications", [])
    if not isinstance(qualifications, list) or len(qualifications) != len(loops):
        raise RuntimeError("frozen loop qualification inventory is incomplete")
    surface = load_surface(surface_npz_path(root))
    for loop, qualification in zip(loops, qualifications):
        if (
            qualification.get("loop_name") != loop.name
            or qualification.get("status") != "PASS_LOOP_GEOMETRY"
            or qualification.get("phase_used_for_selection") is not False
        ):
            raise RuntimeError(f"invalid frozen qualification for {loop.name}")
        attempts = qualification.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError(f"missing mesh attempts for {loop.name}")
        if len(attempts) > len(loop.vertex_levels):
            raise RuntimeError(f"too many mesh attempts for {loop.name}")
        strict_attempts = []
        for level_index, attempt in enumerate(attempts):
            if (
                int(attempt.get("level_index", -1)) != level_index
                or int(attempt.get("npoints", -1))
                != int(loop.vertex_levels[level_index])
                or attempt.get("loop_name") != loop.name
                or attempt.get("signed_edge_overlaps_recorded") is not False
                or attempt.get("phase_product_evaluated") is not False
            ):
                raise RuntimeError(
                    f"mesh schedule or phase embargo changed for {loop.name}"
                )
            if attempt.get("strict") is True:
                strict_attempts.append(attempt)
        if len(strict_attempts) != 2 or attempts[-1] != strict_attempts[-1]:
            raise RuntimeError(
                f"frozen selection is not the first two strict meshes for {loop.name}"
            )
        if (
            qualification.get("selected") != strict_attempts[0]
            or qualification.get("confirmation") != strict_attempts[1]
        ):
            raise RuntimeError(f"selected mesh changed for {loop.name}")
        for mesh_label, mesh in (
            ("selected", strict_attempts[0]),
            ("confirmation", strict_attempts[1]),
        ):
            level_index = int(mesh["level_index"])
            npoints = int(mesh["npoints"])
            expected_points = loop_points(
                root, surface, loop, level_index, npoints
            )
            integrity = mesh.get("point_integrity", [])
            if not isinstance(integrity, list) or len(integrity) != npoints:
                raise RuntimeError(
                    f"incomplete {mesh_label} point inventory for {loop.name}"
                )
            for index, (item, expected_point) in enumerate(
                zip(integrity, expected_points)
            ):
                point_path = Path(str(item.get("path", ""))).resolve()
                if (
                    int(item.get("index", -1)) != index
                    or abs(float(item.get("ux", float("nan"))) - expected_point.ux)
                    > 1.0e-14
                    or abs(float(item.get("uy", float("nan"))) - expected_point.uy)
                    > 1.0e-14
                    or item.get("geometry_hash") != expected_point.geometry_hash
                    or point_path != expected_point.target.resolve()
                    or not point_path.exists()
                    or not point_path.is_relative_to(root.resolve())
                    or sha256_file(point_path) != item.get("sha256")
                ):
                    raise RuntimeError(
                        "frozen loop point changed after geometry freeze: "
                        f"{loop.name}/{mesh_label}/{index}"
                    )
    if source_authorization.get("status") != "PASS_SOURCE_AUTHORIZATION":
        raise RuntimeError("source authorization is no longer valid")
    return decision


def _maximum_cyclic_error(overlaps: Sequence[complex]) -> float:
    reference = ov.normalized_phase_product(overlaps)
    errors = []
    values = list(overlaps)
    for offset in range(len(values)):
        shifted = values[offset:] + values[:offset]
        errors.append(abs(ov.normalized_phase_product(shifted) - reference))
    return float(max(errors, default=0.0))


def phase_mesh_audit(
    args: argparse.Namespace,
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    mesh: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    level_index = int(mesh["level_index"])
    npoints = int(mesh["npoints"])
    _, solutions, molecules = load_loop_objects(
        args, root, surface, loop, level_index, npoints
    )
    forward_overlaps: List[complex] = []
    pair_determinants: List[complex] = []
    for index in range(npoints):
        neighbor = (index + 1) % npoints
        matrix = ov.exact_overlap_matrix(
            solutions[index],
            solutions[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        forward_overlaps.append(complex(matrix[0, 0]))
        pair_determinants.append(
            complex(np.linalg.det(ov.polar_metrics(matrix)["transport"]))
        )

    reverse_order = [0] + list(range(npoints - 1, 0, -1))
    reverse_overlaps: List[complex] = []
    for position in range(npoints):
        left = reverse_order[position]
        right = reverse_order[(position + 1) % npoints]
        matrix = ov.exact_overlap_matrix(
            solutions[left],
            solutions[right],
            molecules[left],
            molecules[right],
        )
        reverse_overlaps.append(complex(matrix[0, 0]))

    forward_product = ov.normalized_phase_product(forward_overlaps)
    reverse_product = ov.normalized_phase_product(reverse_overlaps)
    gauge_audits = [
        ov.scalar_phase_audit(forward_overlaps, seed=seed)
        for seed in GAUGE_SEEDS
    ]
    maximum_gauge_error = max(item["gauge_error"] for item in gauge_audits)
    cyclic_error = _maximum_cyclic_error(forward_overlaps)
    orientation_error = float(abs(reverse_product - np.conj(forward_product)))
    pair_phase = ov.normalized_phase_product(pair_determinants)
    numerical_checks = {
        "phase_real_magnitude": bool(
            abs(float(np.real(forward_product)))
            >= thresholds.minimum_phase_real_magnitude
        ),
        "phase_imaginary_small": bool(
            abs(float(np.imag(forward_product)))
            <= thresholds.maximum_phase_imaginary_abs
        ),
        "all_random_gauges_invariant": bool(
            maximum_gauge_error <= thresholds.maximum_gauge_error
        ),
        "reverse_orientation_consistent": bool(
            orientation_error <= thresholds.maximum_orientation_error
        ),
        "all_cyclic_starts_invariant": bool(
            cyclic_error <= thresholds.maximum_cyclic_start_error
        ),
    }
    expected_sign_pass = bool(
        float(np.real(forward_product)) * loop.expected_sign
        >= thresholds.minimum_phase_real_magnitude
    )
    return {
        "loop_name": loop.name,
        "mesh_role": "frozen_mesh",
        "level_index": level_index,
        "npoints": npoints,
        "expected_sign_preregistered": loop.expected_sign,
        "phase_product": forward_product,
        "phase_angle_rad": float(np.angle(forward_product)),
        "phase_real": float(np.real(forward_product)),
        "phase_imag": float(np.imag(forward_product)),
        "minimum_edge_overlap_abs": float(
            min(abs(value) for value in forward_overlaps)
        ),
        "reverse_phase_product": reverse_product,
        "orientation_error": orientation_error,
        "maximum_cyclic_start_error": cyclic_error,
        "random_gauge_seeds": list(GAUGE_SEEDS),
        "maximum_random_gauge_error": maximum_gauge_error,
        "pair_subspace_determinant_phase_product": pair_phase,
        "numerical_checks": numerical_checks,
        "phase_numerics_pass": all(numerical_checks.values()),
        "expected_sign_pass": expected_sign_pass,
    }


def expose_loop_phase(
    args: argparse.Namespace,
    root: Path,
    surface: Mapping[str, Any],
    loop: LoopSpec,
    qualification: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    selected = phase_mesh_audit(
        args, root, surface, loop, qualification["selected"], thresholds
    )
    confirmation = phase_mesh_audit(
        args, root, surface, loop, qualification["confirmation"], thresholds
    )
    selected_phase = complex(
        selected["phase_product"]["real"], selected["phase_product"]["imag"]
    ) if isinstance(selected["phase_product"], Mapping) else complex(selected["phase_product"])
    confirmation_phase = complex(
        confirmation["phase_product"]["real"], confirmation["phase_product"]["imag"]
    ) if isinstance(confirmation["phase_product"], Mapping) else complex(confirmation["phase_product"])
    mesh_difference = float(abs(selected_phase - confirmation_phase))
    mesh_sign_consistent = bool(
        np.sign(np.real(selected_phase)) == np.sign(np.real(confirmation_phase))
    )
    mesh_checks = {
        "same_scalar_sign": mesh_sign_consistent,
        "phase_difference_small": bool(
            mesh_difference <= thresholds.maximum_mesh_phase_difference
        ),
    }
    numerical_pass = bool(
        selected["phase_numerics_pass"]
        and confirmation["phase_numerics_pass"]
        and all(mesh_checks.values())
    )
    expected_sign_pass = bool(
        selected["expected_sign_pass"] and confirmation["expected_sign_pass"]
    )
    return {
        "loop_name": loop.name,
        "topological_role": loop.topological_role,
        "expected_sign_preregistered": loop.expected_sign,
        "selected_mesh": selected,
        "confirmation_mesh": confirmation,
        "mesh_phase_difference": mesh_difference,
        "mesh_checks": mesh_checks,
        "phase_numerics_pass": numerical_pass,
        "expected_sign_pass": expected_sign_pass,
        "pass": bool(numerical_pass and expected_sign_pass),
    }


def validate_existing_topology_decision(root: Path) -> Dict[str, Any]:
    path = topology_decision_path(root)
    result = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase_inspected": True,
        "manuscript_claim_authorized": False,
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "loop_geometry_decision_sha256": sha256_file(
            loop_geometry_decision_path(root)
        ),
        "surface_npz_sha256": sha256_file(surface_npz_path(root)),
        "thresholds": asdict(Thresholds()),
    }
    if {key: result.get(key) for key in expected} != _jsonable(expected):
        raise RuntimeError(
            "existing topology decision failed restart integrity validation"
        )
    loop_phases = result.get("loop_phases", [])
    expected_names = [loop.name for loop in build_loop_specs()]
    if [item.get("loop_name") for item in loop_phases] != expected_names:
        raise RuntimeError("existing topology decision has an invalid loop inventory")
    allowed = {
        "PROMOTE_RELAXED_MULTIRING_TO_OPPOSED_HISTORY",
        "STOP_PHASE_NUMERICS_FAILURE",
        "STOP_TOPOLOGY_CONTROL_FAILURE",
        "STOP_OUTER_LOOP_NOT_FALSE_NEGATIVE",
    }
    if result.get("status") not in allowed:
        raise RuntimeError("existing topology decision has an unknown terminal status")
    return result


def run_phase(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != BASIS_KEY:
        raise SystemExit("This protocol is locked to cc-pVTZ")
    validate_protocol_declaration()
    root = args.root.resolve()
    source_authorization = validate_source_authorization(
        root, args.source_root, require_pass=True
    )
    create_or_validate_manifest(root, args.basis)
    require_pyscf()
    geometry_decision = validate_loop_geometry_decision(
        root, source_authorization
    )
    output = topology_decision_path(root)
    if output.exists():
        result = validate_existing_topology_decision(root)
        print(f"CPDC_RELAXED_MULTI={result.get('status', 'UNKNOWN')}", flush=True)
        return result

    surface = load_surface(surface_npz_path(root))
    loops = build_loop_specs()
    qualification_by_name = {
        item["loop_name"]: item
        for item in geometry_decision["loop_qualifications"]
    }
    thresholds = Thresholds()
    phases: List[Dict[str, Any]] = []
    print("\nCPDC PHASE EMBARGO RELEASED ON FROZEN POLYGONS", flush=True)
    for loop in loops:
        qualification = qualification_by_name.get(loop.name)
        if qualification is None:
            raise RuntimeError(f"missing frozen qualification for {loop.name}")
        phase = expose_loop_phase(
            args, root, surface, loop, qualification, thresholds
        )
        phases.append(phase)
        selected = phase["selected_mesh"]
        confirmation = phase["confirmation_mesh"]
        print(
            f"PHASE {loop.name}: expected={loop.expected_sign:+d} "
            f"selected={selected['phase_real']:+.12f} "
            f"confirmation={confirmation['phase_real']:+.12f} "
            f"mesh_error={phase['mesh_phase_difference']:.3e} "
            f"pass={phase['pass']}",
            flush=True,
        )

    phase_by_name = {item["loop_name"]: item for item in phases}
    numerical_pass = all(item["phase_numerics_pass"] for item in phases)
    central_pass = bool(phase_by_name["central_inner"]["expected_sign_pass"])
    satellites_pass = all(
        phase_by_name[f"satellite_{replica:02d}"]["expected_sign_pass"]
        for replica in range(N_RING)
    )
    outer_pass = bool(phase_by_name["outer_all_six"]["expected_sign_pass"])
    if not numerical_pass:
        status = "STOP_PHASE_NUMERICS_FAILURE"
        reason = "At least one frozen loop failed mesh/orientation/gauge numerics."
    elif not (central_pass and satellites_pass):
        status = "STOP_TOPOLOGY_CONTROL_FAILURE"
        reason = "The central or a secondary-CI local negative control failed."
    elif not outer_pass:
        status = "STOP_OUTER_LOOP_NOT_FALSE_NEGATIVE"
        reason = "The frozen outer loop did not return the preregistered positive sign."
    else:
        status = "PROMOTE_RELAXED_MULTIRING_TO_OPPOSED_HISTORY"
        reason = (
            "Six negative local CI controls coexist with a positive outer scalar sign."
        )

    checks = {
        "source_ccpVTZ_lift_qualified": True,
        "independent_SA3_rank_arbitration_passed": True,
        "relaxed_Fourier_surface_frozen_without_phase": True,
        "all_seven_loops_have_two_fidelity_qualified_meshes": True,
        "all_phase_numerics_and_mesh_controls_passed": numerical_pass,
        "central_loop_negative": central_pass,
        "all_five_secondary_loops_negative": satellites_pass,
        "outer_loop_positive": outer_pass,
    }
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "reason": reason,
        "phase_inspected": True,
        "manuscript_claim_authorized": False,
        "source_authorization_sha256": sha256_file(source_authorization_path(root)),
        "loop_geometry_decision_file": str(loop_geometry_decision_path(root)),
        "loop_geometry_decision_sha256": sha256_file(
            loop_geometry_decision_path(root)
        ),
        "surface_npz_sha256": sha256_file(surface_npz_path(root)),
        "thresholds": asdict(thresholds),
        "loop_phases": phases,
        "decision_checks": checks,
        "topological_accounting": {
            "central_intersections": 1,
            "secondary_intersections": 5,
            "total_inside_outer_loop": 6,
            "preregistered_product_of_local_signs": 1,
            "observed_local_signs": [
                phase_by_name["central_inner"]["selected_mesh"]["phase_real"]
            ]
            + [
                phase_by_name[f"satellite_{replica:02d}"]["selected_mesh"][
                    "phase_real"
                ]
                for replica in range(N_RING)
            ],
            "observed_outer_sign": phase_by_name["outer_all_six"][
                "selected_mesh"
            ]["phase_real"],
            "false_negative_statement": (
                "A positive scalar Berry sign on the outer loop cannot be interpreted "
                "as absence of enclosed conical intersections because six individually "
                "negative local controls lie on the same frozen nuclear surface."
            ),
        },
        "authorized_next_step": (
            "If promoted, run clockwise and counterclockwise sequential orbital-history "
            "confirmation on these exact frozen meshes before any manuscript claim."
        ),
    }
    write_json(output, result)
    print("\nCYCLOPENTADIENYL CATION RELAXED MULTIRING TOPOLOGY", flush=True)
    print(f"Status: {status}", flush=True)
    print(f"Reason: {reason}", flush=True)
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_RELAXED_MULTI={status}", flush=True)
    return result


def selftest() -> None:
    declaration = validate_protocol_declaration()
    if declaration.get("hypothesis", {}).get("outer_expected_sign") != 1:
        raise RuntimeError("protocol outer-sign hypothesis changed")
    if declaration.get("hypothesis", {}).get("each_local_expected_sign") != -1:
        raise RuntimeError("protocol local-sign hypothesis changed")
    if sha256_file(Path(eng.__file__).resolve()) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("embedded lift engine differs from the validated source engine")
    snapshot = json.loads(frozen_snapshot_path().read_text())
    if canonical_json_sha256(snapshot) != EXPECTED_SOURCE_CANONICAL_SHA256:
        raise RuntimeError("frozen source snapshot fingerprint mismatch")
    coefficients = np.asarray(
        snapshot["candidate_coefficients_angstrom"], dtype=float
    )
    center = np.asarray(eng.REFERENCE_GEOMETRY, dtype=float)
    candidate = eng.geometry_from_coefficients(coefficients)
    anchors = np.asarray(
        [eng.symmetry_replica(candidate, replica) for replica in range(N_RING)]
    )
    frequencies, fourier = _fourier_coefficients(anchors - center[None, :, :])
    errors = []
    for replica in range(N_RING):
        theta = 2.0 * math.pi * replica / N_RING
        reconstructed = center + _surface_displacement_from_arrays(
            theta, frequencies, fourier
        )
        errors.append(float(np.max(np.abs(reconstructed - anchors[replica]))))
    if max(errors) > 2.0e-12:
        raise RuntimeError(f"Fourier anchor selftest failed: {max(errors):.3e}")
    surface = {
        "center_geometry_angstrom": center,
        "frequencies": frequencies,
        "fourier_coefficients": fourier,
    }
    for replica in range(N_RING):
        theta = 2.0 * math.pi * replica / N_RING
        reconstructed = surface_geometry(surface, math.cos(theta), math.sin(theta))
        if np.max(np.abs(reconstructed - anchors[replica])) > 2.0e-12:
            raise RuntimeError("surface parameter-map selftest failed")
    loops = build_loop_specs()
    if len(loops) != 7:
        raise RuntimeError("expected seven preregistered loops")
    if [loop.expected_sign for loop in loops] != [-1] * 6 + [1]:
        raise RuntimeError("loop sign ordering changed")
    thresholds = Thresholds()
    for loop in loops:
        for npoints in loop.vertex_levels:
            for index in range(npoints):
                angle = 2.0 * math.pi * index / npoints
                geometry = surface_geometry(
                    surface,
                    loop.center_ux + loop.radius_u * math.cos(angle),
                    loop.center_uy + loop.radius_u * math.sin(angle),
                )
                metrics = geometry_metrics(geometry, center)
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
                    raise RuntimeError(
                        f"geometry selftest failed for {loop.name}/{npoints}/{index}"
                    )
    complex_probe = _jsonable({"z": -1.0 + 2.0j})
    if complex_probe != {"z": {"real": -1.0, "imag": 2.0}}:
        raise RuntimeError("complex JSON serialization selftest failed")
    ov.selftest()
    negative = ov.scalar_phase_audit([1.0, 1.0, -1.0, 1.0])
    positive = ov.scalar_phase_audit([1.0, -1.0, -1.0, 1.0])
    if negative["phase_real"] > -0.999999999 or positive["phase_real"] < 0.999999999:
        raise RuntimeError("scalar phase selftest failed")
    print("CPDC_RELAXED_MULTIRING_V1_SELFTEST=PASS")


def add_common_arguments(parser: argparse.ArgumentParser, include_jobs: bool) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--basis", default=BASIS_KEY)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-mb", type=int, default=100000)
    if include_jobs:
        parser.add_argument("--max-jobs", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest", help="NumPy-only protocol and surface tests")
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--root", type=Path, required=True)
    qualify.add_argument("--source-root", type=Path, required=True)
    qualify.add_argument("--basis", default=BASIS_KEY)
    surface = subparsers.add_parser("surface")
    add_common_arguments(surface, include_jobs=True)
    phase = subparsers.add_parser("phase")
    add_common_arguments(phase, include_jobs=False)

    worker = subparsers.add_parser("point-worker")
    worker.add_argument("--target", type=Path, required=True)
    worker.add_argument("--surface-npz", type=Path, required=True)
    worker.add_argument("--ux", type=float, required=True)
    worker.add_argument("--uy", type=float, required=True)
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
    if args.command == "surface":
        if args.max_jobs < 1:
            raise SystemExit("--max-jobs must be positive")
        run_surface(args)
    elif args.command == "phase":
        run_phase(args)
    elif args.command == "point-worker":
        point_worker(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
