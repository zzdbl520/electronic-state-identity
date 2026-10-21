#!/usr/bin/env python3
"""Rank-2 projector qualification for the lowest singlet pair of C5H5+.

This protocol is authorized only by two completed phase-blind antecedents:

1. the relaxed-multiring calculation stopped because the *internal* SA2 gap
   on the central loop remained below 2e-4 Eh although every electronic,
   external-isolation, geometry, pair-subspace, and active-space check passed;
2. the independent central-gap landscape screen reproduced the antecedent and
   found that neither the cone nor the smooth quadratic surface reached that
   unchanged internal-gap target.

The present executable therefore changes the scientific object, not the
threshold.  It treats the two lowest singlets as one rank-2 projector P12.
Five symmetry-inequivalent vertices of the existing 40-point central loop are
recalculated with independent equal-weight SA3-CASSCF.  The already completed
20/40/80 SA2 meshes are then reused to audit P12 transport, external isolation
from state 3, U(2) corner-gauge covariance, orientation reversal, cyclic-start
invariance, and mesh convergence.

No scalar Berry product, signed one-state overlap, individual-root assignment,
or conical-intersection count is evaluated or inferred here.  A rank-2 Wilson
spectrum characterizes the isolated manifold; it does not recover the scalar
Berry sign of either unresolved member of the pair.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

import CPDC_ccpVTZ_central_gap_landscape_v1 as gap_screen
import CPDC_ccpVTZ_relaxed_multiring_v1 as antecedent
import cpdc_lift_engine as eng
import cpdc_overlap as ov


SCRIPT_VERSION = "CPDC-CCPVTZ-RANK2-PROJECTOR-v1.0"
PROTOCOL_ID = "CPDC-CCPVTZ-RANK2-PROJECTOR-20260920"
PROTOCOL_FILENAME = "CPDC_ccpVTZ_rank2_projector_protocol_v1.json"
AUTHORIZATION_FILENAME = "CPDC_ccpVTZ_rank2_projector_authorization_v1.json"
MANIFEST_FILENAME = "CPDC_ccpVTZ_rank2_projector_v1_manifest.json"
DECISION_FILENAME = "CPDC_ccpVTZ_rank2_projector_v1_decision.json"

EXPECTED_ANTECEDENT_DRIVER_SHA256 = (
    "ca724ea4f56428f84043b6fd28f442811f9dfb5107039bad5b325c9264cb2942"
)
EXPECTED_SCREEN_DRIVER_SHA256 = (
    "b02c4ac8253fcf6a57bcaf3e5a2c3b69d8b68cc2b77ad6b809992d147cc29d49"
)
EXPECTED_ENGINE_SHA256 = (
    "7ff5d5dab945128e6031e61e2510d836ffd42d0a5e1f86078ef72e9b0ddcd325"
)
EXPECTED_OVERLAP_SHA256 = (
    "f3800d28751ac623bc7c75b8a5ea50d1786754b42ece9e5ca14be6c18b91a429"
)

PYSCF_VERSION_EXACT = "2.13.1"
BASIS_KEY = "cc-pvtz"
NROOTS_SA2 = 2
NROOTS_SA3 = 3
MESH_SIZES = (20, 40, 80)
SPOT_MESH_SIZE = 40
SPOT_INDICES = (0, 1, 2, 3, 4)
GAUGE_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924)
SA3_ROLE = "central_loop_rank2_SA3_symmetry_wedge"


@dataclass(frozen=True)
class Thresholds:
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square_error: float = 1.0e-6
    maximum_mo_orthonormality_error: float = 1.0e-8
    maximum_casci_reproduction_error_Eh: float = 1.0e-7
    minimum_active_reference_sigma: float = 0.90
    minimum_third_singlet_gap_Eh: float = 2.0e-3
    SA2_SA3_projector_sigma_minimum: float = 0.95
    SA2_SA3_projector_leakage_maximum: float = 0.03
    SA3_third_root_SA2_weight_maximum: float = 0.05
    pair_edge_sigma_minimum: float = 0.98
    pair_edge_leakage_maximum: float = 0.01
    active_edge_sigma_minimum: float = 0.98
    active_edge_leakage_maximum: float = 0.01
    maximum_gauge_invariant_error: float = 1.0e-10
    maximum_orientation_invariant_error: float = 1.0e-10
    maximum_cyclic_start_invariant_error: float = 1.0e-10
    maximum_40_80_eigenphase_difference_rad: float = 1.0e-3
    maximum_40_80_trace_difference: float = 1.0e-3
    maximum_40_80_determinant_difference: float = 1.0e-3


@dataclass(frozen=True)
class SpotSpec:
    index: int
    angle_degrees: float
    source_sa2: Path
    source_sa2_sha256: str
    geometry_hash: str
    target_sa3: Path


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


def decision_path(root: Path) -> Path:
    return root / DECISION_FILENAME


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
            f"rank-2 projector protocol requires PySCF {PYSCF_VERSION_EXACT}; "
            f"found {version}"
        )


def validate_protocol() -> Dict[str, Any]:
    path = protocol_path()
    if not path.exists():
        raise RuntimeError(f"missing protocol declaration: {path}")
    declaration = json.loads(path.read_text())
    model = declaration.get("locked_model", {})
    mesh = declaration.get("projector_mesh", {})
    spots = declaration.get("SA3_symmetry_wedge", {})
    expected = {
        "protocol_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "basis": BASIS_KEY,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "production_nroots": NROOTS_SA2,
        "arbitration_nroots": NROOTS_SA3,
    }
    actual = {
        "protocol_version": declaration.get("protocol_version"),
        "protocol_id": declaration.get("protocol_id"),
        "basis": str(model.get("basis", "")).lower(),
        "pyscf_version_exact": model.get("pyscf_version_exact"),
        "production_nroots": model.get("production_nroots"),
        "arbitration_nroots": model.get("arbitration_nroots"),
    }
    if actual != expected:
        raise RuntimeError(f"protocol declaration mismatch: {actual} != {expected}")
    if tuple(mesh.get("existing_mesh_sizes", [])) != MESH_SIZES:
        raise RuntimeError("projector mesh schedule changed")
    if int(spots.get("source_mesh_size", -1)) != SPOT_MESH_SIZE:
        raise RuntimeError("SA3 spot source mesh changed")
    if tuple(spots.get("point_indices", [])) != SPOT_INDICES:
        raise RuntimeError("SA3 symmetry-wedge indices changed")
    if declaration.get("scalar_phase_allowed") is not False:
        raise RuntimeError("scalar phase must remain forbidden")
    if declaration.get("signed_single_state_overlap_allowed") is not False:
        raise RuntimeError("signed one-state overlaps must remain forbidden")
    if declaration.get("CI_count_from_rank2_Wilson_allowed") is not False:
        raise RuntimeError("rank-2 Wilson data may not be used to count internal CIs")
    if declaration.get("thresholds") != jsonable(asdict(Thresholds())):
        raise RuntimeError("protocol thresholds differ from executable")
    return declaration


def _all_point_checks_pass(attempt: Mapping[str, Any]) -> bool:
    checks = attempt.get("point_checks", {})
    if not isinstance(checks, Mapping) or not checks:
        return False
    return all(
        isinstance(item, Mapping) and item and all(bool(value) for value in item.values())
        for item in checks.values()
    )


def _central_attempts(loop_decision: Mapping[str, Any]) -> List[Dict[str, Any]]:
    qualifications = loop_decision.get("loop_qualifications", [])
    if not isinstance(qualifications, list) or len(qualifications) != 1:
        raise RuntimeError("antecedent must contain only the failed central loop")
    central = qualifications[0]
    if (
        central.get("loop_name") != "central_inner"
        or central.get("status") != "STOP_LOOP_GEOMETRY_FAILURE"
        or central.get("phase_used_for_selection") is not False
    ):
        raise RuntimeError("antecedent central-loop qualification is incompatible")
    attempts = central.get("attempts", [])
    if not isinstance(attempts, list) or [
        int(item.get("npoints", -1)) for item in attempts
    ] != list(MESH_SIZES):
        raise RuntimeError("antecedent does not contain the complete 20/40/80 schedule")
    return attempts


def _validate_attempt_inventory(
    loop_root: Path,
    surface: Mapping[str, Any],
    attempt: Mapping[str, Any],
    level_index: int,
) -> List[Dict[str, Any]]:
    npoints = MESH_SIZES[level_index]
    loop = antecedent.build_loop_specs()[0]
    expected_points = antecedent.loop_points(
        loop_root, surface, loop, level_index, npoints
    )
    inventory = attempt.get("point_integrity", [])
    if not isinstance(inventory, list) or len(inventory) != npoints:
        raise RuntimeError(f"incomplete antecedent inventory for {npoints}-point mesh")
    result: List[Dict[str, Any]] = []
    for index, (item, expected_point) in enumerate(zip(inventory, expected_points)):
        path = Path(str(item.get("path", ""))).resolve()
        if (
            int(item.get("index", -1)) != index
            or abs(float(item.get("ux", float("nan"))) - expected_point.ux) > 1.0e-14
            or abs(float(item.get("uy", float("nan"))) - expected_point.uy) > 1.0e-14
            or item.get("geometry_hash") != expected_point.geometry_hash
            or path != expected_point.target.resolve()
            or not path.exists()
            or not path.is_relative_to(loop_root.resolve())
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(
                f"antecedent point integrity failure: mesh={npoints} index={index}"
            )
        antecedent.load_validated_point(expected_point, surface, BASIS_KEY)
        result.append(
            {
                "mesh_npoints": npoints,
                "index": index,
                "angle_degrees": 360.0 * index / npoints,
                "path": str(path),
                "sha256": item["sha256"],
                "geometry_hash": expected_point.geometry_hash,
            }
        )
    return result


def build_authorization(
    root: Path,
    source_root: Path,
    loop_root: Path,
    screen_root: Path,
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    loop_root = loop_root.resolve()
    screen_root = screen_root.resolve()
    source = antecedent.validate_source_authorization(
        loop_root, source_root, require_pass=True
    )
    loop_file = antecedent.loop_geometry_decision_path(loop_root)
    if not loop_file.exists():
        raise RuntimeError(f"missing antecedent loop decision: {loop_file}")
    loop_decision = json.loads(loop_file.read_text())
    screen_authorization = gap_screen.validate_authorization(
        screen_root, source_root, loop_root
    )
    screen_decision = gap_screen.validate_existing_decision(screen_root)
    screen_file = gap_screen.decision_path(screen_root)
    rank_file = antecedent.rank_decision_path(loop_root)
    if not rank_file.exists():
        raise RuntimeError(f"missing antecedent center SA3 arbitration: {rank_file}")
    rank_decision = json.loads(rank_file.read_text())
    embedded_rank = loop_decision.get("rank_arbitration", {})
    surface_file = antecedent.surface_npz_path(loop_root)
    if not surface_file.exists():
        raise RuntimeError(f"missing antecedent surface: {surface_file}")
    surface = antecedent.load_surface(surface_file)
    surface_definition = loop_decision.get("surface_definition", {})
    attempts = _central_attempts(loop_decision)

    checks: Dict[str, bool] = {
        "antecedent_script_version": loop_decision.get("script_version")
        == antecedent.SCRIPT_VERSION,
        "antecedent_protocol_id": loop_decision.get("protocol_id")
        == antecedent.PROTOCOL_ID,
        "antecedent_stopped_only_at_bounded_geometry": loop_decision.get("status")
        == "STOP_BOUNDED_LOOP_GEOMETRY_FAILURE",
        "antecedent_phase_not_inspected": loop_decision.get("phase_inspected") is False,
        "antecedent_phase_product_not_evaluated": loop_decision.get(
            "phase_product_evaluated"
        )
        is False,
        "antecedent_claim_not_authorized": loop_decision.get(
            "manuscript_claim_authorized"
        )
        is False,
        "scalar_topology_decision_absent": not antecedent.topology_decision_path(
            loop_root
        ).exists(),
        "center_SA3_rank_arbitration_passed": rank_decision.get("status")
        == "PASS_SA3_ARBITRATION",
        "center_SA3_rank_embedded_exactly": embedded_rank == rank_decision,
        "surface_hash_matches_STOP_decision": surface_definition.get(
            "surface_npz_sha256"
        )
        == sha256_file(surface_file),
        "source_authorization_passed": source.get("status")
        == "PASS_SOURCE_AUTHORIZATION",
        "source_authorization_hash_matches_STOP_decision": loop_decision.get(
            "source_authorization_sha256"
        )
        == sha256_file(antecedent.source_authorization_path(loop_root)),
        "surface_was_constructed_phase_blind": bool(
            isinstance(surface_definition, Mapping)
            and surface_definition.get("phase_inspected") is False
        ),
        "screen_authorization_passed": screen_authorization.get("status")
        == "PASS_GAP_SCREEN_AUTHORIZATION",
        "screen_selected_projector_pivot": screen_decision.get("status")
        == "REFINE_SPOKE_MAXIMUM_OR_PIVOT_PROJECTOR",
        "screen_reproduction_passed": screen_decision.get(
            "antecedent_reproduction", {}
        ).get("pass")
        is True,
        "screen_target_unchanged": abs(
            float(screen_decision.get("target_internal_gap_Eh", float("nan")))
            - antecedent.Thresholds().minimum_loop_adiabatic_gap_Eh
        )
        <= 1.0e-15,
        "screen_no_surface_crossed_target": bool(
            set(screen_decision.get("surface_summaries", {}))
            == set(gap_screen.SURFACE_KINDS)
            and all(
                summary.get("spoke_target_crossed") is False
                for summary in screen_decision.get("surface_summaries", {}).values()
            )
        ),
        "screen_all_forty_points_pass": bool(
            len(screen_decision.get("point_records", [])) == 40
            and all(
                item.get("all_point_checks_pass") is True
                for item in screen_decision.get("point_records", [])
            )
        ),
        "screen_phase_not_inspected": screen_decision.get("phase_inspected") is False,
        "screen_signed_overlap_not_evaluated": screen_decision.get(
            "signed_overlap_evaluated"
        )
        is False,
    }

    point_integrity: Dict[str, Dict[str, Any]] = {}
    for level_index, attempt in enumerate(attempts):
        npoints = MESH_SIZES[level_index]
        checks[f"mesh_{npoints}_points_pass"] = _all_point_checks_pass(attempt)
        checks[f"mesh_{npoints}_edges_pass"] = attempt.get("all_edges_strict") is True
        checks[f"mesh_{npoints}_internal_gap_fails"] = (
            attempt.get("loop_is_adiabatically_gapped") is False
        )
        checks[f"mesh_{npoints}_strict_fails_only_as_expected"] = (
            attempt.get("strict") is False
            and attempt.get("all_points_pass") is True
            and attempt.get("all_edges_strict") is True
            and attempt.get("loop_is_adiabatically_gapped") is False
        )
        checks[f"mesh_{npoints}_phase_blind"] = (
            attempt.get("signed_edge_overlaps_recorded") is False
            and attempt.get("phase_product_evaluated") is False
        )
        for item in _validate_attempt_inventory(
            loop_root, surface, attempt, level_index
        ):
            key = str(Path(item["path"]).resolve())
            existing = point_integrity.get(key)
            if existing is not None and (
                existing["sha256"] != item["sha256"]
                or existing["geometry_hash"] != item["geometry_hash"]
            ):
                raise RuntimeError(f"nested mesh disagrees on point: {key}")
            point_integrity[key] = item

    checks["eighty_unique_nested_vertices"] = len(point_integrity) == 80
    mesh40 = _validate_attempt_inventory(loop_root, surface, attempts[1], 1)
    spot_integrity = [mesh40[index] for index in SPOT_INDICES]
    checks["five_symmetry_inequivalent_SA3_spots"] = (
        [item["index"] for item in spot_integrity] == list(SPOT_INDICES)
        and [item["angle_degrees"] for item in spot_integrity]
        == [0.0, 9.0, 18.0, 27.0, 36.0]
    )

    status = (
        "PASS_RANK2_PROJECTOR_AUTHORIZATION"
        if all(checks.values())
        else "STOP_RANK2_PROJECTOR_AUTHORIZATION_FAILURE"
    )
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "loop_root": str(loop_root),
        "screen_root": str(screen_root),
        "status": status,
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "source_authorization_file": str(
            antecedent.source_authorization_path(loop_root).resolve()
        ),
        "source_authorization_sha256": sha256_file(
            antecedent.source_authorization_path(loop_root)
        ),
        "loop_decision_file": str(loop_file.resolve()),
        "loop_decision_sha256": sha256_file(loop_file),
        "rank_arbitration_file": str(rank_file.resolve()),
        "rank_arbitration_sha256": sha256_file(rank_file),
        "rank_arbitration_solution_file": rank_decision.get("SA3_solution_file"),
        "rank_arbitration_solution_sha256": rank_decision.get("SA3_solution_sha256"),
        "surface_file": str(surface_file.resolve()),
        "surface_sha256": sha256_file(surface_file),
        "screen_authorization_file": str(
            gap_screen.authorization_path(screen_root).resolve()
        ),
        "screen_authorization_sha256": sha256_file(
            gap_screen.authorization_path(screen_root)
        ),
        "screen_decision_file": str(screen_file.resolve()),
        "screen_decision_sha256": sha256_file(screen_file),
        "source_center_average_energy_Eh": source.get("center_average_energy_Eh"),
        "unique_SA2_point_integrity": sorted(
            point_integrity.values(),
            key=lambda item: (int(item["mesh_npoints"]), int(item["index"])),
        ),
        "SA3_spot_source_integrity": spot_integrity,
    }


def _authorization_artifacts(record: Mapping[str, Any]) -> List[Tuple[str, str, Path]]:
    loop_root = Path(str(record["loop_root"])).resolve()
    screen_root = Path(str(record["screen_root"])).resolve()
    artifacts: List[Tuple[str, str, Path]] = [
        (
            str(record["source_authorization_file"]),
            str(record["source_authorization_sha256"]),
            loop_root,
        ),
        (str(record["loop_decision_file"]), str(record["loop_decision_sha256"]), loop_root),
        (
            str(record["rank_arbitration_file"]),
            str(record["rank_arbitration_sha256"]),
            loop_root,
        ),
        (
            str(record["rank_arbitration_solution_file"]),
            str(record["rank_arbitration_solution_sha256"]),
            loop_root,
        ),
        (str(record["surface_file"]), str(record["surface_sha256"]), loop_root),
        (
            str(record["screen_authorization_file"]),
            str(record["screen_authorization_sha256"]),
            screen_root,
        ),
        (
            str(record["screen_decision_file"]),
            str(record["screen_decision_sha256"]),
            screen_root,
        ),
    ]
    for item in record.get("unique_SA2_point_integrity", []):
        artifacts.append((str(item["path"]), str(item["sha256"]), loop_root))
    return artifacts


def validate_authorization(
    root: Path,
    source_root: Path,
    loop_root: Path,
    screen_root: Path,
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    loop_root = loop_root.resolve()
    screen_root = screen_root.resolve()
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
        "status": "PASS_RANK2_PROJECTOR_AUTHORIZATION",
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "manuscript_claim_authorized": False,
    }
    if {key: record.get(key) for key in expected} != expected:
        raise RuntimeError("rank-2 authorization failed immutable-field validation")
    for raw_path, expected_hash, owner in _authorization_artifacts(record):
        artifact = Path(raw_path).resolve()
        if (
            not artifact.exists()
            or not artifact.is_relative_to(owner)
            or sha256_file(artifact) != expected_hash
        ):
            raise RuntimeError(f"authorized artifact changed: {artifact}")
    antecedent.validate_source_authorization(loop_root, source_root, require_pass=True)
    gap_screen.validate_authorization(screen_root, source_root, loop_root)
    gap_screen.validate_existing_decision(screen_root)
    if antecedent.topology_decision_path(loop_root).exists():
        raise RuntimeError("a forbidden scalar-topology decision appeared after authorization")
    return record


def run_qualify(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = authorization_path(root)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("status") == "PASS_RANK2_PROJECTOR_AUTHORIZATION":
            result = validate_authorization(
                root, args.source_root, args.loop_root, args.screen_root
            )
        elif existing.get("status") == "STOP_RANK2_PROJECTOR_AUTHORIZATION_FAILURE":
            result = existing
        else:
            raise RuntimeError("existing authorization has an unknown status")
    else:
        try:
            result = build_authorization(
                root, args.source_root, args.loop_root, args.screen_root
            )
        except (Exception, SystemExit) as exc:
            result = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "root": str(root),
                "source_root": str(args.source_root.resolve()),
                "loop_root": str(args.loop_root.resolve()),
                "screen_root": str(args.screen_root.resolve()),
                "status": "STOP_RANK2_PROJECTOR_AUTHORIZATION_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}",
                "scalar_phase_evaluated": False,
                "signed_single_state_overlap_recorded": False,
                "individual_root_assignment_used": False,
                "manuscript_claim_authorized": False,
            }
        write_json(path, result)
    print("\nCPDC RANK-2 PROJECTOR AUTHORIZATION", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Decision file: {path}", flush=True)
    print(f"CPDC_RANK2_SOURCE={result['status']}", flush=True)
    return result


def build_manifest(root: Path, authorization: Mapping[str, Any]) -> Dict[str, Any]:
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
            Path(antecedent.__file__).name: sha256_file(Path(antecedent.__file__)),
            Path(gap_screen.__file__).name: sha256_file(Path(gap_screen.__file__)),
            Path(eng.__file__).name: sha256_file(Path(eng.__file__)),
            Path(ov.__file__).name: sha256_file(Path(ov.__file__)),
        },
        "authorized_roots": {
            "source_root": authorization["source_root"],
            "loop_root": authorization["loop_root"],
            "screen_root": authorization["screen_root"],
        },
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
    }


def create_or_validate_manifest(
    root: Path, authorization: Mapping[str, Any]
) -> Dict[str, Any]:
    path = manifest_path(root)
    if path.exists():
        record = json.loads(path.read_text())
        expected_hashes = {
            Path(__file__).name: sha256_file(Path(__file__).resolve()),
            PROTOCOL_FILENAME: sha256_file(protocol_path()),
            Path(antecedent.__file__).name: EXPECTED_ANTECEDENT_DRIVER_SHA256,
            Path(gap_screen.__file__).name: EXPECTED_SCREEN_DRIVER_SHA256,
            Path(eng.__file__).name: EXPECTED_ENGINE_SHA256,
            Path(ov.__file__).name: EXPECTED_OVERLAP_SHA256,
        }
        if (
            record.get("script_version") != SCRIPT_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("authorization_sha256")
            != sha256_file(authorization_path(root))
            or record.get("component_hashes") != expected_hashes
        ):
            raise RuntimeError("existing manifest failed integrity validation")
        return record
    current = {
        Path(antecedent.__file__).name: sha256_file(Path(antecedent.__file__)),
        Path(gap_screen.__file__).name: sha256_file(Path(gap_screen.__file__)),
        Path(eng.__file__).name: sha256_file(Path(eng.__file__)),
        Path(ov.__file__).name: sha256_file(Path(ov.__file__)),
    }
    frozen = {
        Path(antecedent.__file__).name: EXPECTED_ANTECEDENT_DRIVER_SHA256,
        Path(gap_screen.__file__).name: EXPECTED_SCREEN_DRIVER_SHA256,
        Path(eng.__file__).name: EXPECTED_ENGINE_SHA256,
        Path(ov.__file__).name: EXPECTED_OVERLAP_SHA256,
    }
    if current != frozen:
        raise RuntimeError(f"frozen dependency mismatch: {current} != {frozen}")
    record = build_manifest(root, authorization)
    write_json(path, record)
    return record


def spot_specs(root: Path, authorization: Mapping[str, Any]) -> List[SpotSpec]:
    source_records = authorization.get("SA3_spot_source_integrity", [])
    if not isinstance(source_records, list) or len(source_records) != len(SPOT_INDICES):
        raise RuntimeError("authorization lacks the five SA3 spot sources")
    result: List[SpotSpec] = []
    for record, expected_index in zip(source_records, SPOT_INDICES):
        index = int(record["index"])
        if index != expected_index:
            raise RuntimeError("SA3 spot order changed")
        angle = float(record["angle_degrees"])
        result.append(
            SpotSpec(
                index=index,
                angle_degrees=angle,
                source_sa2=Path(str(record["path"])).resolve(),
                source_sa2_sha256=str(record["sha256"]),
                geometry_hash=str(record["geometry_hash"]),
                target_sa3=(
                    root
                    / "sa3_spots"
                    / f"theta_{angle:05.1f}_deg"
                    / "solution.npz"
                ),
            )
        )
    return result


def validate_existing_sa3(target: Path, spot: SpotSpec) -> Dict[str, Any]:
    solution = eng.load_solution(target)
    metadata = solution["metadata"]
    weights = np.asarray(metadata.get("state_average_weights", []), dtype=float)
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": SA3_ROLE,
        "basis": BASIS_KEY,
        "pyscf_version": PYSCF_VERSION_EXACT,
        "ncas": antecedent.NCAS,
        "nelecas": antecedent.NELECAS,
        "nroots": NROOTS_SA3,
        "basis_definition_sha256": eng.BASIS_DEFINITION_SHA256,
        "charge": 1,
        "spin_2S": 0,
        "nact_alpha": antecedent.NACT_ALPHA,
        "nact_beta": antecedent.NACT_BETA,
        "frozen_orbitals": 0,
        "source_SA2_sha256": spot.source_sa2_sha256,
        "source_SA2_file": str(spot.source_sa2),
        "geometry_hash": spot.geometry_hash,
        "spot_index": spot.index,
        "absolute_ground_state_claimed": False,
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
        "basis_definition_sha256": metadata.get("basis_definition_sha256"),
        "charge": metadata.get("charge"),
        "spin_2S": metadata.get("spin_2S"),
        "nact_alpha": metadata.get("nact_alpha"),
        "nact_beta": metadata.get("nact_beta"),
        "frozen_orbitals": metadata.get("frozen_orbitals"),
        "source_SA2_sha256": metadata.get("source_SA2_sha256"),
        "source_SA2_file": metadata.get("source_SA2_file"),
        "geometry_hash": metadata.get("geometry_hash"),
        "spot_index": metadata.get("spot_index"),
        "absolute_ground_state_claimed": metadata.get(
            "absolute_ground_state_claimed"
        ),
    }
    if actual != expected:
        raise RuntimeError(f"existing SA3 spot metadata mismatch: {actual} != {expected}")
    if weights.shape != (NROOTS_SA3,) or not np.allclose(
        weights, [1.0 / NROOTS_SA3] * NROOTS_SA3, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("existing SA3 spot state-average weights changed")
    source = eng.load_solution(spot.source_sa2)
    if np.max(np.abs(solution["geometry_angstrom"] - source["geometry_angstrom"])) > 1.0e-11:
        raise RuntimeError("existing SA3 spot geometry mismatch")
    return metadata


def run_sa3_solution(
    target: Path,
    spot: SpotSpec,
    memory_mb: int,
) -> Dict[str, Any]:
    if target.exists():
        return validate_existing_sa3(target, spot)
    fci, _, mcscf, _ = eng.require_pyscf()
    source = eng.load_solution(spot.source_sa2)
    geometry = np.asarray(source["geometry_angstrom"], dtype=float)
    if eng.array_hash(geometry) != spot.geometry_hash:
        raise RuntimeError("SA3 spot source geometry hash changed")
    if sha256_file(spot.source_sa2) != spot.source_sa2_sha256:
        raise RuntimeError("SA3 spot source file changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.parent / "casscf.log"
    mol = eng.make_mol(geometry, BASIS_KEY, memory_mb, str(log_path), verbose=4)
    mf = eng.run_rhf(mol)
    mo_guess, selection = eng.build_pi_initial_mos(mf)
    ncore_expected = (mol.nelectron - antecedent.NELECAS) // 2
    reference_active = mo_guess[
        :, ncore_expected : ncore_expected + antecedent.NCAS
    ].copy()

    mc = mcscf.CASSCF(
        mf,
        antecedent.NCAS,
        (antecedent.NACT_ALPHA, antecedent.NACT_BETA),
    )
    mc.fcisolver = eng.make_singlet_solver(mol, NROOTS_SA3)
    mc = mc.state_average_([1.0 / NROOTS_SA3] * NROOTS_SA3)
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
    ci_roots = [np.asarray(root) for root in mc.ci]
    if energies.size != NROOTS_SA3 or len(ci_roots) != NROOTS_SA3:
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
    probe_energies, _ = eng.casci_probe(mf, final_mo, NROOTS_SA3)
    reproduction_error = float(np.max(np.abs(probe_energies - energies)))
    spin_square = [
        float(
            fci.spin_op.spin_square0(
                root,
                antecedent.NCAS,
                (antecedent.NACT_ALPHA, antecedent.NACT_BETA),
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
    final_active = final_mo[:, ncore : ncore + antecedent.NCAS]
    active_sv = np.linalg.svd(
        reference_active.conj().T @ metric @ final_active, compute_uv=False
    )
    metadata: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": SA3_ROLE,
        "molecule": "cyclopentadienyl cation",
        "formula": "C5H5+",
        "target": "rank-2 lowest-singlet manifold only",
        "absolute_ground_state_claimed": False,
        "basis": BASIS_KEY,
        "basis_definition_sha256": eng.BASIS_DEFINITION_SHA256,
        "method": "independent equal-weight SA3-CASSCF(4e,5o) full-valence-pi singlet",
        "charge": 1,
        "spin_2S": 0,
        "ncas": antecedent.NCAS,
        "nelecas": antecedent.NELECAS,
        "nact_alpha": antecedent.NACT_ALPHA,
        "nact_beta": antecedent.NACT_BETA,
        "nroots": NROOTS_SA3,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "state_average_weights": [1.0 / NROOTS_SA3] * NROOTS_SA3,
        "source_SA2_file": str(spot.source_sa2),
        "source_SA2_sha256": spot.source_sa2_sha256,
        "spot_index": spot.index,
        "spot_angle_degrees": spot.angle_degrees,
        "geometry_hash": spot.geometry_hash,
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
        "active_orbital_selection": selection,
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
    for index, root_ci in enumerate(ci_roots):
        arrays[f"ci_root_{index}"] = root_ci
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return metadata


def worker_command(args: argparse.Namespace, spot: SpotSpec) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "sa3-worker",
        "--target",
        str(spot.target_sa3.resolve()),
        "--source-sa2",
        str(spot.source_sa2),
        "--source-sha256",
        spot.source_sa2_sha256,
        "--geometry-hash",
        spot.geometry_hash,
        "--spot-index",
        str(spot.index),
        "--angle-degrees",
        f"{spot.angle_degrees:.17g}",
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
            f"SA3 spot worker failed with code {completed.returncode}; see {log_path}"
        )


def run_sa3_batch(args: argparse.Namespace, spots: Sequence[SpotSpec]) -> None:
    pending: List[SpotSpec] = []
    for spot in spots:
        if spot.target_sa3.exists():
            validate_existing_sa3(spot.target_sa3, spot)
        else:
            pending.append(spot)
    if not pending:
        return
    print(
        f"Launching {len(pending)} independent SA3 wedge points with "
        f"max_jobs={args.max_jobs}, threads_per_job={args.threads}",
        flush=True,
    )
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {}
        for spot in pending:
            future = executor.submit(
                subprocess_worker,
                worker_command(args, spot),
                spot.target_sa3.parent / "worker.stdout",
                args.threads,
            )
            futures[future] = spot
        for future in as_completed(futures):
            spot = futures[future]
            try:
                future.result()
                metadata = validate_existing_sa3(spot.target_sa3, spot)
                print(
                    f"DONE SA3 theta={spot.angle_degrees:4.1f} deg "
                    f"gap23={float(metadata['third_singlet_gap_Eh']):.6e} Eh "
                    f"grad={float(metadata['maximum_orbital_gradient']):.3e}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    f"theta={spot.angle_degrees:.1f}: {type(exc).__name__}: {exc}"
                )
                print(f"FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def sa3_worker(args: argparse.Namespace) -> None:
    configure_threads(args.threads)
    require_pyscf()
    spot = SpotSpec(
        index=int(args.spot_index),
        angle_degrees=float(args.angle_degrees),
        source_sa2=args.source_sa2.resolve(),
        source_sa2_sha256=str(args.source_sha256),
        geometry_hash=str(args.geometry_hash),
        target_sa3=args.target.resolve(),
    )
    metadata = run_sa3_solution(spot.target_sa3, spot, args.memory_mb)
    print(
        f"SA3_SPOT index={spot.index} theta={spot.angle_degrees:.1f} "
        f"gap12={float(metadata['internal_pair_gap_Eh']):.9e} Eh "
        f"gap23={float(metadata['third_singlet_gap_Eh']):.9e} Eh",
        flush=True,
    )


def analyze_sa3_spots(
    spots: Sequence[SpotSpec], memory_mb: int, thresholds: Thresholds
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for spot in spots:
        source = eng.load_solution(spot.source_sa2)
        arbitration = eng.load_solution(spot.target_sa3)
        geometry = np.asarray(source["geometry_angstrom"], dtype=float)
        mol_sa2 = eng.make_mol(geometry, BASIS_KEY, memory_mb, verbose=0)
        mol_sa3 = eng.make_mol(geometry, BASIS_KEY, memory_mb, verbose=0)
        overlap = ov.exact_overlap_matrix(source, arbitration, mol_sa2, mol_sa3)
        if overlap.shape != (NROOTS_SA2, NROOTS_SA3):
            raise RuntimeError(f"unexpected SA2/SA3 overlap shape: {overlap.shape}")
        pair_block = overlap[:, :NROOTS_SA2]
        singular_values = np.linalg.svd(pair_block, compute_uv=False)
        leakage = float(1.0 - np.mean(singular_values**2))
        third_weight = float(np.sum(np.abs(overlap[:, 2]) ** 2))
        metadata = arbitration["metadata"]
        checks = {
            "SA3_converged_and_stationary": bool(
                metadata.get("converged")
                and np.isfinite(float(metadata.get("maximum_orbital_gradient", float("nan"))))
                and float(metadata["maximum_orbital_gradient"])
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
            "SA2_pair_retained_in_SA3": bool(
                float(np.min(singular_values))
                >= thresholds.SA2_SA3_projector_sigma_minimum
                and leakage <= thresholds.SA2_SA3_projector_leakage_maximum
            ),
            "SA3_third_root_excluded_from_SA2": bool(
                third_weight <= thresholds.SA3_third_root_SA2_weight_maximum
            ),
            "SA3_third_singlet_isolated": bool(
                float(metadata.get("third_singlet_gap_Eh", -float("inf")))
                >= thresholds.minimum_third_singlet_gap_Eh
            ),
        }
        records.append(
            {
                "index": spot.index,
                "angle_degrees": spot.angle_degrees,
                "source_SA2_file": str(spot.source_sa2),
                "source_SA2_sha256": spot.source_sa2_sha256,
                "SA3_file": str(spot.target_sa3.resolve()),
                "SA3_sha256": sha256_file(spot.target_sa3),
                "geometry_hash": spot.geometry_hash,
                "checks": checks,
                "pass": bool(all(checks.values())),
                "projector_singular_values": singular_values,
                "projector_sigma_min": float(np.min(singular_values)),
                "projector_leakage": leakage,
                "third_root_SA2_weight": third_weight,
                "SA2_SA3_overlap_magnitude_squared_diagnostic": np.abs(overlap) ** 2,
                "SA3_metadata": metadata,
            }
        )
    return {
        "status": (
            "PASS_SA3_WEDGE_ARBITRATION"
            if all(item["pass"] for item in records)
            else "STOP_SA3_WEDGE_ARBITRATION_FAILURE"
        ),
        "symmetry_argument": (
            "The 40-point ring has 9-degree spacing. C5 rotations reduce indices "
            "modulo 8 and reflection maps residue j to 8-j, leaving residues 0..4."
        ),
        "records": records,
        "minimum_projector_sigma": min(
            item["projector_sigma_min"] for item in records
        ),
        "maximum_projector_leakage": max(
            item["projector_leakage"] for item in records
        ),
        "maximum_third_root_SA2_weight": max(
            item["third_root_SA2_weight"] for item in records
        ),
        "minimum_SA3_third_singlet_gap_Eh": min(
            float(item["SA3_metadata"]["third_singlet_gap_Eh"])
            for item in records
        ),
    }


def polar_unitary(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left, singular_values, right_h = np.linalg.svd(np.asarray(matrix))
    return left @ right_h, singular_values


def wilson_product(overlaps: Sequence[np.ndarray]) -> np.ndarray:
    if not overlaps:
        raise ValueError("Wilson product requires at least one edge")
    size = int(np.asarray(overlaps[0]).shape[0])
    result = np.eye(size, dtype=np.complex128)
    for matrix in overlaps:
        transport, _ = polar_unitary(np.asarray(matrix))
        result = result @ transport
    return result


def wrapped_angle_difference(left: float, right: float) -> float:
    return float(abs(np.angle(np.exp(1j * (float(left) - float(right))))))


def eigenphases(matrix: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvals(np.asarray(matrix, dtype=np.complex128))
    values = np.asarray(
        [value / abs(value) if abs(value) > 1.0e-14 else value for value in values]
    )
    return np.sort(np.angle(values))


def phase_set_error(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("phase sets have different dimensions")
    errors = []
    for permutation in itertools.permutations(range(left_array.size)):
        errors.append(
            max(
                wrapped_angle_difference(left_array[index], right_array[target])
                for index, target in enumerate(permutation)
            )
        )
    return float(min(errors))


def wilson_invariants(matrix: np.ndarray) -> Dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return {
        "trace": complex(np.trace(matrix)),
        "determinant": complex(np.linalg.det(matrix)),
        "eigenphases_rad": eigenphases(matrix),
        "unitarity_error": float(
            np.max(np.abs(matrix.conj().T @ matrix - np.eye(matrix.shape[0])))
        ),
    }


def invariant_error(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    conjugate_expected: bool = False,
) -> Dict[str, float]:
    expected_trace = complex(expected["trace"])
    expected_det = complex(expected["determinant"])
    expected_phases = np.asarray(expected["eigenphases_rad"], dtype=float)
    if conjugate_expected:
        expected_trace = np.conj(expected_trace)
        expected_det = np.conj(expected_det)
        expected_phases = -expected_phases
    return {
        "trace_error": float(abs(complex(observed["trace"]) - expected_trace)),
        "determinant_error": float(
            abs(complex(observed["determinant"]) - expected_det)
        ),
        "eigenphase_set_error_rad": phase_set_error(
            observed["eigenphases_rad"], expected_phases
        ),
    }


def maximum_error(record: Mapping[str, float]) -> float:
    return float(max(float(value) for value in record.values()))


def random_unitary(size: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 1.0e-14
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q @ np.diag(np.conj(phases))


def gauge_audit(
    overlaps: Sequence[np.ndarray], reference: Mapping[str, Any]
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    npoints = len(overlaps)
    size = int(np.asarray(overlaps[0]).shape[0])
    for seed in GAUGE_SEEDS:
        rng = np.random.default_rng(seed)
        gauges = [random_unitary(size, rng) for _ in range(npoints)]
        transformed = [
            gauges[index].conj().T
            @ np.asarray(overlaps[index])
            @ gauges[(index + 1) % npoints]
            for index in range(npoints)
        ]
        invariants = wilson_invariants(wilson_product(transformed))
        error = invariant_error(invariants, reference)
        records.append(
            {
                "seed": seed,
                "invariants": invariants,
                "errors": error,
                "maximum_error": maximum_error(error),
            }
        )
    return {
        "records": records,
        "maximum_error": max(item["maximum_error"] for item in records),
    }


def cyclic_start_audit(
    overlaps: Sequence[np.ndarray], reference: Mapping[str, Any]
) -> Dict[str, Any]:
    errors: List[Dict[str, float]] = []
    values = list(overlaps)
    for offset in range(len(values)):
        shifted = values[offset:] + values[:offset]
        invariants = wilson_invariants(wilson_product(shifted))
        errors.append(invariant_error(invariants, reference))
    return {
        "number_of_starts": len(values),
        "maximum_error": max(maximum_error(item) for item in errors),
    }


def analyze_projector_mesh(
    args: argparse.Namespace,
    authorization: Mapping[str, Any],
    npoints: int,
    level_index: int,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    loop_root = args.loop_root.resolve()
    surface = antecedent.load_surface(antecedent.surface_npz_path(loop_root))
    loop = antecedent.build_loop_specs()[0]
    points = antecedent.loop_points(
        loop_root, surface, loop, level_index, npoints
    )
    source_authorization = antecedent.validate_source_authorization(
        loop_root, args.source_root, require_pass=True
    )
    center_geometry = np.asarray(surface["center_geometry_angstrom"], dtype=float)
    center_energy = float(source_authorization["center_average_energy_Eh"])
    base_thresholds = antecedent.Thresholds()
    solutions: List[Dict[str, Any]] = []
    molecules: List[Any] = []
    point_records: List[Dict[str, Any]] = []
    for index, point in enumerate(points):
        solution = antecedent.load_validated_point(point, surface, BASIS_KEY)
        geometry = antecedent.surface_geometry(surface, point.ux, point.uy)
        checks, metrics = antecedent.production_point_checks(
            solution,
            geometry,
            center_geometry,
            center_energy,
            base_thresholds,
        )
        solutions.append(solution)
        molecules.append(
            eng.make_mol(geometry, BASIS_KEY, args.memory_mb, verbose=0)
        )
        point_records.append(
            {
                "index": index,
                "angle_degrees": 360.0 * index / npoints,
                "geometry_hash": point.geometry_hash,
                "solution_file": str(point.target.resolve()),
                "solution_sha256": sha256_file(point.target),
                "checks": checks,
                "all_checks_pass": bool(all(checks.values())),
                "metrics": metrics,
                "internal_pair_gap_Eh": float(
                    solution["metadata"]["adiabatic_gap_Eh"]
                ),
                "third_singlet_gap_Eh": float(
                    solution["metadata"]["third_singlet_gap_Eh"]
                ),
            }
        )

    forward_overlaps: List[np.ndarray] = []
    edge_records: List[Dict[str, Any]] = []
    for index in range(npoints):
        neighbor = (index + 1) % npoints
        matrix = ov.exact_overlap_matrix(
            solutions[index],
            solutions[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        if matrix.shape != (NROOTS_SA2, NROOTS_SA2):
            raise RuntimeError(f"unexpected pair-overlap shape: {matrix.shape}")
        forward_overlaps.append(matrix)
        _, pair_singular_values = polar_unitary(matrix)
        pair_sigma_min = float(np.min(pair_singular_values))
        pair_leakage = float(1.0 - np.mean(pair_singular_values**2))
        active = antecedent.active_edge_metrics(
            solutions[index],
            solutions[neighbor],
            molecules[index],
            molecules[neighbor],
        )
        projector_strict = bool(
            pair_sigma_min >= thresholds.pair_edge_sigma_minimum
            and pair_leakage <= thresholds.pair_edge_leakage_maximum
            and active["sigma_min"] >= thresholds.active_edge_sigma_minimum
            and active["leakage"] <= thresholds.active_edge_leakage_maximum
        )
        edge_records.append(
            {
                "from_index": index,
                "to_index": neighbor,
                "pair_singular_values": pair_singular_values,
                "pair_sigma_min": pair_sigma_min,
                "pair_leakage": pair_leakage,
                "active_singular_values": active["singular_values"],
                "active_sigma_min": active["sigma_min"],
                "active_leakage": active["leakage"],
                "projector_strict": projector_strict,
                "individual_root_assignment_used": False,
                "signed_single_state_overlap_recorded": False,
            }
        )

    forward_matrix = wilson_product(forward_overlaps)
    forward_invariants = wilson_invariants(forward_matrix)
    gauges = gauge_audit(forward_overlaps, forward_invariants)
    cyclic = cyclic_start_audit(forward_overlaps, forward_invariants)

    reversed_indices = [0] + list(range(npoints - 1, 0, -1))
    reverse_overlaps: List[np.ndarray] = []
    for position, index in enumerate(reversed_indices):
        neighbor = reversed_indices[(position + 1) % npoints]
        reverse_overlaps.append(
            ov.exact_overlap_matrix(
                solutions[index],
                solutions[neighbor],
                molecules[index],
                molecules[neighbor],
            )
        )
    reverse_invariants = wilson_invariants(wilson_product(reverse_overlaps))
    orientation_error = invariant_error(
        reverse_invariants, forward_invariants, conjugate_expected=True
    )

    all_points_pass = all(item["all_checks_pass"] for item in point_records)
    all_edges_pass = all(item["projector_strict"] for item in edge_records)
    gauge_pass = gauges["maximum_error"] <= thresholds.maximum_gauge_invariant_error
    cyclic_pass = (
        cyclic["maximum_error"] <= thresholds.maximum_cyclic_start_invariant_error
    )
    orientation_pass = (
        maximum_error(orientation_error)
        <= thresholds.maximum_orientation_invariant_error
    )
    return {
        "mesh_npoints": npoints,
        "level_index": level_index,
        "angular_step_degrees": 360.0 / npoints,
        "point_records": point_records,
        "edge_records": edge_records,
        "all_points_pass": all_points_pass,
        "all_projector_edges_pass": all_edges_pass,
        "minimum_internal_pair_gap_Eh": min(
            item["internal_pair_gap_Eh"] for item in point_records
        ),
        "maximum_internal_pair_gap_Eh": max(
            item["internal_pair_gap_Eh"] for item in point_records
        ),
        "minimum_third_singlet_gap_Eh": min(
            item["third_singlet_gap_Eh"] for item in point_records
        ),
        "minimum_pair_sigma": min(item["pair_sigma_min"] for item in edge_records),
        "maximum_pair_leakage": max(item["pair_leakage"] for item in edge_records),
        "minimum_active_sigma": min(
            item["active_sigma_min"] for item in edge_records
        ),
        "maximum_active_leakage": max(
            item["active_leakage"] for item in edge_records
        ),
        "wilson_invariants": forward_invariants,
        "random_U2_gauge_audit": gauges,
        "cyclic_start_audit": cyclic,
        "reverse_orientation_invariants": reverse_invariants,
        "orientation_errors": orientation_error,
        "gauge_pass": bool(gauge_pass),
        "cyclic_start_pass": bool(cyclic_pass),
        "orientation_pass": bool(orientation_pass),
        "rank2_wilson_spectrum_recorded": True,
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
    }


def mesh_comparison(
    coarse: Mapping[str, Any], fine: Mapping[str, Any]
) -> Dict[str, Any]:
    left = coarse["wilson_invariants"]
    right = fine["wilson_invariants"]
    return {
        "coarse_npoints": coarse["mesh_npoints"],
        "fine_npoints": fine["mesh_npoints"],
        "eigenphase_set_difference_rad": phase_set_error(
            left["eigenphases_rad"], right["eigenphases_rad"]
        ),
        "trace_difference": float(abs(complex(left["trace"]) - complex(right["trace"]))),
        "determinant_difference": float(
            abs(complex(left["determinant"]) - complex(right["determinant"]))
        ),
    }


def classify_result(
    sa3: Mapping[str, Any],
    meshes: Sequence[Mapping[str, Any]],
    convergence: Mapping[str, Any],
    thresholds: Thresholds,
) -> Tuple[str, str]:
    if sa3.get("status") != "PASS_SA3_WEDGE_ARBITRATION":
        return (
            "STOP_SA3_WEDGE_ARBITRATION_FAILURE",
            "At least one independent SA3 wedge point does not preserve an isolated rank-2 pair.",
        )
    if not all(bool(mesh.get("all_points_pass")) for mesh in meshes):
        return (
            "STOP_PROJECTOR_POINT_QUALITY_FAILURE",
            "At least one reused SA2 loop vertex fails the frozen electronic or external-isolation checks.",
        )
    if not all(bool(mesh.get("all_projector_edges_pass")) for mesh in meshes):
        return (
            "STOP_PROJECTOR_EDGE_FIDELITY_FAILURE",
            "At least one rank-2 or active-space edge fails the frozen subspace-fidelity thresholds.",
        )
    if not all(
        bool(mesh.get("gauge_pass"))
        and bool(mesh.get("orientation_pass"))
        and bool(mesh.get("cyclic_start_pass"))
        for mesh in meshes
    ):
        return (
            "STOP_PROJECTOR_INVARIANCE_FAILURE",
            "A U(2) gauge, orientation, or cyclic-start invariant fails its numerical threshold.",
        )
    converged = bool(
        float(convergence["eigenphase_set_difference_rad"])
        <= thresholds.maximum_40_80_eigenphase_difference_rad
        and float(convergence["trace_difference"])
        <= thresholds.maximum_40_80_trace_difference
        and float(convergence["determinant_difference"])
        <= thresholds.maximum_40_80_determinant_difference
    )
    if not converged:
        return (
            "REFINE_RANK2_PROJECTOR_MESH",
            "The isolated projector passes, but its 40/80 Wilson spectrum has not reached the preregistered mesh tolerance.",
        )
    return (
        "PASS_RANK2_PROJECTOR_QUALIFICATION",
        "The isolated two-state manifold passes independent SA3 rank, edge fidelity, U(2) invariance, and mesh-convergence controls.",
    )


def validate_existing_decision(root: Path) -> Dict[str, Any]:
    path = decision_path(root)
    result = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "CI_count_from_rank2_Wilson_claimed": False,
        "manuscript_claim_authorized": False,
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError("existing rank-2 decision failed immutable-field validation")
    for record in result.get("SA3_wedge_arbitration", {}).get("records", []):
        path = Path(str(record.get("SA3_file", ""))).resolve()
        if not path.exists() or sha256_file(path) != record.get("SA3_sha256"):
            raise RuntimeError(f"SA3 spot changed after decision: {path}")
    for mesh in result.get("projector_meshes", []):
        for point in mesh.get("point_records", []):
            path = Path(str(point.get("solution_file", ""))).resolve()
            if not path.exists() or sha256_file(path) != point.get("solution_sha256"):
                raise RuntimeError(f"SA2 mesh point changed after decision: {path}")
    return result


def run_projector(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != BASIS_KEY:
        raise SystemExit("This protocol is locked to cc-pVTZ")
    validate_protocol()
    root = args.root.resolve()
    authorization = validate_authorization(
        root, args.source_root, args.loop_root, args.screen_root
    )
    create_or_validate_manifest(root, authorization)
    require_pyscf()
    output = decision_path(root)
    if output.exists():
        result = validate_existing_decision(root)
        print(f"CPDC_RANK2_PROJECTOR={result['status']}", flush=True)
        return result

    thresholds = Thresholds()
    spots = spot_specs(root, authorization)
    run_sa3_batch(args, spots)
    sa3 = analyze_sa3_spots(spots, args.memory_mb, thresholds)
    print("\nCPDC RANK-2 SA3 SYMMETRY-WEDGE ARBITRATION", flush=True)
    print(f"Status: {sa3['status']}", flush=True)
    print(
        f"sigma={sa3['minimum_projector_sigma']:.9f}; "
        f"leakage={sa3['maximum_projector_leakage']:.3e}; "
        f"third_weight={sa3['maximum_third_root_SA2_weight']:.3e}; "
        f"min_gap23={sa3['minimum_SA3_third_singlet_gap_Eh']:.6e} Eh",
        flush=True,
    )

    meshes: List[Dict[str, Any]] = []
    for level_index, npoints in enumerate(MESH_SIZES):
        print(f"\nPROJECTOR MESH npoints={npoints}", flush=True)
        mesh = analyze_projector_mesh(
            args, authorization, npoints, level_index, thresholds
        )
        meshes.append(mesh)
        phases = np.asarray(mesh["wilson_invariants"]["eigenphases_rad"])
        print(
            f"pair_sigma={mesh['minimum_pair_sigma']:.9f} "
            f"pair_leak={mesh['maximum_pair_leakage']:.3e} "
            f"gap23={mesh['minimum_third_singlet_gap_Eh']:.6e} Eh "
            f"wilson_phases=({phases[0]:+.9e},{phases[1]:+.9e}) rad "
            f"gauge_error={mesh['random_U2_gauge_audit']['maximum_error']:.3e}",
            flush=True,
        )

    comparisons = [mesh_comparison(meshes[0], meshes[1]), mesh_comparison(meshes[1], meshes[2])]
    convergence = comparisons[-1]
    status, reason = classify_result(sa3, meshes, convergence, thresholds)
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "reason": reason,
        "authorization_file": str(authorization_path(root).resolve()),
        "authorization_sha256": sha256_file(authorization_path(root)),
        "manifest_sha256": sha256_file(manifest_path(root)),
        "thresholds": asdict(thresholds),
        "SA3_wedge_arbitration": sa3,
        "projector_meshes": meshes,
        "mesh_comparisons": comparisons,
        "mesh_40_80_convergence": convergence,
        "scalar_phase_evaluated": False,
        "signed_single_state_overlap_recorded": False,
        "individual_root_assignment_used": False,
        "rank2_wilson_spectrum_recorded": True,
        "CI_count_from_rank2_Wilson_claimed": False,
        "manuscript_claim_authorized": False,
        "scientific_scope": (
            "This decision qualifies or rejects identity and transport of the isolated "
            "rank-2 manifold. It neither resolves either individual adiabatic root nor "
            "assigns a scalar Berry phase or an internal CI count."
        ),
        "authorized_next_step": (
            "If PASS, perform an independently seeded opposed-history replication of "
            "the five SA3 wedge points and the rank-2 Wilson spectra before manuscript use."
            if status == "PASS_RANK2_PROJECTOR_QUALIFICATION"
            else "Stop and follow the status-specific diagnostic; do not expose scalar phase."
        ),
    }
    write_json(output, result)
    print("\nCPDC RANK-2 PROJECTOR QUALIFICATION", flush=True)
    print(f"Status: {status}", flush=True)
    print(f"Reason: {reason}", flush=True)
    print(
        "40/80: "
        f"dphase={convergence['eigenphase_set_difference_rad']:.3e} rad "
        f"dtrace={convergence['trace_difference']:.3e} "
        f"ddet={convergence['determinant_difference']:.3e}",
        flush=True,
    )
    print(f"Decision file: {output}", flush=True)
    print(f"CPDC_RANK2_PROJECTOR={status}", flush=True)
    return result


def _synthetic_overlaps() -> List[np.ndarray]:
    angles = (0.07, -0.11, 0.13, -0.05)
    result: List[np.ndarray] = []
    for index, angle in enumerate(angles):
        unitary = np.asarray(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ],
            dtype=np.complex128,
        )
        singular = np.diag([0.995 - 0.001 * index, 0.991 - 0.001 * index])
        result.append(singular @ unitary)
    return result


def selftest() -> None:
    validate_protocol()
    components = {
        "antecedent": sha256_file(Path(antecedent.__file__)),
        "screen": sha256_file(Path(gap_screen.__file__)),
        "engine": sha256_file(Path(eng.__file__)),
        "overlap": sha256_file(Path(ov.__file__)),
    }
    expected = {
        "antecedent": EXPECTED_ANTECEDENT_DRIVER_SHA256,
        "screen": EXPECTED_SCREEN_DRIVER_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
        "overlap": EXPECTED_OVERLAP_SHA256,
    }
    if components != expected:
        raise RuntimeError(f"frozen component selftest failed: {components}")
    if [360.0 * index / SPOT_MESH_SIZE for index in SPOT_INDICES] != [
        0.0,
        9.0,
        18.0,
        27.0,
        36.0,
    ]:
        raise RuntimeError("symmetry-wedge selftest failed")
    covered_indices = set()
    for representative in SPOT_INDICES:
        for rotation in range(antecedent.N_RING):
            covered_indices.add((representative + 8 * rotation) % SPOT_MESH_SIZE)
            covered_indices.add((-representative + 8 * rotation) % SPOT_MESH_SIZE)
    if covered_indices != set(range(SPOT_MESH_SIZE)):
        raise RuntimeError("five SA3 representatives do not cover the 40-point ring")

    snapshot = json.loads(
        Path(__file__)
        .resolve()
        .with_name(antecedent.FROZEN_SOURCE_SNAPSHOT_FILENAME)
        .read_text()
    )
    coefficients_q = np.asarray(
        snapshot["candidate_coefficients_angstrom"], dtype=float
    )
    center = np.asarray(eng.REFERENCE_GEOMETRY, dtype=float)
    candidate = eng.geometry_from_coefficients(coefficients_q)
    anchors = np.asarray(
        [eng.symmetry_replica(candidate, replica) for replica in range(antecedent.N_RING)]
    )
    frequencies, fourier = antecedent._fourier_coefficients(
        anchors - center[None, :, :]
    )
    synthetic_surface = {
        "center_geometry_angstrom": center,
        "frequencies": frequencies,
        "fourier_coefficients": fourier,
    }
    maximum_rotation_error = 0.0
    maximum_reflection_error = 0.0
    for index in range(SPOT_MESH_SIZE):
        theta = 2.0 * math.pi * index / SPOT_MESH_SIZE
        geometry = antecedent.surface_geometry(
            synthetic_surface,
            antecedent.INNER_RADIUS_U * math.cos(theta),
            antecedent.INNER_RADIUS_U * math.sin(theta),
        )
        rotated_theta = theta + 2.0 * math.pi / antecedent.N_RING
        rotated = antecedent.surface_geometry(
            synthetic_surface,
            antecedent.INNER_RADIUS_U * math.cos(rotated_theta),
            antecedent.INNER_RADIUS_U * math.sin(rotated_theta),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(np.max(np.abs(rotated - eng.symmetry_replica(geometry, 1)))),
        )
        reflected = antecedent.surface_geometry(
            synthetic_surface,
            antecedent.INNER_RADIUS_U * math.cos(-theta),
            antecedent.INNER_RADIUS_U * math.sin(-theta),
        )
        expected_reflection = center + eng.reflect_displacement(geometry - center)
        maximum_reflection_error = max(
            maximum_reflection_error,
            float(np.max(np.abs(reflected - expected_reflection))),
        )
    if maximum_rotation_error > 2.0e-12 or maximum_reflection_error > 2.0e-12:
        raise RuntimeError(
            "surface symmetry selftest failed: "
            f"rotation={maximum_rotation_error:.3e}, "
            f"reflection={maximum_reflection_error:.3e}"
        )

    overlaps = _synthetic_overlaps()
    forward = wilson_invariants(wilson_product(overlaps))
    gauges = gauge_audit(overlaps, forward)
    cyclic = cyclic_start_audit(overlaps, forward)
    reverse = [matrix.conj().T for matrix in reversed(overlaps)]
    reverse_invariants = wilson_invariants(wilson_product(reverse))
    orientation = invariant_error(
        reverse_invariants, forward, conjugate_expected=True
    )
    if gauges["maximum_error"] > 1.0e-12:
        raise RuntimeError("synthetic U(2) gauge selftest failed")
    if cyclic["maximum_error"] > 1.0e-12:
        raise RuntimeError("synthetic cyclic-start selftest failed")
    if maximum_error(orientation) > 1.0e-12:
        raise RuntimeError("synthetic orientation selftest failed")

    thresholds = Thresholds()
    good_sa3 = {"status": "PASS_SA3_WEDGE_ARBITRATION"}
    good_mesh = {
        "all_points_pass": True,
        "all_projector_edges_pass": True,
        "gauge_pass": True,
        "orientation_pass": True,
        "cyclic_start_pass": True,
    }
    good_conv = {
        "eigenphase_set_difference_rad": 1.0e-5,
        "trace_difference": 1.0e-5,
        "determinant_difference": 1.0e-5,
    }
    branches = {
        classify_result(good_sa3, [good_mesh] * 3, good_conv, thresholds)[0],
        classify_result(
            {"status": "STOP"}, [good_mesh] * 3, good_conv, thresholds
        )[0],
        classify_result(
            good_sa3,
            [{**good_mesh, "all_points_pass": False}] + [good_mesh] * 2,
            good_conv,
            thresholds,
        )[0],
        classify_result(
            good_sa3,
            [{**good_mesh, "all_projector_edges_pass": False}] + [good_mesh] * 2,
            good_conv,
            thresholds,
        )[0],
        classify_result(
            good_sa3,
            [{**good_mesh, "gauge_pass": False}] + [good_mesh] * 2,
            good_conv,
            thresholds,
        )[0],
        classify_result(
            good_sa3,
            [good_mesh] * 3,
            {**good_conv, "eigenphase_set_difference_rad": 0.1},
            thresholds,
        )[0],
    }
    expected_branches = {
        "PASS_RANK2_PROJECTOR_QUALIFICATION",
        "STOP_SA3_WEDGE_ARBITRATION_FAILURE",
        "STOP_PROJECTOR_POINT_QUALITY_FAILURE",
        "STOP_PROJECTOR_EDGE_FIDELITY_FAILURE",
        "STOP_PROJECTOR_INVARIANCE_FAILURE",
        "REFINE_RANK2_PROJECTOR_MESH",
    }
    if branches != expected_branches:
        raise RuntimeError(f"decision-branch selftest failed: {branches}")
    print("CPDC_RANK2_PROJECTOR_V1_SELFTEST=PASS")


def add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--loop-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)


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

    worker = subparsers.add_parser("sa3-worker")
    worker.add_argument("--target", type=Path, required=True)
    worker.add_argument("--source-sa2", type=Path, required=True)
    worker.add_argument("--source-sha256", required=True)
    worker.add_argument("--geometry-hash", required=True)
    worker.add_argument("--spot-index", type=int, required=True)
    worker.add_argument("--angle-degrees", type=float, required=True)
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
        run_projector(args)
    elif args.command == "sa3-worker":
        sa3_worker(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
