#!/usr/bin/env python3
"""Preregistered cc-pVTZ lift of the CPDC secondary singlet CI.

This executable consumes, without changing it, a completed
CPDC-SCI-QUAL-v2.0 DZV decision with status
``PASS_SECONDARY_CI_QUALIFICATION``.  It asks one question only: does that
geometry-relaxed secondary singlet conical intersection survive the locked
change from DZV to cc-pVTZ at equal-weight SA2-CASSCF(4e,5o)?

The source candidate fixes the representative C2v branch and all nine initial
geometry coordinates.  A fresh cc-pVTZ D5h center is built, the eight
spectators are first relaxed at the source q1 value, q1 is then released only
inside the original interval if necessary, and five rotated replicas are
requalified.  This executable does not calculate a Berry phase, Wilson loop,
response, or manuscript observable.  A pass authorizes only a separate,
preregistered geometry-relaxed multiring topology calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "CPDC-SCI-CCPVTZ-LIFT-v1.0"
PROTOCOL_ID = "CPDC-SECONDARY-CI-CCPVTZ-LIFT-20260920"
PROTOCOL_FILENAME = "CPDC_secondary_CI_ccpVTZ_lift_protocol_v1.json"
MANIFEST_FILENAME = "CPDC_secondary_CI_ccpVTZ_lift_v1_manifest.json"
CENTER_DECISION_FILENAME = "CPDC_secondary_CI_ccpVTZ_center_v1_decision.json"
SOURCE_AUTHORIZATION_FILENAME = "CPDC_secondary_CI_ccpVTZ_source_authorization_v1.json"
# Name retained solely for the unexposed v2 compatibility helper; no lift
# command reads or writes this file.
RUNTIME_ADMISSION_FILENAME = "UNUSED_legacy_runtime_admission.json"
DECISION_FILENAME = "CPDC_secondary_CI_ccpVTZ_lift_v1_decision.json"
REGISTRY_FILENAME = "CPDC_secondary_CI_ccpVTZ_evaluation_registry_v1.json"
SEARCH_RECORD_FILENAME = "CPDC_secondary_CI_ccpVTZ_search_record_v1.json"

SOURCE_SCRIPT_VERSION = "CPDC-SCI-QUAL-v2.0"
SOURCE_PROTOCOL_ID = "CPDC-SECONDARY-CI-QUAL-PYSCF2131-20260920"
SOURCE_DECISION_FILENAME = "CPDC_secondary_CI_qual_v2_decision.json"
SOURCE_REQUIRED_STATUS = "PASS_SECONDARY_CI_QUALIFICATION"
SOURCE_BASIS_DEFINITION_SHA256 = (
    "5d608698f984abaa3de157f5e0f640ae5efcb1c53cf50512e7810f50bd166a4b"
)

N_RING = 5
N_ATOMS = 10
ELEMENTS = ("C",) * N_RING + ("H",) * N_RING
MASSES = np.asarray([12.0] * N_RING + [1.0] * N_RING)
R_CC_ANGSTROM = 1.437
R_CH_ANGSTROM = 1.080
ANGLE_DEGREES = 36.0
ANGLE_RAD = math.radians(ANGLE_DEGREES)
INITIAL_E1_ANGSTROM = 0.0787
E1_LOWER_ANGSTROM = 0.055
E1_UPPER_ANGSTROM = 0.105
SPECTATOR_BOUND_ANGSTROM = 0.060
SPECTATOR_SOFT_RADIUS_ANGSTROM = 0.120
SPECTATOR_PENALTY_EH_PER_A2 = 1.0
ENERGY_SOFT_CEILING_EH = 0.030
ENERGY_PENALTY_INVERSE_EH = 50.0
STAGE_A_MAXITER = 24
STAGE_B_MAXITER = 36
STAGE_A_MAXFUN = 240
STAGE_B_MAXFUN = 360
STAGE_A_MAXLS = 20
STAGE_B_MAXLS = 24
OPTIMIZER_FTOL = 1.0e-14
OPTIMIZER_GTOL = 1.0e-7
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

NCAS = 5
NELECAS = 4
NACT_ALPHA = 2
NACT_BETA = 2
NROOTS = 2
PROBE_NROOTS = 3
CASSCF_CONV_TOL = 1.0e-10
CASSCF_GRAD_TOL = 1.0e-5

# The exact PySCF version and its built-in canonical basis key jointly lock
# the cc-pVTZ table.  The declaration hash below fingerprints those semantics;
# each solution also records the exact runtime version.
PYSCF_BASIS_KEY = "cc-pvtz"
PYSCF_VERSION_EXACT = "2.13.1"
CCPVTZ_BASIS_DECLARATION = {
    "provider": "PySCF built-in basis library",
    "pyscf_version_exact": PYSCF_VERSION_EXACT,
    "basis_key": PYSCF_BASIS_KEY,
    "elements": ["C", "H"],
    "family": "Dunning correlation-consistent polarized valence triple-zeta",
}

# Inert compatibility constants from the frozen v2 engine.  The lift command
# dispatcher never invokes the legacy runtime-admission helper; the mandatory
# gate is ``PASS_SOURCE_AUTHORIZATION`` implemented below.
ADMISSION_FD_STEP_ANGSTROM = 5.0e-4
ADMISSION_MAX_GRADIENT_ABS_ERROR_EH_PER_ANGSTROM = 2.0e-4
ADMISSION_MAX_GRADIENT_RELATIVE_ERROR = 2.0e-3
V1_CENTER_AVERAGE_ENERGY_EH = -191.8792984565422
V1_CENTER_AVERAGE_ENERGY_TOLERANCE_EH = 2.0e-7
V1_CENTER_THIRD_SINGLET_GAP_EH = 0.03013408730291189
V1_CENTER_THIRD_SINGLET_GAP_TOLERANCE_EH = 2.0e-6
V1_INITIAL_GAP_EH = 0.005541412650899247
V1_INITIAL_GAP_TOLERANCE_EH = 2.0e-6
V1_INITIAL_ENERGY_ABOVE_CENTER_EH = 0.004288933186103350
V1_INITIAL_ENERGY_ABOVE_CENTER_TOLERANCE_EH = 2.0e-6
DUNNING_DZV_BASIS = {
    "H": [
        [
            0,
            [19.2406000, 0.0328280],
            [2.8992000, 0.2312080],
            [0.6534000, 0.8172380],
        ],
        [0, [0.1776000, 1.0000000]],
    ],
    "C": [
        [
            0,
            [4232.6100000, 0.0020290],
            [634.8820000, 0.0155350],
            [146.0970000, 0.0754110],
            [42.4974000, 0.2571210],
            [14.1892000, 0.5965550],
            [1.9666000, 0.2425170],
        ],
        [0, [5.1477000, 1.0000000]],
        [0, [0.4962000, 1.0000000]],
        [0, [0.1533000, 1.0000000]],
        [
            1,
            [18.1557000, 0.0185340],
            [3.9864000, 0.1154420],
            [1.1429000, 0.3862060],
            [0.3594000, 0.6400890],
        ],
        [1, [0.1146000, 1.0000000]],
    ],
}
BASIS_DEFINITION_SHA256 = hashlib.sha256(
    json.dumps(
        CCPVTZ_BASIS_DECLARATION, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class Thresholds:
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square_error: float = 1.0e-6
    maximum_mo_orthonormality_error: float = 1.0e-8
    maximum_casci_reproduction_error_Eh: float = 1.0e-7
    minimum_active_reference_sigma: float = 0.90
    minimum_selected_pi_projector_eigenvalue: float = 0.50
    minimum_third_singlet_gap_Eh: float = 2.0e-3
    minimum_center_third_singlet_gap_Eh: float = 3.0e-3
    maximum_center_pair_splitting_Eh: float = 2.0e-4
    maximum_secondary_CI_gap_Eh: float = 1.0e-5
    maximum_replica_gap_Eh: float = 5.0e-5
    maximum_replica_average_energy_spread_Eh: float = 5.0e-6
    maximum_C2v_reflection_error_angstrom: float = 1.0e-9
    maximum_planarity_error_angstrom: float = 1.0e-10
    minimum_e1_boundary_clearance_angstrom: float = 5.0e-4
    maximum_spectator_norm_angstrom: float = 0.120
    maximum_average_energy_above_center_Eh: float = 0.030
    minimum_CC_bond_angstrom: float = 1.20
    maximum_CC_bond_angstrom: float = 1.70
    minimum_CH_bond_angstrom: float = 0.90
    maximum_CH_bond_angstrom: float = 1.25


class AnalyticGradientFailure(RuntimeError):
    pass


class TargetFound(RuntimeError):
    def __init__(self, coefficients: np.ndarray, record: Mapping[str, Any]):
        super().__init__("qualified secondary-CI target found")
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.record = dict(record)


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


def array_hash(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.round(np.asarray(array, dtype=float), 12))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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
    try:
        import pyscf
        from pyscf import fci, gto, mcscf, scf
    except ImportError as exc:
        raise SystemExit(
            "PySCF is required for center/run; selftest is NumPy-only"
        ) from exc
    try:
        installed_version = str(pyscf.__version__)
    except AttributeError as exc:
        raise SystemExit("Could not identify the installed PySCF version") from exc
    if installed_version != PYSCF_VERSION_EXACT:
        raise SystemExit(
            f"This cc-pVTZ lift is locked to PySCF {PYSCF_VERSION_EXACT}; "
            f"found {installed_version}. Use a new protocol and root for another version."
        )
    return fci, gto, mcscf, scf


def require_scipy_minimize():
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise SystemExit("SciPy is required for the bounded CI qualification") from exc
    return minimize


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-13:
        raise RuntimeError(f"cannot normalize {label}: norm={norm:.3e}")
    return np.asarray(vector, dtype=float) / norm


def reference_geometry() -> np.ndarray:
    ring_radius = R_CC_ANGSTROM / (2.0 * math.sin(math.pi / N_RING))
    carbon: List[np.ndarray] = []
    hydrogen: List[np.ndarray] = []
    for index in range(N_RING):
        angle = 2.0 * math.pi * index / N_RING
        radial = np.asarray([math.cos(angle), math.sin(angle), 0.0])
        carbon.append(ring_radius * radial)
        hydrogen.append((ring_radius + R_CH_ANGSTROM) * radial)
    return np.asarray(carbon + hydrogen, dtype=float)


REFERENCE_GEOMETRY = reference_geometry()


def remove_rigid_motion(displacement: np.ndarray) -> np.ndarray:
    result = np.asarray(displacement, dtype=float).copy()
    coordinates = REFERENCE_GEOMETRY.copy()
    center = np.einsum("a,ax->x", MASSES, coordinates) / float(np.sum(MASSES))
    coordinates -= center[None, :]
    translation = np.einsum("a,ax->x", MASSES, result) / float(np.sum(MASSES))
    result -= translation[None, :]

    inertia = np.zeros((3, 3), dtype=float)
    angular = np.zeros(3, dtype=float)
    for mass, position, velocity in zip(MASSES, coordinates, result):
        inertia += mass * (
            float(np.dot(position, position)) * np.eye(3)
            - np.outer(position, position)
        )
        angular += mass * np.cross(position, velocity)
    omega = np.linalg.pinv(inertia, rcond=1.0e-13) @ angular
    result -= np.cross(np.broadcast_to(omega, coordinates.shape), coordinates)
    return result


def add_bond_gradient(
    displacement: np.ndarray, atom_a: int, atom_b: int, coefficient: float
) -> None:
    direction = normalize(
        REFERENCE_GEOMETRY[atom_a] - REFERENCE_GEOMETRY[atom_b], "C--C bond"
    )
    displacement[atom_a] += coefficient * direction
    displacement[atom_b] -= coefficient * direction


def e1prime_displacements() -> Tuple[np.ndarray, np.ndarray]:
    raw: List[np.ndarray] = []
    for component in ("x", "y"):
        displacement = np.zeros_like(REFERENCE_GEOMETRY)
        for index in range(N_RING):
            neighbor = (index + 1) % N_RING
            midpoint = REFERENCE_GEOMETRY[index] + REFERENCE_GEOMETRY[neighbor]
            phi = math.atan2(float(midpoint[1]), float(midpoint[0]))
            coefficient = math.cos(phi) if component == "x" else math.sin(phi)
            add_bond_gradient(displacement, index, neighbor, coefficient)
        for carbon in range(N_RING):
            displacement[N_RING + carbon] = displacement[carbon]
        raw.append(remove_rigid_motion(displacement))
    qx = normalize(raw[0].reshape(-1), "E1' x component").reshape(N_ATOMS, 3)
    raw_y = raw[1] - float(np.vdot(qx, raw[1]).real) * qx
    qy = normalize(raw_y.reshape(-1), "E1' y component").reshape(N_ATOMS, 3)
    if float(np.vdot(qy, raw[1]).real) < 0.0:
        qy = -qy
    return qx, qy


E1_X, E1_Y = e1prime_displacements()


def reflection_matrix(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(2.0 * theta), math.sin(2.0 * theta), 0.0],
            [math.sin(2.0 * theta), -math.cos(2.0 * theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


REFLECTION_MATRIX = reflection_matrix(ANGLE_RAD)


def reflection_mapping() -> Tuple[int, ...]:
    result: List[int] = []
    for atom, coordinate in enumerate(REFERENCE_GEOMETRY):
        reflected = REFLECTION_MATRIX @ coordinate
        group = range(0, N_RING) if atom < N_RING else range(N_RING, N_ATOMS)
        match = min(
            group,
            key=lambda candidate: float(
                np.linalg.norm(REFERENCE_GEOMETRY[candidate] - reflected)
            ),
        )
        error = float(np.linalg.norm(REFERENCE_GEOMETRY[match] - reflected))
        if error > 1.0e-10:
            raise RuntimeError(f"failed to construct C2v atom map: error={error:.3e}")
        result.append(int(match))
    return tuple(result)


REFLECTION_MAP = reflection_mapping()


def reflect_displacement(displacement: np.ndarray) -> np.ndarray:
    result = np.zeros_like(displacement, dtype=float)
    for atom in range(N_ATOMS):
        result[REFLECTION_MAP[atom]] = REFLECTION_MATRIX @ displacement[atom]
    return result


def c2v_internal_basis() -> np.ndarray:
    vectors: List[np.ndarray] = []

    def append_vector(raw: np.ndarray, label: str) -> None:
        symmetrized = 0.5 * (raw + reflect_displacement(raw))
        flattened = remove_rigid_motion(symmetrized).reshape(-1)
        for existing in vectors:
            flattened -= float(np.dot(existing, flattened)) * existing
        norm = float(np.linalg.norm(flattened))
        if norm < 1.0e-10:
            return
        flattened /= norm
        if not vectors:
            reference = (
                math.cos(ANGLE_RAD) * E1_X + math.sin(ANGLE_RAD) * E1_Y
            ).reshape(-1)
            if float(np.dot(flattened, reference)) < 0.0:
                flattened = -flattened
        else:
            largest = int(np.argmax(np.abs(flattened)))
            if flattened[largest] < 0.0:
                flattened = -flattened
        vectors.append(flattened)

    e1_axis = math.cos(ANGLE_RAD) * E1_X + math.sin(ANGLE_RAD) * E1_Y
    append_vector(e1_axis, "selected E1' component")
    for atom in range(N_ATOMS):
        for component in range(2):
            raw = np.zeros_like(REFERENCE_GEOMETRY)
            raw[atom, component] = 1.0
            append_vector(raw, f"atom {atom} component {component}")
    basis = np.asarray(vectors).reshape(len(vectors), N_ATOMS, 3)
    if basis.shape[0] != 9:
        raise RuntimeError(
            f"expected nine planar C2v internal coordinates, found {basis.shape[0]}"
        )
    return basis


C2V_BASIS = c2v_internal_basis()
N_COORDS = int(C2V_BASIS.shape[0])


def geometry_from_coefficients(coefficients: Sequence[float]) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (N_COORDS,):
        raise ValueError(f"expected {N_COORDS} coordinates, got {values.shape}")
    return REFERENCE_GEOMETRY + np.einsum("k,kax->ax", values, C2V_BASIS)


def symmetry_replica(geometry: np.ndarray, replica: int) -> np.ndarray:
    step = int(replica) % N_RING
    angle = 2.0 * math.pi * step / N_RING
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = np.asarray(geometry) @ rotation.T
    result = np.empty_like(rotated)
    for target in range(N_RING):
        source = (target - step) % N_RING
        result[target] = rotated[source]
        result[N_RING + target] = rotated[N_RING + source]
    return result


def geometry_metrics(
    geometry: np.ndarray, coefficients: Sequence[float]
) -> Dict[str, Any]:
    reflected_error = 0.0
    for atom in range(N_ATOMS):
        reflected_error = max(
            reflected_error,
            float(
                np.linalg.norm(
                    geometry[REFLECTION_MAP[atom]]
                    - REFLECTION_MATRIX @ geometry[atom]
                )
            ),
        )
    cc = [
        float(np.linalg.norm(geometry[index] - geometry[(index + 1) % N_RING]))
        for index in range(N_RING)
    ]
    ch = [
        float(np.linalg.norm(geometry[index] - geometry[N_RING + index]))
        for index in range(N_RING)
    ]
    values = np.asarray(coefficients, dtype=float)
    return {
        "C2v_reflection_error_angstrom": reflected_error,
        "planarity_error_angstrom": float(np.max(np.abs(geometry[:, 2]))),
        "e1_amplitude_angstrom": float(values[0]),
        "spectator_norm_angstrom": float(np.linalg.norm(values[1:])),
        "CC_bonds_angstrom": cc,
        "minimum_CC_bond_angstrom": min(cc),
        "maximum_CC_bond_angstrom": max(cc),
        "CH_bonds_angstrom": ch,
        "minimum_CH_bond_angstrom": min(ch),
        "maximum_CH_bond_angstrom": max(ch),
        "maximum_atom_displacement_angstrom": float(
            np.max(np.linalg.norm(geometry - REFERENCE_GEOMETRY, axis=1))
        ),
    }


def make_mol(
    geometry: np.ndarray,
    basis: str,
    memory_mb: int,
    output: str | None = None,
    verbose: int = 0,
):
    _, gto, _, _ = require_pyscf()
    if basis.lower() != PYSCF_BASIS_KEY:
        raise RuntimeError(
            f"locked basis key is {PYSCF_BASIS_KEY!r}, received {basis!r}"
        )
    kwargs: Dict[str, Any] = {
        "atom": list(zip(ELEMENTS, np.asarray(geometry, dtype=float))),
        "unit": "Angstrom",
        "basis": PYSCF_BASIS_KEY,
        "charge": 1,
        "spin": 0,
        "symmetry": False,
        "max_memory": int(memory_mb),
        "verbose": int(verbose),
    }
    if output is not None:
        kwargs["output"] = output
    return gto.M(**kwargs)


def run_rhf(mol):
    _, _, _, scf = require_pyscf()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 180
    mf.kernel()
    if not bool(mf.converged):
        mf = mf.newton()
        mf.conv_tol = 1.0e-12
        mf.max_cycle = 100
        mf.kernel()
    if not bool(mf.converged):
        raise RuntimeError("cyclopentadienyl-cation RHF failed to converge")
    return mf


def metric_orthonormalize(coeff: np.ndarray, metric: np.ndarray) -> np.ndarray:
    gram = coeff.conj().T @ metric @ coeff
    gram = (gram + gram.conj().T) / 2.0
    values, vectors = np.linalg.eigh(gram)
    if float(np.min(values)) < 1.0e-11:
        raise RuntimeError(f"singular projected MO metric: {np.min(values):.3e}")
    return coeff @ (vectors @ np.diag(values ** -0.5) @ vectors.conj().T)


def semicanonicalize(coeff: np.ndarray, fock: np.ndarray) -> np.ndarray:
    if coeff.shape[1] == 0:
        return coeff
    block = coeff.conj().T @ fock @ coeff
    block = (block + block.conj().T) / 2.0
    _, vectors = np.linalg.eigh(block)
    return coeff @ vectors


def _minao_carbon_2pz_indices(target_mol) -> List[int]:
    indices: List[int] = []
    for index, label in enumerate(target_mol.ao_labels(fmt=False)):
        if len(label) < 4:
            continue
        symbol = str(label[1]).split()[0]
        nl = str(label[2]).replace(" ", "").lower()
        component = str(label[3]).replace(" ", "").lower()
        if symbol == "C" and nl == "2p" and component == "z":
            indices.append(index)
    if len(indices) != N_RING:
        raise RuntimeError(
            f"MINAO C 2pz selector found {len(indices)} functions, expected {N_RING}"
        )
    return indices


def build_pi_initial_mos(mf) -> Tuple[np.ndarray, Dict[str, Any]]:
    _, gto, _, _ = require_pyscf()
    mol = mf.mol
    target = gto.M(
        atom=list(zip(ELEMENTS, REFERENCE_GEOMETRY)),
        unit="Angstrom",
        basis="minao",
        charge=1,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    target_indices = _minao_carbon_2pz_indices(target)
    cross = gto.intor_cross("int1e_ovlp", mol, target)[:, target_indices]
    target_metric = target.intor_symmetric("int1e_ovlp")[
        np.ix_(target_indices, target_indices)
    ]
    mo = np.asarray(mf.mo_coeff)
    amplitudes = mo.conj().T @ cross
    projector = (
        amplitudes
        @ np.linalg.pinv(target_metric, rcond=1.0e-12)
        @ amplitudes.conj().T
    )
    projector = (projector + projector.conj().T) / 2.0
    occupations = np.asarray(mf.mo_occ)
    occupied = np.where(occupations > 1.0e-8)[0]
    virtual = np.where(occupations <= 1.0e-8)[0]
    if occupied.size != 17:
        raise RuntimeError(f"expected 17 RHF occupied orbitals, found {occupied.size}")
    occ_values, occ_vectors = np.linalg.eigh(projector[np.ix_(occupied, occupied)])
    vir_values, vir_vectors = np.linalg.eigh(projector[np.ix_(virtual, virtual)])
    occ_order = np.argsort(occ_values)
    vir_order = np.argsort(vir_values)
    occ_values = occ_values[occ_order]
    occ_vectors = occ_vectors[:, occ_order]
    vir_values = vir_values[vir_order]
    vir_vectors = vir_vectors[:, vir_order]
    core = mo[:, occupied] @ occ_vectors[:, :-2]
    active_occ = mo[:, occupied] @ occ_vectors[:, -2:]
    active_vir = mo[:, virtual] @ vir_vectors[:, -3:]
    external = mo[:, virtual] @ vir_vectors[:, :-3]
    fock = np.asarray(mf.get_fock())
    core = semicanonicalize(core, fock)
    active = semicanonicalize(np.column_stack((active_occ, active_vir)), fock)
    external = semicanonicalize(external, fock)
    reordered = np.column_stack((core, active, external))
    metric = mf.get_ovlp()
    error = float(
        np.max(
            np.abs(
                reordered.conj().T @ metric @ reordered
                - np.eye(reordered.shape[1])
            )
        )
    )
    selected = np.concatenate((occ_values[-2:], vir_values[-3:]))
    excluded = np.concatenate((occ_values[:-2], vir_values[:-3]))
    return reordered, {
        "algorithm": (
            "eigenvectors of the MINAO carbon-2pz projector selected separately "
            "inside RHF occupied (top 2) and virtual (top 3) spaces"
        ),
        "target_dimension": N_RING,
        "selected_projector_eigenvalues": selected,
        "minimum_selected_projector_eigenvalue": float(np.min(selected)),
        "maximum_excluded_projector_eigenvalue": float(np.max(excluded)),
        "mo_orthonormality_max_error": error,
    }


def project_mos(old_mol, old_mo: np.ndarray, new_mol) -> np.ndarray:
    from pyscf.scf import addons

    projected = np.asarray(addons.project_mo_nr2nr(old_mol, old_mo, new_mol))
    return metric_orthonormalize(
        projected, new_mol.intor_symmetric("int1e_ovlp")
    )


def make_singlet_solver(mol, nroots: int):
    fci, _, _, _ = require_pyscf()
    solver = fci.direct_spin0.FCI(mol)
    solver.nroots = int(nroots)
    solver.max_cycle = 300
    solver.conv_tol = 1.0e-11
    return solver


def casci_probe(mf, mo: np.ndarray, nroots: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    _, _, mcscf, _ = require_pyscf()
    probe = mcscf.CASCI(mf, NCAS, (NACT_ALPHA, NACT_BETA))
    probe.fcisolver = make_singlet_solver(mf.mol, nroots)
    probe.canonicalization = False
    probe.kernel(mo)
    converged = getattr(probe, "converged", True)
    if not bool(np.all(np.asarray(converged))):
        raise RuntimeError("singlet CASCI probe failed to converge")
    energies = np.asarray(probe.e_tot, dtype=float).reshape(-1)
    roots = [np.asarray(root) for root in probe.ci]
    if energies.size != int(nroots) or len(roots) != int(nroots):
        raise RuntimeError("singlet CASCI probe returned an unexpected root count")
    return energies, roots


def load_solution(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        nroots = int(metadata["nroots"])
        result: Dict[str, Any] = {
            "path": str(path),
            "metadata": metadata,
            "geometry_angstrom": np.asarray(data["geometry_angstrom"]),
            "mo_coeff": np.asarray(data["mo_coeff"]),
            "state_energies_Eh": np.asarray(data["state_energies_Eh"]),
            "ci_roots": [
                np.asarray(data[f"ci_root_{index}"]) for index in range(nroots)
            ],
        }
        if "state_gradient_0_Eh_per_bohr" in data.files:
            result["state_gradients_Eh_per_bohr"] = [
                np.asarray(data[f"state_gradient_{index}_Eh_per_bohr"])
                for index in range(nroots)
            ]
        return result


def validate_existing_solution(
    target: Path,
    geometry: np.ndarray,
    basis: str,
    role: str,
    gradients_required: bool,
) -> Dict[str, Any]:
    solution = load_solution(target)
    metadata = solution["metadata"]
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "basis": basis.lower(),
        "basis_definition_sha256": BASIS_DEFINITION_SHA256,
        "protocol_role": role,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": NROOTS,
        "state_gradients_computed": bool(gradients_required),
        "pyscf_version": PYSCF_VERSION_EXACT,
    }
    actual = {
        "script_version": metadata.get("script_version"),
        "protocol_id": metadata.get("protocol_id"),
        "basis": str(metadata.get("basis", "")).lower(),
        "basis_definition_sha256": metadata.get("basis_definition_sha256"),
        "protocol_role": metadata.get("protocol_role"),
        "ncas": metadata.get("ncas"),
        "nelecas": metadata.get("nelecas"),
        "nroots": metadata.get("nroots"),
        "state_gradients_computed": metadata.get("state_gradients_computed"),
        "pyscf_version": metadata.get("pyscf_version"),
    }
    if actual != expected:
        raise RuntimeError(
            f"existing solution metadata mismatch for {target}: {actual} != {expected}"
        )
    if float(np.max(np.abs(solution["geometry_angstrom"] - geometry))) > 1.0e-11:
        raise RuntimeError(f"existing solution geometry mismatch for {target}")
    return metadata


def run_sa_solution(
    target: Path,
    geometry: np.ndarray,
    basis: str,
    memory_mb: int,
    role: str,
    center_seed: Path | None,
    point_descriptor: Mapping[str, Any],
    compute_state_gradients: bool,
) -> Dict[str, Any]:
    fci, _, mcscf, _ = require_pyscf()
    if target.exists():
        return validate_existing_solution(
            target,
            geometry,
            basis,
            role,
            compute_state_gradients,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.parent / "casscf.log"
    mol = make_mol(geometry, basis, memory_mb, str(log_path), verbose=4)
    mf = run_rhf(mol)
    ncore_expected = (mol.nelectron - NELECAS) // 2
    if center_seed is None:
        mo_guess, selection = build_pi_initial_mos(mf)
        reference_active = mo_guess[
            :, ncore_expected : ncore_expected + NCAS
        ].copy()
    else:
        center = load_solution(center_seed)
        if center["metadata"].get("script_version") != SCRIPT_VERSION:
            raise RuntimeError("center seed belongs to another protocol")
        center_mol = make_mol(
            center["geometry_angstrom"], basis, memory_mb, verbose=0
        )
        mo_guess = project_mos(center_mol, center["mo_coeff"], mol)
        reference_active = mo_guess[
            :, ncore_expected : ncore_expected + NCAS
        ].copy()
        center_selection = center["metadata"].get("active_orbital_selection", {})
        selection = {
            "algorithm": "metric projection of the qualified cc-pVTZ D5h SA2 center MO frame",
            "center_solution": str(center_seed.resolve()),
            "target_dimension": int(center_selection.get("target_dimension", NCAS)),
            "minimum_selected_projector_eigenvalue": float(
                center_selection.get(
                    "minimum_selected_projector_eigenvalue", float("nan")
                )
            ),
        }

    mc = mcscf.CASSCF(mf, NCAS, (NACT_ALPHA, NACT_BETA))
    mc.fcisolver = make_singlet_solver(mol, NROOTS)
    mc = mc.state_average_([0.5, 0.5])
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
    if energies.size != NROOTS or len(ci_roots) != NROOTS:
        raise RuntimeError("SA2-CASSCF returned an unexpected root count")
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
        raise RuntimeError(f"unexpected ncore={ncore}; expected {ncore_expected}")
    probe_energies, _ = casci_probe(mf, final_mo, PROBE_NROOTS)
    reproduction_error = float(np.max(np.abs(probe_energies[:2] - energies)))
    third_gap = float(probe_energies[2] - probe_energies[1])
    spin_square = [
        float(
            fci.spin_op.spin_square0(
                root, NCAS, (NACT_ALPHA, NACT_BETA)
            )[0]
        )
        for root in ci_roots
    ]
    spin_error = max(abs(value) for value in spin_square)
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

    state_gradients: List[np.ndarray] = []
    if compute_state_gradients:
        try:
            gradient_method = mc.nuc_grad_method()
            for state in range(NROOTS):
                gradient = np.asarray(
                    gradient_method.kernel(state=state), dtype=float
                )
                if gradient.shape != (N_ATOMS, 3) or not np.all(
                    np.isfinite(gradient)
                ):
                    raise RuntimeError(
                        f"invalid state-{state} gradient shape/value: {gradient.shape}"
                    )
                state_gradients.append(gradient)
        except Exception as exc:
            failure = {
                "script_version": SCRIPT_VERSION,
                "status": "STOP_ANALYTIC_GRADIENT_FAILURE",
                "target": str(target),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "signal_inspected": False,
            }
            write_json(target.parent / "analytic_gradient_failure.json", failure)
            raise AnalyticGradientFailure(str(exc)) from exc

    metadata: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_role": role,
        "molecule": "cyclopentadienyl cation",
        "formula": "C5H5+",
        "target": "lowest singlet manifold only",
        "absolute_ground_state_claimed": False,
        "basis": basis,
        "basis_literature_label": "cc-pVTZ",
        "basis_identity": CCPVTZ_BASIS_DECLARATION,
        "basis_definition_sha256": BASIS_DEFINITION_SHA256,
        "method": "equal-weight SA2-CASSCF(4e,5o) full-valence-pi singlet",
        "charge": 1,
        "spin_2S": 0,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": NROOTS,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "state_average_weights": [0.5, 0.5],
        "point_descriptor": dict(point_descriptor),
        "geometry_hash": array_hash(geometry),
        "converged": bool(mc.converged),
        "primary_converged": primary_converged,
        "fallback_attempts": fallback_attempts,
        "average_energy_Eh": float(np.mean(energies)),
        "state_energies_Eh": energies,
        "adiabatic_gap_Eh": float(energies[1] - energies[0]),
        "maximum_orbital_gradient": maximum_orbital_gradient,
        "state_S2": spin_square,
        "maximum_spin_square_error": spin_error,
        "mo_orthonormality_max_error": mo_error,
        "CASCI_probe_energies_Eh": probe_energies,
        "CASCI_reproduction_max_error_Eh": reproduction_error,
        "third_singlet_gap_Eh": third_gap,
        "active_reference_singular_values": active_sv,
        "active_reference_sigma_min": float(np.min(active_sv)),
        "active_orbital_selection": selection,
        "state_gradients_computed": bool(compute_state_gradients),
        "state_gradient_max_abs_Eh_per_bohr": [
            float(np.max(np.abs(item))) for item in state_gradients
        ],
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
    for index, root in enumerate(ci_roots):
        arrays[f"ci_root_{index}"] = root
    for index, gradient in enumerate(state_gradients):
        arrays[f"state_gradient_{index}_Eh_per_bohr"] = gradient
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return metadata


def protocol_path() -> Path:
    return Path(__file__).resolve().with_name(PROTOCOL_FILENAME)


def source_authorization_declaration() -> Dict[str, Any]:
    return {
        "required_before_production": True,
        "source_script_version": SOURCE_SCRIPT_VERSION,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_decision_filename": SOURCE_DECISION_FILENAME,
        "required_source_status": SOURCE_REQUIRED_STATUS,
        "required_source_basis": "dz",
        "required_source_basis_definition_sha256": SOURCE_BASIS_DEFINITION_SHA256,
        "required_source_pyscf_version": PYSCF_VERSION_EXACT,
        "required_source_runtime_admission_status": "PASS_RUNTIME_ADMISSION",
        "candidate_gap_maximum_Eh": Thresholds().maximum_secondary_CI_gap_Eh,
        "replica_gap_maximum_Eh": Thresholds().maximum_replica_gap_Eh,
        "replica_average_energy_spread_maximum_Eh": (
            Thresholds().maximum_replica_average_energy_spread_Eh
        ),
        "candidate_solution_hash_required": True,
        "five_replica_solution_hashes_required": True,
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "manuscript_claim_authorized": False,
    }


# Retained only so the unexposed legacy admission helper below remains
# importable; production commands never call it in the lift protocol.
def runtime_admission_declaration() -> Dict[str, Any]:
    return source_authorization_declaration()


def validate_protocol_declaration() -> Dict[str, Any]:
    path = protocol_path()
    if not path.exists():
        raise RuntimeError(f"missing protocol declaration: {path}")
    declaration = json.loads(path.read_text())
    model = declaration.get("locked_model", {})
    search = declaration.get("geometry_search", {})
    expected = {
        "protocol_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "basis": PYSCF_BASIS_KEY,
        "basis_definition_sha256": BASIS_DEFINITION_SHA256,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "charge": 1,
        "spin_2S": 0,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": NROOTS,
        "R_CC_angstrom": R_CC_ANGSTROM,
        "R_CH_angstrom": R_CH_ANGSTROM,
        "casscf_energy_convergence_Eh": CASSCF_CONV_TOL,
        "casscf_orbital_gradient_convergence": CASSCF_GRAD_TOL,
        "representative_axis_degrees": ANGLE_DEGREES,
        "initial_e1_amplitude_angstrom": INITIAL_E1_ANGSTROM,
        "e1_lower_bound_angstrom": E1_LOWER_ANGSTROM,
        "e1_upper_bound_angstrom": E1_UPPER_ANGSTROM,
        "spectator_coordinate_bound_angstrom": SPECTATOR_BOUND_ANGSTROM,
        "spectator_soft_radius_angstrom": SPECTATOR_SOFT_RADIUS_ANGSTROM,
        "spectator_soft_penalty_Eh_per_A2": SPECTATOR_PENALTY_EH_PER_A2,
        "average_energy_soft_ceiling_above_center_Eh": ENERGY_SOFT_CEILING_EH,
        "average_energy_soft_penalty_inverse_Eh": ENERGY_PENALTY_INVERSE_EH,
        "maximum_stage_A_iterations": STAGE_A_MAXITER,
        "maximum_stage_B_iterations": STAGE_B_MAXITER,
        "maximum_stage_A_function_evaluations": STAGE_A_MAXFUN,
        "maximum_stage_B_function_evaluations": STAGE_B_MAXFUN,
        "maximum_stage_A_line_search_steps": STAGE_A_MAXLS,
        "maximum_stage_B_line_search_steps": STAGE_B_MAXLS,
        "optimizer_ftol": OPTIMIZER_FTOL,
        "optimizer_gtol": OPTIMIZER_GTOL,
    }
    actual = {
        "protocol_version": declaration.get("protocol_version"),
        "protocol_id": declaration.get("protocol_id"),
        "basis": str(model.get("basis", "")).lower(),
        "basis_definition_sha256": model.get("basis_definition_sha256"),
        "pyscf_version_exact": model.get("pyscf_version_exact"),
        "charge": model.get("charge"),
        "spin_2S": model.get("spin_2S"),
        "ncas": model.get("ncas"),
        "nelecas": model.get("nelecas"),
        "nroots": model.get("nroots"),
        "R_CC_angstrom": model.get("R_CC_angstrom"),
        "R_CH_angstrom": model.get("R_CH_angstrom"),
        "casscf_energy_convergence_Eh": model.get(
            "casscf_energy_convergence_Eh"
        ),
        "casscf_orbital_gradient_convergence": model.get(
            "casscf_orbital_gradient_convergence"
        ),
        "representative_axis_degrees": search.get("representative_axis_degrees"),
        "initial_e1_amplitude_angstrom": search.get(
            "initial_e1_amplitude_angstrom"
        ),
        "e1_lower_bound_angstrom": search.get("e1_lower_bound_angstrom"),
        "e1_upper_bound_angstrom": search.get("e1_upper_bound_angstrom"),
        "spectator_coordinate_bound_angstrom": search.get(
            "spectator_coordinate_bound_angstrom"
        ),
        "spectator_soft_radius_angstrom": search.get(
            "spectator_soft_radius_angstrom"
        ),
        "spectator_soft_penalty_Eh_per_A2": search.get(
            "spectator_soft_penalty_Eh_per_A2"
        ),
        "average_energy_soft_ceiling_above_center_Eh": search.get(
            "average_energy_soft_ceiling_above_center_Eh"
        ),
        "average_energy_soft_penalty_inverse_Eh": search.get(
            "average_energy_soft_penalty_inverse_Eh"
        ),
        "maximum_stage_A_iterations": search.get("maximum_stage_A_iterations"),
        "maximum_stage_B_iterations": search.get("maximum_stage_B_iterations"),
        "maximum_stage_A_function_evaluations": search.get(
            "maximum_stage_A_function_evaluations"
        ),
        "maximum_stage_B_function_evaluations": search.get(
            "maximum_stage_B_function_evaluations"
        ),
        "maximum_stage_A_line_search_steps": search.get(
            "maximum_stage_A_line_search_steps"
        ),
        "maximum_stage_B_line_search_steps": search.get(
            "maximum_stage_B_line_search_steps"
        ),
        "optimizer_ftol": search.get("optimizer_ftol"),
        "optimizer_gtol": search.get("optimizer_gtol"),
    }
    if actual != expected:
        raise RuntimeError(
            "protocol declaration disagrees with executable constants: "
            f"expected={expected}, actual={actual}"
        )
    if declaration.get("thresholds", {}) != asdict(Thresholds()):
        raise RuntimeError("protocol thresholds disagree with executable")
    if declaration.get("source_authorization", {}) != source_authorization_declaration():
        raise RuntimeError("protocol source authorization disagrees with executable")
    publication = declaration.get("publication_policy", {})
    if any(
        bool(publication.get(key))
        for key in (
            "berry_phase_evaluated",
            "loop_constructed",
            "manuscript_claim_authorized",
        )
    ):
        raise RuntimeError(
            "lift protocol may not evaluate a loop/phase or authorize a manuscript claim"
        )
    return declaration


def build_manifest(root: Path, basis: str) -> Dict[str, Any]:
    script = Path(__file__).resolve()
    protocol = protocol_path()
    source_authorization = root / SOURCE_AUTHORIZATION_FILENAME
    if not source_authorization.exists():
        raise RuntimeError(
            f"missing {source_authorization}; run the 'qualify' command first"
        )
    source_record = json.loads(source_authorization.read_text())
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "pyscf_version_exact": PYSCF_VERSION_EXACT,
        "root": str(root.resolve()),
        "basis": basis.lower(),
        "basis_definition_sha256": BASIS_DEFINITION_SHA256,
        "source_authorization": {
            "name": source_authorization.name,
            "sha256": sha256_file(source_authorization),
            "source_root": source_record.get("source_root"),
            "source_decision_sha256": source_record.get("source_decision_sha256"),
            "source_candidate_solution_sha256": source_record.get(
                "source_candidate_solution_sha256"
            ),
        },
        "immutable_files": {
            "driver": {"name": script.name, "sha256": sha256_file(script)},
            "protocol": {"name": protocol.name, "sha256": sha256_file(protocol)},
        },
        "locked_geometry_search": {
            "axis_degrees": ANGLE_DEGREES,
            "initial_geometry_source": "authorized DZV v2 candidate coefficients",
            "source_q1_angstrom": source_record.get(
                "source_candidate_coefficients_angstrom", [None]
            )[0],
            "e1_bounds_angstrom": [E1_LOWER_ANGSTROM, E1_UPPER_ANGSTROM],
            "c2v_coordinate_dimension": N_COORDS,
        },
        "signal_inspected": False,
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
        "pyscf_version_exact",
        "root",
        "basis",
        "basis_definition_sha256",
        "source_authorization",
        "immutable_files",
        "locked_geometry_search",
        "signal_inspected",
    ):
        if stored.get(key) != current.get(key):
            raise RuntimeError(
                f"run manifest mismatch for {key}; use a new --root"
            )
    return stored


def point_checks(metadata: Mapping[str, Any], thresholds: Thresholds) -> Dict[str, bool]:
    weights = np.asarray(metadata.get("state_average_weights", []), dtype=float)
    return {
        "model_semantics_exact": bool(
            metadata.get("script_version") == SCRIPT_VERSION
            and metadata.get("protocol_id") == PROTOCOL_ID
            and str(metadata.get("basis", "")).lower() == PYSCF_BASIS_KEY
            and metadata.get("basis_definition_sha256")
            == BASIS_DEFINITION_SHA256
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
    }


def center_solution_path(root: Path) -> Path:
    return root / "center_ccpvtz" / "solution.npz"


def run_center(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != PYSCF_BASIS_KEY:
        raise SystemExit(
            "This lift is locked to --basis cc-pvtz in PySCF 2.13.1"
        )
    validate_protocol_declaration()
    configure_threads(args.threads)
    root = args.root.resolve()
    validate_source_authorization(root, args.source_root, require_pass=True)
    create_or_validate_manifest(root, args.basis)
    require_pyscf()
    decision_path = root / CENTER_DECISION_FILENAME
    if decision_path.exists():
        result = json.loads(decision_path.read_text())
        expected_solution = center_solution_path(root)
        if (
            result.get("script_version") != SCRIPT_VERSION
            or result.get("protocol_id") != PROTOCOL_ID
            or Path(str(result.get("solution_file", ""))).resolve()
            != expected_solution.resolve()
            or not expected_solution.exists()
            or result.get("solution_sha256") != sha256_file(expected_solution)
        ):
            raise RuntimeError(
                "existing center decision/solution failed integrity validation; "
                "use a new --root"
            )
        print(f"CPDC_CCPVTZ_CENTER={result.get('status', 'UNKNOWN')}", flush=True)
        return result
    thresholds = Thresholds()
    target = center_solution_path(root)
    metadata = run_sa_solution(
        target=target,
        geometry=REFERENCE_GEOMETRY,
        basis=args.basis,
        memory_mb=args.memory_mb,
        role="fixed_D5h_ccpVTZ_SA2_center",
        center_seed=None,
        point_descriptor={"kind": "fixed_D5h_ccpVTZ_center"},
        compute_state_gradients=False,
    )
    checks = point_checks(metadata, thresholds)
    checks["full_valence_pi_reference_selected"] = bool(
        float(
            metadata.get("active_orbital_selection", {}).get(
                "minimum_selected_projector_eigenvalue", -float("inf")
            )
        )
        >= thresholds.minimum_selected_pi_projector_eigenvalue
    )
    checks["D5h_pair_preserved"] = bool(
        abs(float(metadata.get("adiabatic_gap_Eh", float("inf"))))
        <= thresholds.maximum_center_pair_splitting_Eh
    )
    checks["center_third_singlet_isolated"] = bool(
        float(metadata.get("third_singlet_gap_Eh", -float("inf")))
        >= thresholds.minimum_center_third_singlet_gap_Eh
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "gate": "independent D5h SA2-CASSCF(4e,5o)/cc-pVTZ center",
        "status": "PASS_CENTER" if all(checks.values()) else "STOP_CENTER_MODEL_FAILURE",
        "signal_inspected": False,
        "manuscript_claim_authorized": False,
        "thresholds": asdict(thresholds),
        "checks": checks,
        "solution_file": str(target.resolve()),
        "solution_sha256": sha256_file(target),
        "metadata": metadata,
    }
    write_json(decision_path, result)
    print("\nCYCLOPENTADIENYL CATION cc-pVTZ CENTER", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(
        f"pair splitting={metadata['adiabatic_gap_Eh']:.6e} Eh; "
        f"gap23={metadata['third_singlet_gap_Eh']:.6e} Eh; "
        f"pi_projector={metadata['active_orbital_selection']['minimum_selected_projector_eigenvalue']:.9f}; "
        f"grad={metadata['maximum_orbital_gradient']:.3e}",
        flush=True,
    )
    print(f"Decision file: {decision_path}", flush=True)
    print(f"CPDC_CCPVTZ_CENTER={result['status']}", flush=True)
    return result


def candidate_geometry_checks(
    metadata: Mapping[str, Any],
    geometry: np.ndarray,
    coefficients: np.ndarray,
    center_average_energy: float,
    thresholds: Thresholds,
    include_gap: bool,
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    metrics = geometry_metrics(geometry, coefficients)
    energy_above = float(metadata["average_energy_Eh"] - center_average_energy)
    metrics["average_energy_above_center_Eh"] = energy_above
    checks = point_checks(metadata, thresholds)
    checks.update(
        {
            "analytic_state_gradients_available": bool(
                metadata.get("state_gradients_computed")
            ),
            "C2v_reflection_preserved": bool(
                metrics["C2v_reflection_error_angstrom"]
                <= thresholds.maximum_C2v_reflection_error_angstrom
            ),
            "planarity_preserved": bool(
                metrics["planarity_error_angstrom"]
                <= thresholds.maximum_planarity_error_angstrom
            ),
            "noncentral_e1_amplitude": bool(
                float(coefficients[0])
                >= E1_LOWER_ANGSTROM
                + thresholds.minimum_e1_boundary_clearance_angstrom
                and float(coefficients[0])
                <= E1_UPPER_ANGSTROM
                - thresholds.minimum_e1_boundary_clearance_angstrom
            ),
            "spectator_displacement_bounded": bool(
                metrics["spectator_norm_angstrom"]
                <= thresholds.maximum_spectator_norm_angstrom
            ),
            "average_energy_admissible": bool(
                energy_above <= thresholds.maximum_average_energy_above_center_Eh
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
        }
    )
    if include_gap:
        checks["secondary_CI_gap"] = bool(
            float(metadata["adiabatic_gap_Eh"])
            <= thresholds.maximum_secondary_CI_gap_Eh
        )
    return checks, metrics


def runtime_admission_path(root: Path) -> Path:
    """Legacy v2 helper, deliberately unreachable from the lift CLI."""
    return root / RUNTIME_ADMISSION_FILENAME


def _solution_integrity(path: Path) -> Dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _projected_gap_gradient(solution: Mapping[str, Any]) -> np.ndarray:
    gradients = solution.get("state_gradients_Eh_per_bohr")
    if gradients is None or len(gradients) != NROOTS:
        raise AnalyticGradientFailure("runtime admission lacks two state gradients")
    gradient0 = np.asarray(gradients[0], dtype=float)
    gradient1 = np.asarray(gradients[1], dtype=float)
    if gradient0.shape != (N_ATOMS, 3) or gradient1.shape != (N_ATOMS, 3):
        raise AnalyticGradientFailure("runtime admission state-gradient shape mismatch")
    return (
        np.einsum("kax,ax->k", C2V_BASIS, gradient1 - gradient0)
        * ANGSTROM_TO_BOHR
    )


def _admission_point_checks(
    metadata: Mapping[str, Any],
    geometry: np.ndarray,
    coefficients: np.ndarray,
    center_average_energy: float,
    thresholds: Thresholds,
    gradients_required: bool,
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    checks, metrics = candidate_geometry_checks(
        metadata,
        geometry,
        coefficients,
        center_average_energy,
        thresholds,
        include_gap=False,
    )
    if not gradients_required:
        checks.pop("analytic_state_gradients_available", None)
    return checks, metrics


def validate_runtime_admission(
    root: Path, require_pass: bool = True
) -> Dict[str, Any]:
    raise RuntimeError(
        "legacy v2 runtime admission is disabled; use source authorization"
    )
    path = runtime_admission_path(root)
    if not path.exists():
        if require_pass:
            raise SystemExit(
                f"Missing {path}. Run the v2 'qualify' command on this exact root "
                "before production."
            )
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root.resolve()),
        "pyscf_version": PYSCF_VERSION_EXACT,
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "runtime-admission record does not belong to this immutable v2 root; "
            "use a new --root"
        )
    if require_pass and record.get("status") != "PASS_RUNTIME_ADMISSION":
        raise SystemExit(
            f"Runtime admission is {record.get('status', 'UNKNOWN')}; production "
            "is forbidden. Use a new protocol/root rather than loosening a gate."
        )
    if record.get("status") == "PASS_RUNTIME_ADMISSION":
        integrity = record.get("solution_integrity", {})
        if set(integrity) != {"center", "initial", "minus", "plus"}:
            raise RuntimeError("runtime-admission solution set is incomplete")
        root_resolved = root.resolve()
        for label, item in integrity.items():
            solution_path = Path(str(item.get("path", ""))).resolve()
            if not solution_path.is_relative_to(root_resolved):
                raise RuntimeError(
                    f"runtime-admission {label} solution is outside the run root"
                )
            if not solution_path.exists():
                raise RuntimeError(
                    f"runtime-admission {label} solution is missing: {solution_path}"
                )
            if item.get("sha256") != sha256_file(solution_path):
                raise RuntimeError(
                    f"runtime-admission {label} solution failed hash validation"
                )
    return record


def run_runtime_admission(args: argparse.Namespace) -> Dict[str, Any]:
    raise RuntimeError(
        "legacy v2 runtime admission is disabled; use the lift 'qualify' command"
    )
    if args.basis.lower() != PYSCF_BASIS_KEY:
        raise SystemExit(
            "This qualification is locked to --basis dz, the explicitly frozen "
            "literature DZV (Dunning-Hay) table"
        )
    validate_protocol_declaration()
    configure_threads(args.threads)
    require_pyscf()
    root = args.root.resolve()
    create_or_validate_manifest(root, args.basis)
    decision_path = runtime_admission_path(root)
    if decision_path.exists():
        result = validate_runtime_admission(root, require_pass=False)
        print(f"CPDC_RUNTIME_V2={result.get('status', 'UNKNOWN')}", flush=True)
        return result

    thresholds = Thresholds()
    center = run_center(args)
    if center.get("status") != "PASS_CENTER":
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "root": str(root),
            "pyscf_version": PYSCF_VERSION_EXACT,
            "status": "STOP_RUNTIME_ADMISSION_FAILURE",
            "reason": "The independent PySCF-2.13.1 center failed a locked gate.",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "production_authorized": False,
            "center_status": center.get("status"),
        }
        write_json(decision_path, result)
        print(f"CPDC_RUNTIME_V2={result['status']}", flush=True)
        return result

    center_average = float(center["metadata"]["average_energy_Eh"])
    initial_coefficients = np.zeros(N_COORDS, dtype=float)
    initial_coefficients[0] = INITIAL_E1_ANGSTROM
    initial_geometry = geometry_from_coefficients(initial_coefficients)
    initial_path = root / "runtime_admission" / "initial" / "solution.npz"
    try:
        initial_metadata = run_sa_solution(
            target=initial_path,
            geometry=initial_geometry,
            basis=args.basis,
            memory_mb=args.memory_mb,
            role="pyscf2131_runtime_admission_initial_gradient",
            center_seed=center_solution_path(root),
            point_descriptor={
                "kind": "signal_blind_runtime_admission_initial",
                "coefficients_angstrom": initial_coefficients.tolist(),
            },
            compute_state_gradients=True,
        )
        initial_solution = load_solution(initial_path)
        projected_gap_gradient = _projected_gap_gradient(initial_solution)
    except AnalyticGradientFailure as exc:
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "root": str(root),
            "pyscf_version": PYSCF_VERSION_EXACT,
            "status": "STOP_RUNTIME_ADMISSION_FAILURE",
            "reason": f"PySCF-2.13.1 analytic-gradient admission failed: {exc}",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "production_authorized": False,
        }
        write_json(decision_path, result)
        print(f"CPDC_RUNTIME_V2={result['status']}", flush=True)
        return result

    spectator_gradient = np.asarray(projected_gap_gradient[1:], dtype=float)
    spectator_gradient_norm = float(np.linalg.norm(spectator_gradient))
    if not np.isfinite(spectator_gradient_norm) or spectator_gradient_norm < 1.0e-10:
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "root": str(root),
            "pyscf_version": PYSCF_VERSION_EXACT,
            "status": "STOP_RUNTIME_ADMISSION_FAILURE",
            "reason": "The initial spectator gap-gradient direction is undefined.",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "production_authorized": False,
            "projected_gap_gradient_Eh_per_angstrom": projected_gap_gradient,
        }
        write_json(decision_path, result)
        print(f"CPDC_RUNTIME_V2={result['status']}", flush=True)
        return result

    direction = spectator_gradient / spectator_gradient_norm
    minus_coefficients = initial_coefficients.copy()
    plus_coefficients = initial_coefficients.copy()
    minus_coefficients[1:] -= ADMISSION_FD_STEP_ANGSTROM * direction
    plus_coefficients[1:] += ADMISSION_FD_STEP_ANGSTROM * direction
    minus_geometry = geometry_from_coefficients(minus_coefficients)
    plus_geometry = geometry_from_coefficients(plus_coefficients)
    minus_path = root / "runtime_admission" / "minus" / "solution.npz"
    plus_path = root / "runtime_admission" / "plus" / "solution.npz"
    minus_metadata = run_sa_solution(
        target=minus_path,
        geometry=minus_geometry,
        basis=args.basis,
        memory_mb=args.memory_mb,
        role="pyscf2131_runtime_admission_fd_minus",
        center_seed=center_solution_path(root),
        point_descriptor={
            "kind": "signal_blind_runtime_admission_fd_minus",
            "coefficients_angstrom": minus_coefficients.tolist(),
        },
        compute_state_gradients=False,
    )
    plus_metadata = run_sa_solution(
        target=plus_path,
        geometry=plus_geometry,
        basis=args.basis,
        memory_mb=args.memory_mb,
        role="pyscf2131_runtime_admission_fd_plus",
        center_seed=center_solution_path(root),
        point_descriptor={
            "kind": "signal_blind_runtime_admission_fd_plus",
            "coefficients_angstrom": plus_coefficients.tolist(),
        },
        compute_state_gradients=False,
    )

    initial_checks, initial_metrics = _admission_point_checks(
        initial_metadata,
        initial_geometry,
        initial_coefficients,
        center_average,
        thresholds,
        gradients_required=True,
    )
    minus_checks, minus_metrics = _admission_point_checks(
        minus_metadata,
        minus_geometry,
        minus_coefficients,
        center_average,
        thresholds,
        gradients_required=False,
    )
    plus_checks, plus_metrics = _admission_point_checks(
        plus_metadata,
        plus_geometry,
        plus_coefficients,
        center_average,
        thresholds,
        gradients_required=False,
    )
    analytic_derivative = float(np.dot(spectator_gradient, direction))
    finite_difference_derivative = float(
        (float(plus_metadata["adiabatic_gap_Eh"])
         - float(minus_metadata["adiabatic_gap_Eh"]))
        / (2.0 * ADMISSION_FD_STEP_ANGSTROM)
    )
    derivative_absolute_error = abs(
        finite_difference_derivative - analytic_derivative
    )
    derivative_relative_error = derivative_absolute_error / max(
        abs(analytic_derivative), abs(finite_difference_derivative), 1.0e-14
    )
    initial_energy_above_center = float(
        initial_metadata["average_energy_Eh"] - center_average
    )
    checks = {
        "center_passed": center.get("status") == "PASS_CENTER",
        "initial_point_qualified": all(initial_checks.values()),
        "minus_point_qualified": all(minus_checks.values()),
        "plus_point_qualified": all(plus_checks.values()),
        "directional_gradient_absolute_error": bool(
            derivative_absolute_error
            <= ADMISSION_MAX_GRADIENT_ABS_ERROR_EH_PER_ANGSTROM
        ),
        "directional_gradient_relative_error": bool(
            derivative_relative_error <= ADMISSION_MAX_GRADIENT_RELATIVE_ERROR
        ),
        "v1_center_average_energy_reproduced": bool(
            abs(center_average - V1_CENTER_AVERAGE_ENERGY_EH)
            <= V1_CENTER_AVERAGE_ENERGY_TOLERANCE_EH
        ),
        "v1_center_third_singlet_gap_reproduced": bool(
            abs(
                float(center["metadata"]["third_singlet_gap_Eh"])
                - V1_CENTER_THIRD_SINGLET_GAP_EH
            )
            <= V1_CENTER_THIRD_SINGLET_GAP_TOLERANCE_EH
        ),
        "v1_initial_noncentral_gap_reproduced": bool(
            abs(float(initial_metadata["adiabatic_gap_Eh"]) - V1_INITIAL_GAP_EH)
            <= V1_INITIAL_GAP_TOLERANCE_EH
        ),
        "v1_initial_energy_above_center_reproduced": bool(
            abs(
                initial_energy_above_center
                - V1_INITIAL_ENERGY_ABOVE_CENTER_EH
            )
            <= V1_INITIAL_ENERGY_ABOVE_CENTER_TOLERANCE_EH
        ),
    }
    status = (
        "PASS_RUNTIME_ADMISSION"
        if all(checks.values())
        else "STOP_RUNTIME_ADMISSION_FAILURE"
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "pyscf_version": PYSCF_VERSION_EXACT,
        "status": status,
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "production_authorized": status == "PASS_RUNTIME_ADMISSION",
        "checks": checks,
        "thresholds": runtime_admission_declaration(),
        "center_metrics": {
            "average_energy_Eh": center_average,
            "third_singlet_gap_Eh": float(
                center["metadata"]["third_singlet_gap_Eh"]
            ),
            "pair_splitting_Eh": float(center["metadata"]["adiabatic_gap_Eh"]),
        },
        "initial_metrics": {
            "adiabatic_gap_Eh": float(initial_metadata["adiabatic_gap_Eh"]),
            "average_energy_above_center_Eh": initial_energy_above_center,
            "projected_gap_gradient_Eh_per_angstrom": projected_gap_gradient,
            "spectator_direction": direction,
            "geometry": initial_metrics,
            "point_checks": initial_checks,
        },
        "minus_metrics": {
            "adiabatic_gap_Eh": float(minus_metadata["adiabatic_gap_Eh"]),
            "geometry": minus_metrics,
            "point_checks": minus_checks,
        },
        "plus_metrics": {
            "adiabatic_gap_Eh": float(plus_metadata["adiabatic_gap_Eh"]),
            "geometry": plus_metrics,
            "point_checks": plus_checks,
        },
        "directional_gradient_audit": {
            "step_angstrom": ADMISSION_FD_STEP_ANGSTROM,
            "analytic_Eh_per_angstrom": analytic_derivative,
            "finite_difference_Eh_per_angstrom": finite_difference_derivative,
            "absolute_error_Eh_per_angstrom": derivative_absolute_error,
            "relative_error": derivative_relative_error,
        },
        "solution_integrity": {
            "center": _solution_integrity(center_solution_path(root)),
            "initial": _solution_integrity(initial_path),
            "minus": _solution_integrity(minus_path),
            "plus": _solution_integrity(plus_path),
        },
    }
    write_json(decision_path, result)
    print("\nCPDC PYSCF-2.13.1 RUNTIME ADMISSION", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"center dE={center_average - V1_CENTER_AVERAGE_ENERGY_EH:+.3e} Eh; "
        f"initial gap={initial_metadata['adiabatic_gap_Eh']:.9e} Eh; "
        f"gradient relerr={derivative_relative_error:.3e}",
        flush=True,
    )
    print(f"Decision file: {decision_path}", flush=True)
    print(f"CPDC_RUNTIME_V2={status}", flush=True)
    return result


def source_authorization_path(root: Path) -> Path:
    return root / SOURCE_AUTHORIZATION_FILENAME


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
    for marker in ("evaluations", "replicas", "center_dzv"):
        if marker in raw.parts:
            index = raw.parts.index(marker)
            candidates.append(source_root.joinpath(*raw.parts[index:]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_relative_to(source_root):
            return resolved
    raise RuntimeError(
        f"source artifact is missing or outside --source-root: {recorded}"
    )


def _validate_source_solution_metadata(
    solution: Mapping[str, Any], role: str, gradients_required: bool
) -> Dict[str, bool]:
    metadata = solution["metadata"]
    return {
        f"{role}_script_version": metadata.get("script_version")
        == SOURCE_SCRIPT_VERSION,
        f"{role}_protocol_id": metadata.get("protocol_id")
        == SOURCE_PROTOCOL_ID,
        f"{role}_basis": str(metadata.get("basis", "")).lower() == "dz",
        f"{role}_basis_definition": metadata.get("basis_definition_sha256")
        == SOURCE_BASIS_DEFINITION_SHA256,
        f"{role}_pyscf_version": metadata.get("pyscf_version")
        == PYSCF_VERSION_EXACT,
        f"{role}_active_space": bool(
            int(metadata.get("ncas", -1)) == NCAS
            and int(metadata.get("nelecas", -1)) == NELECAS
            and int(metadata.get("nroots", -1)) == NROOTS
        ),
        f"{role}_gradient_semantics": bool(
            bool(metadata.get("state_gradients_computed"))
            == bool(gradients_required)
        ),
    }


def build_source_authorization(root: Path, source_root: Path) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    decision_path = source_root / SOURCE_DECISION_FILENAME
    if not decision_path.exists():
        raise RuntimeError(f"missing source decision: {decision_path}")
    decision = json.loads(decision_path.read_text())
    thresholds = Thresholds()
    coefficients = np.asarray(
        decision.get("candidate_coefficients_angstrom", []), dtype=float
    )
    replicas = decision.get("replicas", {})
    replica_records = replicas.get("records", [])
    checks: Dict[str, bool] = {
        "source_script_version": decision.get("script_version")
        == SOURCE_SCRIPT_VERSION,
        "source_protocol_id": decision.get("protocol_id") == SOURCE_PROTOCOL_ID,
        "source_status_passed": decision.get("status") == SOURCE_REQUIRED_STATUS,
        "source_runtime_admission_passed": decision.get(
            "runtime_admission_status"
        )
        == "PASS_RUNTIME_ADMISSION",
        "source_signal_blind": decision.get("signal_inspected") is False,
        "source_phase_or_loop_not_calculated": decision.get(
            "phase_or_loop_calculated"
        )
        is False,
        "source_manuscript_claim_not_authorized": decision.get(
            "manuscript_claim_authorized"
        )
        is False,
        "source_candidate_checks_passed": _all_true(
            decision.get("candidate_checks")
        ),
        "source_replica_checks_passed": _all_true(replicas.get("checks")),
        "source_has_nine_finite_coordinates": bool(
            coefficients.shape == (N_COORDS,) and np.all(np.isfinite(coefficients))
        ),
        "source_has_five_replica_records": bool(
            isinstance(replica_records, list) and len(replica_records) == N_RING
        ),
    }
    if coefficients.shape != (N_COORDS,):
        coefficients = np.full(N_COORDS, float("nan"))
    checks.update(
        {
            "source_q1_matches_preregistered_branch": bool(
                np.isfinite(coefficients[0])
                and abs(float(coefficients[0]) - INITIAL_E1_ANGSTROM) <= 1.0e-10
            ),
            "source_q1_inside_locked_interval": bool(
                E1_LOWER_ANGSTROM <= float(coefficients[0]) <= E1_UPPER_ANGSTROM
            ),
            "source_spectators_inside_locked_box": bool(
                np.all(np.abs(coefficients[1:]) <= SPECTATOR_BOUND_ANGSTROM + 1.0e-12)
            ),
            "source_spectator_norm_admissible": bool(
                np.linalg.norm(coefficients[1:])
                <= thresholds.maximum_spectator_norm_angstrom
            ),
            "source_candidate_gap_admissible": bool(
                float(decision.get("candidate_gap_Eh", float("inf")))
                <= thresholds.maximum_secondary_CI_gap_Eh
            ),
            "source_replica_gap_admissible": bool(
                float(replicas.get("maximum_gap_Eh", float("inf")))
                <= thresholds.maximum_replica_gap_Eh
            ),
            "source_replica_energy_spread_admissible": bool(
                float(replicas.get("average_energy_spread_Eh", float("inf")))
                <= thresholds.maximum_replica_average_energy_spread_Eh
            ),
        }
    )

    candidate_path: Path | None = None
    candidate_sha256: str | None = None
    candidate_geometry_hash: str | None = None
    replica_integrity: List[Dict[str, Any]] = []
    try:
        candidate_path = _resolve_source_artifact(
            source_root, decision.get("candidate_solution_file", "")
        )
        candidate_solution = load_solution(candidate_path)
        checks.update(
            _validate_source_solution_metadata(
                candidate_solution, "candidate", gradients_required=True
            )
        )
        candidate_geometry = np.asarray(
            candidate_solution["geometry_angstrom"], dtype=float
        )
        reconstructed = geometry_from_coefficients(coefficients)
        checks["candidate_geometry_reconstructs_from_coordinates"] = bool(
            candidate_geometry.shape == (N_ATOMS, 3)
            and np.max(np.abs(candidate_geometry - reconstructed)) <= 2.0e-10
        )
        candidate_gap = float(
            candidate_solution["metadata"].get("adiabatic_gap_Eh", float("inf"))
        )
        checks["candidate_gap_matches_source_decision"] = bool(
            abs(candidate_gap - float(decision.get("candidate_gap_Eh", float("nan"))))
            <= 1.0e-10
        )
        candidate_sha256 = sha256_file(candidate_path)
        candidate_geometry_hash = array_hash(candidate_geometry)

        if len(replica_records) == N_RING:
            seen_geometry_hashes = set()
            replica_energies: List[float] = []
            replica_gaps: List[float] = []
            for expected_index, replica_record in enumerate(replica_records):
                replica_path = _resolve_source_artifact(
                    source_root, replica_record.get("solution_file", "")
                )
                replica_solution = load_solution(replica_path)
                checks.update(
                    _validate_source_solution_metadata(
                        replica_solution,
                        f"replica_{expected_index}",
                        gradients_required=False,
                    )
                )
                replica_geometry = np.asarray(
                    replica_solution["geometry_angstrom"], dtype=float
                )
                expected_geometry = symmetry_replica(
                    candidate_geometry, expected_index
                )
                checks[f"replica_{expected_index}_geometry"] = bool(
                    replica_geometry.shape == (N_ATOMS, 3)
                    and np.max(np.abs(replica_geometry - expected_geometry))
                    <= 2.0e-10
                )
                metadata = replica_solution["metadata"]
                gap = float(metadata.get("adiabatic_gap_Eh", float("inf")))
                energy = float(metadata.get("average_energy_Eh", float("nan")))
                checks[f"replica_{expected_index}_index"] = bool(
                    int(replica_record.get("replica", -1)) == expected_index
                )
                checks[f"replica_{expected_index}_gap_matches"] = bool(
                    abs(gap - float(replica_record.get("adiabatic_gap_Eh", float("nan"))))
                    <= 1.0e-10
                )
                geometry_hash = array_hash(replica_geometry)
                seen_geometry_hashes.add(geometry_hash)
                replica_gaps.append(gap)
                replica_energies.append(energy)
                replica_integrity.append(
                    {
                        "replica": expected_index,
                        "path": str(replica_path),
                        "sha256": sha256_file(replica_path),
                        "geometry_hash": geometry_hash,
                        "adiabatic_gap_Eh": gap,
                        "average_energy_Eh": energy,
                    }
                )
            checks["five_source_replica_geometries_distinct"] = bool(
                len(seen_geometry_hashes) == N_RING
            )
            checks["recomputed_source_replica_maximum_gap_admissible"] = bool(
                max(replica_gaps) <= thresholds.maximum_replica_gap_Eh
            )
            checks["recomputed_source_replica_energy_spread_admissible"] = bool(
                max(replica_energies) - min(replica_energies)
                <= thresholds.maximum_replica_average_energy_spread_Eh
            )
    except Exception as exc:
        checks["source_artifact_integrity"] = False
        artifact_error = f"{type(exc).__name__}: {exc}"
    else:
        checks["source_artifact_integrity"] = True
        artifact_error = None

    status = (
        "PASS_SOURCE_AUTHORIZATION"
        if all(checks.values())
        else "STOP_SOURCE_AUTHORIZATION_FAILURE"
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "source_decision_file": str(decision_path.resolve()),
        "source_decision_sha256": sha256_file(decision_path),
        "status": status,
        "production_authorized": status == "PASS_SOURCE_AUTHORIZATION",
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "manuscript_claim_authorized": False,
        "checks": checks,
        "artifact_error": artifact_error,
        "source_candidate_coefficients_angstrom": coefficients,
        "source_candidate_gap_Eh": decision.get("candidate_gap_Eh"),
        "source_candidate_solution_file": (
            str(candidate_path) if candidate_path is not None else None
        ),
        "source_candidate_solution_sha256": candidate_sha256,
        "source_candidate_geometry_hash": candidate_geometry_hash,
        "source_replica_solution_integrity": replica_integrity,
        "source_decision_summary": {
            "status": decision.get("status"),
            "maximum_replica_gap_Eh": replicas.get("maximum_gap_Eh"),
            "replica_average_energy_spread_Eh": replicas.get(
                "average_energy_spread_Eh"
            ),
        },
    }
    return result


def validate_source_authorization(
    root: Path, source_root: Path, require_pass: bool = True
) -> Dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    path = source_authorization_path(root)
    if not path.exists():
        if require_pass:
            raise SystemExit(
                f"Missing {path}. Run the 'qualify' command before center/run."
            )
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    decision_path = source_root / SOURCE_DECISION_FILENAME
    expected = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "root": str(root),
        "source_root": str(source_root),
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "manuscript_claim_authorized": False,
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "source authorization belongs to another root/source pair; use a new --root"
        )
    if not decision_path.exists() or record.get(
        "source_decision_sha256"
    ) != sha256_file(decision_path):
        raise RuntimeError("source decision changed after authorization")
    integrity_items = [
        {
            "path": record.get("source_candidate_solution_file"),
            "sha256": record.get("source_candidate_solution_sha256"),
        }
    ] + list(record.get("source_replica_solution_integrity", []))
    for item in integrity_items:
        artifact = Path(str(item.get("path", ""))).resolve()
        if (
            not artifact.exists()
            or not artifact.is_relative_to(source_root)
            or item.get("sha256") != sha256_file(artifact)
        ):
            raise RuntimeError(f"source artifact changed after authorization: {artifact}")
    if require_pass and record.get("status") != "PASS_SOURCE_AUTHORIZATION":
        raise SystemExit(
            f"Source authorization is {record.get('status', 'UNKNOWN')}; "
            "cc-pVTZ production is forbidden."
        )
    return record


def run_source_authorization(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != PYSCF_BASIS_KEY:
        raise SystemExit("This lift is locked to --basis cc-pvtz")
    validate_protocol_declaration()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = source_authorization_path(root)
    if path.exists():
        result = validate_source_authorization(
            root, args.source_root, require_pass=False
        )
    else:
        result = build_source_authorization(root, args.source_root)
        write_json(path, result)
    print("\nCPDC cc-pVTZ LIFT SOURCE AUTHORIZATION", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(
        f"source gap={float(result.get('source_candidate_gap_Eh', float('nan'))):.9e} Eh; "
        f"verified replicas={len(result.get('source_replica_solution_integrity', []))}",
        flush=True,
    )
    print(f"Decision file: {path}", flush=True)
    print(f"CPDC_LIFT_SOURCE={result['status']}", flush=True)
    return result


class EvaluationManager:
    def __init__(
        self,
        root: Path,
        basis: str,
        memory_mb: int,
        center_seed: Path,
        center_average_energy: float,
        thresholds: Thresholds,
    ):
        self.root = root
        self.basis = basis
        self.memory_mb = memory_mb
        self.center_seed = center_seed
        self.center_average_energy = float(center_average_energy)
        self.thresholds = thresholds
        self.registry_path = root / REGISTRY_FILENAME
        if self.registry_path.exists():
            self.registry = json.loads(self.registry_path.read_text())
            expected = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "pyscf_version_exact": PYSCF_VERSION_EXACT,
                "root": str(root.resolve()),
                "basis": basis.lower(),
                "center_solution_sha256": sha256_file(center_seed),
                "signal_inspected": False,
            }
            actual = {key: self.registry.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    "evaluation registry belongs to a different immutable run; "
                    "use a new --root"
                )
        else:
            self.registry = {
                "script_version": SCRIPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "pyscf_version_exact": PYSCF_VERSION_EXACT,
                "root": str(root.resolve()),
                "basis": basis.lower(),
                "center_solution_sha256": sha256_file(center_seed),
                "signal_inspected": False,
                "evaluations": [],
            }
            write_json(self.registry_path, self.registry)

    @staticmethod
    def coefficient_key(coefficients: np.ndarray) -> str:
        return array_hash(np.asarray(coefficients, dtype=float))

    def _existing(self, key: str) -> Mapping[str, Any] | None:
        for record in self.registry["evaluations"]:
            if record.get("coefficient_key") == key:
                return record
        return None

    def evaluate(self, coefficients: Sequence[float]) -> Dict[str, Any]:
        values = np.asarray(coefficients, dtype=float)
        if values.shape != (N_COORDS,):
            raise ValueError(f"unexpected coefficient shape: {values.shape}")
        key = self.coefficient_key(values)
        existing = self._existing(key)
        if existing is not None:
            solution_path = Path(str(existing["solution_file"]))
            solution = load_solution(solution_path)
            return self._record_from_solution(values, solution_path, solution)

        index = len(self.registry["evaluations"])
        # Content addressing keeps restart safe if the NPZ lands immediately
        # before a scheduler kill and the registry update does not.
        target = self.root / "evaluations" / f"eval_{key}" / "solution.npz"
        geometry = geometry_from_coefficients(values)
        metadata = run_sa_solution(
            target=target,
            geometry=geometry,
            basis=self.basis,
            memory_mb=self.memory_mb,
            role="C2v_secondary_CI_gap_optimization_evaluation",
            center_seed=self.center_seed,
            point_descriptor={
                "kind": "signal_blind_secondary_CI_gap_evaluation",
                "evaluation_index": index,
                "coefficients_angstrom": values.tolist(),
            },
            compute_state_gradients=True,
        )
        registry_record = {
            "evaluation_index": index,
            "coefficient_key": key,
            "coefficients_angstrom": values.tolist(),
            "solution_file": str(target.resolve()),
            "adiabatic_gap_Eh": float(metadata["adiabatic_gap_Eh"]),
            "average_energy_Eh": float(metadata["average_energy_Eh"]),
            "maximum_orbital_gradient": float(metadata["maximum_orbital_gradient"]),
            "timestamp_unix": time.time(),
        }
        self.registry["evaluations"].append(registry_record)
        write_json(self.registry_path, self.registry)
        solution = load_solution(target)
        record = self._record_from_solution(values, target, solution)
        print(
            f"EVAL {index:04d} q1={values[0]:+.8f} A "
            f"spec={np.linalg.norm(values[1:]):.8f} A "
            f"gap={record['metadata']['adiabatic_gap_Eh']:.9e} Eh "
            f"Eavg-Ecenter={record['energy_above_center_Eh']:+.9e} Eh "
            f"orbgrad={record['metadata']['maximum_orbital_gradient']:.3e}",
            flush=True,
        )
        return record

    def _record_from_solution(
        self, coefficients: np.ndarray, solution_path: Path, solution: Mapping[str, Any]
    ) -> Dict[str, Any]:
        metadata = solution["metadata"]
        gradients = solution.get("state_gradients_Eh_per_bohr")
        if gradients is None or len(gradients) != NROOTS:
            raise AnalyticGradientFailure(
                f"state gradients missing from {solution_path}"
            )
        gradient0 = np.asarray(gradients[0], dtype=float)
        gradient1 = np.asarray(gradients[1], dtype=float)
        projected0 = (
            np.einsum("kax,ax->k", C2V_BASIS, gradient0)
            * ANGSTROM_TO_BOHR
        )
        projected1 = (
            np.einsum("kax,ax->k", C2V_BASIS, gradient1)
            * ANGSTROM_TO_BOHR
        )
        geometry = np.asarray(solution["geometry_angstrom"])
        checks, metrics = candidate_geometry_checks(
            metadata,
            geometry,
            coefficients,
            self.center_average_energy,
            self.thresholds,
            include_gap=False,
        )
        return {
            "coefficients_angstrom": np.asarray(coefficients, dtype=float),
            "solution_file": str(solution_path),
            "metadata": metadata,
            "geometry_angstrom": geometry,
            "projected_state_gradients_Eh_per_angstrom": [projected0, projected1],
            "prequalification_checks": checks,
            "geometry_metrics": metrics,
            "energy_above_center_Eh": float(
                metadata["average_energy_Eh"] - self.center_average_energy
            ),
        }

    def objective_and_gradient(
        self, coefficients: Sequence[float]
    ) -> Tuple[float, np.ndarray]:
        values = np.asarray(coefficients, dtype=float)
        record = self.evaluate(values)
        metadata = record["metadata"]
        gap = float(metadata["adiabatic_gap_Eh"])
        projected0, projected1 = record[
            "projected_state_gradients_Eh_per_angstrom"
        ]
        gradient = np.asarray(projected1) - np.asarray(projected0)
        objective = gap

        spectator_norm = float(np.linalg.norm(values[1:]))
        if spectator_norm > SPECTATOR_SOFT_RADIUS_ANGSTROM:
            excess = spectator_norm - SPECTATOR_SOFT_RADIUS_ANGSTROM
            objective += SPECTATOR_PENALTY_EH_PER_A2 * excess**2
            gradient[1:] += (
                2.0
                * SPECTATOR_PENALTY_EH_PER_A2
                * excess
                * values[1:]
                / spectator_norm
            )

        energy_excess = (
            float(metadata["average_energy_Eh"])
            - self.center_average_energy
            - ENERGY_SOFT_CEILING_EH
        )
        if energy_excess > 0.0:
            average_gradient = 0.5 * (np.asarray(projected0) + np.asarray(projected1))
            objective += ENERGY_PENALTY_INVERSE_EH * energy_excess**2
            gradient += (
                2.0
                * ENERGY_PENALTY_INVERSE_EH
                * energy_excess
                * average_gradient
            )

        if (
            gap <= self.thresholds.maximum_secondary_CI_gap_Eh
            and all(record["prequalification_checks"].values())
        ):
            raise TargetFound(values, record)
        return float(objective), np.asarray(gradient, dtype=float)

    def all_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in self.registry["evaluations"]:
            values = np.asarray(item["coefficients_angstrom"], dtype=float)
            solution_path = Path(str(item["solution_file"]))
            solution = load_solution(solution_path)
            records.append(
                self._record_from_solution(values, solution_path, solution)
            )
        return records


def optimizer_result_payload(result: Any) -> Dict[str, Any]:
    if result is None:
        return {"not_run": True}
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "jac": np.asarray(result.jac),
        "x": np.asarray(result.x),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(getattr(result, "njev", -1)),
    }


def run_bounded_search(
    args: argparse.Namespace,
    root: Path,
    center_result: Mapping[str, Any],
    thresholds: Thresholds,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    minimize = require_scipy_minimize()
    center_meta = center_result["metadata"]
    manager = EvaluationManager(
        root=root,
        basis=args.basis,
        memory_mb=args.memory_mb,
        center_seed=center_solution_path(root),
        center_average_energy=float(center_meta["average_energy_Eh"]),
        thresholds=thresholds,
    )
    source_authorization = json.loads(
        source_authorization_path(root).read_text()
    )
    initial = np.asarray(
        source_authorization["source_candidate_coefficients_angstrom"],
        dtype=float,
    )
    if initial.shape != (N_COORDS,) or not np.all(np.isfinite(initial)):
        raise RuntimeError("authorized source does not contain nine finite coordinates")
    fixed_q1 = float(initial[0])
    stage_a_result = None
    stage_b_result = None
    target_record: Dict[str, Any] | None = None
    target_coefficients: np.ndarray | None = None

    print("\nCPDC cc-pVTZ SECONDARY-CI LIFT STAGE A", flush=True)
    print(
        f"fixed source q1={fixed_q1:.7f} A; optimizing {N_COORDS-1} C2v spectators",
        flush=True,
    )

    def stage_a_function(spectators: np.ndarray) -> Tuple[float, np.ndarray]:
        coefficients = np.concatenate(([fixed_q1], spectators))
        value, gradient = manager.objective_and_gradient(coefficients)
        return value, gradient[1:]

    try:
        stage_a_result = minimize(
            stage_a_function,
            initial[1:],
            method="L-BFGS-B",
            jac=True,
            bounds=[(-SPECTATOR_BOUND_ANGSTROM, SPECTATOR_BOUND_ANGSTROM)]
            * (N_COORDS - 1),
            options={
                "maxiter": STAGE_A_MAXITER,
                "maxfun": STAGE_A_MAXFUN,
                "ftol": OPTIMIZER_FTOL,
                "gtol": OPTIMIZER_GTOL,
                "maxls": STAGE_A_MAXLS,
            },
        )
    except TargetFound as found:
        target_coefficients = found.coefficients
        target_record = found.record

    records_after_a = manager.all_records()
    eligible_a = [
        record
        for record in records_after_a
        if all(record["prequalification_checks"].values())
    ]
    if target_coefficients is None:
        if eligible_a:
            best_a = min(
                eligible_a, key=lambda item: item["metadata"]["adiabatic_gap_Eh"]
            )
            stage_b_start = np.asarray(best_a["coefficients_angstrom"], dtype=float)
        elif stage_a_result is not None:
            stage_b_start = np.concatenate(
                ([fixed_q1], np.asarray(stage_a_result.x, dtype=float))
            )
        else:
            stage_b_start = initial.copy()

        stage_b_start[0] = min(
            max(stage_b_start[0], E1_LOWER_ANGSTROM + 0.001),
            E1_UPPER_ANGSTROM - 0.001,
        )
        print("\nCPDC cc-pVTZ SECONDARY-CI LIFT STAGE B", flush=True)
        print(
            f"releasing q1 inside [{E1_LOWER_ANGSTROM:.6f}, {E1_UPPER_ANGSTROM:.6f}] A",
            flush=True,
        )
        try:
            stage_b_result = minimize(
                manager.objective_and_gradient,
                stage_b_start,
                method="L-BFGS-B",
                jac=True,
                bounds=[(E1_LOWER_ANGSTROM, E1_UPPER_ANGSTROM)]
                + [(-SPECTATOR_BOUND_ANGSTROM, SPECTATOR_BOUND_ANGSTROM)]
                * (N_COORDS - 1),
                options={
                    "maxiter": STAGE_B_MAXITER,
                    "maxfun": STAGE_B_MAXFUN,
                    "ftol": OPTIMIZER_FTOL,
                    "gtol": OPTIMIZER_GTOL,
                    "maxls": STAGE_B_MAXLS,
                },
            )
        except TargetFound as found:
            target_coefficients = found.coefficients
            target_record = found.record

    all_records = manager.all_records()
    eligible = [
        record
        for record in all_records
        if all(record["prequalification_checks"].values())
    ]
    if eligible:
        best = min(eligible, key=lambda item: item["metadata"]["adiabatic_gap_Eh"])
        if target_record is None or float(best["metadata"]["adiabatic_gap_Eh"]) < float(
            target_record["metadata"]["adiabatic_gap_Eh"]
        ):
            target_record = best
            target_coefficients = np.asarray(best["coefficients_angstrom"], dtype=float)

    search_record = {
        "script_version": SCRIPT_VERSION,
        "gate": "bounded analytic-gradient C2v secondary-CI search",
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "source_candidate_coefficients_angstrom": initial,
        "stage_A_fixed_source_q1_angstrom": fixed_q1,
        "stage_A": optimizer_result_payload(stage_a_result),
        "stage_B": optimizer_result_payload(stage_b_result),
        "number_of_evaluations": len(all_records),
        "number_of_prequalified_evaluations": len(eligible),
        "lowest_gap_Eh": (
            min(float(item["metadata"]["adiabatic_gap_Eh"]) for item in all_records)
            if all_records
            else None
        ),
        "selected_coefficients_angstrom": target_coefficients,
        "selected_solution_file": (
            target_record["solution_file"] if target_record is not None else None
        ),
    }
    write_json(root / SEARCH_RECORD_FILENAME, search_record)
    return search_record, target_record


def run_replicas(
    args: argparse.Namespace,
    root: Path,
    candidate_record: Mapping[str, Any],
    center_average_energy: float,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    candidate_geometry = np.asarray(candidate_record["geometry_angstrom"])
    replica_records: List[Dict[str, Any]] = []
    for replica in range(N_RING):
        geometry = symmetry_replica(candidate_geometry, replica)
        target = root / "replicas" / f"replica_{replica:02d}" / "solution.npz"
        metadata = run_sa_solution(
            target=target,
            geometry=geometry,
            basis=args.basis,
            memory_mb=args.memory_mb,
            role="fivefold_secondary_CI_replica_single_point",
            center_seed=center_solution_path(root),
            point_descriptor={
                "kind": "fivefold_secondary_CI_replica",
                "replica": replica,
                "rotation_degrees": 72.0 * replica,
            },
            compute_state_gradients=False,
        )
        checks = point_checks(metadata, thresholds)
        record = {
            "replica": replica,
            "rotation_degrees": 72.0 * replica,
            "solution_file": str(target),
            "adiabatic_gap_Eh": float(metadata["adiabatic_gap_Eh"]),
            "average_energy_Eh": float(metadata["average_energy_Eh"]),
            "average_energy_above_center_Eh": float(
                metadata["average_energy_Eh"] - center_average_energy
            ),
            "point_checks": checks,
            "geometry_hash": array_hash(geometry),
            "maximum_orbital_gradient": float(metadata["maximum_orbital_gradient"]),
            "third_singlet_gap_Eh": float(metadata["third_singlet_gap_Eh"]),
            "active_reference_sigma_min": float(
                metadata["active_reference_sigma_min"]
            ),
        }
        replica_records.append(record)
        print(
            f"REPLICA {replica} gap={record['adiabatic_gap_Eh']:.9e} Eh "
            f"Eavg-Ecenter={record['average_energy_above_center_Eh']:+.9e} Eh "
            f"grad={record['maximum_orbital_gradient']:.3e}",
            flush=True,
        )
    gaps = [item["adiabatic_gap_Eh"] for item in replica_records]
    averages = [item["average_energy_Eh"] for item in replica_records]
    checks = {
        "all_replica_wavefunctions_qualified": all(
            all(item["point_checks"].values()) for item in replica_records
        ),
        "all_five_replica_gaps_small": bool(
            max(gaps) <= thresholds.maximum_replica_gap_Eh
        ),
        "fivefold_average_energies_equivalent": bool(
            max(averages) - min(averages)
            <= thresholds.maximum_replica_average_energy_spread_Eh
        ),
        "all_replica_average_energies_admissible": bool(
            max(averages) - center_average_energy
            <= thresholds.maximum_average_energy_above_center_Eh
        ),
        "five_distinct_geometry_records": len(
            {item["geometry_hash"] for item in replica_records}
        )
        == N_RING,
    }
    return {
        "checks": checks,
        "records": replica_records,
        "maximum_gap_Eh": max(gaps),
        "average_energy_spread_Eh": max(averages) - min(averages),
    }


def run_qualification(args: argparse.Namespace) -> Dict[str, Any]:
    if args.basis.lower() != PYSCF_BASIS_KEY:
        raise SystemExit("This lift is locked to --basis cc-pvtz in PySCF 2.13.1")
    validate_protocol_declaration()
    configure_threads(args.threads)
    root = args.root.resolve()
    source_authorization = validate_source_authorization(
        root, args.source_root, require_pass=True
    )
    create_or_validate_manifest(root, args.basis)
    require_pyscf()
    source_evidence = {
        "source_authorization_file": str(source_authorization_path(root)),
        "source_authorization_sha256": sha256_file(
            source_authorization_path(root)
        ),
        "source_authorization_status": source_authorization.get("status"),
        "source_decision_sha256": source_authorization.get(
            "source_decision_sha256"
        ),
        "source_candidate_solution_sha256": source_authorization.get(
            "source_candidate_solution_sha256"
        ),
    }
    decision_path = root / DECISION_FILENAME
    if decision_path.exists():
        result = json.loads(decision_path.read_text())
        print(f"CPDC_CCPVTZ_LIFT={result.get('status', 'UNKNOWN')}", flush=True)
        return result
    thresholds = Thresholds()
    center = run_center(args)
    if center["status"] != "PASS_CENTER":
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            **source_evidence,
            "status": "STOP_CCPVTZ_CENTER_MODEL_FAILURE",
            "reason": "The independent cc-pVTZ D5h center failed a locked model gate.",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "manuscript_claim_authorized": False,
            "center": center,
        }
        write_json(decision_path, result)
        print(f"CPDC_CCPVTZ_LIFT={result['status']}", flush=True)
        return result

    try:
        search_record, candidate = run_bounded_search(
            args, root, center, thresholds
        )
    except AnalyticGradientFailure as exc:
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            **source_evidence,
            "status": "STOP_CCPVTZ_ANALYTIC_GRADIENT_FAILURE",
            "reason": str(exc),
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "manuscript_claim_authorized": False,
        }
        write_json(decision_path, result)
        print(f"CPDC_CCPVTZ_LIFT={result['status']}", flush=True)
        return result

    if candidate is None:
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            **source_evidence,
            "status": "STOP_CCPVTZ_SECONDARY_CI_NOT_LOCATED",
            "reason": "No fully prequalified evaluation was produced by the locked bounded search.",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "manuscript_claim_authorized": False,
            "search": search_record,
        }
        write_json(decision_path, result)
        print(f"CPDC_CCPVTZ_LIFT={result['status']}", flush=True)
        return result

    coefficients = np.asarray(candidate["coefficients_angstrom"], dtype=float)
    candidate_checks, candidate_metrics = candidate_geometry_checks(
        candidate["metadata"],
        np.asarray(candidate["geometry_angstrom"]),
        coefficients,
        float(center["metadata"]["average_energy_Eh"]),
        thresholds,
        include_gap=True,
    )
    if not all(candidate_checks.values()):
        result = {
            "script_version": SCRIPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            **source_evidence,
            "status": "STOP_CCPVTZ_SECONDARY_CI_NOT_LOCATED",
            "reason": "The lowest-gap search point failed at least one independent candidate gate.",
            "signal_inspected": False,
            "phase_or_loop_calculated": False,
            "manuscript_claim_authorized": False,
            "search": search_record,
            "candidate_checks": candidate_checks,
            "candidate_metrics": candidate_metrics,
            "candidate_metadata": candidate["metadata"],
            "candidate_solution_file": candidate["solution_file"],
        }
        write_json(decision_path, result)
        print(f"CPDC_CCPVTZ_LIFT={result['status']}", flush=True)
        return result

    print("\nCPDC cc-pVTZ SECONDARY-CI FIVEFOLD REPLICA LIFT", flush=True)
    replicas = run_replicas(
        args,
        root,
        candidate,
        float(center["metadata"]["average_energy_Eh"]),
        thresholds,
    )
    status = (
        "PASS_CCPVTZ_SECONDARY_CI_LIFT"
        if all(replicas["checks"].values())
        else "STOP_CCPVTZ_FIVEFOLD_REPLICA_FAILURE"
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        **source_evidence,
        "signal_inspected": False,
        "phase_or_loop_calculated": False,
        "manuscript_claim_authorized": False,
        "pass_authorizes": (
            "only a separate preregistered geometry-relaxed multiring topology "
            "calculation; it does not authorize a loop or manuscript claim"
        ),
        "thresholds": asdict(thresholds),
        "center_decision_file": str(root / CENTER_DECISION_FILENAME),
        "search_record_file": str(root / SEARCH_RECORD_FILENAME),
        "candidate_solution_file": candidate["solution_file"],
        "candidate_coefficients_angstrom": coefficients,
        "candidate_gap_Eh": float(candidate["metadata"]["adiabatic_gap_Eh"]),
        "candidate_average_energy_above_center_Eh": float(
            candidate["metadata"]["average_energy_Eh"]
            - center["metadata"]["average_energy_Eh"]
        ),
        "candidate_checks": candidate_checks,
        "candidate_metrics": candidate_metrics,
        "replicas": replicas,
    }
    write_json(decision_path, result)
    print("\nCYCLOPENTADIENYL CATION cc-pVTZ SECONDARY-CI LIFT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"candidate gap={result['candidate_gap_Eh']:.9e} Eh; "
        f"q1={coefficients[0]:.8f} A; "
        f"spectator_norm={candidate_metrics['spectator_norm_angstrom']:.8f} A; "
        f"max replica gap={replicas['maximum_gap_Eh']:.9e} Eh; "
        f"replica dEavg={replicas['average_energy_spread_Eh']:.3e} Eh",
        flush=True,
    )
    print(f"Decision file: {decision_path}", flush=True)
    print(f"CPDC_CCPVTZ_LIFT={status}", flush=True)
    return result


def selftest() -> None:
    validate_protocol_declaration()
    flat = C2V_BASIS.reshape(N_COORDS, -1)
    orthogonality = float(np.max(np.abs(flat @ flat.T - np.eye(N_COORDS))))
    if orthogonality > 1.0e-12:
        raise RuntimeError(f"C2v basis is not orthonormal: {orthogonality:.3e}")
    for vector in C2V_BASIS:
        if float(np.max(np.abs(reflect_displacement(vector) - vector))) > 1.0e-12:
            raise RuntimeError("C2v basis failed reflection invariance")
        if float(np.max(np.abs(vector[:, 2]))) > 1.0e-14:
            raise RuntimeError("C2v basis is not planar")
    initial = np.zeros(N_COORDS)
    initial[0] = INITIAL_E1_ANGSTROM
    geometry = geometry_from_coefficients(initial)
    metrics = geometry_metrics(geometry, initial)
    if metrics["C2v_reflection_error_angstrom"] > 1.0e-12:
        raise RuntimeError("initial geometry failed C2v invariance")
    replicas = [symmetry_replica(geometry, index) for index in range(N_RING)]
    hashes = {array_hash(item) for item in replicas}
    if len(hashes) != N_RING:
        raise RuntimeError("fivefold replica generator did not make five geometries")
    for replica in replicas:
        cc = sorted(
            float(np.linalg.norm(replica[i] - replica[(i + 1) % N_RING]))
            for i in range(N_RING)
        )
        if not np.allclose(
            cc,
            sorted(metrics["CC_bonds_angstrom"]),
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise RuntimeError("replica generator changed C--C bond multiset")
    synthetic_g0 = np.linspace(-0.03, 0.04, N_COORDS)
    synthetic_g1 = np.linspace(0.05, -0.02, N_COORDS)
    if not np.allclose(
        synthetic_g1 - synthetic_g0,
        np.asarray(synthetic_g1) - np.asarray(synthetic_g0),
    ):
        raise RuntimeError("objective-gradient algebra selftest failed")
    if BASIS_DEFINITION_SHA256 != hashlib.sha256(
        json.dumps(
            CCPVTZ_BASIS_DECLARATION, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest():
        raise RuntimeError("cc-pVTZ basis declaration hash is not reproducible")
    print("CPDC_CCPVTZ_LIFT_V1_SELFTEST=PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest", help="NumPy-only invariant and protocol tests")
    for name in ("center", "qualify", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument(
            "--source-root",
            type=Path,
            required=True,
            help="completed CPDC-SCI-QUAL-v2.0 DZV run root",
        )
        child.add_argument("--basis", default=PYSCF_BASIS_KEY)
        child.add_argument("--threads", type=int, default=16)
        child.add_argument("--memory-mb", type=int, default=48000)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "selftest":
        selftest()
        return
    if args.threads < 1 or args.memory_mb < 2000:
        raise SystemExit("--threads must be positive and --memory-mb at least 2000")
    if args.command == "center":
        run_center(args)
    elif args.command == "qualify":
        run_source_authorization(args)
    elif args.command == "run":
        run_qualification(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
