#!/usr/bin/env python3
"""Pre-registered LiH S1 locator on a physical R x static-field plane.

S1 follows the successful LiH S0 qualification without changing the electronic
model.  The two coordinates are the Li--H distance R and a uniform laboratory
electric field Fz.  The Hamiltonian convention is

    H(Fz) = H(0) - mu_z Fz,

implemented as ``h_core(Fz) = h_core(0) + Fz * r_z`` for the electrons and
``E_nuc(Fz) = E_nuc(0) - Fz * sum_A Z_A R_A,z`` for the nuclei.  Including both
terms makes the total energy of neutral LiH origin independent.

The fixed electronic model is equal-weight singlet-A1 SA4-CASSCF(2e,5o),
aug-cc-pVDZ, one optimized inactive core, no frozen orbital, and active irrep
content A1^3 B1^1 B2^1.  A fifth A1 CASCI root probes external-root isolation.
All state overlaps are exact active-only CAS-wavefunction overlaps, as qualified
by S0.  Each S1 vertex starts independently from its field-specific RHF
solution; S1 is a locator, not a history-uniqueness or manuscript gate.

The grid and numerical/physical gates are immutable defaults.  A candidate is
an elementary plaquette whose four edges pass strict fidelity and assignment
gates, whose conventional composed assignments agree, and which simultaneously
contains a roots-2/3 near-degeneracy, nonzero continuous holonomy, and a dipole
consequence.  A hit authorizes a separately pre-registered local refinement;
it never authorizes a manuscript claim by itself.

Typical use
-----------
python LiH_S1_physical_field_locator.py selftest
python LiH_S1_physical_field_locator.py all \
  --root ./work/LiH_S1_field_locator_work --threads 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import LiH_S0_SA4_qualification as s0


SCRIPT_VERSION = "LiH-S1-physical-field-locator-v1.7"
TARGET_PAIR = (2, 3)


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


@dataclass(frozen=True)
class Protocol:
    basis: str
    r_grid_angstrom: Tuple[float, ...]
    field_grid_au: Tuple[float, ...]


def field_tag(field_au: float) -> str:
    sign = "m" if field_au < 0.0 else "p"
    return f"F_{sign}{abs(field_au):.8f}".replace(".", "p")


def solution_path(root: Path, r_angstrom: float, field_au: float) -> Path:
    return root / "solutions" / field_tag(field_au) / (s0.float_tag("R", r_angstrom) + ".npz")


def run_rhf_field(mol, field_au: float):
    _, _, _, _, scf, _, _ = s0.require_pyscf()
    mf = scf.RHF(mol)
    hcore_zero = np.asarray(scf.hf.get_hcore(mol))
    r_z = np.asarray(mol.intor("int1e_r", comp=3)[2])
    enuc_zero = float(mol.energy_nuc())
    nuclear_dipole_z = float(np.dot(mol.atom_charges(), mol.atom_coords()[:, 2]))
    hcore_field = hcore_zero + field_au * r_z
    enuc_field = enuc_zero - field_au * nuclear_dipole_z
    mf.get_hcore = lambda *args: hcore_field
    mf.energy_nuc = lambda *args: enuc_field
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.get_hcore = lambda *args: hcore_field
        mf.energy_nuc = lambda *args: enuc_field
        mf.conv_tol = 1.0e-12
        mf.max_cycle = 100
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("field-dressed RHF failed to converge")
    return mf, {
        "field_au": field_au,
        "hamiltonian_convention": "H(Fz)=H(0)-mu_z*Fz",
        "electronic_hcore_term": "+Fz*r_z",
        "nuclear_energy_term_Eh": -field_au * nuclear_dipole_z,
        "nuclear_dipole_z_au": nuclear_dipole_z,
        "origin": "Li and H at z=-R/2 and +R/2",
    }


def casscf_job(
    r_angstrom: float,
    field_au: float,
    protocol: Protocol,
    out_path: Path,
    initial_from: Tuple[Any, np.ndarray] | None = None,
    history: str = "direct",
) -> Dict[str, Any]:
    fci, _, _, mcscf, _, _, _ = s0.require_pyscf()
    log_path = out_path.parent / (out_path.stem + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mol = s0.make_mol(r_angstrom, protocol.basis, str(log_path), verbose=4)
    mf, field_metadata = run_rhf_field(mol, field_au)
    if initial_from is None:
        mo_guess, selection = s0.build_initial_mos(mf)
        guess_semantics = (
            "independent field-specific RHF target-projector selection with fixed irrep quotas"
        )
    else:
        old_mol, old_mo = initial_from
        mo_guess = s0.project_mos(old_mol, old_mo, mol)
        selection = {
            "ncore": (mol.nelectron - s0.NELECAS) // 2,
            "projected_active_irrep_names": s0.orbital_irrep_names(mol, mo_guess)[
                1 : 1 + s0.NCAS
            ],
        }
        guess_semantics = (
            "symmetry-restored projection of the preceding converged field-CASSCF orbitals"
        )

    mc = mcscf.CASSCF(mf, s0.NCAS, s0.NELECAS)
    mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
    mc.fcisolver.wfnsym = s0.WFNSYM
    mc.fcisolver.nroots = s0.NROOTS
    mc = mc.state_average_([1.0 / s0.NROOTS] * s0.NROOTS)
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 1.0e-6
    mc.max_cycle_macro = 100
    mc.max_cycle_micro = 20
    mc.canonicalization = False
    mc.kernel(mo_guess)
    primary_converged = bool(mc.converged)
    fallback_attempts = 0
    while not bool(mc.converged) and fallback_attempts < 3:
        mc = mc.newton()
        mc.conv_tol = 1.0e-10
        mc.conv_tol_grad = 1.0e-6
        mc.max_cycle_macro = 100
        mc.max_cycle_micro = 20
        mc.canonicalization = False
        mc.kernel(mc.mo_coeff)
        fallback_attempts += 1
    fallback_used = fallback_attempts > 0

    ci_roots = [np.asarray(x) for x in mc.ci]
    state_energies = np.asarray(mc.e_states, dtype=float).reshape(-1)
    if len(ci_roots) != s0.NROOTS or state_energies.size != s0.NROOTS:
        raise RuntimeError(
            f"SA4 returned {len(ci_roots)} CI roots and {state_energies.size} energies"
        )
    try:
        grad = np.asarray(mc.get_grad(mc.mo_coeff))
        maximum_gradient = float(np.max(np.abs(grad))) if grad.size else 0.0
    except Exception:
        maximum_gradient = float("nan")

    final_mo = np.asarray(mc.mo_coeff)
    ncore = int(mc.ncore)
    active_irreps = s0.orbital_irrep_names(mol, final_mo)[ncore : ncore + s0.NCAS]
    probe_energies, _ = s0.five_root_probe(mf, final_mo)
    probe_error = float(np.max(np.abs(probe_energies[: s0.NROOTS] - state_energies)))
    excluded_gap = float(probe_energies[s0.NROOTS] - probe_energies[s0.NROOTS - 1])
    dipoles = s0.transition_dipoles(mol, final_mo, ci_roots, ncore)
    state_s2 = [
        float(fci.spin_op.spin_square0(ci, s0.NCAS, (1, 1))[0]) for ci in ci_roots
    ]
    s_ao = mol.intor_symmetric("int1e_ovlp")
    mo_error = float(
        np.max(np.abs(final_mo.conj().T @ s_ao @ final_mo - np.eye(final_mo.shape[1])))
    )
    target_gap = float(state_energies[TARGET_PAIR[1]] - state_energies[TARGET_PAIR[0]])
    metadata = {
        "script_version": SCRIPT_VERSION,
        "R_angstrom": r_angstrom,
        "field_z_au": field_au,
        "history": history,
        "basis": protocol.basis,
        "charge": 0,
        "spin_2S": 0,
        "point_group": str(mol.groupname),
        "target_state_symmetry": s0.WFNSYM,
        "nelecas": s0.NELECAS,
        "ncas": s0.NCAS,
        "nroots": s0.NROOTS,
        "retained_roots": list(range(s0.NROOTS)),
        "state_average_weights": [1.0 / s0.NROOTS] * s0.NROOTS,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "converged": bool(mc.converged),
        "optimizer_protocol": {
            "primary": "symmetry-adapted one-step SA-CASSCF",
            "primary_converged": primary_converged,
            "deterministic_fallback": (
                "symmetry-adapted second-order SA-CASSCF from primary final orbitals"
            ),
            "fallback_used": fallback_used,
            "fallback_attempts": fallback_attempts,
            "final_converged": bool(mc.converged),
        },
        "average_energy_Eh": float(mc.e_tot),
        "state_energies_Eh": state_energies,
        "target_pair": list(TARGET_PAIR),
        "target_pair_gap_Eh": target_gap,
        "five_root_probe_energies_Eh": probe_energies,
        "excluded_root_gap_Eh": excluded_gap,
        "probe_retained_energy_max_error_Eh": probe_error,
        "state_S2": state_s2,
        "maximum_orbital_gradient": maximum_gradient,
        "mo_orthonormality_max_error": mo_error,
        "active_irrep_names": active_irreps,
        "active_irrep_counts": dict(Counter(active_irreps)),
        "initial_guess_semantics": guess_semantics,
        "active_orbital_selection": selection,
        "field_hamiltonian": field_metadata,
        "pyscf_version": __import__("pyscf").__version__,
        "python_version": platform.python_version(),
        "timestamp_unix": time.time(),
    }

    tmp = out_path.with_suffix(".tmp.npz")
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(s0._jsonable(metadata))),
        "mo_coeff": final_mo,
        "state_energies_Eh": state_energies,
        "five_root_probe_energies_Eh": probe_energies,
        "dipole_au": dipoles,
    }
    for i, ci in enumerate(ci_roots):
        arrays[f"ci_root_{i}"] = ci
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)
    return metadata


def load_solution(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": str(path),
            "metadata": json.loads(str(data["metadata_json"].item())),
            "mo_coeff": np.asarray(data["mo_coeff"]),
            "state_energies_Eh": np.asarray(data["state_energies_Eh"]),
            "five_root_probe_energies_Eh": np.asarray(data["five_root_probe_energies_Eh"]),
            "dipole_au": np.asarray(data["dipole_au"]),
            "ci_roots": [np.asarray(data[f"ci_root_{i}"]) for i in range(s0.NROOTS)],
        }


def run_protocol(root: Path, protocol: Protocol, overwrite: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "gate": "LiH S1 physical-field locator",
        "scientific_semantics": (
            "Locate a strict direct-edge R x Fz window in the S0-qualified LiH four-state "
            "manifold; field is a physical Hamiltonian perturbation, not a protocol weight."
        ),
        "protocol": asdict(protocol),
        "thresholds": asdict(Thresholds()),
        "model_guards": {
            "molecule": "LiH",
            "point_group": "C2v",
            "basis": protocol.basis,
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
            "field_hamiltonian": "H(Fz)=H(0)-mu_z*Fz including electronic and nuclear terms",
            "vertex_initialization": "independent field-specific RHF starts",
            "adaptive_window_selection": False,
        },
    }
    s0.write_json(root / "LiH_S1_manifest.json", manifest)
    for field in protocol.field_grid_au:
        for r in protocol.r_grid_angstrom:
            target = solution_path(root, r, field)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                print(f"SKIP existing R={r:.8f} Fz={field:+.8f}: {target}", flush=True)
                continue
            print(f"RUN R={r:.8f} Fz={field:+.8f}", flush=True)
            try:
                md = casscf_job(r, field, protocol, target)
            except Exception as exc:
                s0.write_json(
                    target.with_suffix(".failed.json"),
                    {
                        "script_version": SCRIPT_VERSION,
                        "R_angstrom": r,
                        "field_z_au": field,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            print(
                f"DONE R={r:.8f} Fz={field:+.8f} "
                f"gap23={md['target_pair_gap_Eh']:.6e} "
                f"gap5={md['excluded_root_gap_Eh']:.6e} "
                f"grad={md['maximum_orbital_gradient']:.3e}",
                flush=True,
            )
    print("LiH S1 wavefunctions complete. Run the analyzer or use 'all'.", flush=True)


def route_result(route_r_then_f: np.ndarray, route_f_then_r: np.ndarray, dipole_d: np.ndarray) -> Dict[str, Any]:
    m = route_r_then_f.shape[0]
    delta = route_r_then_f - route_f_then_r
    c_qp = float(np.linalg.norm(delta, "fro") / (2.0 * math.sqrt(m)))
    holonomy = route_r_then_f.conj().T @ route_f_then_r
    phases = np.sort(np.angle(np.linalg.eigvals(holonomy)))
    mu1 = np.asarray(
        [route_r_then_f @ dipole_d[k] @ route_r_then_f.conj().T for k in range(3)]
    )
    mu2 = np.asarray(
        [route_f_then_r @ dipole_d[k] @ route_f_then_r.conj().T for k in range(3)]
    )
    by_axis = np.asarray([np.linalg.norm(mu1[k] - mu2[k], "fro") for k in range(3)])
    dip_norm = float(np.linalg.norm(mu1 - mu2))
    reference_norm = max(float(np.linalg.norm(dipole_d)), 1.0e-16)
    a1 = s0.assignment(route_r_then_f)
    a2 = s0.assignment(route_f_then_r)
    return {
        "C_qp": c_qp,
        "path_difference_F": float(np.linalg.norm(delta, "fro")),
        "holonomy_phases_rad": phases,
        "holonomy_max_abs_phase_rad": float(np.max(np.abs(phases))),
        "route_R_then_F_permutation": a1["permutation"],
        "route_F_then_R_permutation": a2["permutation"],
        "same_composed_assignment": a1["permutation"] == a2["permutation"],
        "dipole_path_difference_au": dip_norm,
        "dipole_path_difference_by_axis_au": by_axis,
        "dipole_path_difference_normalized": dip_norm / reference_norm,
    }


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conj(phases))


def model_semantics_pass(sol: Mapping[str, Any], protocol: Protocol) -> bool:
    md = sol["metadata"]
    return (
        md.get("script_version") == SCRIPT_VERSION
        and md.get("basis", "").lower() == protocol.basis.lower()
        and md.get("point_group") == "C2v"
        and md.get("target_state_symmetry") == s0.WFNSYM
        and int(md.get("nelecas", -1)) == s0.NELECAS
        and int(md.get("ncas", -1)) == s0.NCAS
        and int(md.get("nroots", -1)) == s0.NROOTS
        and int(md.get("ncore", -1)) == 1
        and int(md.get("frozen_orbitals", -1)) == 0
        and md.get("active_irrep_counts") == s0.ACTIVE_IRREP_COUNTS
        and md.get("field_hamiltonian", {}).get("hamiltonian_convention")
        == "H(Fz)=H(0)-mu_z*Fz"
    )


def analyze_protocol(root: Path, protocol: Protocol, thresholds: Thresholds) -> Dict[str, Any]:
    nr = len(protocol.r_grid_angstrom)
    nf = len(protocol.field_grid_au)
    solutions: Dict[Tuple[int, int], Dict[str, Any]] = {}
    mols = {
        i: s0.make_mol(r, protocol.basis, verbose=0)
        for i, r in enumerate(protocol.r_grid_angstrom)
    }
    for i, r in enumerate(protocol.r_grid_angstrom):
        for j, field in enumerate(protocol.field_grid_au):
            solutions[(i, j)] = load_solution(solution_path(root, r, field))

    terms_cache: MutableMapping[Tuple[str, int], Any] = {}
    edge_objects: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]] = {}
    edge_record_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]] = {}
    edge_records: List[Dict[str, Any]] = []

    def edge(a: Tuple[int, int], b: Tuple[int, int], kind: str) -> Dict[str, Any]:
        key = (a, b)
        if key not in edge_objects:
            sa = solutions[a]
            sb = solutions[b]
            matrix = s0.active_overlap_matrix(sa, sb, mols[a[0]], mols[b[0]], terms_cache)
            edge_objects[key] = s0.edge_metrics(matrix)
            obj = edge_objects[key]
            orb = s0.active_orbital_metrics(
                solutions[a], solutions[b], mols[a[0]], mols[b[0]]
            )
            record = {
                "edge": f"{kind}:R{a[0]}F{a[1]}->R{b[0]}F{b[1]}",
                "kind": kind,
                "R_stride_points": abs(b[0] - a[0]),
                "F_stride_points": abs(b[1] - a[1]),
                "elementary": abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1,
                "from": {
                    "R_angstrom": protocol.r_grid_angstrom[a[0]],
                    "field_z_au": protocol.field_grid_au[a[1]],
                },
                "to": {
                    "R_angstrom": protocol.r_grid_angstrom[b[0]],
                    "field_z_au": protocol.field_grid_au[b[1]],
                },
                "state_subspace": obj["metrics"],
                "active_orbital_subspace": orb,
            }
            edge_record_cache[key] = record
            edge_records.append(record)
        return edge_objects[key]

    for j in range(nf):
        for i in range(nr - 1):
            edge((i, j), (i + 1, j), "nuclear")
    for i in range(nr):
        for j in range(nf - 1):
            edge((i, j), (i, j + 1), "physical_field")

    def edge_pass(record: Mapping[str, Any]) -> bool:
        sm = record["state_subspace"]
        om = record["active_orbital_subspace"]
        return bool(
            sm["sigma_min"] >= thresholds.eta_edge_minimum
            and sm["leakage"] <= thresholds.maximum_edge_leakage
            and sm["normalized_margin"] >= thresholds.minimum_normalized_assignment_margin
            and om["sigma_min"] >= thresholds.active_orbital_subspace_sigma_minimum
            and om["leakage"] <= thresholds.active_orbital_subspace_leakage_maximum
        )

    plaquettes: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20260916)
    maximum_gauge_c_error = 0.0
    maximum_gauge_dipole_error = 0.0
    for r_stride in range(1, nr):
        for f_stride in range(1, nf):
            for i in range(nr - r_stride):
                for j in range(nf - f_stride):
                    a = (i, j)
                    b = (i + r_stride, j)
                    c = (i, j + f_stride)
                    d = (i + r_stride, j + f_stride)
                    ab = edge(a, b, "nuclear")
                    bd = edge(b, d, "physical_field")
                    ac = edge(a, c, "physical_field")
                    cd = edge(c, d, "nuclear")
                    edge_keys = [(a, b), (b, d), (a, c), (c, d)]
                    direct_records = [edge_record_cache[k] for k in edge_keys]
                    route1 = ab["Q"] @ bd["Q"]
                    route2 = ac["Q"] @ cd["Q"]
                    dip_d = solutions[d]["dipole_au"]
                    rr = route_result(route1, route2, dip_d)

                    four_edges_strict = all(edge_pass(record) for record in direct_records)
                    enclosed = [
                        (ii, jj)
                        for ii in range(i, i + r_stride + 1)
                        for jj in range(j, j + f_stride + 1)
                    ]
                    enclosed_gaps = [
                        float(solutions[v]["metadata"]["target_pair_gap_Eh"])
                        for v in enclosed
                    ]
                    enclosed_excluded = [
                        float(solutions[v]["metadata"]["excluded_root_gap_Eh"])
                        for v in enclosed
                    ]

                    gauges = {v: random_unitary(s0.NROOTS, rng) for v in (a, b, c, d)}
                    def gauged_q(x: Tuple[int, int], y: Tuple[int, int]) -> np.ndarray:
                        transformed = (
                            gauges[x].conj().T @ edge_objects[(x, y)]["matrix"] @ gauges[y]
                        )
                        return s0.polar_unitary(transformed)[0]
                    gr1 = gauged_q(a, b) @ gauged_q(b, d)
                    gr2 = gauged_q(a, c) @ gauged_q(c, d)
                    gd = gauges[d]
                    dip_g = np.asarray([gd.conj().T @ dip_d[k] @ gd for k in range(3)])
                    grr = route_result(gr1, gr2, dip_g)
                    gauge_c_error = abs(grr["C_qp"] - rr["C_qp"])
                    gauge_dip_error = abs(
                        grr["dipole_path_difference_au"]
                        - rr["dipole_path_difference_au"]
                    )
                    maximum_gauge_c_error = max(maximum_gauge_c_error, gauge_c_error)
                    maximum_gauge_dipole_error = max(
                        maximum_gauge_dipole_error, gauge_dip_error
                    )

                    candidate = bool(
                        four_edges_strict
                        and rr["same_composed_assignment"]
                        and min(enclosed_gaps) <= thresholds.target_pair_maximum_gap_Eh
                        and rr["C_qp"] >= thresholds.minimum_C_qp
                        and rr["dipole_path_difference_normalized"]
                        >= thresholds.minimum_normalized_dipole_path_difference
                    )
                    plaquettes.append(
                        {
                            "plaquette": (
                                f"R{i}->{i+r_stride}:F{j}->{j+f_stride}"
                            ),
                            "cell_type": (
                                "elementary"
                                if r_stride == 1 and f_stride == 1
                                else "cumulative_direct_edge"
                            ),
                            "R_stride_points": r_stride,
                            "F_stride_points": f_stride,
                            "R0_angstrom": protocol.r_grid_angstrom[i],
                            "R1_angstrom": protocol.r_grid_angstrom[i + r_stride],
                            "F0_au": protocol.field_grid_au[j],
                            "F1_au": protocol.field_grid_au[j + f_stride],
                            "target_pair": list(TARGET_PAIR),
                            "minimum_target_pair_gap_inside_window_Eh": min(enclosed_gaps),
                            "minimum_excluded_root_gap_inside_window_Eh": min(enclosed_excluded),
                            "all_four_direct_edges_strict": four_edges_strict,
                            "candidate_window": candidate,
                            **rr,
                            "gauge_test_C_qp_abs_error": gauge_c_error,
                            "gauge_test_dipole_abs_error": gauge_dip_error,
                            "edges": {
                                "AB": direct_records[0],
                                "BD": direct_records[1],
                                "AC": direct_records[2],
                                "CD": direct_records[3],
                            },
                        }
                    )

    all_solutions = list(solutions.values())
    gradients = [float(x["metadata"]["maximum_orbital_gradient"]) for x in all_solutions]
    maximum_gradient = max(gradients)
    maximum_s2 = max(
        abs(float(v)) for x in all_solutions for v in x["metadata"]["state_S2"]
    )
    maximum_mo_error = max(
        float(x["metadata"]["mo_orthonormality_max_error"]) for x in all_solutions
    )
    maximum_probe_error = max(
        float(x["metadata"]["probe_retained_energy_max_error_Eh"]) for x in all_solutions
    )
    minimum_excluded_gap = min(
        (
            {
                "R_angstrom": float(x["metadata"]["R_angstrom"]),
                "field_z_au": float(x["metadata"]["field_z_au"]),
                "gap_Eh": float(x["metadata"]["excluded_root_gap_Eh"]),
            }
            for x in all_solutions
        ),
        key=lambda x: x["gap_Eh"],
    )
    minimum_target_gap = min(
        (
            {
                "R_angstrom": float(x["metadata"]["R_angstrom"]),
                "field_z_au": float(x["metadata"]["field_z_au"]),
                "gap_Eh": float(x["metadata"]["target_pair_gap_Eh"]),
            }
            for x in all_solutions
        ),
        key=lambda x: x["gap_Eh"],
    )
    semantics = all(model_semantics_pass(x, protocol) for x in all_solutions)
    stationary = bool(
        all(bool(x["metadata"]["converged"]) for x in all_solutions)
        and all(np.isfinite(gradients))
        and maximum_gradient <= thresholds.maximum_final_orbital_gradient
        and maximum_s2 <= thresholds.maximum_spin_square
        and maximum_mo_error <= thresholds.maximum_mo_orthonormality_error
    )
    external_isolated = bool(
        minimum_excluded_gap["gap_Eh"] >= thresholds.excluded_root_minimum_gap_Eh
        and maximum_probe_error <= thresholds.probe_retained_energy_maximum_error_Eh
    )
    numerical_validation = bool(
        maximum_gauge_c_error <= thresholds.maximum_numerical_check_error
        and maximum_gauge_dipole_error <= thresholds.maximum_numerical_check_error
    )
    elementary_edges = [e for e in edge_records if e["elementary"]]
    all_elementary_edges_strict = bool(elementary_edges) and all(
        edge_pass(e) for e in elementary_edges
    )
    candidates = [p for p in plaquettes if p["candidate_window"]]
    ranked = sorted(
        plaquettes,
        key=lambda p: (
            p["candidate_window"],
            p["all_four_direct_edges_strict"],
            p["same_composed_assignment"],
            p["C_qp"],
            p["dipole_path_difference_normalized"],
            -p["minimum_target_pair_gap_inside_window_Eh"],
        ),
        reverse=True,
    )
    selected = ranked[0] if candidates else None

    if not numerical_validation:
        status = "STOP_NUMERICAL_VALIDATION_FAILURE"
    elif not semantics:
        status = "STOP_MODEL_SEMANTICS_FAILURE"
    elif not stationary:
        status = "STOP_WAVEFUNCTION_FAILURE"
    elif not external_isolated:
        status = "STOP_EXTERNAL_ROOT_ISOLATION_FAILURE"
    elif not candidates:
        status = "STOP_NO_STRICT_PHYSICAL_HOTSPOT"
    else:
        status = "CANDIDATE_WINDOW_FOUND"

    failed_edges = [e for e in edge_records if not edge_pass(e)]
    nonstationary = [
        {
            "path": x["path"],
            "R_angstrom": x["metadata"]["R_angstrom"],
            "field_z_au": x["metadata"]["field_z_au"],
            "converged": x["metadata"]["converged"],
            "maximum_orbital_gradient": x["metadata"]["maximum_orbital_gradient"],
            "state_S2": x["metadata"]["state_S2"],
            "mo_orthonormality_max_error": x["metadata"]["mo_orthonormality_max_error"],
        }
        for x in all_solutions
        if not bool(x["metadata"]["converged"])
        or not np.isfinite(float(x["metadata"]["maximum_orbital_gradient"]))
        or float(x["metadata"]["maximum_orbital_gradient"])
        > thresholds.maximum_final_orbital_gradient
        or max(abs(float(v)) for v in x["metadata"]["state_S2"])
        > thresholds.maximum_spin_square
        or float(x["metadata"]["mo_orthonormality_max_error"])
        > thresholds.maximum_mo_orthonormality_error
    ]

    decision = {
        "gate": "LiH S1 physical R x static-electric-field locator",
        "status": status,
        "manuscript_claim_authorized": False,
        "script_version": SCRIPT_VERSION,
        "model": {
            "molecule": "LiH",
            "basis": protocol.basis,
            "point_group": "C2v",
            "state_symmetry": s0.WFNSYM,
            "active_space": [s0.NELECAS, s0.NCAS],
            "active_irrep_counts": s0.ACTIVE_IRREP_COUNTS,
            "ncore": 1,
            "frozen_orbitals": 0,
            "optimized_and_retained_roots": s0.NROOTS,
            "target_pair": list(TARGET_PAIR),
            "state_average_weights": [1.0 / s0.NROOTS] * s0.NROOTS,
            "excluded_root_probe": "five-root A1 CASCI on each optimized SA4 orbital set",
            "edge_metric": "exact active-only CAS-wavefunction overlap",
            "physical_axis": "uniform static Fz with H(Fz)=H(0)-mu_z*Fz",
        },
        "pre_registered_grid": {
            "R_angstrom": protocol.r_grid_angstrom,
            "field_z_au": protocol.field_grid_au,
            "adaptive_window_selection": False,
        },
        "thresholds": asdict(thresholds),
        "decision_checks": {
            "numerical_validation": numerical_validation,
            "model_semantics_exact": semantics,
            "all_wavefunctions_stationary": stationary,
            "first_excluded_A1_root_isolated": external_isolated,
            "all_elementary_grid_edges_strict": all_elementary_edges_strict,
            "at_least_one_strict_candidate_plaquette": bool(candidates),
            "candidate_has_same_conventional_composed_assignment": bool(
                selected and selected["same_composed_assignment"]
            ),
            "candidate_has_nonzero_continuous_holonomy": bool(
                selected and selected["C_qp"] >= thresholds.minimum_C_qp
            ),
            "candidate_has_observable_dipole_consequence": bool(
                selected
                and selected["dipole_path_difference_normalized"]
                >= thresholds.minimum_normalized_dipole_path_difference
            ),
        },
        "counts": {
            "wavefunctions": len(all_solutions),
            "elementary_nuclear_edges": (nr - 1) * nf,
            "elementary_physical_field_edges": nr * (nf - 1),
            "direct_edges_evaluated_all_strides": len(edge_records),
            "elementary_plaquettes": (nr - 1) * (nf - 1),
            "cumulative_direct_edge_plaquettes": (
                len(plaquettes) - (nr - 1) * (nf - 1)
            ),
            "plaquettes_all_strides": len(plaquettes),
            "strict_candidate_plaquettes": len(candidates),
            "deterministic_fallback_uses": sum(
                bool(x["metadata"]["optimizer_protocol"]["fallback_used"])
                for x in all_solutions
            ),
        },
        "maximum_final_orbital_gradient": maximum_gradient,
        "maximum_state_S2": maximum_s2,
        "maximum_mo_orthonormality_error": maximum_mo_error,
        "maximum_probe_retained_energy_error_Eh": maximum_probe_error,
        "minimum_target_pair_gap": minimum_target_gap,
        "minimum_excluded_root_gap": minimum_excluded_gap,
        "maximum_gauge_C_qp_error": maximum_gauge_c_error,
        "maximum_gauge_dipole_error": maximum_gauge_dipole_error,
        "worst_elementary_state_edge": min(
            elementary_edges, key=lambda e: e["state_subspace"]["sigma_min"]
        ),
        "worst_evaluated_direct_edge_all_strides": min(
            edge_records, key=lambda e: e["state_subspace"]["sigma_min"]
        ),
        "highest_C_qp_plaquette": max(plaquettes, key=lambda p: p["C_qp"]),
        "selected_candidate_window": selected,
        "failed_edges": failed_edges,
        "nonstationary_wavefunctions": nonstationary,
        "authorized_next_step": (
            "Freeze the selected direct-edge R x Fz window and run opposed continuation-history controls plus an independent reproduction before any manuscript claim."
            if status == "CANDIDATE_WINDOW_FOUND"
            else "Stop S1 and diagnose only the first failed gate; do not move the grid or relax thresholds after inspecting the signal."
        ),
        "authorized_interpretation": (
            "A physical-field LiH hotspot satisfying local fidelity, conventional-assignment agreement, continuous holonomy, and dipole-consequence locator gates has been found."
            if status == "CANDIDATE_WINDOW_FOUND"
            else "S1 has not established a strict physical molecular hotspot."
        ),
        "prohibited_interpretations": [
            "publication-ready molecular false negative from S1 alone",
            "conical intersection proved by a small adiabatic gap",
            "physical effect inferred from a non-strict edge",
            "post hoc grid movement or threshold relaxation",
            "replacement of active-only overlap by scalar inactive-core division",
        ],
    }
    s0.write_json(root / "LiH_S1_decision.json", decision)
    s0.write_json(
        root / "LiH_S1_diagnostics.json",
        {"edges": edge_records, "plaquettes_ranked": ranked},
    )

    print("\nLiH S1 RESULT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"Minimum roots-2/3 gap={minimum_target_gap['gap_Eh']:.12e} Eh at "
        f"R={minimum_target_gap['R_angstrom']:.4f} A, "
        f"Fz={minimum_target_gap['field_z_au']:+.6f} au",
        flush=True,
    )
    print(
        f"Worst elementary edge: eta={min(e['state_subspace']['sigma_min'] for e in elementary_edges):.12f}; "
        f"leakage={max(e['state_subspace']['leakage'] for e in elementary_edges):.12f}; "
        f"margin={min(e['state_subspace']['normalized_margin'] for e in elementary_edges):.12f}",
        flush=True,
    )
    print(
        f"Maximum C_qp={max(p['C_qp'] for p in plaquettes):.12e}; "
        f"strict candidates={len(candidates)}; maximum gradient={maximum_gradient:.3e}",
        flush=True,
    )
    if selected is not None:
        print(
            f"Selected window: R={selected['R0_angstrom']:.4f}->{selected['R1_angstrom']:.4f} A; "
            f"Fz={selected['F0_au']:+.6f}->{selected['F1_au']:+.6f} au; "
            f"C_qp={selected['C_qp']:.6e}; "
            f"dipole_norm={selected['dipole_path_difference_normalized']:.6e}",
            flush=True,
        )
    print(f"Decision file: {root / 'LiH_S1_decision.json'}", flush=True)
    return decision


def selftest() -> None:
    assert s0.parse_grid("4.2:6.0:0.2")[0] == 4.2
    assert s0.parse_grid("4.2:6.0:0.2")[-1] == 6.0
    assert s0.parse_grid("0.010:0.022:0.002")[-1] == 0.022
    assert field_tag(0.012) == "F_p0p01200000"
    assert field_tag(-0.012) == "F_m0p01200000"
    rng = np.random.default_rng(31)
    matrices = [np.eye(s0.NROOTS) + 0.01 * rng.normal(size=(s0.NROOTS, s0.NROOTS)) for _ in range(4)]
    q = [s0.polar_unitary(x)[0] for x in matrices]
    dip = rng.normal(size=(3, s0.NROOTS, s0.NROOTS))
    dip = (dip + dip.transpose(0, 2, 1)) / 2.0
    reference = route_result(q[0] @ q[1], q[2] @ q[3], dip)
    gauges = [random_unitary(s0.NROOTS, rng) for _ in range(4)]
    # Synthetic A,B,C,D gauge transformation of the four overlap matrices.
    transformed = [
        gauges[0].conj().T @ matrices[0] @ gauges[1],
        gauges[1].conj().T @ matrices[1] @ gauges[3],
        gauges[0].conj().T @ matrices[2] @ gauges[2],
        gauges[2].conj().T @ matrices[3] @ gauges[3],
    ]
    tq = [s0.polar_unitary(x)[0] for x in transformed]
    gdip = np.asarray([gauges[3].conj().T @ dip[k] @ gauges[3] for k in range(3)])
    checked = route_result(tq[0] @ tq[1], tq[2] @ tq[3], gdip)
    assert abs(reference["C_qp"] - checked["C_qp"]) < 1.0e-12
    assert (
        abs(reference["dipole_path_difference_au"] - checked["dipole_path_difference_au"])
        < 1.0e-12
    )
    print("SELFTEST=PASS")


def build_protocol(args: argparse.Namespace) -> Protocol:
    r_grid = s0.parse_grid(args.r_grid)
    field_grid = s0.parse_grid(args.field_grid)
    if len(r_grid) < 2 or len(field_grid) < 2:
        raise ValueError("S1 requires at least two points along both R and Fz")
    if tuple(sorted(r_grid)) != r_grid or len(set(r_grid)) != len(r_grid):
        raise ValueError("R grid must be strictly increasing")
    if tuple(sorted(field_grid)) != field_grid or len(set(field_grid)) != len(field_grid):
        raise ValueError("field grid must be strictly increasing")
    return Protocol(args.basis, r_grid, field_grid)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run NumPy-only algebra and protocol tests")
    for name in ("run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument(
            "--root",
            type=Path,
            default=Path.cwd() / "work" / "LiH_S1_field_locator_work",
        )
        p.add_argument("--basis", default="aug-cc-pvdz")
        p.add_argument("--r-grid", default="5.8:6.0:0.05")
        p.add_argument("--field-grid", default="0.0105:0.0115:0.00025")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    s0.require_pyscf()
    import pyscf

    pyscf.lib.num_threads(args.threads)
    protocol = build_protocol(args)
    if args.command in ("run", "all"):
        run_protocol(args.root, protocol, args.overwrite)
    if args.command in ("analyze", "all"):
        analyze_protocol(args.root, protocol, Thresholds())


if __name__ == "__main__":
    main()
