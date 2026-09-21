#!/usr/bin/env python3
"""Pre-registered LiH S0 qualification gate for a later physical-field pilot.

Purpose
-------
This script does *not* search for a two-dimensional holonomy signal.  It first
asks whether a physically motivated LiH model is numerically and variationally
qualified for that search.  The fixed model is

* LiH, aug-cc-pVDZ, C2v subgroup, singlet A1 states;
* equal-weight SA4-CASSCF(2e,5o), no frozen orbital;
* one optimized inactive Li(1s)-like doubly occupied core;
* active-orbital irrep counts A1^3 B1^1 B2^1;
* four optimized/retained A1 roots;
* a five-root CASCI probe on the optimized orbitals to measure isolation from
  the first excluded A1 root;
* R = 2.3...4.2 Angstrom in fixed 0.05 Angstrom increments;
* independent direct starts plus short-bond-outward and long-bond-inward
  continuation histories.

All state-continuity gates use the exact overlap of the CAS active
wavefunctions with the shared inactive-core determinant omitted.  The active
orbitals remain embedded in the full AO Hilbert space, so changes of the
active orbital subspace are retained.  This is a quotient-space definition,
not the non-contractive scalar core division explored in the LiF audit.

S0 can only return ``PROMOTE_TO_PHYSICAL_FIELD_PILOT`` after stationarity,
history uniqueness, external-root isolation, active-orbital continuity, and
strict four-state edge fidelity all pass.  Promotion authorizes a separately
pre-registered R x electric-field pilot; it never authorizes a manuscript
claim.

Typical use
-----------
python LiH_S0_SA4_qualification.py selftest
python LiH_S0_SA4_qualification.py all \
  --root ./work/LiH_S0_SA4_work --threads 8

The calculation requires PySCF and NumPy.  The NumPy-only self-test can run
without PySCF.
"""

from __future__ import annotations

import argparse
import itertools
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


SCRIPT_VERSION = "LiH-S0-SA4-qualification-v1.2"
NELECAS = 2
NCAS = 5
NROOTS = 4
NPROBE_ROOTS = 5
WFNSYM = "A1"
ACTIVE_IRREP_COUNTS = {"A1": 3, "B1": 1, "B2": 1}


@dataclass(frozen=True)
class Thresholds:
    eta_edge_minimum: float = 0.98
    maximum_edge_leakage: float = 0.01
    minimum_normalized_assignment_margin: float = 0.10
    active_orbital_subspace_sigma_minimum: float = 0.90
    active_orbital_subspace_leakage_maximum: float = 0.10
    maximum_final_orbital_gradient: float = 1.0e-4
    history_control_maximum_average_energy_difference_Eh: float = 1.0e-8
    history_control_minimum_state_subspace_sigma: float = 0.9999
    history_control_maximum_state_subspace_leakage: float = 1.0e-6
    excluded_root_minimum_gap_Eh: float = 1.0e-2
    probe_retained_energy_maximum_error_Eh: float = 1.0e-8
    maximum_spin_square: float = 1.0e-7
    maximum_numerical_check_error: float = 1.0e-9


@dataclass(frozen=True)
class Protocol:
    basis: str
    r_grid_angstrom: Tuple[float, ...]
    short_anchor_angstrom: float
    long_anchor_angstrom: float
    history_step_angstrom: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def float_tag(prefix: str, x: float) -> str:
    sign = "m" if x < 0 else ""
    return f"{prefix}_{sign}{abs(x):.8f}".replace(".", "p")


def solution_path(root: Path, history: str, r_angstrom: float) -> Path:
    return root / "solutions" / history / (float_tag("R", r_angstrom) + ".npz")


def parse_grid(spec: str) -> Tuple[float, ...]:
    if ":" not in spec:
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if not vals:
            raise ValueError("empty grid")
        return tuple(vals)
    start, stop, step = (float(x) for x in spec.split(":"))
    if step <= 0.0 or stop < start:
        raise ValueError(f"invalid grid {spec!r}")
    n = int(round((stop - start) / step))
    vals = [round(start + i * step, 10) for i in range(n + 1)]
    if abs(vals[-1] - stop) > 1.0e-8:
        raise ValueError(f"grid endpoint is not reached exactly: {spec!r}")
    return tuple(vals)


def inclusive_path(start: float, stop: float, step: float) -> Tuple[float, ...]:
    direction = 1.0 if stop >= start else -1.0
    signed = direction * abs(step)
    n = int(round((stop - start) / signed))
    vals = [round(start + i * signed, 10) for i in range(n + 1)]
    if abs(vals[-1] - stop) > 1.0e-8:
        raise ValueError(f"history path {start}->{stop} is not divisible by {step}")
    return tuple(vals)


def require_pyscf():
    try:
        import pyscf  # noqa: F401
        from pyscf import fci, gto, lib, mcscf, scf, symm
        from pyscf.scf import addons as scf_addons
    except ImportError as exc:
        raise SystemExit(
            "PySCF is required for 'run', 'analyze', and 'all'. "
            "Run 'selftest' without PySCF or use the calculation environment."
        ) from exc
    return fci, gto, lib, mcscf, scf, scf_addons, symm


def make_mol(r_angstrom: float, basis: str, output: str | None = None, verbose: int = 0):
    _, gto, _, _, _, _, _ = require_pyscf()
    kwargs: Dict[str, Any] = {
        "atom": [("Li", (0.0, 0.0, -r_angstrom / 2.0)), ("H", (0.0, 0.0, r_angstrom / 2.0))],
        "unit": "Angstrom",
        "basis": basis,
        "charge": 0,
        "spin": 0,
        "symmetry": "C2v",
        "verbose": verbose,
    }
    if output is not None:
        kwargs["output"] = output
    return gto.M(**kwargs)


def run_rhf(mol):
    _, _, _, _, scf, _, _ = require_pyscf()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1.0e-12
        mf.max_cycle = 100
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF failed to converge")
    return mf


def ao_indices(mol, element: str, shell: str, component: str | None = None) -> List[int]:
    result: List[int] = []
    element = element.lower()
    shell = shell.lower()
    component = None if component is None else component.lower()
    for i, label in enumerate(mol.ao_labels(fmt=False)):
        parts = [str(x).strip().lower().replace(" ", "") for x in label]
        joined = "|".join(parts)
        if element not in parts:
            continue
        if shell not in joined:
            continue
        if component is not None and component not in joined:
            continue
        result.append(i)
    return result


def subspace_population(mo: np.ndarray, s_ao: np.ndarray, ao_idx: Sequence[int]) -> np.ndarray:
    if not ao_idx:
        raise RuntimeError("AO target subspace is empty")
    idx = np.asarray(ao_idx, dtype=int)
    cross = s_ao[:, idx]
    gram = s_ao[np.ix_(idx, idx)]
    amplitudes = mo.conj().T @ cross
    return np.einsum(
        "pi,ij,pj->p",
        amplitudes.conj(),
        np.linalg.pinv(gram, rcond=1.0e-12),
        amplitudes,
    ).real


def orbital_irrep_names(mol, mo: np.ndarray) -> List[str]:
    _, _, _, _, _, _, symm = require_pyscf()
    labels = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo, check=False)
    return [str(x) for x in labels]


def build_initial_mos(mf) -> Tuple[np.ndarray, Dict[str, Any]]:
    mol = mf.mol
    mo = np.asarray(mf.mo_coeff)
    occ = np.where(np.asarray(mf.mo_occ) > 0.0)[0].tolist()
    expected_ncore = (mol.nelectron - NELECAS) // 2
    if expected_ncore != 1 or len(occ) != expected_ncore + 1:
        raise RuntimeError(
            f"unexpected LiH occupation: ncore={expected_ncore}, occupied MOs={len(occ)}"
        )
    core = [min(occ, key=lambda i: float(mf.mo_energy[i]))]
    candidates = [i for i in range(mo.shape[1]) if i not in core]
    s_ao = mf.get_ovlp()

    targets = {
        "Li_2s": ao_indices(mol, "li", "2s"),
        "Li_2pz": ao_indices(mol, "li", "2p", "z"),
        "H_1s": ao_indices(mol, "h", "1s"),
        "H_2px": ao_indices(mol, "h", "2p", "x"),
        "H_2py": ao_indices(mol, "h", "2p", "y"),
    }
    missing = [name for name, idx in targets.items() if not idx]
    if missing:
        raise RuntimeError(f"could not identify target AO labels: {missing}; labels={mol.ao_labels()}")
    scores_by_target = {name: subspace_population(mo, s_ao, idx) for name, idx in targets.items()}
    total_score = sum(scores_by_target.values())
    irreps = orbital_irrep_names(mol, mo)

    active: List[int] = []
    selected_by_irrep: Dict[str, List[int]] = {}
    for irrep, count in ACTIVE_IRREP_COUNTS.items():
        eligible = [i for i in candidates if irreps[i] == irrep]
        eligible.sort(key=lambda i: (float(total_score[i]), -float(mf.mo_energy[i])), reverse=True)
        if len(eligible) < count:
            raise RuntimeError(f"not enough {irrep} orbitals for active-space quota {count}")
        chosen = eligible[:count]
        selected_by_irrep[irrep] = chosen
        active.extend(chosen)
    active.sort()
    if len(active) != NCAS or len(set(active)) != NCAS:
        raise RuntimeError(f"invalid active selection: {active}")
    active_occupied = [i for i in active if i in occ]
    if len(active_occupied) != 1:
        raise RuntimeError(
            f"CAS(2e,5o) seed must contain exactly one occupied valence MO; got {active_occupied}"
        )
    external = [i for i in range(mo.shape[1]) if i not in core and i not in active]
    order = core + active + external
    if len(order) != mo.shape[1] or len(set(order)) != mo.shape[1]:
        raise RuntimeError("MO reordering is not a permutation")

    diagnostics = {
        "ncore": expected_ncore,
        "core_canonical_index": core[0],
        "active_canonical_indices": active,
        "active_irrep_names": [irreps[i] for i in active],
        "selected_by_irrep": selected_by_irrep,
        "active_target_population_total": [float(total_score[i]) for i in active],
        "target_populations_on_active": {
            name: [float(scores[i]) for i in active] for name, scores in scores_by_target.items()
        },
        "target_ao_indices": targets,
        "mo_reordering": order,
    }
    return mo[:, order], diagnostics


def metric_orthonormalize(mo: np.ndarray, s_ao: np.ndarray) -> np.ndarray:
    gram = mo.conj().T @ s_ao @ mo
    gram = (gram + gram.conj().T) / 2.0
    eig, vec = np.linalg.eigh(gram)
    if float(np.min(eig)) < 1.0e-10:
        raise RuntimeError(f"projected MO Gram matrix is singular: {np.min(eig):.3e}")
    return mo @ (vec @ np.diag(eig ** -0.5) @ vec.conj().T)


def project_mos(old_mol, old_mo: np.ndarray, new_mol) -> np.ndarray:
    _, _, _, _, _, scf_addons, symm = require_pyscf()
    projected = np.asarray(scf_addons.project_mo_nr2nr(old_mol, old_mo, new_mol))
    s_ao = new_mol.intor_symmetric("int1e_ovlp")
    projected = metric_orthonormalize(projected, s_ao)
    projected = np.asarray(symm.symmetrize_space(new_mol, projected, s=s_ao, check=False))
    return metric_orthonormalize(projected, s_ao)


def transition_dipoles(mol, mo: np.ndarray, ci_roots: Sequence[np.ndarray], ncore: int) -> np.ndarray:
    fci, _, _, _, _, _, _ = require_pyscf()
    mo_core = mo[:, :ncore]
    mo_cas = mo[:, ncore : ncore + NCAS]
    r_ao = mol.intor("int1e_r", comp=3)
    core_dm = 2.0 * mo_core @ mo_core.conj().T
    nuclear = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords())
    dip = np.zeros((3, NROOTS, NROOTS), dtype=np.complex128)
    for i in range(NROOTS):
        for j in range(NROOTS):
            dm_act = fci.direct_spin1.trans_rdm1(ci_roots[i], ci_roots[j], NCAS, (1, 1))
            dm_ao = np.einsum("pi,ij,qj->pq", mo_cas, dm_act, mo_cas.conj())
            if i == j:
                dm_ao = dm_ao + core_dm
            electronic = -np.einsum("xpq,qp->x", r_ao, dm_ao)
            dip[:, i, j] = electronic + (nuclear if i == j else 0.0)
    return (dip + dip.conj().transpose(0, 2, 1)) / 2.0


def five_root_probe(mf, mo: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return the five lowest singlet-A1 roots by deterministic dense CASCI.

    PySCF's iterative multi-root solver can occasionally fail to converge a
    fifth root while still returning an energy array.  Dense diagonalization
    is trivial for CAS(2e,5o), but the M_s=0 determinant space also contains
    triplets.  We therefore diagonalize the complete A1 block, evaluate S^2,
    and retain the five lowest roots with S^2=0.  This turns the external-root
    probe into a checked diagnostic rather than trusting an unconverged
    Davidson return value.
    """
    fci, _, _, mcscf, _, _, _ = require_pyscf()
    probe = mcscf.CASCI(mf, NCAS, NELECAS)
    probe.fcisolver = fci.direct_spin1_symm.FCI(mf.mol)
    probe.fcisolver.wfnsym = WFNSYM
    probe.fcisolver.nroots = 10
    probe.fcisolver.davidson_only = False
    probe.kernel(mo)
    if not bool(probe.converged):
        raise RuntimeError("dense singlet-A1 CASCI probe did not converge")
    raw_energies = np.asarray(probe.e_tot, dtype=float).reshape(-1)
    raw_ci = [np.asarray(x) for x in probe.ci]
    singlets: List[Tuple[float, np.ndarray]] = []
    for energy, ci_root in zip(raw_energies, raw_ci):
        s2 = float(fci.spin_op.spin_square0(ci_root, NCAS, (1, 1))[0])
        if abs(s2) <= 1.0e-7:
            singlets.append((float(energy), ci_root))
    singlets.sort(key=lambda item: item[0])
    if len(singlets) < NPROBE_ROOTS:
        raise RuntimeError(
            f"dense A1 probe found only {len(singlets)} singlets among "
            f"{raw_energies.size} M_s=0 roots"
        )
    selected = singlets[:NPROBE_ROOTS]
    energies = np.asarray([item[0] for item in selected], dtype=float)
    ci = [item[1] for item in selected]
    return energies, ci


def casscf_job(
    r_angstrom: float,
    protocol: Protocol,
    history: str,
    out_path: Path,
    initial_from: Tuple[Any, np.ndarray] | None,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    fci, _, _, mcscf, _, _, _ = require_pyscf()
    log_path = out_path.parent / (out_path.stem + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mol = make_mol(r_angstrom, protocol.basis, str(log_path), verbose=4)
    mf = run_rhf(mol)

    if initial_from is None:
        mo_guess, selection = build_initial_mos(mf)
        guess_semantics = "independent RHF target-projector selection with fixed irrep quotas"
    else:
        old_mol, old_mo = initial_from
        mo_guess = project_mos(old_mol, old_mo, mol)
        selection = {
            "ncore": (mol.nelectron - NELECAS) // 2,
            "projected_active_irrep_names": orbital_irrep_names(mol, mo_guess)[1 : 1 + NCAS],
        }
        guess_semantics = "symmetry-restored projection of preceding converged CASSCF orbitals"

    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
    mc.fcisolver.wfnsym = WFNSYM
    mc.fcisolver.nroots = NROOTS
    mc = mc.state_average_([1.0 / NROOTS] * NROOTS)
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 3.0e-6
    mc.max_cycle_macro = 100
    mc.max_cycle_micro = 20
    mc.canonicalization = False
    mc.kernel(mo_guess)
    first_order_converged = bool(mc.converged)
    second_order_fallback_used = False
    if not first_order_converged:
        # Deterministic solver fallback, not a new starting history or a
        # solution-selection rule.  PySCF's symmetry-adapted second-order
        # CASSCF optimizer is restarted from the stalled first-order orbitals
        # and must converge to the same immutable stationary objective.
        mc = mc.newton()
        mc.conv_tol = 1.0e-10
        mc.conv_tol_grad = 3.0e-6
        mc.max_cycle_macro = 100
        mc.max_cycle_micro = 20
        mc.canonicalization = False
        mc.kernel(mc.mo_coeff)
        second_order_fallback_used = True

    ci_roots = [np.asarray(x) for x in mc.ci]
    e_states = np.asarray(mc.e_states, dtype=float).reshape(-1)
    if len(ci_roots) != NROOTS or e_states.size != NROOTS:
        raise RuntimeError(f"SA4 returned {len(ci_roots)} CI roots and {e_states.size} energies")
    try:
        grad = np.asarray(mc.get_grad(mc.mo_coeff))
        maximum_gradient = float(np.max(np.abs(grad))) if grad.size else 0.0
    except Exception:
        maximum_gradient = float("nan")

    ncore = int(mc.ncore)
    final_mo = np.asarray(mc.mo_coeff)
    active_irreps = orbital_irrep_names(mol, final_mo)[ncore : ncore + NCAS]
    probe_energies, _ = five_root_probe(mf, final_mo)
    probe_consistency = float(np.max(np.abs(probe_energies[:NROOTS] - e_states)))
    excluded_gap = float(probe_energies[NROOTS] - probe_energies[NROOTS - 1])
    dipoles = transition_dipoles(mol, final_mo, ci_roots, ncore)
    state_s2 = [float(fci.spin_op.spin_square0(ci, NCAS, (1, 1))[0]) for ci in ci_roots]
    s_ao = mol.intor_symmetric("int1e_ovlp")
    mo_orth_error = float(
        np.max(np.abs(final_mo.conj().T @ s_ao @ final_mo - np.eye(final_mo.shape[1])))
    )
    metadata = {
        "script_version": SCRIPT_VERSION,
        "history": history,
        "R_angstrom": r_angstrom,
        "basis": protocol.basis,
        "charge": 0,
        "spin_2S": 0,
        "point_group": str(mol.groupname),
        "target_state_symmetry": WFNSYM,
        "nelecas": NELECAS,
        "ncas": NCAS,
        "nroots": NROOTS,
        "retained_roots": list(range(NROOTS)),
        "state_average_weights": [1.0 / NROOTS] * NROOTS,
        "ncore": ncore,
        "frozen_orbitals": 0,
        "converged": bool(mc.converged),
        "optimizer_protocol": {
            "primary": "symmetry-adapted one-step SA-CASSCF",
            "primary_converged": first_order_converged,
            "deterministic_fallback": "symmetry-adapted second-order SA-CASSCF from primary final orbitals",
            "fallback_used": second_order_fallback_used,
            "final_converged": bool(mc.converged),
        },
        "average_energy_Eh": float(mc.e_tot),
        "state_energies_Eh": e_states,
        "five_root_probe_energies_Eh": probe_energies,
        "excluded_root_gap_Eh": excluded_gap,
        "probe_retained_energy_max_error_Eh": probe_consistency,
        "state_S2": state_s2,
        "maximum_orbital_gradient": maximum_gradient,
        "mo_orthonormality_max_error": mo_orth_error,
        "active_irrep_names": active_irreps,
        "active_irrep_counts": dict(Counter(active_irreps)),
        "initial_guess_semantics": guess_semantics,
        "active_orbital_selection": selection,
        "pyscf_version": __import__("pyscf").__version__,
        "python_version": platform.python_version(),
        "timestamp_unix": time.time(),
    }

    tmp = out_path.with_suffix(".tmp.npz")
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata))),
        "mo_coeff": final_mo,
        "state_energies_Eh": e_states,
        "five_root_probe_energies_Eh": probe_energies,
        "dipole_au": dipoles,
    }
    for i, ci in enumerate(ci_roots):
        arrays[f"ci_root_{i}"] = ci
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)
    return mol, final_mo, metadata


def load_solution(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "path": str(path),
            "metadata": metadata,
            "mo_coeff": np.asarray(data["mo_coeff"]),
            "state_energies_Eh": np.asarray(data["state_energies_Eh"]),
            "five_root_probe_energies_Eh": np.asarray(data["five_root_probe_energies_Eh"]),
            "dipole_au": np.asarray(data["dipole_au"]),
            "ci_roots": [np.asarray(data[f"ci_root_{i}"]) for i in range(NROOTS)],
        }


def run_sequence(
    root: Path,
    protocol: Protocol,
    history: str,
    r_path: Sequence[float],
    overwrite: bool,
) -> None:
    previous: Tuple[Any, np.ndarray] | None = None
    for r in r_path:
        target = solution_path(root, history, r)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            sol = load_solution(target)
            previous = (make_mol(r, protocol.basis, verbose=0), sol["mo_coeff"])
            print(f"SKIP existing {history} R={r:.8f}: {target}", flush=True)
            continue
        print(f"RUN {history} R={r:.8f}", flush=True)
        try:
            mol, mo, metadata = casscf_job(r, protocol, history, target, previous)
        except Exception as exc:
            write_json(
                target.with_suffix(".failed.json"),
                {
                    "script_version": SCRIPT_VERSION,
                    "history": history,
                    "R_angstrom": r,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        previous = (mol, mo)
        print(
            f"DONE {history} R={r:.8f} Eavg={metadata['average_energy_Eh']:.12f} "
            f"grad={metadata['maximum_orbital_gradient']:.3e} "
            f"gap5={metadata['excluded_root_gap_Eh']:.3e}",
            flush=True,
        )


def run_protocol(root: Path, protocol: Protocol, overwrite: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "gate": "LiH SA4/CAS(2e,5o) one-dimensional S0 qualification",
        "scientific_semantics": (
            "Qualification of the four-state singlet-A1 LiH ionic/covalent manifold before any "
            "R x electric-field holonomy search."
        ),
        "protocol": asdict(protocol),
        "thresholds": asdict(Thresholds()),
        "model_guards": {
            "molecule": "LiH",
            "point_group": "C2v",
            "state_symmetry": WFNSYM,
            "basis": protocol.basis,
            "active_space": [NELECAS, NCAS],
            "active_irrep_counts": ACTIVE_IRREP_COUNTS,
            "optimized_roots": NROOTS,
            "retained_roots": list(range(NROOTS)),
            "state_average_weights": [1.0 / NROOTS] * NROOTS,
            "probe_roots": NPROBE_ROOTS,
            "frozen_orbitals": 0,
            "edge_metric": "exact active-only CAS-wavefunction overlap",
            "adaptive_window_selection": False,
        },
    }
    write_json(root / "LiH_S0_manifest.json", manifest)

    for r in protocol.r_grid_angstrom:
        target = solution_path(root, "direct", r)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            print(f"SKIP existing direct R={r:.8f}: {target}", flush=True)
            continue
        print(f"RUN direct R={r:.8f}", flush=True)
        _, _, metadata = casscf_job(r, protocol, "direct", target, None)
        print(
            f"DONE direct R={r:.8f} Eavg={metadata['average_energy_Eh']:.12f} "
            f"grad={metadata['maximum_orbital_gradient']:.3e} "
            f"gap5={metadata['excluded_root_gap_Eh']:.3e}",
            flush=True,
        )

    short_path = inclusive_path(
        protocol.short_anchor_angstrom, protocol.r_grid_angstrom[-1], protocol.history_step_angstrom
    )
    long_path = inclusive_path(
        protocol.long_anchor_angstrom, protocol.r_grid_angstrom[0], protocol.history_step_angstrom
    )
    run_sequence(root, protocol, "short_up", short_path, overwrite)
    run_sequence(root, protocol, "long_down", long_path, overwrite)
    print("LiH S0 wavefunctions complete. Run the analyzer or use 'all'.", flush=True)


def active_ci_terms(ci: np.ndarray) -> List[Tuple[complex, Tuple[int, ...], Tuple[int, ...]]]:
    fci, _, _, _, _, _, _ = require_pyscf()
    alpha_strings = fci.cistring.make_strings(range(NCAS), 1)
    beta_strings = fci.cistring.make_strings(range(NCAS), 1)
    matrix = np.asarray(ci).reshape(len(alpha_strings), len(beta_strings))
    terms: List[Tuple[complex, Tuple[int, ...], Tuple[int, ...]]] = []
    for ia, sa in enumerate(alpha_strings):
        occ_a = tuple(k for k in range(NCAS) if (int(sa) >> k) & 1)
        for ib, sb in enumerate(beta_strings):
            coeff = complex(matrix[ia, ib])
            if abs(coeff) < 1.0e-14:
                continue
            occ_b = tuple(k for k in range(NCAS) if (int(sb) >> k) & 1)
            terms.append((coeff, occ_a, occ_b))
    return terms


def active_overlap_matrix(
    sol_a: Mapping[str, Any],
    sol_b: Mapping[str, Any],
    mol_a,
    mol_b,
    terms_cache: MutableMapping[Tuple[str, int], Any],
) -> np.ndarray:
    _, gto, _, _, _, _, _ = require_pyscf()
    ncore_a = int(sol_a["metadata"]["ncore"])
    ncore_b = int(sol_b["metadata"]["ncore"])
    if ncore_a != 1 or ncore_b != 1:
        raise RuntimeError(f"S0 requires exactly one inactive core at both vertices: {ncore_a}, {ncore_b}")
    s_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    ca = sol_a["mo_coeff"][:, ncore_a : ncore_a + NCAS]
    cb = sol_b["mo_coeff"][:, ncore_b : ncore_b + NCAS]
    x_active = ca.conj().T @ s_ab @ cb
    result = np.zeros((NROOTS, NROOTS), dtype=np.complex128)
    for i in range(NROOTS):
        key_i = (str(sol_a["path"]), i)
        if key_i not in terms_cache:
            terms_cache[key_i] = active_ci_terms(sol_a["ci_roots"][i])
        for j in range(NROOTS):
            key_j = (str(sol_b["path"]), j)
            if key_j not in terms_cache:
                terms_cache[key_j] = active_ci_terms(sol_b["ci_roots"][j])
            value = 0.0j
            for coeff_a, occ_aa, occ_ab in terms_cache[key_i]:
                for coeff_b, occ_ba, occ_bb in terms_cache[key_j]:
                    det_a = np.linalg.det(x_active[np.ix_(occ_aa, occ_ba)])
                    det_b = np.linalg.det(x_active[np.ix_(occ_ab, occ_bb)])
                    value += np.conj(coeff_a) * coeff_b * det_a * det_b
            result[i, j] = value
    return result


def active_orbital_metrics(sol_a: Mapping[str, Any], sol_b: Mapping[str, Any], mol_a, mol_b) -> Dict[str, Any]:
    _, gto, _, _, _, _, _ = require_pyscf()
    ncore_a = int(sol_a["metadata"]["ncore"])
    ncore_b = int(sol_b["metadata"]["ncore"])
    s_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    ca = sol_a["mo_coeff"][:, ncore_a : ncore_a + NCAS]
    cb = sol_b["mo_coeff"][:, ncore_b : ncore_b + NCAS]
    singular_values = np.linalg.svd(ca.conj().T @ s_ab @ cb, compute_uv=False)
    return {
        "singular_values": singular_values,
        "sigma_min": float(np.min(singular_values)),
        "leakage": float(1.0 - np.mean(singular_values**2)),
    }


def polar_unitary(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u, singular_values, vh = np.linalg.svd(matrix)
    return u @ vh, singular_values


def assignment(matrix: np.ndarray) -> Dict[str, Any]:
    m = matrix.shape[0]
    scored: List[Tuple[float, Tuple[int, ...]]] = []
    for perm in itertools.permutations(range(m)):
        scored.append((float(sum(abs(matrix[i, perm[i]]) for i in range(m))), perm))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_perm = scored[0]
    runner = scored[1][0]
    margin = best_score - runner
    return {
        "permutation": list(best_perm),
        "score": best_score,
        "runner_up_score": runner,
        "margin": margin,
        "normalized_margin": margin / float(m),
    }


def edge_metrics(matrix: np.ndarray) -> Dict[str, Any]:
    q, singular_values = polar_unitary(matrix)
    return {
        "matrix": matrix,
        "Q": q,
        "metrics": {
            "singular_values": singular_values,
            "sigma_min": float(np.min(singular_values)),
            "leakage": float(1.0 - np.mean(singular_values**2)),
            **assignment(matrix),
        },
    }


def model_semantics_pass(sol: Mapping[str, Any]) -> bool:
    md = sol["metadata"]
    return (
        md.get("point_group") == "C2v"
        and md.get("target_state_symmetry") == WFNSYM
        and int(md.get("nelecas", -1)) == NELECAS
        and int(md.get("ncas", -1)) == NCAS
        and int(md.get("nroots", -1)) == NROOTS
        and int(md.get("ncore", -1)) == 1
        and int(md.get("frozen_orbitals", -1)) == 0
        and md.get("active_irrep_counts") == ACTIVE_IRREP_COUNTS
    )


def analyze_protocol(root: Path, protocol: Protocol, thresholds: Thresholds) -> Dict[str, Any]:
    direct = {r: load_solution(solution_path(root, "direct", r)) for r in protocol.r_grid_angstrom}
    short_full = inclusive_path(
        protocol.short_anchor_angstrom, protocol.r_grid_angstrom[-1], protocol.history_step_angstrom
    )
    long_full = inclusive_path(
        protocol.long_anchor_angstrom, protocol.r_grid_angstrom[0], protocol.history_step_angstrom
    )
    short = {r: load_solution(solution_path(root, "short_up", r)) for r in short_full}
    long = {r: load_solution(solution_path(root, "long_down", r)) for r in long_full}
    mols = {
        r: make_mol(r, protocol.basis, verbose=0)
        for r in sorted(set(protocol.r_grid_angstrom) | set(short_full) | set(long_full))
    }
    terms_cache: Dict[Tuple[str, int], Any] = {}

    edge_records: List[Dict[str, Any]] = []
    edge_objects: List[Dict[str, Any]] = []
    for r0, r1 in zip(protocol.r_grid_angstrom[:-1], protocol.r_grid_angstrom[1:]):
        obj = edge_metrics(active_overlap_matrix(direct[r0], direct[r1], mols[r0], mols[r1], terms_cache))
        orb = active_orbital_metrics(direct[r0], direct[r1], mols[r0], mols[r1])
        edge_objects.append(obj)
        edge_records.append(
            {
                "edge": f"R={r0:.8f}->{r1:.8f}",
                "R0_angstrom": r0,
                "R1_angstrom": r1,
                "state_subspace": obj["metrics"],
                "active_orbital_subspace": orb,
            }
        )

    history_records: List[Dict[str, Any]] = []
    for r in protocol.r_grid_angstrom:
        for history, control in (("short_up", short[r]), ("long_down", long[r])):
            obj = edge_metrics(active_overlap_matrix(direct[r], control, mols[r], mols[r], terms_cache))
            history_records.append(
                {
                    "history": history,
                    "R_angstrom": r,
                    "average_energy_difference_Eh": abs(
                        float(direct[r]["metadata"]["average_energy_Eh"])
                        - float(control["metadata"]["average_energy_Eh"])
                    ),
                    "state_subspace": obj["metrics"],
                    "active_orbital_subspace": active_orbital_metrics(
                        direct[r], control, mols[r], mols[r]
                    ),
                }
            )

    target_solutions = (
        list(direct.values())
        + [short[r] for r in short_full]
        + [long[r] for r in long_full]
    )
    gradients = [float(s["metadata"]["maximum_orbital_gradient"]) for s in target_solutions]
    maximum_gradient = max(gradients, default=float("nan"))
    maximum_s2 = max(
        (abs(float(x)) for s in target_solutions for x in s["metadata"]["state_S2"]),
        default=float("nan"),
    )
    maximum_mo_error = max(
        (float(s["metadata"]["mo_orthonormality_max_error"]) for s in target_solutions),
        default=float("nan"),
    )
    stationary = (
        all(bool(s["metadata"]["converged"]) for s in target_solutions)
        and all(np.isfinite(x) for x in gradients)
        and maximum_gradient <= thresholds.maximum_final_orbital_gradient
        and maximum_s2 <= thresholds.maximum_spin_square
        and maximum_mo_error <= thresholds.maximum_numerical_check_error
    )
    semantics = all(model_semantics_pass(s) for s in target_solutions)

    self_overlap_errors: List[float] = []
    reverse_errors: List[float] = []
    polar_errors: List[float] = []
    for r, sol in direct.items():
        self_m = active_overlap_matrix(sol, sol, mols[r], mols[r], terms_cache)
        self_overlap_errors.append(float(np.max(np.abs(self_m - np.eye(NROOTS)))))
    for (r0, r1), obj in zip(
        zip(protocol.r_grid_angstrom[:-1], protocol.r_grid_angstrom[1:]), edge_objects
    ):
        reverse = active_overlap_matrix(direct[r1], direct[r0], mols[r1], mols[r0], terms_cache)
        reverse_errors.append(float(np.max(np.abs(reverse - obj["matrix"].conj().T))))
        polar_errors.append(float(np.max(np.abs(obj["Q"].conj().T @ obj["Q"] - np.eye(NROOTS)))))
    probe_errors = [
        float(s["metadata"]["probe_retained_energy_max_error_Eh"]) for s in target_solutions
    ]
    numerical_error = max(self_overlap_errors + reverse_errors + polar_errors, default=0.0)
    numerical_validation = (
        numerical_error <= thresholds.maximum_numerical_check_error
        and max(probe_errors, default=0.0) <= thresholds.probe_retained_energy_maximum_error_Eh
    )

    state_edges_strict = bool(edge_records) and all(
        e["state_subspace"]["sigma_min"] >= thresholds.eta_edge_minimum
        and e["state_subspace"]["leakage"] <= thresholds.maximum_edge_leakage
        for e in edge_records
    )
    assignments_clear = bool(edge_records) and all(
        e["state_subspace"]["normalized_margin"]
        >= thresholds.minimum_normalized_assignment_margin
        for e in edge_records
    )
    active_orbitals_continuous = bool(edge_records) and all(
        e["active_orbital_subspace"]["sigma_min"]
        >= thresholds.active_orbital_subspace_sigma_minimum
        and e["active_orbital_subspace"]["leakage"]
        <= thresholds.active_orbital_subspace_leakage_maximum
        for e in edge_records
    )
    history_same = all(
        h["average_energy_difference_Eh"]
        <= thresholds.history_control_maximum_average_energy_difference_Eh
        and h["state_subspace"]["sigma_min"]
        >= thresholds.history_control_minimum_state_subspace_sigma
        and h["state_subspace"]["leakage"]
        <= thresholds.history_control_maximum_state_subspace_leakage
        for h in history_records
    )

    gaps = [
        {
            "R_angstrom": r,
            "internal_minimum_gap_Eh": float(np.min(np.diff(direct[r]["state_energies_Eh"]))),
            "excluded_root_gap_Eh": float(direct[r]["metadata"]["excluded_root_gap_Eh"]),
            "state_energies_Eh": direct[r]["state_energies_Eh"],
            "five_root_probe_energies_Eh": direct[r]["five_root_probe_energies_Eh"],
        }
        for r in protocol.r_grid_angstrom
    ]
    minimum_excluded_gap = min(gaps, key=lambda x: x["excluded_root_gap_Eh"])
    minimum_internal_gap = min(gaps, key=lambda x: x["internal_minimum_gap_Eh"])
    external_isolated = (
        minimum_excluded_gap["excluded_root_gap_Eh"] >= thresholds.excluded_root_minimum_gap_Eh
    )

    dipole_records: List[Dict[str, Any]] = []
    heuristic_records: List[Dict[str, Any]] = []
    for r in protocol.r_grid_angstrom:
        energies = direct[r]["state_energies_Eh"]
        dip_z = np.real(np.diag(direct[r]["dipole_au"][2]))
        dipole_records.append(
            {"R_angstrom": r, "state_dipole_z_au": dip_z, "transition_dipole_matrix_au": direct[r]["dipole_au"]}
        )
        for i in range(NROOTS):
            for j in range(i + 1, NROOTS):
                gap = abs(float(energies[j] - energies[i]))
                contrast = abs(float(dip_z[j] - dip_z[i]))
                heuristic_records.append(
                    {
                        "R_angstrom": r,
                        "state_pair": [i, j],
                        "gap_Eh": gap,
                        "diagonal_dipole_contrast_au": contrast,
                        "gap_over_dipole_field_au": gap / max(contrast, 1.0e-12),
                    }
                )
    field_heuristic = min(heuristic_records, key=lambda x: x["gap_over_dipole_field_au"])

    if not numerical_validation:
        status = "STOP_NUMERICAL_VALIDATION_FAILURE"
    elif not semantics:
        status = "STOP_MODEL_SEMANTICS_FAILURE"
    elif not stationary:
        status = "STOP_WAVEFUNCTION_FAILURE"
    elif not history_same:
        status = "STOP_HISTORY_CONTROL_FAILURE"
    elif not external_isolated:
        status = "STOP_EXTERNAL_ROOT_ISOLATION_FAILURE"
    elif not active_orbitals_continuous:
        status = "STOP_ACTIVE_ORBITAL_CONTINUITY_FAILURE"
    elif not state_edges_strict:
        status = "STOP_EDGE_FIDELITY_FAILURE"
    elif not assignments_clear:
        status = "STOP_ASSIGNMENT_FAILURE"
    else:
        status = "PROMOTE_TO_PHYSICAL_FIELD_PILOT"

    failed_edges = [
        e
        for e in edge_records
        if e["state_subspace"]["sigma_min"] < thresholds.eta_edge_minimum
        or e["state_subspace"]["leakage"] > thresholds.maximum_edge_leakage
        or e["state_subspace"]["normalized_margin"]
        < thresholds.minimum_normalized_assignment_margin
        or e["active_orbital_subspace"]["sigma_min"]
        < thresholds.active_orbital_subspace_sigma_minimum
        or e["active_orbital_subspace"]["leakage"]
        > thresholds.active_orbital_subspace_leakage_maximum
    ]
    failed_history = [
        h
        for h in history_records
        if h["average_energy_difference_Eh"]
        > thresholds.history_control_maximum_average_energy_difference_Eh
        or h["state_subspace"]["sigma_min"]
        < thresholds.history_control_minimum_state_subspace_sigma
        or h["state_subspace"]["leakage"]
        > thresholds.history_control_maximum_state_subspace_leakage
    ]
    nonstationary = [
        {
            "path": s["path"],
            "history": s["metadata"]["history"],
            "R_angstrom": s["metadata"]["R_angstrom"],
            "converged": s["metadata"]["converged"],
            "maximum_orbital_gradient": s["metadata"]["maximum_orbital_gradient"],
            "state_S2": s["metadata"]["state_S2"],
            "mo_orthonormality_max_error": s["metadata"]["mo_orthonormality_max_error"],
        }
        for s in target_solutions
        if not bool(s["metadata"]["converged"])
        or not np.isfinite(float(s["metadata"]["maximum_orbital_gradient"]))
        or float(s["metadata"]["maximum_orbital_gradient"])
        > thresholds.maximum_final_orbital_gradient
        or max(abs(float(x)) for x in s["metadata"]["state_S2"])
        > thresholds.maximum_spin_square
        or float(s["metadata"]["mo_orthonormality_max_error"])
        > thresholds.maximum_numerical_check_error
    ]

    decision = {
        "gate": "LiH SA4/CAS(2e,5o) S0 one-dimensional qualification",
        "status": status,
        "manuscript_claim_authorized": False,
        "script_version": SCRIPT_VERSION,
        "model": {
            "molecule": "LiH",
            "basis": protocol.basis,
            "point_group": "C2v",
            "state_symmetry": WFNSYM,
            "active_space": [NELECAS, NCAS],
            "active_irrep_counts": ACTIVE_IRREP_COUNTS,
            "ncore": 1,
            "frozen_orbitals": 0,
            "optimized_and_retained_roots": NROOTS,
            "state_average_weights": [1.0 / NROOTS] * NROOTS,
            "excluded_root_probe": "five-root A1 CASCI on each optimized SA4 orbital set",
            "edge_metric": "exact active-only CAS-wavefunction overlap",
        },
        "pre_registered_grid": {
            "R_angstrom": protocol.r_grid_angstrom,
            "short_anchor_angstrom": protocol.short_anchor_angstrom,
            "long_anchor_angstrom": protocol.long_anchor_angstrom,
            "history_step_angstrom": protocol.history_step_angstrom,
            "adaptive_window_selection": False,
        },
        "thresholds": asdict(thresholds),
        "decision_checks": {
            "numerical_validation": numerical_validation,
            "model_semantics_exact": semantics,
            "all_wavefunctions_stationary": stationary,
            "direct_vs_two_opposed_histories_same_solution": history_same,
            "first_excluded_A1_root_isolated": external_isolated,
            "active_orbital_subspace_continuous": active_orbitals_continuous,
            "all_elementary_active_state_edges_strict": state_edges_strict,
            "all_elementary_assignments_unambiguous": assignments_clear,
        },
        "counts": {
            "direct_wavefunctions": len(direct),
            "short_history_wavefunctions": len(short),
            "long_history_wavefunctions": len(long),
            "analysis_grid_history_controls": len(history_records),
            "elementary_edges": len(edge_records),
        },
        "maximum_final_orbital_gradient": maximum_gradient,
        "maximum_state_S2": maximum_s2,
        "maximum_mo_orthonormality_error": maximum_mo_error,
        "minimum_internal_state_gap": minimum_internal_gap,
        "minimum_excluded_root_gap": minimum_excluded_gap,
        "field_scale_heuristic_not_a_grid_authorization": field_heuristic,
        "numerical_checks": {
            "maximum_self_overlap_error": max(self_overlap_errors, default=0.0),
            "maximum_reverse_overlap_error": max(reverse_errors, default=0.0),
            "maximum_polar_unitarity_error": max(polar_errors, default=0.0),
            "maximum_probe_retained_energy_error_Eh": max(probe_errors, default=0.0),
            "maximum_overlap_algebra_error": numerical_error,
        },
        "failed_edges": failed_edges,
        "failed_history_controls": failed_history,
        "nonstationary_wavefunctions": nonstationary,
        "authorized_next_step": (
            "Design and pre-register a small R x static-electric-field pilot around the fixed avoided-crossing window; "
            "retain the same four A1 roots and active-only overlap definition."
            if status == "PROMOTE_TO_PHYSICAL_FIELD_PILOT"
            else "Stop LiH before any two-dimensional field calculation; diagnose only the first failed gate without relaxing thresholds."
        ),
        "authorized_interpretation": (
            "LiH has passed the one-dimensional model and wavefunction qualification required for a physical-field pilot."
            if status == "PROMOTE_TO_PHYSICAL_FIELD_PILOT"
            else "No molecular local-pass/global-fail or physical-field claim is authorized."
        ),
        "prohibited_interpretations": [
            "publication-ready molecular false negative from S0 alone",
            "conical intersection proved by an avoided crossing",
            "post hoc removal of a difficult root or geometry",
            "replacement of active-only overlap by scalar core division",
            "threshold relaxation after seeing the result",
        ],
    }
    write_json(root / "LiH_S0_decision.json", decision)
    write_json(
        root / "LiH_S0_diagnostics.json",
        {
            "edges": edge_records,
            "history_controls": history_records,
            "gaps": gaps,
            "dipoles": dipole_records,
            "field_scale_heuristics": heuristic_records,
        },
    )

    print("\nLiH S0 RESULT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"Worst active-state edge: eta={min(e['state_subspace']['sigma_min'] for e in edge_records):.12f}; "
        f"leakage={max(e['state_subspace']['leakage'] for e in edge_records):.12f}; "
        f"margin={min(e['state_subspace']['normalized_margin'] for e in edge_records):.12f}",
        flush=True,
    )
    print(
        f"Minimum excluded-root gap={minimum_excluded_gap['excluded_root_gap_Eh']:.12e} Eh; "
        f"maximum gradient={maximum_gradient:.3e}",
        flush=True,
    )
    print(f"Decision file: {root / 'LiH_S0_decision.json'}", flush=True)
    return decision


def selftest() -> None:
    rng = np.random.default_rng(20260916)
    near = np.eye(NROOTS) + 0.02 * rng.normal(size=(NROOTS, NROOTS))
    q, singular_values = polar_unitary(near)
    assert np.max(np.abs(q.conj().T @ q - np.eye(NROOTS))) < 1.0e-12
    assert np.all(singular_values > 0.0)
    assert assignment(np.eye(NROOTS))["permutation"] == list(range(NROOTS))
    exchanged = np.eye(NROOTS)[[1, 0, 2, 3]]
    assert assignment(exchanged)["permutation"] == [1, 0, 2, 3]
    values = parse_grid("2.3:4.2:0.05")
    assert len(values) == 39 and values[0] == 2.3 and values[-1] == 4.2
    assert inclusive_path(6.0, 2.3, 0.05)[-1] == 2.3
    assert _jsonable(1.0 + 2.0j) == {"real": 1.0, "imag": 2.0}
    print("SELFTEST=PASS")


def build_protocol(args: argparse.Namespace) -> Protocol:
    r_grid = parse_grid(args.r_grid)
    if len(r_grid) < 2 and args.command in ("analyze", "all"):
        raise ValueError("analysis requires at least two R points")
    if tuple(sorted(r_grid)) != r_grid or len(set(r_grid)) != len(r_grid):
        raise ValueError("R grid must be strictly increasing")
    if args.short_anchor > r_grid[0] or args.long_anchor < r_grid[-1]:
        raise ValueError("history anchors must bracket the analysis grid")
    inclusive_path(args.short_anchor, r_grid[-1], args.history_step)
    inclusive_path(args.long_anchor, r_grid[0], args.history_step)
    return Protocol(
        basis=args.basis,
        r_grid_angstrom=r_grid,
        short_anchor_angstrom=args.short_anchor,
        long_anchor_angstrom=args.long_anchor,
        history_step_angstrom=args.history_step,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run NumPy-only algebra and protocol tests")
    for name in ("run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument(
            "--root", type=Path, default=Path.cwd() / "work" / "LiH_S0_SA4_work"
        )
        p.add_argument("--basis", default="aug-cc-pvdz")
        p.add_argument("--r-grid", default="2.3:4.2:0.05", help="comma list or start:stop:step in Angstrom")
        p.add_argument("--short-anchor", type=float, default=1.5)
        p.add_argument("--long-anchor", type=float, default=6.0)
        p.add_argument("--history-step", type=float, default=0.05)
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    require_pyscf()
    import pyscf

    pyscf.lib.num_threads(args.threads)
    protocol = build_protocol(args)
    if args.command in ("run", "all"):
        run_protocol(args.root, protocol, args.overwrite)
    if args.command in ("analyze", "all"):
        analyze_protocol(args.root, protocol, Thresholds())


if __name__ == "__main__":
    main()
