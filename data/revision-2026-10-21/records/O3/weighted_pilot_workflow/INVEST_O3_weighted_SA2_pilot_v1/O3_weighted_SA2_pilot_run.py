#!/usr/bin/env python3
"""Run the fixed-root weighted-SA2 O3 pilot.

The protocol coordinate is the excited-root weight p in weights (1-p, p).
Only 0 < p <= 0.5 is allowed, so the ensemble weights remain ordered and the
pilot cannot confuse a weight-inverted SA instability with protocol curvature.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback


SCHEMA_VERSION = 1
STAGE = "O3_weighted_SA2_pilot"


@dataclass(frozen=True)
class Point:
    tag: str
    direction: str
    sign: int
    delta_r: float
    theta: float

    def bonds(self, r_sum: float) -> tuple[float, float]:
        return 0.5 * (r_sum + self.delta_r), 0.5 * (r_sum - self.delta_r)


def build_points(args: argparse.Namespace) -> list[Point]:
    return [
        Point("center", "center", 0, 0.0, args.theta0),
        Point("dr_minus_h_0p010000", "delta_r", -1, -args.delta_r, args.theta0),
        Point("dr_plus_h_0p010000", "delta_r", +1, +args.delta_r, args.theta0),
        Point("theta_minus_h_0p200000", "theta", -1, 0.0, args.theta0 - args.delta_theta),
        Point("theta_plus_h_0p200000", "theta", +1, 0.0, args.theta0 + args.delta_theta),
    ]


def parse_weights(text: str) -> list[float]:
    values = sorted({float(token.strip()) for token in text.split(",") if token.strip()})
    if len(values) < 3:
        raise ValueError("At least three distinct excited-root weights are required.")
    if any(not (0.0 < value <= 0.5) for value in values):
        raise ValueError("Every excited-root weight must satisfy 0 < p <= 0.5.")
    if abs(values[-1] - 0.5) > 1.0e-12:
        raise ValueError("The largest weight must be p=0.5 to reuse the audited Gate4B SA2 seed.")
    return values


def weight_tag(value: float) -> str:
    return f"p{value:.6f}".replace(".", "p")


def atoms_from_values(r1: float, r2: float, theta: float):
    half = math.radians(theta) / 2.0
    s, c = math.sin(half), math.cos(half)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("O", (r1 * s, r1 * c, 0.0)),
        ("O", (-r2 * s, r2 * c, 0.0)),
    ]


def configure_threads(nthreads: int) -> None:
    value = str(nthreads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value


def import_pyscf():
    try:
        import numpy as np
        from pyscf import fci, gto, lib, mcscf, scf, symm
    except ImportError as exc:
        raise RuntimeError("Activate the production PySCF environment before running the pilot.") from exc
    return np, fci, gto, lib, mcscf, scf, symm


def odd_irrep(mol) -> str:
    for candidate in ('A"', "A''", "A2"):
        if candidate in mol.irrep_name:
            return candidate
    raise RuntimeError(f"No reflection-odd Cs irrep found in {mol.irrep_name}.")


def build_mol_values(r1, r2, theta, args, gto, log_path=None):
    mol = gto.Mole()
    mol.atom = atoms_from_values(r1, r2, theta)
    mol.unit = "Angstrom"
    mol.basis = args.basis
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = "Cs"
    mol.max_memory = args.memory_mb
    mol.verbose = args.verbose if log_path is not None else 0
    if log_path is not None:
        mol.output = str(log_path)
    mol.build()
    if mol.groupname != "Cs":
        raise RuntimeError(f"Forced Cs symmetry, but PySCF reports {mol.groupname}.")
    odd_irrep(mol)
    return mol


def build_mol(point: Point, args, gto, log_path=None):
    r1, r2 = point.bonds(args.r_sum)
    return build_mol_values(r1, r2, point.theta, args, gto, log_path)


def close_log(mol) -> None:
    stream = getattr(mol, "stdout", None)
    if stream not in (None, sys.stdout, sys.stderr):
        try:
            stream.flush()
            stream.close()
        except Exception:
            pass


def run_rhf(mol, args, scf, chkfile: Path):
    mf = scf.RHF(mol)
    mf.chkfile = str(chkfile)
    mf.conv_tol = args.rhf_conv_tol
    mf.max_cycle = args.rhf_max_cycle
    mf.kernel()
    if not mf.converged and args.newton_fallback:
        mol.stdout.write("\nRHF Newton fallback\n")
        mol.stdout.flush()
        mf = mf.newton()
        mf.conv_tol = args.rhf_conv_tol
        mf.max_cycle = args.rhf_max_cycle
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")
    return mf


def symmetric_orthonormalize(mo, overlap, np):
    metric = mo.T.conj() @ overlap @ mo
    values, vectors = np.linalg.eigh(metric)
    if float(values.min()) < 1.0e-10:
        raise RuntimeError(f"Projected MO metric is singular ({values.min():.3e}).")
    return mo @ vectors @ np.diag(values ** -0.5) @ vectors.T.conj()


def project_orbitals(seed_mol, seed_mo, mol, np, scf, symm):
    projected = scf.addons.project_mo_nr2nr(seed_mol, seed_mo, mol)
    overlap = mol.intor_symmetric("int1e_ovlp")
    projected = symmetric_orthonormalize(projected, overlap, np)
    return symm.symmetrize_space(mol, projected, s=overlap, check=True)


def ncore_int(ncore) -> int:
    if isinstance(ncore, (tuple, list)):
        if len(ncore) != 2 or int(ncore[0]) != int(ncore[1]):
            raise RuntimeError(f"Restricted ncore required, got {ncore}.")
        return int(ncore[0])
    return int(ncore)


def ci_list(ci) -> list:
    return list(ci) if isinstance(ci, (list, tuple)) else [ci]


def state_energies(mc, np) -> list[float]:
    values = getattr(mc, "e_states", None)
    if values is None:
        candidate = np.asarray(mc.e_tot)
        if candidate.ndim == 1:
            values = candidate
    result = [] if values is None else [float(value) for value in np.asarray(values).reshape(-1)]
    if len(result) != 2:
        raise RuntimeError(f"Expected two state energies, received {result}.")
    return result


def one_step_tail_metrics(mol, window: int):
    stream = getattr(mol, "stdout", None)
    if stream is not None:
        stream.flush()
    output = getattr(mol, "output", None)
    if not output or not Path(output).exists():
        return None
    text = Path(output).read_text(encoding="utf-8", errors="replace")
    number = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?"
    pattern = re.compile(
        rf"macro iter\s+(\d+)[^\n]*?dE\s*=\s*({number})[^\n]*\n"
        rf"\s*\|grad\[o\]\|=\s*({number})\s+\|ddm\|=\s*({number})\s+"
        rf"\|maxRot\[o\]\|=\s*({number})"
    )
    records = [
        {"macro": int(m.group(1)), "dE": float(m.group(2)), "grad": float(m.group(3)),
         "ddm": float(m.group(4)), "maxrot": float(m.group(5))}
        for m in pattern.finditer(text)
    ]
    if not records:
        return None
    tail = records[-window:]
    return {
        "tail_count": len(tail),
        "final_macro": records[-1]["macro"],
        "final_grad": records[-1]["grad"],
        "final_ddm": records[-1]["ddm"],
        "final_maxrot": records[-1]["maxrot"],
        "max_abs_dE": max(abs(row["dE"]) for row in tail),
        "max_grad": max(row["grad"] for row in tail),
        "max_ddm": max(row["ddm"] for row in tail),
        "max_maxrot": max(row["maxrot"] for row in tail),
    }


def run_weighted_sa2(mol, mf, mo_guess, excited_weight, args, fci, mcscf, np):
    mc = mcscf.CASSCF(mf, args.ncas, args.nelecas)
    mc.max_memory = args.memory_mb
    mc.conv_tol = args.mc_conv_tol
    mc.conv_tol_grad = args.mc_conv_tol_grad
    mc.max_cycle_macro = args.mc_max_cycle_macro
    mc.max_cycle_micro = args.mc_max_cycle_micro
    mc.canonicalization = True
    solver = fci.direct_spin0_symm.FCI(mol)
    solver.wfnsym = odd_irrep(mol)
    solver.nroots = 2
    solver.conv_tol = args.fci_conv_tol
    solver.max_cycle = args.fci_max_cycle
    mc.fcisolver = solver
    weights = [1.0 - excited_weight, excited_weight]
    mc.state_average_(weights)
    mc.kernel(mo_guess)
    metrics = one_step_tail_metrics(mol, args.mc_stationary_window)
    convergence_mode = "pyscf"
    if not mc.converged:
        stationary = (
            metrics is not None
            and metrics["tail_count"] == args.mc_stationary_window
            and metrics["max_abs_dE"] <= args.mc_stationary_energy_tol
            and metrics["max_grad"] <= args.mc_stationary_grad_ceiling
            and metrics["max_ddm"] <= args.mc_stationary_ddm_tol
            and metrics["max_maxrot"] <= args.mc_stationary_maxrot_tol
        )
        if not stationary:
            raise RuntimeError(f"Weighted-SA2 failed PySCF and explicit stationarity gates: {metrics}")
        convergence_mode = "stationary_floor"
        mc.converged = True
        mol.stdout.write(
            "\nWEIGHTED-SA2 STATIONARY-FLOOR ACCEPTANCE\n"
            f"p={excited_weight:.6f}, window={args.mc_stationary_window}, "
            f"max|dE|={metrics['max_abs_dE']:.3e}, max|grad[o]|={metrics['max_grad']:.3e}, "
            f"max|ddm|={metrics['max_ddm']:.3e}, max|maxRot[o]|={metrics['max_maxrot']:.3e}\n"
        )
        mol.stdout.flush()
    vectors = ci_list(mc.ci)
    if len(vectors) != 2:
        raise RuntimeError(f"Expected two CI vectors, got {len(vectors)}.")
    energies = state_energies(mc, np)
    occupations = []
    for vector in vectors:
        rdm1 = solver.make_rdm1(vector, mc.ncas, mc.nelecas)
        occupations.append(np.linalg.eigvalsh(rdm1)[::-1])
    mc.pilot_convergence_mode = convergence_mode
    mc.pilot_stationarity_metrics = metrics or {}
    return mc, energies, vectors, occupations


def electronic_dipole_matrix(mol, mo, vectors, ncore, ncas, nelecas, solver, fci, np):
    """Electronic dipole operator matrix in the two-state adiabatic basis."""
    ao_r = mol.intor_symmetric("int1e_r", comp=3)
    core = mo[:, :ncore]
    active = mo[:, ncore:ncore + ncas]
    active_r = np.einsum("pi,xpq,qj->xij", active.conj(), ao_r, active, optimize=True)
    core_r = (
        2.0 * np.einsum("pi,xpq,qi->x", core.conj(), ao_r, core, optimize=True)
        if ncore else np.zeros(3)
    )
    trans_rdm1 = getattr(solver, "trans_rdm1", None)
    if trans_rdm1 is None:
        trans_rdm1 = fci.direct_spin1.trans_rdm1
    matrix = np.zeros((3, 2, 2), dtype=complex)
    for i, bra in enumerate(vectors):
        for j, ket in enumerate(vectors):
            dm1 = trans_rdm1(bra, ket, ncas, nelecas)
            value = np.einsum("pq,xqp->x", dm1, active_r, optimize=True)
            if i == j:
                value = value + core_r
            matrix[:, i, j] = -value
    return 0.5 * (matrix + matrix.transpose(0, 2, 1).conj())


def load_npz_seed(path: Path, point: Point, excited_weight: float, args, gto, np):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if str(data["basis"].item()).lower() != args.basis.lower():
            raise RuntimeError(f"Basis mismatch in seed {path}.")
        if (int(data["ncas"].item()), int(data["nelecas"].item())) != (args.ncas, args.nelecas):
            raise RuntimeError(f"Active-space mismatch in seed {path}.")
        r1, r2 = point.bonds(args.r_sum)
        if max(abs(float(data["r1"].item()) - r1), abs(float(data["r2"].item()) - r2),
               abs(float(data["theta"].item()) - point.theta)) > 1.0e-9:
            raise RuntimeError(f"Geometry mismatch in seed {path}.")
        if "excited_weight" in data and abs(float(data["excited_weight"].item()) - excited_weight) > 1.0e-12:
            raise RuntimeError(f"Weight mismatch in seed {path}.")
        mol = build_mol_values(r1, r2, point.theta, args, gto)
        return {
            "mol": mol,
            "mo": data["mo_coeff"].copy(),
            "ci": [row.copy() for row in data["ci_vectors"]],
            "energies": data["state_energies"].copy(),
            "ncore": int(data["ncore"].item()),
            "convergence_mode": str(data["convergence_mode"].item()),
            "final_grad": float(data["final_orbital_gradient"].item()),
            "final_ddm": float(data["final_density_change"].item()),
            "final_maxrot": float(data["final_max_orbital_rotation"].item()),
            "natural_occupations": data["natural_occupations"].copy(),
        }


def checkpoint_path(workdir: Path, point: Point, excited_weight: float) -> Path:
    return workdir / "wavefunctions" / f"{point.tag}_{weight_tag(excited_weight)}.npz"


def checkpoint_valid(path: Path, point: Point, excited_weight: float, args, np,
                     calculation_role="primary") -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            r1, r2 = point.bonds(args.r_sum)
            return (
                int(data["schema_version"].item()) == SCHEMA_VERSION
                and str(data["stage"].item()) == STAGE
                and str(data["tag"].item()) == point.tag
                and str(data["calculation_role"].item()) == calculation_role
                and abs(float(data["excited_weight"].item()) - excited_weight) < 1.0e-12
                and str(data["basis"].item()).lower() == args.basis.lower()
                and int(data["ncas"].item()) == args.ncas
                and int(data["nelecas"].item()) == args.nelecas
                and abs(float(data["r1"].item()) - r1) < 1.0e-10
                and abs(float(data["r2"].item()) - r2) < 1.0e-10
                and abs(float(data["theta"].item()) - point.theta) < 1.0e-10
                and tuple(data["dipole_electronic_au"].shape) == (3, 2, 2)
            )
    except Exception:
        return False


def save_checkpoint(path, point, excited_weight, mol, mo, vectors, energies, occupations,
                    dipole, ncore, convergence_mode, metrics, source, reused,
                    calculation_role, args, np):
    temporary = path.with_name(path.name + ".tmp.npz")
    r1, r2 = point.bonds(args.r_sum)
    weights = np.asarray([1.0 - excited_weight, excited_weight])
    np.savez_compressed(
        temporary,
        schema_version=SCHEMA_VERSION,
        stage=STAGE,
        tag=point.tag,
        direction=point.direction,
        sign=point.sign,
        delta_r=point.delta_r,
        theta=point.theta,
        r1=r1,
        r2=r2,
        r_sum=args.r_sum,
        basis=args.basis,
        point_group=mol.groupname,
        ncas=args.ncas,
        nelecas=args.nelecas,
        ncore=ncore,
        nroots=2,
        calculation_role=calculation_role,
        excited_weight=excited_weight,
        state_weights=weights,
        mo_coeff=np.asarray(mo),
        ci_vectors=np.stack([np.asarray(vector) for vector in vectors]),
        state_energies=np.asarray(energies),
        weighted_energy=float(np.dot(weights, energies)),
        natural_occupations=np.asarray(occupations),
        dipole_electronic_au=np.asarray(dipole),
        convergence_mode=convergence_mode,
        initial_orbital_seed=source,
        reused_gate4b_seed=bool(reused),
        final_orbital_gradient=float(metrics.get("final_grad", np.nan)),
        final_density_change=float(metrics.get("final_ddm", np.nan)),
        final_max_orbital_rotation=float(metrics.get("final_maxrot", np.nan)),
        rhf_conv_tol=args.rhf_conv_tol,
        fci_conv_tol=args.fci_conv_tol,
        mc_conv_tol=args.mc_conv_tol,
        mc_conv_tol_grad=args.mc_conv_tol_grad,
        mc_stationary_grad_ceiling=args.mc_stationary_grad_ceiling,
    )
    os.replace(temporary, path)


CSV_FIELDS = [
    "tag", "calculation_role", "direction", "sign", "delta_r_A", "theta_deg", "excited_weight",
    "ground_weight", "E_weighted_Eh", "E0_Eh", "E1_Eh", "gap_Eh", "source",
    "reused_gate4b_seed", "convergence_mode", "final_orbital_gradient",
    "wall_time_s", "status", "error",
]


def append_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def rewrap_gate4b_seed(point, args, gto, fci, np):
    source = args.gate4b_workdir / "wavefunctions" / f"{point.tag}_SA2.npz"
    data = load_npz_seed(source, point, 0.5, args, gto, np)
    solver = fci.direct_spin0_symm.FCI(data["mol"])
    solver.wfnsym = odd_irrep(data["mol"])
    solver.nroots = 2
    dipole = electronic_dipole_matrix(
        data["mol"], data["mo"], data["ci"], data["ncore"], args.ncas,
        args.nelecas, solver, fci, np
    )
    metrics = {"final_grad": data["final_grad"], "final_ddm": data["final_ddm"],
               "final_maxrot": data["final_maxrot"]}
    return data, dipole, metrics, str(source)


def run(args: argparse.Namespace) -> None:
    np, fci, gto, lib, mcscf, scf, symm = import_pyscf()
    lib.num_threads(args.threads)
    for directory in (args.workdir, args.workdir / "logs", args.workdir / "rhf_checkpoints",
                      args.workdir / "wavefunctions", args.workdir / "controls"):
        directory.mkdir(parents=True, exist_ok=True)
    points = build_points(args)
    table = args.workdir / "O3_weighted_SA2_calculations.csv"
    print(f"O3 fixed-root weighted-SA2 pilot: {len(points)} geometries x {len(args.excited_weights)} weights", flush=True)
    print(f"Ordered weights (ground, excited): {[(1-p, p) for p in args.excited_weights]}", flush=True)

    for point in points:
        previous = None
        previous_p = None
        for p in sorted(args.excited_weights, reverse=True):
            destination = checkpoint_path(args.workdir, point, p)
            if args.resume and checkpoint_valid(destination, point, p, args, np):
                print(f"SKIP {point.tag:28s} p={p:.3f}", flush=True)
                previous = load_npz_seed(destination, point, p, args, gto, np)
                previous_p = p
                continue
            row = {
                "tag": point.tag, "calculation_role": "primary", "direction": point.direction, "sign": point.sign,
                "delta_r_A": f"{point.delta_r:.10f}", "theta_deg": f"{point.theta:.10f}",
                "excited_weight": f"{p:.8f}", "ground_weight": f"{1-p:.8f}",
            }
            mol = None
            started = time.perf_counter()
            try:
                if abs(p - 0.5) < 1.0e-12:
                    data, dipole, metrics, source = rewrap_gate4b_seed(point, args, gto, fci, np)
                    mol = data["mol"]
                    save_checkpoint(
                        destination, point, p, mol, data["mo"], data["ci"], data["energies"],
                        data["natural_occupations"], dipole, data["ncore"], data["convergence_mode"],
                        metrics, source, True, "primary", args, np,
                    )
                    energies = data["energies"]
                    convergence_mode = data["convergence_mode"]
                    final_grad = data["final_grad"]
                    previous = load_npz_seed(destination, point, p, args, gto, np)
                    previous_p = p
                    reused = True
                else:
                    if previous is None:
                        raise RuntimeError("Weight continuation seed is unavailable.")
                    log_path = args.workdir / "logs" / f"{point.tag}_{weight_tag(p)}.log"
                    mol = build_mol(point, args, gto, log_path)
                    mf = run_rhf(mol, args, scf, args.workdir / "rhf_checkpoints" / f"{point.tag}_{weight_tag(p)}.chk")
                    mo_guess = project_orbitals(previous["mol"], previous["mo"], mol, np, scf, symm)
                    source = f"same-geometry continuation from p={previous_p:.6f}"
                    mol.stdout.write(f"\nInitial orbitals: {source}\nWeights: {1-p:.8f} {p:.8f}\n")
                    mol.stdout.flush()
                    mc, energies, vectors, occupations = run_weighted_sa2(
                        mol, mf, mo_guess, p, args, fci, mcscf, np
                    )
                    metrics = mc.pilot_stationarity_metrics
                    dipole = electronic_dipole_matrix(
                        mol, mc.mo_coeff, vectors, ncore_int(mc.ncore), args.ncas,
                        args.nelecas, mc.fcisolver, fci, np
                    )
                    save_checkpoint(
                        destination, point, p, mol, mc.mo_coeff, vectors, energies, occupations,
                        dipole, ncore_int(mc.ncore), mc.pilot_convergence_mode, metrics,
                        source, False, "primary", args, np,
                    )
                    convergence_mode = mc.pilot_convergence_mode
                    final_grad = float(metrics.get("final_grad", np.nan))
                    previous = load_npz_seed(destination, point, p, args, gto, np)
                    previous_p = p
                    reused = False
                weighted_energy = (1-p) * float(energies[0]) + p * float(energies[1])
                row.update({
                    "E_weighted_Eh": f"{weighted_energy:.14f}",
                    "E0_Eh": f"{float(energies[0]):.14f}", "E1_Eh": f"{float(energies[1]):.14f}",
                    "gap_Eh": f"{float(energies[1]-energies[0]):.12e}", "source": source,
                    "reused_gate4b_seed": reused, "convergence_mode": convergence_mode,
                    "final_orbital_gradient": f"{final_grad:.12e}", "status": "ok", "error": "",
                })
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                with (args.workdir / "failures.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{point.tag} p={p:.6f}]\n")
                    traceback.print_exc(file=handle)
                row["wall_time_s"] = f"{time.perf_counter() - started:.3f}"
                append_row(table, row)
                if mol is not None:
                    close_log(mol)
                raise
            row["wall_time_s"] = f"{time.perf_counter() - started:.3f}"
            append_row(table, row)
            if mol is not None:
                close_log(mol)

        if args.history_controls:
            p = args.excited_weights[0]
            control = args.workdir / "controls" / f"{point.tag}_{weight_tag(p)}_direct_from_equal.npz"
            if args.resume and checkpoint_valid(control, point, p, args, np, "direct_history_control"):
                print(f"SKIP {point.tag:28s} p={p:.3f} direct history control", flush=True)
                continue
            baseline = load_npz_seed(
                checkpoint_path(args.workdir, point, 0.5), point, 0.5, args, gto, np
            )
            row = {
                "tag": point.tag, "calculation_role": "direct_history_control",
                "direction": point.direction, "sign": point.sign,
                "delta_r_A": f"{point.delta_r:.10f}", "theta_deg": f"{point.theta:.10f}",
                "excited_weight": f"{p:.8f}", "ground_weight": f"{1-p:.8f}",
            }
            mol = None
            started = time.perf_counter()
            try:
                log_path = args.workdir / "logs" / f"{point.tag}_{weight_tag(p)}_direct_history_control.log"
                mol = build_mol(point, args, gto, log_path)
                mf = run_rhf(
                    mol, args, scf,
                    args.workdir / "rhf_checkpoints" / f"{point.tag}_{weight_tag(p)}_direct_history_control.chk",
                )
                mo_guess = project_orbitals(baseline["mol"], baseline["mo"], mol, np, scf, symm)
                source = "direct history control from audited p=0.500000 orbitals"
                mol.stdout.write(f"\nInitial orbitals: {source}\nWeights: {1-p:.8f} {p:.8f}\n")
                mol.stdout.flush()
                mc, energies, vectors, occupations = run_weighted_sa2(
                    mol, mf, mo_guess, p, args, fci, mcscf, np
                )
                metrics = mc.pilot_stationarity_metrics
                dipole = electronic_dipole_matrix(
                    mol, mc.mo_coeff, vectors, ncore_int(mc.ncore), args.ncas,
                    args.nelecas, mc.fcisolver, fci, np
                )
                save_checkpoint(
                    control, point, p, mol, mc.mo_coeff, vectors, energies, occupations,
                    dipole, ncore_int(mc.ncore), mc.pilot_convergence_mode, metrics,
                    source, False, "direct_history_control", args, np,
                )
                weighted_energy = (1-p) * float(energies[0]) + p * float(energies[1])
                row.update({
                    "E_weighted_Eh": f"{weighted_energy:.14f}",
                    "E0_Eh": f"{float(energies[0]):.14f}", "E1_Eh": f"{float(energies[1]):.14f}",
                    "gap_Eh": f"{float(energies[1]-energies[0]):.12e}", "source": source,
                    "reused_gate4b_seed": False, "convergence_mode": mc.pilot_convergence_mode,
                    "final_orbital_gradient": f"{float(metrics.get('final_grad', np.nan)):.12e}",
                    "status": "ok", "error": "",
                })
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                with (args.workdir / "failures.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{point.tag} p={p:.6f} direct_history_control]\n")
                    traceback.print_exc(file=handle)
                row["wall_time_s"] = f"{time.perf_counter() - started:.3f}"
                append_row(table, row)
                if mol is not None:
                    close_log(mol)
                raise
            row["wall_time_s"] = f"{time.perf_counter() - started:.3f}"
            append_row(table, row)
            if mol is not None:
                close_log(mol)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "scientific_question": "Can pairwise maximum-overlap tracking return one unambiguous assignment while weighted-SA2 transport has nontrivial path dependence?",
        "model": {
            "molecule": "O3", "forced_symmetry": "Cs", "irrep": "A double prime",
            "basis": args.basis, "active_space": [args.nelecas, args.ncas], "nroots": 2,
            "state_average_weights": [[1-p, p] for p in args.excited_weights],
            "weight_semantics": "p is the excited-root weight; p <= 0.5 preserves ordered ensemble weights",
        },
        "geometry": {
            "r_sum_A": args.r_sum, "theta0_deg": args.theta0,
            "delta_r_A": args.delta_r, "delta_theta_deg": args.delta_theta,
            "points": [point.__dict__ for point in points],
        },
        "gate4b_workdir": str(args.gate4b_workdir),
        "new_primary_casscf_points": len(points) * (len(args.excited_weights) - 1),
        "new_history_control_casscf_points": len(points) if args.history_controls else 0,
        "new_casscf_points_total": len(points) * (len(args.excited_weights) - 1) + (len(points) if args.history_controls else 0),
        "reused_equal_weight_points": len(points),
        "thresholds": {
            "rhf_conv_tol": args.rhf_conv_tol, "fci_conv_tol": args.fci_conv_tol,
            "mc_conv_tol": args.mc_conv_tol, "mc_conv_tol_grad": args.mc_conv_tol_grad,
            "mc_stationary_window": args.mc_stationary_window,
            "mc_stationary_energy_tol": args.mc_stationary_energy_tol,
            "mc_stationary_grad_ceiling": args.mc_stationary_grad_ceiling,
            "mc_stationary_ddm_tol": args.mc_stationary_ddm_tol,
            "mc_stationary_maxrot_tol": args.mc_stationary_maxrot_tol,
        },
    }
    (args.workdir / "O3_weighted_SA2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Weighted-SA2 pilot wavefunctions are complete; run the analysis script next.", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gate4b-workdir", type=Path, required=True,
                        help="Audited Gate4B work directory containing wavefunctions/*_SA2.npz")
    parser.add_argument("--workdir", type=Path, default=Path("O3_weighted_SA2_pilot_work"))
    parser.add_argument("--excited-weights", default="0.20,0.35,0.50")
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--ncas", type=int, default=12)
    parser.add_argument("--nelecas", type=int, default=18)
    parser.add_argument("--r-sum", type=float, default=2.7288)
    parser.add_argument("--theta0", type=float, default=119.70)
    parser.add_argument("--delta-r", type=float, default=0.010)
    parser.add_argument("--delta-theta", type=float, default=0.20)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-mb", type=int, default=120000)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--rhf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--rhf-max-cycle", type=int, default=150)
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--fci-max-cycle", type=int, default=300)
    parser.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument("--mc-max-cycle-macro", type=int, default=60)
    parser.add_argument("--mc-max-cycle-micro", type=int, default=20)
    parser.add_argument("--mc-stationary-window", type=int, default=5)
    parser.add_argument("--mc-stationary-energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-stationary-grad-ceiling", type=float, default=1.0e-4)
    parser.add_argument("--mc-stationary-ddm-tol", type=float, default=1.0e-7)
    parser.add_argument("--mc-stationary-maxrot-tol", type=float, default=1.0e-7)
    parser.add_argument("--newton-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.gate4b_workdir = args.gate4b_workdir.expanduser().resolve()
    args.workdir = args.workdir.expanduser().resolve()
    try:
        args.excited_weights = parse_weights(args.excited_weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.threads < 1 or args.memory_mb < 1000:
        parser.error("threads must be positive and memory-mb must be at least 1000")
    if args.delta_r <= 0.0 or args.delta_theta <= 0.0:
        parser.error("geometry displacements must be positive")
    if args.mc_stationary_window < 3:
        parser.error("mc-stationary-window must be at least 3")
    points = build_points(args)
    if args.dry_run:
        print(f"Gate4B seed directory: {args.gate4b_workdir}")
        print(f"Output directory: {args.workdir}")
        print(f"Weights (ground, excited): {[(1-p, p) for p in args.excited_weights]}")
        primary = len(points) * (len(args.excited_weights) - 1)
        controls = len(points) if args.history_controls else 0
        print(f"Geometries: {len(points)}; reused p=0.5: {len(points)}; new primary CASSCF: {primary}; history controls: {controls}; total new: {primary+controls}")
        for point in points:
            r1, r2 = point.bonds(args.r_sum)
            seed = args.gate4b_workdir / "wavefunctions" / f"{point.tag}_SA2.npz"
            print(f"{point.tag:28s} r1={r1:.6f} r2={r2:.6f} theta={point.theta:.6f} seed={seed}")
        return
    for point in points:
        seed = args.gate4b_workdir / "wavefunctions" / f"{point.tag}_SA2.npz"
        if not seed.exists():
            parser.error(f"Missing audited Gate4B seed: {seed}")
    configure_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
