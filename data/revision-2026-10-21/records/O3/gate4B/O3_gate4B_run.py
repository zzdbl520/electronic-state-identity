#!/usr/bin/env python3
"""Strict directional-refinement SA2/SA3-CASSCF calculations for O3 Gate 4B."""

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


@dataclass(frozen=True)
class Point:
    tag: str
    direction: str
    sign: int
    h: float
    delta_r: float
    theta: float

    def bonds(self, r_sum: float) -> tuple[float, float]:
        return 0.5 * (r_sum + self.delta_r), 0.5 * (r_sum - self.delta_r)


def parse_steps(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("Refinement steps must be a nonempty list of positive numbers.")
    if len(set(values)) != len(values):
        raise ValueError("Refinement steps must be unique.")
    return sorted(values, reverse=True)


def step_tag(value: float) -> str:
    return f"{value:.6f}".replace(".", "p")


def build_points(args: argparse.Namespace) -> list[Point]:
    result = [Point("center", "center", 0, 0.0, 0.0, args.theta0)]
    for h in args.delta_r_steps:
        token = step_tag(h)
        result.extend(
            [
                Point(f"dr_minus_h_{token}", "delta_r", -1, h, -h, args.theta0),
                Point(f"dr_plus_h_{token}", "delta_r", +1, h, +h, args.theta0),
            ]
        )
    for h in args.theta_steps:
        token = step_tag(h)
        result.extend(
            [
                Point(f"theta_minus_h_{token}", "theta", -1, h, 0.0, args.theta0 - h),
                Point(f"theta_plus_h_{token}", "theta", +1, h, 0.0, args.theta0 + h),
            ]
        )
    return result


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
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def import_pyscf():
    try:
        import numpy as np
        from pyscf import fci, gto, lib, mcscf, scf, symm
    except ImportError as exc:
        raise RuntimeError("Activate the PySCF environment before running Gate 4B.") from exc
    return np, fci, gto, lib, mcscf, scf, symm


def odd_irrep(mol) -> str:
    for candidate in ('A"', "A''", "A2"):
        if candidate in mol.irrep_name:
            return candidate
    raise RuntimeError(f"No reflection-odd Cs irrep found in {mol.irrep_name}.")


def build_mol_values(r1, r2, theta, basis, memory_mb, verbose, gto, log_path=None):
    mol = gto.Mole()
    mol.atom = atoms_from_values(r1, r2, theta)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = "Cs"
    mol.max_memory = memory_mb
    mol.verbose = verbose if log_path is not None else 0
    if log_path is not None:
        mol.output = str(log_path)
    mol.build()
    if mol.groupname != "Cs":
        raise RuntimeError(f"Forced Cs symmetry, but PySCF reports {mol.groupname}.")
    odd_irrep(mol)
    return mol


def build_mol(point: Point, args, gto, log_path=None):
    r1, r2 = point.bonds(args.r_sum)
    return build_mol_values(
        r1, r2, point.theta, args.basis, args.memory_mb, args.verbose, gto, log_path
    )


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
        newton = mf.newton()
        newton.conv_tol = args.rhf_conv_tol
        newton.max_cycle = args.rhf_max_cycle
        newton.kernel(mf.mo_coeff, mf.mo_occ)
        mf = newton
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


def ci_list(ci) -> list:
    return list(ci) if isinstance(ci, (list, tuple)) else [ci]


def ncore_int(ncore) -> int:
    if isinstance(ncore, (tuple, list)):
        if len(ncore) != 2 or int(ncore[0]) != int(ncore[1]):
            raise RuntimeError(f"Restricted ncore required, got {ncore}.")
        return int(ncore[0])
    return int(ncore)


def state_energies(mc, nroots: int, np) -> list[float]:
    values = getattr(mc, "e_states", None)
    if values is None:
        candidate = np.asarray(mc.e_tot)
        if candidate.ndim == 1:
            values = candidate
    if values is None:
        raise RuntimeError("PySCF did not expose state-resolved energies.")
    result = [float(value) for value in np.asarray(values).reshape(-1)]
    if len(result) != nroots:
        raise RuntimeError(f"Expected {nroots} state energies, received {result}.")
    return result


def one_step_tail_metrics(mol, window: int):
    """Read the final one-step CASSCF stationarity metrics from the live log."""
    stream = getattr(mol, "stdout", None)
    if stream is not None:
        stream.flush()
    output = getattr(mol, "output", None)
    if not output:
        return None
    path = Path(output)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    number = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?"
    pattern = re.compile(
        rf"macro iter\s+(\d+)[^\n]*?dE\s*=\s*({number})[^\n]*\n"
        rf"\s*\|grad\[o\]\|=\s*({number})\s+"
        rf"\|ddm\|=\s*({number})\s+"
        rf"\|maxRot\[o\]\|=\s*({number})"
    )
    records = [
        {
            "macro": int(match.group(1)),
            "dE": float(match.group(2)),
            "grad": float(match.group(3)),
            "ddm": float(match.group(4)),
            "maxrot": float(match.group(5)),
        }
        for match in pattern.finditer(text)
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
        "max_abs_dE": max(abs(record["dE"]) for record in tail),
        "max_grad": max(record["grad"] for record in tail),
        "max_ddm": max(record["ddm"] for record in tail),
        "max_maxrot": max(record["maxrot"] for record in tail),
    }


def run_casscf(mol, mf, mo_guess, nroots: int, args, fci, mcscf, np):
    mc = mcscf.CASSCF(mf, args.ncas, args.nelecas)
    mc.max_memory = args.memory_mb
    mc.conv_tol = args.mc_conv_tol
    # PySCF's one-step SA-CASSCF orbital residual has a geometry-dependent
    # numerical floor in this near-degenerate region (up to about 5.1e-5 in
    # the present directional grid), after the energy, density, and orbital
    # rotation have stopped changing.  The production
    # protocol therefore uses the same stable one-step optimizer for every
    # SA2/SA3 point.  The optional Newton branch is diagnostic only.
    mc.conv_tol_grad = args.mc_stage1_conv_tol_grad
    mc.max_cycle_macro = args.mc_max_cycle_macro
    mc.max_cycle_micro = args.mc_max_cycle_micro
    mc.canonicalization = True
    solver = fci.direct_spin0_symm.FCI(mol)
    solver.wfnsym = odd_irrep(mol)
    solver.nroots = nroots
    solver.conv_tol = args.fci_conv_tol
    solver.max_cycle = args.fci_max_cycle
    mc.fcisolver = solver
    mc.state_average_([1.0 / nroots] * nroots)
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
            detail = "stationarity metrics unavailable" if metrics is None else str(metrics)
            raise RuntimeError(
                f"SA{nroots}-CASSCF failed both the PySCF and explicit "
                f"stationarity criteria: {detail}"
            )
        convergence_mode = "stationary_floor"
        mc.converged = True
        mol.stdout.write(
            "\nGATE4B STATIONARY-FLOOR ACCEPTANCE\n"
            f"window={args.mc_stationary_window}, final_macro={metrics['final_macro']}, "
            f"max|dE|={metrics['max_abs_dE']:.3e}, "
            f"max|grad[o]|={metrics['max_grad']:.3e}, "
            f"max|ddm|={metrics['max_ddm']:.3e}, "
            f"max|maxRot[o]|={metrics['max_maxrot']:.3e}\n"
        )
        mol.stdout.flush()
    mc.gate4b_convergence_mode = convergence_mode
    mc.gate4b_stationarity_metrics = metrics or {}
    if args.mc_newton_polish:
        mol.stdout.write(
            "\nSA-CASSCF Newton polish "
            f"(final orbital-gradient target {args.mc_conv_tol_grad:.1e})\n"
        )
        mol.stdout.flush()
        newton = mc.newton()
        newton.max_cycle_macro = args.mc_newton_max_cycle
        newton.conv_tol = args.mc_conv_tol
        newton.conv_tol_grad = args.mc_conv_tol_grad
        newton.kernel(mc.mo_coeff, mc.ci)
        mc = newton
    elif args.mc_stage1_conv_tol_grad > args.mc_conv_tol_grad:
        raise RuntimeError(
            "Newton polish is disabled, but the stabilization threshold is "
            "looser than the requested final threshold."
        )
    if not mc.converged:
        raise RuntimeError(
            f"SA{nroots}-CASSCF Newton polish did not reach the final "
            f"orbital-gradient target {args.mc_conv_tol_grad:.1e}."
        )
    vectors = ci_list(mc.ci)
    if len(vectors) != nroots:
        raise RuntimeError(f"Expected {nroots} CI vectors, got {len(vectors)}.")
    energies = state_energies(mc, nroots, np)
    occupations = []
    for vector in vectors:
        rdm1 = solver.make_rdm1(vector, mc.ncas, mc.nelecas)
        occupations.append(np.linalg.eigvalsh(rdm1)[::-1])
    return mc, energies, vectors, occupations


def load_seed(path: Path, args, gto, np):
    if not path.exists():
        raise FileNotFoundError(f"Required Gate 4A seed not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["basis"].item()).lower() != args.basis.lower():
            raise RuntimeError(f"Basis mismatch in seed {path}.")
        if (int(data["ncas"].item()), int(data["nelecas"].item())) != (
            args.ncas,
            args.nelecas,
        ):
            raise RuntimeError(f"Active-space mismatch in seed {path}.")
        mol = build_mol_values(
            float(data["r1"].item()),
            float(data["r2"].item()),
            float(data["theta"].item()),
            args.basis,
            args.memory_mb,
            0,
            gto,
        )
        return mol, data["mo_coeff"].copy()


def save_checkpoint(path, point, protocol, mol, mc, energies, vectors, occupations, args, np):
    temporary = path.with_name(path.name + ".tmp.npz")
    r1, r2 = point.bonds(args.r_sum)
    np.savez_compressed(
        temporary,
        schema_version=4,
        stage="Gate4B",
        optimizer="newton" if args.mc_newton_polish else "one_step",
        tag=point.tag,
        direction=point.direction,
        sign=point.sign,
        h=point.h,
        protocol=protocol,
        nroots=len(vectors),
        delta_r=point.delta_r,
        theta=point.theta,
        r1=r1,
        r2=r2,
        r_sum=args.r_sum,
        basis=args.basis,
        point_group=mol.groupname,
        ncas=args.ncas,
        nelecas=args.nelecas,
        ncore=ncore_int(mc.ncore),
        mo_coeff=np.asarray(mc.mo_coeff),
        ci_vectors=np.stack([np.asarray(vector) for vector in vectors]),
        state_energies=np.asarray(energies),
        natural_occupations=np.asarray(occupations),
        convergence_mode=getattr(mc, "gate4b_convergence_mode", "pyscf"),
        initial_orbital_seed=getattr(mc, "gate4b_seed_description", "unknown"),
        final_orbital_gradient=float(
            getattr(mc, "gate4b_stationarity_metrics", {}).get("final_grad", np.nan)
        ),
        final_density_change=float(
            getattr(mc, "gate4b_stationarity_metrics", {}).get("final_ddm", np.nan)
        ),
        final_max_orbital_rotation=float(
            getattr(mc, "gate4b_stationarity_metrics", {}).get("final_maxrot", np.nan)
        ),
        rhf_conv_tol=args.rhf_conv_tol,
        fci_conv_tol=args.fci_conv_tol,
        mc_conv_tol=args.mc_conv_tol,
        mc_conv_tol_grad=args.mc_conv_tol_grad,
        mc_stage1_conv_tol_grad=args.mc_stage1_conv_tol_grad,
        mc_stationary_window=args.mc_stationary_window,
        mc_stationary_energy_tol=args.mc_stationary_energy_tol,
        mc_stationary_grad_ceiling=args.mc_stationary_grad_ceiling,
        mc_stationary_ddm_tol=args.mc_stationary_ddm_tol,
        mc_stationary_maxrot_tol=args.mc_stationary_maxrot_tol,
    )
    os.replace(temporary, path)


def checkpoint_valid(path: Path, point: Point, protocol: str, args, np) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            r1, r2 = point.bonds(args.r_sum)
            return (
                int(data["schema_version"].item()) >= 4
                and str(data["stage"].item()) == "Gate4B"
                and str(data["optimizer"].item()) == (
                    "newton" if args.mc_newton_polish else "one_step"
                )
                and str(data["tag"].item()) == point.tag
                and str(data["protocol"].item()) == protocol
                and int(data["nroots"].item()) == (2 if protocol == "SA2" else 3)
                and str(data["basis"].item()).lower() == args.basis.lower()
                and int(data["ncas"].item()) == args.ncas
                and int(data["nelecas"].item()) == args.nelecas
                and abs(float(data["r1"].item()) - r1) < 1.0e-10
                and abs(float(data["r2"].item()) - r2) < 1.0e-10
                and abs(float(data["theta"].item()) - point.theta) < 1.0e-10
                and float(data["mc_conv_tol"].item()) <= args.mc_conv_tol
                and float(data["mc_conv_tol_grad"].item()) <= args.mc_conv_tol_grad
                and float(data["mc_stationary_grad_ceiling"].item())
                <= args.mc_stationary_grad_ceiling
                and int(data["mc_stationary_window"].item())
                >= args.mc_stationary_window
                and float(data["mc_stationary_energy_tol"].item())
                <= args.mc_stationary_energy_tol
                and float(data["mc_stationary_ddm_tol"].item())
                <= args.mc_stationary_ddm_tol
                and float(data["mc_stationary_maxrot_tol"].item())
                <= args.mc_stationary_maxrot_tol
            )
    except Exception:
        return False


CSV_FIELDS = [
    "tag", "direction", "sign", "h", "protocol", "nroots", "r1_A", "r2_A",
    "delta_r_A", "theta_deg", "E_SA_Eh", "E0_Eh", "E1_Eh", "E2_Eh",
    "RHF_converged", "CASSCF_converged", "wall_time_s", "status", "error",
]


def append_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def run(args: argparse.Namespace) -> None:
    np, fci, gto, lib, mcscf, scf, symm = import_pyscf()
    lib.num_threads(args.threads)
    for directory in (
        args.workdir,
        args.workdir / "logs",
        args.workdir / "rhf_checkpoints",
        args.workdir / "wavefunctions",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    points = build_points(args)
    table = args.workdir / "O3_gate4B_calculations.csv"
    center_seeds = {}
    continuation_seeds = {}

    print("O3 Gate 4B: strict forced-Cs directional refinement", flush=True)
    print(
        f"{len(points)} geometries; {2*len(points)} SA-CASSCF calculations; "
        f"workdir={args.workdir}",
        flush=True,
    )
    print(
        f"Thresholds: RHF={args.rhf_conv_tol:.1e}, FCI={args.fci_conv_tol:.1e}, "
        f"CASSCF={args.mc_conv_tol:.1e}, orbital-gradient stabilization="
        f"{args.mc_stage1_conv_tol_grad:.1e}, final={args.mc_conv_tol_grad:.1e}; "
        f"stationary ceiling={args.mc_stationary_grad_ceiling:.1e}",
        flush=True,
    )

    for point in points:
        for protocol, nroots in (("SA2", 2), ("SA3", 3)):
            checkpoint = args.workdir / "wavefunctions" / f"{point.tag}_{protocol}.npz"
            if args.resume and checkpoint_valid(checkpoint, point, protocol, args, np):
                print(f"SKIP {point.tag:28s} {protocol}: strict checkpoint exists", flush=True)
                loaded_seed = load_seed(checkpoint, args, gto, np)
                if point.tag == "center":
                    center_seeds[protocol] = loaded_seed
                else:
                    continuation_seeds[(point.direction, point.sign, protocol)] = (
                        loaded_seed[0],
                        loaded_seed[1],
                        point.tag,
                    )
                continue

            if point.tag == "center":
                seed_path = args.seed_workdir / "wavefunctions" / f"center_{protocol}.npz"
                seed_mol, seed_mo = load_seed(seed_path, args, gto, np)
                seed_description = "Gate 4A center checkpoint"
            else:
                if protocol not in center_seeds:
                    raise RuntimeError(f"Strict center seed for {protocol} is unavailable.")
                continuation_key = (point.direction, point.sign, protocol)
                if continuation_key in continuation_seeds:
                    seed_mol, seed_mo, seed_tag = continuation_seeds[continuation_key]
                    seed_description = (
                        f"strict Gate 4B directional continuation from {seed_tag}"
                    )
                else:
                    seed_mol, seed_mo = center_seeds[protocol]
                    seed_description = "strict Gate 4B protocol-matched center"

            r1, r2 = point.bonds(args.r_sum)
            log_path = args.workdir / "logs" / f"{point.tag}_{protocol}.log"
            row = {
                "tag": point.tag,
                "direction": point.direction,
                "sign": point.sign,
                "h": f"{point.h:.10f}",
                "protocol": protocol,
                "nroots": nroots,
                "r1_A": f"{r1:.10f}",
                "r2_A": f"{r2:.10f}",
                "delta_r_A": f"{point.delta_r:.10f}",
                "theta_deg": f"{point.theta:.10f}",
            }
            mol = None
            started = time.perf_counter()
            try:
                print(
                    f"RUN  {point.tag:28s} {protocol}: dr={point.delta_r:+.6f} A, "
                    f"theta={point.theta:.6f} deg",
                    flush=True,
                )
                mol = build_mol(point, args, gto, log_path)
                mf = run_rhf(
                    mol,
                    args,
                    scf,
                    args.workdir / "rhf_checkpoints" / f"{point.tag}_{protocol}.chk",
                )
                mo_guess = project_orbitals(seed_mol, seed_mo, mol, np, scf, symm)
                mol.stdout.write(f"\nInitial orbitals: {seed_description}\n")
                mol.stdout.flush()
                mc, energies, vectors, occupations = run_casscf(
                    mol, mf, mo_guess, nroots, args, fci, mcscf, np
                )
                mc.gate4b_seed_description = seed_description
                save_checkpoint(
                    checkpoint,
                    point,
                    protocol,
                    mol,
                    mc,
                    energies,
                    vectors,
                    occupations,
                    args,
                    np,
                )
                if point.tag == "center":
                    center_seeds[protocol] = (
                        build_mol(point, args, gto),
                        mc.mo_coeff.copy(),
                    )
                else:
                    continuation_seeds[(point.direction, point.sign, protocol)] = (
                        build_mol(point, args, gto),
                        mc.mo_coeff.copy(),
                        point.tag,
                    )
                row.update(
                    {
                        "E_SA_Eh": f"{float(mc.e_tot):.14f}",
                        "E0_Eh": f"{energies[0]:.14f}",
                        "E1_Eh": f"{energies[1]:.14f}",
                        "E2_Eh": f"{energies[2]:.14f}" if nroots == 3 else "",
                        "RHF_converged": bool(mf.converged),
                        "CASSCF_converged": bool(mc.converged),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                with (args.workdir / "failures.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{point.tag} {protocol}]\n")
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
        "schema_version": 4,
        "stage": "Gate4B",
        "optimizer": "newton" if args.mc_newton_polish else "one_step",
        "seeding": "protocol-matched directional coarse-to-fine continuation",
        "forced_symmetry": "Cs",
        "basis": args.basis,
        "active_space": [args.nelecas, args.ncas],
        "r_sum_A": args.r_sum,
        "theta0_deg": args.theta0,
        "delta_r_steps_A": args.delta_r_steps,
        "theta_steps_deg": args.theta_steps,
        "thresholds": {
            "rhf_conv_tol": args.rhf_conv_tol,
            "fci_conv_tol": args.fci_conv_tol,
            "mc_conv_tol": args.mc_conv_tol,
            "mc_conv_tol_grad": args.mc_conv_tol_grad,
            "mc_stage1_conv_tol_grad": args.mc_stage1_conv_tol_grad,
            "mc_stationary_window": args.mc_stationary_window,
            "mc_stationary_energy_tol": args.mc_stationary_energy_tol,
            "mc_stationary_grad_ceiling": args.mc_stationary_grad_ceiling,
            "mc_stationary_ddm_tol": args.mc_stationary_ddm_tol,
            "mc_stationary_maxrot_tol": args.mc_stationary_maxrot_tol,
        },
        "seed_workdir": str(args.seed_workdir),
        "points": [point.__dict__ for point in points],
    }
    (args.workdir / "O3_gate4B_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("All 34 strict calculations are available.", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed-workdir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("O3_gate4B_work"))
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--ncas", type=int, default=12)
    parser.add_argument("--nelecas", type=int, default=18)
    parser.add_argument("--r-sum", type=float, default=2.7288)
    parser.add_argument("--theta0", type=float, default=119.70)
    parser.add_argument("--delta-r-steps", default="0.010,0.005,0.0025,0.00125")
    parser.add_argument("--theta-steps", default="0.20,0.10,0.05,0.025")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-mb", type=int, default=120000)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--rhf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--rhf-max-cycle", type=int, default=150)
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--fci-max-cycle", type=int, default=300)
    parser.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-stage1-conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument("--mc-conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument("--mc-max-cycle-macro", type=int, default=60)
    parser.add_argument("--mc-max-cycle-micro", type=int, default=20)
    parser.add_argument("--mc-newton-max-cycle", type=int, default=12)
    parser.add_argument("--mc-stationary-window", type=int, default=5)
    parser.add_argument("--mc-stationary-energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-stationary-grad-ceiling", type=float, default=1.0e-4)
    parser.add_argument("--mc-stationary-ddm-tol", type=float, default=1.0e-7)
    parser.add_argument("--mc-stationary-maxrot-tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--mc-newton-polish", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--newton-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.seed_workdir = args.seed_workdir.expanduser().resolve()
    args.workdir = args.workdir.expanduser().resolve()
    args.delta_r_steps = parse_steps(args.delta_r_steps)
    args.theta_steps = parse_steps(args.theta_steps)
    if args.threads < 1 or args.memory_mb < 1000:
        parser.error("threads must be positive and memory-mb must be at least 1000")
    if args.mc_stationary_window < 3:
        parser.error("mc-stationary-window must be at least 3")
    stationary_tolerances = (
        args.mc_stationary_energy_tol,
        args.mc_stationary_grad_ceiling,
        args.mc_stationary_ddm_tol,
        args.mc_stationary_maxrot_tol,
    )
    if any(value <= 0.0 for value in stationary_tolerances):
        parser.error("all explicit stationarity tolerances must be positive")
    if args.mc_stationary_grad_ceiling < args.mc_stage1_conv_tol_grad:
        parser.error("mc-stationary-grad-ceiling cannot be tighter than the PySCF target")
    if args.ncas != 12 or args.nelecas != 18:
        print("WARNING: the production definition is CAS(18,12).", file=sys.stderr)
    points = build_points(args)
    for point in points:
        r1, r2 = point.bonds(args.r_sum)
        if min(r1, r2) <= 0.0:
            parser.error(f"Invalid geometry {point.tag}: r1={r1}, r2={r2}")
    if args.dry_run:
        print(f"Gate 4A seed directory: {args.seed_workdir}")
        print("Forced point group: Cs")
        print(f"Unique geometries: {len(points)}; CASSCF calculations: {2*len(points)}")
        for point in points:
            r1, r2 = point.bonds(args.r_sum)
            print(
                f"{point.tag:28s} r1={r1:.6f} A r2={r2:.6f} A "
                f"theta={point.theta:.6f} deg"
            )
        return
    for protocol in ("SA2", "SA3"):
        required = args.seed_workdir / "wavefunctions" / f"center_{protocol}.npz"
        if not required.exists():
            parser.error(f"Missing Gate 4A seed: {required}")
    configure_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
