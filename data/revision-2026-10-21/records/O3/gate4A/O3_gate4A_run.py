#!/usr/bin/env python3
"""Gate 4A: five-point SA2/SA3-CASSCF validation around the O3 CI.

The molecular plane is the xy plane and Cs symmetry is forced at every
geometry, including the C2v center.  This keeps the target state label (A'')
identical on every nuclear and protocol edge.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


@dataclass(frozen=True)
class Point:
    tag: str
    delta_r: float
    theta: float

    def bonds(self, r_sum: float) -> tuple[float, float]:
        return 0.5 * (r_sum + self.delta_r), 0.5 * (r_sum - self.delta_r)


def points(args: argparse.Namespace) -> list[Point]:
    return [
        Point("center", 0.0, args.theta0),
        Point("dr_minus", -args.delta_r_step, args.theta0),
        Point("dr_plus", +args.delta_r_step, args.theta0),
        Point("theta_minus", 0.0, args.theta0 - args.theta_step),
        Point("theta_plus", 0.0, args.theta0 + args.theta_step),
    ]


def atoms(point: Point, r_sum: float):
    r1, r2 = point.bonds(r_sum)
    half = math.radians(point.theta) / 2.0
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
        raise RuntimeError(
            "PySCF is not importable. Activate the intended environment "
            "(for example, `conda activate pyscf`) and rerun."
        ) from exc
    return np, fci, gto, lib, mcscf, scf, symm


def odd_irrep(mol) -> str:
    for candidate in ('A"', "A''", "A2"):
        if candidate in mol.irrep_name:
            return candidate
    raise RuntimeError(f"No reflection-odd Cs irrep found in {mol.irrep_name}.")


def build_mol(point, args, gto, log_path: Path | None):
    mol = gto.Mole()
    mol.atom = atoms(point, args.r_sum)
    mol.unit = "Angstrom"
    mol.basis = args.basis
    mol.charge = 0
    mol.spin = 0
    # A string forces this subgroup and prevents the center from becoming C2v.
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
        raise RuntimeError(
            f"Projected MO metric is singular (min={float(values.min()):.3e})."
        )
    inverse_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T.conj()
    return mo @ inverse_sqrt


def project_orbitals(seed_mol, seed_mo, mol, fallback_mo, np, scf, symm):
    if seed_mol is None or seed_mo is None:
        return fallback_mo, "RHF canonical orbitals"
    projected = scf.addons.project_mo_nr2nr(seed_mol, seed_mo, mol)
    overlap = mol.intor_symmetric("int1e_ovlp")
    projected = symmetric_orthonormalize(projected, overlap, np)
    projected = symm.symmetrize_space(mol, projected, s=overlap, check=True)
    return projected, "projected protocol-matched center orbitals"


def solver_for_roots(fci, mol, nroots: int):
    solver = fci.direct_spin0_symm.FCI(mol)
    solver.wfnsym = odd_irrep(mol)
    solver.nroots = nroots
    solver.conv_tol = 1.0e-10
    solver.max_cycle = 200
    return solver


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
    result = [float(x) for x in np.asarray(values).reshape(-1)]
    if len(result) != nroots:
        raise RuntimeError(f"Expected {nroots} state energies, received {result}.")
    return result


def natural_occupations(mc, solver, np) -> list[list[float]]:
    result = []
    for ci in ci_list(mc.ci):
        rdm1 = solver.make_rdm1(ci, mc.ncas, mc.nelecas)
        result.append([float(x) for x in np.linalg.eigvalsh(rdm1)[::-1]])
    return result


def run_casscf(mol, mf, mo_guess, nroots: int, args, fci, mcscf, np):
    mc = mcscf.CASSCF(mf, args.ncas, args.nelecas)
    mc.max_memory = args.memory_mb
    mc.conv_tol = args.mc_conv_tol
    mc.conv_tol_grad = args.mc_conv_tol_grad
    mc.max_cycle_macro = args.mc_max_cycle_macro
    mc.max_cycle_micro = args.mc_max_cycle_micro
    mc.canonicalization = True
    solver = solver_for_roots(fci, mol, nroots)
    mc.fcisolver = solver
    mc.state_average_([1.0 / nroots] * nroots)
    mc.kernel(mo_guess)
    if not mc.converged and args.newton_fallback:
        mol.stdout.write("\nSA-CASSCF Newton fallback\n")
        mol.stdout.flush()
        newton = mc.newton()
        newton.max_cycle_macro = args.mc_newton_max_cycle
        newton.conv_tol = args.mc_conv_tol
        newton.conv_tol_grad = args.mc_conv_tol_grad
        newton.kernel(mc.mo_coeff, mc.ci)
        mc = newton
    if not mc.converged:
        raise RuntimeError(f"SA{nroots}-CASSCF did not converge.")
    energies = state_energies(mc, nroots, np)
    vectors = ci_list(mc.ci)
    if len(vectors) != nroots:
        raise RuntimeError(f"Expected {nroots} CI vectors, got {len(vectors)}.")
    occupations = natural_occupations(mc, solver, np)
    return mc, energies, vectors, occupations


def save_checkpoint(path, point, protocol, mol, mc, energies, vectors, occ, args, np):
    temporary = path.with_name(path.name + ".tmp.npz")
    r1, r2 = point.bonds(args.r_sum)
    np.savez_compressed(
        temporary,
        schema_version=1,
        tag=point.tag,
        protocol=protocol,
        nroots=len(vectors),
        delta_r=point.delta_r,
        theta=point.theta,
        r1=r1,
        r2=r2,
        r_sum=args.r_sum,
        basis=args.basis,
        point_group=mol.groupname,
        odd_irrep=odd_irrep(mol),
        ncas=args.ncas,
        nelecas=args.nelecas,
        ncore=ncore_int(mc.ncore),
        mo_coeff=np.asarray(mc.mo_coeff),
        ci_vectors=np.stack([np.asarray(x) for x in vectors]),
        state_energies=np.asarray(energies),
        natural_occupations=np.asarray(occ),
    )
    os.replace(temporary, path)


def checkpoint_valid(path: Path, point: Point, protocol: str, args, np) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            r1, r2 = point.bonds(args.r_sum)
            return (
                str(data["tag"].item()) == point.tag
                and str(data["protocol"].item()) == protocol
                and int(data["nroots"].item()) == (2 if protocol == "SA2" else 3)
                and str(data["basis"].item()).lower() == args.basis.lower()
                and int(data["ncas"].item()) == args.ncas
                and int(data["nelecas"].item()) == args.nelecas
                and abs(float(data["r1"].item()) - r1) < 1.0e-10
                and abs(float(data["r2"].item()) - r2) < 1.0e-10
                and abs(float(data["theta"].item()) - point.theta) < 1.0e-10
                and str(data["point_group"].item()) == "Cs"
            )
    except Exception:
        return False


def load_seed(path: Path, point: Point, args, gto, np):
    with np.load(path, allow_pickle=False) as data:
        mo = data["mo_coeff"].copy()
    return build_mol(point, args, gto, None), mo


def append_row(path: Path, row: dict) -> None:
    fields = [
        "tag", "protocol", "nroots", "r1_A", "r2_A", "delta_r_A",
        "theta_deg", "E_SA_Eh", "E0_Eh", "E1_Eh", "E2_Eh",
        "RHF_converged", "CASSCF_converged", "wall_time_s", "status", "error",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fields})


def run(args: argparse.Namespace) -> None:
    np, fci, gto, lib, mcscf, scf, symm = import_pyscf()
    lib.num_threads(args.threads)
    args.workdir.mkdir(parents=True, exist_ok=True)
    logs = args.workdir / "logs"
    rhf_chk = args.workdir / "rhf_checkpoints"
    wavefunctions = args.workdir / "wavefunctions"
    for directory in (logs, rhf_chk, wavefunctions):
        directory.mkdir(exist_ok=True)
    table = args.workdir / "O3_gate4A_calculations.csv"

    center_seeds: dict[str, tuple[object, object]] = {}
    all_points = points(args)
    print("O3 Gate 4A: forced-Cs SA2/SA3-CASSCF(18,12)", flush=True)
    print(f"Five geometries; ten calculations; output={args.workdir}", flush=True)

    for point in all_points:
        for protocol, nroots in (("SA2", 2), ("SA3", 3)):
            chk = wavefunctions / f"{point.tag}_{protocol}.npz"
            if args.resume and checkpoint_valid(chk, point, protocol, args, np):
                print(f"SKIP {point.tag:12s} {protocol}: checkpoint exists", flush=True)
                if point.tag == "center":
                    center_seeds[protocol] = load_seed(chk, point, args, gto, np)
                continue

            log_path = logs / f"{point.tag}_{protocol}.log"
            mol = None
            started = time.perf_counter()
            r1, r2 = point.bonds(args.r_sum)
            row = {
                "tag": point.tag, "protocol": protocol, "nroots": nroots,
                "r1_A": f"{r1:.8f}", "r2_A": f"{r2:.8f}",
                "delta_r_A": f"{point.delta_r:.8f}",
                "theta_deg": f"{point.theta:.8f}",
            }
            try:
                print(
                    f"RUN  {point.tag:12s} {protocol}: r1={r1:.4f} A, "
                    f"r2={r2:.4f} A, theta={point.theta:.2f} deg",
                    flush=True,
                )
                mol = build_mol(point, args, gto, log_path)
                mf = run_rhf(mol, args, scf, rhf_chk / f"{point.tag}_{protocol}.chk")
                if point.tag == "center":
                    if protocol == "SA3" and "SA2" in center_seeds:
                        seed_mol, seed_mo = center_seeds["SA2"]
                    else:
                        seed_mol, seed_mo = None, None
                else:
                    seed_mol, seed_mo = center_seeds.get(protocol, (None, None))
                mo_guess, source = project_orbitals(
                    seed_mol, seed_mo, mol, mf.mo_coeff, np, scf, symm
                )
                mol.stdout.write(f"\nInitial CASSCF orbitals: {source}\n")
                mol.stdout.flush()
                mc, energies, vectors, occ = run_casscf(
                    mol, mf, mo_guess, nroots, args, fci, mcscf, np
                )
                save_checkpoint(chk, point, protocol, mol, mc, energies, vectors, occ, args, np)
                if point.tag == "center":
                    center_seeds[protocol] = (build_mol(point, args, gto, None), mc.mo_coeff.copy())
                row.update({
                    "E_SA_Eh": f"{float(mc.e_tot):.14f}",
                    "E0_Eh": f"{energies[0]:.14f}",
                    "E1_Eh": f"{energies[1]:.14f}",
                    "E2_Eh": f"{energies[2]:.14f}" if nroots == 3 else "",
                    "RHF_converged": bool(mf.converged),
                    "CASSCF_converged": bool(mc.converged),
                    "status": "ok", "error": "",
                })
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                with (args.workdir / "failures.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{point.tag} {protocol}]\n")
                    traceback.print_exc(file=handle)
                append_row(table, {**row, "wall_time_s": f"{time.perf_counter()-started:.3f}"})
                if mol is not None:
                    close_log(mol)
                raise
            row["wall_time_s"] = f"{time.perf_counter()-started:.3f}"
            append_row(table, row)
            if mol is not None:
                close_log(mol)

    manifest = {
        "schema_version": 1,
        "system": "O3",
        "purpose": "Gate 4A mixed nuclear-variational nonclosure pilot",
        "forced_symmetry": "Cs",
        "basis": args.basis,
        "active_space": [args.nelecas, args.ncas],
        "r_sum_A": args.r_sum,
        "theta0_deg": args.theta0,
        "delta_r_step_A": args.delta_r_step,
        "theta_step_deg": args.theta_step,
        "points": [p.__dict__ for p in all_points],
        "protocols": {"SA2": 2, "SA3": 3},
    }
    (args.workdir / "O3_gate4A_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("All ten calculations are available.", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workdir", type=Path, default=Path("O3_gate4A_work"))
    p.add_argument("--basis", default="cc-pvtz")
    p.add_argument("--ncas", type=int, default=12)
    p.add_argument("--nelecas", type=int, default=18)
    p.add_argument("--r-sum", type=float, default=2.7288)
    p.add_argument("--theta0", type=float, default=119.70)
    p.add_argument("--delta-r-step", type=float, default=0.010)
    p.add_argument("--theta-step", type=float, default=0.20)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--memory-mb", type=int, default=120000)
    p.add_argument("--verbose", type=int, default=4)
    p.add_argument("--rhf-conv-tol", type=float, default=1.0e-10)
    p.add_argument("--rhf-max-cycle", type=int, default=100)
    p.add_argument("--mc-conv-tol", type=float, default=1.0e-8)
    p.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-5)
    p.add_argument("--mc-max-cycle-macro", type=int, default=100)
    p.add_argument("--mc-max-cycle-micro", type=int, default=10)
    p.add_argument("--mc-newton-max-cycle", type=int, default=60)
    p.add_argument("--newton-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    if args.threads < 1 or args.memory_mb < 1000:
        raise SystemExit("threads must be positive and memory-mb must be >= 1000")
    if args.ncas != 12 or args.nelecas != 18:
        print("WARNING: production definition is CAS(18,12).", file=sys.stderr)
    for point in points(args):
        r1, r2 = point.bonds(args.r_sum)
        if min(r1, r2) <= 0:
            raise SystemExit(f"Invalid geometry {point.tag}: r1={r1}, r2={r2}")
    if args.dry_run:
        print("Forced point group: Cs")
        print("Protocol/state counts: SA2=2 A'' roots; SA3=3 A'' roots")
        for point in points(args):
            r1, r2 = point.bonds(args.r_sum)
            print(
                f"{point.tag:12s} r1={r1:.4f} A r2={r2:.4f} A "
                f"delta_r={point.delta_r:+.4f} A theta={point.theta:.2f} deg"
            )
        print("Total CASSCF calculations: 10")
        return
    configure_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
