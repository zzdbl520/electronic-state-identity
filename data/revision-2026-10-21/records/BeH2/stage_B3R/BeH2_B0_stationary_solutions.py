#!/usr/bin/env python3
"""Generate the BeH2 B0 stationary-solution control set.

The calculation uses the Be + H2 insertion coordinate already employed in the
original scripts,

    Be = (0, 0, 0)
    H  = (lambda, +(2.540 - 0.46 lambda), 0)
    H  = (lambda, -(2.540 - 0.46 lambda), 0),

with distances in bohr.  Equal-weight SA3- and SA4-CASSCF calculations are
performed separately in the A1 and B2 symmetry blocks.  Four initial orbital
sources are retained at each target geometry: left continuation, right
continuation, common A1/B2 SA2, and fresh RHF.

Typical use
-----------
python BeH2_B0_stationary_solutions.py \
  --workdir BeH2_B0_work \
  --threads 64 | tee BeH2_B0.stdout

The run is restartable: completed NPZ solution files are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_TARGETS = (2.6235, 2.6240, 2.62425, 2.6245, 2.6250)
DEFAULT_IRREPS = ("A1", "B2")
DEFAULT_PROTOCOLS = (3, 4)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "BeH2 B0 same-irrep SA3/SA4 stationary-solution reconnaissance."
        )
    )
    result.add_argument("--workdir", type=Path, default=Path("BeH2_B0_work"))
    result.add_argument("--basis", default="cc-pVTZ")
    result.add_argument("--ncas", type=int, default=6)
    result.add_argument("--nelecas", type=int, default=6)
    result.add_argument("--freeze-core", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--left-anchor", type=float, default=2.6160)
    result.add_argument("--right-anchor", type=float, default=2.6270)
    result.add_argument("--branch-step", type=float, default=0.0005)
    result.add_argument(
        "--targets",
        default=",".join(f"{value:.8f}" for value in DEFAULT_TARGETS),
        help="Comma-separated target lambda values in bohr.",
    )
    result.add_argument(
        "--irreps",
        default=",".join(DEFAULT_IRREPS),
        help="Comma-separated C2v irreps.",
    )
    result.add_argument(
        "--protocols",
        default=",".join(str(value) for value in DEFAULT_PROTOCOLS),
        help="Comma-separated state counts, normally 3,4.",
    )
    result.add_argument("--threads", type=int, default=64)
    result.add_argument("--memory-mb", type=int, default=100000)
    result.add_argument("--hf-conv-tol", type=float, default=1.0e-11)
    result.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    result.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    result.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    result.add_argument("--mc-max-cycle-macro", type=int, default=160)
    result.add_argument("--mc-max-cycle-micro", type=int, default=12)
    result.add_argument("--fci-max-cycle", type=int, default=200)
    result.add_argument("--verbose", type=int, default=4)
    result.add_argument(
        "--newton-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument(
        "--sources",
        default="left,right,common_sa2,rhf",
        help="Subset of left,right,common_sa2,rhf.",
    )
    result.add_argument("--self-test", action="store_true")
    return result


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one target value is required.")
    return tuple(sorted(set(values)))


def parse_text_list(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one item is required.")
    return values


def lambda_tag(value: float) -> str:
    return f"{value:.8f}".replace("-", "m").replace(".", "p")


def insertion_y(lam: float) -> float:
    return 2.540 - 0.46 * lam


def path_points(start: float, stop: float, step: float, extras: Sequence[float]) -> tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("branch step must be positive")
    direction = 1.0 if stop >= start else -1.0
    signed_step = direction * step
    values = [start]
    current = start
    while direction * (stop - current) > 0.25 * step:
        current += signed_step
        if direction * (current - stop) > 0.25 * step:
            current = stop
        values.append(current)
    values.extend(
        value
        for value in extras
        if min(start, stop) - 1.0e-10 <= value <= max(start, stop) + 1.0e-10
    )
    unique = sorted(set(round(value, 10) for value in values), reverse=direction < 0)
    return tuple(float(value) for value in unique)


def configure_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def import_pyscf() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from pyscf import fci, gto, lib, mcscf, scf
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for production. Activate the same PySCF "
            "environment used for the original BeH2 calculations."
        ) from exc
    return fci, gto, lib, mcscf, scf


@dataclass
class RunConfig:
    basis: str
    ncas: int
    nelecas: int
    freeze_core: bool
    threads: int
    memory_mb: int
    hf_conv_tol: float
    mc_conv_tol: float
    mc_conv_tol_grad: float
    fci_conv_tol: float
    mc_max_cycle_macro: int
    mc_max_cycle_micro: int
    fci_max_cycle: int
    verbose: int
    newton_fallback: bool


def make_mol(lam: float, config: RunConfig, log_path: Path, gto: Any) -> Any:
    y = insertion_y(lam)
    mol = gto.Mole()
    mol.atom = f"""
    Be  0.000000000000  0.000000000000  0.000000000000
    H  {lam: .12f}     {y: .12f}        0.000000000000
    H  {lam: .12f}    {-y: .12f}        0.000000000000
    """
    mol.unit = "Bohr"
    mol.basis = config.basis
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = "C2v"
    mol.max_memory = config.memory_mb
    # BeH2/cc-pVTZ is small enough for deterministic in-core AO integrals.
    # This also avoids platform-dependent /proc memory probing in containers.
    mol.incore_anyway = True
    mol.verbose = config.verbose
    mol.output = str(log_path)
    mol.build()
    return mol


def close_mol(mol: Any) -> None:
    try:
        mol.stdout.flush()
    except Exception:
        pass
    try:
        if mol.stdout not in (sys.stdout, sys.stderr):
            mol.stdout.close()
    except Exception:
        pass


def run_rhf(mol: Any, config: RunConfig, scf: Any) -> Any:
    mf = scf.RHF(mol)
    mf.conv_tol = config.hf_conv_tol
    mf.max_cycle = 100
    mf.max_memory = config.memory_mb
    mf.kernel()
    if not mf.converged and config.newton_fallback:
        mf_newton = mf.newton()
        mf_newton.conv_tol = config.hf_conv_tol
        mf_newton.max_cycle = 100
        mf_newton.kernel(mf.mo_coeff, mf.mo_occ)
        mf = mf_newton
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")
    return mf


def scalar_from_env(value: Any) -> float:
    if value is None:
        return math.nan
    array = np.asarray(value)
    if array.size == 0:
        return math.nan
    return float(np.linalg.norm(array)) if array.size > 1 else float(array.reshape(-1)[0])


def ci_state_list(ci: Any) -> list[np.ndarray]:
    if isinstance(ci, (list, tuple)):
        return [np.asarray(item) for item in ci]
    array = np.asarray(ci)
    if array.ndim >= 3:
        return [np.asarray(item) for item in array]
    return [array]


def run_same_irrep_sa(
    mol: Any,
    mf: Any,
    irrep: str,
    nroots: int,
    mo_guess: np.ndarray,
    config: RunConfig,
    fci: Any,
    mcscf: Any,
) -> dict[str, Any]:
    solver = fci.direct_spin0_symm.FCISolver(mol)
    solver.wfnsym = irrep
    solver.nroots = nroots
    solver.conv_tol = config.fci_conv_tol
    solver.max_cycle = config.fci_max_cycle

    mc = mcscf.CASSCF(mf, config.ncas, config.nelecas)
    mc.fcisolver = solver
    if config.freeze_core:
        mc.frozen = 1
    mc.max_memory = config.memory_mb
    mc.conv_tol = config.mc_conv_tol
    mc.conv_tol_grad = config.mc_conv_tol_grad
    mc.max_cycle_macro = config.mc_max_cycle_macro
    mc.max_cycle_micro = config.mc_max_cycle_micro
    mc.canonicalization = True
    mc.state_average_([1.0 / nroots] * nroots)

    trace = {
        "macroiterations": 0,
        "norm_gorb": math.nan,
        "norm_ddm": math.nan,
    }

    def callback(envs: dict[str, Any]) -> None:
        imacro = envs.get("imacro")
        if imacro is not None:
            trace["macroiterations"] = max(trace["macroiterations"], int(imacro) + 1)
        if "norm_gorb" in envs:
            trace["norm_gorb"] = scalar_from_env(envs.get("norm_gorb"))
        if "norm_ddm" in envs:
            trace["norm_ddm"] = scalar_from_env(envs.get("norm_ddm"))

    mc.callback = callback
    mc.kernel(mo_guess)
    if not mc.converged and config.newton_fallback:
        mc_newton = mc.newton()
        mc_newton.max_cycle_macro = config.mc_max_cycle_macro
        mc_newton.conv_tol = config.mc_conv_tol
        mc_newton.conv_tol_grad = config.mc_conv_tol_grad
        mc_newton.kernel(mc.mo_coeff, mc.ci)
        mc = mc_newton
    if not mc.converged:
        raise RuntimeError(f"SA{nroots}-{irrep} CASSCF did not converge.")

    ci_states = ci_state_list(mc.ci)
    if len(ci_states) != nroots:
        raise RuntimeError(f"Expected {nroots} CI vectors; received {len(ci_states)}.")
    energies = np.asarray(mc.e_states, dtype=float).reshape(-1)
    if energies.size != nroots:
        raise RuntimeError(f"Expected {nroots} state energies; received {energies.size}.")
    rdm1_states = np.stack(
        [
            np.asarray(
                fci.direct_spin0_symm.make_rdm1(
                    ci_vector,
                    mc.ncas,
                    mc.nelecas,
                )
            )
            for ci_vector in ci_states
        ]
    )
    return {
        "mc": mc,
        "mo_coeff": np.asarray(mc.mo_coeff),
        "ci_states": np.stack(ci_states),
        "rdm1_states": rdm1_states,
        "energies": energies,
        "e_average": float(np.mean(energies)),
        "ncore": int(mc.ncore),
        "nelecas": tuple(int(value) for value in mc.nelecas),
        "macroiterations": int(trace["macroiterations"]),
        "norm_gorb": float(trace["norm_gorb"]),
        "norm_ddm": float(trace["norm_ddm"]),
    }


def run_common_sa2(
    mol: Any,
    mf: Any,
    config: RunConfig,
    fci: Any,
    mcscf: Any,
) -> dict[str, Any]:
    solvers = []
    for irrep in ("A1", "B2"):
        solver = fci.direct_spin0_symm.FCISolver(mol)
        solver.wfnsym = irrep
        solver.nroots = 1
        solver.conv_tol = config.fci_conv_tol
        solver.max_cycle = config.fci_max_cycle
        solvers.append(solver)
    mc = mcscf.CASSCF(mf, config.ncas, config.nelecas)
    if config.freeze_core:
        mc.frozen = 1
    mc.max_memory = config.memory_mb
    mc.conv_tol = config.mc_conv_tol
    mc.conv_tol_grad = config.mc_conv_tol_grad
    mc.max_cycle_macro = config.mc_max_cycle_macro
    mc.max_cycle_micro = config.mc_max_cycle_micro
    mc.canonicalization = True
    mcscf.state_average_mix_(mc, solvers, weights=[0.5, 0.5])
    mc.kernel(mf.mo_coeff)
    if not mc.converged:
        raise RuntimeError("Common A1/B2 SA2-CASSCF did not converge.")
    return {
        "mo_coeff": np.asarray(mc.mo_coeff),
        "energies": np.asarray(mc.e_states, dtype=float),
        "e_average": float(np.mean(np.asarray(mc.e_states, dtype=float))),
        "ncore": int(mc.ncore),
    }


def solution_path(workdir: Path, irrep: str, nroots: int, source: str, lam: float) -> Path:
    return (
        workdir
        / "solutions"
        / irrep
        / f"SA{nroots}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )


def log_path(workdir: Path, irrep: str, nroots: int, source: str, lam: float) -> Path:
    return (
        workdir
        / "logs"
        / irrep
        / f"SA{nroots}"
        / source
        / f"lambda_{lambda_tag(lam)}.log"
    )


def failure_path(workdir: Path, irrep: str, nroots: int, source: str, lam: float) -> Path:
    return (
        workdir
        / "failures"
        / irrep
        / f"SA{nroots}"
        / source
        / f"lambda_{lambda_tag(lam)}.json"
    )


def save_solution(
    path: Path,
    result: dict[str, Any],
    mol: Any,
    lam: float,
    irrep: str,
    nroots: int,
    source: str,
    config: RunConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lambda_bohr=float(lam),
        irrep=irrep,
        protocol=f"SA{nroots}",
        nroots=int(nroots),
        source=source,
        basis=config.basis,
        ncas=int(config.ncas),
        nelecas=np.asarray(result["nelecas"], dtype=int),
        ncore=int(result["ncore"]),
        freeze_core=bool(config.freeze_core),
        atom_coords_bohr=np.asarray(mol.atom_coords(unit="Bohr")),
        atom_charges=np.asarray(mol.atom_charges(), dtype=int),
        ao_overlap=np.asarray(mol.intor_symmetric("int1e_ovlp")),
        mo_coeff=np.asarray(result["mo_coeff"]),
        ci_states=np.asarray(result["ci_states"]),
        rdm1_states=np.asarray(result["rdm1_states"]),
        energies=np.asarray(result["energies"], dtype=float),
        e_average=float(result["e_average"]),
        macroiterations=int(result["macroiterations"]),
        norm_gorb=float(result["norm_gorb"]),
        norm_ddm=float(result["norm_ddm"]),
    )


def load_mo(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["mo_coeff"])


def write_failure(path: Path, error: BaseException, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **context,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def project_guess(mcscf: Any, mc: Any, previous_mo: np.ndarray, previous_mol: Any) -> np.ndarray:
    return np.asarray(mcscf.project_init_guess(mc, previous_mo, prev_mol=previous_mol))


def run_attempt(
    workdir: Path,
    lam: float,
    irrep: str,
    nroots: int,
    source: str,
    config: RunConfig,
    fci: Any,
    gto: Any,
    mcscf: Any,
    scf: Any,
    mo_seed: np.ndarray | None = None,
    seed_mol: Any | None = None,
) -> tuple[np.ndarray, Any, Path]:
    output = solution_path(workdir, irrep, nroots, source, lam)
    log = log_path(workdir, irrep, nroots, source, lam)
    log.parent.mkdir(parents=True, exist_ok=True)
    mol = make_mol(lam, config, log, gto)
    if output.exists():
        print(f"  REUSE {irrep} SA{nroots} {source} lambda={lam:.8f}", flush=True)
        return load_mo(output), mol, output
    mf = None
    try:
        mf = run_rhf(mol, config, scf)
        if mo_seed is None:
            mo_guess = np.asarray(mf.mo_coeff)
        else:
            temp_mc = mcscf.CASSCF(mf, config.ncas, config.nelecas)
            if config.freeze_core:
                temp_mc.frozen = 1
            mo_guess = (
                project_guess(mcscf, temp_mc, mo_seed, seed_mol)
                if seed_mol is not None
                else np.asarray(mo_seed)
            )
        print(f"  RUN   {irrep} SA{nroots} {source} lambda={lam:.8f}", flush=True)
        result = run_same_irrep_sa(
            mol,
            mf,
            irrep,
            nroots,
            mo_guess,
            config,
            fci,
            mcscf,
        )
        save_solution(output, result, mol, lam, irrep, nroots, source, config)
        failed = failure_path(workdir, irrep, nroots, source, lam)
        if failed.exists():
            failed.unlink()
        return np.asarray(result["mo_coeff"]), mol, output
    except Exception as error:
        write_failure(
            failure_path(workdir, irrep, nroots, source, lam),
            error,
            {
                "lambda_bohr": lam,
                "irrep": irrep,
                "protocol": f"SA{nroots}",
                "source": source,
            },
        )
        close_mol(mol)
        raise


def run_branch(
    workdir: Path,
    points: Sequence[float],
    irrep: str,
    nroots: int,
    source: str,
    config: RunConfig,
    modules: tuple[Any, Any, Any, Any],
) -> None:
    fci, gto, mcscf, scf = modules
    previous_mo = None
    previous_mol = None
    for lam in points:
        mo, mol, _path = run_attempt(
            workdir,
            lam,
            irrep,
            nroots,
            source,
            config,
            fci,
            gto,
            mcscf,
            scf,
            previous_mo,
            previous_mol,
        )
        if previous_mol is not None:
            close_mol(previous_mol)
        previous_mo = mo
        previous_mol = mol
    if previous_mol is not None:
        close_mol(previous_mol)


def common_seed_path(workdir: Path, lam: float) -> Path:
    return workdir / "seeds" / "common_sa2" / f"lambda_{lambda_tag(lam)}.npz"


def build_common_seed(
    workdir: Path,
    lam: float,
    config: RunConfig,
    fci: Any,
    gto: Any,
    mcscf: Any,
    scf: Any,
) -> np.ndarray:
    path = common_seed_path(workdir, lam)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["mo_coeff"])
    log = workdir / "logs" / "common_sa2_seeds" / f"lambda_{lambda_tag(lam)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    mol = make_mol(lam, config, log, gto)
    try:
        mf = run_rhf(mol, config, scf)
        result = run_common_sa2(mol, mf, config, fci, mcscf)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            lambda_bohr=lam,
            mo_coeff=result["mo_coeff"],
            energies=result["energies"],
            e_average=result["e_average"],
            ncore=result["ncore"],
        )
        return np.asarray(result["mo_coeff"])
    finally:
        close_mol(mol)


def run_independent_sources(
    workdir: Path,
    targets: Sequence[float],
    irreps: Sequence[str],
    protocols: Sequence[int],
    sources: set[str],
    config: RunConfig,
    modules: tuple[Any, Any, Any, Any],
) -> None:
    fci, gto, mcscf, scf = modules
    for lam in targets:
        common_mo = None
        if "common_sa2" in sources:
            print(f"\nCOMMON-SA2 seed lambda={lam:.8f}", flush=True)
            common_mo = build_common_seed(workdir, lam, config, fci, gto, mcscf, scf)
        for irrep in irreps:
            for nroots in protocols:
                if "common_sa2" in sources and common_mo is not None:
                    _mo, mol, _path = run_attempt(
                        workdir,
                        lam,
                        irrep,
                        nroots,
                        "common_sa2",
                        config,
                        fci,
                        gto,
                        mcscf,
                        scf,
                        common_mo,
                        None,
                    )
                    close_mol(mol)
                if "rhf" in sources:
                    _mo, mol, _path = run_attempt(
                        workdir,
                        lam,
                        irrep,
                        nroots,
                        "rhf",
                        config,
                        fci,
                        gto,
                        mcscf,
                        scf,
                    )
                    close_mol(mol)


def collect_attempts(workdir: Path) -> None:
    rows = []
    for path in sorted((workdir / "solutions").glob("*/*/*/lambda_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            energies = np.asarray(archive["energies"], dtype=float)
            row = {
                "lambda_bohr": float(archive["lambda_bohr"]),
                "irrep": str(archive["irrep"]),
                "protocol": str(archive["protocol"]),
                "source": str(archive["source"]),
                "e_average": float(archive["e_average"]),
                "macroiterations": int(archive["macroiterations"]),
                "norm_gorb": float(archive["norm_gorb"]),
                "norm_ddm": float(archive["norm_ddm"]),
                "solution_file": str(path.resolve()),
            }
            for index, energy in enumerate(energies):
                row[f"E{index}"] = float(energy)
            rows.append(row)
    if not rows:
        return
    energy_keys = sorted(
        {key for row in rows for key in row if key.startswith("E") and key[1:].isdigit()},
        key=lambda item: int(item[1:]),
    )
    fieldnames = [
        "lambda_bohr",
        "irrep",
        "protocol",
        "source",
        "e_average",
        *energy_keys,
        "macroiterations",
        "norm_gorb",
        "norm_ddm",
        "solution_file",
    ]
    with (workdir / "B0_attempts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def configuration_payload(
    args: argparse.Namespace,
    targets: Sequence[float],
    irreps: Sequence[str],
    protocols: Sequence[int],
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "coordinate": {
            "unit": "Bohr",
            "Be": [0.0, 0.0, 0.0],
            "H1": ["lambda", "+(2.540-0.46*lambda)", 0.0],
            "H2": ["lambda", "-(2.540-0.46*lambda)", 0.0],
        },
        "basis": args.basis,
        "active_space": {"electrons": args.nelecas, "orbitals": args.ncas},
        "freeze_core": args.freeze_core,
        "targets_bohr": list(targets),
        "irreps": list(irreps),
        "protocols": [f"SA{value}" for value in protocols],
        "sources": list(sources),
        "left_anchor_bohr": args.left_anchor,
        "right_anchor_bohr": args.right_anchor,
        "branch_step_bohr": args.branch_step,
        "threads": args.threads,
        "memory_mb": args.memory_mb,
    }


def self_test() -> int:
    targets = DEFAULT_TARGETS
    left = path_points(2.6160, max(targets), 0.0005, targets)
    right = path_points(2.6270, min(targets), 0.0005, targets)
    for value in targets:
        if value not in left or value not in right:
            raise AssertionError(f"Target {value} missing from a branch path.")
    if left != tuple(sorted(left)):
        raise AssertionError("Left path is not ascending.")
    if right != tuple(sorted(right, reverse=True)):
        raise AssertionError("Right path is not descending.")
    if abs(insertion_y(2.6240) - 1.33296) > 1.0e-12:
        raise AssertionError("Coordinate formula check failed.")
    if lambda_tag(2.62425) != "2p62425000":
        raise AssertionError("Lambda tag check failed.")
    print("BeH2 B0 runner self-test: PASS")
    print(f"left path points : {len(left)}")
    print(f"right path points: {len(right)}")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    targets = parse_float_list(args.targets)
    irreps = tuple(value.upper() for value in parse_text_list(args.irreps))
    protocols = tuple(int(value) for value in parse_text_list(args.protocols))
    sources = set(parse_text_list(args.sources))
    invalid_sources = sources - {"left", "right", "common_sa2", "rhf"}
    if invalid_sources:
        raise ValueError(f"Unknown sources: {sorted(invalid_sources)}")
    if args.left_anchor >= min(targets):
        raise ValueError("left anchor must be below every target")
    if args.right_anchor <= max(targets):
        raise ValueError("right anchor must be above every target")
    if any(value < 2 for value in protocols):
        raise ValueError("Every state-averaged protocol must have at least two roots.")

    configure_thread_environment(args.threads)
    fci, gto, lib, mcscf, scf = import_pyscf()
    lib.num_threads(args.threads)
    modules = (fci, gto, mcscf, scf)
    config = RunConfig(
        basis=args.basis,
        ncas=args.ncas,
        nelecas=args.nelecas,
        freeze_core=args.freeze_core,
        threads=args.threads,
        memory_mb=args.memory_mb,
        hf_conv_tol=args.hf_conv_tol,
        mc_conv_tol=args.mc_conv_tol,
        mc_conv_tol_grad=args.mc_conv_tol_grad,
        fci_conv_tol=args.fci_conv_tol,
        mc_max_cycle_macro=args.mc_max_cycle_macro,
        mc_max_cycle_micro=args.mc_max_cycle_micro,
        fci_max_cycle=args.fci_max_cycle,
        verbose=args.verbose,
        newton_fallback=args.newton_fallback,
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    payload = configuration_payload(args, targets, irreps, protocols, sorted(sources))
    (args.workdir / "B0_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("BeH2 B0 stationary-solution reconnaissance", flush=True)
    print(json.dumps(payload, indent=2), flush=True)

    for irrep in irreps:
        for nroots in protocols:
            if "left" in sources:
                left_points = path_points(
                    args.left_anchor,
                    max(targets),
                    args.branch_step,
                    targets,
                )
                print(f"\nLEFT branch {irrep} SA{nroots}: {len(left_points)} points", flush=True)
                run_branch(
                    args.workdir,
                    left_points,
                    irrep,
                    nroots,
                    "left",
                    config,
                    modules,
                )
            if "right" in sources:
                right_points = path_points(
                    args.right_anchor,
                    min(targets),
                    args.branch_step,
                    targets,
                )
                print(f"\nRIGHT branch {irrep} SA{nroots}: {len(right_points)} points", flush=True)
                run_branch(
                    args.workdir,
                    right_points,
                    irrep,
                    nroots,
                    "right",
                    config,
                    modules,
                )

    run_independent_sources(
        args.workdir,
        targets,
        irreps,
        protocols,
        sources,
        config,
        modules,
    )
    collect_attempts(args.workdir)
    failures = list((args.workdir / "failures").glob("**/*.json"))
    print(f"\nCompleted solution files: {len(list((args.workdir / 'solutions').glob('**/*.npz')))}")
    print(f"Failure records         : {len(failures)}")
    print(f"Summary CSV             : {args.workdir / 'B0_attempts.csv'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
