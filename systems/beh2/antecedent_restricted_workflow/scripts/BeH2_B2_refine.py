#!/usr/bin/env python3
"""Refine the BeH2 A1 SA3 fold bracket and retain directional history.

The B1 map brackets a protocol-selective event between 2.52 and 2.53 bohr:
SA3 changes from one stationary solution to multiple solutions, while the SA4
upper branch remains smooth.  B2 resolves that interval on a 0.0005-bohr grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from BeH2_B0_stationary_solutions import (
    RunConfig,
    close_mol,
    collect_attempts,
    configure_thread_environment,
    import_pyscf,
    lambda_tag,
    load_mo,
    run_attempt,
)


LOWER = 2.5200
UPPER = 2.5300
STEP = 0.0005


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BeH2 B2 fine SA3-fold refinement.")
    p.add_argument("--b1-workdir", type=Path, default=Path("BeH2_B1_work"))
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B2_work"))
    p.add_argument("--lambda-min", type=float, default=LOWER)
    p.add_argument("--lambda-max", type=float, default=UPPER)
    p.add_argument("--step", type=float, default=STEP)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--memory-mb", type=int, default=100000)
    p.add_argument("--hf-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    p.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    p.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-max-cycle-macro", type=int, default=200)
    p.add_argument("--mc-max-cycle-micro", type=int, default=12)
    p.add_argument("--fci-max-cycle", type=int, default=240)
    p.add_argument("--verbose", type=int, default=4)
    p.add_argument(
        "--newton-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--self-test", action="store_true")
    return p


def fine_grid(lower: float, upper: float, step: float) -> tuple[float, ...]:
    if not lower < upper or step <= 0.0:
        raise ValueError("Require lambda_min < lambda_max and step > 0.")
    count_float = (upper - lower) / step
    count = int(round(count_float))
    if abs(count_float - count) > 1.0e-8:
        raise ValueError("The B2 interval must be an integer number of steps.")
    return tuple(round(lower + i * step, 10) for i in range(count + 1))


def b1_solution(
    b1_workdir: Path,
    protocol: int,
    source: str,
    lam: float,
) -> Path:
    return (
        b1_workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )


def validate_seed(path: Path, protocol: int, lam: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required B1 seed is missing: {path}")
    with np.load(path, allow_pickle=False) as z:
        if abs(float(z["lambda_bohr"]) - lam) > 1.0e-9:
            raise ValueError(f"Seed geometry mismatch: {path}")
        if str(np.asarray(z["irrep"]).reshape(()).item()) != "A1":
            raise ValueError(f"Seed irrep mismatch: {path}")
        if int(z["nroots"]) != protocol:
            raise ValueError(f"Seed protocol mismatch: {path}")


def run_sequence(
    workdir: Path,
    points: Sequence[float],
    protocol: int,
    source: str,
    initial_mo: np.ndarray,
    config: RunConfig,
    modules: tuple[Any, Any, Any, Any],
) -> bool:
    fci, gto, mcscf, scf = modules
    previous_mo = np.asarray(initial_mo)
    previous_mol = None
    ok = True
    for lam in points:
        try:
            mo, mol, _path = run_attempt(
                workdir,
                lam,
                "A1",
                protocol,
                source,
                config,
                fci,
                gto,
                mcscf,
                scf,
                previous_mo,
                previous_mol,
            )
        except Exception as exc:
            print(
                f"  STOP  A1 SA{protocol} {source} lambda={lam:.8f}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            ok = False
            break
        if previous_mol is not None:
            close_mol(previous_mol)
        previous_mo = np.asarray(mo)
        previous_mol = mol
    if previous_mol is not None:
        close_mol(previous_mol)
    return ok


def locked_model(b1_workdir: Path) -> dict[str, Any]:
    path = b1_workdir / "B1_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing locked B1 configuration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("basis", "")).lower() != "cc-pvtz":
        raise ValueError("B2 expects the locked cc-pVTZ model.")
    active = payload.get("active_space", {})
    if active.get("electrons") != 6 or active.get("orbitals") != 6:
        raise ValueError("B2 expects the locked CAS(6,6) model.")
    return payload


def self_test() -> int:
    grid = fine_grid(LOWER, UPPER, STEP)
    if len(grid) != 21 or grid[0] != LOWER or grid[-1] != UPPER:
        raise AssertionError("B2 fine-grid construction failed.")
    print("BeH2 B2 runner self-test: PASS")
    print(f"grid points: {len(grid)}")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    locked = locked_model(args.b1_workdir)
    grid = fine_grid(args.lambda_min, args.lambda_max, args.step)
    configure_thread_environment(args.threads)
    fci, gto, lib, mcscf, scf = import_pyscf()
    lib.num_threads(args.threads)
    modules = (fci, gto, mcscf, scf)
    config = RunConfig(
        basis=locked["basis"],
        ncas=int(locked["active_space"]["orbitals"]),
        nelecas=int(locked["active_space"]["electrons"]),
        freeze_core=bool(locked.get("freeze_core", True)),
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
    directions = [
        {
            "protocol": "SA3",
            "source": "upper_down",
            "seed": "B1 SA3 upper at 2.5300",
            "direction": "descending",
        },
        {
            "protocol": "SA3",
            "source": "lower_down",
            "seed": "B1 SA3 lower at 2.5300",
            "direction": "descending",
        },
        {
            "protocol": "SA3",
            "source": "unique_up",
            "seed": "B1 SA3 upper (unique solution) at 2.5200",
            "direction": "ascending",
        },
        {
            "protocol": "SA4",
            "source": "upper_down",
            "seed": "B1 SA4 upper at 2.5300",
            "direction": "descending",
        },
        {
            "protocol": "SA4",
            "source": "upper_up",
            "seed": "B1 SA4 upper at 2.5200",
            "direction": "ascending",
        },
    ]
    payload = {
        "gate": "BeH2 B2 fine SA3-fold and mixed-plaquette refinement",
        "parent_b1_workdir": str(args.b1_workdir.resolve()),
        "coordinate": locked["coordinate"],
        "basis": locked["basis"],
        "active_space": locked["active_space"],
        "freeze_core": locked.get("freeze_core", True),
        "irrep": "A1",
        "lambda_min_bohr": args.lambda_min,
        "lambda_max_bohr": args.lambda_max,
        "step_bohr": args.step,
        "grid_bohr": list(grid),
        "directions": directions,
        "threads": args.threads,
        "memory_mb": args.memory_mb,
    }
    (args.workdir / "B2_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("BeH2 B2 fine SA3-fold refinement", flush=True)
    print(json.dumps(payload, indent=2), flush=True)

    plans = [
        (3, "upper_down", "upper", args.lambda_max, tuple(reversed(grid))),
        (3, "lower_down", "lower", args.lambda_max, tuple(reversed(grid))),
        (3, "unique_up", "upper", args.lambda_min, grid),
        (4, "upper_down", "upper", args.lambda_max, tuple(reversed(grid))),
        (4, "upper_up", "upper", args.lambda_min, grid),
    ]
    status = True
    for protocol, source, b1_source, seed_lam, points in plans:
        path = b1_solution(args.b1_workdir, protocol, b1_source, seed_lam)
        validate_seed(path, protocol, seed_lam)
        print(
            f"\nA1 SA{protocol} {source}: {points[0]:.6f} -> {points[-1]:.6f} "
            f"({len(points)} points)",
            flush=True,
        )
        status = run_sequence(
            args.workdir,
            points,
            protocol,
            source,
            load_mo(path),
            config,
            modules,
        ) and status

    collect_attempts(args.workdir)
    old = args.workdir / "B0_attempts.csv"
    if old.exists():
        old.replace(args.workdir / "B2_attempts.csv")
    failures = list((args.workdir / "failures").glob("**/*.json"))
    solutions = list((args.workdir / "solutions").glob("**/*.npz"))
    print(f"\nCompleted solution files: {len(solutions)}")
    print(f"Failure records         : {len(failures)}")
    print(f"Inventory               : {args.workdir / 'B2_attempts.csv'}")
    return 0 if status and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
