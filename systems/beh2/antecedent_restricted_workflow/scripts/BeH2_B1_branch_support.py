#!/usr/bin/env python3
"""Map the support of the two A1 stationary branches found by BeH2 Gate B0.

This stage deliberately does not evaluate a mixed plaquette.  It propagates the
two independently established A1 stationary solutions (the B0 common-SA2 seed
and the B0 right-continuation seed) away from lambda=2.62425 bohr.  The output
therefore locates an energy crossing, a branch merger/spinodal, or the interval
over which both stationary solutions coexist before a plaquette window is
chosen.

The script imports the tested B0 runner placed beside this file.  It is
restartable: existing NPZ files below the B1 work directory are reused.
"""

from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_CENTER = 2.62425
DEFAULT_MIN = 2.40
DEFAULT_MAX = 2.70
DEFAULT_STEP = 0.01


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BeH2 B1 bidirectional support map for the two A1 branches."
    )
    p.add_argument("--b0-workdir", type=Path, default=Path("BeH2_B0_work"))
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B1_work"))
    p.add_argument("--center", type=float, default=DEFAULT_CENTER)
    p.add_argument("--lambda-min", type=float, default=DEFAULT_MIN)
    p.add_argument("--lambda-max", type=float, default=DEFAULT_MAX)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--memory-mb", type=int, default=100000)
    p.add_argument("--hf-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    p.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    p.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-max-cycle-macro", type=int, default=160)
    p.add_argument("--mc-max-cycle-micro", type=int, default=12)
    p.add_argument("--fci-max-cycle", type=int, default=200)
    p.add_argument("--verbose", type=int, default=4)
    p.add_argument(
        "--newton-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--self-test", action="store_true")
    return p


def regular_grid(lower: float, upper: float, step: float, center: float) -> tuple[float, ...]:
    if not lower < center < upper:
        raise ValueError("Require lambda_min < center < lambda_max.")
    if step <= 0.0:
        raise ValueError("step must be positive.")
    count = int(math.floor((upper - lower) / step + 1.0e-9))
    values = [lower + index * step for index in range(count + 1)]
    if values[-1] < upper - 0.25 * step:
        values.append(upper)
    values.extend([lower, center, upper])
    return tuple(sorted({round(value, 10) for value in values}))


def seed_path(
    b0_workdir: Path,
    protocol: int,
    source: str,
    center: float,
) -> Path:
    return (
        b0_workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(center)}.npz"
    )


def validate_seed(path: Path, expected_protocol: int, center: float) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required B0 seed is missing: {path}. Keep the B0 solutions directory."
        )
    with np.load(path, allow_pickle=False) as z:
        actual_lam = float(z["lambda_bohr"])
        actual_irrep = str(np.asarray(z["irrep"]).reshape(()).item())
        actual_nroots = int(z["nroots"])
        if abs(actual_lam - center) > 1.0e-9:
            raise ValueError(f"Seed geometry mismatch in {path}: {actual_lam}.")
        if actual_irrep != "A1" or actual_nroots != expected_protocol:
            raise ValueError(
                f"Seed identity mismatch in {path}: {actual_irrep} SA{actual_nroots}."
            )


def propagate(
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
    for index, lam in enumerate(points):
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
                f"  STOP  A1 SA{protocol} {source} at lambda={lam:.8f}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            ok = False
            break
        if previous_mol is not None:
            close_mol(previous_mol)
        previous_mo = np.asarray(mo)
        previous_mol = mol
        if index == 0 and abs(lam - points[0]) > 1.0e-12:
            raise AssertionError("Unexpected propagation start.")
    if previous_mol is not None:
        close_mol(previous_mol)
    return ok


def run_source_bidirectionally(
    workdir: Path,
    grid: Sequence[float],
    center: float,
    protocol: int,
    source: str,
    seed_mo: np.ndarray,
    config: RunConfig,
    modules: tuple[Any, Any, Any, Any],
) -> bool:
    lower = [value for value in grid if value < center]
    upper = [value for value in grid if value > center]
    down = [center, *sorted(lower, reverse=True)]
    up = [center, *sorted(upper)]
    print(
        f"\nA1 SA{protocol} {source}: center -> lower ({len(down)} points)",
        flush=True,
    )
    ok_down = propagate(
        workdir, down, protocol, source, seed_mo, config, modules
    )
    print(
        f"\nA1 SA{protocol} {source}: center -> upper ({len(up)} points)",
        flush=True,
    )
    ok_up = propagate(workdir, up, protocol, source, seed_mo, config, modules)
    return ok_down and ok_up


def locked_model(b0_workdir: Path) -> dict[str, Any]:
    path = b0_workdir / "B0_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing locked B0 configuration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("basis", "").lower() != "cc-pvtz":
        raise ValueError("B1 expects the locked B0 cc-pVTZ model.")
    active = payload.get("active_space", {})
    if active.get("electrons") != 6 or active.get("orbitals") != 6:
        raise ValueError("B1 expects the locked B0 CAS(6,6) model.")
    return payload


def self_test() -> int:
    grid = regular_grid(DEFAULT_MIN, DEFAULT_MAX, DEFAULT_STEP, DEFAULT_CENTER)
    if DEFAULT_CENTER not in grid or grid[0] != DEFAULT_MIN or grid[-1] != DEFAULT_MAX:
        raise AssertionError("B1 grid construction failed.")
    if len(grid) != 32:  # 31 regular hundredth-bohr points plus the B0 center
        raise AssertionError(f"Unexpected B1 grid size: {len(grid)}")
    print("BeH2 B1 runner self-test: PASS")
    print(f"grid points: {len(grid)}")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    locked = locked_model(args.b0_workdir)
    grid = regular_grid(args.lambda_min, args.lambda_max, args.step, args.center)
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
    payload = {
        "gate": "BeH2 B1 A1 stationary-branch support map",
        "parent_b0_workdir": str(args.b0_workdir.resolve()),
        "coordinate": locked["coordinate"],
        "basis": locked["basis"],
        "active_space": locked["active_space"],
        "freeze_core": locked.get("freeze_core", True),
        "irrep": "A1",
        "protocols": ["SA3", "SA4"],
        "branches": {
            "upper": "B0 common_sa2 stationary solution",
            "lower": "B0 right-continuation stationary solution",
        },
        "center_bohr": args.center,
        "lambda_min_bohr": args.lambda_min,
        "lambda_max_bohr": args.lambda_max,
        "step_bohr": args.step,
        "grid_bohr": list(grid),
        "threads": args.threads,
        "memory_mb": args.memory_mb,
    }
    (args.workdir / "B1_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("BeH2 B1 A1 stationary-branch support map", flush=True)
    print(json.dumps(payload, indent=2), flush=True)

    status = True
    for protocol in (3, 4):
        for branch, b0_source in (("upper", "common_sa2"), ("lower", "right")):
            path = seed_path(args.b0_workdir, protocol, b0_source, args.center)
            validate_seed(path, protocol, args.center)
            status = run_source_bidirectionally(
                args.workdir,
                grid,
                args.center,
                protocol,
                branch,
                load_mo(path),
                config,
                modules,
            ) and status

    collect_attempts(args.workdir)
    old_attempts = args.workdir / "B0_attempts.csv"
    if old_attempts.exists():
        old_attempts.replace(args.workdir / "B1_attempts.csv")
    failures = list((args.workdir / "failures").glob("**/*.json"))
    solution_count = len(list((args.workdir / "solutions").glob("**/*.npz")))
    print(f"\nCompleted solution files: {solution_count}")
    print(f"Failure records         : {len(failures)}")
    print(f"Inventory               : {args.workdir / 'B1_attempts.csv'}")
    return 0 if status and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
