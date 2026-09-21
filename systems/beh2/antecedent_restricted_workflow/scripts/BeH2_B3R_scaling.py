#!/usr/bin/env python3
"""Resolve the BeH2 SA3 spinodal with projection-safe directional anchors.

B2 located the history-selective SA3 collapse between 2.5215 and 2.5220 bohr.
B3R retains only the two histories needed for the positive control and resolves
the event on a 0.00005-bohr grid. Each B2 seed is copied into the new work
directory as an exact-geometry anchor. Continuation begins by reusing that
anchor, so every subsequent geometry receives an explicitly projected orbital
guess. This avoids the unprojected first step that caused B3 to lose the
metastable upper branch before entering the analysis window.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


LOWER = 2.52120
UPPER = 2.52220
STEP = 0.00005


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BeH2 B3R projection-safe scaling grid.")
    p.add_argument("--b2-workdir", type=Path, default=Path("BeH2_B2_work"))
    p.add_argument("--workdir", type=Path, default=Path("BeH2_B3R_work"))
    p.add_argument("--lambda-min", type=float, default=LOWER)
    p.add_argument("--lambda-max", type=float, default=UPPER)
    p.add_argument("--step", type=float, default=STEP)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--memory-mb", type=int, default=100000)
    p.add_argument("--hf-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    p.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    p.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    p.add_argument("--mc-max-cycle-macro", type=int, default=300)
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
        raise ValueError("The B3 interval must be an integer number of steps.")
    return tuple(round(lower + i * step, 10) for i in range(count + 1))


def directed_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Return an inclusive constant-step bridge from an exact seed geometry."""
    if step <= 0.0 or start == stop:
        raise ValueError("Require positive step and distinct bridge endpoints.")
    signed_step = step if stop > start else -step
    count_float = (stop - start) / signed_step
    count = int(round(count_float))
    if abs(count_float - count) > 1.0e-8:
        raise ValueError("Seed-to-window bridge must be an integer number of steps.")
    return tuple(round(start + i * signed_step, 10) for i in range(count + 1))


def b2_solution(
    b2_workdir: Path, protocol: int, source: str, lam: float
) -> Path:
    return (
        b2_workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )


def validate_seed(path: Path, protocol: int, source: str, lam: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required B2 seed is missing: {path}")
    with np.load(path, allow_pickle=False) as z:
        if abs(float(z["lambda_bohr"]) - lam) > 1.0e-9:
            raise ValueError(f"Seed geometry mismatch: {path}")
        if str(np.asarray(z["irrep"]).reshape(()).item()) != "A1":
            raise ValueError(f"Seed irrep mismatch: {path}")
        if int(z["nroots"]) != protocol:
            raise ValueError(f"Seed protocol mismatch: {path}")
        if str(np.asarray(z["source"]).reshape(()).item()) != source:
            raise ValueError(f"Seed history mismatch: {path}")


def install_anchor(
    seed: Path,
    workdir: Path,
    protocol: int,
    source: str,
    lam: float,
) -> Path:
    """Copy an exact B2 solution so the first B3R step has a seed molecule."""
    target = (
        workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(seed, target)
    validate_seed(target, protocol, source, lam)
    return target


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


def locked_model(b2_workdir: Path) -> dict[str, Any]:
    path = b2_workdir / "B2_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing locked B2 configuration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("basis", "")).lower() != "cc-pvtz":
        raise ValueError("B3 expects the locked cc-pVTZ model.")
    active = payload.get("active_space", {})
    if active.get("electrons") != 6 or active.get("orbitals") != 6:
        raise ValueError("B3 expects the locked CAS(6,6) model.")
    return payload


def self_test() -> int:
    grid = fine_grid(LOWER, UPPER, STEP)
    if len(grid) != 21 or grid[0] != LOWER or grid[-1] != UPPER:
        raise AssertionError("B3R fine-grid construction failed.")
    down = directed_grid(2.5225, LOWER, STEP)
    up = directed_grid(2.5210, UPPER, STEP)
    if len(down) != 27 or len(up) != 25:
        raise AssertionError("B3R projection bridge construction failed.")
    print("BeH2 B3R runner self-test: PASS")
    print(f"grid points: {len(grid)}")
    print(f"descending bridge points: {len(down)}")
    print(f"ascending bridge points: {len(up)}")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    locked = locked_model(args.b2_workdir)
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
    plans = [
        # Each sequence starts at the exact B2 seed. run_attempt first reuses
        # that archive and builds its molecule; all later guesses are projected.
        (3, "upper_down", 2.5225, directed_grid(2.5225, args.lambda_min, args.step)),
        (3, "unique_up", 2.5210, directed_grid(2.5210, args.lambda_max, args.step)),
        (4, "upper_down", 2.5225, directed_grid(2.5225, args.lambda_min, args.step)),
        (4, "upper_up", 2.5210, directed_grid(2.5210, args.lambda_max, args.step)),
    ]
    directions = [
        {
            "protocol": f"SA{protocol}",
            "source": source,
            "seed_bohr": seed_lam,
            "direction": "descending" if points[0] > points[-1] else "ascending",
        }
        for protocol, source, seed_lam, points in plans
    ]
    payload = {
        "gate": "BeH2 B3R projection-safe spinodal localization and scaling",
        "parent_b2_workdir": str(args.b2_workdir.resolve()),
        "coordinate": locked["coordinate"],
        "basis": locked["basis"],
        "active_space": locked["active_space"],
        "freeze_core": locked.get("freeze_core", True),
        "irrep": "A1",
        "lambda_min_bohr": args.lambda_min,
        "lambda_max_bohr": args.lambda_max,
        "step_bohr": args.step,
        "grid_bohr": list(grid),
        "analysis_grid_only": True,
        "directions": directions,
        "threads": args.threads,
        "memory_mb": args.memory_mb,
    }
    (args.workdir / "B3R_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("BeH2 B3R projection-safe spinodal scaling grid", flush=True)
    print(json.dumps(payload, indent=2), flush=True)

    status = True
    for protocol, source, seed_lam, points in plans:
        seed = b2_solution(args.b2_workdir, protocol, source, seed_lam)
        validate_seed(seed, protocol, source, seed_lam)
        anchor = install_anchor(seed, args.workdir, protocol, source, seed_lam)
        print(
            f"\nA1 SA{protocol} {source}: {points[0]:.8f} -> "
            f"{points[-1]:.8f} ({len(points)} points)",
            flush=True,
        )
        status = run_sequence(
            args.workdir,
            points,
            protocol,
            source,
            load_mo(anchor),
            config,
            modules,
        ) and status

    collect_attempts(args.workdir)
    old = args.workdir / "B0_attempts.csv"
    if old.exists():
        old.replace(args.workdir / "B3R_attempts.csv")
    failures = list((args.workdir / "failures").glob("**/*.json"))
    solutions = list((args.workdir / "solutions").glob("**/*.npz"))
    print(f"\nCompleted solution files: {len(solutions)}")
    print(f"Failure records         : {len(failures)}")
    print(f"Inventory               : {args.workdir / 'B3R_attempts.csv'}")
    return 0 if status and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
