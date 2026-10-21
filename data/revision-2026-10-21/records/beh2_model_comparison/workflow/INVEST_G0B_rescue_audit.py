#!/usr/bin/env python3
"""G0B closure audit for the INVEST BeH2 JCTC revision.

This low-cost audit closes two loopholes left by G0:
1. test a standard valence CAS(4e,6o) with one inactive doubly occupied core
   orbital but no artificially frozen CASSCF orbital-rotation variable;
2. use deterministic, symmetry-preserving multistart searches at the high-side
   geometry for both clean models.

No expensive spinodal refinement is authorized by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from BeH2_B0_analyze import load_solution
from BeH2_B0_stationary_solutions import (
    RunConfig,
    close_mol,
    collect_attempts,
    configure_thread_environment,
    import_pyscf,
    lambda_tag,
    load_mo,
    make_mol,
    run_attempt,
)
from INVEST_G0_model_arbitration import (
    DEFAULT_LAM_MAX,
    DEFAULT_LAM_MIN,
    DEFAULT_STEP,
    LOWER_SEED,
    TOTAL_ELECTRONS,
    UPPER_SEED,
    Variant,
    analyze_variant,
    constant_grid,
    pair_row,
    seed_path,
    sequence,
    validate_seed,
    write_csv,
)


HIGH_TEST_LAMBDA = DEFAULT_LAM_MAX
RANDOM_STARTS_PER_BASE = 4
RANDOM_AMPLITUDES = (0.02, 0.05)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="INVEST BeH2 G0B closure audit.")
    p.add_argument("--b3r-workdir", type=Path)
    p.add_argument("--g0-workdir", type=Path)
    p.add_argument("--workdir", type=Path)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--memory-mb", type=int, default=100000)
    p.add_argument("--random-seed", type=int, default=20260915)
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


def run_config(args: argparse.Namespace, nelecas: int) -> RunConfig:
    return RunConfig(
        basis="cc-pVTZ",
        ncas=6,
        nelecas=nelecas,
        freeze_core=False,
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


def solution_file(
    workdir: Path,
    protocol: int,
    source: str,
    lam: float,
) -> Path:
    return (
        workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )


def allowed_rotation_pairs(
    mo: np.ndarray,
    mol: Any,
    ncore: int,
    ncas: int,
    symm: Any,
) -> list[tuple[int, int]]:
    overlap = mol.intor_symmetric("int1e_ovlp")
    labels = np.asarray(
        symm.label_orb_symm(
            mol,
            mol.irrep_name,
            mol.symm_orb,
            mo,
            s=overlap,
            check=False,
        )
    )
    nmo = mo.shape[1]
    nocc = ncore + ncas
    pairs: list[tuple[int, int]] = []
    for i in range(ncore):
        for j in range(ncore, nocc):
            if labels[i] == labels[j]:
                pairs.append((i, j))
    for i in range(ncore, nocc):
        for j in range(nocc, nmo):
            if labels[i] == labels[j]:
                pairs.append((i, j))
    if not pairs:
        raise RuntimeError("No symmetry-allowed interspace orbital rotations found.")
    return pairs


def rotate_mo(
    base_mo: np.ndarray,
    pairs: list[tuple[int, int]],
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    mo = np.array(base_mo, copy=True)
    count = min(4, len(pairs))
    chosen = rng.choice(len(pairs), size=count, replace=False)
    for index in np.asarray(chosen).reshape(-1):
        i, j = pairs[int(index)]
        theta = float(rng.uniform(-amplitude, amplitude))
        c = math.cos(theta)
        s = math.sin(theta)
        col_i = np.array(mo[:, i], copy=True)
        col_j = np.array(mo[:, j], copy=True)
        mo[:, i] = c * col_i + s * col_j
        mo[:, j] = -s * col_i + c * col_j
    return mo


def run_valence_scan(
    args: argparse.Namespace,
    workdir: Path,
    seeds: dict[tuple[int, str], Path],
    modules: tuple[Any, Any, Any, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant = Variant(
        "valence_cas46_optimized_core",
        4,
        6,
        False,
        "standard CAS(4e,6o) with one optimized inactive doubly occupied core",
    )
    grid = constant_grid(DEFAULT_LAM_MIN, DEFAULT_LAM_MAX, DEFAULT_STEP)
    workdir.mkdir(parents=True, exist_ok=True)
    config = run_config(args, nelecas=4)
    payload = asdict(variant) | {
        "ncore": (TOTAL_ELECTRONS - 4) // 2,
        "basis": "cc-pVTZ",
        "analysis_grid_bohr": list(grid),
        "semantic_note": (
            "The inactive core is doubly occupied by the CAS electron count; "
            "no MO is removed from orbital optimization."
        ),
    }
    config_path = workdir / "variant_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Existing G0B valence scan has a different model lock.")
    else:
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plans = (
        (3, "upper_down", [UPPER_SEED, *sorted(grid, reverse=True)]),
        (3, "unique_up", [LOWER_SEED, *sorted(grid)]),
        (4, "upper_down", [UPPER_SEED, *sorted(grid, reverse=True)]),
        (4, "upper_up", [LOWER_SEED, *sorted(grid)]),
    )
    ok = True
    for protocol, source, points in plans:
        ok = sequence(
            workdir,
            points,
            protocol,
            source,
            seeds[(protocol, source)],
            config,
            modules,
        ) and ok
    collect_attempts(workdir)
    attempts = workdir / "B0_attempts.csv"
    if attempts.exists():
        attempts.replace(workdir / "G0B_attempts.csv")
    if not ok:
        raise RuntimeError("The standard valence CAS(4,6) scan did not converge.")
    fci, gto, _mcscf, _scf = modules
    return analyze_variant(variant, workdir, grid, fci, gto)


def multistart(
    args: argparse.Namespace,
    label: str,
    nelecas: int,
    base_paths: dict[str, Path],
    workdir: Path,
    modules: tuple[Any, Any, Any, Any],
    symm: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fci, gto, mcscf, scf = modules
    config = run_config(args, nelecas=nelecas)
    ncore = (TOTAL_ELECTRONS - nelecas) // 2
    rng = np.random.default_rng(args.random_seed + nelecas)
    workdir.mkdir(parents=True, exist_ok=True)

    base_mos = {name: load_mo(path) for name, path in base_paths.items()}
    probe_mol = make_mol(
        HIGH_TEST_LAMBDA,
        config,
        workdir / "symmetry_probe.log",
        gto,
    )
    try:
        pairs_by_base = {
            name: allowed_rotation_pairs(mo, probe_mol, ncore, 6, symm)
            for name, mo in base_mos.items()
        }
    finally:
        close_mol(probe_mol)

    starts: list[tuple[str, np.ndarray, str]] = []
    for name, mo in base_mos.items():
        starts.append((f"from_{name}", mo, name))
    for base_name in ("legacy_upper", "clean_upper"):
        base_mo = base_mos[base_name]
        pairs = pairs_by_base[base_name]
        for index in range(RANDOM_STARTS_PER_BASE):
            amplitude = RANDOM_AMPLITUDES[index % len(RANDOM_AMPLITUDES)]
            starts.append(
                (
                    f"random_{base_name}_{index:02d}_a{amplitude:.2f}".replace(".", "p"),
                    rotate_mo(base_mo, pairs, rng, amplitude),
                    base_name,
                )
            )

    paths: list[Path] = []
    start_records = []
    for source, mo_guess, parent in starts:
        mo, mol, path = run_attempt(
            workdir,
            HIGH_TEST_LAMBDA,
            "A1",
            3,
            source,
            config,
            fci,
            gto,
            mcscf,
            scf,
            mo_guess,
            None,
        )
        close_mol(mol)
        paths.append(path)
        start_records.append(
            {
                "source": source,
                "parent_seed": parent,
                "solution_file": str(path.resolve()),
            }
        )

    reference_path = next(
        path for path in paths if path.parent.name == "from_clean_unique"
    )
    reference = load_solution(reference_path)
    rows = []
    for path in paths:
        candidate = load_solution(path)
        row = pair_row(reference, candidate, fci, gto, "cc-pVTZ")
        row["model"] = label
        row["candidate"] = candidate.source
        rows.append(row)

    distinct = [
        row for row in rows if not row["same_stationary_solution"]
    ]
    energies = [load_solution(path).e_average for path in paths]
    summary = {
        "model": label,
        "nelecas": nelecas,
        "ncas": 6,
        "ncore": ncore,
        "frozen_casscf_orbitals": 0,
        "lambda_bohr": HIGH_TEST_LAMBDA,
        "number_of_starts": len(paths),
        "number_of_distinct_solutions_vs_reference": len(distinct),
        "all_starts_same_solution": not distinct,
        "energy_spread_Eh": max(energies) - min(energies),
        "minimum_state_sigma": min(row["state_sigma_min"] for row in rows),
        "maximum_state_leakage": max(row["state_leakage"] for row in rows),
        "minimum_assignment_margin": min(
            row["normalized_assignment_margin"] for row in rows
        ),
        "starts": start_records,
    }
    return rows, summary


def write_multistart_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "candidate",
        "lambda_bohr",
        "delta_e_average_Eh",
        "active_sigma_min",
        "active_angle_max_rad",
        "state_sigma_min",
        "state_leakage",
        "rdm1_procrustes_fro",
        "noon_rms",
        "maximum_overlap_assignment",
        "assignment_score",
        "normalized_assignment_margin",
        "same_stationary_solution",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> int:
    variant = Variant(
        "valence_cas46_optimized_core",
        4,
        6,
        False,
        "test",
    )
    if variant.ncore != 1:
        raise AssertionError("Standard valence model must have ncore=1.")
    grid = constant_grid(DEFAULT_LAM_MIN, DEFAULT_LAM_MAX, DEFAULT_STEP)
    if len(grid) != 6:
        raise AssertionError("Unexpected G0B grid.")
    matrix = np.eye(10)
    pairs = [(0, 1), (2, 8), (3, 9)]
    rng = np.random.default_rng(7)
    rotated = rotate_mo(matrix, pairs, rng, 0.05)
    if np.linalg.norm(rotated.T @ rotated - np.eye(10)) > 1.0e-12:
        raise AssertionError("Givens rotations lost orthonormality.")
    print("INVEST BeH2 G0B closure-audit self-test: PASS")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    if args.b3r_workdir is None or args.g0_workdir is None or args.workdir is None:
        raise ValueError(
            "Production mode requires --b3r-workdir, --g0-workdir, and --workdir."
        )

    decision_path = args.g0_workdir / "G0_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(f"Missing completed G0 decision: {decision_path}")
    g0 = json.loads(decision_path.read_text(encoding="utf-8"))
    if g0.get("status") != "STOP_LEGACY_CONSTRAINT_ONLY":
        raise RuntimeError(
            "G0B is authorized only after STOP_LEGACY_CONSTRAINT_ONLY."
        )

    seed_specs = (
        (3, "upper_down", UPPER_SEED),
        (3, "unique_up", LOWER_SEED),
        (4, "upper_down", UPPER_SEED),
        (4, "upper_up", LOWER_SEED),
    )
    seeds: dict[tuple[int, str], Path] = {}
    for protocol, source, lam in seed_specs:
        path = seed_path(args.b3r_workdir, protocol, source, lam)
        validate_seed(path, protocol, source, lam)
        seeds[(protocol, source)] = path

    configure_thread_environment(args.threads)
    fci, gto, lib, mcscf, scf = import_pyscf()
    from pyscf import symm

    lib.num_threads(args.threads)
    modules = (fci, gto, mcscf, scf)
    args.workdir.mkdir(parents=True, exist_ok=True)

    valence_dir = args.workdir / "valence_cas46_optimized_core"
    valence_rows, valence_summary = run_valence_scan(
        args, valence_dir, seeds, modules
    )
    write_csv(args.workdir / "G0B_valence_scan.csv", valence_rows)

    legacy_dir = args.g0_workdir / "legacy_frozen_mo0"
    all_e_dir = args.g0_workdir / "all_electron_cas66"
    base_all_e = {
        "legacy_upper": solution_file(
            legacy_dir, 3, "upper_down", HIGH_TEST_LAMBDA
        ),
        "legacy_unique": solution_file(
            legacy_dir, 3, "unique_up", HIGH_TEST_LAMBDA
        ),
        "clean_upper": solution_file(
            all_e_dir, 3, "upper_down", HIGH_TEST_LAMBDA
        ),
        "clean_unique": solution_file(
            all_e_dir, 3, "unique_up", HIGH_TEST_LAMBDA
        ),
    }
    base_valence = {
        "legacy_upper": base_all_e["legacy_upper"],
        "legacy_unique": base_all_e["legacy_unique"],
        "clean_upper": solution_file(
            valence_dir, 3, "upper_down", HIGH_TEST_LAMBDA
        ),
        "clean_unique": solution_file(
            valence_dir, 3, "unique_up", HIGH_TEST_LAMBDA
        ),
    }
    for path in [*base_all_e.values(), *base_valence.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing multistart seed: {path}")

    all_e_rows, all_e_summary = multistart(
        args,
        "all_electron_cas66",
        6,
        base_all_e,
        args.workdir / "multistart_all_electron_cas66",
        modules,
        symm,
    )
    valence_multi_rows, valence_multi_summary = multistart(
        args,
        "valence_cas46_optimized_core",
        4,
        base_valence,
        args.workdir / "multistart_valence_cas46_optimized_core",
        modules,
        symm,
    )

    all_rows = all_e_rows + valence_multi_rows
    write_multistart_csv(args.workdir / "G0B_multistart.csv", all_rows)

    clean_branch_found = bool(
        valence_summary["passed"]
        or not all_e_summary["all_starts_same_solution"]
        or not valence_multi_summary["all_starts_same_solution"]
    )
    status = (
        "CLEAN_BRANCH_FOUND_NEEDS_MAPPING"
        if clean_branch_found
        else "CONFIRMED_CONSTRAINT_INDUCED"
    )
    decision = {
        "gate": "G0B model-semantics closure",
        "status": status,
        "old_spinodal_refinement_authorized": False,
        "legacy_BeH2_interpretation": (
            "constrained-variational forensic case"
            if status == "CONFIRMED_CONSTRAINT_INDUCED"
            else "undecided until the clean branch is mapped"
        ),
        "standard_valence_scan": valence_summary,
        "multistart": [all_e_summary, valence_multi_summary],
        "decision_rule": (
            "If no clean model produces a second stationary solution after "
            "symmetry-preserving multistart, treat the legacy event as induced "
            "by freezing active MO 0; do not call it a physical spinodal."
        ),
    }
    (args.workdir / "G0B_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (args.workdir / "G0B_valence_scan_summary.json").write_text(
        json.dumps(valence_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return 0 if status == "CONFIRMED_CONSTRAINT_INDUCED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
