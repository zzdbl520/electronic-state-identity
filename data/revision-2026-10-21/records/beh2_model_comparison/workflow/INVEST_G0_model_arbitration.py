#!/usr/bin/env python3
"""G0 model-semantics arbitration for the INVEST BeH2 JCTC revision.

This runner compares three precisely defined cc-pVTZ models on the same
projection-safe SA3/SA4 continuation window:

1. legacy_frozen_mo0: CAS(6e,6o), lowest MO excluded from orbital rotations;
2. all_electron_cas66: CAS(6e,6o), no frozen orbital;
3. valence_cas46_frozen_core: CAS(4e,6o), one doubly occupied core orbital
   excluded from orbital rotations.

The calculation refuses to call the legacy model "frozen core". A clean
model must reproduce both the SA3 branch event and the smooth SA4 control
before expensive refinement and weighted-SA3 stages are authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from BeH2_B0_analyze import Solution, load_solution, procrustes_rotation
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
from BeH2_B3R_analyze import state_overlap_matrix


TOTAL_ELECTRONS = 6
DEFAULT_LAM_MIN = 2.52170
DEFAULT_LAM_MAX = 2.52195
DEFAULT_STEP = 0.00005
UPPER_SEED = 2.52220
LOWER_SEED = 2.52120


@dataclass(frozen=True)
class Variant:
    name: str
    nelecas: int
    ncas: int
    freeze_lowest_mo: bool
    interpretation: str

    @property
    def ncore(self) -> int:
        remainder = TOTAL_ELECTRONS - self.nelecas
        if remainder < 0 or remainder % 2:
            raise ValueError(f"Invalid active-electron count for {self.name}.")
        return remainder // 2


VARIANTS = (
    Variant(
        "legacy_frozen_mo0",
        6,
        6,
        True,
        "CAS(6e,6o); MO 0 is active but excluded from orbital rotations",
    ),
    Variant(
        "all_electron_cas66",
        6,
        6,
        False,
        "unconstrained all-electron CAS(6e,6o)",
    ),
    Variant(
        "valence_cas46_frozen_core",
        4,
        6,
        True,
        "CAS(4e,6o) with one doubly occupied core MO excluded from rotations",
    ),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="INVEST BeH2 G0 model arbitration.")
    p.add_argument("--b3r-workdir", type=Path)
    p.add_argument("--workdir", type=Path)
    p.add_argument("--lambda-min", type=float, default=DEFAULT_LAM_MIN)
    p.add_argument("--lambda-max", type=float, default=DEFAULT_LAM_MAX)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
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


def constant_grid(lower: float, upper: float, step: float) -> tuple[float, ...]:
    if not lower < upper or step <= 0:
        raise ValueError("Require lambda_min < lambda_max and step > 0.")
    ratio = (upper - lower) / step
    count = int(round(ratio))
    if abs(ratio - count) > 1.0e-8:
        raise ValueError("Arbitration window must contain an integer number of steps.")
    return tuple(round(lower + i * step, 10) for i in range(count + 1))


def seed_path(b3r_workdir: Path, protocol: int, source: str, lam: float) -> Path:
    return (
        b3r_workdir
        / "solutions"
        / "A1"
        / f"SA{protocol}"
        / source
        / f"lambda_{lambda_tag(lam)}.npz"
    )


def scalar_text(archive: Any, key: str) -> str:
    return str(np.asarray(archive[key]).reshape(()).item())


def validate_seed(path: Path, protocol: int, source: str, lam: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exact B3R seed: {path}")
    with np.load(path, allow_pickle=False) as z:
        if abs(float(z["lambda_bohr"]) - lam) > 1.0e-9:
            raise ValueError(f"Seed geometry mismatch: {path}")
        if int(z["nroots"]) != protocol:
            raise ValueError(f"Seed root-count mismatch: {path}")
        if scalar_text(z, "source") != source:
            raise ValueError(f"Seed history mismatch: {path}")
        if scalar_text(z, "basis").lower() != "cc-pvtz":
            raise ValueError(f"Seed basis mismatch: {path}")
        if int(z["ncas"]) != 6:
            raise ValueError(f"Seed active-orbital mismatch: {path}")


def sequence(
    variant_dir: Path,
    points: Sequence[float],
    protocol: int,
    source: str,
    seed: Path,
    config: RunConfig,
    modules: tuple[Any, Any, Any, Any],
) -> bool:
    fci, gto, mcscf, scf = modules
    previous_mo = load_mo(seed)
    previous_mol = None
    ok = True
    for lam in points:
        try:
            mo, mol, _ = run_attempt(
                variant_dir,
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
                f"STOP {variant_dir.name} SA{protocol} {source} "
                f"lambda={lam:.8f}: {type(exc).__name__}: {exc}",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_assignment_margin(overlap: np.ndarray) -> tuple[str, float, float]:
    n = overlap.shape[0]
    if overlap.shape[1] != n:
        raise ValueError("Assignment margin requires a square overlap.")
    candidates = []
    for perm in itertools.permutations(range(n)):
        score = float(sum(abs(overlap[i, perm[i]]) for i in range(n)))
        candidates.append((score, perm))
    candidates.sort(reverse=True)
    best_score, best_perm = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    return (
        ";".join(str(i) for i in best_perm),
        best_score,
        (best_score - second_score) / n,
    )


def pair_row(
    x: Solution,
    y: Solution,
    fci: Any,
    gto: Any,
    basis: str,
) -> dict[str, Any]:
    if abs(x.lam - y.lam) > 1.0e-10:
        raise ValueError("Pair comparison requires one geometry.")
    if x.nroots != y.nroots:
        raise ValueError("Pair comparison requires one root count.")
    overlap, _ = state_overlap_matrix(
        x, y, range(x.nroots), range(y.nroots), fci, gto, basis
    )
    state_singular = np.linalg.svd(overlap, compute_uv=False)
    state_leakage = 1.0 - float(np.sum(state_singular**2)) / x.nroots
    active_overlap = (
        x.active_orbitals.conj().T @ x.ao_overlap @ y.active_orbitals
    )
    active_singular = np.linalg.svd(active_overlap, compute_uv=False)
    active_angle = float(
        np.max(np.arccos(np.clip(active_singular, 0.0, 1.0)))
    )
    rotation = procrustes_rotation(active_overlap)
    rdm_x = np.mean(x.rdm1_states, axis=0)
    rdm_y = np.mean(y.rdm1_states, axis=0)
    aligned_y = rotation.conj().T @ rdm_y @ rotation
    rdm_distance = float(np.linalg.norm(rdm_x - aligned_y))
    noon_x = np.sort(np.linalg.eigvalsh(rdm_x))[::-1]
    noon_y = np.sort(np.linalg.eigvalsh(rdm_y))[::-1]
    noon_rms = float(np.sqrt(np.mean((noon_x - noon_y) ** 2)))
    assignment, assignment_score, assignment_margin = normalized_assignment_margin(
        overlap
    )
    delta_e = abs(x.e_average - y.e_average)
    same = bool(
        delta_e <= 1.0e-7
        and active_angle <= 1.0e-3
        and state_leakage <= 1.0e-6
        and rdm_distance <= 1.0e-4
        and noon_rms <= 1.0e-4
    )
    return {
        "lambda_bohr": x.lam,
        "protocol": x.protocol,
        "source_x": x.source,
        "source_y": y.source,
        "delta_e_average_Eh": delta_e,
        "active_sigma_min": float(np.min(active_singular)),
        "active_angle_max_rad": active_angle,
        "state_sigma_min": float(np.min(state_singular)),
        "state_leakage": state_leakage,
        "rdm1_procrustes_fro": rdm_distance,
        "noon_rms": noon_rms,
        "maximum_overlap_assignment": assignment,
        "assignment_score": assignment_score,
        "normalized_assignment_margin": assignment_margin,
        "same_stationary_solution": same,
    }


def analyze_variant(
    variant: Variant,
    variant_dir: Path,
    grid: Sequence[float],
    fci: Any,
    gto: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparisons = (
        (3, "upper_down", "unique_up"),
        (4, "upper_down", "upper_up"),
    )
    rows: list[dict[str, Any]] = []
    for protocol, source_x, source_y in comparisons:
        for lam in grid:
            x_path = (
                variant_dir / "solutions" / "A1" / f"SA{protocol}" / source_x
                / f"lambda_{lambda_tag(lam)}.npz"
            )
            y_path = (
                variant_dir / "solutions" / "A1" / f"SA{protocol}" / source_y
                / f"lambda_{lambda_tag(lam)}.npz"
            )
            if not x_path.is_file() or not y_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete variant {variant.name} at lambda={lam:.8f}."
                )
            row = pair_row(
                load_solution(x_path),
                load_solution(y_path),
                fci,
                gto,
                "cc-pVTZ",
            )
            row["variant"] = variant.name
            rows.append(row)

    sa3 = sorted(
        (row for row in rows if row["protocol"] == "SA3"),
        key=lambda row: row["lambda_bohr"],
    )
    sa4 = sorted(
        (row for row in rows if row["protocol"] == "SA4"),
        key=lambda row: row["lambda_bohr"],
    )
    brackets = []
    for low, high in zip(sa3[:-1], sa3[1:]):
        if low["same_stationary_solution"] and not high["same_stationary_solution"]:
            brackets.append([low["lambda_bohr"], high["lambda_bohr"]])
    high = sa3[-1]
    low = sa3[0]
    sa3_distinct_high = bool(
        high["delta_e_average_Eh"] >= 1.0e-3
        and high["state_leakage"] >= 1.0e-2
    )
    sa3_same_low = bool(low["same_stationary_solution"])
    sa4_control = bool(all(row["same_stationary_solution"] for row in sa4))
    passed = bool(sa3_distinct_high and sa3_same_low and sa4_control and brackets)
    finite_gradients = []
    for path in variant_dir.glob("solutions/A1/SA*/*/lambda_*.npz"):
        value = load_solution(path).norm_gorb
        if math.isfinite(value):
            finite_gradients.append(value)
    summary = {
        "variant": asdict(variant),
        "expected_ncore": variant.ncore,
        "status": "PASS_EVENT_AND_CONTROL" if passed else "FAIL_MODEL_GATE",
        "passed": passed,
        "SA3_distinct_at_high_end": {
            "passed": sa3_distinct_high,
            "lambda_bohr": high["lambda_bohr"],
            "delta_e_average_Eh": high["delta_e_average_Eh"],
            "state_leakage": high["state_leakage"],
        },
        "SA3_same_at_low_end": {
            "passed": sa3_same_low,
            "lambda_bohr": low["lambda_bohr"],
            "delta_e_average_Eh": low["delta_e_average_Eh"],
            "state_leakage": low["state_leakage"],
        },
        "SA3_event_brackets_bohr": brackets,
        "SA4_history_control_all_same": sa4_control,
        "minimum_assignment_margin": min(
            row["normalized_assignment_margin"] for row in rows
        ),
        "maximum_reported_orbital_gradient": (
            max(finite_gradients) if finite_gradients else None
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "lambda_bohr",
        "protocol",
        "source_x",
        "source_y",
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
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> int:
    grid = constant_grid(DEFAULT_LAM_MIN, DEFAULT_LAM_MAX, DEFAULT_STEP)
    if len(grid) != 6 or grid[0] != DEFAULT_LAM_MIN or grid[-1] != DEFAULT_LAM_MAX:
        raise AssertionError("Grid construction failed.")
    expected_ncore = {
        "legacy_frozen_mo0": 0,
        "all_electron_cas66": 0,
        "valence_cas46_frozen_core": 1,
    }
    for variant in VARIANTS:
        if variant.ncore != expected_ncore[variant.name]:
            raise AssertionError(f"ncore audit failed for {variant.name}.")
    test_overlap = np.diag([0.99, 0.98, 0.97])
    assignment, _score, margin = normalized_assignment_margin(test_overlap)
    # For a 3x3 identity-like matrix, a competing permutation can retain one
    # fixed point, so this normalization has a theoretical maximum of 2/3.
    if assignment != "0;1;2" or margin <= 0.6:
        raise AssertionError("Assignment test failed.")
    print("INVEST BeH2 G0 model-arbitration self-test: PASS")
    print(json.dumps({item.name: item.ncore for item in VARIANTS}, indent=2))
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    if args.b3r_workdir is None or args.workdir is None:
        raise ValueError("Production mode requires --b3r-workdir and --workdir.")

    grid = constant_grid(args.lambda_min, args.lambda_max, args.step)
    if args.lambda_max >= UPPER_SEED or args.lambda_min <= LOWER_SEED:
        raise ValueError("The analysis window must lie strictly between the seeds.")

    config_path = args.b3r_workdir / "B3R_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing B3R configuration: {config_path}")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(source_config.get("basis", "")).lower() != "cc-pvtz":
        raise ValueError("G0 requires the production cc-pVTZ archive.")
    active = source_config.get("active_space", {})
    if active != {"electrons": 6, "orbitals": 6}:
        raise ValueError("G0 requires the production CAS(6,6) seed archive.")

    seed_specs = (
        (3, "upper_down", UPPER_SEED),
        (3, "unique_up", LOWER_SEED),
        (4, "upper_down", UPPER_SEED),
        (4, "upper_up", LOWER_SEED),
    )
    seeds = {}
    for protocol, source, lam in seed_specs:
        path = seed_path(args.b3r_workdir, protocol, source, lam)
        validate_seed(path, protocol, source, lam)
        seeds[(protocol, source)] = path

    configure_thread_environment(args.threads)
    fci, gto, lib, mcscf, scf = import_pyscf()
    lib.num_threads(args.threads)
    modules = (fci, gto, mcscf, scf)

    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gate": "G0 model-semantics arbitration",
        "basis": "cc-pVTZ",
        "coordinate": source_config["coordinate"],
        "source_b3r_workdir": str(args.b3r_workdir.resolve()),
        "analysis_grid_bohr": list(grid),
        "variants": [asdict(item) | {"ncore": item.ncore} for item in VARIANTS],
        "semantic_warning": (
            "PySCF mc.frozen excludes orbitals from CASSCF orbital rotations; "
            "it does not impose double occupation in the FCI space."
        ),
        "source_config_sha256": sha256(config_path),
        "seed_sha256": {
            f"SA{protocol}/{source}": sha256(path)
            for (protocol, source), path in seeds.items()
        },
    }
    locked = args.workdir / "G0_manifest.json"
    if locked.exists():
        previous = json.loads(locked.read_text(encoding="utf-8"))
        stable_keys = (
            "basis",
            "coordinate",
            "source_b3r_workdir",
            "analysis_grid_bohr",
            "variants",
            "source_config_sha256",
            "seed_sha256",
        )
        if any(previous.get(key) != manifest.get(key) for key in stable_keys):
            raise RuntimeError("Existing G0 workdir has a different model lock.")
    locked.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_ok = True
    for variant in VARIANTS:
        variant_dir = args.workdir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        config = RunConfig(
            basis="cc-pVTZ",
            ncas=variant.ncas,
            nelecas=variant.nelecas,
            freeze_core=variant.freeze_lowest_mo,
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
        variant_payload = asdict(variant) | {
            "ncore": variant.ncore,
            "basis": "cc-pVTZ",
            "analysis_grid_bohr": list(grid),
            "note": (
                "The inherited RunConfig field freeze_core maps to PySCF mc.frozen=1 "
                "and is interpreted here only through freeze_lowest_mo."
            ),
        }
        variant_config_path = variant_dir / "variant_config.json"
        if variant_config_path.exists():
            previous_variant = json.loads(
                variant_config_path.read_text(encoding="utf-8")
            )
            if previous_variant != variant_payload:
                raise RuntimeError(
                    f"Existing {variant.name} workdir has a different model lock."
                )
        else:
            variant_config_path.write_text(
                json.dumps(variant_payload, indent=2), encoding="utf-8"
            )

        plans = (
            (3, "upper_down", [UPPER_SEED, *sorted(grid, reverse=True)]),
            (3, "unique_up", [LOWER_SEED, *sorted(grid)]),
            (4, "upper_down", [UPPER_SEED, *sorted(grid, reverse=True)]),
            (4, "upper_up", [LOWER_SEED, *sorted(grid)]),
        )
        print(f"\n===== {variant.name} =====", flush=True)
        print(variant.interpretation, flush=True)
        for protocol, source, points in plans:
            all_ok = sequence(
                variant_dir,
                points,
                protocol,
                source,
                seeds[(protocol, source)],
                config,
                modules,
            ) and all_ok
        collect_attempts(variant_dir)
        attempts = variant_dir / "B0_attempts.csv"
        if attempts.exists():
            attempts.replace(variant_dir / "G0_attempts.csv")

    if not all_ok:
        print("G0 calculation failures occurred; analysis not authorized.", file=sys.stderr)
        return 2

    all_rows = []
    summaries = []
    for variant in VARIANTS:
        rows, summary = analyze_variant(
            variant, args.workdir / variant.name, grid, fci, gto
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"{variant.name}: {summary['status']}", flush=True)

    write_csv(args.workdir / "G0_model_arbitration.csv", all_rows)
    clean_passes = [
        item["variant"]["name"]
        for item in summaries
        if item["passed"] and item["variant"]["name"] != "legacy_frozen_mo0"
    ]
    legacy_pass = next(
        item["passed"]
        for item in summaries
        if item["variant"]["name"] == "legacy_frozen_mo0"
    )
    if "all_electron_cas66" in clean_passes:
        recommendation = "all_electron_cas66"
        status = "PASS_CLEAN_MODEL"
    elif "valence_cas46_frozen_core" in clean_passes:
        recommendation = "valence_cas46_frozen_core"
        status = "PASS_CLEAN_MODEL"
    elif legacy_pass:
        recommendation = None
        status = "STOP_LEGACY_CONSTRAINT_ONLY"
    else:
        recommendation = None
        status = "STOP_EVENT_NOT_REPRODUCED"

    overall = {
        "gate": "G0 model-semantics arbitration",
        "status": status,
        "recommended_model_for_G1": recommendation,
        "variants": summaries,
        "decision_rule": (
            "Proceed to G1 only when an unconstrained all-electron CAS(6,6) or "
            "a correctly defined valence CAS(4,6) reproduces the SA3 event while "
            "the SA4 history control remains smooth."
        ),
    }
    (args.workdir / "G0_decision.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2), flush=True)
    return 0 if status == "PASS_CLEAN_MODEL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
