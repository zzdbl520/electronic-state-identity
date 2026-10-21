#!/usr/bin/env python3
r"""H3 genuine-CI ring validation of mixed nuclear--variational curvature.

This is the next calculation after ``H3_mixed_curvature_gate0.py``.  It uses
the *same* production many-electron overlap and independent common-Fock-space
check, but evaluates a closed distortion ring around equilateral H3.

Default pilot
-------------

* 36 nuclear geometries: phi = 0, 10, ..., 350 degrees;
* equal-weight SA2-CASSCF(3,3) and SA3-CASSCF(3,3) at every geometry;
* the lowest two states retained from both protocols (72 CASSCF calculations);
* 36 mixed nuclear--protocol plaquettes, including the closing 350 -> 0 edge.

Place this file beside ``H3_mixed_curvature_gate0.py`` and run

    python H3_mixed_curvature_ring.py \
        --threads 16 --memory-mb 32000 \
        --workdir H3_ring_36 | tee H3_ring_36.stdout

The calculation is restartable: completed corner ``.npz`` files are loaded by
default.  Use ``--no-resume`` only when every CASSCF calculation must be redone.
For a nested refinement, reuse the old geometries instead of recalculating them:

    python H3_mixed_curvature_ring.py \
        --points 72 --seed-workdir H3_ring_36 --workdir H3_ring_72 \
        --threads 16 --memory-mb 32000

Principal outputs
-----------------

    H3_ring_36/ring_summary.json
    H3_ring_36/ring_profile.csv
    H3_ring_36/ring_matrices.npz
    H3_ring_36/ring_validation.png       (if matplotlib is available)
    H3_ring_36/corners/*.npz
    H3_ring_36/logs/*.log

The pilot is not, by itself, a step-size convergence study.  If it is clean,
repeat with ``--points 72`` and ``--points 144`` in separate work directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import H3_mixed_curvature_gate0 as core
except ImportError as exc:
    raise RuntimeError(
        "Place H3_mixed_curvature_gate0.py in the same directory as this input."
    ) from exc


Array = np.ndarray


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed H3 ring validation of mixed variational curvature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workdir", type=Path, default=Path("H3_ring_36"))
    parser.add_argument(
        "--seed-workdir",
        type=Path,
        default=None,
        help=(
            "Reuse matching corner checkpoints from a coarser nested ring; "
            "for example H3_ring_36 when running --points 72"
        ),
    )
    parser.add_argument("--points", type=int, default=36)
    parser.add_argument("--phi-origin-deg", type=float, default=0.0)
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=32000)
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--rho-angstrom", type=float, default=0.010)
    parser.add_argument("--circumradius-angstrom", type=float, default=0.90)
    parser.add_argument("--hf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--hf-max-cycle", type=int, default=100)
    parser.add_argument("--mc-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--mc-conv-tol-grad", type=float, default=3.0e-6)
    parser.add_argument("--mc-max-cycle-macro", type=int, default=100)
    parser.add_argument("--mc-max-cycle-micro", type=int, default=10)
    parser.add_argument("--mc-newton-max-cycle", type=int, default=60)
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-11)
    parser.add_argument("--fci-max-cycle", type=int, default=200)
    parser.add_argument("--spin-penalty-shift", type=float, default=0.2)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--gauge-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--spin-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--berry-tolerance-rad", type=float, default=5.0e-2)
    parser.add_argument("--pilot-curvature-ceiling", type=float, default=1.0e-2)
    parser.add_argument("--pilot-eta-floor", type=float, default=0.90)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--newton-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--geometry-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.points < 3:
        raise ValueError("--points must be at least 3.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.memory_mb < 1000:
        raise ValueError("--memory-mb is implausibly small.")
    if args.rho_angstrom <= 0.0:
        raise ValueError("--rho-angstrom must be positive.")
    if args.validation_tolerance <= 0.0 or args.gauge_tolerance <= 0.0:
        raise ValueError("Validation tolerances must be positive.")


def ring_angles(args: argparse.Namespace) -> Array:
    step = 360.0 / args.points
    return args.phi_origin_deg + step * np.arange(args.points, dtype=float)


def run_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Parameters that must match before old corner checkpoints are reused."""

    return {
        "points": args.points,
        "phi_origin_deg": args.phi_origin_deg,
        "basis": args.basis,
        "rho_angstrom": args.rho_angstrom,
        "circumradius_angstrom": args.circumradius_angstrom,
        "active_space": [core.N_CAS, list(core.NELE_CAS)],
        "protocol_roots": [2, 3],
        "hf_conv_tol": args.hf_conv_tol,
        "hf_max_cycle": args.hf_max_cycle,
        "mc_conv_tol": args.mc_conv_tol,
        "mc_conv_tol_grad": args.mc_conv_tol_grad,
        "mc_max_cycle_macro": args.mc_max_cycle_macro,
        "mc_max_cycle_micro": args.mc_max_cycle_micro,
        "fci_conv_tol": args.fci_conv_tol,
        "fci_max_cycle": args.fci_max_cycle,
        "spin_penalty_shift": args.spin_penalty_shift,
    }


def establish_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.workdir / "run_manifest.json"
    requested = run_signature(args)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != requested:
            raise RuntimeError(
                f"{manifest_path} does not match the requested calculation. "
                "Use a new --workdir, or deliberately remove/rename the old "
                "work directory after preserving its results."
            )
        return
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(requested, stream, indent=2)
        stream.write("\n")


def checkpoint_key(nroots: int, phi_deg: float) -> tuple[int, float]:
    return nroots, round(float(phi_deg) % 360.0, 10)


def index_seed_checkpoints(args: argparse.Namespace) -> dict[tuple[int, float], Path]:
    """Index compatible completed corners from a coarser nested ring."""

    if args.seed_workdir is None:
        return {}
    seed_manifest_path = args.seed_workdir / "run_manifest.json"
    seed_corner_dir = args.seed_workdir / "corners"
    if not seed_manifest_path.is_file() or not seed_corner_dir.is_dir():
        raise RuntimeError(
            f"--seed-workdir {args.seed_workdir} lacks run_manifest.json or corners/."
        )
    with seed_manifest_path.open("r", encoding="utf-8") as stream:
        seed_signature = json.load(stream)
    requested = run_signature(args)
    for key, value in requested.items():
        if key == "points":
            continue
        if seed_signature.get(key) != value:
            raise RuntimeError(
                f"Seed calculation mismatch for {key!r}: "
                f"seed={seed_signature.get(key)!r}, requested={value!r}."
            )
    if int(seed_signature.get("points", 0)) >= args.points:
        raise RuntimeError(
            "--seed-workdir must contain a coarser ring with fewer points."
        )

    indexed: dict[tuple[int, float], Path] = {}
    for path in sorted(seed_corner_dir.glob("SA*_i*_phi*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                nroots = int(np.asarray(archive["nroots"]).reshape(()))
                phi_deg = float(np.asarray(archive["phi_deg"]).reshape(()))
        except Exception as exc:
            raise RuntimeError(f"Could not index seed checkpoint {path}: {exc}") from exc
        key = checkpoint_key(nroots, phi_deg)
        if key in indexed:
            raise RuntimeError(f"Duplicate seed checkpoint for {key}: {path}.")
        indexed[key] = path
    print(
        f"Indexed {len(indexed)} reusable corners from {args.seed_workdir}",
        flush=True,
    )
    return indexed


def corner_label(nroots: int, index: int, phi_deg: float) -> str:
    wrapped = phi_deg % 360.0
    phi_text = f"{wrapped:010.5f}".replace(".", "p")
    return f"SA{nroots}_i{index:03d}_phi{phi_text}"


def load_corner(
    path: Path,
    label: str,
    phi_deg: float,
    nroots: int,
    args: argparse.Namespace,
    fci: Any,
    gto: Any,
) -> core.Corner:
    with np.load(path, allow_pickle=False) as archive:
        saved_phi = float(np.asarray(archive["phi_deg"]).reshape(()))
        saved_nroots = int(np.asarray(archive["nroots"]).reshape(()))
        if saved_nroots != nroots or not math.isclose(
            saved_phi, phi_deg, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError(
                f"Checkpoint metadata mismatch in {path}: "
                f"(phi,nroots)=({saved_phi},{saved_nroots})."
            )
        coordinates = np.asarray(archive["atoms"], dtype=float)
        atoms = [("H", tuple(float(x) for x in row)) for row in coordinates]
        mo_coeff = np.asarray(archive["mo_coeff"])
        ci_array = np.asarray(archive["ci_states"])
        ci_states = [np.asarray(item) for item in ci_array]
        energies = [float(x) for x in np.asarray(archive["energies"]).reshape(-1)]
        spin_values = [
            float(x) for x in np.asarray(archive["spin_squares"]).reshape(-1)
        ]
        ncore = int(np.asarray(archive["ncore"]).reshape(()))

    if len(ci_states) != nroots or len(energies) != nroots:
        raise ValueError(f"Incomplete checkpoint {path}.")
    if ncore != 0:
        raise ValueError(f"Expected ncore=0 in {path}; received {ncore}.")
    expected_atoms, _metadata = core.h3_distortion_geometry(
        phi_deg, args.rho_angstrom, args.circumradius_angstrom
    )
    expected_coordinates = np.asarray([item[1] for item in expected_atoms])
    if not np.allclose(coordinates, expected_coordinates, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Geometry mismatch in checkpoint {path}.")

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Angstrom"
    mol.basis = args.basis
    mol.charge = 0
    mol.spin = 1
    mol.symmetry = False
    mol.max_memory = args.memory_mb
    mol.verbose = 0
    mol.build()
    return core.Corner(
        label=label,
        protocol=f"SA{nroots}",
        phi_deg=phi_deg,
        atoms=atoms,
        mol=mol,
        mo_coeff=mo_coeff,
        ci_states=ci_states,
        selected_ci=ci_states[: core.SELECTED_M],
        energies=energies,
        spin_squares=spin_values,
        ncore=ncore,
        frame_orthogonality_error=core.selected_frame_orthogonality(
            ci_states[: core.SELECTED_M]
        ),
    )


def obtain_corner(
    index: int,
    phi_deg: float,
    nroots: int,
    args: argparse.Namespace,
    fci: Any,
    gto: Any,
    mcscf: Any,
    scf: Any,
    seed_checkpoints: dict[tuple[int, float], Path],
) -> core.Corner:
    label = corner_label(nroots, index, phi_deg)
    checkpoint = args.workdir / "corners" / f"{label}.npz"
    if args.resume and checkpoint.exists():
        try:
            result = load_corner(
                checkpoint, label, phi_deg, nroots, args, fci, gto
            )
            print(f"Loaded {label}", flush=True)
            return result
        except Exception as exc:
            print(
                f"WARNING: could not load {checkpoint}: {exc}; recalculating.",
                file=sys.stderr,
                flush=True,
            )

    seed_path = seed_checkpoints.get(checkpoint_key(nroots, phi_deg))
    if seed_path is not None:
        result = load_corner(seed_path, label, phi_deg, nroots, args, fci, gto)
        shutil.copy2(seed_path, checkpoint)
        print(f"Seeded {label} from {seed_path}", flush=True)
        return result

    print(f"Running {label} ...", flush=True)
    result = core.run_corner(
        label, phi_deg, nroots, args, fci, gto, mcscf, scf
    )
    # Keep the Mole object, but detach it from the per-corner file stream.
    core.close_mol_log(result.mol)
    result.mol.stdout = sys.stdout
    result.mol.verbose = 0
    return result


def ordered_product(matrices: Sequence[Array]) -> Array:
    result = np.eye(core.SELECTED_M, dtype=np.complex128)
    for matrix in matrices:
        result = result @ np.asarray(matrix, dtype=np.complex128)
    return result


def berry_phase_for_state(edges: Sequence[core.Edge], state: int) -> dict[str, Any]:
    product = 1.0 + 0.0j
    magnitudes: list[float] = []
    for edge in edges:
        value = complex(edge.overlap[state, state])
        magnitude = abs(value)
        if magnitude == 0.0:
            raise RuntimeError(
                f"Zero diagonal overlap for state {state} on edge {edge.label}."
            )
        product *= value / magnitude
        magnitudes.append(magnitude)
    phase = float(np.angle(product))
    return {
        "state_index": state,
        "phase_rad": phase,
        "phase_over_pi": phase / math.pi,
        "pi_error_rad": abs(abs(phase) - math.pi),
        "unit_product": product,
        "minimum_diagonal_overlap": min(magnitudes),
    }


def assignment_diagnostics(edges: Sequence[core.Edge]) -> dict[str, Any]:
    identity_margins = []
    for edge in edges:
        absolute = np.abs(edge.overlap)
        identity = float(absolute[0, 0] + absolute[1, 1])
        exchange = float(absolute[0, 1] + absolute[1, 0])
        identity_margins.append(identity - exchange)
    return {
        "minimum_identity_assignment_margin": min(identity_margins),
        "number_of_exchange_preferred_edges": int(
            np.count_nonzero(np.asarray(identity_margins) < 0.0)
        ),
    }


def ring_wilson(edges: Sequence[core.Edge]) -> dict[str, Any]:
    matrix = ordered_product([edge.transport for edge in edges])
    phases = np.sort(np.angle(np.linalg.eigvals(matrix)))
    return {
        "matrix": matrix,
        "eigenphases_rad": phases,
        "eigenphases_over_pi": phases / math.pi,
        "unitarity_error": core.frobenius(
            core.dagger(matrix) @ matrix - np.eye(core.SELECTED_M)
        ),
    }


def edge_error_maxima(edges: Sequence[core.Edge]) -> dict[str, float]:
    return {
        "production_vs_direct_fock": max(e.direct_overlap_error for e in edges),
        "reverse_overlap": max(e.reverse_overlap_error for e in edges),
        "reverse_transport": max(e.reverse_transport_error for e in edges),
        "polar_unitarity": max(e.unitarity_error for e in edges),
        "ambient_gram_reconstruction": max(e.ambient_gram_error for e in edges),
        "direct_state_norm": max(e.direct_state_norm_error for e in edges),
    }


def write_profile(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_validation_plot(
    path: Path, rows: Sequence[dict[str, Any]], points: int
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    phi = np.asarray([row["phi_start_deg"] for row in rows])
    gap2 = np.asarray([row["SA2_gap_hartree"] for row in rows])
    gap3 = np.asarray([row["SA3_gap_hartree"] for row in rows])
    curvature = np.asarray([row["curvature"] for row in rows])
    eta = np.asarray([row["eta_edge"] for row in rows])
    leakage = np.asarray([row["lambda_edge_max"] for row in rows])

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.2), sharex=True)
    axes[0].plot(phi, gap2, "o-", ms=3, lw=1.2, label="SA2")
    axes[0].plot(phi, gap3, "s-", ms=3, lw=1.2, label="SA3 (lowest pair)")
    axes[0].set_ylabel(r"$E_1-E_0$ / $E_h$")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(phi, curvature, "o-", ms=3, lw=1.2)
    axes[1].set_ylabel(r"$\mathcal{C}_{qp}$")
    axes[2].plot(phi, eta, "o-", ms=3, lw=1.2)
    axes[2].axhline(0.90, color="0.55", ls="--", lw=0.8)
    axes[2].set_ylabel(r"$\eta_{\rm edge}$")
    axes[3].plot(phi, leakage, "o-", ms=3, lw=1.2)
    axes[3].set_ylabel(r"$\lambda_{\max}$")
    axes[3].set_xlabel(r"Start angle $\phi$ / degree")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xlim(float(np.min(phi)), float(np.max(phi)))
    fig.suptitle(f"H3 mixed-curvature ring pilot ({points} geometries)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return None


def print_geometries(args: argparse.Namespace) -> int:
    for index, phi_deg in enumerate(ring_angles(args)):
        _atoms, metadata = core.h3_distortion_geometry(
            float(phi_deg), args.rho_angstrom, args.circumradius_angstrom
        )
        print(
            f"{index:03d} phi={phi_deg:11.6f} "
            f"r12={metadata['r12_angstrom']:.10f} "
            f"r23={metadata['r23_angstrom']:.10f} "
            f"r31={metadata['r31_angstrom']:.10f}"
        )
    return 0


def run_ring(args: argparse.Namespace) -> int:
    core.set_thread_environment(args.threads)
    pyscf, fci, gto, lib, mcscf, scf = core.import_pyscf()
    lib.num_threads(args.threads)
    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError("This PySCF version lacks fci.addons.overlap.")

    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "logs").mkdir(exist_ok=True)
    (args.workdir / "corners").mkdir(exist_ok=True)
    establish_manifest(args)
    seed_checkpoints = index_seed_checkpoints(args)
    angles = ring_angles(args)
    step_deg = 360.0 / args.points

    print("H3 mixed-curvature closed-ring validation", flush=True)
    print(
        f"points={args.points}; calculations={2 * args.points}; "
        f"step={step_deg:.8f} deg; rho={args.rho_angstrom:.8f} Angstrom",
        flush=True,
    )

    corners: dict[int, list[core.Corner]] = {2: [], 3: []}
    for index, phi_deg in enumerate(angles):
        for nroots in (2, 3):
            corners[nroots].append(
                obtain_corner(
                    index,
                    float(phi_deg),
                    nroots,
                    args,
                    fci,
                    gto,
                    mcscf,
                    scf,
                    seed_checkpoints,
                )
            )

    nuclear: dict[int, list[core.Edge]] = {2: [], 3: []}
    vertical: list[core.Edge] = []
    print(
        f"Building and independently checking {3 * args.points} edge overlaps ...",
        flush=True,
    )
    for index in range(args.points):
        next_index = (index + 1) % args.points
        for nroots in (2, 3):
            label = f"N_SA{nroots}_{index:03d}_{next_index:03d}"
            nuclear[nroots].append(
                core.build_edge(
                    label,
                    corners[nroots][index],
                    corners[nroots][next_index],
                    fci,
                    gto,
                )
            )
        vertical.append(
            core.build_edge(
                f"V_SA2_SA3_{index:03d}",
                corners[2][index],
                corners[3][index],
                fci,
                gto,
            )
        )
        if (index + 1) % max(1, args.points // 12) == 0:
            print(f"  checked {index + 1}/{args.points} plaquette positions", flush=True)

    rows: list[dict[str, Any]] = []
    plaquettes: list[dict[str, Any]] = []
    gauge_errors: list[float] = []
    for index in range(args.points):
        next_index = (index + 1) % args.points
        edges = (
            nuclear[2][index],
            vertical[next_index],
            vertical[index],
            nuclear[3][index],
        )
        item = core.plaquette_curvature(*(edge.overlap for edge in edges))
        gauge_error = core.gauge_test(
            [edge.overlap for edge in edges], item["curvature"]
        )
        gauge_errors.append(gauge_error)
        plaquettes.append(item)
        e2 = corners[2][index].energies
        e3 = corners[3][index].energies
        rows.append(
            {
                "index": index,
                "phi_start_deg": float(angles[index] % 360.0),
                "phi_end_deg": float(angles[next_index] % 360.0),
                "phi_mid_deg": float((angles[index] + 0.5 * step_deg) % 360.0),
                "SA2_E0_hartree": e2[0],
                "SA2_E1_hartree": e2[1],
                "SA2_gap_hartree": e2[1] - e2[0],
                "SA3_E0_hartree": e3[0],
                "SA3_E1_hartree": e3[1],
                "SA3_E2_hartree": e3[2],
                "SA3_gap_hartree": e3[1] - e3[0],
                "curvature": item["curvature"],
                "eta_edge": item["eta_edge"],
                "lambda_AB_nuclear_SA2": item["lambda_edges"][0],
                "lambda_BD_protocol": item["lambda_edges"][1],
                "lambda_AC_protocol": item["lambda_edges"][2],
                "lambda_CD_nuclear_SA3": item["lambda_edges"][3],
                "lambda_edge_max": item["lambda_edge_max"],
                "mixed_phase_minus_rad": item["holonomy_phases"][0],
                "mixed_phase_plus_rad": item["holonomy_phases"][1],
                "formula_error": item["formula_error"],
                "gauge_error": gauge_error,
            }
        )

    all_edges = nuclear[2] + nuclear[3] + vertical
    berry = {
        "SA2": [berry_phase_for_state(nuclear[2], state) for state in range(2)],
        "SA3": [berry_phase_for_state(nuclear[3], state) for state in range(2)],
    }
    assignments = {
        "SA2": assignment_diagnostics(nuclear[2]),
        "SA3": assignment_diagnostics(nuclear[3]),
    }
    nuclear_wilson = {
        "SA2": ring_wilson(nuclear[2]),
        "SA3": ring_wilson(nuclear[3]),
    }
    errors = edge_error_maxima(all_edges)
    max_frame_error = max(
        corner.frame_orthogonality_error
        for protocol in corners.values()
        for corner in protocol
    )
    max_spin_error = max(
        abs(value - core.TARGET_S2)
        for protocol in corners.values()
        for corner in protocol
        for value in corner.spin_squares
    )
    curvatures = np.asarray([row["curvature"] for row in rows])
    etas = np.asarray([row["eta_edge"] for row in rows])
    leakages = np.asarray([row["lambda_edge_max"] for row in rows])
    peak_index = int(np.argmax(curvatures))

    numerical_checks = {
        "production_overlap": errors["production_vs_direct_fock"]
        <= args.validation_tolerance,
        "reverse_overlap": errors["reverse_overlap"]
        <= args.validation_tolerance,
        "reverse_transport": errors["reverse_transport"]
        <= args.validation_tolerance,
        "polar_unitarity": errors["polar_unitarity"]
        <= args.validation_tolerance,
        "ambient_gram": errors["ambient_gram_reconstruction"]
        <= args.validation_tolerance,
        "direct_state_norm": errors["direct_state_norm"]
        <= args.validation_tolerance,
        "frame_orthonormality": max_frame_error <= args.validation_tolerance,
        "spin": max_spin_error <= args.spin_tolerance,
        "curvature_formula": max(float(row["formula_error"]) for row in rows)
        <= args.validation_tolerance,
        "gauge_invariance": max(gauge_errors) <= args.gauge_tolerance,
    }
    berry_check = all(
        item["pi_error_rad"] <= args.berry_tolerance_rad
        for protocol in berry.values()
        for item in protocol
    )
    assignment_check = all(
        item["number_of_exchange_preferred_edges"] == 0
        for item in assignments.values()
    )
    pilot_negative_control = (
        all(numerical_checks.values())
        and berry_check
        and float(np.max(curvatures)) <= args.pilot_curvature_ceiling
        and float(np.min(etas)) >= args.pilot_eta_floor
    )

    summary = {
        "test": "H3 genuine-CI closed-ring negative control",
        "status": "PILOT_PASS" if pilot_negative_control else "REVIEW",
        "note": (
            "PILOT_PASS still requires 72- and 144-geometry step refinement "
            "before the molecular negative-control gate is final. Root "
            "permutations or exchange-preferred assignments are reported but "
            "do not veto a gauge-invariant two-state-subspace test."
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
        },
        "parameters": {
            "points": args.points,
            "casscf_calculations": 2 * args.points,
            "step_deg": step_deg,
            "phi_origin_deg": args.phi_origin_deg,
            "rho_angstrom": args.rho_angstrom,
            "circumradius_angstrom": args.circumradius_angstrom,
            "basis": args.basis,
            "active_space": "CAS(3,3)",
            "protocols": ["equal-weight SA2", "equal-weight SA3; lowest pair"],
        },
        "berry_phases": berry,
        "nuclear_wilson_loops": nuclear_wilson,
        "adiabatic_assignment": assignments,
        "mixed_curvature_profile": {
            "maximum": float(np.max(curvatures)),
            "mean": float(np.mean(curvatures)),
            "rms": float(np.sqrt(np.mean(curvatures**2))),
            "minimum": float(np.min(curvatures)),
            "maximum_per_radian": float(np.max(curvatures))
            / math.radians(step_deg),
            "rms_per_radian": float(np.sqrt(np.mean(curvatures**2)))
            / math.radians(step_deg),
            "peak_start_phi_deg": rows[peak_index]["phi_start_deg"],
            "peak_mid_phi_deg": rows[peak_index]["phi_mid_deg"],
            "minimum_eta_edge": float(np.min(etas)),
            "maximum_lambda_edge": float(np.max(leakages)),
        },
        "maximum_numerical_errors": {
            **errors,
            "frame_orthonormality": max_frame_error,
            "spin_square": max_spin_error,
            "curvature_formula": max(float(row["formula_error"]) for row in rows),
            "gauge_invariance": max(gauge_errors),
        },
        "acceptance": {
            "numerical_checks": numerical_checks,
            "berry_pi_check": berry_check,
            "identity_assignment_diagnostic": assignment_check,
            "identity_assignment_required_for_subspace_gate": False,
            "pilot_curvature_check": float(np.max(curvatures))
            <= args.pilot_curvature_ceiling,
            "pilot_eta_check": float(np.min(etas)) >= args.pilot_eta_floor,
            "pilot_negative_control": pilot_negative_control,
        },
    }

    write_profile(args.workdir / "ring_profile.csv", rows)
    with (args.workdir / "ring_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(core.jsonable(summary), stream, indent=2)
        stream.write("\n")

    matrix_data: dict[str, Array] = {}
    for nroots in (2, 3):
        matrix_data[f"nuclear_SA{nroots}_overlaps"] = np.stack(
            [edge.overlap for edge in nuclear[nroots]]
        )
        matrix_data[f"nuclear_SA{nroots}_transports"] = np.stack(
            [edge.transport for edge in nuclear[nroots]]
        )
    matrix_data["vertical_overlaps"] = np.stack([e.overlap for e in vertical])
    matrix_data["vertical_transports"] = np.stack([e.transport for e in vertical])
    matrix_data["mixed_holonomies"] = np.stack(
        [item["holonomy"] for item in plaquettes]
    )
    matrix_data["angles_deg"] = np.asarray(angles)
    np.savez_compressed(args.workdir / "ring_matrices.npz", **matrix_data)

    plot_warning = None
    if args.plot:
        plot_warning = make_validation_plot(
            args.workdir / "ring_validation.png", rows, args.points
        )
        if plot_warning is not None:
            print(f"WARNING: {plot_warning}", file=sys.stderr)

    print("\nRing result", flush=True)
    for protocol in ("SA2", "SA3"):
        phases = [item["phase_over_pi"] for item in berry[protocol]]
        wilson_phases = nuclear_wilson[protocol]["eigenphases_over_pi"]
        print(
            f"{protocol}: adiabatic Berry phases/pi = "
            f"{phases[0]: .8f}, {phases[1]: .8f}; "
            f"Wilson eigenphases/pi = {wilson_phases[0]: .8f}, "
            f"{wilson_phases[1]: .8f}",
            flush=True,
        )
    print(
        f"C_qp: max={np.max(curvatures):.12e}, "
        f"mean={np.mean(curvatures):.12e}, "
        f"rms={np.sqrt(np.mean(curvatures**2)):.12e}",
        flush=True,
    )
    print(
        f"eta_edge min={np.min(etas):.12e}; "
        f"lambda_edge max={np.max(leakages):.12e}",
        flush=True,
    )
    print(
        f"max production/direct error="
        f"{errors['production_vs_direct_fock']:.3e}; "
        f"max gauge error={max(gauge_errors):.3e}",
        flush=True,
    )
    print(f"Status: {summary['status']}", flush=True)
    return 0 if all(numerical_checks.values()) else 2


def main() -> int:
    args = make_parser().parse_args()
    validate_args(args)
    if args.geometry_only:
        return print_geometries(args)
    return run_ring(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
