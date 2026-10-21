#!/usr/bin/env python3
"""Analyze strict O3 Gate 4B directional refinements."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


@dataclass
class Wavefunction:
    tag: str
    direction: str
    sign: int
    h: float
    protocol: str
    basis: str
    r1: float
    r2: float
    theta: float
    ncas: int
    nelecas: int
    ncore: int
    mo: object
    ci: list
    energies: object
    convergence_mode: str
    final_density_change: float
    final_max_orbital_rotation: float


def import_modules():
    try:
        import numpy as np
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError("Activate the PySCF environment before analysis.") from exc
    return np, fci, gto


def atoms(wfn: Wavefunction):
    half = math.radians(wfn.theta) / 2.0
    s, c = math.sin(half), math.cos(half)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("O", (wfn.r1 * s, wfn.r1 * c, 0.0)),
        ("O", (-wfn.r2 * s, wfn.r2 * c, 0.0)),
    ]


def molecule(wfn: Wavefunction, gto):
    mol = gto.Mole()
    mol.atom = atoms(wfn)
    mol.unit = "Angstrom"
    mol.basis = wfn.basis
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = "Cs"
    mol.verbose = 0
    mol.build()
    if mol.groupname != "Cs":
        raise RuntimeError(f"Expected forced Cs for {wfn.tag}, got {mol.groupname}.")
    return mol


def load(path: Path, np) -> Wavefunction:
    with np.load(path, allow_pickle=False) as data:
        return Wavefunction(
            tag=str(data["tag"].item()),
            direction=str(data["direction"].item()),
            sign=int(data["sign"].item()),
            h=float(data["h"].item()),
            protocol=str(data["protocol"].item()),
            basis=str(data["basis"].item()),
            r1=float(data["r1"].item()),
            r2=float(data["r2"].item()),
            theta=float(data["theta"].item()),
            ncas=int(data["ncas"].item()),
            nelecas=int(data["nelecas"].item()),
            ncore=int(data["ncore"].item()),
            mo=data["mo_coeff"].copy(),
            ci=[vector.copy() for vector in data["ci_vectors"]],
            energies=data["state_energies"].copy(),
            convergence_mode=str(data["convergence_mode"].item()),
            final_density_change=float(data["final_density_change"].item()),
            final_max_orbital_rotation=float(
                data["final_max_orbital_rotation"].item()
            ),
        )


def final_gradient(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # One-step CASSCF prints |grad[o]|, whereas Newton CASSCF prints |grad|.
    # Taking both patterns in file order ensures that the reported value is the
    # final Newton-polished residual rather than the preceding stabilization
    # residual.
    values = re.findall(r"\|grad(?:\[o\])?\|=\s*([0-9.eE+\-]+)", text)
    return float(values[-1]) if values else float("nan")


def full_overlap(x: Wavefunction, y: Wavefunction, mol_x, mol_y, fci, gto, np):
    if (x.ncas, x.nelecas, x.ncore) != (y.ncas, y.nelecas, y.ncore):
        raise RuntimeError("Inconsistent CAS/core partitions on one edge.")
    if x.basis.lower() != y.basis.lower():
        raise RuntimeError("Inconsistent AO bases on one edge.")
    overlap_fn = getattr(fci.addons, "overlap", None)
    if overlap_fn is None:
        raise RuntimeError("This PySCF version lacks fci.addons.overlap.")
    sao = gto.intor_cross("int1e_ovlp", mol_x, mol_y)
    core_x = x.mo[:, : x.ncore]
    core_y = y.mo[:, : y.ncore]
    active_x = x.mo[:, x.ncore : x.ncore + x.ncas]
    active_y = y.mo[:, y.ncore : y.ncore + y.ncas]
    scc = core_x.T.conj() @ sao @ core_y
    sca = core_x.T.conj() @ sao @ active_y
    sac = active_x.T.conj() @ sao @ core_y
    saa = active_x.T.conj() @ sao @ active_y
    core_singular = np.linalg.svd(scc, compute_uv=False)
    if float(core_singular.min()) < 1.0e-10:
        raise RuntimeError(f"Singular core overlap on {x.tag}->{y.tag}.")
    effective_active = saa - sac @ np.linalg.solve(scc, sca)
    core_factor = np.linalg.det(scc) ** 2
    matrix = np.empty((len(x.ci), len(y.ci)), dtype=complex)
    for i, ci_x in enumerate(x.ci):
        for j, ci_y in enumerate(y.ci):
            matrix[i, j] = core_factor * overlap_fn(
                ci_x, ci_y, x.ncas, x.nelecas, s=effective_active
            )
    return matrix


def polar(matrix, np):
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh, singular


def leakage(singular, m: int = 2) -> float:
    value = 1.0 - float(sum(abs(item) ** 2 for item in singular)) / m
    return 0.0 if -1.0e-12 < value < 0.0 else value


def random_unitary(m: int, rng, np):
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(phases.conj())


def plaquette_metrics(edges: dict, np, gauge_test: bool = True):
    transports, singular = {}, {}
    for name, matrix in edges.items():
        transports[name], singular[name] = polar(matrix, np)
    k_qp = transports["AB"] @ transports["BD"]
    k_pq = transports["AC"] @ transports["CD"]
    difference = float(np.linalg.norm(k_qp - k_pq, ord="fro"))
    denominator = float(
        np.linalg.norm(k_qp, ord="fro") + np.linalg.norm(k_pq, ord="fro")
    )
    result = {
        "C_qp": difference / denominator,
        "path_difference_F": difference,
        "eta_edge": min(float(values.min()) for values in singular.values()),
    }
    for name in ("AB", "BD", "AC", "CD"):
        result[f"sigma_min_{name}"] = float(singular[name].min())
        result[f"lambda_{name}"] = leakage(singular[name])
    if gauge_test:
        rng = np.random.default_rng(20260811)
        ua, ub, uc, ud = [random_unitary(2, rng, np) for _ in range(4)]
        transformed = {
            "AB": ua.T.conj() @ edges["AB"] @ ub,
            "BD": ub.T.conj() @ edges["BD"] @ ud,
            "AC": ua.T.conj() @ edges["AC"] @ uc,
            "CD": uc.T.conj() @ edges["CD"] @ ud,
        }
        tested = plaquette_metrics(transformed, np, gauge_test=False)
        result["gauge_test_abs_error"] = abs(result["C_qp"] - tested["C_qp"])
    return result


def log_fit(x, y, np):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"alpha": float("nan"), "prefactor": float("nan"), "R2": float("nan")}
    lx, ly = np.log(x[mask]), np.log(y[mask])
    alpha, intercept = np.polyfit(lx, ly, 1)
    predicted = alpha * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"alpha": float(alpha), "prefactor": float(np.exp(intercept)), "R2": r2}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    np, fci, gto = import_modules()
    wavefunction_dir = args.workdir / "wavefunctions"
    paths = sorted(wavefunction_dir.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No Gate 4B checkpoints found in {wavefunction_dir}.")
    wfns, mols, gradients = {}, {}, {}
    for path in paths:
        wfn = load(path, np)
        expected = 2 if wfn.protocol == "SA2" else 3
        if len(wfn.ci) != expected:
            raise RuntimeError(f"{path} contains {len(wfn.ci)} roots, expected {expected}.")
        key = (wfn.tag, wfn.protocol)
        wfns[key] = wfn
        mols[key] = molecule(wfn, gto)
        gradients[key] = final_gradient(args.workdir / "logs" / f"{wfn.tag}_{wfn.protocol}.log")
    tags = sorted({tag for tag, _protocol in wfns})
    if "center" not in tags or len(tags) != 17 or len(wfns) != 34:
        raise RuntimeError(f"Expected 17 geometries/34 wavefunctions, got {len(tags)}/{len(wfns)}.")

    selection, same_geometry, subspace_rows = {}, {}, []
    for tag in tags:
        sa2, sa3 = wfns[(tag, "SA2")], wfns[(tag, "SA3")]
        m23 = full_overlap(
            sa2, sa3, mols[(tag, "SA2")], mols[(tag, "SA3")], fci, gto, np
        )
        _u, singular, vh = np.linalg.svd(m23, full_matrices=True)
        frame = vh.T.conj()[:, :2]
        selection[tag], same_geometry[tag] = frame, m23
        subspace_rows.append(
            {
                "tag": tag,
                "direction": sa2.direction,
                "sign": sa2.sign,
                "h": f"{sa2.h:.12e}",
                "delta_r_A": f"{sa2.r1-sa2.r2:.12e}",
                "theta_deg": f"{sa2.theta:.12e}",
                "SA2_gap01_Eh": f"{float(sa2.energies[1]-sa2.energies[0]):.12e}",
                "SA3_gap01_Eh": f"{float(sa3.energies[1]-sa3.energies[0]):.12e}",
                "SA3_gap12_Eh": f"{float(sa3.energies[2]-sa3.energies[1]):.12e}",
                "sigma_match_1": f"{singular[0]:.12e}",
                "sigma_match_2": f"{singular[1]:.12e}",
                "lambda_protocol": f"{leakage(singular):.12e}",
                "SA3_root3_weight": f"{float(np.sum(abs(frame[2,:])**2)):.12e}",
                "final_grad_SA2": f"{gradients[(tag,'SA2')]:.12e}",
                "final_grad_SA3": f"{gradients[(tag,'SA3')]:.12e}",
                "convergence_mode_SA2": sa2.convergence_mode,
                "convergence_mode_SA3": sa3.convergence_mode,
                "final_ddm_SA2": f"{sa2.final_density_change:.12e}",
                "final_ddm_SA3": f"{sa3.final_density_change:.12e}",
                "final_maxrot_SA2": f"{sa2.final_max_orbital_rotation:.12e}",
                "final_maxrot_SA3": f"{sa3.final_max_orbital_rotation:.12e}",
            }
        )

    center = wfns[("center", "SA2")]
    plaquette_rows, detailed = [], {}
    for tag in tags:
        if tag == "center":
            continue
        neighbor = wfns[(tag, "SA2")]
        if neighbor.sign < 0:
            q0, q1 = tag, "center"
        else:
            q0, q1 = "center", tag
        sa2_0, sa2_1 = wfns[(q0, "SA2")], wfns[(q1, "SA2")]
        sa3_0, sa3_1 = wfns[(q0, "SA3")], wfns[(q1, "SA3")]
        m22 = full_overlap(
            sa2_0, sa2_1, mols[(q0, "SA2")], mols[(q1, "SA2")], fci, gto, np
        )
        m33 = full_overlap(
            sa3_0, sa3_1, mols[(q0, "SA3")], mols[(q1, "SA3")], fci, gto, np
        )
        v0, v1 = selection[q0], selection[q1]
        edges = {
            "AB": m22,
            "BD": same_geometry[q1] @ v1,
            "AC": same_geometry[q0] @ v0,
            "CD": v0.T.conj() @ m33 @ v1,
        }
        metrics = plaquette_metrics(edges, np)
        max_gradient = max(
            gradients[(q0, "SA2")],
            gradients[(q0, "SA3")],
            gradients[(q1, "SA2")],
            gradients[(q1, "SA3")],
        )
        row = {
            "plaquette": f"{q0}_to_{q1}",
            "direction": neighbor.direction,
            "sign": neighbor.sign,
            "h": f"{neighbor.h:.12e}",
            "q0": q0,
            "q1": q1,
            **{name: f"{value:.12e}" for name, value in metrics.items()},
            "max_final_orbital_gradient": f"{max_gradient:.12e}",
        }
        plaquette_rows.append(row)
        detailed[row["plaquette"]] = metrics

    scaling_rows = []
    for direction in ("delta_r", "theta"):
        steps = sorted(
            {float(row["h"]) for row in plaquette_rows if row["direction"] == direction}
        )
        for h in steps:
            selected = [
                row
                for row in plaquette_rows
                if row["direction"] == direction and abs(float(row["h"]) - h) < 1.0e-14
            ]
            by_sign = {int(row["sign"]): row for row in selected}
            if set(by_sign) != {-1, 1}:
                raise RuntimeError(f"Missing +/- pair for {direction}, h={h}.")
            minus, plus = float(by_sign[-1]["C_qp"]), float(by_sign[1]["C_qp"])
            values = np.asarray([minus, plus])
            mean_value = float(values.mean())
            relative_difference = (
                float(abs(minus - plus) / mean_value) if mean_value > 0.0 else 0.0
            )
            scaling_rows.append(
                {
                    "direction": direction,
                    "h": f"{h:.12e}",
                    "C_minus": f"{minus:.12e}",
                    "C_plus": f"{plus:.12e}",
                    "C_max": f"{float(values.max()):.12e}",
                    "C_mean": f"{mean_value:.12e}",
                    "C_rms": f"{float(np.sqrt(np.mean(values**2))):.12e}",
                    "relative_pm_difference": f"{relative_difference:.12e}",
                    "eta_edge_min": f"{min(float(row['eta_edge']) for row in selected):.12e}",
                    "lambda_nuclear_max": f"{max(max(float(row['lambda_AB']),float(row['lambda_CD'])) for row in selected):.12e}",
                    "max_final_orbital_gradient": f"{max(float(row['max_final_orbital_gradient']) for row in selected):.12e}",
                }
            )

    fit_rows = []
    for direction in ("delta_r", "theta"):
        selected = [row for row in scaling_rows if row["direction"] == direction]
        x = [float(row["h"]) for row in selected]
        series = {
            "minus": [float(row["C_minus"]) for row in selected],
            "plus": [float(row["C_plus"]) for row in selected],
            "maximum": [float(row["C_max"]) for row in selected],
            "mean": [float(row["C_mean"]) for row in selected],
            "rms": [float(row["C_rms"]) for row in selected],
        }
        for statistic, values in series.items():
            fit = log_fit(x, values, np)
            fit_rows.append(
                {
                    "direction": direction,
                    "statistic": statistic,
                    "alpha": f"{fit['alpha']:.12e}",
                    "prefactor": f"{fit['prefactor']:.12e}",
                    "R2_loglog": f"{fit['R2']:.12e}",
                    "n_points": len(x),
                }
            )

    write_csv(args.workdir / "O3_gate4B_subspaces.csv", subspace_rows)
    write_csv(args.workdir / "O3_gate4B_plaquettes.csv", plaquette_rows)
    write_csv(args.workdir / "O3_gate4B_scaling.csv", scaling_rows)
    write_csv(args.workdir / "O3_gate4B_fits.csv", fit_rows)
    summary = {
        "schema_version": 3,
        "stage": "Gate4B",
        "definition": "strict polar-unitary mixed nuclear-variational nonclosure refinement",
        "convergence": {
            "pyscf_count": sum(
                wfn.convergence_mode == "pyscf" for wfn in wfns.values()
            ),
            "stationary_floor_count": sum(
                wfn.convergence_mode == "stationary_floor" for wfn in wfns.values()
            ),
            "maximum_final_orbital_gradient": max(gradients.values()),
            "maximum_final_density_change": max(
                wfn.final_density_change for wfn in wfns.values()
            ),
            "maximum_final_orbital_rotation": max(
                wfn.final_max_orbital_rotation for wfn in wfns.values()
            ),
        },
        "center_SA2_gap_Eh": float(center.energies[1] - center.energies[0]),
        "subspaces": subspace_rows,
        "plaquettes": detailed,
        "scaling": scaling_rows,
        "fits": fit_rows,
    }
    (args.workdir / "O3_gate4B_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nGate 4B scaling fits")
    print("direction  statistic   alpha       R2(log-log)")
    for row in fit_rows:
        if row["statistic"] in ("maximum", "mean", "rms"):
            print(
                f"{row['direction']:10s} {row['statistic']:9s} "
                f"{float(row['alpha']):10.6f}  {float(row['R2_loglog']):.8f}"
            )
    print(f"\nResults written to {args.workdir}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("O3_gate4B_work"))
    args = parser.parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    analyze(args)


if __name__ == "__main__":
    main()
