#!/usr/bin/env python3
"""Overlap, holonomy, and observable analysis for the O3 weighted-SA2 pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path


@dataclass
class Wavefunction:
    tag: str
    direction: str
    sign: int
    excited_weight: float
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
    dipole: object
    convergence_mode: str
    final_gradient: float


def import_modules(require_pyscf=True):
    import numpy as np
    if not require_pyscf:
        return np, None, None
    try:
        from pyscf import fci, gto
    except ImportError as exc:
        raise RuntimeError("Activate the production PySCF environment before analysis.") from exc
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
        if str(data["stage"].item()) != "O3_weighted_SA2_pilot":
            raise RuntimeError(f"Unexpected stage in {path}.")
        return Wavefunction(
            tag=str(data["tag"].item()), direction=str(data["direction"].item()),
            sign=int(data["sign"].item()), excited_weight=float(data["excited_weight"].item()),
            basis=str(data["basis"].item()), r1=float(data["r1"].item()),
            r2=float(data["r2"].item()), theta=float(data["theta"].item()),
            ncas=int(data["ncas"].item()), nelecas=int(data["nelecas"].item()),
            ncore=int(data["ncore"].item()), mo=data["mo_coeff"].copy(),
            ci=[row.copy() for row in data["ci_vectors"]],
            energies=data["state_energies"].copy(),
            dipole=data["dipole_electronic_au"].copy(),
            convergence_mode=str(data["convergence_mode"].item()),
            final_gradient=float(data["final_orbital_gradient"].item()),
        )


def full_overlap(x: Wavefunction, y: Wavefunction, mol_x, mol_y, fci, gto, np):
    if (x.ncas, x.nelecas, x.ncore) != (y.ncas, y.nelecas, y.ncore):
        raise RuntimeError("Inconsistent CAS/core partitions on one edge.")
    if x.basis.lower() != y.basis.lower():
        raise RuntimeError("Inconsistent AO bases on one edge.")
    overlap_fn = getattr(fci.addons, "overlap", None)
    if overlap_fn is None:
        raise RuntimeError("This PySCF version lacks fci.addons.overlap.")
    sao = gto.intor_cross("int1e_ovlp", mol_x, mol_y)
    core_x, core_y = x.mo[:, :x.ncore], y.mo[:, :y.ncore]
    active_x = x.mo[:, x.ncore:x.ncore + x.ncas]
    active_y = y.mo[:, y.ncore:y.ncore + y.ncas]
    scc = core_x.T.conj() @ sao @ core_y
    sca = core_x.T.conj() @ sao @ active_y
    sac = active_x.T.conj() @ sao @ core_y
    saa = active_x.T.conj() @ sao @ active_y
    if x.ncore:
        core_singular = np.linalg.svd(scc, compute_uv=False)
        if float(core_singular.min()) < 1.0e-10:
            raise RuntimeError(f"Singular core overlap on {x.tag}->{y.tag}.")
        effective_active = saa - sac @ np.linalg.solve(scc, sca)
        core_factor = np.linalg.det(scc) ** 2
    else:
        effective_active, core_factor = saa, 1.0
    matrix = np.empty((2, 2), dtype=complex)
    for i, ci_x in enumerate(x.ci):
        for j, ci_y in enumerate(y.ci):
            matrix[i, j] = core_factor * overlap_fn(
                ci_x, ci_y, x.ncas, x.nelecas, s=effective_active
            )
    return matrix


def polar(matrix, np):
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh, singular


def leakage(singular) -> float:
    value = 1.0 - float(sum(abs(item) ** 2 for item in singular)) / len(singular)
    return 0.0 if -1.0e-12 < value < 0.0 else value


def assignment(matrix, np):
    scores = []
    for perm in itertools.permutations(range(matrix.shape[1]), matrix.shape[0]):
        score = float(sum(abs(matrix[i, perm[i]]) ** 2 for i in range(matrix.shape[0])))
        scores.append((score, tuple(int(x) for x in perm)))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {
        "permutation": list(best[1]),
        "score": best[0],
        "margin": best[0] - second[0],
        "normalized_margin": (best[0] - second[0]) / matrix.shape[0],
    }


def edge_metrics(matrix, np):
    _transport, singular = polar(matrix, np)
    assigned = assignment(matrix, np)
    return {
        "singular_values": [float(x) for x in singular],
        "sigma_min": float(singular.min()),
        "leakage": leakage(singular),
        **assigned,
    }


def compose_permutations(first, second):
    return [int(second[first[i]]) for i in range(len(first))]


def random_unitary(m: int, rng, np):
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q @ np.diag(phases.conj())


def route_metrics(route_qp, route_pq, dipole_d, np, gauge_test=True):
    """Compare two piecewise polar-transport routes with common endpoints."""
    ordered_edges = list(route_qp) + list(route_pq)
    names = [name for name, _matrix in ordered_edges]
    if len(set(names)) != len(names):
        raise RuntimeError("Route edge names must be unique.")
    transports, singular, assignments = {}, {}, {}
    for name, matrix in ordered_edges:
        transports[name], singular[name] = polar(matrix, np)
        assignments[name] = assignment(matrix, np)

    def transport_product(route):
        result = np.eye(next(iter(transports.values())).shape[0], dtype=complex)
        for name, _matrix in route:
            result = result @ transports[name]
        return result

    def permutation_product(route):
        result = list(range(len(next(iter(assignments.values()))["permutation"])))
        for name, _matrix in route:
            result = compose_permutations(result, assignments[name]["permutation"])
        return result

    k_qp = transport_product(route_qp)
    k_pq = transport_product(route_pq)
    path_difference = float(np.linalg.norm(k_qp - k_pq, ord="fro"))
    denominator = float(np.linalg.norm(k_qp, ord="fro") + np.linalg.norm(k_pq, ord="fro"))
    w_loop = k_qp @ k_pq.T.conj()
    phases = sorted(float(x) for x in np.angle(np.linalg.eigvals(w_loop)))
    pulled_qp = np.asarray([k_qp @ component @ k_qp.T.conj() for component in dipole_d])
    pulled_pq = np.asarray([k_pq @ component @ k_pq.T.conj() for component in dipole_d])
    dipole_axis_abs = [float(np.linalg.norm(pulled_qp[x] - pulled_pq[x], ord="fro")) for x in range(3)]
    dipole_abs = float(np.linalg.norm(pulled_qp - pulled_pq))
    dipole_scale = max(float(np.linalg.norm(pulled_qp)), float(np.linalg.norm(pulled_pq)), 1.0e-15)
    perm_qp = permutation_product(route_qp)
    perm_pq = permutation_product(route_pq)
    result = {
        "C_qp": path_difference / denominator,
        "path_difference_F": path_difference,
        "holonomy_phases_rad": phases,
        "holonomy_max_abs_phase_rad": max(abs(x) for x in phases),
        "eta_edge": min(float(value.min()) for value in singular.values()),
        "maximum_edge_leakage": max(leakage(value) for value in singular.values()),
        "minimum_normalized_assignment_margin": min(
            value["normalized_margin"] for value in assignments.values()
        ),
        "route_q_then_p_permutation": perm_qp,
        "route_p_then_q_permutation": perm_pq,
        "same_composed_assignment": perm_qp == perm_pq,
        "dipole_path_difference_au": dipole_abs,
        "dipole_path_difference_by_axis_au": dipole_axis_abs,
        "dipole_path_difference_normalized": dipole_abs / dipole_scale,
        "edges": {
            name: {
                "singular_values": [float(x) for x in singular[name]],
                "sigma_min": float(singular[name].min()),
                "leakage": leakage(singular[name]),
                **assignments[name],
            }
            for name in names
        },
        "route_q_then_p_edges": [name for name, _matrix in route_qp],
        "route_p_then_q_edges": [name for name, _matrix in route_pq],
    }
    if gauge_test:
        rng = np.random.default_rng(20260915)
        u_start, u_end = random_unitary(2, rng, np), random_unitary(2, rng, np)

        def transform_route(route):
            gauges = [u_start]
            gauges.extend(random_unitary(2, rng, np) for _ in range(len(route) - 1))
            gauges.append(u_end)
            return [
                (name, gauges[index].T.conj() @ matrix @ gauges[index + 1])
                for index, (name, matrix) in enumerate(route)
            ]

        transformed_dipole = np.asarray([u_end.T.conj() @ item @ u_end for item in dipole_d])
        tested = route_metrics(
            transform_route(route_qp), transform_route(route_pq),
            transformed_dipole, np, gauge_test=False,
        )
        result["gauge_test_C_abs_error"] = abs(result["C_qp"] - tested["C_qp"])
        result["gauge_test_dipole_abs_error"] = abs(
            result["dipole_path_difference_au"] - tested["dipole_path_difference_au"]
        )
    return result


def plaquette_metrics(edges: dict, dipole_d, np, gauge_test=True):
    return route_metrics(
        [("AB", edges["AB"]), ("BD", edges["BD"])],
        [("AC", edges["AC"]), ("CD", edges["CD"])],
        dipole_d, np, gauge_test=gauge_test,
    )


def state_id(key):
    tag, weight = key
    return f"{tag}@p={weight:.6f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_cell(row):
    metrics = row["metrics"]
    return {
        "plaquette": row["plaquette"], "direction": row["direction"],
        "cell_type": row["cell_type"], "q0": row["q0"], "q1": row["q1"],
        "p0": f"{row['p0']:.12e}", "p1": f"{row['p1']:.12e}",
        "q_span": row["q_span"], "p_span": f"{row['p_span']:.12e}",
        "C_qp": f"{metrics['C_qp']:.12e}",
        "holonomy_max_abs_phase_rad": f"{metrics['holonomy_max_abs_phase_rad']:.12e}",
        "eta_edge": f"{metrics['eta_edge']:.12e}",
        "maximum_edge_leakage": f"{metrics['maximum_edge_leakage']:.12e}",
        "minimum_normalized_assignment_margin": f"{metrics['minimum_normalized_assignment_margin']:.12e}",
        "same_composed_assignment": metrics["same_composed_assignment"],
        "route_q_then_p_permutation": json.dumps(metrics["route_q_then_p_permutation"]),
        "route_p_then_q_permutation": json.dumps(metrics["route_p_then_q_permutation"]),
        "dipole_path_difference_au": f"{metrics['dipole_path_difference_au']:.12e}",
        "dipole_path_difference_normalized": f"{metrics['dipole_path_difference_normalized']:.12e}",
        "gauge_test_C_abs_error": f"{metrics['gauge_test_C_abs_error']:.12e}",
        "gauge_test_dipole_abs_error": f"{metrics['gauge_test_dipole_abs_error']:.12e}",
    }


def merge_cell_metrics(row):
    """Return one JSON-ready cell record without relying on outer-scope names."""
    return {**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}


def create_figure(rows, output: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    selected = [row for row in rows if row["cell_type"] in ("elementary", "cumulative_boundary")]
    labels = [f"{row['direction']}\n{row['cell_type']}\n{row['q0']}→{row['q1']}\n{row['p0']:.2f}→{row['p1']:.2f}" for row in selected]
    x = list(range(len(selected)))
    colors = ["#3B6FB6" if row["direction"] == "delta_r" else "#55A868" for row in selected]
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.85 * len(selected)), 10), sharex=True)
    axes[0].bar(x, [row["metrics"]["C_qp"] for row in selected], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$C_{qp}$")
    axes[1].bar(x, [row["metrics"]["maximum_edge_leakage"] for row in selected], color=colors)
    axes[1].axhline(1.0e-2, color="black", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("maximum edge leakage")
    axes[2].bar(x, [row["metrics"]["dipole_path_difference_normalized"] for row in selected], color=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("normalized dipole path difference")
    axes[2].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return True


def analyze(args):
    np, fci, gto = import_modules(require_pyscf=True)
    wavefunction_dir = args.workdir / "wavefunctions"
    wfns, mols = {}, {}
    for path in sorted(wavefunction_dir.glob("*.npz")):
        wfn = load(path, np)
        key = (wfn.tag, wfn.excited_weight)
        if key in wfns:
            raise RuntimeError(f"Duplicate wavefunction key {key}.")
        if len(wfn.ci) != 2 or wfn.dipole.shape != (3, 2, 2):
            raise RuntimeError(f"Malformed two-state checkpoint {path}.")
        wfns[key], mols[key] = wfn, molecule(wfn, gto)
    tags = sorted({tag for tag, _p in wfns})
    weights = sorted({p for _tag, p in wfns})
    expected = 5 * len(weights)
    if len(tags) != 5 or len(weights) < 3 or len(wfns) != expected:
        raise RuntimeError(f"Expected 5 geometries on >=3 weights, got {len(tags)} and {len(weights)} ({len(wfns)} states).")
    for tag in tags:
        for p in weights:
            if (tag, p) not in wfns:
                raise RuntimeError(f"Missing checkpoint {tag}, p={p}.")

    control_wfns, control_mols = {}, {}
    for path in sorted((args.workdir / "controls").glob("*.npz")):
        wfn = load(path, np)
        key = (wfn.tag, wfn.excited_weight)
        control_wfns[key] = wfn
        control_mols[key] = molecule(wfn, gto)
    control_weight = weights[0]
    if len(control_wfns) != len(tags) or any((tag, control_weight) not in control_wfns for tag in tags):
        raise RuntimeError(f"Expected one direct p={control_weight} history control for every geometry.")

    overlap_cache, edge_rows = {}, {}
    def overlap(key_x, key_y):
        cache_key = (key_x, key_y)
        if cache_key not in overlap_cache:
            overlap_cache[cache_key] = full_overlap(
                wfns[key_x], wfns[key_y], mols[key_x], mols[key_y], fci, gto, np
            )
            metrics = edge_metrics(overlap_cache[cache_key], np)
            edge_rows[cache_key] = {
                "source": state_id(key_x), "target": state_id(key_y),
                "source_tag": key_x[0], "target_tag": key_y[0],
                "source_p": f"{key_x[1]:.12e}", "target_p": f"{key_y[1]:.12e}",
                "sigma_1": f"{metrics['singular_values'][0]:.12e}",
                "sigma_2": f"{metrics['singular_values'][1]:.12e}",
                "sigma_min": f"{metrics['sigma_min']:.12e}",
                "leakage": f"{metrics['leakage']:.12e}",
                "permutation": json.dumps(metrics["permutation"]),
                "score": f"{metrics['score']:.12e}", "margin": f"{metrics['margin']:.12e}",
                "normalized_margin": f"{metrics['normalized_margin']:.12e}",
            }
        return overlap_cache[cache_key]

    direction_tags = {
        "delta_r": ["dr_minus_h_0p010000", "center", "dr_plus_h_0p010000"],
        "theta": ["theta_minus_h_0p200000", "center", "theta_plus_h_0p200000"],
    }
    cells = []
    seen = set()
    def add_cell(direction, q0, q1, p0, p1, cell_type):
        identity = (direction, q0, q1, p0, p1)
        if identity in seen:
            return
        seen.add(identity)
        a, b, c, d = (q0, p0), (q1, p0), (q0, p1), (q1, p1)
        edges = {"AB": overlap(a, b), "BD": overlap(b, d), "AC": overlap(a, c), "CD": overlap(c, d)}
        metrics = plaquette_metrics(edges, wfns[d].dipole, np)
        q_coordinate = (lambda tag: wfns[(tag, p0)].r1 - wfns[(tag, p0)].r2) if direction == "delta_r" else (lambda tag: wfns[(tag, p0)].theta)
        cells.append({
            "plaquette": f"{direction}:{q0}->{q1}:p{p0:.3f}->p{p1:.3f}",
            "direction": direction, "cell_type": cell_type, "q0": q0, "q1": q1,
            "p0": p0, "p1": p1, "q_span": abs(float(q_coordinate(q1) - q_coordinate(q0))),
            "p_span": p1 - p0, "vertices": {"A": state_id(a), "B": state_id(b), "C": state_id(c), "D": state_id(d)},
            "metrics": metrics,
        })

    def add_composed_boundary(direction, qtags):
        p0, p1 = weights[0], weights[-1]
        q0, q1 = qtags[0], qtags[-1]
        route_qp = []
        route_pq = []
        for index, (left, right) in enumerate(zip(qtags[:-1], qtags[1:])):
            route_qp.append((f"q_bottom_{index}", overlap((left, p0), (right, p0))))
        for index, (low, high) in enumerate(zip(weights[:-1], weights[1:])):
            route_qp.append((f"p_right_{index}", overlap((q1, low), (q1, high))))
        for index, (low, high) in enumerate(zip(weights[:-1], weights[1:])):
            route_pq.append((f"p_left_{index}", overlap((q0, low), (q0, high))))
        for index, (left, right) in enumerate(zip(qtags[:-1], qtags[1:])):
            route_pq.append((f"q_top_{index}", overlap((left, p1), (right, p1))))
        metrics = route_metrics(route_qp, route_pq, wfns[(q1, p1)].dipole, np)
        q_coordinate = (
            (lambda tag: wfns[(tag, p0)].r1 - wfns[(tag, p0)].r2)
            if direction == "delta_r" else
            (lambda tag: wfns[(tag, p0)].theta)
        )
        cells.append({
            "plaquette": f"{direction}:piecewise-boundary:p{p0:.3f}->p{p1:.3f}",
            "direction": direction, "cell_type": "cumulative_boundary",
            "q0": q0, "q1": q1, "p0": p0, "p1": p1,
            "q_span": abs(float(q_coordinate(q1) - q_coordinate(q0))),
            "p_span": p1 - p0,
            "vertices": {
                "A": state_id((q0, p0)), "D": state_id((q1, p1)),
                "q_grid": qtags, "p_grid": weights,
            },
            "metrics": metrics,
        })

    for direction, qtags in direction_tags.items():
        for q0, q1 in zip(qtags[:-1], qtags[1:]):
            for p0, p1 in zip(weights[:-1], weights[1:]):
                add_cell(direction, q0, q1, p0, p1, "elementary")
            add_cell(direction, q0, q1, weights[0], weights[-1], "cumulative_p")
        for p0, p1 in zip(weights[:-1], weights[1:]):
            add_cell(direction, qtags[0], qtags[-1], p0, p1, "cumulative_q")
        add_cell(direction, qtags[0], qtags[-1], weights[0], weights[-1], "cumulative_chord")
        add_composed_boundary(direction, qtags)

    thresholds = {
        "eta_edge_minimum": args.eta_min,
        "maximum_edge_leakage": args.leakage_max,
        "minimum_normalized_assignment_margin": args.assignment_margin_min,
        "minimum_C_qp": args.c_min,
        "minimum_directional_contrast": args.directional_contrast_min,
        "minimum_normalized_dipole_path_difference": args.dipole_difference_min,
        "maximum_final_orbital_gradient": args.gradient_max,
        "history_control_maximum_weighted_energy_difference_Eh": args.history_energy_max,
        "history_control_minimum_state_subspace_sigma": args.history_sigma_min,
        "history_control_maximum_state_subspace_leakage": args.history_leakage_max,
    }
    def strict(metrics):
        return (
            metrics["eta_edge"] >= args.eta_min
            and metrics["maximum_edge_leakage"] <= args.leakage_max
            and metrics["minimum_normalized_assignment_margin"] >= args.assignment_margin_min
        )
    elementary = [row for row in cells if row["cell_type"] == "elementary"]
    focus = next(row for row in cells if row["direction"] == "delta_r" and row["cell_type"] == "cumulative_boundary")
    bend_control = next(
        row for row in cells
        if row["direction"] == "theta" and row["cell_type"] == "cumulative_boundary"
    )
    fm, cm = focus["metrics"], bend_control["metrics"]
    gauge_floor = max(
        1.0e-15,
        max(row["metrics"]["gauge_test_C_abs_error"] for row in cells),
        max(row["metrics"]["gauge_test_dipole_abs_error"] for row in cells),
    )
    directional_contrast = fm["C_qp"] / max(cm["C_qp"], gauge_floor)
    failed_elementary = []
    for row in elementary:
        metrics = row["metrics"]
        reasons = []
        if metrics["eta_edge"] < args.eta_min:
            reasons.append("eta_edge")
        if metrics["maximum_edge_leakage"] > args.leakage_max:
            reasons.append("edge_leakage")
        if metrics["minimum_normalized_assignment_margin"] < args.assignment_margin_min:
            reasons.append("assignment_margin")
        if reasons:
            failed_elementary.append({
                "plaquette": row["plaquette"], "direction": row["direction"],
                "q0": row["q0"], "q1": row["q1"], "p0": row["p0"], "p1": row["p1"],
                "failed_gates": reasons, "eta_edge": metrics["eta_edge"],
                "maximum_edge_leakage": metrics["maximum_edge_leakage"],
                "minimum_normalized_assignment_margin": metrics["minimum_normalized_assignment_margin"],
                "C_qp": metrics["C_qp"],
            })
    history_rows = []
    for tag in tags:
        key = (tag, control_weight)
        primary, control_wfn = wfns[key], control_wfns[key]
        matrix = full_overlap(primary, control_wfn, mols[key], control_mols[key], fci, gto, np)
        metrics = edge_metrics(matrix, np)
        primary_e = (1-control_weight) * float(primary.energies[0]) + control_weight * float(primary.energies[1])
        control_e = (
            (1-control_weight) * float(control_wfn.energies[0])
            + control_weight * float(control_wfn.energies[1])
        )
        history_rows.append({
            "tag": tag, "excited_weight": f"{control_weight:.12e}",
            "weighted_energy_difference_Eh": f"{abs(primary_e-control_e):.12e}",
            "sigma_1": f"{metrics['singular_values'][0]:.12e}",
            "sigma_2": f"{metrics['singular_values'][1]:.12e}",
            "sigma_min": f"{metrics['sigma_min']:.12e}",
            "leakage": f"{metrics['leakage']:.12e}",
            "normalized_assignment_margin": f"{metrics['normalized_margin']:.12e}",
            "primary_final_gradient": f"{primary.final_gradient:.12e}",
            "control_final_gradient": f"{control_wfn.final_gradient:.12e}",
            "same_stationary_solution": (
                abs(primary_e-control_e) <= args.history_energy_max
                and metrics["sigma_min"] >= args.history_sigma_min
                and metrics["leakage"] <= args.history_leakage_max
            ),
        })
    checks = {
        "all_wavefunctions_stationary": all(
            math.isfinite(wfn.final_gradient) and wfn.final_gradient <= args.gradient_max
            for wfn in list(wfns.values()) + list(control_wfns.values())
        ),
        "direct_vs_staged_history_controls_same_solution": all(
            row["same_stationary_solution"] for row in history_rows
        ),
        "all_elementary_edges_strict": all(strict(row["metrics"]) for row in elementary),
        "flagship_cumulative_edges_strict": strict(fm),
        "conventional_composed_assignment_same": fm["same_composed_assignment"],
        "nonzero_geometric_signal": fm["C_qp"] >= args.c_min,
        "antisymmetric_stretch_vs_bend_contrast": directional_contrast >= args.directional_contrast_min,
        "observable_dipole_consequence": fm["dipole_path_difference_normalized"] >= args.dipole_difference_min,
        "gauge_invariance_resolved": gauge_floor <= args.gauge_error_max,
    }
    if not checks["all_wavefunctions_stationary"]:
        status = "STOP_STATIONARITY_FAILURE"
    elif not checks["direct_vs_staged_history_controls_same_solution"]:
        status = "STOP_HISTORY_DEPENDENCE"
    elif not checks["all_elementary_edges_strict"] or not checks["flagship_cumulative_edges_strict"]:
        status = "STOP_EDGE_FIDELITY_FAILURE"
    elif not checks["nonzero_geometric_signal"] or not checks["antisymmetric_stretch_vs_bend_contrast"]:
        status = "HOLD_WEAK_OR_NONDIRECTIONAL_SIGNAL"
    elif not checks["observable_dipole_consequence"]:
        status = "HOLD_NO_OBSERVABLE_CONSEQUENCE"
    elif not checks["conventional_composed_assignment_same"] or not checks["gauge_invariance_resolved"]:
        status = "HOLD_LOGIC_OR_GAUGE_FAILURE"
    else:
        status = "EXPAND_TO_REFINED_WEIGHTED_SA2_MAP"

    summary = {
        "gate": "O3 fixed-root weighted-SA2 pilot",
        "status": status,
        "manuscript_claim_authorized": False,
        "model": {
            "basis": next(iter(wfns.values())).basis,
            "active_space": [next(iter(wfns.values())).nelecas, next(iter(wfns.values())).ncas],
            "nroots": 2, "excited_root_weights": weights,
            "protocol_semantics": "weights=(1-p,p), with p<=0.5",
        },
        "counts": {"geometries": len(tags), "primary_wavefunctions": len(wfns), "history_controls": len(control_wfns), "plaquettes": len(cells)},
        "maximum_final_orbital_gradient": max(wfn.final_gradient for wfn in list(wfns.values()) + list(control_wfns.values())),
        "minimum_state_gap_Eh": min(float(wfn.energies[1] - wfn.energies[0]) for wfn in wfns.values()),
        "thresholds": thresholds,
        "decision_checks": checks,
        "elementary_gate_count": {
            "passed": len(elementary) - len(failed_elementary),
            "failed": len(failed_elementary), "total": len(elementary),
        },
        "failed_elementary_plaquettes": failed_elementary,
        "directional_contrast_C_delta_r_over_theta": directional_contrast,
        "numerical_gauge_floor": gauge_floor,
        "flagship_piecewise_boundary_antisymmetric_stretch": merge_cell_metrics(focus),
        "piecewise_boundary_bend_control": merge_cell_metrics(bend_control),
        "authorized_next_step": (
            "Refine q and p meshes and repeat the same hard gates; do not draft a flagship claim yet."
            if status == "EXPAND_TO_REFINED_WEIGHTED_SA2_MAP" else
            "Stop or redesign the O3 protocol axis according to the failed decision checks."
        ),
        "prohibited_interpretations": [
            "physical conical intersection proved by this pilot",
            "publication-ready molecular false negative before mesh refinement",
            "state-label swap inferred solely from a nonzero continuous holonomy",
        ],
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.workdir / "O3_weighted_SA2_edges.csv", list(edge_rows.values()))
    write_csv(args.workdir / "O3_weighted_SA2_history_controls.csv", history_rows)
    write_csv(args.workdir / "O3_weighted_SA2_plaquettes.csv", [flatten_cell(row) for row in cells])
    (args.workdir / "O3_weighted_SA2_detailed.json").write_text(
        json.dumps({"cells": cells}, indent=2) + "\n", encoding="utf-8"
    )
    (args.workdir / "O3_weighted_SA2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure = args.workdir / "O3_weighted_SA2_pilot.png"
    summary["figure_created"] = create_figure(cells, figure)
    (args.workdir / "O3_weighted_SA2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"PILOT_DECISION={status}")


def self_test():
    np, _fci, _gto = import_modules(require_pyscf=False)
    angle = 0.06
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    edges = {"AB": identity, "BD": rotation, "AC": identity, "CD": identity}
    dipole = np.asarray([np.diag([0.0, 1.0]), np.diag([-0.3, 0.7]), np.diag([0.2, -0.4])], dtype=complex)
    metrics = plaquette_metrics(edges, dipole, np)
    assert metrics["eta_edge"] > 1.0 - 1.0e-12
    assert metrics["maximum_edge_leakage"] < 1.0e-12
    assert metrics["minimum_normalized_assignment_margin"] > 0.99
    assert metrics["same_composed_assignment"]
    assert metrics["C_qp"] > 1.0e-3
    assert metrics["dipole_path_difference_normalized"] > 1.0e-3
    assert metrics["gauge_test_C_abs_error"] < 1.0e-12
    assert metrics["gauge_test_dipole_abs_error"] < 1.0e-12
    half = np.asarray(
        [[math.cos(angle/2), -math.sin(angle/2)],
         [math.sin(angle/2), math.cos(angle/2)]], dtype=complex,
    )
    piecewise = route_metrics(
        [("q0", identity), ("q1", identity), ("p0", half), ("p1", half)],
        [("p2", identity), ("p3", identity), ("q2", identity), ("q3", identity)],
        dipole, np,
    )
    assert abs(piecewise["C_qp"] - metrics["C_qp"]) < 1.0e-12
    assert piecewise["gauge_test_C_abs_error"] < 1.0e-12
    assert piecewise["gauge_test_dipole_abs_error"] < 1.0e-12
    merged = merge_cell_metrics({"direction": "theta", "metrics": {"C_qp": 0.0}})
    assert merged == {"direction": "theta", "C_qp": 0.0}
    print("SELF_TEST=PASS")
    print(json.dumps({k: v for k, v in metrics.items() if k != "edges"}, indent=2))


def make_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("O3_weighted_SA2_pilot_work"))
    parser.add_argument("--eta-min", type=float, default=0.98)
    parser.add_argument("--leakage-max", type=float, default=0.01)
    parser.add_argument("--assignment-margin-min", type=float, default=0.10)
    parser.add_argument("--c-min", type=float, default=1.0e-5)
    parser.add_argument("--directional-contrast-min", type=float, default=10.0)
    parser.add_argument("--dipole-difference-min", type=float, default=1.0e-5)
    parser.add_argument("--gauge-error-max", type=float, default=1.0e-10)
    parser.add_argument("--gradient-max", type=float, default=1.0e-4)
    parser.add_argument("--history-energy-max", type=float, default=1.0e-8)
    parser.add_argument("--history-sigma-min", type=float, default=0.9999)
    parser.add_argument("--history-leakage-max", type=float, default=1.0e-6)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.workdir = args.workdir.expanduser().resolve()
    analyze(args)


if __name__ == "__main__":
    main()
