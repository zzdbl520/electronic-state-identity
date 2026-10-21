#!/usr/bin/env python3
"""Confirm the S1-selected LiH physical-field window with opposed histories.

S2 is deliberately confirmatory.  It does not relocate the window or alter a
threshold.  It reads the fixed S1 grid and selected direct-edge rectangle,
then traverses the complete grid in two exactly opposed snake paths.  Every
history solution is compared with the independent S1 solution at the same
(R,Fz) vertex using energy, exact active-only state-subspace overlap, and
active-orbital overlap.  The selected-window holonomy and dipole consequence
must also be invariant to the starting history.

Only ``PUBLICATION_DEMONSTRATION_READY`` authorizes manuscript integration.
Any earlier failure is a stop condition, not a license to move the window or
relax a gate.

Typical use
-----------
python LiH_S2_physical_field_confirmation.py selftest
python LiH_S2_physical_field_confirmation.py all \
  --root /mnt/LiH/LiH_S1_field_locator_work --threads 8
"""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import LiH_S0_SA4_qualification as s0
import LiH_S1_physical_field_locator as s1


SCRIPT_VERSION = "LiH-S2-physical-field-confirmation-v1.0"
HISTORIES = ("forward_snake", "reverse_snake")


@dataclass(frozen=True)
class Thresholds:
    history_maximum_average_energy_difference_Eh: float = 1.0e-8
    history_minimum_state_subspace_sigma: float = 0.9999
    history_maximum_state_subspace_leakage: float = 1.0e-6
    history_minimum_active_orbital_sigma: float = 0.9999
    history_maximum_active_orbital_leakage: float = 1.0e-6
    maximum_final_orbital_gradient: float = 1.0e-4
    maximum_spin_square: float = 1.0e-7
    maximum_mo_orthonormality_error: float = 1.0e-9
    maximum_probe_retained_energy_error_Eh: float = 1.0e-8
    maximum_history_C_qp_difference: float = 1.0e-8
    maximum_history_normalized_dipole_difference: float = 1.0e-8
    eta_edge_minimum: float = 0.98
    maximum_edge_leakage: float = 0.01
    minimum_normalized_assignment_margin: float = 0.10
    active_orbital_subspace_sigma_minimum: float = 0.90
    active_orbital_subspace_leakage_maximum: float = 0.10
    target_pair_maximum_gap_Eh: float = 1.0e-2
    minimum_C_qp: float = 1.0e-4
    minimum_normalized_dipole_path_difference: float = 1.0e-5


def history_solution_path(
    root: Path, history: str, r_angstrom: float, field_au: float
) -> Path:
    return (
        root
        / "history_solutions"
        / history
        / s1.field_tag(field_au)
        / (s0.float_tag("R", r_angstrom) + ".npz")
    )


def snake_indices(nr: int, nf: int) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    for j in range(nf):
        row = range(nr) if j % 2 == 0 else range(nr - 1, -1, -1)
        path.extend((i, j) for i in row)
    return path


def run_history(
    root: Path,
    protocol: s1.Protocol,
    history: str,
    order: Sequence[Tuple[int, int]],
    overwrite: bool,
) -> None:
    previous: Tuple[Any, np.ndarray] | None = None
    for i, j in order:
        r = protocol.r_grid_angstrom[i]
        field = protocol.field_grid_au[j]
        target = history_solution_path(root, history, r, field)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            sol = s1.load_solution(target)
            previous = (s0.make_mol(r, protocol.basis, verbose=0), sol["mo_coeff"])
            print(
                f"SKIP {history} R={r:.8f} Fz={field:+.8f}: {target}", flush=True
            )
            continue
        print(f"RUN {history} R={r:.8f} Fz={field:+.8f}", flush=True)
        try:
            md = s1.casscf_job(
                r,
                field,
                protocol,
                target,
                initial_from=previous,
                history=history,
            )
        except Exception as exc:
            s0.write_json(
                target.with_suffix(".failed.json"),
                {
                    "script_version": SCRIPT_VERSION,
                    "history": history,
                    "R_angstrom": r,
                    "field_z_au": field,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        previous = (s0.make_mol(r, protocol.basis, verbose=0), s1.load_solution(target)["mo_coeff"])
        print(
            f"DONE {history} R={r:.8f} Fz={field:+.8f} "
            f"gap23={md['target_pair_gap_Eh']:.6e} "
            f"grad={md['maximum_orbital_gradient']:.3e}",
            flush=True,
        )


def run_protocol(root: Path, protocol: s1.Protocol, overwrite: bool) -> None:
    order = snake_indices(len(protocol.r_grid_angstrom), len(protocol.field_grid_au))
    manifest = {
        "script_version": SCRIPT_VERSION,
        "gate": "LiH S2 opposed-history physical-field confirmation",
        "protocol": asdict(protocol),
        "thresholds": asdict(Thresholds()),
        "history_semantics": {
            "forward_snake": order,
            "reverse_snake": list(reversed(order)),
            "first_vertex_of_each_history": "independent field-specific RHF start",
            "subsequent_vertices": "projected preceding converged CASSCF orbitals",
            "window_selection": "read unchanged from LiH_S1_decision.json",
        },
    }
    s0.write_json(root / "LiH_S2_manifest.json", manifest)
    run_history(root, protocol, "forward_snake", order, overwrite)
    run_history(root, protocol, "reverse_snake", list(reversed(order)), overwrite)
    print("LiH S2 history wavefunctions complete. Run the analyzer or use 'all'.", flush=True)


def edge_pass(record: Mapping[str, Any], thresholds: Thresholds) -> bool:
    sm = record["state_subspace"]
    om = record["active_orbital_subspace"]
    return bool(
        sm["sigma_min"] >= thresholds.eta_edge_minimum
        and sm["leakage"] <= thresholds.maximum_edge_leakage
        and sm["normalized_margin"] >= thresholds.minimum_normalized_assignment_margin
        and om["sigma_min"] >= thresholds.active_orbital_subspace_sigma_minimum
        and om["leakage"] <= thresholds.active_orbital_subspace_leakage_maximum
    )


def window_result(
    solution_set: Mapping[Tuple[int, int], Mapping[str, Any]],
    mols: Mapping[int, Any],
    bounds: Tuple[int, int, int, int],
    terms_cache: MutableMapping[Tuple[str, int], Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    i0, i1, j0, j1 = bounds
    a, b, c, d = (i0, j0), (i1, j0), (i0, j1), (i1, j1)

    def one_edge(x: Tuple[int, int], y: Tuple[int, int], name: str) -> Dict[str, Any]:
        matrix = s0.active_overlap_matrix(
            solution_set[x], solution_set[y], mols[x[0]], mols[y[0]], terms_cache
        )
        obj = s0.edge_metrics(matrix)
        return {
            "edge": name,
            "object": obj,
            "state_subspace": obj["metrics"],
            "active_orbital_subspace": s0.active_orbital_metrics(
                solution_set[x], solution_set[y], mols[x[0]], mols[y[0]]
            ),
        }

    ab = one_edge(a, b, "AB:nuclear")
    bd = one_edge(b, d, "BD:physical_field")
    ac = one_edge(a, c, "AC:physical_field")
    cd = one_edge(c, d, "CD:nuclear")
    route1 = ab["object"]["Q"] @ bd["object"]["Q"]
    route2 = ac["object"]["Q"] @ cd["object"]["Q"]
    route = s1.route_result(route1, route2, solution_set[d]["dipole_au"])
    enclosed = [
        (i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
    ]
    direct_edges = [ab, bd, ac, cd]
    min_gap = min(
        float(solution_set[v]["metadata"]["target_pair_gap_Eh"]) for v in enclosed
    )
    candidate = bool(
        all(edge_pass(x, thresholds) for x in direct_edges)
        and route["same_composed_assignment"]
        and min_gap <= thresholds.target_pair_maximum_gap_Eh
        and route["C_qp"] >= thresholds.minimum_C_qp
        and route["dipole_path_difference_normalized"]
        >= thresholds.minimum_normalized_dipole_path_difference
    )
    return {
        **route,
        "minimum_target_pair_gap_inside_window_Eh": min_gap,
        "all_four_direct_edges_strict": all(
            edge_pass(x, thresholds) for x in direct_edges
        ),
        "candidate_window": candidate,
        "edges": {
            name: {k: v for k, v in edge.items() if k != "object"}
            for name, edge in zip(("AB", "BD", "AC", "CD"), direct_edges)
        },
    }


def analyze_protocol(
    root: Path, protocol: s1.Protocol, thresholds: Thresholds
) -> Dict[str, Any]:
    s1_decision_path = root / "LiH_S1_decision.json"
    if not s1_decision_path.exists():
        raise FileNotFoundError(
            f"S2 requires the completed S1 decision at {s1_decision_path}"
        )
    import json

    s1_decision = json.loads(s1_decision_path.read_text())
    selected = s1_decision.get("selected_candidate_window")
    if not selected:
        raise RuntimeError("S1 did not select a candidate window")
    rs = protocol.r_grid_angstrom
    fs = protocol.field_grid_au

    def locate(values: Sequence[float], target: float) -> int:
        matches = [i for i, value in enumerate(values) if abs(value - target) < 1.0e-10]
        if len(matches) != 1:
            raise RuntimeError(f"selected boundary {target} is not unique on the S1 grid")
        return matches[0]

    bounds = (
        locate(rs, float(selected["R0_angstrom"])),
        locate(rs, float(selected["R1_angstrom"])),
        locate(fs, float(selected["F0_au"])),
        locate(fs, float(selected["F1_au"])),
    )
    mols = {i: s0.make_mol(r, protocol.basis, verbose=0) for i, r in enumerate(rs)}
    direct = {
        (i, j): s1.load_solution(s1.solution_path(root, r, field))
        for i, r in enumerate(rs)
        for j, field in enumerate(fs)
    }
    histories = {
        history: {
            (i, j): s1.load_solution(history_solution_path(root, history, r, field))
            for i, r in enumerate(rs)
            for j, field in enumerate(fs)
        }
        for history in HISTORIES
    }

    terms_cache: MutableMapping[Tuple[str, int], Any] = {}
    controls: List[Dict[str, Any]] = []
    for history in HISTORIES:
        for vertex, reference in direct.items():
            control = histories[history][vertex]
            matrix = s0.active_overlap_matrix(
                reference,
                control,
                mols[vertex[0]],
                mols[vertex[0]],
                terms_cache,
            )
            state = s0.edge_metrics(matrix)["metrics"]
            orbital = s0.active_orbital_metrics(
                reference, control, mols[vertex[0]], mols[vertex[0]]
            )
            controls.append(
                {
                    "history": history,
                    "R_angstrom": rs[vertex[0]],
                    "field_z_au": fs[vertex[1]],
                    "average_energy_difference_Eh": abs(
                        float(reference["metadata"]["average_energy_Eh"])
                        - float(control["metadata"]["average_energy_Eh"])
                    ),
                    "state_subspace": state,
                    "active_orbital_subspace": orbital,
                }
            )

    def history_control_pass(control: Mapping[str, Any]) -> bool:
        return bool(
            control["average_energy_difference_Eh"]
            <= thresholds.history_maximum_average_energy_difference_Eh
            and control["state_subspace"]["sigma_min"]
            >= thresholds.history_minimum_state_subspace_sigma
            and control["state_subspace"]["leakage"]
            <= thresholds.history_maximum_state_subspace_leakage
            and control["active_orbital_subspace"]["sigma_min"]
            >= thresholds.history_minimum_active_orbital_sigma
            and control["active_orbital_subspace"]["leakage"]
            <= thresholds.history_maximum_active_orbital_leakage
        )

    all_history_solutions = [x for h in histories.values() for x in h.values()]
    maximum_gradient = max(
        float(x["metadata"]["maximum_orbital_gradient"]) for x in all_history_solutions
    )
    maximum_s2 = max(
        abs(float(value))
        for x in all_history_solutions
        for value in x["metadata"]["state_S2"]
    )
    maximum_mo_error = max(
        float(x["metadata"]["mo_orthonormality_max_error"])
        for x in all_history_solutions
    )
    maximum_probe_error = max(
        float(x["metadata"]["probe_retained_energy_max_error_Eh"])
        for x in all_history_solutions
    )
    stationary = bool(
        all(bool(x["metadata"]["converged"]) for x in all_history_solutions)
        and maximum_gradient <= thresholds.maximum_final_orbital_gradient
        and maximum_s2 <= thresholds.maximum_spin_square
        and maximum_mo_error <= thresholds.maximum_mo_orthonormality_error
        and maximum_probe_error <= thresholds.maximum_probe_retained_energy_error_Eh
    )
    histories_same = all(history_control_pass(x) for x in controls)

    windows = {
        "direct": window_result(direct, mols, bounds, terms_cache, thresholds),
        **{
            history: window_result(histories[history], mols, bounds, terms_cache, thresholds)
            for history in HISTORIES
        },
    }
    c_errors = {
        history: abs(windows[history]["C_qp"] - windows["direct"]["C_qp"])
        for history in HISTORIES
    }
    dipole_errors = {
        history: abs(
            windows[history]["dipole_path_difference_normalized"]
            - windows["direct"]["dipole_path_difference_normalized"]
        )
        for history in HISTORIES
    }
    window_invariant = bool(
        all(windows[name]["candidate_window"] for name in ("direct",) + HISTORIES)
        and max(c_errors.values()) <= thresholds.maximum_history_C_qp_difference
        and max(dipole_errors.values())
        <= thresholds.maximum_history_normalized_dipole_difference
    )
    s1_ready = s1_decision.get("status") == "CANDIDATE_WINDOW_FOUND"

    if not s1_ready:
        status = "STOP_S1_NOT_READY"
    elif not stationary:
        status = "STOP_WAVEFUNCTION_FAILURE"
    elif not histories_same:
        status = "STOP_HISTORY_CONTROL_FAILURE"
    elif not window_invariant:
        status = "STOP_WINDOW_REPRODUCTION_FAILURE"
    else:
        status = "PUBLICATION_DEMONSTRATION_READY"

    failed_controls = [x for x in controls if not history_control_pass(x)]
    nonstationary = [
        {
            "path": x["path"],
            "history": x["metadata"].get("history"),
            "R_angstrom": x["metadata"]["R_angstrom"],
            "field_z_au": x["metadata"]["field_z_au"],
            "converged": x["metadata"]["converged"],
            "maximum_orbital_gradient": x["metadata"]["maximum_orbital_gradient"],
            "state_S2": x["metadata"]["state_S2"],
            "mo_orthonormality_max_error": x["metadata"]["mo_orthonormality_max_error"],
            "probe_retained_energy_max_error_Eh": x["metadata"][
                "probe_retained_energy_max_error_Eh"
            ],
        }
        for x in all_history_solutions
        if not bool(x["metadata"]["converged"])
        or float(x["metadata"]["maximum_orbital_gradient"])
        > thresholds.maximum_final_orbital_gradient
        or max(abs(float(v)) for v in x["metadata"]["state_S2"])
        > thresholds.maximum_spin_square
        or float(x["metadata"]["mo_orthonormality_max_error"])
        > thresholds.maximum_mo_orthonormality_error
        or float(x["metadata"]["probe_retained_energy_max_error_Eh"])
        > thresholds.maximum_probe_retained_energy_error_Eh
    ]

    decision = {
        "gate": "LiH S2 opposed-history confirmation of physical R x Fz hotspot",
        "status": status,
        "manuscript_claim_authorized": status == "PUBLICATION_DEMONSTRATION_READY",
        "script_version": SCRIPT_VERSION,
        "input_S1_status": s1_decision.get("status"),
        "fixed_selected_window": {
            "R0_angstrom": selected["R0_angstrom"],
            "R1_angstrom": selected["R1_angstrom"],
            "F0_au": selected["F0_au"],
            "F1_au": selected["F1_au"],
            "selection_source": "unchanged LiH_S1_decision.json",
        },
        "thresholds": asdict(thresholds),
        "decision_checks": {
            "S1_candidate_window_found": s1_ready,
            "all_history_wavefunctions_stationary": stationary,
            "direct_vs_two_opposed_histories_same_solution": histories_same,
            "selected_window_candidate_under_all_histories": all(
                windows[name]["candidate_window"] for name in ("direct",) + HISTORIES
            ),
            "selected_window_C_qp_history_invariant": max(c_errors.values())
            <= thresholds.maximum_history_C_qp_difference,
            "selected_window_dipole_consequence_history_invariant": max(
                dipole_errors.values()
            )
            <= thresholds.maximum_history_normalized_dipole_difference,
            "conventional_composed_assignment_same": windows["direct"][
                "same_composed_assignment"
            ],
            "nonzero_continuous_holonomy": windows["direct"]["C_qp"]
            >= thresholds.minimum_C_qp,
            "observable_dipole_consequence": windows["direct"][
                "dipole_path_difference_normalized"
            ]
            >= thresholds.minimum_normalized_dipole_path_difference,
        },
        "counts": {
            "grid_vertices": len(direct),
            "opposed_histories": len(HISTORIES),
            "history_wavefunctions": len(all_history_solutions),
            "history_controls": len(controls),
            "failed_history_controls": len(failed_controls),
        },
        "maximum_final_orbital_gradient": maximum_gradient,
        "maximum_state_S2": maximum_s2,
        "maximum_mo_orthonormality_error": maximum_mo_error,
        "maximum_probe_retained_energy_error_Eh": maximum_probe_error,
        "maximum_history_average_energy_difference_Eh": max(
            x["average_energy_difference_Eh"] for x in controls
        ),
        "minimum_history_state_subspace_sigma": min(
            x["state_subspace"]["sigma_min"] for x in controls
        ),
        "maximum_history_state_subspace_leakage": max(
            x["state_subspace"]["leakage"] for x in controls
        ),
        "minimum_history_active_orbital_sigma": min(
            x["active_orbital_subspace"]["sigma_min"] for x in controls
        ),
        "maximum_history_active_orbital_leakage": max(
            x["active_orbital_subspace"]["leakage"] for x in controls
        ),
        "selected_window_results": windows,
        "selected_window_C_qp_history_abs_errors": c_errors,
        "selected_window_normalized_dipole_history_abs_errors": dipole_errors,
        "failed_history_controls": failed_controls,
        "nonstationary_wavefunctions": nonstationary,
        "authorized_interpretation": (
            "In the fixed LiH R x static-field window, all direct edges are faithful and conventional composed assignments agree, yet continuous subspace transport is path dependent and produces a gauge-invariant dipole consequence; the result is stationary and independent of two opposed optimization histories."
            if status == "PUBLICATION_DEMONSTRATION_READY"
            else "The fixed LiH physical-field demonstration has not passed all confirmatory gates."
        ),
        "prohibited_interpretations": [
            "a conical intersection proved solely by the avoided crossing",
            "a state-label swap inferred solely from continuous holonomy",
            "history independence inferred from energy alone",
            "post-confirmation movement of the selected window",
            "replacement of active-only overlap by scalar inactive-core division",
        ],
    }
    s0.write_json(root / "LiH_S2_decision.json", decision)
    s0.write_json(
        root / "LiH_S2_diagnostics.json",
        {"history_controls": controls, "selected_window_results": windows},
    )

    print("\nLiH S2 RESULT", flush=True)
    print(f"Status: {status}", flush=True)
    print(
        f"History controls: max dE={decision['maximum_history_average_energy_difference_Eh']:.3e} Eh; "
        f"min sigma={decision['minimum_history_state_subspace_sigma']:.12f}; "
        f"max leakage={decision['maximum_history_state_subspace_leakage']:.3e}",
        flush=True,
    )
    print(
        f"Selected window: C_qp={windows['direct']['C_qp']:.12e}; "
        f"normalized dipole difference={windows['direct']['dipole_path_difference_normalized']:.12e}; "
        f"max history C error={max(c_errors.values()):.3e}",
        flush=True,
    )
    print(f"Decision file: {root / 'LiH_S2_decision.json'}", flush=True)
    return decision


def selftest() -> None:
    path = snake_indices(5, 5)
    assert len(path) == 25 and len(set(path)) == 25
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    assert list(reversed(path))[0] == path[-1]
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        for a, b in zip(path[:-1], path[1:])
    )
    print("SELFTEST=PASS")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    for name in ("run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument(
            "--root", type=Path, default=Path("/mnt/LiH/LiH_S1_field_locator_work")
        )
        p.add_argument("--basis", default="aug-cc-pvdz")
        p.add_argument("--r-grid", default="5.8:6.0:0.05")
        p.add_argument("--field-grid", default="0.0105:0.0115:0.00025")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    s0.require_pyscf()
    import pyscf

    pyscf.lib.num_threads(args.threads)
    protocol = s1.Protocol(
        basis=args.basis,
        r_grid_angstrom=s0.parse_grid(args.r_grid),
        field_grid_au=s0.parse_grid(args.field_grid),
    )
    if args.command in ("run", "all"):
        run_protocol(args.root, protocol, args.overwrite)
    if args.command in ("analyze", "all"):
        analyze_protocol(args.root, protocol, Thresholds())


if __name__ == "__main__":
    main()
