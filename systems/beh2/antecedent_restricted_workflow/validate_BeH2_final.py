#!/usr/bin/env python3
"""Scientific acceptance checks for the clean BeH2 production rerun."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--work-root", type=Path, required=True)
    return result


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def assert_model(config: dict, label: str) -> None:
    if str(config.get("basis", "")).lower() != "cc-pvtz":
        raise AssertionError(f"{label}: basis is not cc-pVTZ")
    if config.get("active_space") != {"electrons": 6, "orbitals": 6}:
        raise AssertionError(f"{label}: active space is not CAS(6,6)")
    if config.get("freeze_core") is not True:
        raise AssertionError(f"{label}: Be 1s is not frozen")


def exponent(rows: list[dict[str, str]], route: str) -> float:
    selected = [row for row in rows if row["route"] == route]
    if len(selected) != 4:
        raise AssertionError(f"Expected four refinement widths for {route}")
    x = np.log([float(row["delta_q_bohr"]) for row in selected])
    y = np.log([float(row["curvature_max"]) for row in selected])
    return float(np.polyfit(x, y, 1)[0])


def check_figure(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty figure: {path}")
    if path.suffix.lower() not in {".png", ".tiff"}:
        return {"bytes": path.stat().st_size}
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        mode = image.mode
        size = image.size
    if min(float(dpi[0]), float(dpi[1])) < 599.0:
        raise AssertionError(f"Figure is not 600 dpi: {path} ({dpi})")
    if mode != "RGB":
        raise AssertionError(f"Figure is not RGB: {path} ({mode})")
    return {
        "bytes": path.stat().st_size,
        "dpi": [float(dpi[0]), float(dpi[1])],
        "mode": mode,
        "pixels": [int(size[0]), int(size[1])],
    }


def main() -> int:
    args = parser().parse_args()
    root = args.work_root.resolve()
    configs = {
        "B0": read_json(root / "B0" / "B0_config.json"),
        "B1": read_json(root / "B1" / "B1_config.json"),
        "B2": read_json(root / "B2" / "B2_config.json"),
        "B3R": read_json(root / "B3R" / "B3R_config.json"),
    }
    for label, config in configs.items():
        assert_model(config, label)

    failures = []
    for stage in ("B0", "B1", "B2", "B3R"):
        failure_dir = root / stage / "failures"
        failures.extend(str(path) for path in failure_dir.glob("**/*.json"))
    if failures:
        raise AssertionError(f"Failure records remain: {failures}")

    summary = read_json(root / "B3R" / "B3R_summary.json")
    if summary.get("status") != "PASS_BRANCH_RETAINED":
        raise AssertionError(f"B3R branch acceptance failed: {summary.get('status')}")
    if summary.get("solution_count") != 84:
        raise AssertionError("B3R does not contain 84 analysis-grid solutions")
    if summary.get("plaquette_count") != 138:
        raise AssertionError("B3R does not contain 138 multiscale plaquettes")
    validation = summary["validation"]
    for key in (
        "maximum_formula_error",
        "maximum_curvature_gauge_error",
        "maximum_raw_nonclosure_gauge_error",
        "maximum_polar_unitarity_error",
        "maximum_reverse_overlap_error",
    ):
        if float(validation[key]) > 1.0e-9:
            raise AssertionError(f"Numerical validation failed: {key}={validation[key]}")

    with (root / "B3R" / "B3R_scaling.csv").open(newline="", encoding="utf-8") as stream:
        scaling = list(csv.DictReader(stream))
    alpha_upper = exponent(scaling, "upper_down")
    alpha_unique = exponent(scaling, "unique_up")
    if not (-0.20 <= alpha_upper <= 0.20):
        raise AssertionError(f"Refinement-resistant branch was not recovered: alpha={alpha_upper}")
    if not (0.70 <= alpha_unique <= 1.30):
        raise AssertionError(f"Smooth control is not approximately linear: alpha={alpha_unique}")

    finest = [row for row in scaling if int(row["stride"]) == 1]
    by_route = {row["route"]: row for row in finest}
    upper_peak = float(by_route["upper_down"]["curvature_max"])
    unique_peak = float(by_route["unique_up"]["curvature_max"])
    eta_peak = float(by_route["upper_down"]["eta_at_curvature_peak"])
    leakage_upper = float(by_route["upper_down"]["leakage_AB_max"])
    leakage_unique = float(by_route["unique_up"]["leakage_AB_max"])
    if upper_peak <= 10.0 * unique_peak:
        raise AssertionError("Upper-down nonclosure is not localized above the smooth control")
    if eta_peak < 0.85:
        raise AssertionError(f"Peak plaquette is poorly conditioned: eta={eta_peak}")
    if leakage_upper < 1.0e-2 or leakage_unique > 1.0e-4:
        raise AssertionError("Nuclear-edge leakage does not separate collapse and smooth histories")

    figures = {}
    for name in (
        "BeH2_B3R_validation.png",
        "BeH2_B3R_validation.tiff",
        "BeH2_B3R_validation.pdf",
        "BeH2_B3R_components.png",
        "BeH2_B3R_components.tiff",
        "BeH2_B3R_components.pdf",
    ):
        figures[name] = check_figure(root / "figures" / name)

    report = {
        "status": "PASS",
        "locked_model": "SA3/SA4-CASSCF(6,6)/cc-pVTZ; frozen Be 1s; m=3",
        "alpha_upper_down": alpha_upper,
        "alpha_unique_up": alpha_unique,
        "finest_upper_down_curvature_max": upper_peak,
        "finest_unique_up_curvature_max": unique_peak,
        "eta_at_upper_down_peak": eta_peak,
        "finest_upper_down_leakage_AB_max": leakage_upper,
        "finest_unique_up_leakage_AB_max": leakage_unique,
        "matrix_validation": validation,
        "figures": figures,
    }
    (root / "acceptance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# BeH2 final-rerun acceptance: PASS",
        "",
        "- Model: SA3/SA4-CASSCF(6,6)/cc-pVTZ; frozen Be 1s; retained manifold m=3.",
        f"- Upper-down scaling exponent: {alpha_upper:.6f}.",
        f"- Unique-up scaling exponent: {alpha_unique:.6f}.",
        f"- Finest-grid peak conditioning: eta_edge = {eta_peak:.6f}.",
        "- Gauge, polar-unitarity, reverse-overlap, and formula checks: PASS.",
        "- Publication PNG/TIFF files: RGB, 600 dpi.",
    ]
    (root / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("BeH2 final scientific acceptance: PASS")
    print(f"alpha(upper-down) = {alpha_upper:.12f}")
    print(f"alpha(unique-up)  = {alpha_unique:.12f}")
    print(root / "acceptance_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
