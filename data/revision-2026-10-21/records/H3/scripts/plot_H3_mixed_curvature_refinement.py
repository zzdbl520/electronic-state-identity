#!/usr/bin/env python3
"""Create the H3 mixed-curvature refinement figure and source-data tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLORS = {36: "#56B4E9", 72: "#E69F00", 144: "#202020"}
MARKERS = {36: "o", 72: "s", 144: "^"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Plot 36/72/144-point H3 mixed-curvature refinement."
    )
    for n in (36, 72, 144):
        result.add_argument(f"--profile{n}", type=Path)
        result.add_argument(f"--summary{n}", type=Path)
    result.add_argument(
        "--combined-profile",
        type=Path,
        help="Previously generated combined profile source-data CSV.",
    )
    result.add_argument(
        "--combined-scaling",
        type=Path,
        help="Previously generated angular-refinement scaling CSV.",
    )
    result.add_argument(
        "--output-prefix", type=Path, default=Path("H3_mixed_curvature_refinement")
    )
    result.add_argument("--dpi", type=int, default=600)
    return result


def read_profile(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append({key: float(value) for key, value in raw.items()})
    return rows


def read_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_combined_profiles(path: Path) -> dict[int, list[dict[str, float]]]:
    profiles: dict[int, list[dict[str, float]]] = {36: [], 72: [], 144: []}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            points = int(raw["points"])
            profiles[points].append(
                {
                    "index": float(raw["index"]),
                    "phi_start_deg": float(raw["phi_start_deg"]),
                    "phi_end_deg": float(raw["phi_end_deg"]),
                    "phi_mid_deg": float(raw["phi_mid_deg"]),
                    "curvature": float(raw["curvature"]),
                    "eta_edge": float(raw["eta_edge"]),
                    "lambda_AB_nuclear_SA2": float(raw["lambda_nuclear_SA2"]),
                    "lambda_CD_nuclear_SA3": float(raw["lambda_nuclear_SA3"]),
                    "lambda_AC_protocol": float(raw["lambda_protocol_at_start"]),
                    "lambda_BD_protocol": float(raw["lambda_protocol_at_end"]),
                }
            )
    return profiles


def read_scaling_source(path: Path) -> list[dict[str, float | int]]:
    integer_fields = {"points", "curvature_cells_above_2p5e-4", "SA3_exchange_preferred_edges"}
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    key: int(value) if key in integer_fields else float(value)
                    for key, value in raw.items()
                }
            )
    return rows


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def write_profile_source(
    path: Path, profiles: dict[int, list[dict[str, float]]]
) -> None:
    fieldnames = [
        "points",
        "step_deg",
        "index",
        "phi_start_deg",
        "phi_end_deg",
        "phi_mid_deg",
        "curvature",
        "eta_edge",
        "lambda_nuclear_SA2",
        "lambda_nuclear_SA3",
        "lambda_protocol_at_start",
        "lambda_protocol_at_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for n in sorted(profiles):
            for row in profiles[n]:
                writer.writerow(
                    {
                        "points": n,
                        "step_deg": 360.0 / n,
                        "index": int(row["index"]),
                        "phi_start_deg": row["phi_start_deg"],
                        "phi_end_deg": row["phi_end_deg"],
                        "phi_mid_deg": row["phi_mid_deg"],
                        "curvature": row["curvature"],
                        "eta_edge": row["eta_edge"],
                        "lambda_nuclear_SA2": row["lambda_AB_nuclear_SA2"],
                        "lambda_nuclear_SA3": row["lambda_CD_nuclear_SA3"],
                        "lambda_protocol_at_start": row["lambda_AC_protocol"],
                        "lambda_protocol_at_end": row["lambda_BD_protocol"],
                    }
                )


def scaling_rows(
    profiles: dict[int, list[dict[str, float]]], summaries: dict[int, dict]
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for n in sorted(profiles):
        profile = profiles[n]
        curvature = np.asarray([row["curvature"] for row in profile])
        lambda2 = np.asarray([row["lambda_AB_nuclear_SA2"] for row in profile])
        lambda3 = np.asarray([row["lambda_CD_nuclear_SA3"] for row in profile])
        eta = np.asarray([row["eta_edge"] for row in profile])
        summary = summaries[n]
        rows.append(
            {
                "points": n,
                "step_deg": 360.0 / n,
                "step_rad": math.radians(360.0 / n),
                "curvature_max": float(np.max(curvature)),
                "curvature_mean": float(np.mean(curvature)),
                "curvature_rms": float(np.sqrt(np.mean(curvature**2))),
                "curvature_sum": float(np.sum(curvature)),
                "curvature_cells_above_2p5e-4": int(
                    np.count_nonzero(curvature > 2.5e-4)
                ),
                "lambda_nuclear_SA2_max": float(np.max(lambda2)),
                "lambda_nuclear_SA3_max": float(np.max(lambda3)),
                "eta_edge_min": float(np.min(eta)),
                "SA3_exchange_preferred_edges": int(
                    summary["adiabatic_assignment"]["SA3"][
                        "number_of_exchange_preferred_edges"
                    ]
                ),
                "berry_phase_SA2_state0_over_pi": float(
                    summary["berry_phases"]["SA2"][0]["phase_over_pi"]
                ),
                "berry_phase_SA2_state1_over_pi": float(
                    summary["berry_phases"]["SA2"][1]["phase_over_pi"]
                ),
                "berry_phase_SA3_state0_over_pi": float(
                    summary["berry_phases"]["SA3"][0]["phase_over_pi"]
                ),
                "berry_phase_SA3_state1_over_pi": float(
                    summary["berry_phases"]["SA3"][1]["phase_over_pi"]
                ),
            }
        )
    return rows


def write_scaling_source(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    profiles: dict[int, list[dict[str, float]]],
    rows: list[dict[str, float | int]],
    png_path: Path,
    tiff_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.5,
            "mathtext.fontset": "stixsans",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for n in (36, 72, 144):
        phi = np.asarray([row["phi_mid_deg"] for row in profiles[n]])
        curvature = np.asarray([row["curvature"] for row in profiles[n]])
        ax_a.step(
            phi,
            1.0e4 * curvature,
            where="mid",
            color=COLORS[n],
            lw={36: 1.8, 72: 1.35, 144: 0.9}[n],
            alpha={36: 0.78, 72: 0.88, 144: 1.0}[n],
            label=f"{n} points ({360/n:g}°)",
            zorder={36: 1, 72: 2, 144: 3}[n],
        )
    ax_a.set_xlim(0.0, 360.0)
    ax_a.set_ylim(-0.10, 5.30)
    ax_a.set_xticks(np.arange(0.0, 361.0, 60.0))
    ax_a.set_yticks(np.arange(0.0, 5.1, 1.0))
    ax_a.set_xlabel(r"Ring angle, $\phi$ (degree)")
    ax_a.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    legend_a = ax_a.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        title=r"$\gamma_0/\pi=\gamma_1/\pi=1.000$ (SA2 and SA3)",
    )
    legend_a.get_title().set_fontsize(7.4)

    for n in (36, 72, 144):
        phi = np.asarray([row["phi_mid_deg"] for row in profiles[n]])
        lambda3 = np.asarray(
            [row["lambda_CD_nuclear_SA3"] for row in profiles[n]]
        )
        ax_b.step(
            phi,
            lambda3,
            where="mid",
            color=COLORS[n],
            lw={36: 1.8, 72: 1.35, 144: 0.9}[n],
            alpha={36: 0.78, 72: 0.88, 144: 1.0}[n],
            label=f"SA3, {n} points",
            zorder={36: 1, 72: 2, 144: 3}[n],
        )
    phi144 = np.asarray([row["phi_mid_deg"] for row in profiles[144]])
    lambda2_144 = np.asarray(
        [row["lambda_AB_nuclear_SA2"] for row in profiles[144]]
    )
    ax_b.step(
        phi144,
        lambda2_144,
        where="mid",
        color="#777777",
        lw=0.9,
        ls="--",
        label="SA2, 144 points",
    )
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, 360.0)
    ax_b.set_ylim(2.0e-7, 3.0e-2)
    ax_b.set_xticks(np.arange(0.0, 361.0, 60.0))
    ax_b.set_yticks([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    ax_b.set_xlabel(r"Ring angle, $\phi$ (degree)")
    ax_b.set_ylabel(r"$\lambda_{\mathrm{nuc}}$")
    ax_b.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
    )

    steps = np.asarray([float(row["step_deg"]) for row in rows])
    order = np.argsort(steps)
    steps = steps[order]
    c_series = (
        ("curvature_max", r"$\mathcal{C}_{qp}^{\max}$", "o", "#D55E00"),
        ("curvature_mean", r"$\langle\mathcal{C}_{qp}\rangle$", "s", "#0072B2"),
        ("curvature_rms", r"$\mathcal{C}_{qp}^{\mathrm{RMS}}$", "^", "#009E73"),
    )
    for key, label, marker, color in c_series:
        values = np.asarray([float(row[key]) for row in rows])[order]
        alpha = log_slope(steps, values)
        ax_c.plot(
            steps,
            1.0e4 * values,
            marker=marker,
            ms=5,
            color=color,
            label=label + rf"; $\alpha={alpha:.3f}$",
        )
    ax_c.set_xscale("log", base=2)
    ax_c.set_yscale("log")
    ax_c.set_xticks([2.5, 5.0, 10.0], labels=["2.5", "5", "10"])
    ax_c.set_ylim(0.4, 5.6)
    ax_c.set_yticks([0.5, 1.0, 2.0, 5.0])
    ax_c.set_yticklabels(["0.5", "1", "2", "5"])
    ax_c.set_xlabel(r"Angular step, $\Delta\phi$ (degree)")
    ax_c.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    ax_c.legend(frameon=False, loc="lower right")

    leakage_series = (
        ("lambda_nuclear_SA2_max", "SA2", "o", "#0072B2"),
        ("lambda_nuclear_SA3_max", "SA3", "s", "#D55E00"),
    )
    for key, label, marker, color in leakage_series:
        values = np.asarray([float(row[key]) for row in rows])[order]
        alpha = log_slope(steps, values)
        ax_d.plot(
            steps,
            values,
            marker=marker,
            ms=5,
            color=color,
            label=label + rf"; $\alpha={alpha:.3f}$",
        )
    ax_d.set_xscale("log", base=2)
    ax_d.set_yscale("log")
    ax_d.set_xticks([2.5, 5.0, 10.0], labels=["2.5", "5", "10"])
    ax_d.set_ylim(5.0e-7, 3.0e-2)
    ax_d.set_yticks([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    ax_d.set_xlabel(r"Angular step, $\Delta\phi$ (degree)")
    ax_d.set_ylabel(r"$\lambda_{\mathrm{nuc}}^{\max}$")
    ax_d.legend(frameon=False, loc="center right")
    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.grid(False, which="both")
        axis.minorticks_off()
        axis.tick_params(which="major", direction="in", top=True, right=True)
        axis.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.095,
        top=0.88,
        wspace=0.30,
        hspace=0.35,
    )
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    for path, file_format in ((png_path, "PNG"), (tiff_path, "TIFF")):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        options = {"dpi": (dpi, dpi)}
        if file_format == "TIFF":
            options["compression"] = "tiff_lzw"
        temporary_path = path.with_name(f"{path.name}.tmp")
        rgb.save(temporary_path, format=file_format, **options)
        temporary_path.replace(path)


def main() -> int:
    args = parser().parse_args()
    if args.combined_profile is not None or args.combined_scaling is not None:
        if args.combined_profile is None or args.combined_scaling is None:
            raise ValueError("--combined-profile and --combined-scaling must be supplied together.")
        profiles = read_combined_profiles(args.combined_profile)
        rows = read_scaling_source(args.combined_scaling)
        summaries = None
    else:
        missing = [
            option
            for n in (36, 72, 144)
            for option in (f"profile{n}", f"summary{n}")
            if getattr(args, option) is None
        ]
        if missing:
            raise ValueError(
                "Supply all individual profile/summary files or the two combined source files; "
                f"missing: {', '.join(missing)}."
            )
        profiles = {
            n: read_profile(getattr(args, f"profile{n}")) for n in (36, 72, 144)
        }
        summaries = {
            n: read_summary(getattr(args, f"summary{n}")) for n in (36, 72, 144)
        }
        rows = scaling_rows(profiles, summaries)
    for n in (36, 72, 144):
        if len(profiles[n]) != n:
            raise ValueError(f"Expected {n} rows in profile{n}; got {len(profiles[n])}.")
        if summaries is not None and int(summaries[n]["parameters"]["points"]) != n:
            raise ValueError(f"Summary/profile point-count mismatch for {n}.")

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_profile_path = output_prefix.with_name(output_prefix.name + "_profiles.csv")
    source_scaling_path = output_prefix.with_name(output_prefix.name + "_scaling.csv")
    png_path = output_prefix.with_suffix(".png")
    tiff_path = output_prefix.with_suffix(".tiff")
    pdf_path = output_prefix.with_suffix(".pdf")
    write_profile_source(source_profile_path, profiles)
    write_scaling_source(source_scaling_path, rows)
    make_figure(profiles, rows, png_path, tiff_path, pdf_path, args.dpi)
    for path in (
        png_path,
        tiff_path,
        pdf_path,
        source_profile_path,
        source_scaling_path,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
