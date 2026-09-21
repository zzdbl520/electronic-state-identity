#!/usr/bin/env python3
"""Create the publication figure for the BeH2 B3R validation.

Style choices follow the manuscript convention: inward ticks on all sides,
no grid lines, panel letters, explicit scale factors in axis labels, and
600-dpi PNG/TIFF output.  The plotting values are read directly from the B3R
CSV files; no values are hard-coded except axis presentation settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image


ORANGE = "#D55E00"
BLUE = "#0072B2"
GRAY = "#8C8C8C"
EVENT_LEFT = 2.52180
EVENT_RIGHT = 2.52185


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot BeH2 B3R validation results.")
    p.add_argument("--data-dir", type=Path, default=Path("retrieved_B3R"))
    p.add_argument("--output-prefix", type=Path, default=Path("BeH2_B3R_validation"))
    return p


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "mathtext.fontset": "stixsans",
        }
    )


def style_linear_axis(ax: plt.Axes, *, x_label: bool = True) -> None:
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(which="major", direction="in", top=True, right=True)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.set_xlim(2.52115, 2.52225)
    ax.set_xticks(np.arange(2.5212, 2.52221, 0.0002))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    if x_label:
        ax.set_xlabel(r"Insertion coordinate, $\lambda$ (Bohr)")
    ax.axvspan(EVENT_LEFT, EVENT_RIGHT, color=GRAY, alpha=0.18, lw=0, zorder=0)


def power_law_exponent(group: pd.DataFrame, metric: str) -> tuple[float, float]:
    x = np.log(group["delta_q_bohr"].to_numpy(dtype=float))
    y = np.log(group[metric].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main() -> int:
    args = parser().parse_args()
    branch = pd.read_csv(args.data_dir / "B3R_branch_map.csv")
    plaquettes = pd.read_csv(args.data_dir / "B3R_multiscale_plaquettes.csv")
    scaling = pd.read_csv(args.data_dir / "B3R_scaling.csv")

    sa3 = branch.loc[branch["protocol"].eq("SA3")].sort_values("lambda_bohr")
    fine = plaquettes.loc[plaquettes["stride"].eq(1)].copy()
    upper_fine = fine.loc[fine["route"].eq("upper_down")].sort_values("q_mid_bohr")
    unique_fine = fine.loc[fine["route"].eq("unique_up")].sort_values("q_mid_bohr")
    upper_scale = scaling.loc[scaling["route"].eq("upper_down")].sort_values("delta_q_bohr")
    unique_scale = scaling.loc[scaling["route"].eq("unique_up")].sort_values("delta_q_bohr")

    if len(sa3) != 21 or len(upper_fine) != 20 or len(unique_fine) != 20:
        raise RuntimeError("Unexpected B3R row count; refusing to plot incomplete data.")

    alpha_upper, intercept_upper = power_law_exponent(upper_scale, "curvature_max")
    alpha_unique, intercept_unique = power_law_exponent(unique_scale, "curvature_max")

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=False)
    ax_energy, ax_curvature, ax_leakage, ax_scaling = axes.ravel()

    # SA3 history separation in mEh.
    ax_energy.plot(
        sa3["lambda_bohr"],
        1.0e3 * sa3["delta_e_x_minus_y"],
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        label="SA3: upper-down\n$-$ unique-up",
        zorder=3,
    )
    style_linear_axis(ax_energy)
    ax_energy.set_ylim(-0.25, 6.15)
    ax_energy.set_yticks(np.arange(0.0, 6.1, 1.0))
    ax_energy.set_ylabel(r"$\Delta\overline{E}_{\mathrm{hist}}$ (m$E_{\mathrm{h}}$)")
    ax_energy.legend(loc="upper left", handlelength=2.2)

    # Finest-grid polar-unitary nonclosure.  Explicit scaling avoids an offset label.
    ax_curvature.plot(
        unique_fine["q_mid_bohr"],
        1.0e4 * unique_fine["curvature"],
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label="Unique-up",
        zorder=2,
    )
    ax_curvature.plot(
        upper_fine["q_mid_bohr"],
        1.0e4 * upper_fine["curvature"],
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        label="Upper-down",
        zorder=3,
    )
    style_linear_axis(ax_curvature)
    ax_curvature.set_ylim(-0.15, 6.35)
    ax_curvature.set_yticks(np.arange(0.0, 6.1, 1.0))
    ax_curvature.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    ax_curvature.legend(loc="upper left", handlelength=2.2)

    # Nuclear-edge state leakage at the finest step.
    ax_leakage.plot(
        unique_fine["q_mid_bohr"],
        1.0e2 * unique_fine["leakage_AB"],
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label="Unique-up",
        zorder=2,
    )
    ax_leakage.plot(
        upper_fine["q_mid_bohr"],
        1.0e2 * upper_fine["leakage_AB"],
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        label="Upper-down",
        zorder=3,
    )
    style_linear_axis(ax_leakage)
    ax_leakage.set_ylim(-0.15, 5.90)
    ax_leakage.set_yticks(np.arange(0.0, 5.6, 1.0))
    ax_leakage.set_ylabel(r"$10^{2}\,\lambda_{AB}$")
    ax_leakage.legend(loc="upper left", handlelength=2.2)

    # Refinement scaling of the peak polar-unitary nonclosure.
    # Scale the plotted abscissa by 10^4 so all four major ticks remain explicit
    # and legible without Matplotlib's automatic scientific-offset notation.
    unique_width = 1.0e4 * unique_scale["delta_q_bohr"]
    upper_width = 1.0e4 * upper_scale["delta_q_bohr"]
    unique_curvature = 1.0e4 * unique_scale["curvature_max"]
    upper_curvature = 1.0e4 * upper_scale["curvature_max"]
    ax_scaling.loglog(
        unique_width,
        unique_curvature,
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label=rf"Unique-up, $\alpha={alpha_unique:.3f}$",
        zorder=3,
    )
    ax_scaling.loglog(
        upper_width,
        upper_curvature,
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        label=rf"Upper-down, $\alpha={alpha_upper:.3f}$",
        zorder=3,
    )
    xfit = np.geomspace(
        scaling["delta_q_bohr"].min(), scaling["delta_q_bohr"].max(), 100
    )
    ax_scaling.loglog(
        1.0e4 * xfit,
        1.0e4 * np.exp(intercept_unique) * xfit**alpha_unique,
        color=BLUE,
        ls="--",
        lw=1.0,
        alpha=0.7,
        zorder=1,
    )
    ax_scaling.loglog(
        1.0e4 * xfit,
        1.0e4 * np.exp(intercept_upper) * xfit**alpha_upper,
        color=ORANGE,
        ls="--",
        lw=1.0,
        alpha=0.7,
        zorder=1,
    )
    ax_scaling.grid(False, which="both")
    ax_scaling.minorticks_off()
    ax_scaling.tick_params(which="major", direction="in", top=True, right=True)
    ax_scaling.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax_scaling.set_xlim(0.42, 4.8)
    ax_scaling.set_ylim(5.0e-2, 1.0e1)
    ax_scaling.set_xticks([0.5, 1.0, 2.0, 4.0])
    ax_scaling.set_xticklabels(["0.5", "1", "2", "4"])
    ax_scaling.set_yticks([0.1, 1.0, 10.0])
    ax_scaling.set_yticklabels(["0.1", "1", "10"])
    ax_scaling.set_xlabel(r"Plaquette width, $10^{4}\Delta\lambda$ (Bohr)")
    ax_scaling.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}^{\max}$")
    ax_scaling.legend(loc="center left", handlelength=2.2)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    for index, ax in enumerate(axes.ravel()):
        ax.text(
            -0.23,
            1.03,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.985, wspace=0.30, hspace=0.30)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = prefix.with_suffix(".png")
    tiff = prefix.with_suffix(".tiff")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Journal portals are more reliable with RGB rather than RGBA rasters.
    for path, file_format in ((png, "PNG"), (tiff, "TIFF")):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        options = {"dpi": (600, 600)}
        if file_format == "TIFF":
            options["compression"] = "tiff_lzw"
        temporary_path = path.with_name(f"{path.name}.tmp")
        rgb.save(temporary_path, format=file_format, **options)
        temporary_path.replace(path)

    print(f"upper scaling exponent = {alpha_upper:.12f}")
    print(f"unique scaling exponent = {alpha_unique:.12f}")
    print(png)
    print(tiff)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
