#!/usr/bin/env python3
"""Plot production-level microscopic diagnostics for the BeH2 B3R collapse."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

_MPL_CONFIG = Path(tempfile.gettempdir()) / "beh2_b3r_components_mpl"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator
import numpy as np
import pandas as pd
from PIL import Image


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#8C8C8C"
EVENT_LEFT = 2.52180
EVENT_RIGHT = 2.52185


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Plot BeH2 B3R active-space, density, occupation, and iteration diagnostics."
    )
    result.add_argument("--data-dir", type=Path, default=Path("retrieved_B3R"))
    result.add_argument(
        "--output-prefix", type=Path, default=Path("BeH2_B3R_components")
    )
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "legend.fontsize": 8.5,
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
            "legend.frameon": False,
            "mathtext.fontset": "stixsans",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes, *, xlabel: bool) -> None:
    axis.grid(False, which="both")
    axis.minorticks_off()
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.tick_params(which="major", direction="in", top=True, right=True)
    axis.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    axis.set_xlim(2.52115, 2.52225)
    axis.set_xticks(np.arange(2.5212, 2.52221, 0.0002))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axis.axvspan(EVENT_LEFT, EVENT_RIGHT, color=GRAY, alpha=0.18, lw=0, zorder=0)
    if xlabel:
        axis.set_xlabel(r"Insertion coordinate, $\lambda$ (Bohr)")
    else:
        axis.tick_params(labelbottom=False)


def positive(values: np.ndarray) -> np.ndarray:
    if np.any(values <= 0.0):
        raise ValueError("Logarithmic diagnostic contains a nonpositive value.")
    return values


def main() -> int:
    args = parser().parse_args()
    data = pd.read_csv(args.data_dir / "B3R_branch_map.csv")
    sa3 = data.loc[data["protocol"].eq("SA3")].sort_values("lambda_bohr")
    sa4 = data.loc[data["protocol"].eq("SA4")].sort_values("lambda_bohr")
    if len(sa3) != 21 or len(sa4) != 21:
        raise RuntimeError("Unexpected B3R row count; refusing to plot incomplete data.")

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), constrained_layout=False)
    ax_angle, ax_rdm, ax_noon, ax_iter = axes.ravel()

    for axis, column, scale, ylabel in (
        (ax_angle, "active_angle_max_rad", 180.0 / np.pi,
         r"Maximum principal angle (degree)"),
        (ax_rdm, "rdm1_procrustes_fro", 1.0,
         r"Aligned 1-RDM distance, $d_{\gamma}$"),
        (ax_noon, "noon_rms", 1.0,
         r"Occupation distance, $d_n$"),
    ):
        axis.semilogy(
            sa3["lambda_bohr"],
            positive(scale * sa3[column].to_numpy(dtype=float)),
            "s-",
            color=ORANGE,
            markerfacecolor=ORANGE,
            label="SA3 histories",
            lw=1.6,
            ms=4.3,
        )
        axis.semilogy(
            sa4["lambda_bohr"],
            positive(scale * sa4[column].to_numpy(dtype=float)),
            "o-",
            color=BLUE,
            markerfacecolor=BLUE,
            label="SA4 histories",
            lw=1.5,
            ms=4.1,
        )
        axis.set_ylabel(ylabel)

    ax_angle.legend(loc="upper left", handlelength=2.2)
    ax_iter.plot(
        sa3["lambda_bohr"],
        sa3["macroiterations_x"],
        "s-",
        color=ORANGE,
        markerfacecolor=ORANGE,
        label="SA3 upper-down",
        lw=1.6,
        ms=4.3,
    )
    ax_iter.plot(
        sa4["lambda_bohr"],
        sa4["macroiterations_x"],
        "o-",
        color=BLUE,
        markerfacecolor=BLUE,
        label="SA4 upper-down",
        lw=1.5,
        ms=4.1,
    )
    ax_iter.set_ylim(-4, 126)
    ax_iter.set_yticks([0, 30, 60, 90, 120])
    ax_iter.set_ylabel("CASSCF macroiterations")
    ax_iter.legend(loc="upper right", handlelength=2.2)

    for index, axis in enumerate(axes.ravel()):
        style_axis(axis, xlabel=index >= 2)
        axis.text(
            0.02,
            0.96,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.105, top=0.985, wspace=0.31, hspace=0.18
    )
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = prefix.with_suffix(".png")
    tiff = prefix.with_suffix(".tiff")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with Image.open(png) as image:
        rgb = image.convert("RGB")
        rgb.save(tiff, format="TIFF", dpi=(600, 600), compression="tiff_lzw")

    for path in (png, tiff, pdf):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
