#!/usr/bin/env python3
"""Combine the mixed-curvature construction and continuous-limit tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mixed-curvature-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_geometry_figure import product_space_panel, reduction_panel


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "validation_results.json"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"


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
            "lines.markersize": 4.8,
            "legend.frameon": False,
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def scaling_data() -> tuple[dict[str, float], tuple[np.ndarray, ...]]:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    diagnostics = payload["diagnostics"]
    rows = diagnostics["scaling_rows"]
    equal = sorted(
        (row for row in rows if row["delta_q"] == row["delta_p"]),
        key=lambda row: row["delta_q"],
    )
    fixed = sorted(
        (row for row in rows if row["delta_q"] != row["delta_p"]),
        key=lambda row: row["delta_q"],
    )
    h = np.asarray([row["delta_q"] for row in equal], dtype=float)
    c_equal = np.asarray([row["curvature"] for row in equal], dtype=float)
    q = np.asarray([row["delta_q"] for row in fixed], dtype=float)
    c_fixed = np.asarray([row["curvature"] for row in fixed], dtype=float)
    reference_quadratic = c_equal[0] * (h / h[0]) ** 2
    reference_linear = c_fixed[0] * (q / q[0])
    return diagnostics, (
        1.0e2 * h,
        1.0e4 * c_equal,
        1.0e4 * reference_quadratic,
        1.0e2 * q,
        1.0e4 * c_fixed,
        1.0e4 * reference_linear,
    )


def style_scaling_axis(ax: plt.Axes) -> None:
    ax.grid(False, which="both")
    ax.minorticks_off()
    ax.tick_params(which="major", direction="in", top=True, right=True)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def draw_scaling_panels(
    ax_equal: plt.Axes,
    ax_fixed: plt.Axes,
    diagnostics: dict[str, float],
    arrays: tuple[np.ndarray, ...],
) -> None:
    h, c_equal, ref_equal, q, c_fixed, ref_fixed = arrays

    ax_equal.loglog(
        h,
        c_equal,
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label=r"Computed $\mathcal{C}_{qp}$",
        zorder=3,
    )
    ax_equal.loglog(
        h,
        ref_equal,
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
        label=r"Reference $h^2$",
        zorder=4,
    )
    ax_equal.set_xlim(1.05, 23.0)
    ax_equal.set_ylim(0.30, 125.0)
    ax_equal.set_xticks([1.25, 2.5, 5.0, 10.0, 20.0])
    ax_equal.set_xticklabels(["1.25", "2.5", "5", "10", "20"])
    ax_equal.set_yticks([0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    ax_equal.set_yticklabels(["0.5", "1", "5", "10", "50", "100"])
    ax_equal.set_xlabel(r"Equal step, $10^{2}h$ ($h=\Delta q=\Delta p$)")
    ax_equal.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    ax_equal.text(
        0.055,
        0.93,
        rf"fit slope = {diagnostics['equal_step_slope']:.6f}",
        transform=ax_equal.transAxes,
        va="top",
    )
    ax_equal.legend(loc="lower right", handlelength=2.4)

    ax_fixed.loglog(
        q,
        c_fixed,
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        label=r"Computed $\mathcal{C}_{qp}$",
        zorder=3,
    )
    ax_fixed.loglog(
        q,
        ref_fixed,
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
        label=r"Reference $\Delta q$",
        zorder=4,
    )
    ax_fixed.set_xlim(0.42, 9.5)
    ax_fixed.set_ylim(1.5, 40.0)
    ax_fixed.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0])
    ax_fixed.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax_fixed.set_yticks([2.0, 5.0, 10.0, 20.0, 40.0])
    ax_fixed.set_yticklabels(["2", "5", "10", "20", "40"])
    ax_fixed.set_xlabel(r"Nuclear step, $10^{2}\Delta q$ ($\Delta p=0.15$)")
    ax_fixed.set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    ax_fixed.text(
        0.055,
        0.93,
        rf"fit slope = {diagnostics['fixed_delta_p_slope']:.6f}",
        transform=ax_fixed.transAxes,
        va="top",
    )
    ax_fixed.legend(loc="lower right", handlelength=2.4)

    style_scaling_axis(ax_equal)
    style_scaling_axis(ax_fixed)


def save_rgb_rasters(png: Path, tiff: Path, dpi: int) -> None:
    for path, file_format in ((png, "PNG"), (tiff, "TIFF")):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        options = {"dpi": (dpi, dpi)}
        if file_format == "TIFF":
            options["compression"] = "tiff_lzw"
        temporary_path = path.with_name(f"{path.name}.tmp")
        rgb.save(temporary_path, format=file_format, **options)
        temporary_path.replace(path)


def main() -> int:
    configure_style()
    diagnostics, arrays = scaling_data()

    figure = plt.figure(figsize=(7.2, 6.15))
    outer = figure.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.21)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.02)
    bottom = outer[1].subgridspec(1, 2, wspace=0.31)
    ax_plaquette = figure.add_subplot(top[0, 0])
    ax_reduction = figure.add_subplot(top[0, 1])
    ax_equal = figure.add_subplot(bottom[0, 0])
    ax_fixed = figure.add_subplot(bottom[0, 1])

    product_space_panel(ax_plaquette)
    reduction_panel(ax_reduction)
    draw_scaling_panels(ax_equal, ax_fixed, diagnostics, arrays)

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.985)
    prefix = ROOT / "mixed_curvature_framework_validation"
    png = prefix.with_suffix(".png")
    tiff = prefix.with_suffix(".tiff")
    pdf = prefix.with_suffix(".pdf")
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    save_rgb_rasters(png, tiff, 600)

    print(f"equal-step slope = {diagnostics['equal_step_slope']:.12f}")
    print(f"fixed-delta-p slope = {diagnostics['fixed_delta_p_slope']:.12f}")
    print(png)
    print(tiff)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
