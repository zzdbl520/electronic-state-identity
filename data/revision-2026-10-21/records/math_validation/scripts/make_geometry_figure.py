#!/usr/bin/env python3
"""Create the product-space/plaquette schematic for the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mixed-curvature-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#202020"
GREY = "#777777"
LIGHT = "#F5F5F5"


def box(ax, center, width, height, text, *, facecolor="white", edgecolor=INK,
        fontsize=8.2, linewidth=0.9, zorder=3):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.10",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, *, color=INK, linestyle="-", linewidth=1.25,
          mutation_scale=9, connectionstyle="arc3", zorder=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_corner(ax, point, name, coordinate, *, coordinate_offset=(0, -0.54)):
    x, y = point
    ax.add_patch(Circle((x, y), 0.31, facecolor="white", edgecolor=INK,
                        linewidth=1.0, zorder=5))
    ax.text(x, y, rf"$\mathbf{{{name}}}$", ha="center", va="center",
            fontsize=9.4, zorder=6)
    ax.text(x + coordinate_offset[0], y + coordinate_offset[1], coordinate,
            ha="center", va="center", fontsize=7.6, color=INK)


def product_space_panel(ax):
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 8.0)
    ax.axis("off")

    a = (1.65, 1.55)
    b = (7.15, 1.55)
    c = (1.65, 6.35)
    d = (7.15, 6.35)

    # Coordinate axes defining Q_nuc x P_var.
    arrow(ax, (0.55, 0.62), (8.20, 0.62), color=GREY, linewidth=0.85,
          mutation_scale=7)
    arrow(ax, (0.72, 0.42), (0.72, 7.25), color=GREY, linewidth=0.85,
          mutation_scale=7)
    ax.text(4.4, 0.08, r"nuclear coordinate, $q\in\mathcal{Q}_{\rm nuc}$",
            ha="center", va="bottom", fontsize=8.1)
    ax.text(0.03, 4.05, r"protocol, $p\in\mathcal{P}_{\rm var}$",
            rotation=90, ha="left", va="center", fontsize=8.1)

    # Two ordered routes across the same plaquette.
    arrow(ax, (1.98, 1.55), (6.80, 1.55), color=BLUE, linewidth=1.8)
    arrow(ax, (7.15, 1.90), (7.15, 5.98), color=BLUE, linewidth=1.8)
    arrow(ax, (1.65, 1.90), (1.65, 5.98), color=ORANGE, linestyle="--",
          linewidth=1.8)
    arrow(ax, (1.98, 6.35), (6.80, 6.35), color=ORANGE, linestyle="--",
          linewidth=1.8)

    draw_corner(ax, a, "A", r"$(q,p)$")
    draw_corner(ax, b, "B", r"$(q+\Delta q,p)$")
    draw_corner(ax, c, "C", r"$(q,p+\Delta p)$", coordinate_offset=(0, 0.56))
    draw_corner(ax, d, "D", r"$(q+\Delta q,p+\Delta p)$",
                coordinate_offset=(0, 0.56))

    ax.text(4.55, 1.78, r"$\mathbf{T}^{AB}$", color=BLUE, ha="center",
            va="bottom", fontsize=8.8)
    ax.text(7.37, 4.15, r"$\mathbf{T}^{BD}$", color=BLUE, ha="left",
            va="center", fontsize=8.8)
    ax.text(1.43, 4.15, r"$\mathbf{T}^{AC}$", color=ORANGE, ha="right",
            va="center", fontsize=8.8)
    ax.text(4.55, 6.58, r"$\mathbf{T}^{CD}$", color=ORANGE, ha="center",
            va="bottom", fontsize=8.8)

    # Compact path legend inside the plaquette.
    ax.plot([2.65, 3.45], [4.55, 4.55], color=BLUE, lw=1.8)
    ax.text(3.62, 4.55,
            r"$\mathbf{K}_{q\rightarrow p}=\mathbf{T}^{AB}\mathbf{T}^{BD}$",
            ha="left", va="center", fontsize=8.0)
    ax.plot([2.65, 3.45], [3.70, 3.70], color=ORANGE, lw=1.8, ls="--")
    ax.text(3.62, 3.70,
            r"$\mathbf{K}_{p\rightarrow q}=\mathbf{T}^{AC}\mathbf{T}^{CD}$",
            ha="left", va="center", fontsize=8.0)

def reduction_panel(ax):
    ax.set_xlim(0.0, 12.2)
    ax.set_ylim(0.0, 8.0)
    ax.axis("off")
    box(
        ax,
        (4.4, 6.90),
        7.7,
        0.82,
        r"$M_{IJ}^{XY}=\langle\Psi_I^X|\Psi_J^Y\rangle$",
        facecolor=LIGHT,
        fontsize=9.1,
    )
    ax.text(4.4, 6.31, "many-electron cross-overlap on each edge",
            ha="center", va="top", fontsize=7.2, color=GREY)

    arrow(ax, (4.4, 6.05), (4.4, 5.47), color=GREY, linewidth=1.0)
    ax.text(4.65, 5.77, "SVD / polar", fontsize=7.1, color=GREY,
            ha="left", va="center")
    box(
        ax,
        (4.4, 5.02),
        6.6,
        0.82,
        r"$\mathbf{M}^{XY}=\mathbf{T}^{XY}\mathbf{H}^{XY}$",
        fontsize=9.1,
    )

    # Curvature branch from the unitary polar factors.
    arrow(ax, (3.15, 4.57), (2.20, 4.04), color=BLUE, linewidth=1.2)
    arrow(ax, (5.65, 4.57), (6.60, 4.04), color=ORANGE, linestyle="--",
          linewidth=1.2)
    box(ax, (2.15, 3.57), 3.95, 0.78,
        r"$\mathbf{K}_{q\rightarrow p}=\mathbf{T}^{AB}\mathbf{T}^{BD}$",
        edgecolor=BLUE, fontsize=7.9)
    box(ax, (6.65, 3.57), 3.95, 0.78,
        r"$\mathbf{K}_{p\rightarrow q}=\mathbf{T}^{AC}\mathbf{T}^{CD}$",
        edgecolor=ORANGE, fontsize=7.9)
    arrow(ax, (2.15, 3.13), (3.83, 2.62), color=BLUE, linewidth=1.2)
    arrow(ax, (6.65, 3.13), (4.97, 2.62), color=ORANGE, linestyle="--",
          linewidth=1.2)
    box(
        ax,
        (4.4, 2.27),
        5.55,
        0.78,
        r"$\mathbf{W}_{qp}=\mathbf{K}_{q\rightarrow p}"
        r"\mathbf{K}_{p\rightarrow q}^{\dagger}$",
        facecolor=LIGHT,
        fontsize=8.8,
    )
    arrow(ax, (4.4, 1.84), (4.4, 1.35), color=GREY, linewidth=1.0)
    box(
        ax,
        (4.4, 0.91),
        6.25,
        0.82,
        r"$\mathcal{C}_{qp}=\|\mathbf{W}_{qp}-\mathbf{I}\|_{\mathrm{F}}/(2\sqrt{m})$",
        edgecolor=INK,
        linewidth=1.1,
        fontsize=8.7,
    )

    # Conditioning/leakage branch remains separate from curvature.
    arrow(ax, (7.75, 5.02), (9.05, 5.02), color=GREY, linewidth=1.0)
    box(
        ax,
        (10.45, 4.25),
        3.00,
        2.65,
        r"$\eta_{\rm edge}=\min_{XY}\sigma_{\min}$"
        "\n"
        r"$(\mathbf{M}^{XY})$"
        "\n\n"
        r"$\lambda^{XY}=1-\frac{1}{m}\sum_k\sigma_k^2$"
        "\n"
        r"$\lambda_{\rm edge}=\max_{XY}\lambda^{XY}$",
        facecolor=LIGHT,
        edgecolor=GREY,
        fontsize=6.9,
    )
    ax.text(10.45, 2.68, "conditioning and leakage\nreported beside curvature",
            ha="center", va="top", fontsize=7.0, color=GREY)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.55),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
        constrained_layout=True,
    )
    product_space_panel(axes[0])
    reduction_panel(axes[1])
    figure.savefig(ROOT / "mixed_curvature_geometry.pdf", bbox_inches="tight")
    figure.savefig(ROOT / "mixed_curvature_geometry.svg", bbox_inches="tight")
    figure.savefig(ROOT / "mixed_curvature_geometry.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    print("Wrote mixed_curvature_geometry.pdf/svg/png")


if __name__ == "__main__":
    main()
