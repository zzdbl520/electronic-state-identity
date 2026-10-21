#!/usr/bin/env python3
"""Publication plot for the O3 Gate 4B directional-refinement validation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

_mpl_config = Path(tempfile.gettempdir()) / "o3_gate4b_final_matplotlib"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#666666"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    log_x = np.log(x)
    log_y = np.log(y)
    alpha, intercept = np.polyfit(log_x, log_y, 1)
    fitted = intercept + alpha * log_x
    residual = np.sum((log_y - fitted) ** 2)
    total = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - residual / total
    return float(alpha), float(np.exp(intercept)), float(r_squared)


def direction_rows(rows: list[dict[str, str]], direction: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["direction"] == direction),
        key=lambda row: float(row["h"]),
    )


def clean_log_axis(axis, x_ticks, y_ticks) -> None:
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{value:g}" for value in x_ticks])
    axis.set_yticks(y_ticks)
    axis.set_yticklabels([f"{value:g}" for value in y_ticks])
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.yaxis.set_minor_locator(ticker.NullLocator())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", default="O3_directional_refinement")
    args = parser.parse_args()

    workdir = args.workdir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(workdir / "O3_gate4B_scaling.csv")
    radial = direction_rows(rows, "delta_r")
    angular = direction_rows(rows, "theta")

    h_r = np.asarray([float(row["h"]) for row in radial])
    h_t = np.asarray([float(row["h"]) for row in angular])
    c_r_minus = np.asarray([float(row["C_minus"]) for row in radial])
    c_r_plus = np.asarray([float(row["C_plus"]) for row in radial])
    c_r_max = np.asarray([float(row["C_max"]) for row in radial])
    c_t_minus = np.asarray([float(row["C_minus"]) for row in angular])
    c_t_plus = np.asarray([float(row["C_plus"]) for row in angular])
    lambda_r = np.asarray([float(row["lambda_nuclear_max"]) for row in radial])
    lambda_t = np.asarray([float(row["lambda_nuclear_max"]) for row in angular])
    eta_r = np.asarray([float(row["eta_edge_min"]) for row in radial])
    eta_t = np.asarray([float(row["eta_edge_min"]) for row in angular])

    alpha_c, prefactor_c, r2_c = power_fit(h_r, c_r_max)
    alpha_lr, prefactor_lr, r2_lr = power_fit(h_r / h_r.max(), lambda_r)
    alpha_lt, prefactor_lt, r2_lt = power_fit(h_t / h_t.max(), lambda_t)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax_r, ax_t, ax_l, ax_e = axes.flat

    x_r = 1.0e3 * h_r
    y_r_minus = 1.0e5 * c_r_minus
    y_r_plus = 1.0e5 * c_r_plus
    ax_r.plot(x_r, y_r_minus, "o-", color=BLUE, mfc="white", lw=1.5, ms=5,
              label="Negative direction")
    ax_r.plot(x_r, y_r_plus, "s-", color=BLUE, mfc=BLUE, lw=1.5, ms=5,
              label="Positive direction")
    ax_r.plot(
        x_r,
        1.0e5 * prefactor_c * h_r**alpha_c,
        "--",
        color=GREY,
        lw=1.3,
        label=rf"Four-point fit, $\alpha_{{\rm eff}}={alpha_c:.3f}$",
    )
    clean_log_axis(ax_r, x_r, [1, 2, 3, 5])
    ax_r.set_xlabel(r"Antisymmetric-stretch step, $10^{3}\Delta r$ ($\mathrm{\AA}$)")
    ax_r.set_ylabel(r"$10^{5}\mathcal{C}_{qp}$")
    ax_r.legend(loc="upper left", fontsize=7.8, handlelength=2.4)

    x_t = 1.0e2 * h_t
    ax_t.plot(x_t, 1.0e8 * c_t_minus, "o-", color=ORANGE, mfc="white",
              lw=1.5, ms=5, label="Negative direction")
    ax_t.plot(x_t, 1.0e8 * c_t_plus, "s-", color=ORANGE, mfc=ORANGE,
              lw=1.5, ms=5, label="Positive direction")
    clean_log_axis(ax_t, x_t, [0.1, 0.3, 1, 3, 5])
    ax_t.set_xlabel(r"Angular step, $10^{2}\Delta\theta$ (degree)")
    ax_t.set_ylabel(r"$10^{8}\mathcal{C}_{qp}$")
    ax_t.text(
        0.97,
        0.06,
        "No stable power law",
        transform=ax_t.transAxes,
        ha="right",
        va="bottom",
        color=GREY,
        fontsize=7.8,
    )
    ax_t.legend(loc="upper right", fontsize=7.8, handlelength=2.4)

    relative_r = h_r / h_r.max()
    relative_t = h_t / h_t.max()
    ax_l.plot(relative_r, 1.0e3 * lambda_r, "o-", color=BLUE, mfc="white",
              lw=1.5, ms=5,
              label=rf"Antisymmetric stretch, $\alpha={alpha_lr:.3f}$")
    ax_l.plot(relative_t, 1.0e3 * lambda_t, "s-", color=ORANGE, mfc=ORANGE,
              lw=1.5, ms=5,
              label=rf"Bending, $\alpha={alpha_lt:.3f}$")
    ax_l.plot(relative_r, 1.0e3 * prefactor_lr * relative_r**alpha_lr,
              "--", color=BLUE, lw=1.0, alpha=0.8)
    ax_l.plot(relative_t, 1.0e3 * prefactor_lt * relative_t**alpha_lt,
              "--", color=ORANGE, lw=1.0, alpha=0.8)
    clean_log_axis(ax_l, [0.125, 0.25, 0.5, 1], [0.03, 0.1, 0.3, 1, 3, 10])
    ax_l.set_xlabel(r"Relative edge length, $h/h_{\max}$")
    ax_l.set_ylabel(r"$10^{3}\lambda_{\mathrm{nuc}}^{\max}$")
    ax_l.legend(loc="upper left", fontsize=7.5, handlelength=2.4)

    ax_e.plot(relative_r, eta_r, "o-", color=BLUE, mfc="white", lw=1.5, ms=5,
              label="Antisymmetric stretch")
    ax_e.plot(relative_t, eta_t, "s-", color=ORANGE, mfc=ORANGE, lw=1.5, ms=5,
              label="Bending")
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks([0.125, 0.25, 0.5, 1])
    ax_e.set_xticklabels(["0.125", "0.25", "0.5", "1"])
    ax_e.xaxis.set_minor_locator(ticker.NullLocator())
    ax_e.yaxis.set_minor_locator(ticker.NullLocator())
    ax_e.set_ylim(0.9952, 1.0001)
    ax_e.set_yticks([0.996, 0.997, 0.998, 0.999, 1.000])
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_e.set_xlabel(r"Relative edge length, $h/h_{\max}$")
    ax_e.set_ylabel(r"$\eta_{\mathrm{edge}}$")
    ax_e.legend(loc="lower left", fontsize=7.8, handlelength=2.4)

    for axis in axes.flat:
        axis.tick_params(which="major", length=5, width=1.0)
        axis.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
        axis.grid(False)

    prefix = output_dir / args.output_prefix
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"C(delta_r) alpha_eff={alpha_c:.9f}, R2={r2_c:.9f}")
    print(f"lambda(delta_r) alpha={alpha_lr:.9f}, R2={r2_lr:.9f}")
    print(f"lambda(theta) alpha={alpha_lt:.9f}, R2={r2_lt:.9f}")
    print(f"minimum eta_edge={min(eta_r.min(), eta_t.min()):.12f}")
    for suffix in (".png", ".tiff", ".pdf"):
        print(prefix.with_suffix(suffix))


if __name__ == "__main__":
    main()
