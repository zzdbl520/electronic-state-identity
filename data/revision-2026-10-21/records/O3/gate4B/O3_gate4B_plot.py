#!/usr/bin/env python3
"""Plot Gate 4B directional-refinement scaling in the established figure style."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

_mpl_config = Path(tempfile.gettempdir()) / "o3_gate4b_matplotlib"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fit_for(rows: list[dict], direction: str, statistic: str) -> dict:
    for row in rows:
        if row["direction"] == direction and row["statistic"] == statistic:
            return row
    raise RuntimeError(f"Missing {direction}/{statistic} fit.")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("O3_gate4B_work"))
    parser.add_argument("--output-prefix", default="O3_gate4B_refinement")
    args = parser.parse_args()
    workdir = args.workdir.expanduser().resolve()
    scaling = read_rows(workdir / "O3_gate4B_scaling.csv")
    fits = read_rows(workdir / "O3_gate4B_fits.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    specifications = [
        ("delta_r", r"Antisymmetric-stretch step, $10^{3}\Delta r$ (Å)", "#0072B2", 1.0e3),
        ("theta", r"Angular step, $10^{2}\Delta\theta$ (degree)", "#D55E00", 1.0e2),
    ]
    for axis, (direction, xlabel, color, x_scale) in zip(axes, specifications):
        rows = sorted(
            (row for row in scaling if row["direction"] == direction),
            key=lambda row: float(row["h"]),
        )
        h = np.asarray([float(row["h"]) for row in rows])
        x_display = x_scale * h
        minus = np.asarray([float(row["C_minus"]) for row in rows])
        plus = np.asarray([float(row["C_plus"]) for row in rows])
        maximum = np.asarray([float(row["C_max"]) for row in rows])
        fit = fit_for(fits, direction, "maximum")
        alpha = float(fit["alpha"])
        prefactor = float(fit["prefactor"])

        axis.loglog(
            x_display, minus, "o-", color=color, mfc="white", label="Negative direction"
        )
        axis.loglog(
            x_display, plus, "s-", color=color, mfc=color, label="Positive direction"
        )
        axis.loglog(
            x_display,
            prefactor * h**alpha,
            "--",
            color="0.35",
            linewidth=1.2,
            label=rf"Maximum fit, $\alpha={alpha:.3f}$",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"$\mathcal{C}_{qp}$")
        axis.set_xticks(x_display)
        axis.set_xticklabels([f"{value:g}" for value in x_display])
        positive = np.concatenate([minus[minus > 0.0], plus[plus > 0.0]])
        if positive.size:
            y_ticks = np.geomspace(float(positive.min()), float(positive.max()), 4)
            axis.set_yticks(y_ticks)
            axis.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda value, _position: f"{value:.1e}")
            )
        axis.tick_params(which="major", length=5, width=1.0)
        axis.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.yaxis.set_minor_locator(ticker.NullLocator())
        axis.grid(False)
        axis.legend(frameon=False, fontsize=8, loc="best")

    prefix = workdir / args.output_prefix
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {prefix.with_suffix('.tiff')}")
    print(f"Wrote {prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
