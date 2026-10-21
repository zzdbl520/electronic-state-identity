"""Reproduce the explicit m=3 generator test in main Section 2.7 and SI S2.

The historical records/math_validation series uses a different generator setup.
This script leaves those records intact and plots the matrices printed in SI S2.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
AQ = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float)
AP = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def nonclosure(q, p):
    uq, up = expm(q * AQ), expm(p * AP)
    w = uq @ up @ uq.T @ up.T
    return np.linalg.norm(np.eye(3) - w, "fro") / (2 * np.sqrt(3))


def main():
    series = [
        ("equal_steps", np.array([.20, .10, .050, .025, .0125]), None),
        ("fixed_delta_p", np.array([.080, .040, .020, .010, .005]), .15),
    ]
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 10, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    rows, fits = [], {}
    colors = ["#2371a0", "#b9652a"]
    for idx, (name, x, fixed_p) in enumerate(series):
        y = np.array([nonclosure(q, q if fixed_p is None else fixed_p) for q in x])
        coeff = np.polyfit(np.log(x), np.log(y), 1)
        fitted = np.exp(np.polyval(coeff, np.log(x)))
        residual = float(np.max(abs(fitted - y) / y))
        fits[name] = {"slope": float(coeff[0]), "maximum_relative_fit_residual": residual}
        rows.extend({"series": name, "delta_q": float(q),
                     "delta_p": float(q if fixed_p is None else fixed_p),
                     "C_qp": float(c)} for q, c in zip(x, y))
        ax = axes[idx]
        power = 2 if fixed_p is None else 1
        ref = y[-1] * (x / x[-1]) ** power
        ax.loglog(x, ref, "--", color="#60656b", lw=1.2,
                  label=rf"Reference: $\propto {'h^2' if power == 2 else r'\Delta q'}$")
        ax.loglog(x, y, "o-", color=colors[idx], markersize=4.5, lw=1.2,
                  markerfacecolor="white", label="Computed")
        ax.set_title(["(a) Equal edge lengths", "(b) Fixed second edge"][idx],
                     loc="left", fontweight="bold", pad=10)
        ax.text(.05, .95, f"Fitted slope = {coeff[0]:.8f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
        ax.set_xlabel(r"Equal step $h$ ($\Delta q=\Delta p$)" if fixed_p is None
                      else r"Step $\Delta q$ ($\Delta p=0.15$)")
        ax.set_ylabel(r"$\mathcal{C}_{qp}$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(loc="lower right", frameon=False)
    fig.savefig(ROOT / "figures/Figure3_small_plaquette_validation.pdf")
    fig.savefig(ROOT / "figures/Figure3_small_plaquette_validation.png", dpi=450)
    plt.close(fig)
    with (ROOT / "analysis/explicit_generator_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    norm = float(np.linalg.norm(AQ @ AP - AP @ AQ, "fro"))
    report = {"retained_rank": 3, "A_q": AQ.tolist(), "A_p": AP.tolist(),
              "commutator_frobenius_norm": norm, "fits": fits,
              "smallest_equal_step_coefficient_relative_error":
              float(abs(nonclosure(.0125, .0125) / .0125**2 / (norm/(2*np.sqrt(3))) - 1))}
    (ROOT / "analysis/explicit_generator_scaling.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
