#!/usr/bin/env python3
"""Create publication-ready figures and tables from validation_results.json."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mixed-curvature-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "validation_results.json"


def load_results() -> dict[str, object]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def write_scaling_csv(rows: list[dict[str, float]]) -> None:
    path = ROOT / "continuous_limit_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "delta_q", "delta_p", "curvature"])
        for row in rows:
            series = "equal_steps" if row["delta_q"] == row["delta_p"] else "fixed_delta_p"
            writer.writerow(
                [
                    series,
                    f"{row['delta_q']:.10g}",
                    f"{row['delta_p']:.10g}",
                    f"{row['curvature']:.16e}",
                ]
            )


def write_summary_csv(tests: list[dict[str, object]]) -> None:
    path = ROOT / "validation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test", "status", "quantitative_outcome"])
        for test in tests:
            writer.writerow(
                [
                    test["name"],
                    "PASS" if test["passed"] else "FAIL",
                    test["detail"],
                ]
            )


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "^": r"\textasciicircum{}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def write_latex_tables(
    tests: list[dict[str, object]],
    rows: list[dict[str, float]],
    diagnostics: dict[str, object],
) -> None:
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Complete numerical test inventory for the mixed variational curvature.}",
        r"\label{tab:complete_mixed_curvature_tests}",
        r"\begin{tabular}{p{0.25\textwidth}p{0.10\textwidth}p{0.54\textwidth}}",
        r"\toprule",
        r"Test & Status & Quantitative outcome \\",
        r"\midrule",
    ]
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(
            f"{escape_latex(str(test['name']))} & {status} & "
            f"{escape_latex(str(test['detail']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Small-plaquette convergence of $\mathcal{C}_{qp}$.}",
            r"\label{tab:mixed_curvature_scaling}",
            r"\begin{tabular}{ccc}",
            r"\toprule",
            r"$\Delta q$ & $\Delta p$ & $\mathcal{C}_{qp}$ \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['delta_q']:.5f} & {row['delta_p']:.5f} & "
            f"{row['curvature']:.8e} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{3}}{{l}}{{Equal-step slope: {diagnostics['equal_step_slope']:.8f}}} \\",
            rf"\multicolumn{{3}}{{l}}{{Fixed-$\Delta p$ slope: {diagnostics['fixed_delta_p_slope']:.8f}}} \\",
            rf"\multicolumn{{3}}{{l}}{{Coefficient ratio: {diagnostics['coefficient_ratio']:.8f}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (ROOT / "validation_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_scaling(rows: list[dict[str, float]], diagnostics: dict[str, object]) -> None:
    equal = sorted(
        (row for row in rows if row["delta_q"] == row["delta_p"]),
        key=lambda row: row["delta_q"],
    )
    fixed = sorted(
        (row for row in rows if row["delta_q"] != row["delta_p"]),
        key=lambda row: row["delta_q"],
    )

    h = np.asarray([row["delta_q"] for row in equal])
    c_equal = np.asarray([row["curvature"] for row in equal])
    q = np.asarray([row["delta_q"] for row in fixed])
    c_fixed = np.asarray([row["curvature"] for row in fixed])

    reference_quadratic = c_equal[0] * (h / h[0]) ** 2
    reference_linear = c_fixed[0] * (q / q[0])

    # Explicit scale factors keep scientific notation attached to the
    # physical quantity and make every sampled step a labeled major tick.
    h_plot = 1.0e2 * h
    q_plot = 1.0e2 * q
    c_equal_plot = 1.0e4 * c_equal
    c_fixed_plot = 1.0e4 * c_fixed
    reference_quadratic_plot = 1.0e4 * reference_quadratic
    reference_linear_plot = 1.0e4 * reference_linear

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 11,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=False)
    blue = "#0072B2"
    orange = "#D55E00"

    axes[0].loglog(
        h_plot,
        c_equal_plot,
        color=blue,
        marker="o",
        markersize=4.8,
        linewidth=1.35,
        label=r"Computed $\mathcal{C}_{qp}$",
    )
    axes[0].loglog(
        h_plot,
        reference_quadratic_plot,
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label=r"Reference $h^2$",
    )
    axes[0].set_xlim(1.05, 23.0)
    axes[0].set_ylim(0.30, 125.0)
    axes[0].set_xticks([1.25, 2.5, 5.0, 10.0, 20.0])
    axes[0].set_xticklabels(["1.25", "2.5", "5", "10", "20"])
    axes[0].set_yticks([0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    axes[0].set_yticklabels(["0.5", "1", "5", "10", "50", "100"])
    axes[0].set_xlabel(r"Equal step, $10^{2}h$ ($h=\Delta q=\Delta p$)")
    axes[0].set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    axes[0].text(
        0.055,
        0.93,
        rf"fit slope = {diagnostics['equal_step_slope']:.6f}",
        transform=axes[0].transAxes,
        va="top",
    )
    axes[0].legend(loc="lower right", handlelength=2.4)

    axes[1].loglog(
        q_plot,
        c_fixed_plot,
        color=orange,
        marker="s",
        markersize=4.6,
        linewidth=1.35,
        label=r"Computed $\mathcal{C}_{qp}$",
    )
    axes[1].loglog(
        q_plot,
        reference_linear_plot,
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label=r"Reference $\Delta q$",
    )
    axes[1].set_xlim(0.42, 9.5)
    axes[1].set_ylim(1.5, 40.0)
    axes[1].set_xticks([0.5, 1.0, 2.0, 4.0, 8.0])
    axes[1].set_xticklabels(["0.5", "1", "2", "4", "8"])
    axes[1].set_yticks([2.0, 5.0, 10.0, 20.0, 40.0])
    axes[1].set_yticklabels(["2", "5", "10", "20", "40"])
    axes[1].set_xlabel(r"Nuclear step, $10^{2}\Delta q$ ($\Delta p=0.15$)")
    axes[1].set_ylabel(r"$10^{4}\,\mathcal{C}_{qp}$")
    axes[1].text(
        0.055,
        0.93,
        rf"fit slope = {diagnostics['fixed_delta_p_slope']:.6f}",
        transform=axes[1].transAxes,
        va="top",
    )
    axes[1].legend(loc="lower right", handlelength=2.4)

    for axis in axes:
        axis.grid(False, which="both")
        axis.minorticks_off()
        axis.tick_params(which="major", direction="in", top=True, right=True)
        axis.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.97, wspace=0.31)

    pdf = ROOT / "mixed_curvature_scaling.pdf"
    png = ROOT / "mixed_curvature_scaling.png"
    tiff = ROOT / "mixed_curvature_scaling.tiff"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    # Write journal-safe RGB rasters without overwriting an open source file.
    for path, file_format in ((png, "PNG"), (tiff, "TIFF")):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        options = {"dpi": (600, 600)}
        if file_format == "TIFF":
            options["compression"] = "tiff_lzw"
        temporary_path = path.with_name(f"{path.name}.tmp")
        rgb.save(temporary_path, format=file_format, **options)
        temporary_path.replace(path)


def main() -> None:
    payload = load_results()
    tests = payload["tests"]
    diagnostics = payload["diagnostics"]
    rows = diagnostics["scaling_rows"]
    write_scaling_csv(rows)
    write_summary_csv(tests)
    write_latex_tables(tests, rows, diagnostics)
    plot_scaling(rows, diagnostics)
    print("Wrote mixed_curvature_scaling.pdf/png/tiff")
    print("Wrote continuous_limit_data.csv and validation_summary.csv")
    print("Wrote validation_tables.tex")


if __name__ == "__main__":
    main()
