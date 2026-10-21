# Numerical data revision — 21 October 2026

First run `python scripts/unpack_wavefunctions.py` from the repository root to
restore the 1,936 NPZ files from `wavefunctions/`. Each part, archive and restored
file is checked against its recorded hash.

Scientific data and code from the verified JCTC revision package. Original raw
records are unchanged. Figure scripts write to `figures/`.

## Contents

- `records/`: wave functions, logs, calculation plans, scripts and historical tables.
- `analysis/`: source CSVs, numerical summaries and independent replay reports.
- `code/`: complete-overlap implementation, solvers, validators and figure scripts.
- `figures/`: output directory for figure regeneration.
- `SOURCE_PROVENANCE.json`: source-archive hash and per-file source/deposit hashes.
- `SHA256SUMS`: checksums for this data directory.

## LiH evidence

| Evidence | Location |
|---|---|
| Final aug-cc-pVTZ result | `analysis/augccpvtz_summary.json` |
| Original production and nested loops | `records/lih_augccpvtz_production/` |
| Final 154-edge boundary and both histories | `records/lih_augccpvtz_local_refinement/` |
| Independent raw-wave-function audits | `analysis/augccpvtz_production_validation/`, `analysis/augccpvtz_refinement_validation/` |
| Earlier bases | `records/lih_revision_augccpvdz/`, `records/lih_revision_ccpvtz/` |
| All 100 original rectangles | `analysis/lih_100_locator_plaquettes.csv`, `analysis/lih_100_complete_overlap_rectangles.csv` |
| Initial-grid implementation replay | `records/lih_25point_replay/`, `code/lih_complete_overlap.py` |

The final augmented-triple-zeta result is C_RF = 3.191653837e-4 and
D_mu = 6.300611946e-4. Both endpoint permutations are identity. The four
shrinking-loop log-log slope is 1.000243777. The earlier uniform 64-edge
boundary fails the assignment margin and remains a rejected intermediate.
None of the 100 unrefined original rectangles passes complete-overlap fidelity.

The 1,936 NPZ files comprise 736 augmented-triple-zeta production files,
644 local-refinement files, 337 double-zeta revision files, 194 unaugmented
triple-zeta revision files and 25 initial-grid replay files. Counts include
candidate solutions and duplicate baselines, not just independent geometries.

## Reanalysis

From this directory, with dependencies in `requirements.txt` installed:

```bash
python code/summarize_revision.py
python code/summarize_augccpvtz.py
python code/summarize_cpdc.py
python code/make_math_validation_figure.py
python code/make_revision_figures.py
python code/make_cpdc_figure.py
```

These commands regenerate derived files. Check deposited hashes before
regeneration; environment-dependent rounding or plot metadata can change them.

Independent overlap reconstruction without new CASSCF optimizations:

```bash
OPENBLAS_NUM_THREADS=1 python code/validate_revision.py
mkdir -p validation_rerun
OPENBLAS_NUM_THREADS=1 python code/lih_complete_overlap.py records/lih_25point_replay validation_rerun/lih_complete_overlap_audit.json
OPENBLAS_NUM_THREADS=1 python code/validate_augccpvtz_production.py records/lih_augccpvtz_production --out validation_rerun/production --threads 1
OPENBLAS_NUM_THREADS=1 python code/validate_augccpvtz_refinement.py records/lih_augccpvtz_local_refinement --out validation_rerun/refinement --threads 1
```

The production validator preserves the original assignment failure; the
refinement validator verifies its resolution. The 531-file/18-loop report from
`validate_revision.py` covers the two earlier bases only.

Production used PySCF 2.13.1, NumPy 2.3.5 and SciPy 1.17.1. The original
independent replay used SciPy 1.17.0. Plotting used Matplotlib 3.10.8.
Exact runtimes/settings are recorded in the plans and audits.

## Fresh electronic-structure calculations

Always use new output directories; do not change saved plans to force cache reuse.

```bash
python records/lih_augccpvtz_production/calculation_source.py all --root rerun_augccpvtz --threads 8
python records/lih_augccpvtz_local_refinement/code/refine_lih.py all --source rerun_augccpvtz --root rerun_augccpvtz_refinement --threads 8
python code/run_lih_revision.py --workflow code/lih_solver --root rerun_augccpvdz --basis aug-cc-pvdz --nested --histories
python code/run_lih_revision.py --workflow code/lih_solver --root rerun_ccpvtz --basis cc-pvtz --histories
```

## Historical records and limits

- `records/H3/`: ring/refinement tables, scripts, numerical records recovered from returned archives and
  the `STOP_POINTWISE_REFINEMENT_FAILURE` decision.
- `records/O3/`: gate4A/gate4B tables, manifests, logs, workflows, numerical return
  archive and weighted-SA2 pilot. The pilot is `STOP_EDGE_FIDELITY_FAILURE`.
- `records/BeH2/` and `records/beh2_model_comparison/`: tables and summary-level
  model comparison. Cross-model restart arrays and precise historical 12-start
  records are unavailable. Nonclosure ratios/refinement exponents do not
  establish a physical branch transition.
- `records/CPDC/`: point/edge, SA3, mesh/history and Wilson records, scripts and
  logs. Original geometries, relaxed-surface coefficients and orbital/CI arrays
  are unavailable. Record arithmetic/tables can be regenerated; overlap/Wilson
  reconstruction cannot be performed from this deposit alone.

See [Data availability](../../docs/DATA_AVAILABILITY.md). Hashes do not replace
missing raw evidence. Historical status labels must be read with these limits.

`NUMERICAL_ARCHIVES.json` identifies the original H3/O3 return archives and
checksums every retained numerical member. Rendered figures were excluded
when repacking; numerical file contents were preserved byte-for-byte.

The O3 numerical archive is stored as six `.part000`–`.part005` files for
reliable transfer. The reference verifier checks each part and the assembled
archive directly. To reconstruct the original archive manually, run from the
repository root:

```bash
cd data/revision-2026-10-21/records/O3/raw_return_archive
cat O3_gate4B_v5.2_numerical.tar.gz.part0?? > O3_gate4B_v5.2_numerical.tar.gz
```

Its SHA-256 is `3e86b410935e18184c90d9cce0bfa3b84c8a0ccc283b2360c1e7fa6795bf0132`.
The archive and all numerical members are identical to the unsplit deposit.
