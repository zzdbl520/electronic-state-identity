# Reproducibility

First run `python scripts/unpack_wavefunctions.py` to restore the included raw arrays.

1. Transport algebra: `python -m unittest discover -v` after installing the root package.
2. Current records/integrity: `python scripts/verify_reference_results.py` and
   `sha256sum -c SHA256SUMS --quiet`.
3. LiH overlap reconstruction: validators in [the data README](../data/revision-2026-10-21/README.md),
   with PySCF 2.13.1; reads saved wave functions without new optimization.
4. Fresh LiH calculations: archived engines and new output directories; expensive
   CASSCF calculations are not run in GitHub Actions.

Current LiH overlaps include the optimized doubly occupied inactive orbital.
September active-only workflows are historical provenance. Raw source/plan
fingerprints remain immutable.

For the other systems, reproduce only what the available records support;
see [Data availability](DATA_AVAILABILITY.md). Passing record arithmetic does
not replace missing geometries or orbital/CI arrays. Check deposited hashes
before regenerating derived tables and figures.
