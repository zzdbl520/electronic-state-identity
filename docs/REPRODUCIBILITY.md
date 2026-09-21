# Reproducibility levels

This repository distinguishes three levels of reproduction.

## Level 1: transport algebra

`pytest` checks polar transport, edge singular values, bounded route nonclosure,
gauge covariance, gauge-invariant scalars, and back-transported operator
differences using deterministic synthetic matrices. This level needs only NumPy.

## Level 2: archived-result audit

`python scripts/verify_reference_results.py` reads the committed JSON records and
checks the exact statuses and logical gates used for the manuscript claims. In
particular, it asserts the BeH2 `REVIEW_REQUIRED` limitation rather than silently
treating that package as a pass.

## Level 3: electronic-structure rerun

The programs under `systems/*/workflow/` regenerate wave functions, exact
many-electron cross-overlaps, and decision files. Production runs used:

- Python 3;
- NumPy and SciPy;
- PySCF 2.13.1;
- a POSIX shell for the supplied wrappers.

Use a new empty work directory for each protocol and retain the generated
manifests. The workflows can be computationally expensive and are not run in
GitHub Actions.

The committed reference JSON files contain checksums for the original solution
objects. Those large restart files are not in ordinary Git. A byte-for-byte Level
3 audit therefore also needs the external wave-function archive described in
`DATA_AVAILABILITY.md`.

Machine-specific paths in release-facing records were replaced by `${..._ROOT}`
placeholders. The CP+ frozen-source snapshot is the one exception: it is retained
byte for byte because its canonical JSON fingerprint is checked by the code.
Embedded hash fields identify the original artifacts; repository-file integrity
is recorded separately in `SHA256SUMS` (which excludes the manifest itself).

## Why the three levels are separate

A passing matrix-algebra test does not establish wave-function stationarity or
manifold isolation. Conversely, a converged CASSCF point does not establish that
the retained subspace is transported faithfully. The decision records combine
these layers but keep their gates visible.
