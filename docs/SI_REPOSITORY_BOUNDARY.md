# Boundary between the Supporting Information and this repository

The Supporting Information should remain readable without functioning as a code
archive. A practical division is:

## Supporting Information

- derivation of the discrete transport and gauge transformations;
- smooth-limit/area-law statement and assumptions;
- electronic-structure model, basis, active space, state averaging, and
  convergence settings for each molecule;
- the preregistered gates needed to interpret a result;
- compact figures and tables containing the essential evidence;
- a short data/code-availability statement pointing here.

## Repository

- executable calculation and analysis code;
- complete protocol declarations and stopping rules;
- machine-readable decisions and full diagnostics;
- basis, mesh, gauge, opposed-history, and integrity checks;
- provenance-only antecedent workflows, clearly marked when superseded;
- continuous integration for algebraic and record-level verification.

The SI may quote a small number of final metrics, but should link to
`CLAIM_EVIDENCE_MATRIX.md` instead of reproducing long JSON tables or execution
logs. Large restart wave functions belong in a DOI-bearing data archive or Git
LFS, not in the SI PDF.
