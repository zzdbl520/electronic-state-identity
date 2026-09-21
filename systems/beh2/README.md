# BeH2: orbital-constraint forensic comparison

This directory separates the current conclusion from the earlier restricted
workflow that motivated the investigation.

The comparison uses three models: the legacy CAS(6,6) calculation with active MO
0 excluded from orbital rotations, an unconstrained all-electron CAS(6,6), and a
standard valence CAS(4,6). The apparent branch termination is absent in both clean
models. The authorized interpretation is therefore a constraint-induced
termination in the deliberately restricted optimization manifold, not a physical
spinodal or an intrinsic end of the unconstrained CASSCF branch.

## Current forensic package

- `forensic/workflow/`: the three-model comparison and supporting analyzers;
- `reference/G0C_forensic_summary.json`: the machine-readable summary.

The summary status is deliberately `REVIEW_REQUIRED`: not every stride-1 upper
edge passes its local-fidelity gate, so this package is not authorized as a main
quantitative plaquette figure. That limitation does not reverse the clean-model
comparison, but it must remain visible.

The original G0/G0B per-point `solution.npz` trees are not in ordinary Git. The
forensic code and final summary are present; a matrix-level replay additionally
requires the external restart archive described in `../../docs/DATA_AVAILABILITY.md`.

## Antecedent restricted workflow

`antecedent_restricted_workflow/` is retained for provenance because it generated
the event investigated later. Its original physical-collapse interpretation is
superseded by the three-model forensic comparison above. Do not cite the
antecedent README as the current scientific conclusion.
