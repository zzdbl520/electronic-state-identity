# INVEST BeH2 G0C v1.1: zero-new-QC forensic comparison

This package converts the completed G0 and G0B wavefunction archives into the
publication-facing BeH2 comparison. It performs **no new SCF or CASSCF
optimization**. PySCF is used only to evaluate nonorthogonal many-electron
overlaps between archived wavefunctions.

The comparison is deliberately restricted to three cc-pVTZ models:

1. legacy CAS(6e,6o), with active MO 0 excluded from orbital rotations;
2. unconstrained all-electron CAS(6e,6o);
3. standard valence CAS(4e,6o), with one optimized inactive doubly occupied
   core and no orbital excluded from optimization.

## Run

Use the same Python environment that ran G0/G0B:

```bash
unzip INVEST_BeH2_G0C_forensic_v1_1.zip -d INVEST_BeH2_G0C_forensic_v1_1
cd INVEST_BeH2_G0C_forensic_v1_1

python INVEST_G0C_forensic_comparison.py --self-test

bash run_G0C_forensic_comparison.sh \
  ${BEH2_WORK_ROOT}/INVEST_BeH2_G0_work \
  ${BEH2_WORK_ROOT}/INVEST_BeH2_G0B_work \
  ${BEH2_WORK_ROOT}/INVEST_BeH2_G0C_forensic_work \
  | tee ${BEH2_WORK_ROOT}/INVEST_BeH2_G0C_forensic_work.stdout
```

The runner first requires the locked decisions
`STOP_LEGACY_CONSTRAINT_ONLY` and `CONFIRMED_CONSTRAINT_INDUCED`. It then checks
the model semantics and uses only the common six-point grid.

## Outputs

- `G0C_history_separation.csv`: SA3 and SA4 history comparisons for all models;
- `G0C_mixed_plaquettes.csv`: all mixed nuclear--protocol plaquettes, including
  `C_qp`, edge singular values, leakage, assignment margins, holonomy phases,
  and numerical invariance checks;
- `G0C_forensic_summary.json`: machine-readable decision and manuscript-safe
  interpretation;
- `G0C_constraint_forensic_comparison.png`: 600 dpi figure;
- `G0C_constraint_forensic_comparison.pdf`: vector figure.

## Interpretation lock

Authorized wording:

> The archived BeH2 event is a constraint-induced variational branch
> termination within a deliberately restricted orbital-optimization manifold,
> not a physical spinodal of unconstrained CASSCF.

Do not describe the legacy calculation as frozen-core CAS(6e,6o), and do not
revive the physical-spinodal or intrinsic-branch-collapse language.

The summary deliberately applies two different gates:

- `FORENSIC_BEH2_READY` requires the pre-existing BeH2 conditioning criterion
  (eta at the peak at least 0.85), an unambiguous local assignment, a legacy
  event absent from both clean models, and at least tenfold C_qp contrast.
- `strict_molecular_false_negative_candidate` uses eta at least 0.98 and edge
  leakage at most 0.01. This stricter criterion is reserved for the flagship
  molecular false-negative test and is not retroactively imposed on BeH2.

Return the summary, both CSV files, and the PNG if either gate fails. Do not
change thresholds or launch additional calculations before interpreting the
failure.
