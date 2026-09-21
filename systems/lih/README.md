# LiH: bond-length–static-field loops

LiH is the explicit false-negative test for endpoint-based root tracking. The two
routes vary bond length and static field in opposite orders. Conventional local
matches compose to the same endpoint permutation, while the unitary-polar route
transports and back-transported dipole matrices differ.

## Workflow

The scripts are ordered by their original validation stages:

1. `workflow/LiH_S0_SA4_qualification.py` — qualify the SA4 model and state space;
2. `workflow/LiH_S1_physical_field_locator.py` — locate a preregistered candidate
   window;
3. `workflow/LiH_S2_physical_field_confirmation.py` — independent/opposed-history
   confirmation;
4. `workflow/LiH_S3_fixed_window_basis_robustness.py` — repeat the fixed window in
   the larger basis;
5. `workflow/LiH_S4_cross_basis_audit.py` — compare the basis protocols;
6. `workflow/LiH_ccpvtz_one_shot.py` — construct and audit the cc-pVTZ one-shot
   run.

Run `python SCRIPT.py selftest` before a production calculation. The legacy run
notes are retained alongside this file, with machine paths replaced by portable
placeholders.

## Reference records

- `reference/aug-cc-pVDZ/`: locator and opposed-history confirmation;
- `reference/cc-pVTZ/`: fixed-window basis check and independent audit.

The record-level expected statuses are
`PUBLICATION_DEMONSTRATION_READY` and `BASIS_ROBUSTNESS_CONFIRMED`.
