# Claim-to-evidence map

Paths below are relative to [the October data directory](../data/revision-2026-10-21/).

| Statement | Evidence | Qualification |
|---|---|---|
| Polar transport and subspace fidelity | Root `src/` and `tests/`; `analysis/explicit_generator_scaling.json` | Current explicit generators differ from historical `records/math_validation/`. |
| LiH same assignments with different transports/operators | `analysis/augccpvtz_summary.json`, `records/lih_augccpvtz_local_refinement/` | Complete four-electron overlaps, squared-overlap assignments, final 154-edge boundary. |
| LiH mesh and opposed-history checks | `analysis/augccpvtz_refinement_validation/` | Intermediate failures retained. |
| LiH shrinking-loop area scaling | `analysis/lih_augccpvtz_nested.csv`, production records | Four concentric loops; descriptive regression. |
| Three LiH bases agree qualitatively | `records/lih_revision_augccpvdz/`, `records/lih_revision_ccpvtz/`, aug-cc-pVTZ records | Separate within-basis results; not basis convergence. |
| Original selection disclosed | `analysis/lih_100_locator_plaquettes.csv`, `analysis/lih_100_stride_distribution.csv`, `analysis/lih_100_complete_overlap_rectangles.csv` | None of the original unrefined rectangles passes complete-overlap fidelity. |
| BeH2 constraint sensitivity | `records/beh2_model_comparison/`, `records/BeH2/source_data/` | Summary-level energy evidence; failed SA3/SA4 fidelity prevents physical branch-transition inference. |
| C5H5+ externally isolated rank-2 pair | `records/CPDC/`, `analysis/cpdc_sa3.csv` | No individual-root, scalar Berry-phase, CI-count or absolute-ground-state claim. |
| C5H5+ initialization sensitivity | `analysis/cpdc_histories.csv`, `analysis/cpdc_phase_reference_levels.csv` | Reference/history phase difference 6.4154e-5 rad exceeds 40/80 mesh difference 9.6413e-6 rad. |
| H3/O3 historical diagnostics | `records/H3/`, `records/O3/` | Pointwise-refinement/weighted-SA2 failures remain explicit. |

`python scripts/verify_reference_results.py` checks current records and LiH
raw-file hashes. The September audit is preserved as
`scripts/verify_legacy_reference_results.py` for historical checks only.
