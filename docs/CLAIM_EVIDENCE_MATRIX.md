# Claim-to-evidence matrix

This table is the shortest route from a manuscript statement to the numerical
record that supports, qualifies, or limits it.

| Manuscript-level statement | Required checks | Primary record | Repository audit |
|---|---|---|---|
| Unit polar factors provide gauge-covariant edge transports; singular values measure retained-subspace fidelity. | Polar unitarity, endpoint gauge covariance, singular-value invariance. | System workflow self-tests. | `tests/test_transport.py`; `scripts/run_workflow_selftests.py` |
| LiH endpoint matching gives a false negative. | Same composed permutation on both routes; nonzero `C_qp`; nonzero back-transported dipole difference; stationary and faithful edges. | `systems/lih/reference/aug-cc-pVDZ/LiH_S2_decision.json` | `verify_lih()` |
| The LiH observation is not specific to one basis or one optimization history. | Opposed-history agreement and fixed-window larger-basis reproduction. | LiH S2 and `reference/cc-pVTZ/LiH_S3_decision.json`; S5 independent audit. | `verify_lih()` |
| The BeH2 apparent branch end is induced by an active-orbital rotation restriction. | Event in the restricted model; single clean solution in unconstrained all-electron CAS(6,6) and valence CAS(4,6). | `systems/beh2/reference/G0C_forensic_summary.json` | `verify_beh2()` |
| The BeH2 G0C plaquette package is not an unconditional transport pass. | Every stride-1 upper edge would have to satisfy the local gate; it does not. | Same G0C summary: `status=REVIEW_REQUIRED`, `main_figure_candidate=false`. | `verify_beh2()` asserts the limitation. |
| The lower CP+ pair is externally isolated and can be treated as a rank-2 manifold. | Independent SA3 projector arbitration and third-singlet gap. | `CPDC_ccpVTZ_rank2_projector_v1_decision.json` | `verify_cyclopentadienyl_cation()` |
| CP+ transport is faithful, U(2)-gauge invariant, mesh converged, and opposed-history reproducible without individual-root assignments. | Point/edge gates, random U(2), 40/80 mesh comparison, clockwise/counterclockwise replication, root-assignment flag false. | Projector, history, and opposed-history decisions in `systems/cyclopentadienyl-cation/reference/`. | `verify_cyclopentadienyl_cation()` |

Passing the listed checks does not assign a scalar Berry phase to a member of a
near-degenerate pair, count internal conical intersections, or turn route
nonclosure by itself into a topological invariant.
