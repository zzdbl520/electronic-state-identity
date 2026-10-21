# Mixed variational curvature: algebraic validation

This package is the first validation layer for the proposed mixed curvature in
nuclear-coordinate--variational-protocol space. It tests the matrix mathematics
before any result is attributed to H3, BeH2, O3, or an electronic-structure code.

## Run

```bash
python3 mixed_variational_curvature_validation.py
```

To save a machine-readable record:

```bash
python3 mixed_variational_curvature_validation.py --json validation_results.json
```

The only dependency is NumPy.

`validation_subsection.tex`, `mixed_curvature_scaling.pdf`, and the compact
quantitative table are formatted for direct inclusion in a manuscript
Validation section. The complete ten-test table is retained separately for the
Supporting Information.

## What is verified

1. The two definitions
   `||W-I||_F/(2 sqrt(m))` and `||K_qp-K_pq||_F/(2 sqrt(m))` agree numerically.
2. `0 <= C_qp <= 1` on random unitary transports.
3. Independent phases, root permutations, and arbitrary corner-unitary gauges
   cannot create curvature when the underlying state manifold is unchanged.
4. A genuinely nonclosing plaquette retains the same curvature, edge singular
   values, leakage, and holonomy eigenphases after independent corner gauges.
5. Subspace leakage and mixed curvature are distinct: positive leakage can occur
   with exactly zero curvature.
6. A nearly rank-deficient edge is flagged by `eta_edge`, even if its SVD still
   returns a formal polar transport.
7. Reversing plaquette orientation maps `W` to `W^dagger` and leaves scalar
   curvature unchanged.
8. For two constant noncommuting connections, `C(h,h)` scales as `h^2`; when one
   side is fixed, `C(h,p0)` scales as `h`; the small-plaquette coefficient tends
   to `||[A_q,A_p]||_F/(2 sqrt(m))`.
9. A determinant-overlap/CI-contraction calculation agrees with an independent
   direct expansion of the same states in a common Fock basis. The construction
   explicitly includes a nontrivial cross-overlap of the two AO frames and shows
   that the physical many-electron overlap is not a CI-coefficient dot product.

## What remains before a physical claim

This suite validates the gauge geometry, not the chemistry. The next layers are:

- **Production overlap-engine validation:** reproduce the synthetic Fock-space
  equality with the actual AO, active-orbital, determinant, and CI data exported
  by the chosen electronic-structure code.
- **Negative physical control:** H3 around its symmetry-enforced conical
  intersection should show nuclear holonomy while balanced protocols give small
  mixed curvature.
- **Positive variational control:** a known CASSCF stationary-solution switch
  should give a localized mixed-curvature signal and identifiable orbital/CI or
  density changes.
- **Target calculation:** only after those controls should the same locked
  thresholds and diagnostics be applied to the production O3 data.

The decisive falsification condition is simple: if mixed curvature persists
under basis refinement, step-size reduction, tighter convergence, and protocol
balancing in the negative control, it cannot yet be interpreted as a diagnostic
of variational-branch switching.
