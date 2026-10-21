# Validation report: mixed variational curvature

Date: 2026-08-10  
Reference implementation: `mixed_variational_curvature_validation.py`  
Runtime: Python 3, NumPy 2.3.5  
Default tolerance: `1e-12`

## Verdict

The discrete plaquette definition is algebraically self-consistent under the
tests implemented here. All 10 tests pass at the default seed, and the complete
suite also passes for 10 additional random seeds. The stress run contains 2,000
random three-state plaquettes in total.

This establishes a clean mathematical and implementation baseline. It does not
yet establish that a peak in a molecular calculation is caused by a change of
CASSCF stationary solution; that requires the physical controls listed below.

## Default-seed results

| Test | Result | Quantitative outcome |
|---|---:|---|
| Identity plaquette | PASS | `C=0`, `eta_edge=1`, `lambda_edge=0` |
| Phase and root permutation | PASS | No spurious curvature |
| General pure gauge | PASS | Arbitrary independent `U(4)` corner frames close |
| Nonzero-curvature gauge invariance | PASS | `C=0.09003882`; both paths are endpoint-covariant and `W` transforms by similarity |
| Leakage without curvature | PASS | `C=0` with `lambda_edge=0.28` |
| Rank-loss warning | PASS | `eta_edge=1.0e-14` identifies the bad edge |
| Orientation reversal | PASS | `W -> W^dagger`, scalar `C` unchanged |
| Many-electron overlap contraction | PASS | Direct Fock-space error `1.48e-16`; naive CI-dot-product error `0.569` |
| Continuous-limit scaling | PASS | Equal-step slope `1.99892892`; one-side-fixed slope `0.99991434` |
| Random bounds and formula equivalence | PASS | 200 plaquettes; maximum formula discrepancy `2.22e-16` |

The ratio between the observed small-plaquette coefficient and
`||[A_q,A_p]||_F/(2 sqrt(m))` is `0.99998698` at the smallest tested step.

## What the result supports

- The normalization gives `0 <= C <= 1` for unitary path transports.
- The holonomy and path-difference expressions for `C` are numerically
  equivalent to machine precision.
- Root flips, phases, and unitary rotations within the selected state manifold
  cannot by themselves create a finite mixed curvature.
- `eta_edge` and `lambda_edge` carry information that `C` intentionally does
  not: conditioning/leakage must therefore be reported beside curvature.
- The finite plaquette has the stated non-Abelian continuous limit.
- A genuine many-electron cross-overlap, including AO cross-overlap, orbital
  overlap, determinant minors, and CI contraction, agrees with a direct Fock-
  space calculation. A CI-vector dot product does not.

## What would falsify or weaken the physical interpretation

1. A curvature maximum moves or disappears under nuclear-step refinement.
2. It correlates with a collapsing `eta_edge`, so the polar factor is being
   inferred from an ill-conditioned common manifold.
3. It persists in the balanced H3 conical-intersection negative control with no
   protocol-specific branch change.
4. It cannot be reproduced when the production overlap engine is checked
   against a direct small-space contraction.
5. The alleged branch switch is not accompanied by any interpretable change in
   orbital principal angles, one-particle density, occupations, CI character, or
   local energetic residuals.

## Next calculation gate

The next step should be a molecule-level two-control study with all thresholds
fixed in advance:

1. H3: nuclear holonomy present, mixed curvature expected to remain small for
   balanced protocols.
2. A known CASSCF branch-switch example: localized mixed curvature expected,
   with its microscopic source identified by the component diagnostics.
3. Only after both controls behave correctly, apply the unchanged pipeline to
   the target O3 data.

The current result therefore moves the proposal from an attractive definition
to a testable diagnostic. The publishable claim begins only at the control-study
stage.
