# LiH S0 local validation record

Date: 2026-09-16 (UTC)

Driver: `LiH-S0-SA4-qualification-v1.1`

Validation environment: PySCF 2.14.0, NumPy 2.5.3, two CPU threads

## Production-protocol result

```text
LiH S0 RESULT
Status: PROMOTE_TO_PHYSICAL_FIELD_PILOT
Worst active-state edge: eta=0.995479609531;
  leakage=0.003353238868; margin=0.491151598514
Minimum excluded-root gap=8.871453387937e-02 Eh;
  maximum gradient=5.904e-07
```

All eight immutable checks passed:

- numerical validation;
- exact model semantics;
- all wavefunctions stationary;
- direct and two opposed histories give the same solution;
- first excluded A1 root isolated;
- active-orbital subspace continuous;
- every elementary active-state edge strict;
- every elementary assignment unambiguous.

The production calculation comprised 39 independent direct wavefunctions, 55
short-bond-outward continuation wavefunctions, and 75 long-bond-inward
continuation wavefunctions (169 total). Fifteen primary optimizations reached a
small-gradient plateau and invoked the deterministic symmetry-adapted
second-order fallback. No solution was selected from competing starts.

Additional extrema:

- maximum direct/history state-averaged energy difference:
  `9.624212538028587e-11 Eh`;
- minimum direct/history active-state subspace singular value:
  `0.9999999996235768`;
- maximum direct/history active-state subspace leakage:
  `2.299038737163528e-10`;
- minimum fifth-minus-fourth A1 root gap:
  `0.08871453387936867 Eh` at `R = 3.00 Å`;
- failed edges, failed histories, and nonstationary wavefunctions: zero.

## Development mesh rejection

Before v1.1 was frozen, a development-only 0.10 Å mesh was evaluated with the
same scientific thresholds. It was rejected because the 2.30→2.40 Å and
2.40→2.50 Å edges had active-state leakages `0.013290666324` and
`0.010846533057`, above the immutable `0.01` limit. The threshold was not
relaxed. The final 0.05 Å production mesh reduced the worst leakage to
`0.003353238868`.

The primary one-step CASSCF optimizer also stalled on several continuation
points. Restarting the same stationary problem from its own final orbitals with
PySCF's symmetry-adapted second-order optimizer reduced the maximum final
gradient to `5.904e-7` and made both opposed histories coincide with the direct
solution. This deterministic fallback is encoded in v1.1 for every point and
is recorded per wavefunction.

## Authorized conclusion

LiH has passed the one-dimensional numerical, variational, and model-closure
qualification required to design a physical `R x F_z` pilot. This record does
not itself authorize a manuscript claim or establish a molecular
local-pass/global-fail example.
