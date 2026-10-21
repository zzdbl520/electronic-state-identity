# BeH2 Gate B0 verdict

## Decision

- **Switching/multistable block:** A1
- **Internal negative-control block:** B2
- **B0 block-identification gate:** PASS
- **Localization of an abrupt switch near 2.624 bohr:** NOT YET ESTABLISHED
- **Next gate:** B1 bidirectional A1 branch-support map

The `provisional_switching_irrep: B2` field in the original automatic summary
must not be used. The A1 and B2 candidate counts were tied, and the final label
was selected from an unordered set after a clustering test whose tolerances
were below the observed optimization-level numerical scatter.

## Scale-separated evidence

Across 2.6235--2.6250 bohr, comparison of the B0 common-SA2 initialization with
the RHF/right-continuation stationary solution gives:

| Block | Protocol | abs(delta Eavg) | max active angle | state leakage | 1-RDM distance | NOON RMS |
|---|---:|---:|---:|---:|---:|---:|
| A1 | SA3 | 10.34--10.43 mEh | 0.8106--0.8114 rad | 0.3277--0.3280 | 0.9368--0.9377 | 0.3696--0.3699 |
| A1 | SA4 | 9.54--9.61 mEh | 0.8881--0.8886 rad | 0.2427--0.2429 | 0.7246--0.7254 | 0.2549--0.2552 |
| B2 | SA3/SA4, all starts | at most 4.30 microEh | at most 2.39e-4 rad | at most 4.57e-8 | at most 2.23e-5 | at most 1.23e-6 |

Within A1, left continuation, fresh RHF, and right continuation reproduce the
same lower branch: their largest pairwise differences are 4.03 microEh in
state-averaged energy, 2.18e-3 rad in active-space angle, 2.03e-7 in state
leakage, 1.02e-4 in 1-RDM distance, and 3.58e-5 in NOON RMS. The separation
between this group and the common-SA2-derived branch is therefore larger by
roughly 10^3 in energy and active-space measures and by 10^6 in state leakage.

All 156 calculations converged. The largest reported orbital-gradient norm was
1.76e-6 and the largest density-change norm was 2.65e-6, both below the locked
3e-6 CASSCF gradient threshold.

## Protocol conditioning

For the best three-state subset extracted from SA4, the minimum SA3--SA4 state
singular value is about 0.973 on the common-SA2-derived A1 branch and about
0.820 on the lower A1 branch. B2 remains about 0.991. Thus the A1 lower branch
is usable as a strong protocol-sensitivity signal but must carry its edge
conditioning explicitly in any subsequent plaquette analysis.

## What B0 does and does not show

B0 establishes two distinct A1 stationary solutions and an essentially unique
B2 solution. It does not show a localized branch switch in the five-point
window: left and right continuation both track the same lower A1 branch from
2.616 to 2.627 bohr, while the common-SA2-derived upper branch remains distinct
at all five target geometries. A curvature maximum must therefore not be
claimed from B0 alone.
