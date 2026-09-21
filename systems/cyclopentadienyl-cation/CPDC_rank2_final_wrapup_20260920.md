# CPDC cc-pVTZ rank-2 final evidence wrap-up

Date: 2026-09-20

System: cyclopentadienyl cation, C5H5+

Model: equal-weight SA2/SA3-CASSCF(4e,5o)/cc-pVTZ

Final status: `PASS_RANK2_OPPOSED_HISTORY_REPLICATION`

## 1. Archive and provenance audit

The uploaded evidence archive has SHA-256

```text
a8b700b82ba8042068b38a5ec0afcf23c57320e7574cc8870f7d3a07a289a0ec
```

This exactly matches the checksum reported at creation. All 11 payload files
listed in the internal `SHA256SUMS` manifest independently verified as `OK`.
The archive contains the rank-2 decision, opposed-history authorization and
manifest, SA2-history decision, final SA3 decision, protocol, build-validation
record, README, and all three stage logs.

## 2. Executive scientific conclusion

The calculation qualifies the **two-dimensional lowest-singlet projector**
\(P_{12}\), not either individual adiabatic root. On the 80-point loop, the
internal splitting of the pair is only

\[
1.1371\times10^{-5}\text{--}1.8046\times10^{-5}\ E_{\mathrm h}
\quad (0.309\text{--}0.491\ \mathrm{meV}),
\]

whereas the independently rank-controlled SA3 calculation places the third
singlet at least \(2.77536\times10^{-2}\ E_{\mathrm h}\) (0.755 eV) above the
pair. The external isolation is therefore more than 1,537 times the maximum
internal splitting. This is precisely the regime in which individual-root
labels are fragile while the minimal rank-2 electronic manifold remains a
well-defined physical object.

The evidence chain is complete:

1. five independently initialized SA3 calculations recover the same lower-two
   projector and exclude the third state;
2. all 80 SA2 reference vertices and every projector edge pass;
3. random intramanifold \(U(2)\) gauges, reversed orientation, and every cyclic
   starting vertex leave the Wilson invariants unchanged to numerical precision;
4. the 40/80 meshes are converged under the frozen criterion;
5. clockwise and counterclockwise sequential SA2 histories close onto the same
   rank-2 manifold and reproduce the Wilson spectrum;
6. ten history-seeded SA3 calculations reproduce both the independent SA3
   references and one another.

The narrow authorized statement is therefore:

> The two lowest singlets of C5H5+ define an externally isolated,
> mesh-converged rank-2 electronic manifold on the frozen loop whose projector
> and Wilson spectrum are reproducible under opposed sequential CASSCF
> histories and independent/history-seeded SA3 rank controls.

## 3. Independent rank-2 qualification

### 3.1 Independent SA3 rank control

| Quantity | Worst observed value | Frozen gate | Margin |
|---|---:|---:|---:|
| Minimum SA2/SA3 projector singular value | 0.9998952895 | >= 0.95 | pass by 0.04990 |
| Maximum SA2/SA3 projector leakage | 1.48413e-4 | <= 3.0e-2 | 202 times below cap |
| Maximum third-root weight in SA2 | 1.21301e-4 | <= 5.0e-2 | 412 times below cap |
| Minimum SA3 third-singlet gap | 2.77536e-2 Eh | >= 2.0e-3 Eh | 13.88 times above floor |

All five independent SA3 wedge points passed wavefunction stationarity, spin,
MO orthonormality, CASCI reproduction, active-space retention, projector-rank,
third-state exclusion, and external-gap checks.

### 3.2 Eighty-point projector transport

| Quantity | Worst observed value | Frozen gate | Assessment |
|---|---:|---:|---:|
| Edge pair singular value | 0.9999431958 | >= 0.98 | pass |
| Edge pair leakage | 1.11674e-4 | <= 1.0e-2 | 89.55 times below cap |
| Active-space singular value | 0.9999992828 | >= 0.98 | pass |
| Active-space leakage | 4.68620e-7 | <= 1.0e-2 | 21,340 times below cap |
| Random-U(2) invariant error | 1.20027e-14 | <= 1.0e-10 | 8,331 times below cap |
| Reverse-orientation eigenphase error | 2.48065e-16 rad | <= 1.0e-10 rad | numerical precision |
| Cyclic-start invariant error | 1.33227e-15 | <= 1.0e-10 | numerical precision |

The 80-point reference Wilson eigenphases are

\[
\{-8.1932368569\times10^{-4},\ +8.1932368569\times10^{-4}\}\ \mathrm{rad},
\]

with trace 1.9999993287087285 and determinant
0.9999999999999928.

### 3.3 Mesh convergence

| Comparison | Eigenphase-set difference | Trace difference | Determinant difference |
|---|---:|---:|---:|
| 20/40 | 4.12155e-5 rad | 6.50442e-8 | 3.44169e-15 |
| 40/80 | 9.64128e-6 rad | 1.57057e-8 | 5.77316e-15 |
| Frozen 40/80 cap | 1.0e-3 rad | 1.0e-3 | 1.0e-3 |

The decisive 40/80 eigenphase difference is 103.72 times below its cap.

## 4. Opposed-history SA2 replication

Two histories were propagated from the same immutable p000 reference:

```text
clockwise:        p000 -> p001 -> ... -> p079 -> fresh p000 closure
counterclockwise: p000 -> p079 -> ... -> p001 -> fresh p000 closure
```

This comprises 158 non-anchor SA2 optimizations plus two independent closure
optimizations. All wavefunction and geometry checks passed.

### 4.1 Pointwise history/reference agreement

Across 79 non-anchor vertices and all three comparison channels (clockwise vs
reference, counterclockwise vs reference, and clockwise vs counterclockwise):

| Quantity | Global worst value | Location | Frozen gate |
|---|---:|---|---:|
| Pair singular value | 0.9999994478 | p068, 306 degrees, CW vs CCW | >= 0.98 |
| Pair leakage | 5.62254e-7 | p068, 306 degrees, CW vs CCW | <= 1.0e-2 |
| Active-space singular value | 0.9999999814 | p068, 306 degrees, CW vs CCW | >= 0.98 |
| Active-space leakage | 1.05395e-8 | p068, 306 degrees, CW vs CCW | <= 1.0e-2 |
| Average-energy difference | 2.06398e-10 Eh | p075, 337.5 degrees, CW vs reference | <= 1.0e-7 Eh |

The worst pointwise projector leakage is 17,786 times below the frozen cap;
the worst energy mismatch is 484 times below its cap.

### 4.2 Returned-p000 closure

| Comparison | Pair sigma min | Pair leakage | Active sigma min | Active leakage | Absolute dE (Eh) |
|---|---:|---:|---:|---:|---:|
| CW closure vs reference | 0.9999998302 | 1.72954e-7 | 0.9999999942 | 3.28762e-9 | 1.82752e-11 |
| CCW closure vs reference | 0.9999998412 | 1.61733e-7 | 0.9999999946 | 3.07863e-9 | 6.62226e-11 |
| CW closure vs CCW closure | 0.9999996736 | 3.32483e-7 | 0.9999999889 | 6.33216e-9 | 4.79474e-11 |

Thus both sequential histories return to the same starting-geometry manifold,
not merely to similar energies.

### 4.3 History Wilson controls

| Sequence | Minimum edge pair sigma | Maximum pair leakage | Wilson eigenphases (rad) | Gauge max error | Cyclic max error |
|---|---:|---:|---|---:|---:|
| Frozen reference | 0.9999431958 | 1.11674e-4 | +/-8.1932369e-4 | 1.20027e-14 | 1.33227e-15 |
| Clockwise history | 0.9999430112 | 1.11862e-4 | +/-8.8341804e-4 | 3.36009e-15 | 1.11022e-15 |
| Counterclockwise history | 0.9999430025 | 1.11871e-4 | +/-8.8347769e-4 | 5.47195e-15 | 1.77636e-15 |

| Wilson comparison | Eigenphase-set difference | Trace difference | Determinant difference | Result |
|---|---:|---:|---:|---|
| CW vs reference | 6.40943e-5 rad | 1.09136e-7 | 8.54872e-15 | pass |
| CCW vs reference | 6.41540e-5 rad | 1.09242e-7 | 6.10623e-15 | pass |
| CW vs CCW | 5.96546e-8 rad | 1.05406e-10 | 2.44249e-15 | pass |

The most restrictive completed gate is the history/reference Wilson
eigenphase comparison, which still remains 15.59 times below the frozen
1.0e-3-rad limit. The two opposed histories agree with one another much more
tightly: their eigenphase sets differ by only 5.97e-8 rad.

## 5. Final history-seeded SA3 replication

Ten equal-weight SA3 calculations were seeded from the two opposed histories
at 0, 9, 18, 27, and 36 degrees. All ten passed their six wavefunction-quality
checks.

### 5.1 SA2 seed retained inside SA3

| Quantity | Global worst value | Location | Frozen gate | Margin |
|---|---:|---|---:|---:|
| Pair singular value | 0.9998909110 | 36 degrees, CW | >= 0.95 | pass |
| Pair leakage | 1.52876e-4 | 36 degrees, CW | <= 3.0e-2 | 196 times below cap |
| Third-root SA2 weight | 1.29858e-4 | 36 degrees, CW | <= 5.0e-2 | 385 times below cap |

### 5.2 History-seeded SA3 vs independent SA3

| Quantity | Global worst value | Location | Frozen gate |
|---|---:|---|---:|
| Lower-pair singular value | 0.9999999999964 | 36 degrees, CW | >= 0.95 |
| Lower-pair leakage | 5.33784e-12 | 36 degrees, CW | <= 3.0e-2 |
| Pair-to-third cross weight | 4.74076e-12 | 27 degrees, CW | <= 5.0e-2 |
| Third-to-pair cross weight | 4.74076e-12 | 27 degrees, CW | <= 5.0e-2 |
| Average-energy difference | 8.52651e-13 Eh | 0 degrees, CCW | <= 1.0e-6 Eh |

The reference-projector leakage is 5.62 billion times below the cap, and the
largest energy difference is 1.17 million times below its cap.

### 5.3 Clockwise-seeded vs counterclockwise-seeded SA3

| Quantity | Global worst value | Location |
|---|---:|---|
| Lower-pair singular value | 0.999999999999974 | 18 degrees |
| Lower-pair leakage | 4.98490e-14 | 18 degrees |
| Pair/third cross weight | 9.61606e-15 | 9 degrees |
| Average-energy difference | 5.11591e-13 Eh | 0 degrees |

At the final rank-control level, the two deliberately opposed histories are
therefore indistinguishable to approximately double-precision numerical
accuracy.

## 6. Claim boundary

### Supported

- The lowest two singlets constitute an externally isolated rank-2 manifold
  on the frozen central loop.
- This manifold is smooth enough for high-fidelity projector transport.
- Its projector and Wilson spectrum are invariant to intramanifold basis
  rotations and converged with respect to the tested meshes.
- Its identity survives clockwise and counterclockwise sequential CASSCF
  histories, independent returned-p000 closures, and expansion from SA2 to
  SA3.
- The result may be included as **rank-2 manifold evidence**, exactly as
  recorded by `result_authorizes_inclusion_as_rank2_manifold_evidence=true`.

### Not supported by this protocol

- a signed one-state overlap;
- global identity or continuous tracking of either individual adiabatic root;
- a scalar Berry phase;
- odd/even CI parity or the number of internal conical intersections;
- an absolute-ground-state claim.

The retained field `manuscript_claim_authorized=false` is therefore not a
failed numerical result. It is a fail-safe preventing the executable from
authorizing a broader topological or individual-root claim. The narrower final
authorization is the explicit
`result_authorizes_inclusion_as_rank2_manifold_evidence=true`.

The object should consequently be called an **isolated rank-2 lowest-singlet
manifold** or, when the independent geometry/gap evidence is also cited, a
**secondary near-degeneracy candidate**. This projector package alone should
not be described as topological proof of a secondary CI.

## 7. Manuscript-ready main-text paragraph

At the cc-pVTZ cyclopentadienyl-cation near-degeneracy, the individual roots
remain separated by only 1.14-1.80e-5 Eh along the 80-point loop, whereas the
third singlet lies at least 2.78e-2 Eh above the pair. We therefore replaced
individual-root tracking by the minimal projector P12 onto the two lowest
singlets. Exact complete-CASSCF overlaps gave a minimum edge singular value of
0.999943 and a maximum projector leakage of 1.12e-4; random U(2) gauges,
reversed traversal, all cyclic starting vertices, and 40/80-point refinement
preserved the Wilson invariants. To distinguish a physical two-state manifold
from a path-dependent multireference solution, we then propagated independent
clockwise and counterclockwise SA2-CASSCF histories, including fresh closures
at the common starting geometry. Their worst pointwise cross-history leakage
was 5.62e-7, and their Wilson eigenphase sets differed by only 5.97e-8 rad.
Finally, ten history-seeded SA3 calculations reproduced the independently
initialized SA3 lower-pair projectors with leakage below 5.34e-12 and agreed
across the opposed histories to 4.98e-14. Thus the stable electronic object is
the externally isolated rank-2 manifold rather than either individual
adiabatic label; no scalar Berry phase or internal-CI count is inferred.

## 8. Manuscript-ready SI methods/validation paragraph

The two lowest singlets were represented by the rank-2 projector
\(P_{12}=\sum_{a=1}^{2}|\Psi_a\rangle\langle\Psi_a|\) from equal-weight
SA2-CASSCF(4e,5o)/cc-pVTZ wavefunctions. Projector transport was evaluated from exact complete-CASSCF
state-overlap matrices; intramanifold basis dependence was removed through the
unitary polar factor of each 2x2 overlap. The Wilson product was tested on
20-, 40-, and 80-point meshes using five independent random U(2) gauges,
reversed orientation, and every cyclic starting vertex. Rank isolation was
tested independently by equal-weight SA3-CASSCF calculations at five
prequalified symmetry-wedge reference angles (0, 9, 18, 27, and 36 degrees), requiring
retention of the lower-two projector, exclusion of the third root, and an
E3-E2 gap of at least 2e-3 Eh. Optimization-history dependence was then tested
using two sequential SA2 histories initiated from the same p000 solution but
propagated in opposite directions around the 80-point loop; each history ended
in a newly optimized p000 closure. All history projectors were compared with
the frozen reference and with the opposite history, and their Wilson spectra
were recomputed including the returned closures. Only after this stage passed
were ten additional SA3 calculations initialized from the two history-specific
SA2 frames. No signed single-state overlap, individual-root assignment, scalar
Berry product, or CI-count criterion entered any acceptance decision.

## 9. Suggested table caption

**Validation of the isolated lowest-singlet rank-2 manifold of C5H5+.** Exact
complete-CASSCF projector overlaps demonstrate high-fidelity transport on the
80-point loop, convergence of the Wilson spectrum from 40 to 80 points,
closure of independently propagated clockwise and counterclockwise SA2
histories, and reproduction of independently initialized SA3 lower-pair
projectors by ten history-seeded SA3 calculations. Reported values are global
worst cases over the indicated meshes, histories, or rank-control points.

## 10. Archival limitation and preservation action

The uploaded file is a compact, internally checksummed **evidence archive**.
It contains the decisions and logs but not the referenced SA2/SA3 `solution.npz`
wavefunctions. It is therefore sufficient for numerical reporting and audit of
the recorded decisions, but it is not a standalone rerunnable dataset.

For full reproducibility, preserve durable copies of at least these work roots
before any scratch cleanup:

```text
${CPDC_ARCHIVE_ROOT}/CPDC_secondary_CI_ccpVTZ_lift_v1_work
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_relaxed_multiring_v1_work
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_central_gap_landscape_v1_work
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_projector_v1_work
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_opposed_history_v1_work
```

The compact archive should be retained unchanged under its recorded outer
SHA-256.
