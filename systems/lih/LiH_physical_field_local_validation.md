# Local validation audit: LiH physical-field demonstration

This audit records a real PySCF 2.14 execution of the delivered S1 and S2
drivers.  The local data are a development validation, not a substitute for an
independent production run on the author's machine.

## Scientific model

- LiH / aug-cc-pVDZ / C2v singlet A1
- equal-weight SA4-CASSCF(2e,5o), active irreps A1^3 B1^1 B2^1
- one optimized inactive core; no frozen orbital
- four retained roots and a deterministic five-singlet A1 CASCI probe
- physical coordinates: bond length R and uniform static field Fz
- exact active-only CAS-wavefunction overlaps

The final fixed mesh was centered from a separate development locator on the
Stark avoided-crossing ridge.  S1 selected a rectangle from the fixed mesh;
S2 then froze that selection and supplied new opposed-history wavefunctions.
This should be described as discovery followed by confirmation, not as an
unseen prediction.

## S1 result

Status: `CANDIDATE_WINDOW_FOUND`

All S1 decision checks passed:

- numerical and gauge validation;
- exact model semantics;
- all 25 independent wavefunctions stationary;
- first excluded A1 singlet isolated;
- all 40 elementary grid edges strict;
- conventional composed assignments identical on the selected rectangle;
- nonzero continuous holonomy;
- nonzero dipole consequence.

Key grid diagnostics:

| Quantity | Local result |
|---|---:|
| Maximum orbital-gradient component | 2.298e-7 |
| Minimum roots-2/3 gap | 2.388041e-4 Eh |
| Location of minimum gap | R=5.85 A, Fz=0.01150 au |
| Minimum excluded-singlet gap | 6.867778e-2 Eh |
| Maximum retained-root probe error | about 2e-14 Eh |
| Worst elementary-edge eta | 0.999458160218 |
| Worst elementary-edge leakage | 7.594310e-4 |
| Minimum elementary assignment margin | 0.206936799388 |
| Strict candidate rectangles | 33 |

Selected direct-edge window:

- R = 5.80 -> 5.95 A
- Fz = 0.01050 -> 0.01150 au
- minimum roots-2/3 gap anywhere inside = 2.388041e-4 Eh
- minimum excluded-singlet gap inside = 7.070531e-2 Eh
- C_RF = 3.684705101236e-4
- Wilson/holonomy phases = (-9.208838e-4, -4.879931e-4,
  +4.879931e-4, +9.208838e-4) rad
- both conventional composed assignments = [0, 1, 3, 2]
- dipole path difference = 1.7418106e-2 au
- normalized dipole path difference = 7.5861747e-4
- gauge-test C_RF error = 5.00e-16
- gauge-test dipole error = 2.16e-14 au

Direct boundary-edge checks:

| Edge | eta | leakage | normalized margin | assignment |
|---|---:|---:|---:|---|
| AB: R at low field | 0.995145244305 | 0.006773245671 | 0.494923453096 | [0,1,2,3] |
| BD: field at high R | 0.997415113309 | 0.002281025557 | 0.489558295145 | [0,1,3,2] |
| AC: field at low R | 0.997256023348 | 0.002412639277 | 0.484933733109 | [0,1,2,3] |
| CD: R at high field | 0.995149186544 | 0.006698090530 | 0.474600354612 | [0,1,3,2] |

Thus the two routes produce the same discrete assignment while the continuous
subspace transports differ and yield a gauge-invariant dipole consequence.

## S2 result

Status: `PUBLICATION_DEMONSTRATION_READY`

S2 generated 50 new history wavefunctions, one complete grid for each of two
exactly opposed snake traversals.  All confirmatory checks passed.

| Quantity | Local result |
|---|---:|
| Maximum history-wavefunction gradient | 8.604399e-8 |
| Maximum direct/history energy difference | 1.989342e-11 Eh |
| Minimum direct/history state sigma | 0.999999999949 |
| Maximum direct/history state leakage | 3.689471e-11 |
| Minimum direct/history active-orbital sigma | 0.999999998454 |
| Maximum direct/history active-orbital leakage | 1.238330e-9 |
| Maximum history C_RF absolute error | 4.679314e-9 |
| Maximum normalized-dipole absolute error | 3.611289e-9 |

The selected window remained a strict candidate under the independent direct
solutions and both opposed histories.  No threshold was relaxed to obtain the
passing result.

## Authorized manuscript-level conclusion after independent reproduction

In this fixed LiH R x static-field window, every selected boundary edge is
faithful and the two paths give the same conventional composed state assignment,
yet continuous subspace transport is path dependent and produces a nonzero,
gauge-invariant dipole consequence.  The result is stationary, externally
isolated, and independent of two opposed optimization histories.

This supports a physically defined molecular local-pass/global-fail example.
It does not by itself prove a conical intersection, and it should not be framed
as an experimentally routine field strength.
