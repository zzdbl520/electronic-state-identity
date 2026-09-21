# CPDC cc-pVTZ rank-2 opposed-history replication v1

## Why this is the next calculation

The frozen antecedent has passed:

```text
PASS_RANK2_PROJECTOR_QUALIFICATION
40/80: dphase=9.641e-06 rad dtrace=1.571e-08 ddet=5.773e-15
```

The eigenphase-pair difference is about 104 times smaller than the frozen
`1e-3 rad` mesh criterion; the determinant difference is at double-precision
roundoff.  Together with the independent SA3 rank test, edge fidelity, random
`U(2)` gauges, reverse traversal, and cyclic-start controls, this establishes
that the isolated two-state manifold is a numerically stable object.

One question remains before using it as manuscript evidence: does the same
rank-2 manifold survive deliberately opposed orbital-optimization histories?
This package performs that replication without reopening the failed scalar
phase route.

## Frozen two-gate design

### Gate 1: opposed SA2 histories

Starting from the same immutable `p000` solution on the qualified 80-point
`central_inner` ring, two sequential equal-weight SA2-CASSCF(4e,5o)/cc-pVTZ
chains are run:

```text
clockwise:        p000 -> p001 -> ... -> p079 -> new p000 closure
counterclockwise: p000 -> p079 -> ... -> p001 -> new p000 closure
```

Every new point is initialized from the immediately preceding solution in its
own history.  This stage therefore adds 160 SA2 calculations: 79 non-anchor
vertices and one fresh closure in each direction.

At every vertex the two histories are compared with one another and with the
independently seeded reference by exact complete-CASSCF determinant overlaps.
The returned `p000` solutions are also included explicitly in the Wilson
products.  Both histories are expressed in the same geometric orientation
before their unordered Wilson eigenphase pairs, traces, and determinants are
compared.

The SA3 stage is code-blocked unless Gate 1 returns exactly:

```text
CPDC_RANK2_HISTORY=PASS_RANK2_OPPOSED_HISTORY
```

### Gate 2: history-seeded SA3 replication

If and only if Gate 1 passes, each history supplies the MO seed for five
equal-weight SA3-CASSCF(4e,5o)/cc-pVTZ calculations.  The angles are the five
ones already covered by the independent SA3 reference wedge:

| independent 40-point reference index | 80-point history index | angle |
|---:|---:|---:|
| 0 | 0 | 0 degrees |
| 1 | 2 | 9 degrees |
| 2 | 4 | 18 degrees |
| 3 | 6 | 27 degrees |
| 4 | 8 | 36 degrees |

This explicit `0,2,4,6,8` mapping is important: the first five vertices of an
80-point ring would instead be `0,4.5,9,13.5,18` degrees and are not the five
frozen reference geometries.

The ten new SA3 solutions must reproduce the independently initialized SA3
lower-two-state projector, exclude the third state, agree across the opposed
histories, and reproduce the SA3 average energy.

## What a pass means

The terminal pass is:

```text
PASS_RANK2_OPPOSED_HISTORY_REPLICATION
```

It authorizes the narrow statement that the isolated lowest-singlet rank-2
manifold `P12` on the frozen loop is independent of the two deliberately
opposed optimization histories and survives independent/history-seeded SA3
rank controls.

It does **not** authorize any of the following:

- a signed one-state overlap or scalar Berry product;
- an individual-root permutation or assignment;
- a scalar Berry phase;
- the parity or number of conical intersections inside the loop;
- an absolute-ground-state claim.

## Installation and integrity check

```bash
tar -xzf CPDC_ccpVTZ_rank2_opposed_history_v1_20260920.tar.gz
cd CPDC_ccpVTZ_rank2_opposed_history_v1
sha256sum -c SHA256SUMS_CPDC_RANK2_OPPOSED_HISTORY_v1
```

The launcher repeats the checksum and executable self-test before every
stage.

## Paths

The defaults match the completed antecedent runs.  They can be overridden as
shown here:

```bash
export CPDC_OPH_SOURCE_ROOT=${CPDC_ARCHIVE_ROOT}/CPDC_secondary_CI_ccpVTZ_lift_v1_work
export CPDC_OPH_LOOP_ROOT=${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_relaxed_multiring_v1_work
export CPDC_OPH_SCREEN_ROOT=${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_central_gap_landscape_v1_work
export CPDC_OPH_RANK2_ROOT=${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_projector_v1_work
export CPDC_OPH_ROOT=${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_opposed_history_v1_work
```

Use a new `CPDC_OPH_ROOT`.  Do not place this run inside or overwrite any
antecedent root.

## Step 1: freeze authorization

```bash
bash run_CPDC_ccpVTZ_rank2_opposed_history_v1.sh qualify
tail -f CPDC_ccpVTZ_rank2_opposed_history_v1_qualify.stdout
```

Proceed only if the terminal line is:

```text
CPDC_OPPOSED_HISTORY_SOURCE=PASS_OPPOSED_HISTORY_AUTHORIZATION
```

Authorization verifies the exact passed rank-2 decision, its 80 reference
solutions, the five independent SA3 solutions, the antecedent decisions, and
the continued absence of a scalar-topology decision.

## Step 2: run Gate 1

The default launches two concurrent sequential histories.  Each requests 12
threads and 200000 MB, for a maximum simultaneous request of 24 threads and
400000 MB.

```bash
export CPDC_OPH_HISTORY_THREADS=12
export CPDC_OPH_HISTORY_MEMORY_MB=200000

bash run_CPDC_ccpVTZ_rank2_opposed_history_v1.sh history
tail -f CPDC_ccpVTZ_rank2_opposed_history_v1_history.stdout
```

The run is restartable.  Within each direction it remains strictly sequential;
a completed point is reused only after its geometry, descriptor, immediate
seed path/hash, model, and protocol identity are revalidated.

The Gate-1 decision is written to:

```text
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_opposed_history_v1_work/CPDC_ccpVTZ_rank2_history_v1_decision.json
```

If Gate 1 does not pass, stop.  Do not launch `sa3` and do not calculate a
scalar phase.

## Step 3: run Gate 2 only after the Gate-1 pass

The default launches at most four SA3 jobs, each requesting 6 threads and
100000 MB.  The maximum simultaneous request is therefore 24 threads and
400000 MB.

```bash
export CPDC_OPH_SA3_THREADS=6
export CPDC_OPH_SA3_MEMORY_MB=100000
export CPDC_OPH_SA3_MAX_JOBS=4

bash run_CPDC_ccpVTZ_rank2_opposed_history_v1.sh sa3
tail -f CPDC_ccpVTZ_rank2_opposed_history_v1_sa3.stdout
```

The final decision is written to:

```text
${CPDC_ARCHIVE_ROOT}/CPDC_ccpVTZ_rank2_opposed_history_v1_work/CPDC_ccpVTZ_rank2_opposed_history_v1_decision.json
```

## Stop branches

| status | action |
|---|---|
| `STOP_HISTORY_WAVEFUNCTION_FAILURE` | Stop; at least one sequential SA2 solution fails the frozen point checks. |
| `STOP_HISTORY_PROJECTOR_REPRODUCTION_FAILURE` | Stop; an SA2 history fails projector or energy reproduction. |
| `STOP_HISTORY_CLOSURE_FAILURE` | Stop; a returned `p000` solution does not close to the starting manifold. |
| `STOP_HISTORY_EDGE_INVARIANCE_FAILURE` | Stop; an edge, random-`U(2)`, or cyclic-start control fails. |
| `STOP_HISTORY_WILSON_REPRODUCTION_FAILURE` | Stop; the opposed-history Wilson spectra do not reproduce the reference. |
| `STOP_SA3_HISTORY_REPLICATION_FAILURE` | Stop; a history-seeded SA3 rank/reference/cross-history control fails. |

No stop branch falls through to the next stage.
