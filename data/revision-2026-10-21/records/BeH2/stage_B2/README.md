# BeH2 Gate B2: fine SA3 fold and mixed plaquettes

## Why this gate exists

The B1 coarse map does not show a smooth energetic crossing. It brackets an
SA3 stationary-solution fold between 2.52 and 2.53 bohr:

- at 2.52 bohr, the two B1 histories are electronically the same;
- at 2.53 bohr, they differ by 5.16 mEh, 0.745 rad in maximum active-space
  principal angle, and 0.279 in state leakage;
- the SA4 upper solution remains continuous across the same interval.

The clean coarse-grid plaquette follows the upper solution. Its estimated
minimum edge singular value is about 0.899, whereas the lower route falls to
about 0.613. B2 therefore refines the upper route and retains both propagation
directions to expose hysteresis without conflating it with root relabeling.

## Calculation

The default grid is 2.5200--2.5300 bohr at 0.0005-bohr spacing (21 points).
Five directional sequences are computed:

- SA3 `upper_down`, seeded at 2.5300 and propagated downward;
- SA3 `lower_down`, seeded at 2.5300 and propagated downward;
- SA3 `unique_up`, seeded at the unique 2.5200 solution and propagated upward;
- SA4 `upper_down`, seeded at 2.5300 and propagated downward;
- SA4 `upper_up`, seeded at 2.5200 and propagated upward.

This produces 105 A1 SA-CASSCF solutions. The calculation is restartable.

The analyzer then evaluates every adjacent mixed plaquette for two historical
routes (`upper_down` and `unique_up`). The selected manifold has dimension
`m=3`; at each geometry the best three-state subset is extracted from SA4.
Production overlaps include cross-geometry AO overlap, orbital nonorthogonality,
CI contraction, and inactive-core embedding when needed.

## Run

Keep the complete original `BeH2_B1_work`, especially its `solutions/`
directory. Unpack B2, enter the directory, and run:

```bash
python BeH2_B2_refine.py --self-test
python BeH2_B2_plaquettes.py --self-test
bash run_BeH2_B2.sh /path/to/BeH2_B1_work BeH2_B2_work
```

Resource overrides:

```bash
BEH2_B2_THREADS=32 BEH2_B2_MEMORY_MB=80000 \
  bash run_BeH2_B2.sh /path/to/BeH2_B1_work BeH2_B2_work
```

## Return these files

- `BeH2_B2.stdout`
- `BeH2_B2_analysis.stdout`
- `BeH2_B2_work/B2_config.json`
- `BeH2_B2_work/B2_attempts.csv`
- `BeH2_B2_work/B2_branch_map.csv`
- `BeH2_B2_work/B2_plaquettes.csv`
- `BeH2_B2_work/B2_summary.json`
- `BeH2_B2_work/failures/` only if nonempty

The large B2 `solutions/` directory does not need to be uploaded unless a
post-run matrix-level audit is requested.

## Acceptance logic

- The SA3 branch-map rows determine the fold/hysteresis interval without using
  energy as the identity criterion.
- The primary positive-control result is the largest mixed curvature on the
  upper route, reported together with `eta_edge` and all four edge singular
  values.
- A peak with `eta_edge >= 0.85` is retained separately so a large curvature
  cannot be presented without its conditioning.
- Curvature formula agreement, random corner-gauge invariance, polar-unitarity,
  and reverse-overlap Hermiticity are checked for every plaquette.
