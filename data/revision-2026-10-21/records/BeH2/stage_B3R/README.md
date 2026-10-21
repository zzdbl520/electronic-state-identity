# BeH2 Gate B3R: projection-safe spinodal scaling

## Why B3R is required

B2 located a history-selective SA3 stationary-solution collapse on the
2.5215--2.5220 bohr edge. B3 attempted to refine it, but the first B3 point was
2.52220 bohr whereas its B2 seed was stored at 2.52250 bohr. The initial MO
coefficients were passed directly without constructing the B2 seed molecule,
so PySCF could not perform the cross-geometry orbital projection on that first
step. The metastable upper branch was lost immediately: the first SA3
upper-down point required 155 macroiterations and converged to the smooth
solution. Consequently, the subsequent B3 plaquettes contained no upper-branch
collapse and are not a valid refinement test of the B2 event.

B3R changes only the continuation procedure. The definitions of the
many-electron overlap, polar transport, mixed nonclosure, conditioning, and
full-overlap path mismatch are unchanged.

## Projection-safe continuation

For every directional sequence, B3R copies the exact B2 seed archive into a
new work directory and begins the sequence at that same geometry. The first
call therefore reuses the archived solution while constructing its molecular
object. Every later geometry receives a projected orbital guess through
`mcscf.project_init_guess(..., prev_mol=seed_mol)`.

The four sequences are:

- SA3 `upper_down`: 2.52250 to 2.52120 bohr;
- SA3 `unique_up`: 2.52100 to 2.52220 bohr;
- SA4 `upper_down`: 2.52250 to 2.52120 bohr;
- SA4 `upper_up`: 2.52100 to 2.52220 bohr.

The analysis remains restricted to the original 21-point window,
2.52120--2.52220 bohr at 0.00005-bohr spacing. Anchor and bridge points outside
that window are excluded automatically. Direct plaquettes are evaluated at
strides 1, 2, 4, and 8.

## Run

Keep the complete original `BeH2_B2_work`, including its `solutions/`
directory. Use a new B3R work directory; do not reuse the previous B3 work
directory.

```bash
python BeH2_B3R_scaling.py --self-test
python BeH2_B3R_analyze.py --self-test
bash run_BeH2_B3R.sh /path/to/BeH2_B2_work BeH2_B3R_work
```

Resource overrides:

```bash
BEH2_B3R_THREADS=32 BEH2_B3R_MEMORY_MB=80000 \
  bash run_BeH2_B3R.sh /path/to/BeH2_B2_work BeH2_B3R_work
```

The calculation is restartable. Existing B3R solutions are reused.

## Return these files

- `BeH2_B3R.stdout`
- `BeH2_B3R_analysis.stdout`
- `BeH2_B3R_work/B3R_config.json`
- `BeH2_B3R_work/B3R_attempts.csv`
- `BeH2_B3R_work/B3R_branch_map.csv`
- `BeH2_B3R_work/B3R_multiscale_plaquettes.csv`
- `BeH2_B3R_work/B3R_scaling.csv`
- `BeH2_B3R_work/B3R_summary.json`
- `BeH2_B3R_work/failures/` only if nonempty

The large `solutions/` directory does not need to be uploaded unless a
matrix-level audit is requested.

## Acceptance logic

- The SA3 upper-down branch must remain distinct from unique-up at the upper
  end of the analysis window. If it does not, the continuation still failed.
- Its collapse must be co-located with the B2 bracket and with a sharp change
  in SA3 nuclear-edge leakage, active-orbital angle, density, and iteration
  count.
- SA4 and SA3 unique-up must remain smooth on the same grid.
- Step-size behavior is assessed for both the polar-unitary nonclosure
  `C_qp` and the complementary full-overlap mismatch `D_qp^(M)`.
- Gauge invariance, Wilson-loop equivalence, polar unitarity, reverse-overlap
  Hermiticity, and both normalized bounds are checked independently.
