# BeH2 Gate B0: stationary-solution reconnaissance

This package identifies which same-symmetry electronic-state block carries the
abrupt BeH2 multireference stationary-solution change near
`lambda = 2.624 bohr`. It is the positive control paired with the H3 genuine-CI
negative control.

## Locked electronic-structure model

- Coordinate (bohr):
  - `Be = (0, 0, 0)`
  - `H1 = (lambda, +(2.540 - 0.46 lambda), 0)`
  - `H2 = (lambda, -(2.540 - 0.46 lambda), 0)`
- Point group: C2v
- Basis: cc-pVTZ
- Active space: CAS(6,6)
- Frozen core: Be 1s
- Protocols: equal-weight SA3 and SA4
- Symmetry blocks: A1 and B2, calculated separately
- B0 targets (bohr): 2.6235, 2.6240, 2.62425, 2.6245, 2.6250
- Initial orbital sources:
  1. continuation from the left anchor, 2.6160 bohr;
  2. continuation from the right anchor, 2.6270 bohr;
  3. common A1/B2 SA2 orbitals at the target geometry;
  4. fresh RHF orbitals.

CAS(6,12)/6-31G is intentionally not the default. In the supplied legacy
scripts it spans nearly all noncore orbitals and is therefore poorly suited to
a control whose purpose is to expose competing orbital-optimized stationary
solutions. The defaults can nevertheless be changed from the command line if
a strict legacy-level comparison is required.

## Run

Activate the PySCF environment used for the original calculations, unpack this
directory, enter it, and run:

```bash
bash run_BeH2_B0.sh
```

The default uses 64 threads and 100000 MB of memory. Override these without
editing the scripts:

```bash
BEH2_B0_THREADS=32 BEH2_B0_MEMORY_MB=80000 bash run_BeH2_B0.sh
```

The calculation is restartable. Existing solution NPZ files are reused, and a
failure is recorded separately rather than replacing a converged solution.

The full runner, checkpoint serialization, same-geometry many-electron overlap,
SA3--SA4 subspace selection, and analysis paths were smoke-tested with PySCF
2.14.0. See `SMOKE_TEST_REFERENCE.md` for the numerical regression values.

## Fast checks before submission to the queue

These tests do not require PySCF:

```bash
python BeH2_B0_stationary_solutions.py --self-test
python BeH2_B0_analyze.py --self-test
python -m py_compile BeH2_B0_stationary_solutions.py BeH2_B0_analyze.py
```

Then verify the production environment:

```bash
python -c "import pyscf; print(pyscf.__version__)"
```

## Main outputs

The runner writes `BeH2_B0_work/` containing:

- `solutions/`: complete MO, CI-vector, 1-RDM, energy, and convergence records;
- `logs/`: one PySCF log for every attempted stationary solution;
- `failures/`: machine-readable failure reports, if any;
- `B0_attempts.csv`: compact inventory of converged attempts;
- `B0_config.json`: the locked calculation specification.

The analyzer writes:

- `B0_solutions.csv`: energies, convergence residuals, and NOONs;
- `B0_pairwise.csv`: pairwise branch diagnostics for all initializations;
- `B0_clusters.csv`: provisional stationary-solution clusters;
- `B0_SA3_SA4.csv`: the best three-state SA4 subset matched to SA3;
- `B0_summary.json`: provisional switching-block assignment.

The assignment is based jointly on state-averaged energy, active-space
principal angles, genuine many-electron state-subspace overlaps,
Procrustes-aligned 1-RDM distances, and NOON distances. The automatic cluster
label is a screening result; the final A1/B2 decision must also inspect the
convergence logs and left/right reproducibility.

## What to return

After the run, send back:

- `BeH2_B0.stdout`
- `BeH2_B0_analysis.stdout`
- the complete `BeH2_B0_work/B0_*.csv` and `B0_*.json` files
- `BeH2_B0_work/failures/` if it exists and is nonempty

The large `solutions/` directory is needed only if we proceed to the full
mixed-plaquette analysis; it can be archived separately.
