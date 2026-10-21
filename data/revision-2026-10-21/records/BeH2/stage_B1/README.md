# BeH2 Gate B1: support map of the two A1 stationary solutions

Gate B0 identifies A1, not B2, as the block containing two electronically
distinct stationary solutions. It does **not** yet locate a switch near
`lambda = 2.624 bohr`: throughout the B0 interval, the B0 common-SA2 solution
remains on one A1 branch, whereas left continuation, right continuation, and
fresh RHF all return to the other branch.

B1 therefore answers the missing question before any mixed plaquette is
placed. Starting from the two converged B0 A1 solutions at `lambda = 2.62425`
bohr, it propagates both branches in both directions over `2.40--2.70 bohr` at
`0.01 bohr` resolution for equal-weight SA3 and SA4. The calculation contains
only A1 and requires 128 solution files (32 geometries x 2 branches x 2
protocols). No B2 production scan is repeated.

## Required input from B0

Keep the original `BeH2_B0_work` directory, including `solutions/` and
`B0_config.json`. The four center seeds used by B1 are:

- `A1/SA3/common_sa2/lambda_2p62425000.npz` (upper branch)
- `A1/SA3/right/lambda_2p62425000.npz` (lower branch)
- `A1/SA4/common_sa2/lambda_2p62425000.npz` (upper branch)
- `A1/SA4/right/lambda_2p62425000.npz` (lower branch)

## Run

Unpack this package beside the B0 work directory and enter the unpacked
directory. The first argument is the path to `BeH2_B0_work`; the optional
second argument is the new B1 work directory.

```bash
python BeH2_B1_branch_support.py --self-test
python BeH2_B1_analyze.py --self-test
bash run_BeH2_B1.sh ../BeH2_B0_work BeH2_B1_work
```

The calculation is restartable. To change resources without editing files:

```bash
BEH2_B1_THREADS=32 BEH2_B1_MEMORY_MB=80000 \
  bash run_BeH2_B1.sh ../BeH2_B0_work BeH2_B1_work
```

## Return these small files

- `BeH2_B1.stdout`
- `BeH2_B1_analysis.stdout`
- `BeH2_B1_work/B1_config.json`
- `BeH2_B1_work/B1_attempts.csv`
- `BeH2_B1_work/B1_branch_separation.csv`
- `BeH2_B1_work/B1_continuity.csv`
- `BeH2_B1_work/B1_SA3_SA4.csv`
- `BeH2_B1_work/B1_summary.json`
- `BeH2_B1_work/failures/` only if nonempty

Do not upload the large `solutions/` directory yet. The returned summary will
identify the interval to refine and the four corners for Gate B2.

## Decision logic

- A sign change in `delta_e_upper_minus_lower` brackets an energetic crossing
  of two still-distinct stationary solutions.
- A change in `same_stationary_solution` brackets a branch merger or spinodal.
- `B1_continuity.csv` verifies that every propagated branch remains a
  well-conditioned electronic-state manifold between neighboring geometries.
- `B1_SA3_SA4.csv` reports the best three-state SA4 subset matched to SA3. The
  final plaquette will only be placed after both nuclear and protocol edges are
  inspected.
