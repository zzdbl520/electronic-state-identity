# BeH2 final clean rerun: CAS(6,6)/cc-pVTZ

> **Superseded interpretation.** This archived workflow imposed an active-orbital
> rotation restriction. The later three-model forensic comparison in
> `../forensic/` shows that the apparent termination is constraint-induced and
> is not an intrinsic termination of unconstrained CASSCF. This directory is
> retained only for provenance and code-level reproduction of the restricted
> calculation.

This is the submission-production rerun for the BeH2 positive control. It starts
from no electronic-structure checkpoints and runs the full B0 -> B1 -> B2 ->
B3R sequence before regenerating main-text Figure 5 and SI Figure S2.

## Locked scientific model

- equal-weight SA3- and SA4-CASSCF;
- CAS(6,6)/cc-pVTZ;
- frozen Be 1s core;
- C2v geometries and the A1 stationary-solution block;
- the complete SA3 three-state manifold and the optimally matched three-state
  subset of SA4, so the retained dimension is m=3;
- final B3R interval 2.52120--2.52220 Bohr, spacing 0.00005 Bohr, and direct
  plaquette strides 1, 2, 4, and 8.

The wrapper passes the model explicitly, checks it again in every stage's JSON
configuration, and refuses to mix a new run with an existing nonempty work
directory.

## One-command run

Activate the production PySCF environment, unpack the archive, and run:

```bash
cd BeH2_ccpVTZ_final_rerun
BEH2_FINAL_THREADS=64 BEH2_FINAL_MEMORY_MB=100000 \
  bash run_BeH2_ccpVTZ_final.sh ${BEH2_WORK_ROOT}/BeH2_ccpVTZ_final_work
```

For Slurm, edit the resource directives if necessary and submit:

```bash
sbatch submit_BeH2_ccpVTZ_final.slurm ${BEH2_WORK_ROOT}/BeH2_ccpVTZ_final_work
```

The activated Python environment must provide PySCF, NumPy, SciPy, pandas,
Matplotlib, and Pillow. The preflight verifies the required PySCF overlap and
orbital-projection APIs before any expensive calculation begins.

## Restart after an interruption

All stages are checkpointed. The default is deliberately strict and refuses a
nonempty destination. To resume the same interrupted run, use the same path and
set:

```bash
BEH2_FINAL_RESUME=1 BEH2_FINAL_THREADS=64 BEH2_FINAL_MEMORY_MB=100000 \
  bash run_BeH2_ccpVTZ_final.sh ${BEH2_WORK_ROOT}/BeH2_ccpVTZ_final_work
```

Do not use resume mode with checkpoints from another basis, active space, or
work directory.

## Automatic acceptance

The job is successful only if all of the following pass:

- the upper-down SA3 history remains a distinct stationary solution;
- 84 final-grid solutions and 138 multiscale plaquettes are present;
- the upper-down peak is refinement resistant, while unique-up decreases
  approximately linearly;
- the peak remains well conditioned (eta_edge >= 0.85);
- collapse and smooth histories separate in nuclear-edge leakage;
- gauge invariance, polar unitarity, route-formula equivalence, and reverse
  many-electron overlaps agree to 1e-9 or better;
- PNG and TIFF figures are RGB and 600 dpi.

After a pass, the work root contains `ACCEPTANCE.md`,
`acceptance_report.json`, the 600-dpi figures, and the compact file
`BeH2_ccpVTZ_final_return.zip`. Send back that ZIP; the much larger `solutions/`
trees can remain on the cluster unless a matrix-level audit is needed.
