# LiH S0: run instructions and immutable decision contract

## What this run is allowed to answer

This is a **one-dimensional qualification gate**, not the flagship calculation.
It asks whether an equal-weight four-root singlet-A1 LiH model is stationary,
history-independent, isolated from the first excluded A1 root, and locally
continuous under the exact active-space wavefunction metric. Only a complete
pass authorizes a later, separately pre-registered `R x electric field` pilot.

The molecular model is fixed before seeing the result:

- LiH, `aug-cc-pVDZ`, C2v subgroup;
- equal-weight SA4-CASSCF(2e,5o), all four A1 roots retained;
- one optimized inactive doubly occupied Li(1s)-like core; no frozen orbital;
- active-orbital symmetry counts A1^3 B1^1 B2^1;
- five-root A1 CASCI probe on every optimized orbital set;
- direct starts and two opposed continuation histories;
- exact **active-only** CAS-wavefunction overlap for state fidelity.

The production mesh is 0.05 Å. A development-only 0.10 Å qualification run
missed the unchanged leakage gate on two short-bond edges, so the coarse mesh
was rejected rather than the threshold relaxed. Any stalled primary CASSCF
optimization is restarted deterministically with PySCF's symmetry-adapted
second-order optimizer from its own final orbitals; this is not a new history
or a best-of-several-solutions selection.

The four-root choice is deliberate. Published LiH model-space work reports
three low A1 ionic/covalent states but also an avoided crossing with a higher
state near 2.6 Å when that state is omitted. SA4 is the protective closure
test; the fifth root is monitored rather than silently ignored.

## 1. Copy the script to the calculation machine

Place `LiH_S0_SA4_qualification.py` in a clean LiH working directory. Do not
edit the thresholds, grid, roots, symmetry, or active-space definition after
seeing partial output.

## 2. Activate the same validated PySCF environment

```bash
conda activate osmo4-mokit
python - <<'PY'
import numpy, pyscf
print("NumPy", numpy.__version__)
print("PySCF", pyscf.__version__)
PY
```

## 3. Run the local self-test

```bash
python LiH_S0_SA4_qualification.py selftest
```

Required output:

```text
SELFTEST=PASS
```

## 4. Optional three-job API smoke test

This checks PySCF symmetry labels, four-root SA-CASSCF, and the fifth-root
probe before committing to the full one-dimensional scan.

```bash
python LiH_S0_SA4_qualification.py run \
  --root ${LIH_WORK_ROOT}/LiH_S0_smoke \
  --r-grid 2.8 \
  --short-anchor 2.8 \
  --long-anchor 2.8 \
  --history-step 0.05 \
  --threads 8
```

Inspect the three `DONE` lines. Do not analyze the one-point smoke run; the
analyzer intentionally requires at least two nuclear geometries.

## 5. Run the pre-registered S0 qualification

```bash
python LiH_S0_SA4_qualification.py all \
  --root ${LIH_WORK_ROOT}/LiH_S0_SA4_work \
  --threads 8 \
  2>&1 | tee ${LIH_WORK_ROOT}/LiH_S0_SA4_work/LiH_S0_console.log
```

Default workload:

- 39 independent direct points, R = 2.3...4.2 Å by 0.05 Å;
- 55 short-bond-outward history points, 1.5...4.2 Å;
- 75 long-bond-inward history points, 6.0...2.3 Å;
- 169 SA4-CASSCF jobs total, each followed by a five-root CASCI probe.

Existing `.npz` results are resumed automatically. Use `--overwrite` only to
rerun the complete declared target intentionally, never to replace selected
unfavorable points.

If wavefunctions finish but analysis is interrupted:

```bash
python LiH_S0_SA4_qualification.py analyze \
  --root ${LIH_WORK_ROOT}/LiH_S0_SA4_work \
  --threads 8
```

## 6. Return these files and console lines

Return:

- `LiH_S0_decision.json`
- `LiH_S0_diagnostics.json`
- `LiH_S0_manifest.json`
- the final `LiH S0 RESULT` console block
- any `*.failed.json` file if execution stops

## Immutable decision hierarchy

The analyzer stops at the first failed gate:

1. numerical overlap/CASCI consistency;
2. exact model semantics;
3. wavefunction stationarity and singlet purity;
4. direct versus two opposed histories;
5. gap of at least 0.01 Eh to the first excluded A1 root;
6. active-orbital-subspace continuity;
7. strict four-state active-wavefunction edge fidelity;
8. unambiguous state assignment.

Only `PROMOTE_TO_PHYSICAL_FIELD_PILOT` permits the next calculation. Even that
status does **not** authorize a manuscript claim; it only says the physical
`R x F` experiment is worth designing.

## Why the overlap definition changed after LiF

For a CAS wavefunction with a shared inactive determinant, translating the
molecule multiplies the raw all-electron overlap by a spectator-core factor.
LiF showed that this can make nuclear edges look catastrophically nonlocal
while protocol edges remain excellent. Dividing the raw overlap by that scalar
is not acceptable because it can yield singular values greater than one.

S0 instead omits the common inactive determinant and evaluates the exact
overlap of the active CAS expansions using the cross-geometry active-orbital
overlap matrix. This remains contractive and retains genuine changes of the
embedded active orbital space. It is therefore the physically usable identity
metric to test in LiH.

## Method references

- LiH model-space and active-space precedent: *J. Chem. Phys.* **142**, 064106
  (2015), https://doi.org/10.1063/1.4907038
- PySCF state averaging and symmetry-resolved FCI examples:
  https://github.com/pyscf/pyscf/blob/master/examples/mcscf/15-state_average.py
  and https://github.com/pyscf/pyscf/blob/master/examples/1-advanced/030-transition_dipole.py
