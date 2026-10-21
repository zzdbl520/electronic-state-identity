# O3 fixed-root weighted-SA2 pilot

Patch level: v1.0.2. This patch composes cumulative Wilson lines from the audited local mesh edges instead of treating a long direct-overlap chord as one edge. It also reports every failed elementary plaquette explicitly. No electronic-structure calculation needs to be repeated.

This is the next authorized calculation after the BeH2 forensic closure. It asks a deliberately narrow question:

> Can ordinary pairwise maximum-overlap tracking give one unambiguous two-state assignment on every edge, while transport around a nuclear-coordinate/state-average-weight plaquette is measurably path dependent and changes a projected observable?

The package is a pilot, not a manuscript-claim generator. A positive result authorizes a refined O3 map; it does not by itself authorize the flagship molecular claim.

## Locked model

- Molecule/model: O3, forced Cs, two singlet A-double-prime roots.
- Electronic structure: SA-CASSCF(18e,12o)/cc-pVTZ.
- Nuclear center: r1+r2 = 2.7288 Å and theta = 119.70 degrees.
- Nuclear points: center, delta-r = +/-0.010 Å, and theta = 119.50/119.90 degrees.
- Protocol coordinate: excited-root weight `p` in `(1-p,p)`.
- Pilot weights: `p = 0.20, 0.35, 0.50`.
- The lower-energy root always has weight greater than or equal to the upper root. Weight-inverted ensembles are intentionally excluded.

The five equal-weight (`p=0.50`) points are rewrapped from the audited Gate4B SA2 checkpoints. Ten primary `p=0.35` and `p=0.20` wavefunctions are new. Five additional `p=0.20` history controls are started directly from `p=0.50` and compared against the staged `0.50 -> 0.35 -> 0.20` continuation. The total new CASSCF count is 15.

## Required input

An intact Gate4B v5 work directory containing:

```text
wavefunctions/center_SA2.npz
wavefunctions/dr_minus_h_0p010000_SA2.npz
wavefunctions/dr_plus_h_0p010000_SA2.npz
wavefunctions/theta_minus_h_0p200000_SA2.npz
wavefunctions/theta_plus_h_0p200000_SA2.npz
```

If starting from `O3_gate4B_v5.2.tar.gz`, extract it without restoring archived ownership:

```bash
tar --no-same-owner -xzf O3_gate4B_v5.2.tar.gz
```

The default server path is `/mnt/O3/O3_gate4B/O3_gate4B_work_v5`.

## One-command run

Activate the same PySCF environment used for Gate4B, then run:

```bash
chmod +x run_O3_weighted_SA2_pilot.sh
GATE4B_WORKDIR=/mnt/O3/O3_gate4B/O3_gate4B_work_v5 \
PILOT_WORKDIR=/mnt/O3/O3_weighted_SA2_pilot_work \
THREADS=32 MEMORY_MB=120000 \
./run_O3_weighted_SA2_pilot.sh
```

The driver first runs an algebraic gauge-invariance self-test and a dry-run inventory. It is resumable: validated checkpoints are skipped.

## Hard gates

Every elementary plaquette and the cumulative antisymmetric-stretch rectangle are checked against:

| Quantity | Gate |
|---|---:|
| minimum edge singular value, `eta_edge` | >= 0.98 |
| maximum subspace leakage | <= 0.01 |
| minimum normalized assignment margin | >= 0.10 |
| maximum final orbital gradient | <= 1e-4 |
| direct-vs-staged weighted-energy difference | <= 1e-8 Eh |
| direct-vs-staged state-subspace sigma-min | >= 0.9999 |
| direct-vs-staged state-subspace leakage | <= 1e-6 |
| cumulative antisymmetric-stretch `C_qp` | >= 1e-5 |
| `C_qp(delta-r) / C_qp(bend)` | >= 10 |
| normalized dipole path difference | >= 1e-5 |
| gauge-test absolute error | <= 1e-10 |

The conventional false-negative signature is specific: both composed maximum-overlap routes return the same permutation, yet the continuous polar transports disagree (`C_qp > 0`) and the endpoint electronic-dipole operator pulled back along the two routes is different.

## Decision statuses

- `EXPAND_TO_REFINED_WEIGHTED_SA2_MAP`: pilot passed; refine both the nuclear and weight meshes before writing the flagship claim.
- `HOLD_WEAK_OR_NONDIRECTIONAL_SIGNAL`: edge tracking is clean but the geometric signal is too small or not separated from the bend control.
- `HOLD_NO_OBSERVABLE_CONSEQUENCE`: nonzero holonomy did not reach the predeclared dipole consequence gate.
- `STOP_EDGE_FIDELITY_FAILURE`: the example is not a strict false negative because conventional tracking is already visibly unreliable.
- `STOP_STATIONARITY_FAILURE`: at least one wavefunction misses the stationary-point gate.
- `STOP_HISTORY_DEPENDENCE`: direct and staged weight continuations do not reach the same stationary solution.
- `HOLD_LOGIC_OR_GAUGE_FAILURE`: the composed assignment or gauge test invalidates the intended interpretation.

No status from this pilot authorizes the phrases “physical conical intersection proved,” “publication-ready molecular false negative,” or “state-label swap” without the stated refinement and controls.

## Outputs

The work directory contains:

- `O3_weighted_SA2_summary.json`: single stop/go decision and the flagship/control comparison.
- `O3_weighted_SA2_detailed.json`: every elementary and cumulative plaquette, including all four edge diagnostics.
- `O3_weighted_SA2_plaquettes.csv`: compact analysis table.
- `O3_weighted_SA2_edges.csv`: overlap singular values and assignment margins.
- `O3_weighted_SA2_history_controls.csv`: direct-versus-staged stationary-solution audit.
- `O3_weighted_SA2_pilot.png`: overview of curvature, leakage, and dipole path difference (if Matplotlib is available).
- `wavefunctions/*.npz`: orbitals, two CI vectors, state energies, natural occupations, and the two-state electronic-dipole matrix.

The last line printed by the analyzer is machine-readable:

```text
PILOT_DECISION=<status>
```
