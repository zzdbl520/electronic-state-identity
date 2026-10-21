> **Revision deposit status — 21 October 2026:** The October revision data
> package has been prepared but is not yet published in this repository.
> The numerical files below remain the September release at commit
> `6e18f33d40d1e00166278bbb233f12ef8b56f239`. Do not cite the current
> public tree as containing the additional October revision data.

# Electronic-State Identity beyond Root Tracking

Numerical code and verification records for **“Electronic-State Identity beyond
Root Tracking: Gauge-Covariant Closed-Loop Transport of Multireference State
Manifolds.”**

The repository separates three tasks that are easy to conflate:

1. generate stationary multireference wave functions;
2. form exact many-electron cross-overlaps and their unitary polar transports;
3. decide whether a reported route effect survives fidelity, gauge, mesh, and
   opposed-history checks.

The reusable NumPy implementation is in `src/electronic_state_identity/`. The
original, system-specific PySCF workflows and the machine-readable decision
records are kept under `systems/`.

## Evidence map

| System | Question tested | Archived outcome | Location |
|---|---|---|---|
| LiH | Can endpoint root matching miss a difference between two continuous transports? | Yes. Both routes give the same composed assignment, while transport and back-transported dipole diagnostics differ; the result survives basis and opposed-history checks. | `systems/lih/` |
| BeH2 | Is the apparent branch termination intrinsic to unconstrained CASSCF? | No. It disappears when the active-orbital rotation restriction is lifted. The separate G0C plaquette summary remains `REVIEW_REQUIRED` because not every stride-1 upper edge passes the local-fidelity gate. | `systems/beh2/` |
| Cyclopentadienyl cation | Can a near-degenerate pair be validated as a rank-2 manifold without assigning its members? | Yes. Independent SA3, edge-fidelity, U(2)-gauge, mesh, and opposed-history tests pass for the externally isolated lower pair. | `systems/cyclopentadienyl-cation/` |

The BeH2 qualification above is intentional: the constraint comparison supports
the branch-termination interpretation, but the archived G0C plaquette package is
not promoted to an unconditional transport “pass.”

## Quick verification

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -v
python scripts/verify_reference_results.py
python scripts/run_workflow_selftests.py
sha256sum -c SHA256SUMS
```

These commands test the transport algebra and audit the published decision
records. They do not rerun the expensive CASSCF calculations.

## Full calculations

The production workflows were run with PySCF 2.13.1. Install the optional
dependency with:

```bash
python -m pip install -e '.[qc]'
```

Each system README gives its command sequence, inputs, stopping rules, and exact
claim boundary. Restart wave functions are deliberately not tracked by ordinary
Git; see `docs/DATA_AVAILABILITY.md` before attempting a full rerun.

## Repository conventions

- `workflow/`: calculation and analysis programs;
- `reference/`: immutable, machine-readable records from the reported runs;
- `src/`: small shared numerical kernels used by tests and examples;
- `scripts/verify_reference_results.py`: cross-system claim-to-record audit;
- `docs/CLAIM_EVIDENCE_MATRIX.md`: manuscript claim to numerical evidence;
- `docs/REPRODUCIBILITY.md`: verification levels and environment guidance;
- `docs/SCIENTIFIC_SCOPE.md`: what the numerical evidence does and does not show.

Release-facing reference records use sanitized path placeholders. One immutable
CP+ frozen-source snapshot retains historical path strings because its canonical
JSON fingerprint is an executable validation gate. These strings are provenance,
not secrets, and are not required for record-level verification.
