# LiH physical-field demonstration: S1 locator and S2 confirmation

## What this calculation tests

The calculation keeps the S0-qualified electronic model fixed and replaces the
artificial protocol-weight axis by a physical Hamiltonian coordinate: a static
electric field along the Li--H bond,

\[
H(F_z)=H(0)-\mu_z F_z.
\]

The electronic one-body term is `+Fz * r_z`; the nuclear field energy is also
included.  For neutral LiH the resulting total energy is independent of the
choice of coordinate origin.

Model:

- LiH, aug-cc-pVDZ, C2v, singlet A1;
- equal-weight SA4-CASSCF(2e,5o);
- active irreps A1^3 B1^1 B2^1;
- one optimized inactive doubly occupied core, no frozen orbital;
- four optimized and retained roots;
- deterministic dense A1 CASCI probe, filtered to the five lowest singlets;
- exact active-only CAS-wavefunction overlap.

Fixed 5 x 5 mesh:

- `R = 5.80, 5.85, 5.90, 5.95, 6.00 Angstrom`;
- `Fz = 0.01050, 0.01075, 0.01100, 0.01125, 0.01150 au`.

The field interval is approximately 0.540--0.591 V/Angstrom.  It is a strong
but physically defined perturbation, not an optimization weight or a frozen
orbital constraint.

## Files that must be in the same directory

- `LiH_S0_SA4_qualification.py` (v1.2 or later)
- `LiH_S1_physical_field_locator.py` (v1.7)
- `LiH_S2_physical_field_confirmation.py` (v1.0)

S1 imports the robust overlap and five-singlet probe routines from the updated
S0 file.  Replace an older S0 copy before running S1; rerunning the already
passed one-dimensional S0 grid is not required.

## Run S1

```bash
python LiH_S1_physical_field_locator.py selftest

python LiH_S1_physical_field_locator.py all \
  --root ${LIH_ARCHIVE_ROOT}/LiH_S1_field_locator_work \
  --threads 8
```

Do not change the default R grid, field grid, model, or thresholds for the
confirmatory run.  Expected terminal status:

```text
Status: CANDIDATE_WINDOW_FOUND
```

The immutable S1 decision is written to:

```text
${LIH_ARCHIVE_ROOT}/LiH_S1_field_locator_work/LiH_S1_decision.json
```

S1 evaluates the 16 elementary cells and all 84 cumulative rectangles whose
four boundary edges are measured directly from endpoint wavefunctions.  It
does not compose fine edges to manufacture a long-edge fidelity value.

## Run S2 on the same root

Only run S2 after S1 returns `CANDIDATE_WINDOW_FOUND`.

```bash
python LiH_S2_physical_field_confirmation.py selftest

python LiH_S2_physical_field_confirmation.py all \
  --root ${LIH_ARCHIVE_ROOT}/LiH_S1_field_locator_work \
  --threads 8
```

S2 reads the selected S1 window without relocating it.  It regenerates the
entire grid along two exactly opposed snake histories and requires every
history solution to agree with the independently started S1 solution.  Expected
terminal status:

```text
Status: PUBLICATION_DEMONSTRATION_READY
```

The final decision is written to:

```text
${LIH_ARCHIVE_ROOT}/LiH_S1_field_locator_work/LiH_S2_decision.json
```

## Hard stop rules

- Do not call the result publication-ready unless S2 prints
  `PUBLICATION_DEMONSTRATION_READY`.
- Do not relax a threshold after seeing the output.
- Do not move the S1-selected window during S2.
- Do not infer a conical intersection solely from the small roots-2/3 gap.
- Do not replace the active-only overlap by scalar division of an inactive-core
  overlap.
- Preserve all decision, diagnostics, manifest, log, and NPZ files.

If a run stops, report the complete decision JSON and the listed failed records;
do not rerun only favorable vertices.
