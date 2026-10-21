# Gauge-Covariant Closed-Loop Transport of Multireference State Manifolds

Code and numerical data for the JCTC manuscript by Hui Li and Dongbo Zhao.

**Current data revision: 21 October 2026.** Start with the
[data README](data/revision-2026-10-21/README.md).

| System | Current evidence | Data |
|---|---|---|
| LiH | Same endpoint assignments with resolved transport and transported-dipole differences. Three bases are interpreted separately. | [Summary](data/revision-2026-10-21/analysis/augccpvtz_summary.json), [production](data/revision-2026-10-21/records/lih_augccpvtz_production/), [refinement](data/revision-2026-10-21/records/lih_augccpvtz_local_refinement/) |
| BeH2 | Energy separation depends on the orbital-rotation constraint. Failed SA3/SA4 fidelity prevents a physical branch-transition interpretation. | [Tables](data/revision-2026-10-21/records/BeH2/), [model comparison](data/revision-2026-10-21/records/beh2_model_comparison/) |
| C5H5+ | Externally isolated rank-2 pair; initialization sensitivity limits the precision of the small Wilson eigenphase. | [Records](data/revision-2026-10-21/records/CPDC/), [summary](data/revision-2026-10-21/analysis/cpdc_summary.json) |
| H3 | Historical ring/refinement tables and workflows; local pointwise-refinement failure retained. | [Data](data/revision-2026-10-21/records/H3/) |
| O3 | Historical directional-refinement data; weighted-SA2 pilot fails the stated gates. | [Data](data/revision-2026-10-21/records/O3/) |

All 1,936 LiH NPZ files from the revision package are included in checked archive parts, including
optimizer candidates and repeated baselines. This is a file count, not a count
of independent geometries. See [Data availability](docs/DATA_AVAILABILITY.md)
for historical raw-file limitations in the other systems.

## Quick verification

Python 3.10 or newer:

```bash
python -m pip install -e .
python scripts/unpack_wavefunctions.py
python -m unittest discover -v
python scripts/verify_reference_results.py
python scripts/run_workflow_selftests.py
sha256sum -c SHA256SUMS --quiet
```

These commands check transport algebra, current records and integrity without
new CASSCF optimizations. The data README gives overlap-reconstruction and
figure-regeneration commands.

- [Claim-to-evidence map](docs/CLAIM_EVIDENCE_MATRIX.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Scientific scope](docs/SCIENTIFIC_SCOPE.md)
- [SI/repository roles](docs/SI_REPOSITORY_BOUNDARY.md)
- [Revision notes](docs/REVISION_2026-10-21.md)

September files under `systems/` are retained for provenance. The old LiH
active-only overlaps and absolute-overlap assignments are superseded by the
October complete-overlap and squared-overlap analysis. Legacy status labels
are not current manuscript conclusions.
