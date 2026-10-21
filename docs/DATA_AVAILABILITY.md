# Data availability

Current numerical data are committed at
[data/revision-2026-10-21/](../data/revision-2026-10-21/).
Clone or download this repository, then run `python scripts/unpack_wavefunctions.py`.
The checked archive parts are included; no external restart download is needed.

| System | Available here | Remaining limitation |
|---|---|---|
| LiH | All 1,936 NPZ files in the revision package, logs, plans, sources, candidate/history records and validators | File count includes candidates/duplicates; three bases are interpreted separately. |
| H3 | Ring/refinement CSV/JSON, workflows and numerical records recovered from returned archives | Complete original orbital/CI checkpoints unavailable; local pointwise-refinement failure retained. |
| O3 | gate4A/gate4B tables, manifests, logs, scripts, numerical gate4B numerical archive and weighted-SA2 pilot record | Historical wave-function coverage incomplete; weighted-SA2 pilot fails stated gates. |
| BeH2 | Historical tables, constraint-comparison summary and workflows | Cross-model restarts and precise 12-start records unavailable; energy comparison supported at summary level. |
| C5H5+ | Point/edge, SA3, mesh/history and Wilson records, sources and logs | Original geometries, relaxed-surface coefficients and orbital/CI arrays unavailable; independent overlap/Wilson reconstruction cannot be performed. |

The root `SHA256SUMS` covers tracked files except itself. The data directory has
its own checksum list and `SOURCE_PROVENANCE.json`. Original source/plan hashes
and provenance path strings inside raw records are preserved.

The Git commit identifies the deposited version. The September commit does not
contain these additions. This update does not submit the manuscript to a journal.
