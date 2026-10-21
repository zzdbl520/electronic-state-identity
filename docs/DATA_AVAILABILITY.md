> **Revision deposit status — 21 October 2026:** The October revision data
> package has been prepared but is not yet published in this repository.
> The numerical files below remain the September release at commit
> `6e18f33d40d1e00166278bbb233f12ef8b56f239`. Do not cite the current
> public tree as containing the additional October revision data.

# Data availability

The repository contains all currently assembled calculation programs, protocols,
decision records, diagnostics, build validations, and small stdout records used
for numerical verification.

Large per-point wave-function restart files (`solution.npz`) are not committed to
ordinary Git. The decision records retain their SHA-256 digests and use sanitized
path placeholders, so the released archive can be checked without exposing a
machine-specific directory layout.

For full third-party regeneration, choose one of the following before the public
release:

1. deposit the restart archive in a DOI-bearing data repository and record its
   DOI plus archive checksum here; or
2. publish it with Git LFS and document the expected download size.

Until that archive is linked, the repository supports source-level inspection,
transport-algebra tests, record-level verification, and fresh calculations from
the supplied workflows, but not byte-for-byte replay from the original restarts.

The final data DOI or Git LFS location, manuscript author list, and preferred
citation should be added before the archival release.
