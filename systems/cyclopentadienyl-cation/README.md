# Cyclopentadienyl cation: isolated rank-2 manifold

This workflow treats the lower two singlets as a retained rank-2 manifold and
does not assign either individual root. Independent SA3 calculations test
external isolation from the third singlet. Exact many-electron overlaps then test
edge fidelity, Wilson transport, random U(2) gauge invariance, mesh convergence,
and opposed-history reproducibility.

## Contents

- `workflow/`: central screening, relaxed-loop, SA3 projector, and opposed-history
  programs plus their protocols;
- `reference/`: final decisions, authorization, build validation, and stdout
  records;
- `CPDC_rank2_final_wrapup_20260920.md`: the frozen evidence summary.

The standalone cc-pVTZ lift protocol declaration was absent from the assembled
bundle. The copy in `workflow/` is reconstructed directly from the executable's
immutable constants and is tagged as such inside the JSON. The later projector
and opposed-history decision records are original archived outputs.

Fresh production runs also require the source, relaxed-loop, screening, and
projector work roots named by the wrapper. Their large `solution.npz` trees are
part of the external restart archive, not ordinary Git.

The final expected status is `PASS_RANK2_OPPOSED_HISTORY_REPLICATION`.

## Claim boundary

The pass supports history-independent identity and transport of an externally
isolated lower two-state manifold. It does not assign a scalar Berry phase to an
individual state, resolve the two members separately, or count conical
intersections inside the loop.
