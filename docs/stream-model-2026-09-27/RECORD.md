# Record: the stream model (D10), increment 1

Plan: [PLAN.md](PLAN.md). Design: [DESIGN.md](DESIGN.md). G0 answered 2026-09-28
(PLAN §G0); decision 3 (contraction syntax) proceeds on the recommendation, an
enum whose members are defined by einsum strings, pending the user's
confirmation.

## S0. Expressiveness spike

Sketch: [`sketches/s0_roster.py`](sketches/s0_roster.py), on top of
`indexed_ports.py`. Every check compares a derived form with an independent
reference (a form already pinned by tests, or an enumeration of the
operation's index formula).

| Roster member | Result |
|---|---|
| dense, per-channel MatMul | derive (indexed_ports) |
| dense realization of per-channel | derives as a reshaped view: the dense form over `(R, K·C)` re-read over `(R, K, C)` (same flat offsets) |
| tiled MVU (TH > 1) | derives: core activation, core result and weight chunks; both FINN `input_gen` parameter sets fall out of `classify`. The weight chunks are a second split of the lane level (`p = pt·PE/T + pc`), a width conversion of the tile. The tiled core's reduction is *not* an innermost suffix (`t` sits inside `kf`), so it needs its own admission rule, not dotp_axi's |
| SWG / conv-as-matmul, stride and dilation | positions derive (several levels step one axis). **Gap:** `classify` names no adapter for overlapping windows; `input_gen` realizes them with whole-frame coefficients. Not needed until an SWG kernel sits on streams |
| thresholding PE ≤ C | derives `vector_major` |
| thresholding PE > C | derives, as a split of the row index into a beat and a lane level. **Gap:** needs `R` divisible by `PE/C`; otherwise logical padding (S5). Today's port refuses PE > C outright |
| eltwise, broadcast operand | derives: a channel vector repeated per row, and a per-row scalar replayed per channel fold |
| transpose | derives; `classify` names it `LANE_REGROUP` |

Findings that shape S1–S3:

- **Boundary rule (G0.2).** "The receiver adapts" becomes: a boundary input
  presents its internal end's form without *replay* (stride-0 beat loops
  inside a moving loop), keeping whole-pass *repetition* (outermost stride-0
  loops). Activations: `(r, nf↦0, kf)` → `(r, kf)`, replay inside the module.
  External weights: `tile.repeated(R)` unchanged, FINN's `in1_V`. Repetition
  needs a buffer of the whole pass, which no adapter provides (the plan
  refuses `ONCE` feeding `CYCLIC`), so it stays part of the interface.
  Markers are not presented at a boundary (no `TLAST`), as today.
- **Markers as levels (G0.4).** Canonical loops merge contiguous levels, and
  nest names do not cross kernels, so a level marker is named by the number of
  beats it spans and must fall on a loop boundary of its presentation. It is
  `Every` plus that alignment rule. `input_gen`'s `olst[d]` offers exactly the
  levels of its reorder nest.
- **Compound plans.** Width and reorder decompose in either order (width
  first, or reorder first). A lane regroup always has a fallback chain through
  the common lane count, where every permutation is a reorder (slow, but
  correct); `inner_shuffle` is the fast realization of one shape of it.
