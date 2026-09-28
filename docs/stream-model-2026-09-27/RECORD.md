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

## S1. Presentations at the ends (`e158bf798`)

- **Values moved (G0.1).** `finn.dataflow.traversal` is the former
  `finn.kernels.physical.forms`, plus `Presentation` (traversal, repetition,
  markers) and the boundary rule `unreplayed`. `finn.dataflow.tensor` holds
  `Tensor` and `ScalarEncoding`, which moved out of
  `finn.kernels.datatypes.scalar` so a tensor is a logical value;
  `finn.dataflow.datatypes` keeps its no-FINN-imports rule. The untested
  Region/Network model moved to `finn.parked.dataflow.logical_values`; its one
  facade test was deleted and its datatype-ingestion tests were retargeted at
  `ScalarEncoding`, the live boundary.
- **A stream carries a tensor.** `Stream.spec` became `Stream.tensor`; every
  kernel reads only the tensor and exports its own presentation. A new
  constraint `well_formed` (code `stream-tensor`) checks that each end
  traverses the tensor in its element. The FIFO stage presents what arrives at
  it, repetition included.
- **The boundary rule (G0.2),** as found in S0: an input boundary presents its
  consumer's form without replay; an output boundary its producer's form;
  neither with markers. A consequence surfaced at once: a boundary can no
  longer feed dotp directly (dotp needs a frame marker), which S3's adapter
  resolves. The test composite that did so (`test_interfaces.Activated`) gained
  an explicit replay node, its XSim testbench now sends each row once without
  `TLAST`, and it passed.
- **Gate.** Fingerprints and decision keys identical over 14 configurations
  (`identity_dump.py`: dense, cyclic, FIFO, padded, pumped, the input-gen
  replay, memstream plain, pumped, writable and multi-set, per-channel native,
  cyclic and dense). Gates: Space 427, kernels 825, dataflow 11; ruff and mypy
  clean. XSim on the S1 snapshot: dense 40/40, FIFO 6/6 (packed) and 6/6
  (INT8 pumped), per-channel 34/34, input-gen replay 40/40, memstream 14/14 and
  per-channel memstream 12/12, 0 failures.

## S2. The nest (`158830d4d`)

- `finn.dataflow.nest`: `Level`, `Nest`, `Access` (with a row-major view of
  its tensor), `Iteration`, `present`, `frame`, `once`, `period`, and
  `Einsum`/`fold`/`accesses`. The S0 equalities are `tests/dataflow/test_nest.py`
  (16), each against an independent reference.
- **Markers (G0.4).** `Every` became `LevelEnd(beats)`: a marker closes the
  loop level spanning `beats` beats and is refused unless it closes whole
  innermost loops of its presentation (`LevelEnd.aligned`, checked by
  `Presentation` and `StreamContract`).
- **dotp** takes its `iteration` (nest and the accesses of X, W, Y) and
  derives every port through `dotp_presentations`, under dotp_axi's field
  conventions. One admission rule (`dotp-iteration`) replaces the labelled
  relation, `_LABELS`, `lane_reads`, `_operand`, `_frames`, and the walk helpers
  `axis_walk`, `walk_loops`, `split_walk` (deleted with their tests).
  `channel_tile` is gone: it is a derived presentation.
- **MatMul (G0.3 as recommended):** `Contraction` keeps its enum values and
  keys; each member is an einsum (`DENSE = "rk,nk->rn"`,
  `PER_CHANNEL = "rkc,ck->rc"`). PE folds the output index and SIMD the
  reduced one; every presentation derives from that one iteration. The dense
  realization reads the `(R, K, C)` activation tensor through an
  `(R, K·C)` view (the activation stream's tensor is now `(R, K, C)`; it was
  `(R, K·C)`).
- **Thresholding** derives its form from a nest (PE > C is still refused: the
  S0 gap).
- **A cyclic source aligns by elements.** A cyclic producer now repeats its
  pass as many times as the consumer's pass holds its elements, not its beats,
  so a cyclic source may meet a consumer of other lanes.
- **Deviation (G0.5).** The reduction order is declared at the value level:
  `fold(..., reduction_order=...)`, with `tests/dataflow/test_nest.py` showing
  every order of a three-index reduction legal and a `REORDER` apart.
  MatMul's contractions reduce one index, so a Space Decision would have
  exactly one candidate by construction and only add a key; the Decision is
  declared by the first family whose contraction reduces several indices
  (conv as matmul).
- **B1 probes remapped** (`test_port_contracts.py`): a frame crossing rows is
  an admission refusal; a wrong lane count and a transposed tile became stream
  verdicts (plans in S3); a producer's field order (E-048's window-fastest
  order included) is wires.
- **Gate.** Fingerprints and keys identical over the 14 configurations; the
  harnesses read only assembly outputs and instance names, which did not
  change, so the S1 XSim evidence covers the same RTL. Tests 843 passed.

## S3. Plans and adapters

- **`finn.dataflow.plan`.** `plan(source, sink)` returns canonical hops
  (`REORDER`, `WIDTH`, `MARKERS`), each between two presentations; `()` is
  direct, a field permutation included (wires). Lanes and order together
  decompose width-first or reorder-first; a lane regroup goes through the
  common lane count. A marker the sink requires is synthesized after the last
  sequence step. `Unrealizable` names what no chain repairs.
- **Stream.** `plan` (refused as `stream-plan`), `adaptable` (default True),
  a guarded `adapter` Decision over nodes and a guarded `adapter_ram_style`
  Decision (present only when the chain has an `input_gen`), stage lists
  (`stages`, the FIFO after the adapter on a `BufferedStream`), and a chain
  check source → stages → sink. `commit_adapters(point)` commits, per stream,
  the one compatible candidate. `netlist` wires stage lists as
  `u_<stream>_<stage>`.
- **Adapters (`finn.kernels.adapters`).** `realize(plan)` maps hops onto
  FinnLib modules: a reorder is an `input_gen` with `classify`'s parameters and
  the following marker step merged in (its nest split or grouped until the
  required level is some `olst[d]`; a level of one beat adds an innermost loop
  of one); marker synthesis alone is an identity `input_gen`; a width
  conversion a `vpc`. The seven candidates are the seven shapes a plan can
  take (`input_gen`, `vpc`, `vpc_input_gen`, `input_gen_vpc`,
  `input_gen_vpc_input_gen`, `vpc_input_gen_vpc`,
  `vpc_input_gen_vpc_input_gen`), so exactly one carries out any realizable
  plan. `inner_shuffle` stays a node a composite places
  (`finn.kernels.transpose`), not a candidate, until FinnLib's defect is fixed.
- **Replay leaves MatMul (C3 moves here, G0.6).** The `replayed` stream, the
  `replay` Decision, the `buffer`/`input_gen`/`markers` nodes and the replay
  derivations are gone; dotp reads `activations` directly, and that stream's
  `input_gen` adapter replays and frames. `ReplayBuffer` and the
  `replay_buffer_*` functions are deleted (the RTL parser tests still read
  FinnLib's `replay_buffer.sv` as a sample file). `InputGeneratorKernel` is a
  flat module sharing the stage's requirements builder.
- **Keys and names (D7).** Removed: `replay`, `replay.input_gen.ram_style`.
  Added, per stream: `<stream>.adapter`, `<stream>.adapter_ram_style` (each
  applies only under a plan). Instances: `u_replay_buffer`,
  `u_replay_input_gen`, `u_markers` → `u_activations_input_gen`, placed after
  the kernels. `matmul_assembly` lost its `replay` argument. Every MatMul
  fingerprint changes; the dense default is S2's input-gen replay exactly
  (same modules, parameters, wire count and ports; only instance names and
  order differ).
- **Measured (G0.6 asked for it).** `input_gen`'s buffer, elaborated in XSim
  (`BUF_SIZE`): marker synthesis alone, 4 words for frames of 1 to 64 beats
  (`replay_buffer`'s `REP=1` stored none); replay of SF beats,
  `2^⌈log2(2·SF+2)⌉` words: 8, 32 and 64 for SF = 2, 8, 16, against
  `replay_buffer`'s 2, 8 and 16. Replay through `input_gen` costs about four
  times the storage. The scrap stands as decided; this is flagged for review.
- **Evidence.**
  - `tests/dataflow/test_plan.py` (6): steps for known pairs, cyclic
    alignment, what is unrealizable.
  - `tests/kernels/test_stream_plans.py` (5): an independent model of
    `input_gen` and `vpc` from their RTL headers runs every realized chain on
    the source's positions and must reproduce the sink's positions (up to the
    one field permutation the connection wires) and required markers. 600
    random pairs over one to three axes, mismatched lanes, lane axes, beat
    orders, replays and markers (the mutation probes); every realized chain
    is one of the seven candidates.
  - `tests/kernels/test_adapters.py` (12): each candidate between a cyclic
    producer and thresholding, exactly one compatible, `adaptable=False`
    refused.
  - `tests/kernels/test_two_kernels.py` (4): dotp (PE 4) → dotp (SIMD 2, two
    output folds); the hidden stream plans width, replay and frame and places
    `vpc` + `input_gen`; XSim computes `(x·W1ᵀ)·W2ᵀ` free and stalled; a hidden
    stream admitting no adapter refuses the pair.
  - `tests/kernels/test_interfaces.py`: the dotp → thresholding module, now
    replayed by its stream's adapter, passes XSim.
