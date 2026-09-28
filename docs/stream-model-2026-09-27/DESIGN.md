# D10: the stream model for kernel design spaces

Date: 2026-09-27. Status: **design round, for review**. No production code is
changed. Branch `feature/kernel-package-extraction` at `41c30ca0d`.
Companions: [SOURCES.md](SOURCES.md) (what was read, with a verdict on each)
and two executable sketches:

- [`sketches/indexed_ports.py`](sketches/indexed_ports.py) derives the port
  forms from a contraction.
- [`sketches/engine_probe.py`](sketches/engine_probe.py) checks the engine
  assumptions.

Both run with `PYTHONPATH=src` in the kernel venv and pass. They import only
`finn.kernels.physical.forms` and `finn.core.space`.

**Citations** are to HEAD `41c30ca0d`. During this round another session was
landing MatMul's M1 in the same worktree: `mvau.py` becomes `matmul.py`, and
the delivery Decision `implementation` becomes `delivery`. Read `mvau.py:N`
as the same code in `matmul.py` after that rename.

**Addendum (2026-09-28, after this round).** Phase C landed on top of the
cited HEAD without implementing this design: MatMul M2–M5, C3–C5, and C6 in
part (`../matmul-kernel-2026-09-27/RECORD.md`). Two points bear on the
decisions below. The per-channel port relation was built as M0 planned, as one
labelled relation over per-axis walks for both contractions, so decision 1 now
reads "S2 retires that relation" rather than "S2 avoids writing it". C6 built
the `vpc` and `inner_shuffle` adapters as ordinary nodes between two streams
and held the in-stream adapter Decision for S1, as §0 recommends.

## 0. Summary

**The finding that shapes everything else.** B1's dotp checks are of two kinds.

- **Single-stream facts:** this stream has SIMD lanes of consecutive columns, one
  frame marker, PE result lanes.
- **Cross-stream relations:** the weight columns follow the activation columns
  beat by beat; a frame stays inside one activation row and one group of weight
  rows; each result beat holds its frame's row and weight rows.

A stream that knows its whole tensor can own the first kind. It can never own
the second: a relation between three tensors is not a property of any one of
them. What relates them is the kernel's *iteration space*: one loop nest,
with each port reading its tensor through an affine index. With that nest,
both kinds of check stop being checks. The ports' forms are *derived* from
it, and they agree by construction. The sketch proves this for MVAU: one
contraction, one folding and one loop order reproduce all four stream forms
that `mvau.py:216-246` writes by hand. That includes the replay loop, the
weight repetition and the frame period. The per-channel `s·PE + p` relation
falls out of the same derivation with a different contraction.

**The recommended shape.** Four objects, each owning one thing.

| Object | Kind | Owns |
|---|---|---|
| `Tensor` | value (new; canon `Operand`) | shape and element encoding: the fact a stream carries |
| `Traversal` | value (kept, `forms.py`) | one presentation: beat order and field order over the tensor, stride 0 for replay |
| `Nest` + `Access` | values (new; the parked index notation, without its stack) | a kernel's iteration space and each port's affine index into its tensor; the folding is which levels are lanes |
| `Stream` | Space node (revised) | the edge: the tensor as a fact, both ends' presentations (read from `Users(PORT)`), the *plan* that relates them, an `adapter` Decision over nodes that carries it out, and the `transport` (direct/FIFO) Decision it has today |

The form leaves the stream. Today a composite writes one `StreamSpec` per
stream and both ends adopt it (`streams.py:21-23`, `dotp.py:311-325`), so the
stream's compatibility check mostly compares a spec with itself. In the new
model each end *presents* its own traversal, derived from its own nest. The
stream compares the two, and that comparison is what makes adapters possible.

"Which foldings are valid" becomes a conjunction, evaluated where each part
lives:

- the tensor's extents give the folding domains (divisors);
- the owning kernel's admission rule accepts or refuses the nest;
- each stream's plan must be realizable by an admitted adapter candidate.

A refusal stays on the stream or kernel it belongs to, as B1's attribution
already requires.

**The top three decisions for the user** (§11 lists them all):

1. **Adopt the nest (S2) as the foundation of MatMul's per-channel increment**
   (M3 in [M0.md](../matmul-kernel-2026-09-27/M0.md)'s order), so Q7's port
   relation is never written as a stopgap. M0's labelled relation already
   names the indices; S2 turns the check into a derivation. MatMul's Q1 enum
   becomes an einsum-like `Contraction`.
2. **Where the values live.** Move `Traversal`/`classify` and the new
   `Nest`/`Access`/`Tensor` into `finn.dataflow`, as its canonical logical
   values. Leave the untested Region/Network code to the `finn.dataflow`
   model pass, which retires or re-grounds it.
3. **Who decides a module's boundary presentation.** Recommended: an explicit
   composite Param per boundary stream, defaulting to the internal end's
   presentation. Where reuse is realized becomes visible: replay inside
   MatMul, or the boundary presenting the replayed sequence.

**C6 before D10?** C6's form machinery is sound today and D10 keeps it:
`classify` and `Reorder` are already checked against FINN's `input_gen` and
outer-shuffle coefficients (`test_stream_contract.py:174-222`). What C6
cannot sensibly build on is the *ownership* of forms. A stream with an adapter
has two different forms, one per end, and today's stream has exactly one. So
C6 needs increment **S1** (presentation at the ends). S1 is small, and
expected to leave fingerprints unchanged: the forms stay the same and only
their owner moves. C6 does not need S2 (the nest).

**Engine.** No change is needed (§8.4). The probe shows three things:

- ends presenting through per-input exports while reading only the stream's
  tensor;
- the stream deriving a plan from `Users`, and adapter candidates that bind
  to it and refuse themselves;
- a folding domain that depends on a tensor extent read through a reference
  input.

## 1. The problem, concretely

What is fragile today, with the code that shows it:

| # | Problem | Where |
|---|---|---|
| P1 | The composite writes every stream form by hand from its folding: `vector_major`, `replayed`, `tile(...).repeated`, `vector_major`. A composite can write a wrong form, and only a consumer's hand check catches it | `mvau.py:216-246` |
| P2 | Each consumer re-derives what it reads and checks the form it is handed, in bespoke terms. dotp: `_fields`, `_frames`, row/column walks. Thresholding: equality with `vector_major`. Set stream: beat-count equality. Each new contraction adds checks, and per-channel adds Q7's | `dotp.py:327-435`, `thresholding.py:374-399`, SPEC Q7 |
| P3 | Both ends adopt one spec, so a stream cannot hold two different forms. An adapter cannot sit in a stream except as an identity FIFO | `streams.py:21-23`, `dotp.py:311-325`, `thresholding.py:359-372`, `StreamFifo` `streams.py:169-191` |
| P4 | The same fact is written twice. The replayed form is derived by the composite (`mvau.py:223-227`) and by `ReplayBuffer` (`streaming.py:92-115`), then compared. The replay count is wired by hand (`replay_count=neuron_folds`) | `mvau.py:255-260` |
| P5 | Broadcast, replay presence and frame period are properties of the operation, but they are pinned or hand-wired: `ACTIVATION_BROADCASTING=1`, a `replayed` stream, `Every(synapse_folds)` | `dotp.py:242`, `mvau.py:227,251` |

History says these break for real. In post-S4 F1, selecting PE silently
changed the replay count: REP was wired as `MH//pe` while the source requested
fixed `neuron_folds`. In F3, enum-labelled mappings carried no position check.
E-048 is a lane order agreed only by convention between SWG and VVAU.
All three are symptoms of forms written in more than one place.

## 2. First principles

A stream carries **one tensor** from **one producer** to **one consumer**, in
passes. Everything a stream might "know" falls into one of four classes, each
with a natural owner.

1. **What is carried**: the tensor's shape and element encoding. A fact of the
   operation. It exists before any folding and must not depend on either end,
   so it is safe for both ends to read. This is the anti-cycle rule
   `streams.py:21-23` states, applied to the right object.
2. **How each end presents it**: beat order (canon `π`), field order within a
   beat (canon `h`), replayed positions, markers, and pass repetition. This is
   a property of an *end*, derived from that end's iteration space and its
   hardware's field conventions. Two ends may present the same tensor
   differently. That difference is exactly what an adapter exists to bridge.
3. **How the presentations relate**: identical, a free lane permutation, or
   an adapter chain (width conversion, lane regroup, reorder or replay, marker
   synthesis). Or no relation, which is a refusal. This is a property of the
   *edge*, so it belongs to the stream.
4. **How a kernel's ports relate to each other**: the contraction or
   elementwise structure. This belongs to the kernel's *iteration space*, not
   to any stream. It is the part that B1 hand-checks and Q7 would hand-check
   again.

The canon draws the same lines, and the design adopts them:

- REGION §5: a port is an operand plus a beat sequence.
- NETWORK §2/§4: an edge compares complete ordered sequences, with no implicit
  conversion. An adapter is an ordinary unit.
- REGION §1 and README: a Region owns one schedule.
- REGION-PROFILES §3.1 and canon-v1 §7.1: a fold spatializes a *schedule
  level*, never a port, and each port's response follows from its stride at
  that level.

What the canon never had is a compact, comparable encoding with an adapter
classifier. The live `Traversal` and `classify` are that encoding.

## 3. The objects

### 3.1 `Tensor` (value; new)

```python
@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]      # positive extents, row-major positions (canon position_rank)
    element: ScalarEncoding     # as StreamSpec.element today
```

The canon's operand identity is the stream's node identity. Axis names are
optional diagnostics. Padding (§4.7) would extend this value, not the stream.

### 3.2 `Traversal` (value; kept unchanged)

`forms.Traversal` (`forms.py:82-159`) is already the canonical stream form of
REGION-PROFILES §5.1:

- `beat_loops` carry `π`;
- `lane_loops` carry `h`;
- stride 0 carries replay and repetition, which §5.1 excludes and needs a named
  form for.

Construction canonicalizes it, so equality is equality of presented sequences
(`forms.py:13-15`, tested at `test_stream_contract.py:126`). It stays the one
presentation value. So do `pack`, `Every` and `Repetition`, and `classify`,
which grows into `plan` in §5.

### 3.3 `Nest` and `Access` (values; new)

```python
Level(name, extent)                       # one loop
Nest(beats: tuple[Level, ...],            # temporal levels, outer to inner
     lanes: tuple[Level, ...])            # spatial levels: a set; order is each port's
Access(tensor: Tensor,
       index: tuple[Mapping[str, int], ...])   # per tensor axis: level -> coefficient

present(nest, access, *, fields, reduced=()) -> Traversal
frame(nest, reduced) -> Every             # the reduction closes when the reduced suffix wraps
once(form) -> Traversal                   # the form with its stride-0 beat loops removed
period(form) -> Traversal                 # the form with its *outer* stride-0 loops removed
```

`present` is the single derivation rule (`indexed_ports.py:102-125`):

- A beat level becomes a beat loop whose stride is its coefficient summed over
  the tensor's row-major axis strides.
- `fields` orders the lane levels the port carries. That order is the
  hardware's field convention, the canon's `h`.
- A lane level the port does *not* carry must not move the position. That is a
  broadcast, and it is derived rather than declared.
- `reduced` lists the levels an output is presented *after*. They must not
  move the output position; otherwise it is not a reduction.

This is REGION-PROFILES §2.2 (named index expressions) plus §3.1 (a fold is a
spatialized level), in the parked `LocalContract` notation
(`parked/dataflow/kernels/dot_product.py:79-108`), without the authoring,
View and persistence stack that sank it.

A *folding* is how the nest is built from the operation's indices. An index
`n` of extent `N`, folded by `PE`, becomes a beat level `nf` of extent `N/PE`
and a lane level `p` of extent `PE`, with `n = nf·PE + p`. The factors are
Decisions whose domains are divisors of tensor extents (§4.7 covers the
non-divisor case). The *beat order* is also part of the nest.

- Outside the reduction (λ), the operation fixes the beat order: it is the
  output's order.
- Inside λ it is a genuine Decision: the retracted-then-restated "reduction
  order is a dial" (§4.6).

### 3.4 `Presentation` (value; `StreamSpec` renamed and moved)

`Presentation(element, form: Traversal, repetition, markers)` is today's
`StreamSpec` with the same fields (`streams.py:100-111`). What changes is who
holds it. An end holds its presentation and publishes it inside its
`StreamContract` under `PORT`, which it already does. The stream no longer
holds one. A boundary stream holds the one presentation the composite
chooses for its module interface (§6.2).

### 3.5 `Stream` (Space node; revised)

```python
class Stream(Space):
    tensor: Tensor = Param()                   # the fact carried; supplied by the composite
    port: str = Param(required=False)          # boundary ABI name, as today
    boundary: Presentation = Param(required=False)  # a boundary's interface presentation (§6.2)
    ends = Users(PORT)                         # each end's StreamContract, as today

    endpoints  (derived)     # one producer, one consumer; a missing side is the boundary (as today)
    well_formed (constraint) # every end's form traverses `tensor`; element matches
    plan       (derived)     # plan(source, sink): () for direct, else the adaptation steps (§5)
    adapter: None | ReplayBuffer | InputGen | Width | Regroup | Markers | ...
           = Decision over nodes               # each candidate refuses unless it realizes `plan`
    transport: Direct | Fifo = Decision over nodes   # as BufferedStream today; keys unchanged
    connection (view)        # requires well_formed, the adapter and the transport; feeds netlist
```

The stream makes two orthogonal choices, following the canon's line between a
converting unit and a channel (NETWORK §2, §4; bounded-channel §2).

- **`adapter` changes the sequence.**
  - The `None` candidate admits only the empty plan.
  - Every other candidate is a node with two ports. Its input end equals the
    source's presentation and its output end the sink's. The stream checks
    both sides, as it checks the FIFO stage today (`streams.py:282-289`).
  - An adapter may be a chain, e.g. `vpc` then `input_gen`. The engine treats
    a chain as one composite candidate (§5.4).
- **`transport` never changes the sequence.**
  - It is `BufferedStream`'s Decision (`streams.py:307-311`), unchanged:
    `direct`, or a `fifo` with its depth and `ram_style`.
  - A FIFO is a *connection property* in the roster's sense (STREAMS §4),
    which a bounded-channel analysis sizes later.
  - It sits on the consumer side of the adapter.

Keeping `transport` separate keeps today's `weight_stream.transport.*` keys. It
also avoids a clash with the MatMul composite's own `realization` Decision
(`native`/`dense`, M0 X8).

### 3.6 Ownership

| Fact | Kind | Owner | Today |
|---|---|---|---|
| Tensor shape and element | Param (fact) | composite, from the operation → `Stream.tensor` | inside the composite's `StreamSpec.form.shape` |
| Contraction: which indices are free, reduced, shared | Param (fact) | the composite that computes it (MatMul) | pattern enum (Q1), pinned broadcasting |
| Folding factors (PE, SIMD, and later TH) | Decision | the owner of the nest; domains from tensor extents | composite Decisions (`mvau.py:182-183`) |
| Beat order outside λ | derived | the operation | implicit in hand forms |
| Beat order inside λ | Decision (integer arithmetic only) | the owner of the nest | absent |
| Field order of a port (`h`) | fact of the hardware | the end kernel (dotp: weights `(p, s)`; per-channel activation `(s, p)`) | encoded in hand checks |
| An end's traversal | derived | the end: nest + access + fields | adopted from the stream |
| Broadcast (`ACTIVATION_BROADCASTING`) | derived | dotp, from its access (does X use the output lane level?) | pinned 1 |
| Required markers | derived | the consumer (dotp: the reduced suffix) | composite's `Every(synapse_folds)` |
| Offered markers | derived | the producer or an adapter (`replay_buffer`: `olast`, `ofin`) | as today |
| Repetition (cyclic) | property of the producer | the producer (delivery) | as today |
| Boundary presentation | Param, defaulting to the internal end's | the composite (its module interface, D11) | the stream spec |
| Plan (what adaptation is needed) | derived | the stream | refusal only |
| Adapter (which hardware carries out the plan) | Decision over nodes | the stream | absent (refusal) |
| Transport (direct or FIFO) | Decision over nodes | the stream | `BufferedStream.transport` |
| FIFO depth, `ram_style` | Decision | the FIFO candidate | as today |
| Pins, clock and reset roles | fact of the end's module | the end kernel's ABI | as today (B2 revision) |
| Design clocks, target period | Param | the design (D11), forwarded to kernels | a kernel Param |

## 4. Deriving valid foldings: worked examples

All forms below are checked in `sketches/indexed_ports.py` against today's
hand-written forms or by enumerating positions.

### 4.1 Dense MatMul: `Y[r, n] = Σ_k X[r, k] · W[n, k]`

Folding `n = nf·PE + p`, `k = kf·SIMD + s`. Nest: beats `(r, nf, kf)`, lanes
`{p, s}`.

| Port | Access | Fields | Derived form | Equals today's |
|---|---|---|---|---|
| dotp activation | `X[r, kf·SIMD + s]` | `(s)` | beats `(r, nf↦stride 0, kf)`, lanes `s` | `replayed_spec`, `mvau.py:223-227` |
| dotp weights | `W[nf·PE + p, kf·SIMD + s]` | `(p, s)` | beats `(r↦stride 0, nf, kf)`, lanes `(p, s)` | `tile(N, K, PE, SIMD).repeated(R)`, `mvau.py:234-239` |
| dotp results | `Y[r, nf·PE + p]`, reduced `kf` | `(p)` | beats `(r, nf)`, lanes `p` | `result_spec`, `mvau.py:241-246` |
| boundary `in0_V` | `once(activation)` | — | beats `(r, kf)`, lanes `s` | `activation_spec`, `mvau.py:216-221` |

The derivation yields more than the forms:

- **Broadcast.** X does not use `p`, so the activation is broadcast across PE
  and `ACTIVATION_BROADCASTING=1` is derived.
- **Replay.** `nf` has stride 0 in the activation form. Reuse is needed, and
  `classify(once(activation), activation)` is a `REORDER` whose frame is SF
  beats with a coefficient-0 dimension of NF. `replay_buffer(LEN=SF, REP=NF)`
  realizes exactly that special case (§5.3). The replay count is derived
  from the nest, so F1 cannot recur.
- **Frame.** The reduced levels `(kf)` are the innermost beat suffix, so the
  frame is `Every(SF)`: the marker dotp requires.
- **Weight repetition.** `r` has stride 0 outermost in the weight form.
  `period(weights) = tile(N, K, PE, SIMD)` is what a cyclic delivery packs,
  as `weight_period` does today (`mvau.py:229-232`). External delivery
  presents the full form at `in1_V`.

### 4.2 Per-channel (depthwise): `Y[r, c] = Σ_k X[r, c, k] · W[c, k]`

Folding `c = cf·PE + p`, `k = kf·SIMD + s`. Nest: beats `(r, cf, kf)`, lanes
`{p, s}`.

| Port | Access | Fields | Consequence |
|---|---|---|---|
| dotp activation | `X[r, cf·PE + p, kf·SIMD + s]` | `(s, p)`: field `s·PE + p`, FinnLib's order (`dotp_axi.sv:113-117`) | X uses `p`: not broadcast (`ACTIVATION_BROADCASTING=0`), PE·SIMD lanes |
| dotp weights | `W[cf·PE + p, kf·SIMD + s]` | `(p, s)`: `[PE][SIMD]` (`dotp_axi.sv:99`) | `r` stride 0: cyclic repetition, as dense |
| dotp results | `Y[r, cf·PE + p]`, reduced `kf` | `(p)` | `vector_major((R, C), PE)` |

No beat level has stride 0 in the activation form, so no replay is needed. The
consumer still requires `Every(SF)`. The plan from a marker-less producer is
therefore *marker synthesis only*: `replay_buffer` with `REP=1`, or a marker
generator (§5.3). This is MatMul's X5, derived.

The sketch enumerates every beat and field for two operand layouts:

- the SPEC equation's `(r, c, k)`;
- FINN's im2col `(r, k, c)`, channel innermost. This is the layout M0 Q5
  adopts, `(R, K, C)`.

In both, field `s·PE + p` holds `X[r, cf·PE + p, kf·SIMD + s]`, and M0's
declared traversal falls out as `present(...)` over `(R, K, C)`. Q7's
per-channel port relation is therefore zero lines of kernel code. E-048
(hlslib's `(SIMD-1-s)·PE + p`) is a *different field order of the producer*.
If an SWG presents it, the plan names a `LANE_PERMUTATION`, which costs only
wires.

### 4.3 Thresholding: `T'[…, c] = f_c(T[…, c])`

Nest: beats `(…, cf)`, lanes `{p}`. The input and output accesses are both
`T[…, cf·PE + p]` with fields `(p)`, which gives
`vector_major(shape, PE)`. That is what `thresholding.py:378` checks for
today, so the check becomes the derivation. The set-index stream is a tensor
of one index per input beat, shape `(beats,)`, one lane: its form is derived
from the nest's beat levels. The requirement "one index per input beat"
(`thresholding.py:397`) holds by construction. The folding domain "C divides
PE or PE divides C" (`thresholding.py:172`) stays a thresholding admission
rule. A PE larger than C is a fold that wraps across `r`, which a
single-level split cannot express. §11 records it as an open question.

MatMul(PE=4) feeding thresholding(PE=2) in one module: the plan is
`WIDTH_CONVERSION` 4→2 (sketch `check_adapters`). With the same PE it is
`IDENTITY`. The stream refuses a width mismatch only when no admitted adapter
realizes the plan. Whether an adapter is admitted is the stream's `adapter`
Decision.

### 4.4 A FIFO and other transport

A FIFO has no nest. It is an identity stage: its two ends present the
source's presentation. It is therefore a `transport` candidate of the
stream, not a kernel on two streams. That keeps today's `StreamFifo` idea
(`streams.py:169-191`). Depth and `ram_style` stay its Decisions. Sizing
belongs to a future bounded-channel analysis, which needs requirement and
availability (§7). A FIFO never repairs a sequence mismatch
(bounded-channel §6). The plan must be satisfied without it.

### 4.5 What becomes of B1's dotp checks

dotp's port checks shrink to one **admission rule over the nest**
(`indexed_ports.py:dotp_admits`):

- exactly one output lane level (P) and one reduction lane level (S);
- the weights use both lane levels;
- the activation uses S;
- the levels the output does not use (the frame) are the innermost beat
  suffix, and there is at least one.

| B1 check | Where it goes |
|---|---|
| activation lanes = SIMD consecutive columns (`dotp.py:338`) | derived: fields `(s)` of `X[…, kf·S + s]` |
| one frame-marker rule on the activation (`dotp.py:320-324`) | derived requirement `frame(nest, reduced)`; the plan supplies it |
| a frame within one activation row (`dotp.py:341-344`) | admission: the reduced levels are the innermost suffix (`r` cannot be inside) |
| weights are a matrix over the activation's columns (`dotp.py:351-356`) | by construction: W and X share `kf`, `s` |
| weight lanes are PE rows × SIMD, SIMD fastest (`dotp.py:357-360`) | derived: fields `(p, s)` |
| the weight column walk follows the activation's (`dotp.py:361-363`) | by construction: one nest |
| a frame reads one group of weight rows (`dotp.py:364-367`) | admission (the same suffix rule) |
| result lanes = PE (`dotp.py:375-377`) | derived: fields `(p)` |
| results have the weights' rows as columns (`dotp.py:378-380`) | by construction: Y and W share `nf`, `p` |
| each result beat holds its frame's row and weight rows (`dotp.py:382-398`) | by construction |
| element encodings (`dotp.py:314-318`) | unchanged: the tensor's element vs the port scalar |

The B1 regression probes (`test_port_contracts.py:69-117`) map as follows:

- A wrong lane count or a transposed tile can no longer be *written* into dotp,
  because inside a composite the forms are derived. Across a stream, the same
  inputs become a plan, not a dotp refusal: a lane-count mismatch is a
  `WIDTH_CONVERSION`, a transposed tile a `LANE_REGROUP`
  (`test_stream_contract.py:244-253`).
- "A frame crossing rows" is an admission refusal: `order="nf kf r"` in the
  sketch.
- "Results that swap frames" has no analogue, since results are derived.

`beat_walk`, `walk_axis` and `split_walk` (`forms.py:342-400`) then have no
users.

dotp keeps what is truly its own: DSP widths, pumping, accumulator capacity,
segmentation, per-channel requiring DSP58 (SPEC §2), and the admission rule.

### 4.6 The reduction-order dial

Conv as matmul reduces over `k = (kh, kw, c)`. The nest's beats inside λ can
be `(kh, kw, cf)` or `(kh, cf, kw)`. dotp admits both, since the frame is still
the reduced suffix. The two activation forms differ, and `classify` names them
a `REORDER` apart (`check_reduction_order`).

The legal permutations are `Sym(d ≥ λ)`, which is REGION-PROFILES §3.2 and the
memory note. They are a genuine Decision of the nest's owner, with two
consequences:

- **The producer must follow.** An SWG must emit the chosen order, or the plan
  inserts a reorder. The choice is visible in the plan's cost; no rule
  forbids it.
- **It is legal only for exact arithmetic.** Float accumulation reorders
  change numerics, so the Decision's domain collapses to the canonical order
  for float operands. Integer MatMul admits the full group.

Recommended: declare the domain now. Commit the canonical order as the only
candidate until an optimizer exists, because the best legal order is open.

### 4.7 Padding, reshape and non-divisor folds (extensions, not built)

Two kinds of padding need keeping apart:

- **Carrier bit padding.** Byte-aligned AXIS words. Physical, and already
  handled by `Composition.connect` (RECORD B3 "child padding").
- **Logical padding.** A fold that does not divide the extent. The traversal
  then addresses positions outside the tensor.

The second needs `Tensor` to gain padded extents and a pad value, and the
plan to gain two adapter kinds, `pad` and `crop`. Zero padding of a reduced
index is numerically harmless for dotp; padding of a free index produces
results to discard. Neither FINN's MVAU nor this code admits non-divisor
folds today (`mvau.py:122-123`), so nothing forces it. It is open (§11).

A row-major reshape between a producer's tensor and a consumer's view is free,
because `Traversal` works on flat row-major offsets. Supporting it needs only
that `classify` compare flat offsets when the stream declares a reshape view.
A transpose is a real reorder.

## 5. Mismatch: plans, refusals and adapters

### 5.1 What the stream compares

`plan(source, sink)` compares two presentations of one tensor at these levels:

| Level | Compared | Outcome if different |
|---|---|---|
| element | encoding | refusal (a conversion changes values; it is a compute kernel, not an adapter) |
| positions per pass | multiset of presented positions after replay is accounted for | refusal (`INCOMPATIBLE`) |
| beat order (`π`) and replay | beat loop nests | `REORDER` (buffered; replay is the zero-coefficient case) |
| lanes per beat | lane extents | `WIDTH_CONVERSION` (same element order) |
| lane axis | which loop is spatial | `LANE_REGROUP` (banked transpose) |
| field order (`h`) | lane loop order at equal beats | `LANE_PERMUTATION` (wires only) |
| markers | sink's required rules against those offered | marker synthesis, or derived by an adapter that emits them |
| repetition | `ONCE` feeding `CYCLIC` | refusal (no capture-and-cycle adapter exists) |
| protocol | direction, carrier padding, reset polarity | wires (as today) or refusal |
| clock domain | (design level only, §6.4) | CDC adapter, when a design has two domains |

Every row but the last exists today as a verdict of `classify` or `compatibility`
(`forms.py:276-308`, `contract.py:103-168`). Three things change:

1. **A chain instead of one class.** `classify` returns one adaptation. A
   compound mismatch returns `INCOMPATIBLE` even when it is realizable, for
   example different lanes *and* a different beat order. `plan` decomposes it
   canonically: regroup to the greatest common lane count, reorder, regroup to
   the sink's lanes. A plan is a tuple of steps, and `()` is direct. The
   canonical chain is one way to carry it out; adapter candidates may offer
   cheaper fused ones, such as `input_gen` doing replay and reorder at once.
2. **Markers are part of the plan.** Today a missing marker is a refusal
   (`contract.py:146-153`). In the new model it is a step: synthesize
   `Every(k)` on the sink's pass. Reordering invalidates offered markers
   (COMPARISON §5), so a plan recomputes markers after every step, from the
   step's output presentation.
3. **The verdict is not the choice.** `plan` says what must happen. The
   stream's `adapter` Decision says which hardware does it. A plan with no
   admitted candidate refuses at the stream: code `stream-plan`, with the
   plan in the message.

### 5.2 Refusal or adaptation

The rule is the roster's, sharpened. **If the logical sequence is unchanged,
the difference is a connection property, realized as wires**: field
permutation, carrier padding, reset polarity, marker level select. **Otherwise
it is an adapter node**, and the stream refuses unless its `adapter`
Decision selects one that realizes the plan. There is no implicit
adaptation, as NETWORK §4 requires. An adapter is visible as a node, with
its own key, module and fingerprint.

The composite controls whether adapters are *admitted* by narrowing the
`adapter` Decision (AUTHORING "Overrides"). A MatMul that wants the
replay-or-nothing space narrows its activation stream to
`{replay_buffer, input_gen}`. A design that forbids adapters pins its
streams' `adapter` to `None`. Refusals then prune foldings: the plan of a
PE-mismatched pair has no admitted candidate. This is the operational sense in
which "the stream knows which foldings of its tensor are valid".

### 5.3 The adapter kinds that fall out

| Plan step | Candidates | Sequence change | Wrapped today |
|---|---|---|---|
| none | `adapter=None` | none | yes |
| identity buffering | `transport=fifo` (FinnLib `fifo`) | none | yes (`fifo.py`) |
| lane permutation | `adapter=None` (wires) | field order only | yes (`contract.lane_permutation`) |
| replay (one stride-0 loop above the innermost `LEN` beats) | `ReplayBuffer`, `InputGen` | count × REP; `olast`/`ofin` | yes (`streaming.py`) |
| general reorder, incl. replay and broadcast expansion | `InputGen` (`Reorder` → `FM_SIZE/DIMS/COEFS`) | order, count; `olst[d]` | standalone only (`input_generator.py`) |
| width conversion | `Width` over `vpc` | lanes, count; drops markers | no |
| lane regroup (2D transpose with SIMD lanes) | `Regroup` over `inner_shuffle` | lane axis; drops markers | no |
| marker synthesis | `ReplayBuffer(REP=1)` | markers only | yes |
| logical pad / crop (§4.7) | `pad1d`, `crop` | positions | no |
| set-index repetition | `stream_tap` (`TAP_REP`) | count | no |
| clock-domain crossing | an async FIFO `transport` (design level) | none | no |

### 5.3.1 FinnLib components (`deps/finnlib` at `11b5c64b`)

FinnLib has no separate DWC, outer shuffle, transpose, marker generator or SWG
module. `vpc`, `inner_shuffle` and `input_gen` cover those roles, and the
`olst`/`olast`/`ofin` outputs of `input_gen` and `replay_buffer` provide the
markers.

- **`vpc`** (`rtl/shape/vpc.sv:22-29`)
  - *Parameters:* `W`, `N` elements per vector, `PI` → `PO` lanes, `PAD_ZEROS`.
  - *Behaviour:* each `N`-element vector takes `⌈N/PI⌉` beats in and
    `⌈N/PO⌉` beats out. Element order is unchanged, vectors are never packed
    together, and the last beat's unused lanes are zero-filled (`:9-13`).
    There is no marker port.
  - *Hence:* `vpc` realizes a `WIDTH_CONVERSION` exactly when `PI` and `PO`
    both divide the innermost contiguous run `N`. Otherwise it produces a
    *padded* sequence, which is REVALIDATION V41 and the §4.7 extension.
    `Width` refuses that case until padding exists. A required marker must be
    re-synthesized after `vpc`.
- **`inner_shuffle`** (`rtl/shape/inner_shuffle.sv:39-41, 61-67, 84`)
  - *Behaviour:* an `(I, J)` row-major matrix, `SIMD` elements per beat, is
    emitted column by column with the lanes along `i`. Requires `I % SIMD == 0`.
  - *Storage:* double-buffered, `2·I·J` elements in `SIMD` banks.
  - *Hence:* it realizes one specific `LANE_REGROUP` shape. `Regroup` must
    recognize that shape in the plan and refuse others, which fall back to a
    `vpc`/`input_gen` chain.
- **`input_gen`** (`rtl/shape/input_gen.sv:9-34`)
  - *Behaviour:* per frame of `FM_SIZE` words, emits `in[f·FM_SIZE + Σ COEFS[k]·i_k]`
    over `DIMS`. Coefficient 0 replays. Words are opaque, so lanes are packed.
    `olst[d]` marks the completion of level `d` and everything inside it.
  - *Legality:* `Σ COEFS(DIMS-1) < FM_SIZE`.
  - *Storage:* a power-of-two circular buffer, sized conservatively above the
    live window that `input_gen_span_tb.sv` checks.
  - *Hence:* it realizes every `REORDER` that `classify` names (the
    coefficients are checked against FINN, `test_stream_contract.py:174-216`).
    It is also FinnLib's SWG and broadcast expander (`conv2d.sv:114-126`,
    `where.sv`). Its `olst` is a *level* marker, which bears on Q7 in §11.
- **`replay_buffer`** (`rtl/infra/replay_buffer.sv:49-53`)
  - *Behaviour:* `LEN`, `REP`, `W`; emits `olast` per sequence and `ofin` on
    the last repetition. `REP=1` is a combinational identity that still
    generates `olast` (`:107-115`), which makes it the marker synthesizer.
  - *Storage:* `2^⌈log2 LEN⌉` words.
- **`fifo`** (`rtl/infra/fifo.sv`): identity. `fifo_sim.sv` is an unbounded
  queue for simulation-based sizing.
- **`stream_tap`**, **`crop`**, **`pad1d`**: `TAP_REP` value repetition, and
  border crop and pad. Not wrapped yet. They are the candidates for the
  set-index and padding steps.
- **dotp per-channel.** `ACTIVATION_BROADCASTING=0` requires `VERSION==3`
  (DSP58; `dotp_axi.sv:78-80`). The activation field order is `s·PE + p`
  (`:113-117`). The weights are `[PE][SIMD]`, i.e. fields `(p, s)`
  (`:99, :102`). Both are what `sketches/indexed_ports.py` assumes.

### 5.4 How the adapter Decision is built (no engine change)

The probe (`engine_probe.py`) builds exactly this on today's engine:

- **Ends.** Each end exports its presentation per input
  (`exports = {PORT: {stream: view}}`) and reads only `stream.tensor`, a Param.
  No value cycle.
- **Plan.** The stream derives `plan` from `Users(PORT)`.
- **Candidates.** The stream declares
  `adapter = Decision(values={"none": NoAdapter(plan=plan), "dwc": Dwc(plan=plan), ...})`.
  (The probe calls these `realization` and `Direct`.) Each candidate binds
  its formal to the stream's derived plan and refuses itself when it cannot
  realize it. The stream's view obliges the selected candidate's stage.
- **Domains.** A folding Decision's domain can depend on
  `out.extent`, read through the reference input.

A chain is one composite candidate, e.g. `WidthThenReorder`, placing two
stage nodes that `netlist` wires in order. `netlist` today handles one stage
per stream (`streams.py:414-427`). It needs a loop over a stage list (adapter
stages, then the FIFO), a change in the kernel layer, not the engine.

## 6. Instance wiring (D11)

### 6.1 Three kinds of stream, one class

| Stream | Ends | Who derives the presentations | Typical plan |
|---|---|---|---|
| intra-module | two children of one composite | the composite's nest (§4) | identity; derived reuse (replay) |
| boundary | one child and the composite's module interface | the child, and the composite's boundary Param | identity, or reuse realized inside the module |
| inter-module | two kernel instances in a design | each kernel's own boundary presentation | width conversion, FIFO sizing, reorder between independently folded kernels |

One `Stream` class serves all three. The difference is the source of the
presentations. Inside a composite they come from one nest and agree by
construction. Across modules they come from independently folded kernels,
and that is where adapters earn their keep.

### 6.2 The boundary presentation is the module's interface

D11 says a Kernel "exposes a complete pin interface". Its logical sequences
are part of that interface. A boundary stream therefore takes the composite's
chosen `boundary` presentation. When absent, it adopts the internal end's,
with plan identity and no adapter.

This makes the *place of reuse* an explicit decision:

- **MatMul activations.** The boundary presents `once(activation)`, so the
  plan is a replay inside the module. Presenting the replayed sequence at the
  boundary is legal too. It moves the replay upstream, and some designs might
  want that.
- **External weights.** The boundary presents the full consumer form,
  repetition included, as FINN's `in1_V` does.
- **Cyclic weights.** There is no boundary stream at all; the producer is
  internal and presents `period(weights)` cyclically.

Delivery (external or cyclic) and replay (inside or outside) are two instances
of one question: which of the consumer's reuse loops the module realizes
internally. The design keeps them as separate Decisions and does not force
a merge (§11 Q6).

### 6.3 Is a stitched design one more composite? Yes, with two caveats

A kernel composite already does everything a stitch needs:

- places children as `ModuleBuildRequirements` with an ABI;
- wires streams by `Connection`;
- drives clocks and resets by role;
- exports control buses;
- ties unused inputs.

The one thing it lacks, to *be* a child, is a per-stream `PORT` export for its
boundary. That is the `boundary_contract` it already computes
(`streams.py:139-148`). So a `Design` is a Space that places kernel composites
beside inter-module `Stream`s, and a kernel composite gains reference inputs
for the streams it sits on:

```python
class MatMulKernel(Space):
    input_stream: Stream = Param(required=False)    # the design stream it consumes
    output_stream: Stream = Param(required=False)
    ...
    exports = {MODULE: build_requirements,
               PORT: {input_stream: in0_contract, output_stream: out0_contract}}
```

`netlist` wires the design unchanged: children are modules, adapters and
FIFOs are stream stages, and the top pins are the accelerator's interface. The
same idiom then serves all three levels.

**Caveat 1: ownership, not machinery, separates Kernel from Design.** D11's
line holds: a kernel does not wire its own instance. The Design owns the
clocks and frequencies, and forwards the target period to every kernel as a
Param, which it already is. It also owns inter-module adapters and FIFO
sizing, and the *operation-to-module mapping*. MatMul+Thresholding in one
module versus two is a Design Decision over *which composite owns the stream
between them*. The streams, presentations and plans are identical either way.
That answers M-D2: fusion is a composite boundary choice, not a kernel flag.

**Caveat 2: two things are unverified.** Neither is expected to fail.

- *Nested lowering.* A composed module as a child of another composition has
  not been tested. `lower_module_structure` receives children whose
  requirements carry their own generated wrappers. Verify in S4 before
  relying on it.
- *Scale.* A design is hundreds of nodes, and "reusing cached results across
  snapshots" (STATUS Phase D) matters there first.

HLS kernels stay unplaceable until the HLS synthesis stage exists (Phase D).

### 6.4 Clocks

The design has one clock, plus `ap_clk2x` when a child declares one, driven
by role. That is today's `netlist`, which is exactly FINN's
`CreateStitchedIP` (RECORD B2 revision). Nothing about clocks enters the
logical stream.

A design with genuinely separate domains would make a per-instance domain
assignment a Design Decision. The plan would then gain a protocol-level
comparison of the ends' domains, answered by a CDC adapter (async FIFO).
B2's domain nodes were removed because at kernel level they carried no
choice. At design level they may. Leave them out until a design needs two
domains. The scheduling survey finds clock and phase composition under
backpressure an open research question (SURVEY §1.3).

## 7. Relation to `finn.dataflow` and the scratchpad canon

| Canon / `finn.dataflow` concept | Disposition | Reason |
|---|---|---|
| Operand `(identity, element_type, shape)` (REGION §2) | **adopt** as `Tensor`; identity = the stream node | the fact both ends read |
| `BeatSequence` (REGION §5) | **adopt**, encoded as `Traversal` | compact and canonical; `classify` works on it |
| canonical stream form `(t, b, s, π, h)` (PROFILES §5.1) | **adopt**: `π` = beat order, `h` = a port's `fields` | the §5.1 split between traversal and field order is exactly the right one |
| named index expressions, spatialized levels (PROFILES §2.2, §3.1) | **adopt** as `Nest`/`Access` | the fold belongs to a level; each port follows from its stride |
| suffix-permutation capability `Sym(d ≥ λ)` (PROFILES §3.2) | **adopt** as the reduction-order Decision | the reduction-order evidence |
| Region owns one schedule (REGION §1) | **adopt**: one nest per kernel or composite | the anti-duplication rule behind P1/P4 |
| requirement vs presentation (REGION §4–5) | **adopt the distinction**; requirements/availability not materialized yet | needed later for FIFO sizing and live windows; nothing today consumes them |
| edge = exact sequence equality under a position map (NETWORK §2) | **revise**: a plan with explicit adapter nodes; a reshape view as the only position map for now | NETWORK §4 already says adapters are ordinary units; the stream makes them first-class candidates |
| OneToOne pass correspondence (NETWORK §2) | **revise**: add `Repetition.CYCLIC` | MODEL §3 anticipates "compact repeated dataflow" needing an extra contract; delivery needs it |
| markers belong to realization when derived from a static contract (MODEL §5) | **adopt**: markers are derived from the nest and synthesized by the plan | |
| `finn.dataflow.model.logical` code (region, maps, network, validation) | **leave to the `finn.dataflow` model pass**; recommend retire | untested since `0d700b1ab`, no live importer outside `finn.parked`, no markers, no adapters, two map encodings |
| parked index notation (`dot_product.py:79-108`) | **revise** into `Nest`/`Access` | right idea; its authoring stack was the problem |
| parked replay as a copied operand `XR` | **drop** | replay is a stride-0 loop of one tensor |
| TENSOR/BLOCK/STREAM tiers (archived KernelOp) | **drop** the tiers, **keep** "lanes derived across interfaces" | last-axis tiling could not express the weight tile; a nest can |
| physical composition contract: endpoints, routes, one authority for clocks (open/…composition-contract §3.4–3.6) | **revise**: vocabulary for §6 | `netlist` and role-driven clocks already implement most of it |

Where the values should live is decision 2 of §0. The layering
`finn.core.space ← finn.dataflow ← finn.kernels` puts logical values in
`finn.dataflow`. `Traversal`, `Loop`, `classify`, `Reorder`, `Every`,
`Repetition`, `pack`, `Tensor`, `Nest` and `Access` are logical: they carry no
pins or bits. Moving them there makes `finn.dataflow` hold the one live
beat-sequence encoding, instead of an untested second one.

## 8. Migration

### 8.1 Increments

Each increment is independently landable, ends at the usual gates, and records
key or fingerprint changes (D7).

| # | Content | Stays | Evidence | Cost |
|---|---|---|---|---|
| **S1. Presentation at the ends** | `Stream.spec` → `Stream.tensor` + an optional `boundary` presentation. `StreamSpec` → `Presentation`, held by ends. Kernels publish their own presentation: thresholding derives `vector_major(tensor.shape, pe)`; replay, FIFO and delivery as today; dotp takes its three presentations as Params from the composite and keeps B1's checks for now. MVAU computes the same forms and passes them to the ends instead of the streams | all of `forms.py`; B1 checks; `netlist` | both gates; MVAU fingerprints and keys identical; numeric sweep 28/28 | 1–2 days |
| **S2. The nest** | `Tensor`, `Level`, `Nest`, `Access`, `present`, `frame`, `once`, `period`, and a `Contraction` (einsum-like) replacing SPEC Q1's enum. MatMul declares the contraction, PE/SIMD Decisions (tensor-extent domains) and a reduction-order Decision (one candidate for now), and derives every presentation. dotp takes the nest and accesses, and derives its PE, SIMD, broadcasting, frame and presentations plus the admission rule (§4.5). `_fields`, `_frames`, `beat_walk`, `walk_axis`, `split_walk` are removed | `Traversal`, `classify`, `ReplayBuffer`, `CyclicDelivery` | sketch equalities as unit tests; dense fingerprints identical (forms equal); per-channel sweep in M3 | 3–5 days |
| **S3. Plans and adapters (C6; C3 falls out)** | `plan` (chains, markers). `adapter` Decision over nodes beside the unchanged `transport`; `netlist` wires stage lists. Replay leaves MatMul's body and becomes a candidate on the activation stream: `replay_buffer` or `input_gen` is C3. Adapter kernels: `input_gen` onto streams, `vpc`, `inner_shuffle` | everything above | per adapter kind: XSI against an independent golden (fixture-5 lesson); recorded key and name changes (`replay` → `activations.adapter.replay_buffer`) | 1–2 weeks |
| **S4. Design composite (D11 instance wiring)** | kernel composites export boundary `PORT`s; a `Design` Space; target period and clocks as design Params; nested lowering verified | all | two kernels in one design, in XSim | with the artifact-integration layer |
| **S5. Extensions, on demand** | padding (§4.7), reshape views, CDC, requirements/availability for FIFO sizing | | per extension | — |

### 8.2 Order relative to the MatMul spec and Phase C

[M0.md](../matmul-kernel-2026-09-27/M0.md) reorders the MatMul increments:

- M1: the rename (in progress in this worktree during this round);
- M2: the core split (FinnLib `CORE`, two core kernels, a `compute` Decision);
- M3: the INT8 core's per-channel mode with a *labelled* port relation;
- M4: one composite for both contractions;
- M5: the dense realization and `NARROW_WEIGHTS`.

Against that order:

- **M1 and M2 are independent of D10.** Do S1 after M1, to avoid churning the
  renamed files twice. M2 moves dotp's ports onto two core kernels, so S1's
  port change should land after M2 or together with it.
- **S2 replaces M3's port relation.** M0's Q7 already labels operand axes
  with indices (`X[r,k,c]`, `W[c,k]`, `Y[r,c]`) and *checks* that walks agree
  on shared indices. That is S2's `Access` without the nest. With the nest,
  the same labels *derive* the walks, so the check disappears. Doing S2 as
  M3's foundation costs about what the labelled check costs, and Q7's
  stopgap is never written. If M3 lands first, S2 deletes its relation with
  the rest of B1.
- **M4 is where S2 pays off most.** One composite, one contraction; broadcast,
  replay presence and the frame marker are all derived (§4.1, §4.2).
- **M5's dense realization of per-channel (M0 X8) is a reshape view.** It reads
  `(R, K, C)` as `(R, K·C)`, the same row-major sequence (§4.7), over a
  block-diagonal `W'`. In the nest model that is a second contraction over
  derived operands, selected by MatMul's own `realization` Decision.
- **C3 and C6 are S3.** C4/C5 (memstream, multi-set) are orthogonal. The
  set-index stream is an ordinary stream whose tensor is derived from the
  consumer's beat levels.

### 8.3 What changes in the live code

- `streams.py`:
  - `StreamSpec` is renamed and moves to the ends;
  - `Stream` gains `tensor`, `boundary`, `plan` and `adapter` (`transport` stays);
  - `BufferedStream`'s `transport` moves onto `Stream`, keys unchanged (S3);
  - `netlist` handles a stage list (S3).
- `contract.py`: `compatibility` becomes the check of one plan step, and the
  direct case of `plan`.
- `forms.py`: gains `plan` (S3); loses the walk helpers (S2); moves to
  `finn.dataflow` if decision 2 goes that way.
- `dotp.py`: its port views derive instead of check; the admission rule
  replaces `_fields`/`_frames`; `ACTIVATION_BROADCASTING` becomes derived.
- `mvau.py` (MatMul): the four `*_spec` derivations become one contraction
  plus a nest; `_Folding` dissolves into the nest; the `replayed` stream and
  `replay` node go in S3.
- `thresholding.py`: ports derive from the tensor and PE; the set-stream check
  becomes the set tensor's derivation.
- Tests:
  - `test_port_contracts.py` probes are remapped (§4.5);
  - `test_stream_contract.py` gains plan tests;
  - the sketch's checks become `test_nest.py`.

### 8.4 Engine needs

None. The probe (`engine_probe.py`) exercises every engine feature the design
uses:

- per-input exports;
- `Users` aggregation into a derived value;
- a Decision over nodes whose candidates bind a parent's derived value and
  refuse themselves;
- a Decision domain depending on a value read through a reference input.

Two known frictions remain, neither blocking. Optional streams still need
`when=` guards (STATUS §3; the thresholding set stream). A stream refused for
having no users forces boundary names up front.

## 9. Alternatives considered

| Alternative | Why rejected |
|---|---|
| **The stream owns the folding** (a lanes/order Decision per stream; kernels derive PE/SIMD from their streams) | A kernel's parallelism is one level spanning several streams: PE is in the weight and result lanes, SIMD in the activation and weight lanes. Over-determined stream choices must then be reconciled, which is checking again. An adapter makes "one stream, one folding" false. Canon-v1 §7.1: a fold belongs to a level, never a port |
| **Symbolic traversal families on the stream** (the stream holds the set of valid foldings, the ends pick) | Needs symbolic loop algebra and unification the engine does not have. The domain of a folding is already the divisor set of a tensor extent, which the tensor supplies concretely |
| **Keep kernel-side checks, one set per pattern** (Q7 as the end state) | Every new contraction (per-channel, tiled TH, batch interleave, conv reduction orders) adds hand checks that duplicate a derivation, and the composite can still write forms that disagree (F1) |
| **Adopt `finn.dataflow` Region/Network wholesale** | Untested since `0d700b1ab`; strict equality edges and no adapters; no markers; two map encodings; heavier than the kernel layer needs. Its concepts are adopted (§7), its code is not |
| **A full polyhedral (ISL) model** | Rectangular affine nests cover the roster, including sliding windows: an access may give several levels coefficients on one axis. Nothing needs non-affine or non-rectangular sets, and an ISL dependency is heavy |
| **A demand channel** (a consumer exports the form it wants; a polymorphic producer reads it from the stream) | Works on the engine: a separate export key avoids the cycle. Inside a composite, the nest's owner hands producers their form directly, which is simpler. Kept in reserve for design-level reusable producers (the "supply waterfall" memory note) |
| **The composite writes the boundary form and the internal forms separately** (today) | P1 and P4 |

## 10. Risks

| # | Risk | Mitigation |
|---|---|---|
| K1 | `plan` decomposes a compound mismatch wrongly; an adapter then moves data silently wrong | per-kind XSI against an independent golden; mutation tests with NF≠SF and field-order swaps (post-S4 lessons); `classify` already agrees with FINN's coefficient generators |
| K2 | S3 renames instances and keys (replay moves into the stream) | recorded under D7, as the delivery rename was (D2) |
| K3 | The nest cannot express a roster member: tiled MVU (TH), batch interleave, sliding windows with padding, PE > C thresholding | TH and batch interleave are expected to be a second split of a lane level. This is *not sketched*; the tiled-MVU forms exist as `Traversal`s (`test_stream_contract.py:174-206`) and should be the first S2 test. SWG is affine. Padding and wrap-around folds are §4.7 and §11 |
| K4 | Field-order conventions (`h`) are hardware facts and can still be declared wrong | they become one declared tuple per port, not a hand check. E-048 shows the failure is a convention nobody wrote down, which a declared `fields` fixes |
| K5 | Over-derivation hides a choice. The boundary presentation and the place of reuse are choices, not facts | explicit composite Params (§6.2); a Decision where more than one is meaningful |
| K6 | Float MatMul makes the reduction-order dial unsound | the Decision's domain is gated on exact arithmetic (§4.6) |
| K7 | vpc's per-vector padding makes some width conversions change the logical sequence (REVALIDATION V41) | `Width` refuses the non-dividing case unless the plan accounts for padding (§5.3.1) |
| K8 | Nested lowering of a composed module inside a design is untested | verify at the start of S4 |

## 11. Open questions and decisions for the user

Decisions (recommendation first):

1. **S2 as M3's foundation** (§8.2). *Recommended: yes.* Otherwise Q7's
   labelled relation is built and then deleted.
2. **Values in `finn.dataflow`** (§7). *Recommended: yes*, and the old logical
   model goes to the `finn.dataflow` model pass for retirement.
3. **Boundary presentation** (§6.2). *Recommended:* an explicit composite Param,
   defaulting to the internal end's presentation.
4. **Contraction syntax.** An einsum string (`"rk,nk->rn"`) plus a folding map,
   or `Access` objects written directly. *Recommended:* einsum for MatMul,
   because it replaces Q1 and is readable. `Access` stays for everything else.
5. **Reduction-order Decision** (§4.6). *Recommended:* declare the domain,
   one candidate until an optimizer exists.
6. **Delivery and replay as one "where is reuse realized" choice** (§6.2).
   *Recommended: no, keep them separate for now.* Their candidates differ in
   kind: a constant source vs a buffer of a streamed tensor. Revisit when
   memstream (C4) makes "buffer a streamed weight" possible.

Open questions (no recommendation yet):

7. **Markers:** keep `Every(period)` only, or add `Level(d)` (marker on
   completion of a named nest level)? `Level` survives reorders better, and
   `input_gen`'s `olst[d]` is a level marker natively. `replay_buffer` and
   AXIS `TLAST` are periodic. In a rectangular nest the two are
   interconvertible (`Every(∏ inner extents)`), so this is a question of
   which one is canonical.
8. **Cyclic repetition** is matched only as whole passes, outermost
   (`contract.py:171-178`). Is a cyclic source with an inner period (batch
   interleave) ever wanted, or does delivery always pack inner replays into
   its image?
9. **Padding and wrap-around folds** (§4.7, thresholding PE > C): the
   extension's shape, and whether any roster member forces it.
10. **Reshape views across a stream** (§4.7): declared on the stream, or on the
    consumer's `Access`?
11. **`vpc`'s per-vector padding** (§5.3.1): refuse the non-dividing case (the
    recommendation until §4.7), or model it as width conversion plus a pad step?
12. **Design level** (§6.3): per-module IP packaging vs one flat netlist
    wrapper, and where FIFO sizing (bounded-channel analysis) plugs in.
