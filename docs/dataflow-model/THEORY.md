# The logical dataflow model: theory

Date: 2026-09-29. Describes `finn.dataflow` on `feature/kernel-refactor`, with
the kernel-authoring plan built and the code renamed to these terms. The code is
the source of truth. Every worked example here is asserted by
[`check_theory.py`](check_theory.py) against the live code.

**Terminology.** The code uses this document's terms. Where a code name differs
or is worth knowing, it follows in parentheses (e.g. the fold,
`Schedule.steps(i)`). Terms that have an established name elsewhere cite it at
first use ([References](#references)).

This document is the theory: what the objects are, what they mean, and the
rules that relate them. It leaves out how kernels are authored and how
hardware is generated (`finn.kernels`), except where one sentence places a
concept.

## 0. The question

A dataflow accelerator moves tensors between hardware modules over streams.
Each stream transfer, a **beat**, carries a few elements side by side, its
**lanes**. The model answers two questions, in logical order only:

1. In what order do a tensor's elements cross one interface, beat by beat and
   lane by lane?
2. When two interfaces on one stream disagree, what must happen between them,
   or is it impossible?

It says nothing about time: not cycles, stalls, latency, buffer occupancy or
liveness. Those depend on the realization and its environment; logical order
does not determine them (scheduling theory, claim SC-01). A beat is an ordinal,
not a clock cycle.

## 1. Tensors and positions

A **tensor** is a shape `E = (E_0, …, E_{r-1})` of positive extents and one
scalar encoding (a QONNX datatype of positive width) shared by every position.
Its positions are `P = [0,E_0) × … × [0,E_{r-1})`, linearized row-major:

```
lin(p) = Σ_j p_j · Π_{h>j} E_h          (a bijection P → [0, |P|))
```

`lin(p)` is the **flat offset** (numpy's `ravel_multi_index`). All order in the model is expressed in flat
offsets, which is what makes a row-major reshape free: two shapes of equal size
share their offsets.

*Example.* In shape `(2, 4)`, position `(1, 2)` has offset `1·4 + 2 = 6`.

A tensor carries no order and no name. Its identity is the stream that carries
it; in a model it is the graph edge.

## 2. Traversals

### 2.1 Definition

A **traversal** of a tensor is a pair of loop nests over flat offsets, each
listed outer to inner:

```
T = (E, B, L)
B = ((b_1, s_1), …, (b_D, s_D))      beat loops: extent b_i ≥ 1, stride s_i ≥ 0
L = ((l_1, t_1), …, (l_G, t_G))      lane loops: extent l_j ≥ 1, stride t_j ≥ 0
```

It presents `N_B = Π b_i` beats of `N_L = Π l_j` lanes each. Beat `β` has the
mixed-radix digits `(δ_1, …, δ_D)` over the extents `(b_i)`, innermost fastest,
and lane index `φ` has digits `(ε_1, …, ε_G)` over `(l_j)`. Lane `φ` of beat
`β` holds the element at

```
offset(β, φ) = Σ_i δ_i · s_i  +  Σ_j ε_j · t_j,        position = lin⁻¹(offset)
```

**Validity:** `Σ_i (b_i − 1)s_i + Σ_j (l_j − 1)t_j < |P|`, so every offset
names a position. Lane 0 is the least significant word position when a beat is
packed (§2.4).

*Example.* FINN's default order, `vector_major((2,4), 2)`, is `B = ((4, 2))`,
`L = ((2, 1))`:

```
beat 0: (0,0) (0,1)    beat 1: (0,2) (0,3)    beat 2: (1,0) (1,1)    beat 3: (1,2) (1,3)
```

A traversal is FINN's *folded shape*, generalized. Any axis may be split into
lanes, lanes may come from several axes, beats may walk in any order, and
positions may repeat.

### 2.2 Canonical form and equality

Construction rewrites a nest in two ways, and neither changes the sequence it
presents:

- drop a loop of extent 1;
- merge an outer loop `(b_o, s_o)` into the inner one `(b_i, s_i)` beside it
  when `s_o = b_i · s_i` (the two loops walk one contiguous run), giving
  `(b_o·b_i, s_i)`.

**Lemma (canonical form).** Two canonical traversals of one tensor are equal
exactly when they present the same sequence.

*Proof.* Merging preserves the sequence, so equal canonical forms present equal
sequences. Conversely, the beat and lane nests can be read off the sequence
separately: lane 0 across the beats gives the beat term alone, and beat 0
across the lanes gives the lane term alone, since the other term is zero at
digit 0. Take a canonical nest whose innermost loop is `(b, s)`, `b ≥ 2`. Its
offsets advance by `s` for `b − 1` steps. At step `b` the next loop
`(b_o, s_o)` advances by `s_o − (b − 1)s`, which equals `s` only if
`s_o = b·s`: the merge condition, excluded in canonical form. So the run of
equal increments ends at step exactly `b`, and `(b, s)` is determined. The
beats at multiples of `b` present the outer nest, and induction recovers it. ∎

`check_theory.py` keeps a seeded random sample as a regression check: 4,000
nests over a 12-element tensor (extents 1–4, strides 0–6, up to three beat
loops and two lane loops), 696 distinct sequences, each with exactly one
canonical form.

*Example.* `((2, 4), (2, 2))` merges to `((4, 2))`, so a row-by-row, two-chunk
walk equals `vector_major((2,4), 2)`.

### 2.3 Coverage

A traversal need not present every position, or each position once. It
**covers** its tensor when the set of offsets it presents is all of
`[0, |P|)`. Stride-0 loops (§3) and overlapping loops (sliding windows)
present some positions more than once.

### 2.4 Width and packing

A stream's word carries one beat. With element width `w` bits:

```
payload width  W = N_L · w
lane φ         occupies bits [φ·w, (φ+1)·w)          (lane 0 lowest; the physical layer's field)
AXIS carrier   8·⌈W / 8⌉                            (only the whole beat is padded)
```

This is FINN's *stream width* (`get_instream_width`) and its padded form.

### 2.5 Correspondence with layout algebra

A traversal is a **strided layout** in the sense of CuTe's layout algebra [L1]
(and of MLIR's `strided<>` memrefs, or AIE DMA buffer descriptors): a shape and
a stride per mode, mapping a coordinate to an index. Here the coordinate has two
modes, beat and lane, like a CuTe thread-value layout. The correspondences:

| Here | Layout algebra |
|---|---|
| canonical merge (§2.2) | *coalesce* (same condition, `s_o = b_i · s_i`) |
| stride-0 loop (§3) | a broadcast mode |
| aligned marker (§5.1) | `logical_divide` of the beat nest by a contiguous `n`-beat tile is admissible |
| reorder parameters (§6) | the composition `source‡ ∘ sink` restricted to one frame (below) |

For the marker row: `n | N_B` gives the complement `(N_B/n):n`, and
composition's shape divisibility gives the split; Cecka states the admissibility
condition of `logical_divide` as surjectivity [L1].

For the reorder row (`‡` a generalized inverse): the frame factors off the loops
the two nests share, the analogue of `logical_divide` by the shared modes, and
within it the coefficients index source *beats*, matched loop by loop after
common refinement, so the source's offset map is never inverted. CuTe's right
inverse would drop the source's stride-0 modes and choose one representative per
offset; matching loops keeps the source's own beat order instead. A sink that
replays is no obstacle either way, since the composition never inverts the sink.

One difference matters when translating: CuTe numbers coordinates
colexicographically (leftmost mode fastest); this model is row-major (innermost
last).

## 3. Stride zero: replay, repetition, broadcast

Every case in this section is one condition, `σ(i) = 0` (§4.2): the port does
not read index `i`: its projection onto the port's data space does not use `i`
(Timeloop [T1]); `i` lies in the null space of the access. The cases differ in how that is realized, and each gives a
different traversal. (The **moving** loop is the first beat loop with a nonzero
stride.)

| Realization | Here | Reuse taxonomy |
|---|---|---|
| across lanes: no lane loop | broadcast | spatial reuse, multicast |
| drop the loop (the port is outside it) | reduce (output), hold (input) | reduction; stationarity |
| stride-0 beat loop inside the moving loop | replay | temporal reuse, near |
| stride-0 beat loop outside it | repetition | temporal reuse, far (refetch) |

A stride-0 loop presents the same positions again. Where it sits decides what
it means:

- **Repetition:** a stride-0 beat loop *outside* the moving loop. The whole
  pass is presented again (`T.repeated(k)`).
- **Replay:** a stride-0 beat loop *inside* the moving loop. A group of beats
  is presented again before the walk moves on, e.g. each activation row once
  per output fold.
- **Broadcast:** an index with lanes that a port does not read (§4). Its lanes
  would all carry the same element, so no lane loop is created at all.

Two derived traversals follow:

```
unreplayed(T) = T without its replay loops          (repetition kept)
period(T)     = T without its repetition loops      (one pass)
```

*Example.* `vector_major((2,4),2).replayed(2, inner_beats=2)` is
`((2,4), (2,0), (2,2))`: each row's two beats, twice. `unreplayed` of it is
`((4,2))` again.

**Lemma (position is reuse distance).** In a canonical beat nest, a stride-0
loop inside the moving loop re-presents strictly fewer offsets than one pass,
and a stride-0 loop outside it re-presents exactly one pass.

*Proof.* A stride-0 loop re-presents the offsets of the loops nested inside it.
Every loop outside the moving loop has stride 0, so a pass is the moving loop and
everything inside it; a stride-0 loop outside the moving loop therefore
re-presents exactly a pass. Inside the moving loop `(b, s)`, the re-presented
offsets are a subset of `I`, the offsets of the loops inside `(b, s)`. Canonical
form gives `b ≥ 2`, and the moving loop's stride is `s > 0`, so the pass
contains `max(I) + s`, which is not in `I`. With lanes, the same holds for
element offsets (add the lane offsets to both sides; `max(I) + s + max(L)`
exceeds every element of `I + L`). ∎

So the positional rule is already an exact reuse-distance criterion, with the
threshold at one pass: *replay* is reuse whose footprint is smaller than a
pass, *repetition* is reuse of the whole pass. It needs no parameter.
`check_theory.py` asserts the lemma on the sampled canonical nests of §2.2
(116 replay loops, 100 repetition loops).

**Open: an absolute budget.** The one design question left is whether an
absolute buffer budget should ever override the rule: keep at the boundary a
replay whose footprint is too large for the receiver to buffer, or absorb a
repetition of a tiny operand (a broadcast channel vector, say). That override
is what would sit in tension with the boundary presentation being a rule, not a
parameter (§8). It is not settled here.

*Note.* "Repetition" here is unrelated to SDF's *repetitions vector* [S1],
though §7.1's pass count is an SDF-style balance equation.

## 4. Schedules and projection

### 4.1 Index spaces and folding factors

A **schedule** here is a *loop schedule*: an iteration order and its spatial
split, what Timeloop calls a *mapping* [T1]. It is not HLS scheduling (assigning
operations to clock cycles), which §0 places out of scope.

An operation is written over named **indices** (`m`, `n`, `k`), its *iteration
domain*. A schedule gives each index `i`:

- an extent `E_i`;
- a **folding factor** `F_i` dividing `E_i`: the lanes it spreads over each beat
  (code: `Schedule(factors=…)`, `Schedule.factor(i)`);
- its **fold** `S_i = E_i / F_i`: the beats it takes (FINN's neuron fold
  `NF = MH/PE` and synapse fold `SF = MW/SIMD`; code: `Schedule.steps(i)`);

and a **beat order** `π`, a permutation of the indices, outer to inner (code:
`Schedule(order=…)`). An index value splits as

```
i = i_b · F_i + i_ℓ         i_b ∈ [0, S_i) (its beat part),  i_ℓ ∈ [0, F_i) (its lane part)
```

and one **schedule step** is a digit vector `(i_b)` over `π`. There are
`Π S_i` steps. This is strip-mining (Halide's `split` [H1]) with the inner loop
spread over lanes. In `Y[m,n] = Σ_k X·W`, **PE** is `F_n` and **SIMD** is `F_k`.

*Terminology note.* This follows FINN, whose documentation calls PE and SIMD
"folding factors". In VLSI DSP usage [P1] the *folding factor* is instead the
number of operations time-multiplexed onto one unit, which is `S_i` here.
"Fold" alone always means `S_i` in this document.

### 4.2 Accesses

A port reads its tensor through an **access**: one linear expression per axis
(an *access function*, or a row of the *access matrix*, in the polyhedral
sense; the code's class is `Affine`, but it has no constant term) of a view
shape `V` (the tensor's own shape unless a row-major reshape is
declared, `Π V = Π E`). The code's `Access` is one port's tensor shape and its
expressions:

```
a_j = Σ_i c_{j,i} · i          c_{j,i} ≥ 0,  no constant term
```

With axis strides `σ_j = Π_{h>j} V_h`, each index has a **port stride**

```
σ(i) = Σ_j c_{j,i} · σ_j        how far one step of i moves this port's flat offset
```

`σ(i) = 0` means the port does not read `i`.

### 4.3 Projection (`present`)

A port's traversal is the schedule projected through its access:

```
beat loops:  (S_i, F_i · σ(i))   for each i in π, except those the port reduces or holds
lane loops:  (F_i, σ(i))          for each i the port carries as lanes, in its lane order
```

The projection is legal under three conditions:

- **Broadcast:** an index with lanes (`F_i > 1`) that the port does not carry
  as lanes must have `σ(i) = 0`. Otherwise elements would move within a beat
  without a lane to hold them.
- **Reduce:** an index the port *reduces* (an output presented after that
  index runs) must have `σ(i) = 0`.
- **Hold:** an index the port *holds* (an operand presented before that index
  runs, i.e. stationary across it) must have `σ(i) = 0`.

Dropping a reduced or held index leaves positions unchanged and only removes
beats: one beat per completed run.

**Agreement.** Fix a schedule step `(i_b)` and each index's lane digit `i_ℓ`,
so each index takes the value `i = i_b F_i + i_ℓ`. A port presents, at that step
and lane, the offset

```
Σ_i (i_b F_i + i_ℓ) · σ(i)  =  Σ_i i · σ(i)  =  Σ_j a_j σ_j  =  lin(a(i))
```

That is the position its access assigns to the index point. **Every port of one
schedule presents, at each step, the positions of one shared index point.**
So cross-port relations hold by construction and are never checked: a
weight beat follows its activation beat, and an output beat closes its own
reduction. This is why one schedule per kernel matters. Interfaces that each
chose their own order could drift apart.

*Example (dotp).* Take `m=2, n=2, k=4`, `F_k = 2` (SIMD 2), `F_n = 1` (PE 1),
order `m → n → k`, with `X[m,k]`, `W[k,n]` and `Y[m,n]` reducing `k`:

| Port | σ(m), σ(n), σ(k) | Beat loops | Lane loops | Reading |
|---|---|---|---|---|
| x | 4, 0, 1 | (2×4)(2×0)(2×2) | (2×1) | `n` unread and inside `m`: replay each row per output |
| w | 0, 1, 2 | (2×0)(2×1)(2×4) | (2×2) | `m` unread and outermost: repeat all weights per row |
| y | 2, 1, 0 | (4×1) | — | `k` reduced away: one beat per finished sum |

### 4.4 Extent binding

Extents are bound from the tensors, not declared (code: `bind_extents` over the
ports' `Access`es; a kernel reads the result as `Kernel.extents`). An axis
addressed by a plain index `a_j = i` gives `E_i = V_j`. The same index on two
axes must agree.

- An axis addressed by a linear sum of several indices binds nothing. Its
  indices take extents from another axis, or are given explicitly (code:
  `bound_schedule(extents=…)`), and the axis
  is checked: `Σ_i c_{j,i}(E_i − 1) < V_j`. (This is *bounds inference*, as in
  Halide [H1].)
- A port read through a view binds nothing. Its indices must be bound
  elsewhere, and `Π V = Π E` is checked.

**Coverage lemma.** If every axis of a port is addressed by a distinct plain
index bound to that axis's extent, the access is a bijection from those
indices' box onto the positions, so the port's traversal covers its tensor. It
visits each position once per step of the indices it does not read. A
tensor wider than the schedule addresses is then refused, not silently
half-read.

## 5. Markers, repetition and beat sequences

### 5.1 Level-end markers

A marker `LevelEnd(n)` is asserted on beat `β` exactly when `(β + 1) mod n = 0`:
the last beat of every group of `n`. AXIS `TLAST` and a loop-completion bit
(`olst[d]`) are both such markers, as are Tydi's per-dimension end-of-sequence signals [D1]
and the Sparse Abstract Machine's stop tokens [D2].

A marker must close a loop level of its traversal. Write the beat extents inner
to outer as `b_D, b_{D−1}, …`. `LevelEnd(n)` is **aligned** iff `n` divides
`N_B` and, for some `k`,

```
Π_{i>k} b_i  divides n     and     n / Π_{i>k} b_i  divides b_k
```

That is, `n` spans whole inner loops and then a whole divisor of the next one;
the loop can be split there. `check_theory.py` confirms this characterization
against the code on 8,346 (traversal, `n`) pairs.

A reduction's marker comes from the schedule: `closing(R) = LevelEnd(Π_{i∈R} S_i)`,
defined only when the reduced indices `R` are the innermost of `π`. If they
are not, the reduction's outputs interleave, and several partial sums must stay
open at once.

*Example.* dotp's `closing(k)` is `LevelEnd(2)`: with SIMD 2 over `K = 4`, one
reduction is two beats, so TLAST falls on every second beat.

### 5.2 Repetition and beat sequences

A producer presents its pass `ONCE`, or `CYCLIC` (forever, as a read-only
memory does). A **beat sequence** (Tydi would say *stream type* [D1]) is one
interface's complete presentation:

```
BeatSequence = (traversal, repetition ∈ {ONCE, CYCLIC}, markers ⊂ aligned LevelEnds)
```

A producer's markers are guarantees; a consumer's are requirements.

## 6. Comparing two presentations

`classify(A, B)` names what turns one traversal of a tensor into another of
the same shape. It checks in order:

| Verdict | Condition | Realized by |
|---|---|---|
| **identity** | `A = B` | wires |
| **lane permutation** (a spatial permutation [R1]) | same beat loops, same lane count, same multiset of lane offsets | wires (lane order only) |
| **reorder** (a temporal permutation [R1], generalized to duplicating maps: replay) | same lane loops; after common refinement of the beat nests, B's loops are A's plus new stride-0 loops | a buffer: `input_gen` |
| **width conversion** | the flattened nests (beats then lanes) are equal | `vpc` (a data width converter) |
| **lane regroup** (a spatio-temporal permutation [R1]) | the flattened nests are equal as multisets after common refinement | a regroup (§7) |
| **incompatible** | otherwise | nothing |

**Common refinement** splits every loop of both nests at every boundary
`{s, s·b}` of either, when the split is exact. Two nests are then comparable
loop by loop.

**Reorder parameters.** After refinement, drop the shared outer prefix. The
rest of A spans a **frame** of `f = Π` (its extents) beats. For each of B's
remaining loops, its **coefficient** is the beat stride, within A's frame, of
the matching A loop, or 0 for a new stride-0 loop. B's beat at digits `(d_k)`
is then A's frame beat `Σ d_k · coef_k`. These are exactly `input_gen`'s
`FM_SIZE` (the frame), `DIMS` and `COEFS`. This is the composition
`source‡ ∘ sink` of §2.5, restricted to one frame; the frame is also the reorder's buffer
footprint, the reuse distance of §3.

*Examples.* All on the `(2, 4)` tensor unless noted:

- row-major, 1 lane → columns first, 1 lane: **reorder**, frame 8, dims
  `(4, 2)`, coefs `(1, 4)`. Column `j`, row `i` is input beat `j + 4i`.
- the unreplayed dotp `x` → dotp `x`: **reorder**, frame 2, dims `(2, 2)`,
  coefs `(0, 1)`: each two-beat row, twice.
- lanes `(n, k)` → lanes `(k, n)` of a 2×2 tile: **lane permutation**
  `(0, 2, 1, 3)` (sink lane → source lane).
- `(4,4)` row-major with 2 lanes → columns with 2 row-lanes: **lane regroup**
  (a banked transpose).
- a 3-tap, stride-1 sliding window over 6 inputs (`oh + kh`): **incompatible**.
  It has the same positions but multiplicities from overlap rather than from
  stride 0, which the model does not yet name.

## 7. Plans

### 7.1 Unrolling a cyclic source

A cyclic source is presented over one pass of its consumer. It is repeated
`c = (N_B N_L)_sink / (N_B N_L)_source` times, which must be a whole number,
so lanes may differ. This is an SDF balance equation [S1] with the pass as the
firing. A `ONCE` source cannot feed a `CYCLIC` consumer: nothing
captures a pass to replay it forever.

### 7.2 The canonical chain

`plan(source, sink)` is the canonical chain of steps turning the source's beat
sequence into the sink's. The sequence steps are **R** (reorder), **W** (width)
and, last, **M** (marker synthesis):

```
ε  |  R  |  W  |  W·R  |  R·W  |  W·R·W        each optionally followed by M
```

- Lanes and order both differ: `W` then `R` (at the sink's lanes) or `R` then
  `W` (at the source's), whichever classifies.
- A **lane regroup** goes through the common lane count `g = gcd`: `W` to `g`
  lanes, where every permutation is a reorder, then `R`, then `W` to the sink's
  lanes. This is always correct but slow when `g` is small: at `g = 1` the
  stream is serialized to one element per beat. The known alternative realizes
  a streamed permutation at full width without serializing: a spatial network,
  banked RAMs, a spatial network [R1]. FinnLib's `inner_shuffle` is such a
  banked construction for one regroup shape. Its defect under bursty input is
  fixed (FinnLib `99d75e8`), but it is not yet a candidate for a stream's
  adapter: a composite places it explicitly.
- **Markers last.** Every sequence step invalidates the markers before it, so
  a marker the sink requires is synthesized after the last step. Hence `M` only
  at the end.

`plan` refuses (`Unrealizable`) different positions, a non-whole cyclic
repetition, and `ONCE` → `CYCLIC`. An empty plan means the ends connect
directly.

*Examples.* To dotp's `x` (2 lanes, TLAST every 2):

- from 2-lane row-major: `R·M`;
- from 1-lane or 4-lane row-major: `W·R·M`;
- from a cyclic 4-lane source to a 2-lane consumer of two passes: `W`.

On a `(4, 6)` tensor, a 3-lane row-major source whose sink lanes form 2×2
tiles regroups as `W·R·W` (`g = gcd(3, 4) = 1`).

Realization maps each step to a module: `R` is an `input_gen` (frame, dims,
coefs); `W` a `vpc`; `M` merges into a preceding `R`'s `input_gen`, or is an
identity `input_gen` of its own. The plan shapes then fall onto exactly seven
module chains, so a stream's adapter has exactly one candidate among today's
chains for any realizable plan. That candidate is correct, not necessarily the
cheapest (§10).

## 8. Streams, boundaries and composition

A **stream** is one tensor between one producer interface and one consumer
interface. It derives its plan from the two beat sequences and places the
chain that carries it out. It refuses a plan it cannot carry out, and any plan
at all where adapters are forbidden.

**Element.** Each end carries the tensor's encoding. A producer states its own
(never computed from its own output stream), and the stream refuses a
mismatch. Converting values is computation, not adaptation.

**The boundary rule.** Where a stream leaves a composite module, the boundary
presents, for an input, `unreplayed` of its consumer's traversal, and for an
output, its producer's traversal. Neither carries markers, and both are `ONCE`.
The receiver realizes its own replay; repetition stays part of the interface.
This rule is the model's own; the nearest precedent is interface abstraction for
hierarchical dataflow actors [S2]. Finding the plan between two interfaces, or
proving none exists, is the converter-synthesis problem [C1] in a restricted
form.

**Chained plans are not fused.** Chaining is closed: two plans in sequence
compose as functions from sink beat to source beat, and the result is correct.
What the model lacks is *fusing* a chain into the plan of the whole: `plan(A, C)`
need not equal `plan(A, B)·plan(B, C)` step for step. A columns-first producer
feeding a composite through its boundary gets two reorders (to the boundary,
then the boundary's replay), where `plan(producer, consumer)` needs one. Fusing
adjacent reorders is layout composition (§2.5), and it is admissible under
divisibility conditions (CuTe [L1]; power-of-two linear layouts [L2]); the
model does not do it yet. This is a representability question about orders. It
is distinct from the open problem of composing *timing* characterizations, which
this model does not address (§0).

## 9. Correspondence with FINN

| FINN | This model |
|---|---|
| normal shape, datatype | tensor: shape and scalar encoding |
| folded shape `(…, fold, PE)` | a traversal: `vector_major` (§2.1) is FINN's special case; FINN's fold is `S_i` |
| PE, SIMD ("folding factors") | folding factors of indices: `F_n`, `F_k` (§4.1) |
| neuron fold NF, synapse fold SF | folds `S_n`, `S_k` (§4.1) |
| `get_instream_width[_padded]` | `N_L · w`, carrier `8⌈N_L w / 8⌉` (§2.4) |
| MVAU's internal input buffer (replay) | a stride-0 loop in the consumer's traversal (§3), realized on the stream |
| external weight stream (repeated per input vector) | weights' repetition loop, kept at the boundary (§8) |
| `get_exp_cycles` | schedule steps `Π S_i`, at one beat per cycle |
| `MH % PE == 0`, `MW % SIMD == 0` | a folding factor divides its extent (§4.1) |
| `InsertDWC` | a `W` step (§7) |
| "SWG SIMD must equal VVAU PE, or the result is wrong" | the stream's plan: direct if equal, `W` or a refusal if not; a lane-order mismatch is a lane permutation (wires) |

## 10. What the model cannot express

- **Offsets and signs:** constant terms, padding, crops, flips. Every traversal
  starts at offset 0, and all strides and coefficients are non-negative.
- **Folds that do not divide:** ragged last beats, logical padding.
- **Data-dependent order:** gathers, sparsity, runtime-selected weight sets
  (these are visible only as a count of passes).
- **Overlap as a named relation:** sliding windows are expressible as
  traversals, but `classify` calls them incompatible.
- **Markers beyond periodic level ends:** first-beat markers, sidebands; and
  more than one marker per AXIS interface.
- **Cost:** buffer sizes, cycles under stalls, a choice among chains (e.g. a
  full-width permutation instead of a regroup through `g` lanes, §7.2).
- **Fusion:** fusing chained plans into one (§8).

## Appendix: notation

| Symbol | Meaning |
|---|---|
| `E`, `P`, `lin` | tensor shape, positions, row-major flat offset |
| `B`, `L`; `(b, s)`, `(l, t)` | beat and lane loop nests; a loop's extent and stride |
| `N_B`, `N_L` | beats per pass, lanes per beat |
| `E_i`, `F_i`, `S_i`, `π` | index extent, folding factor (lanes), fold (beats), beat order |
| `a_j`, `c_{j,i}`, `σ_j`, `σ(i)` | access expression, coefficient, axis stride, port stride |
| `LevelEnd(n)` | marker on every `n`-th beat |
| `R`, `W`, `M` | reorder, width, marker-synthesis steps |

## References

Verified 2026-09-29 (venue, year, and the specific claim each is cited for;
re-checked in review round 2, `THEORY-REVIEW.md` §6).

- **[C1]** R. Passerone, L. de Alfaro, T. A. Henzinger, A. Sangiovanni-Vincentelli,
  "Convertibility Verification and Converter Synthesis: Two Faces of the Same
  Coin," ICCAD 2002, pp. 132–139.
- **[D1]** J. Peltenburg, J. van Straten, M. Brobbel, Z. Al-Ars, H. P. Hofstee,
  "Tydi: An Open Specification for Complex Data Structures Over Hardware
  Streams," IEEE Micro 40(4), 2020, pp. 120–130; and the Tydi physical-stream
  specification (one `last` bit per nesting dimension; "each replication is
  called a lane").
- **[D2]** O. Hsu, M. Strange, R. Sharma, J. Won, K. Olukotun, J. Emer,
  M. Horowitz, F. Kjølstad, "The Sparse Abstract Machine," ASPLOS 2023,
  pp. 710–726.
- **[H1]** J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand,
  S. Amarasinghe, "Halide: A Language and Compiler for Optimizing Parallelism,
  Locality, and Recomputation in Image Processing Pipelines," PLDI 2013.
- **[L1]** NVIDIA CUTLASS, "CuTe Layout Algebra" (`media/docs/cpp/cute/02_layout_algebra.md`);
  C. Cecka, "CuTe Layout Representation and Algebra," arXiv:2603.02298, v1
  2 March 2026 (v2 29 July 2026).
- **[L2]** K. Zhou et al., "Linear Layouts: Robust Code Generation of Efficient
  Tensor Computation Using 𝔽₂," ASPLOS '26, DOI 10.1145/3760250.3762221
  (arXiv:2505.23819).
- **[P1]** K. K. Parhi, *VLSI Digital Signal Processing Systems: Design and
  Implementation*, Wiley, 1999 (folding, the folding factor).
- **[R1]** M. Püschel, P. A. Milder, J. C. Hoe, "Permuting Streaming Data Using
  RAMs," Journal of the ACM 56(2), Art. 10, 2009.
- **[S1]** E. A. Lee, D. G. Messerschmitt, "Synchronous Data Flow," Proceedings
  of the IEEE 75(9), 1987, pp. 1235–1245.
- **[S2]** S. Tripakis, D. Bui, M. Geilen, B. Rodiers, E. A. Lee,
  "Compositionality in Synchronous Data Flow: Modular Code Generation from
  Hierarchical SDF Graphs," ACM TECS 12(3), Art. 83, 2013.
- **[T1]** A. Parashar et al., "Timeloop: A Systematic Approach to DNN
  Accelerator Evaluation," ISPASS 2019.
