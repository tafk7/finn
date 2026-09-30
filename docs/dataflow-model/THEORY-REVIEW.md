# THEORY.md against established theory: review

Date: 2026-09-29. Reviews [`THEORY.md`](THEORY.md) as of `78a574dd4`. Read-only:
nothing here has been checked against the code beyond what `THEORY.md` itself
asserts.

## Summary

The mathematics is sound, and almost every object in it already has a name in
one of five bodies of work: strided/CuTe layouts, Timeloop-style mappings,
polyhedral/einsum access functions, Püschel's streaming permutations, and
Tydi/SAM stream types. The findings fall into three groups:

- **Four terminology collisions**, one serious: `fold` names the opposite
  factor from FINN's own code and from the VLSI literature (§2.1).
- **Three places that build bespoke machinery** where a standard framework
  exists and is in some cases stronger (§3).
- **One overclaim**: §8's non-closure is layout-composition representability,
  not the open scheduling-composition problem (§3.3).

**Recommendation.** State §2 as "a traversal is a CuTe layout over (beat, lane)
coordinates" and adopt that layout algebra (coalesce, composition, divide). It
absorbs §2.2, the §5.1 alignment condition, and most of §8's reorder question
into named operations with known theorems. Fix the fold/unroll inversion before
it spreads further into `finn.kernels`.

## 1. Correspondence

| THEORY.md | Settled name | Source |
|---|---|---|
| traversal `(E, B, L)`, loops of (extent, stride) | **strided layout** (shape:stride); beats × lanes is a CuTe **TV (thread-value) layout** | CuTe/CUTLASS layout algebra; MLIR `strided<>` memref; AIE DMA buffer descriptors (wrap/stride per dimension) |
| canonical form (§2.2) | **coalesce**: CuTe's merge rule is the same `s_o = b_i·s_i` | CuTe |
| stride-0 loop | **broadcast stride** | numpy `broadcast_to`, CuTe |
| `rank(p)`, flat offset | **linearization**, `ravel_multi_index` | — |
| index space, schedule step, `Π S_i` | **iteration domain**, iteration point, trip count | polyhedral model |
| `i = i_b·F_i + i_ℓ` | **strip-mining** (Halide `split`), inner loop vectorized | Halide, polyhedral |
| schedule = folds + beat order `π` | **mapping**: spatial factors, temporal factors, loop permutation; **space-time mapping** | Timeloop, MAESTRO; systolic synthesis (Moldovan, Quinton); LSGP/LPGS partitioning (Teich & Thiele) |
| access `a_j = Σ c_{j,i}·i` | **access function / access matrix** | polyhedral; einsum |
| projection, `present` | **projection** of the iteration space onto a data space | Timeloop, same word |
| `σ(i) = 0` | **irrelevant dimension**; `i` in the null space of the access matrix | Timeloop; polyhedral reuse analysis |
| replay, repetition, broadcast, hold, reduce | **temporal reuse** at a storage level, **spatial reuse (multicast)**, **stationarity**, **reduction index** | Eyeriss/Timeloop reuse taxonomy; einsum |
| Agreement (§4.3) | follows from one iteration domain under one schedule | polyhedral, systolic: by construction |
| extent binding (§4.4) | **range / bounds inference** | Tensor Comprehensions, Halide |
| `LevelEnd(n)`, markers | per-dimension **`last`** bits; **stop tokens** | Tydi (`last[d]`; Tydi also says "lanes"); SAM, ASPLOS'23 (stop tokens `S0, S1, …`) |
| aligned marker (§5.1) | a **tile boundary**; the divisibility condition of CuTe `logical_divide` | CuTe |
| beat sequence | **stream type** | Tydi |
| lane permutation / reorder / lane regroup | **spatial / temporal / spatio-temporal streaming permutation** | Püschel, Milder, Hoe, *Permuting streaming data using RAMs*, JACM 2009 |
| width conversion | **data width converter**; a **gearbox** when the ratio is not an integer | AXIS DWC IP |
| plan, `Unrealizable` | **converter synthesis**, not convertible | Passerone et al., ICCAD 2002 |
| boundary rule, composite interface | interface abstraction of a hierarchical actor | Tripakis et al., *Compositionality in SDF*, TECS 2013 |

## 2. Terminology collisions

### 2.1 `fold` names the wrong factor (serious)

§4.1 calls `F_i`, the lanes per beat, the *fold*, and says "PE is `F_n`". In
Parhi's VLSI DSP folding, the folding factor is the time-multiplexing count. In
FINN's code, `MH/PE` is the *neuron fold* (NF) and `MW/SIMD` the *synapse fold*
(SF), and the folded shape is `(…, fold, PE)`. Both are this document's `S_i`.

THEORY.md contradicts itself. §9's "folded shape `(…, fold, PE)`" uses fold to
mean `S_i`, and §4.1 uses it to mean `F_i`. The confusion comes from FINN's own
prose, which calls PE and SIMD "folding factors" while its code names the
quotient the fold.

**Change:** `F_i` → *unroll* (or *parallelism*, *spatial factor*). Let `S_i` be
the *fold* (or *temporal factor*). This has to happen before the name spreads
into kernel authoring.

### 2.2 "affine" access with no constant term

§4.2 writes `a_j = Σ_i c_{j,i}·i`, with no constant term, and §10 confirms that
offsets are excluded. That is a *linear* access. **Change:** "linear access",
"access matrix".

### 2.3 "schedule"

To a hardware reader, scheduling is HLS's assignment of operations to clock
cycles, which is exactly what §0 says the model does not do. **Change:**
"mapping" (Timeloop), or "loop schedule" at first use with a sentence that
separates it from HLS scheduling.

### 2.4 "repetition"

This collides with SDF's *repetitions vector*. The collision is worse because
§7.1's `c = (N_B N_L)_sink / (N_B N_L)_source` is effectively an SDF balance
equation. **Change:** "re-presentation", "pass count", or cite SDF and keep the
word deliberately.

### 2.5 Minor

| Term | Problem | Suggestion |
|---|---|---|
| `rank(p)` | tensor rank (`r`) appears in the same sentence | `lin(p)` or "linear index" |
| field / lane | two words for one thing | keep *lane*; *lane index* for `φ` |
| pass / frame | different things; *frame* is also `input_gen`'s feature map (`FM_SIZE`) | keep *pass*; rename the reorder *frame* to *window* or *footprint* |
| hold | reads as hold time | *stationary* |
| traversal | suggests graph traversal | *stream layout*, or *layout* once §2 is framed as a CuTe layout |
| moving loop | ad hoc | goes away under §3.1 |

**Keep as they are:** *projection* and *common refinement* are already the
standard terms. *Beat* is ubiquitous in AXI, though the AXIS specification says
*transfer*.

## 3. Bespoke machinery

### 3.1 Five names for one condition

Broadcast, reduce, hold, replay and repetition are all `σ(i) = 0`: index `i`
is irrelevant to the port. They differ only in where that is realized:

| Realization | THEORY.md | Standard |
|---|---|---|
| across lanes | broadcast | multicast, spatial reuse |
| drop the loop (port outside it) | reduce (output), hold (input) | reduction, stationarity |
| stride-0 beat loop inside the reuse window | replay | temporal reuse at a near buffer |
| stride-0 beat loop outside it | repetition | temporal reuse at a far level; refetch |

The standard framing, an irrelevant index plus the storage level that captures
its reuse, also exposes a defect in §3. Whether a stride-0 loop is replay or
repetition is decided syntactically, by its position relative to the "moving
loop". The meaningful criterion is **reuse distance**: how large a buffer
realizes it. `input_gen`'s frame `f` in §6 already is that reuse footprint.
This is the "replay/repetition split is syntactic" gap from the 2026-09-28
model review, and the reuse-analysis literature gives it a quantitative
criterion to replace the positional one.

### 3.2 §2.2's invariant can be proved rather than sampled

The claim that canonical traversals are equal exactly when they present the
same sequence is supported by 4,000 random samples. It has a short proof.

Take a canonical nest whose innermost loop is `(b, s)` with `b ≥ 2`. The
presented sequence advances by `s` for `b − 1` steps. At step `b` the next
loop `(b_o, s_o)` advances it by `s_o − (b − 1)s`, and that equals `s` exactly
when `s_o = b·s`, which is the merge condition. A canonical nest therefore
breaks its constant run at step `b`, so `(b, s)` is read off the sequence. The
beats at multiples of `b` present the outer nest, and induction recovers it.
The beat nest and lane nest are recovered separately: field 0 across beats
gives `B`, and beat 0 across fields gives `L`, because both offset terms are
zero at digit 0.

**Change:** state it as a lemma with this proof, and keep `check_theory.py`'s
sample as a regression check.

### 3.3 §8 "composition is not closed" is an overclaim

As maps from sink beat to source beat, reorders compose by function
composition. The concept is closed. What can fail is whether the composite can
be *represented* as a single `input_gen`, one `(FM_SIZE, DIMS, COEFS)`. That is
**layout-composition admissibility**:

- CuTe composition is defined under divisibility conditions;
- Triton's F₂ linear layouts are closed for power-of-two sizes.

The §8 example (producer → boundary → replay) does fuse to one `R`, and
`plan(producer, consumer)` shows it. What the model lacks is peephole fusion of
adjacent `R` steps by layout composition, when that composition is admissible.

The scheduling literature's open problem is composing *timing*
characterizations, which is a different object. **Change:** reword §8 and §10
to say "chained plans are not fused; fusing adjacent reorders is layout
composition, which is admissible under divisibility conditions." Keep the
scheduling claim only for timing.

### 3.4 The lane regroup is weaker than known constructions

§7.2 regroups through `g = gcd` lanes. When `g = 1`, as in the `(4, 6)` example
with `gcd(3, 4) = 1`, the stream is serialized to one element per beat. Püschel
et al. realize arbitrary streamed permutations at a fixed width `p` without
serializing: a spatial network, then `p` banked RAMs, then a spatial network.

§10 excludes cost, but §7.2 also claims that each realizable plan has exactly
one module chain. The chain it picks is the slow one. **Change:** at minimum,
cite the spatial-temporal-spatial construction as the known alternative and
note the throughput loss at small `g`.

## 4. What stays

- The `classify` table is a good compact taxonomy. Relabel it with the
  spatial/temporal permutation names and cite Püschel.
- The **boundary rule** (§8: the producer presents its native order, the
  receiver adapts) is the project-specific contribution. The nearest precedent
  to cite is Tripakis et al. on interface abstraction of hierarchical SDF
  actors.
- The §9 FINN correspondence table is useful as it stands, apart from the
  `fold` row, which should follow §2.1.

## 5. Change list

| # | Section | Change | Weight |
|---|---|---|---|
| 1 | §4.1, §9, appendix | `F_i` → unroll; `S_i` → fold | high |
| 2 | §2 | frame traversals as CuTe layouts over (beat, lane); adopt coalesce / composition / divide | high |
| 3 | §3, §8 | replay vs repetition by reuse distance, not loop position | high |
| 4 | §8, §10 | reword non-closure as layout-composition admissibility | medium |
| 5 | §2.2 | replace the sampled invariant with the proof in §3.2 | medium |
| 6 | §4.2 | affine → linear | medium |
| 7 | §0, §4.1 | "schedule" → "mapping", or disambiguate from HLS | medium |
| 8 | §3, §5.2, §7.1 | "repetition" vs SDF repetitions vector | low |
| 9 | §7.2 | note the spatial-temporal-spatial alternative to regrouping through gcd | low |
| 10 | throughout | minor renames (§2.5) | low |
| 11 | §1, §2, §5, §6 | cite the precedents in §1 at first use of each term | low |

## 6. Round 2: the revision and the author's pushback

Re-read `THEORY.md` after the author's revision (same day). The revision
adopts most of the change list. This section records what happened to each item,
which pushback points stand, and what the revision still gets wrong. It is backed
by two research passes: CuTe semantics from the CUTLASS source and docs and from
Cecka's paper, and a check of every reference.

### 6.1 Pushback: all five points accepted

| # | Review said | Pushback | Disposition |
|---|---|---|---|
| 1 | rename `F_i` → unroll | FINN's docs (`tutorials.rst`: "PE & SIMD, also called folding factors"), team vocabulary and `Schedule(folds=…)` all say *folding factor* | **Accepted.** The revision's scheme is sound: `F_i` is the folding factor, `S_i` the fold (NF/SF), a bare "fold" never means `F_i`, and a note gives the VLSI usage. §9 is now consistent. Renaming the code's `folds=` is deferred to A4. |
| 2 | traversal → layout | `finn.kernels.physical.layout.PackedBeatLayout` already uses "layout" for bit packing (§2.4) | **Accepted.** Keep *traversal*; cite CuTe. |
| 3 | replay vs repetition by reuse distance | this is a behavior change to the boundary rule, not documentation | **Accepted**, recorded as an open question in §3. §6.3 below sharpens it. |
| 4 | adopt CuTe's algebra wholesale | correspondence yes, restructuring no; CuTe is colexicographic | **Accepted.** §2.5 is the right size. Its reasoning needs two corrections (§6.2). |
| 5 | "frame" collides with `FM_SIZE` | `Reorder.frame_beats` *is* `input_gen`'s `FM_SIZE` | **Accepted.** This was my error. |

### 6.2 Corrections to the revision

**§2.5, the reason for the frame, is wrong.** THEORY.md says replay makes a
layout non-injective, "which is exactly where layout inverses and complements
stop applying, and why §6 works with a frame". CuTe does not stop there:

- `right_inverse` is defined for layouts with stride-0 modes and drops them.
  Cecka's Table 5 gives `((2,2),(2,4)):((0,1),(0,2))` → `(2,4):(2,8)` with the
  comment "Stride-0 modes do not contribute".
- `left_inverse` explicitly skips stride-0 modes (`layout.hpp`).
- The public `complement` calls `filter()` first, which strips stride-0 and
  size-1 modes. Only the internal routine asserts "Non-injective Layout
  detected".

A *sink* that replays is no obstacle in any case, because composing
`source‡ ∘ sink` never inverts the sink. The code shows the frame's actual
reasons (`Reorder`, `traversal.py:273`), and they are two separate things:

1. **Beat-space indexing.** The coefficients address source *beats*, matched
   loop by loop after common refinement, so the source's offset map is never
   inverted. This is how a source with its own replay loops is handled.
2. **Factoring off the shared outer loops.** Dropping the common outer prefix
   is the analogue of `logical_divide` by the shared modes. The frame is what
   remains, and it is the buffer footprint.

Suggested text: *"The reorder is the composition `source‡ ∘ sink` restricted to
one frame. The frame factors off the loops the two nests share (the analogue of
`logical_divide`), and within it the coefficients index source beats, so the
source's offset map is never inverted. CuTe's right inverse would drop the
source's stride-0 modes and choose one representative per offset; matching
loops keeps the source's own beat order instead."*

**§2.5, the aligned-marker row, is imprecise.** Cecka states logical_divide's
requirement as *surjectivity* of `(B, complement(B))` onto `Z_|A|` (§3.5,
Eq. 30). Composition's own conditions are stride divisibility and shape
divisibility (§3.3.3, Eqs. 20–21). The accurate correspondence is that
`LevelEnd(n)` is aligned iff `logical_divide(beats, n:1)` is admissible:
`n | N_B` gives the complement `(N_B/n):n`, and shape divisibility gives the
split. Suggested row: *"aligned marker (§5.1) | `logical_divide` of the beat
nest by a contiguous `n`-beat tile is admissible"*.

**§6, reorder row.** "A temporal permutation [R1], replay included":
Püschel et al. treat bijections only. Say *"a temporal permutation [R1],
generalized to duplicating maps (replay)"*.

**References.**

| Ref | Finding |
|---|---|
| [L1] Cecka | **Wrong year.** arXiv:2603.02298, v1 2 Mar 2026 (v2 29 Jul 2026), not 2024. |
| [D1] Tydi | The hedge can go. The physical-stream spec: `last` carries one bit per nesting dimension `D` ("to stream two-dimensional sequences, two `last` bits are needed"), replicated per lane (`N·D`) at the highest complexity; "each replication is called a lane". |
| [L2] Linear Layouts | Confirmed at ASPLOS '26 (DOI 10.1145/3760250.3762221). |
| [C1], [D2] | Confirmed. Page ranges come from secondary listings only; the ACM DL returned 403. |
| [H1], [P1], [R1], [S1], [S2], [T1] | Confirmed from primary sources. Parhi 6.2: "N … is also referred to as the folding factor"; Timeloop "projecting the 7D operation points into the 4D dataspace dimensions" and "mapping"; Püschel "decomposing a given permutation into a sequence of three permutations that are either temporal or spatial" at full streaming width. |

My own review had Passerone as DAC'02; it is ICCAD 2002 (now fixed in §1).

### 6.3 The replay/repetition question is narrower than §3 says

THEORY.md's open paragraph says a reuse-distance criterion "needs a threshold, a
buffer budget". It doesn't: the positional rule already *is* an exact
reuse-distance criterion, with the threshold at one pass.

**Claim.** In a canonical beat nest, a stride-0 loop inside the moving loop
re-presents strictly fewer offsets than one pass, and a stride-0 loop outside it
re-presents exactly one pass.

*Proof.* A stride-0 loop re-presents the offsets of the loops nested inside it.
Outside the moving loop, that is the whole pass. Inside it, those offsets are a
subset of `I`, the set of offsets presented by the loops inside the moving loop
`(b, s)`. Canonical form gives `b ≥ 2`, and the moving loop's stride is positive
(`s > 0`), so the pass contains `max(I) + s ∉ I`. Hence the footprint is
strictly smaller than the pass. ∎

So "replay" means reuse whose footprint is smaller than a pass, and
"repetition" means reuse of the whole pass. That is a rule with no parameter,
and it matches `unreplayed`/`period` in `traversal.py:420–436` exactly. The only
real design question left is whether an *absolute* budget should ever override
it: for example, keep at the boundary a replay whose footprint is too large for
the receiver to buffer, or absorb a repetition of a tiny operand. That override
is the part in tension with "boundary presentation is a rule, not a parameter".
Suggested replacement for the §3 paragraph: state the claim above, then give the
absolute-budget override as the open question.

### 6.4 Still open from §5

- Change 7 is done: "schedule" is disambiguated as a loop schedule, what
  Timeloop calls a mapping.
- Change 6 is done: the doc says "linear"; the code's class `Affine` is noted.
- Change 9 is done: §7.2 cites the spatial-temporal-spatial construction and
  notes that FinnLib's `inner_shuffle` is one such construction but is unused
  because of the bursty-input defect.
- Nothing from the change list is outstanding beyond the corrections in §6.2
  and the sharpening in §6.3.
