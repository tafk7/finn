# Parameter-Delivery Design-Space Map (declarative)

*The complete design space of "how parameters reach a compute core" — weight/threshold
delivery, memory access, and Multi-Layer Offload (MLO) — extracted as pure data divorced
from the passes that consume it. Produced 2026-07-17 as the composition acid-test, the
analogue of [[mvau-design-space]] but for the DELIVERY half that MVAU-emit deliberately
deferred (the "compute half only" scope). This is the ANALYSIS artifact for the kernel
COMPOSITION arc; it maps the space, it does not yet map onto the resolve engine.*

Files (ground truth, file:line throughout):
- **RTL** = `finn/finn-rtllib/{memstream,fetch_weights,mvu_tiled,mlo,stream_tap,thresholding}/`
- **finnloop** = `finn/src/finn/custom_op/fpgadataflow/rtl/finn_loop.py`
- **base** = `.../custom_op/fpgadataflow/matrixvectoractivation.py` (`generate_infra_hdl`,
  `code_generation_ipi`), **hwc** = `hwcustomop.py` (`generate_hdl_{memstream,fetch_weights,
  dynload}`), **roll** = `.../transformation/fpgadataflow/loop_rolling.py`
- **proto** = `finn-proto/.../kernels/{mvau/sip,memstream/rtl}` (the `MVAUSIP` composite)
- **ideal** = `../finn-hw-backend-analysis/kernel-model.md` (the to-be design; T3/T6 rulings)
- **doc** = the MLO design doc (`tmp-mlo.md` at repo root, WIP but conceptually sound)

---

## 0. Why this space is separate from the compute space

The MVAU/VVAU/Thresholding design spaces we already modelled ([[mvau-design-space]],
[[impl-bundles]]) are the COMPUTE core: PE×SIMD folding, dtypes, DSP microarch, the
implementation pool. Parameter delivery is a **different, orthogonal space** — FINN's own
stress-test #4 ("a third axis owned by neither compute backend", [[mvau-design-space]] §6).
It is the one thesis claim still unproven: **composition** (a compute kernel co-existing
with a delivery kernel), and the piece FINN was structurally worst at (the `mem_mode`
if/elif scattered across ~16 methods, the 236-line `code_generation_ipi` god-method).

**Central design goal (user, 2026-07-17):** *be able to change a kernel's memory access
cleanly.* FINN's generics are the anti-pattern to improve on, not to reproduce — `mem_mode`
fuses independent concerns into one string switch, and `mlo_max_iter` bolts MLO on as a
parallel flag threaded through a dozen sites. The whole point of this effort is to pull
those apart.

---

## 1. The three orthogonal coordinates

"How parameters reach compute" decomposes into **three independently-variable coordinates**.
This is the core claim of this document; §5 stress-tests it, §2-3 ground it in RTL.

### Coordinate A — STORAGE / TOPOLOGY (where params live + how they reach compute)

| Value | Where params live | How stored | How they reach compute | RTL |
|---|---|---|---|---|
| `embedded` | in the compute core | LUTROM / compiled-in | no port (params.h array) | base `internal_embedded` |
| `on-chip-replay` | on-chip RAM | BRAM / URAM | streamer replays onto weight AXIS | memstream (RTL memstream/) |
| `off-chip-DMA` | off-chip | DDR / HBM | DMA master fetches, on-chip double-buffer | fetch_weights (RTL fetch_weights/) |
| `dynamic-stream` | outside the node | — | arrives on a top-level weight AXIS | base `dynamic` / `external` |
| `generated` *(novel)* | nowhere | procedural | a generator core emits the stream | — (see §5 Case C) |

Sub-properties of this coordinate (NOT separate axes): `ram_style {auto,block,distributed,
ultra}`, `pumpedMemory` (clk2x memory), double-buffering (latency-hiding prefetch), and
**fetch-vs-direct-address** (external/BRAM need a fetch launched by the index; in-core RAM
is directly addressable — see §3). The enum is **OPEN**, not FINN's fixed `mem_mode`.

### Coordinate B — SELECTION-CARDINALITY (one param set, or an index over N)

| Value | Meaning | RTL realization |
|---|---|---|
| `single` (k≡0) | one fixed param set, no runtime selection | `NO_MLO`: index port tied to const-0 |
| `indexed` (N sets) | a runtime index stream selects set `k∈[0,N)` per iteration | `EN_MLO`: index is a live AXIS |

**This is the MLO axis, and it is genuinely orthogonal to A.** Proven by the RTL: a single
`` `define EN_MLO `` (fetch_weights_wrapper.v:34,105-125) flips the module between the two —
when off, `in_idx0_V` becomes an internal wire tied to `tvalid=1,tdata=0`; the DMA path,
aximm master, and weight output are IDENTICAL. memstream does the same via
`SET_BITS = SETS>2 ? clog2(SETS) : 1` with `SETS = mlo_max_iter or 1` (hwc:316). So
"remove MLO" = "set cardinality to 1", a dial ON TOP of storage, not a different storage.

### Coordinate C — STATICNESS (when are the param VALUES known)

| Value | Meaning | Consequence |
|---|---|---|
| `compile-known` | values are graph initializers | `accDataType`/`weightDataType` use ACTUAL values (narrow) |
| `runtime-writable` | values reloadable at runtime (AXI-lite, or PR-swap) | worst-case dtype bounds; runtime-writable port |
| `dynamic` | values arrive as a runtime tensor | worst-case bounds; no static `.dat` |

This coordinate reaches BACK into the compute space: `weights_may_change(p)` gates whether
`accDataType`/`narrow_weights` are computed from real weight values or worst-case type
bounds (base:469-498; already modelled in `fixtures/mvau/shared.py:_acc_datatype`). It is
independent of A and B — see §5 Case D (`embedded × runtime-writable` via partial reconfig).

---

## 2. The couplings — all PREDICATES or DERIVED, never axis-merges

The improvement over FINN's fused generics is that every cross-coordinate relation is a
**feasibility predicate** (reads point+context, returns reason|None) or a **derived scalar**
(computed after axes fixed) — the same two shapes the compute space already uses. None
requires collapsing two coordinates into one axis.

| Coupling | Kind | Statement | Provenance |
|---|---|---|---|
| cardinality × on-chip capacity | **predicate** | high N over on-chip storage may exceed the BRAM/URAM budget → prune | config+device (same shape as URAM gate) |
| `TAP_REP` = f(compute fold) | **derived** (parent→child) | index cadence a delivery kernel needs = how often the core consumes it | roll:558 (`prod(folded_in[:-1])` for Thresholding, else 1) |
| staticness → `accDataType` | **derived** (child→parent) | non-static weights force worst-case accumulator bounds | base:482-498 |
| URAM + non-Versal ⇒ runtime-writable | **predicate** | (existing) URAM weights on UltraScale need runtime_writeable=1 | config+device, hls:147 |
| `off-chip-DMA` ⇒ needs aximm master + address-offset block | **derived** (port set) | off-chip storage publishes an OFFCHIP port + LAYER_OFFS | fetch_weights.sv:36,107 |
| `indexed` ⇒ needs an index port + stream-tap edge | **derived** (port set) | cardinality>1 publishes an INDEX_SINK port | finnloop stream-tap |

The **only** genuine parent↔child data coupling is `TAP_REP` (a clean scalar) plus the
dtype flow-back — both already expressible. Everything else is a port-set consequence or a
capacity gate. This is what "change memory access cleanly" means: swap the coordinate-A
value, and the port set + predicates recompute; no if/elif rewrite.

---

## 3. The uniform index contract (RTL-grounded)

All three of FINN's MLO parameter-streaming paths consume the **same index type** — an
`IDX_BITS`-wide AXIS (`IDX_BITS=16`, or `clog2(SETS)`) distributed by the **same** stream-tap
network (finnloop:538,698-987). They differ ONLY in how each turns an index into params —
which is a property of coordinate A, not a new coordinate:

| Path (doc table) | Operator | Index consumption | Index → params | Storage (coord A) |
|---|---|---|---|---|
| **Fetch Weights** | MVAU | per-layer, standalone AXIS `s_idx` [IDX_BITS] | `addr = idx*LAYER_OFFS` → DMA read | off-chip-DMA |
| **Memstream** | Elementwise | per-layer, set-selector `s_axis_0` [SET_BITS] | sets BRAM read pointer to section | on-chip-replay |
| **Native** | Thresholding | per-ACTIVATION, companion AXIS `s_axis_set` [S_BITS] | `iset` = high address-prefix into threshold RAM | in-core RAM |

Two facts the SV makes precise:

1. **`TAP_REP` is the cadence-adapter (stream_tap.sv:44-87).** The stream-tap repeats each
   index `TAP_REP` times via a signed down-counter (`TCnt` loaded to `-TAP_REP`, tap valid
   while sign bit set, `TLst` at `-2`). MVAU/Elementwise consume once per layer
   (`TAP_REP=1`); Thresholding once per activation beat (`TAP_REP=prod(folded_in[:-1])`).
   **The index cadence is a function of the compute core's fold** — the one real
   parent↔child coupling, and it is a scalar.

2. **fetch-vs-select is WITHIN coordinate A, not a fourth coordinate.** memstream/fetch
   consume the index per-layer to LAUNCH A FETCH (external/BRAM must be addressed);
   thresholding consumes it per-activation as a DIRECT ADDRESS PREFIX into in-core RAM
   (thresholding.sv:197, `iptr[...+:clog2(SETS)]=iset`). "Fetch vs direct-address" is a
   property of the storage primitive, folds into A.

**loop_control** (mlo/loop_control.sv) owns the index LIFECYCLE independently of any param
kernel: the mux GENERATES `idx=0` for a fresh frame (mux.sv:214); intermediate_frames
INCREMENTS it (`m_idx = idx_in + 1`, if:197); the demux DETECTS LAST (`s_idx == N_LAYERS-1`
→ system exit, demux.sv:109). So the index ring is a self-contained structure; the param
kernels are pure consumers of whatever index arrives.

---

## 4. Two composition LEVELS (conflating them was the trap)

The evidence forces a distinction the ideal model blurred and the prototype never reached:

### Level 1 — INTRA-kernel (delivery composition)
One compute core + its parameter-delivery kernel, joined by role-tagged ports. This is
proto's `MVAUSIP` (HLS-compute + RTL-memstream, stitched by `mvau_sip_template.v`) and
ideal's T3 (`WeightDelivery` peer producer: compute publishes `WEIGHT_SINK`, delivery
publishes `WEIGHT_SOURCE`, neither names the other, an op-agnostic `RoleBinding` derives the
net). The stitch here is REGULAR — a small fixed role vocabulary — and very amenable to a
declarative resolver.

### Level 2 — INTER-kernel (MLO / FINNLoop container)
A container over a whole SUB-GRAPH (`body`, a graph-valued node attr, finnloop:85). It adds
THREE networks on top of the body, leaving the body's param nodes as ORDINARY nodes (roll
just sets `mlo_max_iter>0` on each, roll:122-133):
- **data path** — recirculate body output through an off-chip buffer (intermediate_frames,
  DMA-backed) so one physical body copy runs N layers.
- **index path** — the stream-tap graph distributes the iteration counter to every param
  node, in lockstep with the data.
- **per-node param streaming** — unchanged; still each node's own coordinate-A choice.

FINNLoop has ZERO compute — every shape/dtype/width delegates into `body.node[0/-1]` or
`find_consumer(body.input[ind])` (finnloop:146-275). It is a pure container: the closest
existing analogue to "a node whose contents is a resolved sub-DESIGN" — but it composes
over a sub-GRAPH, not over one kernel's delivery slot.

**Why this matters for the engine:** Level 1 is `Derived`-returns-sub-Point (a delivery
kernel resolved against a sub-context from the parent point — verified zero-engine-change,
[[emit-phase]]). Level 2 is a container whose structure is DERIVED FROM BODY ADJACENCY — a
whole coordination graph, not a pairwise constraint. The ideal model's "one `Legal` row
`LoopFetched ⇒ OffChipDma`" (kernel-model T2 tier-3) badly under-models Level 2: the real
thing is an adjacency-derived stream-tap graph with fork/join logic (finnloop:746-967 —
`axis_broadcaster` IP for input forks, discrete AND-gate handshake rebuild for internal
forks, join-pruning to one feeding edge).

---

## 5. Stress test — 4 novel storage cases (the framing holds)

The test: find a STORAGE (coord A) choice that cannot vary without forcing cardinality (B)
or staticness (C) — that would collapse the orthogonality. None found; the model instead
DIAGNOSES three FINN limitations as artificial couplings.

- **Case A — off-chip single-layer (no MLO).** Weights in HBM, DMA'd once, one layer
  (`off-chip-DMA × single × static`). FINN only exposes off-chip THROUGH MLO
  (`external_mem ⇒ mlo_max_iter>0`) — but the RTL `NO_MLO` guard IS this cell (const-0
  index, same DMA). **The model predicts the cell; FINN's inability to express it is an
  artificial coupling.** A large single MVAU whose weights don't fit on-chip wants exactly
  this. → **win (diagnostic).**
- **Case B — ping-pong double-buffered on-chip.** Prefetch set into buffer B while A feeds
  compute. A storage sub-property (latency-hiding), orthogonal to B and C — compose over
  single OR N sets. fetch_weights already double-buffers (doc:221). → **fits.**
- **Case C — compute-generated / procedural params** (weights = f(seed), frozen PRNG
  projection). No memory; a generator core emits the stream. A NEW value on coord A,
  orthogonal to cardinality (index N seeds) and staticness (runtime seed). → **the enum is
  open, not FINN's fixed set.**
- **Case D — partial-reconfiguration weight swap.** Params in LUTs, swapped via a PR
  bitstream at runtime (`embedded × single × runtime-writable`). Looks like a counterexample
  ("embedded ⟹ static" in FINN) but each coordinate takes an independent value — **it is
  expressible PRECISELY BECAUSE staticness is separate from storage.** → **confirms the
  split.**

**The one real strain:** cardinality × on-chip capacity are logically orthogonal but
RESOURCE-coupled (N-set memstream needs N× BRAM; at some N it forces a spill to off-chip).
This is NOT an axis-merge — it lands as a capacity **feasibility predicate**, the same shape
as the URAM gate. The framing survives and even tells us WHERE the coupling lives.

**Net:** the three coordinates correctly label three FINN limitations as artificial, not
hardware laws — (a) off-chip-only-via-MLO, (b) embedded⟹static, (c) closed storage enum —
which is exactly the "improve over FINN's generics" goal.

---

## 6. Prior-art divergence (how the two attempts modelled this)

| Concern | proto (`MVAUSIP`) | ideal (`kernel-model.md`) | this analysis |
|---|---|---|---|
| delivery vs memory | **FUSED** into one `MemstreamRTL` (owns topology + ram_style + pumping + .dat) | **SPLIT**: T3 `WeightDelivery` ⊥ T6 `MemoryStrategy`, one `Legal` coupling | storage is ONE coord (topology+primitive+sub-props); the T3/T6 split over-separates two things the RTL treats as one module |
| composition mechanism | `subkernels` tuple + hand-written Verilog top (`mvau_sip_template.v`, bespoke per-op) | role-tagged `Cell`s + op-agnostic `RegionResolver` (proposed, never built) | Level 1 = role-binding (ideal is right); Level 2 = adjacency-derived graph (both under-model) |
| MLO | **absent** (no MLO in prototype) | one `Legal` row `LoopFetched⇒OffChipDma` (tier-3) | a first-class **cardinality coordinate** + the whole index network — far more than a pairwise row |
| runtime-writable | **stubbed** (AXI-lite hard tied-off, proto sip:231/243) | `runtime-writable` as staticness value | coordinate C, `runtime-writable` |
| stitch | hand-wired, one din/dout assumed | dissolved to `RoleBinding` (Level 1 only) | TWO stitches: Level-1 role net (regular) + Level-2 index net (graph-structural) |

Key correction to the ideal model: **T3⊥T6 (delivery ⊥ memory as two axes) is finer than the
RTL warrants.** memstream and fetch_weights each fuse "topology + memory primitive" into one
module (memstream owns BRAM+replay; fetch_weights owns DMA+buffer). The honest cut is ONE
storage/topology coordinate with the memory primitive as a sub-property — closer to the
prototype's fusion, but WITHOUT the prototype's mistake of also swallowing cardinality and
staticness. The ideal model was right that MLO needs its own treatment; it was wrong that a
pairwise `Legal` row suffices.

---

## 7. Open questions for the engine-mapping phase (deliberately NOT answered here)

1. **Level-1 delivery** as a `Derived`-returns-sub-Point: a `param_delivery` Derived
   resolving a delivery `pool_schema` (embedded/on-chip/off-chip/dynamic bundles) against a
   sub-context built from the parent point (PE/SIMD/WMEM/fold/weight-dtype in; `TAP_REP`,
   depth/width, port set out). Verified mechanism-feasible; needs the sub-schema authored.
2. **Level-2 container** (FINNLoop): does it fit the engine at all, or is it a DIFFERENT
   construct (a resolved sub-graph + a derived index-network)? The stream-tap graph is
   derived from body adjacency — is that a `Derived` returning a structure, or does it need
   a new primitive? This is the deepest question and the real risk.
3. **The two stitches**: Level-1 role-binding is plausibly declarative (`IPICommands` over a
   role-match). Level-2 index-net has fork/join/broadcast — is it declarative-from-adjacency
   or irreducibly imperative? (This is Risk 3, the god-method, at the inter-kernel level.)
4. **Selection nested in composition**: coordinate A is itself a SELECTION (pick one storage
   bundle) — so delivery is `pool_schema` (selection) wrapped in a `Derived` (composition).
   Confirms selection and composition nest; neither subsumes the other.

See [[project-emit-phase]] for the compute-half status this builds on, [[impl-bundles]] for
the selection machinery a delivery pool reuses, and `COMPOSITION-MLO-HANDOFF.md` for the arc
framing. The compute-core emit is proven-equivalent and INSULATED from all of this
(reads none of the mem_mode cluster) — composition ADDS the delivery half, rewrites nothing.
