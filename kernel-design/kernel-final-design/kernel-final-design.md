# The Kernel Backend System — Final Converged Design (v2)

*The improved kernel backend design for FINN, synthesized from a full four-way
analysis of the design space: the FINN baseline (as-is), our theorycrafted ideal,
the real FINN `feature/kernel_flow` prototype, and Microsoft's Brainsmith. This
document commits to one design, resolves the seams where the source designs
collide, and honestly flags the risks that remain open.*

> **v2 re-synthesis.** This version reorganizes the design around a single
> spine — the split between an op's **semantic identity** and its **realizable
> capability envelope** — after a close re-reading of how all three real systems
> handle datatypes, tiling, and DSE knobs. That reading showed none of them
> separates identity from envelope cleanly, and each fails differently (§2). v2
> makes the envelope a first-class, composed, per-implementation concern rather
> than the hand-maintained superset Brainsmith uses. It supersedes v1's
> `ComputeStrategy` enum (now an `Implementation` pool) and carries forward v1's
> inheritance-free skeleton, unified selection, node-representation contract, and
> risk register. The four `seam-*.md` docs remain the frozen original synthesis;
> **where they differ, this document is authoritative.**

## How this design was reached

1. **Baseline** (`../finn-hw-backend-analysis/hw-backend-model.md`) — FINN as-is: base-class leak, `execute_node` diamond, selection god-switch, top-10 pressure list.
2. **Ideal** (`../finn-hw-backend-analysis/kernel-model.md`) — `@final` Kernel + Protocol parts + precondition-AST solver + typed stitch layer. Clean, **never built**.
3. **FINN prototype** (`../kernel-prototype-profile/`) — real `@dataclass(frozen=True)` Kernel replacing HWCustomOp; priority+cost_fn registry; real `sip` static-IP; content-addressed build cache. 0 blocker hacks.
4. **Brainsmith** (`../brainsmith-profile/`) — real, documented. Derives all 8 FINN methods from a declarative `KernelSchema`; DSE first-class; folding unified via a GCD rule. Retains HWCustomOp → inherits FINN's substrate leaks; schema is a hand-maintained capability superset.
5. **This design** — architect + adversary passes resolved the hard seams; a focused three-system re-read of the identity/envelope boundary drove this v2 re-synthesis.

---

## 1. The one-paragraph design

Every hardware op is modelled in two layers. The **semantic identity** — one
`KernelSchema` per op — declares what is true regardless of how the op is built:
its interfaces, which dimensions tile, its golden reference math, its structural
knobs, and the *mathematical* datatype class. The **realizable capability
envelope** — one entry per **`Implementation`** in an open, op-keyed pool —
declares what a *particular* realization can actually build: its concrete
datatype set, its tiling caps, its extra realization knobs, plus how it emits
hardware. These two layers meet in a single `@final`, frozen **`Kernel`** value
object that holds one schema and one bound implementation (plus composed
`weights`/`memory` strategies and a `reference`). The entire 8-method FINN
contract is **derived** from the schema — a kernel author writes zero folding
code for the common case. Selection is one predicate algebra that ranks the pool
by **feasibility** (does any implementation's envelope admit this node?) then
**preference**, and — evaluated over still-free folding variables — doubles as
the DSE valid-range engine. The core imports no qonnx `CustomOp`; a single
generic **`KernelCustomOp(CustomOp)` adapter** keeps qonnx as the retained node
contract and a descriptive, lowering-checkable `op_type` visible in netron —
while **zero per-op subclasses** make the base-class leak and `execute_node`
diamond structurally unnameable. Crucially, **the capability envelope is never
materialized as a superset**: it is *composed* from the pool on demand (union for
feasibility, per-implementation refine for the design space), so adding a
floating-point implementation to an integer-only op is one new pool entry with
**zero edits to the op's identity**.

**In one line:** *Brainsmith's declarative identity + DSE + folding, split from a
per-implementation capability envelope, on the ideal's inheritance-free skeleton,
behind a thin qonnx adapter, with one unified selection algebra over an open
implementation pool.*

---

## 2. The spine: semantic identity vs realizable capability envelope

The single organizing principle of this design is a two-layer split:

| Layer | What it holds | Owner | Cardinality |
|---|---|---|---|
| **Semantic identity** | interfaces; which dims tile; golden reference math; structural knobs (PE/SIMD exist); mathematical dtype class ("numeric") | the op's `KernelSchema` | **one per op** |
| **Realizable envelope** | concrete buildable dtype set; tiling caps; realization knobs (`ram_style`, `mem_mode`, …); the emitter | each `Implementation` | **N per op (a pool)** |

**Why this is the spine and not a detail:** a close re-reading of all three real
systems showed that *none of them separates these cleanly*, and each fails in a
different, instructive way. This design is the correction of all three at once.

- **Baseline (`HWCustomOp`)** — identity is cleanly centralized in the agnostic
  base (`matrixvectoractivation.py:53`; golden `execute_node` at `:132`), **but
  the envelope is scattered across three unsynchronized locations**: the backend
  codegen (`raise`s), the base's own unconditional attr enums (the `mem_mode`
  enum at `matrixvectoractivation.py:87` lists all modes even though HLS/RTL
  differ), and the `specialize_layers` feasibility predicates
  (`_mvu_rtl_possible`, `specialize_layers.py:235`; the SIMD cap
  `mvau_wwidth_max` lives in a *transformation*, `set_folding.py:170`, default 36
  at `:109`). Nothing cross-checks them.
- **FINN prototype (frozen `Kernel`)** — **no per-op identity exists at all**;
  every backend leaf subclasses `Kernel` directly, so shapes and dtype accessors
  are *copy-pasted* into each `MVAUHLS`/`MVAUSIP`, and the envelope is buried in
  per-leaf runtime `raise`s discoverable only by instantiate-and-catch
  (`kernel_registry.py`). The one designed hook for declared constraints
  (`_constraints`) ships **dead** — every op sets it `()`.
- **Brainsmith (`KernelSchema`)** — the only system that **declares the envelope
  as data** (constraint objects like `DatatypeInteger`, `constraints.py:136`,
  used at `thresholding.py:89`) — a genuine advance — **but it attaches that data
  to the identity, as a hand-maintained superset.** Adding an FP backend to an
  int-only op means hand-editing the kernel's constraint
  (`DatatypeInteger` → `DatatypeFloat`/`DatatypeInRange`, `constraints.py:157`/
  `:190`); nothing composes it from backends, and `@backend` metadata carries
  only `language`/`target` (`registry/_metadata.py`), no capability fields. The
  "compose the schema from backend specs" plan is confirmed **purely
  aspirational** — no partial implementation exists.

**The synthesis:** identity belongs centralized (baseline + Brainsmith prove it
works; the prototype proves the pain of omitting it). The envelope belongs **on
the implementation, declared as data** — the thing *none* of the three does.
Brainsmith has the right *representation* (constraint objects) attached to the
wrong *owner* (the identity). This design moves the envelope data onto each
`Implementation` and **composes** it, killing the superset. Notably, that is
Brainsmith's own stated roadmap — so this direction is externally validated, not
speculative.

---

## 3. The Kernel object

```python
@final
@dataclass(frozen=True)                       # prototype's frozen value object (kernel.py:36)
class Kernel:
    # ── IDENTITY (one per op, realization-invariant) ──
    schema:    KernelSchema                    # interfaces, tiling dims, structural knobs,
                                               #   reference math, mathematical dtype class

    # ── COMPOSED realization (solver-bound) ──
    compute:   Implementation                  # ONE entry from pool[op_kind] — carries the envelope (§4)
    weights:   WeightDelivery                  # {Embedded, Decoupled, External, LoopFetched}
    memory:    MemoryStrategy                  # {LutRom, Bram, Uram, OffChipDma}
    reference: ReferenceModel                  # golden semantics — bound by schema.op_kind, never solved

    # ── the 8-method FINN contract: PURE delegation to the schema-derived design_point ──
    def get_folded_input_shape(self, i=0): return self._point().iface_in(i).folded_shape
    def get_instream_width(self, i=0):     return self._point().iface_in(i).stream_width_bits
    def get_exp_cycles(self):              return self._point().initiation_interval
    #    ...all eight derive from (tensor, block, stream). No part consulted. No per-op math.

    # ── realization/execution: PURE delegation ──
    def emit_cell(self, env): return self.compute.emit_cell(env, self._point())
    def runner(self, mode, env): return self.compute.runner(mode, env)   # kills the diamond
    def evaluate(self, inputs): return self.reference.evaluate(inputs)    # never named execute_node
```

Two invariants:
1. **The 8-method contract reads only the schema-derived `design_point`** — the
   Brainsmith derivation (`kernel_op.py:431-490` in spirit), so adding an op
   family needs no folding code.
2. **The design_point is `f(schema, compute.envelope)`** — *not* a pure function
   of the schema alone. This is the correction v1 got wrong and §6/Risk 1 make
   precise: the bound implementation's envelope refines the space before shapes
   and legal ranges are answered.

---

## 4. The Implementation and its capability envelope

`compute` is not an enum. It is one entry drawn from an **open, op-keyed pool**,
carrying both the emitter and — the v2 addition — its **declared capability
envelope**:

```python
@dataclass(frozen=True)
class Implementation:
    name:         str            # "MvuRtlSystolicDsp58", "MvuHlsGemm", "MvuStaticIpInt8"
    op_kind:      str            # "MVAU" — the pool key
    language:     Language       # {hls, rtl, static_ip, subgraph} — picks the emitter only
    microarch:    str            # "systolic" | "gemm" | "dsp48-packed" — descriptive

    # ── the capability ENVELOPE: declared DATA (Brainsmith's representation, correct owner) ──
    dtypes:       Predicate      # buildable datatype set, e.g. AllOf(IntOrUint, BitsInRange(2,8), Signed)
    fold_caps:    FoldConstraint # HARD tiling caps only (systolic geometry, DSP packing).
                                 #   soft heuristics like mvau_wwidth_max go in `cost` — Risk 1
    extra_knobs:  dict[str, ParameterSpec]   # realization DSE knobs this impl adds (ram_style, …)

    precondition: Predicate      # node-level feasibility beyond dtypes (device family, shape, …)
    priority:     int            # coarse preference tier
    cost:         CostFn         # workload-dependent ranking — DEFERRED seam (§7.5)

    def emit_cell(self, env, point): ...   # `language` selects the emitter
    def runner(self, mode, env): ...
```

The registry is keyed by `op_kind`; under each key sits the pool:

```
pool["MVAU"] = { MvuHlsGemm, MvuRtlSystolicDsp58, MvuRtlSystolicDsp48,
                 MvuRtl4sx4u, MvuStaticIpInt8Gemm, … }   # several share a language
```

One `KernelSchema` per op; **N implementations, each declaring its own
envelope**. Systolic-vs-GEMM and HLS-vs-RTL are the *same kind of choice* —
"which pool entry" — distinguished by `dtypes`/`fold_caps`/`precondition`
(feasibility) and `cost` (preference), never by a type hierarchy. FINN's real
`finn-rtllib/mvu/` microarch zoo (`mvu_8sx8u_dsp48`, `mvu_vvu_8sx9_dsp58`,
`mvu_4sx4u`) becomes four pool entries with three declared envelopes, not four
hand-written classes.

---

## 5. The three axes partition identity/envelope differently

The re-read surfaced a subtlety worth stating precisely: dtypes, tiling, and
knobs are all "capability" concerns, but they split across the identity/envelope
line in *different proportions*. One composition mechanism (§6), three partitions:

| Axis | Identity part (schema) | Envelope part (implementation) |
|---|---|---|
| **Datatypes** | the *mathematical class* only — "MVAU is numeric", dtype *propagation* rules (out-dtype from in-dtypes) | **almost all of it** — the concrete buildable set (`Implementation.dtypes`). RTL MVU admits signed 2–8-bit; an FP impl admits fp16/32; identity claims neither |
| **Tiling / folding** | *which dims tile* and how tokens map across interfaces (the GCD structure) | how far — split into *hard* caps (`fold_caps`: systolic geometry, DSP packing — truly unbuildable) and *soft* costs (`cost`: the `mvau_wwidth_max` weight-stream-width heuristic — builds fine, dispreferred). Identity says SIMD divides MW; the impl's hard caps say what won't build, its cost says what to avoid (Risk 1) |
| **DSE knobs** | *structural* knobs that exist regardless of backend — PE, SIMD | *realization* knobs (`extra_knobs`) — `ram_style`, `res_type`, `mem_mode`, `depth_trigger_*`. Different impls expose different sets |

This table is the working definition your question asked for. It also explains
why the three systems were each inconsistent in a *different* place: baseline put
`mem_mode` (envelope) in the identity base for MVAU but the backend for
Thresholding; the prototype put everything (identity and envelope) in the leaf;
Brainsmith put dtypes (envelope) in the schema (identity) as a superset. The
partition above is the consistent rule all three lacked.

---

## 6. Composition: union for feasibility, refine for design space (kill the superset)

The envelope is **never materialized as a superset on the identity**. It is
computed from the pool at two distinct moments:

**(a) Pre-bind — feasibility as a UNION across the pool.** "Can *any*
implementation build this node?" → does the node's dtypes/shape satisfy some
`impl.dtypes ∧ impl.precondition`? This is exactly the integer-vs-FP case: an
int-only op has one impl with `dtypes = IntOrUint(2,8)`; **adding an FP
implementation is one new pool entry** whose `dtypes = FloatBits(16,32)`, and
feasibility auto-admits fp nodes with **zero edits to the schema**. The schema
never claimed fp (it only claimed "numeric"); the *new implementation* is what
makes fp buildable. The superset dissolves into `⋃ impl.dtypes`.

**(b) Post-bind — the design space as a per-implementation REFINE.** Once one
impl is bound, the realizable space is `schema_structure ∩ impl.envelope`:

```
design_space = base_space_from_schema.refine(compute.fold_caps)   # and dtype/knob refinement
```

This is v1's Risk-1 `refine` step, now generalized from folding to all three
axes and given a principled home (the impl's declared envelope). It formalizes
that `design_space = f(schema, bound implementation)` — and therefore that
**"design_space is a pure function of the schema" is false and is dropped.**

**Consequence — nested DSE.** Because `extra_knobs` and `fold_caps` depend on the
bound impl, the design space is genuinely nested: `(which implementation) ×
(that impl's refined sub-space)`. You cannot flatten it into one global space
before fixing the impl; you enumerate per-impl (or fix the impl, then explore).
This is more honest than a flat space and matches how selection actually runs
(§7.4): feasibility unions the pool, then exploration/refinement happens within
the bound choice.

**What stays hand-authored (the irreducible residual):** dtype *propagation*
resolvers (out-dtype from in-dtypes, e.g. Brainsmith's per-op closures) are
op-semantic identity and remain in the schema — that is correct, not a leak.
What is eliminated is the hand-maintained *support superset*.

---

## 7. Resolved seams

### 7.1 Inheritance-free core + thin qonnx adapter (kills the leak, keeps interop)

Two lineages that **never share a base class**. The **pure core** is the `@final
Kernel` (imports no qonnx `CustomOp`, no `onnx_node`, no `HWCustomOp` MRO). The
**adapter** is exactly one generic, op-agnostic `KernelCustomOp(CustomOp)` that
wraps a core Kernel and satisfies FINN's published contract by delegation + a
nodeattr codec. qonnx `CustomOp` is **retained as the boundary** on purpose (it
is the assumption the whole pipeline resolves through); only HWCustomOp's *guts*
move into `Kernel` + `Implementation`. With **zero per-op `CustomOp` subclasses**,
Brainsmith's two leak-vectors have no place to live. `execute_node` is one
concrete dispatch in the shell choosing between two differently-named core
methods (`reference.evaluate` vs `compute.runner`) — diamond unnameable in the
core, resolved by explicit `if` in the adapter.

### 7.2 Declarative folding + the GCD engine

Folding is declared per-interface (`block_tiling`/`stream_tiling`); legal PE/SIMD
is derived: `divisors(gcd(block dims where a token appears))` (`builder.py:644`).
A shared token across interfaces becomes one shared knob whose GCD set satisfies
all interfaces. This is the *identity* half of tiling (§5); the *envelope* half
(`fold_caps`) refines it post-bind (§6). Multi-param single-interface folding
(MVAU MW+MH) is the rare case handled by a declared `FoldMath` escape hatch — see
Risk 3.

### 7.3 DSE-as-contract on the frozen skeleton

Brainsmith's first-class DSE (`design_space`/`design_point`/`get_valid_ranges`/
`OrderedParameter`/sweeps) lifts onto the frozen skeleton by separating
**immutable design *space*** (memoized, now a fact of schema *and bound impl*)
from a **mutable exploration *cursor*** held outside the value object. Structural
knobs come from the schema; realization knobs come from `compute.extra_knobs`
(§5), so the explorable knob set is impl-dependent (the nested DSE of §6).

### 7.4 One selection algebra over the pool (kills the god-switch, closes the drift bug)

One predicate algebra, two evaluation modes, one context: **feasibility and DSE
valid-ranges are the same predicates, differing only in which variables are
bound.** Routing from a graph node to a bound Kernel:

```
CustomOp node (op_type "MVAU", unlowered)
  → pool.for_op("MVAU")                              # the whole pool — no if-ladder
  → drop impls whose (dtypes ∧ precondition) fail vs SelectionContext   ← FEASIBILITY (union, §6a)
  → rank survivors by (priority, cost(ctx, point))                      ← PREFERENCE  (§7.5)
  → bind winner → Kernel(schema, compute=winner, weights=…, memory=…)
  → design_space = base.refine(winner.fold_caps)                       ← per-impl refine (§6b)
  → wrap in KernelCustomOp adapter; relabel node op_type (§8)
```

Routing is identical for HLS-vs-RTL and systolic-vs-GEMM. Grafts: leaf predicate
`check(ctx) -> str | None` (Brainsmith — reason travels with the failure);
combinators `AllOf/AnyOf/Not/Implies` (ideal); **feasibility ⊥ preference**
(ideal — makes the `:60`-vs-`:275` drift bug unrepresentable); `explain()` over
the pool + per-clause reasons; and the fix to Brainsmith's `CustomConstraint`
that swallows exceptions into "infeasible" — exceptions now propagate. Static IP
falls out: a pool entry whose `priority` beats RTL and whose `dtypes`/`precondition`
match its common case → auto-selected, transparent fallback, zero substrate edits.

### 7.5 Open algorithm seams — the point of fixing the structural ones

This design fixes the *structural* seams (identity/envelope, inheritance-free
skeleton, node representation, the feasibility predicate) **precisely so the
*algorithmic* seams can stay open.** The goal is extensibility and exploration
through a tight interface — not framework-finalized control. Wherever a *strategy*
could reasonably vary — how designs are scored, how ties break, how the design
space is searched, how a folding heuristic trades width against timing — the
framework fixes the **interface** and leaves the **policy** to the user.

This is a stated design principle, not a deferred loose end. Four instances:

- **Cost / objective.** `priority` (a coarse tier) expresses "prefer static-IP,
  else RTL, else HLS" but is **not** enough for same-language, same-op,
  both-feasible impls: **systolic vs standard GEMM is workload-dependent**
  (systolic wins large-square, GEMM small-skinny). The framework fixes the
  interface — `cost(ctx, point) → comparable`, with feasibility *strictly
  separated* so a cost function can **never** accidentally forbid a buildable
  design — and leaves the *model* open (static estimate, learned cost, DSE
  measurement, multi-objective area/latency/power, pluggable per-board). The
  prototype's `KernelProjection` is one candidate implementation, not the mandate.
- **The `mvau_wwidth_max` lesson (see Risk 1).** FINN's weight-stream-width cap is
  the canonical example of why this separation matters: it is a *soft* routing/
  timing heuristic (default 36, "set it high then lower until it fits your board"),
  mis-encoded today as a hard feasibility gate in a transformation. In this design
  it is **a cost contribution an implementation declares** (weight-stream width →
  routing pressure), consumed by whatever search the user plugs in — never a
  framework-baked constant.
- **DSE search policy.** The design space is declared (§6); *how* it is explored —
  order, sampling, early-stopping, ML-guided navigation — is a pluggable strategy
  over the same `get_valid_ranges` interface, not a fixed sweep.
- **Selection tie-break.** Fixed only enough to be *deterministic* (the impl
  `name`, never import order); any richer policy is the `cost` seam above.

Until a user supplies a strategy, safe deterministic defaults apply (`priority`
then `name`) so the framework is usable out of the box without pre-committing its
users to one optimization philosophy.

---

## 8. Node representation: qonnx + netron contract

Representation is orthogonal to code structure — keep the visibility FINN users
rely on without per-variant classes.

**Retained:** qonnx `CustomOp` as the node contract; resolution stays
`(domain, op_type, version) → class` via the qonnx registry. A **descriptive,
lowering-checkable `op_type` label**: unlowered = `MVAU`; lowered =
`MVAU_systolic_dsp`. "Has this been lowered?" stays trivial and grep-able; netron
shows it at a glance.

**Changed:** the op_type is a **projection of `op_kind + implementation`**, not an
opaque atom needing its own class. Both `MVAU` and `MVAU_systolic_dsp` resolve —
through the same qonnx registry — to the **one generic `KernelCustomOp` adapter**,
which reconstructs `(schema, Implementation)` from the label + nodeattrs. Many
op_type strings are fine; N hand-written leaf classes are what we avoid. The
inspectability contract: the *name* carries op_kind + the load-bearing
specialization identity; a required `implementation` **nodeattr** carries the full
impl name + resolved folding config, so the specialization stays machine-readable
regardless of naming policy.

---

## 9. Serialization & rehydration (a crucial FINN capability)

FINN must be able to save a graph of *specialized* kernels to ONNX and reload it
losslessly. This design preserves that capability by using the **exact mechanism
FINN already relies on**: the file stores *state*; the registry supplies
*behavior*. No Python object is ever pickled.

**The contract, made explicit** (it was previously only implied by §8):

- **What serializes (state, in the ONNX node):** `op_type` (the human/netron
  label, §8) plus nodeattrs — the authoritative `implementation` name, the bound
  `weights`/`memory` strategy names (for ops that have them), the resolved dtypes,
  and the folding/knob config values. All of this is plain data.
- **What never serializes (behavior, in code):** the `KernelSchema`, the
  `Implementation` object (including its `emit_cell`/`cost` callables and its
  envelope predicates), the strategy parts, the reference model. These are
  reconstructed by name.

**Rehydration** in `KernelCustomOp.__init__ → kernel_from_node(node)`:
1. `op_kind` → the schema, resolved from the registry (schema is code).
2. the **`implementation` nodeattr** → look up `pool[op_kind]` for the entry of
   that name → the `Implementation` object *with its lambdas* (code).
3. remaining nodeattrs → the `config`.
4. `Kernel(schema, compute, config)` is rebuilt; `design_point` is *recomputed*
   from `config` (`space.configure(config)`), never loaded.

This is precisely baseline FINN's `getCustomOp(node)` resolving a *class* from
`(domain, op_type, version)` and wrapping the node's attrs — the **`Implementation`
pool is the registry analogue** for the realization axis. Two deliberate
properties:

- **Reconstruction keys off the `implementation` nodeattr, not the op_type
  string.** The projected op_type (`Thresholding_pipelined_compare`) is for humans;
  string-parsing it back into `(op_kind, microarch)` would be brittle, so the
  machine path uses the nodeattr as the source of truth.
- **Smaller, canonical serialized surface than baseline.** Because `design_point`
  is *derived*, not stored, there is no stale-derived-nodeattr class of bug
  (FINN sometimes persists computed quantities that can drift). The trade is the
  standard, now-explicit FINN assumption: **the pool code must be
  version-compatible with the saved file** — handled by qonnx op versioning
  (`_vN`) exactly as today. Container kernels round-trip recursively: a `Subgraph`
  implementation's nested graph is itself ONNX.

---

## 10. Conformance

- **Simple streaming ops** (softmax, layernorm, single-mode thresholding): schema
  declares interfaces + tiling; 8 methods derive; one impl binds; single streams
  wire. **Holds — the bulk of the op zoo.**
- **Multi-microarch op** (MVAU systolic/GEMM/dsp-packed): one schema, a pool of
  impls with distinct envelopes; solver picks by feasibility then cost. **Holds —
  the case the `Implementation` model exists for.**
- **Adding an FP backend to an int-only op** (the envelope test): one new pool
  entry declaring `dtypes = FloatBits(...)`; feasibility union auto-admits fp;
  **zero schema edits.** **Holds — this is the superset anti-pattern retired.**
- **Static-IP GEMM**: a peer pool entry, auto-selected. **Holds** (prototype ships
  the `sip` mechanism).
- **iodma / non-tensor ports**: typed Port sum where only DATA ports carry a shape
  → memory-mapped ports never raise. **Holds by type.**
- **MVAU (dual-fold + decoupled weights + 236-line IPI stitch)**: **highest-risk
  integration** — three unbuilt mechanisms on one op. See Risks.

---

## 11. Honestly unresolved risks

### Risk 1 (the crux, now cleanly framed) — an implementation refines the space, and *hard caps must be separated from soft costs*
An implementation's envelope narrows what a node can be. The organizing
correction (resolving v1's flagged contradiction): the envelope is declared
per-implementation (§4), composed via union (feasibility) and refine (design
space) (§6), at the price of dropping the "design_space = f(schema)" slogan and
accepting nested DSE.

But the envelope has **two sub-kinds that must not be conflated**, and getting
this wrong is a real bug FINN commits today:

- **Hard caps → `fold_caps` (feasibility).** A truly unbuildable configuration —
  e.g. a systolic array's fixed geometry, a DSP-packing that only supports certain
  widths. These belong in the feasibility predicate; violating them means the
  hardware cannot be generated.
- **Soft costs → `cost` (preference).** A configuration that *builds fine* but is
  *dispreferred*. **`mvau_wwidth_max` is the canonical example** (`set_folding.py:
  170-176`, default 36 at `:109`): it caps the decoupled weight stream at
  `weight_bits × SIMD ≤ 36`, but nothing physically breaks above it — the bus is
  just wider and harder to route/time. It is a *search heuristic* ("first increase
  SIMD while weight-stream width ≤ threshold", docstring `:94`), the FINN examples
  literally being "set it high, then lower until it fits your board." Today it is
  **misfiled as a hard gate inside a transformation**, behind an
  `if op_type in ["MVAU_hls","MVAU_rtl"]` branch (`:155`) with a magic default —
  a soft, board-dependent, realization-and-dtype-coupled *preference* masquerading
  as feasibility.

**Commit:** `fold_caps` is **hard-only**. Anything soft — above all the
weight-stream-width heuristic — is a **cost contribution the implementation
declares** (§7.5), never a feasibility predicate. This keeps the feasibility ⊥
preference law (§7.4) honest: a search heuristic can never render a buildable
design "infeasible," and a user swapping objectives or boards changes cost, not
what is legal. Note the coupling that made this constraint a favorite example: it
is simultaneously folding-legality-shaped, *decoupled-weight-delivery*-specific
(an embedded-weights MVAU has no weight stream and no such term at all), and
dtype-coupled (the bound moves with `weight_bits`) — which is exactly why it can
only be modelled as a *declared term on the specific implementation*, not a fact
of the op.

### Risk 2 — the "thin" adapter owns a selection-policy decision
On an optimization-attr write (`SetFolding` setting SIMD), the adapter must decide
re-select-or-not; both choices break a FINN assumption (re-selecting flips the
impl under a transform assuming folding/backend independence; not-re-selecting
permits a config past the impl's `fold_caps`). **Commit:** re-configure within the
bound impl and add an **explicit post-folding feasibility re-check** — the
infeasibility bug is *contained by an assertion*, not eliminated. FINN's
independent-pass model is preserved at the cost of one guard.

### Risk 3 — MVAU is a bundle of orthogonal features, not an op; it needs its own decomposition pass
MVAU is the biggest monster in FINN, because matmul is load-bearing for every
network and *so many features have been stuffed into one class over the years*.
This one analysis already surfaced, on MVAU alone: the weight-delivery axis
(embedded / decoupled / external / MLO), dual folding (MW×MH), the memory-strategy
`mem_mode` axis, the 236-line IPI stitch, the DSP-packing microarchitectures
(`mvu_8sx8u_dsp48` / `dsp58` / `4sx4u`), threshold fusion, and the
`mvau_wwidth_max` heuristic (Risk 1). That is half a dozen orthogonal concerns
wearing one op name.

The concrete integration hazards are real — the multi-param folding lift is
genuine surgery in the 1030-line DSE core (the "zero folding code" property
evaporates here; per-op fold math relocates into a `FoldMath` callable), the
stitch layer is never-validated, the `pumpedMemory` ClockPlan is unbuilt, and the
stitch `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK)` matches on `(role, proto)` with
**no width/shape equality check**, so a fold-arithmetic error wires a mis-sized
net silently to synthesis, on the op with the least incidental coverage.

**Commit:** do **not** try to land MVAU while building the core abstraction.
(1) **Validate the general design on the *simpler* ops first** — softmax,
layernorm, single-mode thresholding, elementwise — so MVAU's accumulated
complexity does not distort the core contract. (2) Treat MVAU as a **dedicated
follow-up decomposition pass** — the same census → model → design arc this whole
effort ran on the backend, applied to one op, to tease its stuffed-in features
back onto the orthogonal axes this design provides (compute impl, weight delivery,
memory strategy, folding, cost). (3) Only then land it, with the width-consistency
assertion added to the resolver's `match()`. MVAU-dual-fold-decoupled remains the
**single highest-risk milestone** — and it *begins with analysis, not
implementation*.

### Open by design (not a risk) — the algorithm seams
Per §7.5, the *mechanisms* for cost, DSE search policy, and tie-breaking are
committed as tight interfaces; the *strategies* behind them are intentionally left
open for users to explore. This is the design goal (extensibility over finalized
control), not an oversight or a deferred loose end.

### What genuinely holds (calibration)
The reference/runner diamond-kill is unconditionally sound; static-IP
auto-selection with zero substrate edits is clean; the union/refine envelope model
retires the superset anti-pattern all three real systems suffer; the role-binding
resolver dissolves the god-method's coordination for single-stream ops; the
predicate algebra is correct for the 0-and-1-free-variable cases covering most of
the zoo. **The design is real everywhere the four sources ship, and needs building
everywhere they hand off** — the seams above are flagged, not hidden.

---

## 12. Migration

Clean-slate *target*, incremental *delivery* — the ideal's 8-PR strangler-fig,
de-risked because **the adapter (§7.1) is the migration bridge**. Order: (1) core
Kernel + schema derivation + the op_type-projection representation (§8) for 2–3
simple ops behind the adapter; (2) the `Implementation` pool + union-feasibility
selection replacing the god-switch (§4, §6a, §7.4); (3) per-impl envelope refine +
DSE-as-contract (§6b, §7.3); (4) static-IP + cache; (5) **MVAU decomposition
pass** — analysis first (Risk 3), then the dual-fold lift + width-checked stitch as
the highest-risk milestone; (6) retire legacy paths. A user cost/search strategy
(§7.5) can plug in anytime after (2) without blocking the pipeline. Green against
the op regression at every step.

## 13. The lesson carried into the design

> **Structure is better declared than coded, and design-space exploration belongs
> inside the kernel abstraction — but an op's *identity* and its *realizable
> envelope* are two layers, not one.** Every real system conflated them: the
> baseline scattered the envelope, the prototype buried it in exceptions,
> Brainsmith hoisted it onto the identity as a hand-maintained superset. This
> design keeps identity centralized in one schema, moves the envelope onto each
> implementation as declared data, and *composes* it — union for feasibility,
> refine for the design space — so the superset dies and adding a backend never
> edits the op. It pays for that with nested DSE and one dropped slogan
> ("design_space = f(schema)" becomes "f(schema, implementation)") — trading a
> clean claim for a correct one.
