# Handoff: Kernel Composition — memory modes, weight delivery, and MLO

*Working handoff for the composition arc. Written mid-session 2026-07-17 to keep focus
through compaction. This is a DESIGN/ANALYSIS task first, not an implementation task —
the goal of the next phase is to understand the substrate and design how composition
maps onto our engine, THEN build incrementally.*

---

## 0. Why this is the next big thing (and the one unproven claim)

The design-space engine is validated across 3 ops (MVAU/VVAU/Thresholding, 77 tests) and
emit is proven byte-equivalent to FINN (differential test green, re-verified post-#1566).
**Every core thesis claim is now de-risked EXCEPT one: composition** — "kernels compose
kernels." Specifically, the weight-DELIVERY story: a decoupled MVAU is not one kernel, it
is a compute core PLUS a weight-delivery kernel (memstream / dynload / fetch-weights),
co-existing and wired.

This is the design's real value-add over FINN. Per `toy-vs-brainsmith-thresholding.md`
(B2): mem_mode-as-composed-part is THE differentiator, and it is Risk 3 — the least
validated piece. Until composition is shown, the thesis has an asterisk: "composable
except for the thing FINN was worst at, which we haven't demonstrated." So this is
proof-of-thesis work, not coverage work.

**We have deferred this at every prior step** (MVAU emit was scoped to the "compute half"
precisely to keep it insulated from composition). The deferral was principled; now it is
the main event.

---

## 1. The core framing (already settled — do not re-derive)

Two structural operators, established across prior sessions (see `impl-bundles` memory,
`design-space-model.md` §5, and the "selection vs composition" framing):

- **SELECTION = SUM** — pick ONE implementation from a pool (HLS *or* softvec *or* packed).
  Maps to the `Axis` primitive. **BUILT** (`pool_schema` + Implementation bundles).
- **COMPOSITION = PRODUCT** — a kernel co-exists WITH sub-kernels (compute core *and* a
  weight-delivery kernel). Maps to the `Derived` primitive: per `design-space-model.md`
  §5, "a sub-kernel is a `Derived` component whose own sub-space composes" — i.e. a
  `Derived` whose `compute(point, context)` returns a *resolved sub-`Point`*. **NOT BUILT.**

The key insight that makes this tractable: **composition needs NO new engine type.** It
rides the existing `Derived` primitive. A `weight_delivery` Derived resolves a memstream
sub-schema against a sub-context derived from the parent point; parent→child couplings
flow IN via the sub-context, child→parent couplings flow OUT via predicates reading
`p.weight_delivery.<field>`. Namespacing is free (nested under the derived key).

**What this task must NOT do:** invent a recursive `Kernel { pool + components }` type
(rejected before — unexercised scaffolding). Build composition on the `Derived` seam.

---

## 2. Where mem_mode lives today (the thing we're restructuring)

In our current MVAU/VVAU fixtures, the weight-delivery cluster (`mem_mode`, `ram_style`,
`runtime_writeable_weights`, `pumpedMemory`, `dynamic_input`) sits as **op-level shared
axes**, explicitly LABELLED as "borrowed from a future memstream sub-kernel" (see
`fixtures/mvau/shared.py` — the RESERVED COMPOSITION SEAM comment). Thresholding proved
this is per-op (its mem_mode is HLS-bundle-local, deferred entirely). The composition arc
is where that cluster **moves out** of op-level into a composed `weight_delivery` Derived.

`mem_mode` is really a SELECTOR over which delivery sub-kernel is composed in:
- `internal_embedded` → weights baked in (HLS `params.h`; RTL: no ROM, streamed) — no
  sub-kernel, or a trivial one.
- `internal_decoupled` → **memstream** sub-kernel (AXI-stream weight delivery, optional
  runtime-writable via AXI-lite). `matrixvectoractivation.py:generate_hdl_memstream`.
- `external` → weights arrive from outside (dynamic_input path).
- `mlo_max_iter > 0` → **fetch_weights / MLO** path (see §3).

FINN's emit dispatches these in `MVAU_rtl.generate_hdl` (the if/elif on mem_mode /
mlo_max_iter). Our composition replaces that scattered dispatch with "which sub-kernel is
composed," resolved as data.

---

## 3. The MLO substrate — NOW IN-TREE (upstream #1566, the big new reference)

**This is the reason to do composition NOW.** PR #1566 ("Feature/tiling MLO") merged into
our branch (`9adfbe131`) brings the reference implementation of the memory/offload story
we're about to model. Study these — they are the ground truth:

**RTL (`finn-rtllib/`):**
- `mlo/loop_control.sv` + `loop_control_wrapper.v`, `mlo/infrastructure/{mux,demux,
  intermediate_frames}.sv` — the Multi-Layer Offload loop control (iterate a shared
  compute unit over multiple layers' weights).
- `fetch_weights/{fetch_weights,local_weight_buffer}.sv` + wrapper — the weight-fetch
  delivery mechanism for MLO (weights streamed per-iteration from a buffer).
- `mvu_tiled/{mvu_tiled_axi,cu_mvau_tiled,input_gen,acc_stage,weights_buff_tile}.sv` +
  wrapper — a TILED MVU compute core (tiling factor `TH`), the compute side of MLO.
- `memstream/{hdl,sim,doc}` — the standalone memstream weight-delivery RTL (the decoupled
  path's sub-kernel — this is the clearest "delivery kernel as its own thing" example).

**Python:**
- `custom_op/fpgadataflow/rtl/finn_loop.py` — **`FINNLoop(HWCustomOp, RTLBackend)`**: a
  META/CONTAINER node — "a placeholder for a group of fpgadataflow nodes separated into a
  FINN-ONNX model of its own, executed in a loop." **This is FINN already doing a form of
  composition** (a node whose `body` attr is a sub-graph). Study how it models the
  sub-graph, iteration, and weight indexing — it is the closest existing analogue to our
  `Derived`-returns-sub-Point idea, and a reality check on our framing.
- `transformation/fpgadataflow/loop_rolling.py` — the transform that rolls repeated
  layers into a FINNLoop (the MLO lowering).
- `matrixvectoractivation_rtl.py` (post-#1566) — now branches codegen between the plain
  `mvu_vvu_axi_wrapper.v` and the tiled `mvu_tiled_axi_wrapper.v` ($TH$ binding), plus
  `adapt_for_loop_body` / `mlo_max_iter` consumer logic. Read the CURRENT file, not the
  pre-merge handoffs.

**Caveat:** MLO was the ORIGINAL blocker for composition ("needs the MLO analysis").
#1566 is that analysis, shipped. So the blocker is substantially lifted — the reference
now exists to design against.

---

## 4. The design questions to answer FIRST (analysis phase)

Before writing any fixture code, answer these against the real substrate (§3):

1. **What is the memstream sub-kernel's OWN design space?** Its axes (ram_style,
   pumpedMemory, runtime_writeable, depth/width), derived (sip_depth = calc_wmem()/width,
   the padded stream width), predicates (URAM+non-Versal ⇒ runtime_writeable=1 — the
   combination gate). This becomes a `pool_schema`-style sub-schema.
2. **What are the parent↔child COUPLINGS?** Parent→child: MVAU's PE/SIMD/WMEM/weight-dtype
   determine the memstream depth/width. Child→parent: does a delivery choice gate a
   compute predicate? Enumerate them — they define the `subcontext(parent_point)` builder
   and the cross-reading predicates.
3. **How does `Derived`-returns-sub-Point actually resolve?** Concretely: a
   `weight_delivery` Derived whose `compute` builds a sub-Context from the parent point +
   parent Context, calls `resolve(memstream_schema, subctx)`, returns the sub-Point (or
   Illegal → propagates). Does the engine need ANY change to carry a Point-valued derived,
   or does it already work (a Derived can return any value)? **Verify this is zero-engine-
   change, as the framing claims.**
4. **Where does MLO fit vs plain decoupled?** MLO (FINNLoop + fetch_weights + tiled MVU)
   is a MORE composed structure — a loop container over a tiled compute core with per-
   iteration weight fetch. Is it: (a) a different delivery sub-kernel selection, (b) a
   different COMPUTE implementation (tiled) that itself composes a fetch-weights delivery,
   or (c) an outer container composing multiple kernels? This is the deepest modeling
   question — MLO may stress the composition model the way MVAU stressed selection.
5. **Emit for composition** — how do the sub-kernel's Artifacts merge with the parent's?
   (The parent wrapper instantiates the memstream module + the IPI stitch wiring them.)
   The decoupled-mode "stitch god-method" is FINN's `code_generation_ipi` — the known hard
   part, Risk 3. Our `IPICommands` artifact kind is the seam; does composition make the
   stitch declarative, or is it still imperative?

---

## 5. Recommended approach (incremental, proof-first)

Do NOT try to model all of MLO at once. Suggested increments:

1. **ANALYSIS first** — answer §4 Q1-Q3 against the memstream RTL + FINNLoop. Produce a
   short design note (like `mvau-design-space.md` but for the memstream sub-kernel + its
   couplings to MVAU). This is the acid test the way MVAU was for selection.
2. **Simplest composition:** model the memstream sub-kernel as its own schema and compose
   it into MVAU via a `weight_delivery` Derived, decoupled mode only. Prove `resolve`
   yields a parent Point carrying a resolved sub-Point, with couplings honored, venv-only.
   This is the "does composition work at all" gate — analogous to the first emit increment.
3. **Composition emit:** the parent RTL wrapper instantiates the memstream module; extend
   emit to merge sub-kernel Artifacts + the IPI stitch. Differential-test vs FINN's
   decoupled codegen (the harness pattern exists: `diff_mvau_emit_vs_finn.py`).
4. **MLO as the stress test** (§4 Q4) — only after plain decoupled composes cleanly. This
   is where FINNLoop / tiled MVU / fetch_weights come in; likely reveals whether
   composition needs to nest (a kernel composing a kernel that composes a kernel).

Validation environment is READY (Vivado 2025.2, Docker) — see `emit-phase` memory + the
`run_{diff,elaborate}.sh` runners for the working pattern.

---

## 6. Guardrails / what NOT to do

- **Do not invent a recursive `Kernel` supertype.** Composition rides `Derived` (§1).
- **Do not collapse selection into composition or vice versa** — they are distinct
  operators (sum vs product) on distinct primitives (Axis vs Derived). Both nest, neither
  subsumes the other (the prototype's `MVAUSIP` proved an implementation can ITSELF be a
  composite — HLS-compute + RTL-memstream).
- **Do not model to match FINN's scattered mem_mode if/elif** — the whole point is to
  replace "16 methods branch on mem_mode" (per toy-vs-brainsmith A2) with "which sub-kernel
  is composed." Model the clean structure; let the real RTL be the equivalence target.
- **Read the CURRENT code** — #1566 rewrote the MVU RTL codegen; prior handoffs' line refs
  are stale for `matrixvectoractivation_rtl.py`.
- **Keep the compute-core emit untouched** — it's proven equivalent and insulated;
  composition ADDS the delivery half, it does not rewrite the compute half.

---

## 7. State / pointers

- Branch: `feature/dataflow-kernel` @ `9adfbe131` (post-MLO-merge). Split work is a
  separate worktree (`../finn-mvu-split`, `feature/mvu-wrapper-split`) — orthogonal
  (compute-core packaging vs weight delivery); the two can proceed in parallel.
- Trunk: `finn/src/finn/design_space/` (engine `space/`, fixtures, emit, tests — 77 green).
- The reserved seam in code: `fixtures/mvau/shared.py` weight-delivery cluster comment.
- Memory: `emit-phase`, `impl-bundles`, `hermeticity-contract`, `dev-environment`.
- Model spec: `design-space-model.md` §5 (composition deferral + the Derived-sub-Point
  mechanism), §1.2 (primitives). Map: `mvau-design-space.md` §4 (weight-delivery = the
  "third axis owned by neither backend").
- The composition-relevant FINN reference is all under `finn-rtllib/{mlo,mvu_tiled,
  fetch_weights,memstream}/` + `custom_op/fpgadataflow/rtl/finn_loop.py` +
  `transformation/fpgadataflow/loop_rolling.py`.
