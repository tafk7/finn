# Design: Capability / trait-based with constraint-solved selection

> **Essence:** A Kernel declares the traits it REQUIRES and its reference semantics; every implementation (HLS/RTL/static-IP/weight-delivery/container) is a Realization that declares the traits it PROVIDES plus a feasibility precondition expressed as inspectable constraint DATA; a single op-agnostic solver resolves op-requirements against the realization pool by constraint satisfaction, replacing both the specialize_layers god-switch and the base-class op_type leaks with one declarative mechanism.

**Selection strategy:** `separate-pass-queries`

## Core model

## The one idea

Everything the current system encodes as *class identity* (`MVAU_rtl`), *string dispatch* (`optype + "_" + impl_style`), or *imperative feasibility ladders* (`_mvu_rtl_possible`) becomes **declared, matchable DATA on two sides of a solver**:

- **Kernel** (the op) declares the **traits it REQUIRES** and its **reference semantics**. It says *what must be realized*, never *how* or *whether-RTL-is-possible*.
- **Realization** (any implementation kind: HLS, RTL, static-IP, a weight-streamer, a container-stitcher) declares the **traits it PROVIDES** plus a **`Precondition` — a feasibility predicate as constraint data** — plus a **priority** and a pure **emit** function.
- **Solver** is a single op-agnostic pass: `select(Requirements, RealizationPool, SelectionContext) -> Binding | Infeasible(reasons)`. It matches traits (set containment), evaluates preconditions against the concrete node/device context, ranks feasible candidates, and binds the best. It never names an op or a kind.

The substrate's entire vocabulary is `Trait`, `Requirements`, `Capabilities`, `Precondition`, `SelectionContext`, `Binding`, `Realization`. It contains **no `if op_type ==`, no HLS/RTL class axis, no `optype + "_hls" in dict`**. Adding an op adds a Kernel + Realizations (data + one file). Adding a *third kind* adds a Realization (one file). Neither edits shared code — this is how criteria #2 and #3 die together.

```
        REQUIRES (data)                         PROVIDES (data)
   ┌──────────────────────┐               ┌───────────────────────────┐
   │  Kernel (the op)      │               │  Realization pool          │
   │  .semantics()  MatMul │               │  MVU_RTL  provides+precond │
   │  .require(ctx) ──────►│  Requirements │  MVU_HLS  provides+precond │
   │  .reference(ctx) gold │               │  MVU_INT8_IP provides+prec │
   └──────────────────────┘               │  Memstream / Dynload / MLO │
              │                            │  Embedded (delivery reals) │
              ▼                            └─────────────┬─────────────┘
       ┌─────────────────────────────────────────────────▼───────────┐
       │  SOLVER  (single, op-agnostic, constraint satisfaction)       │
       │  1 semantic match  2 trait coverage  3 precondition eval      │
       │  4 rank (pref, priority, cost)  5 bind  → Binding / reasons   │
       └─────────────────────────────┬────────────────────────────────┘
                                     ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  COMPOSER  wires Binding's parts by matching PORT traits       │
       │  (compute.weights_in  ⟷  delivery.weights_out)                 │
       └──────────────────────────────┬───────────────────────────────┘
                                      ▼
              emit(Binding, sink)  pure: typed Binding → Artifacts
              run(Binding, exec_ctx)  cppsim/rtlsim, injected sim
```

## Trait vocabulary — the ubiquitous language spanning the 6 axes

A `Trait` is a **structured, hashable, matchable value** (not a flag). Both sides speak it. The six axes are exactly six trait families:

| Axis | Trait constructor | Example value | Matched by |
|---|---|---|---|
| 1 compute-impl | `Compute(semantic)` | `Compute(MatMul)` | equality on semantic (primary key) |
| 2 weight-delivery | `ParamPort(role, accepted_modes)` req / `Delivers(role, mode)` prov | `ParamPort(weights,{embedded,decoupled,external,dynamic,mlo})` | mode ∈ accepted_modes |
| 3 memory-strategy | `ParamStorage(kind, discipline)` | `ParamStorage(uram, streamed)` | kind ∈ op-accepted ∧ device-precond |
| 4 folding | `Folding(scheme)` | `Folding(PE_SIMD)` | scheme-name equality (interface only) |
| 5 ports | `TensorStream(i,dir)`,`AxiLite(regs)`,`AxiMM(w,dir)`,`Sideband(k)`,`ApNone(sig)` | `AxiMM(64,in)` | complementary-port match (composition) |
| 6 execution | `RunsVia(mode)` on realization; `reference()` slot on Kernel | `RunsVia(rtlsim)` | requested exec mode ∈ provided |
| structural | `Container(body)`, `Infra` | `Container(Subgraph)` | presence |

`Requirements = {required_traits: set[Trait], ports: PortSet, reference: Callable|None}`.
`Capabilities = {provided_traits: set[Trait], precondition: Precondition, priority: int, artifact_kind, emit_fn, run_modes}`.

## Preconditions as DATA — the heart of the prior

The as-is `_mvu_rtl_possible` / `_vvu_rtl_possible` / `_dwc_...` / `_requant_rtl_possible` / `_layernorm_...` / `_elementwise_...` (`specialize_layers.py:235-374`) are the god-switch. In this design **they are not code that lives in a central switch — they are `Precondition` values owned by each Realization**. A `Precondition` is a tree of typed constraint primitives evaluated against a typed `SelectionContext`; on failure each primitive yields its own reason string (killing the per-arm bespoke `warn_str` duplication at `specialize_layers.py:109-211`).

Constraint primitives (data, serializable, `explain()`-able):
`DtypeBits(port, op, n)`, `DtypeSigned(port, bool)`, `DtypeIs(port, dt)`, `DtypeIsInteger(port)`, `AttrEquals(name,val)`, `AttrDivides(a,b)`, `DeviceFamily(any_of)`, `DspBlockIn(any_of)`, `ParamIsNarrow(port)`, `BroadcastCompatible(lhs,rhs,out)`, `FoldingIs(scheme)`.
Combinators: `AllOf`, `AnyOf`, `Not`, `Implies`.

**`_mvu_rtl_possible` (`specialize_layers.py:235-278`) re-expressed as the MVU_RTL realization's precondition:**
```
MVU_RTL.precondition = AllOf([
    AttrEquals("noActivation", 1),          # L246-249: activation off
    AttrEquals("binaryXnorMode", 0),        # L247
    DtypeSigned("weights", True),           # L252-253
    DtypeBits("activations", ">=", 2),      # L274-275  (single source of truth)
    DtypeBits("weights",     ">=", 2),      # L276
    DtypeBits("weights",     "<=", 8),      # header rule 8sx*
    Implies(Not(ParamIsNarrow("weights")),  # L266-269: non-narrow ⇒ not DSP48E1
            Not(DspBlockIn(["DSP48E1"]))),
])
```
The infamous **drift bug** — the `idt.bitwidth() >= 4` pre-gate at `L60/L69` disagreeing with the `2 <=` threshold at `L275` — is *structurally impossible* here: there is ONE precondition object; the "prefer-RTL pre-gate" is not a second copy, it IS this object. Census pain-point #3 drift eliminated by construction.

**`_vvu_rtl_possible` (`specialize_layers.py:281-298`):**
```
VVU_RTL.precondition = AllOf([
    AttrEquals("noActivation", 1),          # L287
    DeviceFamily(["versal"]),               # L289
    AnyOf([ DtypeBits("activations","<=",8),                       # L294
            AllOf([DtypeBits("activations","==",9),
                   DtypeSigned("activations",True)]) ]),
    DtypeBits("weights","<=",8),            # L295
    DtypeSigned("weights", True),           # L296
])
```
**`_dwc` (`L221-232`)** → `DWC_RTL.precondition = AnyOf([AttrDivides("inWidth","outWidth"), AttrDivides("outWidth","inWidth")])`.
**`_requant` (`L363-374`)** → `AllOf([DtypeIsInteger("in"), DtypeSigned("out",False), AttrEquals("narrow",0)])`.
**`_layernorm` (`L350-360`)** → `AllOf([DeviceFamily(["versal"]), DtypeIs("in", FLOAT32)])`.
**`_elementwise` broadcast (`L301-347`)** → `AllOf([DeviceFamily(["versal"]), AnyOf([DtypeIs("lhs",FLOAT32),DtypeIs("rhs",FLOAT32)]), Not(All(const,const)), BroadcastCompatible("rhs","lhs","out")])` — the broadcast loop at `L340-345` folds into one `BroadcastCompatible` primitive.

All six op-family feasibility ladders now live *with their realization*, as data, evaluated by one uniform pass. `specialize_layers.py` collapses to `Solver.select()`.

## SelectionContext — the typed, injected world (kills ambient singletons, criterion #10)

```
SelectionContext = {
  input_dtypes, output_dtypes,        # from graph
  attrs: AttrView,                     # typed read-only node attributes
  fpgapart: FpgaPart,                  # .family, .dsp_block, .is_versal  (INJECTED, not is_versal(global))
  param_stats: ParamStats | None,      # min/max/is_narrow of initializers; None ⇒ dynamic
  folding: FoldingView,                # .pe(), .simd(), .scheme  (shared interface, axis 4)
}
```
`fpgapart`, and later `exec_ctx.simulator` (the finnxsi replacement), `build_dir`, `rtllib_root` are all fields of injected context objects — never module globals (`hwcustomop.py:39`, `is_versal` import). A Kernel/Realization is built, tested, and solved from these inputs alone → hermetic.

## The Solver (single, op-agnostic)

```
select(req, pool, ctx):
  semantic_ok  = [r for r in pool if r.semantic == req.semantic]      # replaces optype+"_hls" in dict
  covering     = [r for r in semantic_ok if req.required_traits ⊆ r.provided_traits]
  feasible, rejected = [], []
  for r in covering:
      fails = r.precondition.eval(ctx)        # returns [] or [reason,...]
      (feasible if not fails else rejected).append((r, fails))
  if user set preferred kind as HARD: feasible = filter to that kind
  feasible.sort(key = (pref_honored, r.priority, r.est_cost(ctx), r.id))   # deterministic
  return Binding(feasible[0]) if feasible else Infeasible(rejected_reasons)
```
**Tie-break, fully specified:** (1) user `preferred_impl_style` as soft boost or hard filter; (2) integer `priority` (defaults static-IP=30 > RTL=20 > HLS=10, encoding the current "prefer RTL for simple layers" as *data*, not the `if optype` ladder); (3) `est_cost` from `estimate(binding)`; (4) realization `id` lexicographic for reproducible builds. `explain(select)` returns the full ranked list with every candidate's feasibility verdict and reasons — the debuggability answer to "why HLS here?"

## Composer — weight delivery & IPI stitching as trait-driven port wiring

A `Binding` can be **composite**: `{compute: MVU_RTL, delivery:{weights: Memstream}, params:{...}}`. Delivery realizations (`Memstream`, `Dynload`, `MLOFetch`, `Embedded`) are solved by the *same* solver against the op's `ParamPort` accepted-modes and the device. Each part declares ports (`Delivers` provides `AxiStream(weights_out)`; `Compute` requires `AxiStream(weights_in)`). The **Composer** wires complementary port traits generically. This is what the 236-line `code_generation_ipi` (`matrixvectoractivation.py:920`) becomes: *composition of declared parts by port matching*, in the substrate, naming no op.

## Kernel & Realization contracts

```
Kernel(Protocol):
  semantics() -> Semantic
  require(ctx) -> Requirements          # traits + ports, pure fn of attrs
  reference(ctx) -> Callable | None     # golden model — SEPARATE slot (kills execute_node diamond)
  serialize_param(role, layout) -> bytes  # unified param serialization (thresholds/weights)

Realization(Protocol):
  provides() -> Capabilities            # traits + Precondition(DATA) + priority + artifact_kind
  emit(binding, sink) -> Artifacts      # PURE: typed Binding → files (no code_gen_dict)
  estimate(binding) -> Resources
  run(binding, exec_ctx, inputs) -> outputs   # cppsim/rtlsim, injected simulator
```
`reference` lives on the Kernel object; `run` lives on the Realization object. They are different names on different objects reached by *composition*, so no MRO can collide them (criterion #4). `emit` consumes a typed `Binding` and returns typed `Artifacts` — the op↔rtllib seam is a **`ParamBinding`** (typed key→value with validation + a declared `ArtifactManifest`), replacing 26 untyped `$KEY$` `str.replace` and the triplicated manifests (criterion #8).

## Registration — a new kind, zero substrate edits

```
@realizes(MatMul)                       # appends to module-level RealizationPool
class MVU_INT8_StaticIP(Realization):
  provides = Capabilities(
    traits = { Compute(MatMul), Delivers(weights, embedded),
               ParamStorage(bram, rom), TensorStream(0,in), TensorStream(0,out),
               Folding(PE_SIMD), RunsVia(rtlsim) },
    precondition = AllOf([ DtypeIs("activations", INT8), DtypeIs("weights", INT8),
                           AttrEquals("noActivation",1), AttrLE("MW",512), AttrLE("MH",512),
                           FoldingIs("PE_SIMD") ]),
    priority = 30, artifact_kind = IPCore )
  def emit(self, b, sink): sink.instantiate_vlnv("finn.int8gemm:1.0", b.port_map)
```
Dropping this file into the realization package makes the solver auto-select it for INT8 512×512 GEMM (priority beats RTL) and fall back to RTL/HLS elsewhere. No edit to the solver, the substrate, or the MVAU Kernel. That single property is the proof the abstraction is real.

## Axis treatment

### compute_impl
Axis 1 is the `Compute(semantic)` trait — the primary match key. HLS, RTL, and static-IP are NOT a class axis; each is a `Realization` that provides `Compute(MatMul)` (etc.) with its own `Precondition` and `emit`. 'Kind' is a value (a priority + emit strategy), never a base class. The as-is `(OpBase, HLSBackend)` multiple-inheritance product (hw-backend-model §1) is replaced by two composed data objects (Kernel + bound Realization). Adding/removing a kind = adding/removing a pool entry.

### weight_delivery
Axis 2 is a FIRST-CLASS trait pair: op declares `ParamPort(role, accepted_modes)` (required), delivery implementations declare `Delivers(role, mode)` (provided) with their own precondition. Delivery is selected by the SAME solver, independently of compute, and composed by port-wiring. This directly evicts the weight-delivery vocabulary (memstream/dynload/fetch_weights, calc_wmem) from the substrate (`hwcustomop.py:307/355/407`) into standalone delivery Realizations. RTL-MVU × memstream and HLS-MVU × embedded are independent choices — the orthogonality the 2-axis model could not express (census MVAU F2).

### memory_strategy
Axis 3 is the `ParamStorage(kind, discipline)` trait carried by delivery Realizations: embedded→(lutrom,rom), decoupled→(bram|uram, streamed), external→(offchip, dma). Device limits (uram-only-on-family) are precondition data on the delivery realization, not `mem_mode` try/except in the width getter (`thresholding.py:179`). One shared concept; ops accept a set of storages and the solver picks a feasible one. Thresholding's embedded-vs-decoupled, lookup's ROM-vs-DMA, MVAU's ram_style all instances of the same trait.

### folding
Axis 4 is `Folding(scheme)` — a SHARED INTERFACE ONLY, per §9.3. The `FoldingView` in SelectionContext exposes `pe()`, `simd()`, `scheme` uniformly, and the trait matches on scheme NAME (`PE_SIMD`, `SIMD_only`, `channel`, `window`, `none`). A realization requires a compatible scheme (static-IP GEMM requires `Folding(PE_SIMD)`). The per-op folding MATH stays inside each Kernel's `require`/`reference` and is explicitly OUT OF SCOPE for v1 — the solver checks scheme-name compatibility (a nominal match), not the arithmetic. Deep unification deferred; I deliberately do not invest here.

### ports
Axis 5: every port is a trait — tensor (`TensorStream(i,dir)`) AND non-tensor (`AxiLite(regs)`, `AxiMM(w,dir)`, `Sideband(tlast|tkeep)`, `ApNone(sig)`). Ports are declared per-port on both sides and are the substrate for the Composer's wiring. Shape/width accessors are defined PER-PORT: an `AxiMM` port simply does not carry a `TensorStream`, so folded-shape is never queried on it — iodma's `raise` (`iodma_hls.py:116/131`) becomes a non-event. checksum AxiLite, tlastmarker Sideband, fmpadding AXI-lite regs all first-class, typed.

### execution
Axis 6 splits into two SEPARATE contract slots on two SEPARATE objects: `Kernel.reference(ctx)` owns golden semantics (numpy/torch/onnxruntime model); the bound `Realization.run(binding, exec_ctx)` owns cppsim/rtlsim, with `RunsVia(mode)` declaring which sims it supports. Because they are different names on different composed objects, no MRO diamond can conflate them (fixes the ~8 execute_node shims and the VVAU indent bug `vectorvectoractivation_rtl.py:89`). `exec_ctx.simulator` injects finnxsi (fakeable), killing the import-time singleton (`hwcustomop.py:39`).

## Conformance sketches

### mvau
Kernel `MatMul`: `semantics()=MatMul`; `require(ctx)={Compute(MatMul), ParamPort(weights,{embedded,decoupled,external,dynamic,mlo}), ParamPort(thresholds,{embedded})?, TensorStream(0,in), TensorStream(0,out), Folding(PE_SIMD)}`; `reference(ctx)=`numpy matmul+xnorpopcount+multithreshold (the genuinely-clean part per census, kept whole). Pool: compute realizations `MVU_RTL` (precondition = the data-ized `_mvu_rtl_possible` shown in core_model) and `MVU_HLS` (broad precondition); delivery realizations `Embedded`/`Memstream`/`Dynload`/`MLOFetch` each with a Delivers trait + precondition. The solver picks compute (RTL if precondition holds, else HLS) AND, per the chosen delivery mode, a delivery realization — two applications of ONE mechanism. The Binding is composite; the Composer wires `Memstream.weights_out ⟷ MVU.weights_in` by matching AxiStream port traits. The 236-line `code_generation_ipi` at `matrixvectoractivation.py:920` (which calls subclass-only `instantiate_ip` — the uninstantiable-by-contract base) is REPLACED by generic port-composition in the substrate: no base method, no `self.instantiate_ip`. The base-class leak `hwcustomop.py:307/355/407` (op_type allowlist) is gone because memstream is a Realization selected by traits, not a substrate branch. 40 overrides → one `require` + one `reference` + per-realization `emit`.

### thresholding
Memory-strategy is the orthogonal axis, not `mem_mode` try/except. Kernel `Threshold`: `require={Compute(Threshold), ParamPort(thresholds,{embedded,decoupled}), TensorStream(0,in/out), Folding(channel)}`. Two delivery realizations provide `Delivers(thresholds,embedded)` (emits thresh.h) and `Delivers(thresholds,decoupled)`+`ParamStorage(bram|uram,streamed)` (emits memstream). The solver chooses; `get_instream_width` (`thresholding.py:173`) is a pure fn of the bound delivery trait, never sniffs `mem_mode` — the base cross-backend leak at `thresholding.py:179/310` is deleted. Threshold serialization is UNIFIED: ONE `Kernel.serialize_param(thresholds, layout)` parameterized by a `layout` value the realization requests (`binary_search_sorted` for the RTL core's sorted-address requirement `thresholding_rtl.py:457`, `linear` for HLS). The three divergent `make_weight_file`/`minimize_weight_bit_width` copies (`thresholding.py:127` vs `_hls.py:781` vs `_rtl.py:566`) collapse to one function + a layout parameter carried as a trait.

### finn_loop
First-class container, not a leaf-kind category error. Kernel `Loop`: `semantics()=Container`; `require(ctx)={Container(body=Subgraph), <ports delegated from body boundary>}` — NO ParamPort, NO Compute. `reference(ctx)`=recursive execute_onnx over the body (legal: containers have reference semantics). Its Realization provides `{Container(Subgraph), RunsVia(rtlsim)}` and `artifact_kind = BlockDesign`. The `get_rtl_file_list`→None stub (`finn_loop.py:1175`) vanishes because the contract admits multiple artifact kinds (`FileList | BlockDesign | IPCore`); a Container emits a BlockDesign, and `prepare_rtlsim` consumes that artifact kind — no nominal-only abstract to dodge. The child-op-type dispatch (`finn_loop.py:421`, `else: raise`) is replaced by reading each child's DECLARED delivery ports/`ParamPort` traits and composing via the Composer — the loop never string-matches `MVAU`/`Thresholding`/`Elementwise`; it composes over declared port traits, so a new child op works with zero loop edits.

### iodma
Infra kernel, honest ports. Kernel `IODMA`: `semantics()=DataMove`, trait `Infra` ⇒ `reference()` may be None (no functional model REQUIRED — legal for Infra, so `execute_node=pass` at `iodma_hls.py:388` is no longer a contract violation). `require(ctx)` bifurcates on the `direction` attr into port sets: in ⇒ `{AxiMM(intfWidth,in), TensorStream(0,out)}`; out ⇒ `{TensorStream(0,in), AxiMM(intfWidth,out)}`. Shape/width accessors are per-port: the AxiMM side carries no `TensorStream`, so folded-shape is never queried there — the `raise ValueError` at `iodma_hls.py:116/131` becomes structurally unreachable, not a landmine for generic passes. The width-conversion (LCM DWC-chaining) is itself a composed sub-Realization (a DWC delivery), reusable by any backend instead of welded into HLS `docompute` strings.

### static_ip_gemm
The forcing function, satisfied by ONE new file. `@realizes(MatMul) class MVU_INT8_StaticIP` provides `{Compute(MatMul), Delivers(weights,embedded), ParamStorage(bram,rom), TensorStream(0,in), TensorStream(0,out), Folding(PE_SIMD), RunsVia(rtlsim)}` with `precondition = AllOf([DtypeIs(act,INT8), DtypeIs(weights,INT8), AttrEquals(noActivation,1), AttrLE(MW,512), AttrLE(MH,512), FoldingIs(PE_SIMD)])`, `priority=30`, `artifact_kind=IPCore`, and `emit` = instantiate the pre-synth IP by VLNV. Because it registers into the pool and the solver ranks by priority, it is AUTO-SELECTED for the INT8 512×512 common case (beating MVU_RTL=20) and the solver transparently falls back to RTL/HLS when its precondition fails. ZERO edits to the substrate (criterion #2), and it is chosen by the same declared-capability mechanism as everything else (criterion #3). It is a peer to HLS/RTL MVU under one contract — proving the abstraction is not a rename of the {HLS,RTL} split.

## Criteria self-assessment (must-fix 1-10)

| # | verdict | how |
|--|--|--|
| 1 | **yes** | Weight delivery is the `ParamPort`/`Delivers` trait pair, an independent axis selected by the solver and composed by port-wiring. The 236-line IPI (`matrixvectoractivation.py:920`) becomes generic Composer port-matching; the substrate weight-delivery leak (`hwcustomop.py:307/355/407`) is deleted — memstream/dynload/MLO are standalone Realizations. |
| 2 | **yes** | The substrate speaks only Trait/Requirements/Capabilities/Precondition/Binding. It contains no op_type allowlist and no HLS/RTL class axis. Adding/removing an op or a kind edits only data pools + one file, never shared code. The static-IP GEMM proves it: zero substrate edits. |
| 3 | **yes** | This IS the prior. `specialize_layers.py:40-211` god-switch and every `_*_rtl_possible` ladder become `Precondition` DATA owned by each Realization, evaluated by one op-agnostic `Solver.select`. Feasibility is queried through a declared contract, not asserted by a central switch. The `L60/L275` drift bug is structurally impossible (one precondition object). |
| 4 | **yes** | `Kernel.reference(ctx)` (golden semantics) and bound `Realization.run(binding, exec_ctx)` (cppsim/rtlsim) are separate named slots on separate composed objects. No inheritance, no MRO, so no diamond. Fixes the ~8 execute_node shims and the VVAU rtlsim indent bug (`vectorvectoractivation_rtl.py:89`). |
| 5 | **yes** | `emit(binding, sink)` is a pure function of a typed `Binding` → typed `Artifacts`. No shared mutable `code_gen_dict` (`hlsbackend.py:136`), no ordered side-effecting steps. Each realization's emit is independently unit-testable from a synthetic Binding. |
| 6 | **yes** | `ParamStorage(kind, discipline)` is one memory-strategy concept shared across MVAU ram_style, thresholding embedded/decoupled, lookup ROM/DMA. Device limits are precondition data, not `mem_mode` try/except re-branched per op (`thresholding.py:179`, `vectorvectoractivation.py:617`). |
| 7 | **yes** | `Infra` trait makes `reference()` optional (iodma/checksum/tlastmarker need no functional model — `iodma_hls.py:388` no longer violates). `Container` semantics + `artifact_kind ∈ {FileList,BlockDesign,IPCore}` admits finn_loop honestly — the `get_rtl_file_list`→None stub (`finn_loop.py:1175`) is gone. Per-port shapes make iodma's folded-shape `raise` (`iodma_hls.py:116`) unreachable. |
| 8 | **yes** | `emit` consumes a typed `ParamBinding` (key→value with validation) and a declared `ArtifactManifest`, replacing 26 untyped `$KEY$` str.replace and the triplicated per-op source manifests (`fmpadding_rtl.py:137/153/165`). Magic-number-in-.v overrides (layernorm `[31:0]`, requant 6-decimal) are caught because dtype values flow through the typed binding, not spelled into templates. |
| 9 | **yes** | StreamingFIFO's hidden 'vivado' backend (`streamingfifo_rtl.py:141`) becomes TWO Realizations providing `Fifo` — `QSrl_RTL` and `VivadoAxisFifo` — each with its own precondition; the solver picks. No runtime `impl_style` switch inside one class, no base try/except reach-through (`streamingfifo.py:90/104`). All kinds are explicit pool peers. |
| 10 | **yes** | `SelectionContext` (fpgapart, param_stats, folding) and `exec_ctx` (simulator, build_dir, rtllib_root) are injected records — no `finnxsi` import-time singleton, no `is_versal` global, no ambient `FINN_ROOT`. A Kernel/Realization is solved, emitted, and run from declared inputs alone. Cross-op couplings (SWG↔VVAU layout) are composed via declared port traits, not sibling introspection. |

## Weaknesses (self-declared)

Honest costs of committing fully to constraint-solved selection: (1) SOLVER OPACITY — 'no feasible realization' can be harder to trace than a linear if-ladder; I mitigate with per-constraint reason strings and `explain(select)` returning the ranked candidate list with verdicts, but the indirection is real and developers must learn to read solver traces. (2) TRAIT VOCABULARY IS A GOVERNANCE BOTTLENECK — the Trait set is itself a shared contract; a genuinely novel capability (new delivery mode, new port kind) requires extending the central vocabulary, a coordinated change. The design moves coupling from code (op_type strings) to vocabulary (trait values). Within the 6 axes this is bounded, but it is not free. (3) PRECONDITION-AS-DATA EXPRESSIVENESS CEILING — most predicates re-express cleanly, but ~2 of 6 (elementwise broadcast loop `specialize_layers.py:340-345`; MVU narrow-from-initializer-values `L259-266`) need first-class composite primitives (`BroadcastCompatible`, `ParamIsNarrow`) or a `Predicate(named_fn)` escape hatch; every escape hatch is a small re-opening of the god-switch, so the vocabulary must actively grow to keep them as data. (4) FOLDING IS ONLY A NOMINAL MATCH — per §9.3 I scope out unified math, so `FoldingIs(PE_SIMD)` trusts the Kernel's folding arithmetic is compatible with what the realization assumes; a real mismatch (op folds differently than the realization's assumption) is NOT caught by the solver in v1. (5) COMPOSITION AMBIGUITY — if two delivery realizations provide overlapping `Delivers(weights)` with overlapping preconditions, wiring must be deterministic; I lean on priority+id, but a truly novel composite (two weight ports, or a wiring topology the port traits under-specify) needs extra port-identity ceremony. (6) SOLVE COST — per-node constraint solving over a pool is costlier than a dict lookup (negligible at compile scale, but non-zero).

## Prior fidelity

This design refuses the generic 'composition' middle and commits to the distinctive engine of the capability/trait prior: a SOLVER over TRAITS + PRECONDITION-DATA. There is no HLS/RTL inheritance axis anywhere — 'kind' is a value (priority + emit strategy) attached to a Realization, and the concrete op is Kernel ∘ bound-Realization, not `(OpBase, Backend)`. Selection is literally constraint satisfaction: I re-expressed ALL SIX god-switch feasibility functions (`_mvu_rtl_possible`, `_vvu_rtl_possible`, `_dwc`, `_requant`, `_layernorm`, `_elementwise`) as declarative `Precondition` values, showing the MVU and VVU ones concretely with file:line correspondence to `specialize_layers.py:235-360`, and demonstrated that the L60/L275 drift bug becomes structurally impossible. I chose SEPARATE-PASS-QUERIES deliberately (not self-advertising): the op declares only REQUIREMENTS (its own business), each realization declares CAPABILITIES + feasibility AS DATA (its own business), and a single op-agnostic solver resolves them — feasibility is never owned by the op, so no per-op `is_feasible()` method scatters back into a switch. The static-IP GEMM (`@realizes` + one precondition + priority=30) proves zero-substrate-edit registration and auto-selection, the forcing function the prior is built to pass. Weight-delivery is shown as the SAME trait-solving mechanism applied at a second axis, unifying the god-switch problem and the MVAU-instantiation-mess under one idea.
