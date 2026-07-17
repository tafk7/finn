# Design: Explicit multi-axis / product type

> **Essence:** A concrete kernel is a typed coordinate in the product space Compute × WeightDelivery × Memory × Folding × Ports × Execution — legal points are a declared subset of the cartesian product, and every artifact is produced by one axis-generic assembler that never names an op or a kind.

**Selection strategy:** `hybrid`

## Core model

## The Kernel = OpContract × Coordinate

A kernel is not a class in an inheritance lattice. It is a **pair**:

```
Kernel  =  OpContract  ×  Coordinate
Coordinate  =  (Compute, WeightDelivery, Memory, Folding, Ports, Execution)
```

- **`OpContract`** carries the op's *semantics* and nothing about hardware: the reference model, the datatype/shape identities, the parameter tensors. It is what makes an MVAU an MVAU regardless of how it is realized. It is written once per op family and is immutable across coordinates.
- **`Coordinate`** is a point in a 6-dimensional typed product space. Each axis is an algebraic sum type whose *values* each expose a small, uniform interface. A kernel instance is literally `OpContract + one value drawn from each axis`.

The as-is system (`hw-backend-model.md` headline #1) is a **2-axis** system: `(OpBase, Backend∈{HLS,RTL})`. Its fatal move was collapsing weight-delivery and memory-strategy onto that single `Backend` inheritance axis, so MVAU's `mem_mode × dynamic_input × mlo_max_iter` explosion had to be *smeared across two backend classes and the substrate* (`matrixvectoractivation.py:920`, `hwcustomop.py:307/355/407`). Here those are **three independent axes**; MVAU's explosion becomes three independent coordinate choices.

```
                          OpContract  (semantics only — one per op family)
                          ├─ reference_model(inputs)        ← pure golden fn
                          ├─ param_tensors()                ← weights/thresholds, untyped-to-HW
                          ├─ datatypes(port_role)           ← dtype identities
                          └─ normal_shape(port_role)        ← unfolded shapes
                                        │
                                        │  combined with a Coordinate:
   ┌────────────┬───────────────┬─────────────┬───────────┬───────────┬────────────┐
   │ Compute    │ WeightDelivery│ Memory      │ Folding   │ Ports     │ Execution  │
   │ (axis 1)   │ (axis 2)      │ (axis 3)    │ (axis 4)  │ (axis 5)  │ (axis 6)   │
   ├────────────┼───────────────┼─────────────┼───────────┼───────────┼────────────┤
   │ HLS        │ NoParams      │ NoMem       │ PeSimd    │ TensorStrm│ {CppSim}   │
   │ RTL        │ Embedded      │ LUTROM      │ (op-spec  │ AxiLite   │ {RtlSim}   │
   │ StaticIP   │ DecoupledStrm │ BRAM(style) │  impl     │ AxiMM     │ (Reference │
   │ Container  │ External      │ URAM        │  behind   │ Sideband  │  always on │
   │ Passthrough│ DynamicLoad   │ OffChipDMA  │  1 iface) │ Clock     │  OpContract│
   │            │ LoopFetched   │             │           │           │  — axis 6  │
   │            │               │             │           │           │  = harness │
   │            │               │             │           │           │  SET)      │
   └─────┬──────┴──────┬────────┴──────┬──────┴─────┬─────┴─────┬─────┴─────┬──────┘
         └─────────────┴───────────────┴────────────┴───────────┴───────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │   Assembler.realize(k)      │  ← ONE function.
                          │   axis-generic; never sees  │    Never branches on
                          │   an op-type or a kind.     │    op or kind (fixes #2)
                          └─────────────┬──────────────┘
                                        ▼
                      Artifact  =  FlatSources | BlockDesign | VendorIP
                                   + PortSet + MemInit + StitchFragments
```

### Each axis is a typed sum with a small uniform interface

Every axis value is an object implementing that axis's interface. The interfaces are the *entire* contract; there is no undeclared host-object surface (the disease of `core-interfaces.md §2`, `rtlbackend.py:188`).

**Axis 1 — `Compute` (how the math is realized).**
```
interface Compute:
    feasible(op: OpContract, target: Target) -> Feasibility   # self-advertise (crit #3)
    core_ports() -> PortSet                                   # the compute core's own ports
    emit_compute(ctx: RealizeCtx) -> ComputeArtifact          # PURE fn (crit #5)
    stitch() -> StitchFragment                                # how this core wires in
    resources() -> ResourceModel
values: HLS(hls_desc) | RTL(rtl_desc) | StaticIP(ip_manifest)
      | Container(subgraph_ref) | Passthrough
```
`emit_compute` takes a *typed* `RealizeCtx` (fpgapart, clk, build_dir, injected simulator, resolved rtllib root) and returns a `ComputeArtifact`. No `self.code_gen_dict` mutation, no ordered side-effecting steps (`hlsbackend.py:136`). `HLS.emit_compute` returns C++ text as data; `RTL.emit_compute` returns a `RtlParamMap` bound against a declared source manifest (the typed replacement for 26 `$KEY$` `str.replace`, `hw-backend-model.md §4.3a`); `StaticIP.emit_compute` returns a `VendorIP` VLNV reference with *no codegen at all*.

**Axis 2 — `WeightDelivery` (how coefficients *move* to the compute).** This is the axis the as-is has no slot for (`hw-backend-model.md` Pattern F2); it is the single largest source of fusion.
```
interface WeightDelivery:
    serialize(params: ParamTensors, layout: ParamLayout) -> ParamArtifact
    delivery_ports() -> PortSet          # weight stream / axilite / aximm ports it adds
    emit_delivery(ctx, mem: MemInit) -> DeliveryFragment   # the streamer/DMA/nothing
    stitch() -> StitchFragment
values: NoParams | Embedded | DecoupledStream | External | DynamicLoad | LoopFetched
```
`DecoupledStream.emit_delivery` is the *one* place the memstream wrapper is emitted — evicting `HWCustomOp.generate_hdl_memstream/fetch_weights/dynload` (`hwcustomop.py:307/355/407`) and its op-type allowlist entirely. `LoopFetched.emit_delivery` is MLO/`fetch_weights`; `DynamicLoad` is dynload. Crucially, **delivery is independent of Compute**: HLS-compute + DecoupledStream compose without the HLS op ever emitting RTL (killing the 13 cross-backend leaks, `matrixvectoractivation_hls.py:144`).

**Axis 3 — `Memory` (where params *live* + read discipline).** Delivery is *motion*; Memory is *storage substrate*. Their conflation is the `mem_mode × ram_style` mess.
```
interface Memory:
    geometry(param_shape, dtypes) -> MemGeometry     # depth/width/sets
    emit_meminit(params: ParamArtifact) -> MemInit    # .dat / .hpp / .npy — one place
    resources() -> ResourceModel
values: NoMem | LUTROM | BRAM(ram_style) | URAM | OffChipDMA
```
The old `mem_mode`/`ram_style`/`mlo` product now factors cleanly: `Embedded×LUTROM` = params.h array; `DecoupledStream×BRAM` = memstream on BRAM; `DecoupledStream×URAM` = `ram_style=ultra`; `LoopFetched×OffChipDMA` = MLO. Each `bram_estimation`/`uram_estimation` (`matrixvectoractivation.py:387/365`) lives on the Memory value, not gated inside the op.

**Axis 4 — `Folding` (parallelism).** Per §9.3, a **shared interface only**, math deliberately *not* unified:
```
interface Folding:
    fold_params() -> dict          # {PE, SIMD, ...} declared vocabulary
    folded_shape(normal, role) -> shape
    stream_width(dtype, role) -> int
    exp_cycles() -> int
```
`PeSimd` is the common value; op-specific folding (SWG windowing, pool reduction) is a distinct value implementing the *same* interface with its own math. The 8 `HWCustomOp` shape/width getters (`hwcustomop.py:261-289`) are *derived* from `Folding × OpContract.normal_shape` — they stop being abstract methods every op re-implements.

**Axis 5 — `Ports` (tensor AND non-tensor, first-class typed).**
```
Port = TensorStream(dir, dtype, folded_shape)
     | AxiLite(regmap)
     | AxiMM(addr_bits, data_width, role)      # role ∈ {source, sink}
     | Sideband(kind)                          # TLAST | TKEEP
     | Clock(rate)                             # clk / clk2x — a detail, not a 7th axis (§9.4)
PortSet = union of ports contributed by Compute.core_ports ∪ WeightDelivery.delivery_ports
```
Only `TensorStream` has a `folded_shape`. A generic pass iterates `port.tensor_shape()` over `TensorStream` variants *only* — so `iodma`'s AXI-MM side never "raises" (`iodma_hls.py:116/131`): it is simply an `AxiMM` port, a different variant with no tensor shape *by type*. This is where the non-tensor port dimension (checksum axilite, tlastmarker sideband, fmpadding regmap) becomes honest.

**Axis 6 — `Execution` (reference vs cppsim vs rtlsim, as distinct slots).** This axis is what dissolves the `execute_node` diamond (`hw-backend-model.md §1 "clean seams" wart`, the single most-repeated census defect, and the latent VVAU bug `vectorvectoractivation_rtl.py:89`).
```
# Reference semantics live on OpContract — ALWAYS present, one slot:
OpContract.reference_model(inputs) -> outputs      # pure golden fn

# Run harnesses are Execution-axis values — a SET the coordinate supports:
interface RunHarness:
    run(ports: PortSet, inputs, ctx.sim) -> outputs
values: CppSim | RtlSim
```
`reference_model` (a field on `OpContract`) and `RunHarness.run` (values on axis 6) are **different fields on different objects** — they can never collide under MRO because there is no shared `execute_node` name and no multiple-inheritance diamond. The rtlsim harness is *axis-generic*: it drives the kernel purely through its declared `PortSet`, so there is one `RtlSim.run`, not one hand-copied rtlsim block per op (`matrixvectoractivation_rtl.py:100` duplicating `_hls.py:569`). The simulator is `ctx.sim` — injected, not the `finnxsi` import-time singleton (`hwcustomop.py:39`).

### Realization: coordinate → artifacts (the axis-generic Assembler)

```
def realize(k: Kernel, ctx: RealizeCtx) -> Artifact:
    layout   = k.compute.param_layout()                 # e.g. LinearLayout | SortedBinSearchLayout
    params   = k.weight_delivery.serialize(k.op.param_tensors(), layout)
    mem      = k.memory.emit_meminit(params)
    compute  = k.compute.emit_compute(ctx)
    delivery = k.weight_delivery.emit_delivery(ctx, mem)
    ports    = k.compute.core_ports() | k.weight_delivery.delivery_ports()
    stitch   = Stitcher.compose([k.compute.stitch(), k.weight_delivery.stitch(),
                                 k.memory.stitch_hint()], ports)
    return Artifact(compute, delivery, mem, stitch, ports)
```
`realize` calls **only axis interfaces**. It contains no `if op_type == ...`, no `isinstance(kind)`, no allowlist. Adding a new op = a new `OpContract`. Adding a new implementation kind = a new `Compute` value. Neither edits `realize`, the substrate, or any other op — this is criterion #2 satisfied by construction. The 236-line `code_generation_ipi` (`matrixvectoractivation.py:920`) is **replaced by `Stitcher.compose` over per-axis `StitchFragment`s** — composition of declared parts, not a base-class method that reaches into subclasses.

### Legality: the product is a *declared subset* of the cartesian product

Not every coordinate is buildable. Illegality is expressed at two typed levels, and an illegal coordinate is **unconstructable** — the `Kernel` smart-constructor returns `IllegalCoordinate(axis_or_pair, reason)` rather than a broken object.

1. **Local (per-value) preconditions** — `Compute.feasible(op, target)`. Example: `RTL(mvu)` for MVAU advertises `noActivation ∧ signed_weights ∧ dtype∈2..8bit ∧ dsp_capable(target)` — the exact predicate that is *today* hard-coded in the `specialize_layers.py:235` god-switch (`_mvu_rtl_possible`), now **co-located with the RTL value** where drift with the inline `>=4` pre-gate (`specialize_layers.py:60` vs `:275`) is impossible because there is one copy.

2. **Relational (pairwise) legality** — a small declared relation `Legal ⊆ Axis_i × Axis_j`:
   - `Embedded` ⇒ Memory ∈ {LUTROM, BRAM}, forbids `OffChipDMA`.
   - `LoopFetched` ⇒ Memory = `OffChipDMA` (MLO must be off-chip).
   - `StaticIP` compute ⇒ WeightDelivery ∈ {Embedded, External} (the hard IP owns its weight port); Ports pinned to the IP's advertised set.
   - `Container` compute ⇒ WeightDelivery = `NoParams` at the container level (params belong to child coordinates).

This declarative relation is what replaces the scattered `try/except mem_mode` sniffing (`thresholding.py:179/310`), the `startswith('Elementwise')` base branches, and the `impl_style` reach-through (`streamingfifo.py:90/104`). The `mem_mode × dynamic × mlo` combinatorial "mess" is now just: *enumerate the legal coordinates; the illegal ones are not in the set.*

### Selection = constraint resolution over the product space

Given `OpContract` + `Target` + user pins (`preferred_impl_style` pins the Compute axis; a runtime-writeable flag pins Ports):
1. Gather, per axis, the values whose local `feasible` passes.
2. Take the cartesian product of survivors; prune by the pairwise `Legal` relation.
3. Rank the surviving legal coordinates by a declared **policy** (prefer RTL compute for simple layers; prefer BRAM over URAM; prefer `StaticIP` for its common case) and pick the top.

`specialize_layers.py:40-211` — the per-op RTL-feasibility ladder + `optype+"_"+impl_style` string concat — collapses into this one op-agnostic resolver. There is no `if optype == "MVAU"` arm anywhere; the resolver only reads declared `feasible` results and the `Legal` relation.

## Axis treatment

### compute_impl
Axis 1 `Compute`, a sum type `HLS | RTL | StaticIP | Container | Passthrough`, each a value implementing `{feasible, core_ports, emit_compute, stitch, resources}`. `emit_compute(ctx) -> ComputeArtifact` is a PURE function of a typed `RealizeCtx` — no `code_gen_dict` side-channel (fixes hlsbackend.py:136). HLS returns C++ as data; RTL returns a typed `RtlParamMap` bound to a declared source manifest (replaces 26 `$KEY$` str.replace, hw-backend-model §4.3a); StaticIP returns a `VendorIP` VLNV with no codegen; Container returns a `BlockDesign`. Kinds are peer VALUES on one axis, never subclasses — so 'the vivado FIFO third backend' (streamingfifo_rtl.py:141) is just another value, not a hidden runtime branch.

### weight_delivery
Axis 2 `WeightDelivery` = `NoParams | Embedded | DecoupledStream | External | DynamicLoad | LoopFetched`. This is the axis the as-is has NO slot for (Pattern F2). It exposes `serialize(params, layout) -> ParamArtifact`, `delivery_ports()`, `emit_delivery(ctx, mem) -> DeliveryFragment`, `stitch()`. `DecoupledStream.emit_delivery` is the SINGLE site that emits the memstream wrapper — evicting `HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` (hwcustomop.py:307/355/407) and its op-type allowlist. It is fully orthogonal to Compute: HLS-compute + DecoupledStream compose without the HLS op emitting any RTL, killing the 13 cross-backend leaks. MVAU's embedded/decoupled/external/dynamic/MLO become 5 values on this ONE axis, not 40 overrides across two backend classes + the base.

### memory_strategy
Axis 3 `Memory` = `NoMem | LUTROM | BRAM(ram_style) | URAM | OffChipDMA`, distinct from WeightDelivery: delivery is the *motion*, memory is the *storage substrate*. Interface `{geometry, emit_meminit, resources}`. The old `mem_mode × ram_style × mlo` product factors: `Embedded×LUTROM`=params.h; `DecoupledStream×BRAM`=memstream-on-BRAM; `DecoupledStream×URAM`=ram_style=ultra; `LoopFetched×OffChipDMA`=MLO. `bram_estimation`/`uram_estimation` (matrixvectoractivation.py:387/365) and threshold `.dat` emission move onto the Memory value, so thresholding/lookup/MVAU stop each re-branching mem_mode independently (crit #6).

### folding
Axis 4 `Folding` — SHARED INTERFACE ONLY per §9.3, math NOT unified. Interface `{fold_params()->{PE,SIMD,...}, folded_shape(normal,role), stream_width(dtype,role), exp_cycles()}`. `PeSimd` is the common value; SWG-windowing and pool-reduction are distinct values implementing the SAME interface with their own math. The 8 `HWCustomOp` shape/width getters (hwcustomop.py:261-289) are DERIVED from `Folding × OpContract.normal_shape`, so they stop being per-op abstract re-implementations. Explicitly out of scope for v1: a unified folding algebra across ops — the product point's Folding component remains the least-orthogonal, most op-specific axis, and I do not over-invest there.

### ports
Axis 5 `Ports`, a typed sum `TensorStream(dir,dtype,folded_shape) | AxiLite(regmap) | AxiMM(addr_bits,data_width,role) | Sideband(kind) | Clock(rate)`. A kernel's `PortSet` is the UNION of `Compute.core_ports ∪ WeightDelivery.delivery_ports` — composition, not a monolithic getter. Only `TensorStream` carries a folded shape, so a generic shape pass iterates `TensorStream` variants only and iodma's AXI-MM side never raises (fixes iodma_hls.py:116/131) — it is an `AxiMM` variant with no tensor shape BY TYPE. checksum's axilite reg, tlastmarker's TLAST/TKEEP, fmpadding's regmap, SWG's axilite-reconfig are all first-class Port variants. clk2x is a `Clock(rate)` port variant — a detail inside this axis, NOT a 7th axis (§9.4).

### execution
Axis 6 `Execution` splits the conflated `execute_node`. REFERENCE semantics are a field on `OpContract` — `reference_model(inputs)->outputs`, always present, ONE slot. RUN harnesses are Execution-axis VALUES `CppSim | RtlSim`, each `run(ports, inputs, ctx.sim)->outputs`; the coordinate declares the SET it supports. Because reference (an OpContract field) and harness (an axis value) are different fields on different objects, there is no shared `execute_node` name and no MRO diamond — dissolving the most-repeated census defect and the latent VVAU indent bug (vectorvectoractivation_rtl.py:89). `RtlSim.run` is axis-generic, driven purely by the declared `PortSet`, so there is ONE rtlsim harness, not per-op copies (matrixvectoractivation_rtl.py:100 vs _hls.py:569). `ctx.sim` is injected, not the finnxsi import-time singleton (hwcustomop.py:39).

## Conformance sketches

### mvau
OpContract(MVAU): `reference_model` = numpy matmul (+ xnorpopcount for bipolar) then multithreshold with NHWC↔NCHW transpose — the genuinely-clean part of matrixvectoractivation.py, kept once. The classic decoupled-BRAM HLS MVAU is the coordinate `(Compute=HLS, WeightDelivery=DecoupledStream, Memory=BRAM, Folding=PeSimd, Ports={in0,out0,weight_stream}, Execution={CppSim,RtlSim})`. The 236-line `code_generation_ipi` (matrixvectoractivation.py:920) that lives in the agnostic base and calls subclass-only `instantiate_ip` is REPLACED by `Stitcher.compose([HLS.stitch(), DecoupledStream.stitch(), BRAM.stitch_hint()], ports)` — composition of declared per-axis StitchFragments, with the IP-cell creation being `Compute.emit_compute` output, not a base method reaching into leaves. The 5 weight-delivery modes are 5 VALUES on axis 2, each combined with a Memory value: embedded=`(HLS,Embedded,LUTROM,...)`; external=`(HLS,External,...)`; dynamic=`(RTL,DynamicLoad,BRAM,...)`; MLO=`(RTL,LoopFetched,OffChipDMA,...)`. Each is ONE coordinate change per axis — the 40 overrides, the pumpedCompute try/except leak (matrixvectoractivation.py:892), the hwcustomop allowlist, and the mlo_max_iter threading through ~12 sites (hwcustomop.py:100) all vanish. clk2x/pumped is a `Clock(rate)` Port variant.

### thresholding
OpContract(Thresholding): `reference_model` = qonnx multithreshold golden, kept once and reused by every coordinate (no `execute_node` re-dispatch shim). mem_mode embedded-vs-decoupled is DECOMPOSED into two axes: `Memory ∈ {LUTROM, BRAM}` × `WeightDelivery ∈ {Embedded, DecoupledStream}` — so the `try/except mem_mode` sniffing in the agnostic base (thresholding.py:179/310) is gone; the base never sees a delivery/memory concept. The three divergent copies of `make_weight_file`/`minimize_weight_bit_width` (thresholding.py:127 / _hls.py:781 / _rtl.py:566) unify into ONE `WeightDelivery.serialize(params, layout)` parameterized by a `ParamLayout` VALUE: RTL uses `SortedBinSearchLayout` (encoding the sorted-threshold binary-search address map), HLS uses `LinearLayout`. The HLS-decoupled-emits-RTL-streamer anomaly (crit fused-no-seam) disappears: the streamer is emitted by `DecoupledStream.emit_delivery` regardless of Compute, so HLS-compute + decoupled-delivery compose with zero cross-backend leak.

### finn_loop
`Compute=Container(subgraph_ref)` is a first-class value, not a stub. `get_rtl_file_list`→None (finn_loop.py:1175) becomes honest: the Artifact type is a sum `FlatSources | BlockDesign | VendorIP`, and `Container.emit_compute` returns `BlockDesign` — it was never supposed to return a flat file list, so there is no nominal-only contract violation and no `prepare_rtlsim` override to dodge a broken abstract. Ports = `{TensorStream(in0), TensorStream(out0), AxiMM(HBM, role=source), Sideband/AxiLite(done_if)}` — all first-class variants. The 8 shape getters legitimately DELEGATE to the subgraph's boundary child kernels via the Folding/OpContract of those children — delegation is a valid `Container` behavior, not a category error. The sibling-op coupling — dispatching on child op_type for `.dat` naming (finn_loop.py:421/712) — is refactored: each child kernel exposes its `ParamArtifact` through `WeightDelivery.serialize`, and the Container reads that DECLARED interface, never hard-coded filenames or `mem_mode` reach-through. The container composes child COORDINATES; it does not reach into child privates.

### iodma
`Compute=Passthrough`; `OpContract.reference_model` = identity (an HONEST identity model, replacing `execute_node: pass` at iodma_hls.py:388, so cppsim data-flow through an IODMA is observable, not a silent no-op). The `direction` bifurcation is expressed as two Port configurations, NOT a per-method branch: dir=in ⇒ Ports `{AxiMM(role=source), TensorStream(out)}`; dir=out ⇒ Ports `{TensorStream(in), AxiMM(role=sink)}`. The folded-shape getters that RAISE today (iodma_hls.py:116/131) are eliminated by TYPING: the AXI-MM side is an `AxiMM` Port variant with no tensor shape, and generic passes iterate `TensorStream` variants only — nothing raises. The LCM DWC-chaining width adaptation is a `Folding`/width-adapter concern expressible independent of HLS, so a future RTL IODMA reuses it as a value rather than reimplementing template-string surgery. iodma is thus a first-class infra kernel with an honest shape/port contract.

### static_ip_gemm
The forcing function, and the cleanest proof of the model. Add ONE new value to axis 1: `StaticIP(ip_manifest)`. `StaticIP.emit_compute` returns a `VendorIP` (VLNV reference, zero codegen). `StaticIP.feasible(op, target)` = `op is matmul ∧ dtype=INT8×INT8 ∧ shapes ∈ ip.supported_range ∧ target.family ∈ ip.families`. Legality relation adds: `StaticIP ⇒ WeightDelivery∈{Embedded,External}` and `Ports = ip.advertised_ports`. SELECTION: the resolver, for an INT8 GEMM whose preconditions match, finds `StaticIP` in the survivors and the policy ranks it FIRST for its common case → it is auto-selected as a peer to HLS/RTL, satisfying the same OpContract, PortSet, and Folding interface (PE/SIMD map to the IP's advertised parallelism). Substrate edits required: the Assembler `realize` — unchanged; `HWCustomOp`/substrate — nonexistent-by-design and unchanged; every other op — unchanged; the resolver — unchanged (it only reads declared `feasible` + `Legal`). Total footprint: one axis-value definition + one legality-pair + one policy-rank entry. This is exactly the criterion-#2 test (zero substrate edits for a third kind) passed by construction, and it is impossible to pass in the 2-axis system where 'backend' is baked into the class hierarchy.

## Criteria self-assessment (must-fix 1-10)

| # | verdict | how |
|--|--|--|
| 1 | **yes** | WeightDelivery is axis 2, a first-class sum type orthogonal to Compute. The memstream/dynload/fetch_weights emission moves out of HWCustomOp (hwcustomop.py:307/355/407) onto the `DecoupledStream`/`DynamicLoad`/`LoopFetched` values, expressible once. The 236-line IPI becomes Stitcher.compose over per-axis StitchFragments. |
| 2 | **yes** | The Assembler `realize` and the resolver call ONLY axis interfaces; they contain no op-type string, no isinstance-on-kind, no allowlist. Adding an op = new OpContract; adding a kind = new Compute value. Neither edits shared code. This is enforced structurally, not by discipline. |
| 3 | **yes** | Selection is constraint resolution: local `Compute.feasible` (self-advertised, co-located with the value — killing the specialize_layers.py:60 vs :275 drift) + a declared pairwise `Legal` relation, ranked by declared policy. The specialize_layers.py:40-211 per-op ladder and `optype+'_'+impl_style` string concat collapse into one op-agnostic resolver. |
| 4 | **yes** | `reference_model` is a field on OpContract; run harnesses are `CppSim\|RtlSim` values on axis 6. Different fields on different objects → no shared `execute_node` name, no MRO diamond, no hand-written shims. RtlSim.run is axis-generic (one copy), dissolving the VVAU indent bug (vectorvectoractivation_rtl.py:89) and the MVAU rtlsim duplication. |
| 5 | **yes** | `emit_compute`/`emit_delivery`/`emit_meminit`/`serialize` are pure functions of typed inputs → typed Artifacts. No `self.code_gen_dict` mutable dict (hlsbackend.py:136), no ordered side-effecting global_includes→defines→docompute sequence. Fragments compose; each is independently unit-testable. |
| 6 | **yes** | Memory is axis 3 (`LUTROM\|BRAM\|URAM\|OffChipDMA`), a single concept shared across all ops, distinct from WeightDelivery. thresholding/lookup/MVAU stop re-branching mem_mode; the base never sniffs a subclass name for storage. |
| 7 | **yes** | `Passthrough` and `Container` are first-class Compute values; non-tensor ports are typed `AxiMM`/`AxiLite`/`Sideband` variants; the Artifact sum admits `BlockDesign`/`VendorIP`. iodma's raising getters and finn_loop's `get_rtl_file_list→None` stub become honest-by-typing, not violations. |
| 8 | **yes** | `RTL.emit_compute` returns a typed `RtlParamMap` bound against a DECLARED source manifest (single source of truth, replacing the 3x-triplicated file lists and 26 untyped `$KEY$` str.replace). dtype/width flow from the OpContract typed values, so magic numbers in .v (layernorm [31:0], requant :.6f) surface as a typed-binding mismatch, not a silent override. |
| 9 | **yes** | Every implementation kind is an explicit Compute VALUE. The hidden 'vivado' FIFO backend (streamingfifo_rtl.py:141) and the base reach-through via try/except (streamingfifo.py:90/104) become a normal peer value selected by the resolver — no runtime attr multiplexing inside a class named `*_rtl`. |
| 10 | **yes** | `RealizeCtx` injects simulator, fpgapart, build dir, and resolved rtllib root — no finnxsi import-time singleton (hwcustomop.py:39), no ambient FINN_ROOT/XILINX_VIVADO reads inside ops. Each kernel builds/tests/sims from its coordinate + ctx alone. (Cross-op cost-model coupling like the shuffle NestSim is confined to a Folding value, not fused to the substrate.) |

## Weaknesses (self-declared)

Honest costs of the product-type framing: (1) LEGALITY EXPRESSIVENESS. Pairwise `Legal ⊆ Axis_i × Axis_j` handles most constraints, but some real constraints are genuinely n-ary (MLO couples WeightDelivery=LoopFetched ∧ Memory=OffChipDMA ∧ a specific Ports/AxiMM ∧ Clock config simultaneously). Where pairwise is insufficient I fall back to a resolver-level n-ary predicate — an escape hatch that is less elegant than the clean pairwise tables and reintroduces a small amount of centralized constraint logic. (2) COMBINATORIAL SURFACE. 6 axes with ~5 values each is a large nominal space; the resolver must enumerate-and-prune. In practice the Legal relation and local feasibility prune it hard, but a naive reader sees a scary cartesian product, and policy ranking across many legal coordinates needs care to stay deterministic. (3) FOLDING IS THE WEAK AXIS. Per §9.3 I only ship a shared interface; the Folding component of a coordinate stays op-specific, so it is the least genuinely-orthogonal axis — two coordinates differing only in Folding may hide substantial per-op math divergence that the 'point in a space' metaphor undersells. (4) `Container`-AS-COMPUTE is a naming stretch: a container is not really 'how the math is done', so axis 1 is more honestly 'realization kind' than 'compute impl' — finn_loop and streamingdataflowpartition sit slightly awkwardly on an axis named Compute. (5) DELIVERY/MEMORY SEPARATION has edge cases: MLO fetch_weights ties motion and off-chip storage tightly; I model it as a legal (LoopFetched, OffChipDMA) pair, but the clean 'motion vs substrate' split is thinnest exactly there. (6) The design assumes every op's real behavior decomposes cleanly onto the 6 axes; a genuinely novel op that needs a cross-axis interaction not anticipated by the Legal relation forces either a new pairwise entry (cheap) or, worst case, a new value on multiple axes at once (a mild re-coupling the product framing was meant to avoid).

## Prior fidelity

This design commits fully to EXPLICIT MULTI-AXIS PRODUCT TYPE and refuses the generic middle. The distinctive, non-negotiable moves: (a) a kernel is literally `OpContract × Coordinate`, a POINT — not a class, not merely an object that 'has-a' strategy (that would be the composition-over-inheritance prior #1). (b) The set of buildable kernels is a DECLARED SUBSET of the cartesian product, with illegality expressed as a typed construction-time error via local `feasible` + a pairwise `Legal` relation — the product's legality relation IS the abstraction, and it is the direct answer to axis-conflation (the assigned attack surface). (c) Realization is a single AXIS-GENERIC assembler that composes one fragment per axis, so the substrate literally cannot name an op or a kind. (d) Selection is CONSTRAINT SOLVING OVER THE PRODUCT SPACE: fix the pinned axes, solve the free axes against the legality relation, rank by policy. (e) The forcing function is dispatched by the cleanest possible product-type move — a third implementation kind is ONE NEW VALUE ON ONE AXIS, with zero edits to the assembler, substrate, resolver, or any other op. The core insight the brief demanded — that the as-is FAILED by collapsing WeightDelivery and Memory onto the HLS/RTL inheritance axis — is answered head-on by making those two SEPARATE axes and showing MVAU's `mem_mode × dynamic_input × mlo` explosion factor into three independent coordinate choices. I deliberately did NOT hedge toward trait-solving (prior #2) or IR-lowering (prior #4); where those overlap (feasibility declaration, typed artifacts) I express them in product-type terms — per-axis-value predicates and per-axis emit-fragments — rather than adopting their framing.
