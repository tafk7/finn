# Design: IR / lowering-target (MLIR-dialect thinking): the Kernel is a typed IR node in a "finn.kernel" dialect with a precise port/folding/semantics/resource contract; HLS, RTL, and static-IP are pure lowerings of that node to artifacts, chosen by a legalization/conversion pass, with no mutable code_gen_dict side-channel and no base-class knowledge of concrete ops.

> **Essence:** A Kernel is a typed IR node (typed ports + folding attrs + a declared reference semantics + a resource interface); every implementation kind — HLS, RTL, static-IP, container — is a pure Lowering(KernelIR, Target) -> Artifact registered as a legalization pattern, so adding a third kind is adding a pattern, never editing the core.

**Selection strategy:** `separate-pass-queries`

## Core model

## The one idea

Treat the HW-op abstraction as a **compiler IR with a lowering framework**, borrowed wholesale from MLIR. The current system fails because it fuses *what a node is* (an ONNX-adjacent typed value with ports, folding, and semantics) with *how it is realized* (C++ text, Verilog text, TCL) — and it realizes it by **mutating shared state** (`code_gen_dict`, `hlsbackend.py:136`) and **branching the substrate on subclass identity** (`hwcustomop.py:307/355/407`). The IR/lowering split makes both impossible by construction:

- **A `KernelOp` is a pure, immutable IR value.** It has *no* codegen methods at all — not `docompute`, not `generate_hdl`, not `code_generation_ipi`. It only *declares*: a typed **port signature**, **folding attributes**, a **reference semantics**, and it exposes a **resource/timing interpretation**. This is the entire contract. It is uninstantiable-into-hardware by design, which is the correct inversion of today's accidental "agnostic base is uninstantiable-by-contract" bug (`matrixvectoractivation.py:920`).
- **A `Lowering` is a pure function** `lower(kernel: KernelOp, target: Target) -> Artifact`. HLS, RTL, static-IP, and container-stitch are each *one lowering*. Nothing the lowering touches is shared mutable state; it consumes the immutable IR + an injected `Target` (fpgapart, clock, toolchain paths, rtllib root, simulator handle) and returns an immutable `Artifact`.
- **Selection is legalization.** A `ConversionTarget` marks `finn.kernel` ops illegal; a driver pass greedily applies the registered lowering *pattern* whose `match(kernel, target)` succeeds with highest benefit. This is exactly MLIR's dialect-conversion framework, and it dissolves `specialize_layers.py:40-211` into declarative patterns.

```
                         finn.kernel  DIALECT  (the IR — pure, immutable)
   ┌───────────────────────────────────────────────────────────────────────┐
   │  KernelOp value                                                         │
   │  ├─ PortSignature :  [ TensorStream(dt, folded_shape, fold_iface)       │
   │  │                     AxiLite(regs)   AxiMM(addr_w, data_w, dir)        │
   │  │                     Sideband(TLAST|TKEEP) ]        ← typed ports      │
   │  ├─ Folding       :  FoldAttr per TensorStream port (PE/SIMD/…)         │
   │  ├─ Semantics     :  SemRef  (declared reference fn: np/qonnx)          │
   │  ├─ Attrs         :  typed attribute dict (MW/MH/dtype/…)               │
   │  └─ Region?       :  optional nested KernelOp graph (containers)        │
   │                                                                         │
   │  Interfaces the value SATISFIES (traits, not base methods):            │
   │     FoldingInterface   ResourceInterface   PortInterface   SemInterface │
   └───────────────────────────────────────────────────────────────────────┘
        │  ↑ queried, never mutated
        │
   ┌────┴───────────────── LOWERING REGISTRY (open, per-(semantic, kind)) ───┐
   │  Pattern = { match(kernel,target)->Legality , lower(kernel,target)->Art}│
   │                                                                         │
   │  HlsLowering      RtlLowering      StaticIpLowering     ContainerLower  │
   │  (emit C++)       (bind rtllib)    (emit IP instance)   (stitch region) │
   │  ParamSourceLower (weight delivery is its OWN kernel+lowering)          │
   └─────────────────────────────┬───────────────────────────────────────────┘
                                 │  pure fn
                                 ▼
   ┌───────────────────────── ARTIFACT (immutable value) ────────────────────┐
   │  sources: [SourceFile]      params: [ParamFile]                          │
   │  instance: HwInstance  (typed port-binding, NOT a text blob)             │
   │  stitch:  BlockDesign  (typed netlist value: cells + typed edges)        │
   └──────────────────────────────────────────────────────────────────────────┘
                                 │
             ┌───────────────────┼────────────────────┐
     RenderTcl(BlockDesign)  RenderCpp(sources)   RenderVerilog(sources)
     (ONE backend-agnostic emitter per artifact type — no per-op TCL)
```

## The IR type system (attacks criteria #7, #8, ports axis)

Ports are the load-bearing type. The 8 shape/dtype accessors (`hwcustomop.py:261-289`) are **not methods on the op** — they are *projections of the port signature*:

```
PortType =
  | TensorStream(elem: FinnDataType, normal_shape: Shape, fold: FoldAttr)
  | AxiLite(regs: [Reg(name, offset, width)])
  | AxiMM(dir: In|Out, addr_w: int, data_w: int)
  | Sideband(kind: TLAST | TKEEP)

get_folded_shape(port)  is defined ONLY on TensorStream.
get_instream_width(port) is defined ONLY on TensorStream.
```

This is why `iodma`'s folded-shape getters can stop `raise`-ing (`iodma_hls.py:116/131`): an IODMA declares `[TensorStream, AxiMM]`. Asking "folded shape of port 1" is a **type query on an AxiMM**, which is statically ill-formed — the framework never asks it, because generic passes iterate `ports.tensor_streams()`, a typed sub-view. `tlastmarker`'s four raising getters (`tlastmarker_hls.py:211`) likewise disappear: it declares a `Sideband` port and *has no tensor streams*, so the tensor-stream accessors are simply not in its interface set. **A contract that admits typed non-tensor ports makes "the getter raises" a category error the type system forbids, not a hack.**

The op↔rtllib coupling (criterion #8) becomes a **typed artifact binding**. An `RtlModule` artifact declares a **typed parameter schema** `{name: Type}` and a **typed port map**. `bind(RtlModule, {ACCU_WIDTH: BitWidth(24), SIGNED: Bool(true), ...})` **type-checks** each value against the schema and produces a parameterized `HwInstance` whose parameters flow only through the wrapper's typed `#()` map. There is no `template.replace("$KEY$", str(v))` (the 26 untyped edges, `matrixvectoractivation_rtl.py:296`); a renamed token or a magic `[31:0]` in a `.v` (`layernorm_wrapper_template.v:20`) is a **schema mismatch caught at bind time**, because the schema, not the `.v` text, is the source of truth for widths. The triplicated source manifests (`fmpadding_rtl.py:137/153/165`) collapse to one field: `Artifact.sources` is a single immutable list, and `get_rtl_file_list`, IPI `add_files`, and rtlsim source discovery are three *readers* of that one value, not three hand-maintained copies.

## Semantics vs execution: three interpretations, zero diamond (attacks criterion #4)

The `execute_node` diamond (`pool_hls.py:116` + ~8 shims, plus the latent bug `vectorvectoractivation_rtl.py:89`) exists because "reference math" and "run this implementation" share one method name under MRO. In the IR model they are **three distinct interpretations over the same IR**, none of them a method on the op:

```
reference(kernel, inputs)          -> outputs   # pure, from kernel.Semantics (SemRef)
run_cppsim(hls_artifact, inputs)   -> outputs   # interpretation of an HLS Artifact
run_rtlsim(rtl_artifact, inputs)   -> outputs   # interpretation of an RTL Artifact
```

`reference` is a **field of the KernelOp** (`SemRef`, the np/qonnx golden model) — it never collides with a backend, because backends are not on the op. `run_cppsim`/`run_rtlsim` are functions of the *Artifact*, dispatched on artifact type, so there is no MRO to resolve and no per-op `execute_node` shim. The `ElementwiseBitShift` attr-schema MRO trap (`elementwise_binary_hls.py:1070`, blocker #15) is gone for the same reason: attributes live in the immutable `Attrs` value on the op, not merged across an inheritance diamond.

Verification is then just **agreement between interpretations**: `assert reference(k, x) ≈ run_rtlsim(lower_rtl(k, target), x)`. rtlsim is hermetic because the simulator is a field of `Target`, injected — not the `finnxsi` import-time singleton (`hwcustomop.py:39`).

## Resource/timing is an interpretation, delegated to the lowering (attacks #10, L5)

Resource and cycle estimates are **queries**, `resources(kernel, folding, target)` and `cycles(kernel, folding)`, returning immutable estimates. Crucially, resource estimation is delegated to the **selected lowering**, not carried by an agnostic base. This fixes the shuffle disaster (`outer_shuffle.py:20-238`) where the "agnostic" base *is* a Python re-sim of the HLS `input_gen.hpp` pipeline: in the IR model, `HlsLowering.resources(shuffle_kernel)` owns the HLS pipeline model and `RtlLowering.resources(shuffle_kernel)` owns the BRAM formula (`inner_shuffle.py:94`). The kernel itself carries only op-count/param-count (a `SemRef`-level quantity). No lowering's microarchitecture leaks into the op.

## Weight/param delivery is a *separate kernel*, not an axis smeared into compute (attacks criteria #1, #2, #6)

This is the decisive move and the heart of the MVAU fix. Today weight delivery is emitted from *both* backends and stitched by 236 lines in the agnostic base (`matrixvectoractivation.py:920`), gated by a substrate op_type allowlist (`hwcustomop.py:310`). In the IR model, **the weight streamer is its own `finn.kernel` node** — `ParamSource` — with its own port signature (an `AxiLite` config port + a `TensorStream` weight-out port) and its own lowerings. The MVAU compute node has a **param-in `TensorStream` port**; an IR edge connects `ParamSource.out -> MVAU.param_in`.

So `mem_mode × dynamic_input × mlo` is not a branch inside MVAU — it is **which `ParamSource` variant** (`EmbeddedParams`, `DecoupledStreamer`, `ExternalDMA`, `DynamicLoad`, `MloFetch`) sits upstream in the IR graph, each with its own lowering and its own `MemoryStrategy` attribute (LUTROM/BRAM/URAM/off-chip). Weight-delivery and memory-strategy are thus **structurally orthogonal** to compute-kind: you pick the compute lowering for MVAU and, independently, the delivery kernel + its memory-strategy attribute. The substrate never sees `calc_wmem`/`ram_style`/MVAU op-type strings — those live entirely inside `DecoupledStreamer`'s lowering.

## Stitching = composition of typed artifacts, not TCL in the base (attacks Pattern F3)

The 236-line `code_generation_ipi` becomes: lower the compute node → `Artifact_c`; lower the upstream `ParamSource` node → `Artifact_p`; then a **structural lowering** `stitch(graph_region) -> BlockDesign` composes them by reading the typed IR edges and producing a `BlockDesign` value (cells + typed port-to-port edges). `BlockDesign` is rendered to Vivado TCL by **one** backend-agnostic `RenderTcl(BlockDesign)` emitter. The stitching is *composition of declared parts*, exactly as the brief demands: MVAU's stitch is `stitch({MVAU_compute, DecoupledStreamer}, edges)`, and the clk2x pin, axilite wiring, etc. are typed cell-ports the renderer handles uniformly — no per-op TCL, no `os.listdir` "last match wins" (`matrixvectoractivation.py:983`).

## Containers = IR regions (attacks criterion #7, finn_loop)

MLIR has regions natively; a `KernelOp` may carry a **`Region`** (a nested `finn.kernel` graph). `finn_loop` is a `KernelOp` with a region = the loop body. Its port signature is *derived by projecting the region boundary* (first-node inputs, last-node outputs) — typed, not stubbed. Its lowering is `ContainerLowering`: recursively lower the region, then `stitch` into a `BlockDesign` with a loop-control shell. `get_rtl_file_list→None` (`finn_loop.py:1175`) is not a violation because the container's Artifact is a `BlockDesign`, whose renderer produces sources structurally — there is no flat-file-list contract to stub. The sibling-op string dispatch (`finn_loop.py:421`) is replaced by walking typed region edges and reading each child's *declared* `ParamSource` port, not `startswith("MVAU")`.

## Why this is the right shape

Every as-is pain reduces to one of two IR invariants: **(a) the op is a pure immutable value with no codegen** (kills `code_gen_dict`, the MRO diamond, the base-class leak, order-dependence), and **(b) realization is a pure typed function into an immutable artifact** (kills the TCL-in-base, the `$KEY$` surgery, the triplicated manifests, the hidden vivado backend). Adding static-IP is adding one pattern to the registry — the forcing function passes trivially.

## Axis treatment

### compute_impl
A first-class IR concept: compute kind = WHICH lowering pattern the legalization pass selects for a KernelOp's semantic. HLS, RTL, static-IP are three registered Patterns keyed by (semantic_id, kind); each is a pure fn `lower(kernel, target) -> Artifact`. The op declares NO compute method (no docompute/generate_hdl) — it declares only ports+folding+semantics. Kinds are peers because they are all just entries in the LoweringRegistry; there is no {HLS,RTL} type baked into any class (fixes hlsbackend/rtlbackend asymmetry, 638 vs 146 LOC). The 'hidden vivado third backend' (streamingfifo_rtl.py:141) becomes an explicit registered VivadoInfraLowering pattern, not a runtime attr branch.

### weight_delivery
Modeled as a SEPARATE KernelOp in the IR graph (ParamSource: EmbeddedParams | DecoupledStreamer | ExternalDMA | DynamicLoad | MloFetch), connected to the compute kernel's typed param-in TensorStream port by an IR edge. Delivery is therefore structurally orthogonal to compute-kind: choosing MVAU's compute lowering and choosing its ParamSource variant are independent selections. All weight-streamer emission (memstream/dynload/fetch_weights, today at hwcustomop.py:307/355/407 and emitted from BOTH backends) lives entirely inside ParamSource lowerings. The substrate never branches on op_type and never sees calc_wmem/ram_style/mlo_max_iter.

### memory_strategy
A typed attribute on the ParamSource kernel (MemoryStrategy = LutRom | Bram | Uram | OffChipDma, plus read-discipline). It is resolved by the ParamSource lowering, not re-branched per compute op. Thresholding, Lookup (ROM vs DMA), and MVAU all reuse ONE ParamSource+MemoryStrategy vocabulary instead of independent mem_mode try/except (thresholding.py:179; lookup ROM/DMA). Because memory-strategy is an attr on a distinct node, changing it never touches the compute op's ports or semantics.

### folding
SHARED INTERFACE ONLY (v1). FoldAttr is a typed attribute attached to each TensorStream port; the FoldingInterface exposes exactly `folded_shape(port, fold) -> Shape` and `fold_factors(port) -> {PE,SIMD,...}`. Every kernel implements this interface uniformly so generic passes (stitching, resource, port-width) rely on one vocabulary. The MATH stays per-kernel — MVAU's PE/SIMD folding and SWG's window folding compute different folded_shape bodies. Unified folding math is EXPLICITLY OUT OF SCOPE for v1 (per brief 9.3); the interface is the only commitment.

### ports
The type system's foundation. PortType = TensorStream(dt, shape, fold) | AxiLite(regs) | AxiMM(dir,addr_w,data_w) | Sideband(TLAST|TKEEP). Tensor-stream accessors (folded_shape, instream_width) are typed projections defined ONLY on TensorStream, so asking them of an AxiMM is statically ill-formed — dissolving iodma's raising getters (iodma_hls.py:116/131) and tlastmarker's 4 raising getters (tlastmarker_hls.py:211). Non-tensor ports (AXI-Lite control, AXI-MM, TLAST/TKEEP) are first-class typed values, not footnotes. Direction (iodma in/out) is just a different PortSignature, not method-wide branching.

### execution
Three DISTINCT interpretations over the same IR, never one method: (1) reference(kernel, x) — a pure fn read from the kernel's declared SemRef field (the golden np/qonnx model); (2) run_cppsim(hls_artifact, x); (3) run_rtlsim(rtl_artifact, x). reference is a field on the op so it never collides with a backend under MRO (kills the ~8 execute_node shims and the vectorvectoractivation_rtl.py:89 indent bug). cppsim/rtlsim dispatch on Artifact type, not class MRO. The simulator is a field of the injected Target, not the finnxsi import-time singleton. Verification = agreement between interpretations.

## Conformance sketches

### mvau
MVAU is TWO IR nodes, not one overloaded class. (a) An `MVAU.compute` KernelOp: ports = [TensorStream(act_in, fold=SIMD), TensorStream(param_in, fold=SIMD*PE), TensorStream(out, fold=PE)]; Attrs = MW/MH/dtypes; SemRef = the clean golden matmul+multithreshold (matrixvectoractivation.py:132, which the census already calls a clean seam). It has NO code_generation_ipi, NO generate_hdl. (b) A `ParamSource` KernelOp upstream, connected `param.out -> MVAU.param_in`. Compute lowerings: HlsLowering(MVAU.compute) emits the hlslib Matrix_Vector_Activate call as a pure fn -> Artifact.sources (no code_gen_dict; global_includes/defines/docompute become fields of the returned Artifact, computed in one pass, not ordered mutations). RtlLowering(MVAU.compute) binds mvu_vvu_axi_wrapper.v via TYPED param schema (ACCU_WIDTH bound from accDataType, fixing the matrixvectoractivation_rtl.py:349 bug where it silently used output dtype). Weight delivery axis = which ParamSource variant is in the graph (embedded/decoupled/external/dynamic/MLO) — fully orthogonal, chosen independently. The 236-line IPI (matrixvectoractivation.py:920) = `stitch({MVAU.compute_artifact, ParamSource_artifact}, typed_edges) -> BlockDesign`, rendered by the ONE RenderTcl emitter. instantiate_ip vanishes: the compute Artifact already carries its HwInstance; stitch composes instances. Zero substrate edits, zero base-class op_type allowlist.

### thresholding
Thresholding.compute KernelOp: ports=[TensorStream(in,fold=PE), TensorStream(out,fold=PE)], plus a param-in TensorStream ONLY when a DecoupledStreamer ParamSource feeds it. Memory-strategy (embedded vs decoupled) is NOT a branch in the op — it is: (embedded) the threshold constants are folded into the compute Artifact by its lowering; (decoupled) a DecoupledStreamer ParamSource node upstream with MemoryStrategy attr. The get_instream_width/get_verilog_top_module_intf_names mem_mode try/except in the agnostic base (thresholding.py:173/292) is gone because the port signature already differs structurally (param-in present or absent) — no sniffing. Threshold serialization: ONE typed `ParamFile` serializer on the ParamSource, parameterized by a ThresholdLayout attr (sorted-binary-search-order for the RTL memory strategy, flat for embedded). The three divergent make_weight_file/minimize_weight_bit_width copies (thresholding.py:127, _hls:781, _rtl:566) collapse into one serializer whose layout is an attribute, not a duplicated method. The 'HLS variant emits RTL streamer' cross-backend leak disappears: the streamer is a separate ParamSource kernel with its OWN rtl lowering.

### finn_loop
A KernelOp carrying a Region (nested finn.kernel graph = loop body) — MLIR regions are the native model for containers, so this is NOT a category error. Its PortSignature is DERIVED by projecting the region boundary (first-node inputs, last-node outputs) — honest typed ports, no delegation hacks (replaces finn_loop.py:146-275 getCustomOp-into-child-with-'param-is-input-1' assumption; the param edge is a typed IR edge, read directly). SemRef = interpret the region 'iteration' times (matches the cppsim re-run at finn_loop.py:300, but as a declared region interpretation). Lowering = ContainerLowering: recursively lower every child node in the region, then stitch into a BlockDesign with the loop-control shell (loop_control_wrapper.v bound via typed schema). get_rtl_file_list->None (finn_loop.py:1175) is not a stub violation — the Artifact IS a BlockDesign whose renderer emits sources structurally; there is no flat-file contract to fake. The sibling op_type string dispatch (finn_loop.py:421/558/712) becomes: for each region edge whose target port is a param-in, read the upstream ParamSource's declared serialization — no startswith('MVAU'/'Thresholding'/'Elementwise').

### iodma
IODMA.compute KernelOp with PortSignature = [TensorStream(stream_side, fold), AxiMM(dir, addr_w, data_w), AxiLite(control_regs)]. Because AxiMM and TensorStream are distinct PortTypes, the folded-shape/instream-width accessors are typed projections over the TensorStream port only — asking them of the AxiMM side is statically ill-formed, so iodma_hls.py:116/131's `raise ValueError` is deleted, not reimplemented: generic passes iterate ports.tensor_streams(). Direction (in vs out) = two PortSignatures (which side is AxiMM vs TensorStream) — not per-method branching across 8 methods. execute_node=pass (iodma_hls.py:388) becomes an honest SemRef: a data-mover identity interpretation over the TensorStream, so cppsim/rtlsim data flow is observable, not a silent no-op. The LCM datawidth-converter topology (docompute template surgery, iodma_hls.py:214) is expressed once as a small IR sub-graph of DWC kernels, backend-agnostic, lowered per-kind.

### static_ip_gemm
The forcing function, and it is a ~40-line addition with ZERO substrate edits. Register ONE new pattern: `StaticIpGemmLowering` keyed by (semantic_id='matmul', kind='static-ip'). Its `match(kernel, target)` advertises feasibility exactly as MLIR dynamic-legality: returns Legal iff kernel.Attrs.act_dt==INT8 and weight_dt==INT8 and MW/MH within the hard IP's supported set and target.fpgapart in the IP's device list — a declared-capability predicate co-located with the lowering, NOT a new `if optype=='MVAU'` arm in a central switch (contrast specialize_layers.py:235 _mvu_rtl_possible). Its `lower()` does NOT emit generated source: it returns an Artifact whose `instance` is an IPInstance referencing a pre-synthesized .xci/VLNV, with the SAME typed PortBinding (act_in/param_in/out TensorStreams at the same fold) as the HLS/RTL MVAU compute. Because it satisfies the identical port+folding contract, stitch() composes it with any ParamSource unchanged, and the legalization driver auto-selects it (highest benefit) whenever match() succeeds — its common INT8 case — falling back to HLS/RTL otherwise. Nothing in the core, the ports, the stitcher, or any other kernel changes: the proof the abstraction is real.

## Criteria self-assessment (must-fix 1-10)

| # | verdict | how |
|--|--|--|
| 1 | **yes** | Weight delivery is a separate KernelOp (ParamSource: embedded/decoupled/external/dynamic/MLO) connected by a typed IR edge to the compute op's param-in port. It is structurally orthogonal to compute-kind and expressed once per variant as a lowering; the 236-line IPI (matrixvectoractivation.py:920) becomes stitch() over declared artifacts. |
| 2 | **yes** | The substrate is the KernelOp IR value + LoweringRegistry; it holds no codegen and no op-type knowledge. Adding/removing an op = registering/removing a pattern. The base-class leaks (hwcustomop.py:307/355/407 op_type allowlists) have no home: memstream/fetch_weights/dynload live inside ParamSource lowerings. |
| 3 | **yes** | Selection is legalization: each lowering pattern carries match(kernel,target)->Legality (declared capability co-located with the lowering). The specialize_layers.py:40-211 god-switch and its stringly-typed optype+'_'+impl_style resolution (L399) become a generic conversion driver picking highest-benefit legal pattern. The drifting dtype pre-gates (L60/69 vs L275) unify into one match predicate. |
| 4 | **yes** | reference (a SemRef field on the op), run_cppsim (fn of HLS Artifact), run_rtlsim (fn of RTL Artifact) are three separate interpretations, none an op method — no MRO diamond, no ~8 execute_node shims, and the vectorvectoractivation_rtl.py:89 indent bug cannot recur since rtlsim is a single artifact-typed fn. |
| 5 | **yes** | Lowering is a pure fn returning an immutable Artifact; global_includes/defines/docompute (hlsbackend.py:136 mutable code_gen_dict) become fields computed in one pass and returned, not ordered side-effects over a shared dict that is populated/cleared. |
| 6 | **yes** | MemoryStrategy is a typed attribute on the ParamSource kernel (LutRom/Bram/Uram/OffChipDma), resolved by its lowering and shared across MVAU/Thresholding/Lookup — replacing the independent mem_mode try/except re-branching (thresholding.py:179; lookup ROM/DMA). |
| 7 | **yes** | Non-tensor ports are typed PortTypes (AxiLite/AxiMM/Sideband) so infra ops declare honest signatures; tensor-stream accessors are typed projections that are simply absent for non-stream ports (kills iodma_hls.py:116/131 and tlastmarker_hls.py:211 raises). Containers are KernelOps with Regions whose Artifact is a BlockDesign — get_rtl_file_list->None (finn_loop.py:1175) is not needed. |
| 8 | **yes** | op<->rtllib is a typed binding: RtlModule declares a {name:Type} param schema + typed port map; bind() type-checks values (ACCU_WIDTH from accDataType, catching matrixvectoractivation_rtl.py:349 and the [31:0] magic in layernorm_wrapper_template.v:20). No $KEY$ str.replace; Artifact.sources is one immutable list, killing triplicated manifests (fmpadding_rtl.py:137/153/165). |
| 9 | **yes** | The 'vivado' impl inside streamingfifo_rtl.py:141 becomes an explicit registered pattern (VivadoInfraLowering) with its own match(); all implementation kinds are peer registry entries, never a runtime impl_style attr branch reached via base try/except (streamingfifo.py:90/104). |
| 10 | **yes** | All ambient deps are fields of the injected Target (fpgapart, clock, toolchain paths, FINN_ROOT/rtllib root, simulator handle) — no finnxsi import-time singleton (hwcustomop.py:39), no ambient FINN_ROOT reads. Cost models are lowering-owned interpretations, so SWG's HLS-pipeline sim (outer_shuffle.py:20-238) lives in HlsLowering.resources, not the op; kernels build/test/sim in isolation from an injected Target. |

## Weaknesses (self-declared)

1) SELECTION VS FOLDING ENTANGLEMENT: making the ParamSource a separate IR node means folding of the param stream must be kept consistent across the compute node and its ParamSource by a graph constraint; if the folding-interface stays per-kernel (v1 scope), the driver needs a small propagation pass to keep param_in fold = f(PE,SIMD) — a real but bounded seam I add, not free. 2) STITCH RENDERER RISK: I concentrate all IPI/TCL knowledge into one RenderTcl(BlockDesign) emitter. That is a strength (one place) but also a single complex component; the current per-op TCL has genuinely bespoke bits (clk2x pins, aperture hotfixes like finn_loop.py:1071) that must be modeled as typed BlockDesign features or the renderer accretes special-cases — risk of re-growing a god-emitter if the BlockDesign type is under-designed. 3) COST OF THE TYPED RTLMODULE SCHEMA: every finn-rtllib module must gain a declared param schema + port map. That is upfront migration work across ~11 rtllib subtrees and is where the clean-slate ideal meets real hand-written Verilog whose params are currently implicit. 4) STATIC-IP MATCH EXPRESSIVENESS: encoding a hard IP's feasibility (device list, dim ranges) as a match predicate is clean, but if two static-IP patterns overlap, benefit-ordering must be made total/deterministic or selection becomes order-sensitive — I rely on an explicit benefit metric that someone must assign. 5) DECOMPOSING OPS INTO IR SUBGRAPHS (iodma's LCM DWC chain, MVAU=compute+ParamSource) shifts complexity from method-branching into graph construction; teams that think in single-class ops face a conceptual tax and the graph must be built by a lowering/inference pass that itself needs testing. 6) REGIONS ADD IR MACHINERY: proper region support (nested graphs, boundary projection, recursive lowering) is more infrastructure than a flat class hierarchy — justified by finn_loop/SDP but real weight for a first version.

## Prior fidelity

This design is MLIR to the bone rather than a generic 'split interfaces' answer. (a) The KernelOp is a pure immutable IR VALUE with declared interfaces (traits), not a base class with methods — codegen is categorically absent from the op, which is the defining IR move. (b) Realization is DIALECT CONVERSION: lowerings are RewritePatterns with match()/rewrite(), selection is a ConversionTarget legalization driver picking by benefit — I chose separate-pass-queries precisely because it IS the MLIR conversion framework, and I justify it as such rather than as a neutral 'capability query'. (c) Verification is INTERPRETATION over IR (reference/cppsim/rtlsim as three interpreters), the classic IR stance that semantics are a property of the node and execution is an interpreter, not a method. (d) Non-tensor ports are TYPES in a type system, so 'the getter raises' becomes 'statically ill-formed' — a type-theory framing no other prior would reach for. (e) Containers are REGIONS, the single most MLIR-specific construct, applied exactly where FINN needs it (finn_loop). (f) The op<->rtllib fix is a TYPED artifact binding with a schema, i.e. lowering targets a typed artifact IR, not text. I deliberately did NOT hedge toward composition-of-parts (prior 1) or a product-type point (prior 3): where those priors would 'compose a ParamDelivery component into a Kernel object', I instead make the ParamSource a PEER IR NODE connected by a typed edge and let a structural lowering stitch — that graph-and-lowering framing is the distinctive commitment of the IR angle.
