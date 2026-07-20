# FINN Build-Flow Consumer Surface — What KernelOp Must Satisfy

*A call-site census of every FINN build-flow consumer of the HWCustomOp op contract:
the 20-step build flow, the fpgadataflow transformations, the analysis/estimation
passes, and the qonnx transforms FINN invokes (read from the finn-pinned deps/qonnx @
f5c9819). Companion to `hw-backend-model.md` — that document models the op SUBSTRATE
being replaced; this one models the CONSUMER SURFACE the replacement (KernelOp) must
present so a partially-migrated graph builds end-to-end (goal G4). Produced by a
43-unit agent census (20 build steps as first-class units, 37 fpgadataflow transforms
in 13 shards, 8 analysis passes, 2 qonnx units); synthesized union with a single
aggregate ground-truth verification pass. Build-step ordering verified against
`default_build_dataflow_steps`.*

---

## 1. The method contract KernelOp must satisfy

This is the union of every method invoked on an op *instance* across all 43 census units (direct calls plus calls that flow through transforms/analyses the units drive). `getCustomOp(node)` resolution and raw-protobuf reads (`get_by_name`, `is_hls_node`) are excluded — they act on the node shell, not the instance. Ranked by aggregate call frequency; `(A)` is the stable contract the adapter **must** expose, `(B)` is the coupling it must **satisfy without reproducing the leak**.

### (A) Stable declared contract — the adapter MUST expose these

| # | method | called by (unit kinds × count) | legitimacy | KernelOp adapter obligation |
|---|--------|-------------------------------|------------|------------------------------|
| 1 | `get_nodeattr(key)` | build_step ×~14, transform ×~11, analysis ×6, qonnx ×1 (≈150+ calls) | declared_contract | Universal untyped accessor. Adapter must resolve *every* key in the nodeattrs union (folding PE/SIMD/parallel_window, dtypes, mem_mode, ram_style/resType, geometry MW/MH/NumChannels/…, container `body`/`model`, side-channels `cycles_estimate`/`io_chrc_*`) as **resolved-config data read off the Point** — never by touching the graph. Tolerate absent keys (many callers wrap in try/except). |
| 2 | `set_nodeattr(key,val)` | build_step ×~11, transform ×~9, analysis ×1 (≈90+ calls) | declared_contract | Must accept writes and **assert the key is a declared nodeattr** (`step_apply_folding_config`, `set_folding`, `set_fifo_depths` rely on this). Writable keys include folding attrs, `cycles_estimate`, `accDataType`/`weightDataType`/`outputDataType`, `instance_name`, `code_gen_dir_*`, `ipgen_path`, `body`. Reconcile with hermeticity: these are config/side-channel writes, acceptable if they mutate resolved state only. |
| 3 | `get_input_datatype(ind)` | `step_specialize_layers`, `specialize_layers`, `step_target_fps_parallelization`, `insert_iodma`, `set_folding` (≈24 calls) | declared_contract | **Getter 1 of 8.** Must accept an optional port index (0=activation, 1=weights; also arg-less in `_mvu_rtl_possible`). Return a `DataType` supporting `.bitwidth()/.signed()/.min()/is_integer()/=="FLOAT32"`. |
| 4 | `onnx_node` (attribute) | `step_make_driver`, `alveo_build`, `make_zynq_proj`, `set_fifo_depths`, `insert_tlastmarker` (≈19 reaches) | declared_contract | Stable base property exposing the wrapped protobuf (`.name/.input/.output/.op_type`). Adapter must expose it for name/tensor-wiring walks. |
| 5 | `get_exp_cycles()` | `annotate_cycles`, `set_folding`, `exp_cycles_per_layer`, `step_create_stitched_ip`, `step_measure_rtlsim_performance`, `step_generate_estimate_reports` (≈13 calls) | declared_contract | Core analytic cost getter. Must return an int-coercible cycle estimate that moves monotonically with the folding attr being swept (drives `SetFolding` search + `AnnotateCycles`→`dataflow_performance`). Derive from the Point's compute geometry + folding. |
| 6 | `get_verilog_top_module_intf_names()` | `create_stitched_ip` ×9, `floorplan`, `step_create_stitched_ip` (≈11 calls) | declared_contract | Must return a dict with keys `clk/rst`(+`clk2x`)/`axilite`/`aximm`/`m_axis`/`s_axis`/`ap_none`, where axis/aximm entries are positionally-indexed `(name,width)` tuples. This **undeclared structural sub-contract** (exact keys + tuple shapes) must be honored verbatim by emit's interface descriptor. |
| 7 | `get_folded_input_shape(ind)` | `insert_fifo`, `insert_iodma`, `insert_dwc`, `set_fifo_depths`, `step_make_driver`, `step_measure_rtlsim_performance`, `insert_tlastmarker` (≈10) | declared_contract | **Getter 2 of 8.** Port-indexed. Return the PE/SIMD-folded input geometry. |
| 8 | `get_folded_output_shape(ind)` | `insert_fifo`, `insert_dwc`, `set_fifo_depths`, `step_make_driver`, `step_measure_rtlsim_performance` (≈9) | declared_contract | **Getter 3 of 8.** Port-indexed. |
| 9 | `get_output_datatype(ind)` | `insert_fifo`, `insert_hook`, `insert_dwc`, `specialize_layers`/`step_specialize_layers` (`_requant_rtl_possible`) (≈5) | declared_contract | **Getter 4 of 8.** Port-indexed. |
| 10 | `node_res_estimation(fpgapart)` | `res_estimation` ×3, `step_generate_estimate_reports` ×2 (5) | declared_contract | Estimation getter; takes an fpgapart string, returns `{BRAM_18K/LUT/URAM/DSP/…}`. Fans out to bram/lut/uram/dsp/`*_efficiency` estimators the callers never name — adapter supplies these under the hood, defaulting sanely. |
| 11 | `make_shape_compatible_op(model)` | `step_tidy_up`, qonnx `InferShapes` (2) | declared_contract | **Base qonnx CustomOp ABC** (lower layer than the 8 getters). Must return a genuinely shape-inferable standard-ONNX surrogate; its `str()` is used as a swap key. |
| 12 | `infer_node_datatype(model)` | `step_tidy_up`, `step_minimize_bit_width`, qonnx `InferDataTypes` (3) | declared_contract | Base ABC. Self-annotate output tensor QONNX datatypes; re-run repeatedly, must be idempotent. |
| 13 | `get_normal_input_shape(ind)` | `set_folding`, `insert_fifo`, `step_target_fps_parallelization` (≈5) | declared_contract | **Getter 5 of 8.** Used for fallback max-PE / LayerNorm dim. |
| 14 | `get_normal_output_shape(ind)` | `insert_fifo`, `insert_dwc`, `set_fifo_depths` (`SplitLargeFIFOs`) (≈6) | declared_contract | **Getter 6 of 8.** |
| 15 | `get_instream_width(ind)` | `insert_dwc`, `insert_tlastmarker` (≈4) | declared_contract | **Getter 7 of 8.** Port-indexed (`get_instream_width(1)` = weight stream on MVAU/2-input Elementwise). |
| 16 | `get_outstream_width(ind)` | `insert_dwc`, `insert_tlastmarker` (≈2) | declared_contract | **Getter 8 of 8.** Port-indexed. |
| 17 | `get_instream_width_padded(ind)` / `get_outstream_width_padded(ind)` | `insert_iodma` (3) | declared_contract | Derived byte-boundary-padded variants of getters 7/8 — **not** in the strict 8 but stable public getters the adapter must still expose (DMA stream widths). |
| 18 | `get_nodeattr_types()` | `res_estimation`, `step_generate_estimate_reports`, `set_fifo_depths` (3) | declared_contract | Must return the attr-spec dict incl. 4-tuple allowed-value sets (`resType∈{dsp,lut}`, `ram_style∈{block,distributed,ultra}`) so `res_estimation_complete` can enumerate variants and `set_fifo_depths` can capability-probe `mem_mode`. |
| 19 | `code_generation_ipi()` | `create_stitched_ip`, `step_create_stitched_ip` (2) | declared_contract | Codegen entrypoint: return the Vivado IPI `create_bd_cell` Tcl list for this node's IP. |
| 20 | `get_number_output_values()` | `insert_tlastmarker` (1) | declared_contract | Declared getter → NumIters for TLastMarker sizing. |
| 21 | `get_op_and_param_counts()` | `op_and_param_counts`, `step_generate_estimate_reports` (2) | declared_contract *(optional, hasattr-guarded)* | Optional analytics; HWCustomOp base default `{}`. Derive op/param counts directly from the Point (compute cardinality + param tensors are already design-space data). Expose with a `{}` default so absence never errors. |
| 22 | `execute_node(context,graph)` | `step_tidy_up` (`FoldConstants`), qonnx `execute_onnx` (1, rare) | declared_contract | Base ABC; only fires for all-constant-input const-folding. Contract obligation exists but rarely reached for dataflow nodes. |

**Codegen / rtlsim entrypoints** (declared as backend-mixin dispatchers, but each fans out to undeclared leaf methods — see (B) notes):

| # | method | called by | legitimacy | obligation |
|---|--------|-----------|------------|------------|
| 23 | `code_generation_ipgen(model,fpgapart,clk)` | `step_hw_codegen`, `prepare_ip` (2) | declared (dispatcher) | Top-level IP-gen entrypoint on both HLSBackend/RTLBackend. Adapter's `emit()` must populate `code_gen_dir_ipgen`. **Absorb its fan-out** (`get_ap_int_max_w`, `generate_params`, `global_includes`, `defines`, `blackboxfunction`, `pragmas`, `docompute`, `ipgen_default/extra_directives`, `generate_infra_hdl` for HLS; `generate_hdl` for RTL) internally — these leaf methods must NOT re-surface as public contract; they are private emit steps driven by the resolved Point. |

### (B) Undeclared coupling — satisfy WITHOUT reproducing the leak

Ranked by frequency. For each: how KernelOp absorbs it.

| # | method | called by (unit) | legitimacy | how KernelOp absorbs it |
|---|--------|------------------|------------|--------------------------|
| 1 | `ipgen_singlenode_code(fpgapart)` | `step_hw_ipgen`, `hlssynth_ip` (2) | undeclared (HLS-leaf only) | **Expose as a clean, backend-declared build entrypoint.** It is the single load-bearing HLS-synth call; make it part of the KernelOp codegen/build contract keyed on impl (HLS), not an `is_hls_node`-gated leaf method. RTL kernels legitimately omit it. |
| 2 | `prepare_rtlsim(behav)` | `prepare_rtlsim` (1) | undeclared (backend mixin + leaf overrides) | **Expose cleanly per-impl** as the rtlsim-lib build entrypoint (sets `rtlsim_so`). Declare it on the backend surface so `FINNLoop`/`StreamingFIFO_rtl`-style overrides are a normal capability, not a duck-typed override. |
| 3 | `code_generation_cppsim(model)` | `prepare_cppsim` (1) | undeclared (HLSBackend only) | **Fold into the codegen entrypoint family** alongside `code_generation_ipgen`; HLS-only, drives cppsim C++. Same private fan-out absorption as ipgen. |
| 4 | `compile_singlenode_code()` | `compile_cppsim` (1) | undeclared (HLS-leaf) | **Clean declared HLS build method** (sets `executable_path`). Pairs with cppsim codegen; RTL kernels omit. |
| 5 | `make_weight_file(w,mode,fname)` | `step_make_driver`, `make_driver` (2) | undeclared (weight-bearing leaf, behind `op_type.startswith('MVAU'/'Thresholding')`) | **Expose as a declared param-delivery method gated by a capability flag, not an op_type allowlist.** This is the weight-delivery coordinate of the Point made concrete; any weight-owning Kernel declares "emits runtime weight blob" and the driver step queries the capability instead of string-matching prefixes (which today silently skips new weight-bearing ops). |
| 6 | `minimize_weight_bit_width(model)` | `step_minimize_bit_width`, `minimize_weight_bit_width` (2) | undeclared (leaf, hasattr-guarded) | **Promote the hasattr duck-type to an explicit optional capability** ("owns minimizable weights"). Derive the tightened `weightDataType` from the Point's param/datatype derivations rather than a graph-mutating leaf method. A Kernel with no weights legitimately declares the capability absent. |
| 7 | `minimize_accumulator_width(model)` | `step_minimize_bit_width`, `minimize_accumulator_width` (2) | undeclared (leaf, hasattr-guarded) | Same pattern as #6 for `accDataType`/`outputDataType`. Declare "owns minimizable accumulator" capability; compute the narrowed dtype from the compute/datatype model, not by reaching into weight/threshold internals. |
| 8 | `adapt_for_loop_body(signature)` | `step_loop_rolling`, `loop_rolling` (2) | undeclared (optional MLO hook, errors swallowed) | **Derive from the Point's param-staticness coordinate, not a mutation hook.** The const→input flip is already a design-space axis (staticness); expose it as an explicit "re-resolve param delivery for streamed loop body" capability with a no-op default, so KernelOps that don't participate simply don't declare it. |
| 9 | `get_scale(model)` / `get_bias(model)` | `absorb_into_requant` (2) | undeclared (Requant-leaf) | **Absorb into generic resolved-param access.** These read scale/bias initializers off a Requant instance; expose them as declared parameter accessors (or fold the absorb rewrite into the Point's param model) rather than Requant-specific methods. |
| 10 | `derive_characteristic_fxns(...)` / `get_io_chrc_in()` / `get_io_chrc_out()` | `derive_characteristic` (3) | undeclared (leaf, needs working rtlsim) | **Prefer eliminating the rtlsim dependency: derive the I/O characteristic analytically** from `get_exp_cycles` + folding where possible. Where measured characterization is still needed, expose it as one declared "characterize" capability returning the `io_chrc_*` arrays as values, replacing the hidden nodeattr side-channel + three leaf accessors. |
| 11 | `get_shifts()` / `get_accum_size()` | `convert_to_hw_layers` (2) | undeclared (but on the **SOURCE** QuantAvgPool2d QONNX op, not a KernelOp) | **Eliminate from the KernelOp contract entirely.** These are read off the frontend op being *converted*, never off the produced HW node. KernelOp creation is "constructible from a flat attribute dict + domain/backend markers" (per `convert_to_hw_layers` finding); no obligation flows to the instance. |

### Key structural takeaways for the adapter

- **The 8 getters are all port-indexed.** `get_input_datatype`, `get_output_datatype`, `get_folded_input/output_shape`, `get_normal_input/output_shape`, `get_instream/outstream_width` are every one called with an explicit port index somewhere (weight stream = index 1). The KernelOp shape/dtype surface must be multi-port from the start (aligns with the port-taxonomy build).
- **`get_nodeattr`/`set_nodeattr` dominate** (~240 combined calls) and are the entire op-contact surface of ~15 units (`create_dataflow_partition`, `floorplan_params`, `fifo_transaction_counts`, `apply_folding_config`, `transpose_decomposition`, etc.). Satisfying the nodeattr *schema* (every key in the corpus) is the single largest obligation — and the cleanest to serve hermetically, since all are resolved-config or side-channel reads/writes, not graph traversals.
- **Two ABC layers, both required.** `step_tidy_up` + the qonnx transforms exercise the *lower* base-CustomOp ABC (`make_shape_compatible_op`, `infer_node_datatype`, `execute_node`) on **un-specialized** nodes; the HWCustomOp getters/estimators/codegen are the *upper* layer used post-specialization. The adapter must satisfy both.
- **Codegen entrypoints hide the real fan-out.** `code_generation_ipgen/cppsim` and `ipgen/compile_singlenode_code` are the only public codegen calls, but each cascades into ~10 undeclared leaf emit-methods (`generate_params`, `pragmas`, `docompute`, `generate_hdl`, …). `emit()` should own that cascade privately, driven by the resolved Point, so none of it re-enters the public contract.

---

## 2. The nodeattr surface (the untyped string side-channel)

Every op-instance interaction in the flow that is *not* a typed shape/datatype getter reduces to two untyped accessors — `get_nodeattr(str)` / `set_nodeattr(str, val)` — plus raw-protobuf `get_by_name` reads for a handful of gate attrs (`backend`, `direction`, `PE`). This is the widest and least-typed part of the contract. The union below is every attr the build flow reads or writes on an op instance, with the classification each consuming unit assigned it. Where a single attr is classified differently by different passes, the split is shown — that split is itself the signal that the attr is doing double duty (genuine config in one pass, cross-pass stash in another).

### 2.1 Design-config attrs (map to Axes / Derived — KernelOp keeps these)

These are genuine design coordinates or values derived from them. A KernelOp exposes them as declared, resolvable data; they are hermetic because they are read-as-configuration, not stashed by an upstream pass.

| attr | read / written by | classification | KernelOp disposition |
|---|---|---|---|
| `PE` | set_folding (rw), apply_folding_config (w), insert_hook (r), target_fps (rw), set_fifo_depths (r); also raw-protobuf in insert_iodma | config | **Axis** (folding) |
| `SIMD` | set_folding (rw), apply_folding_config (w), target_fps (rw), set_fifo_depths (r), transpose_decomp (r) | config | **Axis** (folding) |
| `parallel_window` | set_folding (w), target_fps (rw), apply_folding_config, set_fifo_depths (r) | config | **Axis** (folding) |
| `ram_style` | res_estimation (rw), set_folding (rw), apply_folding_config (w), target_fps (r), estimate_reports (rw) | config | **Axis** (memory/impl) |
| `resType` | res_estimation (rw), set_folding, apply_folding_config (w), target_fps (r), estimate_reports (rw) | config | **Axis** (impl: dsp/lut) |
| `mem_mode` | specialize/convert/floorplan/insert_iodma/make_zynq (r), set_fifo_depths (rw), target_fps/minimize (r) | config | **Axis** (param-delivery) |
| `runtime_writeable_weights` | make_driver, minimize_bitwidth, target_fps, set_fifo_depths (r); loop_rolling | config | **Axis** (param staticness) |
| `depth_trigger_uram` / `depth_trigger_bram` | target_fps (r), set_fifo_depths (r) | config | Derived / memory tuning |
| `depth` | set_fifo_depths (rw), insert_fifo (r) | config | FIFO-op config |
| `impl_style` | specialize/create_stitched_ip (rw), set_fifo_depths (w), floorplan (r) | config | **Axis** (hls/rtl or FIFO impl) |
| `preferred_impl_style` | convert_to_hw (produces), create_dataflow_partition (r), specialize (r, then **dropped**) | config | Axis-preference; consumed+dropped at specialize |
| `inWidth` / `outWidth` | specialize (r), floorplan (r) | config | **Derived** (stream widths) |
| `noActivation` | specialize, minimize, round_thresholds (r) | config | Axis (MVAU/VVAU) |
| `binaryXnorMode` | specialize, minimize (r) | config | Axis (MVAU) |
| `narrow` | specialize, convert_to_hw (r) | config | Axis (quant) |
| `lhs_style` / `rhs_style` | specialize, absorb_into_requant, derive_characteristic, insert_tlastmarker (r); set_fifo_depths / loop_rolling (w) | config | **Axis** (elementwise param-delivery) |
| `lhs_shape` / `rhs_shape` / `out_shape` | specialize (r) | config | **Derived** (elementwise geometry) |
| `MW` / `MH` | set_folding, minimize, target_fps (r) | config | **Derived** (weight-matrix geometry) |
| `NumChannels` `Labels` `Channels` `Kernel` `KernelSize` `IFMChannels` `ChannelsPerStream` `depthwise` | set_folding, target_fps (r) | config | **Derived** (per-op-family geometry) |
| `in_shape` `transpose_in_shape` `transpose_out_shape` `out_shape` `perm` `data_type` (Shuffle) | transpose_decomposition (r) | config | Derived (shuffle geometry) |
| `data_layout` | InferDataLayouts, convert_to_hw (r) | config | Config (layout) |
| `out_scale` / `out_bias` / `out_dtype` / `rounding_mode` | convert_to_hw (r), bipolar_to_xnor (rw) | config | Config (MultiThreshold quant) |
| `pumpedCompute` / `pumpedMemory` | create_stitched_ip (r, try/except — soft) | config | Axis (double-pump), optional |
| `DynIters` | floorplan (r) | config | Config (TLastMarker) |
| `output_hook` | insert_hook (r) | config | Config (checksum) |
| `dataType` / `folded_shape` | set_fifo_depths, insert_fifo (r); fifo_transaction_counts (r) | config | **Derived** (FIFO stream) |
| `iteration` | create_stitched_ip, fifo_transaction_counts (r); loop_rolling (w) | config | Container config (loop count) |
| `backend` | prepare_* (r), create_dataflow_partition, exp_cycles (raw-protobuf gate); loop_rolling (w) | config / ordering | **Discriminator** (must advertise `fpgadataflow`) |
| `inputDataType` / `outputDataType` | absorb_into_requant, loop_rolling (w) | config | Derived (dtypes) |
| `lhs_dtype` / `rhs_dtype` / `out_dtype` | hw_ipgen DSP-conflict, create_stitched_ip (r) | config | Derived (elementwise dtypes) |
| conv/pool source geometry: `kernel` `stride` `kernel_shape` `strides` `kernel_size` `pad_amount` `dilations` `pad_value` | convert_to_hw (r) | config | **Read off SOURCE op, not KernelOp** — no obligation |

### 2.2 Container attrs (subgraph-valued — KernelOp keeps, but they are graph handles)

| attr | read / written by | classification | KernelOp disposition |
|---|---|---|---|
| `body` | transpose_decomp, target_fps, apply_folding, estimate_reports, hw_codegen, hw_ipgen, set_fifo_depths, create_stitched_ip, loop_rolling (rw) | config / **side_channel** (hw_ipgen) | Container attr (FINNLoop subgraph). Legitimate for a container op, but it stashes a raw `GraphProto` and several passes round-trip it — a KernelOp must treat it as an owned subgraph, not a data field |
| `model` | create_dataflow_partition, annotate_cycles, post_synth_res, make_driver, make_zynq, alveo_build, measure_rtlsim (r) | config / **side_channel** | **Container/coupling.** A filesystem *path* to a serialized child graph, re-loaded by many passes. Genuine cross-step side-channel; only StreamingDataflowPartition needs it |
| `mlo_max_iter` | set_fifo_depths (rw), insert_fifo, create_stitched_ip (r); loop_rolling, insert_tlastmarker-shard (w) | config / **side_channel** | Loop-context stash written by loop_rolling for later FIFO/stitch passes — **coupling**, not design config |

### 2.3 side_channel attrs — one pass stashing state for another (KernelOp must NOT rely on these)

These carry no design intent; each is written by one pass purely so a later pass can read it back. A hermetic KernelOp must not depend on their persistence — the state they carry belongs in the build harness / emit context, not on the op node.

| attr | writer → reader (the handshake) | classification | KernelOp disposition |
|---|---|---|---|
| `cycles_estimate` | AnnotateCycles (w) → dataflow_performance (r); ubiquitous across estimate/fifo/stitch/measure steps | **side_channel** | **Drop.** Cost value; recompute via `get_exp_cycles`, don't persist a stamped attr |
| `inFIFODepths` / `outFIFODepths` | set_fifo_depths / SetFolding (w) → insert_fifo, insert_iodma (r); apply_folding_config, derive_characteristic (rw) | side_channel (also labeled config in apply/target passes) | **Drop from op.** Per-port FIFO-depth lists used as a cross-pass channel; belongs to FIFO-sizing state |
| `weightDataType` / `accDataType` / `outputDataType` | minimize_* (w) → round_thresholds / codegen (r) | **side_channel** | Values are *derived dtypes*; the coupling is that minimize stashes them mid-flow. KernelOp derives them, does not treat as durable config |
| `iteration_context_path` | minimize_bitwidth cppsim branch (w, then cleared to "") | **side_channel** | **Drop** (verification scratch) |
| `original_node_name` / `original_simd` | ShuffleDecomposition (w) → InferInnerOuterShuffles / config consolidation (r) | **side_channel** | **Drop** (decomposition provenance) |
| `io_chrc_in_file` / `io_chrc_out_file` / `io_chrc_period` | DeriveCharacteristic (w) → DeriveFIFOSizes (r), then unlinked | **side_channel** | **Drop** (rtlsim characterization scratch; also reached via undeclared leaf methods) |
| `depth_monitor` / `debug_log_path` | set_fifo_depths (w) — StreamingFIFO_rtl only | **side_channel** | **Drop** (rtlsim instrumentation) |
| `res_estimate` / `res_hls` / `res_synth` | AnnotateResources (w) → reporting (r) | **side_channel** | **Drop** (dynamically-named annotation) |
| `executable_path` | CompileCppSim (w) → cppsim exec (r) | **side_channel** | **Drop / harness-owned** |
| `instance_name` | make_zynq, alveo_build (w) → make_driver, deploy (r) | **side_channel** | **Drop / harness-owned** (Vivado bd_cell name) |
| `ipgen_path` | HLSSynthIP (w) → stitch, replace_relpaths, prepare_rtlsim (r) | side_channel / **ordering** | **Drop** (see 2.4) |

### 2.4 ordering / harness-owned attrs — codegen-pipeline handoff paths (KernelOp must NOT rely on these)

These exist *only after* a prior pass populated them; they are pass-to-pass handoff of filesystem paths and execution state, not op configuration. In a hermetic KernelOp these are owned by the emit/build harness (the resolved-config-as-data + Artifacts model), never by the op node.

| attr | read / written by | classification | KernelOp disposition |
|---|---|---|---|
| `code_gen_dir_ipgen` | PrepareIP (rw) → HLSSynthIP, stitch, hls_synth_res_estimation (r); cleanup blanks it; set_fifo_depths reset | **ordering** | **Harness-owned.** Codegen dir path — emit() writes Artifacts, does not stash a dir on the node |
| `code_gen_dir_cppsim` | PrepareCppSim (rw) → CompileCppSim (r); cleanup blanks it | **ordering** | **Harness-owned** |
| `ip_path` | ipgen (w) → create_stitched_ip, make_zynq collect_ip_dirs (r) | **ordering** | **Harness-owned** (per-node IP dir) |
| `rtlsim_so` | prepare_rtlsim (w) → rtlsim exec (r) | **ordering** | **Harness-owned** (compiled sim lib) |
| `rtlsim_trace` | hw_ipgen, set_fifo_depths, verify_step (w) | config / **side_channel** | **Harness-owned** (debug/trace toggle) |
| `exec_mode` | set_exec_mode (rw) → node execution dispatch | config | **Harness-owned** (cppsim/rtlsim selector) |

### 2.5 Placement attrs — floorplan-owned config (KernelOp surfaces, but a placement pass fills them)

| attr | read / written by | classification | KernelOp disposition |
|---|---|---|---|
| `slr` | create_dataflow_partition (rw), floorplan (rw), alveo_build, floorplan_params (r) | config / **ordering** | Surface as readable/writable placement knob; value produced by Floorplan, not by the op |
| `partition_id` | create_dataflow_partition (rw), floorplan (rw), floorplan_params (r) | config / **ordering** / **side_channel** | Same — placement, harness-assigned |
| `mem_port` | create_dataflow_partition (rw), floorplan_params, alveo_build (r) | config / **side_channel** | Same — placement, harness-assigned |
| `burstMode` | create_dataflow_partition, make_driver (r; raw-protobuf) | config | Config (IODMA); only external-DMA container ops |

### 2.6 Traffic summary

- **~50 distinct attr names** cross the op boundary. Of these, **~30 are genuine design-config** (Axes/Derived — §2.1), **3 are container handles** (§2.2), and **~17 are coupling attrs** that a hermetic KernelOp must *not* depend on: **~11 side_channel** (§2.3) + **6 ordering/harness-owned** (§2.4), with `slr`/`partition_id`/`mem_port` straddling config-vs-placement (§2.5).
- **Read/write volume is heavily concentrated:** the folding + FIFO-sizing shard (`set_folding` + `set_fifo_depths`) alone accounts for ~40 `get_nodeattr` reads and ~45 `set_nodeattr` writes; the stitching shard (`create_stitched_ip` + partition) adds ~30 more reads. Across the whole census the untyped accessor is invoked on the order of **150+ reads and 70+ writes**, versus a comparatively small number of typed shape/datatype getter calls — i.e. **the string side-channel, not the typed ABC, is where most op coupling actually lives.**
- **The disposition rule for KernelOp:** config/derived attrs (§2.1) become resolved design-space coordinates the op exposes as data; container attrs (§2.2) become owned subgraphs; and **every side_channel (§2.3) and ordering (§2.4) attr must migrate off the op into the build/emit harness** — these are exactly the "one pass stashing state for another" leaks that break `emit()` hermeticity if a KernelOp keeps reading them off the node.

---

## 3. The build-step orchestration contract

The 20 build steps form a single-threaded pipeline in which each step both *demands* a certain maturity from every op node it touches and *gates* the maturity the next step assumes. A KernelOp cannot satisfy the contract as one flat interface: the surface it must present grows monotonically with pipeline depth, and the pipeline splits into a shallow **estimate-only** sub-flow and a deep **full-build** sub-flow that impose disjoint method sets. For a partially-migrated graph, G4's adapter must present *whichever depth's surface the current step is at* for each node independently.

### 3.1 Ordered walk: precondition ⊳ gate per step

The step body is almost never the direct consumer — nearly every op-contract touch flows through the qonnx/fpgadataflow transforms a step orchestrates (audited as their own units). The chain below states, per step, the **precondition on op nodes** and **what op-state it gates downstream**.

*Ordering verified against `default_build_dataflow_steps` (`build_dataflow_config.py:110–130`): the canonical sequence is minimize_bit_width(9) → transpose_decomposition(10) → generate_estimate_reports(11), and synthesize_bitfile(17) → make_driver(18). `step_loop_rolling` is MLO-gated and is NOT in the default list (retained as #20 here, flagged).*

| # | Step | Precondition on op nodes | Gates downstream |
|---|------|--------------------------|------------------|
| 1 | `step_qonnx_to_finn` | none — runs on QONNX/standard ONNX before any op node exists | FINN-ONNX dialect graph (topology gate, not an op-contract gate) |
| 2 | `step_tidy_up` | node resolvable via `getCustomOp`; implements base CustomOp ABC (`make_shape_compatible_op`, `infer_node_datatype`, `execute_node`) | fully-populated tensor shapes + QONNX datatypes + canonical unique node/tensor names |
| 3 | `step_streamline` | still standard/QONNX ops (pre-HW) | op_type topology that the `Infer*` matchers recognize |
| 4 | `step_convert_to_hw` | frontend op_type spellings match the `Infer*` pattern-matchers (reads **source**-node attrs only) | **abstract HW nodes** (bare op_type, `domain=finn.custom_op.fpgadataflow`, `backend=fpgadataflow`) each exposing `preferred_impl_style` |
| 5 | `step_create_dataflow_partition` | `backend==fpgadataflow`; readable `partition_id`/`slr`/`mem_port`; SDP container carries `model`; readable `preferred_impl_style` | child dataflow model; `template_specialize_layers_config.json`; SDP placement attrs |
| 6 | `step_specialize_layers` | abstract HW node registered for bare op_type; `get_input_datatype(idx)`/`get_output_datatype(idx)` return DataType; `preferred_impl_style ∈ {"","hls","rtl"}`; a registered `_hls`/`_rtl` variant exists | **specialized leaf** (`op_type+_hls/_rtl`, `domain=...hls/.rtl`); `preferred_impl_style` dropped; all other attrs copied verbatim; concrete `getCustomOp` → HLS/RTLBackend leaf |
| 7 | `step_target_fps_parallelization` | nodes `_hls/_rtl`; working `get_exp_cycles()` monotone in the folding attr; per-op folding-source attrs (`MW/MH/NumChannels/…`) settable | resolved `PE/SIMD/parallel_window`; `cycles_estimate` annotated; `auto_folding_config.json` |
| 8 | `step_apply_folding_config` | every JSON attr name is a **declared** nodeattr (`set_nodeattr` asserts membership) | folding/FIFO-depth attrs consumed by estimate, FIFO, and codegen steps |
| 9 | `step_minimize_bit_width` | nodes fpgadataflow-specialized; weight/threshold initializers present; `MW/MH/mem_mode/…` attrs exist; optional `minimize_weight_bit_width`/`minimize_accumulator_width` (hasattr-guarded) | tightened `accDataType`/`weightDataType`/`outputDataType`, re-propagated via `infer_node_datatype`. **In the estimate-only sub-flow** |
| 10 | `step_transpose_decomposition` | folding configured; Shuffle nodes rewritable; FINNLoop exposes `body` subgraph | Inner/OuterShuffle ops already `_hls/_rtl`; Shuffle removed. **Full-build only — NOT in estimate-only** |
| 11 | `step_generate_estimate_reports` | nodes `_hls/_rtl`; `get_exp_cycles()`, `node_res_estimation(fpgapart)`, optional `get_op_and_param_counts()`; `resType`/`ram_style` sweepable | `cycles_estimate` on every node; estimate JSON artifacts. **← last step of the estimate-only sub-flow** (`estimate_only_dataflow_steps` = steps 1–9 + 11, skipping transpose_decomposition) |
| 12 | `step_hw_codegen` | `_hls/_rtl` leaf; declares `code_gen_dir_ipgen`; folding+datatypes+params resolved | `code_gen_dir_ipgen` populated via `code_generation_ipgen(model,fpgapart,clk)` (HLS C++/tcl or filled RTL) |
| 13 | `step_hw_ipgen` | HLS: non-empty `code_gen_dir_ipgen`; implements `ipgen_singlenode_code(fpgapart)`; declares `ipgen_path` (RTL nodes pass-through) | `ipgen_path` → synthesized IP block; Verilog rel-paths absolutized |
| 14 | `step_set_fifo_depths` | nodes specializable & codegen/IP/rtlsim-buildable; per-port `inFIFODepths`/`outFIFODepths`; indexed `get_folded_input_shape(i)`/`get_folded_output_shape(o)` | sized FIFO nodes; `final_hw_config.json`; forces fresh stitched-IP build |
| 15 | `step_create_stitched_ip` | every node `_hls/_rtl` with IP already generated; `get_verilog_top_module_intf_names()` (clk/rst/…/m_axis/s_axis/aximm/ap_none, positional tuples); `code_generation_ipi()` | `vivado_stitch_proj` metadata; stitched IP dir |
| 16 | `step_measure_rtlsim_performance` | stitched IP present; `get_exp_cycles()`; boundary `get_folded_input_shape(ind)`/`get_folded_output_shape(ind)` | `rtlsim_performance.json` (terminal report) |
| 17 | `step_synthesize_bitfile` | SDP exposes recursable `model`; leaf `node.name` matches synth report rows | bitfile/xclbin + `post_synth_resources.json` |
| 18 | `step_make_driver` | top graph all-SDP; boundary IODMA_hls; `get_folded_input/output_shape`; `make_weight_file` on runtime-writable MVAU/Thresholding; `.onnx_node` reach-through | driver dir (`pynq_driver_dir`/`cpp_driver_dir`) + runtime weight blobs |
| 19 | `step_deployment_package` | none (filesystem copy) | `deploy/` dir |
| 20 | `step_loop_rolling` (MLO-gated, not in default list) | body ops are concrete leaves; optional `adapt_for_loop_body(signature)` (errors swallowed); FINNLoop `body` container | FINNLoop op; `mlo_max_iter`/`inFIFODepths` stamped on loop-input consumers |

**The load-bearing dependency chain** (each arrow = a hard gate the left step produces and the right step's precondition consumes):

```
tidy_up ─(shapes+dtypes+names)→ convert_to_hw ─(abstract HW node + preferred_impl_style)→
create_dataflow_partition ─(backend/placement/child-model)→ specialize_layers ─(_hls/_rtl leaf, concrete getCustomOp)→
{ folding: target_fps_parallelization / apply_folding_config } ─(PE/SIMD + cycles_estimate)→
minimize_bit_width ─(final datatypes)→ [estimate-only stops after generate_estimate_reports]
transpose_decomposition ─(shuffles lowered)→ generate_estimate_reports ─→ hw_codegen ─(code_gen_dir_ipgen populated)→
hw_ipgen ─(ipgen_path / synthesized IP)→ set_fifo_depths ─(sized FIFOs)→
create_stitched_ip ─(vivado_stitch_proj)→ measure_rtlsim_performance / make_driver / synthesize_bitfile ─→ deployment_package
```

The single hardest gate is **`specialize_layers`**: everything left of it operates on the *abstract* HW node (base-ABC + config attrs), everything right of it requires a *concrete `_hls`/`_rtl` leaf* resolvable by `getCustomOp` to a backend mixin. This is the boundary at which the op-contract surface roughly doubles.

### 3.2 The nested sub-contracts by pipeline depth

A KernelOp presents a *different* contract at each depth. These are strictly nested — a deeper tier presupposes all shallower ones — but the estimate-only branch (Tier 3) and the full-build branch (Tier 4) fork after specialization and demand disjoint method families. The adapter must be able to serve **any** tier on demand.

**Tier 0 — base CustomOp ABC (steps 2, 9 re-propagation).** The op must be a registered custom op whose `getCustomOp` resolves, and implement `make_shape_compatible_op(model)`, `infer_node_datatype(model)`, `execute_node(context,graph)`, plus `get_nodeattr_types` so the wrapper is constructible. This is the qonnx CustomOp layer — *lower* than the HWCustomOp shape/datatype getters — and is required before any HW conversion exists.

**Tier 1 — abstract HW node / flat config surface (steps 4–5, 8–9).** The op must be constructible from a flat attribute dict tagged `domain=finn.custom_op.fpgadataflow`, `backend=fpgadataflow`, and surface the placement/config knobs read via the universal `get_nodeattr`/`set_nodeattr` accessors: `preferred_impl_style`, `partition_id`, `slr`, `mem_port`, and every declared folding knob (`PE`, `SIMD`, `parallel_window`, `ram_style`, `mem_mode`, …). `set_nodeattr` *asserts declared membership*, so a KernelOp must declare every attr any config JSON might name. No shape/datatype methods and no codegen are exercised yet.

**Tier 2 — the specialization decision (step 6).** The op must answer `get_input_datatype(idx)` and `get_output_datatype(idx)` **in their indexed form** (0=activation, 1=weights), returning DataTypes supporting `bitwidth()/signed()/min()`/`=="FLOAT32"`, and must expose the per-op-family feasibility knobs the central `SpecializeLayers` predicates reach into (`inWidth/outWidth`, `noActivation`, `binaryXnorMode`, `lhs_style/rhs_style/…shape`, `narrow`). It must register both a bare abstract class *and* an `_hls`/`_rtl` variant. Note the current design does **not** delegate the impl-style choice to the op — a KernelOp must either self-declare feasibility or the transform's op_type-family dispatch must be generalized.

**Tier 3 — the ESTIMATE-ONLY sub-contract (steps 7–9, 11; `estimate_only_dataflow_steps` = steps 1–9 + 11).** After specialization, the shallow branch needs **cost and shape, but NO codegen, IP, or rtlsim**. It *includes* `step_minimize_bit_width` (step 9) — so bit-width tightening is part of the estimate-only surface — but *excludes* `step_transpose_decomposition` (step 10, full-build only). The complete op surface is:
- `get_exp_cycles()` → int (the load-bearing cost hook driving `SetFolding`, `AnnotateCycles`, `dataflow_performance`);
- `node_res_estimation(fpgapart)` → resource dict, with `get_nodeattr_types` exposing 4-tuple allowed-value specs for the `resType`/`ram_style` sweep;
- the `minimize_weight_bit_width`/`minimize_accumulator_width` hasattr-guarded mutators (weight-bearing ops), which run in this sub-flow at step 9;
- optional `get_op_and_param_counts()` (hasattr-guarded, silently skipped if absent);
- the `cycles_estimate` **side-channel** nodeattr (written by `AnnotateCycles`, read by `dataflow_performance` — the required handshake);
- placement attrs `slr`/`partition_id`/`mem_port` for `floorplan_params`.

No `code_gen_dir_*`, no `ipgen_path`, no backend codegen method is touched. A KernelOp that implements only Tiers 0–3 can be carried all the way to `estimate_layer_resources.json`/`estimate_network_performance.json`. **This is the cheapest useful migration target** and the natural first increment for G4.

**Tier 4 — the FULL-BUILD sub-contract (steps 10, 12–18).** The deep branch adds the backend-mixin orchestration entry points, none of which are on the base ABC — they live on `HLSBackend`/`RTLBackend` and must be routed distinctly per impl:
- `code_generation_ipgen(model,fpgapart,clk)` emitting into `code_gen_dir_ipgen` (HLS *and* RTL);
- HLS-only `ipgen_singlenode_code(fpgapart)` → `ipgen_path`, and `compile_singlenode_code()`/`code_generation_cppsim` for the cppsim path;
- `prepare_rtlsim(behav)` → `rtlsim_so`;
- `get_verilog_top_module_intf_names()` (with its *undeclared structural sub-contract*: exact dict keys incl. `clk2x`/`ap_none` and positional `(name,width)` tuples) and `code_generation_ipi()` for stitching;
- boundary `get_folded_input_shape(ind)`/`get_folded_output_shape(ind)`, and the `_padded` width variants (`get_instream_width_padded`/`get_outstream_width_padded`) for FIFO/IODMA/DWC sizing;
- weight-bearing leaves only: `make_weight_file(...)` and the `minimize_weight_bit_width`/`minimize_accumulator_width` hasattr-guarded mutators;
- the ordering side-channels `code_gen_dir_ipgen`, `ipgen_path`, `ip_path`, `executable_path`, `rtlsim_so`, plus `io_chrc_*` characterization attrs and their `get_io_chrc_in/out`/`derive_characteristic_fxns` accessors.

Plus the HLS/RTL codegen fan-out (`get_ap_int_max_w`, `generate_params`, `global_includes`, `defines`, `docompute`, `pragmas`, `generate_hdl`, …) reached through `code_generation_ipgen`.

**Container/MLO tier (steps 7, 20, and FINNLoop branches throughout).** Orthogonal to the above: the `FINNLoop` container's `body` subgraph nodeattr (`get_nodeattr("body")`/`set_nodeattr("body", graph)`), the optional `adapt_for_loop_body(signature)` hook, and the `mlo_max_iter`/`inFIFODepths`/`iteration` side-channels. Similarly `StreamingDataflowPartition`'s `model` file-path attr threads through nearly every recursive step. A generic compute KernelOp has no `body`/`model`, so the adapter must either not claim to be a container or answer these container attrs.

### 3.3 Implication for G4's mixed-graph adapter

Because the tiers are nested and the pipeline advances all nodes in lockstep, **the adapter must serve, node-by-node, exactly the tier the current step sits at** — an estimate-only KernelOp and a full-build KernelOp must both survive steps 2–10 presenting identical Tier-0…3 surfaces, then diverge only when a full build proceeds past step 10. Concretely:

1. **Every KernelOp must satisfy Tiers 0–3 unconditionally** — base ABC, flat declared-nodeattr config surface, indexed datatype getters, `get_exp_cycles`, `node_res_estimation`, and the `cycles_estimate` handshake. This is the floor for *any* participation in the flow, old or new.
2. **`specialize_layers` (step 6) is the migration seam.** Left of it, old MVAU-style ops and new KernelOps must be indistinguishable at the abstract-HW-node + `preferred_impl_style` level. Right of it, each must resolve to a concrete backend leaf; a partially-migrated graph will contain both classic `_hls/_rtl` leaves and KernelOp-backed leaves that must answer the *same* Tier-4 backend-mixin surface (`code_generation_ipgen`, `get_verilog_top_module_intf_names`, `code_generation_ipi`, `ipgen_singlenode_code`).
3. **The undeclared/leaf-only couplings are where mixed graphs break.** `make_weight_file`, `minimize_*`, `adapt_for_loop_body`, `get_io_chrc_*`, and the positional-tuple structure of `get_verilog_top_module_intf_names` are duck-typed (hasattr-guarded or structurally implicit), so a KernelOp that omits them is *silently* skipped rather than erroring — producing unminimized resources, unstitchable interfaces, or missing runtime weights with no diagnostic. G4's adapter must implement these explicitly (or declare their absence via a capability query) rather than relying on the classic silent-skip semantics.
4. **The op_type-string dispatch is the pervasive hidden dependency.** Steps and transforms branch on literal op_type prefixes (`MVAU*`, `Thresholding*`, `VVAU*`, `Elementwise*`, `StreamingFIFO*`, `IODMA_hls`, `FINNLoop`, `StreamingDataflowPartition`) for folding bounds, weight emission, DSP-conflict detection, driver generation, and container recursion. A KernelOp with a novel op_type is invisible to all of these unless the adapter either matches an expected spelling or the dispatch is refactored to a capability query — the single largest structural obligation the orchestration contract places on the KernelOp redesign.

---

## 4. Clean-vs-leaky: the adapter verdict

The preceding three sections enumerate everything the flow touches. This section collapses that into a build decision: the exact surface a KernelOp compatibility adapter must expose, the exact surface it must **refuse to carry forward** (deriving instead from the resolved Point), and the per-transform hazards that will bite during a mixed-graph migration. The governing principle: **the adapter is a projection of the resolved Point onto the legacy op-instance API — every method answers from resolved-config-as-data, never by reading state another pass stashed on the node.**

### 4a. THE ADAPTER METHOD LIST — the minimal surface to run the pipeline

This is the closed set. If the adapter exposes exactly these and nothing else, a KernelOp-backed node survives all 20 steps on a mixed graph. Grouped by the tier (Section 3.2) that first demands them, because a partially-migrated node only needs the surface up to the depth the flow has reached.

**Tier 0 — base qonnx CustomOp ABC (unconditional; every node, every graph)**
| method | signature | answers from |
|---|---|---|
| `get_nodeattr_types()` | → dict incl. 4-tuple allowed-value specs | the Point's declared axis/derived schema |
| `make_shape_compatible_op(model)` | → standard-ONNX surrogate node | Point output geometry |
| `infer_node_datatype(model)` | annotates output tensor dtypes; idempotent | Point datatype derivations |
| `execute_node(context, graph)` | functional exec (const-fold path; rare) | Point compute semantics |

**Tier 1 — flat config surface (the untyped accessor pair — carries ~240 of all calls)**
| method | signature | answers from |
|---|---|---|
| `get_nodeattr(key)` | → resolved value for **every** key in the §2.1 union | resolved Point coordinate/derivation; tolerate absent keys |
| `set_nodeattr(key, val)` | asserts `key` is declared, then stores | writes resolved config/derived state only (never graph) |
| `onnx_node` | property → wrapped protobuf (`.name/.input/.output/.op_type`) | the node shell |

**Tier 2 — the specialization decision (indexed dtype getters)**
| method | signature | answers from |
|---|---|---|
| `get_input_datatype(ind=0)` | → DataType (`.bitwidth/.signed/.min/=="FLOAT32"`) | Point port-typed inputs (1 = weights) |
| `get_output_datatype(ind=0)` | → DataType | Point port-typed outputs |

**Tier 3 — estimate-only (the cheapest useful migration target; stops at step 10)**
| method | signature | answers from |
|---|---|---|
| `get_exp_cycles()` | → int, monotone in folding attr | Point compute cardinality ÷ folding |
| `node_res_estimation(fpgapart)` | → `{BRAM_18K/LUT/URAM/DSP/…}` | Point geometry + impl axis; fans out to bram/lut/uram/dsp/`*_efficiency` **internally** |
| `get_op_and_param_counts()` | → dict, default `{}` | Point compute + param cardinality |

**Tier 3 shape/width getters (used by folding/FIFO/DWC/IODMA sizing — all port-indexed)**
| method | note |
|---|---|
| `get_normal_input_shape(ind)` / `get_normal_output_shape(ind)` | getters 5–6 of 8 |
| `get_folded_input_shape(ind)` / `get_folded_output_shape(ind)` | getters 2–3 of 8; PE/SIMD-folded |
| `get_instream_width(ind)` / `get_outstream_width(ind)` | getters 7–8 of 8 |
| `get_instream_width_padded(ind)` / `get_outstream_width_padded(ind)` | byte-padded derived variants (IODMA) |
| `get_number_output_values()` | TLastMarker NumIters |

**Tier 4 — full-build backend surface (routed per impl; declared on the KernelOp backend contract, NOT `is_hls_node`-gated leaf methods)**
| method | impl | answers from |
|---|---|---|
| `code_generation_ipgen(model, fpgapart, clk)` | HLS + RTL | Point → Artifacts into codegen dir; owns the entire private fan-out (`generate_params`, `pragmas`, `docompute`, `generate_hdl`, `get_ap_int_max_w`, `global_includes`, `defines`, `blackboxfunction`, `ipgen_*_directives`, `generate_infra_hdl`) |
| `ipgen_singlenode_code(fpgapart)` | HLS only | drives HLS synth; sets `ipgen_path` |
| `compile_singlenode_code()` | HLS only | cppsim compile; sets `executable_path` |
| `code_generation_cppsim(model)` | HLS only | cppsim C++ emit |
| `prepare_rtlsim(behav)` | HLS + RTL | rtlsim lib; sets `rtlsim_so` |
| `code_generation_ipi()` | HLS + RTL | → Vivado `create_bd_cell` Tcl list |
| `get_verilog_top_module_intf_names()` | HLS + RTL | dict with **exact** keys `clk/rst`(+`clk2x`)/`axilite`/`aximm`/`m_axis`/`s_axis`/`ap_none`; axis/aximm values are positional `(name,width)` tuples — honored verbatim |

**Capability-gated (declared explicitly, never duck-typed)**
| method / capability | replaces today's | answers from |
|---|---|---|
| `make_weight_file(w, mode, fname)` — gated by `owns_runtime_weights` | `op_type.startswith('MVAU'/'Thresholding')` allowlist | Point weight-delivery coordinate |
| `minimize_weight_bit_width(model)` — gated by `owns_minimizable_weights` | `hasattr` guard | Point param/datatype derivation |
| `minimize_accumulator_width(model)` — gated by `owns_minimizable_accumulator` | `hasattr` guard | Point compute/datatype derivation |
| `adapt_for_loop_body(signature)` — gated by `participates_in_loop_body`, no-op default | swallowed-error hook | Point param-staticness axis |
| container: `get_nodeattr("body")` / `set_nodeattr("body", graph)` | — | owned subgraph (FINNLoop only) |
| container: `get_nodeattr("model")` | — | child-graph path (StreamingDataflowPartition only) |

That is the entire width. **~30 methods + the two untyped accessors + one property.** Everything else the census names is either a private fan-out step (absorbed inside `code_generation_ipgen`) or a leak the adapter must *not* reproduce (4b).

### 4b. THE MUST-NOT-REPRODUCE LIST — satisfy by deriving, never by carrying forward

These are the couplings that break `emit()` hermeticity if a KernelOp keeps them on the node. Each is satisfied by the adapter computing the value from the resolved Point (or the harness owning it), so that the consuming pass reads a fresh derivation, not a stashed one.

**Side-channel nodeattrs — one pass stashing state for another (§2.3). Drop from the op; recompute or move to harness.**
- `cycles_estimate` — recompute via `get_exp_cycles()`; do not persist the AnnotateCycles stamp. (The AnnotateCycles→dataflow_performance handshake is served by the adapter answering `get_nodeattr("cycles_estimate")` from a live `get_exp_cycles()` call, not from a stored attr.)
- `inFIFODepths` / `outFIFODepths` — FIFO-sizing state; belongs to the harness's FIFO pass, not the Point.
- `weightDataType` / `accDataType` / `outputDataType` — derived dtypes; the adapter *computes* them on read, never treats the minimize-pass write as durable config.
- `io_chrc_in_file` / `io_chrc_out_file` / `io_chrc_period` — rtlsim characterization scratch; prefer analytic derivation (see Risk R6).
- `original_node_name` / `original_simd` — decomposition provenance; harness-owned.
- `iteration_context_path`, `depth_monitor`, `debug_log_path`, `res_estimate`/`res_hls`/`res_synth` — verification/instrumentation/reporting scratch; harness-owned.
- `instance_name`, `executable_path` — Vivado/build identifiers; harness-owned.

**Ordering / codegen-handoff nodeattrs (§2.4). Harness-owned; emit() returns Artifacts, does not stash paths on the node.**
- `code_gen_dir_ipgen`, `code_gen_dir_cppsim`, `ip_path`, `ipgen_path`, `rtlsim_so`, `rtlsim_trace`, `exec_mode` — every one is a pass-to-pass filesystem/exec-state handoff. The adapter answers `get_nodeattr` for these from the harness's build context (the resolved-config + Artifacts model), so the node never becomes the source of truth for a path.

**Placement nodeattrs (§2.5). Surface as read/write knobs, but the *value* is Floorplan-assigned, not Point-derived.**
- `slr`, `partition_id`, `mem_port` — the adapter must let these be written and read back, but they are harness/placement-owned; the Point declares that the op *has* a placement, not what it is.

**Undeclared leaf methods (§1B). Do not re-expose as `is_hls_node`/`hasattr`-gated duck types.**
- `get_scale(model)` / `get_bias(model)` (Requant) — fold into generic resolved-param access, or absorb the absorb-into-requant rewrite into the Point's param model.
- `get_io_chrc_in/out()` / `derive_characteristic_fxns()` — collapse into one declared "characterize" capability returning arrays as values (or eliminate via analytic derivation).
- `get_shifts()` / `get_accum_size()` — **eliminate entirely**: these are read off the *source* QuantAvgPool2d during conversion, never off the produced KernelOp. No obligation flows to the instance.

### 4c. RISK REGISTER — per-case migration hazards

Ranked by likelihood of a *silent* failure (wrong/missing output with no error) on a mixed graph.

| # | Reach point (transform/step) | What it does | Failure mode if adapter naive | Disposition |
|---|---|---|---|---|
| **R1** | op_type-string dispatch, **pervasive** (`set_folding`, `set_fifo_depths`, `make_zynq_proj`, `make_driver`, `create_stitched_ip`, `insert_iodma`, `specialize_layers`) | branches on literal prefixes `MVAU*`/`Thresholding*`/`VVAU*`/`Elementwise*`/`StreamingFIFO*`/`IODMA_hls`/`FINNLoop`/`StreamingDataflowPartition` for folding bounds, weight emit, DSP-conflict, driver gen, container recursion | novel op_type is **invisible** → unfolded, unminimized, no runtime weights, unstitched — no diagnostic | **Highest-priority refactor.** Replace op_type dispatch with capability queries on the KernelOp. Interim: adapter advertises a legacy-compatible op_type spelling so classic dispatch still fires. This is the single largest structural obligation. |
| **R2** | `step_make_driver` / `make_driver` → `make_weight_file`, behind `op_type.startswith('MVAU'/'Thresholding')` | emits runtime weight `.dat` for weight-bearing nodes | new weight-owning Kernel silently skipped → **missing runtime weights**, driver builds but mis-runs | Adapter declares `owns_runtime_weights` capability; driver step queries it instead of prefix-matching. Must land before any weight-bearing KernelOp ships. |
| **R3** | `step_minimize_bit_width` → `minimize_weight_bit_width` / `minimize_accumulator_width`, `hasattr`-guarded | tightens weight/accumulator dtypes | silent skip → **unminimized resources** (larger, slower), no error | Promote to explicit optional capabilities (`owns_minimizable_weights`/`_accumulator`). Adapter derives tightened dtype from the Point; a weightless Kernel legitimately declares absent. |
| **R4** | `create_stitched_ip` → `get_verilog_top_module_intf_names()` | reads positional `(name,width)` tuples + exact keys incl. `clk2x`/`ap_none` | wrong dict shape → **unstitchable interface** or mis-wired block design; structurally implicit, not type-checked | Adapter's emit interface-descriptor must reproduce the tuple structure and key set verbatim. Add a schema assertion in the adapter to convert silent structural drift into a hard error. |
| **R5** | `step_hw_ipgen` / `hlssynth_ip` → `ipgen_singlenode_code`; `prepare_cppsim`/`compile_cppsim`; `prepare_rtlsim` | HLS-leaf-only synth/compile/rtlsim entrypoints, gated by `is_hls_node` | RTL Kernel correctly omits; but an HLS Kernel that routes these as duck-typed leaves re-leaks the coupling | Declare all four on the KernelOp backend contract, keyed on impl. RTL kernels omit `ipgen_singlenode_code`/`compile_singlenode_code`/`code_generation_cppsim` by declaring HLS-only capability absent — not by `is_hls_node` accident. |
| **R6** | `derive_characteristic` → `derive_characteristic_fxns` + `get_io_chrc_in/out` + `io_chrc_*` nodeattrs | per-node rtlsim characterization for FIFO sizing; needs working rtlsim | hidden nodeattr side-channel + 3 leaf accessors; a Kernel omitting them is skipped → **mis-sized FIFOs** | Prefer analytic derivation from `get_exp_cycles` + folding, eliminating the rtlsim dependency. Where measured characterization is unavoidable, expose one declared `characterize()` capability returning the arrays as values; never persist `io_chrc_*` on the node. |
| **R7** | `annotate_cycles` (w) → `dataflow_performance` (r) via `cycles_estimate` | stamp-then-read handshake across estimate/fifo/stitch/measure | if adapter stores the stamp, it becomes durable node state (leak); if it ignores the write, `dataflow_performance` reads stale | Adapter serves `get_nodeattr("cycles_estimate")` from a live `get_exp_cycles()` computation; accepts the AnnotateCycles write as a harness-cache no-op. The handshake works, the node stays hermetic. |
| **R8** | `step_loop_rolling` / `loop_rolling` → `adapt_for_loop_body(signature)`, errors swallowed | flips param `const→input` for streamed loop bodies | swallowed error → Kernel silently keeps baked constant → **wrong MLO behavior** | Derive from the Point's param-staticness axis. Declare `participates_in_loop_body` with a no-op default; a non-participating Kernel declares absent explicitly rather than relying on the swallow. |
| **R9** | `absorb_into_requant` → `get_scale(model)`/`get_bias(model)` | reads Requant scale/bias initializers | Requant-specific leaf methods on the instance | Fold the absorb rewrite into the Point's param model, or expose scale/bias via generic resolved-param access. Not a per-node hazard for non-Requant Kernels. |
| **R10** | `NodeLocalTransformation` base (`PrepareIP`, `HLSSynthIP`, `PrepareRTLSim`, `PrepareCppSim`) | deepcopies model + `mp.Pool.map` pickles each NodeProto to workers | any non-picklable state the adapter lazily caches on the protobuf **breaks parallel dispatch** | Adapter must reconstruct all cached state from the resolved Point in the worker process; never stash non-picklable handles on the node. Aligns with the hermeticity contract — this risk *disappears* if 4b is honored. |
| **R11** | `step_specialize_layers` — the migration seam (step 6) | central `SpecializeLayers` reaches into per-op-family knobs (`inWidth/outWidth`, `noActivation`, `binaryXnorMode`, `lhs_style/rhs_style/*shape`, `narrow`) to choose impl; does **not** delegate to the op | a KernelOp whose feasibility isn't encoded in these exact knobs gets mis-specialized or unspecialized | Either the adapter self-declares hls/rtl feasibility (preferred — the impl axis is already a Point coordinate), or the transform's op_type-family dispatch is generalized to a `feasible_impls()` query. This is where the contract surface doubles; get it right or everything right of step 6 fails. |
| **R12** | `step_apply_folding_config` / `set_folding` → `set_nodeattr` asserts declared membership | writes JSON-named folding attrs | if the adapter doesn't **declare** every attr a config JSON might name, `set_nodeattr` raises | `get_nodeattr_types()` must enumerate the full §2.1 union (folding + memory + geometry knobs) so any folding/final-config JSON applies cleanly. Under-declaring is a hard failure (good — not silent). |

**Verdict.** The true width of the adapter is **~30 methods, two untyped accessors, one property, and five explicit capability flags** — everything in 4a. Everything in 4b is served by *derivation from the resolved Point*, not by carrying node state forward. The migration's hard seam is **`specialize_layers` (R11)**: identical Tier-0…3 surfaces left of it, disjoint Tier-4 backend surfaces right of it. The migration's silent-failure surface is **R1–R3 (op_type dispatch, weight emit, minimize)**: all three are today `hasattr`/prefix duck-typed, so a conforming-but-incomplete KernelOp is *skipped, not rejected*. G4 must convert every one of those silent skips into an explicit capability query before the corresponding KernelOp class ships.

---

## 5. Verification notes

*Single aggregate ground-truth pass (per-decision: synthesized-aggregate verification
only — individual units were not adversarially re-verified). What was checked, against
the finn tree + pinned `deps/qonnx`:*

**Confirmed.** The method contract (§1), nodeattr classifications (§2), and adapter
surface (§4) are directionally sound and internally consistent. High-traffic op-instance
methods (`get_nodeattr`/`set_nodeattr`, `get_exp_cycles`, `node_res_estimation`,
`code_generation_ipi`, `prepare_rtlsim`, `get_verilog_top_module_intf_names`,
`instantiate_ip`) and the flagged side-channel/ordering nodeattrs (`gen_top_module`,
`code_gen_dir_ipgen`, `ipgen_path`, `mem_mode`) were spot-checked and confirmed real
and correctly classified.

**Corrected — the one substantive error: §3 pipeline ordering.** The pre-verification
draft mis-ordered the middle of the pipeline. Corrected against
`default_build_dataflow_steps` (`build_dataflow_config.py:110–130`) to the canonical
sequence: …target_fps(7) → apply_folding(8) → **minimize_bit_width(9) →
transpose_decomposition(10) → generate_estimate_reports(11)** → hw_codegen(12) →
hw_ipgen(13) → set_fifo_depths(14) → create_stitched_ip(15) → measure_rtlsim(16) →
**synthesize_bitfile(17) → make_driver(18)** → deployment_package(19). `step_loop_rolling`
is MLO-gated and **not** in the default list (retained as #20, flagged). The §3.1 table,
the dependency-chain diagram, and the Tier-3/Tier-4 boundary were all realigned.

**Corrected — estimate-only vs full-build boundary.** `estimate_only_dataflow_steps`
(`build_dataflow_config.py:133–144`) = steps 1–9 + 11: it **includes**
`minimize_bit_width` (step 9) but **excludes** `transpose_decomposition` (step 10). The
draft had placed `minimize_bit_width` in the full-build tier; it is now correctly in the
estimate-only sub-contract (Tier 3), with the `minimize_weight_bit_width`/
`minimize_accumulator_width` mutators moved there.

**Added during the broad sweep.** `gen_top_module` (§2/§4b — an RTL ordering side-channel
set by `generate_hdl`, read by `get_rtl_file_list`/`code_generation_ipi`).
`get_integer_datatype` (folded into the eliminate-list): it is called by
streamline/convert passes on the **source** QONNX `IntQuant`/`BipolarQuant` ops
(`deps/qonnx/.../intquant.py:231`, `bipolar_quant.py:66`), read off the frontend node
being converted — never off a produced KernelOp, so it carries no obligation to the op
instance. `verify_node` (1 call, `analysis/verify_custom_nodes.py:46`) is **not** invoked
anywhere in the build flow — out of scope, no row added.

**Not exhaustively re-verified (out of aggregate-pass scope).** Per-unit call counts
labeled "≈" were spot-checked for direction only, not counted call-by-call; the exact
positional-tuple shape of `get_verilog_top_module_intf_names` and the codegen leaf-method
fan-out were taken from the unit census, not re-read from each op class.
