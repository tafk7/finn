# Census: vectorvectoractivation (VVAU)

*VVAU is the depthwise (vector-vector) MAC-plus-optional-threshold op with a clean agnostic base for the dataflow contract but heavy mem_mode-driven weight-streaming machinery; VVAU_hls wraps finn-hlslib Vector_Vector_Activate(_Stream)_Batch while VVAU_rtl is a Versal-DSP58-only template-fill of finn-rtllib/mvu shared with MVAU. The family's abstraction is undermined by a third (weight-memory) axis that leaks across all layers, base-class knowledge of subclass names, and a copy-paste rtlsim harness containing a real indentation bug.*

**Files:** `src/finn/custom_op/fpgadataflow/vectorvectoractivation.py`, `src/finn/custom_op/fpgadataflow/hls/vectorvectoractivation_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`, `finn-rtllib/mvu/mvu_vvu_axi_wrapper.v`, `finn-rtllib/mvu/mvu_vvu_axi.sv`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `VVAU` | `vectorvectoractivation.py` | `(HWCustomOp)` | yes |
| `VVAU_hls` | `vectorvectoractivation_hls.py` | `(VVAU, HLSBackend)` | yes |
| `VVAU_rtl` | `vectorvectoractivation_rtl.py` | `(VVAU, RTLBackend)` | yes |

## Redesign pressure

The current 2-axis (HLS/RTL) split leaks badly on the WEIGHT-MEMORY axis: mem_mode (internal_embedded / internal_decoupled / external) is an orthogonal third dimension that pervades shapes, widths, param generation, IPI stitching, and the base-class memstream allowlist — it is not modeled by the HLS/RTL abstraction at all, so every method re-branches on mem_mode and the agnostic base even branches on the concrete subclass name (op_type=='VVAU_rtl') to lay out weights. The RTL variant is effectively single-target (Versal DSP58 only; VERSION/PUMPED_COMPUTE/IS_MVU hard-pinned, LUT and DSP48 paths assert out) while the shared wrapper mvu_vvu_axi.sv is fully general, so the python side under-parameterizes what the RTL library already supports. Redesign pressure #1 is promoting mem_mode/weight-streaming to a first-class backend-neutral concern (killing the op_type allowlist leak and make_weight_file string-sniffing); #2 is a shared rtlsim/execute harness so HLS and RTL stop diverging (the RTL execute_node loop-indentation bug is a direct symptom of copy-paste). Otherwise the pure dataflow contract (shapes/widths/folding) is clean and genuinely backend-agnostic.

## Overrides (38)

| method | class | line | reason |
|---|---|---|---|
| `get_input_datatype` | VVAU | 182 | ind=0 -> inputDataType, ind=1 -> weightDataType; weights are a second stream input, not a scalar. |
| `get_output_datatype` | VVAU | 197 | trivial outputDataType lookup. |
| `get_instream_width` | VVAU | 201 | ind-dependent: ind0=i_bits*SIMD*PE, ind1=weight stream width (nonzero only for decoupled/external), ind2=threshold guarded by noActivation. |
| `get_outstream_width` | VVAU | 232 | o_bits*PE only (dot-product reduced, no SIMD in output). |
| `get_folded_input_shape` | VVAU | 237 | SIMD folds kernel dim (k_h*k_w % SIMD==0), PE folds channels; ind1 external weight shape special-cased. |
| `get_folded_output_shape` | VVAU | 260 | folds channels by PE into [1,dim_h,dim_w,nf,pe]. |
| `get_normal_input_shape` | VVAU | 268 | input is im2col-expanded [1,H,W,k_h*k_w*ch]. |
| `get_normal_output_shape` | VVAU | 275 | [1,H,W,ch]. |
| `get_nodeattr_types` | VVAU | 53 | adds PE/SIMD/Dim/Channels/Kernel/mem_mode/ram_style/binaryXnorMode/noActivation/resType/ActVal/accDataType/runtime_writeable_weights. |
| `execute_node` | VVAU | 119 | python golden model: PE reshape/interleave, sparse depthwise weight tensor, matmul + optional multithreshold. |
| `infer_node_datatype` | VVAU | 167 | warns on input dtype change, propagates output dtype. |
| `get_exp_cycles` | VVAU | 382 | (ch*k_h*k_w)/pe/simd * dim_h*dim_w depthwise cycle model. |
| `get_op_and_param_counts` | VVAU | 742 | mac/weight/threshold counts for depthwise conv, canonicalizes mac op-type by bitwidth ordering. |
| `generate_params` | VVAU | 669 | writes params.h/weights.npy/memblock.dat and thresh.h; mem_mode-branched. |
| `get_verilog_top_module_intf_names` | VVAU | 781 | appends in1_V weight AXIS for external, s_axilite for runtime-writeable decoupled. |
| `bram_estimation` | VVAU | 318 | weight-memory BRAM model with mem_mode/ram_style branching. |
| `uram_estimation` | VVAU | 299 | URAM model, only for internal_decoupled+ultra. |
| `code_generation_ipi` | VVAU | 793 | IPI builder that calls self.instantiate_ip() to defer to HLS/RTL; stitches memstream streamer for decoupled. |
| `derive_characteristic_fxns` | VVAU | 766 | adds in1 weight-stream stimulus for decoupled/external before delegating to base. |
| `execute_node` | VVAU_hls | 130 | cppsim/rtlsim harness: npy dump, weight stream replication num_w_reps, bipolar->binary remap. |
| `lut_estimation` | VVAU_hls | 52 | FINN-R paper LUT model (HLS-specific). |
| `dsp_estimation` | VVAU_hls | 116 | P*ceil((W+A)/48) only when resType=dsp. |
| `global_includes` | VVAU_hls | 291 | HLSBackend abstract: weights.hpp/activations.hpp/thresh.h. |
| `defines` | VVAU_hls | 303 | HLSBackend abstract: Channels1/InnerProdDim/SIMD1/PE1/numReps/WP1. |
| `docompute` | VVAU_hls | 380 | HLSBackend abstract: selects Vector_Vector_Activate_Batch vs _Stream_Batch by mem_mode. |
| `blackboxfunction` | VVAU_hls | 459 | HLSBackend abstract: 2-arg vs 3-arg signature by mem_mode. |
| `read_npy_data` | VVAU_hls | 324 | HLSBackend seam: adds weight stream npy2apintstream for decoupled/external. |
| `strm_decl` | VVAU_hls | 366 | HLSBackend seam: adds in1_V stream decl for decoupled/external. |
| `dataoutstrm` | VVAU_hls | 430 | HLSBackend seam: bipolar->binary output remap. |
| `pragmas` | VVAU_hls | 490 | HLSBackend seam: weight/threshold array_partition, in1_V axis interface. |
| `save_as_npy` | VVAU_hls | 456 | empties $SAVEASCNPY$ (no-op stub). |
| `code_generation_ipgen` | VVAU_hls | 239 | after base HLS ipgen, generates memstream HDL for decoupled (relies on base generate_hdl_memstream leak). |
| `minimize_weight_bit_width` | VVAU_hls | 520 | HLS-specific: widens threshold dt to >= accumulator dt to avoid comparison truncation. |
| `execute_node` | VVAU_rtl | 51 | cppsim delegates to VVAU.execute_node; rtlsim path duplicated from HLS. |
| `lut_estimation` | VVAU_rtl | 139 | returns 0 (RTL DSP core). |
| `dsp_estimation` | VVAU_rtl | 142 | P*ceil(SIMD/3) DSP58 3-MAC/DSP model. |
| `generate_hdl` | VVAU_rtl | 192 | RTLBackend abstract: template-fill mvu_vvu_axi_wrapper.v, memstream for decoupled. |
| `get_rtl_file_list` | VVAU_rtl | 290 | RTLBackend abstract: hard-coded list of 6 mvu/*.sv + generated wrapper. |

## Hacks (18 — 2 blocker, 8 major)

- **[blocker/brittle-assumption]** `vectorvectoractivation_rtl.py:89` — INDENTATION BUG: the entire rtlsim block (lines 89-130: get_rtlsim, reset, weight-stream build, rtlsim_multi_io, output writeback, context assignment) is indented 16 spaces INSIDE the `for inputs in node.input:` loop opened at line 62. Unlike VVAU_hls.execute_node where the rtlsim block sits outside the input loop, here rtlsim is re-run once per node input (2-3 times), and sim/export_idt leak from loop iterations. Copy-paste-and-reindent error that only works because the last iteration overwrites context.
- **[blocker/cross-backend-leak]** `vectorvectoractivation.py:617` — make_weight_file (on the backend-AGNOSTIC VVAU base) branches on self.onnx_node.op_type == 'VVAU_rtl' to choose weight layout (unflipped npy at 618, pe_simd_flipped .dat at 628). The base hard-codes knowledge of the RTL subclass string to pick RTL vs HLS memory ordering — inverted dependency defeating the agnostic base.
- **[major/base-class-leak]** `hwcustomop.py:310` — VVAU_hls.code_generation_ipgen (line 250) and VVAU_rtl.generate_hdl (line 229) both call self.generate_hdl_memstream, which in HWCustomOp gates on a hard-coded op_type allowlist ['MVAU_hls','MVAU_rtl','VVAU_hls','VVAU_rtl','Thresholding_hls']. VVAU depends on the base recognizing its own op-type strings; a renamed variant silently no-ops the streamer.
- **[major/duplicated-logic]** `vectorvectoractivation_rtl.py:89` — The rtlsim harness (npy dump, weight replication num_w_reps, bipolar remap, rtlsim_multi_io, output-to-npy) is copy-pasted almost verbatim from VVAU_hls.execute_node (hls lines 148-230). Two divergent copies (out_npy_path 'output.npy' here vs 'output_0.npy' in HLS) with no shared helper.
- **[major/magic-number]** `vectorvectoractivation_rtl.py:240` — _resolve_segment_len hard-codes DSP58 timing constants: 0.741 ns first-DSP delay and 0.605 ns per-subsequent-DSP delay to compute pipeline segment length; also asserts clk>0.741. Device/process-specific magic numbers baked into python.
- **[major/hard-coded-param]** `vectorvectoractivation_rtl.py:266` — _resolve_dsp_version returns hard-coded 3 (DSP58) and asserts resType!='lut' and is_versal(fpgapart). RTL VVU is Versal-only; no DSP48 path exists despite wrapper VERSION supporting 1/2/3.
- **[major/brittle-assumption]** `vectorvectoractivation_rtl.py:282` — $ACCU_WIDTH$ is set from get_output_datatype().bitwidth() (OUTPUT dt), not the accumulator dt (get_accumulator_datatype exists). For noActivation nodes output==acc so it works; with an activation this passes the wrong width to the RTL core. Semantically mislabeled.
- **[major/template-surgery]** `vectorvectoractivation_rtl.py:212` — generate_hdl does raw string .replace() of $KEY$ placeholders across mvu_vvu_axi_wrapper.v with no escaping/validation; correctness relies on exact token spelling and no substring collisions. Fragile template-fill.
- **[major/duplicated-logic]** `vectorvectoractivation_rtl.py:152` — The source-file list ['mvu_pkg.sv','mvu_vvu_axi.sv','replay_buffer.sv','mvu.sv','mvu_vvu_8sx9_dsp58.sv','add_multi.sv'] is hard-coded twice: instantiate_ip (line 152) and get_rtl_file_list (line 298). Two lists to keep in sync.
- **[major/brittle-assumption]** `vectorvectoractivation.py:126` — execute_node inspects producer node op_type ('Im2Col'/'ConvolutionInputGenerator') to decide pe=channels vs attr PE for input reordering. Golden model depends on graph neighbor identity — hidden coupling to sibling ops.
- **[minor/hard-coded-param]** `vectorvectoractivation.py:855` — code_generation_ipi hard-wires the weight streamer's ap_clk2x to the 1x clock with comment '2x clock is not used for decoupled VVAU weights, simply connect input to the 1x clock for now'. Explicit stopgap tying clock topology to an assumption.
- **[minor/hard-coded-param]** `vectorvectoractivation_rtl.py:187` — instantiate_ip unconditionally connects the node clk to the IP's ap_clk2x pin (RTL DSP58 VVU always double-pumps compute). Backend-specific clock wiring coupled to the wrapper's ap_clk2x port.
- **[minor/hard-coded-param]** `vectorvectoractivation_rtl.py:274` — $PUMPED_COMPUTE$ hard-coded to 0 and $IS_MVU$ to 0 in prepare_codegen_default; VVU RTL never exercises the pumped-compute path even though wrapper/mvu_vvu_axi.sv support it.
- **[minor/other]** `vectorvectoractivation.py:636` — make_weight_file comment 'add zeroes to pad out file to 1024 entries' but the code no longer pads (stale/dead comment); decoupled_verilog_dat just flattens. Misleading.
- **[minor/todo-marker]** `vectorvectoractivation.py:891` — '# TODO calculate and pass in segment size here' before assign_bd_address in the axilite path for runtime-writeable weights.
- **[minor/todo-marker]** `vectorvectoractivation_hls.py:271` — get_template_param_values: 'TODO check these with Giulio' and 'TODO handle non-bipolar binary inputs' — the TSrcI/TWeightI recast matrix is admittedly unverified.
- **[minor/todo-marker]** `vectorvectoractivation_hls.py:512` — pragmas: 'TODO find a better way of checking for no pregenerated thresholds' — uses calc_tmem()!=0 as a proxy.
- **[minor/duplicated-logic]** `vectorvectoractivation_hls.py:520` — VVAU_hls.minimize_weight_bit_width re-implements threshold-datatype minimization mirroring the base minimize logic plus an HLS-only widen-to-acc rule; not shared with RTL, so HLS and RTL diverge on threshold dt.

## Hermeticity violations (8)

- **[env-var]** `vectorvectoractivation.py:822` — code_generation_ipi reads os.environ['FINN_ROOT'] to locate finn-rtllib/axi/hdl and finn-rtllib/memstream/hdl for the streamer.
- **[filesystem-path]** `vectorvectoractivation.py:826` — code_generation_ipi does os.listdir(code_gen_dir) and picks the first file ending in '_memstream_wrapper.v' as strm_tmpl — order-dependent filesystem scan; multiple matches -> nondeterministic pick.
- **[sibling-op-coupling]** `vectorvectoractivation.py:126` — execute_node branches on producer node op_type (Im2Col/ConvolutionInputGenerator) pulled from the graph to set effective PE.
- **[hidden-coupling]** `vectorvectoractivation.py:617` — make_weight_file (agnostic base) branches on self.onnx_node.op_type=='VVAU_rtl' string to select weight byte-ordering — base coupled to concrete subclass identity.
- **[env-var]** `vectorvectoractivation_rtl.py:151` — instantiate_ip reads os.environ['FINN_ROOT'] to build finn-rtllib/mvu path for add_files.
- **[env-var]** `vectorvectoractivation_rtl.py:269` — prepare_codegen_default reads os.environ['FINN_ROOT'] for the wrapper template path.
- **[env-var]** `vectorvectoractivation_rtl.py:293` — get_rtl_file_list and get_verilog_paths (line 314) read os.environ['FINN_ROOT'] for the mvu source dir.
- **[order-dependence]** `vectorvectoractivation_rtl.py:207` — generate_hdl mutates the node via set_nodeattr('gen_top_module', ...); later instantiate_ip/get_rtl_file_list depend on this attr being written first — order-dependence between HDL gen and IP instantiation.

## finn-rtllib coupling (9)

- `mvu/mvu_vvu_axi_wrapper.v` via **verilog-template-fill** — rtl/vectorvectoractivation_rtl.py:210-220 reads the wrapper template (path built at :269) and str.replace()s every $KEY$ from prepare_codegen_default (:271-288): $IS_MVU$=0, $VERSION$=3, $PUMPED_COMPUTE$=0, $MW$=prod(Kernel), $MH$=Channels, $PE$, $SIMD$, $ACTIVATION_WIDTH$, $WEIGHT_WIDTH$, $ACCU_WIDTH$(=output dt bitwidth), $SIGNED_ACTIVATIONS$, $SEGMENTLEN$(=_resolve_segment_len), plus $NARROW_WEIGHTS$ (:202) and $MODULE_NAME_AXI_WRAPPER$ (:204). Output written to <gen_top_module>_wrapper.v at :216.
- `mvu/mvu_vvu_axi.sv` via **parameter-passing** — Instantiated by the filled wrapper; parameterized purely through wrapper params (IS_MVU=0 selects VVU input-interleave path at mvu_vvu_axi.sv:159; VERSION=3 selects DSP58). Added via instantiate_ip add_files at rtl/...py:152-165 and listed in get_rtl_file_list :298-308.
- `mvu/mvu_vvu_8sx9_dsp58.sv` via **file-copy** — DSP58 compute core; copied into the build via add_files (:157), selected implicitly by VERSION=3 / _resolve_dsp_version returning 3. No per-node parameterization beyond wrapper params.
- `mvu/mvu.sv` via **file-copy** — add_files at rtl/...py:155; unparameterized support module.
- `mvu/replay_buffer.sv` via **file-copy** — add_files at rtl/...py:154; activation replay buffer used by the VVU input path.
- `mvu/add_multi.sv` via **file-copy** — add_files at rtl/...py:158; adder support module, verbatim copy.
- `mvu/mvu_pkg.sv` via **file-copy** — add_files at rtl/...py:153; SystemVerilog package, verbatim copy.
- `memstream/hdl/memstream_wrapper_template.v` via **verilog-template-fill** — For mem_mode=internal_decoupled, VVAU_rtl.generate_hdl (:229) and VVAU_hls.code_generation_ipgen (:250) call HWCustomOp.generate_hdl_memstream (hwcustomop.py:307), which template-fills $MODULE_NAME$/$SETS$/$DEPTH$(=calc_wmem)/$WIDTH$(=get_instream_width_padded(1))/$INIT_FILE$(memblock.dat)/$RAM_STYLE$/$PUMPED_MEMORY$ — but only because VVAU_hls/VVAU_rtl appear in the hard-coded op_type allowlist at hwcustomop.py:310.
- `memstream/hdl/memstream_axi.sv (+ memstream.sv, axi/hdl/axilite.sv)` via **tcl-instantiate** — vectorvectoractivation.py code_generation_ipi:830-878 adds these files and instantiates the generated <name>_memstream_wrapper (auto-discovered by scanning code_gen_dir at :826) as a bd_cell, wiring m_axis_0->in1_V and clocks/reset; optional s_axilite when runtime_writeable.

## Seams (6)

- **clean-seam** — The 8 HWCustomOp contract methods (shapes/widths/datatypes) plus the execute_node golden model live entirely on the agnostic VVAU base and are backend-independent — a new backend can reuse them unchanged.
- **clean-seam** — instantiate_ip is a genuine polymorphic seam: VVAU.code_generation_ipi (base) calls self.instantiate_ip(cmd), dispatched to VVAU_hls.instantiate_ip (hls:575) or VVAU_rtl.instantiate_ip (rtl:147). Backend swap of the IP-instantiation step without touching the streamer-stitching base logic.
- **fused-no-seam** — make_weight_file (base, vectorvectoractivation.py:550-667) is fused to backend identity via op_type=='VVAU_rtl' branches (:617,:628). Weight byte-ordering (SIMD/PE flip) cannot be separated from the base without moving these branches into the backend classes.
- **fused-no-seam** — generate_hdl_memstream lives on HWCustomOp but is gated by a hard-coded op_type allowlist (hwcustomop.py:310). The decoupled-weight streamer for VVAU is fused across three layers (base allowlist + VVAU.code_generation_ipi tcl + backend ipgen call); no single clean swap point.
- **fused-no-seam** — VVAU_rtl.execute_node rtlsim path is fused/duplicated with VVAU_hls.execute_node rather than sharing a backend-neutral rtlsim harness; the two diverge (loop-indentation bug, output filename) and cannot be swapped without dedup.
- **fused-no-seam** — RTL DSP timing (_resolve_segment_len, _resolve_dsp_version) and clk2x wiring are DSP58/Versal-specific and hard-fused into VVAU_rtl; there is no abstraction for 'RTL on non-DSP58' — it just asserts out.
