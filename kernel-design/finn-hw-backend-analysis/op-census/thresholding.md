# Census: thresholding

*Thresholding is a two-input (activation + threshold tensor) elementwise-per-channel op with a clean agnostic base for shapes/datatypes but a badly overloaded backend story: an orthogonal memory-mode axis (embedded vs decoupled) leaks into the base via mem_mode try/except and an op_type-allowlisted generate_hdl_memstream call, and threshold serialization/width-minimization is triplicated with divergent, incompatible semantics across base/HLS/RTL. The HLS decoupled variant even emits and TCL-stitches finn-rtllib RTL, so the op does not cleanly separate along the HLS/RTL abstraction.*

**Files:** `src/finn/custom_op/fpgadataflow/thresholding.py`, `src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`, `finn-rtllib/thresholding/hdl/thresholding_template_wrapper.v`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Thresholding` | `thresholding.py` | `(HWCustomOp)` | yes |
| `Thresholding_hls` | `thresholding_hls.py` | `(Thresholding, HLSBackend)` | yes |
| `Thresholding_rtl` | `thresholding_rtl.py` | `(Thresholding, RTLBackend)` | yes |

## Redesign pressure

The dominant redesign pressure is that this op does NOT split along the HLS/RTL axis: memory-storage mode (embedded vs decoupled) is an orthogonal third axis the current abstraction has no place for, so it is smeared into mem_mode try/except checks inside the backend-agnostic base (get_instream_width, get_verilog_top_module_intf_names) and into an op_type-allowlisted base-class leak (generate_hdl_memstream) that makes the 'HLS' variant emit and stitch finn-rtllib RTL streamers. Threshold serialization (make_weight_file / minimize_weight_bit_width / generate_params) exists in three same-named but semantically incompatible copies with divergent mode vocabularies and datatype-widening rules, indicating the real seam is 'threshold storage/layout policy', not 'HLS vs RTL'. The RTL path additionally hard-couples to a binary-search hardware address layout (sorted-threshold assumption, per-PE/per-stage .dat bit-reordering, power-of-two channel padding) with no HLS counterpart. A redesign should promote memory-mode and threshold-packing to first-class strategy objects rather than mem_mode string branches and op_type-string base-class dispatch.

## Overrides (30)

| method | class | line | reason |
|---|---|---|---|
| `get_input_datatype` | Thresholding | 113 | Two-input op: ind==0 is activation input, ind==1 is the threshold ('weight') tensor; overloads the datatype accessor by index. |
| `get_output_datatype` | Thresholding | 123 | Standard read of outputDataType attr. |
| `get_instream_width` | Thresholding | 173 | ind==0 is PE*input_bits; ind==1 is the threshold stream width = PE*wp*numSteps, but ONLY when mem_mode==internal_decoupled. |
| `get_outstream_width` | Thresholding | 194 | out_bits*PE. |
| `get_folded_input_shape` | Thresholding | 198 | Fold = TMEM = NumChannels//PE; shape = vecs + [fold, PE]. |
| `get_folded_output_shape` | Thresholding | 205 | Same shape as input (thresholding is elementwise per channel). |
| `get_normal_input_shape` | Thresholding | 209 | vecs + [NumChannels]. |
| `get_normal_output_shape` | Thresholding | 215 | Same as input. |
| `get_exp_cycles` | Thresholding | 219 | prod of folded output minus PE dim (one element/cycle). |
| `get_nodeattr_types` | Thresholding | 47 | Adds PE/NumChannels/numSteps/inputDataType/weightDataType/outputDataType/numInputVectors/ActVal/runtime_writeable_weights. |
| `verify_node` | Thresholding | 89 | Checks backend + required threshold attrs. |
| `execute_node` | Thresholding | 265 | Python golden model via qonnx multithreshold, with 4D NHWC->NCHW transpose and BIPOLAR fixup. |
| `get_verilog_top_module_intf_names` | Thresholding | 292 | Declares clk/rst/in0/out0; conditionally adds in1_V weight stream (MLO) or s_axilite (decoupled+runtime_writeable). |
| `get_nodeattr_types` | Thresholding_hls | 59 | Adds mem_mode {internal_embedded, internal_decoupled}, ram_style, runtime_writeable_weights. |
| `bram_estimation` | Thresholding_hls | 93 | Resource estimates from _threshold_mem_width and ram_style. |
| `get_ap_int_max_w` | Thresholding_hls | 150 | Widen to include the decoupled weight stream width. |
| `generate_params` | Thresholding_hls | 299 | Emits thresh.h (embedded) OR thresholds.npy + memblock.dat (decoupled). |
| `execute_node` | Thresholding_hls | 329 | cppsim/rtlsim npy packing; feeds weight stream in1 with num_w_reps replication for decoupled. |
| `docompute` | Thresholding_hls | 517 | HLSBackend contract; branches on mem_mode (embedded uses Thresholding_Batch, decoupled uses Thresholding_Stream_Batch hlslib template). |
| `code_generation_ipi` | Thresholding_hls | 640 | For decoupled: builds a BD hierarchy stitching a memstream streamer to the HLS IP; embedded falls back to super(). |
| `get_op_and_param_counts` | Thresholding_hls | 750 | Reports threshold param counts as param_threshold_Nb. |
| `minimize_weight_bit_width` | Thresholding_hls | 781 | HLS comparator truncates input to threshold width, so widen tdt to >= input width. |
| `get_nodeattr_types` | Thresholding_rtl | 52 | Adds depth_trigger_uram/bram, uniform_thres (no-op, est only), deep_pipeline. |
| `bram_estimation` | Thresholding_rtl | 139 | Estimates via get_pe_mem_geometries + mem_primitives_versal alternatives. |
| `generate_params` | Thresholding_rtl | 452 | Asserts thresholds sorted (binary search requirement); writes memblock.dat + per-PE/per-stage .dat files. |
| `execute_node` | Thresholding_rtl | 347 | cppsim delegates to Thresholding.execute_node; rtlsim packs single input stream (no weight stream). |
| `get_verilog_top_module_intf_names` | Thresholding_rtl | 445 | Adds s_axilite when runtime_writeable_weights. |
| `generate_hdl` | Thresholding_rtl | 309 | RTLBackend contract: template-fill wrapper .v, copy thresholding.sv/thresholding_axi.sv/axilite.sv. |
| `minimize_weight_bit_width` | Thresholding_rtl | 566 | RTL saturates input to threshold range; tdt must represent [min-1 : max]. |
| `minimize_weight_bit_width` | Thresholding | 127 | Base threshold-width minimization with signed/all-zero special cases. |

## Hacks (17 — 1 blocker, 11 major)

- **[blocker/base-class-leak]** `hwcustomop.py:310` — HWCustomOp.generate_hdl_memstream hard-codes op_type allowlist ['MVAU_hls','MVAU_rtl','VVAU_hls','VVAU_rtl','Thresholding_hls'] and special-cases op_type.startswith('Thresholding') to call calc_tmem() vs calc_wmem(). Thresholding_hls.code_generation_ipgen (thresholding_hls.py:181) relies on this leak to emit the memstream wrapper .v.
- **[major/cross-backend-leak]** `thresholding.py:179` — get_instream_width in the backend-AGNOSTIC base sniffs mem_mode via try/except AttributeError, defaulting to 0, because mem_mode is an HLS-only attribute absent on RTL. HLS memory-mode knowledge leaks into the shared base.
- **[major/cross-backend-leak]** `thresholding.py:310` — get_verilog_top_module_intf_names in the agnostic base again uses try/except on mem_mode, plus branches on mlo_max_iter to inject an in1_V weight stream. Interface topology (decoupled streamer vs embedded) is decided in the base rather than the backend.
- **[major/template-surgery]** `thresholding_rtl.py:323` — generate_hdl reads thresholding_template_wrapper.v and does raw string .replace() for each $KEY$ (16 placeholders: $WI$ $WT$ $N$ $C$ $PE$ $BIAS$ $SIGNED$ $FPARG$ $O_BITS$ $SETS$ $USE_AXILITE$ $THRESHOLDS_PATH$ $DEPTH_TRIGGER_URAM$ $DEPTH_TRIGGER_BRAM$ $DEEP_PIPELINE$ $MODULE_NAME_AXI_WRAPPER$). Fragile un-typed placeholder substitution.
- **[major/magic-number]** `thresholding_rtl.py:515` — make_weight_file decoupled_runtime packing loops over 'range(2 ** (pe - 1).bit_length())' with guard '(c == 0 or c % pe != 0) and c < pe' to pad memory to a power-of-two channel stride. Opaque bit-twiddling with no explanation; brittle for non-power-of-two PE.
- **[major/magic-number]** `thresholding_rtl.py:559` — Per-stage threshold .dat generation indexes t_packed[ch*pe+pe_value][(i << (o_bitwidth - stage)) + 2**sn - 1] to reorder thresholds for the hardware binary-search tree. Deeply coupled to the RTL address layout; no abstraction boundary.
- **[major/brittle-assumption]** `thresholding_rtl.py:457` — generate_params asserts thresholds are pre-sorted ascending (np.diff>=0) because the RTL uses binary search. Silent correctness dependency on an upstream transform; HLS variant has no such requirement.
- **[major/brittle-assumption]** `thresholding_rtl.py:266` — $FPARG$ float check compares get_input_datatype(0) against the string list ['FLOAT32','FLOAT16'] — a DataType object is compared to strings, so this branch is effectively always False (dead code / latent bug).
- **[major/duplicated-logic]** `thresholding_rtl.py:566` — minimize_weight_bit_width duplicates the signed/get_smallest_possible datatype-derivation branch from Thresholding.minimize_weight_bit_width (thresholding.py:151-161) with a -1 tweak. Same logic copied into HLS variant (thresholding_hls.py:781) differently. Three divergent copies of threshold-width logic.
- **[major/hard-coded-param]** `thresholding_rtl.py:468` — make_weight_file is called with weight_file_mode='internal_embedded' string, but the RTL make_weight_file only accepts 'decoupled_runtime'/'internal_embedded' — different mode vocabulary from HLS make_weight_file ('hls_header','decoupled_npy','decoupled_verilog_dat','decoupled_runtime'). Same method name, incompatible semantics across siblings.
- **[major/other]** `thresholding_hls.py:675` — code_generation_ipi scans code_gen_dir with os.listdir for a file ending in '_memstream_wrapper.v' (produced earlier by generate_hdl_memstream) and uses it as a BD reference (hidden order-dependent filesystem coupling). 'strm_tmpl' is undefined if no match, causing a NameError.
- **[major/inheritance-irregularity]** `thresholding.py:133` — Base Thresholding.minimize_weight_bit_width and get_verilog_top_module_intf_names reference get_nodeattr('mlo_max_iter'), an attribute defined on HWCustomOp base for MLO/multi-layer-offload — the agnostic threshold class is entangled with MLO feature flags it does not otherwise implement.
- **[minor/duplicated-logic]** `thresholding_rtl.py:347` — RTL execute_node rtlsim branch re-implements the input-npy packing already present in Thresholding_hls.execute_node (BIPOLAR fixup, reshape, npy_to_rtlsim_input). Large copy-paste between the two backends.
- **[minor/todo-marker]** `thresholding_hls.py:439` — '# TODO check and add whatever missing' above defines(); plus '# TODO add flips/reversals as needed here' (line 245), '# TODO add in/out FIFO contributions' (line 133), '# TODO calculate and pass in segment size here' (line 740).
- **[minor/todo-marker]** `thresholding.py:99` — verify_node contains '# TODO collect automatically from get_nodeattr_types' and hard-codes the required attribute list, referencing 'Threshold_Batch' (legacy hlslib name) in the error string (line 109).
- **[minor/magic-number]** `thresholding_hls.py:277` — decoupled_runtime packing computes words_per_memwidth = 2**ceil(log2(weight_width/32)) then pads to 32-bit words with textwrap.wrap(val,8) + reverse() — hard-coded 32-bit AXI-lite word assumption mixed into weight-file surgery.
- **[minor/brittle-assumption]** `thresholding_hls.py:704` — IPI comment '2x clock is not used for decoupled thresholds / simply connect input to the 1x clock for now' — ap_clk2x wired to ap_clk as a placeholder.

## Hermeticity violations (6)

- **[env-var]** `thresholding_hls.py:671` — code_generation_ipi reads os.environ['FINN_ROOT'] to locate finn-rtllib/axi/hdl and finn-rtllib/memstream/hdl source files.
- **[filesystem-path]** `thresholding_hls.py:675` — os.listdir(code_gen_dir) to discover the *_memstream_wrapper.v produced by a prior generate_hdl_memstream call — order-dependent, silently uses last match, 'strm_tmpl' undefined if none.
- **[env-var]** `thresholding_rtl.py:294` — get_rtl_file_list reads os.environ['FINN_ROOT'] for finn-rtllib/thresholding/hdl and finn-rtllib/axi/hdl.
- **[env-var]** `thresholding_rtl.py:320` — generate_hdl reads os.environ['FINN_ROOT'] for template + sv source copy via shutil.copy into code_gen_dir_ipgen.
- **[hidden-coupling]** `thresholding_hls.py:168` — code_generation_ipgen calls get_vivado_version() and is_versal(fpgapart) to gate URAM support — build-tool/version ambient state affects codegen validity.
- **[sibling-op-coupling]** `thresholding_hls.py:181` — code_generation_ipgen calls self.generate_hdl_memstream(fpgapart) which lives on HWCustomOp base and only fires for a hard-coded op_type allowlist — cross-op shared machinery gated by op_type string.

## finn-rtllib coupling (7)

- `thresholding/hdl/thresholding_template_wrapper.v` via **verilog-template-fill** — rtl/thresholding_rtl.py:322-333 reads thresholding_template_wrapper.v and str.replace()s 16 $KEY$ placeholders from prepare_codegen_rtl_values (thresholding_rtl.py:188-288), then writes <gen_top_module>.v into code_gen_dir_ipgen.
- `thresholding/hdl/thresholding.sv` via **file-copy** — rtl/thresholding_rtl.py:335-337 shutil.copy of thresholding.sv (binary-search core) into code_gen_dir; also listed in get_rtl_file_list (line 303).
- `thresholding/hdl/thresholding_axi.sv` via **file-copy** — rtl/thresholding_rtl.py:335-337 shutil.copy of thresholding_axi.sv into code_gen_dir; listed in get_rtl_file_list (line 304).
- `axi/hdl/axilite.sv` via **file-copy** — rtl/thresholding_rtl.py:338 shutil.copy of axilite.sv (for runtime_writeable AXI-lite); listed in get_rtl_file_list (line 302).
- `thresholding meminit <node>_threshs_<pe>_<stage>.dat` via **parameter-passing** — make_weight_file (thresholding_rtl.py:532-564) writes per-PE per-stage .dat files referenced by $THRESHOLDS_PATH$ (thresholding_rtl.py:234); the .sv $readmemh's them at elaboration.
- `memstream/hdl/memstream_wrapper_template.v` via **verilog-template-fill** — HLS decoupled path: Thresholding_hls.code_generation_ipgen (thresholding_hls.py:181) triggers HWCustomOp.generate_hdl_memstream (hwcustomop.py:307), which template-fills memstream_wrapper_template.v with $DEPTH$=calc_tmem, $WIDTH$=padded weight stream, $INIT_FILE$=memblock.dat — the HLS RTL streamer is emitted via the base-class op_type-allowlisted leak, not by an RTL variant.
- `memstream/hdl/memstream_axi.sv + memstream.sv, axi/hdl/axilite.sv` via **tcl-instantiate** — Thresholding_hls.code_generation_ipi (thresholding_hls.py:679-742) add_files-copies these and create_bd_cell-instantiates the streamer hierarchy, connecting its m_axis_0 to the HLS IP in1_V. The HLS variant emits/wires RTL sources — cross-backend.

## Seams (5)

- **clean-seam** — The 8 shape/width/datatype contract methods (get_normal_*/get_folded_*/get_outstream_width) live entirely in the agnostic Thresholding base and are backend-independent — a new backend can reuse them unchanged.
- **clean-seam** — The Python golden model Thresholding.execute_node (thresholding.py:265) is fully backend-agnostic and is reused by the RTL variant for cppsim (thresholding_rtl.py:351).
- **fused-no-seam** — get_instream_width (thresholding.py:173) and get_verilog_top_module_intf_names (thresholding.py:292) in the agnostic base branch on mem_mode (HLS-only) and mlo_max_iter — the interface/width contract is fused with HLS decoupled-memory semantics and cannot be cleanly separated from the base.
- **fused-no-seam** — make_weight_file and minimize_weight_bit_width exist in all three classes with the SAME name but incompatible mode vocabularies and divergent width logic; there is no shared threshold-serialization interface — base and backend weight formats are fused per-backend, blocking a clean swap.
- **fused-no-seam** — HLS decoupled mode's RTL memstream generation (code_generation_ipgen -> generate_hdl_memstream base leak, code_generation_ipi TCL streamer stitching) means the 'HLS' variant actually emits and wires finn-rtllib RTL — the HLS/RTL axis does not cleanly partition this op.
