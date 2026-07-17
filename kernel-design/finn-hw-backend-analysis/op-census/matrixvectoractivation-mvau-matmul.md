# Census: matrixvectoractivation (MVAU / matmul)

*MVAU is the canonical FINN matmul op: a clean agnostic MVAU base (shapes, datatypes, folding, golden matmul+multithreshold, weight packing) with MVAU_hls (hlslib Matrix_Vector_Activate template calls) and MVAU_rtl (DSP58 mvu_vvu_axi wrapper template-fill) backends. Its complexity is not in the compute but in an orthogonal weight-delivery dimension (embedded/decoupled-memstream/external/dynamic-load/MLO-fetch) whose RTL is emitted from both backends and stitched by a huge IPI/TCL method that lives in the agnostic base and violates the HLS/RTL two-axis model.*

**Files:** `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`, `src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`, `finn-rtllib/mvu/mvu_vvu_axi_wrapper.v`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `MVAU` | `matrixvectoractivation.py` | `(HWCustomOp)` | yes |
| `MVAU_hls` | `matrixvectoractivation_hls.py` | `(MVAU, HLSBackend)` | yes |
| `MVAU_rtl` | `matrixvectoractivation_rtl.py` | `(MVAU, RTLBackend)` | yes |

## Redesign pressure

The MVAU family breaks the 2-axis (HLS/RTL) abstraction along a THIRD, orthogonal axis: weight delivery (internal_embedded vs internal_decoupled/memstream vs external vs dynamic_input/dynload vs mlo_max_iter/fetch_weights). This weight-delivery axis is emitted from both HLS and RTL backends and stitched by a ~236-line code_generation_ipi that lives in the supposedly backend-agnostic base yet calls the backend-only instantiate_ip and reaches directly into eight finn-rtllib subtrees via FINN_ROOT. The agnostic base is thus not agnostic - it hard-depends on Vivado TCL, RTL sources, and subclass methods, and the base HWCustomOp itself leaks MVAU op-type strings (generate_hdl_memstream/fetch_weights) and an MLO feature-flag threaded through ~12 MVAU methods. A redesign needs to (1) make weight-delivery a first-class pluggable component independent of compute backend, (2) pull IPI/TCL emission out of the agnostic op into a backend-owned stitching layer, and (3) eliminate the op_type-string and pumpedCompute/mlo attribute leakage between base and subclasses. The pure compute contract (shapes/datatypes/golden matmul) is genuinely clean and should stay in the agnostic op; almost all of the mess is in the code-gen/stitching and weight-delivery plumbing.

## Overrides (40)

| method | class | line | reason |
|---|---|---|---|
| `get_input_datatype` | MVAU | 237 | ind=1 returns weightDataType (weights are input[1]); handles ext-weight FIFO insertion where ind>0 |
| `get_output_datatype` | MVAU | 252 | returns outputDataType attr |
| `get_instream_width` | MVAU | 256 | ind-dependent: ind0=SIMD*ibits, ind1=weight stream width branching on dynamic_input/mem_mode/mlo_max_iter, ind2=thresholds always embedded (0) |
| `get_outstream_width` | MVAU | 290 | PE * output bitwidth (PE-folded output) |
| `get_folded_input_shape` | MVAU | 295 | SIMD-folded activation (sf,simd); weight input shape differs by dynamic_input/external/mlo |
| `get_folded_output_shape` | MVAU | 319 | PE folding: (vecs, nf, pe) |
| `get_normal_input_shape` | MVAU | 327 | ind1 weight tensor shape (mw,mh) |
| `get_normal_output_shape` | MVAU | 339 | vecs + [mh] |
| `get_nodeattr_types` | MVAU | 59 | adds PE/SIMD/MW/MH/mem_mode/ram_style/runtime_writeable_weights/pumpedMemory/dynamic_input etc. |
| `execute_node` | MVAU | 132 | python golden: matmul with xnorpopcount for bipolar/binary, then multithreshold with NHWC<->NCHW transpose |
| `verify_node` | MVAU | 167 | checks input count depends on noActivation (2 vs 3 inputs) |
| `generate_params` | MVAU | 762 | writes params.h (embedded), input_1.npy + memblock.dat (decoupled) and thresh.h ThresholdsActivation<...> C++ |
| `get_op_and_param_counts` | MVAU | 840 | MAC and weight/threshold param counts, canonicalizes mac_NbxMb op type |
| `get_exp_cycles` | MVAU | 458 | (mh/pe)*(mw/simd)*prod(vecs); mmv hardcoded 1 |
| `bram_estimation` | MVAU | 387 | FINN-R paper RAMB18 model, gated by mem_mode/ram_style/mlo |
| `uram_estimation` | MVAU | 365 | URAM 72x4096 model, only for ram_style=ultra decoupled |
| `get_verilog_top_module_intf_names` | MVAU | 879 | drops super() (commented out line 880); builds intf dict adding weight stream / axilite / aximm(mlo) / clk2x conditionally |
| `code_generation_ipi` | MVAU | 920 | backend TCL block-design surgery living in the agnostic base; wires streamer/dynload/fetch_weights hierarchy; calls self.instantiate_ip which only exists on backend subclasses |
| `derive_characteristic_fxns` | MVAU | 864 | injects weight input stream (in1) for decoupled/external into rtlsim override dict |
| `get_nodeattr_types` | MVAU_hls | 53 | merges MVAU+HLSBackend attrs and overrides resType default to 'lut' |
| `lut_estimation` | MVAU_hls | 61 | FINN-R paper LUT model (mult/adder/acc/threshold LUTs) |
| `dsp_estimation` | MVAU_hls | 124 | P*Q*ceil((W+A)/48) only if resType=dsp |
| `global_includes` | MVAU_hls | 196 | HLS contract: includes weights/activations/mvau.hpp + thresh.h if tmem!=0 |
| `defines` | MVAU_hls | 211 | HLS contract: MW1/MH1/SIMD1/PE1/WMEM1/TMEM1/numReps macros; enforces SIMD>=MW/1024 |
| `docompute` | MVAU_hls | 317 | HLS contract: emits Matrix_Vector_Activate_Batch (embedded) or _Stream_Batch (decoupled) hlslib call |
| `blackboxfunction` | MVAU_hls | 399 | HLS contract: top signature, 2 vs 3 streams by mem_mode |
| `read_npy_data` | MVAU_hls | 247 | npy2apintstream for in0 and (decoupled) in1 with numReps |
| `strm_decl` | MVAU_hls | 295 | declares in0/out0 and conditional in1 weight stream |
| `dataoutstrm` | MVAU_hls | 370 | apintstream2npy, bipolar->binary storage |
| `save_as_npy` | MVAU_hls | 396 | no-op override (empty SAVEASCNPY) |
| `pragmas` | MVAU_hls | 438 | AXIS interface pragmas + ARRAY_PARTITION for weights/thresholds + threshold RAM style RESOURCE pragmas |
| `get_ap_int_max_w` | MVAU_hls | 491 | widen to include weight stream and single-PE weight entry width |
| `execute_node` | MVAU_hls | 504 | cppsim/rtlsim: saves npy, feeds weight stream (in1) num_w_reps times, bipolar<->binary reinterpretation |
| `code_generation_ipgen` | MVAU_hls | 139 | after HLS ipgen, ALSO emits RTL weight-streamer HDL (memstream/dynload/fetch_weights) - cross-backend |
| `generate_params` | MVAU_hls | 626 | actually overrides minimize_weight_bit_width (not generate_params) to widen threshold dt >= acc dt (HLS truncation fix) |
| `execute_node` | MVAU_rtl | 59 | cppsim delegates to MVAU.execute_node (numpy); rtlsim duplicates HLS rtlsim path |
| `lut_estimation` | MVAU_rtl | 149 | returns 0 (RTL MVU is DSP-based) |
| `dsp_estimation` | MVAU_rtl | 152 | DSP58: P*ceil(Q/3) else ceil(P/4)*Q |
| `generate_hdl` | MVAU_rtl | 271 | RTL contract: generate_params + template-fill mvu_vvu_axi_wrapper.v + conditional weight-streamer HDL |
| `get_rtl_file_list` | MVAU_rtl | 357 | RTL contract: lists mvu/*.sv sources + generated wrapper |

## Hacks (18 — 2 blocker, 8 major)

- **[blocker/base-class-leak]** `matrixvectoractivation.py:920` — code_generation_ipi (~236 lines) lives in the backend-AGNOSTIC MVAU base but is entirely Vivado IPI TCL block-design surgery, and it calls self.instantiate_ip() which is defined ONLY on MVAU_hls (line 681) and MVAU_rtl (line 163). The 'agnostic' class is therefore uninstantiable-by-contract and structurally depends on its subclasses.
- **[blocker/cross-backend-leak]** `matrixvectoractivation_hls.py:144` — code_generation_ipgen of the HLS variant calls generate_hdl_dynload/generate_hdl_memstream/generate_hdl_fetch_weights (lines 145,153,155) - i.e. the HLS op emits finn-rtllib Verilog for the weight streamer. HLS compute + RTL weight delivery are fused into one op.
- **[major/base-class-leak]** `hwcustomop.py:310` — generate_hdl_memstream branches on hard-coded op_type string list ["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"] to decide behaviour; MVAU relies on this leak. Same pattern in generate_hdl_fetch_weights (line 358-359, ops=["MVAU_hls","MVAU_rtl"]).
- **[major/duplicated-logic]** `matrixvectoractivation_rtl.py:100` — The rtlsim branch of MVAU_rtl.execute_node (lines 100-140) is a near-verbatim copy of MVAU_hls.execute_node rtlsim branch (hls line 569-617): same npy_to_rtlsim_input, weight-rep feeding, rtlsim_output_to_npy, reshape. Two copies drift independently (rtl writes output.npy, hls writes output_0.npy).
- **[major/duplicated-logic]** `matrixvectoractivation_rtl.py:357` — get_rtl_file_list (357-377) duplicates the exact sourcefile list ['mvu_pkg.sv','mvu_vvu_axi.sv','replay_buffer.sv','mvu.sv','mvu_vvu_8sx9_dsp58.sv','add_multi.sv'] already hard-coded in instantiate_ip (168-175). Two hand-maintained copies of the RTL manifest.
- **[major/cross-backend-leak]** `matrixvectoractivation.py:892` — Agnostic base get_verilog_top_module_intf_names reads pumpedCompute via try/except AttributeError (892-895) and code_generation_ipi does the same (939-942); pumpedCompute is an RTL-ONLY nodeattr (defined only in MVAU_rtl line 53). The base silently defaults to 0 for HLS. RTL-specific attribute leaking into agnostic code.
- **[major/brittle-assumption]** `matrixvectoractivation_rtl.py:349` — $ACCU_WIDTH$ template param is filled from get_output_datatype().bitwidth() NOT accDataType. Assumes the RTL MVU's accumulator width equals the output datatype width (true only because RTL-MVU has no separate activation stage); silently wrong if outputDataType != accDataType.
- **[major/magic-number]** `matrixvectoractivation_rtl.py:233` — _resolve_segment_len hard-codes DSP timing constants 0.741 ns (first DSP) and 0.605 ns (subsequent) and simd_factor 3 (6 when pumped) to compute pipeline chain length for target clk. Device-specific magic tied to DSP58.
- **[major/hard-coded-param]** `hwcustomop.py:379` — generate_hdl_fetch_weights hard-codes n_max_layers = 64 ('upper bound on how many layers can be supported, set to 64 for now') feeding the MLO fetch_weights wrapper $N_LAYERS$.
- **[major/brittle-assumption]** `matrixvectoractivation_rtl.py:281` — narrow_weights determined by np.min(weights)==wdt.min(); relies on actual initializer values (skipped for dynamic_input/mlo). A weight tensor that happens not to reach the datatype minimum flips a hardware-affecting flag.
- **[minor/template-surgery]** `matrixvectoractivation_rtl.py:296` — generate_hdl does raw string .replace() of $KEY$ placeholders across mvu_vvu_axi_wrapper.v; no escaping/validation, order-independent global replace. Same pattern for memstream/fetch_weights wrappers in the base (hwcustomop 343, 395).
- **[minor/magic-number]** `matrixvectoractivation_hls.py:216` — defines() asserts SIMD >= MW/1024 as an HLS-synth-only constraint with no symbolic origin (Vivado HLS array partition limit).
- **[minor/magic-number]** `matrixvectoractivation_hls.py:79` — lut_estimation uses paper-fit constants c0=300, c1=1.1, c2 computed with /64 and /6 magic divisors; bram_estimation similarly (16384/8192/4096/2048/1024/512, mem_width thresholds 1/2/4/9/18/36) lines 416-427.
- **[minor/todo-marker]** `matrixvectoractivation_hls.py:176` — get_template_param_values: 'TODO check these with Giulio' and 'TODO handle non-bipolar binary inputs' (177); True binary non-bipolar raises NotImplemented (168).
- **[minor/brittle-assumption]** `matrixvectoractivation.py:712` — make_weight_file decoupled_verilog_dat: stale comment 'add zeroes to pad out file to 1024 entries' but code no longer pads to 1024; pumpedMemory path (715-729) manually splits each hex word in half and interleaves, raising if pe==simd==1 ('known bug, ask user to increase parallelism').
- **[minor/hard-coded-param]** `matrixvectoractivation_hls.py:206` — global_includes hard-codes '#include "mvau.hpp"'; docompute hard-codes hlslib function names Matrix_Vector_Activate_Batch / _Stream_Batch (lines 332,353).
- **[minor/brittle-assumption]** `matrixvectoractivation_rtl.py:202` — instantiate_ip always connects an ap_clk2x pin: when NOT pumped it connects the wrapper's ap_clk2x input to the regular ap_clk (202-205,225-228). The wrapper (template line 60) always exposes ap_clk2x even when unused - a dead 2x-clock port always present.
- **[minor/todo-marker]** `matrixvectoractivation.py:1126` — code_generation_ipi: 'TODO calculate and pass in segment size here' before blind assign_bd_address for runtime-writeable axilite.

## Hermeticity violations (7)

- **[env-var]** `matrixvectoractivation_rtl.py:167` — instantiate_ip reads os.environ['FINN_ROOT'] to locate finn-rtllib/mvu/; also get_rtl_file_list (360), prepare_codegen_default (329), get_verilog_paths (381).
- **[env-var]** `matrixvectoractivation.py:979` — code_generation_ipi builds absolute finn-rtllib paths from os.environ['FINN_ROOT'] for ram/, dynload/, mlo/, skid/, dwc/, cdma/, axi/, memstream/ (lines 979-1002, 1042-1043).
- **[filesystem-path]** `matrixvectoractivation.py:983` — os.listdir(code_gen_dir) scanned for a file ending in '_dynamic_load_wrapper.v' / '_fetch_weights_wrapper.v' / '_memstream_wrapper.v' (983,1005,1046) and the LAST match wins (no break) - order-dependent, silently picks arbitrary file if multiple match.
- **[filesystem-path]** `matrixvectoractivation.py:1020` — MLO path os.listdir over finn-rtllib/cdma/, cdma_a/, cdma_u/, cdma_x/ globbing *.sv/*.svh to build sourcefiles - directory-content-dependent build.
- **[sibling-op-coupling]** `matrixvectoractivation_rtl.py:66` — MVAU_rtl.execute_node cppsim calls MVAU.execute_node(self, ...) directly (explicit unbound base call) rather than super(), coupling to base MRO.
- **[hidden-coupling]** `hwcustomop.py:100` — Whole MLO feature keyed off base nodeattr mlo_max_iter (defined in HWCustomOp) but threaded through ~12 MVAU sites (instream width 268, folded shape 311, uram 380, bram 410, min-acc 485, min-wbit 534, gen_params 784, intf_names 900, ipi 926/958/996); MVAU semantics silently change based on a base-level attr.
- **[hidden-coupling]** `matrixvectoractivation_rtl.py:66` — generate_hdl (271) calls self.generate_params (base MVAU method that emits HLS params.h/thresh.h and .dat) from the RTL variant - the RTL op reuses HLS-oriented file generation.

## finn-rtllib coupling (10)

- `mvu/mvu_vvu_axi_wrapper.v` via **verilog-template-fill** — prepare_codegen_default (rtl line 328-355) builds code_gen_dict of $IS_MVU$,$VERSION$,$PUMPED_COMPUTE$,$MW$,$MH$,$PE$,$SIMD$,$ACTIVATION_WIDTH$,$WEIGHT_WIDTH$,$ACCU_WIDTH$,$SIGNED_ACTIVATIONS$,$SEGMENTLEN$; generate_hdl adds $NARROW_WEIGHTS$ (288) and $MODULE_NAME_AXI_WRAPPER$ (290); injected via string .replace loop at rtl line 298-301 into the wrapper (placeholders at wrapper lines 35-47).
- `mvu/mvu_pkg.sv` via **file-copy** — instantiate_ip (rtl line 168-181) add_files -norecurse; also listed in get_rtl_file_list (365).
- `mvu/mvu_vvu_axi.sv` via **file-copy** — instantiated by the generated wrapper (wrapper line 78-98, mvu_vvu_axi inst); added via add_files at rtl line 168-181.
- `mvu/replay_buffer.sv` via **file-copy** — add_files in instantiate_ip (168-181) / get_rtl_file_list (365).
- `mvu/mvu.sv` via **file-copy** — add_files in instantiate_ip (168-181).
- `mvu/mvu_vvu_8sx9_dsp58.sv` via **file-copy** — DSP58 compute core; add_files (168-181). $VERSION$ (1=DSP48E1,2=DSP48E2,3=DSP58) from _resolve_dsp_version selects the compute path inside RTL at elaboration.
- `mvu/add_multi.sv` via **file-copy** — add_files (168-181).
- `memstream/hdl/memstream_wrapper_template.v` via **verilog-template-fill** — base generate_hdl_memstream (hwcustomop 307-351) fills $MODULE_NAME$,$SETS$,$DEPTH$,$WIDTH$,$INIT_FILE$,$RAM_STYLE$,$PUMPED_MEMORY$ then emits <name>_memstream_wrapper.v; sources memstream_axi.sv/memstream.sv/axilite.sv added in MVAU.code_generation_ipi (1050-1055). Emitted by BOTH hls (code_generation_ipgen 153) and rtl (generate_hdl 320).
- `mlo/fetch_weights_wrapper.v` via **verilog-template-fill** — base generate_hdl_fetch_weights (hwcustomop 355-405) fills $MW$/$MH$/$PE$/$SIMD$/$N_REPS$/$WEIGHT_WIDTH$/$LAYER_OFFS$/$N_LAYERS$; MLO path in code_generation_ipi (996-1035) also pulls mlo/fetch_weights.sv, mlo/local_weight_buffer.sv, skid/skid.sv, ram/ram_p_c.sv, dwc/hdl/*, and globs cdma/ dirs.
- `dynload/hdl/dynamic_load_wrapper_template.v` via **verilog-template-fill** — base generate_hdl_dynload (hwcustomop 407+) fills the dynamic-load wrapper for dynamic_input weights; sources dynload/hdl/dynamic_load.sv + ram/ram_p_c.sv wired in code_generation_ipi (979-991).

## Seams (5)

- **clean-seam** — The 8 HWCustomOp shape/datatype contract methods (get_*_shape, get_*stream_width, get_*_datatype) plus execute_node golden model, generate_params, weight-tensor packing (get_hw_compatible_weight_tensor/make_weight_file) live wholly in agnostic MVAU and are backend-independent - a new backend inherits them unchanged.
- **clean-seam** — instantiate_ip is the intended per-backend hook: MVAU.code_generation_ipi (base) does all the streamer/hierarchy TCL and defers only the IP-cell creation to MVAU_hls.instantiate_ip (681, ip_vlnv) vs MVAU_rtl.instantiate_ip (163, add_files + gen_top_module). This is the one deliberately clean HLS/RTL substitution point.
- **fused-no-seam** — The weight-delivery subsystem (memstream / dynload / fetch_weights RTL) is NOT on the HLS/RTL axis - it is orthogonal (mem_mode x dynamic_input x mlo_max_iter) yet is emitted from BOTH backends (hls code_generation_ipgen 144-155, rtl generate_hdl 308-322) and wired by the shared base code_generation_ipi. There is no seam separating 'compute backend' from 'weight-delivery backend'; they are entangled across HLS, RTL and the agnostic base.
- **fused-no-seam** — code_generation_ipi (agnostic base, 920) is ~236 lines of Vivado IPI TCL - a backend concern that cannot be lifted out of the agnostic class without also moving the mem_mode/mlo/dynamic branching it performs; base and backend TCL are interleaved.
- **fused-no-seam** — RTL rtlsim execute_node (rtl 100-140) and HLS rtlsim execute_node (hls 569-617) are duplicated rather than shared; the rtlsim harness logic is fused into each backend's execute_node with no common base method to swap.
