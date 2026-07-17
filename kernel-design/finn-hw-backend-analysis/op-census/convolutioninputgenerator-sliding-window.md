# Census: convolutioninputgenerator (Sliding Window Generator / SWG)

*A pure-RTL Sliding Window Generator with a thin, non-self-contained 'agnostic' base (shape/width methods leak the RTL-only parallel_window concept and resource estimators are 0 stubs) and a heavy _rtl subclass that computes SWG controller arithmetic and even whole Verilog blocks in Python, injecting them into five finn-rtllib/swg templates via global string replacement. It carries a third orthogonal axis (impl_style default/parallel x dynamic_mode) that the HLS/RTL abstraction does not represent.*

**Files:** `src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py`, `src/finn/custom_op/fpgadataflow/rtl/convolutioninputgenerator_rtl.py`, `finn-rtllib/swg/swg_template_default.sv`, `finn-rtllib/swg/swg_template_parallel.sv`, `finn-rtllib/swg/swg_template_wrapper.v`, `finn-rtllib/swg/swg_template_axilite.v`, `finn-rtllib/swg/swg_common.sv`, `finn-rtllib/swg/swg_pkg.sv`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `ConvolutionInputGenerator` | `convolutioninputgenerator.py` | `(HWCustomOp)` | NO (backend-only) |
| `ConvolutionInputGenerator_rtl` | `convolutioninputgenerator_rtl.py` | `(ConvolutionInputGenerator, RTLBackend)` | yes |

## Redesign pressure

This family is a pure-RTL op (no HLS variant exists), so the 2-axis HLS/RTL abstraction is already lopsided: the 'agnostic base' exists only to be subclassed by one backend, yet it leaks backend concepts (use_parallel_window_output, parallel_window) into its core shape/width contract, making it non-instantiable and non-reusable on its own. The real complexity is a THIRD hidden axis the abstraction does not model: impl_style in {default, parallel} plus dynamic_mode, which together select among five different rtllib templates and two distinct prepare_codegen paths -- this cuts across the RTL backend rather than fitting under it. The heaviest redesign pressure is the codegen itself: address-increment arithmetic, 5-loop FSM counter values, and (for parallel) whole Verilog module bodies are computed in Python and stitched into swg templates by global $KEY$ string replacement, so op-logic and HDL are fused with no clean seam and correctness rests on assert guards and placeholder-name uniqueness. Finally, get_dynamic_config mutating node geometry and cppsim re-interleaving output to satisfy downstream VVAU are cross-cutting couplings that any new abstraction must either forbid or model explicitly.

## Overrides (24)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | ConvolutionInputGenerator | 51 | Adds SWG-specific geometry attrs (ConvKernelDim, IFMDim/OFMDim as [H,W] lists, Stride, Dilation, SIMD, depthwise, ram_style, parallel_window, is1D, dynamic_mode). |
| `get_normal_input_shape` | ConvolutionInputGenerator | 86 | NHWC image shape (1, IFMDim_H, IFMDim_W, IFMChannels). |
| `get_folded_input_shape` | ConvolutionInputGenerator | 92 | Folds IFMChannels into wf=ifm_ch/simd along an added SIMD axis. |
| `get_normal_output_shape` | ConvolutionInputGenerator | 101 | Output is im2col-expanded: last dim = k_h*k_w*ifm_ch; OFM dims computed via qonnx compute_conv_output_dim. |
| `get_folded_output_shape` | ConvolutionInputGenerator | 113 | Two DIFFERENT folded layouts depending on use_parallel_window_output(): parallel -> (..,wf,k_h*k_w*simd); default -> (..,wf,simd). |
| `get_input_datatype` | ConvolutionInputGenerator | 159 | Reads inputDataType nodeattr. |
| `get_output_datatype` | ConvolutionInputGenerator | 163 | Reads outputDataType nodeattr (SWG is datatype-preserving passthrough). |
| `get_instream_width` | ConvolutionInputGenerator | 167 | simd * input bitwidth. |
| `get_outstream_width` | ConvolutionInputGenerator | 177 | For parallel_window multiplies instream width by k_h*k_w (whole window emitted in parallel); otherwise equal to instream width. |
| `get_exp_cycles` | ConvolutionInputGenerator | 212 | Returns hard 0 (agnostic base has no real cycle model). |
| `bram_estimation` | ConvolutionInputGenerator | 215 | Returns 0 stub. |
| `lut_estimation` | ConvolutionInputGenerator | 218 | Returns 0 stub. |
| `uram_estimation` | ConvolutionInputGenerator | 221 | Returns 0 stub. |
| `execute_node` | ConvolutionInputGenerator | 224 | Functional sim by constructing and executing a qonnx Im2Col node in a throwaway single-node graph. |
| `get_nodeattr_types` | ConvolutionInputGenerator_rtl | 64 | Adds M (parallelization factor, not implemented) and merges ConvolutionInputGenerator + RTLBackend attr dicts. |
| `get_exp_cycles` | ConvolutionInputGenerator_rtl | 124 | Real analytic cycle model split by impl_style (parallel vs default) and 1D vs 2D vs depthwise. |
| `bram_estimation` | ConvolutionInputGenerator_rtl | 181 | Models circular/line-buffer BRAM cascade from buffer_depth and ram_style. |
| `lut_estimation` | ConvolutionInputGenerator_rtl | 241 | LUTRAM model for distributed ram_style. |
| `uram_estimation` | ConvolutionInputGenerator_rtl | 252 | URAM cascade model for ultra ram_style. |
| `execute_node` | ConvolutionInputGenerator_rtl | 312 | cppsim path delegates to base Im2Col exec then re-interleaves channels for depthwise; rtlsim path delegates to RTLBackend.execute_node. |
| `get_verilog_top_module_intf_names` | ConvolutionInputGenerator_rtl | 942 | Adds an s_axilite interface entry only when dynamic_mode is set. |
| `generate_hdl` | ConvolutionInputGenerator_rtl | 823 | RTLBackend abstract impl: selects template, string-fills placeholders, writes _impl.sv/_wrapper.v/_axilite.v, copies static swg_common.sv/swg_pkg.sv. |
| `get_rtl_file_list` | ConvolutionInputGenerator_rtl | 899 | RTLBackend abstract impl: lists swg_pkg.sv, generated wrapper/impl, swg_common.sv (+axilite in dynamic mode). |
| `code_generation_ipi` | ConvolutionInputGenerator_rtl | 917 | RTLBackend abstract impl: emits Vivado add_files + create_bd_cell TCL. |

## Hacks (17 — 0 blocker, 8 major)

- **[major/inheritance-irregularity]** `convolutioninputgenerator.py:124` — Base class get_folded_output_shape (l.124) and get_outstream_width (l.178) call self.use_parallel_window_output(), but that method is ONLY defined in the RTL subclass (rtl file l.79). Instantiating the bare agnostic base and calling these methods raises AttributeError -- the 'agnostic base' is not actually self-consistent.
- **[major/base-class-leak]** `convolutioninputgenerator.py:212` — get_exp_cycles, bram_estimation, lut_estimation, uram_estimation in the agnostic base all return hard 0 (l.212-222). These are meaningless placeholders that only become correct in the RTL subclass; any consumer using the base's estimates silently gets 0.
- **[major/todo-marker]** `convolutioninputgenerator_rtl.py:53` — Header comment 'NOTE: "Parallel" implementation style not yet implemented in this version!' contradicts the code: select_impl_style (l.810-811) can return 'parallel', generate_hdl handles it (l.831-832) via prepare_codegen_parallel, and resource estimators branch on impl_style=='parallel' (l.194,265,298). Stale/contradictory comment about a live code path.
- **[major/hard-coded-param]** `convolutioninputgenerator_rtl.py:67` — 'M' nodeattr declared with comment 'additional parallelization parameter - not yet implemented' yet it is actively used in prepare_codegen_parallel (mmv_in=M*1, mmv_out=M*k_h*k_w l.541-542, and in output-mapping index math l.737-738). Attr documented as not implemented but wired into codegen.
- **[major/template-surgery]** `convolutioninputgenerator_rtl.py:675` — prepare_codegen_parallel builds raw (System)Verilog module bodies via Python str.format(): $GENERATE_REG_FIFOS$ (l.675-698), $GENERATE_BRAM_FIFOS$ (l.700-724), $GENERATE_OUTPUT_MAPPING$ (l.726-744), $GENERATE_BUFFER_CONNECTION$ (l.746-771). HDL instantiation logic lives inside the Python op, not in an rtllib file.
- **[major/template-surgery]** `convolutioninputgenerator_rtl.py:865` — generate_hdl performs global string replace of $KEY$ placeholders across three separately-read template files (impl .sv, wrapper .v, axilite .v) in one loop (l.865-870). Correctness depends on placeholder-name uniqueness/non-overlap; no escaping or structured templating.
- **[major/cross-backend-leak]** `convolutioninputgenerator_rtl.py:321` — execute_node cppsim branch re-interleaves im2col output channels (reshape+transpose+reshape l.323-331) specifically 'because subsequent VVAU_{hls/rtl} expects channels interleaved to match PE parallelism'. This SWG op hard-codes knowledge of a DIFFERENT downstream op's data layout.
- **[major/other]** `convolutioninputgenerator_rtl.py:984` — get_dynamic_config MUTATES persistent nodeattrs (IFMDim, OFMDim, Stride, Dilation l.984-987) as a side effect of computing an axilite register config, then re-runs prepare_codegen_default. A 'get_' accessor permanently rewrites node geometry.
- **[minor/magic-number]** `convolutioninputgenerator_rtl.py:203` — bram_estimation hard-codes Vivado BRAM aspect-ratio thresholds (512/1024/2048/4096/8192 -> ram_width 36/18/9/4/2/1) and 16384 cascade depth (l.203-235). Duplicated verbatim for the cascade-remainder case (l.221-232).
- **[minor/magic-number]** `convolutioninputgenerator_rtl.py:247` — lut_estimation: 'ram_luts = buffer_width * ceil(buffer_depth/38)' and constant '300 +' base LUTs (l.247,250) -- unexplained fitting constants.
- **[minor/magic-number]** `convolutioninputgenerator_rtl.py:272` — uram_estimation hard-codes UltraScale+ URAM geometry ram_depth=4096, ram_width=72 (l.272-273); uram_efficiency_estimation repeats 72*4096 capacity constant (l.309).
- **[minor/brittle-assumption]** `convolutioninputgenerator_rtl.py:406` — prepare_codegen_default asserts on address-increment wrap logic with a user-facing remediation string 'try setting parallel_window=1' (l.406-411). Codegen correctness gated by an assert whose failure mode is a manual folding change.
- **[minor/duplicated-logic]** `convolutioninputgenerator_rtl.py:194` — The parallel-style line-buffer geometry (kernel_width, buffer_depth=(ifm_dim_w-kernel_width)+ifm_dim_w*(dilation_h-1), buffer_count=k_h-1) is copy-pasted across bram_estimation (l.197-199), uram_estimation (l.268-270) and uram_efficiency_estimation (l.301-303).
- **[minor/duplicated-logic]** `convolutioninputgenerator_rtl.py:516` — prepare_codegen_parallel duplicates large blocks of geometry/loop-counter setup from prepare_codegen_default (buffer_min_size, skip_columns/skip_rows, LAST_READ/WRITE_ELEM, cntr_bitwidth, loop iteration -2 offsets) rather than sharing a helper.
- **[minor/brittle-assumption]** `convolutioninputgenerator_rtl.py:737` — $GENERATE_OUTPUT_MAPPING$ access-index math uses integer arithmetic with M (access_idx = len(reg_fifo)-1-int((max(reg_fifo)-access_idx)/M), mmv_idx=(max-access_idx)%M) and a final 'assert out_idx == -1' (l.744) as the only guard that the generated Verilog wiring is complete.
- **[minor/todo-marker]** `convolutioninputgenerator_rtl.py:281` — uram_efficiency_estimation has a TODO about Versal flexible-width URAM (9/18/36/72) not being modeled; current model assumes UltraScale+ 72-bit only.
- **[minor/other]** `convolutioninputgenerator.py:78` — 'is1D' nodeattr is declared (l.78) but never read anywhere in either file; 1D vs 2D is instead detected at runtime via ifm_dim_h==1 or ifm_dim_w==1 (rtl l.145,195, etc.). Dead/misleading attribute.

## Hermeticity violations (9)

- **[env-var]** `convolutioninputgenerator_rtl.py:342` — prepare_codegen_default reads os.environ['FINN_ROOT'] to locate swg_template_default[_dynamic].sv.
- **[env-var]** `convolutioninputgenerator_rtl.py:521` — prepare_codegen_parallel reads os.environ['FINN_ROOT'] for swg_template_parallel.sv.
- **[env-var]** `convolutioninputgenerator_rtl.py:861` — generate_hdl reads os.environ['FINN_ROOT'] for wrapper (l.861), axilite (l.863) templates and to shutil.copy2 swg_common.sv/swg_pkg.sv (l.891-892).
- **[env-var]** `convolutioninputgenerator_rtl.py:902` — get_rtl_file_list joins os.environ['FINN_ROOT']/finn-rtllib/swg for abspath file list.
- **[filesystem-path]** `convolutioninputgenerator_rtl.py:854` — generate_hdl writes generated .sv/.v into nodeattr code_gen_dir_ipgen and copies static sources there; sets ipgen_path/ip_path nodeattrs as side effect (l.896-897).
- **[module-mutable-state]** `convolutioninputgenerator_rtl.py:984` — get_dynamic_config permanently overwrites IFMDim/OFMDim/Stride/Dilation nodeattrs on the node instance as a side effect (order-dependent: later shape queries see mutated geometry).
- **[order-dependence]** `convolutioninputgenerator_rtl.py:842` — generate_hdl writes gen_top_module nodeattr; subsequent get_rtl_file_list/code_generation_ipi depend on it being set first (order dependence).
- **[sibling-op-coupling]** `convolutioninputgenerator.py:224` — execute_node imports/constructs a qonnx Im2Col node via getCustomOp and runs it to produce outputs -- numeric behavior delegated to an external op.
- **[sibling-op-coupling]** `convolutioninputgenerator_rtl.py:321` — cppsim execute_node reshapes output to match downstream VVAU PE interleaving, hard-coupling this op's sim output to another op's expectations.

## finn-rtllib coupling (9)

- `swg/swg_template_default.sv` via **string-replace** — prepare_codegen_default selects this template (rtl l.341) when dynamic_mode=0; generate_hdl reads it (l.855) and str.replace's ~30 $KEY$ placeholders ($BUF_ELEM_TOTAL$, $LOOP_*_ITERATIONS$, $HEAD_INCR_*$, $TAIL_INCR_*$, $CNTR_BITWIDTH$, $INCR_BITWIDTH$, $INNERMOST_STATE$, $IS_DEPTHWISE$, $SIMD$, $ELEM_PER_WINDOW$, etc. set l.370-512), writes to <top>_impl.sv (l.871-875).
- `swg/swg_template_default_dynamic.sv` via **string-replace** — Same placeholder set as default template but selected when dynamic_mode=1 (rtl l.339); the AXI-Lite-reconfigurable variant. get_dynamic_config later maps the same $LOOP_*/$HEAD_INCR_*/$TAIL_INCR_*/$LAST_*_ELEM$ values to axilite register addresses (l.996-1013).
- `swg/swg_template_parallel.sv` via **string-replace** — Selected for impl_style=='parallel' (rtl l.521,832). Placeholders include Python-GENERATED Verilog blocks $GENERATE_REG_FIFOS$/$GENERATE_BRAM_FIFOS$/$GENERATE_BUFFER_CONNECTION$/$GENERATE_OUTPUT_MAPPING$ (built l.675-771) that instantiate swg_reg_buffer/swg_ram_buffer with WIDTH/DEPTH/RAM_STYLE parameters injected from Python.
- `swg/swg_template_wrapper.v` via **string-replace** — Read in generate_hdl (l.861) when dynamic_mode=0; same replace loop fills $TOP_MODULE_NAME$/$IN_WIDTH_PADDED$/$OUT_WIDTH_PADDED$/$BIT_WIDTH$/$RAM_STYLE$ (l.839-851); written to <top>_wrapper.v (l.876-880).
- `swg/swg_template_wrapper_dynamic.v` via **string-replace** — Wrapper variant selected when dynamic_mode=1 (l.858); exposes the extra s_axilite interface added in get_verilog_top_module_intf_names (l.953-954).
- `swg/swg_template_axilite.v` via **string-replace** — Read unconditionally (l.863) but only written (<top>_axilite.v, l.883-888) and added to file lists when dynamic_mode=1 (l.912-913, l.928-929). Provides the runtime-reconfig register file.
- `swg/swg_common.sv` via **file-copy** — Static shared core components (swg_reg_buffer/swg_ram_buffer/controller); shutil.copy2'd verbatim into code_gen_dir (l.891), listed in get_rtl_file_list (l.910) and code_generation_ipi (l.925). No parameterization.
- `swg/swg_pkg.sv` via **file-copy** — Static SystemVerilog package; shutil.copy2'd verbatim (l.892), first in file list (l.907) and IPI sources (l.922). No parameterization.
- `swg/swg_common.sv` via **parameter-passing** — Parallel-style generated Verilog instantiates swg_reg_buffer (#WIDTH=IN_WIDTH, DEPTH) l.682-687 and swg_ram_buffer (#WIDTH, DEPTH, RAM_STYLE) l.706-711 -- Python passes DEPTH per-FIFO and ram_style nodeattr into the module params.

## Seams (6)

- **fused-no-seam** — The agnostic base ConvolutionInputGenerator is NOT backend-independent: get_folded_output_shape (l.124) and get_outstream_width (l.178) call use_parallel_window_output(), which only exists in the RTL subclass. The base's shape/width contract is fused to a backend-specific concept (parallel_window impl style), so the base cannot be reused for a different backend without that method.
- **fused-no-seam** — generate_hdl + prepare_codegen_default/parallel + the swg .sv templates are a tightly-coupled unit: the Python computes address-increment arithmetic and 5-loop counter values that only make sense against the specific finn-rtllib/swg controller FSM (STATE_LOOP_* states, $INNERMOST_STATE$). The template and the Python are two halves of one algorithm; neither is independently swappable.
- **fused-no-seam** — Parallel impl-style Verilog (reg/bram FIFO instantiation, output mapping, buffer connection) is GENERATED inside Python (l.675-771), so there is no clean file boundary between 'op logic' and 'HDL' for that path.
- **clean-seam** — The RTLBackend contract methods (generate_hdl l.823, get_rtl_file_list l.899, code_generation_ipi l.917) are cleanly localized in the _rtl subclass; a hypothetical HLS variant could subclass ConvolutionInputGenerator and supply HLSBackend methods without touching them -- IF the base's use_parallel_window_output dependency were resolved.
- **clean-seam** — execute_node cppsim numerics are delegated to a standalone qonnx Im2Col node (base l.224-261); the functional model is backend-agnostic and could be shared by any backend variant unchanged.
- **clean-seam** — dynamic_mode is handled purely by swapping template file names and conditionally emitting the axilite component (l.338-341, l.857-860, l.912-913) -- a self-contained variant axis orthogonal to HLS/RTL.
