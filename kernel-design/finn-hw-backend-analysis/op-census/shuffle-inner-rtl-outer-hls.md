# Census: shuffle (inner_rtl + outer_hls)

*A three-tier transpose family: a non-lowerable Shuffle placeholder that a transformation decomposes into RTL-only InnerShuffle and HLS-only OuterShuffle sub-ops. The InnerShuffle/OuterShuffle 'agnostic' bases have clean codegen seams but are heavily fused to their single backend via embedded, tool-and-microarchitecture-specific cycle models.*

**Files:** `src/finn/custom_op/fpgadataflow/shuffle.py`, `src/finn/custom_op/fpgadataflow/inner_shuffle.py`, `src/finn/custom_op/fpgadataflow/rtl/inner_shuffle_rtl.py`, `src/finn/custom_op/fpgadataflow/outer_shuffle.py`, `src/finn/custom_op/fpgadataflow/hls/outer_shuffle_hls.py`, `finn-rtllib/inner_shuffle/inner_shuffle_template.v`, `finn-rtllib/inner_shuffle/`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Shuffle` | `shuffle.py` | `(HWCustomOp)` | NO (backend-only) |
| `InnerShuffle` | `inner_shuffle.py` | `(HWCustomOp)` | NO (backend-only) |
| `InnerShuffle_rtl` | `inner_shuffle_rtl.py` | `(InnerShuffle, RTLBackend)` | yes |
| `OuterShuffle` | `outer_shuffle.py` | `(HWCustomOp)` | NO (backend-only) |
| `OuterShuffle_hls` | `outer_shuffle_hls.py` | `(OuterShuffle, HLSBackend)` | yes |

## Redesign pressure

The 2-axis HLS/RTL abstraction assumes an agnostic base holds shape/datatype/functional semantics while backend classes own codegen and cost. This family violates that on both cost and identity axes. Cost estimation is the worst offender: both 'agnostic' bases embed backend-specific cycle models -- OuterShuffle contains a full Python re-simulation of the HLS input_gen.hpp Nest<> pipeline (with WP_DELAY, URAM II thresholds, and a hard XILINX_VIVADO env dependency), and InnerShuffle bakes in the double-buffered-BRAM RTL formula -- so the bases are not backend-agnostic at all. Second, the family is really three-tier: a non-lowerable Shuffle placeholder that depends on a transformation to decompose into InnerShuffle(RTL-only) and OuterShuffle(HLS-only), meaning the backend choice is fixed per-sub-op and asymmetric (no InnerShuffle_hls, no OuterShuffle_rtl), which the flat HLS/RTL variant matrix cannot express. Third, HDL emission is raw $KEY$ string-replace over a .v template plus verbatim .sv file copies with the source list duplicated three times, so there is no structured codegen seam. A redesign needs a cost-model that lives with the backend (not the base), a way to express op decomposition/lowering as a first-class concept, and structured (non-string-surgery) HDL parameterization.

## Overrides (44)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | Shuffle | 32 | adds transpose_in/out_shape, in/out_shape, perm, SIMD, NumChannels plus SIMD-config tracking attrs (original_node_name/original_simd) |
| `get_normal_input_shape` | Shuffle | 79 | reads in_shape nodeattr directly |
| `get_normal_output_shape` | Shuffle | 82 | reads out_shape nodeattr directly |
| `execute_node` | Shuffle | 85 | numpy reshape->transpose(perm)->reshape functional model |
| `get_input_datatype` | Shuffle | 93 | reads data_type nodeattr string |
| `get_output_datatype` | Shuffle | 121 | same as input datatype (transpose preserves type) |
| `get_instream_width` | Shuffle | 111 | ibits*SIMD |
| `get_outstream_width` | Shuffle | 116 | obits*SIMD |
| `get_folded_input_shape` | Shuffle | 133 | splits innermost dim into [fold, SIMD] |
| `get_folded_output_shape` | Shuffle | 125 | splits innermost dim into [fold, SIMD] |
| `verify_node` | Shuffle | 108 | not implemented |
| `get_exp_cycles` | Shuffle | 141 | decomposes transpose into Inner/Outer stages, builds throwaway ONNX nodes for each, returns max of per-stage estimates |
| `get_nodeattr_types` | InnerShuffle | 22 | only data_type/in_shape/SIMD (no perm/transpose shapes - it is a fixed last-2-dim swap) |
| `get_normal_output_shape` | InnerShuffle | 36 | swaps last two dims of input shape (2D transpose semantics) |
| `execute_node` | InnerShuffle | 40 | numpy transpose of last two axes only |
| `get_folded_input_shape` | InnerShuffle | 87 | flattens whole tensor to [prod/SIMD, SIMD] rather than keeping outer dims |
| `get_folded_output_shape` | InnerShuffle | 79 | splits innermost into [fold, SIMD] |
| `get_exp_cycles` | InnerShuffle | 94 | models double-buffered BRAM RTL: 2*total_elems + page_size |
| `get_input_datatype` | InnerShuffle | 50 | reads data_type |
| `get_output_datatype` | InnerShuffle | 75 | same type |
| `get_instream_width` | InnerShuffle | 65 | ibits*SIMD |
| `get_outstream_width` | InnerShuffle | 70 | obits*SIMD |
| `get_nodeattr_types` | InnerShuffle_rtl | 61 | merges InnerShuffle + RTLBackend attrs |
| `generate_hdl` | InnerShuffle_rtl | 78 | RTLBackend abstract; string-replace fills inner_shuffle_template.v and copies 4 .sv files |
| `get_rtl_file_list` | InnerShuffle_rtl | 111 | returns 4 rtllib .sv + generated top .v |
| `code_generation_ipi` | InnerShuffle_rtl | 128 | RTLBackend abstract; add_files + create_bd_cell of module reference |
| `execute_node` | InnerShuffle_rtl | 147 | dispatch: rtlsim -> RTLBackend.execute_node, else InnerShuffle numpy |
| `get_nodeattr_types` | OuterShuffle | 70 | adds loop_coeffs plus perm/transpose shapes/NumChannels |
| `get_normal_input_shape` | OuterShuffle | 87 | attr read |
| `get_normal_output_shape` | OuterShuffle | 90 | attr read |
| `execute_node` | OuterShuffle | 93 | reshape->transpose(perm)->reshape |
| `verify_node` | OuterShuffle | 116 | not implemented |
| `get_folded_input_shape` | OuterShuffle | 141 | split innermost into [fold,SIMD] |
| `get_folded_output_shape` | OuterShuffle | 133 | split innermost |
| `get_instream_width` | OuterShuffle | 119 | ibits*SIMD |
| `get_outstream_width` | OuterShuffle | 124 | obits*SIMD |
| `get_exp_cycles` | OuterShuffle | 149 | full Python re-simulation of the HLS input_gen pipeline via _NestSim, incl. buffer-size / II derivation |
| `get_nodeattr_types` | OuterShuffle_hls | 58 | OuterShuffle \| HLSBackend attrs |
| `global_includes` | OuterShuffle_hls | 61 | HLS abstract; includes input_gen.hpp + ap_int/hls_vector/hls_stream |
| `defines` | OuterShuffle_hls | 69 | HLS abstract; emits SIMD const + TE/TV vector typedefs |
| `docompute` | OuterShuffle_hls | 80 | HLS abstract; instantiates input_gen<...> template with interleaved out_shape/loop_coeffs |
| `blackboxfunction` | OuterShuffle_hls | 100 | HLS abstract; top fn signature |
| `pragmas` | OuterShuffle_hls | 110 | AXIS interface + aggregate + dataflow pragmas |
| `execute_node` | OuterShuffle_hls | 123 | delegates to HLSBackend.execute_node |

## Hacks (17 — 2 blocker, 10 major)

- **[blocker/cross-backend-leak]** `outer_shuffle.py:20` — _NestSim class + OuterShuffle.get_exp_cycles (lines 20-238) are a complete Python reimplementation of the HLS Nest<>/input_gen.hpp template's read-pointer/free-pointer pipeline, living in the supposedly backend-AGNOSTIC OuterShuffle base. The agnostic base is fused to HLS internals: any change to input_gen.hpp silently invalidates this estimator.
- **[blocker/brittle-assumption]** `outer_shuffle.py:189` — get_exp_cycles reads os.environ.get('XILINX_VIVADO') then re.search(r'\b(20\d{2})\.(1|2)\b', vivado_path) at line 190 and unconditionally dereferences match.group(1/2) at line 191 -- crashes with AttributeError if XILINX_VIVADO is unset or the path lacks a YYYY.[12] token. A pure cost model taking a hard dependency on a tool-install path/env-var.
- **[major/duplicated-logic]** `inner_shuffle_rtl.py:19` — auto_size_simd (lines 19-42) is byte-for-byte duplicated in hls/outer_shuffle_hls.py lines 18-41. Two copies of the SIMD-resizing heuristic in sibling backend files.
- **[major/duplicated-logic]** `inner_shuffle_rtl.py:105` — The inner_shuffle .sv source-file list is hard-coded THREE times with inconsistent ordering: generate_hdl line 105 ['inner_shuffle.sv','skid.sv','elasticmem.sv','queue.sv'], get_rtl_file_list line 120 [inner_shuffle, skid, elasticmem, queue], code_generation_ipi line 132 [inner_shuffle, skid, queue, elasticmem] (queue/elasticmem swapped). Drift risk across the three.
- **[major/template-surgery]** `inner_shuffle_rtl.py:94` — generate_hdl does raw str.replace of $KEY$ placeholders into inner_shuffle_template.v (lines 92-99) -- no escaping, no validation that all placeholders were filled; classic string-template surgery for HDL generation.
- **[major/brittle-assumption]** `inner_shuffle_rtl.py:51` — __init__ mutates the SIMD nodeattr at construction time via auto_size_simd (lines 51-59) when I_dim % SIMD != 0. Constructing the CustomOp has a side effect on graph attributes; re-instantiation could re-trigger. Same pattern in outer_shuffle_hls __init__ lines 48-56.
- **[major/hard-coded-param]** `outer_shuffle.py:184` — Magic constants: WP_DELAY=4 (line 184), URAM_DEPTH_THRESHOLD=262144 (line 197), pipeline_ii=3 for URAM (line 198), cycle<total_elems*10 loop guard (line 211). All hard-tied to a specific HLS pipeline and must track the RTL/HLS by hand.
- **[major/inheritance-irregularity]** `inner_shuffle.py:87` — InnerShuffle.get_folded_input_shape (lines 87-92) flattens to [prod/SIMD, SIMD] and OMITS the SIMD-divisibility assert present in every sibling fold method, while get_folded_output_shape (line 79) keeps the outer-dims convention. Input/output fold shapes are structurally inconsistent within the same class.
- **[major/duplicated-logic]** `outer_shuffle_hls.py:82` — docompute (lines 82-85) recomputes the interleaved out_shape/loop_coeffs exactly as OuterShuffle.get_exp_cycles lines 169-180 does -- the loop-nest folding math is duplicated between the HLS emitter and the base cost model.
- **[major/brittle-assumption]** `outer_shuffle_hls.py:83` — docompute mutates the list returned by get_nodeattr('transpose_out_shape') in place (`out_shape[-1] = int(out_shape[-1]/simd)`, line 83). timeout_value similarly relies on get_normal_input_shape. If get_nodeattr returns a live reference this corrupts the stored attr on repeated calls.
- **[major/other]** `shuffle.py:18` — Hidden coupling: Shuffle imports _is_inner_shuffle, decompose_transpose_with_constraints, shuffle_perfect_loopnest_coeffs from finn.transformation.fpgadataflow.transpose_decomposition -- a custom_op importing a transformation module, and get_exp_cycles (lines 154-205) re-runs the decomposition + constructs InnerShuffle/OuterShuffle ONNX nodes via getCustomOp to estimate cost.
- **[major/other]** `outer_shuffle_hls.py:63` — global_includes references input_gen.hpp which is NOT present in finn-rtllib nor under src/finn (grep found no such file in-repo); it is an external finn-hlslib header pulled via the -I$FINN_ROOT/deps/finn-hlslib include path (hlsbackend.py:268). The HLS emitter's core dependency lives outside this repo checkout.
- **[minor/other]** `inner_shuffle_rtl.py:105` — Loop-variable shadowing: `sv_files = [...]` then `for sv_files in sv_files:` (lines 105-107) rebinds the list name to each element. Works by accident (iterator captured first) but leaves sv_files as a string afterward; a latent footgun.
- **[minor/other]** `inner_shuffle_rtl.py:67` — get_template_values (lines 67-76) builds the exact code_gen_dict that generate_hdl re-builds inline at lines 84-91; dead/unused duplicate helper never called.
- **[minor/brittle-assumption]** `inner_shuffle.py:25` — in_shape attr comment: 'Needs to be len==2 can we assert that somewhere?' -- the 2D-transpose contract is documented but never enforced; get_normal_output_shape/get_exp_cycles assume last-two-dim structure.
- **[minor/todo-marker]** `shuffle.py:109` — verify_node raises NotImplementedError('This function is not yet immplemented.') (typo). Duplicated at outer_shuffle.py:117. Neither op implements node verification.
- **[minor/hard-coded-param]** `shuffle.py:72` — NumChannels default hard-coded to 128 (shuffle.py:72, outer_shuffle.py:80); arbitrary magic default unrelated to actual tensor shape.

## Hermeticity violations (5)

- **[env-var]** `outer_shuffle.py:189` — get_exp_cycles reads os.environ['XILINX_VIVADO'] to version-gate pipeline II; cost estimate depends on ambient tool-install path and crashes if unset.
- **[env-var]** `inner_shuffle_rtl.py:79` — generate_hdl (line 79) and get_rtl_file_list (line 114) read os.environ['FINN_ROOT'] to locate finn-rtllib/inner_shuffle sources.
- **[filesystem-path]** `inner_shuffle_rtl.py:105` — generate_hdl shutil.copy's 4 .sv files from the rtllib dir into code_gen_dir_ipgen and mutates ipgen_path/ip_path nodeattrs (lines 107-109); filesystem side effects + attr mutation during HDL gen.
- **[sibling-op-coupling]** `shuffle.py:18` — Shuffle imports the transpose_decomposition transformation and instantiates InnerShuffle/OuterShuffle nodes (getCustomOp) inside get_exp_cycles -- a custom_op depends on a transformation and on its sibling ops to compute its own cost.
- **[module-mutable-state]** `inner_shuffle_rtl.py:51` — __init__ mutates the SIMD nodeattr (auto_size_simd) as a construction side effect; ditto outer_shuffle_hls.py:48. Object construction silently rewrites graph state.

## finn-rtllib coupling (5)

- `inner_shuffle/inner_shuffle_template.v` via **string-replace** — generate_hdl (inner_shuffle_rtl.py:92-99) reads inner_shuffle_template.v and str.replace's $TOP_MODULE_NAME$/$I$/$J$/$SIMD$/$WIDTH$/$STREAM_BITS$ (dict built lines 84-91) to produce the top wrapper .v. Template params I,J,SIMD,BITS map onto the inner_shuffle instance at template.v lines 29-34.
- `inner_shuffle/inner_shuffle.sv` via **file-copy** — shutil.copy'd verbatim into code_gen_dir (inner_shuffle_rtl.py:105-107); the parameterized inner_shuffle #(.BITS,.I,.J,.SIMD) module is instantiated only via the generated wrapper, not re-templated.
- `inner_shuffle/skid.sv` via **file-copy** — copied verbatim (line 105-107) and added via add_files in code_generation_ipi (line 132-139); no parameterization at Python level.
- `inner_shuffle/elasticmem.sv` via **file-copy** — copied verbatim; BRAM double-buffer backing the transpose (referenced in get_exp_cycles page_size model).
- `inner_shuffle/queue.sv` via **file-copy** — copied verbatim (line 105-107); listed in get_rtl_file_list (line 124) and code_generation_ipi (line 132).

## Seams (5)

- **clean-seam** — InnerShuffle (agnostic) / InnerShuffle_rtl (RTLBackend) is a textbook clean split: base holds shape/datatype/functional-exec, RTL variant holds generate_hdl/get_rtl_file_list/code_generation_ipi + an execute_node mode-dispatch (inner_shuffle_rtl.py:147). A second backend could subclass InnerShuffle without touching the base.
- **clean-seam** — OuterShuffle_hls implements exactly the 4 HLSBackend abstracts + pragmas/execute_node/timeout_value; the HLS emission is confined to the _hls file and could be swapped for another HLS impl cleanly at that layer.
- **fused-no-seam** — OuterShuffle.get_exp_cycles + _NestSim (outer_shuffle.py:20-238) fuse the 'agnostic' base to the HLS input_gen.hpp microarchitecture (Nest<> pointer logic, WP_DELAY, URAM II, Vivado version gating). An RTL OuterShuffle backend cannot reuse OuterShuffle as-is because its cost model IS the HLS pipeline. No seam between base and HLS here.
- **fused-no-seam** — InnerShuffle.get_exp_cycles (inner_shuffle.py:94-108) hard-codes the double-buffered-BRAM RTL formula (2*total_elems + page_size) into the agnostic base, so the base is silently RTL-specific despite claiming backend independence.
- **fused-no-seam** — Shuffle op (shuffle.py) has no backend at all -- it is never lowered; its get_exp_cycles reconstructs the decomposition and instantiates sibling ops. It is a pre-lowering placeholder fused to the transpose_decomposition transformation, not a HLS/RTL-swappable op.
