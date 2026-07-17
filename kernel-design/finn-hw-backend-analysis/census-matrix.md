# Phase B — Op-Family Customization Census: Rolled-Up Matrix

*27 op families, all classes under `src/finn/custom_op/fpgadataflow/`. Generated from 27 census agents + 27 adversarial-verify agents (Opus). Verify result: **174 confirmed, 3 partially-confirmed, 0 refuted** across all blocker/major hacks.*

Per-family detail in `op-census/<family>.md`. Sorted by blocker then major hack count.

| Family | overrides | hacks (blk/maj/min) | herm | rtllib | fused seams | backend-only |
|---|--:|--:|--:|--:|--:|:--:|
| [finn_loop (FINNLoop meta/container](op-census/finn-loop-finnloop-meta-container-node.md) | 18 | 22 (3/13/6) | 13 | 5 | 4 | Y |
| [elementwise_binary (ElementwiseBin](op-census/elementwise-binary-elementwisebinaryoper.md) | 26 | 24 (3/12/9) | 10 | 8 | 4 |  |
| [streamingfifo (StreamingFIFO)](op-census/streamingfifo-streamingfifo.md) | 15 | 14 (3/4/7) | 5 | 3 | 2 |  |
| [shuffle (inner_rtl + outer_hls)](op-census/shuffle-inner-rtl-outer-hls.md) | 44 | 17 (2/10/5) | 5 | 5 | 3 | Y |
| [matrixvectoractivation (MVAU / mat](op-census/matrixvectoractivation-mvau-matmul.md) | 40 | 18 (2/8/8) | 7 | 10 | 3 |  |
| [vectorvectoractivation (VVAU)](op-census/vectorvectoractivation-vvau.md) | 38 | 18 (2/8/8) | 8 | 9 | 4 |  |
| [layernorm (LayerNorm)](op-census/layernorm-layernorm.md) | 25 | 12 (2/5/5) | 4 | 6 | 2 |  |
| [thresholding](op-census/thresholding.md) | 30 | 17 (1/11/5) | 6 | 7 | 3 |  |
| [requant](op-census/requant.md) | 30 | 17 (1/5/11) | 4 | 6 | 3 |  |
| [streamingdatawidthconverter (Strea](op-census/streamingdatawidthconverter-streamingdat.md) | 17 | 10 (1/5/4) | 5 | 3 | 2 |  |
| [pool](op-census/pool.md) | 19 | 9 (1/4/4) | 1 | 0 | 1 | Y |
| [convolutioninputgenerator (Sliding](op-census/convolutioninputgenerator-sliding-window.md) | 24 | 17 (0/8/9) | 9 | 9 | 3 | Y |
| [checksum (backend-only, HLS-only)](op-census/checksum-backend-only-hls-only.md) | 22 | 10 (0/7/3) | 3 | 0 | 2 | Y |
| [fmpadding (FMPadding)](op-census/fmpadding-fmpadding.md) | 17 | 10 (0/6/4) | 5 | 4 | 2 |  |
| [lookup (streaming index-to-embeddi](op-census/lookup-streaming-index-to-embedding-look.md) | 25 | 13 (0/6/7) | 4 | 0 | 2 |  |
| [tlastmarker (backend-only)](op-census/tlastmarker-backend-only.md) | 21 | 12 (0/6/6) | 2 | 0 | 2 | Y |
| [iodma (backend-only)](op-census/iodma-backend-only.md) | 19 | 9 (0/5/4) | 1 | 0 | 2 | Y |
| [globalaccpool (GlobalAccPool)](op-census/globalaccpool-globalaccpool.md) | 14 | 9 (0/4/5) | 0 | 0 | 1 |  |
| [crop](op-census/crop.md) | 18 | 9 (0/4/5) | 0 | 0 | 1 |  |
| [fmpadding_pixel](op-census/fmpadding-pixel.md) | 13 | 7 (0/4/3) | 0 | 0 | 1 |  |
| [upsampler (UpsampleNearestNeighbou](op-census/upsampler-upsamplenearestneighbour.md) | 18 | 9 (0/4/5) | 3 | 0 | 1 |  |
| [streamingdataflowpartition (Stream](op-census/streamingdataflowpartition-streamingdata.md) | 4 | 6 (0/4/2) | 3 | 0 | 1 | Y |
| [hwsoftmax (HWSoftmax)](op-census/hwsoftmax-hwsoftmax.md) | 20 | 9 (0/3/6) | 2 | 0 | 2 |  |
| [labelselect](op-census/labelselect.md) | 20 | 10 (0/3/7) | 3 | 0 | 2 |  |
| [split (StreamingSplit)](op-census/split-streamingsplit.md) | 25 | 8 (0/3/5) | 3 | 0 | 2 |  |
| [duplicatestreams](op-census/duplicatestreams.md) | 22 | 7 (0/2/5) | 0 | 0 | 2 |  |
| [concat (StreamingConcat)](op-census/concat-streamingconcat.md) | 13 | 8 (0/2/6) | 0 | 0 | 1 |  |

## Aggregate hack taxonomy

| kind | count |
|---|--:|
| brittle-assumption | 78 |
| magic-number | 48 |
| duplicated-logic | 40 |
| hard-coded-param | 36 |
| other | 33 |
| template-surgery | 27 |
| inheritance-irregularity | 24 |
| todo-marker | 22 |
| cross-backend-leak | 13 |
| base-class-leak | 10 |
| **total** | **331** |

## Aggregate hermeticity-violation taxonomy

| kind | count |
|---|--:|
| env-var | 34 |
| filesystem-path | 26 |
| hidden-coupling | 22 |
| sibling-op-coupling | 8 |
| module-mutable-state | 8 |
| order-dependence | 6 |
| other | 2 |
| **total** | **106** |

## The 21 blocker hacks (actively prevent clean substitution)

- **pool** — [inheritance-irregularity] `pool_hls.py:116` — Pool_hls.execute_node exists ONLY to call HLSBackend.execute_node(self,...). Because MRO is (Pool_hls, Pool, HLSBackend, HWCustomOp), an un-overridden execute_node would resolve to Pool.execute_node (the python behavioral golden model), NOT the HLS cppsim/rtlsim path. The base class overloads execute_node with two incompatible meanings and MRO order silently picks the wrong one without this shim.
- **matrixvectoractivation** — [base-class-leak] `matrixvectoractivation.py:920` — code_generation_ipi (~236 lines) lives in the backend-AGNOSTIC MVAU base but is entirely Vivado IPI TCL block-design surgery, and it calls self.instantiate_ip() which is defined ONLY on MVAU_hls (line 681) and MVAU_rtl (line 163). The 'agnostic' class is therefore uninstantiable-by-contract and structurally depends on its subclasses.
- **matrixvectoractivation** — [cross-backend-leak] `matrixvectoractivation_hls.py:144` — code_generation_ipgen of the HLS variant calls generate_hdl_dynload/generate_hdl_memstream/generate_hdl_fetch_weights (lines 145,153,155) - i.e. the HLS op emits finn-rtllib Verilog for the weight streamer. HLS compute + RTL weight delivery are fused into one op.
- **layernorm** — [magic-number] `layernorm_wrapper_template.v:20` — Input TDATA width is hard-coded [$SIMD$-1:0][31:0] — every element is fixed at 32 bits regardless of inputDataType. get_instream_width() (layernorm.py:98) computes i_bits*SIMD from the datatype, so any input dtype whose bitwidth != 32 silently mismatches the RTL port. Effectively locks input to a 32-bit (float) type.
- **layernorm** — [magic-number] `layernorm_wrapper_template.v:25` — Output TDATA hard-coded [$SIMD$-1:0][31:0] (32-bit float) while get_output_datatype/get_outstream_width read the outputDataType attr. The attr is effectively ignored by the RTL; output is always fp32.
- **streamingfifo** — [base-class-leak] `streamingfifo.py:104` — The 'agnostic' base StreamingFIFO calls self.get_adjusted_depth() wrapped in try/except AttributeError, falling back to raw depth attr. get_adjusted_depth() is defined ONLY on the RTL subclass (streamingfifo_rtl.py:53). Base op depends on a subclass-only method via exception control-flow. Repeated at lines 104, 160, 195, 208, 224, 247.
- **streamingfifo** — [base-class-leak] `streamingfifo.py:90` — Base reads self.get_nodeattr('impl_style') inside try/except AttributeError, but impl_style is declared ONLY on the RTL subclass (streamingfifo_rtl.py:46). AttributeError branch raises 'still in hw abstraction format, run SpecializeLayers'. The base class is thus not backend-agnostic at all. Repeated at 90, 109, 150, 186, 238.
- **streamingfifo** — [cross-backend-leak] `streamingfifo_rtl.py:141` — code_generation_ipi contains a large 'vivado' branch (141-195) that instantiates Xilinx axis_data_fifo:2.0 infrastructure IP via TCL -- a THIRD backend hidden inside the class named *_rtl. impl_style multiplexes {rtl, vivado} at runtime instead of via the class hierarchy; the RTL class carries an entire non-FINN-RTL codegen path.
- **requant** — [magic-number] `requant_rtl.py:78` — format_sv_array emits scale/bias with fixed '{:.6f}' — only 6 decimal places for a 32-bit shortreal. Scale/bias are truncated to 6 decimals in the generated SV, which can perturb the quantization result vs the Python golden model (which uses full float32).
- **streamingdatawidthconverter** — [brittle-assumption] `streamingdatawidthconverter.py:84` — check_divisible_iowidths() is a no-op 'pass' in the agnostic base, silently permitting arbitrary in/out width ratios. The base thus admits configs (non-integer-ratio widths) that only the HLS backend can build via its LCM path; the RTL backend must re-add the constraint by overriding this hook (rtl line 48). Capability differs by backend but the base pretends both are equal.
- **vectorvectoractivation** — [brittle-assumption] `vectorvectoractivation_rtl.py:89` — INDENTATION BUG: the entire rtlsim block (lines 89-130: get_rtlsim, reset, weight-stream build, rtlsim_multi_io, output writeback, context assignment) is indented 16 spaces INSIDE the `for inputs in node.input:` loop opened at line 62. Unlike VVAU_hls.execute_node where the rtlsim block sits outside the input loop, here rtlsim is re-run once per node input (2-3 times), and sim/export_idt leak from loop iterations. Copy-paste-and-reindent error that only works because the last iteration overwrites context.
- **vectorvectoractivation** — [cross-backend-leak] `vectorvectoractivation.py:617` — make_weight_file (on the backend-AGNOSTIC VVAU base) branches on self.onnx_node.op_type == 'VVAU_rtl' to choose weight layout (unflipped npy at 618, pe_simd_flipped .dat at 628). The base hard-codes knowledge of the RTL subclass string to pick RTL vs HLS memory ordering — inverted dependency defeating the agnostic base.
- **elementwise_binary** — [base-class-leak] `hwcustomop.py:311` — HWCustomOp.generate_hdl_memstream branches on op_type.startswith('Elementwise'). The elementwise family RELIES on this leak: both hls (line 141) and rtl (line 177) call self.generate_hdl_memstream(fpgapart). The base op contract hard-codes knowledge of this concrete op family by name.
- **elementwise_binary** — [base-class-leak] `hwcustomop.py:359` — generate_hdl_fetch_weights ALSO branches on op_type.startswith('Elementwise') with a dedicated else-branch computing mw=1, mh=rhs_shape[-1], simd=1, n_reps=prod(rhs_shape[:-1]) plus a 'TODO use broadcast rhs shape here' at line 375. Elementwise-specific weight-fetch logic living inside the shared base class.
- **elementwise_binary** — [inheritance-irregularity] `elementwise_binary_hls.py:1070` — ElementwiseBitShift_hls must override get_nodeattr_types to explicitly call elementwise_binary.ElementwiseBitShift.get_nodeattr_types instead of ElementwiseBinaryOperation's, because the diamond MRO (ElementwiseBinaryOperation_hls, ElementwiseBitShift) would otherwise resolve to the wrong parent and drop the 'direction' attribute. A latent MRO trap for any op that adds attrs.
- **thresholding** — [base-class-leak] `hwcustomop.py:310` — HWCustomOp.generate_hdl_memstream hard-codes op_type allowlist ['MVAU_hls','MVAU_rtl','VVAU_hls','VVAU_rtl','Thresholding_hls'] and special-cases op_type.startswith('Thresholding') to call calc_tmem() vs calc_wmem(). Thresholding_hls.code_generation_ipgen (thresholding_hls.py:181) relies on this leak to emit the memstream wrapper .v.
- **finn_loop** — [inheritance-irregularity] `finn_loop.py:1175` — get_rtl_file_list is a required RTLBackend @abstractmethod but is implemented as a bare 'pass' returning None. Any base-class code path that calls it (RTLBackend.prepare_rtlsim at rtlbackend.py:57) would break; FINNLoop dodges this by overriding prepare_rtlsim (line 277) to read all_verilog_srcs.txt instead. The abstract contract is satisfied only nominally.
- **finn_loop** — [cross-backend-leak] `finn_loop.py:421` — generate_params dispatches on child op_type string prefixes ('MVAU', 'Elementwise', 'Thresholding') to decide how to rename/concatenate .dat files, with 'else: raise Exception' (line 450). This hard-codes knowledge of three sibling ops' internal param-file naming conventions inside the loop op.
- **finn_loop** — [brittle-assumption] `finn_loop.py:712` — adjacency_list is built with a predicate hard-coding exactly three op_type families (Thresholding_rtl, MVAU_rtl, Elementwise*) each gated on a 'mlo_max_iter' attribute > 0. The block-design stream-tap wiring only works for these ops.
- **shuffle** — [cross-backend-leak] `outer_shuffle.py:20` — _NestSim class + OuterShuffle.get_exp_cycles (lines 20-238) are a complete Python reimplementation of the HLS Nest<>/input_gen.hpp template's read-pointer/free-pointer pipeline, living in the supposedly backend-AGNOSTIC OuterShuffle base. The agnostic base is fused to HLS internals: any change to input_gen.hpp silently invalidates this estimator.
- **shuffle** — [brittle-assumption] `outer_shuffle.py:189` — get_exp_cycles reads os.environ.get('XILINX_VIVADO') then re.search(r'\b(20\d{2})\.(1|2)\b', vivado_path) at line 190 and unconditionally dereferences match.group(1/2) at line 191 -- crashes with AttributeError if XILINX_VIVADO is unset or the path lacks a YYYY.[12] token. A pure cost model taking a hard dependency on a tool-install path/env-var.