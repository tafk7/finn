# Census: elementwise_binary (ElementwiseBinaryOperation)

*A large, mostly-clean elementwise binary op family: one backend-agnostic base (ElementwiseBinaryOperation) with 18 pure op-identity subclasses, plus HLS (full generative codegen for 17 ops) and RTL (template-filled finn-rtllib/eltwise, 3 ops) backends. Its customization mass is concentrated in two-input broadcasting, a const/decoupled/MLO parameter-streaming datapath that leaks the family name into the shared base class, and heavily duplicated block-design/datatype logic across the two backends.*

**Files:** `src/finn/custom_op/fpgadataflow/elementwise_binary.py`, `src/finn/custom_op/fpgadataflow/hls/elementwise_binary_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/elementwise_binary_rtl.py`, `finn-rtllib/eltwise/eltwise_template.v`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `ElementwiseBinaryOperation` | `elementwise_binary.py` | `(HWCustomOp)` | yes |
| `ElementwiseAdd/Sub/AbsDiff/Mul/Div/And/Or/Xor/Equal/Less/LessOrEqual/Greater/GreaterOrEqual/BitwiseAnd/BitwiseOr/BitwiseXor/BitShift/Max (18 agnostic subclasses)` | `elementwise_binary.py` | `(ElementwiseBinaryOperation)` | yes |
| `ElementwiseBinaryOperation_hls` | `elementwise_binary_hls.py` | `(ElementwiseBinaryOperation, HLSBackend)` | yes |
| `ElementwiseAdd_hls/Sub_hls/AbsDiff_hls/Mul_hls/Div_hls/And_hls/Or_hls/Xor_hls/Equal_hls/Less_hls/LessOrEqual_hls/Greater_hls/GreaterOrEqual_hls/BitwiseAnd_hls/BitwiseOr_hls/BitwiseXor_hls/Max_hls (17 pass-only subclasses)` | `elementwise_binary_hls.py` | `(ElementwiseBinaryOperation_hls, elementwise_binary.Elementwise<Op>)` | yes |
| `ElementwiseBitShift_hls` | `elementwise_binary_hls.py` | `(ElementwiseBinaryOperation_hls, elementwise_binary.ElementwiseBitShift)` | yes |
| `ElementwiseBinary_rtl` | `elementwise_binary_rtl.py` | `(ElementwiseBinaryOperation, RTLBackend)` | yes |
| `ElementwiseAdd_rtl` | `elementwise_binary_rtl.py` | `(ElementwiseBinary_rtl, elementwise_binary.ElementwiseAdd)` | yes |
| `ElementwiseSub_rtl` | `elementwise_binary_rtl.py` | `(ElementwiseBinary_rtl, elementwise_binary.ElementwiseSub)` | yes |
| `ElementwiseMul_rtl` | `elementwise_binary_rtl.py` | `(ElementwiseBinary_rtl, elementwise_binary.ElementwiseMul)` | yes |

## Redesign pressure

The agnostic op core (shape/dtype/folding/broadcast/execute) is genuinely clean and well-factored, but the backend layer is where the 2-axis HLS/RTL abstraction breaks down. Three forces resist it: (1) the parameter/const-input datapath — a const operand can be embedded (HLS array), decoupled-streamed via a memstream, or MLO-streamed — and this cuts ACROSS the HLS/RTL split while forcing the shared base class (HWCustomOp.generate_hdl_memstream/fetch_weights) to hard-code 'startswith(Elementwise)' branches, i.e. the op family leaks its name into the base contract. (2) Broadcasting: the folded-shape/stream-width special-casing for broadcast axes and the offline PE-lane replication couple the agnostic folding contract to backend buffer/memblock generation, so backends re-derive it rather than consume a clean interface. (3) code_generation_ipi block-design assembly and datatype-width validation are duplicated between hls and rtl with subtle divergences (float16 support, index bugs, clk2x stopgaps), indicating the current backend contracts (4 HLS + 3 RTL abstract methods) do not capture the real variability — a memory-mode / parameter-source axis and a block-design-assembly seam are missing from the abstraction.

## Overrides (26)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | ElementwiseBinaryOperation | 67 | Adds lhs/rhs/out dtype+shape+style, PE, ram_style, mem_mode, and overrides inFIFODepths/outFIFODepths defaults to [2,2]/[2] because the op has TWO inputs. |
| `get_input_datatype` | ElementwiseBinaryOperation | 237 | Two distinct input datatypes (lhs_dtype, rhs_dtype) indexed by ind. |
| `get_output_datatype` | ElementwiseBinaryOperation | 242 | Single out_dtype attribute, ignores ind. |
| `get_normal_input_shape` | ElementwiseBinaryOperation | 247 | Two independent input shapes (lhs_shape, rhs_shape) selected by ind. |
| `get_normal_output_shape` | ElementwiseBinaryOperation | 252 | out_shape is the numpy broadcast of the two inputs. |
| `get_folded_input_shape` | ElementwiseBinaryOperation | 257 | Folding along last axis by PE, but special-cases broadcast axes: if last axis is broadcast (size 1) it returns a dummy (1, num_elems) shape instead of folding. |
| `get_folded_output_shape` | ElementwiseBinaryOperation | 271 | Standard PE folding on last axis. |
| `get_instream_width` | ElementwiseBinaryOperation | 285 | Multiplies elems*bits, but when last axis is broadcast (elems==1) it re-applies PE (elems*=pe) so the stream word is PE-wide even for a broadcast scalar operand. |
| `get_outstream_width` | ElementwiseBinaryOperation | 298 | elems*out_bits from folded output. |
| `make_shape_compatible_op` | ElementwiseBinaryOperation | 170 | Asserts stored lhs/rhs/out shapes match model and broadcast, then emits a generic ONNX 'Add' as a stand-in for shape inference regardless of the actual op. |
| `execute_node` | ElementwiseBinaryOperation | 212 | Pure-numpy functional sim via self.npy_op; casts integer inputs to int64 and forces output to float32. |
| `get_exp_cycles` | ElementwiseBinaryOperation | 429 | prod of folded output shape minus PE dim. |
| `generate_params` | ElementwiseBinaryOperation_hls | 150 | Emits either embedded C++ array init (numpy_to_hls_code + BIND_STORAGE/ARRAY_PARTITION pragmas) OR a memblock.dat hex stream for decoupled/MLO; offline-broadcasts const to PE lanes and left-pads shape to output rank. |
| `generate_params` | ElementwiseBinary_rtl | 467 | Only handles rhs const via make_weight_file into decoupled_npy + decoupled_verilog_dat. |
| `get_verilog_top_module_intf_names` | ElementwiseBinaryOperation_hls | 665 | Conditionally exposes in0_V/in1_V AXIS ports depending on lhs_style/rhs_style and MLO, because the HLS blackbox interface is dynamically generated. |
| `get_verilog_top_module_intf_names` | ElementwiseBinary_rtl | 203 | Same conditional interface logic re-implemented for the RTL path (MLO/input rhs). |
| `execute_node` | ElementwiseBinaryOperation_hls | 818 | cppsim delegates to HLSBackend; rtlsim fully re-implemented because dynamically generated HLS produces variable interfaces (in0/in1 optional). |
| `execute_node` | ElementwiseBinary_rtl | 393 | rtlsim path broadcasts const operands to output shape, folds, feeds only streamed operands; non-rtlsim falls back to agnostic base numpy execute. |
| `code_generation_ipi` | ElementwiseBinaryOperation_hls | 690 | When exactly one side is decoupled (XOR) hand-builds a BD hierarchy with a memstream streamer; otherwise defers to base. |
| `code_generation_ipi` | ElementwiseBinary_rtl | 236 | Full manual BD hierarchy: both-input direct wiring vs const/MLO memstream wrapper, optional axilite for runtime-writeable weights. |
| `generate_hdl` | ElementwiseBinary_rtl | 73 | RTLBackend abstract; validates dtype constraints against RTL param assumptions and template-fills eltwise_template.v, copies 5 .sv files. |
| `get_rtl_file_list` | ElementwiseBinary_rtl | 185 | Returns the 5 finn-rtllib/eltwise .sv sources plus the generated top .v. |
| `get_nodeattr_types` | ElementwiseBinaryOperation_hls | 61 | Merges Elementwise + HLSBackend attrs. |
| `get_nodeattr_types` | ElementwiseBinary_rtl | 35 | Merges attrs and FORCES mem_mode default to internal_decoupled and adds runtime_writeable_weights. |
| `get_nodeattr_types` | ElementwiseBitShift_hls | 1070 | Must re-resolve BitShift attrs because MRO would otherwise pick the wrong get_nodeattr_types under multiple inheritance (drops the 'direction' attr). |
| `minimize_weight_bit_width` | ElementwiseMax (agnostic) | 926 | Overrides to SKIP minimization when either side is FLOAT16/32 to avoid half-vs-ap_int comparison type incompatibility. |

## Hacks (24 — 3 blocker, 12 major)

- **[blocker/base-class-leak]** `hwcustomop.py:311` — HWCustomOp.generate_hdl_memstream branches on op_type.startswith('Elementwise'). The elementwise family RELIES on this leak: both hls (line 141) and rtl (line 177) call self.generate_hdl_memstream(fpgapart). The base op contract hard-codes knowledge of this concrete op family by name.
- **[blocker/base-class-leak]** `hwcustomop.py:359` — generate_hdl_fetch_weights ALSO branches on op_type.startswith('Elementwise') with a dedicated else-branch computing mw=1, mh=rhs_shape[-1], simd=1, n_reps=prod(rhs_shape[:-1]) plus a 'TODO use broadcast rhs shape here' at line 375. Elementwise-specific weight-fetch logic living inside the shared base class.
- **[blocker/inheritance-irregularity]** `elementwise_binary_hls.py:1070` — ElementwiseBitShift_hls must override get_nodeattr_types to explicitly call elementwise_binary.ElementwiseBitShift.get_nodeattr_types instead of ElementwiseBinaryOperation's, because the diamond MRO (ElementwiseBinaryOperation_hls, ElementwiseBitShift) would otherwise resolve to the wrong parent and drop the 'direction' attribute. A latent MRO trap for any op that adds attrs.
- **[major/duplicated-logic]** `elementwise_binary_rtl.py:53` — adapt_for_loop_body is byte-for-byte duplicated between the HLS variant (hls file line 80) and the RTL variant (rtl file line 53) instead of living on the agnostic base. Same for the const/decoupled boolean derivation pattern repeated throughout both files.
- **[major/duplicated-logic]** `elementwise_binary_rtl.py:236` — code_generation_ipi memstream-wiring block (memstream wrapper discovery via os.listdir for '*_memstream_wrapper.v', add_files, create_bd_cell, clk/rst/clk2x wiring) is duplicated between rtl (line 300-364) and hls (line 731-794) with only minor differences.
- **[major/template-surgery]** `elementwise_binary_rtl.py:167` — generate_hdl does naive str.replace of $KEY$ tokens over eltwise_template.v for 13 params. No escaping; a param value containing '$' or a key that is a prefix of another would corrupt output. Relies on exact placeholder spelling in the .v template.
- **[major/magic-number]** `elementwise_binary_rtl.py:116` — Int MUL width cap hardcoded: max_w = 24 if signed else 23 ('DSP58 capacity'). Device-specific DSP58 assumption baked into a Python assert; will silently mis-validate on non-DSP58 fabrics.
- **[major/hard-coded-param]** `elementwise_binary_rtl.py:126` — O_WIDTH derivation hardcoded to mirror the RTL: MUL->2*width, else->width+1, float->32. This duplicates the datatype-derivation logic that _derive_out_dtype already computes in the agnostic classes; the two must be kept in lockstep manually.
- **[major/hard-coded-param]** `elementwise_binary_rtl.py:97` — Float path only supports FLOAT32 (lhs_float = dtype==FLOAT32); asserts float RHS/LHS must be FLOAT32 and output must be exactly 32-bit. No FLOAT16 support in RTL despite agnostic AbsDiff/Max handling FLOAT16.
- **[major/hard-coded-param]** `elementwise_binary_rtl.py:179` — sv_files list ['eltwise.sv','binopf.sv','binopi.sv','int_to_fp32.sv','queue.sv'] hardcoded and duplicated with get_rtl_file_list (line 194). Two sources of truth for the RTL file set.
- **[major/brittle-assumption]** `elementwise_binary_rtl.py:308` — Memstream wrapper filename discovered by scanning code_gen_dir for any file ending '_memstream_wrapper.v' and taking the LAST match (loop does not break). If multiple exist, silently picks last. HLS variant (hls line 736) has the same non-breaking loop.
- **[major/cross-backend-leak]** `elementwise_binary.py:44` — The agnostic base _operation tuple carries an RTL template slot (Identifier, npy, C++, RTL) and exposes rtl_op/cpp_op properties. Backend-specific code strings (C++ '({0} + {1})' and RTL '"ADD"') live in the backend-agnostic definition file, mixing all three backends' concerns in one tuple.
- **[major/todo-marker]** `elementwise_binary.py:231` — execute_node forces output cast to np.float32 with commented-out correct cast and 'TODO: Apparently it is not? Verify this behavior...' — the intended out_dtype cast is disabled, boolean/logical op outputs are floats not the declared BINARY dtype.
- **[major/brittle-assumption]** `elementwise_binary_hls.py:675` — get_verilog_top_module_intf_names: when lhs_style=='input' and rhs const under MLO, it appends in1_V but uses get_instream_width_padded(ind=0) — the LHS width, not ind=1. Likely-wrong width for the rhs interface (copy-paste index bug).
- **[major/cross-backend-leak]** `elementwise_binary.py:918` — ElementwiseMax.cpp_op uses Python %-formatting to inject the HLS datatype name into a C++ ternary ('(%s){0}...' % (odt_hls_name,odt_hls_name)) — a C++ code string built inside the backend-agnostic class, coupling the agnostic op to HLS type syntax.
- **[minor/hard-coded-param]** `elementwise_binary_rtl.py:152` — B_SCALE hardcoded to 1.0 in the code_gen_dict; the eltwise RTL core has a scale parameter that is never actually driven by the op.
- **[minor/cross-backend-leak]** `elementwise_binary_rtl.py:548` — RTL subclasses redefine _operation with a C++ template AND an RTL op name (e.g. Add_rtl sets '({0} + {1})' cpp_op it will never use) purely to satisfy the shared tuple shape, then also implement _get_rtl_op_name returning the same '"ADD"' string. Redundant dual encoding of the op identity.
- **[minor/todo-marker]** `elementwise_binary.py:422` — minimize_weight_bit_width ends with a TODO noting MVAU's return-value convention was copied but makes no sense for two datatypes and the transform ignores the return anyway — logic cargo-culted from MVAU.
- **[minor/todo-marker]** `elementwise_binary.py:658` — ElementwiseDiv marked 'Not tested due to divide by zero from randomly generated inputs...'. Also ElementwiseMod (line 678) and ElementwisePow (lines 897-904, commented out because std::pow/hls::pow fail) are unimplemented/disabled ops.
- **[minor/hard-coded-param]** `elementwise_binary_hls.py:509` — docompute forces the output PE buffer to LUTRAM (impl=LUTRAM) with 'TODO: Maybe reconsider this later?' regardless of the user's ram_style attribute.
- **[minor/brittle-assumption]** `elementwise_binary_hls.py:189` — generate_params offline-broadcasts const to PE lanes via np.broadcast_to with 'TODO: This replicates all parameters and might be inefficient in terms of memory utilization'. Memory-wasteful const replication baked into codegen (repeated for rhs at line 229).
- **[minor/magic-number]** `hwcustomop.py:379` — generate_hdl_fetch_weights (used by Elementwise via the name-leak) hardcodes n_max_layers = 64 'set to 64 for now' as an upper bound on supported layers.
- **[minor/brittle-assumption]** `elementwise_binary_hls.py:770` — code_generation_ipi comment '2x clock is not used for decoupled elementwise ops simply connect input to the 1x clock for now' — ap_clk2x is wired to the 1x clk as a stopgap; the RTL variant does the same at rtl line 338.
- **[minor/duplicated-logic]** `elementwise_binary.py:617` — _derive_out_dtype is re-implemented in ~15 subclasses with near-identical signed/width computation blocks (Add/Sub/AbsDiff/Mul/Div/BitwiseAnd/Or/Xor all repeat the 'signed = any([...])' + width formula). Heavy copy-paste of UG1399 datatype rules.

## Hermeticity violations (10)

- **[env-var]** `elementwise_binary_hls.py:732` — code_generation_ipi reads os.environ['FINN_ROOT'] to locate finn-rtllib/axi/hdl and finn-rtllib/memstream/hdl for the streamer sources.
- **[filesystem-path]** `elementwise_binary_hls.py:736` — os.listdir(code_gen_dir) scans for a '*_memstream_wrapper.v' produced earlier by the base generate_hdl_memstream — order-dependent coupling to a prior codegen step; non-breaking loop takes last match.
- **[env-var]** `elementwise_binary_rtl.py:142` — generate_hdl reads os.environ['FINN_ROOT'] to build rtlsrc path to finn-rtllib/eltwise for template + .sv copy.
- **[env-var]** `elementwise_binary_rtl.py:188` — get_rtl_file_list reads os.environ['FINN_ROOT'] for absolute rtllib paths.
- **[env-var]** `elementwise_binary_rtl.py:303` — code_generation_ipi reads os.environ['FINN_ROOT'] for axi/memstream hdl dirs.
- **[order-dependence]** `elementwise_binary_rtl.py:308` — code_generation_ipi requires generate_hdl_memstream to have already written the wrapper into code_gen_dir; discovers it via os.listdir. Raises if absent.
- **[filesystem-path]** `elementwise_binary_hls.py:265` — generate_params writes memblock.dat directly into code_gen_dir for decoupled/MLO const streaming — implicit file contract consumed later by generate_hdl_memstream.
- **[sibling-op-coupling]** `elementwise_binary_hls.py:40` — Imports LoopBodyInputType from transformation.fpgadataflow.loop_rolling; adapt_for_loop_body mutates lhs_style/rhs_style nodeattrs as a side effect driven by the loop-rolling transform (mutable-state coupling; same in rtl line 17/53).
- **[module-mutable-state]** `elementwise_binary.py:349` — minimize_weight_bit_width silently flips lhs_style/rhs_style nodeattrs to 'const' as a side effect of seeing an initializer (lines 349, 389) and re-annotates model tensors; this mutation drives all downstream codegen branching.
- **[hidden-coupling]** `elementwise_binary_hls.py:109` — _has_embedded_initializer / _check_uram_codegen_support reach into model.get_initializer and gate codegen on is_versal(fpgapart) + Vitis version (get_vivado_version), coupling the op to global toolchain/device state.

## finn-rtllib coupling (8)

- `eltwise/eltwise_template.v` via **string-replace** — rtl/elementwise_binary_rtl.py:165-171 reads eltwise_template.v and does template=template.replace(f'${key}$', str(value)) for a 13-key code_gen_dict (TOP_MODULE_NAME, PE, OP, B_SCALE, A_FLOAT, B_FLOAT, A_WIDTH, A_SIGNED, B_WIDTH, B_SIGNED, A_STREAM_BITS, B_STREAM_BITS, O_STREAM_BITS built at lines 149-163). Writes <top>.v to code_gen_dir. Placeholders live in eltwise_template.v lines 20/24/30/34-45.
- `eltwise/eltwise.sv` via **file-copy** — rtl:179-181 shutil.copy of eltwise.sv into code_gen_dir; instantiated as module 'eltwise' by the filled template (eltwise_template.v line 33). Parameterized purely via the template's #(.PE/.OP/.A_WIDTH/...) param map, no edits to the .sv.
- `eltwise/binopf.sv` via **file-copy** — rtl:179-181 copied verbatim; float binary-op core selected inside eltwise.sv by A_FLOAT/B_FLOAT params. Listed in get_rtl_file_list line 197.
- `eltwise/binopi.sv` via **file-copy** — rtl:179-181 copied verbatim; integer binary-op core, OP param ('ADD'/'SUB'/'MUL' from _get_rtl_op_name) selects operation. Listed line 198.
- `eltwise/int_to_fp32.sv` via **file-copy** — rtl:179-181 copied verbatim; int->float32 converter for mixed int/float datapaths. Listed line 199.
- `eltwise/queue.sv` via **file-copy** — rtl:179-181 copied verbatim; skid/queue buffer. Listed line 200.
- `memstream/hdl/memstream.sv + memstream_axi.sv + memstream_wrapper_template.v` via **tcl-instantiate** — For const/MLO rhs, the RTL op relies on base HWCustomOp.generate_hdl_memstream (hwcustomop.py:307, gated on op_type.startswith('Elementwise')) to template-fill memstream_wrapper_template.v; then rtl code_generation_ipi:303-328 add_files the memstream sources and create_bd_cell the wrapper, wiring m_axis_0 -> core in1_V (line 349). memblock.dat init produced by generate_params/make_weight_file (rtl:467-524).
- `axi/hdl/axilite.sv` via **tcl-instantiate** — rtl code_generation_ipi:303/316-323 add_files axilite.sv when runtime_writeable_weights==1 to expose an aximm/axilite slave for runtime weight update (lines 352-363).

## Seams (6)

- **clean-seam** — The agnostic ElementwiseBinaryOperation cleanly owns shape/datatype/folding/execute contract and the _derive_out_dtype specializations; both HLS and RTL inherit it and only add backend codegen. Adding a third backend that reuses these is structurally straightforward.
- **clean-seam** — The 18 op-identity subclasses (Add/Sub/Mul/...) are near-pure data: they only set _operation and _derive_out_dtype. Op identity is well-factored away from backend mechanics.
- **fused-no-seam** — generate_hdl_memstream / generate_hdl_fetch_weights live in HWCustomOp and dispatch on op_type.startswith('Elementwise') (hwcustomop.py:311,359). The memstream/MLO datapath CANNOT be separated from the base without either subclass-registering or moving this logic — the base is fused to this op family by name.
- **fused-no-seam** — In elementwise_binary_hls.py docompute (lines 378-566), the broadcast semantics, PE unpacking, buffer indexing, stream-read conditions and the C++ op template (self.cpp_op.format) are interleaved as one giant f-string generator. The agnostic broadcast logic and HLS C++ emission are inseparable here.
- **fused-no-seam** — RTL generate_hdl re-derives output width and validates dtype/signedness/width against RTL core assumptions (rtl:102-134). This datatype logic is a shadow copy of the agnostic _derive_out_dtype; the RTL backend and the datatype contract are coupled and must move together.
- **fused-no-seam** — code_generation_ipi in both backends hand-builds the Vivado block design (hierarchy, pins, memstream streamer, clk2x stopgap). This BD-assembly is not expressible through the RTLBackend/HLSBackend contract seams and is bespoke per backend with heavy duplication.
