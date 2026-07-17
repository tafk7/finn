# Census: checksum (backend-only, HLS-only)

*CheckSum_hls is a compact HLS-only, backend-only op (no agnostic base, no RTL variant, no finn-rtllib coupling) that wraps custom_hls/checksum.hpp. Its distinguishing complexity is a non-standard two-output/scalar-plus-AXILite-control shape that the single-stream HWCustomOp contract does not model, forcing a 32-bit checksum width and drain signal to be hard-coded across many codegen methods.*

**Files:** `src/finn/custom_op/fpgadataflow/hls/checksum_hls.py`, `custom_hls/checksum.hpp`, `src/finn/custom_op/fpgadataflow/hlsbackend.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `CheckSum_hls` | `checksum_hls.py` | `(HWCustomOp, HLSBackend)` | NO (backend-only) |

## Redesign pressure

This family is small and, for a single-backend HLS op, fairly clean — but it stresses the 2-axis HLS/RTL abstraction in two structural ways. First, it is a backend-only diamond (HWCustomOp+HLSBackend) with no agnostic base, forcing a redundant execute_node forwarder and manual union of two attr dicts; a redesign should give even single-backend ops a shared op-semantics layer distinct from the HLS emitter. Second, and more importantly, checksum is a two-output node with a scalar side channel plus an AXI-Lite control register (chk/drain), a shape/interface pattern the core HWCustomOp contract (single stream in, single stream out) does not model — so the second output and control interface are smeared across get_normal_output_shape ind==1, npy_to_dynamic_output, dataoutstrm, pragmas, and get_verilog_top_module_intf_names, with a 32-bit width baked in as a magic number in four places. The abstraction has no concept of scalar/control ports or multi-output nodes, which is the main thing this op fights.

## Overrides (22)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | CheckSum_hls | 44 | Adds checksum-specific attrs (words_per_frame, items_per_word, inputDataType, folded_shape) then merges BOTH HWCustomOp and HLSBackend attr dicts (lines 55-56). |
| `get_input_datatype` | CheckSum_hls | 74 | Reads inputDataType nodeattr. |
| `get_output_datatype` | CheckSum_hls | 78 | Output dtype forced identical to input dtype (checksum passes data through unchanged; the scalar checksum is a side channel). |
| `get_instream_width` | CheckSum_hls | 83 | in_width = folded_shape[-1] * dtype.bitwidth(); width derived from innermost folded dim only. |
| `get_outstream_width` | CheckSum_hls | 89 | Pure pass-through node so outstream width == instream width. |
| `get_folded_input_shape` | CheckSum_hls | 92 | Returns the folded_shape nodeattr verbatim. |
| `get_folded_output_shape` | CheckSum_hls | 95 | Returns folded_shape nodeattr verbatim (pass-through). |
| `get_normal_input_shape` | CheckSum_hls | 98 | Reconstructs normal shape by multiplying the two innermost folded dims (folded_shape[-2]*folded_shape[-1]); checksum is inserted between dataflow nodes so it only knows its folded shape. |
| `get_normal_output_shape` | CheckSum_hls | 124 | Two-output node: ind==0 mirrors input shape, ind==1 returns tuple([1]) for the scalar checksum register. |
| `get_ap_int_max_w` | CheckSum_hls | 121 | max(super(), 32) to guarantee AP_INT_MAX_W covers the hard-coded 32-bit checksum accumulator even when data streams are narrower. |
| `npy_to_dynamic_output` | CheckSum_hls | 134 | Calls super() for output_0 then separately loads output_1.npy into context[node.output[1]] to surface the scalar checksum during cppsim. |
| `execute_node` | CheckSum_hls | 141 | Explicitly delegates to HLSBackend.execute_node. |
| `get_verilog_top_module_intf_names` | CheckSum_hls | 244 | Appends axilite interface 's_axi_checksum' so the stitched IP exposes the checksum control register. |
| `global_includes` | CheckSum_hls | 144 | Includes checksum.hpp (repo custom_hls lib). |
| `defines` | CheckSum_hls | 147 | Emits WORDS_PER_FRAME/ITEMS_PER_WORD/WORD_SIZE defines for the checksum<> template. |
| `read_npy_data` | CheckSum_hls | 157 | Custom npy2apintstream call for single input stream in0_V. |
| `strm_decl` | CheckSum_hls | 177 | Declares in0_V, out0_V plus extra ap_uint<32> chk and ap_uint<1> drain=false side signals. |
| `docompute` | CheckSum_hls | 189 | Instantiates checksum<WORDS_PER_FRAME, ITEMS_PER_WORD>(in0_V, out0_V, chk, drain). |
| `dataoutstrm` | CheckSum_hls | 194 | apintstream2npy for output_0 PLUS manual cnpy::npy_save of the scalar chk to output_1.npy. |
| `blackboxfunction` | CheckSum_hls | 223 | Top signature has 4 ports (in0_V, out0_V, chk, drain) instead of the usual 2 streams. |
| `pragmas` | CheckSum_hls | 231 | AXIS on in0_V/out0_V, s_axilite on chk & drain (bundle=checksum), ap_ctrl_none, dataflow + disable_start_propagation. |
| `infer_node_datatype` | CheckSum_hls | 59 | Warns on input dtype change, sets inputDataType, propagates output dtype. |

## Hacks (10 — 0 blocker, 7 major)

- **[major/magic-number]** `checksum_hls.py:122` — get_ap_int_max_w floors the value at literal 32 to cover the checksum accumulator width. The 32 is the checksum register width, duplicated as a magic constant with no shared symbol.
- **[major/magic-number]** `checksum_hls.py:185` — 'ap_uint<32> chk;' hard-codes the 32-bit checksum register width. Same 32 recurs at line 218-220 (std::vector<unsigned int> checksum, cnpy save) and in blackboxfunction/pragmas (chk is ap_uint<32> at line 226). No single source of truth for checksum bitwidth.
- **[major/hard-coded-param]** `checksum_hls.py:187` — 'ap_uint<1> drain = false;' hard-codes drain low with an inline comment 'set drain = false for cppsim'. The drain control has no nodeattr and is fixed off, so cppsim can never exercise the drain path.
- **[major/hard-coded-param]** `checksum_hls.py:247` — axilite interface name 's_axi_checksum' is hard-coded in get_verilog_top_module_intf_names and must stay in sync with the 'bundle=checksum' s_axilite pragmas at lines 235,238. Two string literals coupled by convention only.
- **[major/hard-coded-param]** `checksum_hls.py:138` — npy_to_dynamic_output reads a fixed filename 'output_1.npy' and dataoutstrm writes the matching 'output_1.npy' at line 220. The second-output plumbing is coupled by these bare filename literals, not by a shared helper.
- **[major/brittle-assumption]** `checksum_hls.py:112` — get_normal_input_shape assumes folded_shape has at least 2 dims and that exactly the two innermost dims encode the folding (folded_shape[-2]*folded_shape[-1]). No validation; a 1-D folded_shape would IndexError.
- **[major/brittle-assumption]** `checksum_hls.py:130` — get_normal_output_shape treats ind==1 as a scalar (tuple([1])) second output; the two-output contract is unique to this op and not modeled anywhere in HWCustomOp, so downstream shape/datatype inference for output[1] is unmanaged by the base machinery.
- **[minor/duplicated-logic]** `checksum_hls.py:157` — read_npy_data / strm_decl / dataoutstrm are near-verbatim copies of the generic HLSBackend seam bodies with minor edits (single stream, extra chk/drain). Copy-paste of boilerplate rather than reuse of the base implementations.
- **[minor/inheritance-irregularity]** `checksum_hls.py:141` — execute_node override exists solely to forward to HLSBackend.execute_node, disambiguating the (HWCustomOp, HLSBackend) diamond. A no-op indirection that only exists because the op is backend-only with two bases.
- **[minor/other]** `checksum_hls.py:90` — No make_shape_compatible_op override despite being a two-output node; relies on the base default which is written for single-output ops. Potential mismatch for output[1].

## Hermeticity violations (3)

- **[filesystem-path]** `checksum_hls.py:145` — global_includes references checksum.hpp which lives in the repo at custom_hls/checksum.hpp; compilation depends on that include path being injected by external build config (ambient include-dir coupling).
- **[filesystem-path]** `checksum_hls.py:138` — npy_to_dynamic_output hard-codes '{code_gen_dir}/output_1.npy' load; couples Python execution to a file the generated C++ (dataoutstrm line 220) must have written first (order-dependence between codegen and execute).
- **[filesystem-path]** `checksum_hls.py:220` — dataoutstrm emits cnpy::npy_save to a literal 'output_1.npy' path in code_gen_dir; the second-output side channel is filesystem-mediated rather than through the stream contract.

## Seams (3)

- **clean-seam** — The 4 HLSBackend abstract methods (global_includes/defines/docompute/blackboxfunction) plus the read_npy_data/strm_decl/dataoutstrm/pragmas seams are all implemented in this single class and cleanly generate the HLS wrapper around custom_hls/checksum.hpp. If an RTL variant were desired, the datatype/shape/width contract methods (get_*_datatype, get_*_shape, get_*stream_width) are backend-agnostic and could be lifted into a shared base.
- **fused-no-seam** — There is NO agnostic base and NO RTL variant. The op is a single (HWCustomOp, HLSBackend) diamond class; every shape/datatype method is co-located with HLS codegen in one file, so backend logic and op semantics are fused. The 32-bit checksum register, drain signal, and second scalar output are expressed only in HLS-specific codegen (strm_decl/dataoutstrm/blackboxfunction/pragmas), with no abstraction separating what-it-computes from how-HLS-emits-it.
- **fused-no-seam** — The second (scalar checksum) output and the axilite chk/drain control interface are encoded across get_normal_output_shape (ind==1), npy_to_dynamic_output, dataoutstrm, pragmas, and get_verilog_top_module_intf_names simultaneously. This cross-cutting feature cannot be swapped or relocated without touching all five methods; it is not captured by any base-class concept.
