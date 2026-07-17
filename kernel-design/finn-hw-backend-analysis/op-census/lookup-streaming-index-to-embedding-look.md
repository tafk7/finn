# Census: lookup (streaming index-to-embedding lookup)

*A two-input streaming index-to-embedding lookup with a clean agnostic base (Lookup) and a single HLS backend (Lookup_hls); no RTL variant exists. Its defining complication is a mem_mode axis (internal_embedded ROM vs external AXI-MM DMA) that cross-cuts the entire contract and is fused into both classes via pervasive branching, plus device/tool-version and dtype hard-constraints in the external/URAM paths.*

**Files:** `src/finn/custom_op/fpgadataflow/lookup.py`, `src/finn/custom_op/fpgadataflow/hls/lookup_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Lookup` | `lookup.py` | `(HWCustomOp)` | yes |
| `Lookup_hls` | `lookup_hls.py` | `(Lookup, HLSBackend)` | yes |

## Redesign pressure

The lookup family fits the 2-parent HLS/RTL pattern cleanly on the surface (agnostic Lookup base + Lookup_hls), and there is no RTL variant so the HLS/RTL axis is trivial here. The real pressure is a THIRD orthogonal axis the current abstraction does not model: mem_mode (internal_embedded ROM/BRAM/URAM/LUTRAM vs external AXI-MM DMA). This axis re-derives folded shapes, stream widths, top-level interface names, resource estimates, and every single HLS codegen method via if/elif branches, and the two modes are effectively distinct hardware ops (a baked-in ROM lookup vs a runtime DMA engine with out-of-bounds IRQ). A redesign would want mem_mode as a first-class backend/memory-strategy dimension rather than pervasive intra-method branching. Secondary pressure: execute_node embeds an onnxruntime Gather golden model with hardcoded INT64/FLOAT/opset-13 in the base, and codegen has a device/tool-version guard (Versal + Vivado 2024.2 for URAM) that leaks environment concerns into the op.

## Overrides (25)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | Lookup | 47 | Adds lookup-specific attrs: NumEmbeddings, EmbeddingDim, EmbeddingType, InputType, InputShape, mem_mode(internal_embedded/external), ram_style, ext_mem_width. |
| `get_exp_cycles` | Lookup | 76 | One cycle per input index (throughput = number of lookups). |
| `get_normal_input_shape` | Lookup | 81 | Two inputs: ind0 = index tensor InputShape, ind1 = embedding table (NumEmbeddings, EmbeddingDim). |
| `get_normal_output_shape` | Lookup | 89 | Appends EmbeddingDim to input shape (each index expands to an embedding vector). |
| `get_folded_input_shape` | Lookup | 95 | For ind0 appends trailing 1 (single index per cycle); ind1 returns normal shape. |
| `get_folded_output_shape` | Lookup | 103 | internal_embedded: whole EmbeddingDim per beat. external: splits EmbeddingDim into (emb_dim/elems_per_word, elems_per_word) for AXI-MM word packing. |
| `get_input_datatype` | Lookup | 136 | ind0 = InputType (index dtype), ind1 = EmbeddingType (table dtype). |
| `get_output_datatype` | Lookup | 145 | Output dtype is always EmbeddingType regardless of ind. |
| `get_instream_width` | Lookup | 149 | ind0 = index bitwidth; ind1 = 0 for internal_embedded (baked-in), ext_mem_width for external. |
| `get_outstream_width` | Lookup | 161 | obits * folded_oshape[-1]; for external this is one AXI word, for internal the full embedding. |
| `execute_node` | Lookup | 166 | Python/ONNX-runtime golden: builds a one-node Gather graph (opset 13) and runs onnxruntime to compute the lookup. |
| `bram_estimation` | Lookup | 197 | Only estimates for internal_embedded auto/block ram_style; width/16 x depth/1024 BRAM18 tiling. |
| `uram_estimation` | Lookup | 208 | internal_embedded + ultra ram_style; width/72 x depth/4096. |
| `lut_estimation` | Lookup | 238 | internal_embedded + distributed ram_style: width * ceil(depth/64) LUTRAM. |
| `get_verilog_top_module_intf_names` | Lookup | 248 | external mem_mode adds axilite s_axi_control, aximm (m_axi_gmem, ext_mem_width), and ap_none oob_irq interfaces. |
| `get_nodeattr_types` | Lookup_hls | 47 | Merges Lookup + HLSBackend attrs. |
| `global_includes` | Lookup_hls | 75 | Always includes lookup.hpp; adds embeddings.hpp only for internal_embedded (generated param header). |
| `defines` | Lookup_hls | 83 | Emits distinct macro sets per mem_mode (external: MemBits/EmbeddingSize/EmbeddingAlign/T_SRC/T_DST; internal: NumEmbeddings/EmbeddingDim/InputType/EmbeddingType). |
| `dataoutstrm` | Lookup_hls | 109 | apintstream2npy with folded output shape; BIPOLAR->BINARY storage swap. |
| `docompute` | Lookup_hls | 135 | internal: StreamingLookup<...> template; external: StreamingLookup_ext<EmbeddingSize> with mem/size/oob_count/oob_irq args. |
| `blackboxfunction` | Lookup_hls | 148 | internal: (in0_V,out0_V); external: adds T_DST *mem, unsigned size, unsigned &oob_count, bool &oob_irq to signature. |
| `pragmas` | Lookup_hls | 172 | internal: BIND_STORAGE on embeddings (RAM_S2P/URAM vs ROM_2P for others); external: m_axi + s_axilite bundles + ap_none oob_irq. |
| `generate_params` | Lookup_hls | 194 | internal: writes embeddings.hpp via numpy_to_hls_code with innermost-dim flip; external: zero-pads for burst alignment and writes packed hex .dat. |
| `execute_node` | Lookup_hls | 242 | Asserts internal_embedded only, then delegates to HLSBackend.execute_node. |
| `get_ap_int_max_w` | Lookup_hls | 249 | external mode must account for ext_mem_width in max ap_int width. |

## Hacks (13 — 0 blocker, 6 major)

- **[major/brittle-assumption]** `lookup_hls.py:216` — external mem_mode hard-asserts edt.bitwidth()==8; any non-8-bit embedding in external mode is unsupported. Silently constrains the whole external path to byte embeddings.
- **[major/brittle-assumption]** `lookup.py:174` — execute_node hardcodes TensorProto.INT64 for indices and TensorProto.FLOAT for data/output, and opset 13 (line 189). Golden model correctness depends on these fixed types matching whatever the node actually carries; index dtypes other than those representable are coerced.
- **[major/other]** `lookup.py:166` — execute_node builds and runs a full throwaway ONNX Gather model through onnxruntime (rt.InferenceSession) for every node execution instead of a numpy gather. Heavy, adds onnxruntime import dependency, and reshapes result assuming np.float32 output.
- **[major/hard-coded-param]** `lookup_hls.py:64` — _check_uram_codegen_support hardcodes Vivado version gate (2024, 2) for URAM-backed internal embeddings; brittle version tuple comparison, passes if get_vivado_version() returns None.
- **[major/brittle-assumption]** `lookup_hls.py:208` — generate_params flips innermost dim of embeddings (np.flip(embeddings,-1)) 'to remain compatible with how we normally encode the data in FINN' -- an implicit ordering convention that must be matched by the HLS lookup.hpp reader; silent coupling between Python packing and C++ template.
- **[major/other]** `lookup_hls.py:80` — global_includes emits '#include "embeddings.hpp"' only for internal_embedded, which is exactly the file generate_params writes (line 199). Codegen correctness silently depends on generate_params having run first and written that header.
- **[minor/magic-number]** `lookup.py:202` — BRAM estimation hardcodes 16 (BRAM18 width) and 1024 (depth) tiling constants; line 213-214 URAM hardcodes 72 and 4096; line 244 LUTRAM hardcodes 64. Device-specific magic constants embedded in Python.
- **[minor/magic-number]** `lookup.py:224` — bram_efficiency_estimation uses 18*1024 bits/BRAM and uram_efficiency_estimation (line 235) uses 72*4096; capacity magic numbers duplicated from the estimation constants.
- **[minor/todo-marker]** `lookup.py:205` — TODO 'can we estimate BRAMs for the DMA engine?' (external path); mirrored TODO at line 216 for URAM and multi-line TODO at 228-230 about Versal URAM flexible bit widths. Resource estimation for external/DMA mode is unimplemented (returns 0).
- **[minor/template-surgery]** `lookup_hls.py:179` — pragmas() derives HLS storage type by string test: storage_type = 'RAM_S2P' if ram_style=='URAM' else 'ROM_2P', where ram_style is itself a translated string from RAM_STYLES dict (line 38). Two-hop string mapping (attr -> RAM_STYLES -> storage_type) with no enum.
- **[minor/duplicated-logic]** `lookup_hls.py:226` — external param packing recomputes ext_mem_emb_align = ceil(log2(ext_mem_emb_size)) (also computed in defines() line 96) and folded-shape-derived sizes, duplicating get_folded_output_shape logic inline rather than reusing it.
- **[minor/other]** `lookup_hls.py:143` — external docompute references oob_count/oob_irq streaming out-of-bounds handling; oob_irq surfaced as ap_none top-level interface (lookup.py line 254). OOB-detection machinery only exists in external path -- asymmetric feature between the two mem_modes.
- **[minor/other]** `lookup_hls.py:200` — Dead commented-out code in generate_params (obits/packed_output_hls_type at lines 201-202) left in place.

## Hermeticity violations (4)

- **[hidden-coupling]** `lookup_hls.py:63` — _check_uram_codegen_support calls get_vivado_version() (reads installed tool version from environment/PATH) and is_versal(fpgapart); code generation success depends on ambient Vivado install, not just node attrs.
- **[other]** `lookup.py:30` — Module imports onnxruntime (import onnxruntime as rt) purely to run the golden Gather model in execute_node; heavy external runtime dependency loaded at import for a functional-verification path.
- **[filesystem-path]** `lookup_hls.py:199` — generate_params writes embeddings.hpp (internal) or <node.name>.dat (external, line 232) into the codegen dir; global_includes/docompute later assume those files exist. Order-dependent filesystem side effect coupling param generation to codegen.
- **[order-dependence]** `lookup_hls.py:73` — code_generation_ipgen overrides to run _check_uram_codegen_support(fpgapart) before super(); the URAM/Versal/Vivado-version guard only fires on this specific entry point, not on other codegen paths.

## Seams (3)

- **clean-seam** — Lookup (agnostic base) cleanly separates shape/dtype/estimation contract from HLS emission; Lookup_hls adds only the HLSBackend 4 abstract methods + generate_params. A hypothetical Lookup_rtl could inherit Lookup and implement RTLBackend without touching the base. Standard 2-parent FINN pattern is intact here.
- **fused-no-seam** — The mem_mode axis (internal_embedded vs external) is orthogonal to and cross-cuts the HLS/RTL axis: it changes folded shapes, stream widths, interface names (base class, lookup.py 248-255), AND every HLS codegen method (defines/docompute/blackboxfunction/pragmas/generate_params). mem_mode==external is effectively a second op (DMA lookup with AXI-MM + OOB IRQ) fused into the same class via branching, not separable from the internal ROM lookup.
- **fused-no-seam** — generate_params (HLS) and global_includes/docompute are fused through the on-disk embeddings.hpp/.dat convention plus the innermost-dim flip; the Python packing order and the C++ lookup.hpp reader are a single implicit contract that cannot be swapped independently.
