# Census: iodma (backend-only)

*IODMA_hls is a single HLS-only, backend-only class (inherits HWCustomOp+HLSBackend directly, no agnostic base, no RTL sibling) implementing an AXI-MM <-> AXI-stream data mover whose every method bifurcates on a 'direction' attribute and whose HLS codegen is hand-rolled printf-style template surgery over hard-coded finn-hlslib function names. It has no functional execution model (execute_node is pass) and deliberately violates the folded-shape contract on the memory-mapped side.*

**Files:** `src/finn/custom_op/fpgadataflow/hls/iodma_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `IODMA_hls` | `iodma_hls.py` | `(HWCustomOp, HLSBackend)` | NO (backend-only) |

## Redesign pressure

IODMA is a backend-only op with no HLS/RTL split and no shared agnostic base, yet it is really a hardware infrastructure primitive rather than a compute op — so it fits the 2-axis (op-agnostic base + HLS/RTL backend) model poorly from the start. Its single class is bifurcated end-to-end by the 'direction' attribute (in vs out): folded-shape methods, instream/outstream widths, docompute, blackboxfunction, pragmas, and interface names each branch into two opposite dataflow roles, and the folded-shape methods RAISE rather than answer on the AXI-MM side, breaking the uniform shape contract. execute_node is a no-op, so there is no functional model to share between backends. The width-conversion (LCM DWC-chaining) logic is genuinely backend-agnostic in intent but is implemented entirely as finn-hlslib template-string surgery, meaning any RTL variant would have to reimplement it wholesale. A redesign would want direction split into two distinct op roles and the shape/width semantics lifted into an agnostic base, leaving only the datawidth-converter instantiation as the backend-specific seam.

## Overrides (19)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | IODMA_hls | 83 | Adds DMA primitive attrs (intfWidth, streamWidth, burstMode wrap/increment, direction in/out, intfName, NumChannels); merges HWCustomOp+HLSBackend attr dicts. |
| `get_normal_input_shape` | IODMA_hls | 106 | Shape = numInputVectors + [NumChannels]; DMA moves a flat tensor. |
| `get_normal_output_shape` | IODMA_hls | 112 | Output shape identical to input (pure data mover). |
| `get_folded_input_shape` | IODMA_hls | 115 | Folding only defined on the AXI-stream side; folds NumChannels by streamWidth/datatype-bits. |
| `get_folded_output_shape` | IODMA_hls | 130 | Folding only defined on the AXI-stream side. |
| `infer_node_datatype` | IODMA_hls | 145 | Propagates input tensor DataType to output and stores into dataType attr; warns on change. |
| `get_input_datatype` | IODMA_hls | 158 | Reads dataType attr string. |
| `get_output_datatype` | IODMA_hls | 162 | Same as input datatype (transparent mover). |
| `get_instream_width` | IODMA_hls | 166 | Width depends on direction: 'in' -> intfWidth (AXI-MM side), 'out' -> streamWidth (AXI-stream side). |
| `get_outstream_width` | IODMA_hls | 174 | Mirror of get_instream_width: 'out'->intfWidth, 'in'->streamWidth. |
| `get_number_output_values` | IODMA_hls | 182 | Total tensor bits / streamWidth; asserts word-multiple divisibility. |
| `global_includes` | IODMA_hls | 192 | Pulls finn-hlslib dma.h and streamtools.h. |
| `defines` | IODMA_hls | 196 | Emits NumBytes1/DataWidth1 macros for the DMA template; asserts byte-aligned total bits. |
| `get_ap_int_max_w` | IODMA_hls | 207 | Max ap_int width = LCM(instream,outstream) because width conversion goes through the least-common-multiple width. |
| `docompute` | IODMA_hls | 214 | Instantiates the DMA + up to two StreamingDataWidthConverter_Batch cores, choosing 0/1/2-DWC topology by divisibility of intfWidth vs streamWidth, per direction and burstMode. |
| `blackboxfunction` | IODMA_hls | 322 | Signature differs by direction: 'in' takes ap_uint* in0 + hls::stream out; 'out' takes hls::stream in + ap_uint* out. |
| `pragmas` | IODMA_hls | 349 | Emits m_axi/axis/s_axilite INTERFACE pragmas keyed on direction and intfName, plus DATAFLOW. |
| `execute_node` | IODMA_hls | 388 | No-op: DMA is pure data movement with no functional transform. |
| `get_verilog_top_module_intf_names` | IODMA_hls | 391 | Adds axilite s_axi_control and aximm (m_axi_gmem, intfWidth); blanks out the streaming side that base added (m_axis or s_axis) depending on direction. |

## Hacks (9 — 0 blocker, 5 major)

- **[major/brittle-assumption]** `iodma_hls.py:116` — get_folded_input_shape RAISES ValueError when direction=='in' (and get_folded_output_shape raises when direction=='out', line 131). The op deliberately violates the folded-shape contract on the AXI-MM side. Any generic pass that calls get_folded_input_shape/get_folded_output_shape uniformly across nodes will crash on an IODMA in the wrong direction.
- **[major/template-surgery]** `iodma_hls.py:228` — dma_inst_template and dwc_inst_template are hand-built printf-style format strings ('func<DataWidth1, NumBytes1>(%s, %s, numReps);' etc.) then filled across a 100-line direction x width-divisibility branch tree (lines 239-318). Stream names (dma2dwc, dma2lcm, lcm2out, dwc2dma, in2lcm, lcm2dma) and template widths are string-interpolated by hand; NumBytes1/DataWidth1 must match the macros emitted in defines() by naming convention only.
- **[major/hard-coded-param]** `iodma_hls.py:217` — finn-hlslib function names hard-coded as strings: 'StreamingDataWidthConverter_Batch' (217), 'Mem2Stream_Batch_external_wmem' (220), 'Mem2Stream_Batch' (222), 'Stream2Mem_Batch' (224). No indirection/registry; a rename in finn-hlslib silently breaks codegen.
- **[major/inheritance-irregularity]** `iodma_hls.py:394` — get_verilog_top_module_intf_names calls super() then overwrites intf_names['m_axis']=[] or ['s_axis']=[] to remove the streaming port the base auto-populated, because for a DMA one side is AXI-MM not a FINN stream. Corrects/undoes base behavior rather than the base being direction-aware.
- **[major/other]** `iodma_hls.py:388` — execute_node is a bare 'pass' — IODMA has no functional/simulation model. Node execution silently produces nothing, so any cppsim/rtlsim data-verification flow through an IODMA node is a no-op; correctness cannot be observed at the op level.
- **[minor/magic-number]** `iodma_hls.py:398` — Interface names hard-coded: 'm_axi_gmem' (398), 's_axi_control' (397). These string literals are the contract other stitching/driver-gen passes rely on; not derived from any attr.
- **[minor/duplicated-logic]** `iodma_hls.py:130` — get_folded_output_shape (130-143) is a near-verbatim copy of get_folded_input_shape (115-128). The copy even carries the wrong assertion message 'Input stream width must be a multiple of datatype bits' at line 137 inside the OUTPUT method (copy-paste bug).
- **[minor/brittle-assumption]** `iodma_hls.py:121` — Local variable named 'intfw' is assigned from streamWidth (get_nodeattr('streamWidth')) in both folded-shape methods (lines 121 and 136). Misleading naming (intfw normally = intfWidth) invites confusion between the two width axes that this whole op hinges on.
- **[minor/hard-coded-param]** `iodma_hls.py:193` — global_includes hard-codes 'dma.h' and 'streamtools.h' finn-hlslib headers with no version/path abstraction.

## Hermeticity violations (1)

- **[other]** `iodma_hls.py:154` — infer_node_datatype emits warnings.warn on datatype change (side-effecting global warning state) while also mutating the 'dataType' node attr and the model output tensor datatype in the same call.

## Seams (3)

- **fused-no-seam** — IODMA_hls inherits (HWCustomOp, HLSBackend) directly with NO shared agnostic base and NO rtl sibling. All op-agnostic semantics (shapes, direction-dependent instream/outstream widths, datatype propagation, number_output_values) live in the SAME class as the HLS codegen (global_includes/defines/docompute/blackboxfunction/pragmas/get_ap_int_max_w). There is no seam at which an RTL backend could be attached without duplicating every shape/width method.
- **fused-no-seam** — The width-conversion topology decision (0/1/2 StreamingDataWidthConverter_Batch cores via LCM chaining) is expressed only as HLS template strings inside docompute (214-320). This logic is conceptually backend-agnostic (it's about intfWidth vs streamWidth divisibility) but is welded to finn-hlslib function names and hls::stream declarations, so it cannot be reused by any other backend.
- **clean-seam** — The pure HLS-contract methods (global_includes, defines, docompute, blackboxfunction, pragmas) are cleanly grouped and would be the natural HLS-backend surface IF an agnostic base existed to host the shape/width methods.
