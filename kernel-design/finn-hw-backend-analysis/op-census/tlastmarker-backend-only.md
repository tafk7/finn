# Census: tlastmarker (backend-only)

*TLastMarker_hls is a backend-only, HLS-only shell/DMA-support op that inserts or removes AXI-stream TLAST; it inherits HWCustomOp+HLSBackend directly, deliberately stubs out half the datatype/shape contract (raising or passing), and encodes all real behavior as heavily-branched, protocol-forked C++ string templates.*

**Files:** `src/finn/custom_op/fpgadataflow/hls/tlastmarker_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `TLastMarker_hls` | `tlastmarker_hls.py` | `(HWCustomOp, HLSBackend)` | NO (backend-only) |

## Redesign pressure

This op is a backend-only, HLS-only outlier that treats half the HWCustomOp contract as inapplicable: four of the eight abstract getters (datatypes and normal shapes) raise Exception, and infer_node_datatype/make_shape_compatible_op are no-ops, so the op is opaque to FINN's datatype and shape-inference machinery and must be special-cased by callers. It resists the 2-axis HLS/RTL abstraction not because it spans both backends but because it fits neither cleanly: it is a raw AXI-stream sideband (TLAST/TKEEP) manipulator sized by explicit StreamWidth/ElemWidth bit attrs rather than FINN datatypes+folding, and its real logic lives entirely as per-line ternary string surgery selecting between qdma_axis and ap_axiu APIs (with a latent unclosed-brace bug on the Direction=in+external path). A redesign would want a first-class 'stream sideband / shell-interface' node category with a proper AXIS-protocol parameter object, rather than forcing a shell/DMA artifact through the generic tensor-op contract.

## Overrides (21)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | TLastMarker_hls | 45 | Adds AXI-stream-specific attrs (NumIters, DynIters, Direction, StreamWidth, ElemWidth, Protocol) that have no analog in normal dataflow ops; then manually merges both HWCustomOp and HLSBackend attr dicts. |
| `execute_node` | TLastMarker_hls | 65 | TLAST behavior is only visible in stitched rtlsim/hardware, so single-node execution is a pure pass-through (copies input tensor to output). |
| `make_shape_compatible_op` | TLastMarker_hls | 76 | Op has no meaningful shape transform; returns None. |
| `get_number_output_values` | TLastMarker_hls | 208 | Number of output values equals the static NumIters count (one AXI beat per iteration). |
| `get_input_datatype` | TLastMarker_hls | 211 | Op is datatype-agnostic (raw bit stream), no FINN datatype tracked. |
| `get_output_datatype` | TLastMarker_hls | 215 | Same as input: no FINN datatype. |
| `get_normal_input_shape` | TLastMarker_hls | 219 | No normal (unfolded) tensor shape concept for a stream marker. |
| `get_normal_output_shape` | TLastMarker_hls | 223 | Same; no normal shape. |
| `get_folded_input_shape` | TLastMarker_hls | 227 | Folded shape synthesized from StreamWidth/ElemWidth: n_packed_elems = StreamWidth//ElemWidth, over NumIters beats. |
| `get_folded_output_shape` | TLastMarker_hls | 234 | Output stream identical to input stream (marker is transparent). |
| `get_instream_width` | TLastMarker_hls | 237 | Stream width is the explicit StreamWidth attr. |
| `get_outstream_width` | TLastMarker_hls | 241 | Equal to input StreamWidth. |
| `get_verilog_top_module_intf_names` | TLastMarker_hls | 252 | Declares in0_V/out0_V AXIS ports at StreamWidth and conditionally an s_axi_control axilite when DynIters==1 (dynamic iteration count from AXI-lite). |
| `global_includes` | TLastMarker_hls | 84 | Needs ap_axi_sdata.h for qdma_axis/ap_axiu TLAST-carrying stream types. |
| `defines` | TLastMarker_hls | 87 | Selects stream C++ type (qdma_axis vs ap_axiu vs ap_uint) based on Direction and Protocol. |
| `docompute` | TLastMarker_hls | 122 | Emits the actual TLAST insert/remove loop, forked across direction(in/out) x DynIters(static/dynamic) x Protocol(qdma_axis/ap_axiu API). |
| `blackboxfunction` | TLastMarker_hls | 180 | Top function signature gains a numIters argument only when DynIters==1. |
| `read_npy_data` | TLastMarker_hls | 119 | No npy input file read — data comes from the AXIS stream in stitched context; single-node exec is pass-through. |
| `dataoutstrm` | TLastMarker_hls | 177 | No stream2npy output write. |
| `strm_decl` | TLastMarker_hls | 245 | Declares in0_V/out0_V hls::stream of InDType/OutDType (the marker-typed streams). |
| `pragmas` | TLastMarker_hls | 196 | AXIS interface pragmas on both ports, conditional s_axilite for numIters, ap_ctrl_none on return. |

## Hacks (12 — 0 blocker, 6 major)

- **[major/brittle-assumption]** `tlastmarker_hls.py:132` — Direction=='in' docompute builds the loop body with an inline ternary: the qdma branch emits 'out0_V.write(in0_V.read().get_data());' (NO closing brace) while the ap_axiu branch emits 'out0_V.write(in0_V.read().data);}' (WITH the '}' fused onto the same string). The for-loop's closing brace is smuggled into ONE ternary arm, so for Direction=in + Protocol=external (qdma) the generated C++ for-loop is left UNCLOSED — a latent compile break that only the internal-protocol path avoids.
- **[major/template-surgery]** `tlastmarker_hls.py:129` — docompute is assembled as a list of C++ source lines with per-line Python ternaries switching between qdma_axis setter API (set_data/set_last/set_keep) and ap_axiu member-assignment API (.data/.last/.keep). The same algorithm is duplicated three times across the in / dynamic-out / static-out branches, each re-emitting the protocol ternary — copy-paste HLS surgery with no shared helper.
- **[major/brittle-assumption]** `tlastmarker_hls.py:76` — make_shape_compatible_op is a bare 'pass' (returns None). Any transformation that calls it for shape inference on a graph containing TLastMarker will get None back — the op silently opts out of the shape-inference contract instead of raising or returning a valid node.
- **[major/brittle-assumption]** `tlastmarker_hls.py:80` — infer_node_datatype is a bare 'pass' — the node does not propagate datatypes. Combined with get_input/get_output_datatype raising, the op is entirely opaque to FINN's datatype machinery; any pass that assumes datatype getters work will crash on this node.
- **[major/other]** `tlastmarker_hls.py:211` — The four datatype/normal-shape contract getters (get_input_datatype 211, get_output_datatype 215, get_normal_input_shape 219, get_normal_output_shape 223) all raise Exception('not implemented'). Half of the 8-method HWCustomOp contract is deliberately non-functional; callers must special-case TLastMarker.
- **[major/brittle-assumption]** `tlastmarker_hls.py:142` — Dynamic-iters path relies on a '#pragma HLS protocol fixed' cycle-accurate io_section and a first speculative read to give software time to program numIters over AXI-lite before commit. This is a timing/protocol assumption hand-encoded in string HLS; fragile against HLS tool changes and invisible to the rest of FINN.
- **[minor/magic-number]** `tlastmarker_hls.py:95` — Stream type templates hard-code 'qdma_axis<%d,0,0,0>' and 'ap_axiu<%d,0,0,0>' — the 0,0,0 are TKEEP/TID/TDEST parameter widths baked in with no attr; assumes no user routing/keep sideband. Repeated at lines 97, 104, 106.
- **[minor/magic-number]** `tlastmarker_hls.py:141` — t.set_keep(-1) / t.keep = -1 uses -1 as an all-ones TKEEP mask magic value; also at line 166. No symbolic constant, relies on two's-complement fill.
- **[minor/other]** `tlastmarker_hls.py:225` — Copy-paste error: get_normal_output_shape raises Exception with the wrong message 'get_normal_input_shape not implemented for TlastMarker' — evidence these stubs were duplicated without editing.
- **[minor/hard-coded-param]** `tlastmarker_hls.py:232` — get_folded_input_shape hard-codes batch dimension to 1: returns (1, n_iters, n_packed_elems). No multi-image batching supported at this node.
- **[minor/duplicated-logic]** `tlastmarker_hls.py:93` — Protocol external/internal -> qdma_axis/ap_axiu selection logic exists in defines() (lines 93-108) and is re-derived independently in docompute() via 'use_qdma_axis = Protocol=="external"' (line 125). Two sources of truth for the same protocol decision.
- **[minor/other]** `tlastmarker_hls.py:35` — Whole op exists solely to service the FINN PYNQ/DMA shell (docstring: DMA write / end-of-burst removal). It is a shell/board-specific artifact modeled as a generic dataflow op, coupling the op library to a particular deployment shell.

## Hermeticity violations (2)

- **[hidden-coupling]** `tlastmarker_hls.py:185` — blackboxfunction embeds self.onnx_node.name directly into the emitted C++ top-function name (lines 185-193). The generated HLS symbol is coupled to the ONNX node name; renaming the node changes the RTL top and any downstream stitching that references it.
- **[hidden-coupling]** `tlastmarker_hls.py:35` — Semantics are coupled to the FINN PYNQ shell/DMA behavior (must sit at network start to strip end-of-burst TLAST, or at end to signal DMA write). Correct placement is an out-of-band graph-topology invariant not enforced by the op itself.

## Seams (3)

- **fused-no-seam** — There is no RTL variant and no agnostic base — the class inherits HWCustomOp+HLSBackend directly and all behavior (shape/width contract + HLS codegen) is fused in one class. No HLS/RTL swap seam exists; substituting a backend means rewriting the whole op.
- **fused-no-seam** — defines() and docompute() fuse Direction x Protocol x DynIters branching directly into hand-written C++ string templates (lines 87-175). The 'shape/contract' layer and the 'backend codegen' layer are inseparable; there is no protocol-agnostic op description to reuse.
- **clean-seam** — The npy-IO seam is cleanly overridden: read_npy_data (119), dataoutstrm (177) and execute_node (65) are emptied/pass-through, cleanly decoupling this node from the default HLSBackend cppsim npy plumbing. This is the one place the HLSBackend abstraction is used as a genuine seam (opt-out).
