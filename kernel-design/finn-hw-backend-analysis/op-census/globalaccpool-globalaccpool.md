# Census: globalaccpool (GlobalAccPool)

*Small, clean HLS-only pooling op: a backend-agnostic GlobalAccPool base implementing all shape/datatype/width/cycle logic plus a python golden model, and a thin GlobalAccPool_hls providing the four HLSBackend abstracts that string-fill a finn-hlslib AccPool_Batch call. No RTL variant, no rtllib coupling, no hermeticity issues; the notable risks are hard-coded 4D-layout assumptions in the base (output shape, output datatype, reduction axes) rather than backend-boundary abuse.*

**Files:** `src/finn/custom_op/fpgadataflow/globalaccpool.py`, `src/finn/custom_op/fpgadataflow/hls/globalaccpool_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `GlobalAccPool` | `globalaccpool.py` | `(HWCustomOp)` | yes |
| `GlobalAccPool_hls` | `globalaccpool_hls.py` | `(GlobalAccPool, HLSBackend)` | yes |

## Redesign pressure

This family is one of the cleaner ones: a genuinely backend-agnostic base (all 8 contract methods + cycles + python model) plus a thin HLS variant implementing only the 4 HLSBackend abstracts, with zero RTL variant, no rtllib coupling, and no hermeticity violations. The real friction is not the HLS/RTL axis but the rank/layout assumptions baked into the base: get_normal_output_shape only handles numInputVectors of length 1 or 3 (UnboundLocalError otherwise), get_output_datatype hard-indexes vecs[-1]*vecs[-2], execute_node hard-codes reduction axes [1,2], and docompute hard-indexes get_normal_input_shape()[1]. These would resist any redesign that generalizes tensor rank/layout, and the base-vs-backend execute_node split (python model shadowed by HLS sim) is the only awkward fusion. Otherwise the 2-axis abstraction fits this op well.

## Overrides (14)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | GlobalAccPool | 42 | Adds NumChannels, PE, inputDataType, numInputVectors then merges super() attrs. |
| `get_normal_input_shape` | GlobalAccPool | 57 | Input shape = numInputVectors + [NumChannels]; channels-last layout. |
| `get_folded_input_shape` | GlobalAccPool | 63 | Folds channels into [folds, PE] with assert ch%pe==0. |
| `get_normal_output_shape` | GlobalAccPool | 72 | Global pool collapses spatial dims: for len(vecs)==3 output is [batch,1,1,ch]; for len(vecs)==1 output is vecs+[ch]. |
| `get_folded_output_shape` | GlobalAccPool | 81 | Folds output channel dim into [folds, PE] from normal output shape. |
| `get_input_datatype` | GlobalAccPool | 104 | Reads inputDataType nodeattr. |
| `get_output_datatype` | GlobalAccPool | 108 | Derives accumulator datatype from npixels*idt extreme value via get_smallest_possible; sum-pool grows bitwidth. |
| `get_instream_width` | GlobalAccPool | 120 | PE * input bitwidth. |
| `get_outstream_width` | GlobalAccPool | 127 | PE * output (accumulator) bitwidth. |
| `get_exp_cycles` | GlobalAccPool | 134 | prod(folded_input_shape[:-1]) + folds latency estimate. |
| `execute_node` | GlobalAccPool | 141 | Python golden model via np.apply_over_axes(np.sum, ..., [1,2]). |
| `get_nodeattr_types` | GlobalAccPool_hls | 39 | Merges GlobalAccPool + HLSBackend attr dicts explicitly. |
| `verify_node` | GlobalAccPool_hls | 45 | Checks backend, required attrs, and that numInputVectors is length 3 (2D data). |
| `execute_node` | GlobalAccPool_hls | 72 | Delegates to HLSBackend.execute_node (cppsim/rtlsim path), shadowing GlobalAccPool's python model. |

## Hacks (9 — 0 blocker, 4 major)

- **[major/brittle-assumption]** `globalaccpool.py:72` — get_normal_output_shape only assigns oshape for len(vecs)==1 or len(vecs)==3. Any other numInputVectors length (e.g. 2 or 4) leaves oshape unbound -> UnboundLocalError at return. Silent partial coverage, no else/raise.
- **[major/brittle-assumption]** `globalaccpool.py:113` — get_output_datatype computes npixels = vecs[-1] * vecs[-2], hard-assuming at least 2 trailing spatial dims in numInputVectors. For a length-1 numInputVectors (allowed by get_normal_output_shape branch at line 75) this indexes vecs[-2] out of range or multiplies wrong values, producing a wrong accumulator datatype.
- **[major/magic-number]** `globalaccpool.py:146` — execute_node hard-codes reduction axes [1,2] in np.apply_over_axes(np.sum, inp_values, [1,2]). Only valid for the 4D NHWC batch layout; couples the python golden model to one specific tensor rank rather than deriving axes from numInputVectors.
- **[major/magic-number]** `globalaccpool_hls.py:84` — docompute passes get_normal_input_shape()[1] as the ImgDim template arg to AccPool_Batch. Index [1] assumes numInputVectors starts at position 0 and the spatial dim sits at index 1; brittle magic index tied to the 4D layout.
- **[minor/template-surgery]** `globalaccpool_hls.py:83` — docompute builds the AccPool_Batch<...> C++ call by positional .format() string-fill of 5 template params (ImgDim, NumChannels, in_dt, PE, out_dt) plus a literal trailing '1' (batch/reps). Positional coupling to the finn-hlslib template signature; a reorder in the HLS lib silently miscompiles.
- **[minor/other]** `globalaccpool_hls.py:76` — global_includes pulls '#include "maxpool.h"' even though this op instantiates AccPool_Batch (an accumulation, not max) pool. AccPool_Batch happens to live in maxpool.h in finn-hlslib; the include name is misleading and reveals a header-organization dependency.
- **[minor/other]** `globalaccpool_hls.py:63` — verify_node messages reference 'GlobalAccPool_Batch' (old monolithic node name) rather than the current class name GlobalAccPool_hls; stale naming carried over from pre-split refactor.
- **[minor/duplicated-logic]** `globalaccpool_hls.py:45` — verify_node is the boilerplate backend/attr-existence check copied near-verbatim across many fpgadataflow _hls ops; not specific to this op.
- **[minor/other]** `globalaccpool.py:36` — GlobalAccPool does NOT override make_shape_compatible_op or get_number_output_values, relying entirely on HWCustomOp defaults. Given the spatial-collapse output shape (line 72), the base make_shape_compatible_op default may not reflect the true output shape — noted as a gap, not verified against base impl here.

## Hermeticity violations (0)


## Seams (2)

- **clean-seam** — Backend logic is fully isolated: GlobalAccPool (agnostic base) holds all shape/datatype/width/cycles/python-model logic; GlobalAccPool_hls holds only the 4 HLSBackend abstract methods (global_includes, defines, docompute, blackboxfunction) plus verify_node and sim dispatch. A new backend could subclass (GlobalAccPool, RTLBackend) without touching the base.
- **fused-no-seam** — execute_node is split awkwardly: the agnostic base defines a real python golden model (line 141) but GlobalAccPool_hls overrides it (line 72) to call HLSBackend.execute_node. The base's python reference is thus only reachable if used directly (analytical/transform passes), not through the concrete backend op — the agnostic model and the backend sim are fused via override rather than composed.
