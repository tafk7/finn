# Census: upsampler (UpsampleNearestNeighbour)

*A clean, small HLS-only nearest-neighbour upsampler: a complete HWCustomOp-derived agnostic base (shapes, folding, datatype passthrough, and an onnxruntime-Resize golden reference) plus a thin HLS variant emitting finn-hlslib upsample_nn. Main quirks are the onnxruntime-backed base execute_node and the diamond-inheritance execute_node re-dispatch.*

**Files:** `src/finn/custom_op/fpgadataflow/upsampler.py`, `src/finn/custom_op/fpgadataflow/hls/upsampler_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `UpsampleNearestNeighbour` | `upsampler.py` | `(HWCustomOp)` | yes |
| `UpsampleNearestNeighbour_hls` | `upsampler_hls.py` | `(UpsampleNearestNeighbour, HLSBackend)` | yes |

## Redesign pressure

This is one of the cleaner families: a well-factored agnostic base implementing the full HWCustomOp contract and a thin HLS variant supplying only the four HLSBackend abstracts — the HLS/RTL axis maps naturally and a new backend would slot in without touching the base. The one real friction against the 2-axis abstraction is execute_node: the base carries a heavyweight onnxruntime-backed 'golden reference' execution while backends need cppsim/rtlsim execution, and the two collide under diamond inheritance, forcing the HLS class to manually re-dispatch to HLSBackend.execute_node. A redesign would benefit from separating 'functional/reference semantics' from 'backend simulation' as distinct contract slots rather than overloading a single execute_node resolved by MRO. Secondary pressure: unchecked SIMD-divisibility and integer-scale rounding assumptions live in the shape/base layer with no validation seam.

## Overrides (18)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | UpsampleNearestNeighbour | 45 | Declares op-specific attrs SIMD, HO/WO/HI/WI (in/out feature map dims), NumChannels, inputDataType, batchSize; merges base via super(). |
| `get_exp_cycles` | UpsampleNearestNeighbour | 64 | Cycle estimate = product of folded output spatial dims; one output pixel per cycle. |
| `get_normal_input_shape` | UpsampleNearestNeighbour | 67 | NHWC input shape (batch, HI, WI, NumChannels). |
| `get_normal_output_shape` | UpsampleNearestNeighbour | 75 | NHWC output shape (batch, HO, WO, NumChannels) — spatial upsampled, channels unchanged. |
| `get_folded_input_shape` | UpsampleNearestNeighbour | 83 | Folds channel dim by SIMD: spatial + [NumChannels//SIMD, SIMD]. |
| `get_folded_output_shape` | UpsampleNearestNeighbour | 89 | Same channel folding as input applied to output spatial shape. |
| `get_input_datatype` | UpsampleNearestNeighbour | 109 | Reads inputDataType nodeattr. |
| `get_output_datatype` | UpsampleNearestNeighbour | 114 | Datatype passthrough — output equals input datatype (no compute changes values' range). |
| `get_instream_width` | UpsampleNearestNeighbour | 118 | ibits * SIMD. |
| `get_outstream_width` | UpsampleNearestNeighbour | 123 | obits * SIMD. |
| `infer_node_datatype` | UpsampleNearestNeighbour | 95 | Propagates input datatype unchanged to output; warns if inputDataType attr drifts from actual tensor datatype. |
| `execute_node` | UpsampleNearestNeighbour | 128 | Functional/golden execution: builds a standalone ONNX Resize(mode=nearest) graph and runs it via onnxruntime to produce the reference output. |
| `get_nodeattr_types` | UpsampleNearestNeighbour_hls | 43 | Merges UpsampleNearestNeighbour attrs with HLSBackend attrs explicitly. |
| `global_includes` | UpsampleNearestNeighbour_hls | 49 | HLS contract: include finn-hlslib upsample.hpp. |
| `defines` | UpsampleNearestNeighbour_hls | 52 | Emit #define for HI/WI/HO/WO and CF (=NumChannels//SIMD) for the HLS template. |
| `docompute` | UpsampleNearestNeighbour_hls | 72 | Instantiates upsample_nn<HI, HO, WI, WO, CF, CF>(in0_V, out0_V). |
| `blackboxfunction` | UpsampleNearestNeighbour_hls | 77 | Emits top-level signature using hls::stream<hls::vector<T,SIMD>> in0_V/out0_V. |
| `execute_node` | UpsampleNearestNeighbour_hls | 93 | Delegates to HLSBackend.execute_node for cppsim/rtlsim, overriding the parent's onnxruntime-Resize reference. |

## Hacks (9 — 0 blocker, 4 major)

- **[major/other]** `upsampler.py:128` — Base-class execute_node computes the reference output by constructing a fresh ONNX Resize graph and spinning up an onnxruntime InferenceSession (rt.InferenceSession(model.SerializeToString())). The shape/hw op abstraction pulls in onnxruntime + qonnx_make_model as an execution engine rather than a self-contained numpy kernel.
- **[major/magic-number]** `upsampler.py:137` — scales_val = [1, int(round(HO/HI)), int(round(WO/WI)), 1] assumes an integer upsample factor and silently rounds; a non-integer HO/HI ratio would produce a scale that disagrees with the declared HO/WO output shape.
- **[major/brittle-assumption]** `upsampler.py:86` — folds = NumChannels // simd uses integer floor division with no assertion that SIMD divides NumChannels; a non-divisible SIMD silently drops channels in the folded shape (and in CF for the HLS template).
- **[major/inheritance-irregularity]** `upsampler_hls.py:93` — execute_node explicitly calls HLSBackend.execute_node(self,...) to sidestep the agnostic base's onnxruntime-Resize execute_node that would otherwise win via MRO (UpsampleNearestNeighbour is first parent). The two execute_node implementations serve different purposes (golden ref vs cppsim/rtlsim) and are disambiguated only by this manual override.
- **[minor/magic-number]** `upsampler.py:155` — Hard-coded ONNX opset 13 (helper.make_opsetid("", 13)) baked into the reference execution graph.
- **[minor/brittle-assumption]** `upsampler_hls.py:37` — Class docstring states 'The layer expects square feature maps for the in and output', but the Python code carries independent HI/WI/HO/WO and the HLS template is instantiated with all four; the docstring is stale/misleading about the actual constraint.
- **[minor/duplicated-logic]** `upsampler.py:89` — get_folded_output_shape is a verbatim copy of get_folded_input_shape (same simd, same NumChannels//simd fold); channel folding logic duplicated rather than shared.
- **[minor/hard-coded-param]** `upsampler_hls.py:74` — docompute passes CF twice: upsample_nn<HI, HO, WI, WO, CF, CF> — the finn-hlslib template expects two channel-fold params (input/output) but they are forced equal, hard-coding the assumption that channel count is preserved across the op.
- **[minor/inheritance-irregularity]** `upsampler_hls.py:43` — get_nodeattr_types manually merges UpsampleNearestNeighbour.get_nodeattr_types(self) and HLSBackend.get_nodeattr_types(self) by explicit class calls rather than a cooperative super() chain, a diamond-inheritance workaround.

## Hermeticity violations (3)

- **[hidden-coupling]** `upsampler.py:30` — Module imports onnxruntime (import onnxruntime as rt) at module scope solely to run the reference execution — the shape-agnostic op class hard-depends on an inference runtime.
- **[hidden-coupling]** `upsampler.py:159` — execute_node instantiates rt.InferenceSession and runs it, an order/environment-dependent external execution engine, inside what is otherwise a pure metadata/shape class.
- **[hidden-coupling]** `upsampler.py:34` — Uses qonnx_make_model + onnx.helper to build a throwaway single-node model at execution time; couples node execution to ONNX model-construction machinery.

## Seams (2)

- **clean-seam** — The agnostic base UpsampleNearestNeighbour cleanly owns all 8 contract methods + folding/datatype logic; the HLS variant only supplies the 4 HLSBackend abstracts (global_includes/defines/docompute/blackboxfunction). A new backend (e.g. RTL) could subclass (UpsampleNearestNeighbour, RTLBackend) with no changes to the base — a textbook clean substitution point.
- **fused-no-seam** — execute_node is fused/ambiguous across the diamond: the base provides an onnxruntime golden reference and the HLS variant must explicitly re-dispatch to HLSBackend.execute_node (upsampler_hls.py:93) to avoid MRO picking the base. Any new backend must repeat this manual override; the reference-vs-simulation execution paths are not cleanly separated by the abstraction.
