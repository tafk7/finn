# Census: labelselect

*LabelSelect is a compact, mostly-clean HLS-only op family (agnostic LabelSelect base + LabelSelect_hls) implementing TopK index selection, wrapping finn-hlslib's LabelSelect_Batch. The base/backend seam is clean, but it carries several op-specific quirks: constructor string-surgery to auto-derive outputDataType, onnxruntime-based functional emulation, a hard-coded maxpool.h include, and an int64 re-cast workaround.*

**Files:** `src/finn/custom_op/fpgadataflow/labelselect.py`, `src/finn/custom_op/fpgadataflow/hls/labelselect_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `LabelSelect` | `labelselect.py` | `(HWCustomOp)` | yes |
| `LabelSelect_hls` | `labelselect_hls.py` | `(LabelSelect, HLSBackend)` | yes |

## Redesign pressure

This family is small and largely clean: the agnostic base is genuinely backend-independent and the HLS variant only fills the HLSBackend contract, so the base/backend seam is one of the tidier ones. The main friction points for a redesign are not the two-axis split itself but (1) the op is HLS-only -- no RTL variant exists, so the abstraction is exercised on just one axis and the maxpool.h header + LabelSelect_Batch signature are hard-wired with no parameterization indirection; and (2) op-specific quirks that sit awkwardly on the contract: constructor-time auto-derivation of outputDataType via string surgery, functional emulation via a spun-up onnxruntime TopK graph, and a post-sim int64 re-cast to defeat FINN's float-container assumption. None of these are blockers, but the datatype auto-derivation-in-__init__ and the output-PE-fixed-to-1 folding (get_folded_output_shape/get_outstream_width don't scale with any output PE) are asymmetries a uniform folding model would need to accommodate.

## Overrides (20)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | LabelSelect | 54 | adds Labels/PE/K folding params, input/output datatypes, numInputVectors; merges super(). |
| `get_normal_input_shape` | LabelSelect | 71 | input is numInputVectors + Labels (the full score vector to rank). |
| `get_folded_input_shape` | LabelSelect | 77 | folds Labels axis into (folds, PE); asserts PE divides Labels. |
| `get_normal_output_shape` | LabelSelect | 86 | output is numInputVectors + K (the K selected label indices). |
| `get_folded_output_shape` | LabelSelect | 92 | output folded as (K, 1) -- implicit output PE fixed to 1, one index per cycle. |
| `get_input_datatype` | LabelSelect | 107 | reads inputDataType nodeattr. |
| `get_output_datatype` | LabelSelect | 112 | reads outputDataType nodeattr (auto-derived in __init__ if empty). |
| `get_instream_width` | LabelSelect | 117 | PE * input bitwidth (PE scores packed per beat). |
| `get_outstream_width` | LabelSelect | 124 | just output datatype bitwidth -- output PE is implicitly 1, one index at a time. |
| `get_number_output_values` | LabelSelect | 128 | returns K (number of selected indices). |
| `execute_node` | LabelSelect | 131 | emulates functionality by building a standalone TopK ONNX graph and running onnxruntime. |
| `get_exp_cycles` | LabelSelect | 161 | Labels/PE cycles (one beat per fold). |
| `get_nodeattr_types` | LabelSelect_hls | 41 | merges LabelSelect attrs with HLSBackend attrs explicitly (diamond merge). |
| `verify_node` | LabelSelect_hls | 47 | checks backend attr, required attrs, and 1D-only input constraint. |
| `execute_node` | LabelSelect_hls | 76 | calls HLSBackend.execute_node then casts output to int64 (TopK indices). |
| `global_includes` | LabelSelect_hls | 86 | includes maxpool.h (finn-hlslib LabelSelect_Batch lives there). |
| `defines` | LabelSelect_hls | 89 | no defines needed -- all params passed as template args in docompute. |
| `read_npy_data` | LabelSelect_hls | 92 | custom LE packing (reverse_inner=false) as required by LabelSelect_Batch. |
| `docompute` | LabelSelect_hls | 116 | instantiates LabelSelect_Batch<Labels,PE,K,idt,odt>(in0_V,out0_V,1). |
| `blackboxfunction` | LabelSelect_hls | 127 | declares top with in0_V ap_uint<PE*ibits> and out0_V ap_uint<obits>. |

## Hacks (10 — 0 blocker, 3 major)

- **[major/template-surgery]** `labelselect.py:49` — outputDataType auto-derivation does string surgery on the datatype NAME: new_odt_name = odt.name.replace(str(odt.bitwidth()), str(bw)). Turns e.g. UINT4 -> UINT8 by string-replacing the bitwidth digits inside the name. Brittle: relies on the numeric bitwidth appearing exactly once and only as the width suffix; would misfire on names where the width digits recur.
- **[major/brittle-assumption]** `labelselect.py:52` — __init__ mutates node state via set_nodeattr('outputDataType', ...) at construction time when the attr is empty. Constructing the op has a side effect on the ONNX node; behavior is order/construction dependent.
- **[major/cross-backend-leak]** `labelselect_hls.py:87` — global_includes pulls '#include "maxpool.h"' -- LabelSelect_Batch is defined in finn-hlslib's maxpool.h, an unintuitive cross-op header dependency. A redesign cannot infer the required header from the op name; it is hard-coded to a sibling op's file.
- **[minor/magic-number]** `labelselect.py:48` — roundup_to_integer_multiple(odt.bitwidth(), 8) hard-codes 8-bit rounding of the auto-derived output datatype 'in case this is the last node' -- an implicit assumption about downstream/host container width baked into the op constructor.
- **[minor/duplicated-logic]** `labelselect.py:132` — Copy-paste artifact: comment says 'create a standard add node to help calculate the result' but the code builds a TopK node. Graph name at line 150 is 'single-add-exec' -- both leftovers from a different op, indicating execute_node was copied.
- **[minor/other]** `labelselect.py:155` — execute_node builds a throwaway ONNX model and runs a full onnxruntime InferenceSession per node execution to emulate TopK. Heavyweight and pulls onnxruntime into the op's functional path (import at line 29).
- **[minor/magic-number]** `labelselect_hls.py:118` — docompute hard-codes the trailing numReps argument as literal 1 in LabelSelect_Batch<...>(in0_V, out0_V, 1) -- assumes a single repetition regardless of numInputVectors batch.
- **[minor/brittle-assumption]** `labelselect_hls.py:72` — verify_node raises a bare `raise Exception` (no message, no type) when numInputVectors is not 1D. Uninformative control-flow, and enforces a 1D-only constraint that the agnostic base does not encode.
- **[minor/duplicated-logic]** `labelselect_hls.py:102` — read_npy_data is a hand-rolled copy of the generic npy2apintstream call with a comment about LE packing referencing StreamingDataWidthConverter_Batch -- duplicated boilerplate that overrides the HLSBackend default just to flip reverse_inner=false.
- **[minor/todo-marker]** `labelselect_hls.py:78` — execute_node override exists solely to work around FINN's float-container DataType assumption: it re-casts the TopK index output to int64 after HLSBackend.execute_node. Comment documents that INT64 TopK indices 'can cause issues for the node-by-node simulation'.

## Hermeticity violations (3)

- **[hidden-coupling]** `labelselect.py:155` — execute_node instantiates rt.InferenceSession (onnxruntime, imported line 29) to run a synthesized TopK graph -- functional execution depends on an ambient onnxruntime runtime beyond FINN's own datatype/sim machinery.
- **[module-mutable-state]** `labelselect.py:52` — Constructor writes outputDataType back into the node attributes when unset, so merely instantiating the custom op mutates node state -- an order-dependent side effect.
- **[hidden-coupling]** `labelselect_hls.py:84` — execute_node post-processes context[outp] by casting to np.int64, coupling correct sim results to a manual container-type fixup outside the generic HLSBackend flow.

## Seams (3)

- **clean-seam** — The base/backend split is clean: LabelSelect (labelselect.py) is fully backend-agnostic (shapes, datatypes, cycles, functional execute_node via onnxruntime), and LabelSelect_hls supplies only the four HLSBackend abstract methods plus verify/read_npy/execute overrides. A new backend could subclass LabelSelect without touching the base.
- **fused-no-seam** — The HLS docompute/global_includes/blackboxfunction are tightly fused to the finn-hlslib LabelSelect_Batch signature and its host header maxpool.h (labelselect_hls.py:87,116,127). These are not separable from finn-hlslib -- there is no template/param indirection, the call and header are literal.
- **fused-no-seam** — Asymmetry: there is NO RTL variant. The family is HLS-only, so the '2-axis HLS/RTL' abstraction has only one axis populated; any RTL seam is absent entirely rather than clean.
