# Census: hwsoftmax (HWSoftmax)

*A small, largely clean HLS-only softmax op: an agnostic HWSoftmax (shapes, SIMD folding of the last axis, float32 output, scipy reference model) plus one HWSoftmax_hls backend that string-templates finn-hlslib's SoftMax<> C++ template. Main irregularities are backend facts leaking into the agnostic base (float32-only output, hlslib zero-padding assertion) and an MRO band-aid forcing the correct execute_node.*

**Files:** `src/finn/custom_op/fpgadataflow/hwsoftmax.py`, `src/finn/custom_op/fpgadataflow/hls/hwsoftmax_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `HWSoftmax` | `hwsoftmax.py` | `(HWCustomOp)` | yes |
| `HWSoftmax_hls` | `hwsoftmax_hls.py` | `(HWSoftmax, HLSBackend)` | yes |

## Redesign pressure

This family is small and mostly clean, but it is HLS-only (no RTL variant, zero finn-rtllib coupling) so the 2-axis HLS/RTL abstraction is only half-exercised — the real stress is inside the diamond, not across the HLS/RTL split. The sharpest pressure is the dual execute_node: the agnostic base carries a scipy reference model while the HLS class must explicitly re-dispatch to HLSBackend to override the MRO, revealing that 'agnostic op' and 'backend' both want to own execution semantics and the current design resolves the conflict by hand-wiring. Secondary pressure comes from backend-specific facts (float32-only output, must-represent-zero input constraint) leaking upward into the supposedly backend-agnostic HWSoftmax, so the base class is already silently specialized to finn-hlslib and would mis-constrain any future RTL variant. A redesign should give the agnostic layer a single, backend-neutral execution/datatype contract and push hlslib-specific constraints down into the HLS backend.

## Overrides (20)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | HWSoftmax | 23 | Adds op-specific attrs: ifm_dim (input shape), SIMD (fold factor), input_data_type, NumChannels. Merges super() defaults. |
| `get_normal_input_shape` | HWSoftmax | 34 | Softmax has no separate weight/shape inference; the full input shape is stored verbatim in the ifm_dim attribute. |
| `get_normal_output_shape` | HWSoftmax | 37 | Softmax is shape-preserving (element-wise over last axis), so output shape == input shape. |
| `get_folded_input_shape` | HWSoftmax | 87 | Folds the last (softmax) axis by SIMD: splits normal[-1] into [fold, SIMD]; asserts SIMD divides the axis. |
| `get_folded_output_shape` | HWSoftmax | 84 | Output folding identical to input folding (shape-preserving). |
| `get_input_datatype` | HWSoftmax | 46 | Reads input_data_type attr AND asserts the type can represent zero because the hlslib SoftMax pads with zeros. |
| `get_output_datatype` | HWSoftmax | 80 | SoftMax hardware always emits float32 probabilities regardless of input type. |
| `get_instream_width` | HWSoftmax | 70 | SIMD lanes of input bitwidth. |
| `get_outstream_width` | HWSoftmax | 75 | SIMD lanes of output (float32=32b) bitwidth. |
| `execute_node` | HWSoftmax | 40 | Python functional model: scipy.special.softmax over axis=-1 written straight into context. |
| `make_shape_compatible_op (NOT overridden)` | HWSoftmax | 103 | Relies on HWCustomOp default (const op from get_normal_output_shape). Noted as inherited-clean. |
| `get_number_output_values (NOT overridden)` | HWSoftmax | 254 | Relies on HWCustomOp default prod(folded_output_shape[:-1]). Noted as inherited-clean. |
| `get_nodeattr_types` | HWSoftmax_hls | 20 | Manual diamond merge of HWSoftmax + HLSBackend attr dicts. |
| `execute_node` | HWSoftmax_hls | 82 | Forces HLSBackend.execute_node (cppsim/rtlsim) instead of the scipy model that MRO would otherwise pick from HWSoftmax. |
| `global_includes` | HWSoftmax_hls | 26 | HLS contract: pulls hls_vector.h, softmax.hpp, utils.hpp (external finn-hlslib headers). |
| `defines` | HWSoftmax_hls | 33 | HLS contract: emits SIMD, W=ifm_dim[-1], TI (input hls type), F=float constexprs. |
| `docompute` | HWSoftmax_hls | 46 | HLS contract: instantiates SoftMax<TI,float,W,SIMD> template, uses move() to shuttle AXIS streams. |
| `blackboxfunction` | HWSoftmax_hls | 59 | HLS contract: top function signature with node name interpolated, vector AXIS in/out. |
| `pragmas` | HWSoftmax_hls | 69 | HLS seam: AXIS interfaces, bit-compact aggregate, ap_ctrl_none, dataflow disable_start_propagation. |
| `timeout_value` | HWSoftmax_hls | 85 | rtlsim needs a timeout scaled to total elements (softmax is multi-cycle), not the base constant 1000. |

## Hacks (9 — 0 blocker, 3 major)

- **[major/hard-coded-param]** `hwsoftmax.py:82` — get_output_datatype hard-returns DataType['FLOAT32'] and ignores the ind argument entirely. Output dtype is not configurable; hardware/float coupling baked into the abstract layer.
- **[major/brittle-assumption]** `hwsoftmax.py:51` — get_input_datatype asserts data_type.allowed(0) with comment 'the hlslib op always pads with zeros'. This leaks a specific finn-hlslib SoftMax implementation detail (zero-padding) into the backend-agnostic HWSoftmax class, constraining valid input datatypes for a reason unrelated to the math.
- **[major/inheritance-irregularity]** `hwsoftmax_hls.py:82` — execute_node must explicitly call HLSBackend.execute_node(self,...) because the MRO (HWSoftmax, HLSBackend) would otherwise resolve execute_node to HWSoftmax's scipy python model. This is an MRO band-aid: the two inherited execute_node implementations collide and the fix is hard-coded parent dispatch rather than a single dispatch policy.
- **[minor/magic-number]** `hwsoftmax.py:29` — NumChannels attr default 128 is declared in get_nodeattr_types but is never read anywhere in either file (no self.get_nodeattr('NumChannels')). Dead/misleading attribute; folding actually derives from ifm_dim[-1] and SIMD.
- **[minor/template-surgery]** `hwsoftmax_hls.py:62` — blackboxfunction f-string interpolates self.onnx_node.name directly into the generated C++ top function name; defines() (line 36) and docompute rely on W=ifm_dim[-1] and a TI type string built from get_hls_datatype_str(). All codegen is raw string templating with no escaping/validation.
- **[minor/todo-marker]** `hwsoftmax_hls.py:86` — timeout_value docstring is copied verbatim from HLSBackend: 'Set timeout value for HLS functions defined for one clock cycle', yet the body returns prod(input_shape) precisely because softmax is NOT one clock cycle. Copy-pasted, contradictory docstring.
- **[minor/inheritance-irregularity]** `hwsoftmax_hls.py:20` — get_nodeattr_types manually merges HWSoftmax.get_nodeattr_types(self) and HLSBackend.get_nodeattr_types(self) by explicit class name instead of super(), a common FINN diamond-merge workaround that is fragile if the MRO changes.
- **[minor/duplicated-logic]** `hwsoftmax.py:54` — infer_node_datatype re-implements the standard warn-on-mismatch + set-input/output-dtype pattern copied across many FINN ops (globalaccpool, pool, etc.) rather than a shared helper.
- **[minor/brittle-assumption]** `hwsoftmax.py:35` — get_normal_input_shape returns the raw ifm_dim attribute list; folding (get_folded_input_shape) and execute_node (axis=-1) implicitly assume the softmax axis is always the last dimension of ifm_dim. No validation that ifm_dim is well-formed.

## Hermeticity violations (2)

- **[hidden-coupling]** `hwsoftmax.py:12` — Module-level `from scipy.special import softmax` imports scipy at import time in the backend-agnostic op file, adding a heavy numeric dependency to the HW abstraction layer just for the reference execute_node.
- **[filesystem-path]** `hwsoftmax_hls.py:29` — global_includes references softmax.hpp and utils.hpp which are NOT present in this repo or finn-rtllib (confirmed via find) — they live in the external finn-hlslib include tree. Correct codegen depends on an ambient HLS include path pointing at finn-hlslib; the SoftMax<> C++ template and move() helper are resolved out-of-tree.

## Seams (3)

- **clean-seam** — HWSoftmax (agnostic: shapes, folding, datatypes, scipy reference) is cleanly separated from HWSoftmax_hls (HLS codegen: includes/defines/docompute/blackbox/pragmas). A second backend could subclass (HWSoftmax, RTLBackend) without touching HWSoftmax — the HLS specifics are fully contained in hwsoftmax_hls.py.
- **fused-no-seam** — execute_node is split across the seam: HWSoftmax.execute_node is a scipy functional model while HWSoftmax_hls.execute_node forces HLSBackend cppsim/rtlsim. The agnostic layer owns one execution semantics and the backend owns another, and only an explicit MRO-defeating override (line 82) keeps them from colliding. Which execute_node runs is not policy-driven but hand-wired per backend.
- **fused-no-seam** — get_input_datatype's allowed(0) assertion (hwsoftmax.py:51) embeds finn-hlslib's zero-padding behavior into the agnostic base, so the 'agnostic' contract is silently specialized to the HLS implementation — an RTL backend with different padding would inherit an incorrect constraint.
