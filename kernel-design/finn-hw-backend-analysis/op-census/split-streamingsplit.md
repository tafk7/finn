# Census: split (StreamingSplit)

*A clean, HLS-only fan-out op: StreamingSplit provides fully backend-agnostic split-along-last-axis math and StreamingSplit_hls supplies a thin HLS wrapper around an external split.hpp template. The only friction is multi-output handling, which forces per-output overrides of the base backend's out0-centric pragmas/timeout/drain and depends on an externally-set freerunning hls_style attr.*

**Files:** `src/finn/custom_op/fpgadataflow/split.py`, `src/finn/custom_op/fpgadataflow/hls/split_hls.py`, `src/finn/custom_op/fpgadataflow/hlsbackend.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`, `src/finn/transformation/fpgadataflow/convert_to_hw_layers.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `StreamingSplit` | `split.py` | `(HWCustomOp)` | yes |
| `StreamingSplit_hls` | `split_hls.py` | `(StreamingSplit, HLSBackend)` | yes |

## Redesign pressure

The op math is clean and cleanly separated — StreamingSplit is a model backend-agnostic base with no leakage. The real pressure is the multi-OUTPUT fan-out, which the current abstraction handles only awkwardly: the base HLSBackend hard-codes single-out0 assumptions in pragmas(), timeout_condition(), timeout_read_stream(), and the strm{o} 'freerunning' drain path, forcing this op to override each just to loop over N outputs. Correct cppsim additionally depends on an externally-set hls_style='freerunning' attr rather than an op-owned default, so the multi-output behavior is silently order-dependent on the InferSplitLayer transform. A redesign should make output arity a first-class dimension of the backend contract (per-output stream decl/drain/pragma generation) instead of out0-centric defaults that every fan-out op must patch. It is HLS-only with no RTL variant, so the 2-axis abstraction is under-exercised here — the strain is the N-output axis, not the HLS/RTL axis.

## Overrides (25)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | StreamingSplit | 45 | Adds SIMD, ChannelsPerStream (per-output-stream elem counts), inputDataType, numInputVectors; merges super(). |
| `get_normal_input_shape` | StreamingSplit | 68 | Input channels = sum of all ChannelsPerStream (splits along last axis). |
| `get_folded_input_shape` | StreamingSplit | 74 | folds = total_elems // SIMD along channel axis. |
| `get_normal_output_shape` | StreamingSplit | 80 | Per-output shape indexed by ind = ChannelsPerStream[ind]; one shape per split stream. |
| `get_folded_output_shape` | StreamingSplit | 85 | Per-output fold: ChannelsPerStream[ind] // SIMD. |
| `make_shape_compatible_op` | StreamingSplit | 92 | Emits ONNX Split with axis=-1; asserts input shape and output count. |
| `infer_node_datatype` | StreamingSplit | 102 | Propagates input dtype to ALL outputs (outputs inherit input dtype). |
| `verify_node` | StreamingSplit | 118 | No-op stub. |
| `get_input_datatype` | StreamingSplit | 121 | Reads inputDataType attr. |
| `get_output_datatype` | StreamingSplit | 124 | All outputs forced equal to input datatype; ind ignored. |
| `get_instream_width` | StreamingSplit | 128 | ibits * SIMD. |
| `get_outstream_width` | StreamingSplit | 132 | obits * SIMD; identical formula to instream, ind ignored. |
| `get_number_output_values` | StreamingSplit | 137 | Iterates over all node.output, product of folded_output_shape[1:-1] per stream (multi-output). |
| `get_exp_cycles` | StreamingSplit | 143 | product of folded_input_shape[:-1] (one cycle per SIMD-word streamed in). |
| `execute_node` | StreamingSplit | 146 | Python golden model via np.split on cumsum(ChannelsPerStream[:-1]) along axis=-1. |
| `get_instream_width_padded` | StreamingSplit | 154 | Pads instream width to multiple of 8 for AXIS. |
| `get_nodeattr_types` | StreamingSplit_hls | 37 | Manually merges StreamingSplit + HLSBackend attr dicts (diamond MRO resolution). |
| `execute_node` | StreamingSplit_hls | 43 | Delegates to HLSBackend.execute_node (cppsim/rtlsim), bypassing StreamingSplit's numpy path. |
| `global_includes` | StreamingSplit_hls | 46 | Includes split.hpp (external finn-hlslib). |
| `defines` | StreamingSplit_hls | 49 | Empty $DEFINES$; all params passed as template args instead. |
| `docompute` | StreamingSplit_hls | 52 | Emits StreamingSplit<fold0,fold1,...>(in0_V, out0_V, ...) with per-output folds as template args. |
| `blackboxfunction` | StreamingSplit_hls | 64 | hls::vector<T,SIMD> in0_V + N out streams; signature built from get_n_outputs(). |
| `pragmas` | StreamingSplit_hls | 77 | AXIS interface per output, ap_ctrl_none, aggregate compact=bit per stream. |
| `timeout_condition` | StreamingSplit_hls | 88 | AND of out{i}_V.empty() across all outputs (freerunning cppsim drain). |
| `timeout_read_stream` | StreamingSplit_hls | 95 | Guarded per-output strm{i} << out{i}_V.read() (freerunning drain). |

## Hacks (8 — 0 blocker, 3 major)

- **[major/brittle-assumption]** `split.py:76` — get_folded_input_shape computes folds = total_elems // SIMD with integer division and NO assert that SIMD divides total_elems. get_folded_output_shape (line 88) likewise does ChannelsPerStream[ind] // SIMD. If any per-stream channel count is not a multiple of SIMD, folds silently truncate and the streamed/executed shapes disagree with no error.
- **[major/brittle-assumption]** `split_hls.py:47` — global_includes pulls '#include "split.hpp"' which is NOT present anywhere in this repo (no split.hpp under repo root, finn-rtllib, or deps). It is an external finn-hlslib header resolved at HLS compile time. The StreamingSplit<...> C++ template and its arg order (folds then streams) are an undocumented contract with that external file.
- **[major/brittle-assumption]** `split_hls.py:88` — timeout_condition/timeout_read_stream are only invoked by HLSBackend.code_generation_cppsim when hls_style=='freerunning' (hlsbackend.py:228). StreamingSplit itself sets NO default hls_style (base default is 'ifm_aware'), so correct multi-output cppsim drain depends on the InferSplitLayer transform having set hls_style='freerunning' (convert_to_hw_layers.py:1320). If a node is built without that attr, the multi-output drain logic is dead and only base out0-only timeout runs.
- **[minor/duplicated-logic]** `split.py:154` — get_instream_width_padded reimplements HWCustomOp.get_instream_width_padded (hwcustomop.py:292) but omits the base's `if in_width != 0` zero-width guard and ignores the ind argument. Redundant override that silently diverges from the base contract; unclear why it exists since the base already does roundup_to_integer_multiple(width,8).
- **[minor/brittle-assumption]** `split.py:132` — get_outstream_width returns obits*SIMD identical to get_instream_width and ignores ind; combined with get_output_datatype ignoring ind (line 124), every output stream is assumed identical width/dtype. The per-stream channel differences in ChannelsPerStream are only reflected in fold counts, never in stream width — an implicit invariant that all streams share SIMD and dtype.
- **[minor/template-surgery]** `split_hls.py:61` — docompute string-builds a variadic C++ call 'StreamingSplit<%s>(in0_V, %s)' by joining per-output fold constants and out-stream names. Correctness depends entirely on positional match between the template arg list, the split.hpp template signature, and the blackboxfunction port order — no structural checking.
- **[minor/other]** `split_hls.py:49` — defines(self, var) accepts the 'var' argument (cppsim/ipgen selector) required by the HLSBackend interface but ignores it and emits an empty $DEFINES$ list. Harmless but signals the defines seam is vestigial for this op.
- **[minor/magic-number]** `convert_to_hw_layers.py:1324` — InferSplitLayer hard-codes SIMD=1 and outFIFODepths=[2]*N when creating StreamingSplit nodes, plus cpp_interface='hls_vector' and hls_style='freerunning'. These op-critical defaults live in the transform, not the op, so a node constructed by any other path lacks them.

## Hermeticity violations (3)

- **[filesystem-path]** `split_hls.py:47` — Hard dependency on external header split.hpp (finn-hlslib) not vendored in this tree; resolution depends on ambient HLS include paths set elsewhere in the build.
- **[order-dependence]** `split_hls.py:88` — Multi-output timeout/drain behavior only activates if hls_style attr was set to 'freerunning' by the InferSplitLayer transform (convert_to_hw_layers.py:1320); the op has no self-contained default, so cppsim correctness depends on prior transform ordering.
- **[hidden-coupling]** `split_hls.py:43` — execute_node statically re-dispatches to HLSBackend.execute_node to override the MRO (StreamingSplit is first parent and has its own numpy execute_node at split.py:146). The two execution paths (numpy golden vs cppsim/rtlsim) coexist and the correct one is selected by hard-coded class reference, not polymorphism.

## Seams (3)

- **clean-seam** — StreamingSplit (split.py) is a genuinely backend-agnostic base: it holds ALL op math (shapes, folding, datatypes, numpy execute_node, shape-compat) with zero HLS/RTL references. A backend variant only supplies the 4 HLSBackend abstracts + pragmas. This is a textbook op/backend split and an RTL variant could slot in cleanly.
- **fused-no-seam** — The cppsim drain logic (timeout_condition/timeout_read_stream in split_hls.py) is fused to HLSBackend's hls_style=='freerunning' branching (hlsbackend.py:228,533,587) and to strm{o} declarations. This multi-output freerunning drain is not expressible without the base backend's freerunning code path plus the externally-set hls_style attr — the op cannot control its own drain behavior in isolation.
- **fused-no-seam** — docompute + blackboxfunction + split.hpp template signature are a three-way positional contract (fold args, stream args, port order). Swapping the HLS implementation requires editing all three in lockstep; the C++ template arity is not derivable from the op contract alone.
