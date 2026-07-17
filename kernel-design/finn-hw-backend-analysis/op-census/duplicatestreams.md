# Census: duplicatestreams

*DuplicateStreams is a clean, simple fan-out (stream replication) op: an agnostic HWCustomOp base plus a single HLS backend wrapping finn-hlslib's StreamingDup, with no RTL variant and no rtllib coupling. Its only rough edges are multi-output special-casing (dict output counts, patched shape op) and diamond-MRO friction around execute_node, plus a couple of copy-paste/hard-coded-2-stream leftovers.*

**Files:** `src/finn/custom_op/fpgadataflow/duplicatestreams.py`, `src/finn/custom_op/fpgadataflow/hls/duplicatestreams_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `DuplicateStreams` | `duplicatestreams.py` | `(HWCustomOp)` | yes |
| `DuplicateStreams_hls` | `duplicatestreams_hls.py` | `(DuplicateStreams, HLSBackend)` | yes |

## Redesign pressure

This family is simple and largely clean: a pure fan-out copy op with one agnostic base and one HLS variant, no RTL variant, no finn-rtllib coupling, and no hermeticity violations. The single real friction with the 2-axis abstraction is the diamond-inheritance execute_node dispatch (DuplicateStreams_hls must explicitly re-route to HLSBackend.execute_node to beat the base's python copy via MRO), which is a recurring pain point for ops that carry both a functional reference impl and a backend impl. The secondary pressure is that the op is parametric in NumOutputStreams everywhere except derive_characteristic_fxns, which hard-codes exactly two output streams (out0/out1) - a latent bug if N!=2 and evidence the multi-output shape doesn't fit the single-output-centric base contract (get_number_output_values already had to return a dict, make_shape_compatible_op had to patch ret.output). A redesign should make multi-output a first-class dimension rather than something each method special-cases.

## Overrides (22)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | DuplicateStreams | 42 | Adds op-specific attrs NumChannels/PE/NumOutputStreams/inputDataType/numInputVectors, then merges base defaults. |
| `get_normal_input_shape` | DuplicateStreams | 62 | Shape derived from numInputVectors + NumChannels. |
| `get_folded_input_shape` | DuplicateStreams | 68 | Folds channels by PE with assert ch%pe==0; shape (vecs..., folds, pe). |
| `get_normal_output_shape` | DuplicateStreams | 77 | All output streams identical to input, so returns input shape regardless of ind. |
| `get_folded_output_shape` | DuplicateStreams | 82 | Same folded shape for every output stream, ignores ind. |
| `make_shape_compatible_op` | DuplicateStreams | 87 | Base builds a single-output const op; this op has N outputs so it patches ret.output to the node's full output list. |
| `get_input_datatype` | DuplicateStreams | 107 | Reads inputDataType attr. |
| `get_output_datatype` | DuplicateStreams | 111 | Output datatype equals input datatype (pure copy op). |
| `get_instream_width` | DuplicateStreams | 115 | PE * input bitwidth. |
| `get_outstream_width` | DuplicateStreams | 122 | PE * output bitwidth (same as instream). |
| `get_number_output_values` | DuplicateStreams | 129 | Returns a dict keyed per output stream (out0..outN) instead of a single scalar because the op has multiple output streams. |
| `get_exp_cycles` | DuplicateStreams | 135 | prod of folded output shape minus PE axis = number of transactions. |
| `execute_node` | DuplicateStreams | 139 | Abstraction-layer functional sim: copies single input to every output. |
| `verify_node` | DuplicateStreams_hls | 45 | Checks backend attr and presence of required attrs. |
| `execute_node` | DuplicateStreams_hls | 68 | Delegates to HLSBackend.execute_node for cppsim/rtlsim (overriding the pure-python copy in the agnostic base). |
| `global_includes` | DuplicateStreams_hls | 71 | Includes dup.hpp from finn-hlslib. |
| `defines` | DuplicateStreams_hls | 74 | No defines needed (empty). |
| `docompute` | DuplicateStreams_hls | 77 | Builds StreamingDup(in0_V, out0_V..outN_V) call with variadic output list. |
| `blackboxfunction` | DuplicateStreams_hls | 87 | Emits void top(in0, out0..outN) signature over hls::stream<hls::vector<T,PE>>. |
| `pragmas` | DuplicateStreams_hls | 101 | Adds one AXIS interface + aggregate pragma per output stream plus dataflow disable_start_propagation. |
| `timeout_condition` | DuplicateStreams_hls | 114 | AND of empty() over all N output streams for rtlsim timeout. |
| `timeout_read_stream` | DuplicateStreams_hls | 122 | Drains each of the N output streams into strmN in rtlsim. |

## Hacks (7 — 0 blocker, 2 major)

- **[major/brittle-assumption]** `duplicatestreams.py:158` — derive_characteristic_fxns hard-codes the output rtlsim dict to exactly two streams: {'out0': [], 'out1': []}. If NumOutputStreams != 2 the characterization override is wrong/incomplete despite the rest of the op being fully N-output-generic.
- **[major/inheritance-irregularity]** `duplicatestreams_hls.py:68` — execute_node is overridden solely to force HLSBackend.execute_node over DuplicateStreams.execute_node (the diamond MRO would otherwise resolve to DuplicateStreams' pure-python copy first). Explicit re-dispatch to sidestep the diamond is a fragile pattern.
- **[minor/duplicated-logic]** `duplicatestreams_hls.py:64` — verify_node error string reads 'The required GlobalAccPool_Batch attributes do not exist.' - copy-paste leftover from GlobalAccPool op; wrong op name in a DuplicateStreams error message.
- **[minor/hard-coded-param]** `duplicatestreams.py:156` — io_dict inputs hard-codes only 'in0' and the outputs list is fixed at out0/out1 - inconsistent with the parametric NumOutputStreams used everywhere else.
- **[minor/brittle-assumption]** `duplicatestreams.py:82` — get_normal/folded_output_shape silently ignore the ind argument assuming all outputs identical; a caller requesting a specific out index gets the input shape unconditionally. Comment 'output shape of both out streams' assumes exactly 2 streams even though the op supports N.
- **[minor/inheritance-irregularity]** `duplicatestreams_hls.py:39` — get_nodeattr_types manually calls DuplicateStreams.get_nodeattr_types(self) and HLSBackend.get_nodeattr_types(self) by explicit class name instead of super(), because the diamond inheritance would otherwise not merge both branches cleanly.
- **[minor/magic-number]** `duplicatestreams_hls.py:103` — '#pragma HLS dataflow disable_start_propagation' is an op-specific HLS incantation added ahead of the standard interface pragmas; undocumented why DuplicateStreams needs disable_start_propagation.

## Hermeticity violations (0)


## Seams (3)

- **clean-seam** — DuplicateStreams (agnostic base) is cleanly separable: all HLS specifics live in duplicatestreams_hls.py (global_includes/defines/docompute/blackboxfunction/pragmas/timeout_*). A new backend could subclass DuplicateStreams + <NewBackend> without touching the base. There is no RTL variant at all.
- **fused-no-seam** — execute_node is split across both axes: the base provides a python copy for the abstraction layer, but DuplicateStreams_hls.execute_node (line 68) must explicitly re-route to HLSBackend.execute_node to override the base via MRO. The functional-sim vs backend-sim behavior is fused through diamond inheritance rather than cleanly delegated.
- **fused-no-seam** — derive_characteristic_fxns (base, line 151) hard-codes a 2-output io_dict, coupling the agnostic base to a specific stream count and to rtlsim characterization details - base-layer logic that bakes in a backend/simulation assumption.
