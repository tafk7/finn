# Census: streamingdataflowpartition (StreamingDataflowPartition meta/container node)

*StreamingDataflowPartition is a meta/container placeholder node (inherits qonnx CustomOp directly, not HWCustomOp) that wraps a partitioned FINN sub-model and executes it by recursively calling execute_onnx on a model loaded from a filesystem path; it emits no HLS or RTL and participates in none of the HW backend contract.*

**Files:** `src/finn/custom_op/fpgadataflow/streamingdataflowpartition.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `StreamingDataflowPartition` | `streamingdataflowpartition.py` | `(CustomOp)` | NO (backend-only) |

## Redesign pressure

This family is an outlier that does not belong to the HW backend abstraction at all: a single class inheriting qonnx CustomOp directly (not HWCustomOp), with no HLS or RTL variant, no folding/datatype/stream-width contract, and no code generation. It only carries floorplan/partition metadata and executes by recursively interpreting a referenced ONNX sub-model from a filesystem path. The stale TODO at line 34 ('move to HLSCustomOp base class') is the main pressure signal — it reflects an unresolved discomfort that this container sits inside the fpgadataflow custom_op package yet obeys none of its contracts. A redesign should explicitly model it as a graph-partition/container node outside the HLS/RTL op hierarchy rather than trying to force it onto the backend abstraction; its brittle attribute-count check (line 90 vs. 10 declared attrs) and positional I/O renaming should be cleaned up regardless.

## Overrides (4)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | StreamingDataflowPartition | 43 | Defines container-only attrs (filesystem 'model' path, res_estimate/res_hls/res_synth strings, slr/partition_id/device_id/mem_port floorplan metadata, return_full_exec_context flag) — none are folding/datatype/stream-width params. |
| `make_shape_compatible_op` | StreamingDataflowPartition | 57 | Container node has no intrinsic shape inference; the wrapped sub-model owns shapes. |
| `execute_node` | StreamingDataflowPartition | 63 | Not a HW op: instead of codegen/cppsim/rtlsim it loads the referenced sub-model from disk and runs it recursively via execute_onnx, then copies renamed I/O tensors back into the parent context. |
| `verify_node` | StreamingDataflowPartition | 86 | Validates that the 'model' attribute exists and that there is >=1 input. |

## Hacks (6 — 0 blocker, 4 major)

- **[major/todo-marker]** `streamingdataflowpartition.py:34` — Module-level comment 'TODO move StreamingDataflowPartition to HLSCustomOp base class'. Stale/misguided intent — the op is a pure meta-container with no HLS/RTL emission, so rebasing it onto an HLS backend base would be wrong. Signals the op is an unresolved misfit in the fpgadataflow custom_op package.
- **[major/inheritance-irregularity]** `streamingdataflowpartition.py:37` — Inherits qonnx CustomOp directly, NOT HWCustomOp/HLSBackend/RTLBackend. It lives in the fpgadataflow custom_op family but participates in none of the HW backend contract (no get_input_datatype, get_instream_width, folded shapes, etc.). It is a graph-partition placeholder, not a hardware op.
- **[major/magic-number]** `streamingdataflowpartition.py:90` — verify_node hard-codes 'num_of_attr = 1' and reports 'number of attributes is incorrect' unless exactly one attribute is present, yet get_nodeattr_types (lines 43-55) declares 10 attributes. Any node carrying the optional floorplan/resource attrs (slr, partition_id, res_synth, ...) will fail this brittle count check.
- **[major/brittle-assumption]** `streamingdataflowpartition.py:69` — Input renaming maps parent node.input[i] to model.graph.input[i].name purely by positional index (lines 69-73); likewise output renaming maps node.output[i] to model.graph.output[i].name by index (lines 76-78). Assumes the partition preserved exact positional I/O ordering — any reorder silently binds wrong tensors.
- **[minor/brittle-assumption]** `streamingdataflowpartition.py:57` — make_shape_compatible_op and infer_node_datatype (lines 57-58, 60-61) are bare 'pass' no-ops returning None, unlike the contract expectation that make_shape_compatible_op returns a valid ONNX helper node. Callers must special-case this op.
- **[minor/other]** `streamingdataflowpartition.py:80` — return_full_exec_context path (lines 80-83) prefixes every internal sub-model tensor with node.name + '_' and injects it into the parent context — leaks the entire private execution state of the partition into the caller's namespace, an opt-in but unbounded side-channel.

## Hermeticity violations (3)

- **[filesystem-path]** `streamingdataflowpartition.py:64` — execute_node loads a whole sub-model from a filesystem path stored in the 'model' nodeattr via ModelWrapper(self.get_nodeattr('model')). Execution correctness depends on that .onnx file still existing on disk at the recorded path — ambient filesystem state, not carried in the graph.
- **[hidden-coupling]** `streamingdataflowpartition.py:74` — Calls finn.core.onnx_exec.execute_onnx (imported line 32) to recursively interpret the referenced sub-model, pulling the entire FINN execution engine into this custom_op's execute path — deep coupling well beyond a single-node execute.
- **[hidden-coupling]** `streamingdataflowpartition.py:78` — Mutates the caller-supplied 'context' dict in place (lines 78 and 83) rather than returning outputs; correctness depends on the parent execute_onnx loop's ordering and shared-dict convention.

## Seams (1)

- **fused-no-seam** — There is no HLS/RTL axis to seam here at all: StreamingDataflowPartition has zero backend variants and emits no code. It is a graph-level meta-container executed by recursively running its wrapped sub-model. The '2-axis (HLS/RTL)' abstraction simply does not apply to it — nothing to swap, nothing to separate.
