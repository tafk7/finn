# DataflowOp native state and reconstruction (S2-A)

S2-A replaces the JSON state document with individual native ONNX attributes.
ONNX and build configuration remain the complete reconstruction authority.
There is no summary sidecar, persisted Space, or serialized operand mapping.

S3 subsequently converged MVAU on one flexible `DotProductDesign`. The generic
encoding algorithm and ActivationReplay remain schema 2; MVAU alone is schema
3 because its persisted inventory and qualified paths changed. See
[`dataflow_mvau_convergence.md`](dataflow_mvau_convergence.md).

Base FINN: `dfcede68be390338d7ede718b68995f79ed0b27d`.
QONNX: `e71f1c32ec1cd38e0bd27c3b7be7cb8469a35513`.

## Lifecycle and ownership

`DataflowOp.bind(model, build)` starts a root occurrence, supplies source and
build Problems, and hydrates committed native choices. Successors preserve the
same frozen Problems. `reconstruct()` discards choices while retaining those
Problems; `rebind(model, build)` explicitly reads the current graph. Omitting
`build` on rebind reuses the root's build Problems.

The existing root `ProblemSnapshot` owns source and build facts. The private
`_BoundNode` holds only copied node bytes, scope id, opset version, and output
observations. Each occurrence has its own materialized `NodeProto`. No bound
occurrence retains a ModelWrapper or build configuration. The `source` property
presents the root Problems alongside frozen output observations; it performs no
graph analysis. Querying, assigning, reconstructing, and planning write nothing.

Removed public values and APIs:

- `SourceBinding`, `BuildFacts`, `read_binding`, and the public `binding` property;
- `DataflowState`, `Assignment`, `DecisionCodec`, the JSON readers/writers, and
  `ops.state`;
- `TensorDatatypeFact`, `InitializerAnalysis`, and the initializer fingerprint
  opt-in;
- `SourceAssociation`, `OperandAssociation`, `OperandDestination`,
  `BoundaryDestination`, `StreamDestination`, `RegionStateDestination`, and
  `ops.association`;
- `InputTensor` / `OutputTensor`, replaced by `OpInput` / `OpOutput`;
- MVAU's unused `initializer_excludes_minimum` array reader. Its existing
  criterion is now derived from the QONNX summary at the operation.

The generic model, its pinned facade, Space, engine, Design and Kernel contracts
are unchanged. The operation specialization uses the existing compilation hook
and compiled choice walk to validate its native attribute schema.

## Attribute schema: generic and Replay version 2, MVAU version 3

Only three generic metadata attributes are written:

| Attribute | ONNX kind | Meaning |
| --- | --- | --- |
| `dataflow_scope_id` | STRING | Stable node addressing identity |
| `dataflow_problem_fingerprint` | STRING | Identity of the root Problems |
| `dataflow_schema_version` | INT | Concrete operation attribute encoding version |

The operation-local schema includes every compiled selector and Decision.
`name=` overrides a declaration's local path segment. The full root-relative
qualified path is encoded by replacing every `.` with `__`, preserving all
other legal segment characters. For example, `design.dot_product.pe` becomes
`design__dot_product__pe`; `name="PE"` on that Decision would produce
`design__dot_product__PE`. An existing segment containing `__` can collide with
this spelling: compilation refuses the collision, including collisions with
source attributes and generic metadata. There is no second persistence name.

The S3 MVAU schema is:

| Compiled path | Native attribute | Kind |
| --- | --- | --- |
| `design.case` | `design__case` | STRING |
| `design.dot_product.pe` | `design__dot_product__pe` | INT |
| `design.dot_product.simd` | `design__dot_product__simd` | INT |
| `design.dot_product.weight_supply` | `design__dot_product__weight_supply` | STRING |
| `design.dot_product.compute.kernel` | `design__dot_product__compute__kernel` | STRING |
| `design.dot_product.compute.dotp_axi.compute_pumping` | `design__dot_product__compute__dotp_axi__compute_pumping` | INT |
| `design.batch_interleaved.interleave` | `design__batch_interleaved__interleave` | INT |

MVAU writes `dataflow_schema_version = 3` for both Design alternatives. Replay
has `design__pe` and `design__simd`, no selector, and continues to write schema
2. The generic `DataflowOp.schema_version` also remains 2.

At the revision-pinned S2/C2 baseline, MVAU schema 2 mechanically used
`design.supplied.weight_supply` / `design__supplied__weight_supply` and
`design.supplied.compute.kernel` / `design__supplied__compute__kernel`. Those
are historical evidence paths, not aliases. Schema-2 MVAU records, including
old `dot_product` and batch-interleaved records, are refused and must be
recreated from source ONNX and build configuration.

Missing means unassigned. Only reachable decided choices are written. Committing
a partial point or switching branches removes all known choice attributes that
are absent from the new point. Hydration decodes all assignments before
submitting one compatible batch to the engine. The engine's dependency graph
orders selectors, applicability dependencies and Decision domains, including
parent domains that depend on child exports. Declaration-walk order is not a
replay order. Unsupported kinds, unreachable
assignments, stale fingerprints, duplicate attributes and unknown schema
versions are refused. Legacy JSON state is refused without a migration parser.

Ordinary bools use INT with exactly 0/1; ints must fit signed 64 bits. Strings
and Enum member values use STRING (integer Enum values have decimal spellings).
Simple homogeneous tuples use INTS, FLOATS or STRINGS. FLOAT attributes are
32-bit ONNX values: a Python float that would lose precision is refused. Authors
needing a wider representation must declare an `AttributeCodec`, for example a
STRING codec using exact hexadecimal floats. Structured Decisions likewise
require `canonical=AttributeCodec(identity, version, encode, decode, kind)`;
there is no automatic JSON fallback. Codec functions must round-trip the value.
Codec or declaration-path changes require an operation `schema_version` bump;
per-value codec tags are not stored.

`read_attributes(node)` inspects ordinary native values. `recorded()` reports
decoded reachable choices on a bound occurrence, keyed by qualified paths.

## One model-level initializer analysis

Use `bind_operations(model, build)` to reconstruct a model's DataflowOps or
`analyze_sources(model)` to inspect their sources. The model-level owner runs
QONNX `initializer_value_summaries` exactly once. Node readers consult that
result and never fetch initializer arrays. A standalone `op.bind` delegates to
the same owner for a pass containing that operation.

For an existing model-wide loop, use:

```python
from finn.dataflow.ops.reconstruction import source_analysis

with source_analysis(model):
    operations = [wrapper.bind(model, build) for wrapper in wrappers]
```

Nested calls reuse the one result for that exact ModelWrapper. A pass is a read
snapshot: after changing initializer values, exit it and begin a new pass.
`verify_nodes` establishes one such context around all DataflowOps. A later pass
always analyzes current values; no implicit model cache survives the context.
The commit precondition explicitly starts a fresh analysis even when called
inside an older context.

For model-wide inference, use FINN's explicit pass owners:

```python
from finn.dataflow.ops.inference import InferDataTypes, InferShapes

model = model.transform(InferShapes())
model = model.transform(InferDataTypes())
# Direct InferDataTypes().apply(model) also owns exactly one analysis.
```

These adapters establish a fresh `source_analysis` inside `apply`, around the
actual wrapper processed by QONNX's callbacks. Default `ModelWrapper.transform`
deepcopies its argument and may preprocess float64 initializers before `apply`;
an analysis context around the original wrapper cannot cover those callbacks.
Each `apply` iteration owns one bulk analysis, irrespective of operation count.
QONNX's repeat-until-unchanged behavior is preserved, so a changing datatype
pass followed by its unchanged check performs two analyses for two passes.
Graphs without DataflowOps use the original inference behavior without analysis.

This is an explicit FINN integration boundary, not a change to the accepted
QONNX revision: neither `ModelWrapper` nor QONNX's transformation classes are
patched. Raw QONNX inference classes have no FINN pass owner. Callers requiring
the one-pass guarantee must import the FINN adapters; wrapping a default-copy
`model.transform(...)` call in `source_analysis(model)` is not sufficient.
The DataflowOp inference fixtures now use these pass owners.

Every `OpInput` generates a separate optional `value_summary` Problem with the
explicit versioned `TENSOR_VALUE_SUMMARY_CODEC` (`finn.dataflow.tensor_value_summary@1`).
Every present initializer's digest also enters the operand's canonical identity
(`finn.dataflow.source_operand@2`). Unsupported present initializers propagate
QONNX's `UnsupportedTensorValueError`, including unsupported unused initializers
encountered during the model-level pass.

The summary codec includes all six facts. Extrema have explicit integer, float,
`+inf`, `-inf`, and `none` tags. Integer values remain integers. Finite fractional
floats use `float.hex()`. Integral floats and equal integers share an integer
record with an exact `float_hex` witness when representable; this preserves both
integer precision and equality such as `1 == 1.0`. Signed zero is normalized.
No non-finite JSON number is emitted. Infinite, all-NaN and empty summaries all
have distinct valid encodings and reconstruct from the ONNX initializers.

## Transactions and output authority

A `GraphEffects` plan carries scope id, expected source fingerprint, expected
choice/metadata attributes, attributes to remove/set, output repairs, and
invalidation names. Detached build Problems and the operation class let the
applier re-read source facts without holding an occurrence. Output identities
and source operator identity are narrow preconditions because repairs must not
be applied to an output that has since been retargeted.

The applier rechecks actual source facts, including initializer contents, before
writing. A concurrent choice change refuses the plan. Display-name changes and
unrelated node metadata do not. The write transaction includes the final rebind
returned by `commit`: a failure in hydration restores the complete serialized
model, including native attributes and output repairs, just as a write failure
does. Duplicate attribute names are rejected before filtering ONNX kinds,
including duplicates involving TENSOR attributes. The plan reports
`dataflow.implementation` invalidation when
native state changes; downstream artifact owners must act on that report.
Commitment stage is an in-memory planning check, never a node attribute.

Output shape/datatype annotations remain observations, including for fused MVAU.
MVAU now requires `outputDataType` as a source `DatatypeAttribute`/Problem.
Execution takes fused-threshold scale/bias from that attribute. Datatype
inference repairs the annotation from it. In no-activation mode the source
constraint requires `outputDataType == accDataType`.

## Operand mapping and MVAU continuation

`OpInput` / `OpOutput` declare Region-local `operand=` identity and coordinate
correspondence. `operand_references(accepted_network)` supplies source member
names mapped to tuples of `RegionInputRef` / `RegionOutputRef`. It supplies no
placement table. The generic `operand_mapping` property resolves those qualified
references through `model.refs` and derives exposure and position coverage with
`model.presentation`.

The operation obtains the Network through its accepted Design projection.
Unresolved or refused projections never reach presentation. Direct/synthetic
callers use `derive_operand_mappings`, which validates once before any query.
Individual presentation functions remain unchanged and perform no repeated
whole-Network validation.

Each `OperandMapping` has source operand/tensor, qualified semantic operand,
coordinate correspondence, source/semantic shape, derived placement, and edge,
boundary and unpresented position sets. `Internal` means a Region input with no
port; it makes no storage or physical provisioning claim. A streamed input can
still have unpresented positions, which remain visible in the mapping.

Current MVAU weight references are `compute.W` for external and embedded supply,
and `memory.W` for decoupled supply. The decoupled compute stream is a distinct
qualified reference. Reused bare operand ids never establish lineage. Replay
uses its explicit `replay.X` input/output references. Provenance remains the
operation's source attribute, available through `origin_nodes(source)`.

The S3 concrete MVAU migration preserves these rules:

1. Preserve required `outputDataType`; every accumulator-width transformation
   in no-activation mode must update it together with `accDataType`.
2. Consume `weight.value_summary` for value questions. Do not restore initializer
   array reads or per-operand fingerprint switches. QONNX owns summary and
   smallest-lossless-integer analysis; FINN owns implementation decisions.
3. Supply qualified references when Design roles change. Derive placement from
   the accepted Network, preserving unpresented residue.
4. Use existing `name=` for durable declaration names. MVAU's compiled-path
   change is represented by schema 3; there is no alias or migration reader.
5. The final inventory is `dot_product` and `batch_interleaved`. Dot product
   supply and compute candidate are explicit choices; initializer presence
   does not select either one.

## Software evidence

The complete `scripts/check-dataflow-design.sh --require-parity` gate passed
with these results:

| Check | Result |
| --- | --- |
| `tests/dataflow` with parity required | 1,643 passed, no skips |
| `tests/fpgadataflow/test_mvau_cycle_estimate.py` | 1 passed |
| `mypy --strict -p finn.dataflow -p finn.custom_op.dataflow` | 87 source files clean |
| Gate script's explicit typed-test/source file set | 29 source files clean |
| Ruff format and lint over the gate file set | 195 formatted files; lint clean |
| Diff over model, Space, engine, Design, Kernel contracts and root facade | Empty |
| Five independent pinned dependency clones | Clean, no symlinks or Git alternates |

The dataflow suite includes operation/conformance, real save/reload, inference,
summary fingerprint, scan-count, mapping, transaction, package-boundary and
parity evidence. Review regressions cover parent Decisions whose domains read
child exports, rollback after a post-write hydration failure, duplicate names
with unsupported ONNX kinds, and three real Replay callbacks per inference
pass. Inference coverage includes default deepcopies, optional float64
preprocessing, repeated fixed-point passes, and ordinary QONNX datatype options. Runtime tests used Python 3.10.12 in Docker
`xilinx/finn:env-CONFORMANCE`, with `FINN_SKIP_DEP_REPOS=1`, the target worktree
mounted, and its own QONNX checkout first on the runtime import path. The image
entrypoint was bypassed so no dependency checkout was installed or rewritten.
Missing software test tools were installed under `/tmp` and mounted separately.
Mypy 2.3.1 and Ruff 0.16.6 supplied static checks. Oracle parity used revision
`df2e42a20a2f4a2d852a861f2cf8af78e3e77cc5` from the read-only oracle mount.
No hardware gate or board repository was used.
