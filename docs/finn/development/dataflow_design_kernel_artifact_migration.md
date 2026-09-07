# S2-B: Design, Kernel and module artifact boundary

S2-B implements the accepted Design topology and Kernel artifact notes. A Design
explicitly admits Kernel candidates through `KernelChoice`; a Kernel's physical
projection produces one detached `ModuleBuildSpec`. Permanent physical absence
resolves without unused Decisions, while semantic acceptance still requires the
complete Region/Network constraint sets.

## Revisions and scope

The starting FINN revision is `dfcede68be390338d7ede718b68995f79ed0b27d`.
The implementation commits, in order, are:

| Revision | Change |
| --- | --- |
| `4c4c3b568c5e9883396e16923b91b5ffc05b6cc4` | Rename Design choices, topology declarations and module parameters; migrate consumers. |
| `7f1aed8c14b2b6409166ae7ec87e98089354e4a9` | Remove nominal computation contracts; introduce the detached module spec and stable string provenance. |
| `828648defa81b6e64b0a17fd229904c53ddca431` | Resolve permanent physical absence without choices; pin the Design validation boundary. |

This record's commit adds documentation and strengthens the artifact import-boundary
test. All implementation revisions are importable. Nothing was pushed.

Dependencies remain independent detached clones at their existing pins:

| Dependency | Revision |
| --- | --- |
| QONNX | `e71f1c32ec1cd38e0bd27c3b7be7cb8469a35513` |
| FinnLib | `dfeafac81cd2a6da27e647ee03915ade5532186e` |
| FINN Experimental | `0724be21111a21f0d81a072fccc1c446e053f851` |
| Brevitas | `aad4d5a293db6f2ec622a92a5d3278e47072453e` |
| HLSLib | `8d979e2bdced486dd25d26607d1ff5ae327ed6a8` |

The canonical `finn.dataflow.model`, its exact facade, `_engine`, and generic
`ops` behavior are unchanged. `finn.dataflow` still re-exports nothing.
The existing MVAU and replay Designs have only mechanical declaration/import
changes. MVAU constructor authority remains `ops.mvau.regions`; DotProduct
consolidation and Network-semantic changes remain S3 work. U6 still owns physical
composition and initializer provisioning. No board repositories or hardware tools
were needed or run.

## Public migration

There are no compatibility aliases.

| Previous surface | Current surface / migration |
| --- | --- |
| `designs.Kernels` | `designs.KernelChoice` |
| `designs.Connection` | `designs.NetworkEdge` |
| `designs.Sink` | `designs.EdgeSink` |
| `designs.Boundary` | `designs.NetworkBoundary` |
| `kernels.Parameter` | `kernels.ModuleParameter`, including `.constant(value, why=...)` |
| `kernels.KernelPhysicalResult` | `kernels.ModuleBuildSpec` |
| `ComputationContract` | Deleted; remove imports and declarations. |
| `Kernel.computation`, `KernelChoice(computation=...)`, `design.computation(role)` | Deleted; candidates are explicitly admitted by the Design author. |
| `finn.dataflow.computation` | Deleted, including `DOT_PRODUCT_COMPUTATION` and `ACTIVATION_REPLAY_COMPUTATION`. |
| `parameters.cyclic.computation`, `CYCLIC_PARAMETER_DELIVERY` | Deleted; the cyclic Region constructor remains. |
| Physical result `.kernel_id`, `.kernel_version` | `.implementation_id`, `.implementation_version` |
| Physical result `.build_unit` | `.abi.entry_point` |
| Physical result `.region_family`, `.region_version` | Inspect `KernelClass.region.family/version` or `design.region_family(role)`. |
| Physical result `.assignments` | Read Decisions from the occurrence; expose consumed scalar values with `ModuleParameter`. |
| Physical result `.imported_decisions` | Tuple of root-relative strings; stop calling `.value` on entries. |

`RegionDeclaration` keeps its C1.5 spelling and ownership in `kernels`.
Canonical Network values (`Edge`, `SinkContract`, `BoundaryContract`) keep their
model names; the renamed topology declarations live in `designs`.

Migrate imports to `finn.dataflow.designs` and `finn.dataflow.kernels`, replace
the names above, and remove every `computation=` argument. Keep existing `role=`,
`node_id=`, `name=`, port ids and Input bindings. Candidate ids continue to derive
from `Kernel.id`, with `Subspace(..., name=...)` providing an explicit alias.
A singleton adds no selector, and adding alternatives preserves existing
candidate paths. `ModuleParameter` preserves both sourced and constant forms and
the exact ABI parameter-table check.

## Detached module spec

`ModuleBuildSpec` has exactly these fields:

```text
implementation_id: str
implementation_version: str
region: DataflowRegion
parameters: Mapping[str, Scalar]
abi: ComponentABI
contributions: tuple[Contribution, ...]
render_context: Mapping[str, Scalar]
imported_decisions: tuple[str, ...] = ()
```

The physical evaluator constructs this value from declared resolved inputs.
`component_abi(parameters)` and `render_context(parameters)` receive the scalar
parameter table. The mappings are copied and frozen; non-scalar context values
and non-string provenance are refused. No occurrence, Engine, point, compiler
reference, ONNX node, model wrapper or assignment ledger crosses this boundary.
The Region is a semantic witness, without duplicate family or computation tags.

The compiler retains its root namespace privately so provenance uses the same
root-relative declaration names as persistence. Input references preserve their
supplier's root during direct fragment compilation. These are compiler metadata
only: no change was made to the engine, persisted value encoding, or model facade.
Tests cover explicit declaration names, selectors and multi-segment root namespaces.

Obtain the spec from `kernel.physical.accepted_answer` after checking it is
`Decided`, then pass its value to the existing adapters:

```text
resolve_kernel_contributions(spec, roots=..., template_roots=...)
kernel_source_derivation(spec, resolved)
portable_kernel_component(spec, source_artifact, resolved)
```

Those adapter names remain valid. Artifact services receive detached values;
they import no Space, Engine, model, Kernel, Design, operation or ONNX model
machinery. Initializer arrays are not added to the spec: data bindings remain
inputs to stages that explicitly consume them.

## Semantic acceptance and physical refusal

`design.dataflow.accepted_answer` is the accepted Network boundary. Before it
returns `Decided`, canonical `validate_network`, Design support, selected Region
validity, and selected-Kernel semantic constraints must all accept. Shared
constraints remain semantic even if also listed in `physical_support`.
Unselected alternatives remain inapplicable.

`design.network` and projection `.output` are construction/diagnostic answers.
Use the accepted projection for correspondence and presentation. Presentation
queries consume that validated Network without repeating whole-Network validation.
Regressions cover missing edge sources, mismatched sequences, boundaries aimed at
internal inputs, and structurally valid Networks refused by semantic support.
The valid case resolves reused Region-local operand ids by qualified reference,
distinguishing an unpresented internal requirement from an edge-presented input.

For an implementation that cannot supply any standalone module, declare:

```python
from typing import ClassVar

from finn.dataflow.kernels import PhysicallyUnsupported

physical_unavailable: ClassVar[PhysicallyUnsupported | None] = PhysicallyUnsupported(
    "this implementation has no standalone module"
)
```

This makes the physical projection immediately `Absent`: its output has no
parameter dependencies and its readiness has no unused Decisions or constraints.
The semantic projection still checks every semantic obligation. A containing
realization can consume the Region without evaluating an unavailable child spec.
An unavailable Kernel needs no dummy ABI. A subclass implementing the module
resets `physical_unavailable = None` and supplies `component_abi`.

For parameter-specific physical refusal, continue raising `PhysicallyUnsupported`
from `component_abi`; those parameters must resolve first. Semantic constructor
refusal continues to use `RegionRefused`.

Embedded dot product, batch-interleaved dot product and memstream now declare
permanent absence. Their Regions and Networks remain unchanged, and inherited
pumping choices are no longer prerequisites for discovering absence.

## Artifact identity evidence

The public spec schema changes, so serializers of the old physical-result
dataclass must migrate. Source and package derivation identities do not change:
the adapters already keyed only consumed inputs, and never read the removed
computation, assignment or Region-label fields. `kernel-source`,
`kernel-source-v1` and `finn.kernel.<implementation_id>` remain the registered
artifact kind, schema and producer vocabulary.

A source archive of C1.5 and this implementation were evaluated in separate Python
processes against this worktree's same pinned QONNX/FinnLib clones. The comparison
used the existing DotpAxi and ReplayBuffer fixtures at PE=2, SIMD=4, with both
pumping values for DotpAxi, and the `finn.rtl-module-directory` package format for
`test-part`. Source keys, package keys and every generated file digest matched:

| Fixture | Source key | Package key |
| --- | --- | --- |
| DotpAxi, plain | `16f7471058f4cdda2992519ceecc127eda31f5eb49f198c47935332c45eefef4` | `8cd89a69667b2052c8adcc87cb2eca8a388a59fe585b7c06f6cac72a5bb4c4b3` |
| DotpAxi, pumped | `16f7471058f4cdda2992519ceecc127eda31f5eb49f198c47935332c45eefef4` | `4bd274276137b1f98b79c56b616f7fc98718342d6e4ab0081378c09a8f7f5793` |
| ReplayBuffer | `a99af22351bdeb3b181062aa0c84b989511a60726f5a2364ce0cb77c74f4b1a9` | `8c9811dbb98eb71d5f325128e22b0bb17fefef718e5c16416a169df5fb847b64` |

The existing artifact identity, data-binding and packaging regressions pass.
Additional evidence proves equal specs under distinct root namespaces, source-key
independence from provenance, and frozen render-context snapshots.

## Validation

Software ran in the existing image
`xilinx/finn:v0.10.1-2342-g546538087c71.xrt_202420.2.18.179_22.04-amd64-xrt`
with `FINN_SKIP_DEP_REPOS=1`, network disabled, the target worktree's pinned
dependencies on `PYTHONPATH`, and the oracle mounted read-only. Required parity
used oracle `df2e42a20a2f4a2d852a861f2cf8af78e3e77cc5`.

| Check | Result |
| --- | --- |
| `tests/dataflow`, with `FINN_DATAFLOW_REQUIRE_PARITY=1` | 1,637 passed; one mypy-dependent test skipped in the image. |
| Skipped descriptor typing test, with the existing mypy environment mounted | 1 passed. All 1,638 tests are covered across the two runs. |
| `tests/fpgadataflow/test_mvau_cycle_estimate.py` | 1 passed. |
| Strict mypy: `finn.dataflow` and `finn.custom_op.dataflow` | 82 source files clean. |
| Strict mypy: gate script's typed test set, including the new boundary regressions | 31 source files clean. |
| Ruff format/check over the gate script's source set | 188 files formatted; all lint checks passed. |

The checking tools were mypy 2.3.1 and ruff 0.16.4. Python was 3.10.12 in
Docker; host strict typing used Python 3.12.3. The strengthened package tests
also pass separately, including fresh imports and the unchanged exact model
facade. No hardware evidence is claimed.

The local evidence bundle is `/tmp/finn-s2b-evidence/`: software and descriptor
typing logs, ruff logs, and the C1.5/current artifact key and generated-digest
comparison script/results. `scripts/check-dataflow-design.sh --require-parity`
remains the reproducible software gate; its strict test list includes both new
regression modules.
