# Dataflow design architecture

*Current as of FINN `f0e06a96bbd4a05b913a1e6f33d34844115d3947`, the C1.5–S3
simplification checkpoint. Sections that describe a migration are dated and
name the revision they record.*

FINN's dataflow stack separates the semantic model, the generic design-space
language, physical Kernels, Designs, operations, and artifacts.

```text
finn.dataflow.model      detached DataflowRegion / DataflowNetwork values
finn.dataflow.space      the generic Space language, compiler and occurrences
                              |
              +---------------+---------------+
              |                               |
finn.dataflow.kernels            finn.dataflow.designs
    Kernel, RegionDeclaration,       DataflowDesign, KernelChoice,
    ModuleParameter,                 NetworkEdge, NetworkBoundary
    detached ModuleBuildSpec                     |
              |                                  |
              +---------------+------------------+
                              |
ONNX/QONNX --> ops.base.DataflowOp  (root Space *and* CustomOp)
                              |
                              v
                  finn.dataflow.artifacts   (detached; one-way)
```

`model` and `space` are independent. Generic `space` does not import `model`;
its one model-aware bridge is the private `space.dataflow_value_semantics`,
which the public `space` facade excludes. `finn.dataflow` itself re-exports
nothing, so every value has exactly one import path.

## Canonical packages

`finn.dataflow.model`
: The canonical `DataflowRegion` and `DataflowNetwork` values, their datatype
  boundary, construction profiles, validation and presentation queries. It
  imports neither `space` nor the engine.

`finn.dataflow.space`
: The generic Space declaration language, compiler and occurrence runtime:
  `Problem`, `Input`, `Decision`, `@derived`, `@constraint`, `ConstraintGroup`,
  `Readiness`, `Projection`, `ProjectionAssessment`, `Subspace`,
  `SubspaceChoice`, `ChoiceView`, `BranchCatalog` and `compile_space_model`.
  `finn.dataflow._engine` remains private and domain-neutral.

`finn.dataflow.kernels`
: The `Kernel` contract — one `RegionDeclaration`, physical-only Decisions,
  `ModuleParameter`, `ComponentABI`, an ordered source closure — the two
  projections, the detached `ModuleBuildSpec` they hand downstream, and the
  reusable `DotpAxiKernel`, `ReplayBufferKernel` and `MemstreamKernel`.

`finn.dataflow.designs`
: The generic `DataflowDesign` contract and topology vocabulary: `KernelChoice`,
  `NetworkEdge`, `EdgeSink`, `NetworkBoundary` and the synthesized
  `SelectedNetwork`. MVAU-specific Designs do not live here.

`finn.dataflow.ops`
: `ops.base.DataflowOp` — simultaneously a QONNX `CustomOp` and the root Space
  for one ONNX node — plus the source schema, native persistence,
  reconstruction, `OperandMapping` and the FINN inference adapters. The package
  `__init__` re-exports nothing (`__all__` is empty), so import from the leaf
  that owns the value. `ops.mvau` and `ops.replay` are the two production
  operations and own their Design inventories.

`finn.dataflow.artifacts`
: Generic artifact identity, contributions, derivation, packaging and checked
  storage. It imports nothing from `space`, `model`, `kernels`, `designs` or
  `ops`, and receives detached specifications only.

`finn.dataflow.conformance`
: One lifecycle check every `DataflowOp` must pass, written so that a *third*
  operation would pass it.

## Authoring boundary

A contributor authors ordinary Python classes. Declarations name their
dependencies explicitly and callbacks receive resolved declared values only.
No public operation returns an `Engine`, a `DesignPoint`, a `_Ref`, a compiled
record or an unrestricted path lookup, and contributor code may not:

- construct raw engine `Decision`, `DerivedProperty`, `Constraint`,
  `ProblemField`, `DependencyRef` or `EvaluatorSpec` values;
- reconstruct a qualified path from a path/kind/semantics triple;
- inspect the flat spec's declarations to recover handles; or
- re-aggregate constraints, readiness or selected-Kernel metadata that the
  Design occurrence already owns.

`assign`, `answer`, `assess` and `project` accept declarations, never path
strings. The generic implementation still compiles class bodies into one
ordinary flat `DesignSpaceSpec`; no new engine primitive is introduced.

A choice that changes any selected Region belongs to the enclosing Design and
reaches a Kernel as a typed `Input`. A Kernel-local `Decision` may only change
physical realization, and the compiler refuses one that appears in the Region
property's transitive closure. That refusal is a migration guard against
duplicating one shared folding axis across sibling Kernels; relaxing it needs a
Design-level shared-axis ownership contract that does not exist yet.

## Network-only operation results

Every production `DataflowOp` resolves to a `DataflowNetwork`. A Design with one
Region returns the canonical singleton Network; there is no Region-or-Network
result alternative.

Consumers use `design.dataflow.accepted_answer` (or `op.network`, which is that
answer). `design.network` and a projection's raw `output` are construction and
diagnostic values: they do not establish semantic acceptance. Presentation and
operand-correspondence queries run on the already-validated Network and do not
repeat whole-Network validation.

The Network is semantic and flat: nodes are Regions, edges carry exact
position/beat correspondence, and boundaries expose logical interfaces.
Physical components, clocks, buses and tool state appear only in the separate
physical projection.

## Kernel candidates and admission

A `KernelChoice` segment owns its candidate Kernel classes explicitly. There is
no global Kernel registry — and, equally, **no graph-stage candidate-elimination
pass**. Admission is two things and no more:

1. **Authoring-time membership.** A candidate is admitted because it is listed
   in a `KernelChoice`, checked by `KernelChoice.validate_candidate` when the
   Design class is compiled. Nothing filters candidates against graph facts, and
   nothing scores or ranks them.
2. **Validation of what was selected.** Once the selector is committed, the
   selected candidate's constraints are evaluated as part of the enclosing
   projection — semantic ones through `design.dataflow`, physical ones through
   `kernel.physical`.

Choosing among candidates is a specialization policy, and no such service exists
here; U7 owns it. Until then a candidate is selected by an explicit assignment
like any other Decision, and `finn.dataflow.ops.inference` supplies only FINN's
shape and datatype adapters — it admits nothing and eliminates nothing.

The Design's Network constraint set holds the Design's own constraints **and
every candidate constraint that is not physical-only**, where physical-only is
the difference `physical_support − dataflow_support`. A Kernel refusing its
Region therefore refuses the Network, while one that merely cannot be *built*
at this configuration still contributes a Region. An unselected candidate's
constraints report as not applicable, not as refusals.

A permanently unrealizable implementation declares
`physical_unavailable = PhysicallyUnsupported("reason")`; its physical
projection is `Absent` immediately, without a dummy ABI and without first
resolving inherited physical Decisions. Its semantic projection still runs.

DotpAxi, ReplayBuffer and the RTL memstream are reusable concrete Kernels. None
imports a `DataflowOp`, a `DataflowDesign` or an MVAU operation implementation,
and direct evidence configures each from a flat engine point.

They do each import one thing from `ops.mvau`: the pure Region constructors in
`finn.dataflow.ops.mvau.regions`
(`construct_activation_replay_region`, `construct_weight_stream_region`, and the
dot-product constructors). That is deliberate — C1.5 established one MVAU
Region-constructor authority rather than letting each Kernel restate the same
canonical construction — and it is a dependency on a pure function over
scalars and datatypes, not on an operation, an occurrence or a graph. A Kernel
that needed a *different* Region would declare its own constructor.

## MVAU design inventory

After S3, MVAU has exactly two production Designs, and weight supply is a
Design-owned choice on the dot-product one:

```text
design.case                            = dot_product | batch_interleaved
design.dot_product.weight_supply       = external | embedded | decoupled
design.dot_product.compute.kernel      = dotp_axi | dotp_axi_embedded
design.dot_product.pe, .simd
```

`dot_product@2` is the production replay-plus-dot-product Design. External
supply presents `compute.W` at a boundary, embedded supply retains `compute.W`
as an unpresented `InternalInput`, and decoupled supply admits an unpresented
`memory.W` plus an edge-presented, distinct `compute.W`. Supply and compute
candidate are both explicit choices; initializer presence only constrains the
two modes that retain weights locally.

`batch_interleaved` is a valid semantic singleton-Network design without a
production physical candidate. It may be selected for modeling, but automatic
build admission does not claim it is physically realizable.

Inactive supplier Regions and segments are `Absent`. `BatchInterleavedDesign`
retains its deliberately Design-owned `interleave` choice; the audit of that
placement — Region locality, independent Kernel reuse, forwarding cost and
shared-axis duplication — is recorded and deferred, not discharged.

## Persistence and identities

Current DataflowOp persistence writes native Decision attributes. The generic
default and Replay schema remain 2; MVAU is schema 3 after the Design inventory
and qualified-path migration. Regions, Networks, configured Kernels, and
artifacts are recomputed rather than serialized into ONNX.

Artifact keys contain only values the corresponding artifact stage reads.
Occurrence namespaces, source-node identity and filesystem locations do not
enter portable identity. Root-relative imported-Decision provenance travels on
`ModuleBuildSpec` but is deliberately excluded from those keys, because no
artifact stage consumes it; S3's provenance rename therefore changed no key.

## Deliberate boundary

Physical composition is a contract, not an implementation. There is no
`DesignPhysicalResult`, no `ModuleInstance`, no wrapper generation and no
composed packaging in this tree, and no specialization policy, ONNX lowering or
Region CustomOp either. `RegionDeclaration.family`/`version` and the Design's
role-to-node metadata preserve the seam a future annotated-ONNX carrier needs;
no ONNX object reaches the engine.

The internal Provider/semantic-Kernel framework, the old MVAU compatibility
tree, the old supply Kernels and the Region-or-Network result alternative are
deleted without aliases, and package-boundary tests pin the retired module
paths as unimportable and unreferenced.

External legacy FINN `MVAU_hls`/`MVAU_rtl` HWCustomOps remain available as
independent comparison oracles. They are not a production compatibility path
for the `DataflowOp` stack.

## Verification

Run the software gate with:

```bash
scripts/check-dataflow-design.sh                    # developer run
scripts/check-dataflow-design.sh --require-parity   # gate evidence
```

Without `--require-parity` the oracle-parity tests skip and the script says so;
with it, an absent or wrong-revision oracle is a failure rather than a silent
pass. The gate runs `tests/dataflow`, the MVAU cycle regression, Ruff
format/lint and strict mypy over the production packages and a focused test set.

Changes to consumed RTL source, module parameters, an ABI, a source derivation
or numeric behaviour additionally require the standalone Vivado XSI and
out-of-context synthesis matrices; see `CLAUDE.md` for how to run them. Closure
evidence records one clean FINN revision, the pinned FinnLib revision,
Python/pytest/Ruff/mypy versions and the Vivado version.

## Engine migration provenance

The private engine was migrated from the Project Kernels scratchpad at revision
`fba51ae01f26c1d53cf89ec51cf6bb88b2e4cbec`. The combined SHA-256 digest of
the source, tests, and examples used as the migration baseline was
`1b4c3156594c8b1ef1f067b0e25e083b12f0145ef970f84e534374f410432e96`.

On 2026-08-26, the original author confirmed that the engine was their original
work, written for inclusion in FINN, and authorized its distribution under
FINN's BSD-3-Clause license. The standalone `design_space` package is retained
only as historical migration provenance; `finn.dataflow._engine` is the live
private implementation.
