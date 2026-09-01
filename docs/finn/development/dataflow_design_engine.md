# Dataflow design architecture

FINN's dataflow stack separates semantic models, declaration-time authoring,
evaluation, physical Kernels, and artifacts.

```text
Region / Network model -----------+
                                   |
private design engine ------------+--> declaration-time authoring
                                         |
                                         +--> flat DesignSpaceSpec
                                         +--> DataflowDesign inventory
                                                   |
ONNX/QONNX integration --> DataflowOp -------------+
                                                   |
                                                   v
                              DesignPoint / NetworkRef / realization
                                                   |
                                                   v
                                   elaboration --> artifacts
```

The Region/Network model and the private engine are independent inputs to the
authoring layer. Neither depends on the other.

## Canonical packages

`finn.dataflow`
: Semantic Region and Network values plus validation. Importing these values
  does not load the engine, authoring, Kernels, or operations.

`finn.dataflow.authoring`
: Declaration-time API for `DataflowOp`, `DataflowDesign`, scopes, typed
  `Ref` values, domains, rejection/unresolved helpers, and conditional input
  supply.

`finn.dataflow.design`
: Evaluation-time API for `Engine`, `DesignPoint`, answers, findings,
  requests, diagnostics, `NetworkRef`, and `ResolvedDataflowOp`.
  `finn.dataflow._engine` remains private and domain-neutral.

`finn.dataflow.kernels`
: Physical `Kernel`, `KernelScope`, coverage, configured bindings, components,
  and source manifests. Compiled declarations and candidate-selection records
  are private implementation details.

`finn.dataflow.artifacts`
: Generic artifact identities and checked storage. Operation-specific payload
  rendering and tool-stage state remain with the operation that owns them.

`finn.dataflow.ops.mvau`
: The public MVAU `DataflowOp` and build-context façade. Internal declarations,
  projection, persistence, realization, elaboration, and artifact payloads live
  in precise leaves beneath this package.

`finn.dataflow.testing`
: Reusable contributor conformance checks, including the Network-only
  operation lifecycle and fresh-import/declaration-boundary helpers.

## Authoring boundary

Operation-specific authoring may declare domain decisions, properties,
constraints, and explicit dependencies. It may not:

- construct raw engine `Decision`, `DerivedProperty`, `Constraint`,
  `ProblemField`, `DependencyRef`, or `EvaluatorSpec` values;
- reconstruct a `Ref` from a path/kind/semantics triple;
- inspect `spec.decisions` or `spec.constraints` to recover handles;
- manually aggregate constraints, readiness, or selected Kernel metadata that
  the design inventory already owns; or
- call `assemble_specs` to rebuild generic inventory structure.

`DataflowDesignInventory` is authoritative for its design-selection handle,
selected Network, active placements, configured Kernel identities, active
decision metadata, constraint aggregation, and realization readiness.
Structural metadata is available before physical choices such as compute
pumping are committed; fully configured parameters remain realization/artifact
facts.

The generic authoring implementation may use raw engine declarations internally
to compile the contributor-facing scopes into one ordinary flat
`DesignSpaceSpec`. No new engine primitive is introduced.

## Network-only operation results

Every production `DataflowOp` resolves to `finn.dataflow.design.NetworkRef`.
A design with one Region returns a singleton Network. `RegionRef` and the old
`RegionRef | NetworkRef` operation boundary do not exist.

The Network is semantic and flat: nodes are Regions, edges carry exact
position/beat correspondence, and boundaries expose logical interfaces.
Physical components, clocks, buses, and tool state are introduced only after a
design point is resolved.

## Kernel candidates and admission

A design placement owns its candidate Kernel classes. There is no global
Kernel registry. Candidate-backed admission answers three separate questions:

1. **Semantic recognition:** does the graph describe the source operation?
2. **Graph-stage build admission:** does every active placement retain at least
   one candidate after evaluating all graph-answerable coverage constraints?
3. **Resolved physical feasibility:** after target/build/design/Kernel choices
   are known, do all selected Kernel constraints pass?

Constraint classification follows transitive problem provenance. A graph-pure
rejection eliminates a candidate. Missing graph-owned information is
unresolved. Target/build-dependent constraints are explicitly deferred during
inference and must later return a positive verdict. Both stages invoke the same
Kernel-owned predicates.

DotpAxi, ReplayBuffer, and FINN RTL memstream are reusable concrete Kernels.
Their input records contain computation/interface, target, and physical facts,
not MVAU node roles, source policy, decision paths, or persisted attributes.
Synthetic non-MVAU designs bind all three as promotion conformance cases.

## MVAU design inventory

MVAU declares one operation-owned weight-supply policy and two designs:

```text
design = dot_product | batch_interleaved
weight supply = external | finn_rtl_memstream
```

`dot_product` is the production replay-plus-dot-product Network. Its compute and
replay placements use the shared DotpAxi and ReplayBuffer Kernels. Selecting
`finn_rtl_memstream` conditionally adds the cyclic parameter-delivery Region,
edge, boundary changes, local-state association, and memstream placement.

`batch_interleaved` is a valid semantic singleton-Network design without a
production physical candidate. It may be selected for modeling, but automatic
build admission does not claim it is physically realizable.

Inactive supplier Regions and placements are `Absent`. Active placement
realization validates exact node coverage, absorbed edges (including complete
fan-out), unabsorbed connection obligations, boundaries, and configured Kernel
candidates.

## Persistence and identities

MVAU persistence is v6 at the node-attribute layer and v11 at the source
envelope layer. v5/v10 is rejected explicitly. Regions, Networks, configured
Kernels, and artifacts are recomputed rather than serialized into ONNX.

Artifact identity excludes source occurrence and placement. Stable encoded
tokens preserve pre-move identities such as
`finn.dataflow.mvau_problem.MVAUDspBlock`; Python package cleanup does not move
v6 fingerprints or artifact keys.

MVAU's artifact implementation has explicit modules for wrapper rendering,
source staging, packaged-unit state, OOC synthesis, IP-XACT packaging, and
memstream-specific payloads. Prepared and completed tool states are distinct
types, and every completed artifact is checked against its declared layout.

## Compatibility boundary

The internal Provider/semantic-Kernel framework, old MVAU compatibility tree,
old supply Kernels, and Region-or-Network result alternative are deleted. The
`finn.dataflow.kernels` name now means physical Kernels only.

External legacy FINN `MVAU_hls`/`MVAU_rtl` HWCustomOps remain available as
independent comparison oracles. They are not a production compatibility path
for the new `DataflowOp` stack. The fused `MvuVvuAxiKernel` remains test-only.

## Verification

Run the focused software gate with:

```bash
./scripts/check-dataflow-design.sh
```

Physical or artifact-stage changes additionally require sequential Vivado
fixtures 5–9 and both memstream gates. Closure evidence records one clean FINN
revision, the pinned FinnLib revision, Python/pytest/Ruff/mypy versions, and
Vivado 2025.2.

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
