# Dataflow adapter architecture

FINN's dataflow stack has one contributor-facing hierarchy and two supporting
layers:

```text
DataflowOp declaration
    -> DataflowDesign declaration
        -> Kernel declaration
            -> private adapter compiler -> one flat DesignSpaceSpec

DataflowRegion / DataflowNetwork
    immutable engine-independent semantic IR values

ResolvedDataflowOp -> DesignRealization -> physical composition
    -> artifact-native handoff -> artifact lifecycle
```

## Canonical packages

`finn.dataflow.authoring`
: Public immutable declarations for Op, Design, and Kernel class bodies. It
  intentionally does not export mutable scopes, bound references, inventories,
  or compiled declaration records.

`finn.dataflow.design`
: Evaluation API for `Engine`, `DesignPoint`, answers, findings, requests, and
  `ResolvedDataflowOp`. The engine implementation remains private and
  domain-neutral.

`finn.dataflow.kernels`
: Public `Kernel`, `PhysicalComponent`, and parameter projection helper. Kernel
  compilation, coverage records, and binding helpers are internal.

`finn.dataflow`
: Immutable Region and Network values plus validation. Importing the semantic
  model does not load the engine, authoring layer, Kernels, or operations.

`finn.dataflow.artifacts`
: Downstream artifact identity and storage. The adapter-side MVAU handoff now
  constructs its existing identity values from immutable Kernel-origin
  projections rather than passing configured Kernels or a
  `DesignRealization`. The separately owned artifact-substrate integration
  will finish making this package a strict leaf; this adapter change does not
  modify that package.

`finn.dataflow.ops.mvau`
: The public MVAU operation and build-context façade. MVAU-specific logical
  association and physical composition remain in focused private modules.

## Compiler boundary

Class declarations are immutable templates. The compiler collects inherited
members in deterministic MRO order, validates layer capabilities, binds every
template beneath its use-site namespace, and never writes bound state back to
the class. A reusable Design or Kernel can therefore be compiled more than once
without aliasing paths.

The private compiled-operation record owns:

- the validated flat `DesignSpaceSpec`;
- graph/build projection plans and problem provenance;
- portable decision codecs;
- the closed Design inventory and selected Network/association handles;
- Design and Kernel exports, placements, constraints, and readiness groups;
- realization dispatch; and
- optional Design-owned physical composers and their declared inputs.

Operation-specific code does not construct raw engine declarations, rebuild
projection mappings, scan specifications to rediscover handles, or implement a
second selection/persistence lifecycle.

## Runtime states

The runtime values are deliberately separate:

1. A `DesignPoint` is an immutable problem snapshot plus sparse choices.
2. A `ResolvedDataflowOp` stores the selected Design id, resolved Network, and
   logical source association once. `NetworkRef` no longer exists.
3. Binding produces configured Kernels from exact Region and edge values.
4. A `DesignRealization` proves exact whole-Network coverage and preserves
   unabsorbed edge and boundary obligations.
5. The selected Design's registered composer receives a restricted
   `PhysicalCompositionContext`. It has no `Engine` or raw `DesignPoint`.
6. The resulting physical values are projected into artifact-owned values;
   configured Kernels and semantic design-space objects do not cross into the
   artifact package.

Partial specialization remains normal. Selecting a Design can leave folding,
supply, or Kernel-local choices unresolved. A semantic-only Design can resolve
its Network while physical realization reports that no Kernel is available.

## Region and Network

`DataflowRegion` owns one normalized logical schedule, input requirements,
output availability, operands, and typed interfaces. `DataflowNetwork` owns the
flat topology, edges, boundaries, and position maps. Neither contains Design,
Kernel, engine, source-occurrence, or artifact identity.

This separation permits:

- several Designs to share an equal Network but partition it differently;
- one Design to produce different Networks under conditional input supply;
- a Region to have zero, one, or several Kernel candidates; and
- one Kernel to cover several Regions and an edge.

## Admission and realization

Semantic recognition, graph-stage build admission, and resolved physical
feasibility are distinct questions. Candidate constraints are classified by
transitive problem provenance: graph-answerable rejections remove candidates,
while target/build-dependent questions remain deferred until their facts are
available. The same Kernel-owned constraints answer both stages.

Realization validates that every active Network node is covered exactly once,
absorbed edges have the declared endpoints and full fan-out, configured Kernels
belong to their placements, and remaining edges/boundaries are explicit
composition obligations.

## Persistence and provenance

The generic adapter owns one canonical fingerprint and portable codec path.
MVAU uses family version `mvau-dataflow-op-v7` and persistence format `1`; v6
and the retired MVAU-only v11 transport are rejected rather than reinterpreted.

Logical source association stops at semantic operands and coordinate mappings.
Kernel choices, placement identities, configured parameters, physical
components, connections, and local-state loading belong to physical
provenance. Equal semantic Networks and mappings therefore retain equal logical
associations when an equal-coverage Kernel changes.

## Verification

Run:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/check-dataflow-design.sh
```

Changes reaching physical composition also require the sequential fixtures
5–9 and both D6a numerical/hardware gates. Evidence must identify one clean
FINN revision, the pinned FinnLib revision, and the Python, pytest, Ruff, mypy,
and Vivado versions used.
