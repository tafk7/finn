# Dataflow design engine

The supported authoring surface for dataflow design spaces is
`finn.dataflow.design`. Domain adapters should not import
`finn.dataflow._engine` directly. The private engine remains domain-neutral and
uses only the Python standard library; region knowledge is confined to the
public design adapter.

The model-aware source-operation surface is `finn.dataflow.authoring`.
`DataflowOp` subclasses are loaded through
`ModelWrapper.get_customop_wrapper`, project live graph and build facts into a
problem instance, and persist only explicitly committed decisions as node
attributes. See `../implementation/dataflow-op.rst` for the contributor guide.
The implementation began from reviewed FINN baseline
`28ed4d9736b2447471995f139db8f3e068d262ce`. Its model-aware QONNX dependency
is pinned to `tafk7/qonnx` commit
`46b69021e3a38b57c636f6941a52d809c8928a7b`.

A concrete microarchitecture is selected once, as a Kernel. `MVAU_COMPUTE_SELECTION`
is the compute pool and holds four: `legacy_hls`, `rtl_softvec`, `rtl_packed`,
and `rtl_batch_interleaved_dsp58`. Each derives exactly one Region per complete
set of its own local choices, so there is no Region-declaration choice above the
Kernels and no binding choice below them. The batch-interleaved Kernel derives
the `(batch, nf, sf, t)` schedule with exact activation and weight requirements,
ordinary vector-major activation/output sequences, and a chunked weight
sequence. Legacy HLS keeps embedded versus streamed weights as one local
boundary choice, because its generated unit reads the weight array inside its
own schedule either way.

Several Kernels may derive equal Regions: soft-vector and packed DSP do, for
equal semantic choices. Region equality deliberately does not erase Kernel
identity, constraints, cost, or provider inventory.

PE, SIMD, interleave, compute pumping, and the legacy HLS arithmetic resource
are Kernel-local decisions, owned by the Kernel that uses them. Structural
validation remains a separate derived report and constraint, so structural
readiness never requires any feasibility answer.

Kernel feasibility records conservative capability checks, not formal
realizability witnesses. A `KernelInstance` can therefore be constructed for a
fully selected point whose separately queried feasibility constraint set is
false; callers must not treat instance construction as buildability.

The compatibility module `finn.dataflow.mvau_design` continues to expose the
earlier constructor names, and `MVAU_DESIGN_SPACE_SPEC` in
`finn.dataflow.mvau.legacy_design` retains the original six-field problem
schema and `mvau.pe`/`mvau.simd` paths as a standalone fixture. It is not part
of the Kernel pool.

`accumulator_element_type` is projected problem data computed by FINN's
existing numeric-range analysis before this design-space query; it is not an
independent design choice. Accumulator-output profiles require it to equal the
selected output element type. Fused-threshold profiles additionally require a
threshold source, its real shape, initializer availability, and a representation
at least as wide as the accumulator.

`finn.dataflow.parameters.supply_kernels` holds the weight-supply pool:
`finn_rtl_memstream` and `finnlib_hls_memstream`. Each derives a rank-zero
local-state source from the demand the selected compute Kernel published, so
there is no duplicated delivery tile. One supplier-local choice remains because
it is a real alternative: serve the demand exactly, or emit the natural
full-tile sequence and require the separately selected adapter Kernel. RAM
style, memory pumping, and initialization rules belong to the supplier that
exposes them.

`finn.dataflow.kernels` is the public Kernel authoring surface. A `Kernel`
declares one microarchitecture family with its local decisions, feasibility and
source-admission constraints, Region derivation, interface demands, exports,
and provider inventory. A `KernelSelection` is one static Op-class-owned pool
behind a single identity decision; `SelectedKernel` is the durable identity a
design point records, and `KernelInstance` is the operation-bound view binding
that Kernel to its local assignments, Region, demands, and providers. An
optional pool may select the reserved `NO_KERNEL` member, which is how an
absent supplier is expressed without inventing a topology choice.

Source admission is decision-free by construction: `Kernel` refuses a
source-admission constraint that reads one of its own decisions, which is what
makes `admissible_kernels` a genuine existential answer over problem data
alone. Target facts are therefore excluded from admission and asked at
selection time, so inference coverage does not change with the board.

`finn.dataflow.spec_algebra` contains the generic flat-spec algebra those
surfaces share: path-prefixed placement with explicitly shared problem fields,
applicability gating, and assembly of ordinary flat `DesignSpaceSpec` values.
The authoring layer does not add a Kernel primitive to the engine.

`finn.dataflow.network` and `finn.dataflow.network_validation` implement the
flat acyclic network contract with qualified endpoints, explicit tensor-position
maps, exact beat-sequence compatibility, exposed boundaries, and ordered
channels without physical capacity fields. `finn.dataflow.ops.mvau` uses that
foundation to assemble the selected Kernels, while keeping source-to-region
coordinate mappings outside the normalized region.

The parameter topology is derived, not decided. Whether the result is a
`RegionRef` or a `NetworkRef` follows from which Kernels were selected: an
embedded compute Region, or a streamed one with no supplier, yields a region
reference; a selected supplier yields a network. Direct connection is
established by exact endpoint compatibility, and the adapter is an ordinary
optional Kernel that is refused when the endpoints already match.

`finn.transformation.fpgadataflow.infer_mvau_dataflow` lowers recognized MVAU
sources to unresolved logical nodes. It decides only whether a subgraph is a
source form; whether any implementation supports it is the compute pool's
answer, so the pass contains no datatype, width, target, or language switch.
`finn.transformation.fpgadataflow.select_dataflow_design` is the replaceable
selection-policy seam. It is operation-generic: each `DataflowOp` family names
its own Kernel pools, constraint set, and readiness profiles, and a policy sees
the whole model and every operation scope in one call.

MVAU source associations are topology-aware. Embedded weights identify
compute-local binding state, direct weights identify the compute region's `W`
operand, and cyclic weights identify delivery-local state while the network
edge separately relates the delivery output to the compute input. Semantic
destinations are region- or node-qualified; source identities never enter the
normalized region.

The legacy MVAU cycle estimate now follows the analytically derived logical work
count for both standard and batch-interleaved execution: interleaving
reorganizes the `R * NF * SF` points but does not multiply them by `TH`. The
focused regression checks this formula; comparison against tiled RTL simulation
remains deferred to hardware-enabled measurement work.

Run the focused local verification with:

```bash
./scripts/check-dataflow-design.sh
```

## Engine migration provenance

The private engine was migrated from the Project Kernels scratchpad at revision
`fba51ae01f26c1d53cf89ec51cf6bb88b2e4cbec`. The combined SHA-256 digest of the
source, tests, and examples used as the migration baseline was
`1b4c3156594c8b1ef1f067b0e25e083b12f0145ef970f84e534374f410432e96`.

On 2026-08-26, the original author confirmed that the engine was their original
work, written for inclusion in FINN, and authorized its distribution under
FINN's BSD-3-Clause license. Migrated Python files use the approved FINN header:

```text
Copyright (C) 2026, Advanced Micro Devices, Inc.
SPDX-License-Identifier: BSD-3-Clause
```

The standalone `design_space` package is retained only as the historical
migration source. FINN's implementation under `finn.dataflow._engine` is the
live authority.

The migrated core and parity fixtures are excluded from FINN's Black/isort
hooks so their source remains mechanically comparable with the recorded
baseline. The focused check applies Ruff formatting and linting to those paths
using the repository configuration.

The recorded migration baseline is:

| Check | Result |
|---|---|
| Standalone tests | 68 passed |
| Package-root API | 36 exported names |
| Public `Engine` API | 15 methods |
| Runtime imports | Python standard library only |
| Ruff formatting and lint | clean |
| Strict mypy | clean |
| Supported private-core Python versions | 3.10–3.13 |
