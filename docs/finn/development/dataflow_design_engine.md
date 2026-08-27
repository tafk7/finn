# Dataflow design engine

The supported authoring surface for dataflow design spaces is
`finn.dataflow.design`. Domain adapters should not import
`finn.dataflow._engine` directly. The private engine remains domain-neutral and
uses only the Python standard library; region knowledge is confined to the
public design adapter.

The first connected Kernel definitions are the MVAU compute Kernel and cyclic
parameter-delivery Kernel. `MVAU_COMPUTE_KERNEL_SPEC` declares three complete
region branches: `standard.embedded`, `standard.streamed`, and
`batch_interleaved.streamed`. The interleaved branch has schedule
`(batch, nf, sf, t)`, exact activation and weight requirements, ordinary
vector-major activation/output sequences, and a chunked weight sequence. The
compatibility module `finn.dataflow.mvau_design` continues to expose the
earlier constructor and specification names without retaining duplicate logic.

Computation profiles and implementation bindings are separate from region
selection. The five initial binding identities distinguish legacy HLS LUT,
legacy HLS DSP, RTL soft-vector, RTL packed, and RTL batch-interleaved DSP58
implementations. Target, width, narrowing, and pumping rules are binding
constraints; they are not inputs to any region constructor. Structural
validation remains a separate derived report and constraint, so
`model_structural` readiness never requires a binding.

`CYCLIC_PARAMETER_KERNEL_SPEC` is the second concrete Kernel definition. It
constructs a rank-zero local-state source whose output port exactly preserves
the requested full-tile or chunked `BeatSequence`. RAM style, runtime
writability, pumping, and implementation identity remain binding-owned.

`finn.dataflow.kernel` contains the generic authoring layer extracted from those
two definitions. It records Kernel, region-declaration, and binding identities,
constructs `KernelInstance` values, and assembles ordinary flat
`DesignSpaceSpec` values. It does not add a Kernel primitive to the engine.

`finn.dataflow.network` and `finn.dataflow.network_validation` implement the
flat acyclic network contract with qualified endpoints, explicit tensor-position
maps, exact beat-sequence compatibility, exposed boundaries, and ordered
channels without physical capacity fields. `finn.dataflow.ops.mvau` uses that
foundation to select an embedded/direct `RegionRef` or a cyclic-delivery
`NetworkRef`, while keeping source-to-region coordinate mappings outside the
normalized region.

The legacy MVAU cycle estimate now follows the selected logical work count for
both standard and batch-interleaved execution: interleaving reorganizes the
`R * NF * SF` points but does not multiply them by `TH`.

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
