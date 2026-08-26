# Dataflow design engine

The supported authoring surface for dataflow design spaces is
`finn.dataflow.design`. Domain adapters should not import
`finn.dataflow._engine` directly. The private engine remains domain-neutral and
uses only the Python standard library; region knowledge is confined to the
public design adapter.

The first connected specification is `finn.dataflow.mvau_design`. It derives a
concrete streamed-weight MVAU `DataflowRegion` after the `mvau.pe` and
`mvau.simd` decisions are committed. Structural validation is represented by a
separate `RegionValidationReport` derived property and a Boolean constraint, so
a failed constraint never discards the model-local issues that explain it.

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
