# Complete dataflow adapter: AC0 baseline

This record freezes the implementation boundary before the class-centered
adapter migration. The decisions are:

- retire the MVAU-only v11 graph-metadata transport; the repository contains
  test and compatibility-facade callers only, and no required external caller
  has been identified;
- publish the migrated MVAU operation as `mvau-dataflow-op-v7` and reject v6
  explicitly rather than reinterpreting its stored choices;
- use the adapter-first merge order;
- this branch owns adapter, operation, Design, Kernel, resolution, association,
  and physical-composition changes;
- the parallel artifact effort exclusively owns
  `src/finn/dataflow/artifacts/**` and `tests/dataflow/artifacts/**`; and
- preserve the current artifact-facing values until that effort performs its
  integration. No artifact-substrate implementation is copied or recreated by
  this migration.

## Revisions

| Repository | Revision | State |
|---|---|---|
| FINN | `15b2fe56adec68b8d279e8aa1cf41b432567dc77` | clean |
| FinnLib | `97cdc4ee2961354c17792eec9bf72365553eb55f` | clean |
| artifact-substrate worktree | `1e948127e355486bbe721104bbea38cd62efb668` | clean and not merged |

The FINN and artifact branches diverge from
`9cf9a7b0a1c3c128899265f7689a48eef0ec97f1`. At this baseline FINN is 18
commits ahead on its side and the artifact branch is 11 commits ahead on its
side. The artifact branch edits several adapter-facing files, so later
integration must treat this branch as the owner of the adapter implementation
rather than resolving the overlap as an unreviewed textual merge.

## Software gate

Run on 2026-09-01 with:

```text
Python 3.10.19
pytest 6.2.5
Ruff 0.16.4
mypy 2.3.1
```

Results:

```text
994 dataflow tests passed
MVAU cycle regression passed
Ruff format check passed
Ruff lint passed
strict mypy passed over 145 source files
```

The command was:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/check-dataflow-design.sh
```

## Hardware evidence

The existing logs identify the same FINN and FinnLib revisions and Vivado
2025.2. Their SHA-256 digests at AC0 are:

| Log | SHA-256 |
|---|---|
| `review3-software.log` | `d287d2c9ae6a88779960d479d7c44d844f98b2c00bc9ac23833fa917afb7fea3` |
| `review3-fixture5.log` | `4a57b2b91836606012542b82dc49e1998f8ba523389a50724107b3589c67a8b7` |
| `review3-fixture6.log` | `f3b3df1daf409f4ba533afea228aa994ccdf6ca519ce187561c99666b43e6aa8` |
| `review3-fixture7.log` | `f4bebe66f2a648e587372f791829303cb687992de8b17741d0e21435fca1c55b` |
| `review3-fixture8.log` | `7bebdc393f305642c4cbc0fc73e5f03d4bec0021d7b9773145068129d298c44e` |
| `review3-fixture9.log` | `f01daadc7c3afeb18715a2eedb2715d0c480410206dc190346bb8cef74295b66` |
| `review3-d6a-numeric.log` | `642977bfb9fc9d4ecc6d5baa1fa3c4ca3d64c4fefc3c4f4e86a9420aab1ff0af` |
| `review3-d6a-hardware.log` | `361571c5a786b3823090f904ed0a531988fa8f6a9443dfa3f206aa14a6ea1822` |

## Persistence and semantic baselines

The v6 operation problem schema has 22 fields and ten persistent decisions.
The exact paths, requiredness, codecs, node-attribute bytes, and the v11 JSON
transport are pinned by the migration tests. Existing normalized migration
tests continue to pin external and supplied Regions and Networks, logical and
physical associations, configured parameters, source manifests, wrapper text,
and artifact identities.

The old logical `MVAUSourceAssociation` deliberately mixes source/semantic
content with these physical fields, which AC4 will split out:

```text
compute_kernel_id
supply_kernel_id
adapter_kernel_id
design_id
decision_paths
kernel_ids
BindingLocalStateDestination
```

The exact v6 and v11 bytes recorded by the tests are migration evidence only.
They are not accepted payloads for v7.

## AC3 proposed v7 problem identity

The shadow class-authored MVAU frontend projects 39 problem fields. Tensor
identity, shape, datatype, initializer presence, and requested initializer
fingerprints are owned by the four operand declarations; `r`, `mw`, `mh`, the
computation profile, source description, and effective narrow-weight flag are
derived properties rather than separately projected copies.

For the canonical unfused logical-MVAU fixture, the proposed problem
fingerprint is:

```text
8e96ddc2535ee0e97d8df67f5186b7247b03acb528d8b6d99a4f1dc786159b34
```

The shadow lifecycle uses the same ten reviewed decision paths and storage
attribute names as v6. Tests compare its selected Region, Network, association,
candidate admission, and configured-Kernel inputs against the production v6
path before the atomic v7 cutover.

## AC4–AC7 migration record

The canonical logical operation now publishes `mvau-dataflow-op-v7` with
persistence format `1`. A v6 family header is rejected explicitly, and the v11
graph-metadata projection/persistence modules and their test-only lifecycle have
been removed rather than reinterpreted.

`ResolvedDataflowOp` now stores the selected Design id, Network, logical source
association, source-scope id, immutable point, and private compiler context
directly. The MVAU logical association contains only source provenance,
semantic operand mappings, coordinate mappings, and the selected semantic
parameter topology. Kernel ids, placement identities, decision paths, and
binding-local state destinations remain in realization/physical provenance.

`MvauDataflowOp`, `DotProductDesign`, `BatchInterleavedDesign`,
`DotpAxiKernel`, `ReplayBufferKernel`, and `FinnRtlMemstreamKernel` use the
class-local declaration compiler. The canonical operation no longer calls a
manual MVAU projection, persistence implementation, or inventory-aggregation
bridge. Candidate-backed source admission reads the compiler-owned inventory
and provenance.

The software gate at this checkpoint is:

```text
1000 dataflow tests passed
MVAU cycle regression passed
Ruff format and lint passed
strict mypy passed over 149 source files
```

## AC8–AC9 composition and retirement record

Physical composition is now registered by `DotProductDesign` through a
`PhysicalComposition` declaration. Generic operation code selects the compiled
Design, realizes it, resolves only the declared composition facts, and invokes
the composer with a `PhysicalCompositionContext`. The MVAU composer cannot see
an `Engine` or `DesignPoint`.

The context is reduced to immutable `PhysicalCompositionProvenance` before it
is retained by the MVAU physical result. The MVAU artifact handoff derives the
legacy artifact-owned identities and source paths from `KernelOrigin`
projections; neither configured Kernels nor `DesignRealization` cross that
handoff. The generic artifact package remains unchanged for the separately
owned artifact-substrate integration.

The public façades now expose only the intended contributor vocabulary:

```text
finn.dataflow.authoring
    immutable Op/Design/Kernel declarations and generic runtime entry types

finn.dataflow.design
    evaluation values, including ResolvedDataflowOp

finn.dataflow.kernels
    Kernel, PhysicalComponent, scalar_parameters

finn.dataflow.ops.mvau
    MVAUDataflowBuildContext, MvauDataflowOp
```

`OpDesign`, `Scope`, `Ref`, `DataflowDesignScope`, `KernelScope`, compiled
declarations, inventories, placement/coverage records, and legacy assembly
functions remain reachable only from private implementation modules where old
equivalence tests still require them. `DataflowDesign` and `Kernel` no longer
advertise scope-callback methods. `NetworkRef`, `MVAUNetworkRef`, and
`DataflowOpResult` have been removed.

The public MVAU import path no longer loads the retired `problem`, `inventory`,
physical-composition, or artifact-stage modules. `problem.py` and
`inventory.py` remain private migration fixtures only; production v7 source
types live in `ops.mvau.contracts`, and production selection uses the compiled
operation inventory.

## Final MVAU responsibility map

| Module | Responsibility |
|---|---|
| `ops/mvau/op.py` | Source schema, generic projections, operation facts, closed Designs, persistence declarations, and QONNX behavior |
| `ops/mvau/contracts.py` | Stable family ids and pure typed Op-to-Design facts |
| `ops/mvau/designs/dot_product.py` | Dot-product Regions, Network, mappings, placements, and registered composition |
| `ops/mvau/designs/batch_interleaved.py` | Semantic-only batch-interleaved Region, Network, mappings, and readiness |
| `ops/mvau/input_supply.py` | Closed external/memstream weight-supply policy |
| `kernels/dotp_axi.py` | Dot-product Kernel coverage, choices, parameters, sources, and component elaboration |
| `kernels/replay_buffer.py` | Replay Kernel coverage, parameters, sources, and component elaboration |
| `kernels/finn_rtl_memstream.py` | Memstream Kernel coverage, choices, parameters, sources, and component elaboration |
| `ops/mvau/elaboration.py` | Restricted Design-owned MVAU physical composition |
| `ops/mvau/physical.py` | MVAU-specific physical records and association validation |
| `ops/mvau/artifacts/*` | Existing adapter-first rendering, staging, synthesis, packaging, and IP-XACT compatibility surface |
| `ops/mvau/problem.py`, `ops/mvau/inventory.py` | Private pre-v7 equivalence fixtures; absent from production imports |

Intentional schema/API changes are the v7 family identifier, format-1 generic
persistence, direct Network result, logical/physical association split,
removal of `provider_ids`, and retirement of the public scope/compiler-record
exports. Region/Network values, candidate verdicts, configured parameters,
source manifests, generated RTL/Tcl/XDC, and legacy artifact identities remain
covered by exact regression tests.

## AC10 clean-revision closure

Final evidence was produced from the clean implementation revision:

```text
FINN     e8c436d781697020fbfd169c92bf0a04dd167dd1
FinnLib  97cdc4ee2961354c17792eec9bf72365553eb55f
Python   3.10.19
pytest   6.2.5
Ruff     0.16.4
mypy     2.3.1
Vivado   2025.2
```

The implementation history is:

| Phase | Commit |
|---|---|
| AC0 baseline | `291839e0f` |
| AC1 immutable declarations | `72fed1a67` |
| AC2 complete generic frontend | `747777253` |
| AC3 MVAU shadow declaration | `368afffa2` |
| AC4–AC7 v7 cutover and compiler consolidation | `7f1068885` |
| AC8–AC10 composition, retirement, documentation | `e8c436d78` |

All required gates passed:

| Gate | Result | Log | SHA-256 |
|---|---|---|---|
| Complete software | 1003 dataflow tests; MVAU cycle regression; Ruff format/lint; strict mypy over 151 files | `adapter-final-software.log` | `666d5fe547c6ad98b7c40d7ca4853ed8f15c6c9c3325f2d1bd110f6d890f9bb9` |
| Fixture 5 | 8/8 fused-versus-composed configurations, free-running and stalled | `adapter-fixture5.log` | `52f9567304f286474c94c219fbe156bd3cad03864c874d5d3460f87ebc907ab6` |
| Fixture 6 | soft-vector DSP48E2, packed DSP58, and DSP48E1 synthesis/resource checks | `adapter-fixture6.log` | `5d4f56a1b7ffac0f32d4ba917447f172c01f7059b7dfa04dfba13ee6684cc3b1` |
| Fixture 7 | two-cell module stitching and reported-pin checks | `fixture7.log` | `d5829af1fe5f961dab28a22d97684f2ebe7523ea2ef280bc1568d1e6c2089fc1` |
| Fixture 8 | 15/15 numerical cases, free-running and stalled | `fixture8.log` | `817430a6fa1d6b9903edaf377f76f8e76d17864fa9b8af93f2829db05550777a` |
| Fixture 9 | IP-XACT packaging, catalog resolution, two-cell stitching, wrapper generation | `fixture9.log` | `31078823a9194903d1107706afdb04e03047837eac1e41e9104ced10220252c5` |
| D6a numerical | unpumped/pumped, free-running/stalled; six output beats in every mode | `d6a-memstream-numeric.log` | `d4c3342c8f197775ed8418e1e2bb71e66ef242a4c3852b9d37fed1937aff3fef` |
| D6a hardware | 2 DSP primitives, one RAMB18E2, IP-XACT package, catalog stitch | `d6a-memstream-hardware.log` | `7a4ee04c08e4055894028a36f9f7b34366cfadaa0143d93e383e0aa2a643fe58` |

The artifact-substrate branch remained unmerged and unchanged at
`1e948127e355486bbe721104bbea38cd62efb668`; no file under
`src/finn/dataflow/artifacts/**` or `tests/dataflow/artifacts/**` was edited by
this implementation.
