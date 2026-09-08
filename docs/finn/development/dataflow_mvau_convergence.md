# MVAU Design convergence (S3)

S3 consolidates the production MVAU inventory at implementation commit
`fe6d0a179` from C2 base `c12a53b1429c22d5aeffb5eb659d914485c52a03`.
The operation now offers exactly:

```text
dot_product       DotProductDesign@2
                    weight_supply = external | embedded | decoupled
batch_interleaved BatchInterleavedDesign@1
```

The former external-only `DotProductDesign@1`,
`SuppliedDotProductDesign`, the `supplied` operation alternative and the
`designs.supplied_dot_product` module are removed without aliases. The shared
`WeightedDotProductDesign` remains an abstract declaration base. Reusing its
Python members avoids duplicated authoring, but compilation gives each Design
occurrence distinct root-relative coordinates.

## Explicit dot-product selection

`DotProductDesign` receives the source fact `initializer_present` and requires
callers to choose both a supply and a compatible compute candidate:

| Supply | Compute candidate | Source weight entry |
| --- | --- | --- |
| `external` | `dotp_axi` | boundary-presented `compute.W` |
| `embedded` | `dotp_axi_embedded` | unpresented `compute.W` |
| `decoupled` | `dotp_axi` | unpresented `memory.W` |

In the decoupled Network, `memory.W` and the edge-presented `compute.W` are
distinct qualified operands. In the embedded Network, `compute.W` remains an
`InternalInput` with the same requirements as the streamed Region; only its
dataflow port is absent. `InternalInput` itself says nothing about storage or
initialization. Requiring an initializer for embedded and decoupled supply is
this concrete Design's admission policy. External supply is available with or
without an initializer, and initializer presence never selects a mode.

The supply/candidate pairing is checked through the existing semantic Network
boundary. A point missing either choice is unresolved; an inconsistent pairing
is refused. Physical availability remains separate: embedded compute,
batch-interleaved compute and memstream are immediately physical-`Absent`, while
their accepted semantic Networks and operand mappings remain available.

## Native schema migration

`MvauDataflowOp.schema_version` is 3. Its mathematical `family_version` remains
`"1"`. The generic `DataflowOp` default and ActivationReplay remain schema 2;
the encoding algorithm did not change.

| Meaning | Schema-3 native attribute |
| --- | --- |
| Schema | `dataflow_schema_version = 3` |
| Design selector | `design__case = "dot_product" | "batch_interleaved"` |
| Dot-product folding | `design__dot_product__pe`, `design__dot_product__simd` |
| Weight supply | `design__dot_product__weight_supply` |
| Compute candidate | `design__dot_product__compute__kernel` |
| Streamed pumping | `design__dot_product__compute__dotp_axi__compute_pumping` |

Other candidate-qualified paths follow the same compiled-name rule.
Batch-interleaved paths keep their local spelling but the containing MVAU node
also writes schema 3. The C2 schema-2 paths under `design.supplied.*` are
retired, and even schema-2 records using retained branch spellings are refused.
There is no old-schema reader, alias, missing-choice default or automatic
rewrite; pre-release models are recreated from source ONNX and build
configuration.

Switching among external, embedded and decoupled supply uses immutable
successors and prunes the unreachable compute-candidate attributes. Switching
between `dot_product` and `batch_interleaved` similarly prunes the old Design
family. Transaction, stale-source and output-repair behavior remains generic.

## Semantic and artifact equivalence

Before editing, C2 source was exported with `git archive` and loaded in a
separate process with both `FINN_ROOT` and `PYTHONPATH` pointing at that export.
The evidence encoder records every dataclass field, ordered tuples and beat or
requirement sequences, plus stable Enum and QONNX datatype identities.

Exact C2/S3 equality holds for:

- all nine external cases in `test_dot_product_design.MATRIX`;
- external, embedded and decoupled supply at representative legal folding;
- all seven legal BatchInterleaved matrix cases.

C2 also directly proved that its narrow external Design and flexible external
Design produced equal Networks. The S3 unified external witnesses equal both.
Source execution, output authority, qualified operand mapping and the retained
Region/Network constructors remain unchanged.

The standalone plain/pumped DotpAxi and ReplayBuffer source keys, package keys,
resolved source order and generated file digests are unchanged. Unified
external compute and replay children also retain the C2 realized module fields
after excluding provenance, along with exact source/package keys and generated
bytes. Whole `ModuleBuildSpec` equality is intentionally not claimed:

```text
C2 flexible compute  supplied.compute.kernel, supplied.pe, supplied.simd
S3 compute           dot_product.compute.kernel, dot_product.pe, dot_product.simd

C2 flexible replay   supplied.pe, supplied.simd
S3 replay            dot_product.pe, dot_product.simd
```

The S3 paths are actual persistable declarations. Artifact derivation remains
independent of provenance and keys only the inputs it consumes.

## Deferred BatchInterleaved audit

`interleave` remains Design-owned as a deliberate lift, not because Region
visibility alone proves that owner. A later architectural review must answer:

1. whether the choice is local to one Region;
2. whether the Kernel should be reusable independently of this Design;
3. whether Kernel ownership removes meaningful forwarding cost;
4. how a local rule prevents duplicated shared axes; and
5. whether the weight-delivery model needs correction first.

S3 does not introduce a generic local-semantic-Decision rule or redesign the
BatchInterleaved Region.

## Scope and hardware disposition

The canonical model, generic Space and engine, generic Design/Kernel/artifact
packages, source schema, inference adapters, Region constructors, Networks and
mathematical MVAU execution are unchanged from C2. No physical composition,
production routing, lowering, initializer provisioning or authority fold is
included.

No hardware run is required for this convergence: ABI, parameters, source
closure, artifact keys and generated standalone bytes are unchanged. Composed
hardware evidence remains U6-owned, and the long-lived authority fold remains
S4-owned.
