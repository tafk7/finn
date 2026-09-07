# Dataflow model and Space packages: the C1.5 migration

*Status: complete. This migration landed after the C1 semantic decision and
before S2-A or S2-B began. It was a complete internal API cutover, not a
compatibility phase.*

*Date: 2026-09-06.*

This note records what C1.5 decided, what it moved, and what it did not do. It
supersedes the proposal that circulated before implementation; where the two
differ, this is the record and the scratchpad C1.5 plan is the authority it was
implemented against.

## 1. Decision

Assign the two package names to the concepts they literally denote:

```text
finn.dataflow.model
    the canonical DataflowRegion and DataflowNetwork model

finn.dataflow.space
    the generic Space declaration language, compiler and occurrence runtime
```

The former contents of `finn.dataflow.model` are the generic Space frontend and
moved to `finn.dataflow.space`. Every generic Region/Network value, validator,
reference, construction profile, datatype boundary and pure presentation query
moved into the newly vacated `finn.dataflow.model`. Every repository consumer
was updated and the former paths deleted.

At the same time, without aliases:

```text
UnportedInput                  -> InternalInput
unported_inputs                -> internal_inputs
internally_presented_positions -> edge_presented_positions
externally_presented_positions -> boundary_presented_positions
Region (Kernel declaration)    -> RegionDeclaration
```

There is no compatibility alias, deprecated name, forwarding module,
`finn.dataflow.semantic` intermediate package or dual import path in the
completed migration. This feature surface has not been released; internal
imports moved together and the old API was removed outright.

The migration also completed the semantic adoption C1 deliberately left to
concrete owners:

- embedded MVAU retains its weight requirement as an `InternalInput`;
- cyclic parameter delivery declares an `InternalInput` for the operand it
  presents; and
- the duplicate MVAU Region-constructor implementations are consolidated behind
  one operation-owned authority.

## 2. Why it happened before S2

S2-A and S2-B both depend on the Region/Network model and own different sides of
it:

```text
S2-A   source schema, qualified source correspondence, OperandMapping,
       operation reconstruction and native persistence

S2-B   KernelChoice and topology declarations, Region and Network projection
       consumers, ModuleParameter and ModuleBuildSpec
```

Starting them from the previous layout would have given both workstreams imports
and public names already known to be transitional, and moving the model after
they began would have forced parallel workers to edit the same operation, Kernel
and Design imports while performing their substantive refactors.

C1.5 establishes one stable dependency boundary first:

```text
                 +-------------------------+
                 | finn.dataflow.model     |
                 | Region / Network model  |
                 +-------------------------+
                              ^
                              |
                 +-------------------------+
                 | finn.dataflow.space     |
                 | declarations / compiler |
                 | occurrences             |
                 +-------------------------+
                              ^
                              |
                 +------------+------------+
                 |            |            |
              kernels      designs        ops
                 \            |            /
                  \           v           /
                         physical
                              |
                              v
                          artifacts
```

Arrows point from a consumer to a dependency. Nothing in `finn.dataflow.model`
imports `finn.dataflow.space`, `finn.dataflow._engine` or a higher layer.

## 3. Meaning of `InternalInput`

The normalized Region input is a sum:

```python
@dataclass(frozen=True)
class InputInterface:
    port: Port
    requirements: ScheduledInputRequirements

    @property
    def operand(self) -> Operand:
        return self.port.operand


@dataclass(frozen=True)
class InternalInput:
    operand: Operand
    requirements: ScheduledInputRequirements


RegionInput = InputInterface | InternalInput
```

`InternalInput` has exactly one meaning:

> The Region requires this operand according to this scheduled requirement map,
> and the Region exposes no input dataflow port for it.

"Internal" is relative to the Region's dataflow boundary. It does **not** mean:

- physically local to one module;
- embedded RAM or ROM;
- private rather than shared storage;
- initialized from an ONNX initializer;
- compile-time constant;
- resident in a `DataSlot`; or
- served without an external memory system.

A future MLO realization may service several `InternalInput` values from shared
off-chip storage without changing any Region. An embedded core may service one
from private module state. A parameter supplier may service one from a module
data slot. Those are physical bindings of the same dataflow statement.

`InternalInput` carries exactly `operand` and `requirements`. It has no storage,
initializer, placement, module, locality or physical-service field.

### 3.1 Two orthogonal axes

The proposal kept "internal" for both the Region-input kind and the
Network-topology axis, and that collision is the one place this record
deliberately departs from it. The queries were renamed:

```text
Region input kind      InputInterface | InternalInput
Network presentation   edge-presented | boundary-presented | unpresented
```

`unpresented_positions` keeps its name. A position can be unpresented at a
Region port while being serviced by storage outside the eventual physical
module; presentation is a dataflow fact and physical locality is not.

An `InternalInput` has no endpoint and therefore no edge- or boundary-presented
positions. The converse does not hold: a ported input fed by an edge can still
have an unpresented residue when its beat sequence covers less than its
requirements name, and `test_presentation` carries that case.

## 4. Final package tree

```text
src/finn/dataflow/
|
+-- model/
|   +-- __init__.py
|   +-- datatypes.py
|   +-- region.py
|   +-- region_profiles.py
|   +-- region_validation.py
|   +-- network.py
|   +-- network_validation.py
|   +-- refs.py
|   `-- presentation.py
|
+-- space/
|   +-- __init__.py
|   +-- declarations.py
|   +-- compiler.py
|   +-- occurrence.py
|   +-- branching.py
|   +-- domains.py
|   +-- spec_algebra.py
|   `-- dataflow_value_semantics.py
|
+-- kernels/
+-- designs/
+-- ops/
+-- parameters/
+-- artifacts/
`-- _engine/
```

There is no package named `finn.dataflow.semantic`: the model package itself is
the semantic authority. This is an intentional correction to the C0 package map,
amended in the scratchpad rather than left as two current documents assigning
different meanings to `finn.dataflow.model`.

`space.dataflow_value_semantics` is an explicit bridge module, not generic core
vocabulary. It owns the engine `ValueSemantics` adapters and the Problem codec
for canonical dataflow-model values, and it is the only module under `space`
that names the model. `space.__init__` does not import it, so importing the
Space frontend does not drag the dataflow model in behind it.

## 5. Exact module migration

### 5.1 Generic Space package

| Former module | Final module |
|---|---|
| `finn.dataflow.model.__init__` | `finn.dataflow.space.__init__` |
| `finn.dataflow.model.branching` | `finn.dataflow.space.branching` |
| `finn.dataflow.model.compiler` | `finn.dataflow.space.compiler` |
| `finn.dataflow.model.declarations` | `finn.dataflow.space.declarations` |
| `finn.dataflow.model.domains` | `finn.dataflow.space.domains` |
| `finn.dataflow.model.occurrence` | `finn.dataflow.space.occurrence` |
| `finn.dataflow.model.spec_algebra` | `finn.dataflow.space.spec_algebra` |
| `finn.dataflow.model.semantics` | `finn.dataflow.space.dataflow_value_semantics` |

The corresponding test package moved in full: `tests/dataflow/model` became
`tests/dataflow/space`.

The bridge module is named `dataflow_value_semantics` rather than
`value_semantics`, so that a reader of the module list sees which one crosses
the boundary without opening it.

### 5.2 Canonical dataflow-model package

| Former module | Final module |
|---|---|
| `finn.dataflow.datatypes` | `finn.dataflow.model.datatypes` |
| `finn.dataflow.region` | `finn.dataflow.model.region` |
| `finn.dataflow.region_profiles` | `finn.dataflow.model.region_profiles` |
| `finn.dataflow.region_validation` | `finn.dataflow.model.region_validation` |
| `finn.dataflow.network` | `finn.dataflow.model.network` |
| `finn.dataflow.network_validation` | `finn.dataflow.model.network_validation` |
| refs in `finn.dataflow.network_operands` | `finn.dataflow.model.refs` |
| presentation queries in `finn.dataflow.network_operands` | `finn.dataflow.model.presentation` |
| endpoint-ownership helper | `finn.dataflow.model.presentation`, private |
| `RegionRefused` from `finn.dataflow.kernels.kernel` | `finn.dataflow.model.region` |

`network_operands` was the C1 successor to `input_service`; both names are
retired, because the reference and presentation responsibilities now live
separately. `NetworkOperandError` lives with the resolution half in `model.refs`,
which exports the reference values, the two resolution functions and that error
and nothing else. Endpoint ownership -- whether an edge or a boundary feeds a
resolved port -- is presentation logic and is private to `model.presentation`:
the Boolean is a step in computing the position sets, not an answer, and a caller
treating "fed by an edge" as a disposition would miss the partial-presentation
case exactly.

`RegionRefused` moved because the constructors that raise it are pure model
functions. `ops.mvau.regions` is the one MVAU constructor authority and must stay
importable without the engine, which `kernels.kernel` is not; a refusal type
reachable only through the Kernel package would have dragged the compiler in
behind every semantic constructor. It is no longer exported from
`finn.dataflow.kernels`.

`finn.dataflow.__init__` re-exports nothing. Callers import explicitly from
`finn.dataflow.model`, `finn.dataflow.model.<module>`, `finn.dataflow.space` or
`finn.dataflow.space.<module>`. One canonical import path per concept prevents
equal classes from appearing to have two owners.

The former root-level modules and the former generic `model` contents are
deleted. `tests/dataflow/test_package_boundaries.py` carries a `MIGRATED_MODULES`
tuple asserting that none of them is importable and that no source file names
one.

## 6. Public API

`finn.dataflow.model.__init__` exports the detached model surface: the datatype
boundary, the Region model, construction profiles, Region validation, the
Network model, Network validation, qualified references and the
exposure/presentation queries. It exports no `Space`, `Problem`, `Input`,
`Decision`, `RegionDeclaration`, `Kernel`, `DataflowDesign`, `DataflowOp`,
`OperandMapping`, `ComponentABI`, `ModuleBuildSpec`, physical value or artifact
derivation.

The facade was decided by auditing what each module already declared public
rather than from an illustrative list: the datatype helpers, `is_element_type`,
`element_width` and `EdgeTransport` are exported although the pre-migration root
facade omitted them, because dropping them would have made `model.<module>` a
second path for part of the surface. `QONNX_DATATYPE_TOKEN` and
`DATATYPE_PAYLOAD_KEY` are deliberately *not* exported: they are codec plumbing
that `space.dataflow_value_semantics` imports from `model.datatypes` directly,
not vocabulary a Region author uses.

`finn.dataflow.space.__init__` exports the generic contributor surface: `Space`,
`Problem`, `Input`, `Decision`, `Derived`, constraints, readiness, projections,
`Subspace`, `SubspaceChoice`, `ChoiceView`, occurrence diagnostics and the
compiler entry points. It exports no Region, Network, physical, ONNX or artifact
value.

## 7. `RegionDeclaration` versus `DataflowRegion`

The Kernel authoring declaration is `RegionDeclaration`:

```python
class DotpAxiKernel(Kernel):
    region = RegionDeclaration(
        family="mvau.dot_product",
        version="1",
        construct=construct_dot_product_region,
        repetitions=repetitions,
        ...
    )
```

The names now state the lifecycle without explanation:

```text
RegionDeclaration   an authored compiler declaration and construction recipe
DataflowRegion      the detached normalized value produced from it
```

There is no compatibility alias named `Region`.

## 8. One MVAU Region-constructor authority

`finn.dataflow.ops.mvau.regions` owns the pure MVAU semantic constructors:

```text
construct_activation_replay_region
construct_dot_product_region
construct_embedded_dot_product_region
construct_batch_interleaved_streamed_mvau_region
construct_standard_streamed_mvau_region
construct_standard_embedded_mvau_region
construct_weight_stream_region
construct_standard_mvau_weight_port
construct_batch_interleaved_mvau_weight_port
```

There is no nested `ops.mvau.model` package. The proposal suggested one; a
one-file package is not justified, and a nested `model` grouping for MVAU is a
separately reviewable migration of the whole cohesive group (`computation`,
`regions`, `networks`).

Kernel modules retain candidate declarations, physical constraints, parameters,
ABIs and source contributions, and import the one constructor authority. The
duplicate implementations were proved value-identical over several foldings and
then deleted; historical behavior remains available through revision-pinned
tests and oracles, not a second live implementation.

Two facades were removed with them: `construct_mvau_compute_region` with its
`MVAUWeightInterface` selector, and `construct_streamed_weight_mvau_region`, a
second name for the standard streamed constructor with no callers.

The authority refuses with `RegionRefused` throughout, so a Kernel reading these
constructors gets a rejecting absence at an infeasible folding rather than an
`EvaluationError`.

One restatement survives deliberately. `construct_weight_stream_region` writes
its own beat order instead of calling `construct_standard_mvau_weight_port`: a
supplier deriving its output order from its consumer's constructor would be a
supplier bound to that consumer, and the two agreeing would be guaranteed by an
import rather than checked on the edge. It is checked, position by position.

The generic cyclic parameter constructor stays in `finn.dataflow.parameters`
because it is not MVAU-specific; what belongs to MVAU is the order.

## 9. Concrete `InternalInput` adoption

### 9.1 Embedded dot product

The embedded constructors derive the standard weight operand and scheduled
requirements once and change only presentation:

```text
streamed   InputInterface(weight Port, weight requirements)
embedded   InternalInput(W, the same weight requirements)
```

Equal between the two: schedule; activation operand and requirements; weight
operand, datatype, shape and requirements; output operand, availability and beat
sequence. The sole difference is the port, so the embedded Network has neither a
weight edge sink nor a weight boundary.

The old docstring claim that "the matrix itself is not an operand here" is
deleted. The matrix is a logical operand and is not stream traffic at this
Region boundary.

### 9.2 Cyclic parameter delivery

The cyclic parameter Region declares:

```text
schedule   rank zero: one point ()
input      InternalInput(output operand, one requirement at () for every
           position in the output beat image)
output     the existing Port and BeatSequence, every image position available
           at ()
```

Repeated output presentation does not create repeated input requirement
occurrences by itself. One internal position may be retained and emitted several
times by the binding, just as one presented input position may satisfy several
scheduled uses. A more restrictive supplier can author a different requirement
relation if its computation semantics require one.

### 9.3 Decoupled evidence

The corrected decoupled Network makes both qualified requirements real:

```text
memory.W    InternalInput,   unpresented
compute.W   InputInterface,  edge-presented by the weight edge
```

Physical provisioning is decided later by U6.

## 10. Source correspondence remains outside the package

The model package answers questions about an explicit qualified ref. It cannot
determine which source tensor that ref corresponds to, and it does not scan bare
`Operand.id` values to guess.

S2-A owns `finn.dataflow.ops.mapping.OperandMapping`: the operation authors or
selects the correspondence, and the compiler checks its qualified targets against
the selected Network. The model package supplies only qualified target
resolution, port and boundary exposure, edge/boundary presentation sets and
unpresented positions. It supplies no placement enum and names no storage
technology.

## 11. Dependency rules, enforced

`finn.dataflow.model` may import the Python standard library, its own siblings
and QONNX datatype identity through `model.datatypes`. It may not import
`finn.dataflow.space`, `finn.dataflow._engine`, `kernels`, `designs`, `ops`,
`parameters`, `artifacts` or ONNX graph types.

`tests/dataflow/model/test_facade.py` pins `model.__all__` exactly, in both
directions: an accidental export and an accidentally dropped one both show up as
a set difference. It is the counterpart of the `space` facade test that has
existed since U1.5, and it exists because C1.5 made this facade permanent and
deliberately decided what it does *not* carry -- a decision that lives only in a
commit message is one that quietly erodes.

`tests/dataflow/test_package_boundaries.py` asserts this statically over every
file in the package, asserts the reverse direction for the generic `space` core
with the bridge module as the one declared exception, asserts that
`finn.dataflow` itself exports nothing, and asserts by fresh subprocess import
that `finn.dataflow.model` loads none of the forbidden packages and that
`finn.dataflow.space` loads neither the bridge nor the model.

### 11.1 Presentation's precondition

The presentation queries are pure over a Network that `validate_network` has
already accepted. They do not revalidate, and the docstrings now say so on each
function rather than implying a guarantee the code does not make.

The contract is deliberate: validity is a whole-Network property established
once, while presentation is scoped to one reference and is meant to be asked many
times -- per qualified reference during S2-A correspondence, per obligation in
U6. Folding validation into each query would make the common case pay for it
repeatedly and still not be a validator.

So these functions do not check that an edge's source exists or is an output,
that its position map is total, that element types or beat sequences agree across
it, that the referenced Region is valid, or that the Network is acyclic. Given a
Network carrying one of those defects they answer from the consumer side, and the
answer looks authoritative.

The evidence establishes the two halves of that arrangement without freezing the
undefined part. `test_presentation` proves that `validate_network` catches the
defects the precondition excludes — a dangling edge source, and an edge whose
sides disagree — so the precondition is discharged rather than merely asserted;
and it proves the queries do not re-run it, by breaking `validate_network`
outright and then running every query over a Network that is in fact valid, plus
a static check that the module names no validator under any spelling. What it
deliberately does *not* do is assert what a query returns for an invalid
Network. That is undefined by contract, and pinning today's answer would make it
normative: an implementation that later detected one of these defects cheaply and
raised would be legal under the precondition and would read as a regression.

The one condition `_owned_endpoint` re-checks locally is endpoint ownership,
because that single fact is what selects between the edge and boundary arms.
Without exactly one owner there is no edge-versus-boundary answer to give, so it
refuses rather than picking. That is arm selection, not validation, and the
docstring no longer claims otherwise.

## 12. Canonical equality and validation

The migration changed names and module ownership without weakening canonical
value behavior:

- datatypes are re-resolved through canonical QONNX identity;
- requirements and availability retain canonical sorted representations;
- beat order and repeated beat positions remain exact value state;
- ported inputs remain ordered by port id;
- `InternalInput` values follow ported inputs and are ordered by operand id;
- outputs remain ordered by port id;
- Network nodes, edges and boundaries remain ordered by stable id; and
- all values remain immutable.

Region validation changed one diagnostic label with the type:

```text
unported_input['W'] -> internal_input['W']
```

Existing semantic issue codes are unchanged. In particular
`input.operand_duplicate` remains the one-input-per-operand rule, and no
`internal_input.operand_duplicate` parallel code was introduced.

Network endpoint ownership continues to range only over `InputInterface` values.
An `InternalInput` has no port and is not a Network endpoint.

## 13. Tests

The generic Space tests moved with their implementation to
`tests/dataflow/space/`. The canonical model tests occupy the package name that
matches their subject:

```text
tests/dataflow/model/
    supply_networks.py       the three supply forms, built once
    test_datatypes.py
    test_region_primitives.py
    test_region_profiles.py
    test_region_validation.py
    test_network.py
    test_network_validation.py
    test_refs.py             resolution only
    test_presentation.py     exposure and edge/boundary/unpresented
    test_facade.py           the exact public surface of the package
```

The refs/presentation split follows the module split: `test_refs` asks which
value a qualified name denotes, `test_presentation` asks which of its positions
arrive over a port. They share the three supply Networks rather than each
rebuilding them.

Semantic evidence covered by the suite:

1. ported-only Region equality and ordering are unchanged, and `inputs[:n]`
   still equals `input_interfaces` for them;
2. `InternalInput` participates in every operand and requirement rule and no
   port-shaped rule;
3. streamed and embedded dot product retain equal weight operands and
   requirements;
4. embedded dot product has no weight endpoint, edge or boundary;
5. cyclic parameter delivery declares an `InternalInput` and preserves its exact
   output sequence, availability and repeated presentation;
6. decoupled `memory.W` is unpresented and `compute.W` is edge-presented;
7. partial edge presentation retains its unpresented residue;
8. repeated requirements over one presentation remain valid;
9. Region-local operand-id reuse across Network nodes remains legal;
10. endpoint ownership ranges only over `InputInterface` values;
11. semantic validity remains independent of physical availability; and
12. production MVAU Kernels name one constructor authority, asserted by
    identity.

## 14. Evidence gate

Run from one clean revision:

```text
FINN_ROOT=$PWD PYTHONPATH=src:tests:deps/qonnx/src python -m pytest tests/dataflow
python -m pytest tests/fpgadataflow/test_mvau_cycle_estimate.py
ruff format --check and ruff check over src/finn/dataflow and tests/dataflow
env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow \
    -p finn.custom_op.dataflow
```

`scripts/check-dataflow-design.sh` runs all of these together and is the gate for
changes under `src/finn/dataflow/`. Its typed-test file list follows the moved
paths.

No hardware rerun was required. The migration changed no ABI, no RTL source
closure, no actual RTL parameter, no `ModuleBuildSpec` identity and no generated
source; package movement and Region-model corrections do not justify a hardware
claim.

## 15. Non-goals

C1.5 did not implement S2-A, S2-B, S3 or U6 features merely because their final
dependencies were being established. It introduced no `OpInput`, `OpOutput` or
`OperandMapping`; no native ONNX Decision persistence; no `KernelChoice`,
`NetworkEdge` or `NetworkBoundary`; no `ModuleParameter` or `ModuleBuildSpec`; no
Design physical composition; no binding-realizability witness; no multi-port
service for one operand; no dynamic beats, feedback or stateful output versions;
and no physical storage, interface, timing or artifact field in a Region.

## 16. Completion statement

The codebase has one visible dataflow-model center:

```text
finn.dataflow.model
    owns what DataflowRegion and DataflowNetwork mean

finn.dataflow.space
    owns the generic declaration language, compiler and occurrence runtime

RegionDeclaration
    declares how one Kernel derives a DataflowRegion

ops.mvau.regions
    is the one authority for the MVAU Region families

finn.dataflow.ops.mapping
    will own source-to-qualified-model correspondence

physical composition
    will bind semantic obligations without changing them
```

S2-A and S2-B proceed against final package ownership and terminology rather
than extending transitional paths that another migration must later remove.
