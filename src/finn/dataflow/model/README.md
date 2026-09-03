# Declarative dataflow model

`finn.dataflow.model` is an isolated experiment in declaring design spaces as
ordinary Python classes. It does not wrap or preserve the existing
`finn.dataflow.authoring` and `finn.dataflow.kernels` APIs.

Three layers share one frontend and lower to one flat spec:

```text
Space                  ordinary declarations, direct child composition,
                       and OneOf(Case(...), ...) exclusive branching
   |
Kernel(Space)          semantic Inputs, physical-only Decisions,
                       one Region(family, version, construct, **deps)
   |
DataflowDesign(Space)  semantic Decisions, Kernels segments,
                       explicit Connections and Boundaries,
                       one selected canonical DataflowNetwork
   |
   -> flat DesignSpaceSpec -> existing _engine validation and evaluation
   -> configured Design: one Kernel per role, one validated Network
```

`DesignSpaceSpec`, `Engine`, and `DesignPoint` remain the normalized IR and
runtime. They are private implementation details of the occurrence API for
ordinary contributors; the model package remains a source-language frontend,
not another design-space evaluator.

## A closed Space

A root `Space` may declare problem fields and decisions directly on its class.
Derived values and constraints name their dependencies explicitly.

```python
from finn.dataflow.model import Decision, Problem, Space, compile_space, derived


class Folding(Space):
    extent = Problem(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, extent=extent, lanes=lanes)
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes


spec = compile_space(
    Folding,
    "folding",
    problem_namespace="problem.folding",
)
```

This produces the same raw declaration records that could be written by hand:
`ProblemField`, `Decision`, and `DerivedProperty`. `Engine.validate(spec)` is
still the authority that validates those records.

A constraint-free `Space` is valid. `ConstraintGroup` and `Readiness` are
available only when a caller needs named aggregate checks.

## Class-centered occurrences and projections

Starting a root returns an instance of the authored class. Child navigation
returns instances of the authored child classes, all backed by the root's one
private immutable point:

```python
from finn.dataflow.model import ConstraintGroup, Projection, Readiness, constraint


class Tile(Space):
    # declarations as above
    @constraint(lanes=lanes)
    def positive(*, lanes: int) -> bool:
        return lanes > 0

    legal = ConstraintGroup(positive)
    ready = Readiness(decisions=(lanes,), properties=(cycles,), constraints=legal)
    schedule = Projection(cycles, readiness=ready, constraints=(legal,))


root = Pair.start({Pair.extent: 16})
left = root.child(Pair.left)
left = left.assign(Tile.lanes, 4)

assert type(root) is Pair
assert type(left) is Tile
assert left.root is not root  # immutable successor root
assert left.cycles == 4  # descriptor shorthand for a decided answer

assessment = left.project(Tile.schedule)
assert assessment.accepted_answer.value == 4
```

`ProjectionAssessment` retains its compiled name plus four separate facts:
readiness, every constraint-group assessment, the raw output answer, and the
accepted answer. The reduction order is the contract:

```text
1. any required readiness, output or constraint dependency unresolved
       -> Unresolved
2. the distinguished output is finally inapplicable
       -> the output's own non-rejecting Absent
3. any projection constraint refuses
       -> rejecting Absent
4. otherwise
       -> Decided(snapshot(output))
```

Absence precedes refusal deliberately. Reporting "a constraint refused this" for
a value that simply does not arise is a different and misleading sentence. The
two policies behind steps 2 and 4 -- propagate final inapplicability, snapshot
through the output declaration's own `ValueSemantics` -- live in the compiled
metadata and are not constructor arguments, because in U1 neither has a second
legal value. Every `Projection` declares a `Readiness`; a projection with no
further obligation declares an empty profile rather than omitting the concept.

Navigation keys are exact declarations. A Python class names a family, not one
of its occurrences:

```python
root.child(Pair.left).assign(Tile.lanes, 2)  # exact use site
root.child(Tile)  # refused: a class is not an occurrence
```

`branch(OneOfDeclaration)` exposes case ids, selection, and exact case views
without exposing its generated selector path. `answer`, `assign`, `assess`,
`project`, `branch`, and `child` accept declarations, not path strings.

Thirteen member names are reserved -- `start assign answer assess project
diagnostics branch child root problem_snapshot problem_fingerprint is_stale
reconstruct` -- because each is a lifecycle operation every authored class
inherits. The check is on the Python member name only, so
`choice = OneOf(..., name="branch")` keeps the compiled path `<ns>.branch`.

Two error kinds and no third. A malformed or out-of-scope declaration, a
malformed `Projection`, a reserved name or a Problem value with no canonical
encoding is an `AuthoringError`. Refused Problem data, a rejected assignment, an
unavailable value and an incompatible persisted fingerprint are the engine's own
`RequestError` carrying findings.

Problem values are snapshotted once. A callable problem projector makes an
explicit staleness check possible without making ordinary queries reread live
state:

```python
source = {"extent": 16, "unrelated": "a"}
root = Pair.start(lambda: {Pair.extent: source["extent"]})

source["unrelated"] = "b"
assert not root.is_stale()

source["extent"] = 32
assert root.is_stale()
assert root.extent == 16  # the old snapshot is unchanged
fresh = root.reconstruct()  # strict: old assignments are not migrated
```

`problem_fingerprint` guards future persisted assignment hydration;
`expected_problem_fingerprint=` refuses an incompatible payload. The digest
carries the Space identity, the ordered Problem names and value semantics,
explicit absence, and each field's canonical encoding. Encoding is owned by the
declaration, not by a global registry and not by a magic method on a value class
this project does not own:

```python
QONNX_DATATYPE = CanonicalValueCodec("qonnx.datatype", 1, lambda value: value.name)


class Op(Space):
    activation = Problem(DataType, canonical=QONNX_DATATYPE)
```

The default structural codec covers built-ins, containers, enums and dataclasses
and refuses everything else by name; arbitrary `str()` or `repr()` never enters
a fingerprint. The codec identity and version travel in the digest, so changing
an encoding cannot be mistaken for a changed value.

Diagnostics translate findings into root-to-child scope, authored class and
member, branch case, and the projection asked for, while keeping the original
finding, path, values and causal trace intact. Ownership comes from an index
built once with the compiled model, parents before children, so a child `Input`
is attributed to the supplier that declares it and a generated path that no
declaration owns is reported as unattributed rather than guessed at.

Compilation is reused: one immutable `SpaceModel` per authored class, namespace
pair and declaration structure, holding the compiled tree, the validated
`DesignSpace`, the branch catalog, the projection metadata and the diagnostic
ownership index. `compile_space_model(...).start(problem)` is the
compiler-service entry and returns the authored class, exactly as
`Space.start(...)` does. What the model does *not* own is the runtime: every
start mints a fresh `Engine`, point, frozen problem and lock, so two independent
roots reuse compilation without sharing an evaluation cache or serializing on
each other.

Allocation goes through `Space._new_occurrence(OccurrenceContext)`. The default
allocates without calling `__init__`; a subclass whose instances need external
context -- a future `DataflowOp` around a `NodeProto` -- overrides it and
initializes from the context, which carries identity and the root but never the
Engine or the point.

The supported capability guarantee is the public surface: no public operation
returns an `Engine`, a `DesignPoint`, a `_Ref`, a compiled record or an
unrestricted path lookup, and contributor callbacks receive resolved declared
values only. `QualifiedPath` still appears inside every immutable `Finding`,
which is where a diagnostic needs it. Underscore-private attributes remain
inspectable by deliberately hostile code; this is normal Python privacy, not a
sandbox, and is not claimed to be one.

Each root lineage serializes access to its engine's evaluation caches with one
re-entrant lock, so concurrent reads and successor creation are deterministic
without
changing `_engine` semantics.

## Reusable Space fragments

Reusable spaces use `Input`, not `Problem`. `Use` binds each child input to a
typed value in the parent and flattens the child beneath a fresh namespace.

```python
from finn.dataflow.model import Decision, Input, Problem, Space, Use, derived


class Tile(Space):
    extent = Input(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, extent=extent, lanes=lanes)
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes

    exports = (lanes, cycles)


class Pair(Space):
    extent = Problem(int)
    left = Use(Tile, extent=extent)
    right = Use(Tile, extent=extent)

    @derived(int, left=left.cycles, right=right.cycles)
    def total(*, left: int, right: int) -> int:
        return left + right
```

The result is one flat `DesignSpaceSpec`, one `Engine`, and one `DesignPoint`.
There are no nested engine instances or runtime subspace objects. A child may
not introduce a `Problem`, every `Input` must be bound exactly once, and only
the values named in `exports` are visible to its parent. Reusing the same class
twice creates independently rebased declarations without mutating the class.

`Use(..., when=condition)` applies the existing engine applicability semantics
to the entire child fragment.

## Inheritance

Declarations are collected from the oldest `Space` base to the leaf class.
An override keeps the inherited position. It must retain the declaration
category and compatible value semantics; replacing a declaration with an
ordinary class value is an authoring error. New leaf declarations follow in
class-definition order.

The class body stores immutable templates with relative references only.
Compilation allocates paths and dependency references afresh, so compiling the
same class under several namespaces is deterministic and thread-safe.

## A one-Region Kernel

`Kernel` is the first specialization of `Space`. A Kernel class declares:

- a stable `id` and `version`;
- one `ComputationContract`;
- exactly one `Region(...)` member named `region`;
- physical-only decisions and its feasibility constraints;
- scalar physical `Parameter`s;
- an exact `ComponentABI`; and
- an ordered source closure.

```python
from typing_extensions import Self

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.model import Decision, Input, Kernel, Parameter, Region, RegionRefused
from finn.dataflow.region import DataflowRegion


def build_region(extent: int, lanes: int) -> DataflowRegion: ...


class ExampleKernel(Kernel):
    id = "example"
    version = "1"
    computation = ComputationContract("example.copy")

    extent = Input(int)
    lanes = Input(int)
    pipelined = Decision(bool, values=(False, True))

    region = Region(
        family="example.copy",
        version="1",
        construct=build_region,
        extent=extent,
        lanes=lanes,
    )

    LANES = Parameter(lanes)

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI(
            "example",
            (),
            (("LANES", str(configured.LANES)),),
        )
```

Kernel compilation automatically adds structural Region validation, a
whole-Kernel feasibility set, and a readiness profile. Configuration succeeds
only after all owned decisions, derived physical parameters, and constraints
resolve. The configured object retains its Region, local assignments, imported
decision provenance, physical parameter table, ABI, and source contributions.
It does not retain an `Engine`, `DesignPoint`, Network, node, edge, filesystem,
tool, or artifact store.

The ABI parameter table must exactly match the Kernel's resolved physical
parameter table. A physical constant uses `Parameter.constant(value, why=...)`
so the reason it is not a design-space value is explicit.

### RTL source ownership

Reusable RTL components are declared beneath their owning named source root;
the root is resolved by the caller and never embedded as an absolute path in a
Kernel or artifact identity. `replay_buffer` is owned by FinnLib at
`rtl/infra/replay_buffer.sv`. Both ReplayBufferKernel APIs and the legacy
MVAU/VVA custom operations consume that one source. FINN's
`finn-rtllib/mvu/mvu_vvu_axi.sv` continues to instantiate the unchanged module
ABI, but FINN no longer carries a second editable module definition.

`Region(...)` is one ordinary `DerivedProperty` that also names the compact
semantic family the resolved value belongs to. `@derived` is mechanically
sufficient but cannot say which family produced a Region, and a family field
parked beside a separate `@derived` drifts away from the value it labels. The
constructor stays a pure canonical function; it signals an infeasible request by
raising `RegionRefused`, which becomes a rejecting absence. Any other exception
is a defect and stays an `EvaluationError`.

### Design-owned semantics, Kernel-owned physics

A choice that changes any selected Region belongs to the enclosing Design and
reaches a Kernel as a typed `Input`. A Kernel-local `Decision` may only change
physical realization. The compiler enforces this: it walks the Region property's
transitive closure -- values *and* applicability -- and refuses any Kernel-owned
Decision in it, including one nested in a helper `Space`. A local Decision that
merely gates a helper the Region reads is still a refusal, because it makes the
Region present or absent.

The Kernel-to-Design interface has exactly one automatic value, the Region. A
concrete Kernel may not add public `exports`; a value a peer needs is a
Design-owned fact.

## Exclusive branching

`OneOf(Case(A, ...), Case(B, ...), outputs=(...))` embeds exactly one of several
child Spaces. Several cases generate one ordinary selector `Decision` over
stable case ids and gate every case fragment through it; a singleton generates
no selector but keeps the same selected-output paths, so adding an alternative
later renames nothing that already existed. Each `Case` owns its own exact
`Input` bindings, so alternatives may have unrelated Input vocabularies.

The declaration stores no selection algorithm. `compile_space_model()` returns
the ordinary spec plus a `BranchCatalog` of namespaces, selector paths, case ids,
and each case's decision, property, constraint, and readiness paths. An external
algorithm reads that, trials immutable successor points, and commits the
selector like any other decision. Nothing about that is Kernel-specific.

## A DataflowDesign

`DataflowDesign` owns every choice that changes a selected Region or the Network
they form. It consumes external facts only through `Input`, never `Problem`.

```python
from finn.dataflow.model import (
    Boundary,
    Case,
    Connection,
    DataflowDesign,
    Decision,
    Input,
    Kernels,
    Sink,
    configure_design,
    divisors_of,
)


class ExampleDesign(DataflowDesign):
    id = "example"
    version = "1"

    extent = Input(int)
    lanes = Decision(int, domain=divisors_of(extent))

    produce = Kernels(
        Case(ProducerKernel, extent=extent, lanes=lanes),
        computation=PRODUCE,
    )
    consume = Kernels(
        Case(ExampleKernel, extent=extent, lanes=lanes),
        Case(AlternativeKernel, width=extent, parallel_lanes=lanes),
        computation=CONSUME,
    )

    stream = Connection(produce.output("stream"), Sink(consume.input("input")))
    source = Boundary(produce.input("source"))
    result = Boundary(consume.output("output"))
```

`Kernels` is a thin `OneOf` specialization. It adds the required
`ComputationContract` that every candidate must declare, a default case id taken
from `Kernel.id`, the implicit selected Region output, and the stable Design role
and Network node id. Roles, node ids, and case ids are single path segments.

`Connection` and `Boundary` generate one ordinary property,
`semantic.<design>.network`, from the exact selected Regions. A `Sink` owns its
position map because a canonical fan-out is one Edge with several sink
contracts; omitting the map means the identity over the selected source port's
image. Endpoints carry only segment, port id, and expected direction -- the
Region owns the real port list -- so a claim the Region cannot honour is named
by canonical `validate_network` with the canon's own issue code. The one
Design-specific supplement is `segments_match_network`.

`when=` on a segment, a Connection, or a Boundary makes it conditional.
Complementarity is not proved syntactically; canonical endpoint ownership
rejects both-active and neither-active at every point.

### Configured Design

`configure_design(engine, compiled, point)` checks readiness, checks
feasibility, resolves each active segment's selection, configures exactly those
Kernels, resolves the Network, and proves each configured Kernel realizes its
node's Region. The result is an instance of the authored class holding the
selected Network, one Kernel per active role, the selected case ids,
Design-owned assignments, external decision provenance, and static
role/node/Region-family metadata -- and no Engine, point, compiled record,
unselected candidate, or branch catalog.

## The production slice

`DotProductDesign` composes `ReplayBufferKernel` and `DotpAxiKernel`. The Design
owns PE and SIMD once, because each changes both Regions and the beat contract on
the edge between them; DotpAxi keeps pumping, which preserves its Region exactly.
Neither Kernel imports a DataflowOp, a Design, an MVAU operation implementation,
or legacy Kernel authoring machinery. Direct evidence configures each from a flat
engine point and tests its RTL numerically and through OOC synthesis.

## Artifact boundary

The artifact substrate remains downstream and does not import this package.
The one-way helpers in `kernel_artifacts.py` turn a configured Kernel into
artifact-native values:

```text
configured Kernel + declared source roots
    -> ResolvedContributions
    -> Derivation for the reusable source closure
    -> PortableComponent carrying source ArtifactRef + ComponentABI
```

Artifact keys contain only values read by the corresponding artifact stage.
Occurrence namespaces and filesystem locations do not enter portable
identity. Stores, packaging formats, tool requests, and synthesis stay outside
the Kernel object.

Evaluator callbacks currently run under that lineage lock. Re-entry from the
same thread is safe because the lock is re-entrant; a callback that hands work
to another thread and waits for it would deadlock. Moving synchronization into
the engine's own cache operations is U7 work, and U1 deliberately changes no
`_engine` semantics.

## Deliberate boundary of this slice

U1 does not migrate Kernel, DataflowDesign, or DataflowOp to occurrences. This
package does not yet attach input supply, project ONNX graph context, compose
artifacts across Kernels, or persist a selection. `Region.family`/`version` and
the configured Design's role-to-node metadata preserve the seam a future
annotated-ONNX carrier would need; no ONNX object exists in the stack. Existing
upper-stack collection failures caused by retired artifact interfaces are
non-gating here.
