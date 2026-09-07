# The declarative dataflow stack

Design spaces are declared as ordinary Python classes. This is the canonical
stack: the experimental authoring, Kernel, Design and operation APIs it
replaces were removed in U1.5 and survive only as behavior oracles in their own
worktrees.

Five packages, one frontend, one flat spec:

```text
finn.dataflow.model     the canonical DataflowRegion and DataflowNetwork model
finn.dataflow.space     the generic Space language, compiler and occurrences
finn.dataflow.kernels   the Kernel contract and the reusable Kernels
finn.dataflow.designs   the generic Design contract and topology
finn.dataflow.ops       DataflowOps and their own Design inventories
```

The first two names say what they own and nothing else. `model` is the detached
semantic value -- what a Region and a Network *are*; `space` is the language a
contributor authors a design space in. The dependency runs one way: `space` may
import `model`, and `model` imports neither `space` nor the engine. Nothing is
re-exported from `finn.dataflow` itself, so every value has one import path.

The distinction has a vocabulary consequence worth stating once:

```text
RegionDeclaration   a Kernel's authoring recipe, in finn.dataflow.kernels
DataflowRegion      the detached normalized value it produces, in .model
```

Four layers share that frontend and lower to one flat spec:

```text
Space                  ordinary declarations, direct Subspace composition,
                       and SubspaceChoice structural choice
   |
Kernel(Space)          semantic Inputs, physical-only Decisions,
                       one RegionDeclaration(family, version, construct, **deps)
   |
DataflowDesign(Space)  semantic Decisions, Kernels segments,
                       explicit Connections and Boundaries,
                       one selected canonical DataflowNetwork
   |
DataflowOp             one ONNX node, frozen source facts, a closed set of
   (holds a root)      Designs, and the choices that survive a save
   |
   -> flat DesignSpaceSpec -> _engine validation and evaluation
   -> two projections per layer: what it *means*, and what it *builds*
```

`DesignSpaceSpec`, `Engine`, and `DesignPoint` remain the normalized IR and
runtime. They are private implementation details of the occurrence API for
ordinary contributors; the space package remains a source-language frontend,
not another design-space evaluator.

## A closed Space

A root `Space` may declare problem fields and decisions directly on its class.
Derived values and constraints name their dependencies explicitly.

```python
from finn.dataflow.space import Decision, Problem, Space, compile_space, derived


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
from finn.dataflow.space import ConstraintGroup, Projection, Readiness, constraint


class Tile(Space):
    # declarations as above
    @constraint(lanes=lanes)
    def positive(*, lanes: int) -> bool:
        return lanes > 0

    legal = ConstraintGroup(positive)
    ready = Readiness(decisions=(lanes,), properties=(cycles,), constraints=legal)
    schedule = Projection(cycles, readiness=ready, constraints=(legal,))


root = Pair.start({Pair.extent: 16})
left = root.left
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

Navigation is descriptor-based and therefore exact. A Python class names a
family, not one of its occurrences, so there is no class-keyed navigation verb
at all:

```python
root.left.assign(Tile.lanes, 2)  # the exact member that places the child
choice = root.implementation  # the bound ChoiceView
choice.alternatives  # ("fast", "small")
choice.selected()  # Answer[str]
choice = choice.select("fast")  # ordinary selector assignment
fast = choice.alternative("fast")
```

A `ChoiceView` exposes alternative ids, selection, and exact alternative views
without exposing its generated selector path. `answer`, `assign`, `assess` and
`project` accept declarations, never path strings.

`assess` takes a `Readiness` profile or a named `ConstraintGroup`. A bare
`Constraint` is an engine and compiler unit; only the named group can be shared
between projections, renamed independently of its call sites, and pointed at in
a diagnostic.

Eleven member names are reserved -- `start assign answer assess project
diagnostics root problem_snapshot problem_fingerprint is_stale reconstruct` --
because each is a lifecycle operation every authored class inherits. Six private
protocol names are reserved the same way -- `_new_occurrence`
`_finalize_compilation` `_occurrence_state` `_space_value` `_implicit_exports`
`exports` -- against a *declaration* bound to them, never against overriding the
method or metadata itself. Both checks are on the Python member name only, so
`choice = SubspaceChoice(..., name="branch")` keeps the compiled path
`<ns>.branch`.

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

Compilation is reused by holding it, not by hiding it.
`compile_space_model(Pipeline, ...)` returns a `SpaceModel[Pipeline]` -- the
compiled tree, the validated `DesignSpace`, the branch catalog, the projection
metadata and the diagnostic ownership index -- and `model.start(problem)` is the
compiler-service entry, returning a `Pipeline` exactly as `Pipeline.start(...)`
does. `Space.start(...)` is the ergonomic one-shot spelling and compiles a fresh
model each time. There is deliberately no implicit class-level cache: a class
body is ordinary mutable Python, and no key can honestly cover `exports`,
`_implicit_exports` and specialization metadata until class immutability is a
real contract. What the model does *not* own is the runtime: every start mints a
fresh `Engine`, point, frozen problem and lock, so two independent roots reuse
compilation without sharing an evaluation cache or serializing on each other.

Allocation goes through `Space._new_occurrence(OccurrenceContext)`. The default
allocates without calling `__init__`; a subclass whose instances need external
context -- a future `DataflowOp` around a `NodeProto` -- overrides it and
initializes from the context, which carries identity and the root but never the
Engine or the point. Every root, child and successor must be a *fresh* instance:
a hook that returns something it made earlier is refused before any state is
written, because overwriting an attached occurrence would silently mutate the
old one into the new. The concrete `DataflowOp` attachment channel is
intentionally deferred to U4; U1 proves the seam, not the payload.

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

Reusable spaces use `Input`, not `Problem`. `Subspace` binds each child input
to a typed value in the parent and flattens the child beneath a fresh namespace.

```python
from finn.dataflow.space import Decision, Input, Problem, Space, Subspace, derived


class Tile(Space):
    extent = Input(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, extent=extent, lanes=lanes)
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes

    exports = (lanes, cycles)


class Pair(Space):
    extent = Problem(int)
    left = Subspace(Tile, extent=extent)
    right = Subspace(Tile, extent=extent)

    @derived(int, left=left.cycles, right=right.cycles)
    def total(*, left: int, right: int) -> int:
        return left + right
```

The result is one flat `DesignSpaceSpec`, one `Engine`, and one `DesignPoint`.
There are no nested engine instances or runtime subspace objects. A child may
not introduce a `Problem`, every `Input` must be bound exactly once, and only
the values named in `exports` are visible to its parent. Reusing the same class
twice creates independently rebased declarations without mutating the class.

`Subspace(..., when=condition)` applies the existing engine applicability
semantics to the entire child fragment. Class access returns the declaration and
instance access returns the bound child occurrence, so `Pair.left` is a
`Subspace` and `pair.left` is a `Tile`.

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
- exactly one `RegionDeclaration(...)` member named `region`;
- physical-only decisions and its feasibility constraints;
- scalar physical `Parameter`s;
- an exact `ComponentABI`; and
- an ordered source closure.

```python
from typing_extensions import Self

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.kernels.kernel import Kernel, Parameter, RegionDeclaration
from finn.dataflow.model import DataflowRegion, RegionRefused
from finn.dataflow.space import Decision, Input


def build_region(extent: int, lanes: int) -> DataflowRegion: ...


class ExampleKernel(Kernel):
    id = "example"
    version = "1"
    computation = ComputationContract("example.copy")

    extent = Input(int)
    lanes = Input(int)
    pipelined = Decision(bool, values=(False, True))

    region = RegionDeclaration(
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

`RegionDeclaration(...)` is one ordinary `DerivedProperty` that also names the compact
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

## Structural variation

A `SubspaceChoice` owns an ordered mapping from stable alternative id to `Subspace`:

```python
implementation = SubspaceChoice(
    {
        "fast": Subspace(FastImplementation, extent=extent),
        "small": Subspace(SmallImplementation, extent=extent),
    },
    outputs=("cycles",),
)
```

Several alternatives generate one ordinary selector `Decision` over those ids
and gate every alternative fragment through it; a singleton generates no
selector but keeps the same selected-output paths, so adding an alternative
later renames nothing that already existed. Each alternative `Subspace` owns its
own exact `Input` bindings, so alternatives may have unrelated Input
vocabularies and unrelated Space classes.

There is one nested-space declaration, not two. The mapping key *is* the
alternative id, so nothing restates it; the exclusivity, the selector and the
selected outputs all belong to the container. A per-alternative `when=` is
refused during this phase: the choice owns the outer condition, and
candidate-specific applicability is a separate question no case has yet forced.
Internal compiler vocabulary -- `BranchCatalog`, `BranchInfo`, `CaseInfo` --
still describes the lowered branch IR and is not the authoring vocabulary.

A layer specializes a choice through three methods and nothing else:
`candidate_id(subspace)` names a candidate that already carries its own stable
id, so nothing writes that id twice; `validate_candidate(owner, subspace)`
refuses a candidate the layer does not admit; `default_outputs()` supplies the
selected outputs the layer implies. A specialization that takes positional
candidates initializes through `_from_candidates`, and both spellings pass
through the same structural check, so supplying ids is not permission to supply
something that is not a `Subspace`.

`validate_candidate` is reached only through the private `_admit_candidate`,
which attaches the declaring member's name -- one wrapper, called by the
compiler and by any layer that must admit a candidate before compilation. It is
not public authoring vocabulary. Everything else -- selector creation, gating,
namespaces, forwarding, the view, and the persistence identity -- is the one
generic implementation, and a specialization neither overrides nor duplicates
it. `selector_name` renames the generated selector; it does not add a second
one.

The compiled root-relative declaration name is the stable persistence
*identity*. `name=` overrides the local segment, and there is no second naming
mechanism: `occurrence_persistable()` reports every selector and `Decision`
beneath one root under that identity, selectors first, with the root namespace
removed so the same document reloads under any root.

`name=None` means “use the Python member name.” Any explicit `name=` is one
non-empty `QualifiedPath` segment using only ASCII letters, digits, `_`, and
`-`; a dot is refused because structural nesting comes from actual `Subspace`
or `SubspaceChoice` declarations. An invalid explicit name never falls back to
the member name.

That identity is not a serialized spelling. This package writes nothing down and
fixes no attribute name; the operation persistence layer maps the identity
deterministically onto native ONNX attribute names and owns the collision rule
for them, so how a nested selector finally reads to a user is its decision, not
this one's.

The declaration stores no selection algorithm. `compile_space_model()` returns
the ordinary spec plus a `BranchCatalog` of namespaces, selector paths, case ids,
and each case's decision, property, constraint, and readiness paths. An external
algorithm reads that, trials immutable successor points, and commits the
selector like any other decision. Nothing about that is Kernel-specific.

## A DataflowDesign

`DataflowDesign` owns every choice that changes a selected Region or the Network
they form. It consumes external facts only through `Input`, never `Problem`.

```python
from finn.dataflow.designs import Boundary, Connection, DataflowDesign, Kernels, Sink
from finn.dataflow.space import Decision, Input, Subspace, divisors_of


class ExampleDesign(DataflowDesign):
    id = "example"
    version = "1"

    extent = Input(int)
    lanes = Decision(int, domain=divisors_of(extent))

    produce = Kernels(
        Subspace(ProducerKernel, extent=extent, lanes=lanes),
        computation=PRODUCE,
    )
    consume = Kernels(
        Subspace(ExampleKernel, extent=extent, lanes=lanes),
        Subspace(AlternativeKernel, width=extent, parallel_lanes=lanes),
        computation=CONSUME,
    )

    stream = Connection(produce.output("stream"), Sink(consume.input("input")))
    source = Boundary(produce.input("source"))
    result = Boundary(consume.output("output"))
```

`Kernels` is a thin `SubspaceChoice` specialization written through the
three-method seam -- `candidate_id`, `validate_candidate`, `default_outputs` --
and nothing else. It adds the required `ComputationContract` that every
candidate must declare, the implicit selected Region output, and the stable
Design role and Network node id. Its alternatives
are positional rather than a mapping because a Kernel already carries its own
stable `id`, and that id *is* the alternative id; `Subspace(..., name=...)`
aliases it, which is what lets one Kernel class fill two candidate slots. Roles,
node ids, and candidate ids are single path segments.

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

### Asking a Design

`design.dataflow` is the Design's one question and there is no resolved-Design
wrapper in between. Everything else is asked of the same attached occurrence:

```python
design.dataflow  # ProjectionAssessment[DataflowNetwork]
design.roles  # ("replay", "compute")
design.region(role)  # Answer[DataflowRegion]
design.region_family(role)  # Answer[(family, version)]
design.computation(role)  # ComputationContract
design.node_id(role)  # the stable Network node
design.is_active(role)  # Answer[bool]
design.selected(role)  # Answer[str], the candidate id
design.kernel(role)  # Answer[Kernel], the child occurrence
design.assignments  # Design-owned, including its selectors
design.imported_decisions  # provenance
```

The Network resolves from semantics alone. Its readiness profile lists the
Network and the selected Regions and nothing else. Its constraint set holds the
Design's own constraints **and every candidate constraint that is not
physical-only**, so a Kernel refusing its Region refuses the Network, while one
that cannot be *built* at this configuration still contributes a Region.
Physical-only is a difference — `physical_support` minus `dataflow_support` — so
a constraint an author classifies twice gates both projections rather than
neither. An unselected candidate's constraints report as *not applicable*, not
as refusals. `design_dataflow(engine, compiled, point)` is the same projection
for a caller holding a compiled fragment.

## The production slice

`DotProductDesign` -- MVAU's own, under `ops/mvau/designs/` -- composes
`ReplayBufferKernel` and `DotpAxiKernel`. The Design
owns PE and SIMD once, because each changes both Regions and the beat contract on
the edge between them; DotpAxi keeps pumping, which preserves its Region exactly.
Neither Kernel imports a DataflowOp, a Design, or an MVAU operation
implementation. Direct evidence configures each from a flat
engine point and tests its RTL numerically and through OOC synthesis.

## Artifact boundary

The artifact substrate remains downstream and does not import this package.
The one-way helpers in `kernels/artifacts.py` take a **detached**
`KernelPhysicalResult` -- never an occurrence -- and turn it into
artifact-native values:

```text
kernel.physical -> KernelPhysicalResult + declared source roots
    -> ResolvedContributions
    -> Derivation for the reusable source closure
    -> PortableComponent carrying source ArtifactRef + ComponentABI
```

A build unit therefore holds no handle back into the design space, and an
implementation with no realization for a configuration its Region accepts says
so by raising `PhysicallyUnsupported` rather than by inventing an ABI.

Artifact keys contain only values read by the corresponding artifact stage.
Occurrence namespaces and filesystem locations do not enter portable
identity. Stores, packaging formats, tool requests, and synthesis stay outside
the Kernel object.

Evaluator callbacks currently run under that lineage lock. Re-entry from the
same thread is safe because the lock is re-entrant; a callback that hands work
to another thread and waits for it would deadlock. Moving synchronization into
the engine's own cache operations is U7 work, and U1 deliberately changes no
`_engine` semantics.

## Operations

`finn.dataflow.ops.base.DataflowOp` is a QONNX `CustomOp` that *wraps* a root
occurrence rather than being one:

```text
NodeProto + ModelWrapper + build config
    -> read once   SourceNode        (frozen; the graph is not reread)
    -> freeze      Problem snapshot  + fingerprint
    -> start       root occurrence
    -> hydrate     node attributes replayed as ordinary assignments
    -> ask         network(), association()
```

Persistence is one authority, on the node: scope id, family, version, problem
fingerprint, and one attribute per persistent Decision. A persisted Decision
declares its *route* -- a navigator from the root plus the declaration object --
so it survives the root being started under another namespace. A changed
problem is refused, never rebased.

`SourceAssociation` is logical only. It records the tensor, the Network
boundary (or none), the node and port reached, and the coordinate
correspondence; it carries no Kernel identity, component id or artifact key,
and it is read off the resolved Network, so a matrix produced internally
reports no boundary while a pumped core and an unpumped one associate the same.

MVAU and ActivationReplay are the two operations, deliberately unalike -- two
Design alternatives against one fixed `Subspace`, three operands against two --
because the only way to know the layer is not MVAU-shaped is for something that
is not MVAU to use it unchanged.

## Deliberate boundary

Physical composition is a contract, not an implementation: there is no
`DesignPhysicalResult`, no wrapper generation and no composed packaging here.
Nor is there a specialization policy, ONNX lowering, or a Region CustomOp.
`RegionDeclaration.family`/`version` and the Design's role-to-node metadata preserve the
seam a future annotated-ONNX carrier would need; no ONNX object reaches the
engine.

Evaluator callbacks run under the per-lineage lock. Re-entry from the same
thread is safe because the lock is re-entrant; a callback that hands work to
another thread and waits for it would deadlock. Moving synchronization into the
engine's own cache operations is later work, and nothing so far changes
`_engine` semantics -- its diff from the pre-unified baseline is empty.
