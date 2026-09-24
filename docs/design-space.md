# Space authoring and immutable specialization

`finn.kernels.space` provides the supported language and runtime for kernel
families. A Space class describes facts, choices, computations and views. Its
compiled model can create many independently bound occurrences. Parent and
child occurrences share one immutable specialization snapshot.

## A runnable family

```python
from finn.kernels.space import (
    Const, Decided, Decision, Inapplicable, Param, Space, Subspace,
    Unresolved, View, ViewKey, compile_space, constraint, derived,
    divisors_of, view,
)


class Tiles(Space):
    extent = Param(int)
    factor = Decision(int, domain=divisors_of(extent))

    @derived
    def cycles(*, extent: int, factor: int) -> int:
        return extent // factor

    @constraint
    def supported(*, extent: int) -> bool:
        return extent > 0

    @view(constraints=(supported,))
    def shape(*, factor: int, cycles: int) -> tuple[int, int]:
        return factor, cycles


tile_model = compile_space(Tiles)
tile_base = tile_model.start({Tiles.extent: 12})
assert isinstance(tile_base.answer(Tiles.factor), Unresolved)
tile_point = tile_base.assign(Tiles.factor, 3)
assert tile_point.cycles == 4
assert tile_point.shape().accepted_answer == Decided((3, 4))
```

`Tiles.start(parameters)` is the one-shot equivalent. Retain a model for repeated
starts with different inputs. Compilation validates structure without running
author callbacks. It freezes the meaning of its references; later class edits
cannot rewrite an existing model.

| Declaration | Class access | Occurrence access |
|---|---|---|
| `Param[T]` | Typed binding/reference handle | Supplied `T` |
| `Const[T]` | Typed definition-owned value handle | Frozen `T` |
| `Decision[T]` | Typed owning choice | Committed `T` |
| `Derived[T]` | Typed computation handle | Computed `T` |
| `View[T]` | Typed assessment declaration | Callable bound view returning `ViewAssessment[T]` |
| `Subspace[S]` | Typed placement and member references | Child occurrence of type `S` |

Use `point.answer(reference)` when a value may be unresolved, inapplicable or
refused. Direct descriptor reads require a decided value. A decision-state query
can return a decided **unassigned state** while the corresponding value query is
unresolved. Queries never choose a candidate or adopt a default.

## Signatures and value semantics

Derived functions, constraints and function-authored views bind arguments by
name after effective inherited members have been collected. Explicit aliases,
such as `@derived(a=activation.dtype, b=weights.dtype)`, override those names.
Postponed annotations are resolved with the function module and effective class
namespace. Arguments cannot have defaults, positional-only parameters, variadic
parameters or implicit `self`/`cls`.

Ordinary return annotations retain their value types. Custom values use an
explicit `ValueSemantics[T]` containing a compatibility token, name, recognition,
equality and snapshot callbacks. Snapshot callbacks must detach mutable state.
The default nominal policy copies values; generic return annotations such as
`tuple[int, ...]` use their nominal container origin. Supply custom semantics
when element recognition or domain equality is stronger than that policy.

```python
from dataclasses import dataclass
from finn.kernels.space import Answer, ValueSemantics, reject


@dataclass(frozen=True)
class Encoding:
    bits: int


ENCODING = ValueSemantics.immutable_nominal(Encoding)


class Encoded(Space):
    source = Param(ENCODING)

    @derived(semantics=ENCODING)
    def result(*, source: Encoding) -> Answer[Encoding]:
        if source.bits <= 0:
            return reject("encoding-width", "an encoding needs positive width")
        return Decided(source)
```

Explicit semantics retain the underlying `T` for answer-returning functions;
an annotation of `object` does not infer a narrower value type. Constraints may
return `True`, `False` or explicit semantic answers. Bare `False` becomes an
attributable refusal. A derived Boolean `False` remains an ordinary value.

Integer references support `+`, `-`, `*`, `//`, `%`, unary `-`, and reflected
literal forms. They create ordinary dependency nodes and reject symbolic
truthiness. Only this bounded integer vocabulary is folded; arbitrary Python
callbacks are not traced or executed during compilation.

## Binding and reusable scopes

A root binding map supplies every required exposed Param. Values are validated
and snapshotted before evaluation. `Param(required=False)` deliberately permits
omission: that absence is frozen for the start and is distinct from supplying
`None`. A later decision commitment cannot fill a missing input.

A child formal must be bound explicitly to a literal, an existing value
reference, a fresh exposed Param, or a fresh local Decision. The following uses
`Tiles` from the first example:

```python
class Pair(Space):
    extent = Param(int)
    first = Subspace(Tiles, extent=extent)
    second = Subspace(Tiles, extent=Param(int))


pair = Pair.start({Pair.extent: 12, Pair.second.ref(Tiles.extent): 18})
first_selected = pair.assign(Pair.first.decision_ref(Tiles.factor), 3)
assert first_selected.first.cycles == 4
assert isinstance(first_selected.second.answer(Tiles.factor), Unresolved)
assert isinstance(pair.first.answer(Tiles.factor), Unresolved)
```

`placement.ref(Child.member)` preserves the member's value type.
`placement.decision_ref(...)` is editable only when that placement owns the
Decision, including a fresh Decision bound to a child Param. An ordinary Param
alias stays read-only even if its supplier is a parent Decision. Editing a child
returns a successor child whose `.root` sees the same successor snapshot.

Nested exposed parameters can be supplied by an enclosing definition without
flat duplicate fields or constructed paths:

```python
class TileBlock(Space):
    tile = Subspace(Tiles, extent=Param(int))


class Board(Space):
    size = Param(int)
    block = Subspace(
        TileBlock,
        bindings={TileBlock.tile.ref(Tiles.extent): size},
    )


board = Board.start({Board.size: 12})
assert board.block.tile.extent == 12
```

Mapped targets must be deliberately exposed Params. Existing aliases, constants
and owning Decisions cannot be overwritten by that map. Repeated placements
retain independent local choices and share only their explicitly supplied values.

## Views, guards and choices

`View(value, constraints=(condition,))` and `@view(...)` share one reducer. An
assessment exposes `output_answer`, `readiness`, `constraints`, and
`accepted_answer`. Output and acceptance prerequisites are always included;
additional `requires=` obligations default to empty. Named `Readiness` and
`ConstraintGroup` declarations are optional reuse mechanisms.

Known constraint refusals remain visible in `assessment.constraints.answers`
while other unresolved obligations may keep `accepted_answer` unresolved. A raw
output is not a claim of acceptance. `placement.accepted(Child.view)` consumes
the exact accepted answer used by a direct child view call.

`when=` is an explicit Boolean control reference on computations, Decisions,
constraints, views, placements and structural choices. An outer false guard
settles inapplicability before inner guards or callbacks are demanded:

```python
class OptionalTiles(Space):
    enabled = Param(bool)
    implementation = Subspace(Tiles, extent=12, when=enabled)


disabled = OptionalTiles.start({OptionalTiles.enabled: False})
assert isinstance(disabled.implementation.answer(Tiles.factor), Inapplicable)
```

`SubspaceChoice` uses typed export keys to share contracts across heterogeneous
alternatives. It needs no common concrete child class:

```python
from finn.kernels.space import SubspaceChoice

OUTPUT = ViewKey("output", int)


class Small(Space):
    width = Const(4)
    physical = View(width)
    exports = {OUTPUT: physical}


class Wide(Space):
    width = Const(8)
    physical = View(width)
    exports = {OUTPUT: physical}


class Implementation(Space):
    choice = SubspaceChoice(
        {"small": Subspace(Small), "wide": Subspace(Wide)},
        exports=(OUTPUT,),
    )
    physical = View(choice.accepted(OUTPUT))


implementation = Implementation.start()
selected = implementation.choice.select("wide")
assert selected.occurrence.assess(Implementation.physical).accepted_answer == Decided(8)
```

Only the selected alternative is demanded. Rejection from its accepted view is
preserved. A singleton choice needs no selector commitment. Choices, fields and
views retain source ownership for diagnostics; potential dependencies are still
validated conservatively for cycles.

## Extension authoring

`ScopeBuilder` expands ordinary declarations into one child template. It can
add Params, Consts, Decisions, derived functions, constraints and views; `.add`
also accepts ordinary declaration objects. Typed export helpers fix the key's
type before accepting a value or view.

```python
from finn.kernels.space import ScopeBuilder, ValueKey

LATENCY = ValueKey("latency", int)


def latency(*, cycles: int, overhead: int) -> int:
    return cycles + overhead


builder = ScopeBuilder(Tiles, name="BufferedTiles")
builder.const("overhead", 1)
latency_ref = builder.derived("latency", latency)
builder.export(LATENCY).value(latency_ref)
builder.bind(Tiles.extent, Param(int))
template = builder.finish()
assert builder.finish() is template


class TileArray(Space):
    first = builder.place()
    second = builder.place()


array = TileArray.start({
    TileArray.first.ref(Tiles.extent): 12,
    TileArray.second.ref(Tiles.extent): 18,
})
array = array.assign(TileArray.first.decision_ref(Tiles.factor), 3)
assert array.answer(TileArray.first.ref(LATENCY)) == Decided(5)
```

Finishing seals mutations; repeated placements reuse the template and own
independent occurrence state. Inherited fields retain the base Space's static
type. Wrapper properties may delegate to standard `ref` and `accepted` handles.
Use `.binding(nested_parameter_ref).to(supplier)` for a builder's nested binding.
Dynamic admission limits must be local Params with explicit suppliers. Capturing
a parent reference in a child callback does not create a dependency binding.
The builder has no runtime access and cannot replace compiled records.

## Atomic refinement and sparse replay

Edits retain their exact base snapshot. All requests are checked and snapshotted
before evaluators run; dependent edits are assessed in dependency order, not
the submitted order. Publication is atomic:

```python
batch = pair.refine(
    pair.edit(Pair.first.decision_ref(Tiles.factor), 3),
    pair.edit(Pair.second.decision_ref(Tiles.factor), 6),
)
assert batch.accepted
assert batch.point.first.cycles == 4
assert batch.point.second.cycles == 3
```

`RefinementReport.outcomes` distinguishes unchanged, provisional, refused and
committed items. A refused batch retains the base point; equal recommits are
no-ops and changed recommits conflict. The strict `assign` convenience raises
`RefinementError` with its report for a refused candidate. Malformed binding or
edit requests raise `RequestError`; programmer failures remain `EvaluationError`
with owner, evaluator role and original cause. These exceptions are available
from `finn.kernels.space.errors`.

Selections are detached, immutable sparse commitments. Capture includes each
committed owning Decision and nontrivial selector once, including a value equal
to the first candidate. It omits aliases, derived values and evaluator state:

```python
from finn.kernels.space import selections

saved = selections.capture(tile_point)
edited = saved.with_changes([saved.edit(Tiles.factor, 4)])
replayed = selections.restore(tile_base, edited)
assert replayed.accepted and replayed.point.factor == 4
changed_inputs = tile_model.start({Tiles.extent: 10})
assert not selections.restore(changed_inputs, edited).accepted
```

`Selection.remove(reference)` creates a detached removal request. Changing a
selector does not silently prune old case entries; remove them explicitly or
replay reports their inapplicability. Restore is a root operation using ordinary
atomic refinement. In-process selections identify the exact compiled model.

Portable encoding additionally requires an explicit family/version and a codec
for every saved choice. No codec, fingerprint or global registry is needed to
start, query or capture an ordinary model:

```python
from finn.kernels.space import JSONValue, SelectionSchema, ValueCodec, codec_for, codecs


def decode_integer(value: JSONValue) -> int:
    if type(value) is not int:
        raise ValueError("expected an integer")
    return value


integer_codec: ValueCodec[int] = ValueCodec("integer", 1, lambda value: value, decode_integer)
schema = SelectionSchema(
    tile_model, family="example.tiles", version=1,
    bindings=(codec_for(Tiles.factor, integer_codec),),
    owned_keys=("retired-factor",),
)
document = codecs.encode(saved, schema)
assert codecs.decode(document, schema) == saved
```

Encoding checks that each codec preserves declared value equality. Decoding
checks family/version, stable keys, duplicates and codec identifiers before
decoding values; replay validates applicability and domains. Public selection
values and codec inputs are detached snapshots. `selections.replace_owned`
returns a mapping with explicitly owned current/obsolete keys replaced while
preserving unrelated entries. It performs no external write or graph transaction.

## Answers and inspection

`Answer[T]` is `Decided[T] | Inapplicable | Rejected | Unresolved`. Scalar answers
reject Boolean coercion. Inspect their variant and findings. Unresolved findings
distinguish missing immutable inputs from commitment blockers.

Ordinary dependencies require values. `optional(source)` admits explicit
`MissingInput` / `NotApplicable` markers while preserving rejection;
`full_answer(source)` supplies the entire `Answer[T]`. Advanced answer-aware
callbacks still owe deterministic, monotone behavior. A decided fallback that
later changes after commitment violates that contract.

`inspection.members`, `decisions` and `choices` discover frozen metadata without
running evaluators. Typed `value_handle` and `decision_handle` references belong
to exactly one compiled model. `inspection.dependencies` reports static direct
dependencies; `inspection.explain` evaluates a query and returns its demanded
evidence, including parameter presence, owning decisions, guards/selectors and
constraint causes. Cached reads retain the same evidence. `statistics(model)`
reports structural counts without evaluating the model.

The optional `conformance.MonotonicityHarness().verify(base, samples)` compares
settled observations across caller-provided edits/batches. It reports violations,
skipped samples and no-ops using declared equality, including unhashable values.
It is empirical checking, not a general proof or a solver.

## Kernel boundary and retired interfaces

`finn.kernels.base.Kernel` adds identity and public capability metadata. It does
not require streams, clocks or an RTL ABI. [Physical kernels](../src/finn/kernels/README.md)
return explicit detached requirements: HLS source interfaces and synthesized RTL
interfaces remain different products. Artifact roots, stores, rendering and
physical algorithms remain in their existing layers.

The old `_engine` runtime, `Engine`/`DesignPoint`/`DesignSpaceSpec`,
`Input`/`Problem`/`Projection` spellings, property-style view access, root-factory
reconstruction hooks and compiled-record replacement hooks have been retired.
There is one supported runtime. Proposal scheduling, solvers, cross-snapshot
cache sharing and graph integration are outside this delivery.

The experimental `finn.dataflow` consumers still require an intentional port.
ONNX parsing, graph staleness checks, nodeattr writes and graph transactions are
adapter responsibilities. The independent kernel gate and historical dataflow
validation records do not establish post-cutover dataflow compatibility.
