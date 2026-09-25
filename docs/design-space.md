# Space authoring and immutable specialization

`finn.core.space` provides the supported language and runtime for kernel
families and other consumers. A Space class describes facts, choices, computations
and views. Its prepared definition can create many independently bound configurations. Parent and
child configurations share one immutable specialization snapshot.

## A runnable family

```python
from finn.core.space import (
    Const, Available, Decision, Inapplicable, Param, Space, Subspace,
    Unresolved, View, ViewKey, compile_space, constraint, derived,
    divisors_of, view,
)


class Tiles(Space):
    extent = Param(int)
    factor = Decision(int, domain=divisors_of(extent))

    @derived
    def cycles(self) -> int:
        return self.extent // self.factor

    @constraint
    def supported(self) -> bool:
        return self.extent > 0

    @view(constraints=(supported,))
    def shape(self) -> tuple[int, int]:
        return self.factor, self.cycles


tile_model = compile_space(Tiles)
tile_base = tile_model.bind(extent=12)
assert isinstance(tile_base.query(Tiles.factor), Unresolved)
tile_configuration = tile_base.with_choices(factor=3)
assert tile_configuration.cycles == 4
assert tile_configuration.shape() == (3, 4)
assert tile_configuration.shape.inspect().accepted_result == Available((3, 4))
assert tile_configuration.view(Tiles.shape)() == (3, 4)
assert tile_configuration.field(Tiles.cycles).get() == 4
assert tile_configuration.field(Tiles.factor).query() == Available(3)
```

`Tiles(extent=12)` is the normal one-shot form. Retain a prepared definition and call `bind`
for repeated construction with different inputs. Both paths reuse the same
prepared definition. Preparation validates structure without running author callbacks
and finalizes declaration structure for that family.

| Declaration | Class access | Configuration access |
|---|---|---|
| `Param[T]` | Typed binding/reference handle | Supplied `T` |
| `Const[T]` | Typed definition-owned value handle | Frozen `T` |
| `Decision[T]` | Typed owning choice | Committed `T` |
| `Derived[T]` | Typed computation handle | Computed `T` |
| `View[T]` | Typed output declaration | Callable bound view returning accepted `T` |
| `Subspace[S]` | Typed placement and member references | Child configuration of type `S` |

Use `configuration.query(reference)` when a value may be unresolved, inapplicable or
refused. `configuration.field(reference)` binds typed inspection to the current snapshot:
its `get()` returns a value and `query()` returns a structured result.
Views use one `BoundView[T]` through descriptor access, `view(reference)`, or
`field(reference)`, with call/get, query, and inspection operations. Decisions
also expose `state`, `candidates()`, `change(value)`, and `clear()` on that bound
field. These operations avoid repeating the configuration at each call. Direct descriptor
reads require an available value. `require_value(result)` provides the same
explicit unwrap and raises `ValueUnavailableError` for a non-value result.

## Signatures and value semantics

Derived computations, constraints and views are ordinary methods with `self`.
Read declared values through `self.member`; a child such as `self.first` has its
concrete authored Space type. Reads schedule prerequisites as they are reached.
No dependency list, symbolic-value proxy or type-checker plugin is needed.
Postponed annotations resolve using the function module and effective class
namespace.

Generic builders and domain providers also support explicit-input pure
functions. Their arguments bind by name after inherited members have been
collected; explicit aliases such as `@derived(a=activation.dtype)` override the
names. This advanced construction facility uses the same evaluator. Combining
`self` binding with explicit argument aliases is an error. Explicit-input
functions cannot have argument defaults or variadic parameters.

Ordinary return annotations retain their value types. Custom values use an
explicit `ValueSemantics[T]` containing a compatibility token, name, recognition,
equality and snapshot callbacks. Recognition and snapshot hooks are pure transformations of supplied values;
they must not read configuration state. Snapshot callbacks must detach mutable
state without changing the input. Equality receives defensive copies.
The default nominal policy copies values; generic return annotations such as
`tuple[int, ...]` use their nominal container origin. Supply custom semantics
when element recognition or domain equality is stronger than that policy.

```python
from dataclasses import dataclass
from finn.core.space import QueryResult, ValueSemantics, reject


@dataclass(frozen=True)
class Encoding:
    bits: int


ENCODING = ValueSemantics.immutable_nominal(Encoding)


class Encoded(Space):
    source = Param(ENCODING)

    @derived(semantics=ENCODING)
    def result(self) -> QueryResult[Encoding]:
        if self.source.bits <= 0:
            return reject("encoding-width", "an encoding needs positive width")
        return Available(self.source)
```

Explicit semantics retain the underlying `T` for answer-returning functions;
an annotation of `object` does not infer a narrower value type. Constraints may
return `True`, `False` or explicit semantic answers. Bare `False` becomes an
attributable refusal. A derived Boolean `False` remains an ordinary value.

Integer references support `+`, `-`, `*`, `//`, `%`, unary `-`, and reflected
literal forms. They create ordinary dependency nodes and reject symbolic
truthiness. Expressions are type-checked during preparation and evaluated lazily
inside each snapshot. Preparation does not evaluate arithmetic or authored callbacks.

Ordinary pure Python remains available inside declared functions:

```python
class Geometry(Space):
    depth = Param(int)

    @derived
    def address_bits(self) -> int:
        return max(1, (self.depth - 1).bit_length())


assert Geometry(depth=17).address_bits == 5
```

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


pair = Pair({Pair.second.ref(Tiles.extent): 18}, extent=12)
first_selected = pair.with_choices(
    pair.field(Pair.first.decision_ref(Tiles.factor)).change(3)
)
assert first_selected.first.cycles == 4
assert isinstance(first_selected.second.query(Tiles.factor), Unresolved)
assert isinstance(pair.first.query(Tiles.factor), Unresolved)
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


board = Board(size=12)
assert board.block.tile.extent == 12
```

Mapped targets must be deliberately exposed Params. Existing aliases, constants
and owning Decisions cannot be overwritten by that map. Repeated placements
retain independent local choices and share only their explicitly supplied values.

## Views, guards and choices

`View(value, constraints=(condition,))` and `@view(...)` share one reducer. A bound view
call returns its accepted value or raises `ValueUnavailableError`. Use
`configuration.member.inspect()` or `configuration.inspect(ViewReference)` for
an assessment; `configuration.view(ViewReference)` binds a view generically.
The assessment exposes `output_result`, `readiness`, `constraints`, and
`accepted_result`. Output and constraint dependencies determine acceptance. `ConstraintGroup` can
name a reusable set of constraints; readiness is a diagnostic derived from the
output and constraint results.

Known constraint refusals remain visible in `assessment.constraints.results`
while other unresolved obligations may keep `accepted_result` unresolved. A raw
output is not a claim of acceptance. `placement.accepted(Child.view)` consumes
the exact accepted answer used by a direct child view call.

`when=` is an explicit Boolean control reference on computations, Decisions,
constraints, views, placements and structural choices. An outer false guard
settles inapplicability before inner guards or callbacks are demanded:

```python
class OptionalTiles(Space):
    enabled = Param(bool)
    implementation = Subspace(Tiles, extent=12, when=enabled)


disabled = OptionalTiles(enabled=False)
assert isinstance(disabled.implementation.query(Tiles.factor), Inapplicable)
```

`SubspaceChoice` uses typed export keys to share contracts across heterogeneous
alternatives. It needs no common concrete child class:

```python
from finn.core.space import SubspaceChoice

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


implementation = Implementation()
selected = implementation.choice.select("wide")
assert selected.instance.view(Implementation.physical)() == 8
assert selected.instance.inspect(Implementation.physical).accepted_result == Available(8)
```

Only the selected alternative is demanded. Rejection from its accepted view is
preserved. A singleton choice needs no selector commitment. Choices, fields and
views retain source ownership for diagnostics. Preparation validates known
structural references and explicit dependencies. Cycles reached through method
body reads fail during evaluation with scoped context.

## Extension authoring

`ScopeBuilder` expands ordinary declarations into one child template. It can
add Params, Consts, Decisions, derived functions, constraints and views; `.add`
also accepts ordinary declaration objects. Typed export helpers fix the key's
type before accepting a value or view.

```python
from finn.core.space import ScopeBuilder, ValueKey

LATENCY = ValueKey("latency", int)


def latency(*, cycles: int, overhead: int) -> int:
    return cycles + overhead


builder = ScopeBuilder(Tiles, name="BufferedTiles")
builder.add("overhead", Const(1))
latency_ref = builder.add("latency", derived(latency))
builder.export(LATENCY).value(latency_ref)
builder.bind(Tiles.extent, Param(int))
template = builder.finish()
assert builder.finish() is template


class TileArray(Space):
    first = builder.place()
    second = builder.place()


array = TileArray({
    TileArray.first.ref(Tiles.extent): 12,
    TileArray.second.ref(Tiles.extent): 18,
})
array = array.with_choices(array.field(TileArray.first.decision_ref(Tiles.factor)).change(3))
assert array.query(TileArray.first.ref(LATENCY)) == Available(5)
```

Use `builder.add(name, declaration)` for Params, Consts, Decisions, computations,
constraints and views. The returned declaration retains its exact type. Python
constructs the declaration before calling `add`: in particular, `Const(value)`
snapshots its value even if `add` subsequently rejects a sealed builder.

Finishing seals mutations; repeated placements reuse the template and own
independent configuration state. Inherited fields retain the base Space's static
type. Wrapper properties may delegate to standard `ref` and `accepted` handles.
Use `.binding(nested_parameter_ref).to(supplier)` for a builder's nested binding.
Dynamic admission limits must be local Params with explicit suppliers. Capturing
a parent reference in a child callback does not create a dependency binding.
The builder has no runtime access and cannot replace compiled records.

## Configuration replacement and sparse replay

`with_choices` creates a revised configuration over the same frozen facts. It
can add, replace, or clear choices in one order-independent batch. The original
configuration remains intact, retained choices are revalidated, and views stay
lazy. Equal updates return the original receiver. Child updates return the same
scoped type over a new root snapshot:

```python
configured = pair.with_choices(
    pair.field(Pair.first.decision_ref(Tiles.factor)).change(3),
    pair.field(Pair.second.decision_ref(Tiles.factor)).change(6),
)
assert configured.first.cycles == 4
assert configured.second.cycles == 3
```

`try_with_choices` returns a `ConfigurationResult`; strict `with_choices` raises
`ConfigurationError` with the same report on refusal. Malformed request keys,
foreign snapshots, duplicates, and nominal type errors raise `RequestError`
before domain evaluators run. A replacement is published only when every
retained and new choice is admitted; failed or inactive retained choices are
not silently removed. New snapshots have independent caches and preserve the
exact frozen facts and prepared definition.

Selections are detached, immutable sparse commitments. Capture includes each
committed owning Decision and nontrivial selector once, including a value equal
to the first candidate. It omits aliases, derived values and evaluator state:

```python
from finn.core.space import selections

saved = selections.capture(tile_configuration)
edited = selections.capture(tile_configuration.with_choices(factor=4))
replayed = selections.restore(tile_base, edited)
assert replayed.accepted and replayed.instance.factor == 4
changed_inputs = tile_model.bind({Tiles.extent: 10})
assert not selections.restore(changed_inputs, edited).accepted
```

Edit configurations with `with_choices`, then capture them. Use a bound decision's
`clear()` to remove a choice. Changing a selector does not silently prune old case
commitments; clear them explicitly in the same batch.

`selections.restore(base, saved)` requires a root with no committed choices and
returns `ConfigurationResult`. The base may already have been queried. Replay is
atomic and revalidates all saved choices under the bound facts; failure leaves the
base intact. A configured receiver raises `RequestError`, including equal replay
and replay of an empty selection. Bind a fresh root to replay different inputs.
In-process selections identify the exact compiled model. Selection entries and
`value(reference)` provide detached inspection; selections have no separate editor.

Portable encoding additionally requires an explicit family/version and a codec
for every saved choice. No codec, fingerprint or global registry is needed to
start, query or capture an ordinary model:

```python
from finn.core.space import JSONValue, SelectionSchema, ValueCodec, codec_for, codecs


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

## Results and inspection

`QueryResult[T]` is `Available[T] | Inapplicable | Rejected | Unresolved`. Scalar answers
reject Boolean coercion. Inspect their variant and findings. Unresolved findings
distinguish missing immutable inputs from commitment blockers.

Computations consume ordinary values through `self` reads or explicit required
arguments. Missing optional Params remain unresolved and distinct from a supplied
`None`; the driver can inspect these outcomes with `query` and `inspect`.
Callbacks may return explicit semantic results, including refusal. Status
inspection, configuration updates and reads from another snapshot inside a
computation fail contextually. Catching a blocked read cannot publish a fallback.
Pure deterministic computations preserve settled results when compatible choices
are added under the same facts; general configuration replacement can change them.

`inspection.members`, `decisions` and `choices` discover frozen metadata without
running evaluators. Typed `value_handle` and `decision_handle` references belong
to exactly one compiled model. `inspection.dependencies` reports known structural and explicit direct
dependencies; it cannot predict every read in an arbitrary method body.
`inspection.explain` evaluates a query and returns observed demanded evidence,
including parameter presence, owning decisions, guards/selectors and constraint
causes. Observed reads include cache hits and do not claim a complete static
dependency graph. `statistics(model)`
reports structural counts without evaluating the model.

## Kernel boundary and validation

`finn.kernels.base.Kernel` adds identity and public capability metadata. It does
not require streams, clocks or an RTL ABI. [Physical kernels](../src/finn/kernels/README.md)
return explicit detached requirements: HLS source interfaces and synthesized RTL
interfaces remain different products. Artifact roots, stores, rendering and
physical algorithms remain in their existing layers.

There is one supported runtime under `finn.core.space`, with its own `py.typed`
marker and a declared native dependency on `greenlet==3.2.4`. No public scheduler
selection is exposed. The generic package depends only on Python, typing support
and that continuation dependency; hardware and QONNX adapters stay in kernels. Proposal scheduling, solvers, cross-snapshot
cache sharing and graph integration are outside this delivery.

The experimental `finn.dataflow` consumers still import the removed
`finn.kernels.space` path and use incompatible retired interfaces. They require
an intentional future port; no alias or copied legacy runtime is provided.
Existing core graph-execution modules are outside this migration.
ONNX parsing, graph staleness checks, nodeattr writes and graph transactions are
adapter responsibilities. The independent kernel gate and historical dataflow
validation records do not establish post-cutover dataflow compatibility.

Run `scripts/check-space.sh` for standalone Space validation without hardware or
graph dependencies. Install its prerequisites from `requirements-space-test.txt`.
See [Maintaining Space](space-internals.md) for the implementation map and
validation contracts, and [migration notes](space-migration.md) for retired APIs.

Run `scripts/check-kernels.sh` for the independent generic Space and kernel
gates, strict typing, formatting, import/wheel boundaries and these executable
examples. It does not run or skip the parked dataflow/graph tests and makes no
full-repository compatibility claim.
