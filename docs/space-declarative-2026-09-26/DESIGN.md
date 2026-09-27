# Declarative design spaces: nodes, references, overrides, one compile step

Date: 2026-09-26. Branch: `spike/space-declarative-4` (iteration 4), on
`spike/space-declarative-3` at `426882411` (iteration 3), on
`spike/space-declarative-2` at `43d4576a6` (iteration 2), itself on
`spike/space-declarative` at `35e59b442` (iteration 1). Status: **design
spike, at the human review gate.** Not for merge as is.

This spike replaces the composition layer of `finn.core.space` with a
declarative graph of design spaces. The engine has no domain notions: no
ports, directions or carried values. Streams, their direction and their
boundaries live in `finn.kernels.streams`.

This document describes the **current** model. §1 lists what iteration 4
changed (views read as values), §2 what iteration 3 changed after the second
human review, and §3 is the **next pass** that review asked to flag. Earlier
iterations' text is kept only where it still applies (git has the rest);
where iteration 4 supersedes iteration 3's text, the older text says so.

## 0. Summary

| Question | Answer |
|---|---|
| What is a declaration? | Calling a family, `Room(area=12)` or bare `Room()`, returns a **declaration**: a template holding its assignments. Nothing is compiled. |
| What does an attribute read give? | On a **declaration**, a symbolic reference typed as the member's value (typing option A); on a **configuration**, the value. Every member follows this rule, **views included** (§1): `kitchen.cost` in a class body is an `int` reference to the view's accepted value, and `point.total` is the accepted `int`. |
| How is a member typed? | By its **annotation**: `area: int = Param()`, `finish: int = Decision(values=(1, 2, 3))`, `output: Stream = Param()`. The annotation is the single source of the value type. A view is typed by its method's return (`@view def total(self) -> int`) or its source (`View(kitchen.area)`). |
| How is a view assessed? | By explicit calls on the configuration that take the view declaration: `point.inspect(House.total) -> ViewAssessment[int]`, `point.query(House.total) -> QueryResult[int]`; a child's view as `point.kitchen.inspect(Room.cost)` (§1.2). |
| What is an edge? | A binding at the call (`Room(area=kitchen.cost)`) or an **assignment** (`hall.area = kitchen.area`), from any reference: a formal, a Decision, a derived value or a view. |
| Who may assign what? | Any body may assign any **Param, Decision or child node** of any descendant, at any depth (`middle.kernel.port.lanes = lanes`). The **outermost** assignment wins; one body assigning a target twice is an error; behaviour (derived values, constraints, views, `Members`/`Users`) is not assignable. |
| What does overriding a Decision do? | A value **pins** it (its key disappears); another Decision **replaces** it under the same key. Both are checked against the declared domain. |
| Who set a value? | Every supplied member records its **provenance**: `kitchen.area = 16 (set by House at house.py:42; declared 12 at room.py:10)`, in `inspection` and in refusals. |
| How is a design space opened? | `design_space(House(budget=100))`, typed `House`. The vocabulary is declaration → model → design space → configuration → design point (§2.5). |
| What does compilation optimize? | Chains of forwarding aliases collapse to their source: scopes, keys and names are unchanged, evaluation does not walk the chain (§2.3). |
| Engine verdict | Iteration 4 changed **no** IR, linker or runtime code: a view node already answered its accepted result, and a formal bound to a view already read it. The change is the access path (`View.__get__`) and the typed surface (`inspect`, `query`, `field`), plus the removal of `BoundView`, `Space.view` and `accepted`. |
| Evidence | Both gates green with XSim executed: Space **418 passed** (384 in iteration 3), kernels **763 passed, 0 skipped** (9 XSim, also run separately), dataflow **16 passed**. The 6 MVAU fingerprints are identical to iteration 3 and, with the cyclic instance renamed, bit-identical to `0d700b1ab`. No MVAU decision key, node key, node kind or top-level ABI port name changed. Scale and collapse probes unchanged (§9). |

**Flagged deviations and additions, iteration 4.** Each is the closest
workable form of the decision; the section named gives the reason.

1. **A view's own name in its class body is its declaration, `View[T]`.** In
   the body that declares it, `total` is the `View` object (the class does not
   exist yet, and class access `House.total` must stay `View[T]` for
   `inspect`). It may be required, exported and wrapped, and at runtime it
   supplies a formal like any reference, but statically a formal refuses it:
   `Room(area=cast(int, total))` (§1.1, R30).
2. **`point.inspect(House.kitchen.cost)` is typed by a value-typed overload.**
   A reference through a node is typed as its value, so the path form is
   `inspect(T) -> ViewAssessment[T]` and cannot reject a non-view statically
   (`inspect(House.kitchen.area)` is refused at runtime, naming the member).
   The precise form is `point.kitchen.inspect(Room.cost)` (§1.2, R31).
3. **`requires=` is typed `Sequence[object]`.** A view reference is typed as
   its value, so no static type separates it from a Param reference; the
   linker refuses anything but constraints, groups, views, references to them,
   `Members` and `Users`, as before (§1.1, R32).
4. **`field()` of a view is a `BoundValue[T]`**: the accessor of what the
   attribute reads, like a derived value's (§1.2).

**Carried from iteration 3** (unchanged): a located formal keeps a descriptor
annotation, `a: LocatedParam[int] = LocatedParam()` (§2.2); a Decision over
nodes is typed by its annotation only (§2.2, R21); it cannot be pinned by a
value, only narrowed (§2.1); a pin or a replacement keeps the declared domain
as a contract (§2.1); additions: attribute projection, `Space.present()`,
`composite(annotations=)`, `inspection.provenance()`/`pinned()`,
`EvidenceNode.via` (§2). Iteration 3's flagged `accepted(view)` bridge is
**gone** (§1).

## 1. Iteration 4 changes: views read as values

The lifecycle split makes every object of a family class one of two things:
on a declaration, attribute reads are symbolic references typed as their
values; on a configuration, they are the values. Params, Decisions and derived
values followed that rule already. Views were the one exception (a callable
`BoundView`, and iteration 3's `accepted(view)` bridge); the exception is gone.

| # | Decision | Implemented as |
|---|---|---|
| 1 | On a configuration a view read is its accepted value, typed `T` | `View.__get__` on a configuration returns `read_value(point, view)`, the read `BoundView.__call__` made; `@view` methods and `View(source)` alike (§1.1) |
| 2 | An unaccepted read raises `ValueUnavailableError` with its result; inside computations it blocks | Unchanged runtime path: the read goes through `_read_value`, so it raises from the driver and halts a method as a blocked read (§1.1) |
| 3 | On a declaration `kitchen.cost` is a reference typed `T`; remove `accepted()` | `View.__get__` on a declaration returns the `MemberRef`, typed `T`; the linker already resolved a view reference as its accepted result wherever a reference is used (§1.1) |
| 4 | Assessment and query are explicit calls | `point.inspect(House.total) -> ViewAssessment[T]`, `point.query(House.total) -> QueryResult[T]`; into a child: `point.kitchen.inspect(Room.cost)` or `point.inspect(House.kitchen.cost)` (§1.2) |
| 5 | Remove `BoundView`, `Space.view()` and what existed only for callable views | §1.3 |
| 6 | Decide what `field()` returns for a view | `BoundValue[T]` (§1.2) |
| 7 | Migrate everything | engine, kernels, `mvau_assembly`, README, benchmark, doc scripts, tests and fixtures (§1.5) |

### 1.1 Semantics

```python
class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def small_enough(self) -> bool | Rejected: ...

    @view(requires=(small_enough,))
    def cost(self) -> int:
        return self.area * self.finish

class Hall(Space):
    want_garage: bool = Decision(values=(False, True))
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=30, when=want_garage)
    hall = Room()
    hall.area = kitchen.cost                    # a reference to kitchen's accepted cost: int
    annex = Room(area=dining.cost)              # at the call too
    porch = Room(area=Present(garage.cost))     # in Present
    doubled = View(kitchen.cost * 2)            # in an expression
    checked = View(dining.area, requires=(kitchen.cost, dining.small_enough))

    @view(requires=(kitchen.cost, dining.cost))  # obligations: their acceptance only
    def total(self) -> int:
        return self.kitchen.cost + self.dining.cost   # accepted values; blocks until both are

point = design_space(Hall()).with_choices({Hall.kitchen.finish: 2, Hall.dining.finish: 1})
point.total                   # 40, an int
point.inspect(Hall.total)     # ViewAssessment[int]
point.query(Hall.total)       # Available(40)
```

**On a configuration**, a view read is its **accepted** value. A view that
is not accepted raises `ValueUnavailableError` whose `result` is the accepted
result (`Unresolved` while an obligation or the output waits, `Rejected` with
the refusal, `Inapplicable` when the view's node is absent), exactly as any
other unavailable read, and `result == point.query(Family.view)`. Inside a
`@derived`, `@constraint` or `@view` method the read is recorded as a
dependency and blocks the method like any other read; the per-view obligation
results of an assessment are unchanged (`{"kitchen.cost": Available(True),
"dining.cost": Unresolved(...)}`: an obliged view contributes only its
acceptance).

**On a declaration**, `kitchen.cost` is a reference to the view's accepted
value, typed `T` (the `MemberRef` it always was at runtime). It is usable
wherever a reference is: it supplies a formal at the call or by assignment,
it is a `Present` source, an operand of integer arithmetic (`kitchen.cost *
2`), and a `requires=` obligation (contributing only its acceptance, with its
own result in the assessment). Every value-like use (`bool`, comparison,
calling, `str`, ...) raises `ReferenceUseError` naming the reference, as for
any reference; `kitchen.cost.inspect` is an `AttributeError` (a reference
projects only attributes its value type annotates).

**Class access**, `House.total`, is the view declaration, typed `View[T]`:
the key `inspect`, `query`, `field`, `inspection.explain` and
`inspection.value_handle` take. In **its own class body** a view's name is
that declaration too (flag 1, R30).

**Nothing in the runtime changed.** A view node's evaluation already answered
its accepted result, `BoundView.__call__` was already `read_value`, and a
formal, `Present` source or obligation that named a view reference already
read its accepted result (which is why `accepted()` was an identity at
runtime). The change is the access path and the static types.

| Iteration 3 | Iteration 4 |
|---|---|
| `point.total()` | `point.total` |
| `self.kitchen.cost()` in a method | `self.kitchen.cost` |
| `point.total.inspect()` / `point.total.query()` | `point.inspect(House.total)` / `point.query(House.total)` |
| `point.kitchen.cost.inspect()` | `point.kitchen.inspect(Room.cost)` or `point.inspect(House.kitchen.cost)` |
| `point.view(House.total)()` | `point.total` (`point.field(House.total).get()`) |
| `point.field(House.total)` → `BoundView[int]` | `BoundValue[int]` |
| `hall.area = accepted(kitchen.cost)` | `hall.area = kitchen.cost` |
| MVAU: `point.build_requirements()`, `self.buffer.interfaces()` | `point.build_requirements`, `self.buffer.interfaces` |
| `mvau_assembly`: `point.structure.query()` | `point.query(MVAU.structure)` |

### 1.2 Assessment, query and `field`

```python
@overload
def inspect(self, view: View[T]) -> ViewAssessment[T]: ...
@overload
def inspect(self, view: Constraint | ConstraintGroup) -> ConstraintAssessment: ...
@overload
def inspect(self, view: T) -> ViewAssessment[T]: ...      # a reference through a node

@overload
def query(self, value: ValueRef[T] | View[T]) -> QueryResult[T]: ...
@overload
def query(self, value: T) -> QueryResult[T]: ...
```

- `point.inspect(House.total)` is the view's `ViewAssessment` (raw output,
  readiness, obligation results, accepted result); `point.query(House.total)`
  is its accepted result, the answer the read would raise with.
- **A view of a child** has two forms, both implemented and typed:
  `point.kitchen.inspect(Room.cost)` on the child's configuration, typed by the
  view declaration (**the precise form**: a non-view is a type error), and
  `point.inspect(House.kitchen.cost)` through a path, typed by the value-typed
  overload (a non-view such as `House.kitchen.area` type-checks and is refused
  at runtime: *"kitchen.area is not a view or a constraint: inspect()
  assesses those; read or query() any other member"*). They return equal
  assessments. `query` takes both forms the same way.
- **`field()` of a view returns a `BoundValue[T]`**: `get()` is the read,
  `query()` is `point.query(view)`. Reason: `field(x)` is the bound accessor of
  whatever `point.x` reads, and a view now reads as a value exactly like a
  derived value, whose field is a `BoundValue`. Rejected: refusing `field` for
  a view (it would make `field` partial over value-typed members for no gain),
  and a view-specific accessor carrying `inspect()` (that is `BoundView` again,
  a second route to the assessment). The assessment has one route,
  `point.inspect`.

### 1.3 What was removed

`BoundView` (and its `get`, `inspect`, `query`, `__call__`), `Space.view()`,
`occurrence.bind_view`, `accepted()` and its export, and every `BoundView`
arm of a signature: `View.__init__`, `Present.__new__`,
`LocatedParam.__set__`, the `Obligation` alias, `Space.query`, `Space.field`,
`bind_field`. `bind_field` no longer special-cases views. No alias or shim
remains; `finn.core.space` exports neither name.

### 1.4 Typing results

Under `mypy --strict` (`tests/core/space/typing/positive.py`, section
"iteration 4", `negative.py.txt`, `extensions.py`, and the kernels'
`typing/` fixtures):

| Claim | Result |
|---|---|
| `point.total`, `house.total` on a configuration | `int` |
| `self.kitchen.cost` in a method | `int` |
| `kitchen.cost` in a class body | `int` |
| `Present(kitchen.cost, dining.cost)` in a class body | `int` |
| `hall.area = kitchen.cost`, `Room(area=dining.cost)` | accepted |
| `@view(requires=(kitchen.cost, dining.cost))`, `View(kitchen.cost, requires=(dining.cost, total))` | accepted |
| `House.total` | `View[int]` |
| `point.inspect(House.total)` | `ViewAssessment[int]` |
| `point.query(House.total)` | `QueryResult[int]` |
| `point.kitchen.inspect(Room.cost)`, `point.inspect(House.kitchen.cost)` | `ViewAssessment[int]` |
| `point.field(House.total)` | `BoundValue[int]` |
| `fifo.inspect(Fifo.supported)` (a constraint) | `ConstraintAssessment` |
| `point.physical()`, `point.physical(3)` (calling a view), `child.physical()` in a class body | **errors** |
| `point.physical.inspect()`, `point.view(...)` (the removed surface) | errors |
| `wrong: str = point.physical`, `wrong: int = point.inspect(Child.physical)`, `QueryResult[str] = point.query(Child.physical)` | errors |
| `Child(label=child.physical)` (an `int` view into a `str` formal) | error |
| `Child(width=own)` with `own` a view of the same body | error (R30; `cast` it) |
| `point.inspect(Child.width)` (a Param) | not an error (R31; refused at runtime) |

**Gained:** a view supplies a formal, feeds `Present` and is read in a method
with no bridge or call; `kitchen.cost + 1` is typed; the assessment and query
of a view are typed through the same two calls as every other member.
**Lost:** R30–R32 below. `negative.py.txt` now holds 40 expected-error lines
(34 in iteration 3), matched line for line.

### 1.5 Migration

- **Engine** (`finn.core.space`): `declarations.py` (`View.__get__`,
  `Obligation`, removal of `accepted`), `_configuration.py` (`inspect`,
  `query`, `field`; removal of `BoundView`, `view`), `occurrence.py`
  (`bind_field`; `inspect` names a non-assessable member before evaluating
  it), `__init__.py`.
- **Kernels** (`finn.kernels`, 12 files): view reads on 26 lines of 10
  modules lost their call (`streams.py`, `streaming.py`, `dotp.py`,
  `delivery.py`, `fifo.py`, `eltwise.py`, `input_generator.py`,
  `int_to_fp32.py`, `thresholding.py`, `mvau.py`); the two port helpers (`physical/ports.py`,
  `physical/axi_stream.py`) pass `element.encoding` instead of
  `accepted(element.encoding)`; `mvau_assembly` queries `MVAU.structure`.
  `finn.kernels.configure.commit` needed no change (it takes keys, never
  views).
- **Tests**: every engine and kernel test file that read, inspected or
  queried a view (20 engine test files and fixtures, 19 kernel test files,
  including script strings run in subprocesses by `test_boundaries.py` and
  `test_installed_package.py`); no test was deleted. A parametrized kernel
  test whose factory yields several families inspects
  `type(point).build_requirements`. Tests of the removed surface now cover
  the replacement: `point.view(...)` in `test_runtime.py` and
  `test_public_snapshots.py` became `field(...).get()`, reads and
  `point.query(...)`; the kernels' FIFO test compares the read, `field`,
  `query` and the assessment's accepted result, where it compared
  `point.inspect(F.x)` with `point.x.inspect()`; `accepted(...)` edges in `test_design_graph.py` and
  `test_collapse.py` are plain references; the misuse test of a called view
  reference is kept.
- **New tests**: `tests/core/space/test_view_values.py` (9) and two in
  `test_reference_misuse.py` (every value-like use of a view reference
  refused, 24 cases; no `inspect`/`query`/`get` on a reference).
- **Docs and scripts**: `src/finn/kernels/README.md` (every example runs),
  `scripts/benchmark-space.py`, `collapse_probe.py`, `scale_probe.py`.

### 1.6 Evidence

`tests/core/space/test_view_values.py`: an unaccepted read raises with its
result (unresolved, rejected and inapplicable, each equal to `query`); a read
inside a `@derived` halts it at the first unavailable view while the
assessment's per-view obligation results stay acceptance-only;
`hall.area = kitchen.cost`, `Room(area=dining.cost)`, `Present(garage.cost)`
and `kitchen.cost * 2`; view references in `requires=` of both view forms,
including a refused one; a non-obligation refused at link; `inspect`/`query`
on the child and through a path, a non-view refused by name; `field` of a view
is a `BoundValue`; a view's own name in its body supplies a formal through a
cast; the removed surface is absent. `test_reference_misuse.py`: all 24
value-like misuses of a view reference raise `ReferenceUseError`, and a
reference has no `inspect`/`query`/`get`. Gates, fingerprints, keys and probes:
§9.

## 2. Iteration 3 changes

| # | Decision | Implemented as |
|---|---|---|
| 1 | Parents override the data of any descendant | §2.1. `NodeDecl.overrides` keyed by member path; layered at link, outermost wins; same-body double assignment, behaviour and incompatible families refused; pins, replacements, child and choice replacement; provenance in `inspection`, refusals, stale selections |
| 2 | Annotate formals with their value type | §2.2. `Param`/`Decision` take options only and are typed as their value; every family (engine tests, kernels, MVAU, streams, `Constants`, scripts) migrated; key-taking APIs accept keys typed as values |
| 3 | Collapse value-derivation chains, never Spaces | §2.3. A post-link pass rewrites evaluation edges; method reads take the same shortcut at runtime; `explain` lists the aliases read through as `via` |
| 4 | Fix the presence oddity | §2.4. `query(K.output)` answers the referenced configuration; `point.present(node) -> bool` |
| 5 | Terminology | §2.5. `SpaceModel` → `Model`, `configure` → `design_space`, `compile_space` → `compile_model` |
| 6 | Flag the next pass | §3 |

### 2.1 Overrides and provenance

```python
class Room(Space):
    area: int = Param(default=12)
    finish: int = Decision(values=(1, 2, 3))

class LargeRoom(Room):                       # a subclass may replace a Room
    windows: int = Decision(values=(2, 4))

class Wing(Space):
    kitchen = Room(area=14)                  # Wing's body sets 14
    study = Room()

class House(Space):
    wing = Wing()
    wing.kitchen.area = 16                   # overrides Wing's 14
    wing.study.finish = 2                    # pins a grandchild's Decision

class Estate(Space):
    home = House()
    home.wing.kitchen.area = 18              # the outermost assignment wins
    home.wing.kitchen.finish = Decision(values=(1, 2))   # narrows: same key
    home.wing.study = LargeRoom(area=9)      # replaces a child node
```

**Where an assignment lives.** `a.b.c = x` in a body is stored on the first
node of the path (`a`), keyed by the member path below it (`"b.c"`). A call
keyword is exactly an assignment by the calling body. Keys are attribute
paths, not record identities, so an assignment reaches whatever node an
intermediate body placed at that path (a replacement included). A path
through a Decision candidate is allowed (`mvau.cyclic.rom_style`, keyed
`implementation.cyclic.rom_style`); a path through a reference input is not
(assign the referenced node where it is placed).

**Layering.** When a node is placed, the linker collects every setting of each
of its members: the family's declaration (a `Param` default, the `Decision`, the
child node), the node's own record (the body that declared it), and the record
of every enclosing node that assigns through a path. Settings are ordered by
how deep the writing body is; the **outermost** wins. Depth, not position,
decides: a replacement node's own settings were written by the outer body that
replaced it.

**Rules.**

- *Same-body double assignment* is a `DefinitionError`: at the assignment when
  it hits the same record and path ("`kitchen.area is already assigned at
  x.py:3 in this body`"), and at link when two settings of one member come from
  one body by different routes (a replacement's keyword and a path assignment).
- *Behaviour is not assignable*: a derived value, constraint, group, view,
  `Members`, `Users`, `Const` or class-body alias is refused where it is
  assigned: "behaviour belongs to the family; subclass it to change it".
  A candidate handle is refused too ("override the Decision instead").
- *A Decision member* overridden by a value is **pinned**: a `const` node (an
  `alias` for a reference) under the same node key, with no decision key. By
  another Decision it is **replaced** under the same key (the attribute path).
  Either way the declared domain stays as a contract: a pinned value outside it
  is refused where it is read, a replacement's candidate is admitted only if the
  declared domain admits it too, and its advisory enumeration lists only what
  the contract admits. The declared guard (`when=`) is kept.
- *A formal* overridden where an inner body opened a coordinate with an inline
  Decision closes that coordinate: its key disappears the same way.
- *A child node* may be replaced by a **fresh** node of its family **or a
  subclass**. A subclass keeps every member an enclosing body can name (Python
  inheritance, plus collection's "override changes declaration kind / value
  semantics" checks), so every reference into the slot still resolves with the
  same types; mypy enforces the same rule (`wing.kitchen = Garden()` is an
  error). A structurally similar family is refused: nothing would check the
  names enclosing bodies use. The replacement takes the slot's guard and may
  not declare its own `when=`; it keeps the slot's node name, and every
  reference to the declared node reaches it.
- *A Decision over nodes* may be replaced by a narrower one: a subset of the
  cases, each a node of the declared candidate's family or a subclass (or
  `None` where the declared case is `None`). Its key and candidate names stay.

**Provenance.** Each supplied member's layers are kept in the model
(`LinkedModel.provenance` by node, `scope_provenance` by replaced child,
`pinned` by removed key) as a `Provenance(key, layers)`; each `Layer` has the
body, the source line and the value as text:

```
home.wing.kitchen.area = 18 (set by Estate at estate.py:12; overrides 16 set by
House at house.py:9; overrides 14 set by Wing at wing.py:4; declared 12 at room.py:2)
```

It is exposed by `inspection.provenance(subject, reference)`,
`inspection.pinned(subject)`, and `NodeInfo.provenance` (so `members`,
`dependencies` and `explain` carry it). Diagnostics:

- a pinned value outside the declared domain: `domain-membership`, *"room.finish
  = 7 (set by Wrong at x.py:5; declared Decision(values=(1, 2, 3)) at
  room.py:3): 7 is outside the declared domain"*; a replacement's candidate the
  contract refuses reads the same way;
- a computation's own refusal appends the provenance of every **overridden**
  value it read (two or more layers): *"area 30 exceeds 20;
  home.wing.kitchen.area = 30 (set by Mansion at ...; overrides 16 set by House
  at ...; declared 12 at ...)"*, also as `details["provenance"]`;
- editing a pinned key: *"wing.study.finish is not an owned Decision: an
  enclosing body pinned it (wing.study.finish = 2 (set by House at ...))"*;
- a persisted selection holding a pinned key is refused on decode as **stale**,
  with the provenance; `finn.kernels.configure.commit` refuses it the same way;
  an in-memory `Selection` belongs to its model and is refused elsewhere.

Removed: the "a formal has one supplier / already supplied" rule across bodies,
the "assign formals only" rule, and "unknown formals" (now "unknown members",
since a call may pin a Decision: `Room(finish=2)`).

### 2.2 Annotated members and what typing gains and loses

```python
class Room(Space):
    area: int = Param()                          # required
    label: str = Param(required=False)           # optional, unsupplied if nobody binds it
    finish: int = Decision(values=(1, 2, 3))
    spec: StreamSpec = Param(semantics=STREAM_SPEC)   # custom value semantics

class Kernel(Space):
    output: Wire = Param()                       # a reference input
    buffer = Buffer(word_bits=output.spec.payload_bits)   # typed in the class body
    heating: Boiler | HeatPump | None = Decision(values={...})
```

- `Param(*, default=..., required=..., semantics=...)` and
  `Decision(*, values=... | domain=..., semantics=, when=, name=)` take options
  only. `UNSUPPLIED` is gone from the API: `required=False` says it.
- The annotation is read when the class is created (or, for a forward
  reference, when it is collected), evaluated in the module, the class and
  the enclosing function's locals. A plain class gives default semantics; a
  union, protocol or special form needs `semantics=`, which must agree with it.
  A member without an annotation is a `DefinitionError` ("annotate the param
  with its value type, as in `area: int = Param()`"). A member built as data
  (outside a class body) takes its annotation from
  `composite(..., annotations={"choice": int})`, or its value type from
  `semantics=`.
- A `Param` annotated with a family is a reference input (`FamilyFormal` is
  gone). In the class body `output.spec` is a reference through it, and an
  attribute of a referenced value projects: `output.spec.payload_bits` is a
  derived node that reads the property, typed by its annotation.
- An inline Decision (`Fifo(depth=Decision(values=(4, 8)))`) takes its value
  type from the formal it supplies.
- `@derived` members are typed as their value too, so a derived value supplies
  a formal (`Stream(spec=activation_spec)`). `Present(...)` is typed as its
  value. (Iteration 3 added `accepted(view)` to supply a formal with a view's
  accepted value; iteration 4 removed it: a view reference is typed as its
  value, §1.)

Key-taking APIs accept keys typed as their values: `with_choices({Room.finish:
2})`, `field(Room.finish) -> BoundDecision[int]`, `query(K.output) ->
QueryResult[Wire]`, `inspection.decision_handle(point, House.kitchen.finish) ->
DecisionHandle[int]`, `decision_info`, `value_handle`, `explain`, `codec_for`
(and `commit`, which takes string keys, is unchanged).

| Claim (mypy `--strict`) | Result |
|---|---|
| `Room.area` | `int` |
| `Room()` | accepted; every member is optional at the call |
| `Room(area="x")`, `hall.area = "x"`, `Room(finish="two")` | errors |
| `Room(area=kitchen.area)`, `hall.area = kitchen.area` | accepted |
| `output.spec.payload_bits` in the class body | `int` (**gained**: iteration 2 left `output.spec` untyped) |
| `wing.kitchen = Garden()` (not a `Room`) | error |
| `heating.kw` with `heating: Boiler \| HeatPump` | `int` (**gained**: no `Decision[...]` subscript) |
| `point.query(K.output)` | `QueryResult[Wire]`, and it is true now |
| `point.present(3)` | error |
| `inspection.decision_handle(house, House.kitchen.finish)` | `DecisionHandle[int]` (**gained**: was `Any`) |

What is **lost**, each with the reason:

| Loss | Why |
|---|---|
| A `Param` key cannot be told from a `Decision` key: `point.field(Child.width).change(2)` type-checks and fails at runtime (`field` of a non-decision returns a `BoundValue`) | both are typed `int` |
| A declaration's own attributes (`Family.x.semantics`, `.domain`, `__set_name__`) need a `cast` | class-level members are typed as values |
| A Decision over nodes: a wrong annotation is not a mypy error | the call is `Any` (R21); collection checks each candidate against the annotation |
| ~~A view does not supply a formal without `accepted(...)`~~ (resolved in iteration 4, §1) | `BoundView[T]` was not `T` (R27) |
| Class-body arithmetic is `int`, not `Expr`: `number / 2`, `number ** 2` and `int + float` are not mypy errors | the operands are `int`; refused at runtime (`/` and `**` raise a `TypeError`, a non-`int` operand is a `DefinitionError` when linked) |
| A missing required formal is not a mypy error (unchanged) | an assignment may still supply it |
| `LocatedParam[int]` is the one descriptor annotation | its suppliers and value differ in type |
| `Const(...)` members stay typed `Const[T]` | not asked; supplying a formal with a `Const` needs a cast |

Evidence: `tests/core/space/typing/positive.py` (a new section: reads through
a reference input, bare calls, pins, narrowing and replacement typed in a
class body, keys), `negative.py.txt` (34 expected-error lines, matched line for
line, including wrong-type pins, a wrong replacement family, a typo through a
reference input, `present(3)` and a mistyped query), `extensions.py`,
`expressions_*`, and the kernels' `typing/` fixtures, all under `--strict`.

### 2.3 Collapsed forwarding chains

A formal bound to a reference, a formal forwarded through composites, a named
shared decision's uses and a class-body alias are `alias` nodes. After linking
and type checks, `collapse()`:

1. points each alias's `output` at the end of its chain, following the next
   alias only when that alias's guard holds whenever this one's does (its
   guard is this guard or an outer one);
2. points every reader's edge (arguments, domain and contract arguments, view
   and guard outputs, alternatives) straight at that source when the alias's
   guard holds whenever the **reader's** does: a reader is evaluated only when
   its own guard holds, so the alias could not have answered "inapplicable".

Method reads (`self.width`) take the same shortcut at runtime, against the
reading node's guard. An alias that pins a Decision (it has a domain) is not
forwarding and is never bypassed. A reference input forwarded through
composites already resolves, at link, to the node it finally references.

**Unchanged:** every scope, node, node key, decision key, instance name and the
netlist's hierarchy. Every node still answers when queried by name. The
answers of **every node** are identical with and without collapse, for the
house toy, the reference toys, a forwarding chain with a guarded middle, data
pipelines and MVAU (`tests/core/space/test_collapse.py`,
`tests/kernels/test_mvau_collapse.py`), findings and owners included, since an
alias passes its source's answer through unchanged.

**`explain`.** A bypassed alias is not evaluated, so it is not an evidence
node; each evidence node lists the aliases it read through in `via` (by their
authored key), and its dependency is the alias's source. Rejected: keeping
aliases as synthetic evidence nodes (they were not evaluated, and their
"result" would be a copy); and hiding them (the authored name a method read
would vanish).

**`Users`** is unchanged: the nodes that directly reference the node, in the
body that wrote the reference. Whether a forwarding composite or its inner
nodes count as users is deferred to the query-tools pass (§3).

**Measured** (`collapse_probe.py`, [`evidence/collapse-probe.txt`](evidence/collapse-probe.txt)):

| Case (read) | nodes | aliases | edges into aliases before → after | nodes evaluated before → after | aliases evaluated before → after |
|---|---|---|---|---|---|
| MVAU external (`structure`) | 217 | 32 | 12 → 0 | 168 → 142 | 26 → 0 |
| MVAU cyclic + FIFO (`structure`) | 217 | 32 | 12 → 0 | 197 → 165 | 32 → 0 |
| pipeline N=800 (last width) | 3200 | 799 | 0 → 0 | 3200 → 2401 | 799 → 0 |
| composites N=200 (last width) | 1400 | 399 | 199 → 0 | 1400 → 1001 | 399 → 0 |

Node counts are unchanged by construction; evaluated nodes drop by 15–29 %
(every alias). The scale probe's time moves little (§9): an alias frame is
cheap next to a callback.

### 2.4 Presence

`point.query(K.output)` on a reference input now answers the **referenced
node's configuration**, as its static type says: `Available(<Stream>)` when
present, `Inapplicable` when the referenced node is absent, `Unresolved` while
its presence is undecided or the optional input is unsupplied. The same holds
for any node reference: a child (`query(House.garage)`) or a candidate handle.

`point.present(node) -> bool` is the typed presence accessor. It reads like a
value: `True`/`False`, `False` for an unsupplied optional input, and an
undecided presence raises `ValueUnavailableError` (inside a method it halts the
method as unresolved, like any read). Inside a method it does not halt on an
absent node: `self.present(K.output)` is `False`.

### 2.5 Terminology

| Term | Meaning | In the API |
|---|---|---|
| declaration | `Room(area=12)`: a template, not compiled | a family call; `inspection.declaration()`, `NodeDeclaration` |
| model | compiled structure, no inputs | `Model` (was `SpaceModel`), `inspection.model()`, internal `compile_model()` |
| design space | a model with inputs bound and every choice open | `design_space(House(budget=100))` (was `configure`) |
| configuration | a design space after some choices: a partial point | the same `Space`-typed object; `with_choices`, `ConfigurationResult` |
| design point | a configuration complete for the question asked | no type: completeness is relative to the question |

- **`SpaceModel` → `Model`**: the package already qualifies it
  (`finn.core.space.Model`); "Space" said nothing.
- **`configure` → `design_space`**: the call makes no choice. It binds the
  inputs and opens every choice, which is exactly the vocabulary's design
  space; `configure` read as "make choices". The result is typed as the family
  and is the empty configuration, so no second type appears.
- **`compile_space` → `compile_model`**: it compiles a model.
- **Kept:** `Space` (a family of design spaces), `with_choices` (it makes a
  configuration), `ConfigurationResult`/`ConfigurationError` (they are about
  configurations), and `finn.kernels.configure.commit` (it configures: commits
  choices by key). No `DesignPoint` type: a point complete for one query is
  partial for another.

## 3. Next pass: query and search tools

**Goal.** Now that compilation is decoupled from authoring, determine which
query and search tools are needed and useful over Spaces: for authors in class
bodies, for compiler passes over declarations, and for search over
configurations. This pass only flags it; nothing below is implemented.

**The idea under consideration.**

- **One engine primitive, a presence-aware gather**: `Collect(refs)` returns
  located values, each with its own obligation (its acceptance counts
  separately), omitting absent sources.
- **`Present` becomes `Collect(..., exactly_one=True)`**: unresolved while any
  source is, refused on two, unsupplied on none, as today.
- **Structural queries become plain Python over declarations at compile time**:
  `children(space, exporting=K)`, `users(node)`, paths. They serve authors and
  compiler passes alike, and the linker lowers their result to a `Collect`.
- **`Members`, `Users` and `Present` would become compositions of these**: a
  structural selection, then a gather.
- **Open: how a class body names its own structure before the class exists**,
  for example a deferred selector (`Collect(children(Self, exporting=COST))`
  resolved at collection).

**Deferred questions recorded for it.**

- `Users` through forwarding composites: closure (a composite is the user and
  re-exports its children's ports, today) or flattening (the inner nodes are
  users, named from the referenced node's parent)?
- Should `Ports`-style convenience views (a kernel's keyed `PORTS` record, one
  export per user) stay domain-level, or does a keyed gather belong in the
  engine (it would also fix R16's attribution)?

## 4. The model

A **family** is a `Space` subclass; its class body declares members and nodes.
The complete toy is [`tests/core/space/test_house.py`](../../tests/core/space/test_house.py):

```python
class House(Space):
    budget: int = Param()
    want_garage: bool = Decision(values=(False, True))
    hall = Room()                                      # its area is assigned below
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=20, when=want_garage)           # guarded node
    heat_pump = HeatPump(kw=8)                         # a handle naming a candidate
    heating: Boiler | HeatPump = Decision(values={"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    thermostat = Thermostat(kw=heating.kw)             # the selected candidate's kw
    hall.area = kitchen.area                           # an edge after its nodes
    matched = Match(a=kitchen.finish, b=dining.finish)
    costs = Members(COST)

class Estate(Space):
    """An enclosing body customizes the house it contains: data, never behaviour."""
    home = House(budget=300)
    home.kitchen.area = 14                                          # overrides 12
    home.kitchen.finish = 2                                         # pins: key gone
    home.heating = Decision(values={"heat_pump": HeatPump(kw=6)})   # narrows the choice
    home.garage = Room(area=24, finish=1)                           # replaces a child

point = design_space(Estate()).with_choices({Estate.home.want_garage: True, ...})
```

### 4.1 Concepts and their lowering

| Concept | Written | Lowers to (evaluation graph) |
|---|---|---|
| Declaration | `Room(area=12)`, placed by a class attribute | one scope; members become nodes keyed `kitchen.area`, ... |
| Formal | `area: int = Param()` | `param` at the root (a runtime input); in a child: `const` (literal), `alias` (reference), `decision` (fresh Decision) |
| Unsupplied formal | never assigned | optional: `const` of the default, or `present` with no alternatives; required: a definition error when preparing |
| Assignment | `hall.area = kitchen.area`, at any depth | a layer of the member; the outermost wins |
| Pinned Decision | `room.finish = 2` from an enclosing body | `const` (or `alias`) with the declared domain as a check; no key |
| Replaced Decision | `room.finish = Decision(values=(1, 2))` | `decision` under the same key, with a contract domain |
| Replaced child | `wing.kitchen = LargeRoom()` | the new node's scope at the slot's name and guard |
| Reference input | `output: Stream = Param()` | a presence node plus `Scope.references` (or a placement for a fresh node) |
| Projection | `output.spec.payload_bits` | a `derived` node `...$project.N` reading the attribute |
| Users | `claims = Users(SPEND)` | a `members` node over the users' exports |
| Decision over nodes | `heating: A \| B = Decision(values={...})` | a selector keyed `heating`, one guarded scope per candidate |
| Member through a choice | `heating.kw` | a `select` node `heating.$member.kw` |
| Whichever is present | `Present(a.out, b.out)` | a `present` node |
| View | `@view def cost(self) -> int`, `View(source, requires=...)` | a `view` node whose output is its source or a `derived` `...$output`, with its obligations; it answers its **accepted** result |
| View reference | `kitchen.cost` in a body (typed `int`) | resolves to the `view` node: a formal's `alias`, a `present` source or an expression operand reads its accepted result; an obligation contributes its acceptance (§1) |
| Forwarding alias | any `alias` without a domain | kept as a node; readers bypass it (§2.3) |

The compile step: `design_space(node)` checks that `node` is an unplaced
declaration and that the root's required formals are supplied. If every root
setting is a plain value for its own formals, it reuses the family's model
(compiled once, cached on the family) and binds the values as runtime inputs;
otherwise (a node, a reference, a fresh Decision, a pin, or any path
assignment) the model is compiled for that declaration and cached on it.
Preparing a model freezes every declaration it instantiates, and
`design_space` freezes its root: an assignment after that is refused, because
the cached model would silently miss it (unchanged from iteration 2).

### 4.2 Reference inputs and `Users`

A `Param` annotated with a family is a **reference input**. Its supplier is a
declaration; whether it is placed there or referenced is decided at link,
because class-body placement happens in `__set_name__`, after the body:

- **fresh** (no class attribute, candidate, replacement or composite member
  placed it): placed at the input, keyed below the node that placed it
  (`k.output.spec`). A fresh node supplied to two inputs is refused.
- **placed**: a reference, resolved after every scope is allocated, in the body
  that wrote it: a sibling, a candidate, or a forwarded reference input of that
  body. A forwarded input resolves to the node it finally references.

Either way the input gets a presence node keyed like the input (a constant
guarded by the reached node's guard; `present` with no alternatives when an
optional input is unsupplied). In a method `self.output` is the referenced
configuration, inapplicable when that node is absent. A reference written by an
enclosing body through a path (`department.team.account = central`) resolves
in that body, so the user is named `department.team`.

`Users(key)` is declared in the referenced node: one `Located(node=<user's
name beside this node>, member=<the user's input>, value=<the user's export of
key>)` per (user, input), in declaration order; users that do not export `key`
are omitted at link, absent ones at run time; as an obligation each user counts
once. A composite forwarding its input is itself the user (closure); see §3.

### 4.3 Streams (`finn.kernels.streams`)

A `Stream` is an ordinary Space that kernels reference: `spec: StreamSpec =
Param(semantics=STREAM_SPEC)`, `port: str = Param(required=False)` (the AXIS
name when it is a boundary), `ends = Users(PORTS)`. A kernel has one reference
input per stream it sits on and exports `PORTS`, a `Ports` record keyed by
those input names, each a `Port(Flow.IN | Flow.OUT, contract)`. The stream
refuses a second producer or consumer and an unused stream, and turns a
missing side into the composite's boundary named by `port`. A
`BufferedStream` owns `transport: _Direct | StreamFifo = Decision(...)`.
**The anchoring rule:** a stream's `spec` must not depend on its users (kernels
read it to build their contracts); a violation is a dependency cycle reported
with its path. MVAU now reads:

```python
class MVAU(Space):
    repetitions: int = Param()
    matrix_height: int = Param()
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    pe: int = Decision(domain=divisors_of(matrix_height))
    ...
    activations = Stream(spec=activation_spec, port="in0_V")
    weight_stream = BufferedStream(spec=weight_spec, port="in1_V")
    compute = DotpAxiKernel(..., activation_stream=replayed, weights_stream=weight_stream)
    cyclic = CyclicDelivery(dtype=weights_dtype, form=weight_period, values=weights,
                            output_stream=weight_stream)
    implementation: CyclicDelivery | None = Decision(values={"external": None, "cyclic": cyclic})
    delivery = selected(implementation)
```

`Constants` (`tests/kernels/test_declared_streams.py`) and every kernel family
use the same form.

### 4.4 Rules carried from iterations 1–2 (unchanged)

- **Shared decisions are named.** An inline Decision supplying one member is
  keyed by that member's path; supplying two or more it must be named: a class
  attribute, or `Decision(..., name="lanes")`, owned by the lowest scope
  containing its uses and keyed `<owner>.<name>`.
- **Reference visibility is lexical:** a reference resolves in the body that
  wrote it; a family reaches an ancestor's node only through an input.
- **Place once:** a declaration is placed exactly once: a class attribute, a
  Decision candidate, a replacement, or (when nothing else places it) the one
  reference input it is supplied to. A class attribute naming a candidate of a
  Decision in the same body is a handle.
- **Boundary names** come from a stream's `port` input, not its node name (a
  node name is identity and prefixes persisted keys).
- **`Present(a, b)`** is the only way to write alternative suppliers.
- **Literals** are recognized and frozen once, where they are written.
- **Where errors appear:** unknown members, positional arguments, bad literals,
  wrong families, behaviour targets and same-record double assignment at the
  call or assignment; missing formals, double assignment by one body through
  two routes, unresolvable references, unnamed shared decisions and
  unresolvable annotations when preparing.
- **`configure` rejected names** (iteration 1) still apply to its successor:
  `compile` shadows a builtin, `instantiate` collides with calling a family.

## 5. Public API

- **Families and declarations:** `Space`, a family call `F(**members, when=...)`
  (any member may be left out; a keyword may pin a Decision), assignment
  `node.member = value` at any depth, `design_space(node) -> F`, `Model`,
  `composite(name, members, base=, annotations=, exports=)`.
- **Members:** `Param(default=, required=, semantics=)`, `LocatedParam()`,
  `Const`, `Decision(values= | domain=, semantics=, when=, name=)`,
  `selected(decision)`, `Derived`/`@derived`, `Constraint`/`@constraint`,
  `ConstraintGroup`, `View`/`@view(requires=...)`, `ViewKey` + `exports`.
  A view reads as its accepted value; `kitchen.cost` in a body is a reference
  to it, typed as that value.
- **Graph primitives:** `Present(*refs)`, `Members(key)`, `Users(key)`, `Located`.
- **Configurations:** reads (a view reads as its accepted value, raising
  `ValueUnavailableError` with its result otherwise), `query` (a view answers
  its accepted result; nodes answer their configuration), `present`,
  `inspect(View[T]) -> ViewAssessment[T]` and
  `inspect(Constraint | ConstraintGroup) -> ConstraintAssessment` (a child's
  view as `point.child.inspect(Child.view)` or through a path), `field` (a
  view binds as a `BoundValue`), `root`, `with_choices` / `try_with_choices`.
  Removed in iteration 4: `view`, `BoundView`, `accepted`.
- **Inspection:** as before, plus `provenance`, `pinned`, `Provenance`,
  `Layer`, `NodeInfo.provenance`, `EvidenceNode.via`.
- **Errors:** as before; a pinned key is refused with its provenance, a stale
  persisted key on decode.

## 6. Resistance log: where the existing engine and mypy pushed back

| # | Resistance | Change | Reading |
|---|---|---|---|
| R1 | Sibling declarations have no owner until `__set_name__` runs after the body | Classification (fresh decision, member, fresh node vs reference) at link | Unchanged |
| R2 | `dataclass_transform` fields must be annotated | Members annotated with their **value type** | Now the one source of the value type |
| R3 | A field specifier without a default makes a field required | `Param`/`Decision` are no field specifiers and are typed as their value (`__new__` returns `T`) | Their call is a default: every member optional at the call |
| R3b | A declared `__setattr__` accepts any attribute | Hidden under `if not TYPE_CHECKING` | Unchanged; the annotation types assignment |
| R4 | Dict displays are joined; `None` makes the union Optional | Resolved by annotation (`heating: A \| B \| None`) | See R21 |
| R14 | Assignment through a path would mutate a shared class-body declaration | Stored on the path's first node, keyed by **member path** | Paths survive a replaced intermediate node |
| R15 | A method halts only through a blocked node read | Presence nodes; `present()` reads without halting on absence | |
| R16 | `Users` reads one export per user | Unchanged: a kernel's port refusal reaches all its streams | Deferred to §3 (keyed gather) |
| R17 | A `members` node carried one member name | Tuple of member names | Unchanged |
| R18 | Model caches assume immutable declarations | Freezing on preparation and on `design_space` | Unchanged |
| R19 | `query(K.output)` typed as the family, answered `True` | **Resolved**: answers the configuration; `present()` | |
| R20 | A stream's `spec` must not derive from its users | Anchoring rule, detected at evaluation | Unchanged |
| R21 | mypy infers a generic call's type variable from its arguments before the declared type, so a dict display of candidates joins to their base | A Decision over nodes returns `Any`; collection checks candidates against the annotation | The annotation types the choice |
| R22 | Annotations are strings (`from __future__ import annotations`) and families declared in functions name function locals | The declaring frame's globals, class namespace and enclosing locals are captured when a member is created | Resolution works for local families and forward references |
| R23 | `Param()` runs before its name or annotation exists, yet `output.spec` in the body needs the family | The Param finds itself in the class namespace being built and reads its annotation there | `__getattr__` only for members of the family: introspection is unaffected |
| R24 | mypy honours a `__new__` that returns a non-instance only when there is no `__init__` | `Present` builds itself in `__new__` | |
| R25 | A node frame could make only one native call | A frame may make another after its first returns | Contract checks call the declared domain after the replacement's |
| R26 | A replacement node's own settings sit innermost structurally but were written by an outer body | Layers ordered by the writing body's depth | Outermost is a property of bodies, not records |
| R27 | A view read as `BoundView[T]` through a node | Iteration 3: `accepted(view) -> T` | **Resolved** in iteration 4: a view reads as its value (§1); `accepted` and `BoundView` removed |
| R28 | A key typed as its value cannot say whether it is a Param or a Decision | `field(T) -> BoundDecision[T]` | Decision operations on a Param fail at runtime |
| R29 | Bypassing an alias drops its guard | Bypass only when the alias's guard is the reader's or an outer one | A guarded node's alias read from outside keeps its hop |
| R30 | In its own class body a view's name is the `View` object, and mypy types the body name and class access (`House.total`, which must stay `View[T]` for `inspect`) from the one assignment | The body name stays `View[T]` | Supplying a formal from a view of the **same** body needs `cast(int, total)`; the runtime accepts it. Through a node (`kitchen.cost`) there is no cast |
| R31 | A reference into a child (`House.kitchen.cost`) is typed as its value, so `inspect` cannot see that it names a view | An `inspect(T) -> ViewAssessment[T]` overload for the path form; the child form `point.kitchen.inspect(Room.cost)` is typed by the declaration | A non-view through a path is refused at runtime, naming the member (the same trade as R28) |
| R32 | A view reference and a Param reference are both typed `T` | `requires=` is typed `Sequence[object]` | A non-obligation is refused at link, as before; mypy no longer rejects `requires=(3,)` |

Carried over and still open: the design-graph spike's R6 (a refusal reaching a
view through both its output and an obligation is reported twice) and R7
(whole-snapshot caches). R6 is easier to meet now that a method reads a view
without a call: `Hall.total` in `test_view_values.py` requires `kitchen.cost`
and reads `self.kitchen.cost`, so while `kitchen.finish` is open its accepted
result lists that blocker twice (once through the obligation, once through the
output). Iteration 3 behaved identically with `self.kitchen.cost()`.

## 7. What was removed

Iteration 4: `BoundView`, `Space.view()`, `accepted()`, `occurrence.bind_view`,
and the `BoundView` arms of every signature (§1.3). Iteration 3: `Param(T)` / `Param(Family)` / `Param(Located)` value-type
arguments, `Param[T]` annotations, `Decision(T, ...)` and `Decision[T](...)`,
`UNSUPPLIED` (public), `FamilyFormal` and `family_formal`, `Param.__set__`,
`PlacementPlan`/`placement_plan`, `NodeDecl.supplied_at` and the identity-keyed
`nested` map, the across-bodies "already supplied" rule, `SpaceModel`,
`configure`, `compile_space`. Earlier removals stand (iteration 2: `OPEN`,
`Bind`, first-use keying, `StreamLink`; iteration 1: `Subspace`, `ScopeBuilder`,
`ValueKey`, `External`, ...).

## 8. Key and identity changes

Iteration 4 changed no key, node, kind, name or port: views reading as values
is an access-path and typing change (§1.1). The facts below were re-checked on
this branch.

- **MVAU decision keys: none changed.** `compute.compute_pumping`,
  `implementation`, `implementation.cyclic.rom_style`, `pe`, `simd`,
  `weight_stream.transport`, `weight_stream.transport.fifo.buffer.depth`,
  `weight_stream.transport.fifo.buffer.ram_style`.
- **MVAU node keys and kinds: none changed** (`mvau_keys.py` on
  `426882411` (iteration 3) and on this branch: identical output, 227 lines,
  35 `view` nodes; [`evidence/mvau-keys.diff`](evidence/mvau-keys.diff) is
  empty. Iteration 3 made the same comparison against `43d4576a6`).
- **Top-level ABI port names: none changed.** `in0_V`, `in1_V` (external),
  `out0_V`.
- **Instance names: none changed** (`u_implementation_cyclic` since iteration 1;
  `0d700b1ab` named it `u_weights`).
- **Generic:** a pinned Decision's key disappears (by design) and is listed by
  `inspection.pinned`; a projection adds `...$project.N` nodes; the root of a
  declaration that pins or overrides below it is compiled per declaration.
- **Evidence shape:** `explain` no longer lists a bypassed alias as a node; it
  appears in the reader's `via`.
- **Messages:** "unknown formals" → "unknown members"; "while configuring" →
  "while opening the design space of"; "already supplied" → "already assigned
  ... in this body" (iteration 3); "this declaration is not assessable" →
  "`<key>` is not a view or a constraint: inspect() assesses those; read or
  query() any other member" (iteration 4).

## 9. Evidence

All runs are on `spike/space-declarative-4`; transcripts are under
[`evidence/`](evidence/) (`evidence/README.md` lists them). The iteration-3
baseline is `426882411`, extracted with `git archive` and run in the same
session.

| Evidence | Result |
|---|---|
| Gate `check-kernels.sh` (normal `PATH`, Xilinx 2025.2 `xelab`/`xsim`) | Space **418 passed**; kernels **763 passed, 0 skipped**, including the 9 XSim tests; format, lint, strict mypy clean ([`evidence/gate-check-kernels.txt`](evidence/gate-check-kernels.txt)) |
| XSim tests, run separately | **9 passed** in 247 s, `Vivado Simulator v2025.2` ([`evidence/xsim-tests.txt`](evidence/xsim-tests.txt)) |
| Gate `check-dataflow-design.sh` | **16 passed**, clean |
| Views as values | §1.6: `test_view_values.py` (9), `test_reference_misuse.py` (+25) |
| Typing | §1.4: `positive.py` (a new iteration-4 section), `negative.py.txt` (40 expected errors, line for line), `extensions.py`, the kernels' `typing/` fixtures, all under `--strict` |
| MVAU fingerprints | [`evidence/fingerprints.txt`](evidence/fingerprints.txt) **identical to iteration 3**; [`evidence/fingerprints-renamed.txt`](evidence/fingerprints-renamed.txt) (cyclic instance renamed to `u_weights`) **bit-identical to `0d700b1ab`** |
| MVAU keys | [`evidence/mvau-keys.diff`](evidence/mvau-keys.diff) empty against `426882411`: every decision key, node key and node kind unchanged; [`evidence/mvau-abi-ports.txt`](evidence/mvau-abi-ports.txt) top-level ports unchanged |
| Collapse | [`evidence/collapse-probe.txt`](evidence/collapse-probe.txt): every count identical to iteration 3 |
| README | every Python block of `src/finn/kernels/README.md` runs, as it did at `426882411` |
| Benchmark | `scripts/benchmark-space.py` at small sizes: `"semantic_assertions": "passed"` |
| Overrides, presence (iteration 3) | still green: `test_overrides.py` (14), `test_collapse.py` (4), `tests/kernels/test_mvau_collapse.py` (2), `test_references.py` |

**The scale probe** (same session, round 2 of
[`evidence/scale-probe.txt`](evidence/scale-probe.txt), in ms):

| | prepare (it. 3) | prepare (it. 4) | full read (it. 3 / 4) | read after one local edit (it. 3 / 4) | callbacks re-run |
|---|---|---|---|---|---|
| N=50 | 36.3 | 36.2 | 15.0 / 14.9 | 15.6 / 15.5 | 50/50 |
| N=200 | 139.1 | 137.1 | 60.3 / 59.8 | 58.6 / 58.1 | 200/200 |
| N=800 | 536.5 | 532.0 | 235.6 / 236.0 | 226.0 / 226.1 | 800/800 |

Within noise, as expected: the runtime did not change, and a view read is the
same `read_value` the old `BoundView.__call__` made (one fewer object per
read). One local edit still re-runs the whole graph (R7).

## 10. Open questions for human review

Iteration 4 (views read as values):

1. **A view named in its own class body.** It is its declaration, `View[T]`,
   so supplying a formal from it needs `cast` (R30). Options: accept the cast;
   type `@view` as its value in the body and reach the declaration some other
   way (`inspect` would then need a typed key other than `House.total`, for
   example `House.views.total`); or forbid the same-body edge and ask for a
   `@derived` beside the view. Which cost is preferred?
2. **The path form of `inspect`.** `point.inspect(House.kitchen.cost)` is
   typed by a value-typed overload and accepts any member statically (R31);
   `point.kitchen.inspect(Room.cost)` is precise. Keep both, or keep only the
   child form so `inspect` stays statically exact? The same question stands
   for `field` (R28), which it mirrors.
3. **Duplicate blockers when a view is both read and required (R6).** Reading
   views without a call makes the pattern common. Should a view's reduction
   de-duplicate findings by owner and code, or should an obligation that the
   output already reads be refused as redundant?
4. **Statically untyped obligations (R32).** `requires=` is `Sequence[object]`.
   Is the link-time refusal enough, or should obligations be written as a
   distinct marker (for example `requires=(accepted_by(kitchen.cost),)`)? That
   would reintroduce a bridge, which this iteration removed.

Carried from iteration 3:

5. **The declared domain as a contract.** A pin or a replacement is checked
   against the declared domain (widening refused). Is that the intended reading
   of "behaviour is not overridable", or should a replacement Decision replace
   the domain outright?
6. **Pinning a Decision over nodes.** Only narrowing is offered (down to one
   case, key kept). Is a true pin (key removed, the other candidates never
   placed) needed, and how should it be typed?
7. **Persisted vs in-memory selections.** A persisted selection with a pinned
   key is refused as stale; an in-memory `Selection` is bound to its model.
   Should `restore` accept another model's selection by key?
8. **Which values a refusal names.** A computation's refusal names the
   overridden values it read directly. Transitively (through the derived values
   it read) would be more complete but noisier; single-layer supplies are not
   named at all.

Further: `Users` through forwarding composites and keyed gathers (§3); should
`Const` members be typed as their values like `@derived` (in its own body a
`Const` is its declaration, `Const[T]`, the same position as a view's, R30);
should a path assignment through a reference input be allowed (today: assign
the node where it is placed); should the kernels gate run its XSim tests as a
separate target.
