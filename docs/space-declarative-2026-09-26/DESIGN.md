# Declarative design spaces: nodes, references, one compile step

Date: 2026-09-26. Base: `spike/space-design-graph` at `0acc51e1e`. Branch:
`spike/space-declarative`. Status: **design spike, at the human review gate.**
Not for merge as is.

This spike replaces the composition layer of `finn.core.space` with a
declarative graph of design spaces. It builds on the previous spike
(`docs/space-design-graph-2026-09-26/DESIGN.md`, which is untracked in the main
checkout and not on the base branch: open formals, `Bind`, `Present`, located
values, `Members`, view obligations),
and keeps its verdict: the value graph and runtime stay as they are, and only
the authoring and linking layer changes. There are no domain notions here:
no ports, directions or carried values.

## 0. Summary

| Question | Answer |
|---|---|
| What is a node? | Calling a family, `Room(area=12)`, returns a **node declaration**: a declaration-mode `Room` object holding its bindings. Nothing is compiled. |
| What is an edge? | A binding at the call (`Room(area=kitchen.area)`), or a `Bind` declared after its nodes. |
| What is a reference? | Attribute access on a node declaration: `kitchen.finish`. `House.kitchen.finish` from class level is the same reference. |
| What is a structural choice? | `Decision(values={"boiler": Boiler(), "heat_pump": HeatPump(), "none": None})`. |
| How is a configuration made? | Only by `configure(House(budget=100))`, typed `House`. |
| Typing | Option A. Nodes are typed as their family, references as their values, and `self` in methods as a configuration. `dataclass_transform` types every family call. |
| Engine verdict | The runtime (scheduler, snapshots, admission, selections, codecs) is unchanged, except that a `select` of an absent candidate member is `Inapplicable`. Linking was rewritten around node records. The IR gained three fields (`Node.origin`, `Scope.record`, `LinkedModel.editable_aliases`), and `Choice.exports` became `Choice.members`. |
| Evidence | Both gates green. The house toy, the 14 generic acid tests, and 28 misuse tests pass. With the one instance renamed, the 6 MVAU fingerprints are bit-identical to `0d700b1ab`, and no MVAU decision key changed. |

**Flagged deviations from the decided design.** Each is the closest workable
form; §5 and §6 give the reasons.

1. **`OPEN`.** A *required* formal that a later `Bind` supplies must be
   written `Room(area=OPEN)`. mypy cannot see the later `Bind`, so a bare
   `Room()` is a "missing formal" error, which is exactly the typing the design
   asked for. The runtime enforces the same rule. Optional formals are open
   implicitly.
2. **Union typing needs a type application.** `Decision(values={...})` over
   different families is joined by mypy (to `Space`). The union comes only from
   `Decision[Boiler | HeatPump](values={...})`.
3. **`None` in the union.** With a `None` candidate the class-body type is
   `N | None`, so mypy rejects `heating.kw` (the design wanted `None` ignored).
   Use a **candidate handle** instead: a class attribute that names a candidate
   (`cyclic = CyclicDelivery(...)`, then `values={"cyclic": cyclic, ...}`). It is
   an addition to the design.
4. **`heating.kw` lowers to a `select` node**, not a `present` node. It is
   unresolved until the decision is made. It is `Inapplicable` (rather than
   "unsupplied") when the selected candidate lacks the member or is `None`.
5. **Unnamed decisions may only supply formals.** A fresh `Decision` used by
   several nodes is shared and its owner inferred (§4.4). Nodes are never
   inferred: a node is placed explicitly or it is an error.

## 1. The model

A **family** is a `Space` subclass. Its class body declares members and nodes:

```python
class House(Space):
    budget: Param[int] = Param(int)                    # formal (annotated: typed call)
    want_garage = Decision(bool, values=(False, True))
    hall = Room(area=OPEN)                             # node; formal left for a Bind
    kitchen = Room(area=12)                            # node
    dining = Room(area=16)
    garage = Room(area=20, when=want_garage)           # guarded node
    heat_pump = HeatPump(kw=8)                         # a handle naming a candidate
    heating = Decision[Boiler | HeatPump](values={"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    thermostat = Thermostat(kw=heating.kw)             # the selected candidate's kw
    hall_area = Bind(hall.area, kitchen.area)          # an edge after its nodes
    matched = Match(a=kitchen.finish, b=dining.finish) # Match.a: LocatedParam[int]
    costs = Members(COST)                              # present children exporting COST

    @view(requires=(costs, within_budget, matched.agreed))
    def total(self) -> int: ...

house = configure(House(budget=100))
point = house.with_choices({House.heating: "heat_pump", House.heat_pump.cop: 3,
                            House.kitchen.finish: 2})
```

The complete toy is [`tests/core/space/test_house.py`](../../tests/core/space/test_house.py).

### Concepts and their lowering

| Concept | Written | Lowers to (evaluation graph) |
|---|---|---|
| Node declaration | `Room(area=12)`, placed by a class attribute | one scope; its members become nodes keyed `kitchen.area`, ... |
| Formal | `area: Param[int] = Param(int)` | `param` at the root (a runtime input); in a child: `const` (literal), `alias` (reference), `decision` (fresh Decision), or `present` (open) |
| Optional formal | `Param(int, default=UNSUPPLIED)` or `default=<value>` | open: `present` over its binds; with none, unsupplied, or a `const` of the default |
| Open required formal | `Room(area=OPEN)` | `present` over its `Bind` edges; with none, a definition error |
| Reference | `kitchen.finish`, `House.kitchen.finish`, `h.kitchen.finish` | resolved by node identity through `Scope.children` to the member's node |
| Guard | `Room(..., when=c)` | a `guard` node on the scope (as before) |
| Decision over nodes | `Decision(values={"a": A(), "b": B(), "n": None})` | a `decision` node keyed `heating` (str domain = keys), one scope per candidate keyed `heating.a`, guarded by a `$selected` node |
| Member through a choice | `heating.kw` | a `select` node `heating.$member.kw` keyed by the selector |
| Selected key | `selected(heating)` | an `alias` of the selector (read-only) |
| Candidate handle | `cyclic = C(...)` then `values={"cyclic": cyclic}` | nothing: the candidate's scope is reachable by node identity |
| Edge | `Bind(hall.area, kitchen.area)` | an `alias` node that is one alternative of the target's `present` |
| Whichever is present | `Present(a.out, b.out)` | a `present` node |
| Located formal | `a: LocatedParam[int] = Param(Located)`; `Match(a=kitchen.finish)` | a `locate` node; value `(node name relative to the reading scope, member name)` |
| Quantification | `Members(COST)` | a `members` node over the children's exported views |
| Obligation | `@view(requires=(costs, within, matched.agreed))` | view constraints; a `Members` obliges each member |
| Family-typed formal | `home: House = Param(House)`; `Estate(home=House(budget=1))` | the supplied node's scope, placed at `home` |
| Fresh (unnamed) Decision | `Fifo(depth=Decision(int, values=(4, 8)))` | a `decision` at `second.depth`; shared by several nodes, one decision at their lowest common scope (§4.4) |
| Graph as data | `composite("Pipeline", {"s0": s0, "e1": Bind(...)})` | a family, collected like a class body |

The compile step: `configure(node)` checks that `node` is an unplaced
declaration (a root). If every root binding is a plain value, it reuses the
family's model (compiled once, cached on the family) and binds the values as
runtime inputs. That keeps one model per family, so selections and codecs
replay across roots. If a root binding supplies structure (a node, a
reference, a fresh Decision or `OPEN`), the model is compiled for that
declaration and cached on it.

## 2. Public API

- **Families and nodes:** `Space`, a family call `F(**formals, when=...)`,
  `configure(node) -> F`, `composite(name, members, base=, exports=)`.
- **Members:** `Param` (plus `LocatedParam` via `Param(Located)`, and a
  family-typed formal via `Param(Family)`), `OPEN`, `UNSUPPLIED`, `Const`,
  `Decision` (over values, or over nodes via `values={key: node | None}`),
  `selected(decision)`, `Derived`/`@derived`, `Constraint`/`@constraint`,
  `ConstraintGroup`, `View`/`@view(requires=...)`, `ViewKey` + `exports`.
- **Graph primitives:** `Bind(target, source, when=)`, `Present(*refs)`,
  `Members(key)`, `Located`.
- **Configurations** (unchanged apart from edits): reads, `query`, `inspect`,
  `view`, `field`, `root`, `with_choices` / `try_with_choices` taking
  `{reference: value}` mappings, `Change` objects, or keywords for the point's
  own decisions.
- **Inspection:** everything as before, and every function accepts a family
  class too (compiled on demand). New:
  - `inspection.model(subject)`;
  - `inspection.declaration(node)` (family, name, placement, members,
    bindings, open formals, guard, origin, all *before* compiling);
  - `inspection.reference(ref)` (path, member, origin);
  - `inspection.candidate(point, decision, key)`;
  - `NodeInfo.origin`;
  - `CaseInfo.scope` / `space_type`, which are `None` for a `None` candidate.
- **Errors:** `ReferenceUseError` (a `TypeError`), raised on value-like use of a
  reference.
- **Kernels:** `finn.kernels.configure.commit(point, {key: value})` replaces the
  old `configure(space_type, facts, choices)`. Facts are now the root node's
  typed formals.

## 3. References

`kitchen.finish` is a `MemberRef(path=(kitchen,), member=Room.finish)`. `path` is
a tuple of node records, outermost first. `house.kitchen.finish` (with `house`
a declaration) is `MemberRef((house, House.kitchen), ...)`, and resolution skips
the element that is the scope's own node. So both forms resolve against a
`House` configuration. Resolution is by identity through `Scope.children`,
never by name. The place-once rule makes node identity equal to placement
identity.

- **Class access is the schema key.** `House.kitchen` returns the kitchen node
  itself, so `House.kitchen.finish` is the same reference as the class-body
  `kitchen.finish`. `Room.finish` is the member declaration ("this member of
  every Room").
- **Hash and equality** are structural, over (path identities, member). This is
  enough for mapping keys (`{House.kitchen.finish: 2}`). `==` against another
  reference is a `bool`; it is not an expression.
- **Integer arithmetic** (`+ - * // %`, unary `-`) still builds `Expr` nodes.
- **Every other value-like use** raises `ReferenceUseError`: truthiness, `==`
  with a value, ordering, `len`, iteration, `in`, indexing, calling, `str`,
  `format`/f-strings, `int`/`float`/`complex`/`index`, `round`/`floor`/`ceil`/
  `trunc`/`abs`, `max`/`min`/`sorted`, `/`, `**`, bitwise and shift operators.
  The message names the reference, where it was written, and what to do:

  ```
  <Room node (declared at test_reference_misuse.py:44)>.area (declared at
  test_reference_misuse.py:44) is a declaration reference, not a value: a truth
  value is not available while a Space is declared. Compute with it in a
  @derived or @view method instead (or pass the reference where the value is needed).
  ```

  Inside a class body a node has no name yet (`__set_name__` runs after the
  body), so the message identifies it by its declaration line. Once the node is
  placed, the message uses its path (`kitchen.area`).

## 4. Decisions, with reasons and rejected alternatives

### 4.1 The compile step is `configure`

The result is a *configuration*, and the codebase already says so:
`ConfigurationResult`, `ConfigurationError`, and "configuration" throughout
`_changes`. So `configure(House(budget=100))` names what comes back, and it
is edited with `with_choices`.

- **`compile`** shadows a builtin.
- **`space.compile`** forces a qualified import, and its result is not a
  compiled model.
- **`instantiate`** collides with the idea that calling a family already
  instantiates a template.
- **`realize`** is vague.

The old kernels helper named `configure` became `commit(point, choices)`. It
now only commits choices by key, because facts are typed formals.
`compile_space(family)` survives in `finn.core.space.compiler` as the internal,
cached template compiler behind `configure` and `inspection.model(family)`. It
is no longer exported: `SpaceModel.bind` is gone, and configurations come only
from `configure`.

### 4.2 `Members(ViewKey)` over the structural alternatives

I evaluated two alternatives against MVAU and the house toy.

- **By member name (the Protocol form), `Members(HasBuildRequirements)`.** For
  MVAU's children, matching the name `build_requirements` selects the same three
  nodes as `MODULE`, and `connection` the same four streams. So the form would
  work, but it fails in two ways:
  1. **It cannot be typed.** Python typing cannot project an attribute type out
     of a Protocol, so the element type must be restated
     (`Members[ModuleBuildRequirements](HasBuildRequirements)`). A `ViewKey`
     carries the name and the type in one object.
  2. **It matches by accident.** Every same-named member joins in. Placing
     another MVAU as a child would silently add its composed
     `build_requirements`, and every `StreamLink` has a `stage` view.
- **By declaration, `Members(Kernel.build_requirements)` (nominal).** This
  needs a common base that declares the member. Space has no abstract members,
  and inventing them only for this is new machinery. It also forces
  inheritance on unrelated families (Dotp, ReplayBuffer, CyclicDelivery).

**Kept: `ViewKey` + `exports`.** An export is an explicit, typed opt-in, and it
is independent of the family hierarchy. That is what heterogeneous graphs need.
The choice exports that `SubspaceChoice` had are gone: members are read through
a choice by name (`transport.stage`).

### 4.3 `requires=` instead of `constraints=`

A view's list now holds constraints, groups, other views, references to child
views (`matched.agreed`) and `Members`. `constraints=` described one kind of
entry out of five. `requires=` says what the list does (the view is accepted
only if they are); it does not say what the entries are.
`obligations=` is accurate but heavy. The assessment's result field keeps its
name (`ViewAssessment.constraints`).

### 4.4 Ownership inference, for fresh decisions only

A *fresh* Decision is one that no class attribute names. It may supply formals
at node calls.

- **The unit is one family body.** The decision is identified by (the
  instantiated scope of the family body that contains it, the Decision object).
  A family placed twice gets two decisions, exactly as for member decisions.
  This matters: `StreamFifo`'s inline `depth=Decision(...)` must be
  per-placement.
- **One use:** the decision belongs to the node whose formal it supplies. Its
  key is `second.depth`, as before.
- **Several uses:** it is **one decision owned by the lowest common scope of
  the nodes it supplies**. So it applies whenever any use applies, and it takes
  the key of its first use in declaration order. Every use becomes an alias of
  it that keeps its own node's guard: an inactive use reads `Inapplicable`
  while the decision stays editable through the others. It can be edited
  through any use (`{Twin.left.area: 4}` or `{Twin.right.area: 5}`), and
  through the Decision object from the owner scope.
- **Nodes are never inferred.** A reference to a node that was never placed
  is an error that names where the node was declared:
  `x.area: Room node (declared at probe.py:72) is not placed in <root>`.
  Nodes need names, and a name inferred from a use would be arbitrary.
- **Rejected: one decision per object across the whole graph.** That shares a
  class-body inline decision across every placement of its family.

A fresh Decision may only supply formals: a node-call keyword argument, or the
formal of a candidate. A `Bind` source or a guard needs a named decision; this
is a scope limit of the spike.

### 4.5 The place-once rule

A node declaration is placed exactly once:

- by a class attribute (`__set_name__`),
- by a family-typed formal (`Estate(home=House(...))`), or
- as a Decision candidate.

A second placement is refused, and the message names both sites:

```
Room node placed at X.a (declared at probe1.py:58) cannot also be placed at X.b:
a node declaration is placed exactly once. Declare a fresh node for each
placement, for example from a function that returns one.
```

On Python 3.10, an exception raised in `__set_name__` reaches the caller as a
`RuntimeError` whose `__cause__` is the `DefinitionError`; this was already
true of declarations. `composite` unwraps it. Reuse goes through functions
that return fresh nodes, as `integer_scalar(...)`, `native_stream(...)` and
`axi_stream(...)` do.

**One addition: a candidate handle.** A class attribute that names a node
already adopted by a Decision *in the same body* does not place it again. It is
a typed handle to the candidate (`MVAU.cyclic`, `House.heat_pump`), and the
Decision still places it (`implementation.cyclic`). Collection refuses a handle
whose Decision lives in another family. The handle is how a class body reaches
one candidate's member with an exact type (§5).

### 4.6 Family-typed formals: implemented

`home: House = Param(House)` returns a formal node (a declaration-mode
`House`). In the class body `home.kitchen.finish` is therefore a reference
typed `int`. The caller supplies a node, `Estate(home=House(budget=5))`, and
that node is placed at `home`: its keys are `home.kitchen.finish` and it reads
as `self.home` in methods.

It composes cleanly with the rest:

- references through the formal resolve by identity;
- guards and fresh decisions inside the supplied node are authored in the
  caller's scope;
- mypy types the call (`Estate(home=Room(...))` is an error).

Limits:

- a family-typed formal is required, and cannot be open or bound by `Bind`;
- a family that has one cannot be compiled alone (`compile_space` refuses),
  because its structure depends on the node supplied;
- `configure` compiles a model per root declaration and caches it on the node.

### 4.7 `ScopeBuilder` is replaced by `composite`

Nodes are plain Python values now, so a graph built as data is a list of
nodes plus `Bind` edges. The only remaining job is to *name* them as the
members of a family, and a class statement already does that. So
`composite(name, members, base=, exports=)` is `type(name, (base,), members)`,
plus name validation and one eager collection for early errors. Python's class
construction places the nodes (`__set_name__`).

`ScopeBuilder`'s other features (a mutable session, sealing, bind/export
recorders, typed `BindingTarget.to`) existed because declarations could not be
built outside a class body. Its tests were ported to intent: purity and
repeatability, early errors, duplicate/foreign exports, typed extension over a
base family, and literal snapshots. Rejected alternatives:

- **Collect by reachability from a root.** Nodes have no names until a
  container gives them names.
- **An explicit graph container object.** It would be a second family concept.

### 4.8 Smaller decisions

- **Literals are frozen once, at the node call.** Previously this happened at
  compile time. An unrecognized literal is a `DefinitionError` at the call; an
  adapter that raises is an `EvaluationError` with the role `parameter
  snapshot`/`recognition`. `inspection.declaration` hands out detached copies.
- **Where node-call errors appear.** Unknown formals, missing required formals
  and positional arguments are `DefinitionError`s at the call. A family call
  takes keywords only. Everything that needs the graph fails at `configure`.
- **Classification waits for linking.** Whether a supplier is a member or a
  fresh Decision is decided at link time: inside a class body no declaration
  has an owner yet (R1).
- **`decision.member` at runtime.** A name several candidates share is linked
  as a `select` node. A name only one candidate has resolves to that
  candidate's own member, which is present exactly when that candidate is
  selected. For an edit, `{House.heating.cop: 3}` names the only candidate that
  has `cop`; an ambiguous name must go through a candidate handle.
- **A singleton choice is an ordinary one-value Decision** and needs a
  commitment. This removes the special selector-less choice.

## 5. Typing results

The mechanism: `SpaceMeta` is decorated with `dataclass_transform(kw_only_default=True,
field_specifiers=(Param, when-field))`, and **formals are annotated**
(`area: Param[int] = Param(int)`). Two things follow:

- The constructor keyword type of each formal is `Param.__set__`'s value type:
  `T | ValueRef[T] | View[T] | BoundView[T]`, and for a `LocatedParam[T]`
  additionally `Located[T]`.
- Because references are typed as values, `kitchen.area` (an `int`) is
  accepted wherever an `int` is.

`when: ValueRef[bool] | bool | None` is a keyword-only field on `Space`.
Evidence: [`tests/core/space/typing/positive.py`](../../tests/core/space/typing/positive.py)
and `negative.py.txt` (24 expected errors, matched line for line by
`test_typing.py`).

| Claim | mypy result |
|---|---|
| `kitchen.finish` in a class body is `int` | **exact** (`assert_type(kitchen.finish, int)`) |
| `kitchen.cost` (a view) in a class body | `BoundView[int]`; accepted by `int` formals through `__set__` |
| `self.kitchen.finish` in a method is `int` | **exact**; `self.kitchen.cost()` is `int` |
| `House.kitchen.finish` from class level | `int` (option A); `Room.finish` is `Decision[int]` |
| Family call keywords typed; missing required formal is an error | **exact**: `Missing named argument "width"`, `Unexpected keyword argument "depth"`, `incompatible type "str"` |
| A reference of the wrong type is rejected | **exact** (`Child(width=child.label)` with `label: str`) |
| `when=` needs a Boolean | **exact** |
| `configure(House(...))` is `House` | **exact** |
| Family-typed formal | **exact**: `home.kitchen.finish: int`; `Estate(home=Room(...))` is an error |
| `heating.kw` with candidates `Boiler \| HeatPump` | **exact only with `Decision[Boiler \| HeatPump](...)`**. mypy joins a dict display: plain `Decision(values={...})` is typed `Space`, so `heating.kw` is `"Space" has no attribute "kw"`. Annotating the attribute does not help: the overloaded `__new__` gets no type context from the annotation. |
| `heating.cop` (only on `HeatPump`) | **correctly rejected**: `Item "Boiler" ... has no attribute "cop"` |
| A `None` candidate | typed `N \| None` (mypy infers this from the dict). In a method, `self.heating` is exact (`Boiler \| None`). In a class body, `heating.kw` is rejected (`Item "None"`), so reach the member through a candidate handle. |
| `selected(heating)` | `str`, exact in methods |
| Edits `{House.kitchen.finish: 2}` | keys are `Any` in the mapping: **no static check** of values against keys |
| `point.field(Family.decision)` | exact `BoundDecision[T]`, also for a `DecisionHandle[T]`. A nested reference is typed as its value, so `point.field(House.kitchen.finish)` is `BoundValue[int]`; use the mapping or `point.kitchen.field(Room.finish)` |
| `codec_for` / `Selection.value` | exact for a class-level `Decision[T]` and for handles; a Decision over nodes takes a `str` codec. A reference is typed as its value (`T`), so a `Param` passed where a decision is expected is **not** a static error. |
| `Bind(target, source)` | **approximate**: mypy infers `T` from both arguments (a join), so `Bind(child.port.width, "wrong")` is not an error. `Bind[int](...)` is. |
| `Present(a.out, b.out)` | exact (overloads for references and for values) |
| Arithmetic on a reference in a class body | typed `int`, so a class member `x = child.width + 1` is `int` at class and instance level (at runtime an `Expr`) |
| Unannotated formal | invisible to mypy: its keyword is `Unexpected keyword argument` |

What mypy could not do:

1. Know that a later `Bind` supplies a formal, hence `OPEN`.
2. Infer a union from a dict display (it joins).
3. Type-narrow `N | None` for class-body member access.
4. Take a type variable from only one argument (`Bind`).
5. Tell a reference from its value (the premise of option A).

Two mypy behaviours shaped the engine code:

- mypy 2.3 applies a descriptor's `__get__` to property return types and to
  class-level annotations of descriptor type. Every `Space` node is now a
  descriptor, so engine records keep instance-level attributes, and
  `Space.__get__` accepts any owner instance.
- `__new__` overloads that return non-instances need `# type: ignore[misc]` in
  the library. Call sites see the declared return types.

## 6. Resistance log: where the existing engine pushed back

| # | Resistance | Change | Reading |
|---|---|---|---|
| R1 | In a class body, sibling declarations have no owner until `__set_name__` runs after the body. At a node call, `Room(finish=member_decision)` and `Room(finish=Decision(...))` look the same | Supplier classification (fresh decision, inline Param, member) moved from the call to the link | Node calls store raw suppliers; validation splits between the call (names, literals, missing formals) and the link (structure) |
| R2 | `dataclass_transform` fields must be *annotated*; the engine's idiom `x = Param(int)` is invisible to mypy | Formals annotated (`x: Param[int] = Param(int)`): 63 kernel formals, and the families of every test | The one real authoring cost of typed calls. Runtime accepts unannotated formals; mypy then rejects their keywords |
| R3 | mypy cannot see a later `Bind` | `OPEN` sentinel (typed `Any`); runtime enforces the same rule | A typed call and free edges conflict; `OPEN` makes openness explicit |
| R4 | Dict displays are joined; `None` makes the union Optional | `Decision[A \| B](...)`; candidate handles | The class-body type of a choice cannot be both its candidates and "or nothing" |
| R5 | Making every `Space` node a descriptor (for placement) changed mypy's view of any class-level annotation or property typed as a `Space` | Instance-level attributes in records; `Space.__get__(instance: object)` | Option A spreads: once nodes are descriptors, *containers* of nodes must avoid class-level annotations |
| R6 | Python 3.10 wraps `__set_name__` exceptions in `RuntimeError` | Accepted (pre-existing for declarations); `composite` unwraps | Place-once at class creation is exact on 3.12+, wrapped on 3.10 |
| R7 | The linked model is frozen, so `decision.member` cannot create a selection at query time | Shared names linked eagerly; a single-candidate name resolves to that candidate; an incompatible unreferenced name is dropped at semantics check | Lowering happens at link time, which forces completeness decisions up front |
| R8 | `Node.scope` meant both "owning scope (applicability)" and "whose member map holds it" | A shared fresh decision gets its own node in the owner scope; every use becomes an editable alias | Owner and use are different identities; the IR needed the alias set `LinkedModel.editable_aliases` |
| R9 | Constraints and groups were not descriptors, so `child.supported` on a node returned the child family's declaration (the wrong scope) | A node-reference descriptor on `Constraint`/`ConstraintGroup` | Every member kind reached through a node must return a reference |
| R10 | One Python type now covers a declaration and a configuration (option A); every configuration method must detect the mode | `state()` refuses a declaration with "is a node declaration, not a configuration; configure() its root" | The price of option A at runtime is small and local |
| R11 | Selections and codecs require one model per family (`restore` checks model identity) | Root literals remain runtime inputs; only structural roots compile per node | The compile cache key is "family plus structural bindings" |
| R12 | A candidate's name contains a dot (`implementation.cyclic`), which is not a Verilog identifier | `netlist` maps `.` to `_` (`u_implementation_cyclic`) | The only fingerprint change (§8) |
| R13 | `SubspaceChoice`'s selector was a generated node `choice.$selector` with a key mapping | The selector *is* the member node `choice` | Removes `decision_key` indirection; persisted keys are unchanged |

Carried over and still open: the previous spike's R6 (a refusal that reaches a
view through both its output and an obligation is reported twice) and R7
(whole-snapshot caches, §9).

## 7. What was removed

`finn.core.space`:

- **Structural declarations:** `Subspace`, `SubspaceChoice` (with `exports=`,
  singleton selector-less choices and `ChoiceView.select/alternative/alternatives`).
- **Builders:** `ScopeBuilder` (with `BindingTarget`, `ValueExport`,
  `ViewExport`).
- **Keys and references:** `ValueKey`, `DecisionRef`, `AcceptedViewRef`,
  `ScopedValueRef`, `LocatedRef`, `located()`, `ChoiceCaseRef`.
- **Placement internals:** `collect_placement`, `PlacementPlans`,
  `_TargetResolver`.
- **Old ways to configure:** `SpaceModel.bind`, the top-level `compile_space`
  export, and calling a family to get a configuration.
- **Binding forms:** exposed inline `Param` suppliers (root parameters named
  `child.x`), nested `bindings={...}` maps (now a `Bind` to an open nested
  formal), and `required=` (now `default=`).
- **Keywords:** `constraints=` (now `requires=`).

`finn.kernels`:

- `External` (now a `None` candidate);
- the `STAGE` key;
- `configure(space_type, facts, choices)` (now core `configure` plus `commit`).

## 8. Key and identity changes

- **MVAU decision keys: none changed.** `pe`, `simd`,
  `compute.compute_pumping`, `implementation`,
  `implementation.cyclic.rom_style`, `weight_stream.transport`,
  `weight_stream.transport.fifo.buffer.depth` and
  `weight_stream.transport.fifo.buffer.ram_style` are all verified by
  `tests/kernels/test_mvau_delivery_choice.py`.
- **Cyclic delivery naming.** Its node name (`Located.node`, `Members`) is
  `implementation.cyclic` (base spike: `implementation`). Its netlist instance
  is `u_implementation_cyclic` (base spike: `u_implementation`; `0d700b1ab`:
  `u_weights`).
- **Generated nodes:**
  - the selector node is `implementation`, not `implementation.$selector`
    (the persisted key is the same);
  - choice-member nodes are `<choice>.$member.<name>`, not `$export`;
  - a shared fresh decision's uses are `<first use>.$use`.
- **Singleton structural choices** now persist their selector key (a one-value
  Decision).
- **Exposed nested root parameters** such as `first.depth` no longer exist:
  formals are declared on the root family.
- **`Located.node` for a choice's member** is the selected candidate's path
  (`heating.boiler`), not the choice name.

## 9. Evidence

All runs are on `spike/space-declarative`. Transcripts are under
[`evidence/`](evidence/).

| Evidence | Result |
|---|---|
| Gate `check-kernels.sh` | Space **351 passed** (base 314), kernels **749 passed, 9 skipped** (base 758 passed), format, lint and strict mypy clean. See the note below |
| Gate `check-dataflow-design.sh` | **16 passed**, clean |
| House toy | `tests/core/space/test_house.py`: 5 tests |
| Generic acid tests | `tests/core/space/test_design_graph.py`: 14 tests. Relation node, quantification, pipeline built as data (`composite`), `Present`, multiple suppliers, anchored and unanchored cycles, closure, reducibility (Decision over nodes vs `Decision` + guarded nodes + `Present`, with identical assertions), candidates as nodes, located own formal, open-formal rules |
| Misuse | `tests/core/space/test_reference_misuse.py`: 28 tests, one per class of misuse, plus class-body, key and repr checks |
| Typing | `typing/positive.py` clean under `--strict`; `typing/negative.py.txt` gives exactly its 24 expected errors |
| MVAU fingerprints | [`evidence/fingerprints.txt`](evidence/fingerprints.txt): external, fifo-external, padded-output and pumped-dsp58 are identical to `0d700b1ab`. cyclic-block and fifo-cyclic differ. [`fingerprints_renamed.py`](fingerprints_renamed.py) renames `u_implementation_cyclic` to `u_weights`, and then **all six are bit-identical to `0d700b1ab`** ([`evidence/fingerprints-renamed.txt`](evidence/fingerprints-renamed.txt)) |
| Scale probe | [`scale_probe.py`](scale_probe.py), the previous spike's probe ported to `composite`; [`evidence/scale-probe.txt`](evidence/scale-probe.txt) |

**Gate note.** The instructions say not to run Vivado or XSim, but
`check-kernels.sh` runs XSim-backed tests whenever the Xilinx tools are on
`PATH`. The recorded gate ran with Xilinx removed from `PATH`, so those 9 tests
skip. With them, the count is the base's 758. They do not touch the Space
layer: they are the native stream, FIFO and flat-kernel RTL checks, and their
Python-side configurations run in the same files.

**The scale probe did not move materially.** In ms:

| | prepare (prev spike) | prepare (now) | full read | read after one local edit | callbacks re-run |
|---|---|---|---|---|---|
| N=50 | 41 | 43 | 18 | 17 | 50/50 |
| N=200 | 150 | 157 | 70 | 67 | 200/200 |
| N=800 | 584 | 619 | 276 | 287 | 800/800 |

Preparation is about 5% slower, because it includes the node call and
placement plans, and family formals are cached per link. Reads are unchanged.
One local edit still re-runs the whole graph. Cross-snapshot cache reuse is a
separate decision and was not attempted.

## 10. Open questions for human review

1. **`OPEN`, or untyped open formals?** Keep the explicit `OPEN` for required
   formals supplied by a later `Bind`? The alternative is to let `Room()` pass
   and give up "a missing required formal is a mypy error".
2. **Annotated formals.** Accept `area: Param[int] = Param(int)` as the price of
   typed calls? Or infer the value type from the annotation alone
   (`area: Param[int] = Param()`, SQLAlchemy style), which halves the
   repetition but moves semantics into annotations?
3. **Choice typing.** Is `Decision[A | B](values=...)` plus candidate handles
   acceptable? Or should a choice's class-body type deliberately drop `None`
   (unsound in methods) so that `implementation.output` type-checks?
4. **`decision.member` semantics.** `select` gives `Inapplicable` when the
   selected candidate lacks the member. Keep it, or make it
   `Present`-unsupplied as the design sketched? And should unreferenced
   single-candidate names stay unlinked (today they resolve to the candidate's
   own member)?
5. **Ownership inference scope.** Fresh decisions are shared within one family
   body and keyed by their first use. Is first-use keying stable enough for
   persisted selections, given that reordering declarations changes it? Should
   a shared fresh decision require a name instead?

Further questions: should unnamed *nodes* be adopted by the lowest composite
using them (rejected here)? Does `inspection.declaration` belong on the node
object itself (member-name collisions argue no)? Should the kernels gate run
its XSim tests in a separate target?
