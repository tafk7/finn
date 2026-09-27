# Declarative design spaces: nodes, references, one compile step

Date: 2026-09-26. Branch: `spike/space-declarative-2` (iteration 2), on
`spike/space-declarative` at `35e59b442` (iteration 1, itself on
`spike/space-design-graph` at `0acc51e1e`). Status: **design spike, at the
human review gate.** Not for merge as is.

This spike replaces the composition layer of `finn.core.space` with a
declarative graph of design spaces. The value graph and runtime stay as they
are; only the authoring and linking layer changes. The engine has no domain
notions: no ports, directions or carried values. Streams, their direction and
their boundaries live in `finn.kernels.streams`.

This document describes the **current** model. §1 lists what iteration 2
changed after the human review; iteration 1 text is kept only where it still
applies (git has the rest).

## 0. Summary

| Question | Answer |
|---|---|
| What is a node? | Calling a family, `Room(area=12)` or bare `Room()`, returns a **node declaration**: a declaration-mode `Room` object holding its bindings. Nothing is compiled. |
| What is an edge? | A binding at the call (`Room(area=kitchen.area)`), or an **assignment** after the call (`hall.area = kitchen.area`). |
| What is a reference? | Attribute access on a node declaration: `kitchen.finish`. `House.kitchen.finish` from class level is the same reference. |
| How does one node use another? | A **reference input**, `output: Param[Stream] = Param(Stream)`. A fresh node supplied to it is placed there; a node placed beside it is referenced. Several nodes may reference one node. |
| How does a node see who uses it? | `Users(key)`, the mirror of `Members(key)`: every present node whose input references it, located by name and input. |
| What is a structural choice? | `Decision(values={"boiler": Boiler(), "heat_pump": heat_pump, "none": None})`. |
| How is a configuration made? | Only by `configure(House(budget=100))`, typed `House`. Preparing a model freezes the declarations in it. |
| Typing | Option A. Nodes are typed as their family, references as their values, `self` in methods as a configuration. Every formal is optional at the type level; each keyword and each assignment is typed by `Param.__set__`. |
| Engine verdict | The runtime (scheduler, snapshots, admission, selections, codecs) is unchanged except that a `members` node now carries one member name per entry. The IR gained `Scope.references` in iteration 2. |
| Evidence | Both gates green with XSim executed: Space 363 passed, kernels 761 passed (0 skipped), dataflow 16 passed. The 6 MVAU fingerprints are identical to iteration 1, and with the cyclic instance renamed they are bit-identical to `0d700b1ab`. No MVAU decision key and no top-level ABI port name changed. |

**Flagged deviations and additions.** Each is the closest workable form of
the decided design; §4 and §6 give the reasons.

1. **Assignment through a path is an addition.** `kernel.port.dtype = dtype`
   supplies a formal of a node *inside* `kernel`, for this placement of
   `kernel` only. It keeps what `Bind` could do (reach an unsupplied formal
   of a reusable family's internal node) without mutating the shared
   class-body declaration (§4.6).
2. **Reference inputs are typed only in methods.** `output: Param[Stream]`
   (the annotation the review asked for) makes `self.output.spec` exact in a
   method, but `output.spec` in the class body is not typed (`Param[Stream]`
   has no attribute `spec`); it works at runtime (§5).
3. **A kernel's `PORTS` export is one value.** A stream reads its users'
   whole port records, so a refusal of one of a kernel's ports reaches every
   stream the kernel sits on (§6, R16). The only visible case is dotp's
   element-type refusal, which previously reached the weight stream alone.
4. **A boundary stream without a `port` is unresolved, not refused**: its
   `port` input is unsupplied (owner `x.port`), which says what to supply.
5. **The formal of a reference input has a node**: a presence node (`k.output`,
   a constant guarded by the referenced node's guard). A query of it is typed
   as the family but answers `True` (§6, R19).

## 1. Iteration 2 changes

Each decision from the human review, and what implements it.

| # | Decision | Implemented as |
|---|---|---|
| 1 | Bare `Room()`; remove `OPEN` | `OPEN` is gone. A node call may leave any formal unsupplied. A required formal that nothing supplies is a `DefinitionError` when a family containing the node is prepared, naming the formal's declaration line and the node's call line. `Param` is no longer a `dataclass_transform` field specifier, so every formal is optional at the type level (§5). |
| 2 | Kept: annotated formals, candidate handles, `None` in a choice's union, `decision.member` reads `Inapplicable` | Unchanged. |
| 3 | Shared decisions must be named | An unnamed Decision supplies one formal and is keyed by it (`second.depth`). Used at two or more sites it is a `DefinitionError` naming every site and saying how to name it. `Decision(..., name="lanes")` is owned by the lowest common scope of its uses and keyed `<owner>.<name>`. First-use keying is removed (§4.4). |
| 4 | Assignment replaces `Bind` | `Bind` is gone. `hall.area = kitchen.area`, `adder.back = register.q` (class body) and `current.width_in = previous.width_out` (a loop over plain nodes) all supply a formal. Assigning a supplied formal, or a formal of a frozen declaration, is refused. Alternatives are `x.formal = Present(a.out, b.out)`. `Param.__set__` types the value (§4.5). |
| 5 | Reference-valued inputs | `Param(Family)` returns a reference input typed `Param[Family]`. A fresh (unplaced) node is placed at it; a node placed beside it is referenced; several nodes may reference one; place-once holds. In methods `self.output` is the referenced configuration. A reference resolves in the body that wrote it: a sibling, a candidate, or a forwarded input of that body (§4.7). |
| 6 | `Users(key)` | A `members` node over the exports of the nodes whose inputs reference this one: `Located(node, member=<input name>, value)`, in declaration order, absent users omitted, obliged per user (§2.3). |
| 7 | Streams are Spaces referenced by kernels | `Stream`, `BufferedStream`, `Port`/`Ports`/`PORTS`, `Flow`. Kernels (dotp, replay, cyclic delivery) have stream inputs and a `ports` export. MVAU and `Constants` are rewritten. Boundary names come from the stream's `port` input (§4.8). The anchoring rule is demonstrated (§2.4). |
| 8 | Keep everything else | `Members`, `Present`, `Located` auto-location, `requires=`, `configure`/`commit`, `ReferenceUseError`, source locations, place-once and pre-compile inspection all stay. |

## 2. The model

A **family** is a `Space` subclass. Its class body declares members and nodes:

```python
class House(Space):
    budget: Param[int] = Param(int)                    # formal (annotated: typed call)
    want_garage = Decision(bool, values=(False, True))
    hall = Room()                                      # node; area assigned below
    kitchen = Room(area=12)                            # node
    dining = Room(area=16)
    garage = Room(area=20, when=want_garage)           # guarded node
    heat_pump = HeatPump(kw=8)                         # a handle naming a candidate
    heating = Decision[Boiler | HeatPump](values={"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    thermostat = Thermostat(kw=heating.kw)             # the selected candidate's kw
    hall.area = kitchen.area                           # an edge after its nodes
    matched = Match(a=kitchen.finish, b=dining.finish) # Match.a: LocatedParam[int]
    costs = Members(COST)                              # present children exporting COST

    @view(requires=(costs, within_budget, matched.agreed))
    def total(self) -> int: ...

house = configure(House(budget=100))
point = house.with_choices({House.heating: "heat_pump", House.heat_pump.cop: 3,
                            House.kitchen.finish: 2})
```

The complete toy is [`tests/core/space/test_house.py`](../../tests/core/space/test_house.py).
Reference inputs and `Users`, in a toy with nothing about hardware
([`tests/core/space/test_references.py`](../../tests/core/space/test_references.py)):

```python
class Budget(Space):
    limit: Param[int] = Param(int)
    rate: Param[int] = Param(int)
    claims = Users(SPEND)                  # every present node that references this budget

    @constraint
    def covered(self) -> bool | Rejected: ...   # refuses as "sales.budget=30, research.budget=20 exceed 40"

class Department(Space):
    budget: Param[Budget] = Param(Budget)  # a reference input
    staff = Decision(int, values=(1, 2, 3))

    @view
    def spend(self) -> int:
        return self.staff * self.budget.rate    # the referenced node's configuration

    exports = {SPEND: spend}

class Company(Space):
    open_lab = Decision(bool, values=(False, True))
    shared = Budget(limit=100, rate=10)
    sales = Department(budget=shared)      # placed beside it: a reference
    research = Department()
    research.budget = shared               # assigned like any formal
    lab = Department(budget=shared, when=open_lab)  # absent: not a user
```

### 2.1 Concepts and their lowering

| Concept | Written | Lowers to (evaluation graph) |
|---|---|---|
| Node declaration | `Room(area=12)`, placed by a class attribute | one scope; its members become nodes keyed `kitchen.area`, ... |
| Formal | `area: Param[int] = Param(int)` | `param` at the root (a runtime input); in a child: `const` (literal), `alias` (reference), `decision` (fresh Decision) |
| Unsupplied formal | left out of the call, never assigned | optional: `const` of the default, or `present` with no alternatives (unsupplied); required: a definition error when preparing |
| Assignment | `hall.area = kitchen.area` | exactly a binding at the call: the formal's node is an `alias` |
| Assignment through a path | `kernel.port.dtype = dtype` | a binding of the node `kernel.port` in this placement of `kernel`; its supplier is read in the body that declared `kernel` |
| Reference | `kitchen.finish`, `House.kitchen.finish`, `h.kitchen.finish` | resolved by node identity through `Scope.children` (and `Scope.references`) to the member's node |
| Reference input, fresh node | `Estate(home=House(budget=1))` | the node's scope placed at `home`; a presence node `home` |
| Reference input, placed node | `Department(budget=shared)` | `Scope.references[formal] = <shared's scope>`; a presence node `sales.budget` (a `const` guarded by `shared`'s guard) |
| Forwarded input | `team = Department(budget=budget)` inside a family with `budget: Param[Budget]` | the same scope reference as the family's own input |
| Users | `claims = Users(SPEND)` | a `members` node whose alternatives are the users' `SPEND` views, with one member name (the user's input) per entry |
| Guard | `Room(..., when=c)` | a `guard` node on the scope |
| Decision over nodes | `Decision(values={"a": A(), "b": B(), "n": None})` | a `decision` node keyed `heating` (str domain = keys), one scope per candidate keyed `heating.a`, guarded by a `$selected` node |
| Member through a choice | `heating.kw` | a `select` node `heating.$member.kw` keyed by the selector |
| Selected key | `selected(heating)` | an `alias` of the selector (read-only) |
| Candidate handle | `cyclic = C(...)` then `values={"cyclic": cyclic}` | nothing: the candidate's scope is reachable by node identity |
| Whichever is present | `Present(a.out, b.out)` | a `present` node |
| Located formal | `a: LocatedParam[int] = Param(Located)`; `Match(a=kitchen.finish)` | a `locate` node; value `(node name relative to the reading scope, member name)` |
| Quantification | `Members(COST)` | a `members` node over the children's exported views |
| Obligation | `@view(requires=(costs, within, matched.agreed))` | view constraints; `Members` and `Users` oblige each entry |
| Unnamed Decision | `Fifo(depth=Decision(int, values=(4, 8)))` | a `decision` at `second.depth` (one use only) |
| Named shared Decision | `Decision(int, values=(1, 2), name="lanes")` at several calls | one `decision` keyed `<lowest common scope>.lanes`; each use an editable `alias` |
| Graph as data | `composite("Pipeline", {"s0": s0, ...})` after assigning in a loop | a family, collected like a class body |

The compile step: `configure(node)` checks that `node` is an unplaced
declaration (a root) and that every required formal of the root is supplied.
If every root binding is a plain value, it reuses the family's model (compiled
once, cached on the family) and binds the values as runtime inputs. If a root
binding supplies structure (a node, a reference, a fresh Decision, or a
binding through a path), the model is compiled for that declaration and
cached on it. Preparation then freezes the declarations (§4.5).

### 2.2 Reference inputs

`Param(Stream)` returns, at runtime, a declaration-mode `Stream` object whose
record is a `FamilyFormal`; it is typed `Param[Stream]`. Its supplier is a
node declaration. Whether that node is placed *there* or *referenced* cannot
be decided at the call: in `edge = Stream(); k = K(output=edge)` the stream is
not yet placed when `K(...)` runs (class-body placement happens in
`__set_name__`, after the body). The linker decides:

- **fresh** (no class attribute, candidate or composite member placed it): the
  node is placed at the input, nested, keyed below the node that placed it
  (`k.output.spec`). A fresh node supplied to two inputs is refused ("placed
  by none: place it so that each input references it").
- **placed** (a class attribute or a candidate): a reference. The linker
  resolves it after every scope is allocated, in scope order, in the body that
  wrote it (§4.7).

Either way the input gets a **presence node** keyed like the input
(`compute.weights_stream`): a constant guarded by the reached node's guard. An
unsupplied optional input (`Param(Stream, default=UNSUPPLIED)`) is a `present`
node with no alternatives. Reading `self.output` in a method attaches the
referenced scope; every read through it is inapplicable when that node is
absent, exactly as for a guarded child. An unsupplied optional input reads its
presence first, which halts the method as unsupplied.

### 2.3 `Users(key)`

`Users(key)` is declared in the node that is referenced. It lists every
present node whose input references this node directly: one entry per
(user, input) pair, `Located(node=<user's name beside this node>,
member=<the user's input name>, value=<the user's export of key>)`, in
declaration order (scope order, then the user's formal order). Users that do
not export `key` are omitted at link time; absent users (guarded off, an
inactive candidate) are omitted at run time. As an obligation, each user's
export is obliged separately; a user referencing the node through two inputs
is obliged once. The engine attaches no meaning to the entries: direction,
roles and cardinality are the reading family's business.

A **user** is the node whose input names this node. When a composite forwards
its own input to a child (`Division(budget=shared)` whose `team =
Department(budget=budget)`), the user of `shared` is `division`, not
`division.team`: a composite is a node like any other, and presents its
children's exports itself if it wants to be seen (closure, acid test 6). A
fresh node placed at an input has its placing node as user, named `None`
(the placing node is its parent).

### 2.4 Streams (finn.kernels.streams)

```python
class Stream(Space):                         # a relation between its users
    spec: Param[StreamSpec] = Param(STREAM_SPEC)
    port: Param[str] = Param(str, default=UNSUPPLIED)   # the ABI name, if a boundary
    ends = Users(PORTS)                      # each kernel exports its ports keyed by input name

    @derived(semantics=ENDPOINTS)
    def endpoints(self) -> Endpoints | Rejected: ...    # one producer "out", one consumer "in"
    @constraint
    def compatible(self) -> bool | Rejected: ...        # contracts (and a FIFO stage) agree
    connection = View(link, requires=(compatible,))
    exports = {CONNECTION: connection}

class BufferedStream(Stream):
    transport = Decision[_Direct | StreamFifo](
        values={"direct": _Direct(), "fifo": StreamFifo(spec=Stream.spec)})
    stage = View(transport.stage)
```

A kernel has one reference input per stream it sits on and exports
`PORTS: ports`, a `Ports` record keyed by those input names; each `Port` is
`Port(Flow.OUT | Flow.IN, contract)`. The stream classifies its users by flow,
refuses a second producer or consumer (`stream-users`) and an unused stream
(`stream-unused`), and turns a missing side into the composite's boundary:
`boundary_contract(self.port, self.spec, ...)`, the AXIS target on the input
side and the initiator on the output side. `netlist` is unchanged.

MVAU now reads:

```python
    activations = Stream(spec=activation_spec, port="in0_V")
    replayed = Stream(spec=replayed_spec)
    weight_stream = BufferedStream(spec=weight_spec, port="in1_V")
    results = Stream(spec=result_spec, port="out0_V")

    replay = ReplayBuffer(input_stream=activations, output_stream=replayed,
                          sequence_length=synapse_folds, replay_count=neuron_folds)
    compute = DotpAxiKernel(..., activation_stream=replayed,
                            weights_stream=weight_stream, result_stream=results)
    cyclic = CyclicDelivery(dtype=weights_dtype, form=weight_period, values=weights,
                            output_stream=weight_stream)
    implementation = Decision(values={"external": None, "cyclic": cyclic})
    delivery = selected(implementation)
    modules = Members(MODULE)
    streams = Members(CONNECTION)
```

In the external case `weight_stream` has one user (`compute`, a consumer), so
it is the boundary `in1_V`. In the cyclic case the `cyclic` candidate
references it as its producer and it is internal. The old `in0_V`/`in1_V`/
`out0_V` views, the `external` derivation and the `Present(in1_V,
cyclic.output)` are gone: "whichever driver is present" is now what `Users`
sees.

**The anchoring rule.** A stream's `spec` must not depend on its users: kernels
read it to build their port contracts (dotp's `_port` reads
`self.weights_stream.spec`). `tests/kernels/test_declared_streams.py` declares
a stream that derives its spec from its producer's contract, with a producer
whose contract reads the spec. The engine refuses it at evaluation with the
cycle path:

```
edge.spec (dependency cycle): edge.spec (derived) -> edge.ends (value)
  -> producer.ports (value) -> producer.ports (derived) -> edge.spec
```

The replay buffer shows the anchored alternative: its output contract derives
from its *input* stream's spec, not from `replayed.spec`, so nothing cycles.

## 3. Public API

- **Families and nodes:** `Space`, a family call `F(**formals, when=...)` (any
  formal may be left out), assignment `node.formal = value`, `configure(node)
  -> F`, `composite(name, members, base=, exports=)`.
- **Members:** `Param` (plus `LocatedParam` via `Param(Located)`, and a
  reference input via `Param(Family)`, optional with `default=UNSUPPLIED`),
  `UNSUPPLIED`, `Const`, `Decision` (over values, with `name=` for a shared
  one, or over nodes via `values={key: node | None}`), `selected(decision)`,
  `Derived`/`@derived`, `Constraint`/`@constraint`, `ConstraintGroup`,
  `View`/`@view(requires=...)`, `ViewKey` + `exports`.
- **Graph primitives:** `Present(*refs)`, `Members(key)`, `Users(key)`, `Located`.
- **Configurations** (unchanged): reads, `query`, `inspect`, `view`, `field`,
  `root`, `with_choices` / `try_with_choices` taking `{reference: value}`
  mappings, `Change` objects, or keywords for the point's own decisions.
- **Inspection:** as in iteration 1, and `inspection.declaration(node)` now
  reports `bindings`, `nested` (formals assigned through a path, keyed
  `"port.dtype"`), `unsupplied` (required formals not yet supplied) and
  `frozen` (why assignment is closed), all before compiling.
- **Errors:** `ReferenceUseError` (a `TypeError`) on value-like use of a
  reference; `DefinitionError` for a formal nothing supplies, a formal
  supplied twice, an assignment after freezing, an unnamed shared decision,
  and an unresolvable reference.
- **Kernels:** `finn.kernels.streams`: `Stream`, `BufferedStream`, `Port`,
  `Ports`, `PORTS`, `Flow`, `produces`/`consumes`, `Endpoints`,
  `CONNECTION`, `MODULE`, `netlist`. `commit(point, {key: value})` as before.

## 4. Decisions, with reasons and rejected alternatives

### 4.1 The compile step is `configure` (iteration 1, unchanged)

The result is a *configuration*, and the codebase already says so
(`ConfigurationResult`, `ConfigurationError`). `compile` shadows a builtin,
`space.compile` forces a qualified import, `instantiate` collides with calling
a family, `realize` is vague. `compile_space(family)` survives as the
internal, cached template compiler.

### 4.2 `Members(ViewKey)` over structural alternatives (iteration 1, unchanged)

Matching by member name cannot be typed (Python cannot project an attribute
type out of a Protocol) and matches by accident; matching by a declaration
forces a common base. A `ViewKey` export is an explicit, typed opt-in,
independent of the family hierarchy. `Users` reuses it for the same reasons:
`Users(PORTS)` sees only users that opted into `PORTS`.

### 4.3 `requires=` instead of `constraints=` (iteration 1, unchanged)

The list holds constraints, groups, views, references to child views,
`Members` and now `Users`.

### 4.4 Shared decisions are named

A Decision that no class attribute names is *unnamed*. Every place it
supplies a formal (a call keyword, an assignment, a path assignment) is
recorded on the Decision as a site when the call or assignment runs.

- **One site:** it belongs to the node whose formal it supplies, keyed by that
  formal's path (`second.depth`). A family placed twice gets two decisions,
  as for member decisions (`StreamFifo`'s inline depth is per placement).
- **Two or more sites:** a `DefinitionError` when a model using it is
  prepared, listing the sites: *"Decision (declared at probe.py:6) supplies 2
  formals (Port.lanes (declared at probe.py:8), Port.lanes (declared at
  probe.py:9)): a shared decision must be named. Make it a class attribute, or
  pass Decision(..., name="..."); while configuring Shared node (declared at
  probe.py:10)"*. The check uses the recorded sites, so it also catches one
  object used in two different family bodies.
- **Named by a class attribute:** owned by that family's scope, as always; its
  uses are plain aliases.
- **Named with `name=` (created outside a class body):** the ownership rule of
  iteration 1 is kept. It is one decision owned by the **lowest common scope
  of the nodes it supplies** (per instance of the authoring body), keyed
  `<owner>.<name>`. It applies whenever its owner does; each use is an alias
  that keeps its own node's guard and through which the decision may be
  edited. A name that collides with a member of the owner is refused. A
  single-use named decision is owned by the node it supplies
  (`s.<name>`). Giving `name=` to a class attribute with a different
  attribute name is refused.

**Rejected: first-use keying** (iteration 1). Reordering declarations changed
a persisted key. **Rejected: owning named decisions by the authoring body**
rather than the lowest common scope: simpler, but the review asked to keep
the rule, and it gives the tighter applicability (a decision used only inside
`x` applies with `x`).

### 4.5 Assignment, and how freezing happens

`node.formal = value` goes through `Space.__setattr__` (hidden from mypy, see
§5) to the same validation as a call keyword: unknown formals, literal
recognition and snapshot, reference semantics, and node family checks happen
at the assignment; everything that needs the graph happens at link. A binding
records where it was written, so errors name both sites:

```
<Sink node (declared at test_design_graph.py:450)>.width (assigned at
test_design_graph.py:451): the formal is already supplied at
test_design_graph.py:450; a formal has one supplier (write alternatives as
Present(a, b))
```

A formal has **one supplier**. Several suppliers are `Present(a.out, b.out)`.

**Freezing.** A declaration is mutable until one of two things happens:

1. a model containing it is prepared: after a successful link, `_publish`
   freezes every node declaration instantiated as a scope in that model
   (class-body nodes of every family involved, candidates, nested nodes, and
   the root record when it is compiled per declaration);
2. `configure()` takes it as a root (also when it reuses the family's model).

An assignment checks the node that holds the binding (the path's first node)
and is refused with the reason: *"... is frozen (Parent was prepared); a
declaration can be assigned only until its family is prepared or configure()
takes it"*. A failed preparation freezes nothing, so a definition error can be
fixed by assignment and retried.

The reason is the model cache: `compile_space` caches one model per family
and `configure` one per structural root, and selections replay by model
identity. An assignment after preparation would otherwise be silently
invisible to the cached model, or produce two models of one declaration.

- **Rejected: freeze at class creation.** Nothing is cached before
  preparation, so there is nothing to protect yet; and it forbids joining a
  graph built as data after `composite()` has named its nodes.
- **Rejected: invalidate and relink on mutation.** Configurations and
  selections already handed out would describe a declaration that no longer
  exists.
- **Rejected: snapshot at preparation, keep mutating.** A later assignment
  would be accepted and ignored: a silent trap.

### 4.6 Assignment through a path (addition)

`Reusable` places `port = Port()` and leaves its formals to whoever places a
`Reusable`. With `Bind` gone, the enclosing family writes `kernel.port.dtype =
dtype`. Setting the binding on the `port` declaration itself would change
*every* `Reusable`, so the binding is stored on the path's first node
(`kernel`), keyed by the rest of the path, and applies to that placement
only. Its supplier is read in the body that declared `kernel`. Two bodies
supplying the same formal through different paths meet only at link, where
it is the same "already supplied" definition error. Paths through a reference
input or a Decision are refused: assign the node itself.

- **Rejected: refuse path assignment** and require re-exposure by formals.
  That works (`test_reexposed_nested_slot...` shows both), but loses a form
  the review did not ask to remove.

### 4.7 Reference visibility: where a reference input may point

**Decided: a reference resolves lexically, in the body that wrote it.** It may
name a node placed in that body (a sibling class attribute, a candidate or a
candidate handle, a composite member), or forward one of that body's own
reference inputs. It may not name a path into another node
(`K(input=sub.inner)`), and it cannot name an ancestor's node except through a
forwarded input.

- **Hermeticity.** A family's meaning depends only on its formals. An upward
  reference would make a family's graph depend on where it is placed; the
  explicit way to reach an ancestor's node is to declare an input and have
  the parent pass it down.
- **One resolution rule.** It is the rule of value references (`kitchen.area`
  resolves in the writing body by node identity), and place-once gives node
  identity equal to placement identity.
- **Users stay nameable.** Every user is then a descendant of the referenced
  node's parent, so `Users` names it relative to that body (`"compute"`,
  `"implementation.cyclic"`), the same frame as `Members`.
- **Rejected: anything visible by path** (siblings' descendants). A user would
  sit outside the referenced node's parent, `Users` would need upward names,
  and `sub`'s internals would become part of its interface.
- **Rejected: siblings only.** Forwarding is what makes a composite a node
  like any other: `Division(budget=shared)` passes `shared` on to its team.

The error for a node placed elsewhere: *"team.budget: references Budget node
placed at Other.held (declared at ...), which is not placed in <root>: a
reference input names a node placed beside it, or forwards an input of the
enclosing family"*.

### 4.8 Boundary streams are named by a `port` input

**Decided: `Stream.port: Param[str] = Param(str, default=UNSUPPLIED)`**, the
AXIS name presented when the stream is a boundary. MVAU:
`weight_stream = BufferedStream(spec=weight_spec, port="in1_V")`.

- A node name is identity: it prefixes persisted keys
  (`weight_stream.transport`, `weight_stream.transport.fifo.buffer.depth`)
  and instance names (`u_weight_stream_fifo`). An ABI name is an external
  contract. They vary independently: `weight_stream` is `in1_V` only in the
  external case.
- An ABI name is legitimate design data of the stream, like its spec. It is
  supplied where the stream is declared, next to the spec.
- **Rejected: name the node by its port** (`in1_V = BufferedStream(...)`).
  It renames the persisted FIFO and transport keys to `in1_V.transport...`
  and gives an internal stream (the cyclic case) an ABI name as identity.
- **Rejected: derive it from the user's input** (`weights_stream`). A
  kernel's internal names would leak into the composite's ABI, and two
  boundaries could collide.
- **Rejected: a composite-level map** `{stream: name}`: a second place to keep
  in sync with the streams.

Direction is a `Flow` in the kernel's `Port` record, stated by the kernel.
Deriving it from the transport's endpoint (initiator = produces) would work
here, but conflates a signal role with a stream relation.

### 4.9 What happens to `Present`

`Present` is now the **only** way to write alternative suppliers
(`sink.width = Present(a.out, b.out)`). Several `Bind`s to one formal used to
be an implicit `Present`; several assignments are an error. `Present` stays
usable wherever a value reference is (a call keyword, an assignment, a
`View`'s source) and keeps its semantics: unresolved while any source is,
refused if two are present, unsupplied if none is. In MVAU it disappeared:
which weight driver is present is now a question the stream answers through
`Users`, structurally, rather than a value-level choice between two contract
references.

### 4.10 The place-once rule (iteration 1, extended)

A node declaration is placed exactly once: by a class attribute, as a
Decision candidate, or (only when nothing else places it) at the one reference
input it is supplied to. A class attribute naming a candidate of a Decision in
the same body is a handle, not a placement. A second placement is refused and
names both sites. A node supplied to a reference input *and* placed elsewhere
is referenced, not placed twice. `configure()` refuses a node that is placed
or supplied to an input.

### 4.11 Smaller decisions

- **Literals are frozen once, at the call or assignment.** An unrecognized
  literal is a `DefinitionError` there.
- **Where errors appear.** Unknown formals, positional arguments, bad
  literals and wrong node families: at the call or assignment. Missing
  formals, double supply across bodies, unresolvable references and unnamed
  shared decisions: when preparing.
- **The root's required formals** are checked by `configure` before
  compiling: *"width is not supplied: Sink.width (declared at x.py:3) is
  required; supply it at the call (width=...) or assign it before configure();
  while configuring Sink node (declared at x.py:10)"*.
- **`decision.member`** (iteration 1): a name several candidates share is
  linked as a `select` node; one candidate's name resolves to that
  candidate's own member.

## 5. Typing results

The mechanism: `SpaceMeta` is decorated with `dataclass_transform(kw_only_default=True,
field_specifiers=(when-field,))`, and **formals are annotated**
(`area: Param[int] = Param(int)`). `Param` is deliberately *not* a field
specifier any more, so every annotated formal has a default at the type level
and may be left out of a call. The type of each keyword, and of each
assignment, is `Param.__set__`'s value type: `T | ValueRef[T] | View[T] |
BoundView[T]` (plus `Located[T]` for a `LocatedParam[T]`). `Space.__setattr__`
exists only at runtime (`if not TYPE_CHECKING`): a declared `__setattr__`
would make mypy accept assignment to *any* attribute.

Evidence: [`tests/core/space/typing/positive.py`](../../tests/core/space/typing/positive.py),
`negative.py.txt` (28 expected errors, matched line for line),
`extensions.py` / `extensions_negative.py.txt` (16), and the fixture in
`test_nested_parameter_bindings.py`, all under `mypy --strict`.

| Claim | mypy result |
|---|---|
| A bare `Room()` type-checks | **exact** (`configure(Room())` is `Room`). Accepted loss: a missing required formal is no longer a mypy error |
| `hall.area = kitchen.area` in a class body, and `current.width_in = previous.width_out` in a loop | **checked** by `Param.__set__`: `bare.width = "four"` is an error |
| Assignment to something that is not a formal | **error** (`"Child" has no attribute "depth"`), because `__setattr__` is hidden |
| Reference input keyword and assignment | **exact**: `Producer(output=child)` and `later.output = child` with a non-`Stream` are errors |
| `self.output` / `self.output.spec` in a method | **exact** (`Stream`, `int`); `return self.output.spec` from a `str` method is an error |
| `output.spec` in a class body | **not typed**: `output` is `Param[Stream]`. It works at runtime (a `MemberRef` through the input) |
| `Users(COST)` in a method | `tuple[Located[int], ...]`, exact |
| `Decision(int, values=(1, 2), name="shared")` | `Decision[int]`, exact |
| `kitchen.finish` / `kitchen.cost` in a class body, `self.kitchen.finish` | unchanged from iteration 1: `int`, `BoundView[int]`, `int` |
| `heating.kw` with candidates `Boiler \| HeatPump` | exact only with `Decision[Boiler \| HeatPump](...)` (a dict display is joined) |
| A `None` candidate | `N \| None` (kept, sound); class-body member access goes through a candidate handle |
| Edits `{House.kitchen.finish: 2}` | keys are `Any` in the mapping: no static check |
| `point.query(K.output)` (a reference input's presence) | typed `QueryResult[Stream]`, answers `Available(True)`: a typing lie (R19) |
| `Department(budget=holder.inner)` (a path into another node) | **not an error** (the path is typed `Budget`); refused at the call at runtime |

What mypy could not do: know that a later assignment supplies a formal (so
missing formals are a preparation error); infer a union from a dict display;
narrow `N | None` in a class body; tell a reference from its value (the
premise of option A); type a member of a reference input in a class body.

## 6. Resistance log: where the existing engine pushed back

| # | Resistance | Change | Reading |
|---|---|---|---|
| R1 | In a class body, sibling declarations have no owner or placement until `__set_name__` runs after the body; `Room(finish=member)` and `Room(finish=Decision(...))` look alike, and so do `K(output=edge)` with `edge` a sibling and with `edge` fresh | Classification (fresh decision, member, and now fresh node vs reference) happens at the link | Node calls store raw suppliers; the call checks names, literals and families, the link checks structure |
| R2 | `dataclass_transform` fields must be *annotated* | Formals annotated (`x: Param[int] = Param(int)`) | The one authoring cost of typed calls |
| R3 | A field specifier call without `default=` makes the field required, so a bare `Room()` was a mypy error (iteration 1 needed `OPEN`) | `Param` removed from `field_specifiers`: every formal has a default at the type level; `__set__` still types values | The type level cannot know about later assignments; the runtime reports the missing formal when preparing |
| R3b | A declared `__setattr__` makes mypy accept assignment to any attribute | `Space.__setattr__`/`__delattr__` defined under `if not TYPE_CHECKING` | Assignment is typed through the descriptors alone |
| R4 | Dict displays are joined; `None` makes the union Optional | `Decision[A \| B](...)`; candidate handles | Kept as decided |
| R5 | Every `Space` node is a descriptor, so class-level annotations and properties typed as a `Space` read through `__get__` | Instance-level attributes in records | Unchanged |
| R6 | Python 3.10 wraps `__set_name__` exceptions in `RuntimeError` | `composite` unwraps | Assignment errors in a class body are not wrapped: they are raised by the statement |
| R7 | The linked model is frozen, so `decision.member` cannot create a selection at query time | Shared names linked eagerly | Unchanged |
| R8 | `Node.scope` meant both "owning scope" and "whose member map holds it" | A named shared decision gets its own node in the owner; uses are editable aliases | Now only for `name=` decisions |
| R9–R13 | (iteration 1: constraint descriptors, mode detection, one model per family, dotted candidate names, selector identity) | Unchanged | |
| R14 | Assignment through a path would mutate a class-body declaration shared by every placement | Stored on the path's first node, merged at link per placement; double supply across bodies detected at link | Per-placement bindings are link-time data, like `Bind` was |
| R15 | A method halts only through a *blocked node read*; raising `ValueUnavailableError` manually is a programmer failure | Every reference input has a presence node; an unsupplied optional input reads it | A node reference needs a node in the value graph to be unsupplied |
| R16 | `Users` reads one export per user | A kernel exports all its ports as one `Ports` value | A refusal of one port reaches every stream of the kernel (dotp's element check); finer attribution needs per-input exports |
| R17 | A `members` node carried one member name (the key) | Its value is a tuple of member names, one per entry | The only runtime change; `Members` passes the key name for each entry |
| R18 | Model caches (per family, per root) assume immutable declarations | Freezing on preparation and on `configure` | Mutability is bounded by the cache, not by class creation |
| R19 | A reference input's presence node is found by the formal's declaration, which mypy types as `Param[Family]` | Accepted: `query(K.output)` is typed as the family but answers `True` | Presence could get its own typed accessor |
| R20 | A stream's `spec` read by its users must not be derived from them | Documented and tested (anchoring rule); detected at evaluation with the cycle path | Method reads are discovered at run time, so the static order cannot see this cycle |

Carried over and still open: the design-graph spike's R6 (a refusal that
reaches a view through both its output and an obligation is reported twice)
and R7 (whole-snapshot caches, §9).

## 7. What was removed

Iteration 2:

- `finn.core.space`: `OPEN`, `Bind`, the "missing formals" error at the call,
  first-use keying of shared unnamed decisions, `NodeDeclaration.open`.
- `finn.kernels.streams`: `StreamLink`, `BufferedStreamLink`, `_end`; located
  `source`/`sink` formals.
- `finn.kernels.mvau`: the boundary views `in0_V`, `in1_V`, `out0_V`, the
  `external` derivation, and the `Present` over the weight stream's drivers.
- `finn.kernels.dotp`, `streaming`: stream *spec* formals (now stream inputs).

Iteration 1 (still gone): `Subspace`, `SubspaceChoice`, `ScopeBuilder`,
`ValueKey`, `DecisionRef`, `AcceptedViewRef`, `ScopedValueRef`, `LocatedRef`,
`located()`, `ChoiceCaseRef`, `collect_placement`, `SpaceModel.bind`, the
top-level `compile_space`, calling a family to get a configuration, inline
`Param` suppliers, nested `bindings={...}`, `required=`, `constraints=`,
`External`, the `STAGE` key, and `configure(space_type, facts, choices)`.

## 8. Key and identity changes

- **MVAU decision keys: none changed.** `compute.compute_pumping`,
  `implementation`, `implementation.cyclic.rom_style`, `pe`, `simd`,
  `weight_stream.transport`, `weight_stream.transport.fifo.buffer.depth`,
  `weight_stream.transport.fifo.buffer.ram_style`: identical to iteration 1
  ([`evidence/mvau-keys.diff`](evidence/mvau-keys.diff), and
  `tests/kernels/test_mvau_delivery_choice.py`).
- **Top-level ABI port names: none changed.** `in0_V`, `in1_V` (external),
  `out0_V`, from the streams' `port` inputs.
- **Instance names: none changed.** `u_replay`, `u_compute`,
  `u_implementation_cyclic`, `u_weight_stream_fifo`, and connection names
  `activations`, `replayed`, `weight_stream`, `results`.
- **MVAU node keys (not persisted) that changed**, from the full diff in
  [`evidence/mvau-keys.diff`](evidence/mvau-keys.diff):
  - removed: `in0_V`, `in1_V`, `in1_V.$guard`, `out0_V`, `external`, and per
    stream `<s>.contracts`, `<s>.source`, `<s>.sink` and their `$located` /
    `$present` nodes;
  - added: per stream `<s>.ends` (`members`), `<s>.endpoints`, `<s>.port`;
    per kernel `ports` (and `$output`); presence nodes
    `replay.output_stream`, `implementation.cyclic.output_stream`;
  - changed kind: `compute.activation_stream`, `compute.weights_stream`,
    `compute.result_stream`, `replay.input_stream` from `alias` (a spec) to
    `const` (a presence).
- **Generic keys.** A named shared decision is keyed `<owner>.<name>` (was:
  its first use). An unnamed shared decision no longer links. Edge
  declarations (`hall_area`, `loop`, `e4`) no longer have nodes: the formal's
  own node is the alias.
- **Iteration 1 changes, still in force:** the cyclic candidate's node name is
  `implementation.cyclic` and its instance `u_implementation_cyclic`
  (`0d700b1ab`: `u_weights`); the selector node is the Decision itself;
  choice-member nodes are `<choice>.$member.<name>`.

## 9. Evidence

All runs are on `spike/space-declarative-2`. Transcripts are under
[`evidence/`](evidence/); `evidence/README.md` lists them.

| Evidence | Result |
|---|---|
| Gate `check-kernels.sh` (normal `PATH`, Xilinx 2025.2 `xelab`/`xsim` present) | Space **363 passed**; kernels **761 passed, 0 skipped**, including the 9 XSim tests; format, lint and strict mypy clean |
| XSim tests, run separately | **9 passed** ([`evidence/xsim-tests.txt`](evidence/xsim-tests.txt)) |
| Gate `check-dataflow-design.sh` | **16 passed**, clean |
| House toy | `tests/core/space/test_house.py`: 5 tests, bare `hall = Room()` + `hall.area = kitchen.area` |
| Generic acid tests | `tests/core/space/test_design_graph.py`: 14 tests (13 functions). Accumulator loop by `adder.back = register.q`; pipeline built as data by assignment in a loop; `Present` by assignment; closure; reducibility; supplied-formal and unsupplied-formal errors |
| References and users | `tests/core/space/test_references.py`: 9 tests. Shared `Budget` with `Users` refusing with located names; a guarded department dropping out; a choice candidate referencing the shared node; absent referenced node; fresh node placed at the input; forwarding; visibility, reach-into-a-node and placed-by-none errors |
| Assignment rules | `test_nested_parameter_bindings.py`: unnamed shared decision error, named shared decision (and its name checks), path assignment, double supply, assignment after freezing, `configure` freezes its root |
| Streams | `tests/kernels/test_declared_streams.py`: 8 tests. `Constants` in the stream form, two producers refused, a boundary without a port, the anchoring cycle |
| Typing | as §5, all under `--strict` (`test_typing.py`, `test_extensions.py`, `test_nested_parameter_bindings.py`) |
| MVAU fingerprints | [`evidence/fingerprints.txt`](evidence/fingerprints.txt): identical to iteration 1. `external`, `fifo-external`, `padded-output`, `pumped-dsp58` equal `0d700b1ab`; `cyclic-block` and `fifo-cyclic` differ only by the cyclic instance's name: [`fingerprints_renamed.py`](fingerprints_renamed.py) renames `u_implementation_cyclic` to `u_weights` before wiring, and then **all six are bit-identical to `0d700b1ab`** ([`evidence/fingerprints-renamed.txt`](evidence/fingerprints-renamed.txt)) |
| Scale probe | [`scale_probe.py`](scale_probe.py) on the assignment API; [`evidence/scale-probe.txt`](evidence/scale-probe.txt) |

**The scale probe got cheaper** (same session, same machine, in ms):

| | prepare (iteration 1) | prepare (now) | full read (it. 1) | full read (now) | read after one local edit (now) | callbacks re-run |
|---|---|---|---|---|---|---|
| N=50 | 42.5 | 31.6 | 18.0 | 15.5 | 15.3 | 50/50 |
| N=200 | 156.2 | 119.2 | 70.0 | 61.5 | 60.7 | 200/200 |
| N=800 | 616.9 | 462.5 | 277.9 | 243.2 | 235.1 | 800/800 |

An assigned edge is the formal's own `alias`; a `Bind` was an extra node, and
its open formal a `present` node over it. At N=800 the graph has 3200 nodes
and 1599 edges (iteration 1: 3999 and 2398). One local edit still re-runs the
whole graph: cross-snapshot cache reuse is a separate decision and was not
attempted.

## 10. Open questions for human review

1. **Path assignment.** Keep `kernel.port.dtype = x` as a per-placement
   binding (the reach `Bind` had), or require every nested formal to be
   re-exposed as a formal of the enclosing family?
2. **Users granularity.** One export per user makes a kernel's port refusal
   reach all its streams. Should `Users` project one entry of a keyed export
   (the user's own input), which needs a typed "export keyed by input" in the
   engine, or should kernels export a port refusal inside the record?
3. **Hierarchy of streams.** A composite that forwards a stream input to a
   child is the stream's user and must re-export its child's ports. Is that
   the closure we want, or should `Users` see through forwarding (transitive
   users, named from the stream's parent)?
4. **Class-body typing of reference inputs.** Accept `output.spec` untyped in
   a class body, or annotate reference inputs as `output: Stream =
   Param(Stream)` (typed in the body, but then the annotation is not a
   `Param`)?
5. **Ownership of named shared decisions.** Keep the lowest common scope
   (a single-use named decision is keyed `node.<name>`), or own every
   `name=` decision by the body that wrote it (`<body>.<name>`)?

Further questions: should a boundary stream without `port` be refused rather
than unresolved? Should a reference input's presence get a typed accessor
(R19)? Should a failed preparation freeze nothing (today) or freeze what
linked? Should the kernels gate run its XSim tests as a separate target?
