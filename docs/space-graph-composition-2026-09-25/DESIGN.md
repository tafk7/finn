# Graph composition for `finn.core.space`

Date: 2026-09-25/26. Base: `feature/kernel-package-extraction` at `0d700b1ab`.
Status: **superseded** by [`../space-design-graph-2026-09-26/DESIGN.md`](../space-design-graph-2026-09-26/DESIGN.md).
Review found the Port/Net/Interface/Fold model shaped by streams rather than by
design spaces. The record is kept for its evidence and rejected alternatives.
A working spike validates it; see §9. The spike is commit `759ee0e17` on the local branch
`spike/space-graph-composition`. It is not merged, and nothing is pushed.

## 0. Summary

A composite Space is a graph. The **nodes** are its existing placements
(`Subspace`) and structural choices (`SubspaceChoice`). **Ports** are a new
declaration kind: typed attachment points on a Space. **Nets** are hyperedges.
Each net is placed as a scope of a domain `Link` family, so it owns decisions,
constraints, views and findings like any child. A child attaches its port to a
net when it is placed. The composite attaches its *own* port to a net from
inside, which is how a child's endpoint becomes a boundary port (promotion).

Each port carries one value along its net. Some ends, or the parent, **publish**
that value; the others **adopt** it; the engine unifies and checks it. Each
end can also **offer** a facet, such as its contract, which the net's own
checks read. **Folds** apply a domain `Interpretation`, such as a netlist, to
the composite's active topology. The domain defines each interpretation once,
and the engine builds the topology and attributes every refusal.

Everything lowers onto the existing evaluation graph, adding four node kinds
and no new evaluation mechanism.

| Question | Decision |
|---|---|
| 1 Nodes | Placements and structural choices, unchanged. A choice is one node; each case binds its own ports; an inactive case's ports are absent ends |
| 2 Ports | `Port(interface, direction, carry=, offer=, net=, when=)`. Published with `carry=`, otherwise adopted. The domain type comes from an `Interface` (carried and offered semantics). An unconnected adopting port reads as `Unresolved` (`port-unconnected`) and needs no binding |
| 3 Edges | A net is a scope (`Net(LinkFamily)`) with decisions, constraints, views and attributed findings. Nets are hyperedges. Direction is per end, relative to the net. Role cardinality is the link's own constraint |
| 4 Promotion | `Port(..., net=inner_net)` attaches the composite's own port from inside, with its direction flipped. `TopInput`/`TopOutput` disappear. A conditional port uses `when=` |
| 5 Type flow | Anchors are `carry=` on the net plus publishing ends. One anchor: the others adopt. Several anchors: they must agree, or the carried value is refused (`net-disagree`). A family becomes a net-owned `Decision`. Evaluation is demand-driven through aliases |
| 6 Interpretations | `Interpretation(name, node=, net=, result=, reduce=)` is defined once per domain; `Fold(interp, **args)` places it. Every contribution is an obligation, so refusals are per node and per net, all at once. A reducer reads only the `Topology` and its bound arguments |
| 7 Names | Node, net and port names are declaration names; instance names derive from them. Selections and codecs are unchanged; net-owned decisions are ordinary decisions |
| 8 Conditional topology | `when=` on ports, nets and placements; choice cases; presence propagates to ends. Adapter insertion becomes a choice inside a link, as the FIFO transport is today |
| 9 Topology as data | `Net` and `Port` are ordinary declarations, so `ScopeBuilder.add` builds topology unchanged |
| 10 Repetition | Not in scope, and not precluded. A literal N works now through ScopeBuilder. A parametric N means allocating the maximum with `when=` guards |
| Cycles | Declared topology may be cyclic. The evaluation graph stays acyclic: a value cycle fails preparation (static) or evaluation (dynamic), both with scoped context. Domains restrict topology with ordinary constraints |

## 1. Why the previous attempts failed

The engine composed only as a tree:

- a `Subspace` is a product and a `SubspaceChoice` a sum;
- bindings flow down, and views flow up through accepted refs.

Each of the three earlier attempts rebuilt the graph outside the engine:

- the parked `NetworkEdge`/MRO walk;
- `assemble_streams`;
- `Stream` + `connected` + `compose` + `TopInput`/`TopOutput`.

Outside the engine they had no identity (literal instance names, `__set_name__`
injection), no attribution (one opaque callback, or a `return True`
constraint), no readiness (the hand-written `structure` view restates the
topology) and no conditional presence (`Module(None)` for a boundary case).
The design moves exactly those four responsibilities into the engine and
leaves every stream, AXIS, clock and RTL notion in the domain.

## 2. The model

### 2.1 Concepts

| Concept | What it is | Owner |
|---|---|---|
| `Interface[C, F]` | A port kind. `carries` gives the semantics of the value unified along a net; `offers` gives the optional semantics of a per-end facet | Domain (engine class) |
| `Port[C]` | A declaration on a Space. Reading it yields the carried value `C` | Engine |
| `Net[L]` | A `Subspace` of a `Link` family, placed in a composite. It is one scope with a key, guard, decisions, constraints and views | Engine placement, domain family |
| `Link` | Base family of net scopes. `Carried(interface)` reads the unified value and `Ends(interface)` the active ends | Engine base, domain subclass |
| `End[F]` | Detached record of one active end: `node` (child declaration name, or None for the composite's own port), `port`, `direction` relative to the net, `publishes`, `offer` | Engine value |
| anchor | The parent's `carry=` on the net, or a publishing end | Engine rule |
| `Interpretation[N, E, R]` | A fold: the node view key, the net view key, the result semantics, and a pure `reduce(topology=..., **args)` | Domain |
| `Fold[R]` | A `View` placing an interpretation in a composite, with bound arguments and optional extra obligations | Engine |
| `Topology[N, E]` | Detached active topology: `nodes` (name, contribution), `nets` (name, contribution, end refs) and `ports` (name, direction) | Engine value |

### 2.2 Semantics

**Attachment.** A child attaches its port by binding it to one of its parent's
nets at placement: `Subspace(Kernel, input_stream=net)`. The binding is a
keyword like a Param binding, but it is *optional*. A port with no binding is
unconnected, which is an ordinary state, not a definition error. The
composite attaches its own port from inside with `Port(..., net=net)`. A port
has at most one outside attachment and at most one inside attachment.

**Direction.** A port declares `"in"` or `"out"` as seen from outside its
Space. On a net, an end's direction is relative to the net: `out` drives it
and `in` reads it. A composite port's inside end has the opposite direction:
an input of the composite drives its inner net. The engine records direction
and imposes nothing else. Roles and cardinality (one producer and one
consumer for a stream; one driver and many loads for a clock) are the link
family's own constraints over `Ends`.

**Carried value.** Each net has one unified carried value. Its anchors, in
declaration order, are:

1. the parent's `carry=` on the net: a reference, a literal, or a fresh
   `Decision` that the net owns;
2. every end whose port publishes: a child port with `carry=`, or a composite
   child whose port exposes its inside (see below);
3. a relaying composite port (see below).

Inactive anchors are absent. With one active anchor, its value is the carried
value. With several, they must be equal under the interface's semantics, or
the carried value is `Rejected` with `net-disagree`, owned by the net. With
none active, it is `Rejected` with `net-unanchored`. An unresolved anchor makes
it `Unresolved`. A net that has no potential anchor at all fails preparation
("no end publishes the carried value"). Every adopting port aliases the
carried value, so adopters see the net's own refusal, attributed to the net.

**A composite port** has the same surface as a primitive port (closure). Its
mode is decided statically from its inner net:

| Declared | Inner net has another anchor | Mode | Outer value | Inner net |
|---|---|---|---|---|
| `carry=x` | any | publish | `x` (anchors the outer net) | anchored by `x` |
| no `carry` | yes | expose | the inner carried value (anchors the outer net) | anchored inside |
| no `carry` | no | relay | adopted from the outer net | anchored by the relay |

A relaying port that is unconnected outside is `Unresolved`
(`port-unconnected`), so a composite that needs its environment stays
unresolved when standalone. That is the right outcome, not an error.

**Offers.** `Port(..., offer=view)` names one of the Space's own views.
`Ends` reads its accepted value as the end's facet. The composite's own ports
have no offer inside the net; the link derives what a boundary presents (for
streams, the AXIS contract) from the carried value and the port name.

**Ends.** `Ends(interface)` is the tuple of *active* ends in declaration order:
the composite's own ports first, then children in placement order. An end is
active when its port's guard holds. The guard already includes the owning
placement's guard and, for a choice case, its selection.

**Nodes.** Nodes are the composite's `Subspace` placements (other than nets)
and `SubspaceChoice`s. A choice is one node. Its cases may differ completely in
their ports: each case placement binds its own ports to the composite's nets,
and an unselected case's ends are inactive. A case may place nothing
(`Subspace(Space)`, or a named empty family such as MVAU's `External`).

**Topology and folds.** A `Topology` holds the composite's active nodes, nets
and ports, in declaration order:

- a node appears when it exports the interpretation's node key; for a choice,
  the key is looked up on the selected case;
- a net appears when its link family exports the interpretation's net key;
- anything inactive or without a contribution is omitted.

`Fold(interp, constraints=..., **arguments)` is a view. Its raw output is
`interp.reduce(topology=..., **arguments)`, and its obligations are the
declared constraints plus *every* potential contribution view.

### 2.3 Lowering onto the evaluation graph

Every concept lowers onto nodes the existing scheduler already evaluates.
The four new kinds are small, pure frames:

| Declaration | Lowered nodes | Node kind (evaluation) |
|---|---|---|
| `Port` with `carry=` | the port node aliases the carry reference | `alias` |
| `Port` attached and adopting | the port node aliases the net's `$carried` | `alias` |
| composite `Port` (expose or relay) | aliases the inner or outer `$carried` | `alias` |
| unconnected adopting `Port` | the port node | **`port`**: `Unresolved(port-unconnected)` |
| `Net(L, carry=...)` | a child scope of `L`, plus `<net>.$carried` and `<net>.$ends`; `carry=` becomes a reference, a `<net>.carry` constant, or a `<net>.carry` decision owned by the parent scope | scope |
| `<net>.$carried` | its arguments are the anchors, in order | **`unify`**: first value if all active anchors agree |
| `<net>.$ends` | per potential end: its presence guard and offer view | **`ends`**: tuple of active `End`s |
| `Carried` / `Ends` members of a `Link` | alias to `$carried` / `$ends` | `alias` |
| `Fold` | a view node; `<fold>.$output`, an explicit-input derived calling `reduce`; and `<scope>.$topology.<interp>` | `view`, `derived`, **`topology`** |
| view obligations | `constraints=` may name a `View` or an accepted ref; its acceptance counts as `True` | `view` frame |
| `choice.case()` | a read-only reference to the choice's selector | resolves to the selector |

The linker adds one phase, `attach`, after allocation and nested parameter
overrides:

1. collect each net's ends from placement plans and `net=` ports;
2. resolve each net's anchors bottom-up, since an inner net decides whether a
   composite port exposes or relays;
3. reserve `$carried` and `$ends`;
4. record every port's alias target.

After members are linked, `link_graph` fills the anchors, end slots (whose
presence guards now exist) and topologies. The existing
`dependency_order` then sees every alias and anchor edge, so a value cycle
through declared bindings fails preparation like any other.

### 2.4 Type flow and evaluation order

The three flows the brief names:

| Flow | Expressed as | Evidence |
|---|---|---|
| publish → adopt | one anchor (a publishing port, a composite port with `carry=`, or `Net(carry=)`) | MVAU `in0_V` publishes `activation_spec`; replay adopts it |
| both publish → check | several anchors, compared with interface equality; refusal `net-disagree` owned by the net | `test_two_publishers_are_checked_and_the_refusal_belongs_to_the_net` |
| both support a family → the edge decides | `Net(L, carry=Decision(...))`: a net-owned decision `<net>.carry`, persisted like any decision | `test_a_net_owned_decision_anchors_the_value_and_persists` |

There is no global propagation pass. A read of an adopted port is an alias read
of `$carried`, which demands its anchors, which demand whatever the publisher
computes. The scheduler's native suspension handles this like any other
dependency. The engine's existing rules settle order:

- **Static cycles** go through declared bindings, for example two echo kernels
  each publishing what they adopt. They fail preparation with
  `DefinitionError: model contains cyclic dependencies`.
- **Dynamic cycles** go through a method body. They fail evaluation with the
  full scoped path, for example:
  `adder.out (value) -> adder.sum_width (derived) -> adder.back (value) ->
  feedback (value) -> echo.q (value) -> echo.d (value) -> result (value) ->
  adder.out`.
  Note that the net names `feedback` and `result` appear in the path.

A family negotiated from the ends' offers, rather than from a static domain,
needs the decision's domain to read `Ends`. The design places that decision in
the link family as its anchor, not on the parent. See §8, Q9. It is not built.

### 2.5 Readiness, acceptance and attribution

A fold's assessment uses the existing view reducer unchanged:

- **Per-contribution results.** `assessment.constraints.results` has one entry
  per node and per net contribution, keyed by that view's key: for example
  `first.connection`, `left.part` or `implementation.cyclic.build_requirements`.
- **Inactive contributions** are `Inapplicable` and never refuse. An inactive
  case's contribution is evaluated only as far as its guard; its body never
  runs.
- **Several refusals** and nothing unresolved: `accepted_result` is `Rejected`
  and carries *every* refusal's findings, each with its own owner (for example
  `first.compatible` and `second.compatible`).
- **Some obligation unresolved**: `accepted_result` is `Unresolved`, and the
  known refusals remain visible in `results`. This is the existing precedence.
- **The reducer's own refusal** belongs to the fold view. An example is a
  clock-domain conflict that no single net can see (`stream-composition`).
- **Readers.** `inspection.explain` shows `$topology.<interp>`, and per net its
  `$carried`, `$ends`, constraints and contribution.

**What an interpretation may read.** Only the `Topology` value and the
arguments its `Fold` binds (references or literals). The reducer is an
explicit-input function: it has no `self`, cannot read configuration state
and receives detached values. It is therefore hermetic and reusable across
every composite in its domain. Contributions are whatever node and link
families export under the interpretation's keys.

## 3. Proposed public API

All names are exported from `finn.core.space`; the value types are also in
`finn.core.space.graph`.

```python
class Interface(Generic[C, F]):
    def __init__(self, name: str, *, carries: type[C] | ValueSemantics[C],
                 offers: type[F] | ValueSemantics[F] | None = None) -> None: ...

class Port(ValueDecl[C]):                      # cfg.port -> C
    def __init__(self, interface: Interface[C, Any], direction: Literal["in", "out"], *,
                 carry: ValueRef[C] | None = None,     # publish; otherwise adopt
                 offer: View[Any] | None = None,       # this end's facet
                 net: Net[Space] | None = None,        # the Space's own port, attached inside
                 when: ValueRef[bool] | None = None) -> None: ...

class Net(Subspace[L]):                        # cfg.net -> the link scope, typed L
    def __init__(self, link: type[L], *, carry: object = None,   # ref | literal | Decision
                 when: ValueRef[bool] | None = None, bindings=None, **parameters) -> None: ...

class Link(Space): ...                         # base of net families
class Carried(ValueDecl[C]):  def __init__(self, interface: Interface[C, Any]) -> None: ...
class Ends(ValueDecl[tuple[End[F], ...]]): def __init__(self, interface: Interface[Any, F]) -> None: ...

class Interpretation(Generic[N, E, R]):
    def __init__(self, name: str, *, node: ViewKey[N] | None, net: ViewKey[E] | None,
                 result: type[R] | ValueSemantics[R],
                 reduce: Callable[..., R | QueryResult[R]]) -> None: ...

class Fold(View[R]):
    def __init__(self, interpretation: Interpretation[Any, Any, R], *,
                 constraints: Sequence[Obligation] = (), when=None, **arguments: object) -> None: ...

# values: End[F], EndRef, NetEntry[E], PortEntry, Topology[N, E]
# generalized: View(..., constraints=(Constraint | ConstraintGroup | View | AcceptedViewRef, ...))
# new: SubspaceChoice.case() -> ChoiceCaseRef (read-only selected case, a str)
# narrowed: Subspace.accepted / SubspaceChoice.accepted return AcceptedViewRef[T] (was ValueRef[T]),
#           so an accepted ref type-checks as a view obligation
# placement: Subspace(Kernel, <port name>=<Net>) attaches a port; unbound ports stay unconnected
```

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **Topology outside the engine** (all three prior attempts) | Identity, attribution, readiness and presence are reimplemented, badly. See §1 |
| **Pairwise edges naming endpoints**: `Connect(a.port(X.out), b.port(Y.inp))`, Chisel's `<>` | Fan-out and a clock net need extra syntax. A choice case's port must be reached through a path into the case. Topology and placement bindings would be two places to read. Nets bound at placement (Verilog `.port(wire)`, and the user's own `s1 = AxiStream(); DotP(input_1=s1)` sketch) reuse the existing binding machinery and make hyperedges native |
| **Ports as child Spaces** (a port-kind Space placed per port) | A scope per port, bindings from the kernel's own views into its port scopes, and no gain over one carried value plus one offered facet. Structured values cover richer protocols |
| **Ports as optional `Param` formals** (the status quo) | Every standalone placement must bind them. There is no unconnected state, no direction and no attribution |
| **Boundary pseudo-nodes** (`TopInput`/`TopOutput`) | They stand in for the parent's own port. The inside face of a real port is that port, and closure follows |
| **A separate `agree` constraint** beside a first-anchor-wins value | Adopters would silently compute with a disputed value. Refusing the carried value makes every reader see it, attributed to the net |
| **Engine-level edge direction or roles** | Roles and cardinality are domain facts (stream versus clock versus AXI-MM). The engine records `in`/`out` per end and nothing more |
| **`connected(stream)`-style proxy constraints** | Views may now oblige views directly. That is the proper extension the proxy was faking |
| **A fold hand-called per parent** (`compose(...)` inside a `structure` view) | The author restates the topology. `Fold` receives it from the engine |
| **A declared common port surface on `SubspaceChoice`** (port exports) | Not needed: cases bind their own ports to the enclosing composite's nets. Revisit only if a *choice itself* must be attached by an outer composite without a wrapping composite |

## 4. Cycles: DAG or unbounded?

**Recommendation: allow cyclic declared topology. Require only that the value
flow be acyclic, which the engine already enforces. Domains restrict topology
with ordinary constraints over `Ends` or `Topology`.**

The two graphs are kept apart. Declared topology (nodes, ports, nets) never
becomes evaluation structure by itself; only aliases and anchors do. A cycle
of nets therefore creates an evaluation cycle only if values actually flow
around it without an anchor.

| | Unbounded topology, acyclic value flow (recommended) | DAG-only topology |
|---|---|---|
| Engine cost | None beyond the model: the engine never traverses topology to evaluate. Cycle diagnostics are the existing ones, with net-named paths | A topology acyclicity check. With conditional presence it must evaluate guards, so it cannot be a preparation check. It would be an engine-generated constraint, which is domain policy inside the engine |
| Diagnostics | Static value cycle: preparation `DefinitionError`. Dynamic: evaluation error with the scoped path through the nets | A "cycle" error even when the value flow is anchored and well defined |
| Adopting-port cycle | No anchor means refusal: static cycle, `no end publishes`, or `net-unanchored`. No unification and no fixed point | Same |
| Expressiveness | Feedback, request/response (AXI-MM), and a shared resource with clients | Each needs an escape hatch |

**What a cycle of adopting ports means.** It means refusal, not unification or
a fixed point. A fixed point would need lattice-typed values and would break
the invariant that settled observations stay stable. Unification would make
the answer depend on every reader of the loop.

**Near-term consumers.** None needs a cycle. MVAU and `Constants` are DAGs.
The roster's AXI-Lite-writable delivery is the first plausible
request/response pair. Canon Networks are acyclic by definition (NETWORK.md
§4.8); `finn.dataflow` adds that as a constraint in its own fold.

**Spike evidence.** `Accumulator` (adder → register → adder, anchored on the
feedback net) evaluates and folds. `Unanchored` fails evaluation with the
scoped cycle path, and `Mirror` fails preparation. See §9.

**What would change the recommendation:**

- a consumer that needs *type inference around a loop*, for example an
  accumulator width that must converge. That needs a fixed-point Interface
  semantics, and a DAG restriction would not help either;
- a need for the *engine* to provide a topological order of nets, for example
  for scheduling. A fold can compute that from `Topology` today, so this
  alone does not justify the restriction.

## 5. Worked examples

These are the spike's actual sources, abridged only where marked.

### 5.1 MVAU: external or cyclic weights, and an optional FIFO

```python
class External(Space):
    """External weight delivery: nothing is placed; the weights enter at ``in1_V``."""


class MVAU(Space):
    ...                                  # facts, pe/simd, folding, specs: unchanged
    activations = Net(StreamLink)
    replayed = Net(StreamLink, carry=replayed_spec)
    weight_stream = Net(BufferedStreamLink, carry=weight_spec)   # transport: direct | fifo
    results = Net(StreamLink)

    in0_V = Port(STREAM, "in", net=activations, carry=activation_spec)
    out0_V = Port(STREAM, "out", net=results, carry=result_spec)
    replay = Subspace(ReplayBuffer, sequence_length=synapse_folds, replay_count=neuron_folds,
                      input_stream=activations, output_stream=replayed)
    compute = Subspace(DotpAxiKernel, ..., activation_stream=replayed,
                       weights_stream=weight_stream, result_stream=results)
    implementation = SubspaceChoice({
        "external": Subspace(External),
        "cyclic": Subspace(CyclicDelivery, dtype=weights_dtype, form=weight_period,
                           values=weights, output_stream=weight_stream),
    })
    delivery = implementation.case()

    @derived
    def external(self) -> bool:
        return self.delivery == "external"

    in1_V = Port(STREAM, "in", net=weight_stream, when=external)   # a conditional boundary port

    @derived
    def module_name(self) -> str:
        return "finn_mvau_" + self.delivery

    @derived(semantics=default_semantics(ProducerIdentity))
    def producer(self) -> ProducerIdentity:
        return ProducerIdentity("finn.mvau." + self.delivery, "1")

    structure = Fold(NETLIST, constraints=(dimensions,), module=module_name, producer=producer)

    @view(semantics=default_semantics(ModuleBuildRequirements), constraints=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure().requirements
```

`weight_stream` has two potential drivers: `in1_V` (guarded by `external`) and
`implementation.cyclic.output_stream` (guarded by the case's selection).
Exactly one is active, and `StreamLink.endpoints` refuses anything but one
producer and one consumer. The FIFO is still a choice inside the net's own
link family. Its keys are unchanged:
`weight_stream.transport` and `weight_stream.transport.fifo.buffer.{depth,ram_style}`.

The domain side, defined once in `finn.kernels.streams`:

```python
STREAM = Interface("stream", carries=STREAM_SPEC, offers=STREAM_CONTRACT)
MODULE = ViewKey("module", default_semantics(ModuleBuildRequirements))

class StreamLink(Link):
    spec = Carried(STREAM)
    ends = Ends(STREAM)
    @derived(semantics=ENDPOINTS)
    def endpoints(self) -> Endpoints | Rejected: ...    # one producer, one consumer; boundary -> AXIS
    @constraint
    def compatible(self) -> bool | Rejected: ...        # compatibility(), both halves around a FIFO
    connection = View(link, constraints=(compatible,))
    exports = {CONNECTION_VIEW: connection}

NETLIST = Interpretation("netlist", node=MODULE, net=CONNECTION_VIEW,
                         result=COMPOSED, reduce=netlist)   # netlist(*, topology, module, producer)

class DotpAxiKernel(Kernel):
    ...
    activation_stream = Port(STREAM, "in", offer=activation_port)
    weights_stream = Port(STREAM, "in", offer=weights_port)
    result_stream = Port(STREAM, "out", offer=result_port)
    exports = {MODULE: build_requirements}
```

### 5.2 The two-stream `Constants` test

```python
class Constants(Space):
    first_spec = Param(STREAM_SPEC)
    second_spec = Param(STREAM_SPEC)
    first = Net(StreamLink)
    second = Net(StreamLink)
    out0_V = Port(STREAM, "out", net=first, carry=first_spec)
    out1_V = Port(STREAM, "out", net=second, carry=second_spec)
    first_source = Subspace(CyclicDelivery, dtype=INT4, form=PRODUCED, values=(1, 2, 3, 4),
                            output_stream=first)
    second_source = Subspace(CyclicDelivery, dtype=INT4, form=PRODUCED, values=(5, 6, 7, -8),
                             output_stream=second)
    build = Fold(NETLIST, module="constants", producer=ProducerIdentity("test.constants", "1"))
```

This has no `compose`, no `connected`, no `TopOutput`, no instance literals and
no `structure` view. With two mismatched streams, `build.inspect()` reports
`first.connection` and `second.connection` both `Rejected`, owned by
`first.compatible` and `second.compatible`. The accepted result is `Rejected`
carrying both. With one mismatch, the other net's `connection` is `Available`.

### 5.3 Not a stream, and not a DAG: an accumulator with feedback

```python
WORD = Interface("word", carries=int, offers=str)

class Adder(Space):
    inp = Port(WORD, "in", offer=label)
    back = Port(WORD, "in", offer=label)
    @derived
    def sum_width(self) -> int:
        return max(self.inp, self.back) + 1
    out = Port(WORD, "out", carry=sum_width, offer=label)

class Register(Space):
    d = Port(WORD, "in", offer=label)
    q = Port(WORD, "out", offer=label)            # adopts: it publishes nothing of its own

class Accumulator(Space):
    width = Param(int)
    total = Param(int)
    inp = Net(Wire, carry=width)
    result = Net(Wire)
    feedback = Net(Wire, carry=total)             # the anchor that makes the loop well defined
    x = Port(WORD, "in", net=inp)
    adder = Subspace(Adder, inp=inp, back=feedback, out=result)
    register = Subspace(Register, d=result, q=feedback)
    listing = Fold(LISTING, name="acc")
```

`Accumulator(width=4, total=12).listing()` returns
`acc[adder=adder,register=register|inp=4:<top>.x,adder.inp;result=13:adder.out,register.d;feedback=12:adder.back,register.q|in:x]`.
Replacing the register by an `Echo` that publishes on `q` what it adopts on `d`
(`q = Port(WORD, "out", carry=d)`) removes the anchor and fails with the cycle
path quoted in §2.4.

### 5.4 Not a stream, built as data: a clock net

```python
CLOCK = Interface("clock", carries=int)

class ClockNet(Link):
    period = Carried(CLOCK)
    ends = Ends(CLOCK)
    @view
    def loads(self) -> int:
        return sum(1 for end in self.ends if end.direction == "in")

class Timed(Space):
    clk = Port(CLOCK, "in")
    @derived
    def frequency(self) -> int:
        return 1000 // self.clk

builder = ScopeBuilder(Space, name="ClockTree")
clk = builder.add("clk", Net(ClockNet, carry=4))
for index in range(3):
    builder.add(f"unit{index}", Subspace(Timed, clk=clk))
point = builder.finish()()
assert point.clk.loads() == 3 and point.unit2.frequency == 250
```

The net has one driver and many loads, and it is a hyperedge. A driver-count
constraint would live in `ClockNet`, as `Wire.one_driver` does in the toy
domain.

### 5.5 Nested composites: expose and relay

`Pair` places a source fanned out to two sinks on net `data`. It also exposes
a third sink's input as its own port `feed`, which relays because nothing
inside anchors `spare`. Standalone, `Pair` leaves `extra.inp` unresolved
(`port-unconnected`). `Closed` places `Pair` and drives `feed` from its own
source on net `feed`; `pair.extra.inp == 5`. `pair` is then an ordinary `in`
end of the outer net. See `tests/core/space/test_graph.py`.

## 6. Migration

What is deleted from `finn.kernels`, as done in the spike:

| Deleted | Replaced by |
|---|---|
| `compose(module, producer, instances, connections)` as public API | `netlist(*, topology, module, producer)`, reached only through `NETLIST`/`Fold` |
| `connected(stream)` | view obligations; `Fold` obliges every contribution itself |
| `Stream(Subspace)`: literal `(instance, port_view)` endpoints and `__set_name__` injection of `name` | `Net(StreamLink)` / `Net(BufferedStreamLink)`; ports attach at placement; names come from declarations |
| `StreamLink.{name, source, sink, source_instance, sink_instance}` params | `Carried(STREAM)`, `Ends(STREAM)`; `Connection.name` removed (the net name is in `Topology`) |
| `TopInput`, `TopOutput`, `_top` | composite `Port(..., net=)`; `boundary_contract(name, spec, endpoint)` inside `StreamLink` |
| `streams.Port` (an optional `Param` subclass) | core `Port(STREAM, direction, offer=)` |
| `Module`, `MODULE_SEMANTICS`, `OUTPUT_PORT`; `CyclicDelivery.module` | `MODULE = ViewKey(ModuleBuildRequirements)`; kernels export `{MODULE: build_requirements}` |
| MVAU's hand-written `structure` view, `source`/`sink` placements, `*_connected` constraints, `streams` group, choice `exports` | `structure = Fold(NETLIST, ...)`; `in0_V`/`in1_V`/`out0_V` ports; `External` empty case |

The engine adds:

- `graph.py`: 125 lines;
- declarations: `Interface`, `Port`, `Net`, `Carried`, `Ends`, `Fold`,
  `ChoiceCaseRef`, view obligations;
- the linker's `attach`/`link_graph`/`topology`/`link_fold`: about 350 lines;
- runtime frames: about 130 lines;
- in total, about 880 added engine lines, including `graph.py`, and 24 changed;
- `ir`: `EndSlot`, `TopologyPlan`;
- placement plans gain `ports`.

Nothing is removed from the engine. The node-kind protocol, the scheduler,
selections, codecs and the existing inspection API are unchanged.

**Keys and identity changes:**

- Every decision key is unchanged, so selections and codecs replay as before.
- The cyclic weight instance becomes `u_implementation` instead of `u_weights`,
  because it now derives from the node's declaration name.
- Gone node keys: `source.*`, `sink.*`, `implementation.external.*`,
  `*_connected`, `streams`, `<stream>.name`.
- New node keys: `in0_V`, `in1_V`, `out0_V`, `<net>.$carried`, `<net>.$ends`,
  `$topology.netlist` and `structure.$output`.
- Constraint result keys in `structure.inspect()` are per contribution: for
  example `activations.connection` and `compute.build_requirements`.

**Tests changed:**

- `test_declared_streams.py` is rewritten as nets.
- `test_mvau_delivery_choice.py`:
  - instance names change;
  - the external case is `External`;
  - an unresolved cyclic module is now visible as its own obligation;
  - the inactive family is reached only to settle its guard.
- `test_dotp.py`: ports are no longer bound, and are listed as `port`
  members.
- `test_mvau_assembly.py`: placement binds the nets.
- `tests/core/space/typing/positive.py` and
  `tests/kernels/typing/axi_stream_types.py`: `accepted()` now asserts
  `AcceptedViewRef[...]`.
- `tests/core/space/test_graph.py` is new: 15 tests on a toy, non-stream domain.

## 7. Acceptance criteria for the implementation pass

1. **Authoring.** MVAU and `Constants` are authored with no `compose`,
   `connected`, literal instance names or hand-written `structure` view.
   *(The spike meets this.)*
2. **Attribution.** Refusals are attributed per net and per node, and
   independent refusals are visible at once in one assessment.
   `test_each_stream_owns_its_refusal_and_independent_refusals_are_all_visible`
   and `test_independent_refusals_are_all_visible_and_owned_per_node_and_net`
   show this. *(The spike meets this.)*
3. **Evidence.** `inspection.explain` of a fold shows `$topology.<interp>`, and
   per net `$carried`, `$ends`, `compatible` and `connection`. *(The spike
   meets this.)*
4. **Build identity.** MVAU `module_build_fingerprint` is identical for equal
   configurations, or any change is recorded with its cause. In the spike,
   four of six configurations are identical. The two cyclic configurations
   differ only by instance id: they are bit-identical once `u_implementation`
   is mapped back to `u_weights` (§9). The gate decides whether to accept the
   new name (Q1).
5. **Simulation.** MVAU numeric XSim passes, direct (20) and through a FIFO (8).
   *Not run in the spike.* The harness goes through `mvau_assembly`, and only
   the wrapper's internal net names change for cyclic delivery.
6. **Gates.** `check-kernels.sh` and `check-dataflow-design.sh` pass. *(The
   spike's gate results are in §9.)*
7. **Engine hardening:**
   - type `Ends` as `tuple[End[F], ...]` end to end;
   - add an `Interface` overload so that an interface without `offers=` needs
     no explicit `Interface[C, None]` annotation;
   - give fold literal arguments explicit semantics;
   - add a diagnostic when a port's binding names a net of another scope;
   - add behavioural tests for guarded anchors, relay under a guarded outer
     net, and choice cases on both sides of a net;
   - check that `benchmark-space.py` shows no regression.
8. **Documentation.** Update the scratchpad `space/` guides (DESIGN,
   AUTHORING, INTERNALS and MIGRATION) and the kernel README (already updated
   in the spike).

## 8. Open questions for the human gate

1. **Instance naming.** Accept `u_implementation` for the cyclic weight source,
   or rename the choice? Renaming the choice changes the selection key
   `implementation`. *Recommendation: accept the new name.* It is what
   declaration-derived identity means, and nothing outside the wrapper sees it.
2. **Composite port mode.** Keep the static publish/expose/relay rule of §2.2,
   or require an explicit `relay=True`? With the static rule, a guarded inside
   anchor that is switched off leaves the net `net-unanchored`. It does not
   fall back to relaying. *Recommendation: keep the static rule.*
3. **Refusal placement.** A disagreement is a refusal of the carried value, not
   a separate `agree` constraint. Is that acceptable?
4. **Clocks and resets as nets now?** The netlist still routes clock and reset
   pins by role and by name (`"clk2x" in name`) inside the reducer. Declaring
   clock ports on kernels and `ClockNet`s in composites (§5.4) removes that. It
   also touches every kernel's ABI declaration and wrapper wiring, so it may
   change fingerprints. *Recommendation: make it the next increment, not part
   of this pass.*
5. **Offer closure.** A composite port's *outward* offer (MVAU's `in0_V`
   contract, seen by an outer composite) is not built. The proposal is that the
   inner link exports it under an interface-level key; it is needed only for
   nested stream composites.
6. **Empty cases.** Keep the empty `External` case, or model delivery as a
   `Decision` plus `when=` guards? The keys are identical either way. The
   choice keeps `inspection.choices` meaningful.
7. **Negotiation hook.** For a family negotiated from ends' offers, is a
   link-level anchor `Decision` the right place? The alternative is a parent
   `carry=Decision` whose domain reads `net.ref(Link.ends)`. Settle this before
   adapter work.

## 9. Spike evidence

The branch is `spike/space-graph-composition`, one commit (`759ee0e17`) on `0d700b1ab`. The
spike is complete enough to answer the design questions. It is not reviewed
production code.

- **Engine suite.** 316 tests pass: the 301 existing plus 15 new
  `test_graph.py` tests. The new tests cover publish/adopt, fan-out, promotion,
  expose/relay, unconnected ports, `net-disagree`, preparation refusal of an
  unanchored net, conditional ends, a net-owned decision persisted by
  selections, the cyclic accumulator, dynamic and static value cycles,
  `ScopeBuilder` topology, view obligations and interface mixing.
- **Kernel suite.** 759 tests pass: the 758 at the base revision, minus the
  old `Constants` tests, plus the rewritten ones. Documentation examples: 18 of
  18 pass (`scratchpad/space/check-examples.py`, including the updated kernel
  README).
- **Fingerprints.** [`fingerprints.py`](fingerprints.py) prints
  `module_build_fingerprint` for six MVAU configurations through the public
  adapter:

  | Configuration | Baseline `0d700b1ab` vs spike | With `u_implementation` → `u_weights` |
  |---|---|---|
  | external | identical | identical |
  | cyclic, block ROM | differs | identical |
  | FIFO, external | identical | identical |
  | FIFO, cyclic | differs | identical |
  | padded output | identical | identical |
  | pumped DSP58 | identical | identical |

  The raw outputs are in `evidence/fingerprints-*.txt`, and the remapping
  harness is [`fingerprints_renamed.py`](fingerprints_renamed.py). An earlier
  spike run also differed in instance *order*, until the MVAU declarations were
  ordered replay, compute, implementation. Fold order is declaration order,
  which is the only ordering rule.
- **Gates.** See §9.1.

### 9.1 Gate results

Logs are in [evidence/](evidence/), with a [README](evidence/README.md).

| Gate | Base `0d700b1ab` | Spike |
|---|---|---|
| `check-kernels.sh` | pass: Space 301, kernels 758; format, lint and strict mypy clean | pass: Space 316, kernels 759; format, lint and strict mypy clean |
| `check-dataflow-design.sh` | pass: 16 | pass: 16 |

The spike's first gate runs failed on:

- import sorting (`check-space.sh` adds `--extend-select I`);
- strict typing of the new test file;
- two typing fixtures that pinned `accepted()` to `ValueRef[T]`.

All three were fixed before the recorded run. XSim was not run; see §7.5.

## 10. Deferred work this design touches

- **`Traversal` vs `BeatSequence`.** The stream interface's carried value is a
  `StreamSpec` over `Traversal`. Adopting the canon `BeatSequence` changes
  `STREAM`'s carried semantics and `StreamLink`, not the engine.
- **Adapter negotiation and insertion.** A link family may own a `Decision`
  anchor over a domain derived from `Ends` (Q7), and a choice of adapters, as
  `BufferedStreamLink.transport` does today. Canon's "adapters are ordinary
  nodes" maps to adapter placements inside the link scope. Whether they should
  instead become nodes of the composite, by expanding the net into two nets and
  a node, is a realization question.
- **Realization strategy** (one wrapper module or flattened). The
  interpretation is the seam. `NETLIST` produces one wrapper per composite. A
  flattening interpretation would need composite nodes to contribute their own
  `Topology` (for example through an identity interpretation exported as a
  view) and then merge the topologies recursively. No engine change is
  foreseen.
- **Dataflow model.** `finn.dataflow` can supply a `LOGICAL` interpretation
  producing a Network, with acyclicity as a constraint on its fold. Canon
  stays a domain, not an engine dependency.
