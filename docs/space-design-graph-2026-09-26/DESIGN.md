# A graph of design spaces

Date: 2026-09-26. Base: `feature/kernel-package-extraction` at `39482b480`.
Spike: `spike/space-design-graph` at `0acc51e1e`. Status: **design
spike for human review.** It supersedes the Port/Net/Interface/Fold model in
[`../space-graph-composition-2026-09-25/DESIGN.md`](../space-graph-composition-2026-09-25/DESIGN.md).
That model was shaped by streams rather than by design spaces.

The question this spike answers: *to get a generic, graph-based design engine,
does the existing Space system need to be broken down or transformed, or only
extended?* Hardware kernels are the stress case, not the goal.

## 0. Answer

| Part of the system | Verdict | Evidence |
|---|---|---|
| **Value graph and runtime**: nodes, demand-driven scheduler, snapshots, admission, selections, codecs | **Keep.** It already is a graph engine. The spike added three small frames and changed nothing else | §4, §6 |
| **Composition layer**: how Spaces are joined | **Transform.** Today a placement owns its incoming edges, and a choice owns its own export mechanism. Invert this to *nodes + edges + presence*, and make `Subspace` bindings, nested bindings and `SubspaceChoice` sugar over five primitives | §3, §6 |
| **Snapshot cache model** | **Transform, as a separate decision.** Each snapshot has its own cache, so one local edit in an N-node graph re-runs all N computations (measured). This is the real structural pressure a graph engine puts on Space | §7 |

Nothing that was built needed Ports, Nets, an Interface, direction or a fold
concept. Everything below is stated in terms of `Space`.

## 1. What a graph of Spaces already is, and what it lacked

```
 ownership (a tree)                        value flow (already a DAG)
 ──────────────────                        ──────────────────────────
 Chain                                     head.width_out ──► body.width_in
  ├─ head   : Stage                        body.first.width_out ──► body.second.width_in
  └─ body   : Pair                         body.second.width_out ──► body.width_out
       ├─ first  : Stage
       └─ second : Stage
```

A `Space` is a node. `Subspace` places a node, and `SubspaceChoice` places one
node from a set. Bindings and accepted references already carry values between
any nodes a scope can see, siblings included. The tree provides identity,
ownership and presence; it never limited value flow.

The earlier workarounds each mark one missing capability, and none of them is
"edges":

| Workaround in the kernel layer | Missing capability |
|---|---|
| literal instance names, `__set_name__` injection | **identity**: a value can't say which node produced it |
| hand-listed `instances={...}`, one `connected()` per stream | **quantification**: a computation can't range over the members that are present |
| `TopInput` in the external case, `Module(None)`, ports that must be bound | **presence**: an absent node can't be read as absent, and a formal can't be left open |
| edges only at placement, in class-body order | **free edges**: an edge can't be declared after its nodes (forward, around a cycle, or as data) |

## 2. The primitives

Five primitives, plus one generalization of views:

| Primitive | Meaning | Lowers to |
|---|---|---|
| **open formal** | A child `Param` left unbound at placement. It can be supplied by a `Bind`. If nobody supplies it, an optional one is *unsupplied* (`Unresolved`, `input-unsupplied`) and a required one is the usual definition error, raised where it must be supplied: the parent | `present` node (new) |
| **`Bind(target, source, when=)`** | An edge declared in the parent: it supplies a descendant's open formal. It may point forward, around a cycle, or be added as data. Several binds to one formal act like `Present` | `alias` node (the bind), plus alternatives of the target's `present` |
| **`Present(*sources)`** | The value of whichever single source is present. It is *unresolved while any source is unresolved*, because a later commitment could make a second one present. Two present sources: `multiple-suppliers`. None: unsupplied | `present` node (new) |
| **`node.at(member)` / `located(member)`** | The value together with where it lives: `Located(node, member, value)`. `node` is the child's declaration name (for a choice case, the choice's name), or None for the enclosing Space itself | `locate` node (new) |
| **`Members(key)`** | Every present child node that exports `key`, as `Located` values in declaration order. For a choice, the key is read on its selected case | `members` node (new) |
| **view obligations** | A view's `constraints=` may name another view, an accepted ref, or a `Members`. A `Members` obligation obliges *each* member separately, so refusals keep their owners and an absent member never refuses | `view` frame |

Also added:

- `SubspaceChoice.case()`: a read-only reference to the selected case;
- `SubspaceChoice.at(key)`;
- case placements reachable as ordinary references
  (`choice.alternatives["cyclic"].at(...)`, `.decision_ref(...)`);
- `accepted()` now returns `AcceptedViewRef[T]` instead of `ValueRef[T]`, so an
  accepted ref type-checks as an obligation.

Evaluation is unchanged. Every primitive is a node read on demand through the
existing dispatcher; cycles are detected exactly as before.

## 3. Reducibility: what becomes sugar

| Existing construct | Expressed with primitives |
|---|---|
| `Subspace(F, x=ref)` | a node, plus an unguarded `Bind` of `x` |
| nested `bindings={F.child.ref(G.x): v}` | a `Bind` to a nested open formal |
| `SubspaceChoice({...}, exports=(K,))` | a selector `Decision`, one `Subspace` per case guarded by `selector == case`, and `Present(case.accepted(K), ...)` for each export |
| `choice.accepted(K)` | `Present` over the cases' `K` |
| kernel `connected(stream)` | a view obligation |
| `compose(instances=..., connections=...)` | `Members(MODULE)` and `Members(CONNECTION)` |
| `TopInput` / `TopOutput` | `located(in0_V)`: the composite's own member |
| literal instance names | `Located.node` |

**Evidence.** `test_a_structural_choice_reduces_to_primitives` runs the same
assertions against `WithChoice` (a `SubspaceChoice`) and `WithPrimitives`
(`Decision` + guarded `Subspace`s + `Present`). Both pass:

- unresolved before selection;
- the tuned value;
- a stale case-local choice refused when switching;
- the fixed value after clearing, and the inactive choice `Inapplicable`;
- replay through `selections`.

The differences are cosmetic:

- decision keys: `choice.tuned.extra` versus `tuned.extra`;
- the `inspection.choices` metadata.

**Recommendation.** Keep `SubspaceChoice` as the authoring form, because it
names cases and keeps their keys. Lower it onto the primitives in the linker,
so there is one mechanism for presence (`select` nodes become `present` nodes)
and one for supplying formals (placement, nested and late bindings all become
binds). The spike did *not* rewrite that lowering. It only proved the
equivalence behaviourally; see open question 1.

## 4. Acid tests (generic, not hardware)

These are in `tests/core/space/test_design_graph.py`, 13 tests, all passing.

1. **A relation is a node.** `Same(left=first.at(Tiles.factor),
   right=second.at(Tiles.factor))` refuses with `aligned.equal`:
   `"first.factor=3, second.factor=4"`. The names come from the graph.
2. **Quantification.** `Members(COST)` ranges over present tiles. Obliging it
   gives per-member results (`a.cost`, `b.cost`), and the guarded-off `c.cost`
   is `Inapplicable` and never refuses. A budget constraint reads the located
   costs by node name.
3. **A graph built as data.** A 5-stage pipeline from `ScopeBuilder` leaves each
   `width_in` open and supplies it with `Bind` edges `e1..e4`. `explain` shows
   the edges.
4. **Whichever source is present.**
   `Sink(width=Present(a.accepted(...), b.accepted(...)))`, with `a` and `b`
   guarded by a mode `Decision`. Two unguarded binds are refused with
   `multiple-suppliers`, owned by the formal.
5. **Cyclic topology.** Adder → Register → Adder, with the loop edge a `Bind`.
   It evaluates when the register's raw width anchors the loop. With an `Echo`
   instead, it fails with the scoped cycle path.
6. **Closure.** A `Pair` composite's `width_in` is an open formal supplied by a
   `Bind` in `Chain`. `Members(WIDTH)` in `Chain` sees `head` and `body` as
   peers.
7. **Reducibility.** See §3. Case members are now ordinary typed references.
8. Supporting tests cover `located(own member)`, and a `Bind` to an
   already-supplied formal being refused.

## 5. Stress case: MVAU and `Constants`

```python
class MVAU(Space):
    ...                                   # facts, folding, specs: unchanged
    in0_V = View(activation_spec)         # the composite's own boundary members
    out0_V = View(result_spec)
    replay = Subspace(ReplayBuffer, input_stream=activation_spec, ...)
    compute = Subspace(DotpAxiKernel, ..., weights_stream=weight_spec, ...)
    implementation = SubspaceChoice({"external": Subspace(External),
                                     "cyclic": Subspace(CyclicDelivery, ...)})
    delivery = implementation.case()
    @derived
    def external(self) -> bool: return self.delivery == "external"
    in1_V = View(weight_spec, when=external)

    activations = Subspace(StreamLink, spec=activation_spec,
                           source=located(in0_V), sink=replay.at(ReplayBuffer.input_port))
    replayed = Subspace(StreamLink, spec=replayed_spec,
                        source=replay.at(ReplayBuffer.output_port),
                        sink=compute.at(DotpAxiKernel.activation_port))
    weight_stream = Subspace(BufferedStreamLink, spec=weight_spec,
        source=Present(located(in1_V),
                       implementation.alternatives["cyclic"].at(CyclicDelivery.output)),
        sink=compute.at(DotpAxiKernel.weights_port))
    results = Subspace(StreamLink, spec=result_spec,
                       source=compute.at(DotpAxiKernel.result_port), sink=located(out0_V))
    modules = Members(MODULE)
    streams = Members(CONNECTION)

    @view(semantics=COMPOSED, constraints=(dimensions, modules, streams))
    def structure(self) -> Composed | Rejected:
        return netlist(self.modules, self.streams, module="finn_mvau_" + self.delivery,
                       producer=ProducerIdentity("finn.mvau." + self.delivery, "1"))
```

- **A stream is an ordinary Space** (`StreamLink`): two `Located` formals, a
  `compatible` constraint and an exported `connection` view.
  `BufferedStreamLink` still owns the FIFO transport choice, and its keys are
  unchanged.
- **Kernel ports** are plain optional `Param`s again: the `Port` subclass is
  deleted. They no longer need binding when placed standalone.
- **Deleted:** `Stream` (literal names and `__set_name__`), `connected`,
  `compose` as the parent's hand-called reduction, `TopInput`/`TopOutput`,
  `Module`/`OUTPUT_PORT`, and the per-case exports.
- **`streams.py`** is 92 lines shorter.
- **Fingerprints.** They are identical to the base for external, FIFO-external,
  padded-output and pumped-DSP58. The two cyclic configurations differ only
  because the instance is now `u_implementation` (the node's name) instead of
  `u_weights`. With that remapped, all six are bit-identical to `0d700b1ab`;
  see `evidence/`.
- **Every decision key is unchanged.**
- **`Constants`** (`tests/kernels/test_declared_streams.py`) is two relations and
  two `Members`. It has no `compose`, `connected`, literals or pseudo-nodes.
  Independent refusals are visible together, owned by `first.compatible` and
  `second.compatible`.

## 6. Resistance log: where the existing engine pushed back

This is the evidence for the verdict. Each entry records what the spike had to
change, and what that says about the architecture.

| # | Resistance | Change needed | Reading |
|---|---|---|---|
| R1 | *Every child formal must be bound at placement* (`collect_placement`) | The check moves to the parent. `ScopeBuilder.place()` no longer diagnoses an incomplete placement, and two tests were rewritten | The placement-owns-its-edges rule is the core assumption to invert (§3) |
| R2 | Choice cases weren't reachable as nodes; case members were reachable only through `inspection` handles | Register case placements in the parent's child map (2 lines) | Cases want to be ordinary guarded nodes (§3) |
| R3 | Each reference-producing expression needs its own anonymous-node path in the linker: `Expr` already had one, and `Present` and located refs needed another each | Three ad hoc paths | The linker wants **one** lowering hook for reference expressions |
| R4 | New node kinds need late semantics inference (a `Present` over function views) | Infer from the first alternative, as for aliases | A small, local fix |
| R5 | Anchoring a loop on an *accepted* view whose constraint reads the loop closes the cycle | Anchor on a raw value | Not a defect: acceptance is part of value flow. Document it |
| R6 | A refusal that reaches a view through both its output and an obligation is reported twice | None (pre-existing) | A reducer wart to fix when lowering lands |
| R7 | One local edit re-runs the whole graph | None | §7 |
| R8 | Class-body order: a placement's bindings can only name earlier declarations | `Bind` solves forward and cyclic edges | Expected: edges become declarations |

Untouched: the scheduler and continuations, snapshots and trial admission,
`with_choices`, selections, codecs, and the inspection APIs. All 301 existing
Space tests pass. Two were edited for R1, and two typing fixtures for the
narrower `accepted()` return type.

## 7. The runtime pressure: whole-graph snapshots

[`scale_probe.py`](scale_probe.py) builds an N-stage pipeline, evaluates it,
edits one decision of the *last* stage, and reads again:

| N | prepare | full read | read after one local edit | callbacks re-run |
|---|---|---|---|---|
| 50 | 41 ms | 17 ms | 16 ms | 50 / 50 |
| 200 | 150 ms | 67 ms | 67 ms | 200 / 200 |
| 800 | 584 ms | 260 ms | 267 ms | 800 / 800 |

Each `with_choices` creates a new root snapshot with an empty cache, so an edit
costs O(N), not O(affected). For a design-space exploration over a graph of
kernels this dominates. The engine already records each evaluation's *observed*
dependencies, so an entry whose dependency closure contains no changed
assignment could be carried to the successor. That changes a stated invariant
("separate configurations do not share evaluation caches") and the
reclamation contracts. It therefore needs its own design; this spike only
measures it.

## 8. Compared with the previous spike

| | Previous (`759ee0e17`) | This spike |
|---|---|---|
| New concepts | `Interface`, `Port` (direction, carry, offer, net), `Net`, `Link`, `Carried`, `Ends`, `Fold`, `Interpretation`, `Topology`, `End`, publish/expose/relay | open formal, `Bind`, `Present`, `Located`/`at`, `Members` |
| New node kinds | 4 (`port`, `unify`, `ends`, `topology`) | 3 (`present`, `locate`, `members`) |
| New IR payload | `EndSlot`, `TopologyPlan` | none |
| Engine diff | about 880 added lines | about 480 added lines |
| Domain concepts in the engine | direction, one carried and one offered value | none |
| Composite boundary | port promotion with three modes | the composite's own members, `located(...)` |
| MVAU fingerprints | same result | same result |

## 9. What the implementation pass would do

1. Land the five primitives and view obligations. Fix R3 with a single
   reference-expression lowering hook, and fix R6.
2. Lower `Subspace` bindings, nested bindings and `SubspaceChoice` onto binds,
   guarded nodes and `present` nodes. Delete the duplicate `select` and
   nested-binding paths, preserving decision keys and `inspection.choices`.
3. Port the kernels as in §5, and run the MVAU XSim (direct and FIFO). XSim was
   not run in this spike.
4. Update the scratchpad `space/` documents.
5. **Separately:** design incremental cache reuse across snapshots (§7).

## 10. Open questions for review

1. **Lowering depth.** Should the implementation pass rewrite `SubspaceChoice`
   onto the primitives (one mechanism, larger change), or keep both mechanisms
   and document the equivalence?
2. **Unsupplied formals.** An open optional formal reads `input-unsupplied`.
   Should the root's omitted optional inputs, today `input-missing`, merge into
   the same code?
3. **`Present` stability rule.** It is unresolved while any source is
   unresolved. This protects settled observations, but it means a supplier
   behind an undecided guard delays every reader. Is that acceptable as the
   default?
4. **Instance naming.** Accept `u_implementation` (the node name)?
5. **Runtime reuse.** Take on incremental cache reuse (§7) next, or keep
   whole-snapshot caches until graphs grow?

### Evidence

`evidence/` holds the gate transcripts and fingerprint outputs, and
[`evidence/README.md`](evidence/README.md) describes them. The remapping harness
is [`fingerprints_renamed.py`](fingerprints_renamed.py); the fingerprint driver
is the previous design's
[`fingerprints.py`](../space-graph-composition-2026-09-25/fingerprints.py),
unchanged.
