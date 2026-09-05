# S1-C — semantic-core redesign, comparison and recommendation for C1

*Workstream: S1-C. Base revision `546538087`, branch
`work/dataflow-gate2-s1-semantics`. Nothing under `src/` or `tests/` is
modified by this submission; the only new files are this directory.*

Inputs: `dataflow-gate2-simplification-c0-decisions.md` §7,
`dataflow-gate2-semantic-requirements-design-note.md`,
`dataflow-gate2-simplification-implementation-plan.md` §5 S1-C.

---

## 1. Recommendation in one page

Adopt **schema A**: one new value on the canonical Region, and placement
derived from Region plus Network.

```python
# src/finn/dataflow/region.py
@dataclass(frozen=True)
class LocalStateInput:
    operand: Operand


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[InputInterface, ...]
    outputs: tuple[OutputInterface, ...]
    local_state: tuple[LocalStateInput, ...] = ()  # new, last, defaulted
```

```python
# src/finn/dataflow/placement.py   (new; imports region + network only)
@dataclass(frozen=True, slots=True)
class External:
    boundary_id: str
    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class LocalState:
    node_id: str
    operand_id: str


OperandPlacement = External | LocalState


def derive_input_placement(network: DataflowNetwork, operand_id: str) -> OperandPlacement: ...
def derive_output_placement(network: DataflowNetwork, operand_id: str) -> OperandPlacement: ...
```

The derivation rule is one sentence:

> **The entry site of a source input operand is the site carrying that operand
> which nothing inside the Network feeds.** A streamed input that is the sink
> of an `Edge` is a continuation, not an entry; a local-state input is fed from
> outside by construction. Exactly one site must survive.

Three consequences worth stating up front, because they are what C1 is
actually deciding:

1. **Two derived placement cases, not three.** `InternalStream` cannot fire
   over a Network that passes `validate_network`; the public destination union
   shrinks from three cases to two. Evidence: `run.py` §6.
2. **The decoupled answer changes.** Today the source weight is reported as
   `StreamDestination('compute', 'weight')` — the *consumer*. Under the rule it
   is `LocalState('memory', 'W')` — where the bytes are installed. One current
   test asserts the old answer (`tests/dataflow/ops/test_dataflow_op.py:933`).
3. **Matching moves from port id to operand id.** `OpInput(index=1,
   operand="weight")` must name a Region `Operand.id`, i.e. `"W"`, not a port
   id. Today the match works only because MVAU's port ids happen to equal its
   source member names.

The names come from the canon, not from me. `REGION.md` already says
**"declared local state"** three times — §3.7, §5.2 condition 1, and the model
map — and the design note's own placement case is `LocalState`. C0's
placeholder `StateInput` drops the qualifier that keeps it distinct from engine
state and occurrence lifecycle state, so I recommend `LocalStateInput` /
`DataflowRegion.local_state` / `LocalState`, which is the canon's phrase
unchanged.

I recommend **omitting** `requirements` from `LocalStateInput` for now. See §5.

---

## 2. What the current model actually does

Three findings, all read off `546538087`, all reproduced by `run.py`.

**The embedded Region does not say it consumes the matrix.**
`construct_embedded_dot_product_region` (`kernels/dotp_axi.py:239`) builds the
streamed Region and filters out the weight interface — operand, port,
requirements and all. Its docstring states the current position explicitly:
"The matrix itself is not an operand here. It is not traffic; it is state the
realization carries." That is the sentence this redesign reverses: whether the
matrix crosses a *port* is a transport fact, but that the arithmetic *consumes*
it is a mathematical one, and the Region is the value that states the
mathematics.

**The memstream Region does not say it holds the matrix.**
`construct_cyclic_parameter_region` (`parameters/cyclic/region.py:17`) returns
a Region with zero inputs and one output whose every position is "available at
the sole schedule point". Read as a value, it produces a 64×64 matrix from
nothing.

**Placement is therefore a guess.** `_internal_destination`
(`ops/mvau/op.py:613`) scans for an input port whose id equals the source
operand name and, failing that, returns `RegionStateDestination(node.id, ...)`
for the node literally named `"compute"`. Both branches are wrong in a way the
model cannot currently notice:

| case | today | schema A |
|---|---|---|
| external | `BoundaryDestination('weight','compute','weight')` | `External('weight','compute','weight')` |
| decoupled | `StreamDestination('compute','weight')` | `LocalState('memory','W')` |
| embedded | `RegionStateDestination('compute','weight')` | `LocalState('compute','W')` |

The decoupled row is the load-bearing difference. `_internal_destination`
finds compute's weight input port before it considers the memory node, so the
source tensor is reported as arriving where it is *consumed*. U6 asks the
opposite question — where does the initializer image have to be installed —
and gets a port with no storage behind it. The embedded row is right only
because the fallback hardcodes the node id; rename the compute segment and the
answer becomes `DataflowOpError`.

### 2.1 The canon already assumes the field the value model lacks

`scratchpad/dataflow/canon/REGION.md` — which I did not edit and do not own —
uses "declared local state" as an established concept in three places:

```text
§3.7  "a boundary sequence can present that position once, repeatedly, or not
       at all when declared local state supplies it"
§5.2  "The selected input beat sequences and declared local state can satisfy
       every canonical input-requirement occurrence"
map   "Input service | boundary beats · local state · scheduled requirements"
```

But `DataflowRegion = (S, Inputs, Outputs)` in §2.1 gives it nowhere to be
declared. The canon resolves this by making local state a *binding-witness*
property: §5.2 lists "streaming, replay, parameter memory, constant operands"
as things the witness may use. That was sufficient while placement was a
physical question. It is not sufficient now, because C0 asks whether a source
operand became external, internally streamed or local — and answers that
question *before* a binding witness exists, from Region plus Network alone.

So schema A is not importing a foreign concept. It promotes a term the canon
already uses from binding prose to a canonical field, and leaves the
*realization* of local state exactly where §5.2 has it. The fold consequence
is precise and belongs to S4:

- §2.1 becomes `DataflowRegion = (S, Inputs, LocalState, Outputs)`;
- §3.7's "when declared local state supplies it" gains a cross-reference to
  the declaration instead of pointing only at §5.2;
- §5.2 condition 1 keeps its wording; "declared local state" now names
  something the reader can point at.

This also answers a question C1 might otherwise ask about schema C: the canon
has no requirement object distinct from the operand, and inventing one would
put the canon and the implementation into two vocabularies.

---

## 3. The three schemas

### A — local-state inputs on `DataflowRegion` *(recommended)*

The Region states every mathematical input it consumes; the ones with no port
are listed separately because "has no port" is exactly the fact. Placement is
computed from Region plus Network, so a contributor authors topology once.

*Advantages*

- Reuses `Operand`, `RegionEndpoint`, `Edge`, `BoundaryContract` unchanged.
- The Design compiler needs **no** change: `_network_property`
  (`designs/design.py:851`) copies whole Region values into `NetworkNode`s, so
  local-state inputs ride along for free, and `_correspondence_constraint`
  (`designs/design.py:1053`) compares Region values, so it starts comparing
  local-state inputs for free too. Confirmed by reading both evaluators; neither names a
  Region field.
- `DATAFLOW_REGION_SEMANTICS` is `immutable_nominal`
  (`model/semantics.py:71`) — no field-level codec, so equality, hashing and
  fingerprinting absorb the new field with no engine change.
- A defaulted fourth field keeps all 36 positional `DataflowRegion(...)` call
  sites in `src` and `tests` compiling.
- The `ModuleBuildSpec` Region witness stays one value.

*Costs*

- The canonical Region definition grows a field, and `REGION.md`'s §5.1 rule
  list grows two entries.
- Region equality changes: an embedded Region that declares its local state is
  not equal to today's. This is intended — they say different things — but any
  fixture holding a literal embedded Region must be updated.
- `REGION.md` §2.1, §3.7 and §5.2 need the fold described in §2.1 below.

### B — adjacent immutable Region semantic metadata

Keep `DataflowRegion`; return a companion value beside it.

The companion itself is small (`RegionResidency`, `AnnotatedRegion`,
`AnnotatedNetwork` — 3 values). What is not small is that from the Kernel
declaration to the artifact boundary, one value becomes a pair. Nine seams in
the current code have to learn about it, listed in
`alternatives.py:B_THREADING_SITES` and read off the source, not estimated:

```text
kernels/kernel.py     Region declaration: a second exported member, or a Region
                      that returns a pair
kernels/kernel.py     the no-local-Decision audit and the region type token
model/semantics.py    a second ValueSemantics
designs/design.py     DataflowDesign.region(role): a second accessor
designs/design.py     _network_property: a second dependency per segment, a
                      second returned value, gated by the same segment `when`
designs/design.py     _correspondence_constraint: must compare companions too,
                      or the pair can silently disagree
designs/design.py     SelectedNetwork / the `network` export: consumers no
                      longer receive one value
network_validation.py validate_network takes a Network and would need the
                      companion map to check residency at all
artifacts/*           the ModuleBuildSpec Region witness becomes a pair
```

The sixth line is the disqualifying one. `_correspondence_constraint` exists
precisely to prove that a Network node holds its segment's selected Region. If
residency is a second value, that proof has to be duplicated for the companion,
and a Design that resolved a Region and a mismatched companion would be
*structurally valid*. That is the wrapper ladder the Unified Space effort
removed, re-entering through the Region.

### C — separate requirement/disposition graph

`SemanticRequirement`, `RequirementDisposition`, `DispositionKind`,
`DispositionGraph` — 4 values, of which `SemanticRequirement` is `Operand`
with the fields renamed (`id`, `element_type`, `shape`).

Two authorities. `alternatives.py` builds a disposition table that says the
decoupled matrix is local state of `compute` rather than `memory`: it is
well-formed, it satisfies every rule the graph itself can state, and it is
false. Detecting it requires deriving the truth from Region plus Network —
which is schema A. So C is A plus a table that can disagree with A, and the
only way to make C safe is to implement A underneath it.

C would earn its keep if requirements existed that no Region can express — a
requirement satisfied by several Networks at once, or a requirement with no
consuming Region. Neither exists in the four forcing cases, and none is on the
U6 or MLO path as described in the design note §7.

---

## 4. Exact recommended schema

### 4.1 `src/finn/dataflow/region.py`

```python
@dataclass(frozen=True)
class LocalStateInput:
    """One mathematical input the Region consumes without a stream port.

    Not storage.  A local-state input says the Region's computation consumes the
    operand and that this factorization gives it no port; it names no memory,
    technology, slot, image, file or module, and a physical choice can never
    add or remove one.
    """

    operand: Operand

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")

    @property
    def id(self) -> str:
        """A local-state input is named by its operand; it has no channel."""
        return self.operand.id


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[InputInterface, ...]
    outputs: tuple[OutputInterface, ...]
    local_state: tuple[LocalStateInput, ...] = ()

    def __post_init__(self) -> None:
        ...  # unchanged
        held = tuple(self.local_state)
        if not all(isinstance(item, LocalStateInput) for item in held):
            raise TypeError("local_state must contain only LocalStateInput values")
        object.__setattr__(
            self, "local_state", tuple(sorted(held, key=lambda item: item.operand.id))
        )

    def local_state_input(self, operand_id: str) -> LocalStateInput:
        """Return the uniquely identified local-state input."""
        matches = tuple(item for item in self.local_state if item.operand.id == operand_id)
        if len(matches) != 1:
            raise KeyError(f"expected one local-state operand {operand_id!r}, found {len(matches)}")
        return matches[0]
```

`interfaces` is **not** extended — it is the port-bearing collection and every
one of its consumers assumes a `.port`. Local state is reached through
`local_state` and `local_state_input()`.

Field order: `local_state` last and defaulted, so the 36 positional
constructions keep working and the migration is one field, not 36 edits.

### 4.2 `src/finn/dataflow/region_validation.py`

Two new codes, plus the requirement-domain checks if C1 takes the
`requirements` variant:

```text
local_state.operand_duplicate       two local-state entries share an operand id
local_state.operand_also_streamed   an operand is declared both streamed-in and local state
```

Everything else in `validate_region` is untouched. Local state deliberately does
**not** participate in `port.id_duplicate` (they have no port id) and **do**
participate in `operand.identity_conflict` (an operand id must mean one type
and shape throughout a Region, whether it arrives on a port or not).

### 4.3 `src/finn/dataflow/network_validation.py`

**No change.** This is worth stating explicitly: `endpoint.input_ownership`
requires every region input *endpoint* to be consumed or exposed exactly once.
Local state is not an endpoint, so it is correctly outside that rule, and no
exemption has to be written.

### 4.4 `src/finn/dataflow/placement.py` (new)

```python
@dataclass(frozen=True, slots=True)
class External:
    boundary_id: str
    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class LocalState:
    node_id: str
    operand_id: str


OperandPlacement = External | LocalState


class PlacementError(ValueError):
    """No entry site for an operand, or more than one."""


def derive_input_placement(network: DataflowNetwork, operand_id: str) -> OperandPlacement: ...
def derive_output_placement(network: DataflowNetwork, operand_id: str) -> OperandPlacement: ...
def placed_shape(network: DataflowNetwork, placement: OperandPlacement) -> tuple[int, ...] | None:
    """The shape at the placed port, or ``None`` when there is no port."""
```

Imports: `finn.dataflow.region`, `finn.dataflow.network`. Nothing else — no
ONNX, Space, Kernel, artifact or engine import, so the module joins the
`_assert_fresh_import_avoids(..., ("finn.dataflow._engine",))` list in
`tests/dataflow/test_package_boundaries.py:178`.

`placed_shape` returns `None` for a `LocalState`, preserving the current
`selected_shape` contract: `None` rather than `()`, because `()` is the shape
of a scalar and a consumer comparing it to the source shape would report a
mismatch instead of "there is no port".

---

## 5. The one sub-decision C1 must make: does `LocalStateInput` carry requirements?

| | A1 `LocalStateInput(operand)` | A2 `LocalStateInput(operand, requirements)` |
|---|---|---|
| placement derivation | works | works, identically |
| new fields | 1 | 2 |
| says "the embedded core reads what the streamed core reads" | no | yes, provably |
| embedded MVAU Region size, R=4 64×64 PE=SIMD=8 | 0 extra entries | +16 384 entries (~3 MB) |
| … R=8 512×512 PE=SIMD=16 | 0 | +2 097 152 entries (~352 MB) |

The prototype implements A2 and proves both halves: the requirement function
lifted from the streamed weight interface is *equal* to what the embedded core
needs, and every placement in every case is unchanged when the local-state inputs are
rebuilt without it (`run.py` §6).

**Recommend A1.** The requirement term is already the model's dominant memory
cost — `region.py`'s own module docstring is about keeping it sparse — and A2
doubles it for a claim that has no consumer today. `requirements` can be added
later as a second defaulted field, which is the same cheap migration as this
one; removing it later would not be. The trigger that would justify adding it:
a shared-MLO bandwidth or cycle analysis that needs a local-state input operand's access
schedule, which is exactly the U6/MLO work item, not this one.

---

## 6. Source-operand matching

The design note §5 proposes `OpInput(index=1, operand="weight")`. Two things
must be locked with it.

**The namespace is `Operand.id`, not a port id.** MVAU's Region operands are
`X`, `W`, `Y`; its port ids are `activation`, `weight`, `output`; its source
member names are `activation`, `weight`, `output`. The current matching in
`_internal_destination` compares source names against **port ids** and works
only because two of those three namespaces coincide by accident. Under the
recommended schema the declaration reads:

```python
activation = OpInput(index=0, operand="X", correspondence=FLATTEN_LEADING)
weight = OpInput(index=1, operand="W", correspondence=TRANSPOSE_2D)
output = OpOutput(index=0, operand="Y", correspondence=FLATTEN_LEADING)
```

Rename a port and nothing breaks; rename an operand and the mapping fails
loudly at the exactly-one check.

**Cross-node operand-id agreement is not required and must not be validated.**
It happens to hold in the decomposed MVAU (`X` → `X`, `W` → `W`), but a fused
pipeline would legitimately connect a producer's `Y` to a consumer's `X`. The
entry-site rule does not need the agreement: it filters candidate sites by
operand id *and* by "nothing inside feeds this", and in the decoupled Network
that leaves exactly one even though `W` names three sites. I recommend adding
no `edge.operand_identity_mismatch` rule.

---

## 7. The design note's eight validation questions

1. **Unique by id within one Region?** Local-state operand ids: yes, enforced
   (`local_state.operand_duplicate`). Streamed port ids: yes, already. Streamed
   *operand* ids: no, and unchanged — the replay Region uses `X` on both its
   input and its output, which is what a replay is.
2. **Can one logical operand be both streamed and local state?** No, for inputs —
   two contradictory transport claims about one mathematical input
   (`local_state.operand_also_streamed`). Yes, for local-state-plus-*output* — that
   is precisely what a parameter source is, and forbidding it would forbid the
   memstream. The prototype asserts both.
3. **Several nodes with equal operand ids?** The entry-site rule. Candidate
   sites = {streamed inputs carrying the operand that are not the sink of any
   edge} ∪ {local-state inputs carrying it}. Exactly one must survive; zero and
   many are both `PlacementError`, with all candidates named. No preference
   order, no node-name special case.
4. **Decoupled: memory state, not the compute port?** Falls out of (3).
   compute's `weight` input is the sink of `weight_supply`, so it is excluded
   as a continuation; memory's local-state `W` is the only survivor. This is the
   answer the prototype produces and it differs from production today.
5. **Must every local-state input appear in computation metadata?** No. There is no
   computation metadata on a Region — `family`/`version` live on the Kernel's
   `Region` declaration, and `ComputationContract` is being removed by C0 §9.
   Region structural validity checks local-state inputs exactly as far as it checks
   streamed inputs: identity, uniqueness, non-contradiction, and (under A2)
   requirement domains. Whether the local state is *appropriate* for the
   computation is explicit Kernel candidate admission's job, unchanged.
6. **How does a local-state input acquire contents without physical storage in the
   Network?** It does not, at this layer. The lineage is: source operand →
   `OperandMapping` → `LocalState(node_id, operand_id)` → U6 reads the mapping
   and the initializer summary and produces a `ModuleBuildSpec` with a data
   slot / parameter image. The Region is never consulted about storage and
   never gains a field that could be.
7. **Can several local-state inputs be grouped by an MLO realization without changing
   their Region values?** Yes. A grouping is a set of `(scope, LocalState)`
   pairs held by whoever owns the realization. `run.py` §7 groups the local-state inputs
   of two independently constructed decoupled Networks and asserts both
   Networks are equal to freshly constructed ones — the grouping wrote nothing
   back. No memstream Kernel learns that it was grouped.
8. **Family/version, or only value identity?** Only value identity.
   `family`/`version` are arguments to the Kernel's `Region` declaration
   (`kernels/kernel.py:171`), not fields of `DataflowRegion`, so a local-state input
   changes the value, its hash and any fingerprint over it, and changes no
   family. Separately: once local-state inputs make the delta explicit, the
   `mvau.dot_product` / `mvau.dot_product.embedded` family split arguably
   becomes redundant — same arithmetic, different transport. That is
   Kernel-owned (S2-B/S3), not mine, and I am not proposing it here.

---

## 8. Forcing examples

All four are built in `cases.py` from the production constructors —
`construct_dot_product_region`, `construct_embedded_dot_product_region`,
`construct_activation_replay_region`, `construct_weight_stream_region` — with
only the proposed local-state input added.

```text
external    replay --activation_replay--> compute
            boundaries: activation, weight, output
            W -> External('weight', 'compute', 'weight')          shape (64, 64)

embedded    replay --activation_replay--> compute (local state W)
            boundaries: activation, output
            W -> LocalState('compute', 'W')                         shape None

decoupled   replay --activation_replay--> compute
            memory (local state W) --weight_supply--> compute.weight
            boundaries: activation, output
            W -> LocalState('memory', 'W')                          shape None

MLO         two independent decoupled Networks, local-state inputs collected into one
            realization group; both Networks unchanged afterwards
```

`validate_network` reports zero issues on all three canonical projections.

**Prospective MLO.** The seam the design note §7 asks for is `LocalState`. A
shared off-chip realization is a set of `LocalState` placements plus the
initializer summaries their source operands already carry — both available
without opening a memstream Kernel, and neither stored in a Region. If MLO
later needs to say "these three local-state inputs share one physical bank", that is a
physical-projection value in U6 keyed on `(scope_id, node_id, operand_id)`,
which the placement already provides.

---

## 9. Type and authority accounting

| | new public values | values removed | net | second authority? |
|---|---|---|---|---|
| A | `LocalStateInput`, `External`, `LocalState` = 3 | `BoundaryDestination`, `StreamDestination`, `RegionStateDestination` = 3 | **0** | no |
| B | `RegionResidency`, `AnnotatedRegion`/pair, second `ValueSemantics` = 3 | 0 | +3 | yes — the pair can disagree |
| C | `SemanticRequirement`, `RequirementDisposition`, `DispositionKind`, `DispositionGraph` = 4 | 0 | +4 | yes — the table can lie |

Schema A is net type-neutral: `LocalStateInput` enters the Region, and the
three-case destination union becomes a two-case placement union. It adds one
field, one accessor, one module, two validation codes, and removes one
hand-written fallback (`_internal_destination`) and one class.

---

## 10. Prototype evidence

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```

Ten sections, all asserting; final line `all assertions passed`. What each one
establishes:

1. external / embedded / decoupled placements for `X`, `W`, `Y`, derived, no
   node-name special case.
2. the production `_internal_destination` answer for decoupled, side by side
   with schema A's, asserted to be `StreamDestination('compute','weight')` so
   the difference is a fact and not a claim.
2b. with the local-state inputs stripped, the embedded case has **no** entry site —
   the current model genuinely cannot answer, and production only appears to
   because of the hardcoded `"compute"`.
3. `validate_network` and `validate_region` report zero issues on all three
   canonical projections; the added local-state rules report zero.
4. all four new diagnostics fire on constructed negatives, and the
   local-state-plus-output-port case correctly does not.
5. an ambiguous Network (`W` local state on two nodes) raises `PlacementError`
   naming both candidates.
6. the embedded local-state input's requirement function equals the streamed weight
   interface's (16 384 entries); every placement is identical under A1; the
   A2 entry cost is tabulated across four foldings; `InternalStream` never
   fires over a valid Network.
7. MLO grouping over two Networks leaves both Network values unchanged.
8. schema B reaches the same answer and needs the companion at 9 seams.
9. schema C's table gives the right answer when written correctly and a
   well-formed wrong answer when not; catching that needs A's derivation.
10. the type accounting above.

Suite status on this revision, unchanged by this submission:

```
tests/dataflow (minus kernels/, artifacts/, parity/)   786 passed
tests/dataflow/kernels, artifacts, parity              not collected: the
                                                       interpreter available
                                                       here lacks msgspec and
                                                       pyslang
```

`git status` shows `prototypes/` as the only addition; no `src/` or `tests/`
file is touched. Ruff check and format are clean over `prototypes/gate2_s1c`.

I did **not** run strict mypy over the prototype: it imports `finn.dataflow`
with `PYTHONPATH` including `deps/qonnx/src`, which CLAUDE.md documents as the
configuration that turns every qonnx import into a false positive. The
production implementation will run the real gate.

---

## 11. Migration consequences

**Mine to implement after C1** (`region.py`, `region_validation.py`, new
`placement.py`, their tests):

- `LocalStateInput`, the fourth `DataflowRegion` field, `local_state_input()`;
- two validation codes;
- `placement.py` and its tests over hand-built and MVAU-derived Networks;
- `REGION.md` §5.1 rule list, if the canon document is mine — **flagging**: I
  did not locate a `REGION.md` in this worktree and did not go looking outside
  it. If it lives in the scratchpad authorities, S4 owns the fold.

**S2-A (`ops`) inherits:**

- `_internal_destination` and `_selected_shape` (`ops/mvau/op.py:613,634`)
  delete; `association` calls `derive_input_placement` /
  `derive_output_placement`;
- `OperandAssociation.destination` → `OperandMapping.placement`, and
  `BoundaryDestination` / `StreamDestination` / `RegionStateDestination` are
  deleted, not aliased;
- `OpInput(..., operand="W")` must name the Region operand id; the
  `operand=` argument is new on `InputTensor`/`OpInput`
  (`ops/schema.py:138`), which today has only `index`, `optional`,
  `fingerprint`.

**S2-B / S3 inherit:**

- `construct_embedded_dot_product_region` (`kernels/dotp_axi.py:239`) gains
  the local-state input and **its docstring must be rewritten** — it currently asserts
  "The matrix itself is not an operand here", which the accepted schema
  contradicts;
- `construct_cyclic_parameter_region` (`parameters/cyclic/region.py:17`) takes
  a local-state input for the operand it emits, so the memstream stops producing a
  matrix from nothing;
- `tests/dataflow/ops/test_dataflow_op.py:914` — the test named
  *"a decoupled matrix is traffic and an embedded one is state"* asserts the
  decoupled weight is a `StreamDestination`. Its expectation flips to
  `LocalState('memory','W')` and its name and docstring should change with it;
  the distinction it was defending (embedded has no port) survives intact.
- `tests/dataflow/parity/correspondence.py:530` describes
  `OperandAssociation.destination`; the prose needs the new vocabulary. This is
  parity-record text, not a behavioural change.
- the `mvau.dot_product` vs `mvau.dot_product.embedded` family split is
  *arguably* redundant once local-state inputs exist. Recommend recording it, not
  acting on it, in this simplification.

**Explicitly unaffected**, verified by reading the code rather than assumed:
`designs/design.py` `_network_property` and `_correspondence_constraint`
(Region values are copied and compared whole); `model/semantics.py`
(`immutable_nominal`, no field codec); `network.py` and `network_validation.py`
(local-state inputs are not endpoints); all 36 positional `DataflowRegion(...)`
constructions (defaulted last field).

---

## 12. What I need from C1

Five decisions, in the order they block work:

1. **Schema A, B or C.** Recommend A.
2. **`LocalStateInput(operand)` or `LocalStateInput(operand, requirements)`.**
   Recommend the former; §5 has the numbers either way.
3. **Names:** `LocalStateInput` / `local_state` / `LocalState`, or C0's
   `StateInput` / `state_inputs` / `LocalState`. Recommend the former; either
   is a mechanical substitution in my package.
4. **Two placement cases or three.** Recommend dropping `InternalStream`,
   since it cannot fire over a valid Network. If C1 keeps it, say what a
   consumer does with it, because the derivation would then be returning a
   case that only appears alongside a `validate_network` failure.
5. **Where `OperandPlacement` lives.** Recommend a new canonical
   `finn.dataflow.placement`, so the derivation is testable without `ops` and
   `ops/association.py` becomes purely the reporting value S2-A owns. The
   alternative — putting it in `network.py` — would work but grows the module
   that Design compilation imports.

Two things C1 should note that are *not* mine to decide:

- the `OpInput(operand=...)` namespace correction (§6) is S2-A's surface, but
  the schema is unusable without it;
- whether the embedded dot-product Kernel keeps its own Region family (§7.8).

## 13. Follow-up prompt I would need

> Implement the C1-accepted semantic core in
> `{{SEMANTICS_WORKTREE}}` from the reconciled C1 revision. You own
> `src/finn/dataflow/region.py`, `src/finn/dataflow/region_validation.py`, the
> new `src/finn/dataflow/placement.py`, and
> `tests/dataflow/test_region_primitives.py`,
> `tests/dataflow/test_region_validation.py`,
> `tests/dataflow/test_placement.py`. Do not edit `network.py`,
> `network_validation.py`, `designs/`, `kernels/`, `ops/`, or `artifacts/`;
> report any change you believe they need instead of making it.
>
> Land, as separate commits where imports stay valid:
> (1) `LocalStateInput`, the `DataflowRegion.local_state` field and
> `local_state_input()`, with the canonical sort and type checks;
> (2) the two (or four) new `validate_region` codes;
> (3) `placement.py` with `External`, `LocalState`, `OperandPlacement`,
> `PlacementError`, `derive_input_placement`, `derive_output_placement`,
> `placed_shape`, and tests covering: the three MVAU supply modes built from
> the production constructors, zero entry sites, two entry sites, an operand id
> shared by three sites in the decoupled Network, and a local-state input coinciding
> with an output operand.
> (4) delete `prototypes/gate2_s1c/`.
>
> Evidence: `tests/dataflow/test_region_*`, `tests/dataflow/test_network_*`,
> `tests/dataflow/test_package_boundaries.py`, `tests/dataflow/designs`,
> `tests/dataflow/model`, plus `ruff check`, `ruff format --check` and
> `env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow`.
> Add `finn.dataflow.placement` to the `_assert_fresh_import_avoids` list.
> Do not push.
