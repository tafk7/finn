# S1-C — dataflow-model redesign, revised for C1

*Third pass. Base revision `546538087`, branch
`work/dataflow-gate2-s1-semantics`. Nothing under `src/` or `tests/` is
modified; the only files are this directory.*

Inputs: `dataflow-gate2-simplification-c0-decisions.md` §7,
`dataflow-gate2-semantic-requirements-design-note.md`, the C1 feedback on the
first two passes, and `scratchpad/dataflow/canon/REGION.md`.

---

## 1. Recommendation

```python
# src/finn/dataflow/region.py
@dataclass(frozen=True)
class RegionInput:
    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]  # element type changes; signature does not
    outputs: tuple[OutputInterface, ...]
```

Today an input is `(Port, ScheduledInputRequirements)`, so requirements cannot
exist without a port and an operand no port presents is absent from the value
entirely. The fix is to make the **port** the optional part, not the
requirements.

Everything else follows: exposure is a derived comparison, source mappings are
references into the Region model and may be plural, and no placement value is
stored anywhere.

---

## 2. Framing

```text
functional model        what mathematical function the source operation computes
dataflow model          what logical data is required and produced, on what
                        schedule, through which interfaces and Network
                        connections
physical realization    which modules, memories, protocols and timing realize
                        that dataflow
artifact system         how those physical requirements are built, stored and
                        measured
```

`DataflowRegion` and `DataflowNetwork` *are* the dataflow model — the connecting
contract between the source operation and the physical realization. They do not
own the operation's mathematics and must not own storage placement. "Semantic"
appears below only where logical facts are contrasted with physical ones.

The first pass crossed that line: it went from "the embedded compute Region does
not mention W" straight to a `LocalStateInput` collection, which is a statement
about where data lives.

---

## 3. The missing fact

Two different kinds of fact, welded into one value today:

```text
requirement    this Region requires positions of operand W at these schedule
               points with these multiplicities
exposure       this Region presents some or all of W through this ordered
               stream interface, in this beat order
```

`REGION.md` already treats them as separable:

```text
§3.7   "a boundary sequence can present that position once, repeatedly, or not
        at all when declared local state supplies it"   -- per position
§5.2.1 "The selected input beat sequences and declared local state can satisfy
        every canonical input-requirement occurrence"   -- jointly
§3.7   inputs deliberately have no domain/image equality, unlike outputs
```

and `AUTHORING.md` §4.2, having written `required_W`, adds: *"Embedded weights
can be local state and omit the input port. A streamed-weight alternative
retains the input interface and declares its own beat sequence."* Same
requirement in both alternatives; only the port differs.

"Declared local state" stays where the canon puts it — with the **binding
witness**, whose options §5.2 lists as "streaming, replay, parameter memory,
constant operands". This redesign does not move it. It moves the *requirement*,
so that "what does this Region consume" has an answer either way.

### 3.1 Findings that stand

- `construct_embedded_dot_product_region` (`kernels/dotp_axi.py:239`) filters
  the weight interface out of the streamed Region — port, operand and
  requirements together — so the Region no longer says the arithmetic consumes
  a matrix.
- `construct_cyclic_parameter_region` (`parameters/cyclic/region.py:17`)
  declares zero inputs and one output whose every position is available at the
  sole schedule point. Read as a value, it produces a 64×64 matrix from nothing.
- `_internal_destination` (`ops/mvau/op.py:613`) therefore guesses: it scans for
  an input *port* whose id equals the source member name, and otherwise returns
  state of the node literally named `"compute"`. In the decoupled case the scan
  finds the downstream consumer first (`run.py` §2).

### 3.2 Two things the earlier passes got wrong

**`local_state.operand_also_streamed` rejects a canonical Region.** The rule
treated "streamed" and "local state" as mutually exclusive classifications of a
whole operand. `REGION.md` §3.7 permits the mix at position granularity.
`run.py` §4 builds a Region whose `W` port presents half the required positions
and shows the rule rejecting it.

**The multi-port argument was not evidence.** The second pass recommended two
separate collections — requirements and ports — on the strength of one operand
presented by two ports, and cited `REGION.md` §5.1 condition 3 ("within one
region, operands with the same identity have the same element type and shape")
as presuming that recurrence. The replay Region's `X` — an input port and an
output port — fully accounts for condition 3. It says nothing about the input
side. That argument is withdrawn; §4 replaces it with a judgement.

---

## 4. Why one collection, not two

The structural claim the recommendation makes is **at most one input port per
operand**. Not 1:1 — zero is the embedded and parameter-source case, and the
output side is untouched, which is why the replay Region's `X` on both an input
and an output remains legal.

The alternative — `input_requirements` and `input_ports` as separate collections
(`candidates.SplitRegion`) — holds one operand on several ports. The question is
whether that case is worth its permanent cost.

**Where the case would come from.** Wide parameters exceeding a stream width is
packing: physical, one port, a wider beat. Double-pumping is physical. The one
genuine driver is two suppliers feeding one matrix — two memstreams, two edges,
one `W` — which is the MLO/shared-storage direction the feedback explicitly told
me to treat as an extension seam and not over-claim. Using a future topology to
buy a permanent structural cost is the mistake the second pass made.

**What splitting an operand would cost instead.** If two channels ever carry one
tensor and we model them as `W_lo` and `W_hi`, the source mapping has to describe
a *partition of one ONNX tensor across several dataflow operands* — new
vocabulary in a layer that currently needs none. That, not "two suppliers exist",
is the trigger to widen; it is recorded as `candidates.WIDENING_TRIGGER`.

**The asymmetry decides it.**

```text
choose one collection and be wrong later
    port: Port | None = None   ->   ports: tuple[Port, ...] = ()
    item.port is None          ->   not item.ports
    one beat image             ->   the union of the ports' beat images
    one equality check         ->   the same check in a loop
    a mechanical widening of one field inside one existing type

choose two collections and never need it
    two collections joined by operand id before anything can be said about an
    input, on every read, forever
    a rule that every port's operand is declared -- structurally impossible when
    the two live in one value
    a requirement map keyed by operand while REGION.md §3.1 keys it by
    interface: a change to the canon's notation, not just to a definition
```

The second cost is paid continuously for a case with no instance. The first is
paid once, if the case arrives. `run.py` §6 triggers the refusal on demand
(`MultiPortLimit`) so the risk is executable rather than argued.

### 4.1 The other schemas

**The first pass's local-state form** — rejected by §3.2: it classifies whole
operands, incompatibly with `REGION.md` §3.7, and states no requirements for the
operand it names.

**Adjacent companion metadata (D)** — turns one value into a pair at nine seams
read off the current code (`alternatives.D_THREADING_SITES`). The disqualifying
one is `designs/design.py:_correspondence_constraint`, which exists to prove a
Network node holds its segment's Region; with a companion, a Design that
resolved a Region and a mismatched companion would be structurally valid. It is
also worse now than it was in the first pass: the companion has to carry the
requirements, so it is half the Region rather than a small annotation.

**Separate disposition graph (E)** — authors what the recommendation derives.
`alternatives.py` writes a disposition table that is well-formed and false;
catching it needs the derivation underneath, so E is the recommendation plus a
table that can disagree with it.

---

## 5. Exact schema

### 5.1 `src/finn/dataflow/region.py`

```python
@dataclass(frozen=True)
class RegionInput:
    """One operand the Region requires, and at most one port presenting it.

    ``requirements`` is never optional.  A streamed input and an unported one
    are equally complete statements about the computation -- which positions, at
    which schedule points, how often -- and differ only in whether an ordered
    channel carries any of them.

    Not storage.  A ``None`` port says no ordered channel presents this operand
    in this factorization; it names no memory, technology, slot, image or
    module, and a physical choice can never add or remove one.
    """

    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None

    @property
    def id(self) -> str: ...  # the operand id
    @property
    def occurrence_count(self) -> int: ...
    @property
    def required_positions(self) -> frozenset[Coordinate]: ...
    @property
    def presented_positions(self) -> frozenset[Coordinate]: ...
    @property
    def unpresented_positions(self) -> frozenset[Coordinate]: ...


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]  # canonical order: by operand id
    outputs: tuple[OutputInterface, ...]  # canonical order: by port id

    @property
    def input_ports(self) -> tuple[Port, ...]: ...  # the ports that exist
    @property
    def ports(self) -> tuple[Port, ...]: ...
    def input(self, operand_id: str) -> RegionInput: ...
    def input_port(self, port_id: str) -> Port: ...
    def output_interface(self, port_id: str) -> OutputInterface: ...
```

`InputInterface` is deleted; `RegionInput` replaces it. The name change is
deliberate — a value that may have no port should not be called an interface.

`OutputInterface` is untouched. Availability stays port-local because
`REGION.md` §3.7 binds it to the port's beat image (`domain(available_k) =
image(beat_k)`) and explicitly refuses the mirror equality for inputs. Changing
the input side and leaving the output side alone *implements* the canon's
asymmetry rather than inventing one.

`Port` keeps its `operand`: a port is the complete external account of one pass
(§3.5) and an edge compares two of them without a Region in hand. The apparent
duplication with `RegionInput.operand` is closed by a **local** check inside one
value — which is the concrete advantage of one collection over two.

`unpresented_positions` is derived and never stored. It is a report, not a
classification: empty does not mean "streamed" and full does not mean
"embedded".

### 5.2 `src/finn/dataflow/mapping.py` (new)

```python
@dataclass(frozen=True, slots=True)
class RegionInputRef:
    node_id: str
    operand_id: str


@dataclass(frozen=True, slots=True)
class RegionOutputRef:
    node_id: str
    operand_id: str


DataflowOperandRef = RegionInputRef | RegionOutputRef


class MappingError(ValueError): ...


def derive_input_mappings(net: DataflowNetwork, operand_id: str) -> tuple[RegionInputRef, ...]: ...
def derive_output_mappings(
    net: DataflowNetwork, operand_id: str
) -> tuple[RegionOutputRef, ...]: ...
def exposing_ports(net: DataflowNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]: ...
def exposing_boundaries(
    net: DataflowNetwork, ref: DataflowOperandRef
) -> tuple[BoundaryContract, ...]: ...
```

Imports `finn.dataflow.region` and `finn.dataflow.network` only.

**No stored placement value.** `External`, `LocalState` and `InternalStream` are
all withdrawn. Exposure is answered on demand with values that already exist:
`exposing_ports` returns `RegionEndpoint`, `exposing_boundaries` returns the
`BoundaryContract` itself. Nothing repackages a canonical record into a
near-identical one.

`InternalStream` does not survive as a source-entry case. Internal stream
continuation *is* the `Edge`; an unfed, unexposed port is a `validate_network`
failure or an unfulfilled requirement, not a source location.

### 5.3 The derivation rule

> A source input operand's dataflow targets are the Region inputs for that
> operand that the Network does not itself supply. An input is supplied
> internally when it has a port and that port is the sink of an edge.

Mirrored for outputs: a produced operand is a target unless every port emitting
it is an edge source.

| case | `derive_input_mappings(net, "W")` |
|---|---|
| external | `(RegionInputRef('compute','W'),)`, exposed by boundary `weight` |
| embedded | `(RegionInputRef('compute','W'),)`, exposed by no port |
| decoupled | `(RegionInputRef('memory','W'),)`, exposed by no port |
| plural | `(RegionInputRef('compute_a','W'), RegionInputRef('compute_b','W'))` |

The decoupled compute Region still *requires* `W`; its requirement is fed by the
`weight_supply` edge, so the source tensor corresponds to the memory Region's
requirement. Where the memory's unpresented positions come from is the binding's
question and is answered nowhere in this model.

---

## 6. Validation

Not "two new rules". One condition changes what it ranges over, the port-shaped
conditions learn to skip an input with no port, and two rules are new.
`dataflow_model.validate_region` implements the whole list, and `run.py` §3
asserts that production and the recommendation report the **same codes** for the
same Region broken the same way.

| REGION.md §5.1 | status | over what |
|---|---|---|
| 1 schedule extents, level names | unchanged | schedule |
| 2 port identity uniqueness | unchanged | the ports that exist |
| 2b input operand uniqueness | **new** | `inputs` |
| 2c a port presents its input's operand | **new** | `inputs` with a port |
| 3 operand identity / type / shape / width / extents | **expanded** | every input's operand **and** output port operands |
| 4 beat field domain | unchanged | the ports that exist |
| 5 requirement iteration domain, position domain, non-negative multiplicity | unchanged in form | now reached for an input with no port |
| 6 availability domains | unchanged | outputs |
| 7 beat positions | unchanged | the ports that exist |
| 8 availability/image equality | unchanged | outputs |

New codes: `input.operand_duplicate`, `input.port_operand_mismatch`.

Condition 2b is what makes "at most one input port per operand" structural
rather than merely observed: two inputs for one operand would be two requirement
maps for one computation's use of it, with no defined relation between them.

Why condition 3's expansion matters, demonstrated rather than asserted
(`run.py` §3b): declare the embedded Region's `W` with a zero extent and the
recommendation reports `operand.extent_not_positive`; today's embedded Region
reports nothing, because the operand is not in the value to be validated.

Deliberately **not** added: any rule comparing required occurrences against
presented fields. `REGION.md` §3.7 refuses that equality for inputs and §5.2
makes joint satisfaction a binding-realizability obligation with a witness, not
a structural one. `unpresented_positions` reports the gap; nothing rejects it.

`network_validation.py` needs one change: `_resolve_port` reaches inputs through
`region.input_port(...)` instead of `region.input_interface(...).port`.
`endpoint.input_ownership` still ranges over *ports*, so an unported input is
correctly outside it and no exemption is written.

---

## 7. The source mapping contract

```python
@dataclass(frozen=True, slots=True)
class OperandMapping:
    source_operand: str  # the OpInput/OpOutput member name
    tensor: str  # the ONNX tensor
    correspondence: CoordinateMapping
    targets: tuple[DataflowOperandRef, ...]  # plural


@dataclass(frozen=True, slots=True)
class SourceMapping:
    scope_id: str
    source_node: str
    family: str
    family_version: str
    operands: tuple[OperandMapping, ...]
    origin_nodes: tuple[str, ...] = ()
```

It answers *which selected dataflow requirement or product corresponds to this
source operand*. It does not answer where bytes are installed, and carries no
Kernel, module, artifact, storage, path or occurrence.

```text
zero targets      an error -- MappingError
one target        the ordinary case
several targets   explicit and legal; two Regions initialized from one source
                  tensor are two mappings
singularity       an OpInput whose operation contract requires exactly one may
                  say so on its own declaration; the dataflow model does not
```

The singularity flag is `ops/schema.py`'s surface and belongs to S2-A. MVAU
would set it on every operand; that is MVAU's contract, not the model's.

**Matching namespace.** `OpInput(index=1, operand="W")` names a Region
`Operand.id`. MVAU's operands are `X`/`W`/`Y`, its ports are
`activation`/`weight`/`output`, and its source members are
`activation`/`weight`/`output`; today's matching compares source names against
*port ids* and works only because two of those namespaces coincide. Rename a
port and nothing breaks; rename an operand and the mapping fails loudly.

---

## 8. What remains for the physical binding

Stated by the dataflow model, per Region and operand: which positions are
required, at which schedule points, with what multiplicity; the operand's
element type and shape; which positions the port presents and in what beat
order; whether a boundary exposes that port or an edge feeds it; and
consequently, by subtraction, the size and shape of the gap.

`run.py` §7:

```text
embedded compute   occurrences 16384   unpresented positions 4096   INT8 (64, 64)
decoupled memory   occurrences  4096   unpresented positions 4096   INT8 (64, 64)
partial service    occurrences    24   unpresented positions    4   INT8 (2, 4)
```

Left entirely to the binding and to U6: which service covers each unpresented
occurrence (embedded ROM, shared off-chip memory, constant generation, replay
register, parameter memory, or another supported mechanism); storage datatype,
packing, alignment, addressing, banking; access conflict and bandwidth analysis;
any generated external interface; and the §5.2 binding-realizability witness
itself. None of these has a field, an enum case or a reserved name in the
dataflow model.

### 8.1 MLO — an extension seam, narrowly

What this model gives a future shared-memory realization: per-Region,
per-operand scheduled requirement maps (so access pattern, occurrence count and
per-schedule-point demand are computable), operand element type and shape, and
plural source mappings so several Regions' requirements can be recognised as
naming one source tensor.

What it does **not** establish: shared addressing, common or per-tensor storage
datatypes, packing and alignment, access conflicts, bandwidth budgets, or one
generated external interface. Those need a physical MLO value keyed on
`RegionInputRef` plus the source tensor summary. The claim tested here is only
that such a value can be written *without* changing any Region.

---

## 9. Canon fold

`REGION.md` §2.3 currently defines

```text
InputInterface_k = (Port_k, ScheduledInputRequirements_k)
```

which is why requirements cannot exist without a port. The amendment:

```text
§2.3   InputInterface_k = (Operand_k, ScheduledInputRequirements_k, Port_k?)
§3.1   required_k : I x P_k -> N        -- UNCHANGED
§5.1   port-shaped conditions (2, 4, 7) range over interfaces that have a port
§5.1   new: one interface per operand; a port presents its interface's operand
§3.7   cross-reference the declaration, not only §5.2
```

`required_k` is untouched because there is still exactly one operand per
interface, so `P_k` still means what it meant. That is a direct benefit of one
collection: the two-collection form would have re-keyed the requirement map by
operand and rewritten §3.1's notation.

`AUTHORING.md` §4.2 needs no change; it already writes `required_W` keyed by
operand and already describes the embedded alternative as omitting the port.

I own no file under `scratchpad/dataflow/canon/` and have edited none. S4
carries the fold; C1 should record the amendment as accepted before S2 begins,
because S2-B compiles Designs against the changed value.

---

## 10. Type and authority accounting

| | added | removed | net |
|---|---|---|---|
| **recommended** | `RegionInput`, `RegionInputRef`, `RegionOutputRef` | `InputInterface`, `BoundaryDestination`, `StreamDestination`, `RegionStateDestination`, `OperandDestination` | **−2** |
| two collections | `InputRequirement`, and the same two refs | the same five | −2, plus a permanent join and one extra rule |
| local-state form | `LocalStateInput`, `External`, `LocalState`, `InternalStream` | the three destinations | +1, and rejects a canonical Region |
| companion (D) | a companion value, a paired Region, a second `ValueSemantics` | none | +3, and a pair that can disagree |
| disposition graph (E) | `SemanticRequirement`, `RequirementDisposition`, `DispositionKind`, `DispositionGraph` | none | +4, one duplicating `Operand` |

`DataflowRegion` keeps its three fields, their names and their order; only the
element type of `inputs` changes.

Authorities: one. Each requirement is declared once, each port is declared once
beside it, and every other statement — exposure, gap, boundary, mapping — is
computed. No stored derived value survives anywhere in the proposal.

---

## 11. Prototype evidence

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```

Exits 0; final line `all assertions passed`.

- **§1** source mappings for external / embedded / decoupled, with exposure
  derived as `RegionEndpoint` and `BoundaryContract`;
- **§2** the production `_internal_destination` answer for decoupled, asserted
  to be `StreamDestination('compute','weight')`, beside
  `RegionInputRef('memory','W')`;
- **§3** zero issues on every lifted production Region; the same Region broken
  the same way yields the *same code set* from production `validate_region` and
  from the recommendation; the round trip back to today's value silently loses
  `compute.W` and `memory.W`;
- **§3b** operand validation reaches an unported operand — extent 0 caught,
  invisible today;
- **§3c** the two new rules and the expanded identity rule fire on constructed
  negatives;
- **§4** a canonical Region with partial service both ways — `X` required 24
  times and presented 8, `W` required over 8 positions and presented over 4 —
  validates clean, and the first pass's `local_state.operand_also_streamed`
  rejects it;
- **§5** one source tensor mapping to two Regions' requirements, plural, no
  error;
- **§6** the multi-port refusal triggered on demand, with the widening cost and
  its trigger printed;
- **§7** the requirement/gap accounting and the binding boundary;
- **§8** schema D reaching the same answer through a companion at nine seams,
  and schema E's well-formed lying disposition table;
- **§9** type accounting.

The five forcing cases in `cases.py` lift the production constructors —
`construct_dot_product_region`, `construct_activation_replay_region`,
`construct_weight_stream_region` — and change only what is proposed. The
embedded case is the streamed Region with `drop_ports=("weight",)`: one argument
where today there is a separate constructor.

Suite status on this revision, unchanged by this submission:

```
tests/dataflow (minus kernels/, artifacts/, parity/)   786 passed
tests/dataflow/kernels, artifacts, parity              not collected: the only
                                                       interpreter here with
                                                       numpy lacks msgspec and
                                                       pyslang
```

Ruff check and format are clean over the prototype. Strict mypy was not run on
the prototype for the `PYTHONPATH`/qonnx reason CLAUDE.md documents; the
production implementation runs the real gate.

---

## 12. Migration consequences

**Mine after C1** — `region.py`, `region_validation.py`, one function in
`network_validation.py`, the new `mapping.py`, and their tests.

**Cross-boundary, and it needs authorization.** `DataflowRegion`'s signature is
unchanged — same three fields, same order, same names — so there is no
constructor migration and no keyword-only cutover. What changes is the element
type of `inputs`. Measured on this revision: 45 `InputInterface` references and
33 `input_interface(` calls across 14 files in
`src/finn/dataflow/{designs,kernels,network_validation,ops/mvau,parameters}` and
`tests/dataflow/{designs,kernels,model,ops,parameters}`, most owned by S2-A and
S2-B. Each is mechanical: `InputInterface(port, requirements)` becomes
`RegionInput(port.operand, requirements, port)`, and `interface.port` accesses
must tolerate `None`. `cases.lift` performs exactly this rewrite on the real
production Regions, losslessly, so the migration is demonstrated before it is
requested.

The implementation plan already provides for this shape of change (S1-B's
`Variant` rename is "an atomic downstream cutover"). I propose one atomic commit
that changes the value and migrates every site with no semantic change to any
Region, followed by the owners' real work on top.

**S2-A (`ops`) inherits:** `_internal_destination` and `_selected_shape` delete;
`association` becomes `mapping` over `derive_input_mappings` /
`derive_output_mappings`; `OperandAssociation.destination` becomes
`OperandMapping.targets`; the three destination classes are deleted, not
aliased; `OpInput` gains `operand=` naming a Region `Operand.id`, and optionally
a singularity flag.

**S2-B / S3 inherit:** `construct_embedded_dot_product_region` becomes the
streamed constructor with the weight port set to `None` and the requirement
kept, and **its docstring must be rewritten** — it currently asserts "The matrix
itself is not an operand here", which the accepted model contradicts.
`construct_cyclic_parameter_region` gains the requirement for the operand it
emits. `tests/dataflow/ops/test_dataflow_op.py:914` — *"a decoupled matrix is
traffic and an embedded one is state"* — changes: the decoupled weight now maps
to the memory Region's requirement. The distinction it defended (embedded
exposes no port) survives as `exposing_ports(...) == ()`.

**Unaffected**, verified by reading rather than assumed: `designs/design.py`
`_network_property` and `_correspondence_constraint` copy and compare whole
Region values and name no Region field; `model/semantics.py` registers
`DataflowRegion` as `immutable_nominal`, so equality, hashing and fingerprinting
absorb the change with no engine work.

---

## 13. What I need from C1

1. **Accept the `REGION.md` amendment in §9** — one line in §2.3 plus the §5.1
   wording. Without it the implementation and the canon disagree about whether a
   requirement can exist without a port.
2. **Authorize the atomic cutover in §12** — one commit touching 14 files owned
   mostly by S2-A and S2-B, mechanical, no semantic change — or say the owners
   should do it themselves and I will ship the value and its tests only.

Not mine to decide, but the schema is unusable without them: `OpInput(operand=)`
naming a Region `Operand.id`, and the optional per-declaration singularity flag.

---

## 14. Follow-up implementation prompt

> Implement the C1-accepted dataflow-model change in `{{SEMANTICS_WORKTREE}}`
> from the reconciled C1 revision.
>
> You own `src/finn/dataflow/region.py`, `src/finn/dataflow/region_validation.py`,
> the `_resolve_port` change in `src/finn/dataflow/network_validation.py`, the
> new `src/finn/dataflow/mapping.py`, and `tests/dataflow/test_region_primitives.py`,
> `tests/dataflow/test_region_validation.py`,
> `tests/dataflow/test_network_validation.py`, `tests/dataflow/test_mapping.py`.
>
> C1 has authorized one atomic cutover commit that also mechanically migrates
> every `InputInterface(...)` construction and every `region.inputs` /
> `input_interface(...)` accessor in `src/` and `tests/`, including those in
> `designs/design.py`, `kernels/`, `ops/mvau/`, `parameters/` and their tests.
> `InputInterface(port, requirements)` becomes
> `RegionInput(port.operand, requirements, port)`; accessors that reach a port
> must tolerate `None`. Change no Region's meaning in that commit, and let every
> migrated test assert what it asserted before. Beyond that migration do not
> edit `designs/`, `kernels/`, `ops/` or `artifacts/`; report any further change
> you believe they need instead of making it.
>
> Land, in order:
> 1. `RegionInput` with `operand`, `requirements` and `port: Port | None`, the
>    derived position properties in §5.1, `DataflowRegion.inputs` retyped,
>    `input()` / `input_port()` / `input_ports` / `ports`, `InputInterface`
>    deleted, and the mechanical migration.
> 2. `validate_region` per the table in §6 — the expanded operand condition, the
>    port-shaped conditions skipping unported inputs, and the two new codes —
>    plus `_resolve_port`.
> 3. `mapping.py` with `RegionInputRef`, `RegionOutputRef`, `DataflowOperandRef`,
>    `MappingError`, `derive_input_mappings`, `derive_output_mappings`,
>    `exposing_ports`, `exposing_boundaries`. No stored placement value.
> 4. Delete `prototypes/gate2_s1c/`.
>
> Tests must cover: the three MVAU supply modes built from the production
> constructors; a Region whose port presents a subset of required positions; a
> Region whose port presents every position but not every occurrence; zero
> mappings raising `MappingError`; two mappings returned as two; an unported
> operand caught by every operand rule; two inputs for one operand; a port whose
> operand differs from its input's.
>
> Evidence: `tests/dataflow/test_region_*`, `tests/dataflow/test_network_*`,
> `tests/dataflow/test_mapping.py`, `tests/dataflow/test_package_boundaries.py`,
> `tests/dataflow/designs`, `tests/dataflow/model`, `tests/dataflow/parameters`,
> plus `ruff check`, `ruff format --check`, and
> `env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow`.
> Add `finn.dataflow.mapping` to the `_assert_fresh_import_avoids` list in
> `tests/dataflow/test_package_boundaries.py`. Do not push.
