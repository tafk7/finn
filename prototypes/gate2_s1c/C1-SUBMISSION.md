# S1-C — dataflow-model redesign, revised for C1

*Second pass, after the C1 feedback on the first submission. Base revision
`546538087`, branch `work/dataflow-gate2-s1-semantics`. Nothing under `src/` or
`tests/` is modified; the only files are this directory.*

Inputs: `dataflow-gate2-simplification-c0-decisions.md` §7,
`dataflow-gate2-semantic-requirements-design-note.md`, the C1 feedback, and
`scratchpad/dataflow/canon/REGION.md`.

---

## 1. Framing

Four strata, and this workstream owns exactly one of them:

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

`DataflowRegion` and `DataflowNetwork` *are* the dataflow model. It is the
connecting contract between the source operation and the physical realization:
it does not own the operation's mathematics, and it must not own storage
placement. "Semantic" appears below only where logical facts are being
contrasted with physical ones.

The first submission crossed that line. It went from "the embedded compute
Region does not mention W" straight to `LocalStateInput`, which is a statement
about where data lives. The correct dataflow fact is one step earlier and one
step more general.

---

## 2. The actual missing fact

Today the two facts are welded into one value:

```python
InputInterface = (Port, ScheduledInputRequirements)
```

so an operand with no port has no requirements, and a Region that consumes it
says nothing at all. But they are different kinds of fact:

```text
requirement    this Region requires positions of operand W at these schedule
               points with these multiplicities
exposure       this Region presents some or all of W through this ordered
               stream interface, in this beat order
```

`REGION.md` already treats them as separable and already says so three times:

```text
§3.7   "a boundary sequence can present that position once, repeatedly, or not
        at all when declared local state supplies it"   -- per position
§5.2.1 "The selected input beat sequences and declared local state can satisfy
        every canonical input-requirement occurrence"   -- jointly
§3.7   inputs deliberately have no domain/image equality, unlike outputs
```

and `AUTHORING.md` §4.2, having written `required_W`, adds: *"Embedded weights
can be local state and omit the input port. A streamed-weight alternative
retains the input interface and declares its own beat sequence."* The
requirement is the same declaration in both alternatives. Only the port differs.

The canon leaves "declared local state" to the **binding witness** — §5.2 lists
"streaming, replay, parameter memory, constant operands" as the witness's
options. That is the right home for it and this redesign does not move it. What
the redesign moves is the *requirement*, out of the interface, so that the
question "what does this Region consume" has an answer whether or not a port
exposes it.

So the redesign question is:

> How should `DataflowRegion` represent scheduled logical input requirements
> independently of stream-interface exposure?

### 2.1 Three findings from the first pass that stand

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
  finds the downstream consumer first. Reproduced in `run.py` §2.

### 2.2 One finding the first pass got wrong

The proposed rule `local_state.operand_also_streamed` — an operand may not be
both streamed and local state — contradicts `REGION.md` §3.7, which permits
exactly that at position granularity. `run.py` §4 builds a canonical Region
whose `W` port presents half the required positions and shows that rule
rejecting it. "Streamed" and "locally supplied" are not classifications of an
operand at all; they are a per-occurrence comparison between what the ports
present and what the requirement asks for, and the difference is the binding's
to cover.

---

## 3. Candidates

### A — requirements and ports as separate collections *(recommended)*

```python
@dataclass(frozen=True)
class InputRequirement:
    operand: Operand
    requirements: ScheduledInputRequirements


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    input_requirements: tuple[InputRequirement, ...]
    input_ports: tuple[Port, ...]
    outputs: tuple[OutputInterface, ...]
```

One requirement per `(Region, Operand)`. Ports say which positions cross a
boundary and in what order. Exposure is a derived comparison, never a stored
classification.

### B — one Region input with optional stream exposure

```python
@dataclass(frozen=True)
class RegionInput:
    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None
```

Compact, and it holds every current FINN Region, the embedded case, the
decoupled case **and** partial service — `run.py` §6 builds the partial-service
Region under candidate B successfully, so partial exposure is *not* what
separates them.

What separates them is one operand presented by **two** ports: a matrix
delivered by two suppliers, or a tile split across two channels.
`region_b_from_requirements` raises on it:

```text
CandidateBLimit: operand 'W' is presented by 2 input ports (w_hi, w_lo);
                 RegionInput holds one
```

The workarounds are worse than the problem. Splitting into `W_lo`/`W_hi` loses
the single requirement map and the single source correspondence — the source
tensor would map to two operands that are not the operand. Widening `port` to a
tuple makes candidate B into candidate A with the collections nested one level
deeper, and then `RegionInput` is a grouping with no invariant of its own.

`REGION.md` §5.1 condition 3 — *"within one region, operands with the same
identity have the same element type and shape"* — exists precisely because an
operand identity may recur across interfaces. Candidate B makes that condition
unreachable on the input side.

**A over B, on that case.** If C1 judges multi-port supply out of scope, B is
smaller by one collection and everything else in this document is unchanged.

### C — the first submission's local-state form

Rejected by §2.2 above: it classifies whole operands, and the classification is
incompatible with `REGION.md` §3.7. It also states no requirements for the
operand it names, which §5 of the feedback correctly refuses.

### D — adjacent companion metadata; E — separate disposition graph

Unchanged from the first submission, and no new evidence has appeared.

D turns one value into a pair at nine seams read off the current code
(`alternatives.py:D_THREADING_SITES`); the disqualifying one is
`designs/design.py:_correspondence_constraint`, which exists to prove a Network
node holds its segment's Region — with a companion, a Design that resolved a
Region and a mismatched companion would be structurally valid.

E authors what A derives, and `alternatives.py` writes a disposition table that
is well-formed and false. Catching that needs A's derivation underneath, so E is
A plus a table that can disagree with it.

---

## 4. Recommended normalized schema

### 4.1 `src/finn/dataflow/region.py`

```python
@dataclass(frozen=True)
class InputRequirement:
    """What one Region's computation logically requires of one operand.

    Stated once per operand, not once per port: ``required(i, p)`` counts the
    computation's uses, and a computation does not use a position twice because
    two ports happen to deliver it.  Whether any port presents those positions
    is a separate fact, and covering what the ports do not present is the
    binding's obligation under REGION.md 5.2, not a field here.
    """

    operand: Operand
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")

    @property
    def id(self) -> str:
        return self.operand.id


@dataclass(frozen=True, kw_only=True)
class DataflowRegion:
    schedule: LogicalSchedule
    input_requirements: tuple[InputRequirement, ...]
    input_ports: tuple[Port, ...]
    outputs: tuple[OutputInterface, ...]

    # canonical order: requirements by operand id, input ports by port id,
    # outputs by port id.

    @property
    def ports(self) -> tuple[Port, ...]: ...
    def input_requirement(self, operand_id: str) -> InputRequirement: ...
    def input_port(self, port_id: str) -> Port: ...
    def output_interface(self, port_id: str) -> OutputInterface: ...
    def ports_for(self, operand_id: str) -> tuple[Port, ...]: ...
    def presented_positions(self, operand_id: str) -> frozenset[Coordinate]: ...
    def unpresented_positions(self, operand_id: str) -> frozenset[Coordinate]: ...
```

Four consequences, each deliberate:

**`InputInterface` is deleted.** With requirements gone it holds only a port,
and a one-field wrapper is the ladder this effort removed. An input interface
*is* a port. `region.inputs` becomes `region.input_ports` so every call site
fails loudly rather than silently changing type.

**`OutputInterface` is not touched.** Availability stays port-local because
`REGION.md` §3.7 binds it to the port's beat image (`domain(available_k) =
image(beat_k)`) and explicitly refuses the mirror equality for inputs. Moving
requirements out and leaving availability in *implements* the canon's asymmetry
rather than inventing one. If outputs ever need the same split — several ports
emitting one produced operand — it is the same change again, and it is not
needed by any case here.

**`Port` keeps its `operand`.** A port is the complete external account of one
pass (§3.5) and an edge compares two of them without a Region in hand. The
apparent duplication with `InputRequirement.operand` is closed by validation,
not by a reference: `operand.identity_conflict` now ranges over requirement
operands too, which is the generalization of §5.1 condition 3.

**Keyword-only construction.** The field list changes shape, so all 36
positional `DataflowRegion(...)` sites must be edited regardless. Making the
canonical constructor keyword-only means the next field addition reorders
nothing. This is the opposite of the first submission's reasoning, which put a
field last to avoid edits; the feedback is right that migration convenience is
not an architectural argument.

### 4.2 `src/finn/dataflow/mapping.py` (new)

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

There is **no stored placement value**. `External`, `LocalState` and
`InternalStream` are all withdrawn. Exposure is answered on demand and answered
with values that already exist: `exposing_ports` returns `RegionEndpoint`,
`exposing_boundaries` returns the `BoundaryContract` itself. A caller that wants
the boundary's beat sequence or pass correspondence already has it, and nothing
repackages a canonical record into a near-identical one.

`InternalStream` in particular does not survive as a source-entry case. Internal
stream continuation *is* the `Edge`; an unfed, unexposed port is a
`validate_network` failure or an unfulfilled requirement, not a source location.

### 4.3 The derivation rule

> A source input operand's dataflow targets are the input requirements for that
> operand that the Network does not itself supply. A requirement is supplied
> internally when it has at least one exposing port and *every* exposing port is
> the sink of an edge.

Mirrored for outputs: a produced operand is a target unless every port emitting
it is an edge source.

Checked in `run.py` §1 and §5:

| case | `derive_input_mappings(net, "W")` |
|---|---|
| external | `(RegionInputRef('compute', 'W'),)`, exposed by boundary `weight` |
| embedded | `(RegionInputRef('compute', 'W'),)`, exposed by no port |
| decoupled | `(RegionInputRef('memory', 'W'),)`, exposed by no port |
| plural | `(RegionInputRef('compute_a','W'), RegionInputRef('compute_b','W'))` |

The decoupled compute Region still *requires* `W`; its requirement is fed by
the `weight_supply` edge, so the source tensor corresponds to the memory
Region's requirement. That is a dataflow statement. Where the memory's
unpresented positions come from is the binding's question and is not answered
anywhere in this model.

---

## 5. Validation

This is not "two new rules". Two existing conditions change what they range
over, two are new, and the rest are unchanged. `dataflow_model.validate_region`
implements the whole list, and `run.py` §3 asserts that production and candidate
A report the *same codes* for the same Region broken the same way.

| REGION.md §5.1 | status | over what |
|---|---|---|
| 1 schedule extents, level names | unchanged | schedule |
| 2 port identity uniqueness | unchanged | input ports + output ports |
| 2b requirement operand uniqueness | **new** | `input_requirements` |
| 3 operand identity / type / shape / width / extents | **expanded** | requirement operands **and** port operands |
| 4 beat field domain | unchanged | all ports |
| 5 requirement iteration domain, position domain, non-negative multiplicity | **expanded** | keyed by operand; now reached for unported operands |
| 5b every input port exposes a declared requirement | **new** | input ports |
| 6 availability domains | unchanged | outputs |
| 7 beat positions | unchanged | all ports |
| 8 availability/image equality | unchanged | outputs |

New codes: `requirement.operand_duplicate`, `input_port.requirement_missing`.
Moved paths: requirement codes are now keyed `input_requirements['W']` rather
than `input['weight']`.

Why condition 3's expansion matters, demonstrated rather than asserted
(`run.py` §3b): declare the embedded Region's `W` with a zero extent and
candidate A reports `operand.extent_not_positive`; today's embedded Region
reports nothing, because the operand is not in the value to be validated.

Deliberately **not** added: any rule comparing required occurrences against
presented fields. `REGION.md` §3.7 refuses that equality for inputs, and §5.2
makes joint satisfaction a binding-realizability obligation with a witness, not
a structural one. `unpresented_positions()` reports the gap; nothing rejects it.

`network_validation.py` needs one change only: `_resolve_port` reaches inputs
through `region.input_port(...)` instead of `region.input_interface(...).port`.
`endpoint.input_ownership` still ranges over *ports*, so an unported requirement
is correctly outside it and no exemption is written.

---

## 6. The source mapping contract

Renaming `SourceAssociation` and reducing it, per C0 §8:

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
source operand*. It does not answer where bytes are installed, and it carries no
Kernel, module, artifact, storage, path or occurrence.

Cardinality rules:

```text
zero targets      an error -- MappingError, the Network has no unfed
                  requirement for the operand
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

## 7. What remains for the physical binding

Stated by the dataflow model, per Region and operand:

- which positions are required, at which schedule points, with what
  multiplicity;
- the operand's element type and shape;
- which ports present which positions, in what beat order;
- which of those ports a boundary exposes or an edge feeds;
- consequently, by subtraction, the size and shape of the gap.

`run.py` §7 prints the gap for three cases:

```text
embedded compute   occurrences 16384   unpresented positions 4096   INT8 (64, 64)
decoupled memory   occurrences  4096   unpresented positions 4096   INT8 (64, 64)
partial service    occurrences    24   unpresented positions    4   INT8 (2, 4)
```

Left entirely to the binding and to U6:

- which service covers each unpresented occurrence — embedded ROM, shared
  off-chip memory, constant generation, replay register, parameter memory, or
  another supported mechanism;
- storage datatype, packing, alignment, addressing, banking;
- access conflict and bandwidth analysis;
- any generated external interface;
- the §5.2 binding-realizability witness itself.

None of these has a field, an enum case or a reserved name in the dataflow
model.

### 7.1 MLO — an extension seam, narrowly

What this model gives a future shared-memory realization: per-node, per-operand
scheduled requirement maps (so access pattern, occurrence count and
per-schedule-point demand are computable), operand element type and shape, and
plural source mappings so several Regions' requirements can be recognised as
naming one source tensor.

What it does **not** establish, and the first submission overstated: shared
addressing, common or per-tensor storage datatypes, packing and alignment,
access conflicts, bandwidth budgets, or one generated external interface. Those
need a physical MLO value keyed on `RegionInputRef` plus the source tensor
summary. The claim here is only that such a value can be written *without*
changing any Region — `run.py` §5 shows two Regions' requirements naming one
tensor, and nothing was written back.

---

## 8. Canon fold — this requires a REGION.md amendment

C1 must accept a canon change, not only an implementation change. `REGION.md`
currently indexes the requirement map **by interface**:

```text
§2.3   InputInterface_k = (Port_k, ScheduledInputRequirements_k)
§3.1   required_k : I x P_k -> N
§5.1.5 "For every input interface, required_k is total on I x P_k"
```

Under the recommendation it is indexed by operand:

```text
§2.1   DataflowRegion = (S, InputRequirements, InputPorts, Outputs)
§2.3   an input interface is a Port; requirements are declared per operand
§3.1   required_O : I x P_O -> N
§5.1.3 operand rules range over requirement operands and port operands
§5.1.5 "For every input requirement, required_O is total on I x P_O"
§5.1   new: one requirement per operand; every input port's operand is declared
§3.7   cross-reference the declaration, not only §5.2
```

Per-interface indexing is not merely inconvenient: with two ports for one
operand it is ambiguous, because `required_1` and `required_2` over the same
`P_W` have no defined relation to the computation's actual use. Per-operand
indexing removes the ambiguity and matches §3.1's own wording — *"the number of
logical uses of operand position `p` at iteration point `i`"* — which is a
statement about the computation, not about a channel.

`AUTHORING.md` §4.2 needs no change; it already writes `required_W` keyed by
operand.

I do not own `scratchpad/dataflow/canon/` and have edited nothing there. S4
carries the fold; C1 should record the amendment as accepted before S2 begins,
because S2-B compiles Designs against the changed value.

---

## 9. Type and authority accounting

| | added | removed | net |
|---|---|---|---|
| A *(recommended)* | `InputRequirement`, `RegionInputRef`, `RegionOutputRef` | `InputInterface`, `BoundaryDestination`, `StreamDestination`, `RegionStateDestination`, `OperandDestination` | **−2** |
| B | `RegionInput` | `InputInterface`, and the same four | −4, at the cost of §3 |
| C (local state) | `LocalStateInput`, `External`, `LocalState`, `InternalStream` | the same three destinations | +1, and rejects a canonical Region |
| D (companion) | `RegionResidency` + a paired Region + a second `ValueSemantics` | none | +3, and a pair that can disagree |
| E (disposition) | `SemanticRequirement`, `RequirementDisposition`, `DispositionKind`, `DispositionGraph` | none | +4, one duplicating `Operand` |

Authorities under A: one. Requirements are declared once per operand, exposure
is declared once per port, and every other statement — placement, gap, boundary,
mapping — is computed. No stored derived value survives anywhere in the
proposal.

---

## 10. Prototype evidence

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```

Exits 0; final line `all assertions passed`. Sections:

- **§1** source mappings for external / embedded / decoupled, with exposure
  derived as `RegionEndpoint` and `BoundaryContract`;
- **§2** the production `_internal_destination` answer for decoupled, asserted
  to be `StreamDestination('compute','weight')`, beside candidate A's
  `RegionInputRef('memory','W')`;
- **§3** candidate A reports zero issues on every lifted production Region; the
  same Region broken the same way yields the *same code set* from production
  `validate_region` and candidate A; the round trip back to today's value
  silently loses `compute.W` and `memory.W`;
- **§3b** operand validation reaches an unported operand — extent 0 caught by A,
  invisible today;
- **§3c** the two new rules and the expanded identity rule fire on constructed
  negatives;
- **§4** a canonical Region with partial service both ways — `X` required 24
  times and presented 8, `W` required over 8 positions and presented over 4 —
  validates clean, and the first submission's
  `local_state.operand_also_streamed` rejects it;
- **§5** one source tensor mapping to two Regions' requirements, plural, no
  error;
- **§6** one operand on two ports: candidate A validates clean, candidate B
  raises `CandidateBLimit`, and candidate B *does* hold the partial-service
  Region — so multi-port supply is the discriminator, not partial exposure;
- **§7** the requirement/gap accounting and the binding boundary;
- **§8** schema D reaching the same answer through a companion at nine seams,
  and schema E's well-formed lying disposition table;
- **§9** type accounting.

Suite status on this revision, unchanged by this submission:

```
tests/dataflow (minus kernels/, artifacts/, parity/)   786 passed
tests/dataflow/kernels, artifacts, parity              not collected: the only
                                                       interpreter here with
                                                       numpy lacks msgspec and
                                                       pyslang
```

`git status` shows `prototypes/` as the only addition. Ruff check and format are
clean over the prototype. Strict mypy was not run on the prototype for the
`PYTHONPATH`/qonnx reason CLAUDE.md documents; the production implementation
runs the real gate.

---

## 11. Migration consequences

**Mine after C1** — `region.py`, `region_validation.py`, `network_validation.py`
(one function), new `mapping.py`, and their tests.

**Cross-boundary, and it needs C1's authorization.** Deleting `InputInterface`
and making the constructor keyword-only is a destructive change. Measured on
this revision: 36 `DataflowRegion(...)` constructions, 33 `input_interface(`
calls and 45 `InputInterface` references, across 14 files in
`src/finn/dataflow/{designs,kernels,network_validation,ops/mvau,parameters}` and
`tests/dataflow/{designs,kernels,model,ops,parameters}` — most of them owned by
S2-A and S2-B. The implementation plan already provides for this shape of change
(S1-B's `Variant` rename is "an atomic downstream cutover"). I propose the same:
one atomic commit that changes the value and mechanically migrates every
constructor and accessor, with no semantic change to any Region, followed by the
owners' real work on top. If C1 prefers, the alternative is a defaulted
additional field and a deprecated `inputs` property — which is exactly the
positional-compatibility reasoning the feedback rejected, so I am not
recommending it.

**S2-A (`ops`) inherits:** `_internal_destination` and `_selected_shape` delete;
`association` becomes `mapping` over `derive_input_mappings` /
`derive_output_mappings`; `OperandAssociation.destination` becomes
`OperandMapping.targets`; the three destination classes are deleted, not
aliased; `OpInput` gains `operand=` naming a Region `Operand.id`, and optionally
a singularity flag.

**S2-B / S3 inherit:** `construct_embedded_dot_product_region` becomes the
streamed constructor with the weight *port* dropped and the requirement kept —
`cases.py` does this in one argument — and **its docstring must be rewritten**,
since it currently asserts "The matrix itself is not an operand here", which the
accepted model contradicts. `construct_cyclic_parameter_region` gains the
requirement for the operand it emits. `tests/dataflow/ops/test_dataflow_op.py:914`
— *"a decoupled matrix is traffic and an embedded one is state"* — changes: the
decoupled weight now maps to the memory Region's requirement. The distinction it
defended (embedded exposes no port) survives as `exposing_ports(...) == ()`.

**Unaffected**, verified by reading rather than assumed: `designs/design.py`
`_network_property` and `_correspondence_constraint` copy and compare whole
Region values and name no Region field; `model/semantics.py` registers
`DataflowRegion` as `immutable_nominal`, so equality, hashing and fingerprinting
absorb the shape change with no engine work.

---

## 12. What I need from C1

1. **Candidate A or candidate B** — A unless multi-port supply for one operand
   is out of scope.
2. **Accept the `REGION.md` amendment in §8** (requirements keyed by operand),
   or reject it, in which case candidate A must be re-derived with per-interface
   requirements and the multi-port case becomes unrepresentable.
3. **Authorize the atomic cutover** in §11, or choose the deprecated-property
   alternative.
4. **`input_ports` / `input_requirements` naming**, and confirmation that
   `InputInterface` is deleted rather than kept as a one-field wrapper.
5. **`finn.dataflow.mapping` as the module home** for the refs and derivations,
   leaving `ops/association.py` to become S2-A's reporting value.

Not mine to decide, but the schema is unusable without them: `OpInput(operand=)`
naming a Region `Operand.id`, and the optional per-declaration singularity flag.

---

## 13. Follow-up implementation prompt

> Implement the C1-accepted dataflow-model change in `{{SEMANTICS_WORKTREE}}`
> from the reconciled C1 revision.
>
> You own `src/finn/dataflow/region.py`, `src/finn/dataflow/region_validation.py`,
> the `_resolve_port` change in `src/finn/dataflow/network_validation.py`, the new
> `src/finn/dataflow/mapping.py`, and `tests/dataflow/test_region_primitives.py`,
> `tests/dataflow/test_region_validation.py`,
> `tests/dataflow/test_network_validation.py`, `tests/dataflow/test_mapping.py`.
>
> C1 has authorized one atomic cutover commit that also mechanically migrates
> every `DataflowRegion` construction and every `region.inputs` /
> `input_interface(...)` accessor in `src/` and `tests/`, including the ones in
> `designs/design.py`, `kernels/`, `ops/mvau/`, `parameters/` and their tests.
> Change no Region's meaning in that commit: every migrated constructor must
> produce a Region whose requirements and ports are the ones its interfaces
> held, and every migrated test must assert the same thing it asserted before.
> Beyond that mechanical migration do not edit `designs/`, `kernels/`, `ops/` or
> `artifacts/`; report any further change you believe they need instead of
> making it.
>
> Land, in order:
> 1. `InputRequirement`; `DataflowRegion` as
>    `(schedule, input_requirements, input_ports, outputs)`, keyword-only, with
>    the canonical orders and the accessors in §4.1; `InputInterface` deleted;
>    the mechanical migration of all call sites.
> 2. `validate_region` per the table in §5 — the expanded operand and
>    requirement conditions and the two new codes — plus `_resolve_port`.
> 3. `mapping.py` with `RegionInputRef`, `RegionOutputRef`, `DataflowOperandRef`,
>    `MappingError`, `derive_input_mappings`, `derive_output_mappings`,
>    `exposing_ports`, `exposing_boundaries`. No stored placement value.
> 4. Delete `prototypes/gate2_s1c/`.
>
> Tests must cover: the three MVAU supply modes built from the production
> constructors; a Region whose ports present a subset of required positions; a
> Region whose ports present every position but not every occurrence; one operand
> on two ports; zero mappings raising `MappingError`; two mappings returned as
> two; an unported operand caught by every operand rule; a port presenting an
> undeclared operand; two requirements for one operand.
>
> Evidence: `tests/dataflow/test_region_*`, `tests/dataflow/test_network_*`,
> `tests/dataflow/test_mapping.py`, `tests/dataflow/test_package_boundaries.py`,
> `tests/dataflow/designs`, `tests/dataflow/model`, `tests/dataflow/parameters`,
> plus `ruff check`, `ruff format --check`, and
> `env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow`.
> Add `finn.dataflow.mapping` to the `_assert_fresh_import_avoids` list in
> `tests/dataflow/test_package_boundaries.py`. Do not push.
