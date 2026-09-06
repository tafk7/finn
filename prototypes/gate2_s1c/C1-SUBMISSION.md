# S1-C — final dataflow-model proposal for C1

*Base revision `546538087`, branch `work/dataflow-gate2-s1-semantics`. Nothing
under `src/` or `tests/` is modified; the only files are this directory.*

Inputs: `dataflow-gate2-simplification-c0-decisions.md` §7,
`dataflow-gate2-semantic-requirements-design-note.md`, the three rounds of C1
feedback, and `scratchpad/dataflow/canon/{REGION,AUTHORING}.md`.

---

## 1. Recommendation

```python
# src/finn/dataflow/region.py
@dataclass(frozen=True)
class InputInterface:  # name, fields, constructor unchanged
    port: Port
    requirements: ScheduledInputRequirements

    @property
    def operand(self) -> Operand:
        return self.port.operand


@dataclass(frozen=True)
class UnportedInput:  # the one genuinely new case
    operand: Operand
    requirements: ScheduledInputRequirements


RegionInput = InputInterface | UnportedInput


@dataclass(frozen=True)
class DataflowRegion:  # three fields, same names, same order
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]
    outputs: tuple[OutputInterface, ...]
```

Today a Region input is `(Port, ScheduledInputRequirements)`, so requirements
cannot exist without a port and an operand no port presents is absent from the
value entirely. The fix adds the missing case rather than making the existing
one nullable.

---

## 2. Nullable port versus the sum type

The nullable form `RegionInput(operand, requirements, port: Port | None)`
authors the operand twice whenever a port exists — `item.operand` and
`item.port.operand` — so a Region in which they disagree is constructible and
needs a validation rule to catch it. `run.py` §1 constructs one.

The comparison C1 asked for is consumer and validation change, not type count.
Measured on `546538087`:

| | sum type | nullable |
|---|---:|---:|
| `InputInterface(...)` constructions to rewrite | **0** | 20 |
| `region.input_interface(port_id)` calls to rewrite | **0** | 33 |
| sites reading `.port` off a Region input | 15 | 15 |
| validation rules for operand/port agreement | **0** | 1 |
| new public dataclasses | 1 | 1 |

The sum type keeps `InputInterface`'s name, fields and construction order, so
those 53 sites are untouched — `cases.lift` performs the whole rewrite and the
ported branch is the identity for every current FINN Region. The 15 `.port`
reads change under either shape, and change *better* under the sum type: mypy
narrows `isinstance(item, InputInterface)` to a non-optional `Port`, where the
nullable form yields `Port | None` and needs a guard for the same reason with
none of the discrimination.

**Recommend the sum type.** It is smaller at the sites that already exist, it
deletes a rule rather than adding one, and it makes ported-versus-unported
explicit at the point of reading.

One consequence worth stating: `region.inputs` becomes a union, so every
existing iteration that reaches `.port` fails type checking until it discriminates.
That is the desired failure mode — mypy enumerates the 15 sites rather than
letting an unported input reach code that assumes a port.

---

## 3. Framing

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

`DataflowRegion` and `DataflowNetwork` *are* the dataflow model. They do not own
the operation's mathematics and must not own storage placement.

Two kinds of fact, welded into one value today:

```text
requirement    this Region requires positions of operand W at these schedule
               points with these multiplicities
exposure       this Region presents some or all of W through this ordered
               stream interface, in this beat order
```

`REGION.md` already treats them as separable — §3.7 ("a boundary sequence can
present that position once, repeatedly, or **not at all** when declared local
state supplies it", per position), §5.2 condition 1 (beat sequences and local
state satisfy the requirement occurrences *jointly*), and §3.7's deliberate
absence of the output-side domain/image equality for inputs. `AUTHORING.md` §4.2
writes `required_W` once and then says the embedded alternative omits the port.

"Declared local state" stays where the canon puts it: with the **binding
witness**, whose options §5.2 lists as "streaming, replay, parameter memory,
constant operands". This proposal does not move it, and the words do not appear
in the value.

### 3.1 Findings that stand

- `construct_embedded_dot_product_region` (`kernels/dotp_axi.py:239`) filters the
  weight interface out of the streamed Region — port, operand and requirements
  together — so the Region no longer says the arithmetic consumes a matrix.
- `construct_cyclic_parameter_region` (`parameters/cyclic/region.py:17`) declares
  zero inputs and one output whose every position is available at the sole
  schedule point: read as a value, it produces a 64×64 matrix from nothing.
- `_internal_destination` (`ops/mvau/op.py:613`) therefore guesses — it scans for
  an input *port* whose id equals the source member name, and otherwise returns
  state of the node literally named `"compute"`. In the decoupled case the scan
  finds the downstream consumer first (`run.py` §3).

---

## 4. What `OperandMapping.targets` means

**One meaning, chosen and stated: every dataflow requirement or product in the
selected Network that this source operand corresponds to.** Correspondence.
Not source-entry, not provisioning.

`derive_input_mappings(net, operand_id)` returns every Region input requiring
that operand, whether a port presents it, whether an edge feeds that port, and
whether anything supplies it at all. No filtering happens inside the mapping,
because a filtered set is a different question and giving both the same name is
how the meaning drifted in the earlier passes.

For decoupled MVAU the three questions separate cleanly:

```text
mathematical correspondence
    the source weight corresponds to compute's W requirement and to memory's
    W requirement -- both Regions require W, and it is the same tensor
    -> derive_input_mappings(net, "W") == (compute.W, memory.W)

dataflow provenance
    the Network supplies all 4096 of compute's required positions through the
    weight_supply edge, and none of memory's
    -> internally_supplied_positions / externally_supplied_positions /
       unsupplied_positions

physical provisioning
    initializer bytes are installed in the memory realization
    -> U6.  Not in this model, not derivable from it, and not named by it.
```

The earlier pass's rule — "targets are the requirements the Network does not
feed" — silently answered the provenance question under the correspondence
name. §5 shows a case where that rule also gets provenance itself wrong.

### 4.1 Provenance is position-granular, never a boolean

Three disjoint sets per referenced requirement, whose union is its required
position set:

```python
internally_supplied_positions(net, ref)  # presented by a port an edge feeds
externally_supplied_positions(net, ref)  # presented by a port a boundary exposes
unsupplied_positions(net, ref)  # required, presented by no port
```

They are **position**-granular, not occurrence-granular, and that is a canon
consequence, not a convenience. `REGION.md` §3.7 refuses any required-versus-
presented equality for inputs precisely because a position presented once may be
required three times — the multi-visit kernels the deadlock proofs name
(softmax, LayerNorm) declare re-reads as multiplicity, not as extra beats. Such
a tensor **has** entered the construction; serving the re-reads is the binding's
business. A position no port presents has **not** entered, and something must
supply it. `run.py` §5 shows both in one Region: `X` required 24 times and
presented 8, fully covered; `W` required over 8 positions and presented over 4,
half owed.

---

## 5. The partial internally-fed case

`cases.partial_internal_network` — `compute` requires all four positions of `W`;
its `w_hi` port **is** the sink of the `weight_supply` edge; and that port
presents two of the four.

```text
W @ compute   internal 2   external 0   owed 2
W @ memory    internal 0   external 0   owed 2
```

Under a boolean `fed_internally = endpoint in edge_sinks`, compute's requirement
is dropped from the mapping entirely and the two positions the Network does not
supply disappear with it. That is the concrete failure the reviewer predicted,
and it is why the boolean is gone: an edge feeding a port says nothing about
whether the port presents everything the requirement asks for.

The full case set, all in `cases.py` over the production constructors where they
exist:

| case | `derive_input_mappings(net, "W")` | provenance |
|---|---|---|
| external | `(compute.W,)` | external 4096, exposed at boundary `weight` |
| embedded | `(compute.W,)` | owed 4096 |
| decoupled | `(compute.W, memory.W)` | compute internal 4096; memory owed 4096 |
| partial internal | `(compute.W, memory.W)` | compute internal 2 / owed 2; memory owed 2 |
| plural target | `(compute_a.W, compute_b.W)` | both owed |

The activation is plural too, and honestly so: `derive_input_mappings(decoupled,
"X")` returns replay's `X` (boundary-fed) and compute's `X` (edge-fed). One
tensor, two requirements, told apart by provenance rather than by filtering the
mapping.

---

## 6. Plural targets and operand-id collisions

```text
zero targets      MappingError
one target        the ordinary case
several targets   valid, and returned as several
singularity       an OpInput whose operation contract requires exactly one may
                  say so on its own declaration
```

The tuple is ordered by node id for determinism of the *return value* only.
Nothing is selected by ordering — the earlier `_internal_destination` behaviour
is exactly what that forbids.

**Collisions.** `Operand.id` is Region-local, so two unrelated Regions may each
call something `W`. Three answers, and I recommend the first two together:

1. **A Network-scoped operand identity rule.** `REGION.md` §5.1 condition 3
   already says one operand identity means one type and shape *within* a Region.
   Lift it to the Network: `network.operand_identity_conflict`. This is what
   makes `Operand.id` usable as the matching namespace at all, and it catches
   unrelated tensors that differ in type or shape (`run.py` §6).
2. **Declaration-side qualification for the rest.** Two unrelated tensors that
   agree on type *and* shape are invisible to any structural rule. The answer is
   an optional qualifier on the source declaration —
   `OpInput(index=1, operand="W", node="memory")` — which is `ops/schema.py`'s
   surface and belongs to S2-A. It is an escape hatch, not the normal path.
3. A stable operation-wide operand-id convention across a Design's alternatives
   is what (1) enforces in practice; the Design author owns every Region in
   their Network, so the convention is enforceable at authoring time.

Not an answer: node ordering.

---

## 7. Requirements are mandatory

`UnportedInput.requirements` is a required field, as is
`InputInterface.requirements`. A ported input and an unported one are equally
complete statements about the computation — which positions, at which schedule
points, how often.

The measured cost is recorded and does not change the recommendation:

```text
R=1  64x64    PE=SIMD=8    4,096 entries
R=4  64x64    PE=SIMD=8   16,384 entries
R=1  1024x1024 PE=SIMD=32  1,048,576 entries
R=8  512x512  PE=SIMD=16   2,097,152 entries
```

These identify a scaling concern in the explicit `ScheduledInputRequirements`
relation, which the streamed Region already pays today. The correction, if it is
ever needed, is a compact canonical requirement representation — a named rule or
profile that normalizes to `required`, which `REGION.md` §3.1 already
contemplates ("Construction profiles can use structured occurrence coordinates
… but they normalize to `required_k`"). **Recorded as separate future work.** It
is not a reason to drop positions, schedule points or multiplicities.

---

## 8. The one-port limitation

The recommendation's structural claim is **at most one input port per operand**,
enforced by `input.operand_duplicate`. Zero is the embedded and parameter-source
case. The output side is untouched, so the replay Region's `X` on an input port
*and* an output port stays legal.

Acceptable for the first implementation: no current forcing case needs more, and
the second pass's argument for multi-port (that `REGION.md` §5.1 condition 3
presumes recurrence) was wrong — the replay Region's `X` fully accounts for that
condition on the *output* side and says nothing about inputs.

**Widening is not guaranteed to be mechanical**, and the earlier characterization
was too optimistic. `candidates.MULTI_PORT_WIDENING`:

```text
mechanical
    port: Port -> ports: tuple[Port, ...]
    one beat image -> the union of the ports' beat images

not mechanical -- a service/partition relation the model does not have
    do two ports' position sets have to be disjoint, or may they overlap?
    if they overlap, is a position delivered twice, or is one delivery
        authoritative?
    is there an order across streams, or are the ports independent?
    which occurrences does which interface serve, when the requirement map
        counts uses and the ports count deliveries?
```

`REGION.md` §3.7 refuses a required-versus-presented equality for one port. With
several the question is joint rather than repeated, and §5.2's realizability
witness would have to speak about interfaces rather than about one beat
sequence. So widening introduces a relation, not just a field.

**Trigger** (`candidates.MULTI_PORT_TRIGGER`): not "two suppliers exist" — two
suppliers can always be modelled as two operands. The trigger is that modelling
them as two operands forces `OperandMapping` to describe a *partition of one
ONNX tensor across several dataflow operands*, which is new vocabulary in a
layer that currently needs none.

`run.py` §8 raises `MultiPortLimit` on demand, so the refusal is executable
rather than argued.

---

## 9. Validation

Complete list. One condition changes what it ranges over, the port-shaped
conditions range over the ports that exist, and one rule is new.
`dataflow_model.validate_region` implements all of it, and `run.py` §7 asserts
that production and the recommendation report the **same codes** for the same
Region broken the same way.

| REGION.md §5.1 | status | over what |
|---|---|---|
| 1 schedule extents, level names | unchanged | schedule |
| 2 port identity uniqueness | unchanged | the ports that exist |
| 2b one input per operand | **new** (`input.operand_duplicate`) | `inputs` |
| 3 operand identity, datatype, shape, positive bit width, positive extents | **expanded** | every input's operand **and** output port operands |
| 4 beat field domain | unchanged | the ports that exist |
| 5 requirement iteration domain, position domain, non-negative multiplicity | unchanged in form | now reached for an unported input |
| 6 availability domains | unchanged | outputs |
| 7 beat positions | unchanged | the ports that exist |
| 8 availability/image equality | unchanged | outputs |

`input.port_operand_mismatch` **does not exist**: under the sum type a ported
input's operand *is* its port's operand, so disagreement is unrepresentable
rather than reportable. That is the rule the nullable form would have needed.

Why condition 3's expansion matters, demonstrated rather than asserted
(`run.py` §7b): declare the embedded Region's `W` with a zero extent and the
recommendation reports `operand.extent_not_positive`; today's embedded Region
reports nothing, because the operand is not in the value to be validated.

Deliberately **not** added: any rule comparing required occurrences against
presented fields, for the §4.1 reason.

Network validation gains one rule — `network.operand_identity_conflict` (§6) —
and changes one function: `_resolve_port` reaches inputs through
`region.input_interface(...)`, whose signature is unchanged because a port
lookup can only ever find a ported input. `endpoint.input_ownership` continues
to range over actual ports only, so an unported input is correctly outside it
and no exemption is written.

---

## 10. The source mapping contract

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


@dataclass(frozen=True, slots=True)
class OperandMapping:
    source_operand: str  # the OpInput/OpOutput member name
    tensor: str  # the ONNX tensor
    correspondence: CoordinateMapping
    targets: tuple[DataflowOperandRef, ...]
```

`src/finn/dataflow/mapping.py` (new) holds the refs, `MappingError`, the two
`derive_*` functions, the three position queries and:

```python
def exposing_ports(net, ref) -> tuple[RegionEndpoint, ...]
def exposing_boundaries(net, ref) -> tuple[BoundaryContract, ...]
```

Imports `finn.dataflow.region` and `finn.dataflow.network` only.

**No stored placement value.** `External`, `LocalState`, `BoundaryPlacement`
and `InternalStream` are all absent. Exposure is answered on demand with values
that already exist — `RegionEndpoint` and the `BoundaryContract` itself — and
the position queries return plain `frozenset`s. Nothing repackages a canonical
record into a near-identical one, so there is no `BoundaryPlacement` and no
`UnportedInputRef`: whether a referenced requirement has a port is a property of
the Region, queryable, not a second ref type.

`InternalStream` does not survive as a source-entry case. Internal stream
continuation *is* the `Edge`, and it now shows up where it belongs — as
`internally_supplied_positions`.

**Matching namespace.** `OpInput(index=1, operand="W")` names a Region
`Operand.id`. MVAU's operands are `X`/`W`/`Y`, its ports are
`activation`/`weight`/`output`, its source members are
`activation`/`weight`/`output`; today's matching compares source names against
*port ids* and works only because two of those namespaces coincide.

---

## 11. What remains for the physical binding

Stated by the dataflow model, per Region and operand: which positions are
required, at which schedule points, with what multiplicity; the operand's
element type and shape; which positions a port presents and in what beat order;
whether an edge feeds that port or a boundary exposes it; and by subtraction the
positions still owed.

```text
embedded compute   occurrences 16384   owed positions 4096   INT8 (64, 64)
decoupled memory   occurrences  4096   owed positions 4096   INT8 (64, 64)
partial internal   occurrences     8   owed positions    2   INT8 (2, 2)
```

Left entirely to the binding and to U6: which service covers the owed positions
(embedded ROM, shared off-chip memory, constant generation, replay register,
parameter memory, or another supported mechanism); storage datatype, packing,
alignment, addressing, banking; access conflict and bandwidth analysis; any
generated external interface; and the §5.2 realizability witness itself. None of
these has a field, an enum case or a reserved name in the dataflow model.

### 11.1 MLO — an extension seam, narrowly

What this model gives a future shared-memory realization: per-Region,
per-operand scheduled requirement maps (so access pattern, occurrence count and
per-schedule-point demand are computable), operand element type and shape, the
owed-position sets, and plural correspondence so several Regions' requirements
can be recognised as naming one source tensor.

What it does **not** establish: shared addressing, common or per-tensor storage
datatypes, packing and alignment, access conflicts, bandwidth budgets, or one
generated external interface. Those need a physical MLO value keyed on
`RegionInputRef` plus the source tensor summary. The claim tested here is only
that such a value can be written without changing any Region.

---

## 12. Canon amendment

Not a one-line edit — the previous submission characterized it wrongly. It is a
small conceptual refactor with five parts.

```text
1. normalized Region input definition   §2.3
   InputInterface_k = (Port_k, ScheduledInputRequirements_k)
       becomes a sum:
   RegionInput_k = InputInterface_k | UnportedInput_k
   InputInterface_k = (Port_k, ScheduledInputRequirements_k)
   UnportedInput_k  = (Operand_k, ScheduledInputRequirements_k)

2. terminology                          §2.1, §2.3, §5.1
   "input interface" no longer names every Region input.  "Region input" is the
   union; "input interface" keeps its current meaning for the ported case.

3. scope of structural operand validation   §5.1 condition 3
   ranges over every Region input's operand, not over port operands only.

4. port-specific rule domains           §5.1 conditions 2, 4, 7
   range over the ports that exist, so an unported input contributes none.

5. requirement-versus-exposure explanation   §3.1, §3.7
   §3.1's required_k : I x P_k -> N is UNCHANGED -- there is still exactly one
   operand per Region input, so P_k means what it meant.  §3.7 gains a
   cross-reference to the declaration rather than pointing only at §5.2.
```

`AUTHORING.md` §4.2 needs no change; it already writes `required_W` and already
describes the embedded alternative as omitting the port.

I own no file under `scratchpad/dataflow/canon/` and have edited none. S4
carries the fold; C1 should record the amendment as accepted before S2 begins,
because S2-B compiles Designs against the changed value.

---

## 13. Type and authority accounting

| | added | removed | net |
|---|---|---|---|
| **recommended** | `UnportedInput`, `RegionInputRef`, `RegionOutputRef` (`RegionInput` is a union alias) | `BoundaryDestination`, `StreamDestination`, `RegionStateDestination`, `OperandDestination` | **−1** |
| nullable port | `RegionInput`, the same two refs | `InputInterface`, the same four | −2, plus one validation rule and 53 rewritten sites |
| local-state form | `LocalStateInput`, `External`, `LocalState`, `InternalStream` | the three destinations | +1, and rejects a canonical Region |
| companion (D) | a companion value, a paired Region, a second `ValueSemantics` | none | +3, and a pair that can disagree |
| disposition graph (E) | 4 values, one duplicating `Operand` | none | +4, and the table can lie |

`InputInterface` and `DataflowRegion`'s three fields, their names and their
order are all kept.

Authorities: one. Each requirement is declared once, each port is declared once
with it, and every other statement — exposure, provenance, owed positions,
boundary, mapping — is computed. No stored derived value survives anywhere.

D and E are unchanged in substance and worse in this round: D's companion now
has to carry the requirements, making it half the Region rather than a small
annotation; and because correspondence is plural, E's table has to restate the
whole correspondence set. `run.py` §10 runs both.

---

## 14. Prototype evidence

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```

Exits 0; final line `all assertions passed`.

- **§1** the nullable-vs-sum migration table, and a `NullableInput` whose operand
  disagrees with its port's, constructed;
- **§2** source mappings for external / embedded / decoupled with the three
  position sets printed per target;
- **§3** the production `_internal_destination` answer for decoupled, asserted;
- **§4** the partial internally-fed case: an edge-fed port presenting half;
- **§5** positions versus occurrences, and the first pass's rule rejecting a
  canonical Region;
- **§6** plural targets, and the Network-scoped operand collision rule firing;
- **§7 / 7b / 7c** validation parity with production, the expanded operand rule
  reaching an unported operand, and the one new rule;
- **§8** `MultiPortLimit` and the non-mechanical widening questions;
- **§9** the binding boundary;
- **§10** schemas D and E on the same cases;
- **§11** type accounting, and the plural activation correspondence.

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

## 15. Migration accounting

**Mine after C1** — `region.py`, `region_validation.py`, `network_validation.py`
(one function plus one rule), the new `mapping.py`, and their tests.

**Cross-boundary.** `DataflowRegion`'s signature is unchanged and
`InputInterface`'s constructor is unchanged, so the 20 constructions and 33
`input_interface(` lookups need no edit. What changes:

```text
15   sites iterating a Region's inputs and reaching .port
      -> discriminate with isinstance(item, InputInterface); mypy --strict
         enumerates every one of them
 2   constructors that currently express "no weight" by deleting the interface
      -> construct_embedded_dot_product_region, construct_cyclic_parameter_region
 7   InputInterface type annotations / isinstance checks / imports
      -> widen to RegionInput where they mean "any input"
```

Most of the 15 are in files owned by S2-A and S2-B. Each is mechanical and
type-checked; none changes any Region's meaning. I propose one commit that lands
the value, its validation and its tests, and performs those discriminations —
or, if C1 prefers, I ship the value and its tests and the owners discriminate
their own call sites, at the cost of a red branch in between.

**S2-A (`ops`) inherits:** `_internal_destination` and `_selected_shape` delete;
`association` becomes `mapping` over `derive_input_mappings` /
`derive_output_mappings` plus the position queries; `OperandAssociation.destination`
becomes `OperandMapping.targets`; the three destination classes are deleted, not
aliased; `OpInput` gains `operand=` naming a Region `Operand.id`, plus the
optional node qualifier and singularity flag from §6.

**S2-B / S3 inherit:** `construct_embedded_dot_product_region` becomes the
streamed constructor with the weight interface turned into an `UnportedInput`
carrying the same requirements, and **its docstring must be rewritten** — it
currently asserts "The matrix itself is not an operand here", which the accepted
model contradicts. `construct_cyclic_parameter_region` gains the requirement for
the operand it emits. `tests/dataflow/ops/test_dataflow_op.py:914` — *"a
decoupled matrix is traffic and an embedded one is state"* — changes: the
decoupled weight now corresponds to both compute's and memory's `W`
requirements, told apart by provenance. The distinction it defended survives as
`exposing_ports(...) == ()`.

**Unaffected**, verified by reading: `designs/design.py` `_network_property` and
`_correspondence_constraint` copy and compare whole Region values and name no
Region field; `model/semantics.py` registers `DataflowRegion` as
`immutable_nominal`, so equality, hashing and fingerprinting absorb the change
with no engine work.

---

## 16. What I need from C1

1. **Accept the sum type** (§1–§2) or direct me to the nullable form.
2. **Accept the canon amendment** in §12 — five parts, small but not one line.
3. **Decide who discriminates the 15 `.port` sites** (§15): me in one commit, or
   their owners.

Not mine to decide, but the schema is unusable without them: `OpInput(operand=)`
naming a Region `Operand.id`, the optional node qualifier for same-shape
collisions, and the optional singularity flag.

---

## 17. Follow-up implementation prompt

> Implement the C1-accepted dataflow model in `{{SEMANTICS_WORKTREE}}` from the
> reconciled C1 revision.
>
> You own `src/finn/dataflow/region.py`,
> `src/finn/dataflow/region_validation.py`,
> `src/finn/dataflow/network_validation.py`, the new
> `src/finn/dataflow/mapping.py`, and `tests/dataflow/test_region_primitives.py`,
> `tests/dataflow/test_region_validation.py`,
> `tests/dataflow/test_network_validation.py`, `tests/dataflow/test_mapping.py`.
>
> C1 has authorized discriminating the ~15 sites in `src/` and `tests/` that
> iterate a Region's inputs and reach `.port`. `InputInterface(...)`
> constructions and `region.input_interface(port_id)` lookups do not change.
> Change no Region's meaning, and let every migrated test assert what it
> asserted before. Beyond those discriminations do not edit `designs/`,
> `kernels/`, `ops/` or `artifacts/`; report any further change you believe they
> need instead of making it.
>
> Land, in order:
> 1. `UnportedInput`, `InputInterface.operand` as a property over its port,
>    `RegionInput` as the union, `DataflowRegion.inputs` retyped, `input()`,
>    `input_ports`, `ports`; `input_interface(port_id)` unchanged in signature.
>    Discriminate the call sites in the same commit.
> 2. `validate_region` per the table in §9 — the expanded operand condition, the
>    port-shaped conditions over the ports that exist, and
>    `input.operand_duplicate`. No operand/port agreement rule. Add
>    `network.operand_identity_conflict` to `validate_network` and update
>    `_resolve_port`.
> 3. `mapping.py` with `RegionInputRef`, `RegionOutputRef`, `DataflowOperandRef`,
>    `MappingError`, `derive_input_mappings`, `derive_output_mappings`,
>    `internally_supplied_positions`, `externally_supplied_positions`,
>    `unsupplied_positions`, `exposing_ports`, `exposing_boundaries`. No stored
>    placement value; `derive_*` returns correspondence and filters nothing.
> 4. Delete `prototypes/gate2_s1c/`.
>
> Tests must cover: the three MVAU supply modes over the production
> constructors; an edge-fed port presenting only part of its requirement, with
> the three position sets asserted; a port presenting every position but not
> every occurrence, asserted as fully supplied; zero targets raising
> `MappingError`; two targets returned as two; an unported operand caught by
> every operand rule; two inputs for one operand; a Network-scoped operand
> identity conflict.
>
> Evidence: `tests/dataflow/test_region_*`, `tests/dataflow/test_network_*`,
> `tests/dataflow/test_mapping.py`, `tests/dataflow/test_package_boundaries.py`,
> `tests/dataflow/designs`, `tests/dataflow/model`, `tests/dataflow/parameters`,
> plus `ruff check`, `ruff format --check`, and
> `env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow`.
> Add `finn.dataflow.mapping` to the `_assert_fresh_import_avoids` list in
> `tests/dataflow/test_package_boundaries.py`. Do not push.
