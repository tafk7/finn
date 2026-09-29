# Plan: kernel authoring — one port, one schedule, checked against the RTL

Date: 2026-09-29. Status: **G0 answered; P0 done and reviewed** (2026-09-29,
[`p0/REPORT.md`](p0/REPORT.md)); its corrections are folded in below. **A1 done,
for review** ([`RECORD.md`](RECORD.md)). Branch
`feature/kernel-authoring`, worktree `/home/tkeller/prj-kernels/finn-kernel-authoring`,
from `347f24f6d` (`feature/kernel-package-extraction` with the code-quality pass
merged). Nothing in `src/` is built yet (A1 is test-side).

## Goal

A hardware engineer integrating an RTL module states its **hardware facts once**;
the model derives or checks everything else. Today an author also writes model
plumbing (extent getters, fold domains, hand-built traversals, per-port idle
widths, `semantics=` on most values) and hand-written checks (shape and element
agreement), and nothing checks the one fact Python cannot see: that the RTL
really walks the loop order the kernel declares.

The target, for a new module (global sum pooling, `accpool_axi`), is:

```python
b, s, c = Index("b"), Index("s"), Index("c")

class AccPoolKernel(Kernel):
    id, version, module = "example.accpool_axi", "1", "accpool_axi"
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    channels = extent_of(c)                              # bound from the ports' tensors
    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def schedule(self) -> Schedule | Rejected:           # the RTL's loop nest
        return self.bound_schedule(beats=(b, s, c), folds={c: self.pe})

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def sum_dtype(self) -> QONNXDataType: ...            # what the RTL emits

    x = AxiStreamPort(name="s_axis_input", endpoint=Endpoint.TARGET, stream=x_stream,
                      schedule=schedule, index=(b, s, c), lanes=(c,), admits=Integer(max_bits=16))
    y = AxiStreamPort(name="m_axis_output", endpoint=Endpoint.INITIATOR, stream=y_stream,
                      schedule=schedule, index=(b, c), lanes=(c,), reduces=(s,), dtype=sum_dtype)

    admission = ConstraintGroup(accumulator_supported)   # the RTL's own limits only
    def parameters(self): ...
    def sources(self): ...

# tests/kernels/test_accpool.py
conformance(AccPoolKernel, inputs={"x_stream": ...}, outputs={"y_stream": (2, 8)},
            reference=lambda x: x.sum(axis=1), folds=SAMPLED)
```

About 40 lines against 69 on today's code ([`sketches/accpool_today.py`](sketches/accpool_today.py)
runs on this branch's base; [`sketches/accpool_proposed.py`](sketches/accpool_proposed.py) is the sketch). What remains is the
datasheet: module and sources, generics, pin names and direction, loop order,
lane order, which interfaces need TLAST, and the RTL's limits.

## Settled in discussion (2026-09-29)

| # | Decision |
|---|---|
| S1 | **One primitive port.** Every kernel interface that carries a scheduled tensor is an `AxiStreamPort`: logical (schedule projection → traversal and TLAST rule), element (producer states, consumer admits) and physical (AXIS packing and pins, or FinnLib's native ready/valid names) in one node. It replaces `StreamPort`, `ScheduledPort` and `GivenPort`. |
| S2 | **`WordPort` stays below it**, for stream stages (`input_gen`, `vpc`, `fifo`): their words are opaque and their sequences come from the plan hop they carry out, not from a schedule. Kernel authors never see it. |
| S3 | **Conformance harness, sampled folds**: per kernel, the smallest fold, the largest, one interior, and one that forces the stream in front to place an adapter. |
| S4 | **Semantics inference**: a `T \| Rejected` return annotation infers the semantics of `T` (the engine knowing its own refusal type, not a registry). Protocol types (QONNX datatypes) stay explicit by design. |
| S5 | **Extents are bound from the ports**: an index that alone addresses a tensor axis takes that axis's extent; two ports disagreeing on an index is a refusal. Probed on today's engine: a kernel's `schedule` can read its own ports' `index` declarations without a cycle ([`sketches/bound_probe.py`](sketches/bound_probe.py), reproduced as a test in A3). |
| S6 | **Producers state their element, consumers admit it, the stream refuses a mismatch** — code-quality C generalized to every port. |

Standing decisions this plan keeps: no families, registries or wrapper Spaces
(refine the core); compatibility filters, DSE later; producers present native
data and receivers adapt; the boundary presentation is a rule; `replay_buffer`
stays scrapped; weights stored `(k, n)`; reduction order value-level; the RTL
checker only refuses and never derives a kernel (`artifacts/rtl.py`).

## Human gate G0: decisions before building (all answered 2026-09-29)

| # | Decision | Recommendation | Blocks |
|---|---|---|---|
| 1 | Name of the primitive | **Answered (user, 2026-09-29): `AxiStreamPort`.** Code-quality A deleted a test-only class of that name; note it in the change log | A4 |
| 2 | The escape hatch for traversals no schedule derives | **Answered (user, 2026-09-29): `sequence=` on `AxiStreamPort` itself**, exactly one of `schedule` or `sequence`, rather than a `GivenPort` subclass | A4 |
| 3 | **Flat (unplaced) kernel builds.** eltwise's and thresholding's module parameters need no tensor extents (streaming modules), so a flat build of them is meaningful; dotp's (MW, MH) need its streams. | **Answered (user, 2026-09-29): keep flat builds where the module needs no extents** (K2 stands). An idle port's lanes derive from the fold of its `lanes` indices; `idle_lanes` goes; a kernel whose `parameters()` need extents is refused unplaced (`kernel-extents`). | A4 |
| 4 | Folds that are `Param`s today (thresholding and eltwise `pe`, transpose's SIMD inside `input_form`) | **Answered (user, 2026-09-29): folding factors are Decisions, always.** A parent may pin or narrow a child's fold (E1's pin-by-key and narrowing), but a standalone or top-level kernel's fold is an independent optimization choice. New keys (`<node>.pe`, `<node>.simd`) are recorded under D7; flat tests commit the fold as a choice (`point_for(..., pe=2)`). Sub-question 4a below. | A5 |
| 4a | Thresholding's PE domain. Its RTL accepts PE dividing C *or* a multiple of C (rows folded into lanes); the stream model cannot present the latter (the S0 gap), so a placed kernel already refuses it, while a flat build with PE = 2C is accepted today (`test_flat_kernels`: `threshold(pe=4)` on 2 channels). | **Answered (user, 2026-09-29): domain = divisors of C** (read from the threshold table, so it is known flat). PE > C returns with the fold-across-rows extension. That flat case becomes a refusal. | A5 |
| 4b | A fold whose extent only a stream gives (eltwise's last axis) in a flat build | **Answered (user, 2026-09-29): as recommended.** Probe P0.5: a Decision whose `divisors_of` domain reads an absent stream's extent. If the engine cannot commit it flat, the domain is the RTL's own bound (1 ≤ PE < 2^32) when unplaced and divisors when placed. | A5 |
| 5 | Where a produced tensor's element comes from, for the compiler | **Reframed (user, 2026-09-29): aim at the future compiler-facing KernelOp, not the provisional `finn.graph` shim.** See *Forward interface: KernelOp*. Consequence here: A6a stays (producers state `dtype`), A6b (a stream taking its element from its producer inside the engine) is dropped, and a rule replaces it: **a producer's `dtype` must not read its own output stream**, so an op can ask a kernel for its output types before the downstream tensor exists. | A6 |

## Design

### D1. Conformance harness (`tests/kernels/conformance.py`)

```python
def conformance(
    family: type[Kernel],
    *,
    inputs: Mapping[str, Tensor],              # reference input -> tensor
    outputs: Mapping[str, tuple[int, ...]],    # reference input -> shape; element from the kernel
    reference: Callable[..., Mapping[str, np.ndarray]],  # named inputs -> named outputs (a future execute)
    folds: Folds = SAMPLED,
    choices: Mapping[str, object] = {},        # the kernel's other Decisions, pinned
    facts: Mapping[str, object] = {},          # the kernel's other Params
) -> None
```

For each sampled fold configuration:

1. **Place** the kernel in a generated `Design` between boundary streams, commit
   the fold and `choices`, `settle` (the adapter chains).
2. **ABI check, no Vivado needed**: `artifacts.rtl.check_abi` on the kernel's
   build requirements against its materialized sources, with the declared
   parameter binding. `Declined` fails the test only under `--strict-rtl`; a
   refusal always fails.
3. **Model check**: every port's traversal covers its tensor (until D3 makes
   this structural), beat counts match `schedule.beat_count` for dropped
   indices, `parameters()` keys equal the RTL's parameter names (from the
   extraction in 2), and, with its outputs unplaced, the kernel still states
   every output element (D5's rule).
4. **XSim** (`requires_xsim`): random integers in each input's range; the
   reference output; each input packed in the order its *boundary* presents
   (`unreplayed` of the port's form) and each output in its port's order
   (`traversal.pack`); `xsim.stream_through`, stalled and free.
5. **The adapter case**: the fourth sample feeds the first input from a
   read-only `MemStreamKernel` presenting `vector_major` at a lane count other
   than the port's, so the stream plans a width conversion and the adapter is
   part of the simulated path.

`Folds`: `SAMPLED` (the four above, deduplicated), `ALL`, or explicit tuples.
Sampling enumerates each scalar fold Decision's domain with
`point.field(<fold>).candidates()` (`compatible_cases` serves Decisions over
nodes only; P0 correction 5), so a refused fold is never sampled. A fold whose
domain is empty while the kernel is flat (eltwise's, G0.4b) is sampled placed. One simulation per
process holds: `simulate` runs the xsim tools in a subprocess.

### D2. Engine rule: `T | Rejected` infers `T` (`core/space/_signatures.py`)

`output_semantics` today raises "output annotation needs explicit semantics="
for any union. New rule: a union whose members are one value type plus any of
the engine's own result markers (`Rejected`, `Inapplicable`, `Unresolved`)
infers the value type's default semantics, as `_answer_value_type` already does
for `QueryResult[T]`. Explicit `semantics=` still overrides and is still
checked against the annotation. Protocols and other unions still need it.

The rule is [`p0/p1_output_semantics.patch`](p0/p1_output_semantics.patch)
(a `_marked_value_type` helper applied after the `QueryResult` branch). It also
closes a gap: today an explicit `semantics=` on a union return is never checked
against its annotation; with the rule it is.

Then remove every explicit `semantics=` the annotation implies: 88 of the 115
in `finn.kernels` and `finn.dataflow` (P0 count, `p0/count_semantics.out`).
Of those, **50 are redundant on today's engine** (every `CLOCKING`, `INDICES`
and `SCHEDULE`, `STAGES`, most `TENSOR`, the 10 plain `BEAT_SEQUENCE`/`TRAVERSAL`)
and **38 need the rule** (`T | Rejected` returns: `TENSOR`, `SCALAR_ENCODING`,
`TRANSPORT`, the adapter facts, 7 `BEAT_SEQUENCE`, ...). Constants that also key
a `ViewKey` (`TIEOFFS_SEMANTICS`, `CONTROL_SEMANTICS`, `CONNECTION_SEMANTICS`,
`PARTS_SEMANTICS`, `MODULE_REQUIREMENTS`, `EXPORTED_SEMANTICS`) stay as
constants; only their `semantics=` uses go. What stays (27): the QONNX
datatypes (a Protocol, 19), `INTEGER_POLICY` (`Integer | None`), and named
integer tensors (`INTEGER_TENSOR`, `INTEGER_VECTOR`, `THRESHOLD_TABLE`). `BEAT_SEQUENCE` and `TRAVERSAL` differ from
their defaults only in `name` (used in messages; compatibility compares
`type_token`) and in snapshotting by identity rather than `deepcopy` (both are
frozen values), so they are deleted with their uses; A2 confirms the identity
dump is unchanged.

Space docs: scratchpad `space/AUTHORING.md` (outputs and semantics) and
`space/MIGRATION.md`, with the documentation examples re-run.

### D3. Extent binding (`finn.dataflow.schedule`)

```python
Access = tuple[Sequence[int], Sequence[Index | Affine]]   # a tensor shape, the port's index

def bind_extents(accesses: Sequence[Access]) -> dict[Index, int]   # raises Refused
```

- An axis addressed by a plain `Index` gives that index its extent.
- An index seen on two axes (one port or two) with different extents is refused.
- An index no axis addresses alone, and no explicit extent gives, is refused
  (`kernel-extents`).
- An axis addressed by an `Affine` of several indices (a sliding window) binds
  nothing; its indices need extents from another axis or from the author
  (`bound_schedule(extents={...})`), and the axis is checked:
  `reach < extent`.
- A port read through a view (`reshaped`) **binds nothing** (its view's
  extents are its indices' extents, so binding from them would be circular). It
  is checked after binding: every index it reads is bound by another port (or
  given), and `prod(view) == prod(shape)`. dotp's `x` under the dense
  realization is bound by `w` (`k`) and `y` (`m`). (P0 correction 2.)
- **Coverage**: an index bound this way walks its whole axis, so a port's
  traversal covers its tensor exactly when every axis is bound. That closes the
  finding that a too-wide tensor settles silently.

### D4. `AxiStreamPort` (`finn.kernels.port`)

One class replacing `StreamPort`, `ScheduledPort` and `GivenPort`:

| Param | Meaning |
|---|---|
| `name`, `endpoint`, `clock`, `reset` | as today |
| `stream` | the stream it sits on (`required=False`: an optional interface) |
| `schedule`, `index`, `lanes`, `reduces`, `holds`, `closes`, `reshaped` | the logical projection, as `ScheduledPort` today |
| `sequence` | the escape hatch (G0.2): a given `BeatSequence`; exactly one of `schedule` or `sequence` |
| `dtype` | a producer's element (an initiator must give it; a target may, to pin it) |
| `admits` | a consumer's integer policy |
| `signals` | FinnLib-native pin names (data, valid, ready) instead of an AXIS bus |

Derived as today: `sequence` (from the schedule unless given), `element`
(`dtype` for a producer, the stream's tensor element for a consumer),
`axis`/`transport`/`pins`, `contract`. An idle port (no stream: an optional interface, or a flat build) takes its
lanes from the folds of its `lanes` indices, known without extents (G0.3);
`idle_lanes` is deleted.

Kernel base (`finn.kernels.base`) gains three helpers, reading only its own
ports' declarations and their streams' tensors:

- **The access export.** Each `AxiStreamPort` with a schedule exports its access
  (its stream's shape, `index`, `reshaped`) under a new `ACCESS` view key; an
  idle port exports nothing. The base collects them with `Members(ACCESS)`, as
  it collects pins with `Members(PINS)`. The export reads Params only, never the
  port's sequence, so it does not cycle. (Review of P0: the probe's class walk
  through the private `_nodes.node_record` is not carried into `src`; A4's first
  test confirms this spelling.)
- `extents` (derived): `bind_extents` over `Members(ACCESS)`; a `Refused`
  becomes `reject("kernel-extents", …)`.
- `bound_schedule(beats, folds, extents=None)`: a `Schedule` over the bound
  extents (plus any explicit ones).
- `extent_of(index)`: a factory returning a derived member that reads
  `extents[index]` (refused as `kernel-extents` when no port binds it), for a
  fold Decision's `divisors_of` domain. **It must be named in the class body**
  (`channels = extent_of(c)`); inline use inside `divisors_of(...)` is refused
  when the model is linked (P0 correction 3). (A fold's domain cannot read the
  schedule, which reads the fold.)
- **Flat folds (G0.4b).** The engine cannot commit a fold whose domain reads an
  absent stream's extent (P0.5). A kernel whose module needs no extents
  (eltwise) takes a domain over `extents` itself: the RTL's own bound
  (`1 <= pe < 2**32`) while unplaced, the divisors once placed. Its name and
  place (a base helper or not) are settled in A4/A5; it is a domain, not a
  family. Every fold is a Decision (G0.4); a kernel whose fold
  extent is a fact of its own (thresholding's channels, from its table) uses that
  fact instead, which keeps its flat build.

### D5. Element flow (producer states, consumer admits)

Every producing `AxiStreamPort` states `dtype`; the stream's `well_formed`
refuses a tensor of another element (`stream-tensor`), as code-quality C does
for `GivenPort`. The producer's element is `dtype`, placed or idle (P0.4: a
3-line override on the port). One mismatch is refused today under two codes,
`stream-tensor` (logical) and `stream-element` (`physical/contract.py`
`compatibility`); A6 keeps the logical one. The `accumulator_fits`-style checks of a type someone else
chose become the producer's own `dtype`.

**Rule:** a producer's `dtype` depends on the kernel's facts, choices and input
elements only, never on its own output stream. Conformance (D1) checks it: the
kernel is built with its outputs unplaced and must still state every output
element. That is what lets a compiler-level op infer output datatypes node by
node (*Forward interface*), which replaces the dropped A6b.

`finn.graph`'s provisional adapter keeps calling `exact_result_dtype` for now;
it now meets the MatMul's stated `dtype` at the stream, so a disagreement is a
refusal rather than a silent second authority.

### D6. Authoring guide (`src/finn/kernels/AUTHORING.md`)

The datasheet-to-model checklist (indices; folds; the loop nest; per interface
the index, lanes, reduces and TLAST), the worked example (`accpool_axi`, model
only: no RTL exists), the loop-order trade-off (accumulators in the kernel
against a reorder buffer on the stream), and the conformance test. Its code
blocks run under the scratchpad `check-examples.py --finn-root`.

## Forward interface: KernelOp (not built)

The provisional `finn.graph` adapter turns a whole ONNX graph into one `Design`
before anything is chosen, so it must pre-infer every tensor. The compiler-facing
op that replaces it is a QONNX `CustomOp` wrapping one kernel, living above
`finn.kernels` (which never reads a graph). This plan does not build it; it
makes the kernel layer answer what that op will ask:

| The op's compiler duty (FINN today) | What the kernel layer must answer | After this plan |
|---|---|---|
| Build from the node (`convert_to_hw_layers`) | its facts as Params, and one stream reference per interface whose tensor is the ONNX edge | yes: the Model owns tensors |
| Datatype inference (`infer_node_datatype`, run node by node in topological order) | its output elements from its facts, choices and input elements only | yes: D5's rule |
| Shape inference (`make_shape_compatible_op`) | its output shapes from its inputs and facts | gap: today a stream's shape is given; a derived output shape per kernel is future work |
| Folding (`SetFolding`, the folding config JSON, `ApplyConfig`) | every fold a Decision with a stable key and a finite domain; a parent can pin it | yes: G0.4; keys are the persisted nodeattr names (D7 keeps them stable) |
| Persistence (nodeattrs) | committed choices encoded and decoded by key (`finn.core.space.codecs`) | exists |
| Folded shapes and widths (`get_folded_*_shape`, `get_*stream_width[_padded]`) | each port's contract: traversal, lanes, element, packed and padded width | yes: `AxiStreamPort.contract` |
| Width conversion and FIFO insertion (`InsertDWC`, `InsertFIFO`) | the plan between two ports, and its adapter chain | exists (`Stream.plan`, adapters); whether the compiler inserts adapter nodes per edge or builds a partition `Design` is the op's decision |
| Cycles (`get_exp_cycles`) | the schedule's beat count | yes, per port; resource estimates are a gap |
| Code generation and stitching (`code_generation_ipi`, `CreateStitchedIP`) | build requirements, materialized sources, composite netlist | exists |
| Numerical execution (`execute_node`, python mode) | the operation's numeric semantics | the conformance `reference`, written as `(named input arrays) -> named output arrays`, is the seed of a kernel-owned `execute`; rtlsim mode is the conformance XSim path |
| `verify_node` | refusals with codes | exists (admission, stream refusals) |

From the parked `DataflowOp` (`finn.parked.dataflow.ops.base`): keep its
shape (the op attaches its model, rebuilds the kernel's space from the node,
serializes committed choices into nodeattrs, derives datatypes and
verification from the space). Leave its scope identifiers, graph-effect
transactions and persistence stack, which D10 records as what sank it.
The op is designed after this plan lands, from the table above.

## Kernel migration

| Kernel | Ports today | After |
|---|---|---|
| dotp (both cores) | three `ScheduledPort`; extents from `y.shape[0]`, `y.shape[-1]`, `w.shape[0]` | `AxiStreamPort`s; `rows`/`outputs`/`reduction` become `extent_of(m/n/k)`; `x` reshaped under the dense realization binds from its view. `y` states `dtype` from a new `result_dtype` Param: the accumulator width is the parent's choice (MatMul binds its `result_type`) |
| thresholding | `GivenPort`s; input `vector_major(shape, pe)`, output = input, set port one index per input beat; `pe` a Param | input and output on one schedule (`a0..`, `c` folded by `pe`); `pe` a **Decision** over the divisors of the table's channel count (G0.4a); **set port keeps `sequence=`**: it indexes beats (`c`'s fold), which no affine index of the tensor expresses |
| eltwise | `GivenPort`s; `rhs` broadcast built by hand (trailing shape, `.repeated`); `pe` a Param | one schedule over `lhs`'s indices; `rhs` reads the trailing indices, so its repetition derives; `result` = `lhs`'s projection; `pe` a **Decision** (G0.4b) |
| transpose | `GivenPort`s from `input_form` (its SIMD is the form's lane count) | two schedules: in `(…, i, jf)` with lanes of `j`, out `(…, j, if)` with lanes of `i`; `input_form` is replaced by a `simd` **Decision** over the common divisors of I and J |
| memstream | `GivenPort`s; output in its consumer's order | **`sequence=`** for the output (the consumer's `period`, a demand channel) and for the set port |
| input_gen, vpc, fifo | `WordPort` | unchanged (S2) |

Identity target: module parameters, memory images, top ports, wire and beat
counts and wrapper fingerprints unchanged (`identity.py --api=k1` against
`evidence/identity-norom.txt`); decision keys unchanged except the new fold Decisions of
G0.4 (thresholding and eltwise `pe`, transpose `simd`), recorded under D7.

## Increments

| # | Content | Gate | Evidence | Cost |
|---|---|---|---|---|
| **P0** | Probes, in a scratch file: (1) the D2 rule on a copy of `output_semantics`; (2) `bind_extents` read by a kernel's `schedule` (the `sketches/bound_probe.py` result, as a test); (3) `divisors_of(extent_of(c))` with `extent_of` reading a derived dict; (4) a producer port's `dtype` read with its output stream absent; (5) a fold Decision whose `divisors_of` domain reads an absent stream's extent (G0.4b) | report | `docs/kernel-authoring-2026-09-29/p0/` | 0.5 d |
| **A1** | `fetch-repos.sh` in this worktree first (the kernel gate needs `deps/`). D1 on today's API: `conformance` for dotp (packed, INT8), thresholding, eltwise, transpose, memstream (identity reference) | kernel gate (green with `deps/`); XSim from the commit | sampled sweeps pass; a deliberately wrong loop order in a test kernel fails in XSim and passes every Python check (the harness's reason to exist) | 2 d |
| **A2** | D2: the engine rule (the P0 patch) with P0.1's tests moved into the Space suite, then the 88 removals | Space + kernel + dataflow gates with `deps/`; identity dump identical (the removals, not the rule, are what could move it); 27 doc examples | counts removed per constant, against `p0/count_semantics.out` | 1 d |
| **A3** | D3: `bind_extents` + tests (plain, shared, affine, view, disagreement, coverage) | dataflow gate | the S0 roster's shapes bind; the too-wide-x probe is refused | 1 d |
| **A3b** | The RTL checker (`artifacts/rtl.py`) establishes parameter names without values, so an array or real parameter no longer declines the whole module (review of A1: it bound 13 of 32 conformance samples); still refusal-only | kernel gate | conformance's parameter-name check binds for thresholding and eltwise; the decline table re-measured | 0.5 d |
| **A4** | D4: `AxiStreamPort` with its `ACCESS` export, base `extents`/`bound_schedule`/`extent_of`; dotp migrated; G0.3 applied (idle lanes from folds; flat builds kept for extent-free modules) | all gates; identity; A1's conformance for dotp | `shapes_agree`-class checks deleted; dotp getters gone | 2 d |
| **A5** | thresholding, eltwise, transpose, memstream migrated; their folds become Decisions (G0.4); `StreamPort`/`ScheduledPort`/`GivenPort` deleted | all gates; identity; A1's conformance for each | one port class in `finn.kernels`; eltwise's hand-built broadcast gone | 2 d |
| **A6** | D5: every producer states `dtype`; the unplaced-output rule checked by conformance; one refusal code for an element mismatch (`stream-tensor`) | all gates; graph XSim | every output element stated by its kernel; the graph shim's inference now checked against it | 1 d |
| **A7** | D6: the authoring guide, `accpool_axi` sketch as its example | doc examples | — | 0.5 d |
| **Close** | STATUS, RECORD, XSim sweeps from the final commit | — | — | — |

Order: P0 → A1 (the safety net everything after is checked against) → A2
(independent; may run beside A3) → A3 → A3b → A4 → A5 → A6 → A7. Each increment is
committed after its fast gates; XSim runs from a snapshot of the commit while
work continues, as in the composition plan.

## Environment

This worktree owns its `deps/` (fetched with `fetch-repos.sh`, never a
symlink). The kernel gate needs them, not only XSim: without `deps/`, 46
`tests/kernels` and 2 `tests/graph` tests fail reading FinnLib or qonnx sources
(P0 correction 6). Gates (kernel venv):

```
PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python bash scripts/check-kernels.sh
PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python bash scripts/check-dataflow-design.sh
```

`pyslang` 11.0.0 is in the venv (D1 step 2). The P0 probes run without
`deps/` (see `p0/REPORT.md`, "How to run").

## Non-goals

- Deriving a kernel from its RTL (the checker refuses, never supplies).
- New families, registries or wrapper Spaces; a port *family* per protocol.
- DSE over folds; the harness samples, it does not choose.
- Changing plans, adapters, the boundary rule or the adapter chain table.
- Integrating `inner_shuffle` as an adapter (FinnLib defect unchanged).
- Buffering kernels beyond transpose (SWG waits for `classify` to name
  overlapping windows).

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | The port merge changes ABI order or pin names, so fingerprints move | identity dump every commit; pins are built by the same `AxiStream`/`ReadyValidStream` code |
| R2 | `pyslang` declines FinnLib tops (no parameter defaults, `default_nettype` inside modules) | the declaration's binding is passed; `Declined` is reported, fails only under `--strict-rtl` |
| R3 | Binding refuses a legitimate kernel (an axis only an `Affine` addresses) | explicit `extents=` on `bound_schedule`; the S0 roster as tests |
| R4 | Flat builds and bound extents interact badly (an idle port with a schedule but no stream) | an idle port uses only the folds of its `lanes`, never the schedule's extents; `test_flat_kernels` stays as the check |
| R5 | XSim cost of four samples per kernel | `SAMPLED` deduplicates; the full domain stays opt-in (`ALL`) |
| R6 | A producer's `dtype` genuinely needs its output stream (P0.4) | the kernel takes the missing fact as a Param bound by its parent, as dotp's `result_dtype` does |
