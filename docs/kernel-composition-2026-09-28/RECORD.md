# Record: kernel choices, one interface per port, composable kernels

The execution record of [PLAN.md](PLAN.md). Each increment appends what landed,
the gate results as observed, key and name changes (D7), and deviations.

## G0 (2026-09-28)

| # | Answer |
|---|---|
| 1 | Weight memory is a discrete `memory` Decision in the composite (F10) |
| 2 | Open; V1 proceeds with `(k, n)` unless the user says otherwise |
| 3 | `required()` ships with E1 |
| 4 | Registration dropped; candidates are listed in the Decision |
| 5 | A port's contract is named after the nearest ancestor that owns a module |
| 6 | `replay_buffer` stays scrapped |

## E0: engine probe

- **By** a subagent; probe, tests and report in [`e0/`](e0/); the report is
  [`e0/REPORT.md`](e0/REPORT.md), the review [`e0/REVIEW.md`](e0/REVIEW.md).
- **Result.** The refined Decision builds today's modules and keys over the
  identity configurations for the dot-product core, weight memory, the stream
  adapter and transport; case 5 (disjoint choices) holds on stubs. Narrowing
  and pinning by key needed an engine change (spiked).
- **Gates.** Probe 52 passed (rerun by the reviewer); kernel gate Space 427,
  kernels 819; dataflow 33; unchanged by the probe.
- **Review verdicts** fixed the E1 design: the report's engine changes, with
  `settle` split into a generic engine service (admission passed in) and the
  kernels' convention (`finn.kernels.configure.settle`, the `admission`
  member).
- Commit `1051948f5`.

## E1: the engine refinement

- **Engine** (`finn.core.space`):
  - `Decision(entries, /, *, optional=False, when=None, **shared)`: entries are
    families or calls; shared bindings strict (every candidate declares each;
    no double assignment); the scalar form's argument names are refused as
    shared bindings; `optional=True` adds a `None` candidate keyed `"none"`,
    first. The `values={...}` mapping form is unchanged (migrated in K2).
  - Direct reads through an entry Decision need a member every candidate
    declares, with one value type (checked at class creation, from each
    candidate's collected semantics); `choice["key"]` reads one candidate's
    node (in a body, from class access, and for enclosing assignments).
  - An enclosing body pins a Decision over nodes with a key (the selector
    becomes a constant, the key is listed as pinned) or narrows it with a
    Decision over keys (the key stays, its domain shrinks); either keeps the
    declared candidates with their declared bindings. The earlier
    fresh-node narrowing still works.
  - `required(T)`, `Required`, `unmet_required`: a family with an unmet member
    cannot be placed (checked in `declare_node`) and cannot be an entry.
  - `settle(point, admission=None)` and `compatible_cases` in
    `finn.core.space.settling`; `settle` and `Settlement` exported.
  - `inspection.choices()` reports a pinned choice's selector as `None`;
    `codecs.decode` refuses a narrowed-out case.
- **Kernels.** `finn.kernels.configure.settle` and `admission` (the member
  named `admission`: a constraint group, a constraint or a view).
- **Docs.** Scratchpad `space/AUTHORING.md` ("Choices over nodes": entries,
  shared bindings, reads, `required`, `settle`; the overrides table; the house
  example in the entries form and the estate narrowing by key),
  `space/DESIGN.md` §4.3–4.4, and a migration note in `space/MIGRATION.md`.
  27 documentation examples pass.
- **Deviation from the plan.** `required()` is spelled `required(T)` (for
  example `schedule = required(Schedule)`), not `schedule: Schedule =
  required()`. As a dataclass field specifier (E0's Finding 7), mypy refuses a
  subclass that meets the member with a derived value ("Dataclass attribute
  may only be overridden by another attribute") and demands the member as a
  call keyword. Typed by its argument, like a derived value, it checks cleanly
  under `--strict`, and a Param, derived value, view or constant meets it. No
  settled decision changes.
- **Changed test.** `test_overrides.py`: a key now pins a Decision over nodes
  (it was refused); `test_inspection.py`: a choice's selector is typed
  `DecisionHandle[str] | None`.
- **Keys.** None changed.
- **Gates.** `scripts/check-kernels.sh`: Space 446 passed (427 plus 19 in
  `tests/core/space/test_candidate_entries.py`), kernels 819 passed, ruff and
  mypy clean, exit 0. `scripts/check-dataflow-design.sh`: 33 passed, exit 0.
  The kernel gate is unchanged, as E1 requires (nothing migrated).

## V1: values

- **`finn.dataflow`:**
  - `schedule`: `Index` (named, with affine arithmetic: `Affine`), `Schedule`
    (extents, folds, beat order; `present`, with `lanes`, `reduces`, `holds`
    and `view`; `closing`), `Refused`.
  - `gemm`: `m`, `n`, `k`, `Signature`, `Form` (`DENSE`, `DEPTHWISE`).
  - `traversal`: `Presentation` renamed `BeatSequence` (`BEAT_SEQUENCE`);
    `once` and `period` moved here.
  - `stream`: the logical `Stream` (tensor, `adaptable`, `ends =
    required(Ends)`, `well_formed`, `plan`, `adapting`, `realizable`); `End`,
    `Ends`. `finn.kernels.streams.Stream` subclasses it, finds the ends among
    its `users` (was `ends = Users(PORT)`) and defines `ends` from its
    contracts.
- **Removed.** `finn.dataflow.nest` (`Level`, `Nest`, `Access`, `Iteration`,
  `Einsum`, `fold`, `accesses`), `Contraction`, `contraction_iteration`,
  `DotpPresentations`/`dotp_presentations` (now `DotpSequences`/
  `dotp_sequences` over a schedule and a form, until K1 removes them).
- **Weights stored `(k, n)`** (G0.2, proceeding on the recommendation):
  MatMul's `weights`, the block-diagonal dense realization
  (`W'[k * N + c, n]`), the tests' literals (written by output, transposed)
  and the numeric harness (generates by output, passes the transpose).
- **Renamed.** Per-channel is depthwise throughout (`Form.DEPTHWISE`,
  `test_matmul_depthwise.py`, the harness's `--depthwise`); `contraction=`
  is `form=`; refusal codes `dotp-iteration` → `dotp-schedule`,
  `dotp-contraction` → `dotp-form`; the stream's `ends` (users) → `users`.
- **Tests.** `test_nest.py` → `test_schedule.py` (19: the S0 roster over
  `Schedule`, including the weights' MVAU tile order as `(k, n)` positions and
  a held operand); `test_stream.py` (4) for the logical stream.
- **Identity.** `identity.py` (13 configurations: D10's 14 less
  `replay-input-gen`, which S3 made the dense default): module parameters,
  memory images, top ports, wire counts, beat counts, wrapper fingerprints and
  decision keys are identical before and after V1
  (`evidence/identity-d10.txt`, `evidence/identity-v1.txt`).
- **Gates.** Kernel gate: Space 446, kernels 819, ruff and mypy clean.
  Dataflow gate: 40 passed, ruff and mypy clean.
- **Not run.** XSim: no module parameter, image or wrapper changed.

## K1: the Kernel protocol, ports, and MatMul

- **The protocol** (`finn.kernels.base`): a kernel declares its RTL `module`,
  `sources()`, `parameters()`, `clocking` (a `Clocking` value: `ap_clk`,
  `ap_rst_n`, and a doubled clock held low while unused) and an `admission`
  group; the base derives `codegen` (clocking, then every port's bus, then
  parameters), `build_requirements` (accepted under `admission`), `tieoffs`
  and the `MODULE`/`TIEOFFS` exports. `MODULE`, `TIEOFFS` and `Tieoffs` moved
  here from `streams`. Kernels not yet on the protocol (eltwise, FIFO,
  int-to-fp32, the HLS memstream) declare `exports = {}` until K2.
- **Ports** (`finn.kernels.port`): `Port` (stream reference, element from the
  stream's tensor admitted by an `Integer` policy, `sequence = required(...)`,
  AXIS pins; exports `PORT` for its stream and `BUS`) and `ScheduledPort`
  (the kernel's schedule through `index`, `lanes`, `reduces`, `holds`,
  `closes`, `reshaped`).
- **dotp**: `DotpAxiKernel` on three ports `x`, `w`, `y` over `x_stream`,
  `w_stream`, `y_stream`; extents from the streams' tensors; `pe`, `simd` and
  `compute_pumping` are its Decisions; `schedule` derived; one `admission`
  group (target, core, accumulator, stream widths, pumping). Removed: the
  dtype Params and scalar nodes, `axi_stream` ports, `schedule`/`iteration`
  inputs, `DotpSequences`/`dotp_sequences`, the element cross-check,
  `support` (now `admission`).
- **MatMul**: facts `m`, `n`, `k` (were `rows`, `outputs`, `reduction`);
  `compute = Decision({"packed": packed, "int8_dsp58": Int8Dsp58DotpKernel},
  form=..., target_dsp=..., target_period_ns=..., reshape_activations=...,
  x_stream=..., w_stream=..., y_stream=...)` with the `packed` handle carrying
  `narrow_weights`; `memory = Decision({"rom": RomKernel, "memstream":
  MemStreamKernel(set_stream=..., control=...)}, optional=True, dtype=...,
  form=weight_period, contents=..., writable=..., sets=..., output_stream=...)`.
  `weight_period` reads the selected core's weight port. `matmul_assembly`
  commits facts and the caller's choices, settles the core, commits its
  folds, settles the adapters. Removed: `pe`/`simd`/`compute_pumping` Params,
  `matmul_schedule`, `_Folding`, `sequences`, `delivery`, `commit_adapters`.
- **Memories**: `CyclicDelivery` → `RomKernel` (`rom.py`), refusing writable
  weights or several sets (`rom-writable`, `rom-sets`); `values` → `contents`
  on both memories; memstream's `support` → `admission`. `WeightDelivery`
  keeps its names with the `memory` cases as values (`none`, `rom`,
  `memstream`).
- **Streams**: `netlist` names a stream end after its nearest module owner
  (G0.5), so a core's ports keep the core's instance (`u_compute_packed`).
  Stream adapters have an `admission` group (`realizes`).
- **Engine additions** (not in E0's list):
  - `Users` sees a node that reaches the referenced node through a forwarded
    input (the deferred "Users through forwarding composites"), unless a node
    it forwards through exports the key for that input (that node answers
    for it, as `test_references` requires).
  - `settle` counts a candidate compatible unless its admission is
    `Rejected`; an admission waiting on an open choice does not refuse.
    `finn.kernels.configure.admission` refuses a group as soon as one of its
    constraints refuses (the engine's group result waits for the open ones).
- **Protocol views on every kernel.** `build_requirements` and `tieoffs` are
  capabilities of every `Kernel`; a kernel without a module refuses
  `build_requirements` (`kernel-module`). `test_kernel_extensions` updated.
  The HLS memstream's `build_requirements` view is renamed `sources` (an HLS
  source bundle is not a module).
- **Keys (D7).** `pe`, `simd`, `compute_pumping` → `compute.<core>.pe`,
  `.simd`, `.compute_pumping`; `delivery` → `memory` with cases `none`,
  `rom`, `memstream`; `delivery.cyclic.rom_style` → `memory.rom.rom_style`;
  `delivery.memstream.*` → `memory.memstream.*`.
- **Names.** Instances `u_delivery_cyclic` → `u_memory_rom`,
  `u_delivery_memstream` → `u_memory_memstream`; top modules
  `finn_matmul_external`/`_cyclic` → `finn_matmul_none`/`_rom`; stream
  users are port nodes (`compute.packed.y`).
- **Tests.** `test_dotp.py` rewritten over placed cores (60);
  `test_port_contracts.py`, `test_two_kernels.py`, `test_interfaces.py`,
  the MatMul tests (`test_matmul_delivery_choice.py` →
  `test_matmul_memory_choice.py`), `test_memstream.py`,
  `test_boundaries.py`, `test_installed_package.py`, the typing test, and the
  harnesses migrated; `helpers.placed_dotp` and `helpers.settled`. Probes
  with no analogue dropped: a schedule handed to dotp in another beat order
  or with other folds (dotp derives its own), a dense core handed depthwise
  accesses.
- **Identity.** Over the 13 configurations, every module's parameters, the
  memory images, the top ports, wire counts and beat counts are identical to
  V1 (`evidence/identity-k1.txt`); only top module names, wrapper
  fingerprints and keys change.
- **Gates.** Kernel gate: Space 447, kernels 810, ruff and mypy clean.
  Dataflow gate: 40. Documentation examples: 27.
- **XSim.** Every numeric sweep from a snapshot of `1e44982cb`, one
  simulation per process: dense 40, FIFO transport 6 (packed) and 6 (INT8
  pumped), depthwise 34, memstream 14 and 12 (depthwise), pumped memory 14,
  writable 14, three sets 14, pure dotp 27 and its stress set 17, adapters 26;
  224 passes, no failures.
- **G0.2 answered after the fact** (user, 2026-09-28): weights stay `(k, n)`
  for now, to be revisited in detail, specifically how weight files are
  generated.

## K2: every other kernel on the protocol; adapters and transport

- **Ports** (`finn.kernels.port`). `Port` is the base: a required
  `transport`, pins under `PINS`, and what it holds while `idle` under
  `HELD`. `WordPort`: opaque words on FinnLib's native `idat`/`ivld`/`irdy`
  or `odat`/`ovld`/`ordy`, with markers. `StreamPort` (K1's `Port`): on a
  stream, an AXIS bus, or with `signals` (data, valid, ready) those native
  pins carrying the lanes' bits exactly; idle without a stream, its pins
  then from `idle_dtype` and `idle_lanes`; `admits` defaults to `None` (the
  kernel admits the element itself). `ScheduledPort` and `GivenPort` as in K1.
- **The protocol** (`finn.kernels.base`). `PORT`, `PINS`, `HELD` beside
  `MODULE` and `TIEOFFS`; `Clocking.active_low` and `NATIVE_CLOCKING` (`clk`,
  active-high `rst`); `other_pins()` (an AXI-Lite bus) and `held()` (which may
  refuse); `parameters()` may refuse (an `inspect` evaluates `codegen` even
  when admission refuses); `sources()` takes any requirement contribution (an
  INIT_FILE). The ABI is clocking, other pins, then ports' pins in
  declaration order; `tieoffs` merges the doubled clock, idle ports and
  `held()`. A kernel adding exports extends the base's:
  `{**Kernel.exports, CONTROL: ...}`.
- **Kernels on the protocol.** `FifoKernel` (two `WordPort`s);
  `InputGeneratorKernel` (`olst` of its rank on its output); a new
  `VpcKernel` (`finn.kernels.vpc`); `RomKernel` (the `cyclic_stream` module;
  id `cyclic_stream`, the build identity it always had; native output pins
  through `signals`); `MemStreamKernel` (set port idle with one set, AXIS
  output, AXI-Lite through `other_pins()`/`held()`, `clk2x` through
  `Clocking`); `ThresholdingAxiKernel` (three AXIS ports, the selector idle
  with one set, AXI-Lite likewise); `EltwiseKernel` (three native ports, now
  with optional `lhs_stream`/`rhs_stream`/`result_stream`: each walked
  row-major, PE a beat, a trailing-shape `rhs` presented once per `lhs`
  element it meets); `TransposeKernel` (`inner_shuffle`, native ports). Each
  checks that a placed stream carries the element it stores or takes
  (`memory-element`, `threshold-stream-element`, `eltwise-stream-element`,
  `transpose-element`). Off the protocol, recorded: `IntToFp32Kernel`
  (combinational, no clock) and `MemStreamHlsKernel` (an HLS source bundle).
- **Adapters.** Each chain places its modules as kernel children named by
  stage (`input_gen`, `vpc`, `input_gen_1`, `vpc_1`), their facts derived
  from the realization (`input_gen_facts`, `vpc_facts`, ...); `stages` builds
  each `Stage` from the child's `build_requirements` and port transports.
  Q3 option C: `ram_style` is the `InputGeneratorKernel`'s own Decision. The
  stream's `adapter` and `transport` Decisions are in the entries form
  (shared `tensor`, `plan`; `when=adapting`); `StreamFifo` reads its FIFO's
  port transports. `ADAPTER_RAM_STYLES` (`*.adapter.*.ram_style`) is the key
  pattern `matmul_assembly` and `helpers.settled` commit to `auto`.
- **Engine.** A class body reads an attribute of its own derived value
  (`facts.word_bits`), as it already could of a reference input; a value
  type names a field's semantics with `Annotated[T, SEMANTICS]`; projection
  finds dataclass fields without defaults. Q2: `Decision(values={...})` is
  refused, pointing at entries; every site (kernels, 39 Space test sites and
  four by hand, the typing fixtures, the Space docs) is migrated, and an
  empty entry mapping says it needs a candidate.
- **Removed.** `finn.kernels.streaming` (`cyclic_stream_requirements`, whose
  tests moved onto `RomKernel`); the adapters' `input_gen_requirements`,
  `vpc_requirements`, `input_gen_interfaces`, `clock_reset`, `native`,
  `buffers`; the stream's `buffering` and `adapter_ram_style`;
  `physical.ports.NativeStreamPort` and `native_stream`; every hand-written
  ABI and tie-off (memstream, thresholding, ROM, FIFO, `input_gen`, `vpc`,
  `inner_shuffle`, eltwise).
- **Keys and names (D7).** `<stream>.adapter_ram_style` →
  `<stream>.adapter.<chain>.<stage>.ram_style` (one per `input_gen` of each
  chain). Memory and transpose stream users are port nodes
  (`memory.rom.output`, `first_source.output`); their contracts read as
  `.output.contract`. Removed views: the ROM's `output` contract (now a
  port), memstream's `output_port`, `set_port`, `output_interface`,
  `set_interface`, thresholding's `interfaces`, `input_port`, `output_port`,
  `set_port`, `input_form`, `implementation_supported` (now `admission`),
  the transpose's `output_form` (now `output_sequence`).
- **Deviation.** The flat-kernel tests were not moved onto streams: an
  unplaced kernel's ports are idle and take their pins from the kernel's own
  dtypes, so the flat build is the placed one. Streams are exercised by the
  composite tests, and eltwise on streams by `test_port_contracts`.
- **Identity.** Over the 13 configurations, module parameters, images, top
  ports, wire and beat counts are unchanged from K1; wrapper fingerprints are
  unchanged except the four memstream configurations (its `m_axis_0` is an
  AXIS bus in its ABI, and its tie-offs are in protocol order); keys change
  as above (`evidence/identity-k2.txt`).
- **Gates.** Space 448, kernels 806 (XSim in pytest included), dataflow 40;
  ruff and mypy clean; documentation examples 27.

## S4: composable kernels and the Design

- **Nested lowering, verified first, failed as the plan's risk anticipated.**
  `_flatten_contributions` refused a rendered child and validation required
  fixed child names: a generated module's name is known only when its build
  is prepared. The fix is in the artifact layer, additive:
  `RenderedSourceRequirement.values` (a rendered source carrying its own
  bindings, `MODULE_NAME` among them, with a fixed output) and
  `nested_module_name` (stem and requirements fingerprint, so equal nested
  modules share one name). `finn.kernels.physical.lowering.nested` turns a
  composed child into a fixed module whose wrapper renders from its own
  values; `netlist` nests every composed child, and flattening accepts such
  wrappers (deduplicated by output).
- **Composites** (`finn.kernels.composite`). `Composite(Kernel)`: `modules`,
  `streams`, `tied` (children's tie-offs), `controls`, `flattened`
  (children's parts), `structure` and `build_requirements`; `stem()` and
  `producer_identity()` name the module. Placed in a parent through the
  reference inputs its `boundaries` pair with internal streams, it exports
  each boundary stream's contract under `PORT` (`Stream.boundary`) and, by
  its `fused` Decision (applicable only when placed), `MODULE` or `PARTS`.
  `seated` refuses a parent stream of another tensor (`composite-tensor`);
  `uncontrolled` refuses a fused composite exposing a control bus
  (`composite-control`), which its parent cannot export yet. `Design` is a
  composite at the top.
- **Parts and splicing** (`finn.kernels.streams`). `Parts` carries a
  composite's netlist inputs; `merge_parts` names a child's parts below it
  (instances `u_<child>_<node>`, control ports `<child>_<port>`) and splices
  each child boundary stream with the parent stream it sits on, matched by
  the reference input now recorded on each end (`Connection.source_input`,
  `sink_input`); stages keep the stream that placed them (`Stage.stream`).
  `netlist` refuses any input nothing drives, bus members included (a nested
  child's unexported bus is no longer skipped silently).
- **MatMulKernel** is a `Composite`: `x_stream`, `w_stream`, `y_stream`,
  `set_stream`; `admission` (was `dimensions`).
- **Test composites** in `test_adapters`, `test_two_kernels` (the D10
  two-kernel demo) and `test_interfaces` are `Design`s.
- **Keys.** `fused` is a new key of every composite (inapplicable at the
  top). **Identity.** Module parameters, images, top ports, wire and beat
  counts unchanged; every composed wrapper fingerprint changes once, because
  `RenderedSourceRequirement` gained a field (its canonical form); the
  rendered wrapper inputs are byte-identical to K2's
  (`evidence/identity-s4.txt`). The wrapper template's stale header comment
  is corrected.
- **Fast gates** (XSim tests skipped, run separately from the commit): Space
  448, kernels 800 + 15 XSim skipped, dataflow 40; ruff and mypy clean;
  documentation examples 27. Before the commit, both XSim variants of the
  Design gate passed locally (fused and unfused).

## G1: the graph adapter, for MatMul

- **Layer.** `finn.kernels` never reads a graph (`test_boundaries` forbids
  `qonnx.core.modelwrapper` there), so the adapter is a new layer above it,
  `finn.graph` (`core.space <- dataflow <- kernels <- graph`), tested in
  `tests/graph` by the kernel gate; `finn.kernels` may not import it.
- **`graph_design(model, target_dsp=, target_period_ns=)`** reads the model
  once into a generated `Design`: a stream per exchanged tensor (graph inputs
  and outputs are `inN_V`/`outN_V` in graph order; leading axes are rows), a
  `MatMulKernel` per `MatMul` node. An initializer is the kernel's weights,
  `(k, n)` as ONNX stores them; without one the weights are a stream and
  `memory` is pinned to `none` (`GraphDesign.pinned`). Each MatMul's exact
  result type flows downstream; a narrower annotation of that tensor is
  refused (`FLOAT32` admits anything), as is any other operator
  (`GraphError`).
- **FINN attributes.** `MatMulKernel.finn_attributes` (a view) gives the MVAU
  attributes from the configuration alone: `MW`, `MH`, `SIMD`, `PE`,
  `numInputVectors`, the three data types and `accDataType`, `mem_mode`
  (standardization, recorded: none → `external`, ROM → `internal_embedded`,
  memstream → `internal_decoupled`), `ram_style` (the memory's style),
  `runtime_writeable_weights`, `pumpedMemory`, `resType` `dsp`,
  `noActivation` 1; a depthwise MatMul (FINN's VVAU) is refused.
  `finn_model(model, point, kernels)` rewrites each MatMul as an `MVAU` node
  with them and annotates its result; FINN's own `MVAU` reads them back.
- **Tests.** `tests/kernels/xsim.py` is a shared one-in, one-out XSim harness
  (the Design test uses it). `tests/graph/test_adapter.py`: two MatMuls with
  initializers compute in XSim what `execute_onnx` computes, fused and
  unfused; weights without an initializer become `in1_V`; refusals.
- **Fast gates.** Space 448, kernels 800 + 15 XSim skipped, graph 4 + 2 XSim
  skipped, dataflow 40; ruff and mypy clean. The XSim tests passed locally
  before the commit and run again from it.

## XSim from the commits

- **K2** (`8d68a7c4b`, snapshot): every numeric sweep passes: dense 40, FIFO
  6 and 6, depthwise 34, memstream 14 and 12, pumped 14, writable 14, sets
  14, dotp 27, dotp-stress 17, adapters 26 (224).
- **S4** (`c7764f7f9`, snapshot): the kernel suite with its XSim tests passes
  (815), the Design gate fused and unfused among them. Its numeric sweeps did
  not run: the snapshot lacked the locally built XSI extension
  (`finn_xsi/xsi.so`, ignored build output), so every sweep failed to start.
  The runner now copies it; the sweeps run from the next commit instead,
  whose hardware includes S4's.

## ROM removed (user, 2026-09-28)

The cyclic ROM (`RomKernel`, `cyclic_stream.sv`) embedded no eliminable
constants: its words reached the DSP cores through a registered stream, as a
read-only memstream's do, so it duplicated the memstream with fewer options.
Removed: `finn.kernels.rom`, `resources/cyclic_stream.sv`,
`WeightDelivery.CYCLIC`, `matmul_assembly(rom_style=)`, `ROM_STYLE`, the
`rom` candidate of `memory` (now `none` or `memstream`), and
`test_streaming_components` (the ROM's own tests). `stored_element` moved to
`finn.kernels.memstream`. Tests and harnesses that used the ROM as a constant
producer use a read-only `MemStreamKernel`; hand-wired compositions tie off
what it holds; XSim harnesses place its INIT_FILE where `$readmemh` reads it
(the shared `tests/kernels/xsim.py`, the adapter harness, the
stream-contract test); `test_two_kernels` uses the shared harness. FINN's
`mem_mode` for the memstream stays `internal_decoupled`. The real embedded
path is a design note: `docs/constant-weights-2026-09-28/DESIGN.md`.

- **Identity** (`evidence/identity-norom.txt`): every remaining configuration
  is identical to S4's, fingerprints included; the four ROM configurations
  and the key `memory.rom.rom_style` are gone.
- **Fast gates**: Space 448, kernels 792 + 14 XSim skipped, graph 4 + 2 XSim
  skipped, dataflow 40; ruff and mypy clean; documentation examples 27.
