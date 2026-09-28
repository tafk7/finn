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
