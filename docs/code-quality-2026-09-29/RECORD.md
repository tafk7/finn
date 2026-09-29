# Record: code quality pass over finn.kernels and finn.dataflow

Goal: the same behaviour with less code and fewer concepts. Branch
`feature/code-quality-2026-09-28` (worktree `finn-code-quality`), from
`9118a3cb1`; local commits only.

Each increment passes the fast gates (both gate scripts, Vivado off `PATH` so
XSim tests skip), the 27 documentation examples, and the identity dump
(`docs/kernel-composition-2026-09-28/identity.py --api=k1`), diffed against
`evidence/identity-norom.txt`. XSim runs from a snapshot of the commit.

Baseline at `9118a3cb1`: Space 448, kernels 792 + 14 XSim skipped, graph 4 + 2
XSim skipped, dataflow 40; identity identical to `identity-norom.txt`.

## A: dead code and test-only views

- **Removed.**
  - `physical/ports.py` (`TypedStream`), `AxiStreamPort` and `axi_stream()`:
    no kernel used them since the Port nodes; `lane_layout` moved into
    `physical/axi_stream.py`.
  - The `interfaces` views of `FifoKernel`, `InputGeneratorKernel` and
    `EltwiseKernel`, and `STREAM_INTERFACES`: tests read each port's
    `transport`. Eltwise's view carried a PE bound (`eltwise-interface`) that
    `implementation_supported` already refuses (`eltwise-operation`).
  - `configure.compatible`, `Stream.adapter_admitted` and
    `StreamAdapter.admitted`: the one test using them reads the engine's
    `compatible_cases(point, key, admission)` and `admission(candidate)`.
  - `finn.dataflow`: `canonical_loops`, `is_repetition`, `once` (one test
    assertion), `gemm.FORM`; `split_beats` is private (`_split_at`).
  - `base.merged` (one caller), inlined.
- **Tests.** `test_axi_stream_declaration.py` becomes
  `test_scalar_declaration.py`: its scalar-admission tests read each
  scalar's `encoding` view instead of an `AxiStreamPort` bound to it; the one
  test only about that port's views is dropped.
- **Docs.** Stale docstrings: `configure` (keys), `IntToFp32Kernel` (off the
  protocol for good), `physical/axi_stream`.
- **Names removed.** `AxiStreamPort`, `axi_stream`, `TypedStream`,
  `physical.ports`, `STREAM_INTERFACES`, `*.interfaces`,
  `configure.compatible`, `Stream.adapter_admitted`, `StreamAdapter.admitted`,
  `canonical_loops`, `is_repetition`, `once`, `split_beats`, `FORM`, `merged`.
  No key changed.
- **Lines.** src −209 net, tests −38 net (20 files, +100 −347).
- **Gates.** Space 448, kernels 791 + 14 skipped (the dropped port test),
  graph 4 + 2, dataflow 40; ruff and mypy clean; examples 27; identity
  unchanged.

## B: the wiring moves to the composite; one Connection

- **Split.** `streams.py` held two things: the stream family and the wiring
  of a composite's module. The wiring (`Parts`, `merge_parts`, `netlist`,
  `Composed`, and the clock, tie-off and wrapper helpers) moves unchanged to
  `composite.py`, its only consumer; `streams.py` keeps `Stream`,
  `BufferedStream`, `StreamFifo`, `Connection` and the boundary contract
  (776 → 415 lines; `composite.py` 176 → 535).
- **One value.** `Endpoints` was `Connection` without its stages: `endpoints`
  now derives a `Connection` and `link` adds the stages with `replace`.
  `boundary_sequence` (one caller) is inlined into the stream's boundary.
- **Re-exports removed.** `streams` re-exported `MODULE`, `PORT`, `TIEOFFS`,
  `TIEOFFS_SEMANTICS`, `Tieoffs` and `Stage`; `port` re-exported `PINS` and
  `HELD`. Each is imported from its owner (`base`, `adapters`).
- **Names moved.** `finn.kernels.streams.{netlist, merge_parts, Parts, PARTS,
  PARTS_SEMANTICS, Composed, COMPOSED}` → `finn.kernels.composite`.
  **Removed:** `Endpoints`, `ENDPOINTS`, `boundary_sequence`. No key changed.
- **Lines.** src −47 net, tests −10 net (12 files, +445 −502, mostly the move).
- **Gates.** As A: Space 448, kernels 791 + 14 skipped, graph 4 + 2,
  dataflow 40; ruff and mypy clean; examples 27; identity unchanged.

## C: a port carries its kernel's element; the stream checks it

- **Found.** Every stream end's contract took its element from the stream's
  own tensor, so `Stream.well_formed`'s element check (and `compatibility`'s
  `stream-element`) could never fail on a live composition. Instead four
  kernels each re-checked their dtypes against their placed streams:
  memstream (`carried`, `stored_element`, and the element half of
  `selected`), thresholding, eltwise and transpose.
- **Changed.** A `GivenPort` presents the element its kernel gives it
  (`dtype`, placed or idle); the stream's `well_formed` refuses an end of
  another element (`stream-tensor`, naming the end). A `ScheduledPort`
  (dotp) still carries its stream's element, which its `admits` policy
  admits. `TransposeKernel` gives both ports its input's element, so a
  transpose between streams of different elements is refused by its output
  stream. The five kernel checks are deleted.
- **Names.** `StreamPort.idle_dtype` → `dtype`. Removed:
  `memstream.stored_element`, the `carried` constraints of memstream,
  thresholding and eltwise. Refusal codes `memory-element`,
  `threshold-stream-element`, `eltwise-stream-element` and
  `transpose-element` are gone: the stream refuses with `stream-tensor` (and
  its hops with `stream-element`). The refusal moves from the kernel's
  `build_requirements` and `admission` to the stream's `connection`; a
  composite still refuses, through its `structure`. No key changed.
- **Tests.** `test_port_contracts`: the eltwise operand of another element is
  refused by its stream (`stream-tensor`).
- **Lines.** src −53 net, tests −2 net (6 files, +48 −100).
- **Gates.** Space 448, kernels 791 + 14 skipped, graph 4 + 2, dataflow 40;
  ruff and mypy clean; examples 27; identity unchanged.

## D: the adapter chains are a table

- **Changed.** The seven adapter classes differed only in their stage tuple.
  `CHAINS` lists the tuples; `ADAPTERS` builds each candidate with the
  engine's `composite(...)` over `StreamAdapter`, keyed by its modules joined
  (`vpc_input_gen`), each child named by its stage (`input_gen`, `vpc`,
  `input_gen_1`, `vpc_1`) and bound to that stage's facts, as before. The
  stream's `adapter` Decision takes `ADAPTERS`.
- **Names.** Removed: `InputGenAdapter`, `WidthAdapter`,
  `WidthReorderAdapter`, `ReorderWidthAdapter`, `ReorderWidthMarkersAdapter`,
  `RegroupAdapter`, `RegroupMarkersAdapter`. New: `CHAINS`, `ADAPTERS`. The
  adapter keys (`<stream>.adapter` cases and
  `<stream>.adapter.<chain>.<stage>.ram_style`) are unchanged.
- **Tests.** `test_stream_plans` checks each realized chain against `CHAINS`.
- **Lines.** src −74 net, tests −7 net (3 files, +34 −115).
- **Gates.** Space 448, kernels 791 + 14 skipped, graph 4 + 2, dataflow 40;
  ruff and mypy clean; examples 27; identity unchanged.
