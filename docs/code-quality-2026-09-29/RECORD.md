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
