# Plan: kernel families, one interface per port, composable kernels

Date: 2026-09-28. Status: **proposed**. Follows the D10 stream model
([`../stream-model-2026-09-27/RECORD.md`](../stream-model-2026-09-27/RECORD.md)).
Plan of record: [`../kernel-status-2026-09-27/STATUS.md`](../kernel-status-2026-09-27/STATUS.md).

## Goal

Every level of the hardware stack is a `Kernel`, and every choice among
kernels is a `Decision`. When done:

- an operation (MatMul) is a kernel family: its facts, its ports and a
  schedule every implementation must define; its implementations are its
  registered subclasses (a dot-product array, later a systolic array, a tiled
  MVU, float and LUT cores);
- a choice among implementations is a Decision over a family, bound once;
  compatibility filters the candidates and `settle` commits the single
  survivor, while several survivors stay a design choice;
- a kernel declares one node per interface (a `Port`), one schedule, its
  admission and its parameters; the `Kernel` base derives the ABI, build
  requirements, tie-offs and exports;
- kernels compose into kernels: a Design is a composite kernel, and an ONNX
  node becomes Decisions over families inside it.

## Settled in discussion (2026-09-28)

| # | Decision |
|---|---|
| F1 | A realization of an operation **is** a Kernel; no separate realization level. An operation is a kernel family whose concrete subclasses are its kernels |
| F2 | A Design **is** a composite Kernel: one generated top module, built from children wired by streams. No separate Design or Op class kind |
| F3 | Choices among families are **Decisions**, refined: a Decision over a family (candidates are its registered subclasses), bindings written once on the Decision, `optional=True` for a `None` candidate, reads typed by the family, candidate members reached qualified (`compute["dotp"].pe`), local keys namespaced (`compute.dotp.pe`). Explicit `values={...}` stays for ad-hoc choices |
| F4 | No `Contract` or `Composite` classes. A family's required outputs are `required()` members; leaf or composite follows from what a kernel declares (its module and `parameters()`, or children wired by streams) |
| F5 | One name per concept: `Index`, `Schedule`, `BeatSequence` (was `Presentation`), `Tensor`. Retired: `Level`, `Nest`, `Access`, `Iteration`, `Einsum`, `fold`, `accesses`, `Contraction` |
| F6 | Canonical GEMM notation: indices `m`, `n`, `k`; `Form.DENSE` (X `(m, k)`, W `(k, n)`, Y `(m, n)`) and `Form.DEPTHWISE` (X `(m, k, n)`). Ports are written over the operation's indices; the schedule folds them |
| F7 | The schedule and its folding Decisions belong to the kernel, not the operation (PE/SIMD are the dot-product array's folds of `n` and `k`) |
| F8 | One `Port` node per interface: its indices, lane order, element admission, pins and stream reference; it exports its own `PORT`. The element comes from the stream's tensor |
| F9 | The `Kernel` base standardizes codegen and composition plumbing: one `admission` group, `module`/`sources`/`clocking`/`parameters()`, derived ABI, `build_requirements`, tie-offs of unbound ports and buses, and exports; a composite's netlist likewise |

## Human gate G0: decisions before building

| # | Decision | Recommended | Blocks |
|---|---|---|---|
| 1 | Weight delivery moves up to the composite that holds the op's weight stream (a `WeightSupply` Decision), rather than living inside each MatMul kernel; whether a ROM shares the compute kernel's module becomes the composite's fusion Decision | yes | K1 |
| 2 | Weights are stored `(k, n)` (ONNX `MatMul`), not FINN's `(n, k)` | yes | V1 |
| 3 | `required()` members ship with the Decision refinement (a link-time error for a family that forgets its schedule), rather than later | yes; it is small | E0 |
| 4 | Registration: a subclass registers with `key=` in its class statement; the candidate set is resolved when the model compiles and persisted as family id and version. Entry-point discovery for out-of-tree kernels is later work | yes | E0 |
| 5 | A port node's contract is named, in `netlist`, after the nearest ancestor that owns a module (`u_compute_dotp` stays the dotp instance, not `u_compute_dotp_x`) | yes | K1 |
| 6 | Still open from D10: keep `replay_buffer` scrapped (G0.6; `input_gen` replays at about four times its storage), or restore it as a second replay adapter | the user's call | K2 |

## Increments

Each ends at the kernel and dataflow gates plus the XSim sweeps it touches,
records key, name and fingerprint changes (D7), and lands on its own.

### E0. Engine probe (2–3 days, no production code)

- **Work.** A probe in `docs/` building the refined Decision on today's
  engine, or on a spike branch where the engine must change, over the five
  cases:
  1. a dot-product core (two families, shared bindings);
  2. op → kernel (`MatMul` → `DotpMatMul`, a second stub kernel);
  3. `WeightSupply`, optional, with one qualified binding;
  4. a stream's adapter (seven families, shared bindings, guarded);
  5. a stream's transport.

  It also covers:
  - reads through the Decision typed by the family;
  - qualified reads, inapplicable when another candidate is selected;
  - namespaced keys, narrowing and pinning;
  - switching candidates (the stale-choice rule);
  - selections that capture and restore by key, and refuse an unknown family;
  - `required()` refused at link time.
- **Gate.** The user reviews the probe and its API, as for B1's per-input
  exports. The engine design is fixed here.

### E1. The engine refinement (3–5 days)

- **Work.**
  - `Decision(Family, **bindings, optional=..., among=..., when=...)` and path
    assignment through it.
  - Family-typed reads, qualified access, and the registry (`key=`, resolved
    at compile).
  - `required()`.
  - `settle(point)`: commit every Decision with exactly one feasible
    candidate. It is a helper beside `commit`/`compatible`.
- **Docs.** The canonical Space documentation's "Choices over nodes" section
  (scratchpad `space/AUTHORING.md`) and its examples, plus a migration note.
- **Gate.** Space gate (427 today, plus new tests); examples pass; the kernel
  gate still passes unchanged (nothing migrated yet).

### V1. Values (2–3 days)

- **Work.**
  - `finn.dataflow`: `Index` (with affine arithmetic), `Schedule` (extents,
    folds, beat order; `present`, `closing`), `BeatSequence`, `Form`, and
    `reduces`/`holds` in `present`.
  - The logical `Stream` Space moves to `finn.dataflow` (tensor, ends'
    sequences, plan, `adaptable`); `finn.kernels` subclasses it.
- **Removed.** `Level`, `Nest`, `Access`, `Iteration`, `Einsum`, `fold`,
  `accesses`, `Presentation`.
- **Gate.** The S0/S2 equalities re-expressed over `Index`/`Schedule`
  (`test_nest.py` becomes `test_schedule.py`); plan tests unchanged.

### K1. The Kernel protocol, ports, and MatMul as a family (1–1.5 weeks)

- **Work.**
  - The `Kernel` base (F9) and `Port` nodes (F8), including `netlist`'s
    instance naming (G0.5).
  - dotp: the `DotpCore` family (`PackedDotp`, `Int8Dsp58Dotp`), each a leaf
    kernel with its admission.
  - `MatMul`: facts, `Form`, ports, required `schedule`.
  - `DotpMatMul`: PE and SIMD, its schedule, a `DotpCore` Decision; its
    activation stream's adapter replays as today.
  - The op composite places a `MatMul` Decision and a `WeightSupply` Decision
    (`RomSupply`, `MemStreamSupply`, optional; G0.1).
  - `matmul_assembly` becomes facts, then `commit`, then `settle`.
- **Removed.**
  - dotp's `*_dtype`/`*_type`/`axi_stream`/`*_stream` slots, `iteration`,
    `dotp_presentations`/`DotpPresentations`, and the element cross-check.
  - `contraction_iteration`, `_Folding`'s checks, and `realization`/core
    selection code.
  - `delivery` inside MatMul, and `commit_adapters` (subsumed by `settle`).
- **Gate.**
  - Module parameters identical per configuration over the identity dump's
    14 configurations. Wrapper fingerprints change with instance and key
    names; each change is recorded.
  - All MatMul XSim sweeps. The numeric harness runs per registered
    implementation of the family (conformance by family).

### K2. Every other kernel on the protocol; adapters and transport as families (1 week)

- **Work.**
  - Thresholding, eltwise, FIFO, memstream, `input_gen`, `vpc` (a new leaf
    `VpcKernel`), `inner_shuffle` and the cyclic stream move onto the
    `Kernel` protocol and `Port` nodes.
  - A stream's `adapter` becomes a Decision over the `StreamAdapter` family
    and `transport` a Decision over `Transport` (direct, FIFO), both bound
    once. Adapter chains place `InputGeneratorKernel`/`VpcKernel` children
    instead of calling requirement builders.
  - Test composites shrink to children and streams.
- **Removed.** The hand-written tie-offs (dotp, memstream, thresholding), the
  duplicated clock/reset helpers, and the per-kernel parameter-to-ABI
  conversions.
- **Gate.** Flat-kernel, adapter and two-kernel XSim evidence unchanged in
  behaviour; all sweeps.

### S4. Composable kernels and the Design (1–2 weeks)

- **Work.**
  - A composite exports its boundary streams' contracts under `PORT` and
    takes stream references; nested lowering (a composed module as a child of
    `netlist`) is verified first.
  - The composite's fusion Decision decides which children generate their own
    module.
  - A Design is a composite kernel placing kernel families and the streams
    between them.
- **Gate.** Two MatMuls and a thresholding in one Design, fused and unfused
  variants equal in XSim; the D10 two-kernel demo re-expressed as a Design.

### G1. The graph adapter, for MatMul (1 week)

- **Work.**
  - An ONNX `MatMul` node (with or without a weight initializer) becomes
    Decisions over `MatMul` and `WeightSupply` in a Design, built from the
    node's facts with real initializers (the model-context lesson).
  - FINN node attributes are emitted from the selected families' decisions
    ("kernel owns nodeattrs"), hermetically.
  - Programmatic declaration of a Design body is the one engine question
    here; E0 checks it can be done with class creation and still yield stable
    keys.
- **Gate.** A small ONNX model through the adapter, `settle`, and XSim.

## Order and dependencies

```text
G0 ──► E0 ─(review)─► E1 ──► V1 ──► K1 ──► K2 ──► S4 ──► G1
```

V1 can start beside E1, since it touches values only. K1 needs E1 and V1.

## Risks

| Risk | Mitigation |
|---|---|
| The Decision refinement interacts badly with guards, collapse or selections | E0 covers the five real cases, collapse on and off, and selection round trips before any migration |
| Key and name churn across every kernel | recorded per increment (D7); module parameters must stay identical, so only names and wrappers change |
| Port nodes exporting contracts break instance naming or `Users` attribution | G0.5 decides the naming rule; K1's gate checks every instance name and per-port refusal |
| Nested lowering fails | S4 verifies it first, before building on it |
| Programmatic Design declaration conflicts with class-body compilation | probed in E0; G1 is last |
| Scope creep into new kernels (systolic, float) | out of scope: E0 uses a stub second kernel, and new kernels come after K1 |

## Out of scope

- New kernels (systolic, float, tiled MVU, LUT): after K1, each is a
  registered implementation.
- DSE among several feasible candidates.
- FIFO sizing.
- The `inner_shuffle` fix (FinnLib).
- The HLS synthesis stage.

## Estimate

About five to seven weeks:

| Increment | Size |
|---|---|
| E0 | 2–3 days |
| E1 | 3–5 days |
| V1 | 2–3 days |
| K1 | 1–1.5 weeks |
| K2 | 1 week |
| S4 | 1–2 weeks |
| G1 | 1 week |

The human gates are G0 and E0's review; the rest are reviewed as they land.
