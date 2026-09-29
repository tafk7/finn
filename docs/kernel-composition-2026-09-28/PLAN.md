# Plan: kernel choices, one interface per port, composable kernels

Date: 2026-09-28. Status: **approved; executing** (see Execution). Follows the D10 stream model
([`../stream-model-2026-09-27/RECORD.md`](../stream-model-2026-09-27/RECORD.md)).
Plan of record: [`../kernel-status-2026-09-27/STATUS.md`](../kernel-status-2026-09-27/STATUS.md).

## Goal

Every level of the hardware stack is a `Kernel`, and every choice among
kernels is a `Decision`. No new abstraction level is added: the work refines
`Decision` and standardizes `Kernel`. When done:

- a choice among kernels is one `Decision` over kernel classes, with the
  bindings every candidate shares written once and each candidate's own
  bindings on its entry;
- a composite binds facts (its ports' streams); each kernel owns its own
  choices (a dot-product core's PE and SIMD), keyed under it;
- a kernel declares one node per interface (a `Port`), one schedule, its
  admission and its parameters; the `Kernel` base derives the ABI, build
  requirements, tie-offs and exports;
- kernels compose into kernels: a Design is a composite kernel, and an ONNX
  node becomes a composite kernel with its Decisions.

## The refined Decision

```python
class MatMulKernel(Kernel):
    compute = Decision(
        {
            "packed": PackedDotpKernel(narrow_weights=narrow_weights),  # entry: its own bindings
            "int8_dsp58": Int8Dsp58DotpKernel,                          # entry: the bare class
        },
        x=activations, w=weight_stream, y=results,                     # shared: merged into each entry
    )
    memory = Decision(
        {"rom": RomKernel(rom_style=rom_style), "memstream": MemStreamKernel(writable=writable)},
        optional=True,                                                  # None: weights come from outside
        contents=datapath_weights, output=weight_stream,
    )
```

- **Entries.** A candidate entry is a kernel class, or a call on it carrying
  the bindings only that candidate takes. An entry places a fresh node when
  selected, as `values={...}` does today.
- **Shared bindings.** Keyword arguments are merged into every entry. Each
  must be declared by every candidate, or compilation fails naming the
  candidates that lack it (no silent skipping). A shared binding also written
  on an entry is a double assignment, refused as in any body.
- **`optional=True`** adds a `None` candidate: nothing is placed, and the
  bindings go nowhere.
- **Reads.** `self.compute.y` reads a member every candidate declares with the
  same type, checked at compile. Any other member is read qualified,
  `self.compute["packed"].narrow_weights`, and is inapplicable when another
  candidate is selected.
- **Candidate choices.** A candidate's own Decisions are never mentioned by the
  composite. They are keyed under it (`compute.packed.pe`), pinnable from
  outside, and inapplicable when another candidate is selected.
- **Overrides.** An outer layer can still assign a candidate's input through
  the qualified path (the engine already lets a path reach through a
  candidate).
- **Errors, not refusals.** An unsupplied required input of a candidate is an
  authoring error naming the candidate, not an incompatibility.
- **`required()`.** A base declares members every subclass must define
  (`schedule = required(Schedule)` on `Kernel`); a class that leaves one
  unmet cannot be placed, and naming it in a Decision is a compile error.
- **`settle(point)`** commits every Decision with exactly one compatible
  candidate. Several survivors remain a design choice (future DSE).

Once ports read streams (F8), a stream's tensor carries the shapes and element
types, so shared bindings shrink to the streams: the "same interface" of the
candidates.

## Settled in discussion (2026-09-28)

| # | Decision |
|---|---|
| F1 | A realization of an operation **is** a Kernel; there is no separate realization or operation level, and no family, interface or registry concept. Reuse is by importing the same kernel classes into another Decision (a shared set may be a module constant). Registration is revisited later |
| F2 | A Design **is** a composite Kernel: one generated top module, built from children wired by streams |
| F3 | A choice among kernels is the refined `Decision` above: class or call entries, shared bindings merged strictly, `optional=`, direct reads of common members, qualified reads of the rest |
| F4 | No `Contract` or `Composite` classes. Required members are `required()`; leaf or composite follows from what a kernel declares (its module and `parameters()`, or children wired by streams) |
| F5 | One name per concept: `Index`, `Schedule`, `BeatSequence` (was `Presentation`), `Tensor`. Retired: `Level`, `Nest`, `Access`, `Iteration`, `Einsum`, `fold`, `accesses`, `Contraction`. The same concept under two names in two candidates is renamed, not mapped |
| F6 | Canonical GEMM notation: indices `m`, `n`, `k`; `Form.DENSE` (X `(m, k)`, W `(k, n)`, Y `(m, n)`) and `Form.DEPTHWISE` (X `(m, k, n)`) |
| F7 | A composite binds facts; each kernel owns its choices. The schedule and its folding Decisions belong to the kernel that folds (PE and SIMD are the dot-product core's) |
| F8 | One `Port` node per interface: its indices, lane order, element admission, pins and stream reference; it exports its own `PORT`. Shapes and elements come from the stream's tensor |
| F9 | The `Kernel` base standardizes codegen and composition plumbing: one `admission` group, `module`/`sources`/`clocking`/`parameters()`, derived ABI, `build_requirements`, tie-offs of unbound ports and buses, and exports; a composite's netlist likewise |
| F10 | Weight memory stays a discrete kernel choice in the composite (a `memory` Decision over `RomKernel` and `MemStreamKernel`), not part of `Stream`, for now. Its rules (writable and several sets need memstream) become the memory kernels' admission |

## Human gate G0: decisions before building

| # | Decision | Answer | Blocks |
|---|---|---|---|
| 1 | Where weight memory lives | **settled (F10)**: a `memory` Decision in the composite | K1 |
| 2 | Weights stored `(k, n)`, as ONNX `MatMul` and FINN's MVAU initializer, not the kernel layer's current `(n, k)`. Bank conflicts do not bear on it: the memory image follows the consumer's traversal, not the tensor's axis order | **kept for now** (user, 2026-09-28, after V1 built it); to be revisited in detail, specifically how weight files (memory images) are generated | V1 |
| 3 | `required()` ships with the Decision refinement | **agreed** | E0 |
| 4 | Registration of kernels | **dropped**: candidates are listed in the Decision; revisit registration later | — |
| 5 | A port node's contract is named, in `netlist`, after the nearest ancestor that owns a module (`u_compute_packed` stays the core's instance, not `u_compute_packed_x`) | **agreed** | K1 |
| 6 | `replay_buffer` | **stays scrapped** | — |

## Execution

The whole plan is executed in one run, increment by increment:

| Step | Done by | Review |
|---|---|---|
| E0 | a dedicated subagent, briefed with this plan | an independent review (the lead, or a second agent) against F3 and the E0 questions; findings resolved before E1 |
| E1 | the lead, on the reviewed E0 API | gates below |
| V1 … G1 | the lead | each increment's gate; reviewed as it lands |

- The user is not a gate at E0; the reviewed API and its record are reported
  with E1. The user may stop the run at any increment.
- Each increment commits locally with explicit paths; nothing is pushed
  without asking.
- Each increment appends to `RECORD.md` beside this plan: what landed, gate
  results as observed (XSim runs in the background, one simulation per
  process; results not seen are not reported), key and name changes, and
  deviations from this plan.
- A deviation that changes a settled decision (F1–F10) stops the run for the
  user.

## Increments

Each ends at the kernel and dataflow gates plus the XSim sweeps it touches,
records key, name and fingerprint changes (D7), and lands on its own.

### E0. Engine probe (2–3 days, no production code)

- **Work.** A probe in `docs/` building the refined Decision on today's
  engine, or on a spike branch where the engine must change, over five cases:
  1. the dot-product core: shared streams, a binding unique to one core;
  2. weight memory: optional, unique bindings per entry;
  3. a stream's adapter: seven entries, shared `tensor` and `plan`, guarded,
     and a binding (`ram_style`) that six of the seven take;
  4. a stream's transport;
  5. two candidates with disjoint choices (the packed core against a stub
     core with other folds, keys `compute.packed.pe` against
     `compute.stub.rows`).

  It also covers:
  - the strict errors: a shared binding a candidate lacks, a double
    assignment, an unsupplied required input, an unmet `required()`;
  - direct reads of common members, qualified reads inapplicable when another
    candidate is selected;
  - namespaced keys, narrowing, pinning, and switching candidates (the
    stale-choice rule);
  - selections that capture and restore by key and refuse an unknown entry;
  - collapse on and off;
  - declaring a composite programmatically (for G1) with stable keys.
- **Questions it answers.**
  - Shared bindings named like the Decision's own arguments (`when`,
    `optional`, `values`, `domain`): today's memory kernels take `values`.
    Either those Params are renamed (`contents`) or shared bindings are
    passed differently.
  - Whether `values={...}` over nodes is kept beside the positional entries or
    migrated to them (`values=` then means scalar choices only).
  - A binding most but not all candidates take (case 3): written on each
    entry, or taken from a stream-level Decision the entries reference.
- **Deliverables.** Under `docs/kernel-composition-2026-09-28/e0/`: the probe
  code and its tests (runnable with the kernel venv), and `REPORT.md` with the
  API as built, each case's result, answers to the three questions, the exact
  engine changes E1 must make (files, functions, new errors), and open risks.
  No change to `src/`.
- **Gate.** An independent review of the probe and its API (see Execution).
  The engine design is fixed here.

### E1. The engine refinement (3–4 days)

- **Work.** The refined `Decision` (entries, shared bindings, `optional=`,
  reads), `required()`, and `settle(point)` beside `commit`/`compatible`.
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
  - Weights `(k, n)` if G0.2 is agreed.
- **Removed.** `Level`, `Nest`, `Access`, `Iteration`, `Einsum`, `fold`,
  `accesses`, `Presentation`.
- **Gate.** The S0/S2 equalities re-expressed over `Index`/`Schedule`
  (`test_nest.py` becomes `test_schedule.py`); plan tests unchanged.

### K1. The Kernel protocol, ports, and MatMul (1–1.5 weeks)

- **Work.**
  - The `Kernel` base (F9) and `Port` nodes (F8), including `netlist`'s
    instance naming (G0.5).
  - The dot-product cores (`PackedDotpKernel`, `Int8Dsp58DotpKernel`): leaf
    kernels on ports, with their own PE and SIMD Decisions, schedule and
    admission.
  - `MatMulKernel`: facts, `Form`, ports; a `compute` Decision over the cores
    and an optional `memory` Decision (F10), in the refined form.
  - `matmul_assembly` becomes facts, then `commit`, then `settle`.
- **Removed.**
  - dotp's `*_dtype`/`*_type`/`axi_stream`/`*_stream` slots, `iteration`,
    `dotp_presentations`/`DotpPresentations`, and the element cross-check.
  - MatMul's `pe`/`simd` Params, `contraction_iteration`, `_Folding`'s checks,
    the per-candidate repeated bindings, and `commit_adapters` (subsumed by
    `settle`).
  - `delivery`, replaced by `memory`; `CyclicDelivery` becomes `RomKernel`.
- **Keys.** `pe`, `simd` → `compute.<core>.pe`, `.simd`; `delivery.*` →
  `memory.*`. Recorded under D7.
- **Gate.**
  - Module parameters identical per configuration over the identity dump's
    14 configurations. Wrapper fingerprints change with instance and key
    names; each change is recorded.
  - All MatMul XSim sweeps, with the numeric harness run per core.

### K2. Every other kernel on the protocol; adapters and transport (1 week)

- **Work.**
  - Thresholding, eltwise, FIFO, memstream, `input_gen`, `vpc` (a new leaf
    `VpcKernel`), `inner_shuffle` and the cyclic stream move onto the
    `Kernel` protocol and `Port` nodes.
  - A stream's `adapter` and `transport` Decisions move to the refined form.
    Adapter chains place `InputGeneratorKernel`/`VpcKernel` children instead
    of calling requirement builders.
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
    module (for example, whether the ROM shares the compute module).
  - A Design is a composite kernel placing kernels, Decisions over kernels,
    and the streams between them.
- **Gate.** Two MatMuls and a thresholding in one Design, fused and unfused
  variants equal in XSim; the D10 two-kernel demo re-expressed as a Design.

### G1. The graph adapter, for MatMul (1 week)

- **Work.**
  - An ONNX `MatMul` node (with or without a weight initializer) becomes a
    `MatMulKernel` in a Design, built from the node's facts with real
    initializers (the model-context lesson).
  - FINN node attributes are emitted from the selected kernels' decisions
    ("kernel owns nodeattrs"), hermetically.
- **Gate.** A small ONNX model through the adapter, `settle`, and XSim.

## Order and dependencies

```text
G0 ──► E0 ─(review)─► E1 ──► V1 ──► K1 ──► K2 ──► S4 ──► G1
```

V1 can start beside E1, since it touches values only. K1 needs E1 and V1.

## Risks

| Risk | Mitigation |
|---|---|
| The refined Decision interacts badly with guards, collapse or selections | E0 covers the five real cases, collapse on and off, and selection round trips before any migration |
| Shared-binding names collide with the Decision's own arguments | E0 decides the rule (rename the Params, or a different way of passing shared bindings) |
| Key and name churn across every kernel | recorded per increment (D7); module parameters must stay identical, so only names and wrappers change |
| Port nodes exporting contracts break instance naming or `Users` attribution | G0.5's rule; K1's gate checks every instance name and per-port refusal |
| Nested lowering fails | S4 verifies it first, before building on it |
| Programmatic composite declaration conflicts with class-body compilation | probed in E0; G1 is last |
| Scope creep into new kernels (systolic, float) | out of scope: E0 uses a stub core, and new kernels come after K1 |

## Out of scope

- New kernels (systolic, float, tiled MVU, LUT): after K1, each becomes an
  entry in the Decisions that offer it.
- Kernel registration and discovery (revisit later).
- Weight memory as part of `Stream` (revisit after S4).
- DSE among several compatible candidates.
- FIFO sizing.
- The `inner_shuffle` fix (FinnLib).
- The HLS synthesis stage.
- The detailed revisit of weight layout and weight-file generation (G0.2).

## Estimate

About five to six and a half weeks:

| Increment | Size |
|---|---|
| E0 | 2–3 days |
| E1 | 3–4 days |
| V1 | 2–3 days |
| K1 | 1–1.5 weeks |
| K2 | 1 week |
| S4 | 1–2 weeks |
| G1 | 1 week |

G0.2 is kept for now and flagged for a detailed revisit (weight-file generation).
E0's review is done by an agent; the rest are reviewed as they land. The user
accepted E1/K1's engine additions (`Users` through forwarded inputs, `settle`
treating a pending admission as not refused), the `build_requirements` and
`tieoffs` views on every kernel, and retiring the mapping form at K2's close
(2026-09-28).
