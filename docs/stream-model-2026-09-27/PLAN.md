# Plan: the stream model (D10), increment 1

Date: 2026-09-28. Status: **built, S0–S3** (record: [RECORD.md](RECORD.md)).
Design: [DESIGN.md](DESIGN.md).
Plan of record: [`../kernel-status-2026-09-27/STATUS.md`](../kernel-status-2026-09-27/STATUS.md).

## Goal

Streams carry a tensor. Each kernel presents its own traversal of that
tensor, derived from its iteration nest. The stream derives a plan relating
the two presentations, and realizes it with an adapter it chooses. When done:

- no composite writes a stream form by hand, and no kernel hand-checks one;
- a stream can join two differently folded or ordered ends through an admitted
  adapter, or refuses with the plan it could not realize;
- the same stream serves inside a module, at its boundary, and (next
  increment) between modules.

## Scope

**In:** design steps S1–S3 (presentations at the ends, the nest, plans and
adapters), across every kernel on streams (dotp cores, MatMul, replay,
delivery, memstream, FIFO, thresholding, input generator, adapters).

**Out, next increment:** S4, the `Design` composite (instance wiring across
modules, clocks, and packaging). Also out: S5 extensions (padding and
non-divisor folds, clock-domain crossing, FIFO sizing), HLS kernels, and
the `inner_shuffle` fix (a FinnLib task that runs in parallel).

## Human gate G0: decisions before building

| # | Decision | Answer (2026-09-28) | Blocks |
|---|---|---|---|
| 1 | Where the values live: `Tensor`, `Traversal`, `Nest`, `Access`, `classify`, `plan` in `finn.dataflow`; retire the untested logical model into `finn.parked` | **agreed** | S1 (the move) |
| 2 | Who fixes a module's boundary presentation | **a rule, not a Param**: a producer presents its data honestly, in its native form, and is never pre-adapted to a consumer; the receiver adapts. See below | S1 |
| 3 | Contraction syntax | **pending confirmation**: keep the `Contraction` enum as the Param, each member defined by an einsum; the nest derives everything from the einsum | S2 |
| 4 | Canonical marker | **agreed: `Level`**, anchored to the presentation's beat loops (not nest names, which do not cross kernels); `Every` derived at periodic pins (AXIS `TLAST`) | S2 |
| 5 | Reduction order | **agreed**: declare the Decision; compatibility leaves the canonical order only until DSE exists; float collapses to canonical | S2 |
| 6 | Delivery and replay | **delivery stays a Decision; flag unifying it with stream reuse for revisit after S3.** Replay moves onto the stream (S3), and **`replay_buffer` is scrapped**: `input_gen` is a superset and the resource saving is not worth a second kernel | S3 |

**Decision 2 in detail.** Compatibility does most of the filtering; what
remains is a design choice, and choosing among survivors is future DSE (as for
compute cores). Hence:

- A kernel's output presents its native traversal, derived from its nest.
- A kernel's input boundary presents what it consumes before any adaptation:
  the `once` form of its internal end. For MatMul that is today's `in0_V`.
- Any adaptation (replay, width, reorder, markers) is a stream adapter on the
  receiving side. Where that adapter is packaged at the design level is an S4
  question.
- Cheaper fixes on the producer's side, such as refolding it to match, are a
  producer folding Decision picked by future DSE, not a presentation knob.

## Increments

Each ends at both gates, records key, name and fingerprint changes (D7), and
lands on its own.

### S0. Expressiveness spike (2–3 days, no production code)

This retires the design's largest risk, K3: whether one nest model covers the
roster.

- **Work.** Extend `sketches/indexed_ports.py` to derive, from a nest and
  accesses, the forms of:
  - dense and per-channel MatMul, and the dense realization as a reshaped view;
  - tiled MVU (TH > 1), the forms already in `test_stream_contract.py`;
  - SWG / conv-as-matmul with stride and dilation;
  - thresholding, with PE ≤ C and PE > C;
  - eltwise with a broadcast operand;
  - a transpose.
- **Gate.** Each derives the known form, or is recorded as a gap with its
  extension (padding, wrap-around folds). A gap that the next two increments
  cannot tolerate returns to the design.

### S1. Presentations at the ends (1–2 days)

- **Work.**
  - `Stream.spec` becomes `Stream.tensor` plus an optional `boundary`
    presentation.
  - `StreamSpec` becomes `Presentation`, held by each end in its
    `StreamContract`.
  - Every kernel publishes its own presentation: MatMul computes the same
    forms as today but hands them to the ends, and thresholding derives its
    own.
  - Boundary contracts come from the G0.2 rule (derived, not a Param).
  - The values move to `finn.dataflow`; `finn.dataflow.model.logical` moves
    to `finn.parked`.
- **Stays.** All form checks, `netlist`, and every Decision.
- **Gate.**
  - Fingerprints and decision keys are identical.
  - The dense, per-channel, memstream and input-generator sweeps pass.
  - Both gates pass, plus the documentation examples.

### S2. The nest (3–5 days)

- **Work.**
  - `Nest`, `Level`, `Access`, and the operations `present`, `frame`, `once`,
    `period`.
  - MatMul declares its contraction as an einsum. PE, SIMD and reduction order
    are Decisions over nest levels, with tensor-extent domains, and every
    presentation is derived. `realization = dense` becomes a second
    contraction over a reshaped view.
  - The dotp cores take the nest and accesses, and derive broadcasting, the
    frame and their presentations. An admission rule replaces the labelled
    relation.
  - Thresholding and the set stream are derived.
- **Removed.**
  - The labelled relation, `axis_walk`, `walk_loops` and `split_walk`.
  - The hand-written `*_spec` derivations.
  - `channel_tile`, which becomes a derived presentation.
- **Gate.**
  - The S0 sketch equalities become unit tests.
  - Dense and per-channel fingerprints are identical, or every change is
    explained.
  - Every XSim sweep passes.
  - The B1 port probes are remapped: an admission refusal, or a plan (see
    DESIGN §4.5).

### S3. Plans and adapters (1–2 weeks)

- **Work.**
  - `plan(source, sink)`: chains of steps with markers recomputed after each.
    `compatibility` becomes a check of one plan step.
  - `Stream.adapter`, a Decision over nodes beside `transport`. Candidates
    bind to the plan and refuse themselves.
  - `netlist` wires a list of stages per stream.
  - Adapter candidates:
    - `input_gen`: replay leaves MatMul and becomes a reorder on the
      activation stream, so C3 moves here. It also synthesizes markers
      (identity coefficients, `olst[d]`), which replaces the per-channel
      `markers` node;
    - `vpc`;
    - `inner_shuffle`, once FinnLib is fixed.
  - `replay_buffer` is removed (G0.6): the `ReplayBuffer` kernel, MatMul's
    `replay` Decision and `markers` node, their exports, tests and the
    harness's `--replay` option. Before removing it, measure `input_gen`'s
    buffer for the identity (marker-only) case against `replay_buffer`'s
    zero-storage `REP=1`, and record the result.
- **Keys and names.**
  - Keys: `replay` and `markers` → `activations.adapter.input_gen.*`, and
    `BufferedStream.transport` moves onto `Stream` with its keys unchanged.
  - Instance names follow the adapter stages.
  - Both are recorded under D7.
- **Gate.**
  - For each adapter kind: XSim against an independent golden, free and
    stalled.
  - Mutation probes for compound plans (NF ≠ SF, swapped field orders).
  - A two-kernel demonstration: a producer and a consumer with different PE
    and order, joined by `vpc` + `input_gen`, correct in XSim, and refused
    when adapters are pinned to `None`.

### Close

- **Record and status.** The record, the evidence, and a STATUS update.
  C6's held part is closed by S3.
- **Canonical Space documentation.** Only if the engine is touched; none is
  expected.
- **Handoff.** A plan for S4, together with the artifact-integration work.

## Order and dependencies

```text
G0 ──► S0 ──► S1 ──► S2 ──► S3 ──► close ──► (next) S4 Design composite
        │                          ▲
        └─ inner_shuffle fix (FinnLib, parallel) ─┘
```

S0 can start before G0 closes; S1 needs G0.1 and G0.2. The FinnLib fix
gates only the transpose candidate.

## Risks

| Risk | Mitigation |
|---|---|
| The nest can't express a roster member (tiled MVU, PE > C) | S0 first; gaps are recorded as S5 extensions |
| A compound plan decomposes wrongly and an adapter moves data silently wrong | independent goldens per adapter kind, mutation probes, `classify` already checked against FINN's generators |
| Churn from moving replay into the stream (S3) | record the key and name changes; fingerprint diffs explained |
| Scope creep into the Design level | S4 is explicitly next; S3 stops at streams inside and at the boundary of one module |
| An engine change turns out to be needed | the design's engine probe says no; if one appears, it becomes a separate proposal, as in B1 |

## Estimate

About three weeks of work: S0 (2–3 days), S1 (1–2 days), S2 (3–5 days) and
S3 (1–2 weeks), each followed by an XSim sweep. The human gate is G0; the
later increments are reviewed as they land, as in Phase C.
