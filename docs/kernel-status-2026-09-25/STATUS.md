# Kernel layer status and direction

Date: 2026-09-25. Worktree `finn-kernels-extraction`, branch
`feature/kernel-package-extraction`, head `ee5e8852e`. FinnLib
`b17eae6a` on `kernel-contract-refinement-20260925`, unchanged by this work.
Nothing is pushed. The generic Space engine (`src/finn/core/space`) has not
changed since `b2d01750a`.

## 1. Where things stand

| Commit | Increment | Record |
|---|---|---|
| `6ff394315` | Scalars and typed ports as handwritten Spaces; MVAU weight delivery as a `SubspaceChoice` | [composition](../kernel-composition-2026-09-25/REVIEW.md) |
| `31dcee3a1` | Review corrections found by the roster pass | same |
| `ae7f8a0cd` | Stream contracts, checked `Composition.connect`, reusable `CyclicDelivery` | [stream contract](../kernel-stream-contract-2026-09-25/REVIEW.md) |
| `240c33a11` | Canonical loop-nest `Traversal` replacing the ad hoc forms; `classify()` adapter classification | same, "Revision" section |
| `ee5e8852e` | Declared `Stream`s, kernel `Port`s, `assemble_streams`, optional FIFO transport slot | [declared streams](../kernel-declared-streams-2026-09-25/REVIEW.md) |

Validation at `ee5e8852e`:
- 301 Space tests and 757 kernel tests pass, with no skips.
- Format, lint and strict typing are clean.
- 18 documentation examples pass.
- MVAU XSim: 20 of 20 with direct transport, and 8 of 8 through a depth-2
  weight FIFO.
- Out-of-context synthesis confirms that `rom_style` changes memory inference.
- The eltwise + `CyclicDelivery` composition passes in XSim.

Supporting analysis (in scratchpad, untracked there):
- `open/kernel-roster-map/`: FINN family → FinnLib mapping, delivery and
  stream evidence.
- `space/{DESIGN,AUTHORING,MIGRATION}.md` updates.

## 2. Current architecture

```text
finn.core.space            generic Space engine (unchanged)
finn.kernels
  datatypes/               QONNX identity, Integer policy, Scalar Spaces, ScalarEncoding
  physical/forms.py        Traversal (canonical loop nest), classify(), pack(), Every
  physical/contract.py     StreamContract, compatibility()
  physical/composition.py  Composition: drive(), connect() → PhysicalStructure
  physical/ports.py, axi_stream.py   typed native / AXIS ports over scalars
  streams.py               StreamSpec, Stream, Port, TopInput/TopOutput, StreamFifo,
                           assemble_streams()
  delivery.py              CyclicDelivery (any traversal, rom_style)
  streaming.py             ReplayBuffer kernel, replay/cyclic requirements
  dotp.py, mvau.py, ...    kernels; MVAU declares streams and placements
  artifacts/               detached build specifications, preparation, store
```

What the evidence settled:
- Handwritten Spaces with small factories beat generated builders for kernel
  interfaces.
- `SubspaceChoice` needed no new API for a real heterogeneous consumer.
- One canonical traversal reproduces the tiled MVU's hard-coded `input_gen`
  parameters and FINN's OuterShuffle coefficients.
- Delivery works equally inside or beside an operation.

## 3. Known problems

| Problem | Why it matters |
|---|---|
| **The `assembly` construct.** `MVAU.assembly` calls `assemble_streams`, which rediscovers the topology by inspecting class attributes, reads every component, raises on the first mismatch and returns a bundle (`MVAUAssembly`) | It bypasses the linker, which already knows the bindings. Connection checks are not constraints, so refusals aren't attributed to streams, and one failure hides the others. One opaque callback depends on everything. The bundle mixes folding facts, choice state and build products. See Section 4 |
| **Ports are formals.** Placing a stream-capable kernel outside any stream still requires binding its optional ports | An ergonomic cost on every standalone placement (dotp test fixture) |
| **Selector access.** A selector or case decision is reached through `inspection.choices` / `inspection.decisions` keyed lookups | Repeated in the adapter, tests and README; three consumers now |
| **Generic helpers inside MVAU.** `_encoding`, `_selector`, `_findings` and the adapter's commit-by-key plumbing | They belong in `datatypes.scalar` and a keyed configuration helper (`tests/kernels/helpers.py::point_for` already does this) |
| **Physical-profile limits** | A padded AXIS child cannot feed another child (partial padding disposal is not expressible); no data fan-out |
| **Forms are declared, not negotiated** | Ports don't publish the traversals they support; `classify()` names adapters but nothing inserts them |
| **Hygiene** | FinnLib `b17eae6a` (which adds `replay_buffer.sv`) is on no remote. Scratchpad records are untracked. The pinned `space/MVAU-EXAMPLE.md` predates these APIs |

## Update (same day)

- **Section 4 is implemented.** See the declared-streams REVIEW, section
  "Revision: Space-native streams". Streams now have explicit endpoints,
  per-stream constraints and connection views, a pure `compose` reduction and
  the `configure` helper. `ScalarEncoding.admit` is added, and `MVAUAssembly`
  is adapter-only. Build identity is unchanged.
- **Decisions taken.**
  1. The restructure landed before the artifact-integration SPEC, which is
     untouched.
  2. Instance names are explicit on stream endpoints, so no engine change was
     needed.
  3. FinnLib `b17eae6a` is pushed as `origin/kernel-contract-refinement-20260925`
     on the personal fork.
  4. Only `open/kernel-roster-map/` is committed in scratchpad (`f601063`).
     `space/` remains untracked there.
- **Still open.** Negotiation and adapter insertion, the robust-MVAU items, MX
  (Section 5), and the physical-profile limits.

## 4. Direction: Space-native streams

**Goal.** Each connection is an ordinary part of the Space: it has its own
endpoints, its own check and its own wires. The parent's physical output is a
thin reduction over them, not a procedure that rediscovers and decides.

```text
Stream weight_stream = Stream(spec,
                              producer=implementation.accepted(OUTPUT_PORT),
                              consumer=compute.accepted(DotpAxiKernel.weights_port))
  spec          logical sequence (as now)
  compatible    @constraint, owns its refusal ("weight_stream: needs a reorder adapter …")
  wires         @derived, this stream's wires, FIFO halves included
  transport     direct | fifo (as now)

MVAU
  streams + placements          (as now)
  structure  @view  = modules of the placements + every stream's wires   (reduction only)
  build_requirements @view = lower(structure)
```

### Steps

1. **Per-port accepted views.** Kernels publish one accepted view per port
   (for example `weights_port`), replacing the dict-shaped `component`. Each
   port's module comes from the kernel's existing `build_requirements`. The
   views are explicit and typed, matching the artifact-integration SPEC's
   preference for explicit typed view references over discovery by name.
2. **Explicit stream endpoints.** A `Stream` names its producer and consumer
   through accepted refs. The linker owns the topology, and class introspection
   is deleted. Binding a port to `stream.spec` stays, so consumers still receive
   the logical spec.
3. **Checks as constraints, wires as derived values.** `compatible` becomes a
   `StreamLink` constraint and `wires` a derived value computed with the
   existing `Composition` rules, one stream at a time. The FIFO case
   contributes its module and splits the wires.
4. **A thin parent reduction.**
   - `structure` collects modules and wires, derives the top ABI from the
     boundary ports, and validates.
   - `build_requirements` lowers it.
   - `MVAUAssembly` dissolves:
     - beat counts become `folding` fields;
     - delivery mode and initializer are read from the choice and its case;
     - `mvau_assembly()` keeps returning the same record, built from those.
5. **Helper cleanup.**
   - Move `ScalarEncoding` admission into `datatypes.scalar`.
   - Promote a keyed configuration helper (facts plus choices by decision
     key, with readable refusals).
   - Rewrite the adapter, tests and README on it.

### Open question to settle first

A stream's wires need instance names for its endpoints, and today they are
placement names. Check whether a computation can learn a placement's scope name
through public API. If it can't, the choices are a small, justified engine
addition (a scope-name read) or explicit instance names on placements.
Everything else needs no engine change.

### Acceptance criteria

- Refusals are attributed to individual streams, and independent streams settle
  independently. A test shows two streams refusing at once.
- `inspection.explain` for `build_requirements` shows per-stream nodes, not one
  callback.
- MVAU's structure, wrapper bytes and requirements identity are unchanged for
  equal configurations, or any change is recorded with its cause.
- All existing MVAU structural assertions, XSim (direct and FIFO), the
  eltwise + delivery composition and the gate pass.
- No class-attribute discovery remains in `streams.py`.

### Coordination with the artifact-integration SPEC

`docs/kernel-artifact-integration-2026-09-25/SPEC.md` (untracked, from another
session) is based on `240c33a11`, which predates declared streams.

- Its section 4 (nested generated-module composition) consumes the same
  lowering boundary this work reshapes.
- Its section 1 handoff (`point.build_requirements()`) is unchanged by it.

Land the Space-native streams restructure first, or coordinate explicitly, so
that section 4 builds on explicit per-stream wires rather than on
`assemble_streams`.

## 5. After that

- **Negotiation and adapters.** Ports publish the traversal families they
  support. A stream owns a `form` Decision over their intersection. Adapter
  kernels (`input_gen` reorder/replay, inner shuffle, width converter) become a
  declared category with transform signatures, and insertion becomes a choice
  of adapters on a stream. `classify()` already supplies the parameters.
- **Robust MVAU**, per the roster:
  - multi-set delivery with a set-index sideband (serves MLO and eltwise);
  - AXI-Lite-writable delivery;
  - replay through `input_gen` with `Level(d)` markers;
  - dotp soft-vector/packed as an explicit choice (needs the FinnLib wrapper
    split);
  - MVAU → Thresholding composition.
- **MX.** A `BlockEncoding` at the datatype level; the scale travels as a
  derived second operand.
- **Carried defects:**
  - the census thresholding contract (V·NF threshold beats, not NF);
  - the SWG → VVAU lane order must be re-derived for FinnLib dotp.

## 6. Decisions needed

1. Proceed with Section 4, before or coordinated with the artifact-integration
   SPEC?
2. If a placement's scope name is not publicly readable from a computation:
   add a small engine read, or declare explicit instance names on placements?
3. Push FinnLib `b17eae6a` (or merge it) before further work depends on
   `replay_buffer.sv`?
4. Commit the scratchpad records (`space/`, `open/kernel-roster-map/`), which
   are untracked there?
