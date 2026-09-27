# Kernel layer status and plan

Date: 2026-09-27. Worktree `finn-kernels-extraction`, branch
`feature/kernel-package-extraction`, head `e218f9f76` plus this record. Nothing
is pushed. This supersedes [the 2026-09-25 status](../kernel-status-2026-09-25/STATUS.md).

## 1. Where things stand

| Commit(s) | Increment | Record |
|---|---|---|
| `b244ec8be`..`545981eea` | Declarative Space model landed: engine, tests, kernels, design record, R6 duplicate-finding fix, landing record | [design](../space-declarative-2026-09-26/DESIGN.md), [landing](../space-declarative-2026-09-26/LANDING.md) |
| `902efb29f` | Robust MVAU task spec | [spec](../robust-mvau-2026-09-26/SPEC.md) |
| `6aa0383cf` | Kernel README: `BufferedStream` | — |
| `e218f9f76` | Kernel audit on the landed model | [audit](../kernel-audit-2026-09-26/AUDIT.md), [coverage](../kernel-audit-2026-09-26/COVERAGE.md), [roster reconciliation](../kernel-audit-2026-09-26/ROSTER-RECONCILIATION.md) |

**Validation at the landing (`545981eea`):**

- Space 423, kernels 763 with 0 skipped, dataflow 16; format, lint and strict
  mypy clean.
- XSim 9/9. MVAU numeric XSI 28/28: 20 direct, 8 through a depth-2 weight FIFO.
- MVAU fingerprints are bit-identical to `0d700b1ab` once the one accepted
  rename is mapped back: the cyclic delivery instance `u_weights` is now
  `u_implementation_cyclic`.
- No decision-key or top-level ABI change.
- The canonical Space documentation in `scratchpad/space/` is rewritten, and
  24/24 examples pass. It is untracked there.

**The Space model in one paragraph.**

- *Nodes and choices.* Calling a family declares a node (`Room(area=12)`).
  `design_space(...)` compiles it, and configurations follow by `with_choices`.
  A structural choice is a `Decision` over nodes.
- *Typing.* Declarations are typed as the values they stand for, and formals are
  annotated with their value type (`area: int = Param()`).
- *References and edges.* References are attribute access (`kitchen.finish`).
  Edges are assignments, and parents may override any descendant's data, with
  provenance recorded.
- *Graph primitives.* Reference inputs, with `Users`/`Members`, carry streams as
  ordinary Spaces that kernels reference.
- *Views and compilation.* Views read as values. Value-derivation chains
  collapse at compile time; Space scopes never do.

## 2. Decisions taken this cycle

| # | Decision |
|---|---|
| D1 | The declarative Space model is adopted and landed. The design history stays on the `spike/*` branches and in the design records |
| D2 | The cyclic delivery instance rename (`u_weights` → `u_implementation_cyclic`) is accepted |
| D3 | **Every worktree or clone has its own `deps/`.** A shared symlinked FinnLib is not allowed. Environment-management checkouts that don't use `deps/` are exempt |
| D4 | **FinnLib is consolidated on one commit on a remote:** the newer `rtl/infra/…` layout, with the `replay_buffer` and dotp-backpressure fixes carried onto it |
| D5 | **Runtime-writable and multi-set weights use the RTL memstream, ported into FinnLib.** The source is FINN's `finn-rtllib/memstream/` (`memstream_axi`, `memstream`, and the `axilite` adapter): AXI-Lite write, `SETS` with a set-index stream, `INIT_FILE`, `RAM_STYLE`, `clk2x`. *Revised 2026-09-27:* this was first "HLS memstream now, RTL later". An HLS memstream needs an HLS-synthesis stage before `netlist` can place it, and has no `SETS`. The HLS synthesis stage becomes separate future work (Phase D) |
| D6 | **The dotp core choice (R4) is deferred.** FinnLib's `dotp_axi` picks its INT8/DSP58 or soft-vector core internally. Soon, examine and decide how to split them into separate kernels or choices properly |
| D7 | Key and ABI changes the increments need are accepted when recorded: replay becoming a `Decision`, the fused-activation stream, and dropping the unpumped `ap_clk2x` top pin (audit H3) |
| D8 | Per-port refusal attribution is fixed now, in J2. A stream input names the one port it presents, so a refusal reaches only its own stream (audit H4) |
| D9 | The audit's reordered plan is approved: VVAU before thresholding, with J2 and J4 added (audit H6) |

## 3. Known problems

All of these are verified in the audit (§/probe references there).

| Problem | Why it matters |
|---|---|
| **dotp accepts streams it can't consume** (`k/dotp.py:302-317` checks only the element type) | A wrong lane count or a transposed weight order still builds a netlist. MVAU is correct only because its own specs are |
| **Refusals are reported by every stream a kernel touches** | One `ports` export per kernel: a bad weight dtype shows on 3 streams |
| **No non-stream interfaces** | AXI-Lite and sideband buses can't be driven or exported, so thresholding can't compose (68 undriven bits). A padded AXIS child can't feed a child |
| **FinnLib pinned three ways** | The kernels use `b17eae6a` (flat layout, with the fixes). `fetch-repos.sh` pins `dfeafac8` (new layout, no fixes). Upstream has neither |
| **Shared `deps/finnlib`** | A symlink to a clone that other sessions use |
| **Clock and reset routed by pin name** | `"clk2x" in name` in `streams.netlist`, and an unpumped design still exposes `ap_clk2x` |
| **Only 3 of 11 kernels use the stream idiom** | Thresholding, eltwise and the input generator are standalone |
| **HLS kernels can't be placed by `netlist`** | They yield HLS source requirements with no pin ABI. No HLS kernel is on the current path (D5); the HLS synthesis stage is future work (Phase D) |
| **An unused stream is refused**, and a boundary needs its port name declared up front | Optional streams need explicit `when=` guards |

## 4. Plan

Each step is independently landable and ends at a review gate. Nothing below
has started.

### Phase A: foundations

| Step | Content | Exit criteria |
|---|---|---|
| **A1. Per-worktree `deps/`** (D3) | Make `fetch-repos.sh` produce real per-checkout clones. Replace the `deps/finnlib` symlink in this checkout and in the spike worktree. Document the `FINNLIB_ROOT` override for local FinnLib development | Each checkout builds from its own pinned `deps/`; gates green |
| **A2. FinnLib consolidation, J0** (D4, D5) | One commit on the fork, pinned in `fetch-repos.sh`, which also ports `memstream_axi`, `memstream` and `axilite` from `finn-rtllib`, with their testbench if it's reusable. Update the kernels' source manifests to the new layout. Re-baseline the fingerprints once and record the reason (source identities change). Open upstream PRs for `replay_buffer`, the dotp fix and the memstream port if wanted | One pin everywhere; gates, XSim and the MVAU numeric sweep green; fingerprint change recorded |

### Phase B: correctness and infrastructure

| Step | Content | Depends on |
|---|---|---|
| **B1. J2: ports that check their streams** | Every stream port verifies lanes, form, repetition and markers against its stream. Per-port attribution (D8): the small engine addition where a stream input names its presented port, and `Users` yields only that port | A2 |
| **B2. J3: clock and reset as a Space** | A clock-domain family referenced by kernels. The netlist drives pins from references, not names, and the unpumped `ap_clk2x` is dropped (D7) | A2 |
| **B3. J4: non-stream interfaces** | AXI-Lite export and tie-off, sideband buses, and padding disposal between children | B2 |

### Phase C: robust MVAU capabilities (the audit's order)

| Step | Content | Depends on |
|---|---|---|
| **C1. J5: VVAU reuse**, plus a reusable delivery slot | Same families, `ACTIVATION_BROADCASTING=0`, a marker generator instead of replay. Needs E-048 (SWG→VVAU lane order) re-derived | B1 |
| **C2. J6: fused thresholding** | Migrate thresholding to the stream idiom; compose MVAU → Thresholding | B1, B3 |
| **C3. J7: replay as a choice** | `replay_buffer` or `input_gen`, as a `Decision` over nodes (D7) | B1, B2 |
| **C4. J8: an RTL memstream kernel, plus runtime-writable weights** (D5) | A fixed-interface kernel over FinnLib `memstream_axi`: `DEPTH`, `WIDTH`, `SETS`, `RAM_STYLE`, and pumped memory through the clock-domain Space (B2). Initial contents go through `INIT_FILE`: check whether the artifact layer's data-file contribution supports it, and add one if not. Its AXI-Lite bus is exported at the MVAU boundary (B3). MVAU gains a writable delivery candidate, and the numeric XSI harness gains an AXI-Lite write driver. Decide whether it replaces the kernel-local `cyclic_stream.sv` for read-only delivery, which needs equivalence evidence | A2, B2, B3 |
| **C5. J9: multi-set delivery** (R7, MLO) | `SETS > 1`, with the set-index stream as an ordinary stream reference input of the memstream kernel. MLO drives the index from outside the op (V10) | C4 |
| **C6. J10: stream adapters** | Width conversion, lane regroup and reorder as a `Decision` over adapter nodes inside a stream, chosen by `classify()` | B1, C3 |

### Phase D: planned and deferred work

| Item | Plan |
|---|---|
| **HLS synthesis stage** (future work) | Makes HLS kernels placeable, and is the path to HLS backends as choices, e.g. HLS MVAU and HLS thresholding (the audit's "HLS backend as a whole" gap). It needs five parts: (1) a background Vitis HLS runner with store caching keyed by sources, part, clock and tool version; (2) a generated-source contribution type, with tool version in build identity; (3) the pin interface predicted from the HLS interface directives for `netlist`, and verified after synthesis with `artifacts/rtl.check_abi` (option a, which reverses today's "no predicted HLS ABI" stance; the alternative, option b, is a two-stage flow binding the synthesized interface as a fact); (4) control-mode handling (`ap_ctrl_none` for free-running blocks); (5) an opt-in slow test tier. Plan it when HLS backends are wanted |
| **dotp core split** (D6, R4) | Near term: examine FinnLib's `dotp_axi` core selection and the FINN wrapper-split precedent. Then decide how the cores become separate kernels or a `Decision` |
| **Query and search tools** | The next Space-engine pass (design record §3). The audit's "queries wanted" list is its input |
| **Reusing cached results across snapshots** | A local edit re-runs the whole graph. It matters once search runs over graphs of many kernels |
| **Other standalone kernels** (eltwise, input generator) | Move them to the stream idiom when a composite needs them |
| **`finn.dataflow` model pass** | After robust MVAU |
| **Diagrams** in `scratchpad/space/diagrams/` | Regenerate them for the new API |

### Coordination

- **Artifact-integration SPEC** (`docs/kernel-artifact-integration-2026-09-25/`,
  owned by another session): §1 and §6 need re-baselining on the landed API.
  The mapping is in [LANDING.md §5](../space-declarative-2026-09-26/LANDING.md).
- **External-resources merge (`integration/er-kpe`):** re-merging it conflicts
  in 11 files. Its removal of `typing_extensions` conflicts with the engine,
  which already depended on it before this landing.

## 5. Open questions for later

1. **The memstream boundary:** whether the RTL memstream also replaces the
   kernel-local `cyclic_stream.sv` for read-only delivery (C4).
2. **The upstream path for FinnLib:** which fixes go upstream, and on what
   schedule?
3. **The dotp core split (D6):** separate kernels, or a `Decision` within one
   dotp family?
4. **The Space model's open questions** (design record §10): a view used in its
   own class body (`cast`), the path form of `inspect`, and redundant
   obligations.
