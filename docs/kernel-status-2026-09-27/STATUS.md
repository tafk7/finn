# Kernel layer status and plan

Date: 2026-09-27. Worktree `finn-kernels-extraction`, branch
`feature/kernel-package-extraction`. Nothing in this repository is pushed; the
consolidated FinnLib branch is pushed to the `tkeller/finnlib` fork (A2). This
supersedes [the 2026-09-25 status](../kernel-status-2026-09-25/STATUS.md).

**Update (same day): phases A and B are done.** Record, evidence and the
recorded fingerprint and ABI changes:
[`../kernel-phase-ab-2026-09-27/RECORD.md`](../kernel-phase-ab-2026-09-27/RECORD.md).

**Update (2026-09-28): Phase C is done through C6**, with one part of C6 held
for D10 (below). Record and evidence:
[`../matmul-kernel-2026-09-27/RECORD.md`](../matmul-kernel-2026-09-27/RECORD.md).
The D10 design round is written for review:
[`../stream-model-2026-09-27/DESIGN.md`](../stream-model-2026-09-27/DESIGN.md).

## 1. Where things stand

| Commit(s) | Increment | Record |
|---|---|---|
| `b244ec8be`..`545981eea` | Declarative Space model landed: engine, tests, kernels, design record, R6 duplicate-finding fix, landing record | [design](../space-declarative-2026-09-26/DESIGN.md), [landing](../space-declarative-2026-09-26/LANDING.md) |
| `902efb29f` | Robust MVAU task spec | [spec](../robust-mvau-2026-09-26/SPEC.md) |
| `6aa0383cf` | Kernel README: `BufferedStream` | — |
| `e218f9f76` | Kernel audit on the landed model | [audit](../kernel-audit-2026-09-26/AUDIT.md), [coverage](../kernel-audit-2026-09-26/COVERAGE.md), [roster reconciliation](../kernel-audit-2026-09-26/ROSTER-RECONCILIATION.md) |
| `56475970c` | A1: per-checkout `deps/` | [record](../kernel-phase-ab-2026-09-27/RECORD.md) |
| `34d734de8` | A2: FinnLib consolidated on `11b5c64b` (fork branch `kernels/consolidated-20260927`), memstream ported | same |
| `d42dc05c6`, `d7a64b0be` | B1: per-input exports (engine); ports check their streams; per-port attribution | same, [engine proposal](../kernel-phase-ab-2026-09-27/PROPOSAL-per-input-exports.md) |
| `bf9dfe0c9` | B2: clock domains as referenced Spaces; unpumped `ap_clk2x` dropped. *Revised after B3 (D11): domain nodes removed, clocks driven by role, target period added* | same |
| `75595eedb` | B3: control buses, tie-offs, sidebands, child padding; thresholding on streams | same |
| `538b53951`, `f815f4c20` | C0 M0: the user's answers and the collapse/split analysis; M1: `MVAU` becomes `MatMulKernel` | [spec](../matmul-kernel-2026-09-27/SPEC.md), [M0](../matmul-kernel-2026-09-27/M0.md), [record](../matmul-kernel-2026-09-27/RECORD.md) |
| `1182e8025` | C0 M2–M4: one kernel per dotp core (FinnLib `CORE`, pin `b9262df`), the per-channel mode, one composite for dense and per-channel | same |
| `f5dc725c2` | C0 M5: the dense realization of per-channel operations; derived NARROW_WEIGHTS | same |
| `ae68b558a` | C3: replay as a choice (`replay_buffer` or `input_gen`); marker bits | same |
| `cca9ae9e3` | C4–C6: memstream delivery (writable, pumped, several sets); adapter kernels (`vpc`, `inner_shuffle`) | same |

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
| D11 | *Added after B3.* **A Kernel owns the design choices of one generated module.** It may compose sub-kernels inside that module and exposes a complete pin interface for it: pin roles, clock relations and buses. It does not wire its own instance into a design. Instance wiring (the design's clocks and frequencies, clock-domain crossing, edges between separately generated modules) consumes that interface and belongs to the dataflow or artifact-integration layer. Consequences: B2's `ClockDomain`/`DerivedClock` nodes and `Stream.clock` are removed. Composites drive child clocks by pin role, and the only clock choices are per-kernel pumping. The target clock period is a kernel input: dotp derives its DSP58 segmentation from it, as FINN's RTL MVAU does |
| D12 | *Added after the B2 revision.* **`MatMulKernel`**: MVAU is renamed and dissolved with VVAU into one design space; thresholding is cut out; fused HLS designs are left behind. Where RTL combines what should be design-space choices, and where kernel distinctions should collapse, is decided first. Task spec: [`../matmul-kernel-2026-09-27/SPEC.md`](../matmul-kernel-2026-09-27/SPEC.md) |
| D10 | *Added after B1.* B1's port checks are kept, but they are a kernel-side stopgap: dotp checks what it reads against the forms it is given. A stream that knows the whole tensor it iterates, and so which foldings of it are valid, is the robust model. That opens design questions about the `Stream` object and how much of the original dataflow modeling corpus to adopt or revise. **Parked** as the stream and dataflow modeling revision (Phase D) |

## 3. Known problems

All of these are verified in the audit (§/probe references there).

| Problem | Why it matters |
|---|---|
| ~~dotp accepts streams it can't consume~~ | Fixed in B1: dotp checks lanes, lane order, column walk and frame rows |
| ~~Refusals are reported by every stream a kernel touches~~ | Fixed in B1: per-input `PORT` exports |
| ~~No non-stream interfaces~~ | Fixed in B3: control buses exported or tied off, sideband streams, child padding |
| ~~FinnLib pinned three ways~~ | Fixed in A2: one pin, `11b5c64b`, on the fork. Upstream has none of the carried commits yet |
| ~~Shared `deps/finnlib`~~ | Fixed in A1 |
| ~~Clock and reset routed by pin name~~ | Fixed in B2 and its revision: child clocks and resets are driven by declared role; an unpumped design has no `ap_clk2x` |
| **Some kernels are still standalone** | The stream idiom now covers the dotp cores, replay, cyclic and memstream delivery, FIFO, thresholding, the input generator and the adapters. Eltwise and int-to-float remain standalone |
| **FinnLib `inner_shuffle` emits undefined lanes** (found in C6) | SIMD 4 with a side of 4 or 8, input in bursts with idle cycles. Reproduced in FinnLib's own testbench by changing only its input timing. Blocks the transpose adapter until fixed upstream |
| **A stream holds one form** | Both ends adopt the composite's form, so a stream cannot place an adapter between two presentations. D10's S1 is the fix (C6's held part) |
| **Parameter images are in the build identity** | Cyclic `INIT_DATA` and memstream `INIT_FILE` (`GeneratedData`) key the module by its weights. The artifact layer's intent is late-bound slots (`DataSlot`/`DataBinding`), so one component serves every weight value |
| **HLS kernels can't be placed by `netlist`** | They yield HLS source requirements with no pin ABI. No HLS kernel is on the current path (D5); the HLS synthesis stage is future work (Phase D) |
| **An unused stream is refused**, and a boundary needs its port name declared up front | Optional streams need explicit `when=` guards |

## 4. Plan

Each step is independently landable and ends at a review gate. Phases A and B
are done (see the record). Phase C is done through C6 (2026-09-28), except the
in-stream adapter Decision, which waits for D10's S1.

### Phase A: foundations (done)

| Step | Content | Exit criteria |
|---|---|---|
| **A1. Per-worktree `deps/`** (D3) | Make `fetch-repos.sh` produce real per-checkout clones. Replace the `deps/finnlib` symlink in this checkout and in the spike worktree. Document the `FINNLIB_ROOT` override for local FinnLib development | Each checkout builds from its own pinned `deps/`; gates green |
| **A2. FinnLib consolidation, J0** (D4, D5) | One commit on the fork, pinned in `fetch-repos.sh`, which also ports `memstream_axi`, `memstream` and `axilite` from `finn-rtllib`, with their testbench if it's reusable. Update the kernels' source manifests to the new layout. Re-baseline the fingerprints once and record the reason (source identities change). Open upstream PRs for `replay_buffer`, the dotp fix and the memstream port if wanted | One pin everywhere; gates, XSim and the MVAU numeric sweep green; fingerprint change recorded |

### Phase B: correctness and infrastructure (done)

| Step | Content | Depends on |
|---|---|---|
| **B1. J2: ports that check their streams** | Every stream port verifies lanes, form, repetition and markers against its stream. Per-port attribution (D8): the small engine addition where a stream input names its presented port, and `Users` yields only that port | A2 |
| **B2. J3: clock and reset** | As built: a clock-domain family referenced by kernels. Revised (D11): child clocks and resets are driven by pin role, not by domain nodes or names; the unpumped `ap_clk2x` is dropped (D7); the target clock period sets dotp's segmentation | A2 |
| **B3. J4: non-stream interfaces** | AXI-Lite export and tie-off, sideband buses, and padding disposal between children | B2 |

### Phase C: robust matmul capabilities

| Step | Content | Depends on |
|---|---|---|
| **C0. `MatMulKernel`** (D12) | *Done: M0–M5 (the spec's M4 became M4 and M5).* The [task spec](../matmul-kernel-2026-09-27/SPEC.md), increments M0–M4: the collapse and split analysis at a human gate, then the rename, dotp's per-channel mode, and one composite for the dense and per-channel contractions | B1–B3 |
| ~~C1. J5: VVAU reuse~~ **Replaced** by the [`MatMulKernel` spec](../matmul-kernel-2026-09-27/SPEC.md): MVAU and VVAU dissolve into one design space; thresholding is cut out | — | — |
| *(was C1)* VVAU reuse, plus a reusable delivery slot | Same families, `ACTIVATION_BROADCASTING=0`, a marker generator instead of replay. Needs E-048 (SWG→VVAU lane order) re-derived | B1 |
| ~~C2. J6: fused thresholding~~ **Dropped** by the `MatMulKernel` spec: thresholding stays its own kernel; placing it with a matmul in one module is an operation-to-module mapping decision of the dataflow layer (D10, D11) | — | — |
| *(was C2)* fused thresholding | Thresholding already sits on streams and a control bus (B3). Remaining: compose MVAU → Thresholding as an optional node with `when=`-guarded streams, and the numeric sweep over the fused output | B1, B3 |
| **C3. J7: replay as a choice** | *Done (`ae68b558a`).* `replay_buffer` or `input_gen`, as a `Decision` over nodes (D7) | B1, B2 |
| **C4. J8: an RTL memstream kernel, plus runtime-writable weights** (D5) | *Done (`cca9ae9e3`); both deliveries stay candidates (user, 2026-09-27).* A fixed-interface kernel over FinnLib `memstream_axi`: `DEPTH`, `WIDTH`, `SETS`, `RAM_STYLE`, and pumped memory as a Decision of the memstream kernel (its 2x clock pin driven by role). Initial contents go through `INIT_FILE`: check whether the artifact layer's data-file contribution supports it, and add one if not. Its AXI-Lite bus is exported at the module boundary (B3). `MatMulKernel` gains a writable delivery candidate, and the numeric XSI harness gains an AXI-Lite write driver. Decide whether it replaces the kernel-local `cyclic_stream.sv` for read-only delivery, which needs equivalence evidence | A2, B2, B3, C0 |
| **C5. J9: multi-set delivery** (R7, MLO) | *Done (`cca9ae9e3`).* `SETS > 1`, with the set-index stream as an ordinary stream reference input of the memstream kernel. MLO drives the index from outside the op (V10) | C4 |
| **C5.5. D10 design round** | *Done: written for review.* A first-principles design of the dataflow `Stream` for kernel Spaces, by a fresh-context review of the prior corpus (scratchpad `dataflow/`, `space/`, `open/`) taken as wary inputs. Output: [`../stream-model-2026-09-27/DESIGN.md`](../stream-model-2026-09-27/DESIGN.md), for review. Design only; implementing it is Phase D | B1–B3 |
| **C6. J10: stream adapters** | *Done in part (`cca9ae9e3`): the `vpc` and `inner_shuffle` adapter kernels, as nodes between two streams. The in-stream Decision waits for D10's S1; see the record.* Width conversion (FinnLib `vpc`), lane regroup and reorder (`inner_shuffle`) as a `Decision` over adapter nodes inside a stream, chosen by `classify()`. Within one generated module only; edges between modules are D10's. Built consistently with the C5.5 design | B1, C3, C5.5 |

### Phase D: planned and deferred work

| Item | Plan |
|---|---|
| **HLS synthesis stage** (future work) | Makes HLS kernels placeable, and is the path to HLS backends as choices, e.g. HLS MVAU and HLS thresholding (the audit's "HLS backend as a whole" gap). It needs five parts: (1) a background Vitis HLS runner with store caching keyed by sources, part, clock and tool version; (2) a generated-source contribution type, with tool version in build identity; (3) the pin interface predicted from the HLS interface directives for `netlist`, and verified after synthesis with `artifacts/rtl.check_abi` (option a, which reverses today's "no predicted HLS ABI" stance; the alternative, option b, is a two-stage flow binding the synthesized interface as a fact); (4) control-mode handling (`ap_ctrl_none` for free-running blocks); (5) an opt-in slow test tier. Plan it when HLS backends are wanted |
| **dotp core split** (D6, R4) | Near term: examine FinnLib's `dotp_axi` core selection and the FINN wrapper-split precedent. Then decide how the cores become separate kernels or a `Decision` |
| **Query and search tools** | The next Space-engine pass (design record §3). The audit's "queries wanted" list is its input |
| **Reusing cached results across snapshots** | A local edit re-runs the whole graph. It matters once search runs over graphs of many kernels |
| **Other standalone kernels** (eltwise, input generator) | Move them to the stream idiom when a composite needs them |
| **Stream and dataflow modeling revision** (D10, parked) | Streams that understand the full tensor they iterate and the valid folding configurations over it, instead of kernels checking the forms they are handed. Decide the shape of `Stream`, and how much of the dataflow modeling corpus (`finn.dataflow`, the roster's S1 contract, the parked dataflow model) to adopt or iterate on. It subsumes B1's dotp form checks and informs adapters (C6). It also owns instance wiring (D11): edges between separately generated modules, and the design's clocks. Whether a stitched design is one more composite wired by the same `netlist` is a question for this pass. Plan it together with the `finn.dataflow` model pass |
| **D10 implementation** (after review) | The design round recommends increments S1 (presentation at the ends; then C6's in-stream adapter Decision), S2 (the loop nest that derives port forms; retires the labelled-relation stopgap) and onward; §11 lists the user's decisions |
| **FinnLib `inner_shuffle` fix** | Characterize and fix the defect found in C6 (repro in the MatMul record's evidence), then pin it |
| **Late-bound parameter images** | Move cyclic `INIT_DATA` and memstream `INIT_FILE` to `DataSlot` + `DataBinding`, with the artifact-integration work |
| **MatMul device coverage** (flagged important) | A LUT dot-product core; native per-channel on DSP48E1/E2; further device and behaviour permutations. Each lands as another core kernel or candidate in `MatMulKernel`'s space |
| **Choosing among valid candidates** | Compatibility filters candidates; where several remain, choosing is optimization (DSE passes), future work |
| **Revisit: `NARROW_WEIGHTS`** | Provisionally derived from known weights (cyclic delivery), 0 otherwise. The user wants to revisit it |
| **Revisit: cyclic vs memstream** | Both stay delivery candidates. Whether to eliminate or consolidate one is a future optimization question |
| **`finn.dataflow` model pass** | After robust MVAU; to be planned with the stream and dataflow modeling revision |
| **Diagrams** in `scratchpad/space/diagrams/` | Regenerate them for the new API |

### Coordination

- **Artifact-integration SPEC** (`docs/kernel-artifact-integration-2026-09-25/`,
  owned by another session): §1 and §6 need re-baselining on the landed API.
  The mapping is in [LANDING.md §5](../space-declarative-2026-09-26/LANDING.md).
- **External-resources merge (`integration/er-kpe`):** re-merging it conflicts
  in 11 files. Its removal of `typing_extensions` conflicts with the engine,
  which already depended on it before this landing.

## 5. Open questions for later

1. ~~The memstream boundary~~ *Answered 2026-09-27:* both stay delivery
   candidates; consolidation is a future optimization question (Phase D).
2. **The upstream path for FinnLib:** which fixes go upstream, and on what
   schedule? `replay_buffer` and two testbenches carry BSD-3-Clause headers in
   an MIT library; relicensing is the author's call before a PR.
3. ~~The dotp core split (D6)~~ *Answered 2026-09-27:* each meaningfully
   different core is its own kernel, selected by a Decision (MatMul spec, M0).
4. **The Space model's open questions** (design record §10): a view used in its
   own class body (`cast`), the path form of `inspect`, and redundant
   obligations.
