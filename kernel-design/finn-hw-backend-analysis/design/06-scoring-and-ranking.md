# Comparative Ranking — The Kernel Backend Panel

## Verdict at a glance

| Rank | Design | Score | Adversary | MVAU | Verdict |
|------|--------|-------|-----------|------|---------|
| **1** | **composition** | 31 | no collapse | survives | **WINNER — synthesis backbone** |
| 2 | capability-trait | 29 | no collapse | survives | Best selection engine; graft wholesale |
| 3 | multi-axis | 29 | no collapse | survives | Best orthogonality + port typing; graft |
| 4 | ir-lowering | 31 | **COLLAPSES** | survives | Best single idea (ParamSource node), but breaks at the anchor test |
| 5 | evolutionary | 25 | no collapse | survives | Honest floor; owns the migration plan |

The decisive move in this ranking is that **ir-lowering's raw score of 31 is tied for first, but its adversary collapses precisely at the forcing function** (static-IP GEMM composition) and re-grows a god-emitter — so it is penalized hard, from a co-leader down to fourth. Everything else follows the scores modulated by how each design's weakest seam behaves under pressure.

---

## 1. The ranking, justified

### #1 — composition (31, no collapse)

Tied for the top score and the only top-scorer whose adversary does **not** collapse. It is the canonical answer to the master property: one `final`, never-subclassed `Kernel` that *has* six Protocol-typed parts, and a substrate of generic drivers that never `isinstance` or `op_type`-branch. It resolves the two disqualifiers *structurally*, not nominally:

- **Diamond kill (#4):** `execute_node` becomes two differently-named slots on two different objects — `kernel.reference.evaluate` (pure golden, "NEVER named execute_node") vs `kernel.compute.runner(mode)`. No shared method name can ever collide under MRO. This also structurally kills the latent VVAU indent bug (`vectorvectoractivation_rtl.py:89`) and collapses the copy-pasted per-op rtlsim block (`matrixvectoractivation_rtl.py:100` ≈ `_hls.py:569`) into one shared `RtlsimRunner`.
- **Base-class leak kill (#2):** memstream Verilog moves out of `HWCustomOp.generate_hdl_memstream` and its op-type allowlist (`hwcustomop.py:307/355/407`) into `DecoupledStream.emit_artifacts`; `calc_wmem`/`ram_style` move onto the `MemoryStrategy` that actually owns the geometry.
- The static-IP GEMM plugs in as one new `ComputeStrategy` with zero substrate edits — the forcing-function test passes by construction.

Its adversary confirms it "does NOT collapse back into op_type/isinstance special-casing." The one real hole is displaced, not eliminated (see tensions §6): cross-component constraints route through an under-specified `negotiate(spec)` handshake that the design itself admits could "quietly become a new god-object." That caps its ceiling but does not sink it — and it is exactly the hole the #2/#3 designs fill.

### #2 — capability-trait (29, no collapse)

The most complete answer to criterion #3, the selection god-switch. Its load-bearing move — **preconditions as inspectable constraint DATA** (`AllOf/AnyOf/Not` over typed primitives), co-located on each Realization and resolved by *one* op-agnostic solver with `explain(select)` — genuinely deletes the `specialize_layers.py:40-211` ladder and makes the `:60/:69`-vs-`:275` drift bug **structurally impossible** (one source of truth). Substrate purity (5/5) falls out by construction, proven by the zero-edit static-IP GEMM (`class MVU_INT8_StaticIP`, one new file, `priority=30` beats RTL's 20). Its adversary clears all four disqualifiers "on the merits, not by renaming."

Two honest soft spots keep it at 29, not higher: (a) memory is **not** a truly independent axis — `ParamStorage(kind,discipline)` rides on the delivery Realization (`Memstream` provides both decoupled-delivery *and* `bram|uram`), so axis-3 orthogonality is partly nominal; (b) the "preconditions are DATA" claim leaks — ~2 of 6 predicates need a `Predicate(named_fn)` escape hatch, a small re-opening of the god-switch (still Realization-local, so #3 still passes). Its Composer for the 236-line MVAU IPI is "asserted, never demonstrated."

### #3 — multi-axis (29, no collapse)

The purest realization of the explicit-orthogonal-axes prior and the **only design that makes Memory a genuine peer axis** (5/5 orthogonality — its crown). Its cleanest, most portable win is the **typed Port sum where only `TensorStream` carries `folded_shape`**: iodma's AXI-MM side (`iodma_hls.py:116/131`) never raises because it is an `AxiMM` variant with *no tensor shape by type* — the category error is eliminated by typing, not by branch. Its conformance sketches (iodma-by-typing, MVAU IPI as `Stitcher.compose`) are the most concrete of the panel.

It sits just behind capability-trait for one reason: its cross-component constraints live in a **central `Legal ⊆ Axis_i × Axis_j` relation** — a shared surface "that grows with each new kind," which the judge flags as "a softer, data-driven echo of the specialize_layers god-switch it replaces." And it self-admits the relation is really *n-ary* (MLO couples `LoopFetched ∧ OffChipDMA ∧ AxiMM ∧ Clock` at once), forcing a resolver-level escape-hatch predicate. Capability-trait's *distributed* per-realization preconditions are the more principled answer to the identical problem, so multi-axis lands at #3. Its "coordinate = self-contained point in a product space" metaphor leaks at exactly the n-ary/cross-kernel cases (MLO, pumpedMemory, container) the brief cares most about.

### #4 — ir-lowering (31, but adversary COLLAPSES)

This is the ranking's sharpest call. On raw score it ties composition for first (5s on substitutability, hermeticity, axis-orthogonality, substrate-purity, expressive-completeness). It owns the single best MVAU idea in the panel (below). **But its adversary returns `collapses=true`, and the collapse is not incidental — it lands on the two hardest cases the brief privileges:**

1. **The forcing function itself.** ir-lowering has *no declared relational/cross-node legality mechanism*. The static-IP GEMM's weight-ownership constraint (the hard IP owns its weight port, so it constrains which upstream `ParamSource` is legal) is a *relation between two nodes* and is therefore **unexpressible as a local `match()`**. It forces "ad-hoc graph-construction/rewrite logic" — the exact centralized special-casing the brief exists to kill, relocated into a graph-builder pass. Both #2 and #3 express this constraint cleanly (cap-trait: provided/required trait matching; multi-axis: a `Legal` pair). ir-lowering cannot, at the anchor test.
2. **The stitch god-emitter.** All bespoke IPI/TCL (clk2x pins, the `finn_loop.py:1071` aperture hotfix, `os.listdir` last-match wiring) is concentrated into one `RenderTcl(BlockDesign)` emitter. The design *asserts* these become "typed BlockDesign features" but never demonstrates the type is expressive enough. Under pressure this "re-grows special-cases unless BlockDesign is exhaustively typed" — the god-method moved from a base method to a renderer, not dissolved.

Per the ranking rule (a high score with adversary collapse must be penalized hard), a 31-with-collapse-at-the-anchor drops below two clean 29s that both handle the anchor. It stays ahead of evolutionary because its ideas are more valuable and it does cleanly beat the as-is on substrate purity (immutable-IR-value + pure-lowering genuinely kills the `code_gen_dict` side-channel and the MRO diamond via `SemRef`/Artifact-type dispatch).

### #5 — evolutionary (25, no collapse)

The honest floor, and it does its job: it concretely passes all four disqualifiers *without* a rewrite (op_type genuinely evicted from `hwcustomop.py` into a `WeightDelivery` collaborator; `specialize_layers` replaced by co-located `feasible()` classmethods killing the `:60`-vs-`:275` drift; `execute_node` split into two named slots; static-IP added as a third registry with zero substrate edits). Its adversary confirms it does not secretly reintroduce the defects. It is capped by *deliberate* under-investment, which it declares honestly: the mixin diamond **survives structurally** (any new name collision reopens it — patched by a registration-time assertion, not prevented); HLS/RTL are never made peers (638 vs 146 LOC, `code_gen_dict` vs verilog-param); compute and weight-delivery still share a live host protocol so they are not independently substitutable; the 236-line IPI is decomposed but `instantiate_ip` still lives on the leaf and is called from the agnostic tier. It sets an **unbeatable migration bar (5/5)** and a high substrate-purity bar — which is exactly its role: the control the ambitious four must clearly beat. They do, on substitutability, uniformity, and structural elimination of the diamond.

---

## 2. The winner: composition

**composition is the synthesis backbone.** Reasons:

1. **It is the only top-scorer (31) that survives its adversary intact.** ir-lowering ties on score but collapses at the anchor; composition does not.
2. **It attacks the two disqualifiers at the structural level, not the nominal level.** One `final` Kernel + Protocol dispatch makes criterion #2 hold *by construction* (there is no base to leak *from* and no subclass to branch *on*). The reference/runner split makes the #4 diamond *unnameable*, not merely refactored. This is the strongest possible reading of the modular-dev master property: the taxonomy that was ~80 classes becomes one class + a library of interchangeable parts.
3. **It is the most graftable host.** Its one weakness — where cross-axis constraints live — is a *hole*, not a wrong answer, so the panel's best constraint mechanisms (from #2 and #3) drop straight in. The judge verdict is explicit: "Grafting the multi-axis Legal relation onto this composition core would produce the best of the panel." I would go further and graft capability-trait's *distributed* precondition data instead (see tensions §2), which fits composition's spirit better than a central table.
4. **Its conformance sketches are honest and complete** — the reference/runner split, the `WeightPort.stream_width` delegation that makes embedded-params yield width 0 with no `try/except AttributeError`, and the `Parameterization` typed token-binding that kills the 26 untyped `$KEY$` replacements (criterion #8) — all without inventing IR/region machinery it then can't fully type (ir-lowering's trap).

---

## 3. Ideas to graft into the winner

### From capability-trait — the selection engine (highest-value graft)
- **Preconditions as inspectable DATA + one op-agnostic solver + `explain(select)`.** Replace composition's `can_realize(spec, tgt) -> Feasibility` *method* with capability-trait's constraint-DATA representation (`AllOf/AnyOf/Not/Implies` over typed primitives like `DtypeBits`, `DeviceFamily`, `DspBlockIn`). This buys debuggability the god-switch never had (per-constraint reason strings, ranked candidate lists) and makes the `:60/:275` drift *structurally* impossible rather than merely relocated. This is the definitive answer to criterion #3 and is graftable into any core.
- **`priority`-ranked auto-selection** for the static-IP common case (`MVU_INT8_StaticIP priority=30` beats `MVU_RTL=20`, transparent fallback on precondition failure).

### From multi-axis — orthogonality + typed ports
- **The typed Port sum where only `TensorStream` carries `folded_shape`.** This is the cleanest fix for criterion #7 in the whole panel: iodma/tlastmarker's raising getters (`iodma_hls.py:116/131`) are eliminated *by type* — generic passes iterate `TensorStream` variants only, so non-tensor ports never raise. Graft this directly into composition's `PortSet`/`Port` Protocol (composition already has `AxiMmPort`/`SidebandPort`, but multi-axis's "shape-by-type" discipline is the load-bearing rule).
- **Memory as a genuinely independent axis** (`Memory ∈ {LUTROM, BRAM(ram_style), URAM, OffChipDMA}`, distinct from delivery-motion). This fixes capability-trait's and composition's residual memory-into-delivery coupling and is why multi-axis is the only 5/5 on orthogonality. Adopt the `Embedded×LUTROM` / `DecoupledStream×BRAM` / `LoopFetched×OffChipDMA` factoring so `bram_estimation`/`uram_estimation` (`matrixvectoractivation.py:387/365`) live on the Memory object.
- **The pairwise `Legal` relation** as the fallback for *relational* constraints that per-realization preconditions cannot express alone (e.g. `StaticIP ⇒ WeightDelivery ∈ {Embedded, External}`) — but see tension §2 for the distributed-vs-central decision.

### From ir-lowering — two structural ideas worth stealing without the IR framing
- **Weight/param delivery modeled as its own upstream node** (`ParamSource: Embedded/Decoupled/External/Dynamic/Mlo` connected by a typed `param_in` edge). This is the cleanest MVAU fix among the five: `mem_mode × dynamic × mlo` becomes "which upstream node is in the graph," structurally orthogonal to compute-kind, and the substrate never sees `calc_wmem`/`ram_style`/`mlo_max_iter`. Even if composition keeps delivery as a *composed part* rather than a full graph node, adopt the framing that delivery is a peer producer with its own ports, not a service the compute reaches into.
- **Regions as the native container model** for finn_loop/SDP (a `KernelOp` carrying a nested graph; `PortSignature` derived by projecting the region boundary). This is the most principled answer to conformance-op #3 and beats composition's `SubgraphCompute` and multi-axis's admitted "Container-on-an-axis-named-Compute is a stretch." Graft the region *idea* for containers only, not the whole IR dialect.
- The typed `RtlModule` param schema (binds `ACCU_WIDTH` from `accDataType`, fixing the `matrixvectoractivation_rtl.py:349` bug where output dtype was silently used) reinforces composition's `Parameterization`/criterion #8 story.

### From evolutionary — the migration plan
- **The 8-PR decomposition** as the derived migration path (the brief defers migration until after a winner is chosen — this is that path). Each of the 8 refactors is independently landable and green against the 27-family regression. The clean-slate winner needs exactly this to avoid a big-bang rewrite; steal it wholesale.
- The `feasible()`-classmethod + tiny `preference_order` data table is the minimal shadow of capability-trait's solver — useful as the *first* PR before the full solver lands.

---

## 4. Disqualification

**No design should be formally disqualified.** All five express MVAU (the "cannot model MVAU is dead" gate), and all five concretely pass disqualifier criteria #1–#4 — including evolutionary, which does it surgically rather than structurally.

**However, ir-lowering is on the bubble and must be flagged.** Its static-IP GEMM story — *the* anchor and "single most important design test" (§3) — does **not** hold up as a clean, local, declared-capability mechanism. Because it has no relational legality, the anchor's weight-ownership constraint forces ad-hoc graph-rewrite logic, and its single `RenderTcl` emitter risks re-becoming the exact god-switch being eliminated. That is the closest thing to a disqualifier in the panel. It is not formally disqualified because the node still *type-checks* against the port/folding contract and MVAU survives — but the criterion-#2 "zero substrate edits" and criterion-#3 "contract-driven selection" claims **leak at the anchor**, which is why it drops from a co-leader to fourth.

The general caution the brief raised — "a design that merely RENAMES the current split must be caught" — applies to none of the five: all genuinely separate weight-delivery and reference/execution from compute-kind rather than renaming HLS/RTL. Evolutionary is the closest to a rename (it keeps the mixin shape) but it *honestly declares* this and still evicts the actual leaks, so it is a legitimate floor, not a disguised no-op.

---

## 5. Unresolved tensions the D3 synthesis must decide

These are the questions the five designs *disagree* on or *all dodge*. D3 cannot ship without ruling on each.

**T1 — Feasibility as CODE vs DATA (self-advertise vs separate-pass).** The brief (§9.2) permits both. Selection strategies split: evolutionary self-advertises (`feasible()` classmethod); composition and multi-axis are hybrid (`can_realize`/`feasible` method queried by a generic pass); capability-trait and ir-lowering are separate-pass over declared data. The deeper axis is whether feasibility is an opaque **method** (simple, but no `explain()`) or an inspectable **precondition AST** (debuggable, but needs a governed vocabulary + admits `Predicate(named_fn)` escape hatches). **Recommendation:** adopt capability-trait's DATA form — the debuggability and the structural impossibility of `:60/:275` drift are worth the vocabulary-governance cost, and the winner's hybrid `can_realize` is trivially reskinned as "evaluate this precondition."

**T2 — Where cross-axis / relational constraints live (composition's actual hole).** Four different answers: composition's `negotiate` handshake (self-admittedly under-specified, "could become a new god-object"); multi-axis's **central** `Legal` table (a shared surface that grows); capability-trait's **distributed** per-realization provided/required trait matching; ir-lowering's *nothing* (forces graph-rewrite — its collapse). D3 must specify the constraint algebra composition left blank. **Recommendation:** distributed preconditions (cap-trait) for local feasibility + a *small* declared pairwise relation (multi-axis `Legal`) only for genuinely cross-component constraints (`StaticIP ⇒ Embedded|External`), with an explicit, bounded n-ary escape hatch for the 2–3 known cases (MLO, pumpedMemory). Do not let the relation become the god-table.

**T3 — Weight delivery: composed part vs graph node.** composition/capability-trait/multi-axis model delivery as a composed component the compute negotiates with; ir-lowering models it as a *separate upstream graph node* (`ParamSource`) wired by a typed edge — the most structurally orthogonal option, but it drags in a folding-propagation pass (keep `param_in` fold consistent with compute `PE×SIMD`) and region/IR machinery. D3 must decide how far toward "delivery is a peer producer, not a service" to push. **Recommendation:** keep delivery a composed part (lighter, no IR framework), but adopt the ParamSource *port model* so the compute never reaches into the delivery object's privates.

**T4 — The stitch/IPI coordination layer (NO design solved this — all displaced it).** Every design's adversary flags the same unsolved problem: the 236-line `code_generation_ipi` is **not** pure fragment concatenation. Cross-port wiring (weight-streamer → `in1_V`), cross-clock (pumpedMemory → memstreamer clk2x), and aperture hotfixes (`finn_loop.py:1071`) force a typed coordination layer. composition's `Stitcher`/`negotiate`, capability-trait's `Composer` (asserted, never demonstrated), multi-axis's `Stitcher.compose`, and ir-lowering's `RenderTcl` (its *actual* collapse point) are all the same under-designed component wearing four hats. **This is the deepest tension and the one most likely to sink the synthesis if unaddressed.** D3 must exhaustively type the `BlockDesign`/`StitchFragment` model — cross-port, cross-clock, and hotfix wiring as first-class typed features — or the god-method simply relocates. Whichever design's stitch type is most complete should be the graft source; on the evidence, none is yet, so this needs fresh design work, not a graft.

**T5 — Purity ceiling vs migration cost.** Four designs sit at migration-tractability 2; evolutionary at 5. The clean-slate stance (§0) makes migration secondary and *disqualifies nothing* on it — but evolutionary proves the disqualifiers are fixable surgically and green against the regression at every step. D3 must decide the interregnum: adopt the composition core (clean-slate ceiling, zero-substrate-edit purity) *but* execute it via evolutionary's 8-PR path to avoid a big-bang rewrite with no rollback. The tension is real (evolutionary warns the codebase has "two ways to do everything" mid-migration), but the combination — clean-slate target, incremental delivery — dominates either pure strategy alone.

**T6 — Memory: peer axis or sub-choice of delivery.** multi-axis (5/5 orthogonality) makes Memory a fully independent axis; capability-trait explicitly fails this (storage rides on the delivery Realization); composition has a `MemoryStrategy` Protocol but its coupling to `WeightDelivery` is not fully specified. D3 must rule whether `LUTROM/BRAM/URAM/OffChipDMA` is chosen independently of `Embedded/Decoupled/External/LoopFetched`, or whether the `(LoopFetched, OffChipDMA)`-style pairing means the clean "motion vs substrate" split is thinnest exactly at MLO. **Recommendation:** adopt multi-axis's independent-Memory-axis with a small `Legal` pairing for the genuinely-coupled MLO case — this is the single highest-orthogonality decision available and directly raises the winner's weakest sub-score (axis orthogonality 4 → 5).