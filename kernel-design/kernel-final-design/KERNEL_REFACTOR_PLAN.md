# Kernel Refactor Plan — Replacing the FINN HW Backend

*The plan: goals, integration strategy, and staging for landing the KernelOp design-space
model in the FINN flow. The motivation — why this is needed, what the prior systems prove —
is in `MOTIVATION.md`.*

---

## 1. Goals — the deliverable, and the bound on scope

Four goals define what we build: a KernelOp abstraction (G1–G3) integrated into the FINN
build flow (G4). The adapter path (§2) lets new kernels and classic `HWCustomOp` leaves
coexist in one graph, so **retiring `HWCustomOp` is an optional end-state, not the
objective**: a cleanup we can take once the last family migrates, or leave indefinitely. The
first three goals are properties of the new abstraction; the fourth makes it a real
in-place replacement.

**G1 — The design space is data (declarative, resolvable, explorable, enforced).** A kernel
declares its space as *free choices* (Axes, guarded existence + point-dependent domains),
*computed quantities* (Derived), and *feasibility gates* (Predicates). A resolver turns
(space + device context + choices) into a legal build Point — or an *explained* illegality.
Two motivation findings collapse into this one goal:
- **The specialization axis is named correctly** (MOTIVATION §1.1). The root free choice is
  `implementation` — *which microarchitecture* realizes the kernel — with `language`
  (HLS/RTL) a *derived* property of the chosen realization, not the axis itself. MVU's
  softvec vs DSP58-packed cores become two pool members, not a hardcoded `generate` fork;
  the prototype's *dropped* device feasibility becomes the per-member Predicate
  (`mvau_dsp_packed`: DSP58 ∧ w≤8 ∧ a≤9). This dissolves the `specialize_layers` god-switch.
- **The space is the enforced, documented contract** (MOTIVATION §1.5). Silent, unchecked
  conventions (decorative datatypes, the overloaded `ind` port parameter, the positional
  interface-tuple shape, valid `mem_mode` combinations) become declared Axes/Derived/
  Predicates. `resolve` *rejects* an illegal configuration with a reason, so a standard can
  no longer be violated without error — the machine checks what was previously learned by
  tracing call sites.

**G2 — Composition is first-class and declarative.** A kernel is a graph of role-tagged
blocks: compute co-existing with parameter-delivery sub-kernels (Level 1) and, above that,
an outer container over a sub-graph (MLO/FINNLoop, Level 2). Wiring is *derived from
declared ports by role*, not hand-written Tcl — the direct replacement for the 236-line
`code_generation_ipi`.

**G3 — The substrate is hermetic and knowledge-free.** Emit reads its resolved Point + a
frozen Context as *data* and never touches the graph, ambient env, or its siblings. The
substrate names no op. Reference (golden) and emit (backend) are separate contract slots —
no diamond. Codegen is typed artifacts with validated slots — no `code_gen_dict`, no silent
`$KEY$` no-op. **Beyond the contract, the source model** (MOTIVATION §1.4): sources become
first-class artifacts with provenance and composition rules — which core → which shared
includes, per resolved Implementation — replacing the `shutil.copy`-per-node, 3×-duplicated,
`os.listdir`-scan source handling. The typed-emit foundation begins this; a full
source/provenance/composition model is **long-tail work** the `mvu-wrapper-split`
exploration surfaces and Stage 5 finishes.

**G4 — It replaces the subsystem in place (the integration goal).** The new system must
satisfy the *real* consumer contract, not just the ABC, so a partially-migrated graph builds
end-to-end. The shape (§2): **share the frontend, replace the selection seam, adapter the
backend tail.**

**Non-negotiable cross-cutting requirement: correctness is proven against FINN, not
asserted.** Every emit is validated byte-equivalent to FINN's own codegen; composed designs
are validated in rtlsim against a golden reference.

**Explicitly out of scope** (honest residuals every design shares): per-op HLS
`$DOCOMPUTE$` compute-text and vendor IP-packaging Tcl stay per-op. We derive *numerics* and
*stitching* declaratively; we do not generate the compute kernel's inner C++.

---

## 2. Integration strategy — depth-keyed, hinged on `specialize_layers`

**We adapter over the existing flow rather than build out the prototype's parallel one.**
The prototype already proved where a parallel flow ends: clean, but *out of the FINN
ecosystem* — the 20-step flow never integrated, so it was never a real replacement
(MOTIVATION §2.1). The adapter keeps mixed graphs building end-to-end at every stage — no
flag day, each stage ships on its own — at no cost to purity: Stage 6 still deletes
`specialize_layers` and retires `HWCustomOp`. A transitional bridge, not a permanent one.

The op-contract surface is **tiered and grows with pipeline depth**, and the tiers before
and after `specialize_layers` (build step 6) are structurally different consumers — so the
integration shape is not one global choice but keyed to pipeline depth:

| Pipeline region | Steps | Strategy | Why |
|---|---|---|---|
| **Frontend** | 1–5 | **Integrated / shared** | Operates on the *abstract* HW node (base ABC + config nodeattrs + dtype-from-Context). `convert_to_hw` reads *source*-op attrs and never calls a kernel method. Old ops and kernels are indistinguishable here — no fork. |
| **Selection seam** | 6 | **Replaced** | `specialize_layers` IS the god-switch. `resolve` supersedes it: feasibility as per-bundle predicates, selection as resolution. Deleted, not adaptered. |
| **Backend tail** | 7–20 | **Adaptered** | A thin adapter projects `resolve`/`emit`/`stitch` onto the legacy method names the tail calls. Mixed graph works because classic leaves and new kernels answer the *same* calls. |

**`specialize_layers` is the point of complete divergence** — unsalvageable for three
reasons: it *is* the `if optype==` god-switch; it is where the contract surface doubles
(abstract HW node → concrete `_hls/_rtl` leaf); it does not delegate feasibility to the op.
Our resolve engine already *is* its replacement. (Separate the selection *logic* — deleted —
from its mechanical plumbing: node-naming, SDP templates, FINNLoop-body recursion — which is
harness cleanup.)

**Data still lives on the node, but the kernel structure is the authority.** Forced by
`NodeLocalTransformation` (it pickles each NodeProto to `mp.Pool` workers): per-node state
must be serializable nodeattrs or reconstructable from them. So the kernel data structure is
a **typed cached view** over that state, reconstructed as a pure function of (nodeattrs +
graph + Context) — the frozen-Point-from-Context model. Three kinds of nodeattr, three
dispositions: genuine *config* (Point-owned), *side-channel* (served from a live derivation,
write is a harness no-op), *ordering/handoff* (harness-owned; `emit()` returns Artifacts, not
stashed paths).

**The resolution-staging decision: incremental per tier** (see §5.1). `resolve` fires
incrementally — commit `implementation` at step 6, folding at 7–9, bit-width at 9 — each a
further axis assignment producing a more-complete partial Point. This matches the flow's
existing staging and the guarded-axis engine supports it natively.

---

## 3. Foundation — proven on the hardest vertical slice [done]

MVAU is the deliberate first target: the op whose unmodeled third axis *produces* the
236-line IPI, so a model that holds for MVAU holds. Four capabilities, each validated
against FINN:

- **Resolve engine + selection (G1)** — four-primitive engine (Context / Axis / Derived /
  Predicate → Point); MVAU as an implementation-bundle pool; the god-switch dissolved into
  per-bundle predicates.
- **Hermetic emit (G3)** — typed artifacts replacing `code_gen_dict` + `$KEY$`;
  reference/emit split. **Byte-equivalent to FINN** on MVAU compute.
- **Composition — parameter delivery (G2)** — weight/threshold delivery as three orthogonal
  coordinates; memstream emit **byte-equivalent to FINN** in Vivado.
- **Composition — the stitch (G2)** — role-tagged ports + op-agnostic resolver replacing
  `code_generation_ipi`. Validated by a **functional rtlsim of a composed MVAU computing a
  correct matmul end-to-end**.

This proves the model shape — but it is **not** a replacement yet: one op, standalone,
outside the build flow. The staging below plugs it into the flow and widens it.

---

## 4. Staging — build the ladder once, climb it many times

Two orthogonal axes of progress: **depth** (how far down the pipeline a kernel rides:
estimate → codegen → ipgen → stitch → rtlsim → bitfile) and **breadth** (how many op
families are on the new substrate). The integration machinery — the `resolve`-seam + the
adapter — is **built along depth and reused along breadth.** So: depth-first on MVAU (build
the whole ladder on the op we understand best), then breadth across the op zoo, each new op
climbing a ladder that already exists.

**The reframe that governs sequencing:** Stage 2's deliverable is the reusable harness, not
"MVAU estimates" — MVAU is just the first rider on machinery every later op reuses.

### Stage 0 — Polish the foundation into a proper FINN submodule

Before integration, harden what the foundation built. The vertical slice (§3) was grown as
a siloed proof — its own fixtures, Docker validation runners, and scratch harnesses,
deliberately *outside* the FINN pipeline to prove the model without entangling it. Stage 0
lifts it from proof-of-concept to a clean submodule the effort builds on.
**Build / clean up:**
- **Structure** — settle the package layout (engine `space/` vs op `fixtures/` vs adapter
  vs harness), API surface, and naming so later contributors read one coherent module, not
  a spike. Fold the superseded `kernels/` reference spike out of the way.
- **Test consolidation** — the one-off Docker differential + rtlsim scripts (`diff_*`,
  `elaborate_*`, `rtlsim_*`) become a coherent, documented validation suite with a stable
  entry point; pin the qonnx import to the finn-pinned `deps/qonnx` (the sibling-checkout
  drift found this session).
- **Docs** — a module README + the design-doc set cross-linked and current, so the module is
  legible without archaeology.
- **Toolchain hygiene** — capture the Vivado-2025.2 accommodations (§6) as documented,
  contained shims rather than scattered workarounds.
**Why here:** every later stage adds contributors and surface to this module; the cleanup
now is far cheaper than migrating 29 op families on top of a spike-shaped foundation. Also
the natural point to commit the foundation as a reviewed, landed submodule rather than a
working branch.
**Not in scope:** no new capability — purely consolidation of what §3 proved.

### Stage 1 — The KernelOp model layer (op-native surface) [done]

A model-layer increment between Stage 0 (consolidation, no new capability) and Stage 2
(FINN-flow integration): the **op-native** WHAT-surface, built and tested purely in the venv
against a hand-built Context — no `ModelWrapper`, no build flow, no adapter. Stress-tested on
MVU + LayerNorm + Elementwise from both FINN and brainsmith source. **Built:**
- the stream-tiling **expression evaluator** (`space/tiling.py`) — acid-tested on MVU's
  weight port `WSIMD=PE·SIMD/TH` and the elementwise broadcast replicate;
- the **`KernelOp` façade** (`space/kernel_op.py`) projecting the op surface from a resolved
  Point: the port-indexed normal/folded shapes + stream widths, and a rough monotone
  `get_exp_cycles` (the reduction-aware op `cost_model` where BLOCK demands it);
- **impl-owned tiling** (`Implementation.tiling`) + the BLOCK/STREAM ownership split + the
  unified GIVEN/op-default/backend-override principle across tiling, cost, and datatype;
- **MVAU routed through the façade** (`mvau_kernel_op`; `mvau_schema` now delegates to it),
  with the KernelOp getters cross-validated to agree exactly with the emit-side op-derived
  stream widths. The stale pre-KernelOp **VVAU fixture was deleted** (see Stage 5.1 note).

**NOT built (this is Stage 2, not here):** the `resolve` transform replacing
`specialize_layers`; the `get_nodeattr`/`set_nodeattr` adapter view over the Point;
`node_res_estimation` (only `get_exp_cycles` exists — the *resource* getter is unbuilt);
mixed-graph; the `estimate_layer_resources.json` proof.

### Stage 2 — The seam + estimate-only tier (harness; MVAU as first rider)

**Build:** the `resolve` transform replacing `specialize_layers` (step 6); the adapter's
estimate-tier surface (base qonnx ABC; the `get_nodeattr`/`set_nodeattr` view over the Point;
indexed dtype getters; `get_exp_cycles`; `node_res_estimation`); the shared-frontend wiring
(steps 1–5 untouched; `convert_to_hw` produces an *unresolved* kernel).
**Prove:** a real network with **MVAU on the new substrate, other ops classic** rides
`estimate_only_dataflow_steps` (through step 11) to `estimate_layer_resources.json` — mixed
graph, no codegen.
**Why first:** cheapest milestone that proves the whole integration thesis (shared frontend
+ replaced seam + shallow adapter + mixed graph) with zero full-build surface; and
independently useful (estimates for new kernels).

### Stage 3 — The full-build tail (MVAU to bitfile)

**Build:** the adapter's full-build tier — `code_generation_ipgen` → emit,
`code_generation_ipi` → stitch, `prepare_rtlsim`, `get_verilog_top_module_intf_names` (the
exact-tuple sub-contract). Convert MVAU's duck-typed couplings to explicit capabilities:
`make_weight_file` → `owns_runtime_weights`; minimize → `owns_minimizable_{weights,
accumulator}`.
**Prove:** MVAU-on-new-substrate builds codegen → ipgen → stitch → rtlsim → **bitfile** in a
mixed graph. This is where the foundation's byte-equivalent emit + composed rtlsim plug into
the *real flow*.
**Why here:** with Stages 1–2 done, the entire adapter ladder exists and is proven on one op;
everything after is additive.

### Stage 4 — Capability-query refactor (infrastructure gate)

**Build:** replace the pervasive op_type-string dispatch (~57 call sites across
`set_folding`, `set_fifo_depths`, `make_zynq_proj`, `make_driver`, `create_stitched_ip`,
`insert_iodma`) with **capability queries** on the kernel.
**Why a dedicated stage:** this is the single largest structural obligation and every breadth
op depends on it — a novel op_type is *invisible* to folding/driver/stitch dispatch until it
lands. Interim: Stages 1–2 run on the adapter advertising a legacy-compatible op_type so
classic dispatch still fires. Doing the real refactor here (not buried in the first breadth
op) keeps the infrastructure change clean.

### Stage 5 — Widen across ops (breadth; climbing the built ladder)

Each op family is now additive — declare its design space, ride the existing adapter
(estimate-only first, then full-build). Order by leverage:
1. **VVAU + the source model** — a clean weights sibling of MVAU (near-zero new compute
   surface; proves the ladder isn't MVAU-shaped), but it shares MVU's RTL, so it forces the
   **source/provenance/composition model** (G3, MOTIVATION §1.4). *Note:* the pre-KernelOp
   VVAU fixture (resolve-only, no emit) was **deleted** in Stage 1, so this builds VVAU
   fresh as a KernelOp, not a migration. This is where the `mvu-wrapper-split` exploration
   lands: the monolithic `mvu_vvu_axi.sv` splits into shared `base_head/base_tail.svh` +
   per-microarchitecture `packed.sv`/`softvec.sv`, and each resolved Implementation declares
   *which core → which shared includes* as composable artifacts, not copied strings. This is
   what makes the microarchitecture axis (§1.1) real in RTL, not just in the Python model.
2. **Param-KIND extension → Thresholding + context A** (Requant, LayerNorm, Elementwise) —
   the largest, least-hermetic context; turns the *synthetic* multi-parameter proof into a
   real one.
3. **Port-model extension → contexts B/D/F** (streaming reshape, reduction, backend-only
   shells): output arity + non-AXIS port kinds (AXI-MM master, AXI-lite, TLAST sideband) for
   iodma, checksum, split, DWC.

### Stage 6 — Containers + flag-off

FINNLoop/MLO (Level-2 composition, built on the Stage-2 stitch) and
StreamingDataflowPartition on the container concept; the last families migrate; **delete
`specialize_layers` + the old substrate.** `HWCustomOp` retired.

---

## 5. The cross-cutting schedule — the risk register IS the ordering

Certain couplings are duck-typed today, so a conforming-but-incomplete kernel is **skipped,
not rejected** (unminimized resources, missing runtime weights, unstitched IP — no error).
The migration rule: **convert every silent skip into an explicit capability query before the
kernel class ships.**

| Risk | Coupling | Gate |
|---|---|---|
| **R11** | `specialize_layers` feasibility delegation | **IS Stage 2's seam** — get it right or nothing right of step 6 works. |
| **R2** | `make_weight_file` op_type allowlist | Before any weight-bearing kernel ships full-build → hard gate inside **Stage 3** (MVAU) and again at **Stage 5.2** (Thresholding). |
| **R3** | bit-width minimize `hasattr` guard | Same gate as R2 — capability flags `owns_minimizable_{weights,accumulator}`. |
| **R4** | interface-name tuple shape | **Stage 3** (stitching) — add a schema assertion so structural drift is a hard error. |
| **R1** | pervasive op_type-string dispatch | **Stage 4** — the dedicated infrastructure refactor before breadth. |
| **R10** | non-picklable node state | Disappears if the §2 nodeattr discipline is honored — a design invariant, not a stage. |

## 5.1 Open decisions

Two decisions shape the stages and are recorded here for resolution:

1. **Estimate-only as a shipped increment, or internal milestone only?** Staged as a real
   first target (Stage 2) because it is cheap and independently useful (resource/cycle
   estimates for new kernels, no codegen). If the team values only end-to-end bitfile,
   Stage 2 collapses into Stage 3 and we lose the cheapest proof point. **Recommendation:
   keep separate** — it de-risks the seam before the heavy tail. *Informed by Stage 1:* the
   op-native estimate surface (shapes/widths/rough cost) is already built and venv-tested as
   a cleanly separable layer with zero Vivado — strong reinforcement that estimate-only stands
   alone. (Caveat: `node_res_estimation` is still unbuilt; the cycle side of estimate-only
   exists, the resource side does not.)

2. **Resolution staging — incremental vs atomically-deferred** (introduced in §2).
   Incremental: `resolve` commits `implementation` at step 6, folding at 7–9, as partial
   Points. Atomic-deferred: steps 6–9 gather config, one `resolve` fires later.
   **Recommendation: incremental** — matches the flow's staging, keeps mixed-graph steps
   uniform, natively supported by the guarded-axis model. Atomic-deferred would force the
   frontend into a config accumulator that breaks the "each step commits what it owns"
   symmetry. This determines what "replace `specialize_layers`" emits (a partial Point, not a
   config bag), so it is load-bearing for Stage 2. *Informed by Stage 1:* the estimate
   getters as built presuppose a **fully-resolved** point (they read `implementation` +
   folding dials), so incremental staging must guarantee each getter is only *called* after
   its required axes are committed — a small ordering constraint the depth-keyed staging
   already satisfies (folding-dependent getters are consumed at step 7+).

---

## 6. Where this leaves us

The foundation is proven: the three-legged thesis (G1 declarative space, G2 declarative
composition, G3 hermetic substrate) holds on the hardest op, to a functionally correct
composed rtlsim, byte-equivalent to FINN at every emit boundary. The **resolve core** has
needed **no new primitive** since the first milestone — selection, parameter-delivery
composition, the stitch, and (on paper) MLO all ride the same four primitives (Context /
Axis / Derived / Predicate → resolve). *Above* that core, the KernelOp folding/projection
layer added two things — a derived stream-tiling **expression sub-language** (`space/
tiling.py`: `derive`/`param`/`Mul`/`Div`/`broadcast_aware`, an introspectable AST so
cross-interface fold deps stay orderable) and the **`KernelOp` façade** (`space/
kernel_op.py`: op-side `Interface`, `Implementation.tiling`, `cost_model`). These are a
projection layer *on top of* resolve, not new resolve primitives — the core is untouched.

What remains is the majority of the effort, but it is *scoped, not exploratory*: the consumer
surface is audited (a fixed ~30-method adapter list + a risk register that orders the
migration), the integration shape is decided (share/replace/adapter by depth), and the op zoo
is mapped. The foundation converts a backend rewrite from "rewrite and hope" into "carry
known work onto a proven substrate, along a ladder built once."

**Three honest caveats for planning:** (1) the functional proof is one op at one geometry —
cross-cardinality generality (weights *and* thresholds) is proven synthetically, not yet on a
built multi-param op; closing that is Stage 5.2. (2) the source/provenance/composition model
(G3, MOTIVATION §1.4) is designed but not built — today's emit still leans on FINN's
copied-strings source handling; the RTL microarchitecture split is proven in
`mvu-wrapper-split` but not landed, and closing it is Stage 5.1. (3) the composed rtlsim runs
in the FINN Docker container because of two finn-rtllib/Vivado-2025.2 realities (the un-split
`mvu_vvu_axi` referencing both DSP cores; `axilite.sv`'s use-before-declare that 2025.2
rejects without `-relax`) — documented toolchain accommodations, not model defects, captured
as contained shims in Stage 0.
