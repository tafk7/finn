# Design Brief — The "Kernel" Backend System

*The fixed rubric for the Kernel design panel. Every architecture agent optimizes
against THIS document, and every design is scored against it. This is the
"interfaces first" discipline applied to the redesign: get the brief right and
the five parallel designs become comparable; get it wrong and the panel's work
is wasted.*

**Status:** DRAFT for user review (Phase D0 gate) before the design panel runs.
**Grounded in:** `../hw-backend-model.md` (the verified as-is model) and the
27-family census.

---

## 0. Goal

Replace the current **2-axis** HW-op abstraction (`HWCustomOp` substrate + one of
`{HLSBackend, RTLBackend}` mixed in by multiple inheritance) with a single
**unified Kernel model** in which HLS, RTL, and **static IP** (pre-optimized
hard IP for common cases) are peer *implementation kinds* under one contract —
not a fixed inheritance axis.

**Stance:** clean-slate ideal. Design the right abstraction first; the migration
path is derived *after* a winner is chosen, not baked into the design. The
current system is a thing to be replaced, not preserved.

**Scope:** full unification. One coherent design must cover the backend
abstraction, variant selection, folding parameterization, resource estimation,
the execution/rtlsim harness, and non-tensor ports. These are not separate
follow-on passes — the whole point is that they stop being separate.

---

## 1. Invariants (the regression set — non-negotiable)

A design is only valid if the system it describes could still compile every op
that compiles today. The **27 op families** in `../op-census/` are the regression
set. Concretely, the Kernel model must be able to express, without special-casing:

- All 24 op families that split HLS/RTL, plus the 4 backend-only ops
  (iodma, checksum, tlastmarker, finn_loop), plus the meta/container node
  (streamingdataflowpartition).
- The 8 datatype/shape/stream-width accessors that constitute the real op
  contract today (`get_{input,output}_datatype`, `get_{normal,folded}_{input,
  output}_shape`, `get_{in,out}stream_width`).
- Both execution modes (cppsim, rtlsim) and the ipgen/IPI build flow.

The design need not preserve the *class structure* or *method names* — only the
externally observable capability set.

---

## 2. Must-fix acceptance criteria (the design's test suite)

These are the top-10 redesign-pressure points from the as-is model, restated as
pass/fail criteria. **A design that does not resolve items 1–4 is disqualified.**

| # | Criterion | Fails today because | A passing design… |
|---|-----------|--------------------|-------------------|
| **1** | **Weight-delivery is a first-class axis, not smeared across backends** | MVAU: `mem_mode × dynamic_input × mlo` weight delivery is emitted from *both* HLS and RTL backends and stitched by a 236-line `code_generation_ipi` in the "agnostic" base (`matrixvectoractivation.py:920`) | models weight/param delivery as its own dimension, orthogonal to compute-impl, expressible once |
| **2** | **Substrate never branches on subclass identity** | `HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` hard-code op-type allowlists and call subclass-only methods (`hwcustomop.py:307/355/407`) — 10 base-class leaks | the substrate depends only on declared contracts; adding/removing an op never edits shared base code |
| **3** | **Selection is contract-driven, not a per-op god-switch** | `specialize_layers.py:40-211` is a hard-coded per-op RTL-feasibility ladder; variant resolved by string concat (`optype+"_"+impl_style`) | backends *declare* what they can realize; selection is constraint resolution over declared capabilities |
| **4** | **`execute_node` is not an inheritance diamond** | functional-golden-model vs backend-exec collide under MRO; ~8 hand-written shims; latent correctness bug (`vectorvectoractivation_rtl.py:89`) | separates "reference semantics" from "how this implementation runs" into distinct contract slots |
| **5** | **HLS codegen has no mutable-dict side-channel** | `code_gen_dict` is populated by ordered side-effecting steps (`hlsbackend.py:136`); non-composable, non-unit-testable | implementation emission is a pure function of typed inputs → artifacts |
| **6** | **Memory-strategy (embedded/decoupled/external/ROM-vs-DMA) is modeled, not re-branched** | thresholding/lookup/MVAU each re-branch `mem_mode` independently; base sniffs subclass name | one memory-strategy concept shared across ops |
| **7** | **Backend-only + container ops satisfy the contract honestly** | iodma/tlastmarker have shape getters that `raise`; finn_loop stubs `get_rtl_file_list`→None | the contract admits infra/container kernels as first-class, not as violations |
| **8** | **op↔finn-rtllib coupling is a typed adapter, not `str.replace`** | 26 untyped `$KEY$` replacements, triplicated source manifests, magic numbers in `.v` overriding the Python dtype contract | a typed parameterization interface between op and its HDL/IP artifact |
| **9** | **No hidden runtime "third backend"** | `streamingfifo_rtl.py:141` selects a "vivado" impl by runtime attr, base reaches through via try/except | all implementation kinds are explicit peers under the Kernel contract |
| **10** | **No ambient singletons / cross-op coupling in the core** | `finnxsi` import-time singleton; SWG↔VVAU layout coupling; shuffle cost-model fused to HLS pipeline | dependencies (simulator, fpgapart, build dir) are injected; ops build/test/sim in isolation |

---

## 3. The forcing function: static IP as a third implementation kind

The single most important design test. Today "backend" is a binary
{HLS, RTL} baked into the class hierarchy. The Kernel model must treat
**static IP** — a pre-synthesized, pre-optimized hard IP block for a common
parameterization (e.g. a hand-tuned INT8 GEMM, a vendor FFT) — as a **peer
implementation kind**, selected when its preconditions match, with the same
port/folding/resource contract as an HLS or RTL realization.

If a design only abstracts "HLS vs RTL," it fails the moment static IP is added.
Any design that requires touching the substrate to add a third kind fails
criterion #2. **The third kind is how we know the abstraction is real and not
just a rename of the current split.**

---

## 4. Axes that must stay orthogonal (do not re-conflate)

The as-is system's core failure is collapsing several independent axes onto the
single HLS/RTL inheritance axis. A passing design keeps these separable:

1. **Compute implementation kind** — {HLS, RTL, static-IP, …} — how the math is realized.
2. **Weight/param delivery** — embedded / decoupled-streamed / external / dynamic-load / loop-fetched — how coefficients reach the compute.
3. **Memory strategy** — where params live (LUTROM / BRAM / URAM / off-chip DMA) and the read discipline.
4. **Folding / parallelism** — PE/SIMD (and op-specific folding) behind a
   *uniform interface*. **v1 requires a shared foundation (one interface, common
   vocabulary), NOT full unification of the folding math** — per-op folding
   *implementation* may differ. Deep folding unification is deferred (see §9.3).
5. **Port model** — tensor streams *and* non-tensor ports (AXI-Lite control, AXI-MM, TLAST/TKEEP sideband) as first-class, typed.
6. **Execution/verification** — reference semantics vs cppsim vs rtlsim as distinct concerns over a common port contract.

The design should state, for each axis, what the abstraction is and how it
composes with the others.

---

## 5. Evaluation criteria (how D2 scores each design)

Each design is scored 1–5 on:

- **Substitutability** — can any implementation satisfying a Kernel's contract replace another with no change elsewhere? (the modular-dev master property)
- **Hermeticity** — are dependencies declared/injected, or ambient? Can a Kernel be built/tested/simulated in isolation?
- **Axis orthogonality** — are the six axes above genuinely independent, or does the design secretly re-couple them?
- **Substrate purity** — does the core ever need to know a concrete op/kind? (criterion #2)
- **Uniformity** — one folding model, one selection mechanism, one exec harness — vs per-op variation.
- **Expressive completeness** — can it express all 5 conformance ops (§6) without special-casing?
- **Migration tractability** — *secondary*, scored but not disqualifying (clean-slate stance): how painful is the path from today?

---

## 6. Conformance set (every design MUST express these)

A design is judged by whether it can model the five hardest real cases. Each
design doc must include a concrete sketch of how these are expressed in its
model. **A design that cannot cleanly model MVAU is dead.**

1. **MVAU (matmul)** — the worst case. Must show: compute (HLS/RTL/static-IP)
   separated from weight-delivery (all of embedded/decoupled/external/dynamic/MLO),
   with the 236-line IPI stitching (`matrixvectoractivation.py:920`) expressed as
   *composition of declared parts*, not a base-class method. 40 overrides today.

2. **Thresholding** — memory-strategy (embedded vs decoupled) as an orthogonal
   axis, not `mem_mode` try/except in the base. Threshold serialization
   (`make_weight_file`/`minimize_weight_bit_width`) unified, not triplicated.

3. **finn_loop (container)** — a meta-node wrapping a subgraph, stitched into a
   block design. Must show the contract admits container/hierarchical kernels
   without stubbing (`get_rtl_file_list`→None today) or category errors.

4. **iodma (non-tensor infra)** — direction-bifurcated AXI-MM primitive whose
   folded-shape getters `raise` today. Must show non-tensor ports and infra
   kernels are first-class, with an honest shape/port contract.

5. **Static-IP GEMM (hypothetical, new)** — a hand-optimized INT8 GEMM hard IP
   as a third implementation kind for MVAU's common case. Must show: it plugs in
   as a peer to HLS/RTL MVAU, is *selected* when its preconditions match (via the
   declared-capability mechanism, criterion #3), and requires **zero** substrate
   edits (criterion #2). This op does not exist today — it is the forcing
   function that proves the abstraction generalizes.

---

## 7. What each design agent must return

A structured design document containing:
- **Core model** — the Kernel contract(s), the module structure, how the six
  axes (§4) are represented and composed. Diagram encouraged.
- **Axis treatment** — for each of the 6 axes, what the abstraction is.
- **Conformance sketches** — how each of the 5 ops (§6) is expressed.
- **Criteria self-assessment** — how it satisfies must-fix items 1–10 (§2), with
  explicit attention to the disqualifiers (1–4).
- **Known weaknesses** — where this design pays a cost (honesty required; the
  judge will find them anyway).
- **Assigned-prior fidelity** — how this design embodies its assigned angle.

---

## 8. The five assigned priors (D1 panel)

Each agent explores one region deliberately, so the panel spans the solution
space instead of converging on one generic answer:

1. **Composition-over-inheritance** — a Kernel *is* a data-contract object that
   *has* a pluggable implementation-strategy (+ weight-delivery, memory,
   folding as composed parts). Directly attacks the mixin diamond and base leak.
2. **Capability / trait-based** — ops declare capabilities (has-weights,
   multi-stream, needs-thresholds, non-tensor-ports); implementation kinds
   declare what they can realize; selection is constraint solving over the two.
   Attacks the god-switch (#3).
3. **Explicit multi-axis / product type** — model the Kernel as an explicit
   product of the six orthogonal axes (§4); a concrete op is a point in that
   space. Attacks axis-conflation head-on.
4. **IR / lowering-target** — the Kernel is a typed compilation target with a
   port/folding/resource contract; HLS, RTL, and static-IP are *lowerings* of
   it. MLIR-dialect thinking; attacks the codegen side-channel and the
   op↔artifact coupling.
5. **Evolutionary control (baseline)** — the smallest change to the *current*
   structure that satisfies must-fix items 1–10, with no clean-slate rewrite.
   The baseline the ambitious four must *beat* — if an ambitious design isn't
   clearly better than this, that's a finding.

---

## 9. D0 gate decisions (confirmed with user)

1. **Static-IP GEMM is the anchor.** The hypothetical hand-optimized INT8-GEMM
   hard IP (§6.5) is confirmed as the forcing-function conformance case. Designs
   express static IP as a peer implementation kind through it.
2. **Both selection strategies are in scope.** A design MAY have the Kernel own
   its own feasibility (self-advertising) OR keep selection a separate pass that
   queries declared capabilities — both are viable. Each design should state
   which it chooses and why; the judge compares across the two strategies rather
   than mandating one. (Satisfies criterion #3 either way.)
3. **Folding: shared foundation only, not full unification.** v1 needs one
   *uniform interface* and common vocabulary for folding/parallelism so the rest
   of the model can rely on it — the deep unification of per-op folding math is a
   complex topic deferred to a later pass. Designs should define the interface
   and explicitly scope out the math. Do not over-invest here.
4. **The six axes (§4) are sufficient.** No seventh axis. Clock/multi-clock
   (MVAU clk2x) is treated as a detail within the existing axes, not a
   first-class dimension.
