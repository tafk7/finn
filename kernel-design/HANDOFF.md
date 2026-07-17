# Handoff: Fresh Design Session — a FINN-native Kernel implementation

You are starting a fresh design session to design a **new, FINN-native Kernel
backend implementation**, borrowing from and building on two real systems
(Microsoft's **Brainsmith** and the FINN **`feature/kernel_flow` prototype**).
This document is the single most important thing to read first: it carries
orientation that came out of a long prior analysis+design conversation and lives
**nowhere in the artifact files**. The artifacts are evidence; this is the map.

Working directory root: `/home/tkeller/prj-kernels/`. All paths below are from there.

---

## 0. The one-paragraph goal

Design a Kernel backend abstraction for FINN that keeps **Brainsmith's
declarative schema + design-space derivation + DSE** (its proven front-end),
adopts the **prototype's composed-backend structure** (which dissolves FINN's
base-class leak and `execute_node` diamond *by construction*), and adds a
**selection model that can express device-dependent feasibility** (which *neither*
real system managed). It must stay **FINN-native**: a real qonnx `CustomOp` on the
graph, saveable/reloadable to ONNX, interoperable with existing FINN transforms.
This is a **hybrid neither existing system is** — that is the whole reason to
build rather than just adopt Brainsmith.

---

## 1. Why build at all (the improvement over "just use Brainsmith")

A four-way analysis (FINN baseline, a theorycrafted "ideal", the prototype,
Brainsmith) established that **Brainsmith and the prototype each solved a disjoint
half of FINN's structural problems, and neither can reach the other's half
without the move this design makes.** Concretely, mapped against the baseline's
top-10 "pressure points":

| Capability | Brainsmith | Prototype |
|---|---|---|
| Declarative schema-derivation of the shape/width contract | ✅ | ❌ (copy-pastes shapes into every op) |
| First-class DSE (design space / design point) | ✅ | ❌ |
| Behavior reuse across an op's backends | ✅ (a shared `KernelOp` base) | ❌ (duplicates `make_weight_file`, the ~50-line `execute_rtlsim` body, etc. per op) |
| `execute_node` diamond dissolved | ❌ inherited from FINN | ✅ by construction (single-inheritance value object, 0 `execute_node`, 0 shims) |
| `code_gen_dict` mutable side-channel removed | ❌ inherited | ✅ by construction (pure-return producers, dict-merge) |
| Base-class op_type leak removed | ❌ inherited | ✅ (substrate has zero op_type branching) |
| Device-dependent selection (e.g. DSP48-vs-DSP58, Versal-only) | ❌ (FINN's `_mvu_rtl_possible` ladder ported verbatim) | ❌ (dropped — constraints are `Callable[[Kernel],bool]` with no fpgapart/model) |

**The structural reasons the two cannot converge on their own:**
- Brainsmith can't get the prototype's column because `KernelOp(HWCustomOp, ABC)`
  *inherits* `HWCustomOp` — the mixin **is** the diamond, the shared base **is**
  the leak. It can't shed them without leaving `HWCustomOp`, and it can't leave
  without breaking FINN interop.
- The prototype can't get Brainsmith's column because it has **no shared op
  base** (every backend is a bare `class …(Kernel)` value object), so it pays a
  **behavior-duplication tax** — the direct cost of "value object, no base".
- **Neither** solved device-dependent selection.

So the design is a genuine three-legged synthesis: **Brainsmith's derivation +
the prototype's composed backend (but re-based so behavior is shared) + a
contextful selection model.** No single existing system is a substitute. That is
the answer to "why not just use Brainsmith."

---

## 2. Corrections that OVERRIDE the artifacts (read before the design docs)

The prior conversation produced a design doc
(`kernel-final-design/kernel-final-design.md`, "v2") that **over-reached**, and
this session walked much of it back. **Where v2 and this handoff disagree, this
handoff wins.** The corrections, each hard-won:

1. **KEEP per-op classes.** v2's "no per-op classes, ops as pure schema-data" was
   wrong. `ThresholdingOp(KernelOp)` — single inheritance, per-op, holding
   `build_schema`, the golden reference, `can_infer_from`/`infer_from` — is
   correct and is how ONNX/qonnx is meant to work. An op has real per-op behavior;
   a class is its natural home. **The objection is narrow: only the *backend
   mixin* (`Thresholding_hls(Thresholding, HLSBackend)`) is toxic** — the second
   inheritance edge is what creates the diamond and the leak. Replace *that* with
   a **composed** backend the op holds; keep everything else about the class.

2. **The `execute_node` diamond is a WEAK argument — do not lead with it.** A
   shared `execute_node` dispatching on `exec_mode` (python golden / cppsim /
   rtlsim) is a *legitimate, useful* multi-fidelity design. The composed
   "reference vs runner" split just *relocates* that dispatch into an adapter; it
   is a minor cleanup (name the reference model, share the sim harness), not a
   headline win. **Lead the design's value on axis-separation and the stitch
   god-method instead** (see #3, #4).

3. **The strongest concrete win is AXIS-SEPARATION, shown by `mem_mode`.** In real
   Brainsmith Thresholding, `mem_mode ∈ {internal_embedded, internal_decoupled}`
   is a nodeattr that **branches ~16 methods** in the HLS backend (an orthogonal
   memory-delivery axis smeared through the op as a cross-cutting conditional).
   Making it a **composed part** (a `WeightDelivery`/`MemoryStrategy` object)
   collapses the 16 branches to "which part is bound." *This* is the design's real
   argument on a real op — not the dtype-envelope story.

4. **The stitch / `code_generation_ipi` god-method is real and unsolved.** ~130
   lines of block-design TCL string-surgery (hardcoded pins, `os.listdir`,
   `FINN_ROOT`, dead `ap_clk2x` workaround) sit in the HLS backend, inherited from
   FINN. The prior "ideal" proposed a typed role-binding stitch layer to dissolve
   it but **never validated it against real Vivado TCL** — treat it as *open
   design work*, the highest-risk piece, not a solved problem.

5. **The dtype-"envelope/superset" story does NOT fire on most ops.** The idea
   (each backend declares its buildable dtypes; feasibility = union across
   backends; adding an FP backend = one new entry, no schema edit) is sound *in
   general*, and killing the hand-maintained-superset is even Brainsmith's own
   stated roadmap — **but** real Thresholding's HLS and RTL backends share the
   same integer dtype support, so nothing is exercised. **Do not showcase the
   design on dtype-envelope; it needs an op whose backends genuinely differ in
   dtype support.** (An earlier toy example *fabricated* such a difference — that
   file has been deleted for being misleading.)

6. **The prototype REPLACES `HWCustomOp` but via an ad-hoc bridge, not a clean
   adapter.** `class Kernel:` inherits nothing (verified). But it still calls
   `getCustomOp(node).set_nodeattr(...)` to write results back onto a *still-present
   legacy CustomOp node* (`kernel.py:355-367,475`). So it's a transitional
   dual-world, not a clean break. **Our contribution on this axis is formalizing
   that bridge into ONE generic `KernelCustomOp(CustomOp)` adapter** — the single
   boundary where the composed core meets the FINN graph. This is not a new
   invention; it's cleaning up a coupling the prototype already needed.

7. **Serialization is preserved and must stay a first-class constraint.** FINN
   must save a specialized-kernel graph to ONNX and reload it. The mechanism (both
   real systems use it): the **file stores state** (op_type + nodeattrs incl. an
   `implementation` key + config), the **registry/pool supplies behavior** by
   name — **nothing is pickled**. The composed `Implementation` (with its
   callables) is reconstructed by name from a registry, exactly as qonnx resolves
   a class from `(domain, op_type)`. Keep this property; do not design anything
   that requires serializing behavior.

---

## 3. Design ingredients that are settled enough to build on

- **`Implementation` is an open, op-keyed POOL, not a language enum.** FINN ships
  multiple RTL microarchitectures for one op (`finn-rtllib/mvu/`:
  `mvu_8sx8u_dsp48`, `mvu_vvu_8sx9_dsp58`, `mvu_4sx4u` — systolic vs GEMM vs
  DSP-packed). So "HLS/RTL" is a *language field*, not the identity. Selection
  picks "which entry in `pool[op_kind]`" by feasibility then preference — the same
  path for systolic-vs-GEMM as for HLS-vs-RTL.
- **Node representation stays visible & lowering-checkable.** op_type is a
  *projection* of `op_kind + implementation` (`MVAU` unlowered →
  `MVAU_systolic_dsp` lowered) resolving to the one generic adapter — many op_type
  strings are fine, N hand-written classes are what to avoid. A required
  `implementation` nodeattr is the machine-readable source of truth; the op_type
  string is for humans/netron.
- **Selection = feasibility ⊥ preference.** Feasibility is a hard predicate over
  an injected context (dtypes, **fpgapart**, model) — this is the piece both real
  systems lacked (prototype's constraint has no fpgapart; Brainsmith ported FINN's
  ladder). Preference (`priority` + a cost function) is an **intentionally OPEN
  algorithm seam** — fix the interface, leave the strategy pluggable. Do not try
  to finalize the cost model this iteration.
- **Template-filling: Jinja for the regular tier, imperative for `docompute`.**
  Brainsmith already uses Jinja to code-generate backend *class skeletons*
  (`tools/kernel_integrator/…/auto_rtl_backend.py.j2`) — evidence backends are
  mechanically derivable from declared metadata. Jinja cleanly handles the
  value-substitution / structural-repetition tier (`defines`, `pragmas`,
  `blackboxfunction`, interface lines). It does **not** help `docompute` (a
  variable-depth PE-folded loop nest / structure-dependent function call) —
  Brainsmith's own elementwise writes that imperatively despite having Jinja. Plan
  for: declared bindings + template for the regular tier, a small imperative
  escape for `docompute`.

---

## 4. What NOT to over-invest in (deferred / out of scope for a first cut)

- **The stitch god-method** (#4 above) — real but open; scope it as a milestone,
  don't block the core on it.
- **The cost/preference model** — keep it a named open seam.
- **MVAU** — it is a bundle of half-a-dozen orthogonal features stuffed into one
  op (weight delivery, dual folding MW×MH, mem_mode, the 236-line IPI, DSP
  microarchs, threshold fusion). It deserves its **own** later census→design pass.
  **Validate the core on simpler ops first** (thresholding embedded-only,
  channelwise, addstreams).
- **DSE search policy, build cache** — real but orthogonal; adopt Brainsmith's DSE
  engine, don't redesign search.

---

## 5. Reading list (rooted at `prj-kernels/`)

### Read with confidence — real code, adversarially verified (the ground truth)
- `finn-hw-backend-analysis/hw-backend-model.md` + `census-matrix.md` — the
  **baseline problem statement**: FINN's as-is defects and the top-10 pressure
  list everything is measured against.
- `kernel-prototype-profile/` (`prototype-profile.md`, `core-model.md`,
  `build-flow.md`) — the **prototype**, profiled from real code.
- `brainsmith-profile/` (`brainsmith-profile.md`, `core-model.md`,
  `folding-and-dse.md`, `docs-consistency.md`, `as-designed-from-docs.md`) — 
  **Brainsmith**, profiled + docs-diffed.
- `compare-prototype-vs-baseline/pressure-point-matrix.md` — **the single most
  useful synthesis artifact**: which pressure points each real system resolved,
  as-built vs as-intended, with residuals. Start here after this handoff.
- `compare-brainsmith/four-way-synthesis.md` and `vs-prototype.md` — real-vs-real
  positioning.

### Read with the caveats in §2 — useful but superseded or paper-biased
- `kernel-final-design/kernel-final-design.md` — the **prior "v2" design attempt.**
  Most developed, but **substantially walked back** by §2 above. Read it for the
  identity/envelope framing and the Implementation-pool shape, **not** as the
  spec. It over-reaches on per-op-classes, the diamond, and dtype-envelope.
- `kernel-final-design/toy-vs-brainsmith-thresholding.md` — the honest correction
  of the (now-deleted) toy against real Brainsmith code; good on where the real
  wins are (axis-separation, not dtypes).
- `finn-hw-backend-analysis/kernel-model.md` — the theorycrafted **"ideal"**
  (never built, paper only). Hold loosely. Its one genuinely useful contribution
  the real systems lack: the **selection model** (feasibility-as-data, a *closed*
  governed predicate vocabulary, the two named n-ary couplings
  `PumpedClockCoupling`/`MloCoupling`). Mine it for that; ignore its "by
  construction" guarantees, which are unproven.
- `compare-prototype-vs-ideal/`, `compare-brainsmith/vs-ideal.md` — comparisons
  *against* the paper ideal; the `fairness-audit.md` files flag home-team bias.

### Deleted on purpose (do not go looking for them)
- The `seam-*.md` docs and the `example-thresholding.py` toy were **deleted**: the
  seams are frozen first-synthesis output predating every §2 correction, and the
  toy **fabricated** envelope differences that don't exist in real code. Both would
  mislead a fresh reader.

### The real code (the actual subjects — read directly, don't trust summaries alone)
- **Brainsmith**: `brainsmith/brainsmith/dataflow/` (esp. `kernel_op.py`,
  `schemas.py`, `constraints.py`, `dse_models.py`) and
  `brainsmith/brainsmith/kernels/thresholding/` (all four files — the cleanest
  cross-design anchor).
- **Prototype**: on git branch `feature/kernel_flow` — it lives on the **`upstream`
  remote** (Xilinx/finn), NOT as a local branch and NOT in the working tree.
  Access via `git -C finn show upstream/feature/kernel_flow:src/finn/kernels/<path>`
  (list a dir with `git -C finn ls-tree upstream/feature/kernel_flow:src/finn/kernels/<dir>`).
  Key: `kernel.py`, `kernel_registry.py`, `kernels/thresholding/`.
  **Ignore `feature/new_kernel_flow`** — despite the name it is an older precursor
  (`kernel_flow` is 32 commits ahead of it; `new_kernel_flow` has no
  `src/finn/kernels/` dir at all). `feature/kernel_flow` is the prototype our
  analysis profiled and the one to use.
- **Baseline**: `finn/src/finn/custom_op/fpgadataflow/` (`hwcustomop.py`,
  `matrixvectoractivation.py`, and the `hls/`+`rtl/` variants).

---

## 6. Suggested first move for the fresh session

Interfaces before internals (modular-dev). The irreducible core is the **seam
between op-identity and implementation**, and the value that crosses it
hermetically is the **design_point** (an immutable value the identity derives,
the implementation consumes). Define, minimally:
- `KernelOp` (identity): `build_schema`, `reference`, `can_infer_from`/`infer_from`,
  derives `design_point`. **Keep it a class; consider basing it on Brainsmith's
  `KernelOp` so derivation is inherited, not duplicated.**
- `Implementation` (composed backend): `precondition(ctx)` (feasibility, with
  fpgapart), declared `knobs`, `emit(design_point) → artifacts`. **The hermeticity
  test IS the deliverable**: if `emit` can be written against nothing but the
  `design_point` — no `self.onnx_node`, no reaching into a base — the seam is
  clean and the thesis holds. If it can't, we learn that cheaply.
- `Registry` + `KernelCustomOp(CustomOp)` adapter: the one boundary to the FINN
  graph; select an `Implementation` from `pool[op_kind]`, wrap for FINN, round-trip
  to ONNX.

Prove it on **one simple op, embedded-only, with two implementations** (so
selection is actually exercised). Explicitly out of scope for the first cut:
stitch layer, `mem_mode`-as-part, DSE search, cache, MVAU, the cost model.

---

## 7. Repo / git setup

- `prj-kernels/finn/` is the working repo (tafk7/finn fork), checked out on
  **`dev`** at the latest Xilinx `upstream/dev`. Start the fresh implementation on
  a new branch off `dev`.
- Remotes: **`origin`** = `git@github.com:tafk7/finn.git` (your fork),
  **`upstream`** = `git@github.com:Xilinx/finn.git`.
- The **prototype** and other reference branches (`feature/kernel_flow`, the
  various `feature/*`) live on **`upstream`** — reachable with
  `git show upstream/feature/kernel_flow:...`, no checkout needed.
- `brainsmith/` is a **sibling repo** under `prj-kernels/`, read directly.
- The analysis+design artifacts (`finn-hw-backend-analysis/`, `*-profile/`,
  `compare-*/`, `kernel-final-design/`, this `HANDOFF.md`) are plain folders under
  `prj-kernels/`, **not** tracked in the `finn` repo — read them in place.
