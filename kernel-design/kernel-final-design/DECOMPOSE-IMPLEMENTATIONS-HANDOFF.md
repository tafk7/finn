# Handoff: Decompose the MVAU Implementation Pool into Composable Bundles

## The larger project (orientation — where this task sits)

We are designing a **new hardware-kernel backend for FINN** (an FPGA dataflow compiler
for quantized neural nets). In FINN, each neural-net layer becomes a hardware "kernel,"
and each kernel can be built by several **implementations** — HLS vs RTL, and within
RTL, multiple compute microarchitectures (e.g. MVAU's soft-vectorized DSP core vs its
DSP58 INT8-packed core). Today FINN tangles op-identity, backend choice, device
constraints, and codegen together; the redesign separates them.

The current phase builds a **generic design-space model**: represent *all the ways a
kernel could be built* as data — **Axes** (free choices like folding factors PE/SIMD,
memory mode), **Derived** quantities (computed, e.g. accumulator datatype), and
**Predicates** (legality, e.g. "this core needs a Versal part") — and a `resolve`
engine that, given a device Context, yields the legal design points. The **thesis** is
that this makes the backend *composable*: adding a new implementation (a new HLS/RTL
core, or a future AI-engine backend) should be a purely additive change — declare its
axes/derived/predicates/sources, and it slots into the design space with no edits to
existing code. If that holds, FINN can grow backends cleanly; today it cannot.

**MVAU** (matrix-vector-activation — the matmul kernel) is the hardest, richest kernel
and is being used as the acid test. A generic `resolve` engine already exists
(`finn/src/finn/design_space/`) with MVAU encoded as an executable fixture. **Your
task tests the composability thesis directly** by restructuring that fixture.

Full model spec: `kernel-final-design/design-space-model.md`. This is one task in a
longer arc (model → fixtures → codegen/emit → real kernels); you are in the
fixture/model-validation stage — no hardware synthesis in this task.

---

## This task

You are restructuring a design-space fixture so each hardware backend (HLS, and each
RTL core) is a **self-contained implementation bundle** rather than scattered `if
implementation == X` branches across shared lists. The goal is to test — concretely
and falsifiably — whether the model composes: **can you add a new implementation by
adding one bundle and touching nothing else?**

Working root: `/home/tkeller/prj-kernels/`. All paths from there.

---

## 0. READ THESE YOURSELF, FIRST — do not trust any summary (including this doc's)

This handoff gives you orientation and a claim about each file. **Your first job is to
open each file and verify the claim against the actual code.** A prior session's agent
leaned on a prose summary and encoded things that weren't in the source — that is the
exact failure this restructure exists to fix, so do not repeat it. For every file
below: read the cited lines, confirm the claim, and if the code disagrees with this
doc, the CODE wins and you flag the discrepancy.

**Read in this order:**

1. `kernel-final-design/design-space-model.md` — the model. Read §1 (the four
   primitives: Context / Axis / Derived / Predicate), **§1.2.1 and §1.2.2** (the
   implementation-pool and the pool-vs-Derived rule — this is the crux; read it
   twice), and **§8** (the worked MVAU decision tree). *Verify:* that §1.2.2 says the
   pool is defined by distinct buildable source templates, and "forced" is a
   resolve-time outcome, not a design property.

2. `kernel-final-design/mvau-design-space.md` — MVAU's full design space as data,
   every axis/dependency/predicate with `file:line` into real FINN. This is the
   source-of-truth the fixture must match. *Verify:* skim §1 (axis inventory), §2
   (dependency graph), §3 (feasibility catalogue). You will cross-check the fixture
   against this.

3. `finn/src/finn/design_space/space/` — the resolve engine (SMALL, read all of it):
   `context.py`, `axis.py`, `derived.py`, `predicate.py`, `point.py`, `schema.py`,
   `resolve.py`. *Verify:* how a `Schema` holds `axes`/`derived`/`predicates` as flat
   tuples, how `resolve` walks them (`resolve.py`), and how `schema.py` topo-sorts by
   `deps`. This is what you're restructuring *around* — you likely add an
   `Implementation`/bundle construct here.

4. `finn/src/finn/design_space/fixtures/mvau.py` — the fixture under restructure
   (~540 lines). Read it ALL. *Verify:* the flat structure — one `implementation`
   axis (`REALIZATION_POOL = {mvau_hls, mvau_dsp_softvec, mvau_dsp_packed}`), and note
   how every backend's behavior is smeared across `mvau_axes()`, `mvau_derived()`,
   `mvau_predicates()` as `if p.implementation in _DSP_REALIZATIONS` guards and
   early-returns. That smearing is what you are pulling apart into bundles.

5. The real RTL, to ground the source-file decomposition — **read the actual
   SystemVerilog, do not infer it:**
   - `finn/finn-rtllib/mvu/mvu_vvu_axi.sv:305-320` — the `generate` fork that selects
     between the two RTL cores. *Verify:* the exact condition
     `(!IS_MVU) || ((VERSION>2) && (NUM_LANES<=3) && (WEIGHT_WIDTH<=8) &&
     (ACTIVATION_WIDTH<=9))`, and the `NUM_LANES` formula just above it
     (`A_WIDTH = 25 + 2*(VERSION>1)`, `NUM_LANES = A_WIDTH==W? 1 : 1 + (A_WIDTH -
     !NARROW - W)/(W+A-1)`). **This shared-wrapper fork is the whole problem** (see §2).
   - `finn/finn-rtllib/mvu/mvu.sv` — the soft-vectorized core (`mvau_dsp_softvec`).
   - `finn/finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv` — the DSP58 INT8-packed core
     (`mvau_dsp_packed`).
   - `finn/src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py` — the
     Python that fills the wrapper and declares the source-file list (`instantiate_ip`,
     `prepare_codegen_default`). *Verify:* which `.sv` files it lists, and that
     `$VERSION$` comes only from `fpgapart`.
   - `finn/src/finn/transformation/fpgadataflow/specialize_layers.py` — search
     `_mvu_rtl_possible`. *Verify:* the real RTL-feasibility gate the fixture's
     `_rtl_mvu_feasible` predicate mirrors.

---

## 1. Why this restructure (the root cause it fixes)

The fixture's flat structure produced a real defect (audit finding F1, §4 below): the
`_packed_feasible` predicate tried to replicate FINN's *internal* `NUM_LANES`
elaboration math — and got the arithmetic wrong — **because softvec and packed are not
actually separate implementations in FINN.** They share `mvu_vvu_axi.sv` and fork
inside it via `generate`. Reverse-engineering a shared wrapper's fork into a Python
predicate is inherently error-prone.

Decomposing at the source dissolves the whole class of bug: each implementation owns
its feasibility, its axes, its derived, its predicates, and its source files. The
composability thesis then has a **concrete, falsifiable test**: add a new
implementation as one bundle, touch nothing else.

---

## 2. The work

### 2a. Restructure the fixture into implementation bundles (venv-testable — do now)

Today (flat): shared `mvau_axes()`/`mvau_derived()`/`mvau_predicates()` with
`if implementation == X` branches throughout.

Target (bundled): op-level shared elements + a pool of self-contained bundles.

```
Implementation (bundle):
    name          "mvau_dsp_packed"
    feasible(ctx) its OWN requirement, returns reason|None
    axes          the axes it introduces      (e.g. resType={dsp}, pumpedCompute)
    derived       what it computes            (dsp_primitive, dsp_version, SEGMENTLEN)
    predicates    its OWN legality             (SEGMENTLEN>0.741ns, RTL-no-lut, ...)
    sources       the RTL/HLS files it owns    (declared association — see 2b)

Schema  =  op-level shared axes/derived/predicates   (PE, SIMD, MW, MH, mem_mode,
                                                       noActivation, weight-delivery
                                                       cluster, accDataType, ...)
        +  a pool of Implementation bundles
resolve includes ONLY the selected bundle's axes/derived/predicates.
```

Decide (this is genuine design work, not mechanical): does the *engine* (`space/`)
gain a first-class `Implementation` construct, or is a bundle just a convention/helper
that assembles a `Schema`? Let the composability test decide — whichever makes "add a
4th backend = add one bundle" cleanest.

The three bundles to produce, one per **distinct source template** (per §1.2.2):
- `mvau_hls` — HLS compute core.
- `mvau_dsp_softvec` — `mvu.sv`, any DSP part.
- `mvau_dsp_packed` — `mvu_vvu_8sx9_dsp58.sv`, DSP58 with its own feasibility.

### 2b. Associate RTL-lib sources per bundle (declarative — do now)

Each bundle names the `.sv`/`.cpp` files it owns. This surfaces whether the sources
cleanly separate. **Expected finding:** softvec and packed cannot own *disjoint* file
sets today because they share `mvu_vvu_axi.sv` — the shared wrapper is the physical
manifestation of the non-separation. Document this; it motivates 2c.

### 2c. Physically split the shared RTL wrapper (SCOPING DECISION — likely DEFER)

Actually splitting `mvu_vvu_axi.sv` into per-core wrappers (so there is no shared
`generate` fork) is **synthesizable-RTL surgery** that needs Vivado to validate and
belongs to the codegen/`emit` phase, which does not exist yet. **This handoff does not
decide 2c.** Present the design for the split; flag it for the user; default to
deferring the actual `.sv` change until codegen. Do NOT attempt Vivado-dependent work
in the venv.

---

## 3. Acceptance — the composability thesis test

The restructure is proven when:
1. `resolve` still produces identical results for existing MVAU test points (no
   regression — run the existing `finn/src/finn/design_space/tests/`).
2. Each bundle is self-contained: its axes/derived/predicates/feasibility/sources live
   in one place, not scattered.
3. **The thesis test:** add a *hypothetical* 4th implementation (e.g. a stub
   `mvau_lut_rtl` or `mvau_aie`) as one bundle and show it composes into the pool with
   **zero edits to the other three bundles or the op-level shared elements.** Write a
   test asserting it appears as a feasible/infeasible pool member per its own
   `feasible`. This is the whole point — make it a real, passing test.

---

## 4. Fold in the open audit findings (land them in the right bundle)

A prior audit (ground-truth-verified) found these; the restructure is the natural time
to fix them, each landing in its correct home. **Re-verify each against the cited code
before fixing — do not trust these descriptions:**

- **F1 (packed feasibility, wrong math) — FIX in the packed bundle's `feasible`.**
  The real condition is `VERSION>2 ∧ NUM_LANES≤3 ∧ W≤8 ∧ A≤9` with the `NUM_LANES`
  formula at `mvu_vvu_axi.sv:311`. The current predicate DROPS the `NUM_LANES≤3` term
  on a false lemma ("lanes≤3 follows from w≤8∧a≤9") — verified false: small widths
  yield MORE lanes (`W=2,A=2` → 9 lanes → FINN routes to softvec, not packed). Compute
  `NUM_LANES` for real. This becomes packed's OWN `feasible`, not a fork-replication.

- **F2 (bipolar-threshold predicate reads wrong quantity) — likely DROP as
  out-of-scope.** `_bipolar_thresholds_nonneg` checks `p.ActVal >= 0`, but FINN
  (`matrixvectoractivation.py:576-580`) asserts on the **threshold tensor values**
  (`orig_thres_matrix >= 0`), a different object from the scalar `ActVal` bias. The
  fixture's Context models `inp`/`weights`/`out` — **thresholds are not a modeled
  tensor**, so the correct fix is to DROP the predicate with a note "reinstate when
  thresholds are a first-class Context tensor," not to check the wrong scalar.
  (ActVal *is* the threshold activation's bias — related — but it is not the tensor the
  assertion guards. Verify at `matrixvectoractivation.py:156` (ActVal=out_bias) vs
  `:578` (assert on orig_thres_matrix).)

- **F3 (true-binary predicate wrong both ways) — FIX in the hls bundle.** Real
  condition (`hls .../matrixvectoractivation_hls.py`, and `matrixvectoractivation.py`
  ~:574): reject when `(input_binary OR weight_binary) AND NOT binaryXnorMode`. The
  fixture checks only input, ignores the xnor escape (over-rejects) and binary weights
  (under-rejects). Fix to the full condition.

- **F4 (mlo_max_iter domain {0..64} invents a bound at the wrong entity) — REMOVE the
  bound from the axis.** The `64` is `n_max_layers` in the MLO fetch-weights HDL
  (`hwcustomop.py:378`) — the size of a generated hardware table, a *fabric-wide*
  resource budget across all MLO layers. It does NOT bound THIS node's `mlo_max_iter`
  (a per-node iteration count). Verify at `hwcustomop.py:317-319` (`sets=mlo_max_iter`,
  unbounded) and `:378` (`n_max_layers=64`, a separate template constant). Make
  `mlo_max_iter`'s domain unbounded non-negative int; the 64 is a different concern
  that does not belong on this axis.

(Lower-severity audit notes F5–F8 exist — accDataType round-up-to-8 for terminal
no-activation nodes; a narrow_weights notion mismatch in `_rtl_mvu_feasible`;
`_weight_stream_width` dynamic_input case; dynamic_input/mlo/external mutual-exclusion.
Address if cheap during the restructure; otherwise leave a note. Get them from the map
§2/§3 and the real code, not from memory.)

---

## 5. Environment

Venv at `prj-kernels/.kernel-venv/` (pure Python + data; no Docker for the fixture/
engine work). Run:
```
PYTHONPATH="qonnx/src:finn/src" .kernel-venv/bin/python -m pytest \
  finn/src/finn/design_space/tests/ -q
```
28 tests currently pass. Keep them green; add the thesis test (§3.3).

Do NOT pip-install into the venv or attempt Vivado/synthesis work — 2c is deferred.

---

## 6. What NOT to do

- Do not re-litigate the softvec-vs-packed *pool membership* — that is settled (two
  members, per §1.2.2). You are restructuring HOW they are expressed, not WHETHER they
  are separate.
- Do not attempt the physical `mvu_vvu_axi.sv` split (2c) — design it, defer it.
- Do not trust this doc's line numbers or claims blindly — open the file, confirm, and
  flag any discrepancy. That discipline is the point.
