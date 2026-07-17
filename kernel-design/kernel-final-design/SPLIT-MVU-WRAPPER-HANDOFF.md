# Handoff: Physically split the shared MVU RTL wrapper into per-core standalone wrappers (the "2c" work)

## 0. TL;DR

The RTL-MVAU backend fuses two distinct compute cores (`mvu.sv` soft-vectorized,
`mvu_vvu_8sx9_dsp58.sv` DSP58 INT8-packed) inside ONE shared wrapper
(`mvu_vvu_axi.sv`), selecting between them with an internal `generate` fork. Our
design-space model treats them as two **separate pool members** that each *own* one
core `.sv`. Those two views are in conflict: the model says "two standalone designs,"
the RTL says "one wrapper, both cores compiled in, fork picks."

**This task resolves the conflict in favour of the model** — split the shared wrapper
into two standalone per-core wrappers so each pool member owns a genuinely disjoint,
independently-elaboratable source set, and delete the fork. This is synthesizable-RTL
surgery validated by rtlsim (Vivado), NOT modeling work.

**Decision already made (do not re-litigate):** we split (Option A), we do not collapse
the model to match FINN's fusion (Option B). Rationale is in §6. The user chose this
explicitly after weighing both.

---

## 1. Why this task exists (the concrete trigger)

We built hermetic `emit` for MVAU (see `project_emit_phase` in memory, and
`finn/src/finn/design_space/fixtures/mvau/emit_rtl.py`). A **differential test** against
FINN's own codegen passed (all 15 wrapper params + HLS `params.h` byte-match — see
`finn/src/finn/design_space/tests/diff_mvau_emit_vs_finn.py`, runs green in Docker).

Then an **RTL elaboration check** (`elaborate_mvau_emit.py`) failed at `xelab`:

```
ERROR: [VRFC 10-2063] Module <mvu_vvu_8sx9_dsp58> not found while processing
       module instance <blkDsp.genINT8.core> [mvu_vvu_axi.sv:315]
```

Cause: our `mvau_dsp_softvec` bundle ships only `mvu.sv` (its owned core), but the
shared wrapper's `generate` fork **textually references BOTH** cores, so Verilog
elaboration demands both `.sv` present regardless of which branch is reachable. FINN
sidesteps this by shipping ALL six files for every RTL-MVAU node
(`matrixvectoractivation_rtl.py:168-175`) — the fused-reality workaround.

The split makes each bundle's owned source set == its compile set, so
`mvau_dsp_softvec` elaborates against `{wrapper_softvec, mvu, plumbing}` alone, and the
elaboration check passes honestly.

---

## 2. The current RTL structure (read these YOURSELF, first)

Working root: `/home/tkeller/prj-kernels/`. All paths from there. **Read the actual
SystemVerilog before touching it — do not trust this summary.**

- `finn/finn-rtllib/mvu/mvu_vvu_axi.sv` — THE shared wrapper (~398 lines). Structure:
  - **AXI-lite param wrapper + I/O** (module header, three AXI-Stream interfaces).
  - **Replay buffer** for activation reuse (`replay_buffer` instance, ~line 141).
  - **Input unflattening / VVU interleave** (~line 147-170).
  - **`blkDsp` block (lines ~177-342)** — the double-pump machinery (unpumped +
    `genPumpedCompute` clk2x logic, ~177-303) that prepares `dsp_w`, `dsp_a`,
    `dsp_last`, `dsp_zero`, then the **`generate` fork (lines 305-340)**:
    `if(!IS_MVU || (VERSION>2 && NUM_LANES<=3 && w<=8 && a<=9)) genINT8: mvu_vvu_8sx9_dsp58`
    `else genSoftVec: mvu`. **Both branches receive IDENTICAL signals** (`dsp_w`,
    `dsp_a`, `dsp_last`, `dsp_zero`, the `PUMPED_COMPUTE? ap_clk2x : ap_clk` mux) and
    produce IDENTICAL outputs (`dsp_vld`, `dsp_p`). This is the clean cut line.
  - **`blkOutput` block (lines ~344-396)** — output FIFO/queue, `ovld`/`odat` → AXI out.
  - The fork carries a `@todo Push core selection decision entirely into mvu.sv`
    (line 305) — the RTL authors themselves flag the fork as debt, AND it DUPLICATES
    `mvu.sv`'s `NUM_LANES` math (lines 308-311 ≈ `mvu.sv:113-115`) — the duplication
    that caused audit finding F1.
- `finn/finn-rtllib/mvu/mvu.sv` — soft-vec core. Params `VERSION, PE, SIMD, *_WIDTH,
  ACCU_WIDTH, SIGNED_ACTIVATIONS, NARROW_WEIGHTS, FORCE_BEHAVIORAL`. Structured ports
  `w[PE][SIMD][WEIGHT_WIDTH]`, `a[SIMD][ACTIVATION_WIDTH]`. Instantiates DSP48E1/E2/DSP58
  via `case(VERSION)`.
- `finn/finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv` — packed core. Params `IS_MVU, PE, SIMD,
  *_WIDTH, ACCU_WIDTH, SIGNED_ACTIVATIONS, SEGMENTLEN, FORCE_BEHAVIORAL`. FLAT packed
  ports `w[PE*SIMD]`, `a[(IS_MVU?1:PE)*SIMD]`. DSP58 cascade chains.
- `finn/finn-rtllib/mvu/mvu_vvu_axi_wrapper.v` — the outer IP-packaging wrapper (the
  `$KEY$` template our emit fills). Instantiates `mvu_vvu_axi`. 14 params.
- Supporting: `mvu_pkg.sv`, `replay_buffer.sv`, `add_multi.sv`.

**Note IS_MVU / VVU coupling:** `mvu_vvu_axi.sv` also serves VVAU (`IS_MVU=0` forces the
`genINT8` branch always — VVU is DSP58-packed-only). Whatever you do to the split must
keep the VVU path working. VVU only ever uses the packed core, so a `_packed` wrapper
serves both MVU-packed and all VVU. See `fixtures/vvau/` for the VVU model.

---

## 3. The target structure

Replace the one shared wrapper + internal fork with a **core-agnostic base** + **two
thin per-core wrappers**:

```
mvu_vvu_axi_base.sv   (NEW)  — everything shared: AXI I/O, replay_buffer,
                               input unflatten, blkDsp's double-pump prep,
                               blkOutput. Exposes the prepared compute signals
                               (dsp_w, dsp_a, dsp_last, dsp_zero, pumped clk) as
                               an interface to a core, and takes dsp_vld/dsp_p back.
                               NO generate fork, NO NUM_LANES computation.
mvu_vvu_axi_softvec.sv (NEW) — base + `mvu` core only. No reference to the packed core.
mvu_vvu_axi_packed.sv  (NEW) — base + `mvu_vvu_8sx9_dsp58` core only. No ref to softvec.
```

Two viable factorings — pick whichever elaborates cleanly and rtlsim-matches:

- **(a) Base-as-module:** `mvu_vvu_axi_base` is a real module exposing the compute
  handoff as ports; each per-core wrapper instantiates base + its core and wires them.
  Cleanest separation, but the base↔core interface (the `dsp_*` bundle, which is
  parameterized by PE/SIMD/widths) must be expressed as ports — non-trivial with the
  packed vs structured port shapes.
- **(b) Base-as-include / parameterized top:** each per-core wrapper is a full
  `mvu_vvu_axi`-shaped module with the shared body inlined (via `` `include `` of a
  shared `.svh`, or copy) and only the core instantiation differing. Less elegant, but
  sidesteps the port-interface problem. FINN-style pragmatism.

Delete the `NUM_LANES`/`generate` fork entirely. **The core-selection decision now lives
100% in the Python model** — `mvau_dsp_packed.feasible` already computes NUM_LANES
correctly (see `fixtures/mvau/impl_dsp_packed.py`, audit F1 fix) and preference
(packed > softvec) is a ranking concern, not RTL. This is the whole point: selection is
data, not silicon.

---

## 4. Python/emit changes (small, once RTL is split)

- **`fixtures/mvau/dsp_common.py`** — `SHARED_SOURCES` currently `{mvu_pkg, mvu_vvu_axi,
  replay_buffer, add_multi}`. After split, the shared plumbing set changes: replace
  `mvu_vvu_axi.sv` with `mvu_vvu_axi_base.sv` (+ whatever `.svh` the factoring needs).
- **`fixtures/mvau/impl_dsp_softvec.py`** — `sources = SHARED_SOURCES + (mvu_vvu_axi_softvec.sv, mvu.sv)`.
- **`fixtures/mvau/impl_dsp_packed.py`** — `sources = SHARED_SOURCES + (mvu_vvu_axi_packed.sv, mvu_vvu_8sx9_dsp58.sv)`.
  Now the two bundles' `.sources` are **genuinely disjoint** on the core + core-wrapper
  (they still share the base plumbing — that IS shared, correctly).
- **`fixtures/mvau/emit_rtl.py`** — the emitted top wrapper (`mvu_vvu_axi_wrapper.v`
  template) currently instantiates `mvu_vvu_axi`. It must now instantiate the per-core
  wrapper named by the selected bundle. Two options: (i) a per-bundle wrapper template,
  or (ii) bind the instantiated module name from `point` (e.g. a new derived
  `rtl_top_module` = softvec/packed wrapper name). Prefer (ii) — keeps one template,
  the module name is data. Add that derived in `dsp_common.py`.
- **`fixtures/vvau/impl_rtl.py`** — VVU uses the packed wrapper; update its `.sources`
  to the `_packed` set. VVU's `sources` currently lists `mvu.sv` as a dead file
  (documented) — after split it should list the `_packed` wrapper + packed core only,
  and `mvu.sv` drops out entirely (a nice cleanup — the dead-file note goes away).

---

## 5. Validation (this is the real work — Vivado required)

Environment is READY: Vivado/Vitis 2025.2 at `$FINN_XILINX_PATH=/home/tkeller/Xilinx`,
FINN Docker image built. One-off container command pattern (word-splitting caveat —
pass a single script path, not an inline `bash -c` with spaces):

```
# from finn/ :
bash run-docker.sh bash src/finn/design_space/tests/<your_runner>.sh
# runner cds to $FINN_ROOT, runs python, tees output to a file in the mounted tree.
```
(See `src/finn/design_space/tests/run_diff.sh` / `run_elaborate.sh` for the working
pattern. They tee to `$FINN_ROOT/_diff_out.txt` / `_elab_out.txt` — gitignored,
host-visible via the bind mount. run-docker.sh rebuilds + reinstalls deps each call —
~2-4 min startup; budget for it.)

**Acceptance gates (all must pass):**
1. **Elaboration:** `xvlog -sv` + `xelab` on each per-core wrapper's OWN source set
   (base + one core + one wrapper) succeeds with NO "module not found." Extend
   `finn/src/finn/design_space/tests/elaborate_mvau_emit.py` to cover both softvec AND
   packed (it currently only does softvec and FAILS — that failure is the trigger for
   this task; it should pass after the split).
2. **rtlsim behavioural equivalence (the critical gate):** for a representative config
   per core, the split wrapper must produce BIT-IDENTICAL output to the original fused
   `mvu_vvu_axi.sv` for the same stimulus. Use FINN's xsi rtlsim (`finn/src/finn/xsi/`,
   `finn_xsi/xsi.so` present in container). Approach: drive the original `mvu_vvu_axi`
   (FORCE_BEHAVIORAL) and the new `_softvec`/`_packed` wrapper with the same random
   weight+activation stream, assert equal output streams. A softvec config (any DSP part)
   and a packed config (DSP58, w≤8, a≤9, lanes≤3) at minimum. Bit-exact or the split
   changed behaviour.
3. **FINN differential still green:** `diff_mvau_emit_vs_finn.py` must still pass
   (our emit params unchanged; only the instantiated module name + source list change).
   Note: FINN's own `matrixvectoractivation_rtl.py` still uses the OLD fused wrapper —
   so after the split our emit intentionally DIVERGES from FINN on the wrapper filename
   and instantiated module. Update the diff to compare the param BINDINGS (still equal)
   not the module name (now intentionally different), OR gate that one field out.
4. **Venv suite still green:** the 77 `design_space` tests
   (`PYTHONPATH="qonnx/src:finn/src" .kernel-venv/bin/python -m pytest
   finn/src/finn/design_space/tests/ -q`) — the `.sources` changes will touch the MVAU
   emit tests' expected static-file sets; update them.

---

## 6. Why split (A), not collapse (B) — the settled rationale

Recorded so it is not re-litigated. Two options were weighed:
- **A (this task): split the SV into two standalone components.**
- **B: accept them as two specializations of one FINN component, collapse the model to
  one pool member with the packed core as an internal Derived specialization.**

**A chosen because:**
1. **They ARE two designs** — two source files, two microarchitectures (soft-vec
   lane-slicing vs DSP58 INT8 cascade), two cost/precision envelopes. They **overlap**
   (both build on DSP58+INT8; the `else genSoftVec` branch proves softvec is valid
   there) — overlap + a real tradeoff = a genuine choice, not a sub-mode. The fork is a
   *packaging* accident (packed bolted onto the wrapper as an optimization — the source's
   own `@todo` admits it), orthogonal to the identity of the things selected.
2. **We already committed to two pool members, deliberately** (design-space-model.md
   §1.2.2: "model the wider menu; let predicates narrow it" — chosen precisely so a
   future device with more DSP options turns a forced branch into a real choice with
   zero model change). B would REVERSE that decision because the RTL is inconvenient; A
   makes the RTL true to the model we committed to.
3. **Kills the F1 bug class permanently** — F1 happened because our predicate replicated
   the fork's NUM_LANES math. Split wrappers have no fork to replicate.
4. **Makes the composability thesis literally true for MVU** — post-split, a 3rd DSP core
   is a 3rd standalone wrapper + 3rd pool member, additive, zero edits. Under B, a 3rd
   core means editing a fork/Derived condition — reintroducing the non-composability we
   are eliminating.

**The one fact that would have flipped to B** (and did NOT): if packed *strictly
dominates* softvec whenever eligible — i.e. no one would EVER pick softvec on a
packed-eligible config — the "choice" is vacuous and an honest collapse beats a fake
axis. Read: packed is *nearly* always preferred (3 MACs/DSP58) but not *vacuously* so
(resource-balance / pipeline-depth / SIMD-constrained corners can favour softvec). That
is exactly the "usually-decided menu" §1.2.2 was written for. If during this work you
find hard evidence softvec is NEVER worth choosing on a packed-eligible part, STOP and
raise it — that would reopen B.

---

## 7. Scope boundaries

**In scope:** split `mvu_vvu_axi.sv` → base + 2 per-core wrappers; delete the fork; wire
the Python `.sources`/emit to the new files; elaboration + rtlsim-equivalence validation;
keep VVU working; keep the 77 venv tests + the differential green.

**Out of scope:** the weight-DELIVERY composition arc (memstream/decoupled/dynload/mlo)
— that is a separate, larger effort (the `Derived`-returns-sub-Point compose mechanism).
This task is purely the compute-core wrapper split. HLS is untouched (no fork there).

**Risk / bail conditions:** if the base↔core port interface (factoring 3a) proves too
gnarly given the packed-vs-structured port shapes, fall back to the include/copy
factoring (3b) — FINN-pragmatic, still achieves disjoint elaboratable source sets. If
rtlsim equivalence FAILS, the split changed behaviour — do not paper over; the shared
`blkDsp` prep or `blkOutput` likely has a subtlety not cleanly separable, and that is a
finding worth surfacing before forcing the split.

## 8. Files

- EDIT (surgery) `finn/finn-rtllib/mvu/mvu_vvu_axi.sv` → factor into NEW
  `mvu_vvu_axi_base.sv` (+ maybe `.svh`), `mvu_vvu_axi_softvec.sv`,
  `mvu_vvu_axi_packed.sv`. Original may be retired or kept until VVU migrated.
- EDIT `fixtures/mvau/dsp_common.py` (SHARED_SOURCES + new `rtl_top_module` derived),
  `impl_dsp_softvec.py`, `impl_dsp_packed.py`, `emit_rtl.py` (instantiate per-core wrapper).
- EDIT `fixtures/vvau/impl_rtl.py` (packed wrapper sources; drop dead `mvu.sv`).
- EDIT `tests/test_mvau_emit.py`, `tests/test_vvau_resolve.py` (static-source expectations),
  `tests/diff_mvau_emit_vs_finn.py` (gate the now-divergent module name),
  `tests/elaborate_mvau_emit.py` (cover both cores; must pass).
- NEW rtlsim-equivalence test (split wrapper vs original fused, bit-exact) under
  `finn/src/finn/design_space/tests/`.

## 9. Starting context (memory + prior handoffs)

Read these for grounding: `kernel-final-design/design-space-model.md` §1.2.2 (pool-vs-
Derived, the split's theoretical basis) and §8; `kernel-final-design/mvau-design-space.md`;
`kernel-final-design/DECOMPOSE-IMPLEMENTATIONS-HANDOFF.md` (the original 2c deferral, §2c);
memory notes `impl-bundles`, `emit-phase`, `dev-environment`. The differential +
elaboration harnesses (`finn/src/finn/design_space/tests/{diff,elaborate}_mvau_emit*.py`
+ `src/finn/design_space/tests/run_{diff,elaborate}.sh`) are your working Docker/Vivado examples.
