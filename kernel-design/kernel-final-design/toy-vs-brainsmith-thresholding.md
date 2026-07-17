# Toy example vs. real Brainsmith Thresholding — an honest comparison

*Compares `example-thresholding.py` (the toy illustrating the converged design)
against Brainsmith's actual `kernels/thresholding/{thresholding,thresholding_hls,
thresholding_rtl}.py` (365 + 750 + 580 = 1695 LOC). Two questions: (A) what does
the toy get **wrong or miss** as a model of reality; (B) which is the **better
design**, judged on the real code, not the toy's own marketing.*

Verified by full reads of all three Brainsmith files (not grep).

---

## Part A — Where the toy is wrong or incomplete

### A1. The toy FABRICATED its headline feature difference (the important one)

The toy's centerpiece is "HLS admits 1–32-bit ints, **RTL caps at 8-bit** — a real
envelope difference." **This is invented. It does not exist in the code.**

- Both backends inherit the schema's single `DatatypeInteger(("input","output"))`
  constraint (`thresholding.py:82`). Neither `@backend` class narrows dtypes.
- Worse for the toy's claim: the RTL backend actually handles **floating-point
  input** — `prepare_codegen_rtl_values` emits `$FPARG$` = "Is input datatype
  non-integer? (assume this means floating-point)" (`thresholding_rtl.py`
  ~`is_integer()` check). So if anything RTL is *broader*, not narrower.
- The toy's `fold_caps = "PE<=64"` hard cap on RTL is **also invented** — there is
  no such cap; PE need only divide NumChannels (`DimensionDivisible`,
  `thresholding.py:85`).

**Consequence:** the toy's showcase of "union feasibility picks HLS when RTL's
envelope excludes the node" never fires for real thresholding, because the two
envelopes are identical on dtypes. Thresholding is a **weak example** for the
design's headline (envelope-as-data / superset-killing) feature — I should not
have used dtype as the differentiator here. (It *is* a strong example for a
different feature — see B2.)

### A2. The toy OMITS the feature that actually dominates the real code: `mem_mode`

The single biggest thing in real Thresholding is the **memory-strategy axis**:
`mem_mode ∈ {internal_embedded, internal_decoupled}` (`thresholding_hls.py:73`).
- `internal_embedded`: thresholds baked into a `thresh.h` header / `.dat` ROM.
- `internal_decoupled`: thresholds **streamed** over an AXI-Stream via a generated
  **memstream** submodule, optionally **runtime-writable** over AXI-Lite.

This one knob branches **~16 methods** in the HLS backend alone: `get_instream_width`,
`code_generation_ipgen`, `get_ap_int_max_w`, `make_weight_file`, `generate_params`,
`execute_node`, `global_includes`, `defines`, `read_npy_data`, `strm_decl`,
`docompute`, `blackboxfunction`, `pragmas`, `code_generation_ipi`,
`get_verilog_top_module_intf_names`, `derive_characteristic_fxns`. The toy models
**none** of it — no `weights`/`memory` parts, no streaming, no AXI-Lite.

### A3. The toy OMITS all the real HW parameter marshaling

Neither identity nor envelope — genuine realization logic the toy hand-waved as
`params=["thresholds"]`:
- `get_hw_compatible_threshold_tensor` (`thresholding.py`): PE-interleaving,
  reshape to `(1, PE, TMEM, n_steps)`, unsigned-threshold assertions.
- `make_weight_file` — **four** weight-file modes in HLS (`hls_header`,
  `decoupled_npy`, `decoupled_verilog_dat`, `decoupled_runtime`), **two** in RTL,
  each packing to hex `.dat`/headers differently.
- **Narrow-range quantization** (`thresholding_rtl.py`): the `expected_thresholds
  != n_thres_steps` dance — insert a dummy threshold, bump bias, widen the weight
  dtype. Subtle, correctness-critical, entirely absent from the toy.
- **BIPOLAR** handling (store bipolar as binary, `2*y-1` on the way out).

### A4. The toy OMITS inference (`can_infer_from` / `infer_from`)

Real kernels declare how a raw ONNX node becomes this kernel: Thresholding
pattern-matches a `MultiThreshold` node, validates `out_scale==1.0`, extracts
`act_val`, emits the HW node (`thresholding.py:can_infer_from/infer_from`). The
toy starts from an already-specialized node — it never shows lowering-from-ONNX,
which is half of what a real kernel is for. (Our design has a home for this — a
`can_infer_from`/`infer_from` on the schema — but the toy didn't show it.)

### A5. The toy OMITS resource estimation

`bram_estimation`/`lut_estimation`/`uram_estimation` are real, per-backend, and
non-trivial (`get_pe_mem_geometries`, `get_memutil_alternatives`). The toy has
`cost` as a one-line stub. In the real design these feed the (deferred) cost seam
— so the *shape* is right, but the toy shows none of the substance.

### A6. What the toy got RIGHT

- **Schema-derived contract** — the toy's `_point()` mirrors Brainsmith's "Arete
  principle" (`ki = self.design_point; num_channels = ki.inputs["input"].
  tensor_shape[-1]`) exactly. This is the genuinely-shared good idea.
- **RTL's distinct knobs** — the toy guessed `depth_trigger_uram/bram`,
  `deep_pipeline` for RTL; those are **real** (`thresholding_rtl.py:62-69`). The
  *shape* of "backends expose different realization knobs" is correct even though
  the dtype-cap example was not.
- **Node representation / round-trip** — Brainsmith's real node uses
  `domain="brainsmith.kernels"`, op_type `"Thresholding"`, and reconstructs via
  `getCustomOp` — exactly the by-name behavior the toy's §7 save/rehydrate shows.

---

## Part B — Which is the better design

Judged on the **real** code, the design still wins on structure — but for
*different reasons* than the toy advertised, and with real humility about what
Brainsmith already does well.

### B1. Brainsmith's genuine strengths (the toy/design must not pretend otherwise)
- **It is real and complete** — it handles mem_mode, streaming, runtime-writable
  weights, narrow-range, bipolar, four weight-file formats, resource models, and
  ONNX inference. The toy handles none of these.
- **Schema-driven derivation works** — the "deleted 116+ lines of redundant code…
  removed shape/stream methods that duplicate KernelOp base" note
  (`thresholding.py` header) is the design's own thesis, *shipping*. Our design
  keeps this wholesale; we did not improve on it.
- **The `@backend` + registry split is clean** at the language level.

### B2. Where the converged design is genuinely better — and it's NOT dtypes

The real win on *this* op is **the memory-strategy axis (A2)**, not the dtype
envelope. `mem_mode` branching ~16 methods is the textbook symptom the design
targets: it is an **orthogonal axis conflated into the op**. In the converged
design, `internal_embedded` vs `internal_decoupled` is a **`WeightDelivery` /
`MemoryStrategy` composed part**, so:
- the 16 `if mem_mode == …` branches collapse to *which part is bound*;
- the streamer stitch (below) lives in the part's `emit_cell`, not smeared across
  the op;
- adding a third delivery mode is a new part, not edits to 16 methods.

This is the design's real value proposition on thresholding, and it is strong —
I just pointed the toy at the wrong axis (dtypes) to dramatize it.

### B3. The `execute_node` diamond is real and the design fixes it
`Thresholding(KernelOp → HWCustomOp → CustomOp)`, and **both** backends carry a
near-identical ~50-line `execute_node` (npy save + rtlsim plumbing), with RTL's
cppsim path calling `Thresholding.execute_node(self, …)` explicitly
(`thresholding_rtl.py`). That is the diamond, live: reference semantics and
backend execution fused under one method name, duplicated per backend. The
design's **reference/runner split** (`evaluate` vs `runner(mode)`) dedupes this
into one shared runner + one reference — a concrete, real improvement.

### B4. The stitch god-method is real and inherited — design points at it, but does NOT yet beat it
`Thresholding_hls.code_generation_ipi` (`thresholding_hls.py:565-695`) is a
~130-line block-design god-method: hardcoded pin names, `os.listdir` to find the
memstream wrapper, `os.environ["FINN_ROOT"]`, `connect_bd_intf_net` string
surgery, a dead `ap_clk2x`→`ap_clk` workaround. This is exactly the baseline leak
Brainsmith inherited by keeping `HLSBackend`. The design's role-tagged Cell/Region
stitch *aims* to dissolve this — **but the toy does not implement it**, and the
design itself flags the stitch layer as its least-validated piece (Risk 3). So on
this axis the design shows *direction*, not a demonstrated win. Honest score: TBD,
not victory.

### B5. Honest verdict
- **As a model of reality, the toy is thin and partly fabricated** (A1/A2/A3) —
  it should be corrected: drop the invented dtype/PE caps, and if it keeps one
  differentiator, make it `mem_mode`-as-composed-part (the real one).
- **As a design, the converged approach still wins on the real code**, but the
  winning margin comes from **axis separation** (mem_mode → part, B2) and the
  **reference/runner split** (B3), *not* the dtype-envelope story the toy led
  with. The envelope/superset feature is real in general but **thresholding is a
  poor showcase** for it (shared integer dtype, nothing to compose).
- **Brainsmith wins on realness and completeness**, and its schema-derivation is
  the part the design adopts unchanged.
- Net: the design is the better *target*; Brainsmith is the better *artifact
  today*; and this exercise usefully falsified the toy's headline while
  confirming the design's structural claims against 1695 lines of real code.

---

## What this implies for the design doc

1. **Fix the toy** — remove the fabricated RTL dtype/PE caps; either use a real
   dtype-differentiated op (e.g. an op with an int-only RTL and an fp HLS) for the
   envelope showcase, or re-anchor the thresholding toy on `mem_mode`-as-part.
2. **Add a `mem_mode`/weight-delivery walkthrough** — it is the strongest concrete
   evidence for the composed-parts thesis, and the toy currently hides it.
3. **State honestly in §9/conformance** that the envelope-superset win needs an op
   whose backends genuinely differ in dtype support; thresholding does not.
4. **The stitch-layer claim stays aspirational** (already Risk 3) — this
   comparison confirms the god-method it targets is real, but the design has not
   yet beaten it in code.
