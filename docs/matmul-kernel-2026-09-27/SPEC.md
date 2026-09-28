# Task spec: `MatMulKernel`, one design space for the dot-product units

Date: 2026-09-27. Status: **specified, not started.** It replaces Phase C's C1
(VVAU reuse) and C2 (fused thresholding) in
[`STATUS.md`](../kernel-status-2026-09-27/STATUS.md). It supersedes the
MVAU framing of the [robust MVAU spec](../robust-mvau-2026-09-26/SPEC.md),
whose remaining capabilities (R3, R6, R7, R8) now apply to this kernel.

## 1. Decisions this spec starts from

| # | Decision |
|---|---|
| M-D1 | The kernel is **`MatMulKernel`**: a matrix-multiply unit. MVAU's name, and the "A" (activation) in it, go |
| M-D2 | **Thresholding is cut out.** It stays its own kernel (already on streams since B3). Placing it after a matmul in one generated module is composition of adjacent kernels, and whether to do so is an operation-to-module mapping decision owned by the dataflow layer (D10, D11), not a flag of this kernel |
| M-D3 | **MVAU, VVAU and their subcases dissolve into one composed design space.** What distinguishes them is derived from the operation where the operation determines it, and is a Decision where it is a genuine choice |
| M-D4 | **The fused HLS designs are left behind for now**: HLS MVAU/VVAU with thresholds in the loop, binary/xnor, `resType=lut`. The kernel is RTL on FinnLib |
| M-D5 | Before building, **decide where to collapse and where to split**: which choices RTL modules make internally that should be design-space choices, and which kernel-level distinctions are really one thing (§4) |

Standing constraints: the engine stays generic (an engine change is a separate
proposal); a kernel owns one generated module and exposes its pin interface
(D11); key and ABI changes are accepted when recorded (D7); FinnLib
dependencies are pinned commits on a remote; the `CLAUDE.md` RTL rules apply.

## 2. What differs between MVAU and VVAU today

Both compute, per PE lane and beat, a dot product over SIMD elements, on the
same FinnLib `dotp_axi`.

| | Dense (MVAU: matmul, fully connected, conv as matmul) | Per-channel (VVAU: depthwise) |
|---|---|---|
| Operation | `Y[r, n] = Σ_k X[r, k] · W[n, k]` | `Y[r, c] = Σ_k X[r, c, k] · W[c, k]` |
| Activations across PE lanes | shared: one vector broadcast to PE outputs | distinct: each lane is its own channel |
| Activation reuse across output folds | yes: replay the activation vector per output fold | none: each output reads its own inputs |
| Frame marker for dotp | from the replay stage | needed without replay: `replay_buffer` with REP=1, or a marker generator |
| Activation beat | SIMD fields | PE × SIMD fields, channel fastest (`s·PE + p`) |
| Weight traversal | `tile(N, K, PE, SIMD)` | channel × window, PE channels × SIMD per beat |
| `dotp_axi` | `ACTIVATION_BROADCASTING=1` | `ACTIVATION_BROADCASTING=0`, DSP58 only, always the INT8 core |
| Baseline FINN | `MVAU` (HLS, RTL), nodeattrs `MW, MH, PE, SIMD, …` | `VVAU` (HLS, RTL on DSP58), nodeattrs `Channels, Kernel, Dim, PE, SIMD, …` |

The first two rows are properties of the operation, and every other row
follows from them. The hypothesis this spec tests: **the two units are one
design space whose sharing pattern is derived from the operation.**

## 3. The target design space (first cut; §4 and §5 revise it)

| Axis | Kind | Notes |
|---|---|---|
| Operation: operand shapes and the contraction (which axes are reduced, which are shared across outputs) | Params (facts) | How to describe it is Q1 |
| Operand dtypes; result dtype | Params; derived | As today: exact accumulator derived |
| Target DSP; target clock period | Params (facts) | Period sets segmentation (B2 revision) |
| Activation broadcast; replay needed; stream traversals | **derived** | From the contraction and the folding |
| PE, SIMD | Decisions | Divisor domains over the output and reduction extents |
| Weight delivery | Decision over nodes | external / cyclic; later writable (C4), multi-set (C5) |
| Weight stream transport | Decision over nodes | direct / FIFO (depth, `ram_style`) |
| Activation reuse realization | Decision over nodes, when reuse exists | `replay_buffer`; later `input_gen` (C3) |
| Frame marker source, when no reuse | derived node | `replay_buffer(REP=1)` or a marker generator (§4) |
| Compute pumping | Decision | As today |
| dotp core | Decision or derived | Q2, with D6 |
| Dense realization of a per-channel operation | Decision (candidate) | Block-diagonal weights on the dense path: wasteful in general, valid, sometimes attractive when channels are few (Q3) |

## 4. Collapse and split analysis (M-D5; the first deliverable)

For each item: where the choice lives today, and a proposed disposition
(**derived**, **Decision**, **pinned**, or **FinnLib change needed**). The
analysis confirms or overturns each proposal with evidence (RTL reading,
baseline FINN behaviour, resource or timing effect), and the human gate
decides.

| # | Item | Where today | Proposed |
|---|---|---|---|
| X1 | dotp core: INT8 DSP58 (`dotp_8sx9_dsp58`) vs soft-vector (`dotp`) | Inside `dotp_axi`, from widths, lane count and broadcasting (`dotp_axi.sv:254-289`; its own `@todo`) | Decision where both cores are valid (DSP58, narrow operands), derived elsewhere; needs the FinnLib wrapper split or a core-select parameter (D6) |
| X2 | Activation broadcasting | `dotp_axi` parameter, pinned to 1 by the kernel | derived from the contraction |
| X3 | Lane packing (`NUM_LANES`) and `NARROW_WEIGHTS` | Inside `dotp.sv`; `NARROW_WEIGHTS` pinned 0 | `NUM_LANES` derived (it follows from widths); `NARROW_WEIGHTS` derived from known weights (cyclic delivery: no most-negative value), else 0 |
| X4 | SEGMENTLEN | derived from the target period (B2 revision) | keep derived, or a Decision bounded by the period (more segments: more registers and latency, easier timing) |
| X5 | Replay stage vs frame-marker source | MVAU composite (replay); VVAU would need a marker source | one slot, derived presence: reuse needs replay; no reuse needs only markers |
| X6 | Weight memory style: `rom_style` (cyclic), FIFO `ram_style`, memstream `RAM_STYLE`, all offering `auto` | Kernel Decisions whose `auto` defers to synthesis or to RTL selection (`fifo.sv` picks shift/LUTRAM/BRAM/URAM by depth and width) | decide whether `auto` belongs in the space or the kernel resolves the style itself |
| X7 | Memory pumping (`PUMPED_MEMORY`) | `memstream_axi` parameter | a Decision on the memstream delivery candidate (C4) |
| X8 | Dense realization of per-channel | not offered | candidate Decision (Q3) |
| X9 | `FORCE_BEHAVIORAL` | pinned 0 | pinned (simulation only) |
| X10 | MVAU vs VVAU composites | two kernels in baseline FINN | one composite (M-D3) |

The analysis also lists anything found along the way that RTL combines and
the space should split, or that the space splits and should collapse.

## 5. Open questions (answer before building)

1. **Q1: How is the contraction described?** Options:
   - a pattern enum (`DENSE`, `PER_CHANNEL`);
   - operand shapes plus named shared and reduced axes (an einsum-like description);
   - traversals over the full operand tensors, from which sharing is derived.

   The last is the robust form, and it depends on the parked stream and dataflow
   revision (D10). A first cut may take the smallest description that derives
   everything in §3, designed so D10 can replace it.
2. **Q2: The dotp core (X1, D6).** A Decision needs FinnLib to expose the
   choice. Is the finn-rtllib MVU wrapper split (packed and soft-vector wrappers,
   bit-equivalent to the fused golden) the model to port, or a `CORE` parameter
   on `dotp_axi`?
3. **Q3: Is the dense realization of a per-channel operation offered?**
4. **Q4: Folding for the per-channel case.** PE divides channels and SIMD
   divides the window. Confirm FinnLib's constraints (e.g. DSP58-only VVU, SIMD
   limits of the INT8 core) and whether any baseline VVAU folding is lost.
5. **Q5: E-048.** The per-channel activation order is `s·PE + p` in FinnLib, not
   the hlslib `(SIMD-1-s)·PE + p`. It is declared as a traversal here; the SWG
   producer's order is re-derived when an SWG composite exists.
6. **Q6: Keys and names.** Accept renaming the MVAU decision keys
   (`implementation`, `weight_stream.transport`, `compute.compute_pumping`,
   `pe`, `simd`) where the composite changes, and the module name
   (`finn_mvau_*` to `finn_matmul_*`). Recorded as D7 changes.
7. **Q7: dotp's port checks.** B1's checks encode the dense relation. The
   per-channel relation (lane `p`'s activations, weight row and result column
   are one channel) is added in the same form, and both are recorded as the
   stopgap D10 replaces.

## 6. Scope

**In scope.** `MatMulKernel` covering the dense and per-channel contractions;
external and cyclic delivery; direct or FIFO weight transport; replay or marker
source; compute pumping; the §4 dispositions accepted at the gate. The
convenience adapter is replaced by one for the new kernel.

**Out of scope.**

- thresholding in the kernel (M-D2);
- fused HLS designs, binary/xnor, `lut` (M-D4);
- writable and multi-set weights, and the `input_gen` replay (C3, C4 and C5 add them to this kernel later);
- stream adapters (C6);
- `TH>1`, MMV, dynamic weights, `external_mem`/MLO;
- the SWG and conv composites;
- the stream and dataflow modeling revision (D10).

## 7. Evidence required

1. **Tests:** the space's derived structure for both contractions; each
   Decision and its refusals; port checks for both relations; attribution per
   stream.
2. **Numeric XSI:** the MVAU sweep ported to `MatMulKernel` (28 cases, dense),
   plus a per-channel sweep (DSP58: several PE/SIMD foldings, external and
   cyclic, free and stalled output, pumped and not).
3. **Fingerprints:** the dense configurations' structures compared with the
   current MVAU. Expected changes (names, keys, module name) are recorded;
   anything else is explained.
4. **Coverage:** `COVERAGE.md` gains the per-channel (VVAU) rows next to the
   MVAU rows, mapped to this space.
5. **Both gates green**, XSim executed.

## 8. Increments (draft)

| # | Content | Gate |
|---|---|---|
| M0 | The §4 analysis and answers to §5 | **Human gate**: dispositions and Q1–Q7 decided |
| M1 | `MVAU` becomes `MatMulKernel` for the dense case: rename, adapter, keys and module name; thresholding references removed; numeric sweep and fingerprints | recorded rename; 28/28 |
| M2 | dotp's per-channel mode: broadcasting derived, PE×SIMD activation lanes, the per-channel port relation, DSP58 refusal otherwise | tests |
| M3 | One composite for both contractions: derived broadcast, derived replay or marker source, per-channel weight traversal for delivery; per-channel numeric harness | per-channel sweep green |
| M4 | The Decisions accepted in M0 (e.g. the dotp core with its FinnLib change, `NARROW_WEIGHTS`, the dense realization) | per decision |
