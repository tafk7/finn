# Roster reconciliation: kernel-roster-map (2026-09-25) against the landed Space model

This file reconciles every claim of
`scratchpad/open/kernel-roster-map/` against `6aa0383cf`: claims V01–V45,
Decisions 1–3 and open questions 1–10. That roster was written against kernels
`6ff394315` and FinnLib `b17eae6a`. Labels and abbreviations are as in
[`COVERAGE.md`](COVERAGE.md).

**Status codes:**

| Code | Meaning |
|---|---|
| **holds** | Still true, and still relevant to the kernel layer |
| **holds (baseline)** | A fact about baseline FINN or FinnLib that is still true; the kernel layer does not depend on it |
| **resolved** | The redesign or a later commit removed the problem |
| **changed** | Still relevant, but the facts have moved |
| **obsolete** | No longer relevant to kernel work |

## 1. Claims V01–V45

| Id | Status | One line of evidence |
|---|---|---|
| V01 | changed | Baseline's memstream allowlist is unchanged (`fpd/hwcustomop.py:313-314`). On the kernel side, `CyclicDelivery` is now parameterized by any `Traversal` (`k/delivery.py:48-58`), so one family can serve MVAU, VVAU, THR and EW. In `src` only MVAU uses it; eltwise uses it only in a hand-built test (`tests/kernels/test_stream_contract.py:305-345`). [V] |
| V02 | holds (baseline) | `fpd/hwcustomop.py:313-334` sets `SETS=mlo_max_iter or 1` and `DEPTH=calc_wmem·TH`, and blanks the init file for URAM off Versal. The kernel has no SETS; that is R7. [V] |
| V03 | holds (baseline) | fetch is MVAU-only; out of scope (deferred). [C] |
| V04 | changed | The kernel's MVAU port set is `in0_V`, `out0_V`, plus `in1_V` only for external delivery. Cyclic delivery drops `in1_V` through placement-by-visibility (`k/mvau.py:276-278`, `k/streams.py:253-256`). There is no `s_axilite`/`axi_mm`. [V] |
| V05 | obsolete | HLS `inputBuf` replay; the kernel uses RTL `replay_buffer`. [V] |
| V06 | holds | The weight stream is `tile(MH,MW,PE,SIMD).repeated(repetitions)` (`k/mvau.py:227-237`), which is `n·NF·SF` beats. [V] |
| V07 | holds (baseline) | dynload is deferred. [C] |
| V08 | holds (baseline) | fetch_weights is deferred. [C] |
| V09 | holds (baseline) | It still argues that MLO changes only delivery. [C] |
| V10 | holds | Thresholding's per-beat set index is still unmodeled: the `s_axis_set` pins are always present and have no driver (`k/thresholding.py:281-286`; probe P4). [V] |
| V11 | changed | "The period must divide the pass" is now a generic contract check: `_presented` refuses when it does not (`k/physical/contract.py:125-133`, `:176-183`). [V] |
| V12 | changed | The kernel `EltwiseKernel` does not refuse int/int or in×in (`k/eltwise.py:106-127`), so it diverges from baseline RTL. [V] |
| V13 | holds, but obsolete for the kernel path | The census still says `beat_count = NF` (`scratchpad/proofs/baseline-finn-design-census/families/thresholding.md:236`). No FinnLib thresholding consumes a threshold stream, and the kernel keeps an internal table, so V13 does not block R5. [V] |
| V14 | not rechecked | It is not load-bearing here. |
| V15 | holds (baseline) | `fpd/vectorvectoractivation.py:74-79` has three values. [V] |
| V16 | obsolete | Lookup is out of scope. |
| V17 | obsolete | Pad1D is out of scope. |
| V18 | holds | The kernel wraps FinnLib `rtl/input_gen.sv` (`k/input_generator.py`). [V] |
| V19 | holds | The MVAU composite externalizes exactly this replay (`k/mvau.py:253-258`). [V] |
| V20 | holds, with a caveat | The `dotp_axi` contract is unchanged at `b17eae6a`. The output-queue fix is **not** on `upstream/dev` (`probes/finnlib-remote-status.txt`). [V] |
| V21 | holds | `hls/memstream.hpp` is free-running and is wrapped by `MemStreamHlsKernel`. [V] |
| V22 | obsolete | A classification nit about `mem2stream`. |
| V23 | holds (baseline) | HLS dotp is `flit_t`. The kernel layer has no HLS compute. [V] |
| V24 | holds | Thresholding's `cfg`/`iset` pins exist; the kernel emits them but they are unmodeled. [V] |
| V25 | obsolete | Pad1D. |
| V26 | changed | `upstream/dev` has reorganized FinnLib into `rtl/{arith,infra,linalg,nonlin,shape}` and merged `queue` into `fifo` (commit `21a3c45`), so the roster's flat paths are stale. [V] |
| V27 | resolved | External delivery has two instances (`tests/kernels/test_mvau_assembly.py:56`); the REVIEW text was corrected in `31dcee3a1`. [V] |
| V28 | changed | `rom_style` is now keyed `implementation.cyclic.rom_style` (`k/delivery.py:58`); UltraRAM is still excluded (`k/streaming.py:233-235`). [V] |
| V29 | resolved | Marker *meaning* now exists: `StreamSpec.markers` / `StreamContract.markers` hold `Every(k)` rules, which are checked and wired (`k/streams.py:86`, `k/physical/contract.py:64`, `:144-151`). Multi-bit markers are still excluded (`:81`). [V] |
| V30 | changed | `b17eae6a` is now on `origin/kernel-contract-refinement-20260925` (the personal fork) but not on upstream. **New:** `fetch-repos.sh:53` pins a different commit, `dfeafac8`, with an incompatible layout (AUDIT §5, D1). [V] |
| V31 | holds | `resources/` has no dotp copy. [V] |
| V32 | holds | Upstream still has only the HLS memstream (`hls/infra/memstream.hpp`). [V] |
| V33 | holds | `PeriodicLast` is used only in `finn.parked`; live code uses `Every`. [V] |
| V34 | holds (stale corpus) | — |
| V35 | holds | `conv2d.sv` folds NF into the `input_gen` nest. This decides SPEC §7 Q2 (AUDIT §4). [C, relied on] |
| V36 | holds (baseline) | `deconv.hpp` embeds its weights. [C] |
| V37 | holds | The MVAU kernel boundaries carry no TLAST (`k/mvau.py:214-244`: no markers on the boundary specs). [V] |
| V38 | changed | The kernel `EltwiseKernel` uses native unpadded ports (`k/eltwise.py:85-89`). Inside kernels, child padding cannot be dropped at all (`k/physical/contract.py:153-159`), so the V38 question moves to that rule (AUDIT §5). [V] |
| V39 | holds (baseline) | — [C] |
| V40 | holds | The kernel uses the `THRESHOLDS` parameter (`k/thresholding.py:237`). [V] |
| V41 | changed | `vpc` was refined upstream: optional clean-zero padding and an identity path (`upstream/dev` log `8b08607`, `9167c2d`). [V] |
| V42 | holds | — [V by listing] |
| V43 | obsolete | topk is out of scope. |
| V44 | holds | The kernel analogue is the `BufferedStream.transport` slot on `weight_stream` (`k/mvau.py:250`). [V] |
| V45 | holds | FinnLib HEAD is `b17eae6a`, with only the two untracked `input_gen_lift*` files (`probes/finnlib-remote-status.txt`). [V] |

## 2. Decisions 1–3

| Decision | Status | Evidence and reading |
|---|---|---|
| **1. Delivery placement** (A inside, B beside, C by visibility) | **C is realized for the external/cyclic pair.** | `implementation: CyclicDelivery \| None = Decision(...)` (`k/mvau.py:276-278`). A stream with a user on one side only becomes the composite's boundary (`k/streams.py:253-256`), so "external" is simply "nothing placed". "Beside" is the same `CyclicDelivery` node declared by a parent (`tests/kernels/test_declared_streams.py:54-80`). **Still open** for the cases that need non-stream interfaces: writable (AXI-Lite), multi-set (a set-index sideband), dynamic and fetch. The model has no place for a control bus or a sideband yet (AUDIT §5, D4). [V] |
| **2. Robust MVAU scope** | holds as a scope, revised in order | AUDIT §9 revises SPEC §3/§6. [V] |
| **3. Stream abstraction** (S1 minimal contract) | **S1 adopted and realized.** | `StreamSpec` carries element, form, repetition and markers (`k/streams.py:79-90`). `StreamContract` + `compatibility` + `Composition.connect` wire every data, padding, handshake and marker wire (`k/physical/contract.py`, `composition.py:131-190`). **Not yet on the stream:** clock-domain identity (still pin *names*, `k/streams.py:97`, `:446-463`) and multi-bit `Level(d)` markers. Reset polarity lives on the ABI signal and inversion is derived (`composition.py:73-86`). [V] |

## 3. Open questions 1–10

| # | Question | Status now |
|---|---|---|
| 1 | Push or merge `b17eae6a`? | **changed.** It is pushed to the personal fork only. `fetch-repos.sh` pins `dfeafac8`, which has a different layout and no dotp fix. Needs a human decision (AUDIT §9). [V] |
| 2 | Inside vs beside | **resolved by the model.** Both are expressible and chosen by case: the Decision over nodes with a `None` boundary (`k/mvau.py:276-278`). [V] |
| 3 | Thresholding census contract (V13) | **open in the census, irrelevant to the kernel path** (V13 above). [V] |
| 4 | REVIEW errors (V27, V29) | **resolved.** |
| 5 | Replay location | **still a composition decision**, now expressible as a Decision over nodes. AUDIT §4 Q2 recommends it. [V/I] |
| 6 | Owner of the image function | **resolved.** `pack(form, values, bits)` in `k/physical/forms.py:364-382`, used by `CyclicDelivery.image` (`k/delivery.py:61-73`). [V] |
| 7 | Padded memstream → unpadded eltwise `in1_V` | **moved.** Inside kernels, a child's padding cannot be dropped (`k/physical/contract.py:153-159`, probe P3). Native cyclic → native eltwise is exact, so the question does not arise there; it arises for `dotp → thresholding` (R5). [V] |
| 8 | Valid-only cores | **open.** Only ready/valid and AXIS transports exist (`k/physical/stream.py`). [V] |
| 9 | SWG→VVAU lane order (E-048) | **open, now checkable.** FinnLib VVU reads field `simd*PE + pe`, with no SIMD reversal (`FL/rtl/dotp_axi.sv:109-120`). `classify` can express a lane permutation and `connect` wires it for free, **once dotp declares its own form** (probe P5 shows it does not). [V] |
| 10 | Stale corpus | **holds** (V34, V31, V22). |
