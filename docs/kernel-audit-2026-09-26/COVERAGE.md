# Coverage: baseline FINN custom ops against the kernel layer

Date: 2026-09-26. This file holds the coverage table that the robust MVAU SPEC
§5.6 asks for, plus shorter tables for VVAU, Thresholding, Elementwise, SWG,
FIFO and DWC. [`AUDIT.md`](AUDIT.md) holds the conclusions.

**Revisions:**

- Kernels: `6aa0383cf`, on `feature/kernel-package-extraction`.
- Baseline custom ops: `src/finn/custom_op/fpgadataflow/`, in the same checkout.
- FinnLib: `b17eae6a`, the working clone behind `deps/finnlib`.

**Path abbreviations:**

| Prefix | Path |
|---|---|
| `fpd/` | `src/finn/custom_op/fpgadataflow/` |
| `k/` | `src/finn/kernels/` |
| `FL/` | the pinned FinnLib |

**Labels:**

| Label | Meaning |
|---|---|
| **V** | Verified by reading the code or running it. For the non-MVAU tables this means read by a delegated reader and spot-checked here. |
| **I** | Inferred. |

**Mapping kinds.** Each row maps a baseline attribute or variant to one of:

- a **Param** (a fact the caller supplies);
- a **Decision** (a choice);
- a **node** (a kernel placed in the Space);
- a **Decision over nodes**;
- **gap**, with the reason.

The "FinnLib" column says whether FinnLib can realize the row at all:

| Value | Meaning |
|---|---|
| RTL | FinnLib has an RTL part |
| HLS only | Only an HLS part exists. The kernel layer cannot netlist HLS parts yet (§3 of AUDIT). |
| none | No FinnLib realization |
| kernel-local | A kernel resource, not FinnLib |

## 1. MVAU (MVAU_hls, MVAU_rtl)

The baseline attributes are at `fpd/matrixvectoractivation.py:61-131`, `fpd/rtl/matrixvectoractivation_rtl.py:50-57` and `fpd/hls/matrixvectoractivation_hls.py:52-57`. RTL eligibility is decided at `transformation/fpgadataflow/specialize_layers.py:239-283`.

| Baseline attribute or variant | Space coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| `MW`, `MH` | `MVAU.matrix_width`, `matrix_height`: Params (`k/mvau.py:173-174`) | — | V |
| `numInputVectors` (a list) | `MVAU.repetitions`: a Param holding the flattened product (`k/mvau.py:172`). The per-axis shape is lost, which is fine because nothing downstream reads it. | — | V |
| `PE`, `SIMD` | `MVAU.pe`, `MVAU.simd`: Decisions whose domains are the divisors of MH and MW (`k/mvau.py:180-181`) | RTL | V |
| `inputDataType`, `weightDataType` | `activation_dtype`, `weights_dtype`: Params (`k/mvau.py:175-176`). The admission policy is dotp's: INT or UINT activations of at least 2 bits, **signed** weights of at least 2 bits (`k/dotp.py:97-98`, `:133-157`) | RTL | V |
| `outputDataType`, `accDataType` | `MVAU.result_type`: derived as the exact full-range dot-product width (`k/mvau.py:94-104`, `:183-188`). It **cannot be supplied**, so it is never narrowed by weight values (baseline minimizes the accumulator from the actual weights). | RTL | V |
| `mem_mode=external` | `implementation="external"`: the Decision over nodes places nothing, so `weight_stream` becomes the `in1_V` boundary (`k/mvau.py:276-278`, `k/streams.py:253-256`) | RTL | V |
| `mem_mode=internal_decoupled` (read-only) | `implementation="cyclic"`: a `CyclicDelivery` node (`k/mvau.py:273-275`) with the image as `INIT_DATA` | kernel-local `cyclic_stream.sv`; FinnLib has no RTL memstream (V32) | V |
| `mem_mode=internal_embedded` | gap: HLS-only in baseline (weights as C++ initializers). Functionally `cyclic` covers it. | HLS only (`FL/hls/dotp.hpp` + a const array, as in `deconv.hpp`) | V/I |
| `mem_mode=dynamic` | gap, deferred: there is no `dynamic_load`. | none | V |
| `mem_mode=external_mem` (+ `address_offset`, `base_address`, `in_idx0_V`) | gap, deferred: there is no fetch/cdma. | none | V |
| `ram_style` auto/block/distributed | `implementation.cyclic.rom_style` Decision (`k/delivery.py:58`, `k/streaming.py:214`) | kernel-local | V |
| `ram_style=ultra` | gap: URAM cannot be initialized on every device (`k/streaming.py:233-235`). Baseline requires `runtime_writeable_weights` for it off Versal (V02). It belongs with R6. | none in RTL | V |
| `runtime_writeable_weights` | gap: R6. `MemStreamHlsKernel` has an AXI-Lite array (`k/memstream_hls.py:73`), but it produces `HlsSourceRequirements`, which `netlist` cannot place. | HLS only (`FL/hls/memstream.hpp`) | V |
| `pumpedMemory` | gap: no pumped memory source exists, and there is no clock model to express one (R1). | none | V |
| `mlo_max_iter` (SETS of memstream, or a per-iteration index) | gap: R7. No multi-set source exists and there is no sideband model. | none | V |
| `noActivation=0` with a threshold input (fused activation) | gap: R5. HLS-only in baseline; baseline RTL requires `noActivation=1` (`specialize_layers.py:250-253`). | RTL (`thresholding_axi`), composed | V |
| `ActVal` | gap now; it would become `ThresholdingAxiKernel.bias` under R5 (`k/thresholding.py:78`). | RTL | V |
| `ram_style_thresholds` | gap now; under R5 it would become thresholding's `depth_trigger_bram`/`depth_trigger_uram` Params (`k/thresholding.py:99-100`). This is not a like-for-like style. | RTL | V |
| `binaryXnorMode`, BIPOLAR/BINARY operands | gap: the dotp scalars need at least 2 bits (`k/dotp.py:97-98`). Baseline is HLS-only here too (`specialize_layers.py:250-253`). | HLS only (`FL/hls/dotp.hpp` is type-generic) | V/I |
| Unsigned weights | gap: dotp needs signed weights (`k/dotp.py:98`, `:152`). Baseline RTL refuses them as well (`specialize_layers.py:255-258`). | HLS only | V |
| `resType` auto/dsp | `target_dsp`: a Param (`k/mvau.py:177`, `k/target.py`). Only DSP is covered. | RTL | V |
| `resType=lut` | gap: HLS-only in baseline (`fpd/rtl/matrixvectoractivation_rtl.py:286-289`). | HLS only | V |
| `pumpedCompute` (RTL) | `compute.compute_pumping` Decision (`k/dotp.py:92`), with pumping constraints at `k/dotp.py:176-182` | RTL | V |
| RTL `SEGMENTLEN` (baseline derives it from the clock period, `fpd/rtl/matrixvectoractivation_rtl.py:260-281`) | `segment_length` Param supplied by the caller (`k/mvau.py:178`). The target clock is not a fact of the Space. | RTL | V |
| RTL `VERSION` | derived from `target_dsp` (`k/dotp.py:74`, `:232`) | RTL | V |
| RTL `SIGNED_ACTIVATIONS` | derived from the activation dtype (`k/dotp.py:228`) | RTL | V |
| RTL `NARROW_WEIGHTS` (baseline derives it from the weight values) | pinned to 0 (`k/dotp.py:229`). The kernel admits full-range weights on DSP48E1 (baseline refuses them, `specialize_layers.py:271-273`); this passed the landing's XSim sweep. | RTL | V |
| RTL `IS_MVU`, `ACTIVATION_BROADCASTING` | pinned to broadcasting (`k/dotp.py:233`). The VVU form is gap R9. | RTL (VVU needs DSP58, `FL/rtl/dotp_axi.sv:78-82`) | V |
| RTL internal replay (`mvu_vvu_axi.sv` `replay_buffer(LEN=SF, REP=NF)`, V19) | the `MVAU.replay` node (`ReplayBuffer`, `k/mvau.py:253-258`) | RTL (`FL/rtl/replay_buffer.sv`, not upstream; see AUDIT §5) | V |
| dotp core (packed INT8 vs soft-vector) | not a coordinate: `dotp_axi` picks the core with an internal `generate` (`FL/rtl/dotp_axi.sv:263-289`), and the kernel ships both cores (`k/dotp.py:266-290`). R4 needs a FinnLib split. | RTL (split only in FINN `finn-rtllib`, AUDIT §4 Q4) | V |
| `TH>1` (tiled MVU) | gap, deferred (C4) | none | V |
| MMV / OUT_TILED | gap, deferred | none | V |
| `inFIFODepths` / FIFO on the weight input | `weight_stream.transport` Decision over nodes `direct`\|`fifo`, plus `...fifo.buffer.depth` and `...ram_style` (`k/streams.py:130-152`, `:304-308`) | RTL (`FL/rtl/fifo.sv`) | V |
| `inFIFODepths` on `in0`, `outFIFODepths` | gap: out of scope. Boundary FIFOs are a graph concern. | — | V |
| HLS backend as a whole (`preferred_impl_style`) | gap: there is no HLS compute kernel, and `HlsSourceRequirements` has no pin ABI for `netlist` (`k/memstream_hls.py:63-105`). With both backends it would be a Decision over compute nodes. | HLS only | V |
| Exec, rtlsim, IP-path and estimate attributes (`fpd/hwcustomop.py:52-101`) | n/a: adapter or tool concerns | — | V |
| Output TLAST | none, which matches baseline (V37). The `results` spec has no markers (`k/mvau.py:239-244`). | — | V |

**Definitively deferred for MVAU.** No FinnLib realization exists in any form
(SPEC §7 Q5):

- `TH>1`;
- MMV / OUT_TILED;
- `dynamic`;
- `external_mem` with its fetch, set index and address offset;
- `pumpedMemory`;
- a multi-set memstream (`SETS`).

**RTL writable memstream.** FinnLib has only the HLS `memstream.hpp`. It needs
an RTL part, which is a human decision (AUDIT §9).

**Available in FinnLib as HLS only, so they wait for an HLS netlisting path:**

- `internal_embedded`;
- `resType=lut`;
- binary, bipolar and xnor operands;
- unsigned weights;
- fused thresholds in HLS.

## 2. VVAU (VVAU_hls, VVAU_rtl)

The attributes are at `fpd/vectorvectoractivation.py:47-105`.

| Baseline attribute or variant | Kernel coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| The family as a whole | gap: no VVAU composite. dotp is pinned to broadcasting (`k/dotp.py:233`) and its activation port is SIMD lanes wide (`k/dotp.py:105`). | RTL: `dotp_axi` with `ACTIVATION_BROADCASTING=0`, DSP58 only; `conv2d_dw.sv` | V |
| `PE`, `SIMD`, `Dim`, `Channels`, `Kernel` | gap. dotp `pe`/`simd` are the nearest coordinates (`k/dotp.py:88-89`). | RTL | V |
| `mem_mode` embedded / decoupled / external | gap. The MVAU `implementation` pattern (`k/mvau.py:276-278`) is the reuse target. | kernel-local cyclic ROM | V |
| `runtime_writeable_weights`, `ram_style` | as for MVAU: gap / `rom_style` | as MVAU | V |
| `noActivation`, `ActVal`, thresholds | gap (R5). Baseline RTL VVU requires `noActivation=1` (`specialize_layers.py:291`). | RTL (composed) | V |
| `binaryXnorMode`, `resType=lut` | gap (HLS only) | HLS only | V |
| RTL `SEGMENTLEN`, `NARROW_WEIGHTS`, `VERSION` | as for MVAU | RTL | V |
| Reduction marker (baseline `replay_buffer` with REP=1, V19) | `ReplayBuffer(replay_count=1)` still produces `olast` (`FL/rtl/replay_buffer.sv:107-113`), so the MVAU `replay` node already expresses it. | RTL | V |
| SWG→VVAU lane order (E-048) | gap. FinnLib's VVU input order is field `simd*PE + pe`, with no SIMD reversal (`FL/rtl/dotp_axi.sv:109-120`). This must be declared as the port's form (AUDIT §6). | RTL | V |

## 2a. After Phase C (2026-09-28): MVAU and VVAU as one `MatMulKernel`

Sections 1 and 2 are the audit as of 2026-09-26. Phase C dissolved MVAU and
VVAU into `MatMulKernel` (`k/matmul.py`; record:
`../matmul-kernel-2026-09-27/RECORD.md`). The rows below are the attributes
whose mapping changed; the rest of sections 1 and 2 stand, with `MVAU.x`
read as `MatMulKernel.x` and `matrix_width`/`matrix_height`/`repetitions`
as `reduction`/`outputs`/`rows`.

| Baseline attribute or variant | MatMulKernel coordinate | FinnLib |
|---|---|---|
| MVAU vs VVAU (`IS_MVU`, `ACTIVATION_BROADCASTING`) | `contraction` Param, `DENSE` or `PER_CHANNEL`; broadcasting is derived | RTL |
| VVAU `Channels`, `Kernel` (window), `Dim` | `outputs` (channels), `reduction` (window), `rows` (pixels) | — |
| VVAU `PE`, `SIMD` | `pe` divides the channels, `simd` the window | RTL |
| VVAU activation order (E-048) | declared: `channel_tile`, field `s·PE + p` | RTL |
| VVAU reduction marker | `markers`, a derived one-repetition replay buffer | RTL |
| VVAU on non-DSP58 targets | `realization = "dense"` (block-diagonal weights), known weights only | RTL |
| dotp core (packed vs INT8) | `compute` Decision over `PackedDotpKernel` and `Int8Dsp58DotpKernel` | RTL (`CORE`, pin `b9262df`) |
| `pumpedCompute` | `compute_pumping` Decision (per-channel too, which baseline RTL VVAU never offered) | RTL |
| RTL `SEGMENTLEN` | derived from `target_period_ns` (B2 revision) | RTL |
| RTL `NARROW_WEIGHTS` | derived from known weights (cyclic, read-only memstream); provisional | RTL |
| RTL internal replay | `replay` Decision: `buffer` or `input_gen` | RTL |
| `mem_mode=internal_decoupled` | `delivery`: `cyclic` (ROM) or `memstream` (RAM, `ram_style`) | RTL (`memstream_axi`) |
| `runtime_writeable_weights` | `writable_weights`, memstream only; AXI-Lite exported as `s_axilite` | RTL |
| `pumpedMemory` | `delivery.memstream.pumped_memory` | RTL |
| `mlo_max_iter` / memstream `SETS` | `weight_sets`, memstream only; one index per row on `in2_V` | RTL |
| `ram_style=ultra` | `delivery.memstream.ram_style = "ultra"` | RTL |
| `noActivation=0` (fused thresholds) | not in the kernel (M-D2): adjacent kernels, composed by the dataflow layer | — |

Still gaps (flagged future work): `resType=lut` and HLS backends,
per-channel natively on DSP48E1/E2, binary/xnor and unsigned weights,
`internal_embedded`, `dynamic`, `external_mem`, `TH>1`, MMV.

## 3. Thresholding (Thresholding_hls, Thresholding_rtl)

| Baseline attribute or variant | `ThresholdingAxiKernel` coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| `PE` | `pe` Param (`k/thresholding.py:96`). It also admits PE as a multiple of C (`:150`). | RTL | V |
| `NumChannels`, `numSteps` | derived from the shape of the `thresholds` table (`k/thresholding.py:77`, `:211`) | RTL | V |
| `inputDataType`, `weightDataType` | `input_dtype`, `threshold_dtype` Params: integers of equal signedness (`k/thresholding.py:73-74`, `:102-113`) | RTL | V |
| `outputDataType` | derived `result_dtype`, UINT or INT only (`k/thresholding.py:80-94`) | RTL | V |
| `ActVal` | `bias` Param. A bias below −N−1 is refused because of a native RTL defect (`k/thresholding.py:162-174`). | RTL | V |
| `numInputVectors` | gap: the kernel has no stream and no geometry (it has no `StreamSpec`) | — | V |
| `runtime_writeable_weights` | `use_axilite` Decision (`k/thresholding.py:97`), refused with more than one set (`:176-183`) | RTL | V |
| `mlo_max_iter` → SETS, with a per-beat set stream (V10) | SETS is `len(thresholds)` (`k/thresholding.py:211`). The `s_axis_set` port is always present (`:281-286`) but has no model (probe P4). | RTL | V |
| RTL `depth_trigger_*`, `deep_pipeline` | Params and a Decision (`k/thresholding.py:98-100`) | RTL | V |
| FPARG (float or fixed-point input) | gap: pinned to 0 (`k/thresholding.py:234`) | RTL | V |
| Threshold file | inlined as the `THRESHOLDS` parameter, with `THRESHOLDS_FILE=""` (`k/thresholding.py:214-238`) | RTL | V |
| HLS variant; decoupled thresholds (THR-C2) | gap. FinnLib has no stream-consuming threshold core (V40, V42). | HLS only (an embedded table) | V |
| Composability | gap: no stream references and no `PORTS` export. Always-present buses have no driver (probe P4). | — | V |

## 4. ElementwiseBinaryOperation

| Baseline attribute or variant | `EltwiseKernel` coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| Add / Sub / Mul (RTL) | `operation` Param ADD/SUB/SBR/MUL (`k/eltwise.py:66`, `:114`) | RTL | V |
| The other HLS ops (AbsDiff, Div, logic, comparisons, shifts, Max) | gap | none | V |
| `lhs_dtype`, `rhs_dtype`, `out_dtype` | Params plus a derived result (`k/eltwise.py:68-80`) | RTL | V |
| Shapes and broadcasting, `lhs_style`/`rhs_style` const | gap. A `CyclicDelivery` rhs was composed by hand in a test (`tests/kernels/test_stream_contract.py:305-345`), not as a Space. `CyclicDelivery` is integer-only (`k/delivery.py:53`). | kernel-local cyclic ROM | V |
| `PE` | `pe` Param (`k/eltwise.py:67`) | RTL | V |
| `mem_mode`, `ram_style`, `runtime_writeable_weights` | gap, except the delivery's `rom_style` | as MVAU | V |
| RTL `B_SCALE` (fixed at 1.0) | `b_scale` Param (`k/eltwise.py:91-102`). The kernel exposes more than baseline. | RTL | V |
| RTL int MUL width limit, ≤24 signed / ≤23 unsigned (`fpd/rtl/elementwise_binary_rtl.py:115-119`) | **not checked**. The kernel admits up to 128 bits (`k/eltwise.py:57`). | — | V (missing check) |
| RTL refuses int/int and in×in without MLO (V12) | not refused: the kernel admits int/int RTL | RTL | V |
| Composability | gap: native ports but no stream references and no `PORTS` | — | V |

## 5. ConvolutionInputGenerator (SWG)

| Baseline attribute or variant | `InputGeneratorKernel` coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| Conv geometry (`ConvKernelDim`, `IFMDim`, `OFMDim`, `Stride`, `Dilation`, `IFMChannels`, `is1D`) | only a generic loop nest: `frame_words`, `extents`, `strides` (`k/input_generator.py:45-48`). No conv façade exists. | RTL (`FL/rtl/input_gen.sv`, `conv2d.sv`) | V |
| `SIMD`, input and output dtypes | gap: opaque `word_bits` (`k/input_generator.py:45`) | — | V |
| `depthwise` | gap as a flag. It is a nest order (`conv2d_dw.sv`). | RTL | I |
| `ram_style` | Decision (`k/input_generator.py:75`) | RTL | V |
| `parallel_window` / impl_style parallel | gap. It would be `input_gen` + `vpc` (V41). | partial | V/I |
| `dynamic_mode` | gap | none | V |
| Output markers | a multi-bit `olst` LOOP_END (`k/input_generator.py:95`). This **cannot** be written as a `StreamContract` rule, because rules require 1-bit markers (`k/physical/contract.py:81`). | RTL | V |
| Composability | gap: no stream references and no `PORTS` | — | V |

## 6. StreamingFIFO

| Baseline attribute or variant | Kernel coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| `depth` | `FifoKernel.depth` Param (`k/fifo.py:50`). Inside a stream it is a Decision over 2..2³² (`k/streams.py:141`). | RTL | V |
| `ram_style` | Decision that adds `shift` (`k/fifo.py:60`). The `storage` view models capacity (`:63-93`). | RTL | V |
| `folded_shape`, `dataType` | opaque: `word_bits` is derived from the stream's payload bits (`k/streams.py:135-137`), so it is **unpadded**. Baseline uses the padded AXIS width. | — | V |
| impl_style vivado, `depth_monitor`, `debug_log_path` | gap | partial (`FL/rtl/fifo_sim.sv`) | V |
| FIFO insertion | the `BufferedStream.transport` slot, on MVAU's `weight_stream` only (`k/mvau.py:250`) | — | V |

## 7. StreamingDataWidthConverter (DWC)

| Baseline attribute or variant | Kernel coordinate, or gap | FinnLib | V/I |
|---|---|---|---|
| The family as a whole (`inWidth`, `outWidth`, HLS LCM two-stage, RTL divisible) | gap: no kernel. `classify` names `WIDTH_CONVERSION` (`k/physical/forms.py:297-304`) and `compatibility` refuses it (`k/physical/contract.py:135-142`), but nothing inserts a converter. | RTL `vpc.sv` (element lanes, per-vector padding, V41; refined upstream since, AUDIT §5) | V |
