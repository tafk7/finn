# MVAU Design-Space Map (declarative)

*The complete design space of MVAU (baseline + prototype), extracted as pure data
divorced from the passes that consume it. Produced 2026-07-15 as the rigorous test
case for the generic design-space model. Every branch classified as: AXIS,
DEPENDENCY, or FEASIBILITY PREDICATE (tagged by the context it reads). file:line
throughout in the source extraction; this is the persisted reference.*

Files: **base** = `finn/src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`;
**hls/rtl** = its `hls/`+`rtl/` variants; **hwc** = `hwcustomop.py`; **util** =
`finn/util/basic.py`; **p-sip/p-hls/p-ms** = prototype
`finn-proto/.../mvau/{sip,hls}` + `memstream/rtl/memstream_rtl.py`.

## 1. Axis inventory

| Axis | Domain | Default | Defined | Free/Derived |
|---|---|---|---|---|
| PE | int, divides MH (MH%PE==0) | required | base:61 | free |
| SIMD | int, divides MW (MW%SIMD==0) | required | base:62 | free |
| MW | int (matrix width) | required | base:63 | free (tensor shape) |
| MH | int (matrix height) | required | base:64 | free (tensor shape) |
| inputDataType | any DataType | required | base:68 | free |
| weightDataType | any DataType | required | base:69 | free, derivable-down (minimize_weight_bit_width base:529) |
| outputDataType | any DataType | required | base:70 | free, or **derived** = accDataType when noActivation (base:525) |
| accDataType | any DataType | "INT32" | base:72 | **derived** (minimize_accumulator_width base:469-527) |
| resType | {auto,lut,dsp} | "auto"; hls→"lut" (hls:58); rtl→{dsp} (rtl:257) | base:65 | free (device-constrained) |
| ActVal | int | 0 | base:66 | free |
| binaryXnorMode | {0,1} | 0 | base:75 | free |
| noActivation | {0,1} (2 vs 3 node inputs) | 0 | base:77 | free |
| numInputVectors | list[int] | [1] | base:82 | free |
| mem_mode | {internal_embedded, internal_decoupled, external} | decoupled | base:87 | free |
| ram_style | {auto,block,distributed,ultra} | "auto" | base:99 | free |
| ram_style_thresholds | {auto,block,distributed} | "auto" | base:109 | free |
| runtime_writeable_weights | {0,1} | 0 | base:124 | free |
| pumpedMemory | {0,1} (clk2x weight mem) | 0 | base:125 | free |
| dynamic_input | {0,1} (runtime weight stream) | 0 | base:127 | free |
| pumpedCompute | {0,1} (clk2x DSP; RTL-only baseline) | 0 | rtl:53 | free |
| mlo_max_iter | int (MLO fetch-weights; base-level, all HWCustomOp) | 0 | hwc:100 | free |
| $VERSION$ (DSP microarch) | {1=DSP48E1,2=DSP48E2,3=DSP58} | — | rtl:253 | **DERIVED from device** |
| SEGMENTLEN (DSP chain) | int, ≤(SIMD+2)/3 on DSP58 | — | rtl:230 | **derived** (clk target) |
| narrow_weights | {0,1} | — | rtl:281 | **derived** (data-dependent) |
| backend impl_style | {hls,rtl} baseline; {sip} prototype | — | class identity | **structural axis** |

Prototype: same attribute set as frozen `@dataclass`; `pumpedCompute` first-class
(not RTL-only); no `mlo_max_iter`; weights/thresholds captured as `np.ndarray`.

## 2. Dependency graph (inter-axis relations)

Weight-delivery cluster:
- **ram_style meaningful only when mem_mode==internal_decoupled** (base:93,377,407).
- **runtime_writeable_weights exists only when mem_mode==internal_decoupled** (base:115,913).
- **memstream subkernel exists only when mem_mode==internal_decoupled** (base:1040; p-sip:102).
- **memblock.dat exists only for static internal_decoupled** (¬dynamic_input ∧ decoupled; p-sip:123).
- **in1_V weight port exists iff dynamic_input ∨ mem_mode∈{decoupled,external} ∨ mlo_max_iter** (base:900,260). internal_embedded → instream_width(1)=0, no port.
- **weight folded-shape defined ONLY for dynamic_input ∨ external** (else **raises**, base:315; p-sip:586).
- **dynamic_input, mlo_max_iter, mem_mode==external are mutually-exclusive weight-source selectors** (if/elif base:971/996/1040 → dynload/fetch_weights/memstream).
- **mlo_max_iter overrides mem_mode**: adds axi_mm master + index stream, suppresses weight port (base:900).

Double-pumping cluster:
- **pumpedCompute requires SIMD≠1** (rtl:334; sv assert).
- **pumpedMemory ∧ (PE==SIMD==1) illegal** ("known bug", base:717; p-sip:303).
- **clk2x port exists iff pumpedCompute ∨ pumpedMemory** (base:897; rtl:195).
- pumpedMemory couples memstream clk wiring (base:1100); pumpedCompute couples MVU clk (rtl:195).

Threshold cluster:
- **noActivation controls node-input count (2 vs 3)** and thresholds/TMEM/thresh.h existence (base:197).
- **ram_style_thresholds matters only when noActivation==0** (calc_tmem≠0, hls:468).

## 3. Feasibility predicate catalogue (tagged by context read)

**config-only:** MH%PE==0, MW%SIMD==0 (base:351); SIMD≥MW/1024 HLS array-partition
limit (hls:216); AP_INT_MAX_W≤8191 (p-hls:283); pumpedCompute⇒SIMD≠1 (rtl:334);
pumpedMemory⇒¬(PE==SIMD==1) (base:717); weight initializer must exist unless
external∨mlo∨dynamic (base:782); true-binary non-bipolar unsupported (hls:167);
bipolar×bipolar⇒nonneg int thresholds (base:576).

**device (fpgapart):** DSP microarch (is_versal⇒DSP58; fpgapart[2]=='7'⇒DSP48E1; else
DSP48E2, util:391); **ram_style==ultra ∧ decoupled ∧ ¬is_versal ⇒
runtime_writeable_weights==1 required** (config+device, hls:147); resType==dsp DSP
count depends on DSP block (rtl:152); VVU only on DSP58 (sv:121); narrow-weight DSP
packing is DSP58 (device+data, rtl:281).

**toolchain/clk:** SEGMENTLEN feasibility — ref_clk>0.741ns else raise
(toolchain+config(SIMD,pumpedCompute)+device(DSP58), rtl:242).

**combination:** resType legal domain depends on backend (RTL rejects lut, rtl:256;
HLS default lut, hls:58); n_max_layers=64 MLO bound (hwc:379).

## 4. Derived vs free

**Free:** PE, SIMD, mem_mode, ram_style, ram_style_thresholds,
runtime_writeable_weights, pumpedMemory, pumpedCompute, dynamic_input, mlo_max_iter,
resType, noActivation, binaryXnorMode, numInputVectors, ActVal, initial datatypes.

**Derived (must NOT be axes):** WMEM=MW·MH/(PE·SIMD) (base:345); TMEM=MH/PE or 0
(base:356); SF=MW/SIMD, NF=MH/PE (base:300); **accDataType** (data-dependent:
actual weight values unless external/mlo/dynamic → worst-case bounds, base:469-498);
outputDataType when noActivation (base:517); weightDataType narrowing (data-dependent,
base:529); stream widths (base:256); $VERSION$/$SEGMENTLEN$/$NARROW_WEIGHTS$/
$ACCU_WIDTH$; all resource/cycle estimates; prototype sip_depth=calc_wmem()/
sip_padded_width injected into memstream (p-sip:109).

## 5. Baseline-vs-prototype divergence

- **Backend axis:** baseline impl_style∈{hls,rtl} as two selectable classes;
  prototype registers only MVAUSIP(sip), fuses HLS-compute+RTL-memstream, MVAUHLS
  unregistered. **No rtl-vs-hls choice, no monolithic RTL MVU** in prototype.
- **mem_mode:** baseline branches it inside both backends + shared IPI; prototype
  makes it intra-kernel selecting *which subkernels exist* (p-sip:99), orthogonal to
  (absent) backend choice.
- **Weight delivery:** baseline emits memstream/dynload/fetch from both backends,
  wired in 236-line agnostic-base IPI; prototype promotes memstream to standalone
  MemstreamRTL Kernel composed via generic SIP stitching.
- **Expressiveness gaps:** baseline has dynamic_input+mlo+pumpedCompute-on-RTL;
  prototype has no MLO and, lacking an RTL compute core, **cannot realize
  pumpedCompute** nor a pure RTL-DSP58 MVU. Prototype's memstream drops projection
  and its runtime AXI-lite is stubbed (nonfunctional reload).

## 6. Stress-test summary — 4 features that break a naive model

1. **Variable-shape space (conditional axis existence).** Whether ram_style,
   runtime_writeable_weights, the memstream subkernel, in1_V, weight-folded-shape,
   clk2x, ActVal/thresholds, even pumpedCompute *exist as axes* depends on the
   values of mem_mode, dynamic_input, mlo, noActivation, pumped*, and backend.
   Absence is a **hard error** (base:315 raises), not a default. **Requirement:**
   dependent/guarded axis presence — not a flat fixed vector.

2. **Device-dependent feasibility.** URAM+Ultrascale forces runtime_writeable=1; DSP
   microarch/counts/narrow-weight/VVU read fpgapart. **Requirement:** feasibility
   predicates read a device context; carry per-predicate provenance
   {config|device|toolchain|data}.

3. **Derived-not-free, some data-dependent.** WMEM/TMEM/accDataType/weightDataType/
   SEGMENTLEN/VERSION/narrow_weights/widths are computed; several read the actual
   weight tensor. **Requirement:** distinguish free axes from derived; derivations
   may read external data, not just other axes.

4. **Weight-delivery: a third axis owned by neither compute backend.**
   mem_mode×dynamic_input×mlo×runtime_writeable×ram_style×pumpedMemory is an
   orthogonal sub-space, with its own device predicate (URAM), clocking
   (pumpedMemory/clk2x), and derived geometry (sip_depth). **Requirement:**
   composable, independently-constrained components — compute-backend and
   delivery-backend as two independent coordinates joined by explicit couplings.
