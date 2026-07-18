# Generality Gaps — model coverage checklist across all FINN ops

*A durable checklist of where the compute-pool + parameters-pool model holds, strains,
or breaks across FINN's full op set. Produced 2026-07-17 from an audit of all 28
op-census docs (`../finn-hw-backend-analysis/op-census/`) against the built engine +
MVAU/parameters fixtures. Purpose: inventory what's coming BEFORE generalizing, so no
op surprises us. Nothing here invalidates the built model — every item is ADDITIVE
surface. Update the status boxes as gaps are closed.*

## The model being tested

An op = a **COMPUTE selection pool** (`implementation` root axis; each bundle owns
axes/derived/predicates/sources/emit) **+ optionally a composed PARAMETERS pool**
(`parameters.topology` × cardinality × staticness, namespaced, guarded root so
param-free ops omit it via `compose(..., guard=)`) **+ a few cross-coordinate coupling
Derived/Predicates** (read both the compute fold AND the topology). MLO/FINNLoop is a
separate OUTER container level (proven mappable, unbuilt). Built + validated on MVAU
(compute byte-equivalent; memstream delivery byte-equivalent). See
[[param-delivery-design-space]], [[mvau-design-space]], [[design-space-model]].

**Implicit assumptions the audit found (the real gaps):** 1 AXIS-in / 1 AXIS-out / 1
weight-stream port shape; exactly ONE weight-like param tensor on a fixed input index;
staticness applies only to param VALUES.

---

## Tier 1 — genuine model EXTENSIONS we will need (additive; none invalidate the built model)

### G1. A real port model  — `[ ] open`
**The biggest gap.** The model implicitly assumes 1 AXIS-in / 1 AXIS-out / 1 weight
stream. Reality needs input/output **arity** axes + non-AXIS **port KINDS**: AXI-MM
master (`m_axi_gmem`), AXI-lite control register, TLAST/TKEEP sideband, `oob_irq`,
scalar side-channels. Ports should be a Derived CONSEQUENCE of the resolved point (the
param-delivery doc already treats the weight port that way — generalize to all ports).
- Forces it: `iodma-backend-only.md` (AXI-MM, folded-shape getters RAISE on the MM side),
  `checksum-backend-only-hls-only.md` (scalar `chk` output + AXI-lite reg, two outputs),
  `tlastmarker-backend-only.md` (TLAST sideband, 4/8 getters raise), `split-streamingsplit.md`
  + `duplicatestreams.md` (multi-output), `lookup-...md` (aximm + oob_irq).

### G2. Param KIND + multiplicity  — `[ ] open`
"parameters = topology × cardinality × staticness" silently assumes ONE weight-like
tensor. Missing a fourth facet: **param KIND** (weights / thresholds / embedding-table /
scale+bias) governing packing, layout, emit, and dtype-derivation — these are NOT
interchangeable payloads. Also: MULTIPLE param tensors (requant), and **param-role as a
CHOICE** (which input carries the const, if any).
- Forces it: `thresholding.md` (sorted binary-search address layout, per-PE/per-stage
  `.dat`, power-of-2 padding — utterly unlike weight `.dat`), `requant.md` (scale AND bias,
  two fp32 params), `lookup-...md` (embedding table, no accDataType narrowing),
  `elementwise-binary-...md` (const may be lhs OR rhs — `lhs_style`/`rhs_style`).
- **Directly gates the "generalize parameters pool to VVAU/Thresholding" task:** VVAU is a
  clean weights sibling (cheap); Thresholding needs param-KIND first (not cheap).

### G5. Runtime-reconfigurable NON-param config  — `[ ] open`
Staticness is currently scoped to param VALUES only. But GEOMETRY can be runtime-
programmable too — a staticness-like coordinate over config, not weights.
- Forces it: `convolutioninputgenerator-sliding-window.md` (`dynamic_mode` → AXI-lite port
  for runtime-reprogrammable window geometry), `tlastmarker-backend-only.md` (`DynIters`).

---

## Tier 2 — structurally DIFFERENT (correctly OUTSIDE the single-op pool model)

### G3. Two OUTER-container flavors  — `[ ] open (FINNLoop proven-mappable, unbuilt)`
Container nodes wrap sub-GRAPHS; the single-op (compute+parameters) taxonomy does not
apply. TWO distinct kinds:
- **FINNLoop** (`finn-loop-...md`) — MAPPABLE: delegates 8 accessors into child nodes, adds
  data-recirculation + stream-tap index-net + per-node param streaming. The index network is
  a Derived-returning-STRUCTURE from body adjacency (fork/join/broadcast, TAP_REP per-child
  fold-dependent). Proven in `test_composition_mapping.py`; the known frontier
  ([[param-delivery-space]] §7). Bounded but real.
- **StreamingDataflowPartition** (`streamingdataflowpartition-...md`) — the audit's #1 concern:
  a partition PLACEHOLDER with NO hardware contract at all (plain qonnx `CustomOp`, not even
  `HWCustomOp`; loads an `.onnx` off disk and recurses). Not even FINNLoop's kind of container.
  **Do NOT force a pool onto it** — needs an explicit graph-partition concept outside the HW
  taxonomy.

### G4. Op DECOMPOSITION + compute⊗compute composition  — `[ ] open`
The built composition is compute⊗**parameters**. Some ops are intrinsically TWO COMPUTE
backends stitched — a different join, plus a lowering step.
- Forces it: `shuffle-inner-rtl-outer-hls.md` (a non-lowerable `Shuffle` placeholder a
  TRANSFORMATION decomposes into RTL-only `InnerShuffle` + HLS-only `OuterShuffle`). Also the
  historical MVAU-SIP shape (HLS-compute ⊗ RTL-memstream). Needs op-lowering as a first-class
  concept and composition generalized beyond compute+parameters.

---

## Fits cleanly (compute pool; usually NO parameters pool via the param-free guard)

`layernorm-layernorm.md`, `hwsoftmax-hwsoftmax.md`, `pool.md`,
`globalaccpool-globalaccpool.md`, `crop.md`, `upsampler-upsamplenearestneighbour.md`,
`fmpadding-fmpadding.md`, `fmpadding-pixel.md`, `concat-streamingconcat.md`,
`labelselect.md`, `streamingdatawidthconverter-streamingdat.md`,
`vectorvectoractivation-vvau.md` (clean weights+memstream sibling of MVAU; SIMD folds
the kernel dim; RTL is DSP58-only = one feasibility predicate).

Notes: these are weightless (or derive output dtype as a `Derived` — globalaccpool
accumulator, labelselect TopK width). Their pervasive census complaint — the
`execute_node` MRO diamond (reference-vs-backend-sim) — is IRRELEVANT to the
design-space model, which already separates golden-reference/identity from `emit`.
DWC's per-backend width-ratio is a plain feasibility predicate. Decorative fp32 dtype
attrs (layernorm/hwsoftmax) map to a pinned `DatatypeSpec`. `streamingfifo-...md` is a
DIAGNOSTIC WIN: its runtime FINN-RTL-vs-Vivado-IP multiplex is native to the pool model
(Vivado-IP = its own pool member), provided "IP-instantiation" counts as a realization
kind like HLS/RTL.

---

## How this steers the roadmap

- **Composition stitch (Level-1 close-out)** is UNTOUCHED by every gap here — it wires the
  already-byte-equivalent memstream + compute wrappers into one block design. Highest-value
  next step, no dependency on G1–G5.
- **Generalize parameters pool to VVAU** — cheap (clean weights sibling). **To Thresholding**
  — blocked on **G2 (param KIND)**.
- **G1 (port model)** and **G2 (param kind)** are the two deliberate model-extension tracks
  that multi-output / backend-only / threshold ops will force; plan them as first-class work,
  not per-op patches.
- **G3/G4** are the outer-container + compute⊗compute frontiers — after Level-1 is closed.

*Cross-refs: [[param-delivery-space]] (the three coordinates + two composition levels),
[[impl-bundles]] (the selection machinery), [[emit-phase]] (compute-half status).*
