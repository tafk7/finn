# FINN HW Backend — As-Is Modular Model

*A faithful `/modular-dev` model of the `HLSBackend`/`RTLBackend` abstraction exactly as it exists today, including op-specific customizations and hacks. This is the input to a future redesign — it describes the system as-is, and does NOT propose the new design.*

## Provenance

Built by a 3-phase read-only agent analysis (all Opus), one artifact per phase:

- **Phase A** — `core-interfaces.md`: 4 agents characterized the core contract. Fidelity gate passed (abstract-method sets 8/4/3 match `grep`).
- **Phase B** — `census-matrix.md` + `op-census/*.md`: 27 census agents + 27 adversarial-verify agents over all op families. **174 hack claims confirmed, 3 partially-confirmed, 0 refuted.** 351 hacks + 106 hermeticity violations catalogued; every claim carries a `file:line`.
- **Phase C** — this document: 4 synthesis agents (one per model dimension) + 1 completeness critic.

## Headline findings

1. **The abstraction is 2-axis** — `HWCustomOp` substrate + one of `{HLSBackend, RTLBackend}` mixed in by multiple inheritance; a concrete op is `(OpBase, Backend)`.
2. **The substrate is not hermetic w.r.t. its children** — `HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` branch on hard-coded subclass op-type strings and call methods that exist only on subclasses. This back-edge (top module reaching into leaves) is the concrete root of the "MVAU instantiation mess." **10 base-class-leak hacks.**
3. **HLS and RTL are not peers** — 638 vs 146 LOC, 4 vs 3 abstract methods, sharing nothing but the substrate. HLS codegen mutates a shared `code_gen_dict` side-channel in a required order; RTL parameterizes handwritten `finn-rtllib` Verilog. "Backend" is a modeling fiction over two different things.
4. **The selection layer replicates the anti-pattern** — `specialize_layers.py` hard-codes per-op RTL-feasibility guards and instantiates variants by string concatenation, mirroring the base-class leak at the selection seam.
5. **21 blocker hacks** actively prevent clean substitution; the ranked redesign-pressure list is at the end of this document.

---


# Module Structure (DAG) + Variant-Selection Adapter

*As-is model of the HW-backend abstraction as a /modular-dev module DAG, plus the `SpecializeLayers` machinery that selects a concrete variant. Cited to `file:line`.*

## 1. The module DAG

The abstraction is **2-axis**: one agnostic op module supplies the datatype/shape/schema contract, a backend mixin supplies codegen, and the concrete op is the product of the two. In /modular-dev terms there are three tiers of module plus two downstream *artifact* dependencies (handwritten HDL/C++ libraries that are consumed as data, not imported as code).

```
                         ┌───────────────────────────────────────────┐
                         │  SUBSTRATE  (bounded context: "an HW node") │
                         │  HWCustomOp(CustomOp)   hwcustomop.py       │
                         │  8 abstract methods = THE downstream        │
                         │  contract (get_*_datatype/shape/width)      │
                         │  + generate_hdl_{memstream,fetch_weights,   │
                         │    dynload}  ← LEAK: substrate branches on   │
                         │    subclass op_type (L307/355/407)          │
                         └───────────────┬─────────────────────────────┘
                                         │ inherits
                 ┌───────────────────────┼───────────────────────────┐
                 │ (agnostic op modules — one bounded context per op) │
                 │  MVAU, VVAU, Thresholding, StreamingFIFO, DWC,     │
                 │  ElementwiseBinaryOperation, LayerNorm, Requant,   │
                 │  Pool, CIG, FMPadding, Lookup, ... (27 families)   │
                 │  define the 8 abstracts in terms of op semantics.  │
                 └──────┬──────────────────────────────┬─────────────┘
                        │ multiple-inheritance mix-in   │ multiple-inheritance mix-in
             ┌──────────▼──────────┐          ┌─────────▼───────────┐
             │ HLSBackend(ABC)     │          │ RTLBackend(ABC)     │
             │ hlsbackend.py 638LOC│          │ rtlbackend.py 146LOC│
             │ 4 abstract+25 concr │          │ 3 abstract+5 concr  │
             │ side-channel:       │          │ parameterizes hand- │
             │ mutable code_gen_dict│         │ written Verilog     │
             └──────────┬──────────┘          └─────────┬───────────┘
                        │ concrete = (OpBase, Backend)  │
             ┌──────────▼──────────┐          ┌─────────▼───────────┐
             │ MVAU_hls, VVAU_hls, │          │ MVAU_rtl, VVAU_rtl, │
             │ Thresholding_hls,   │          │ FIFO_rtl, DWC_rtl,  │
             │ Elementwise*_hls... │          │ LayerNorm_rtl ...   │
             │ registered in       │          │ registered in       │
             │ hls/__init__        │          │ rtl/__init__        │
             │  (hls_variants dict)│          │  (rtl_variants dict) │
             └──────────┬──────────┘          └─────────┬───────────┘
                        │ emits/reads (artifact dep)    │ emits/reads (artifact dep)
             ┌──────────▼──────────┐          ┌─────────▼───────────┐
             │  finn-hlslib        │          │  finn-rtllib        │
             │  (C++ templates,    │          │  (handwritten .v/.sv│
             │   Vitis HLS)        │          │   modules + TCL)    │
             └─────────────────────┘          └─────────────────────┘

   templates.py  ── shared string constants (docompute/ipgen/ipgentcl/
                    ip_package_tcl/ip_gen_loop_op). NOT a class in the DAG;
                    a cross-cutting data module imported by HLSBackend codegen
                    and by finn_loop. Embeds $::env(FINN_ROOT) + hard-coded
                    finn-rtllib source paths (templates.py:331-357) → couples
                    the "shared strings" module directly to the RTL artifact.
```

Key structural notes on the DAG:

- **The substrate is not hermetic w.r.t. its children.** `HWCustomOp.generate_hdl_memstream` (`hwcustomop.py:307`), `generate_hdl_fetch_weights` (:355) and `generate_hdl_dynload` (:407) live *above* the split yet branch on concrete op-type strings and call `calc_wmem`/`calc_tmem`/`ram_style`/MW/MH/PE/SIMD that exist only on subclasses. This is a **back-edge** in what should be a DAG: the top module reaches down into leaves (10 base-class-leak hacks).
- **HLS and RTL are not peers.** 638 vs 146 LOC, 4 vs 3 abstracts (`core-interfaces.md` §2–3). They share nothing but the substrate; HLS communicates through a mutable `code_gen_dict` side-channel (`hlsbackend.py:136`), RTL parameterizes Verilog. Treating them as one "Backend" interface is a modeling fiction.
- **`templates.py` is a data module, not a node in the inheritance DAG**, but it is a real edge: it bakes `finn-rtllib` paths into strings, so the "shared templates" module has a hard dependency on the RTL artifact tree even for the HLS path.

## 2. The variant-selection adapter (`specialize_layers.py`)

`SpecializeLayers` (a QONNX `Transformation`, `specialize_layers.py:377`) is the **seam** that turns an agnostic node (`domain == "finn.custom_op.fpgadataflow"`) into a concrete `_hls`/`_rtl` node. It is the substitution point of the whole abstraction. The decision is centralized in `_determine_impl_style(node, fpgapart, model)` (L40).

**Availability probe (L44–45).** It first asks the two registries whether a variant exists, by *string concatenation*:

```python
hls_variant = optype + "_hls" in hls_variants.keys()   # L44
rtl_variant = optype + "_rtl" in rtl_variants.keys()   # L45
```

**Policy.** Reading `preferred_impl_style` (L49):
- `""` (unset): prefer RTL for "simple" layers when an RTL variant exists (L53–90), else HLS (L92), else raise (L97).
- `"hls"`: honor if HLS exists, else warn-and-fall-back to RTL (L105–121).
- `"rtl"`: honor if the per-op feasibility guard passes, else warn-and-fall-back to HLS (L122–211).

**Per-op RTL-feasibility guards.** The generic "prefer RTL" rule is overridden by a hard-coded `if optype == ...` ladder that dispatches to op-specific predicates. These encode the physical capability gap between each op's RTL module and its HLS module:

| op_type (string-matched in `_determine_impl_style`) | guard predicate | line | gate summary |
|---|---|---|---|
| `StreamingDataWidthConverter` | `_dwc_determine_impl_style` | L221 | RTL only if in/out widths are integer ratios (`% == 0`) |
| `MVAU` | `_mvu_rtl_possible` | L235 | needs activation off, signed weights, DSP-block-dependent narrow-weight rule, 2..8-bit dtypes; also inline `bitwidth() >= 4` pre-gate at L60 |
| `VVAU` | `_vvu_rtl_possible` | L281 | Versal-only, `noActivation`, ≤8-bit signed weights, ≤8/9-bit acts; inline `>=4` pre-gate at L69 |
| `ElementwiseAdd/Sub/Mul` | `_elementwise_rtl_possible` | L301 | Versal-only DSP58, float/float or int/float, shape/broadcast checks; int/int forced HLS |
| `LayerNorm` | `_layernorm_rtl_possible` | L350 | Versal-only, input must be `FLOAT32` |
| `Requant` | `_requant_rtl_possible` | L363 | integer input, unsigned output, `narrow == 0` |

**Instantiation (L399–405).** Having chosen `impl_style ∈ {"hls","rtl"}`, it rebuilds the node purely by string manipulation:

```python
optype = node.op_type + "_" + impl_style                       # L399
new_node = helper.make_node(
    optype, node.input, node.output,
    domain="finn.custom_op.fpgadataflow." + impl_style)        # L401-405
```

The concrete class is then resolved out-of-band by `getCustomOp` against that synthesized `domain`. Attributes are copied verbatim except `preferred_impl_style` is dropped (L409). Two op families are excluded from the loop: `Shuffle` (specialized later by `InferInnerOuterShuffle`, L395) and anything not in the fpgadataflow domain (L390).

## 3. Modular-dev characterization: adapter, or replicated anti-pattern?

`SpecializeLayers` is *structurally* an adapter — a single seam that maps an abstract node onto one of two implementation modules, exactly where you'd want the substitution boundary. But it is **not a clean adapter**; it replicates the same *base-knows-subclasses* inversion that plagues the substrate:

1. **It hard-codes per-op knowledge that belongs in the ops.** The whole L54–194 ladder is a list of `if optype == "MVAU"/"VVAU"/"LayerNorm"/"Requant"/...` special-cases, each with an op-specific predicate (`_mvu_rtl_possible` etc.) that reaches into that op's *node attributes and datatypes* (`noActivation`, `binaryXnorMode`, `lhs_style`, `narrow`, weight signedness, DSP block). The selection module thus embeds detailed, private knowledge of six op families' RTL capability envelopes. This is the **same leak as `HWCustomOp.generate_hdl_memstream` branching on op_type** — only relocated from the substrate to the adapter. A capability that is a property of *the RTL variant of op X* is asserted by a central switch, not queried through an interface (there is no `rtl_variant.is_feasible(node, fpgapart)` contract method).

2. **The interface it adapts against is stringly-typed, not a contract.** Variant existence is `optype + "_hls" in dict` (L44) and instantiation is `optype + "_" + impl_style` with `domain + impl_style` (L399–405). There is no declared interface object — the "seam" is a naming convention. Any op whose class name doesn't follow `<Optype>_<style>` silently fails to resolve; adding a third backend means editing this string logic plus every `if optype ==` arm.

3. **Duplication / drift risk.** The inline pre-gates (`idt.bitwidth() >= 4` at L60/L69) live *next to* but are *not the same as* the thresholds inside `_mvu_rtl_possible` (`>= 2`, L275) — two copies of "is this dtype RTL-able" that can and do disagree. That is the classic symptom of policy that should be co-located with the module it governs.

**Verdict:** the layer occupies the right position for an adapter but implements it as a **god-switch**. A hermetic design would invert control: each RTL variant module *exposes* a `rtl_feasible(node, fpgapart) -> bool` (or richer capability descriptor) as part of the `RTLBackend` contract, and `SpecializeLayers` becomes a thin, op-agnostic dispatcher — `if rtl_variant and rtl_cls.feasible(node): "rtl"`. Today the adapter *is* the coupling.

## 4. Circular / irregular edges

Two families of irregularity break the clean DAG:

**(a) Backend-only ops — missing the agnostic tier.** Four ops have **no agnostic base module**; the op *is* a backend class directly (`census-matrix.md`, `backend-only` column, cross-referenced with the established facts): **`iodma`, `checksum`, `tlastmarker`, `finn_loop`**. In DAG terms the middle tier is absent — the concrete node inherits substrate + backend with no op-semantics module between them. Consequences:

- These ops are invisible to `SpecializeLayers`: they are already `_hls`/`_rtl` (or exist only in one backend, e.g. `checksum` is HLS-only), so there is no `<optype>` to `<optype>_<style>` transition. The selection seam simply doesn't apply — an asymmetry the adapter's string convention silently tolerates because the un-suffixed name is never in `hls_variants`/`rtl_variants`.
- `finn_loop` is the worst offender: it is backend-only *and* a container. `get_rtl_file_list` (a required `RTLBackend` abstract) is a bare `pass` returning `None` (`finn_loop.py:1175`), satisfied only nominally; it dodges the broken contract by overriding `prepare_rtlsim` (:277). This is an **irregular edge**: the node claims to implement the `RTLBackend` interface but doesn't honor it, and the base path that would call the abstract (`rtlbackend.py:57`) is bypassed.

**(b) Base ↔ backend / base ↔ sibling cycles (true back-edges in the DAG).** The census confirms several edges pointing *upward or sideways*, i.e. an agnostic/base module depending on a concrete backend or a sibling op:

- **Substrate → subclass:** `HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` (`hwcustomop.py:307/355/407`) branch on op_type and call subclass-only methods. Elementwise, MVAU, VVAU, Thresholding *rely* on this leak (e.g. `hwcustomop.py:311` startswith `Elementwise`; :310 allowlist includes `Thresholding_hls`). Cycle: base needs child, child needs base.
- **Agnostic MVAU → its own subclasses:** `matrixvectoractivation.py:920` — `code_generation_ipi` (~236 lines) sits in the *agnostic* MVAU base yet calls `self.instantiate_ip()`, defined only on `MVAU_hls` (:681) and `MVAU_rtl` (:163). The agnostic class is **uninstantiable by contract** — a structural back-edge to both leaves.
- **Agnostic op branches on its own RTL class name:** `vectorvectoractivation.py:617` — `make_weight_file` (agnostic VVAU) branches on `op_type == "VVAU_rtl"` to pick memory ordering; `streamingfifo.py:90/104` — agnostic `StreamingFIFO` reaches for `impl_style` / `get_adjusted_depth()` that exist only on `StreamingFIFO_rtl` (:46/:53), via `try/except AttributeError`. These are agnostic-tier modules with hard back-edges to their RTL leaf.
- **Cross-backend edge (HLS leaf → RTL artifact):** `matrixvectoractivation_hls.py:144` and `elementwise_binary_{hls,rtl}` call `generate_hdl_*` to emit finn-rtllib weight-streamer Verilog — an HLS-compute op with an RTL-delivery edge fused in (13 cross-backend-leak hacks total). The two "backend" subtrees, drawn as disjoint in the DAG, are in fact cross-wired.
- **Sibling → sibling edge:** `finn_loop.py:421/712` — the loop container dispatches on child op_type prefixes (`MVAU`/`Elementwise`/`Thresholding`) and hard-codes their `.dat` naming and stream-tap wiring. `finn_loop` (itself backend-only, no agnostic tier) has direct edges into three sibling op families' private conventions.

Net: the intended DAG (substrate → op → backend → artifact) is in practice a **cyclic graph** — the substrate and several agnostic op modules carry back-edges to their own backend leaves, the two backend subtrees are cross-wired, `finn_loop` edges sideways into siblings, and four ops skip the middle tier entirely. The variant-selection adapter, rather than absorbing this irregularity behind a clean interface, encodes a sixth copy of the per-op knowledge as a central switch.


---


# Dimension: Nominal vs Real Interface — The Contract Gap

*As-is model of the distance between what the FINN HW-backend ABCs **declare** as their contract and what each backend **actually demands** of its host object at runtime. Every abstract method named here is catalogued in `core-interfaces.md`; every override/hack count is drawn from `census-matrix.md` and `_census_digest.json`.*

## 1. The nominal contract: 15 abstract methods across three ABCs

The declared interface — the sum of every `@abstractmethod` a concrete op must satisfy — is small and clean:

| ABC | Abstract methods (nominal contract) | Source |
|---|---|---|
| `HWCustomOp(CustomOp)` | `get_input_datatype`, `get_output_datatype`, `get_normal_input_shape`, `get_normal_output_shape`, `get_folded_input_shape`, `get_folded_output_shape`, `get_instream_width`, `get_outstream_width` (**8**) | `hwcustomop.py:261–289` |
| `HLSBackend(ABC)` | `global_includes`, `defines`, `docompute`, `blackboxfunction` (**4**) | `hlsbackend.py:418, 425, 542, 605` |
| `RTLBackend(ABC)` | `generate_hdl`, `get_rtl_file_list`, `code_generation_ipi` (**3**) | `rtlbackend.py:53, 77, 82` |

Read literally, this says: *"Implement 8 shape/datatype methods, pick a backend, implement 3–4 codegen methods, and you have a working op."* This is the contract the redesign would naively preserve. It is **almost entirely fictional** — it captures neither the real coupling nor the real behavior.

## 2. The real contract: the undeclared host-object interface

Both `HLSBackend` and `RTLBackend` are **mixins**, never instantiated alone. A concrete op is `(OpBase, Backend)` under multiple inheritance, and the backend half calls back into `HWCustomOp` (and into methods that live *only on the concrete op*) without declaring any of it. `core-interfaces.md:228` names this the "undeclared host-object interface"; `rtlbackend.py:188` flags the same for the RTL side. The mixin's *real* precondition is a rich host object supplying at least the following — none of which appear in any ABC:

**Node-attribute & ONNX substrate (called everywhere):**
- `get_nodeattr` / `set_nodeattr` / `get_nodeattr_types` — the entire attribute schema is accessed dynamically by string key, not typed.
- `onnx_node` — the raw protobuf node; backends read `.op_type`, `.input`, `.name` directly.
- `get_input_datatype` / `get_output_datatype` — the abstracts, but called with `ind` arguments the signature under-specifies (see §4).

**Stream/width helpers (concrete on `HWCustomOp`, consumed by both mixins):**
- `get_instream_width_padded` / `get_outstream_width_padded` (`hwcustomop.py:292, 301`) — the memstream wrapper fills `$WIDTH$` from `get_instream_width_padded(1)` (`_census_digest.json:1348`).
- `get_number_output_values`, `get_folded_input_shape`/`get_folded_output_shape` (called index-free — a latent multi-input bug, `hlsbackend.py:470`).

**rtlsim lifecycle (concrete on `HWCustomOp`, reaching the `finnxsi` singleton):**
- `get_rtlsim`, `reset_rtlsim`, `rtlsim_multi_io`, `close_rtlsim` — `RTLBackend.execute_node` and `prepare_rtlsim` assume all of these exist on the host (`rtlbackend.py:56, 88`).

**Codegen/IP plumbing that the ABC declares but the base then *calls back into*:**
- `instantiate_ip` — declared on **no** ABC; defined only on `MVAU_hls:681` / `MVAU_rtl:163`, yet called by the *agnostic base* `MVAU.code_generation_ipi` (`matrixvectoractivation.py:920`). The agnostic class is uninstantiable-by-contract: it structurally depends on a method its own subclasses invent.
- `generate_params` — a no-op hook on `HWCustomOp:248`, but `MVAU_rtl.generate_hdl:271` calls it to emit HLS `params.h`/`.dat` (`_census_digest.json:339`).

**Op-specific methods the substrate calls but never declares (the base-class leak):**
- `calc_wmem` / `calc_tmem`, `get_nodeattr('ram_style')`, `MW`/`MH`/`PE`/`SIMD`, `rhs_shape` — invoked by `HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` (`hwcustomop.py:307/355/407`) on `self`, though **none exist on the base**. The substrate hard-codes knowledge of its subclasses (see §4's op_type allowlist).
- `get_adjusted_depth` (`streamingfifo.py:104`), `use_parallel_window_output` (`convolutioninputgenerator`, `_census_digest.json:1049`), `pumpedCompute`/`impl_style`/`mem_mode` nodeattrs — all read by "agnostic" base methods via `try/except AttributeError`, i.e. the base reaches *up* into subclass-only state and papers over the missing declaration with exception control-flow.

**Net gap:** the nominal contract is 15 methods; the real contract is those 15 **plus ~15+ undeclared host methods/attrs**, several of which are defined only on the concrete leaf, not on any class in the declared hierarchy. The ABCs describe roughly half the true interface, and the missing half is exactly the part that carries the coupling.

## 3. The `code_gen_dict` side-channel: a hidden, ordering-dependent contract

The four HLS abstracts (`global_includes`, `defines`, `docompute`, `blackboxfunction`) *appear* to be independent pure functions. They are not. Each **mutates a shared instance dict** `self.code_gen_dict` keyed by `$PLACEHOLDER$` (`hlsbackend.py:136`), and the orchestrators (`code_generation_ipgen:130`, `code_generation_cppsim:213`) populate → consume → **clear** it (`hlsbackend.py:154/175/244`). The real HLS contract is therefore not "implement 4 functions" but "implement 4 *ordered side-effecting steps* that collectively leave one mutable dict in a fillable state":

- `global_includes` must run before `defines` before `docompute` before `blackboxfunction`; the template fill at the end assumes every `$KEY$` slot has been populated.
- The ordering dependence extends *into the ops*: `thresholding_hls` sets `_scale_is_one`/`_bias_is_zero` in `generate_params` and reads them in `docompute` via `getattr(..., False)` (`_census_digest.json:776, 848`) — if `docompute` runs first the optimization silently defaults off. `lookup_hls.global_includes` `#include`s a header that `generate_params` must have written first (`_census_digest.json:588`).

This is the exact opposite of a hermetic interface: the "contract" is a required call sequence over hidden mutable state, un-declared and un-testable in isolation. `core-interfaces.md:150` records it as a top-level smell; `census-matrix.md` counts **6 `order-dependence` + 8 `module-mutable-state`** hermeticity violations, most of them instances of this pattern. RTL has a milder analogue: `generate_hdl` writes the `gen_top_module` nodeattr that `get_rtl_file_list`/`code_generation_ipi` later read (`_census_digest.json:1401`, `fmpadding_rtl.py:121`) — the abstract methods are ordered even though the ABC presents them as peers.

## 4. Honored uniformly vs subverted: which abstracts actually hold

The 8 `HWCustomOp` shape/datatype abstracts are the **only** part of the nominal contract that most families honor as intended — the census repeatedly notes "the pure dataflow/compute contract (shapes/widths/folding) is genuinely clean" (MVAU `_census_digest.json:185`, VVAU `:1246`, elementwise `:1592`). But a hard core of ops subvert even these, and the subversions cluster on `get_instream_width`/`get_input_datatype` and their `ind` parameter — the signature says `(self, ind=0)`, implying uniform per-index streams, but the worst offenders make the *meaning* of the method depend on which subclass, memory-mode, or input index you are in:

**Worst offenders (from the digest):**
- **`get_instream_width` branching on undeclared mode state.** `thresholding.py:173` and `thresholding.py:292` (`get_verilog_top_module_intf_names`) branch on `mem_mode` (an HLS-only attr) and `mlo_max_iter` via `try/except`, defaulting to 0 — the *agnostic base's* width contract is fused to HLS decoupled-memory semantics (`_census_digest.json:1844, 1951`). Same pattern verbatim in the elementwise base.
- **`get_instream_width` returning 0 for non-stream inputs.** `requant` is multi-input with embedded scale/bias constants; `get_instream_width` returns 0 for `ind!=0` and `get_folded_input_shape` special-cases `ind`, purely to stop the backend loops from trying to stream a constant (`_census_digest.json:758`). The `ind` axis, nominally uniform, is overloaded to mean "is this input actually a stream."
- **Datatype abstracts that ignore their own return contract.** `hwsoftmax.py:82` `get_output_datatype` hard-returns `FLOAT32` and ignores `ind` entirely; `layernorm` computes widths from the datatype attrs while the RTL wrapper hard-codes `[31:0]` and HLS hard-codes `TO=float` (`layernorm_wrapper_template.v:20/25`, `layernorm_hls.py:46`) — the datatype attributes are **decorative**; the declared width contract is silently overridden by the emitted hardware (2 of the 21 blocker hacks).
- **The abstract that the substrate itself subverts.** `HWCustomOp.generate_hdl_memstream` branches on a hard-coded `op_type` allowlist `["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"]` (`hwcustomop.py:310`) and `startswith("Elementwise")`/`startswith("Thresholding")` — the base class dispatches on concrete subclass identity instead of a declared method. `make_weight_file` (VVAU base) does the same with `op_type=='VVAU_rtl'` (`vectorvectoractivation.py:617`). This is the inversion that makes "backend-agnostic base" a misnomer: **10 base-class-leak hacks** total.
- **`get_rtl_file_list` satisfied only nominally.** `finn_loop.py:1175` implements this *required* RTL abstract as a bare `pass` returning `None`, then overrides `prepare_rtlsim` to dodge the code path that would call it (`_census_digest.json` finn_loop). The abstract contract is honored on paper and violated in fact.

**Uniformly honored:** `global_includes`/`defines`/`blackboxfunction` are populated by essentially every HLS op as intended; the shape abstracts hold for the clean families (concat, crop, pool, globalaccpool, duplicatestreams). The subversion is concentrated in the high-pressure families — the same ops that carry a hidden third axis (weight-delivery / mem_mode / impl_style) the 2-axis abstraction cannot express.

## 5. Quantifying the gap: overrides as a proxy for contract failure

If the nominal 15-method contract captured real behavior, ops would implement ~11–12 methods (8 shape + 3–4 backend) and override little else. Instead, override counts run **4× to 6× the nominal contract**, and the excess is precisely the undeclared/leaked interface:

| Family | overrides | ≈ nominal (11–12) | excess | what the excess encodes |
|---|--:|--:|--:|---|
| shuffle | 44 | 12 | **+32** | nested-sim reimpl of HLS internals in the agnostic base |
| matrixvectoractivation | 40 | 12 | **+28** | weight-delivery axis + 236-line IPI in the base |
| vectorvectoractivation | 38 | 12 | **+26** | mem_mode + `make_weight_file` op_type sniffing |
| thresholding | 30 | 12 | **+18** | 3 divergent threshold-serialization copies |
| requant | 30 | 12 | **+18** | embedded-param inputs vs stream assumption |
| lookup | 25 | 12 | **+13** | mem_mode ROM-vs-DMA third axis |
| layernorm | 25 | 12 | **+13** | fp32 pinning across 3 layers |
| elementwise_binary | 26 | 12 | **+14** | broadcast + param-source axis |
| finn_loop | 18 | 12 | **+6** | container op — wrong taxonomy entirely |

Across all 27 families the census records **~570 overrides** and **331 hacks** (of which 21 blockers, 10 base-class-leak, 13 cross-backend-leak) plus **106 hermeticity violations**. The override count is the direct measure of how much real behavior the nominal interface fails to name: every override beyond the 15 abstracts is a piece of the *real* contract that the declared contract does not model — most of it either (a) an undeclared third axis (weight-delivery, mem_mode, impl_style) branched into methods that pretend to be backend-uniform, or (b) reach-through between base and subclass that the ABC declares as a clean boundary. Even the *cleanest* families (concat 13, streamingdataflowpartition 4) confirm the ceiling: the ones that stay near nominal are exactly the ones with no third axis and no cross-backend leak.

**Conclusion for the redesign.** The ABC layer is a *nominal* seam, not a *real* one. The real interface is the undeclared host-object surface (§2), threaded through a mutable-dict side-channel with required call ordering (§3), with the "uniform" abstracts subverted wherever an unmodeled axis intrudes (§4), at a measured 4–6× override tax (§5). Any substitution boundary drawn at the current abstract methods will leak, because the load-bearing contract — `instantiate_ip`, `calc_wmem`, `get_adjusted_depth`, `mem_mode`, `code_gen_dict` ordering, the `op_type` allowlists — lives entirely outside what the three ABCs declare.


---


# Hermeticity Violation Catalogue & Bounded Contexts

*As-is model of the FINN HW-backend abstraction along the hermeticity dimension. Hermeticity — the modular-dev property that a module can be built, tested, and reasoned about with only its declared inputs — is the precondition for **substitution**, the master property. Every violation below is a place where an op reaches outside its declared interface into ambient process state, the filesystem, a sibling module, or the substrate, and thereby cannot be swapped, faked, or relocated without editing something else.*

Two population layers matter here and should not be conflated:

- **Substrate-level smells (31)** live in the four core files (`hwcustomop.py` 9, `hlsbackend.py` 11, `rtlbackend.py` 4, `templates.py` 7). These are shared by *every* op by inheritance, so a single substrate violation multiplies across all 27 families.
- **Family-level violations (106)** are the per-op census hermeticity hits, taxonomised as: `env-var` 34, `filesystem-path` 26, `hidden-coupling` 22, `sibling-op-coupling` 8, `module-mutable-state` 8, `order-dependence` 6, `other` 2.

---

## 1. Ranked catalogue of violation classes (by substitution-blocking force)

The census taxonomy sorts by *mechanism*. Ranked instead by *how hard each blocks substitution*, the order inverts: the rarest classes (base-branches-on-subclass, sibling coupling) are the most fatal, because they defeat substitution structurally rather than merely making a module awkward to isolate.

| Rank | Violation class | Count | What it blocks | Worst-case cite |
|---|---|--:|---|---|
| 1 | Substrate-branches-on-subclass (`hidden-coupling` subset) | ~4 sites, 10 base-class-leak hacks | Adding/renaming/removing an op requires editing the substrate | `hwcustomop.py:307/355/407` |
| 2 | Sibling-op coupling | 8 | One op cannot be built/tested without its neighbours present | `finn_loop.py:421`, `shuffle.py:18` |
| 3 | `finnxsi` import-time singleton | 3 substrate sites | rtlsim simulator cannot be injected/faked; every rtlsim op couples to it | `hwcustomop.py:39` |
| 4 | Ambient env-var config | 34 | Op cannot run outside a configured Vitis/FINN install; not parameterizable | `outer_shuffle.py:189` |
| 5 | Hard-coded finn-rtllib / Vitis filesystem layout | 26 | RTL library cannot be reorganised or substituted; brittle path/filename welds | `matrixvectoractivation.py:979` |
| 6 | `code_gen_dict` mutable side-channel + `module-mutable-state` | 1 substrate + 8 | Individual codegen steps cannot be composed, swapped, or unit-tested | `hlsbackend.py:136`, `requant_hls.py:59` |
| 7 | Order-dependence | 6 | Methods silently break unless called in a hidden sequence | `matrixvectoractivation.py:983` |

### 1.1 Substrate-branches-on-subclass — the hardest blocker

The substrate is supposed to be a context-free interface. Instead, three HDL-wrapper generators on `HWCustomOp` name concrete subclasses:

- `generate_hdl_memstream` (`hwcustomop.py:307`) branches on a hard-coded allowlist `["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"]` and `op_type.startswith("Elementwise")` (`:310`/`:311`), calling `calc_wmem`/`calc_tmem`/`ram_style` — **none defined on the base**.
- `generate_hdl_fetch_weights` (`hwcustomop.py:355`) has a dedicated `Elementwise` else-branch (`:359`) with a `TODO use broadcast rhs shape here` (`:375`) and a magic `n_max_layers=64` (`:379`).
- `generate_hdl_dynload` (`hwcustomop.py:407`) unconditionally assumes MVAU-style `MW/MH/PE/SIMD`.
- The MLO feature flag `mlo_max_iter` is declared on the base (`hwcustomop.py:100`) but is meaningful only to weight-bearing ops, threaded through ~12 MVAU sites and Thresholding (`thresholding.py:133`).

This is the root of the "MVAU instantiation mess": you cannot substitute a new op family, or rename an existing one, without editing the substrate — the antithesis of an open module boundary. Six families (MVAU, VVAU, Thresholding, Elementwise, plus the inverse `make_weight_file` op-type sniff at `vectorvectoractivation.py:617` and `matrixvectoractivation.py:892` `pumpedCompute` read) **depend** on this leak to function.

### 1.2 Sibling-op coupling — modules that cannot stand alone

Eight sites import or introspect a *different* op, so the two cannot be built or versioned independently:

- `finn_loop.py:421`/`712`/`558` — the container dispatches on child `op_type` prefixes (`MVAU`/`Thresholding`/`Elementwise`) and reads child `mlo_max_iter` to lay out `.dat` files and stream taps; `else: raise` (`:450`).
- `shuffle.py:18` — a `custom_op` imports the `transpose_decomposition` **transformation** and instantiates `InnerShuffle`/`OuterShuffle` via `getCustomOp` inside `get_exp_cycles`.
- `vectorvectoractivation.py:126` — `execute_node` inspects the **producer** node (`Im2Col`/`ConvolutionInputGenerator`) to set effective PE.
- `convolutioninputgenerator_rtl.py:321` — cppsim reshapes output specifically to match downstream `VVAU` PE-interleaving.
- `matrixvectoractivation_rtl.py:66` — calls unbound `MVAU.execute_node(self,...)` and reuses base `generate_params`.
- `elementwise_binary_hls.py:40` — imports `LoopBodyInputType` from the `loop_rolling` transform.

### 1.3 `finnxsi` import-time singleton

Resolved once at module import in all three substrate files (`hwcustomop.py:39`, `hlsbackend.py:44`, `rtlbackend.py:37`; `AttributeError` if xsi absent, `rtlbackend.py:64`, no guard). Because it is a module global, no op's rtlsim path (`get_rtlsim`, `reset_rtlsim`, `rtlsim_multi_io`, `close_rtlsim`) can accept an injected or faked simulator — every substitution or test of the rtlsim seam is blocked at the substrate. `rtlsim_multi_io` also reads a process-wide `get_liveness_threshold_cycles()` (`hwcustomop.py:230`).

### 1.4 Ambient env-var config (34)

`FINN_ROOT` dominates (in the three substrate `generate_hdl_*` helpers at `hwcustomop.py:313/360/409`, and in nearly every RTL variant: `matrixvectoractivation_rtl.py:167`, `matrixvectoractivation.py:979`, `fmpadding_rtl.py:111/148`, `layernorm_rtl.py:37/80`, `streamingfifo_rtl.py:78/204`, `streamingdatawidthconverter_rtl.py:85/117`, `convolutioninputgenerator_rtl.py:342/521/861/902`, `vectorvectoractivation.py:822`+`_rtl.py:151/269/293`, `elementwise_binary_hls.py:732`/`_rtl.py:142/188/303`, `thresholding_hls.py:671`/`_rtl.py:294/320`, `requant_rtl.py:90/136`, `inner_shuffle_rtl.py:79`, `finn_loop.py:390/541/698/1082`). The Vitis toolchain vars sit in the HLS substrate: `XILINX_VIVADO` (`hlsbackend.py:256`, regex, no None-guard), `HLS_PATH` (`:261`), `VITIS_PATH` (`:263`), and `FINN_ROOT`-relative include flags (`:267–274`). The starkest case is `outer_shuffle.py:189–191`: a **pure cost model** reads `XILINX_VIVADO`, regex-extracts the year, and unconditionally dereferences `match.group()` — crashing if the var is unset. Env-var config blocks substitution because a module cannot be exercised in isolation; its behaviour is a hidden function of process ambient state.

### 1.5 Hard-coded finn-rtllib / Vitis filesystem layout (26)

Every RTL op welds itself to exact `finn-rtllib/<subtree>/hdl` paths and literal `.sv`/`.v` filenames, usually via `shutil.copy` into `code_gen_dir` plus a source list duplicated 3× (e.g. `layernorm_rtl.py:69/85/98`, `fmpadding_rtl.py:137/153/165`, `streamingdatawidthconverter_rtl.py:106/122/134`, `inner_shuffle_rtl.py:105/120/132` — with inconsistent ordering). The substrate templates embed `$::env(FINN_ROOT)` directly in generated TCL (`templates.py:134/136/331`) and hard-code the full `ip_gen_loop_op` rtllib subtree list (`:331–357`). Any reorganisation of the RTL library silently breaks IP-gen. The `os.listdir`-scan variant is worse than a fixed path because it is also non-deterministic: `matrixvectoractivation.py:983` picks the **last** file ending `_memstream_wrapper.v` (no `break`); same non-breaking scan at `vectorvectoractivation.py:826`, `elementwise_binary_hls.py:736`, `thresholding_hls.py:675`. External headers not vendored in-tree (`split.hpp` `split_hls.py:47`, `softmax.hpp` `hwsoftmax_hls.py:29`, `input_gen.hpp` `outer_shuffle_hls.py:63`, `checksum.hpp` `checksum_hls.py:145`) mean correct codegen depends on an ambient `-I` include path — the module's real dependency is invisible to the module.

### 1.6 `code_gen_dict` side-channel + module-mutable-state (1 + 8)

HLS codegen is a sequence of methods (`global_includes`→`defines`→`docompute`→…) that communicate **only** by mutating one shared instance dict, populated then cleared at `hlsbackend.py:136/154/175/244`. Steps are non-composable: you cannot run or substitute `docompute` in isolation because it consumes state another method deposited. The family-level `module-mutable-state` hits are the same pathology at op scope: `requant_hls.py:59` (`_scale_is_one`/`_bias_is_zero` written in `generate_params`, read in `docompute`), `elementwise_binary.py:349` (`minimize_weight_bit_width` silently flips `lhs_style`/`rhs_style`, driving all downstream branching), `labelselect.py:52` and `inner_shuffle_rtl.py:51` (constructors mutate node attrs as a side effect).

### 1.7 Order-dependence (6)

The consequence of §1.5–1.6: methods that silently break unless a prior method ran first. `fmpadding_rtl.py:121`, `streamingdatawidthconverter_rtl.py:90`, `vectorvectoractivation_rtl.py:207`, `convolutioninputgenerator_rtl.py:842` all set `gen_top_module` in `generate_hdl` that `get_rtl_file_list`/`code_generation_ipi` later require; `lookup_hls.py:73` gates a URAM check on one specific entry point; `split_hls.py:88` multi-output drain only activates if a prior transform set `hls_style='freerunning'`.

---

## 2. Bounded contexts

The 27 families cluster into seven bounded contexts, each with its own shared vocabulary. The current 2-axis abstraction (`HWCustomOp` + `{HLSBackend|RTLBackend}`) cuts *across* these clusters rather than aligning with them, which is why several contexts need an orthogonal third axis the substrate does not model.

| Context | Families | Shared vocabulary (ubiquitous language) | Missing axis |
|---|---|---|---|
| **A. Weight/parameter-bearing compute** | MVAU, VVAU, Thresholding, Elementwise, Requant, LayerNorm | `MW/MH/PE/SIMD`, `mem_mode` (embedded/decoupled/external), `accDataType`, `make_weight_file`, `calc_wmem`/`calc_tmem`, `.dat`/`memblock.dat`, memstream/dynload/`mlo_max_iter`, `narrow_weights` | weight-delivery / memory-mode |
| **B. Streaming reshape & routing** | DWC, Concat, Split, DuplicateStreams, Crop | `SIMD` folding, `ChannelsPerStream`, `NumOutputStreams`, stream width, dtype-preserving, `hls_style='freerunning'` drain | output arity |
| **C. Windowing / spatial** | ConvolutionInputGenerator (SWG), FMPadding, FMPadding_Pixel, Upsampler | `IFMDim`/`OFMDim`, `Stride`, `Dilation`, `Kernel`, `ImgDim`, `numInputVectors`, NHWC, im2col, `impl_style` | impl-style + dynamic reconfig |
| **D. Reduction / classification** | Pool, GlobalAccPool, LabelSelect, HWSoftmax | reduction axes, onnxruntime/scipy golden model, TopK/softmax, `Function` string dispatch | behavioural-ref vs backend-exec |
| **E. Embedding lookup** | Lookup | `mem_mode` (ROM vs external DMA), embeddings `.hpp`/`.dat`, Gather golden model | memory-strategy |
| **F. Backend-only shell infrastructure** | IODMA, Checksum, TLastMarker | `direction` (in/out), AXI-MM / AXI-Lite, TLAST/TKEEP sideband, `StreamWidth`/`ElemWidth` bits, **no agnostic base, no functional model** | scalar/control ports; direction as role |
| **G. Meta / container** | FINNLoop, StreamingDataflowPartition | subgraph-as-nodeattr (`body`/`model`), recursive `execute_onnx`, block-design assembly, child-op introspection | hierarchical/container node concept |
| **(H. Transpose, decomposed)** | Shuffle → InnerShuffle (RTL-only), OuterShuffle (HLS-only) | `transpose_out_shape`, loop-nest coeffs, decomposition/lowering | op lowering/decomposition |

Notes on context internal-consistency:

- **Context A** is the largest and least hermetic. `mem_mode` is an orthogonal third axis (embedded array vs decoupled memstream vs external vs `dynamic_input`/dynload vs MLO `fetch_weights`) that the HLS/RTL split does not capture, so every method re-branches on it. Requant and LayerNorm are a sub-dialect ("embedded parameter input": scale/bias/thresholds as constants not streams) that forces `get_instream_width→0` for `ind!=0` and full `execute_node` re-implementations in all three layers (`requant.py:136`, `requant_hls.py:253`, `requant_rtl.py:169`).
- **Context F** ops deliberately reject the substrate contract: four of the eight `HWCustomOp` abstract getters raise `Exception` (`tlastmarker_hls.py:211–223`), `execute_node` is `pass` (`iodma_hls.py:388`), and folded-shape getters raise on the AXI-MM side (`iodma_hls.py:116/131`). They are shell/plumbing primitives mis-filed under the tensor-op taxonomy — a `bounded context` that shares almost no vocabulary with A–E.
- **Context G** nodes inherit the wrong base entirely: `StreamingDataflowPartition` extends qonnx `CustomOp` directly (`streamingdataflowpartition.py:37`), `FINNLoop` stubs its required `RTLBackend.get_rtl_file_list` to `None` (`finn_loop.py:1175`). Neither is a leaf kernel; both need a first-class container concept.

---

## 3. Where contexts leak into each other

The seams between contexts are not clean adapters — they are hard couplings that make the map non-modular. Ranked by blast radius:

**L1 — Context A leaks into the substrate.** The single most damaging leak. The weight-delivery vocabulary of Context A (`memstream`, `fetch_weights`, `dynload`, `calc_wmem`/`calc_tmem`, `mlo_max_iter`) is embedded *in the substrate* (`hwcustomop.py:307/355/407/100`). The substrate — which should be context-free and shared by all seven clusters — hard-codes the names and internal methods of exactly one context. Every other context inherits this dead weight, and Context A cannot be substituted or extended without editing the shared base. This is where the "bounded context" model breaks: A has no boundary; it is smeared into the substrate.

**L2 — HLS↔RTL backend leak inside Context A.** The HLS/RTL axis does not partition Context A. HLS ops emit RTL weight-streamers: `MVAU_hls.code_generation_ipgen` (`matrixvectoractivation_hls.py:144`) calls `generate_hdl_dynload/memstream/fetch_weights`; `Thresholding_hls.code_generation_ipgen` (`thresholding_hls.py:181`) and `Elementwise` (`_hls.py:141` / `_rtl.py:177`) do the same. A single "HLS" op straddles both backends — 13 cross-backend-leak hacks total. The seam meant to separate HLS from RTL runs *through* the op, not around it.

**L3 — Context C ↔ Context A bidirectional coupling.** The windowing context and the compute context reach into each other's internals: SWG hard-codes VVAU's PE-interleaved data layout in its cppsim output (`convolutioninputgenerator_rtl.py:321`), and VVAU inspects its producer's `op_type` (`Im2Col`/`ConvolutionInputGenerator`) to reconstruct PE (`vectorvectoractivation.py:126`). Neither can be substituted without the other's layout convention.

**L4 — Context G reaches into Context A internals.** `FINNLoop` (Context G) hard-codes Context A's private param-file naming (`memblock.dat`, `<node>_threshs_<pe>_<stage>.dat`) and op-type dispatch (`finn_loop.py:419/421/712/558`), and reads child `mlo_max_iter`. The container context depends on the private filesystem contract of the compute context — a leak across two context boundaries at once.

**L5 — Context H's "agnostic" base *is* a backend cost model.** `OuterShuffle.get_exp_cycles` + `_NestSim` (`outer_shuffle.py:20–238`) is a full Python re-simulation of the HLS `input_gen.hpp` pipeline; `InnerShuffle.get_exp_cycles` bakes in the RTL double-buffered-BRAM formula (`inner_shuffle.py:94`). The supposedly backend-agnostic base is fused to one backend's microarchitecture, so the base cannot be reused for the other backend — the HLS/RTL seam has collapsed into the base.

**L6 — Context D drags heavyweight runtimes into the agnostic layer.** HWSoftmax (`hwsoftmax.py:12` scipy), Upsampler (`upsampler.py:30` onnxruntime), LabelSelect (`labelselect.py:155`), Lookup (`lookup.py:30`) import inference engines at module scope purely for golden `execute_node`s, coupling the shape/datatype abstraction to external execution runtimes. Combined with the pervasive `execute_node` MRO collision (base numpy/torch model vs `HLSBackend.execute_node`, patched by hand in ≥8 families), this shows Context D really needs a *separate* behavioural-reference vs backend-execution seam that the single `execute_node` name does not provide.

**Synthesis for redesign.** Substitution is blocked at three structural levels: the substrate names one context (L1), the backend axis leaks inside a context (L2), and contexts reach into each other's private state (L3–L6). A hermetic redesign must (a) evict Context A's weight-delivery vocabulary from the substrate into a pluggable memory-strategy module, (b) make `finnxsi`, `FINN_ROOT`, the Vitis toolchain paths, and the finn-rtllib layout **injected inputs** rather than ambient globals, (c) replace the `code_gen_dict` side-channel with composable, individually-substitutable codegen steps, and (d) model Contexts F and G as distinct node categories (shell-interface primitives; hierarchical containers) outside the leaf HLS/RTL kernel taxonomy.


---


# Seam Map + Op↔finn-rtllib Coupling Surface

*As-is model of where a backend variant can be substituted cleanly versus where base and backend logic are welded, plus the aggregated surface where RTL ops parameterize `finn-rtllib`. Built on the Phase A core-interface contract and the Phase B census (`_census_digest.json`, 27 families). Vocabulary per `/modular-dev`: a **seam** is a boundary where a module can be substituted without editing its neighbors; a **fused** point is a boundary that has collapsed — the modules cannot be separated.*

---

## 1. Clean seams — where backend substitution is (nearly) local

A seam is *clean* when the agnostic base carries the full op semantics (the 8 `HWCustomOp` getters + a functional model) and the backend class only fills its contract (4 `HLSBackend` abstracts or 3 `RTLBackend` abstracts) without the base reaching back into it. In these families the base has **zero rtllib coupling, zero cross-backend leak, and no base-class-leak hack** — a new backend variant could in principle slot in by implementing the backend contract alone.

The clean set (all HLS-only today, one HLS variant + agnostic base, no RTL sibling to stress the seam):

- **globalaccpool** — agnostic base owns all 8 contract methods + python model; HLS variant is codegen-only; no rtllib, no hermeticity violations.
- **concat (StreamingConcat)** — base has zero HLS/RTL leakage; HLS variant is pure codegen.
- **crop** — dtype-preserving streaming op, no env/filesystem/global state, no rtllib.
- **fmpadding_pixel** — thin HLS backend fills only the four hooks; kernel from hlslib.
- **upsampler** — well-factored base + thin HLS variant; "a new backend would slot in without touching the base."
- **duplicatestreams**, **pool**, **hwsoftmax**, **labelselect**, **split** — same shape: agnostic base + single HLS variant, no rtllib coupling.

**The one wart these clean seams share** (`inheritance-irregularity`, 24 occurrences aggregate): the `execute_node` diamond. The agnostic base carries a functional/golden model under the name `execute_node`; the backend carries cppsim/rtlsim under the *same* name; and because concrete ops are declared `(OpBase, HLSBackend)`, Python MRO resolves `execute_node` to the base's python model. Every one of these families patches it with a 2-line shim that explicitly re-dispatches to `HLSBackend.execute_node` (`pool_hls.py:116`, `crop_hls.py:88`, `duplicatestreams_hls.py:68`, `upsampler_hls.py:93`, `fmpadding_pixel_hls.py:91`, `hwsoftmax_hls.py:82`, `split_hls.py:43`). So even the *cleanest* seam is not clean at the execution boundary: which `execute_node` runs is decided by base-ordering accident, not policy. **This is the single most repeated seam defect in the census** and the strongest signal that "functional reference" and "backend execute" are two contract slots masquerading as one.

**Caveat on cleanliness:** these are clean only because they have exactly one backend. No RTL sibling exists to test whether the base truly generalizes. Where a second backend *does* appear (DWC below), the base's silent assumptions are immediately exposed.

---

## 2. Fused (no-seam) points — where base and backend are welded

~50 `fused_seams` entries across the census reduce to **six recurring patterns**. In each, you cannot substitute a backend without editing the agnostic base (or a sibling), because logic that belongs to one module lives in another.

### Pattern F1 — Agnostic base calls backend-only methods (the structural inversion)
The base class is *uninstantiable by contract*: it invokes methods that exist only on its subclasses.
- **MVAU/VVAU** (blocker): `MVAU.code_generation_ipi` (~236 lines of Vivado IPI TCL) lives in the backend-agnostic base but calls `self.instantiate_ip()`, defined *only* on `MVAU_hls:681` / `MVAU_rtl:163` (`matrixvectoractivation.py:920`). The "agnostic" op structurally depends on its subclasses — an inverted dependency.
- **streamingfifo** (blocker ×2): the agnostic base reaches `self.get_adjusted_depth()` and `self.get_nodeattr('impl_style')` — both RTL-subclass-only — wrapped in `try/except AttributeError` (`streamingfifo.py:90,104`, repeated 11×). The base is "not backend-agnostic at all."
- **convolutioninputgenerator (SWG)**: base `get_folded_output_shape`/`get_outstream_width` call `use_parallel_window_output()`, defined only on the RTL subclass (`convolutioninputgenerator.py:124`). The bare base raises `AttributeError`.

This is the same defect the Phase A model flagged as the "base-knows-subclasses leak" (`HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}`), reproduced at the *per-op* level. There is no seam here at all — base and backend are one welded object split across two files.

### Pattern F2 — A hidden third axis cutting across HLS/RTL (weight/memory delivery)
The 2-axis model (HWCustomOp × {HLS,RTL}) does not have a slot for **how parameters reach the compute core**, yet this is the single largest source of fusion:
- **MVAU**: weight delivery (`internal_embedded` / `internal_decoupled`/memstream / `external` / `dynamic_input`/dynload / `mlo_max_iter`/fetch_weights) is emitted from *both* HLS (`code_generation_ipgen:144-155`) and RTL (`generate_hdl:308-322`) and stitched by the shared base `code_generation_ipi`. "There is no seam separating 'compute backend' from 'weight-delivery backend'."
- **thresholding**: `mem_mode` (embedded vs decoupled) smeared into the agnostic base via `try/except` (`thresholding.py:179,310`); the HLS variant emits and wires finn-rtllib RTL streamers, so "the HLS/RTL axis does not cleanly partition this op."
- **elementwise_binary**: const operand can be embedded / decoupled-streamed / MLO-streamed — cutting across HLS/RTL and forcing the `startswith('Elementwise')` base leak.
- **VVAU**, **lookup** (`internal_embedded` ROM vs `external` DMA — "effectively a second op"): same shape.

The welding mechanism is concrete: `HWCustomOp.generate_hdl_memstream` gates on a hard-coded op_type allowlist `['MVAU_hls','MVAU_rtl','VVAU_hls','VVAU_rtl','Thresholding_hls']` and `startswith('Elementwise')` (`hwcustomop.py:310-311`). The memstream/dynload/fetch_weights datapath *cannot* be separated from the base without either subclass-registration or moving the logic out — it is fused to these op families *by name*.

### Pattern F3 — Block-design/IPI assembly with no contract slot
Vivado IPI block-design assembly (add_files, create_bd_cell, pin wiring, clk2x stopgaps) is bespoke, heavy, and **not expressible through any RTLBackend/HLSBackend abstract method**, so it lands wherever the op author put it — usually duplicated per backend:
- **elementwise_binary**: `code_generation_ipi` memstream-wiring block duplicated between rtl (`:300-364`) and hls (`:731-794`).
- **MVAU/VVAU**: `code_generation_ipi` in the agnostic base interleaves `mem_mode`/mlo/dynamic branching with TCL; "cannot be lifted out without also moving the branching."
- **finn_loop**: the *entire* op is a ~540-line Vivado-TCL block-design assembler (`ipgen_singlenode_code`) fused with child-subgraph introspection — no HLS/RTL axis to abstract at all.
- **streamingfifo** (blocker): `code_generation_ipi` hides a whole *third* backend — a `vivado` branch instantiating Xilinx `axis_data_fifo:2.0` infra IP (`streamingfifo_rtl.py:141`) — multiplexed by the runtime `impl_style` attr inside a class named `*_rtl`.

### Pattern F4 — Divergent numeric algorithm behind one op name
Where an op *does* have two backends, they frequently implement **genuinely different math/capabilities**, not two codegens of one algorithm — so they are not transparently substitutable:
- **requant** (blocker): HLS does float multiply-add + `hls::lrint`; RTL derives fixed-point params in SV from **6-decimal-truncated** literals (`requant_rtl.py:78`). "The 'same op' has two independent implementations that can disagree."
- **streamingdatawidthconverter** (blocker): HLS supports arbitrary widths via an LCM two-stage stream; RTL only supports integer width ratios. The base's `check_divisible_iowidths()` is a no-op `pass` (`streamingdatawidthconverter.py:84`) that "pretends both backends are equal" — backend choice can silently invalidate a node.
- **layernorm** (blocker ×2): datatype attrs are decorative — the RTL wrapper hard-codes `[31:0]` for both TDATA ports (`layernorm_wrapper_template.v:20,25`) and HLS hard-codes `TO=float`, while `get_instream_width` computes from the dtype attr. fp32 is fused across three layers with no single source of truth.

### Pattern F5 — Agnostic base fused to backend microarchitecture via cost models
`get_exp_cycles` and friends encode backend-specific pipeline depths into the "agnostic" base:
- **shuffle** (cross-backend-leak): `OuterShuffle.get_exp_cycles` + `_NestSim` (`outer_shuffle.py:20-238`) are a *complete python reimplementation* of the HLS `input_gen.hpp` pipeline living in the agnostic base; `InnerShuffle.get_exp_cycles` hard-codes the RTL double-buffered-BRAM formula. "Its cost model IS the HLS pipeline."
- **layernorm**: `get_exp_cycles` magic constants (`+7`, `+24`, `+5`) fused to `accuf.sv`/`rsqrtf.sv`/`queue.sv` pipeline depths (`layernorm_rtl.py:134`).
- **MVAU/VVAU**: `_resolve_segment_len` DSP58 timing constants `0.741ns`/`0.605ns` (`matrixvectoractivation_rtl.py:233`, `vectorvectoractivation_rtl.py:240`).

### Pattern F6 — Backend-only ops with no agnostic base (inheritance asymmetry)
Four ops inherit `(HWCustomOp, HLSBackend)` or `(HWCustomOp, RTLBackend)` directly with **no agnostic base** — there is no seam at which a second backend could attach without duplicating every shape/width method: **iodma**, **checksum**, **tlastmarker**, **finn_loop**. tlastmarker makes this explicit — half the 8-method contract raises `Exception('not implemented')` (`tlastmarker_hls.py:211`); the op is opaque to FINN's datatype/shape machinery. These are infrastructure/shell primitives forced through a tensor-op contract.

---

## 3. The op↔finn-rtllib coupling surface

Eleven of 27 families reach into `finn-rtllib`. The aggregated surface is **75 coupling edges** across the census. Three things characterize it: the mechanism is primitive, the module surface is concentrated on a few subtrees, and the binding is convention-only (brittle).

### 3a. Coupling mechanisms (how an op parameterizes RTL)
| Mechanism | Count | What it is |
|---|--:|---|
| **file-copy** | 40 | `shutil.copy` an unmodified `.sv`/`.v` into the codegen dir; parameters flow only through the generated top wrapper's `#(...)` param map |
| **verilog-template-fill** | 15 | read a `*_wrapper_template.v`/`.sv`, `str.replace('$KEY$', val)` in a loop, write `<top>.v` |
| **string-replace** | 11 | same as above but explicitly flagged as raw untyped `$KEY$` surgery (swg, dwc, fifo, requant, eltwise, inner_shuffle) |
| **tcl-instantiate** | 6 | emit `add_files` + `create_bd_cell` TCL to wire the module into the block design |
| **parameter-passing** | 3 | pass DEPTH/WIDTH/RAM_STYLE into a submodule instantiation, or a `.dat` meminit file |

**There is no structured binding anywhere.** All 26 `verilog-template-fill`/`string-replace` edges are the *same* pattern: `template.replace(f'${key}$', str(value))` in a loop over a `code_gen_dict`. No escaping, no validation, silent no-op if a token is renamed or a value contains `$`. The template's `$NAME$` tokens and the Python dict keys are coupled *by spelling only*. This is `template-surgery` (27 aggregate hacks) as the universal op↔rtllib adapter.

### 3b. finn-rtllib module subtrees touched
| Subtree | Edges | Consumers |
|---|--:|---|
| **mvu/** | 14 | MVAU, VVAU (shared `mvu_vvu_axi_wrapper.v` + 6 `.sv` DSP cores) |
| **swg/** | 9 | convolutioninputgenerator (5 templates: default/dynamic/parallel/wrapper/axilite + common/pkg) |
| **memstream/** | 7 | MVAU, VVAU, thresholding, elementwise — the shared weight/param streamer |
| **layernorm/** | 6 | layernorm (wrapper + 5 verbatim `.sv`) |
| **requant/** | 6 | requant |
| **eltwise/** | 6 | elementwise_binary |
| **inner_shuffle/** | 5 | shuffle (inner RTL) |
| **fmpadding/** | 4 | fmpadding |
| **fifo/**, **dwc/**, **thresholding/** | 3 each | streamingfifo, DWC, thresholding |
| **mlo/**, **axi/**, **stream_tap/** | 2 each | MVAU/finn_loop (fetch_weights, loop_control, stream_tap), axilite |
| **dynload/**, **skid/** | 1 each | MVAU dynamic-load, finn_loop/mlo |

`memstream/`, `axi/hdl/axilite.sv`, and `mlo/` are the **shared-infrastructure hubs** — reached by 3-4 unrelated op families each, always through the `HWCustomOp.generate_hdl_*` base leak + per-op `code_generation_ipi` TCL. They are the closest thing to a reusable rtllib "library seam," but access is un-brokered: every consumer independently reads `FINN_ROOT`, globs the directory, and hand-lists source files.

### 3c. Where the coupling is brittle

**Magic numbers baked into `.v`/`.sv` templates (the RTL disregards the Python contract):**
- `layernorm_wrapper_template.v:20/25` — TDATA hard-coded `[31:0]`, silently overriding the `inputDataType`/`outputDataType` attrs (two blockers).
- `requant_rtl.py:78` — `{:.6f}` truncates scale/bias to 6 decimals in generated SV vs full float32 golden model (blocker).
- MVAU/VVAU `$ACCU_WIDTH$` filled from output-dtype bitwidth, not `accDataType` (`matrixvectoractivation_rtl.py:349`, `vectorvectoractivation_rtl.py:282`) — silently wrong when `outputDataType != accDataType`.
- fmpadding AXI-lite register byte offsets `0*4..5*4` hand-mirrored to `axi2we.sv` (`fmpadding_rtl.py:101`); any HW register reorder silently breaks runtime reconfig.

**Triplicated source-file manifests (no single source of truth):** the hard-coded `.sv` list is duplicated across `generate_hdl` / `get_rtl_file_list` / `code_generation_ipi` in **fmpadding** (3×: `137/153/165`), **layernorm** (3×: `69/85/98`), **dwc** (3×: `106/122/134`), and duplicated 2× in **MVAU** (`168` vs `357`), **VVAU** (`152` vs `298`), **elementwise** (`179` vs `194`). Adding or renaming one RTL file requires editing 2-3 Python methods.

**Ambient path resolution (34 `env-var` + 26 `filesystem-path` hermeticity violations):** every rtllib-coupled op reads `os.environ['FINN_ROOT']` — usually 2-4 times per op in independent methods — and hard-codes the relative subpath (`/finn-rtllib/<module>/hdl`). Several then `os.listdir` the codegen dir for a `*_memstream_wrapper.v` and take the **last match with no break** (MVAU `:983,1005,1046`, VVAU `:826`, elementwise `:308`, thresholding `:675`) — order-dependent, nondeterministic if multiple match, `NameError`/`strm_tmpl` undefined if none. MVAU's MLO path even globs `cdma/`, `cdma_a/`, `cdma_u/`, `cdma_x/` for `*.sv` — a directory-content-dependent build.

**Order-dependent node state:** nearly every RTL op mutates `gen_top_module`/`ipgen_path`/`ip_path` nodeattrs in `generate_hdl`, which `get_rtl_file_list`/`code_generation_ipi` then read — the coupling breaks if `generate_hdl` hasn't run first (fmpadding, dwc, layernorm, streamingfifo, SWG, MVAU, VVAU). SWG's `get_dynamic_config` goes further and *permanently rewrites* IFMDim/OFMDim/Stride/Dilation as a side effect of a `get_` accessor (`convolutioninputgenerator_rtl.py:984`).

**Python-generated HDL (no file boundary at all):** SWG's parallel impl-style builds whole (System)Verilog module bodies — `$GENERATE_REG_FIFOS$`, `$GENERATE_BRAM_FIFOS$`, `$GENERATE_OUTPUT_MAPPING$`, `$GENERATE_BUFFER_CONNECTION$` — via `str.format()` inside the Python op (`convolutioninputgenerator_rtl.py:675-771`). There is no clean file boundary between "op logic" and "HDL" for that path; the template `.sv` and the Python are two halves of one algorithm.

---

## Seam-map summary for redesign

- **Clean, low-risk seams:** the 10 single-HLS-backend streaming ops (globalaccpool, concat, crop, fmpadding_pixel, upsampler, duplicatestreams, pool, hwsoftmax, labelselect, split). Their only shared defect is the `execute_node` diamond — fixable globally by splitting "functional reference" from "backend execute" into two contract slots.
- **The seam that must be *created*, not repaired:** a **parameter/weight-delivery module** orthogonal to compute backend (Pattern F2). It is currently fused into `HWCustomOp` by op-type-string allowlist and touches the `memstream/`, `mlo/`, `dynload/` hubs from MVAU, VVAU, thresholding, elementwise, lookup simultaneously.
- **The seam that must be *lifted out*:** block-design/IPI assembly (Pattern F3) has no contract slot today and is duplicated per backend or welded into the agnostic base; a backend-owned "stitching" layer would absorb it.
- **The op↔finn-rtllib adapter needs replacing wholesale:** 26 template-fill edges are untyped `$KEY$` `str.replace`; the source manifest is triplicated per op; path resolution is ambient `FINN_ROOT` + directory globbing. A declarative rtllib-source manifest + typed template/parameter binding would collapse the 60 env-var/filesystem hermeticity violations and the brittle magic-number-in-`.v` failures (layernorm, requant, fmpadding) into one brokered seam.


---


# Completeness Critique & Redesign-Pressure List


*Independent critic pass over the four sections above, checking for unmodeled families, unsupported claims, and contradictions, and consolidating the ranked pain points.*


I've cross-checked all four sections against the census matrix, the core-interface contract, the 21-blocker list, and the per-family `redesign_pressure` fields. Findings below.

## (1) Gap list — what to add / fix

**Blocker hacks not (or barely) reflected in the model:**
- **VVAU rtlsim indentation correctness bug is entirely missing.** Blocker #11 (`vectorvectoractivation_rtl.py:89`) — the whole rtlsim block is indented *inside* the `for inputs in node.input:` loop, so rtlsim re-runs 2-3× per node and `sim`/`export_idt` leak across iterations. This is one of only two genuine correctness blockers in the set, and none of the four sections mentions it. Add it — it is the direct symptom of the missing shared HLS/RTL execute harness (Pattern F1/F4).
- **ElementwiseBitShift_hls `get_nodeattr_types` MRO trap under-covered.** Blocker #15 (`elementwise_binary_hls.py:1070`) — diamond MRO drops the `direction` attr unless explicitly re-dispatched. Sections only cover the *`execute_node`* diamond; this shows the diamond defect also hits **attribute schema**, not just execution. Call it out as a second face of the inheritance-diamond problem.

**Modeling gaps:**
- **No consolidated ranked redesign-pressure list exists.** Section 1 gives "circular edges," Section 3 gives L1-L6 "by blast radius," Section 4 gives a "seam-map summary" — but there is no single MVAU-first ranked list a redesign can act on. Drafted below in (2).
- **Non-stream port dimension is scattered.** Context B captures "output arity" (fan-out streams), but the **scalar/AXI-Lite-control/AXI-MM/sideband port** axis is only in a Context-F footnote. It actually spans checksum (2-output + `chk/drain` AXI-Lite reg, 32-bit magic in 4 places), iodma (direction-bifurcated AXI-MM, folded-shape getters *raise*, `iodma_hls.py:116/131`), tlastmarker (TLAST/TKEEP sideband), fmpadding (`axi2we.sv` register offsets, `fmpadding_rtl.py:101`), and SWG (axilite dynamic reconfig). Consolidate as a first-class "non-tensor port" gap.
- **FIFO characteristic-function subsystem unmodeled.** `HWCustomOp.derive_characteristic_fxns` (`hwcustomop.py:440`, 125 lines, writes `io_chrc_*.npy` sidecars, stores abs paths as nodeattrs `hwcustomop.py:556`) is a substrate concern touched by every op and carries hermeticity smells; `duplicatestreams` hard-codes exactly 2 output streams in it. No section addresses it.
- **`streamingdataflowpartition` distinctiveness thin.** It inherits qonnx `CustomOp` *directly* (`streamingdataflowpartition.py:37`), owns none of the 8 contract methods, and has a stale `TODO: move to HLSCustomOp base` (line 34). Covered only in a Context-G note; it is a *different* kind of outlier than the 4 backend-only ops and deserves its own "graph-partition node, wrong package" line.

**Contradictions / unsupported claims:**
- **Override total is off.** Sections 2 & 5 say "~570 overrides"; summing the census-matrix column gives **~597**. Minor, but fix the number.
- **Task-prompt "351 hacks" vs census "331."** The four sections correctly use **331** (matches `census-matrix.md` total and the taxonomy sum). The "351" in the task's established-facts is the outlier — sections are right, don't propagate 351.
- Numbers that *do* check out (verified): substrate smells 31 (9+11+4+7); family hermeticity 106 (34+26+22+8+8+6+2); rtllib edges 75 (both the per-family column sum and the mechanism table 40+15+11+6+3); base-class-leak 10, cross-backend-leak 13, 21 blockers. No cross-section contradictions on abstract-method counts (8 / 4 / 3).

**Modality coverage (all four present, confirmed):** selection layer ✔ (Section 1), rtllib coupling ✔ (Section 4 §3), `code_gen_dict` side-channel ✔ (Section 2 §3 / Section 3 §1.6), backend-only ops ✔ (Section 1 §4a / Section 3-F / Section 4-F6). All 27 families appear in the Context table — none entirely unmodeled.

## (2) Consolidated Redesign-Pressure List (top 10, MVAU first)

| # | Pain point | Anchors | Blocks |
|---|---|---|---|
| 1 | **MVAU weight-delivery is an unmodeled 3rd axis; agnostic base is uninstantiable-by-contract** (236-line IPI TCL in base calls subclass-only `instantiate_ip`; HLS op emits RTL weight-streamers) | `matrixvectoractivation.py:920`; `matrixvectoractivation_hls.py:144`; `hwcustomop.py:307/355/407` | Substitution of compute vs weight-delivery; the "instantiation mess" root |
| 2 | **Substrate branches on subclass op_type** (`generate_hdl_{memstream,fetch_weights,dynload}` hard-code allowlist + `startswith`) | `hwcustomop.py:310/311/359/407` | Adding/renaming/removing *any* op requires editing the shared base |
| 3 | **`SpecializeLayers` god-switch** — per-op RTL-feasibility ladder + stringly-typed variant resolution; duplicated dtype pre-gates that can drift | `specialize_layers.py:40-211, 399-405`; drift `:60/:69` vs `:275` | A 6th copy of per-op knowledge; no `rtl_feasible()` contract |
| 4 | **`execute_node` diamond** — functional-reference vs backend-exec collide under MRO; ~8 hand-shims; latent correctness bug when un-shimmed | `pool_hls.py:116`, `crop_hls.py:88`, `duplicatestreams_hls.py:68`, +5; **VVAU indent bug `vectorvectoractivation_rtl.py:89`** | Most-repeated census defect; two contract slots masquerading as one |
| 5 | **`code_gen_dict` mutable side-channel** — 4 HLS abstracts are ordered side-effecting steps over one dict; op-level `module-mutable-state` (8) | `hlsbackend.py:136/154/175/244`; `requant_hls.py:59`; `elementwise_binary.py:349` | Non-composable, non-unit-testable codegen |
| 6 | **`mem_mode` / memory-strategy 3rd axis** (embedded/decoupled/external/MLO ROM-vs-DMA) re-branched everywhere; base sniffs subclass name | `thresholding.py:179`; `vectorvectoractivation.py:617`; `lookup.py` (ROM/DMA); `elementwise_binary_hls.py:141` | The HLS/RTL axis does not partition these ops |
| 7 | **Backend-only ops, no agnostic base, contracts violated** (iodma/checksum/tlastmarker/finn_loop) | `tlastmarker_hls.py:211` (4/8 getters raise); `iodma_hls.py:388` (`pass`); `finn_loop.py:1175` (`get_rtl_file_list`→None) | No seam for a 2nd backend; contract "satisfied" only nominally |
| 8 | **op↔finn-rtllib adapter is convention-only** — 26 untyped `$KEY$` `str.replace`, triplicated source manifests, ambient `FINN_ROOT` + `os.listdir` "last-match" globbing, magic numbers in `.v` overriding the Python dtype contract | manifests `fmpadding_rtl.py:137/153/165`; glob `matrixvectoractivation.py:983`; `layernorm_wrapper_template.v:20/25`; `requant_rtl.py:78` | 60 env-var/filesystem hermeticity violations; silent dtype/precision mismatch |
| 9 | **Hidden "vivado" 3rd backend inside `*_rtl`, base reach-through via try/except** | `streamingfifo_rtl.py:141`; `streamingfifo.py:90/104` | Backend selected by runtime `impl_style` attr, not class hierarchy; base inseparable from subclass |
| 10 | **Cross-context couplings + ambient singletons** — SWG↔VVAU layout, finn_loop→sibling `.dat` naming, shuffle cost-model fused to HLS pipeline (`XILINX_VIVADO` crash), `finnxsi` import-time singleton | `convolutioninputgenerator_rtl.py:321` / `vectorvectoractivation.py:126`; `finn_loop.py:421/712`; `outer_shuffle.py:20/189`; `hwcustomop.py:39` | Ops can't be built/tested/simulated in isolation; rtlsim can't be injected/faked |

*(Below the top 10, secondary pressures the sections already surface: `execute_node`/attr diamond as a 2nd face of MRO (#4/gap); block-design IPI assembly has no contract slot (Pattern F3, `finn_loop` ~540 lines); divergent numeric algorithms behind one op name (requant, DWC, layernorm — Pattern F4); non-stream/AXI-Lite port dimension (checksum/iodma/tlastmarker/fmpadding); characteristic-fxn subsystem on the substrate.)*

