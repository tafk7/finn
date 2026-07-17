# Design: Evolutionary control (baseline) — the smallest, most surgical set of changes to the CURRENT structure (HWCustomOp substrate + HLSBackend/RTLBackend mixins + specialize_layers registries) that satisfies must-fix items 1-10, with NO clean-slate rewrite. The class hierarchy, the name-keyed registries, and the multiple-inheritance (OpBase, Backend) shape are all preserved; the fixes are targeted extractions, method-splits, and a registered-predicate table.

> **Essence:** Keep the mixin substrate and the name-keyed variant registries exactly as they are, but (a) evict the three op_type-branching generate_hdl_* methods into an injected WeightDelivery collaborator, (b) split execute_node into two named contract slots, (c) replace the specialize_layers ladder with a feasible() classmethod queried by an op-agnostic dispatcher, and (d) add a third staticip registry as a peer to hls/rtl — eight independently-landable refactors, each green against the 27-family regression.

**Selection strategy:** `kernel-self-advertises`

## Core model


# Evolutionary Baseline — "The smallest diff that passes the test suite"

## 0. Stance and what is deliberately NOT touched

This is the control. Its job is to establish the bar: any clean-slate design must be clearly better than this, or the rewrite isn't justified. So I preserve every structural asset that already works and only cut where a must-fix criterion forces it.

**Preserved as-is (zero change):**
- The three-tier inheritance shape: `HWCustomOp(CustomOp)` substrate -> agnostic op module -> `{HLSBackend|RTLBackend}` mixin -> concrete `(OpBase, Backend)` leaf.
- The name-keyed registries: `hls/__init__.py::custom_op` and `rtl/__init__.py::custom_op` dicts populated by the `@register_custom_op` decorator (`hls/__init__.py:32-49`).
- The 8 shape/datatype abstracts on `HWCustomOp` (`hwcustomop.py:261-289`) — the census confirms these are the one clean part of the contract; kept verbatim.
- `SpecializeLayers` stays a QONNX `Transformation` that rewrites nodes in place.

**The insight that makes a surgical fix viable:** the census shows the 10 pressure points are NOT diffuse — they concentrate in ~5 code sites in 4 files. The substrate leak is exactly 3 methods (`hwcustomop.py:307/355/407`). The selection god-switch is exactly one function (`specialize_layers.py:40-211`). The execute diamond is one method name colliding under MRO. You do not need a new abstraction to fix a localized leak; you move it behind a declared seam. That is the whole design.

```
                    BEFORE (as-is)                          AFTER (evolutionary)
   HWCustomOp                                    HWCustomOp
   |- 8 shape abstracts       (KEEP)             |- 8 shape abstracts           (unchanged)
   |- generate_hdl_memstream  -+ op_type         |- execute_node -- dispatcher --+  (NEW: split)
   |- generate_hdl_fetch_wts   | allowlist       |     |- "functional" -> execute_reference()
   |- generate_hdl_dynload    -+ LEAK (crit 2)   |     +- cppsim/rtlsim -> execute_impl()
   |- execute_node  <- MRO diamond (crit 4)      |- get_param_delivery() -> None  (NEW hook)
   +- finnxsi (import singleton, crit 10)        +- build_ctx  (INJECTED, crit 10)
        (OpBase, HLSBackend|RTLBackend)                      |
                                                 WeightDelivery collaborator  (NEW, crit 1/2/6)
   specialize_layers._determine_impl_style       |  {Embedded, DecoupledStream, External,
     if optype=="MVAU": _mvu_rtl_possible(...)   |   DynamicLoad, MLOFetch} x {ROM,BRAM,URAM,DMA}
     if optype=="VVAU": _vvu_rtl_possible(...)   |  owns the 3 evicted generate_hdl_* methods
     ...  6-arm ladder (GOD-SWITCH, crit 3)      |
                                                 specialize_layers  (op-AGNOSTIC dispatcher)
   hls_variants / rtl_variants (2 registries)      for kind in preference_order(optype):
                                                       cls = registry[kind].get(optype+"_"+kind)
                                                       if cls and cls.feasible(node,part,model): pick
                                                 hls_variants / rtl_variants / STATICIP_variants (3)
```

## 1. Fix A — split execute_node into two named slots (criterion 4)

**As-is defect:** the agnostic op base defines a numpy/torch golden model named `execute_node`; `HLSBackend.execute_node` (`hlsbackend.py:315`) and `RTLBackend.execute_node` (`rtlbackend.py:88`) define cppsim/rtlsim under the same name. Because a leaf is `(OpBase, Backend)`, MRO silently resolves to the base golden model, so ~8 families hand-write a shim `def execute_node(...): return HLSBackend.execute_node(self,...)` (`pool_hls.py:116`, `crop_hls.py:88`, `duplicatestreams_hls.py:68`, +5). And `vectorvectoractivation_rtl.py:89` has the latent correctness bug where the rtlsim block is nested inside the input loop.

**Surgical fix (one method, one rename, delete 8 shims):**
- Rename each agnostic base golden model `execute_node` -> `execute_reference`.
- Rename `HLSBackend.execute_node`/`RTLBackend.execute_node` -> `execute_impl`.
- Add ONE concrete dispatcher on the substrate:
```python
# hwcustomop.py — the sole execute_node in the MRO
def execute_node(self, context, graph):
    mode = self.get_nodeattr("exec_mode")   # "" | "cppsim" | "rtlsim"
    if mode in ("", "functional"):
        return self.execute_reference(context, graph)   # golden semantics
    return self.execute_impl(context, graph)             # backend run
```
Which code runs is now policy in one place, not an MRO accident. The diamond is gone because the two collided names are distinct contract slots — exactly what brief §2.4 demands. The VVAU indent bug is fixed for free by centralizing the loop in the shared `execute_impl` harness. Highest-value refactor; ~11 files mechanical.

Second face of the diamond (blocker #15, `elementwise_binary_hls.py:1070`): `get_nodeattr_types` drops attrs under MRO. Same discipline — an assertion in `register_custom_op` that a class `get_nodeattr_types` cooperates with `super()` (key-set check at registration). A lint, not a redesign; honest limitation in weaknesses.

## 2. Fix B — evict the substrate leak into a WeightDelivery collaborator (criteria 1, 2, 6)

**As-is defect (the root):** `HWCustomOp.generate_hdl_memstream/fetch_weights/dynload` (`hwcustomop.py:307/355/407`) branch on a hard-coded op_type allowlist `["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"]` + `startswith("Elementwise")` and call `self.calc_wmem/calc_tmem/ram_style/MW/MH/PE/SIMD` — none defined on the base. This is the "MVAU instantiation mess."

**Surgical fix — Extract Collaborator (canonical refactoring), not a rewrite:**
```python
class WeightDelivery:                 # base: embedded is the trivial default
    def __init__(self, host: "WeightDeliveryHost", memory: "MemoryStrategy"): ...
    def generate_hdl(self, build_ctx): ...    # emits memstream/fetch/dynload verilog
    def stitch_ipi(self, cmd, host): ...      # the IPI wiring for this delivery
    def make_param_file(self): ...            # unifies make_weight_file/threshold serialize

class EmbeddedDelivery(WeightDelivery):        ...  # params in compute array, no HDL
class DecoupledStreamDelivery(WeightDelivery): ...  # owns old generate_hdl_memstream body
class ExternalDelivery(WeightDelivery):        ...
class DynamicLoadDelivery(WeightDelivery):     ...  # owns generate_hdl_dynload body
class MLOFetchDelivery(WeightDelivery):        ...  # owns generate_hdl_fetch_weights body
```
`MemoryStrategy` is the second facet (criterion 6): `{LUTROM, BRAM, URAM, OffChipDMA}` carrying `ram_style` + read discipline; a FIELD of the delivery object, not a new branch. Thresholding embedded-vs-decoupled and Lookup ROM-vs-DMA become `(delivery, memory)` pairs — one concept, chosen once.

**The host protocol makes the coupling declared instead of leaked.** The strategy needs `MW/MH/PE/SIMD/calc_wmem/ram_style` — declared as a narrow, typed `WeightDeliveryHost` protocol. Substrate residue is only:
```python
# hwcustomop.py — the ONLY residue on the substrate
def get_param_delivery(self):    # default: op has no params
    return None
```
`generate_hdl_memstream/fetch_weights/dynload` are DELETED from `hwcustomop.py`. Criterion 2 becomes literally true: the base has no op_type string anywhere. Criterion 1: weight-delivery is a composed dimension, expressed once, orthogonal to compute-impl.

## 3. Fix C — HLS abstracts become pure fragment-returners (criterion 5)

**As-is defect:** `global_includes/defines/docompute/blackboxfunction` each mutate `self.code_gen_dict` (`hlsbackend.py:136`) in a required order; `thresholding_hls`/`requant_hls.py:59` stash `_scale_is_one` between steps.

**Surgical fix — change 4 signatures, not the architecture:** each abstract returns its fragment dict; the orchestrator merges (`d.update(self.global_includes()); ...`). Each step becomes a pure function `typed inputs -> fragment`, unit-testable in isolation (brief §2.5). Cross-step state passed as explicit returns. A one-line back-compat shim (`if ret is None: ret = self.code_gen_dict`) lets un-migrated ops land the change file-by-file.

## 4. Fix D — typed rtllib adapter (criterion 8)

**As-is defect:** 26 untyped `template.replace(f"${k}$", str(v))` loops, triplicated source manifests (`fmpadding_rtl.py:137/153/165`), `os.listdir`-last-match globs (`matrixvectoractivation.py:983`), magic `[31:0]` in `.v` overriding the dtype contract.

**Surgical fix — one helper + one declared manifest per op:** `RtlTemplate.render(params)` validates every `$KEY$` against a typed param dict (raises on drift). Each RTL op declares `RTL_SOURCES = [...]` ONCE as a class attribute; `generate_hdl`/`get_rtl_file_list`/`code_generation_ipi` all read that single list (kills triplication). `RtllibLocator` is injected (holds finn-rtllib root) replacing ambient `FINN_ROOT` + globbing — collapsing the 60 env/filesystem hits. Magic-number-in-`.v` cases surface as validation errors because render() now requires the dtype-derived param.

## 5. Fix E — registered-predicate selection + third registry (criteria 3, 9)

**As-is defect:** `_determine_impl_style` (`specialize_layers.py:40-211`) is a 6-arm `if optype==` ladder calling `_mvu_rtl_possible` etc., with drift between inline pre-gates (`:60/:69`) and predicates (`:275`). Variant chosen by string concat `optype+"_"+impl_style` (`:399`).

**Surgical fix — invert control into a feasible() classmethod (kernel-self-advertises):**
```python
class RTLBackend:
    @classmethod
    def feasible(cls, node, fpgapart, model) -> bool:
        return True          # default: if a variant is registered, it's usable
```
Each op with a ladder arm MOVES its predicate onto its own class: `MVAU_rtl.feasible` IS the old `_mvu_rtl_possible` + the `bitwidth()>=4` pre-gate, co-located so drift is impossible. `specialize_layers` becomes op-agnostic:
```python
def _determine_impl_style(node, fpgapart, model):
    for kind in preference_order(node):        # tiny data table, e.g. ["rtl","staticip","hls"]
        cls = REGISTRY[kind].get(node.op_type + "_" + kind)
        if cls and cls.feasible(node, fpgapart, model):
            return kind
    raise NoVariant(node)
```
The ladder is deleted. `preference_order` is a small declarative policy dict.

**Criterion 9 falls out for free:** the streamingfifo "vivado" branch (`streamingfifo_rtl.py:141`) becomes a first-class registered variant `StreamingFIFO_vivado` selected by its own feasible() — no runtime attr reach-through, no base try/except. Same mechanism as static IP.

**REGISTRY = `{"hls": hls_variants, "rtl": rtl_variants, "staticip": STATICIP_variants}`** — adding the third key is the only new structure, same shape as the two that exist. The crux: because Fix B removed op_type from the substrate and Fix E made selection a predicate query, a third kind needs ZERO substrate edits.

## 6. Fix F — honest infra & container bases (criterion 7)

**As-is defect:** iodma/checksum/tlastmarker/finn_loop have no agnostic tier; `tlastmarker_hls.py:211` raises in 4/8 getters; `iodma_hls.py:116/131` raise in folded getters; `finn_loop.py:1175` stubs `get_rtl_file_list`->None; `streamingdataflowpartition` extends qonnx `CustomOp` directly (`:37`).

**Surgical fix — two thin intermediate ABCs under the substrate:**
- `InfraKernel(HWCustomOp)`: implements the 8 getters in terms of `get_port_map() -> list[PortSpec]` (typed AXI-MM/AXI-Lite/AXIS/sideband, direction a field). iodma describes its AXI-MM burst honestly instead of raising; tlastmarker declares TLAST/TKEEP.
- `ContainerKernel(HWCustomOp)`: `get_rtl_file_list` legitimately returns `[]`; new `get_body() -> ModelWrapper` + `stitch_children(...)` replaces the pass->None stub. finn_loop sibling dispatch (`finn_loop.py:421/712`) narrows to querying each child declared `get_param_delivery().make_param_file()` instead of hard-coded `.dat` names.

Two ~40-line base classes, not a new taxonomy — leaves keep their mixins.

## 7. Fix G — inject ambient deps (criterion 10)

`finnxsi` (`hwcustomop.py:39`), `fpgapart`, build dir become fields of an injected `BuildContext` threaded through `execute_impl`/`generate_hdl`. The module global is kept as a default so call sites work unchanged until migrated, and rtlsim can accept a fake for isolated testing. The `outer_shuffle.py:189` `XILINX_VIVADO` crash reads from `BuildContext`, not `os.environ`.

## 8. Folding — shared interface only (brief §9.3, in scope)

Add one substrate method `get_folding() -> FoldingSpec` (dataclass: `pe`, `simd`, op-specific `extra: dict`), default derived from `PE`/`SIMD` nodeattrs. Shared vocabulary that resource-est/port-widths/selection depend on. Per-op folding MATH untouched — explicitly out of scope for v1.

## 9. Migration = 8 independently-landable refactors (each green on the 27-family regression)

| PR | Refactor | Criteria | Blast radius |
|----|----------|----------|--------------|
| R1 | Split execute_node -> execute_reference/execute_impl dispatcher; delete 8 shims; fix VVAU indent bug | 4 | ~11 files, mechanical |
| R2 | Extract WeightDelivery/MemoryStrategy; delete 3 substrate methods + op_type allowlist | 1,2,6 | 4 files + 6 weight ops |
| R3 | HLS 4 abstracts return fragments (back-compat shim) | 5 | hlsbackend + incremental |
| R4 | RtlTemplate + RtllibLocator + declarative RTL_SOURCES | 8 | 11 RTL ops, mechanical |
| R5 | feasible() classmethod + op-agnostic dispatcher + staticip registry key | 3,9 | specialize_layers + 6 predicates moved |
| R6 | InfraKernel / ContainerKernel bases | 7 | 5 outlier ops |
| R7 | BuildContext injection (finnxsi/fpgapart/builddir) | 10 | 3 substrate files, default-preserving |
| R8 | Add MVAU_staticip INT8-GEMM proof | forcing fn | +1 file, ZERO substrate edits |

Every PR keeps the registries and mixin shape, so each lands, runs the full op test suite, and merges before the next starts. That is the definition of an evolutionary path.


## Axis treatment

### compute_impl
Stays the multiple-inheritance mixin {HLSBackend, RTLBackend} PLUS a new peer StaticIPBackend, each populating a name-keyed registry (hls_variants/rtl_variants/STATICIP_variants) via the existing @register_custom_op decorator (hls/__init__.py:32). I do NOT unify HLS and RTL into one interface — the census proves they are genuinely different (638 vs 146 LOC, code_gen_dict vs verilog-param), and forcing them together is the modeling fiction to avoid. The three kinds share only the thin feasible()/generate_hdl-or-code_generation_ipgen/code_generation_ipi surface that selection and stitching actually need. Compute kind is chosen by the predicate dispatcher, never by class identity in shared code.

### weight_delivery
First-class, and the biggest structural change. Extracted from the substrate (hwcustomop.py:307/355/407) into a composed WeightDelivery strategy object owning the three former generate_hdl_* bodies, selected by a factory from (mem_mode, dynamic_input, mlo_max_iter). Variants: Embedded / DecoupledStream / External / DynamicLoad / MLOFetch. It couples to its host through a narrow, DECLARED WeightDeliveryHost protocol (MW/MH/PE/SIMD/calc_wmem/ram_style) instead of a substrate op_type allowlist — coupling made explicit and narrow rather than eliminated (honest baseline limitation). Orthogonal to compute_impl: any of the 3 compute kinds composes any delivery.

### memory_strategy
A MemoryStrategy facet {LUTROM, BRAM, URAM, OffChipDMA} carried as a FIELD of the WeightDelivery object (not a separate branch), holding ram_style + read discipline. Thresholding embedded/decoupled and Lookup ROM/DMA become (delivery, memory) pairs chosen once by the factory, replacing the independent mem_mode re-branches at thresholding.py:179 / lookup.py / vectorvectoractivation.py:617. Threshold/weight serialization unified into WeightDelivery.make_param_file(), de-triplicating make_weight_file/minimize_weight_bit_width.

### folding
Shared interface ONLY, per brief §9.3. One new substrate method get_folding() -> FoldingSpec (dataclass: pe, simd, op-specific extra: dict), default derived from existing PE/SIMD nodeattrs. Common vocabulary that resource estimation, port widths, and selection can depend on. Per-op folding MATH explicitly untouched and out of scope for v1 — no unification of SWG/MVAU/Pool folding formulas. The deliberate under-investment the gate decision asks for.

### ports
Two-tier. Tensor streams keep the 8 shape/datatype getters unchanged (the clean part). Non-tensor ports get a new typed get_port_map() -> list[PortSpec] where PortSpec carries kind {AXIS, AXI-MM, AXI-Lite, sideband}, direction, and width — provided honestly by the new InfraKernel base so iodma/tlastmarker/checksum stop raising in the shape getters (iodma_hls.py:116, tlastmarker_hls.py:211). Default get_port_map on the substrate derives AXIS ports from the 8 tensor getters, so existing ops need no change. Minimal, not a full port-graph IR.

### execution
Split into two named contract slots by a single substrate dispatcher execute_node that branches on exec_mode: execute_reference (golden numpy/torch semantics, on the agnostic op) vs execute_impl (cppsim/rtlsim, on the backend mixin). Dissolves the MRO diamond and the 8 hand shims, and fixes the VVAU rtlsim indent bug (vectorvectoractivation_rtl.py:89) by centralizing the harness. rtlsim finnxsi is injected via BuildContext so a fake simulator can drive execute_impl in isolation. Reference vs cppsim vs rtlsim are now three distinct, individually-invocable concerns over the same port contract.

## Conformance sketches

### mvau
Compute and weight-delivery are separated: MVAU_hls/MVAU_rtl/MVAU_staticip are three registered compute kinds; EmbeddedDelivery/DecoupledStreamDelivery/ExternalDelivery/DynamicLoadDelivery/MLOFetchDelivery are the composed delivery axis. The 236-line code_generation_ipi (matrixvectoractivation.py:920) is DECOMPOSED, not deleted: the op-generic hierarchy/clock/port skeleton stays a short method on the agnostic MVAU, and the mem_mode/mlo/dynamic branching that made it 236 lines is delegated to self.get_param_delivery().stitch_ipi(cmd, host) — stitching becomes composition of the compute kind instantiate_ip (still on the leaf, called through the declared contract) plus the delivery stitch_ipi. The 40 overrides collapse because weight-delivery, memory-strategy, and param-file serialization each live in ONE collaborator method. Honest caveat: instantiate_ip still lives on the leaf and is invoked from the agnostic tier — I convert it from an undeclared reach-through (matrixvectoractivation.py:920 calling MVAU_hls:681) into a DECLARED contract method (raising default), the smallest fix that removes the uninstantiable-by-contract inversion without moving the whole IPI body.

### thresholding
mem_mode embedded-vs-decoupled becomes a (delivery, memory) pair from the same factory used by MVAU — the try/except branches at thresholding.py:179/310 are deleted and get_instream_width stops sniffing HLS-only mem_mode state. Threshold serialization is the SAME WeightDelivery.make_param_file() method (thresholds are params with calc_tmem depth), unifying the three divergent copies. Thresholding_hls emitting an RTL weight-streamer (thresholding_hls.py:181) is no longer a cross-backend leak: it is Thresholding_hls composed with DecoupledStreamDelivery, which owns the streamer HDL — HLS/RTL axis and delivery axis are now orthogonal, so HLS compute with RTL-streamed params is a legal expressible point, not a hack.

### finn_loop
Becomes a ContainerKernel(HWCustomOp) — a new thin base making container-ness first-class. get_rtl_file_list legitimately returns [] (a container emits a block design, not leaf RTL) instead of the pass->None stub (finn_loop.py:1175), and prepare_rtlsim (finn_loop.py:277) no longer dodges the abstract. The subgraph is exposed via get_body() -> ModelWrapper; child stitching goes through a stitch_children slot querying each child DECLARED get_param_delivery().make_param_file() and get_port_map() rather than hard-coding .dat names and op_type prefixes (finn_loop.py:421/712). Honestly satisfies the contract because the contract now has a category for containers; no getter raises, no abstract stubbed.

### iodma
Becomes an InfraKernel(HWCustomOp). Its folded-shape getters (iodma_hls.py:116/131) stop raising: InfraKernel implements the 8 getters in terms of get_port_map(), and iodma declares its ports as [PortSpec(AXI-MM, direction=in/out, width=...), PortSpec(AXIS, ...)]. The direction bifurcation becomes a direction field on the PortSpec, not a subclass split. Non-tensor ports are first-class through PortSpec, so the honest shape/port contract holds — iodma is a legitimate infra kernel, not a contract violator forced through the tensor-op taxonomy. checksum (AXI-Lite reg) and tlastmarker (TLAST/TKEEP sideband) migrate the same way.

### static_ip_gemm
The forcing function, and the payoff of the baseline. MVAU_staticip is a new file registering into a new STATICIP_variants dict via @register_custom_op — structurally identical to the existing hls/rtl registries. It implements the thin compute-kind surface: generate_hdl copies the pre-synthesized INT8-GEMM IP, code_generation_ipi instantiates it, and feasible(node, fpgapart, model) returns True only when input+weight dtypes are INT8 and the shape matches the hard IP parameterization. Selection auto-picks it: preference_order lists ['staticip','rtl','hls'] for MVAU, the op-agnostic dispatcher (Fix E) asks each kind feasible(), and static-IP wins for its common case. It composes EmbeddedDelivery (weights baked into the IP) via the SAME delivery axis. Critically: adding it required editing ZERO substrate code — because Fix B removed op_type from hwcustomop.py and Fix E made selection a predicate query, the only new artifacts are one registry dict key and one op file. This is the concrete proof the abstraction is real and not a rename of the HLS/RTL split. If any substrate edit had been needed, criterion 2 would have failed — that it doesn't is what the whole baseline is engineered to demonstrate.

## Criteria self-assessment (must-fix 1-10)

| # | verdict | how |
|--|--|--|
| 1 | **yes** | Weight-delivery extracted into a composed WeightDelivery strategy (Embedded/DecoupledStream/External/DynamicLoad/MLOFetch) owning the 3 former generate_hdl_* bodies; orthogonal to compute kind; the 236-line MVAU IPI (matrixvectoractivation.py:920) becomes op-skeleton + delivery.stitch_ipi composition. |
| 2 | **yes** | The op_type allowlist and startswith checks are DELETED from hwcustomop.py:307/355/407; substrate residue is only get_param_delivery()->None. Adding MVAU_staticip proves zero substrate edits are needed to add a kind/op. |
| 3 | **yes** | specialize_layers ladder (specialize_layers.py:40-211) replaced by an op-agnostic dispatcher querying each variant feasible() classmethod; the 6 per-op predicates move onto their own classes (drift between :60/:69 and :275 becomes structurally impossible). |
| 4 | **yes** | Single substrate execute_node dispatcher delegates to execute_reference (golden) vs execute_impl (backend); 8 MRO shims deleted; VVAU indent bug (vectorvectoractivation_rtl.py:89) fixed by centralizing the harness. |
| 5 | **yes** | The 4 HLS abstracts return typed fragment dicts merged by the orchestrator instead of mutating self.code_gen_dict (hlsbackend.py:136); cross-step state passed as explicit returns; back-compat shim allows incremental per-op migration. |
| 6 | **yes** | MemoryStrategy {LUTROM,BRAM,URAM,OffChipDMA} is a field of the WeightDelivery object; thresholding/lookup/MVAU mem_mode re-branches collapse to one (delivery,memory) choice; make_weight_file/threshold serialization unified in make_param_file(). |
| 7 | **yes** | New InfraKernel and ContainerKernel bases: infra getters implemented via typed get_port_map() so iodma/tlastmarker stop raising; finn_loop get_rtl_file_list returns [] legitimately; streamingdataflowpartition reparented off bare qonnx CustomOp. |
| 8 | **yes** | RtlTemplate.render() validates every $KEY$ against a typed param dict; declarative RTL_SOURCES class attr de-triplicates manifests; injected RtllibLocator replaces ambient FINN_ROOT + os.listdir last-match globs; magic-number-in-.v overrides surface as validation errors. |
| 9 | **yes** | streamingfifo 'vivado' branch (streamingfifo_rtl.py:141) becomes a registered StreamingFIFO_vivado variant selected by its own feasible() — same mechanism as static-IP; no runtime impl_style reach-through, no base try/except. |
| 10 | **partial** | finnxsi/fpgapart/builddir injected via BuildContext (default-preserving), enabling faked rtlsim and fixing outer_shuffle XILINX_VIVADO crash. Partial: the multiple-inheritance mixin diamond itself remains, so full cross-op isolation is bounded by residual host-protocol coupling — see weaknesses. |

## Weaknesses (self-declared)

Honesty is this baseline's entire value, so these are where a clean-slate design can beat it:

1. THE MIXIN DIAMOND SURVIVES. I split execute_node and lint get_nodeattr_types super()-cooperation, but a leaf is still (OpBase, Backend) under multiple inheritance. Any NEW method-name collision between the agnostic tier and a backend mixin reintroduces the MRO accident (blocker #15, elementwise_binary_hls.py:1070, is patched by a registration-time assertion, not structurally prevented). A composition-over-inheritance design (prior 1) eliminates this class of bug by construction; I only whack the two instances the census found.

2. THE HOST->STRATEGY COUPLING IS NARROWED, NOT REMOVED. WeightDelivery still needs MW/MH/PE/SIMD/calc_wmem from its host. I converted an undeclared op_type allowlist into a declared WeightDeliveryHost protocol — strictly better and testable — but the compute op and its weight delivery are still not independently substitutable; they share a live interface. A product-type design (prior 3) that makes delivery a truly free axis pays less here.

3. HLS AND RTL ARE STILL NOT PEERS. I deliberately refuse to unify them (638 vs 146 LOC, code_gen_dict vs verilog-param). StaticIPBackend looks like RTL; HLS keeps its fragment-dict codegen. So 'backend' remains three different-shaped things sharing only a thin selection/stitch surface. An IR/lowering design (prior 4) that makes HLS/RTL/static-IP genuine lowerings of one typed target achieves real uniformity I don't.

4. instantiate_ip STILL LIVES ON THE LEAF and is called from the agnostic MVAU IPI skeleton. I make it a declared contract method (raising default) instead of an undeclared reach-through — the uninstantiable-by-contract inversion is fixed — but the 236-line method is decomposed, not dissolved; a chunk of IPI orchestration remains agnostic-tier code that knows there IS a compute leaf to call.

5. INCREMENTALISM LEAVES A LONG TAIL OF HALF-MIGRATED STATE. The back-compat shims (code_gen_dict fallback, finnxsi default global) mean mid-migration the codebase has TWO ways to do everything. That is the price of safe landability; a clean-slate design has no such interregnum but also no safe rollback.

6. FOLDING AND PORTS ARE INTENTIONALLY SHALLOW. get_folding() is vocabulary-only and get_port_map() defaults derive from tensor getters — adequate for v1 per the gate decisions, but a design that invests in folding/port unification scores higher on Uniformity if the judge weights that axis, and I explicitly chose not to.

## Prior fidelity

This design embodies the evolutionary-control angle without drifting toward the ambitious middle in four concrete ways.

First, it changes NO structural asset that works: the three-tier inheritance, the @register_custom_op name-keyed registries, the QONNX Transformation selection pass, and the 8 clean shape abstracts are all byte-preserved. Every fix is a named, textbook refactoring — Extract Collaborator (Fix B), Rename/Extract Method (Fix A), Introduce Parameter Object (Fix G), Replace Conditional With Polymorphism (Fix E) — applied to a localized site, not a re-conception.

Second, it treats each criterion as a failing TEST to make pass with minimum diff, not a principle to maximize. Where the census showed HLS and RTL are genuinely different, I REFUSE to unify them, because unifying would be a rewrite and the baseline's job is to prove whether a rewrite is even necessary. That refusal is the control condition: if the ambitious designs' unification doesn't buy substantially more than my thin three-kind registry, that is a finding the panel needs.

Third, it makes the third-kind test pass through subtraction, not addition: static-IP plugs in with zero substrate edits precisely because Fixes B and E REMOVED code (the op_type allowlist, the ladder) rather than adding a new abstraction layer. The smallest change that makes the forcing function work is to stop the substrate from knowing op identity — exactly what an evolutionary fix should isolate.

Fourth, the migration is delivered as 8 independently-landable, individually-test-green PRs against the 27-family regression — the defining property of an evolutionary path and the thing no clean-slate design can claim. The deliberately-honest weaknesses (surviving diamond, narrowed-not-removed host coupling, non-peer backends) are left IN rather than papered over, because the baseline's contribution to the panel is a truthful floor: it shows exactly how much residual leak remains after the cheapest possible fixes, so the judge can measure what each ambitious design's extra complexity actually buys.
