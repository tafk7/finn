# Design: Composition over inheritance

> **Essence:** There is exactly ONE final Kernel class in the whole system; all 27 families are instances of it that differ only in the strategy objects they compose (a DataContract + a ComputeStrategy + a WeightDelivery + a MemoryStrategy + a FoldingSpec + a PortSet + an ExecutionHarness), and the substrate is a set of generic drivers that only ever call declared component interfaces — never `isinstance`, never `op_type`.

**Selection strategy:** `hybrid`

## Core model

## The one-sentence thesis

Replace `(OpBase, Backend)` multiple inheritance with **object composition + delegation**. There is a single, `final`, never-subclassed `Kernel` class. A `Kernel` *is* a data-contract holder and *has* six pluggable component objects. Every substrate operation (build, stitch, select, simulate, estimate) is a generic driver that walks a `Kernel`'s components through narrow declared interfaces. Nothing in the substrate branches on op identity, subclass name, or `mem_mode`, because there are no subclasses and no modes — only composed components.

This directly attacks the two disqualifier-class defects:
- The **base-class leak** (`hwcustomop.py:307/355/407`, the `op_type` allowlist at `:310`) exists because a *shared base* holds behavior that belongs to *one context*. Composition has no shared base: weight-streaming behavior lives inside a `WeightDelivery` component, invoked through an interface.
- The **`execute_node` diamond** (`pool_hls.py:116`, `crop_hls.py:88`, +6 shims; latent bug `vectorvectoractivation_rtl.py:89`) exists because MRO collides two meanings of one method name. Composition gives them two *named slots on two objects*: `kernel.reference` (a `ReferenceModel`) and `kernel.compute.runner(mode)` (a `Runner`). No MRO, no shim, no accidental resolution.

---

## 1. The Kernel object (the whole taxonomy is one class)

```python
@final
class Kernel:
    contract:  DataContract      # op SEMANTICS  (per-op, the only irreducible per-family object)
    compute:   ComputeStrategy   # axis 1: how the math is realized  {HLS, RTL, StaticIP, Subgraph}
    weights:   WeightDelivery    # axis 2: how coefficients reach compute
    memory:    MemoryStrategy     # axis 3: where params live + read discipline
    folding:   FoldingSpec        # axis 4: PE/SIMD vocabulary (shared interface)
    ports:     PortSet            # axis 5: tensor AND non-tensor typed ports
    reference: ReferenceModel     # axis 6a: pure golden semantics
    # axis 6b (cppsim/rtlsim runners) is owned BY the ComputeStrategy — see §7
```

`Kernel` has **no** abstract methods and is **never subclassed**. All 27 op families are `Kernel` instances. MVAU is not a class; it is `make_mvau(attrs)` returning a `Kernel` whose `contract` is an `MVAUContract`. This is the strongest possible statement of "favor composition": the type that used to be an inheritance lattice of ~80 classes collapses to *one* concrete class + a library of interchangeable parts.

The `Kernel` implements the 8-method downstream contract (`core-interfaces.md §1`) by **pure delegation** — it owns no compute, no codegen, no simulation:

```python
def get_input_datatype(self, ind=0):   return self.contract.input_dtype(ind)
def get_output_datatype(self, ind=0):   return self.contract.output_dtype(ind)
def get_normal_input_shape(self, ind=0):return self.contract.normal_in(ind)
def get_normal_output_shape(self,ind=0):return self.contract.normal_out(ind)
def get_folded_input_shape(self, ind=0):
    return self.folding.fold(self.contract.normal_in(ind), self.ports.input(ind))
def get_folded_output_shape(self,ind=0):
    return self.folding.fold(self.contract.normal_out(ind), self.ports.output(ind))
def get_instream_width(self, ind=0):    return self.ports.input(ind).stream_width()
def get_outstream_width(self,ind=0):    return self.ports.output(ind).stream_width()
```

Two subtle wins are already visible here:
- **Folded-shape is `folding × port`, not per-op math sprinkled through backends.** `use_parallel_window_output()` (called by the SWG *base* but defined only on the RTL subclass, `convolutioninputgenerator.py:124`) becomes a property of the SWG `FoldingSpec` object that the RTL `ComputeStrategy` supplies at composition time — the `Kernel` never reaches into a subclass.
- **Stream width is a question you ask a `Port`, not a mode you sniff.** MVAU `get_instream_width` branching on `dynamic_input/mem_mode/mlo_max_iter` (`matrixvectoractivation.py:256`), Thresholding branching on `mem_mode` via try/except (`thresholding.py:173/179`), and Requant returning `0` for `ind!=0` (`requant`, digest 758) are all the *same* question: "is input `ind` a live stream, and how wide?" The `WeightPort` answers it by delegating to `self.weights`, so an embedded param yields width 0 with no branch and no `try/except AttributeError`.

---

## 2. Component interfaces (six narrow Protocols)

Each axis is a `typing.Protocol` (structural, no inheritance obligation). A component satisfies it by having the methods — this is the "declared contract" the substrate depends on (criterion #2).

```python
class DataContract(Protocol):        # op semantics — the ONLY per-family object
    def input_dtype(self, ind)->DataType; def output_dtype(self, ind)->DataType
    def normal_in(self, ind)->Shape;      def normal_out(self, ind)->Shape
    def attr_schema(self)->AttrSchema     # typed, replaces get_nodeattr_types stringly access

class ComputeStrategy(Protocol):     # axis 1
    def can_realize(self, spec:KernelSpec, tgt:Target)->Feasibility   # self-advertised (§8)
    def emit(self, e:EmitInputs)->Artifacts        # PURE: typed in -> artifacts out (criterion #5)
    def instantiate(self, bd:BDContext)->BDCell     # was the base-called instantiate_ip
    def contribute_ipi(self, bd:BDContext)->None    # this strategy's slice of stitching
    def runner(self, mode:ExecMode, env:BuildContext)->Runner         # axis 6b
    def layout_constraints(self)->list[LayoutConstraint]  # e.g. RTL wants sorted thresholds

class WeightDelivery(Protocol):      # axis 2
    def is_streamed(self)->bool                      # -> WeightPort.stream_width == 0 or PE*w
    def emit_artifacts(self, e:EmitInputs)->Artifacts   # was generate_hdl_memstream/dynload/fetch_weights
    def contribute_ipi(self, bd:BDContext)->None
    def extra_ports(self)->list[Port]                # weight stream / axilite / aximm(mlo)

class MemoryStrategy(Protocol):      # axis 3
    def primitive(self)->{"lutrom","bram","uram","dma"}
    def layout(self, params:ParamTensor, req:LayoutConstraint)->MemImage  # unified serializer
    def resource_estimate(self, geom)->ResRow

class FoldingSpec(Protocol):         # axis 4 (SHARED INTERFACE only; math per-op, §9.3)
    def parallelism_axes(self)->dict           # {"PE":pe,"SIMD":simd,...} common vocabulary
    def fold(self, normal:Shape, port:Port)->Shape
    def exp_cycles(self)->int

class Port(Protocol):                # axis 5 (tensor AND non-tensor)
    def role(self)->PortRole   # TENSOR_STREAM | AXI_MM | AXI_LITE | SIDEBAND
    def stream_width(self)->int
    def intf_decl(self)->IntfSpec               # typed replacement for get_verilog_top_module_intf_names

class ReferenceModel(Protocol):      # axis 6a (split from runners — kills the diamond)
    def evaluate(self, inputs:dict)->dict       # pure golden numpy/qonnx; NEVER named execute_node
```

The substrate holds a `Kernel` and calls only these methods. Adding an op family = add one `DataContract` + reuse the strategy library. Adding an implementation kind = add one `ComputeStrategy`. **Neither edits shared code** — the definition of criterion #2 passing.

---

## 3. Module structure (a DAG with no back-edges)

```
kernel/
  kernel.py          Kernel (final) + KernelSpec
  contract.py        DataContract Protocol; per-family: MVAUContract, ThresholdingContract, IODMAContract...
  folding.py         FoldingSpec Protocol; PeSimdFolding, WindowFolding, PassthroughFolding
  ports.py           Port Protocol + PortSet; TensorStreamPort, WeightPort, AxiMmPort, AxiLitePort, SidebandPort
  compute/           ComputeStrategy Protocol + HlsCompute, RtlCompute, StaticIpCompute, SubgraphCompute
  weights/           WeightDelivery Protocol + Embedded, DecoupledStream, External, DynamicLoad, LoopFetched
  memory/            MemoryStrategy Protocol + LutRom, Bram, Uram, OffChipDma  (+ unified ParamSerializer)
  exec/              ReferenceModel Protocol; runners: CppsimRunner, RtlsimRunner; Simulator (injected)
  build/
    context.py       BuildContext  (INJECTED: RtllibLocator, fpgapart, build_dir, Simulator, tool paths)
    artifacts.py     Artifact, ArtifactManifest, Parameterization (typed template binding — criterion #8)
    stitcher.py      Stitcher  (generic, op-agnostic IPI/BD driver — replaces the 236-line base method)
  select/
    selector.py      Selector  (generic pass; queries can_realize — replaces specialize_layers god-switch)
```

Every edge points strictly downward: `Kernel → components → BuildContext`. There is no substrate→subclass edge because there are no subclasses. The as-is cyclic graph (`hw-backend-model.md §Module-Structure ¶4`: substrate↔backend, agnostic-MVAU↔its leaves, HLS-leaf→RTL-artifact, finn_loop→siblings) is acyclic by construction.

---

## 4. Construction & wiring — the `KernelSpec` → `Kernel` pipeline

A `Kernel` is never hand-built with `new`. It is assembled by a builder from an op's typed spec:

```python
spec = KernelSpec(contract=MVAUContract(attrs), target=Target(fpgapart, clk))
kernel = Selector(registry).resolve(spec, env)      # picks compute + wires compatible weights/memory
```

`Selector.resolve` (fully generic, §8) chooses the `ComputeStrategy`, then asks it which `WeightDelivery`/`MemoryStrategy` it is compatible with (a `negotiate` handshake, §Weaknesses), and constructs the single `Kernel`. Wiring is explicit and inspectable — you can print a Kernel's six parts. This is the antithesis of the as-is stringly-typed instantiation (`optype + "_" + impl_style`, `specialize_layers.py:399`).

---

## 5. How composition dissolves the base-class leak (worked)

As-is: `HWCustomOp.generate_hdl_memstream` (`hwcustomop.py:307`) branches on the allowlist `["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"]` and `startswith("Elementwise")`, calling `calc_wmem`/`calc_tmem`/`ram_style` that live only on leaves. Six families *depend* on this leak.

Composed: the memstream Verilog is emitted by `DecoupledStream.emit_artifacts(e)`. `calc_wmem`/`calc_tmem` are methods on the `MemoryStrategy` (the thing that actually knows the geometry). The substrate build driver does:

```python
for part in (kernel.compute, kernel.weights, kernel.memory):   # generic; no names
    artifacts += part.emit(e) if hasattr-declared else part.emit_artifacts(e)
```

There is no `op_type` string anywhere. Renaming or adding a family cannot touch this code. The `mlo_max_iter` feature flag threaded through ~12 MVAU sites (`hwcustomop.py:100`, census MVAU) becomes the `LoopFetched` `WeightDelivery` component — present only when composed in, invisible otherwise. Thresholding's entanglement with `mlo_max_iter` in its *agnostic base* (`thresholding.py:133`) simply cannot occur: the `ThresholdingContract` has no knowledge of and no reference to weight delivery.

---

## 6. How composition dissolves the execute_node diamond (worked)

As-is: concrete op is `(OpBase, Backend)`; `execute_node` resolves by MRO to the base golden model, so every clean family patches it (`pool_hls.py:116` +7). VVAU's rtlsim block is mis-indented *inside* the input loop (`vectorvectoractivation_rtl.py:89`) because the rtlsim harness is copy-pasted per backend (MVAU too: `matrixvectoractivation_rtl.py:100` ≈ `_hls.py:569`).

Composed: two distinct slots on two distinct objects.
- `kernel.reference.evaluate(inputs)` — the pure golden model. Named `evaluate`, never `execute_node`, so no name can collide.
- `kernel.compute.runner(mode, env).run(inputs)` — cppsim or rtlsim. The `RtlsimRunner` is **one shared object** parameterized by the strategy's artifacts + an injected `Simulator`; it is written once, so the VVAU indentation bug is structurally impossible and the MVAU HLS/RTL rtlsim duplication collapses to one runner. The attribute-schema face of the same diamond (`ElementwiseBitShift_hls.get_nodeattr_types` dropping `direction`, blocker #15) is gone too: `attr_schema()` is a `DataContract` method returning a typed merged schema, not an MRO-order-dependent dict merge.

---

## 7. Reference vs cppsim vs rtlsim as three distinct concerns (axis 6)

- **Reference** lives on `kernel.reference` (a `ReferenceModel`) — pure, injectable, testable with zero tool state. iodma's `execute_node = pass` (`iodma_hls.py:388`) becomes an honest `IdentityMover` reference so cppsim data actually flows.
- **cppsim / rtlsim** live on `kernel.compute.runner(mode)`. The runner receives the injected `Simulator` from `BuildContext`, so the `finnxsi` import-time singleton (`hwcustomop.py:39`, `hlsbackend.py:44`, `rtlbackend.py:37`) is replaced by a constructor argument — a `FakeSimulator` can be injected for unit tests (criterion #10).

---

## 8. Selection: hybrid (components self-advertise, one generic pass orchestrates)

`Selector.resolve` is ~30 lines, fully op-agnostic:

```python
def resolve(self, spec, env):
    cands = self.registry.compute_for(spec.contract.kind())          # all peer kinds, no if-ladder
    feasible = [c for c in cands if c.can_realize(spec, spec.target)] # knowledge lives IN the component
    chosen = policy_rank(feasible, spec.contract.attr("preferred_impl_style"))
    w, m = chosen.negotiate(spec)                                    # compute picks compatible weights/memory
    return Kernel(spec.contract, chosen, w, m, folding_for(spec), ports_for(spec, w), reference_for(spec))
```

The `specialize_layers.py` god-switch — the `if optype == "MVAU"/"VVAU"/"LayerNorm"/...` ladder (`:40-211`) with `_mvu_rtl_possible`, `_vvu_rtl_possible`, etc., plus the drifting dtype pre-gates (`:60/:69` disagreeing with `:275`) — is deleted. `_mvu_rtl_possible`'s logic *moves into* `RtlCompute(mvau).can_realize`, co-located with the strategy it governs. There is exactly one feasibility method per strategy, so the pre-gate drift is impossible. StaticIP is selected by the identical mechanism.

I choose **hybrid** because it is what composition naturally yields and it is the strongest reading of criterion #3: the *capability knowledge* is self-advertised by each component (satisfying "backends declare what they can realize"), while a single generic pass does the *orchestration* (satisfying "selection is constraint resolution," with no per-op code in the pass).

---

## 9. Typed op↔rtllib adapter (criterion #8) & hermetic build (criterion #10)

`Parameterization` replaces the 26 untyped `$KEY$` `str.replace` edges (`hw-backend-model.md §Seam-Map §3a`). It is a typed `dict[Token, TypedValue]` bound to an `ArtifactManifest` (a declarative source list, single source of truth — killing the triplicated manifests at `fmpadding_rtl.py:137/153/165`, etc.). Token binding validates that every template token is provided and every provided value type-checks, so `layernorm`'s hard-coded `[31:0]` overriding the dtype contract (`layernorm_wrapper_template.v:20/25`) and requant's 6-decimal truncation (`requant_rtl.py:78`) become declared, validated parameters instead of silent magic. Source location goes through an injected `RtllibLocator` in `BuildContext`, evicting the ~34 `os.environ["FINN_ROOT"]` reads and the `os.listdir`-last-match globbing (`matrixvectoractivation.py:983`).

---

## Diagram

```
                         ┌──────────────────────────────────────────────┐
                         │  Kernel  (final, one class, never subclassed) │
                         │  implements 8-method contract by DELEGATION   │
                         └───┬───┬───┬───┬───┬───┬────────────────────────┘
        ┌────────────┬───────┘   │   │   │   │   └──────────┬─────────────┐
        ▼            ▼           ▼   ▼   ▼   ▼              ▼             ▼
   DataContract  Compute    Weight  Mem Fold Ports    Reference    (Runner via
   (per-family)  Strategy   Delivery Strat Spec (typed)  Model      compute.runner)
   MVAU/Thr/     HLS/RTL/   Embedded/ LUTROM Pe   Tensor  golden      Cppsim/Rtlsim
   IODMA/...     StaticIP/  Decoupled/BRAM  Simd  Stream/            (+ injected
   attr_schema   Subgraph   External/ URAM  Window AxiMM/            Simulator)
                 can_realize Dynamic/ DMA         AxiLite/
                 emit(pure)  LoopFetch            Sideband
                    │           │       │
                    └─────┬─────┴───────┘
                          ▼
                 Stitcher (generic): composes contribute_ipi() from each part
                          ▲
                 BuildContext (INJECTED: RtllibLocator, fpgapart, build_dir, Simulator)
```

## Axis treatment

### compute_impl
Axis 1 is the `ComputeStrategy` Protocol with peer implementations `HlsCompute`, `RtlCompute`, `StaticIpCompute`, `SubgraphCompute`. It is a COMPOSED reference on the Kernel, not an inherited mixin — this is the core move that makes HLS/RTL/static-IP peers (fixing the 638-vs-146-LOC non-peer asymmetry, `core-interfaces.md §Summary`). Each strategy exposes: `can_realize` (self-advertised feasibility), `emit` (PURE typed-in→artifacts-out, killing the `code_gen_dict` side-channel `hlsbackend.py:136`), `instantiate` (the old base-called `instantiate_ip`, now honestly on the component), `contribute_ipi` (its slice of stitching), and `runner` (cppsim/rtlsim). HlsCompute internally builds a typed `CppTranslationUnit` value (includes/defines/docompute/blackbox as fields) instead of four ordered side-effecting methods mutating a shared dict — no ordering, unit-testable in isolation.

### weight_delivery
Axis 2 is the `WeightDelivery` Protocol — its own first-class dimension ORTHOGONAL to compute (criterion #1). Peers: `Embedded`, `DecoupledStream`, `External`, `DynamicLoad`, `LoopFetched`(MLO). It owns `emit_artifacts` (the memstream/dynload/fetch_weights Verilog previously smeared across BOTH backends and the base, `matrixvectoractivation.py:920`+`_hls.py:144`+`hwcustomop.py:307/355/407`), `contribute_ipi`, `is_streamed` (drives WeightPort width so `get_instream_width` needs no mode branch), and `extra_ports`. Because it is composed independently, an `HlsCompute` MVAU no longer emits RTL weight-streamers — the cross-backend leak (13 hacks) is dissolved: compute and delivery are separate objects wired at construction. The MVAU common-case matrix `mem_mode × dynamic_input × mlo` becomes: pick one WeightDelivery component; there is no matrix in code.

### memory_strategy
Axis 3 is the `MemoryStrategy` Protocol — `LutRom`, `Bram`, `Uram`, `OffChipDma` — owning `primitive()`, `layout()` (the UNIFIED param/threshold serializer), and `resource_estimate()`. This replaces the per-op re-branching of `mem_mode` in thresholding/lookup/MVAU (`thresholding.py:179`, lookup ROM-vs-DMA). `calc_wmem`/`calc_tmem` (leaked onto the substrate via `hwcustomop.py:307`) move here, where the geometry actually lives. Crucially the three divergent threshold serializers (base/HLS/RTL `make_weight_file`, `thresholding.py:617`+`_hls.py:299`+`_rtl.py:452`) collapse to ONE `ParamSerializer.layout(params, req)`; the RTL binary-search 'sorted' requirement is expressed as a `LayoutConstraint` the compute strategy requests, honored by the one serializer — not a fourth copy.

### folding
Axis 4 is the `FoldingSpec` Protocol — SHARED INTERFACE ONLY per §9.3, explicitly NOT unified math. Common vocabulary: `parallelism_axes()->{PE,SIMD,...}`, `fold(normal,port)->folded`, `exp_cycles()`. Implementations differ per family (`PeSimdFolding`, `WindowFolding`, `PassthroughFolding`) — the per-op fold math stays in its own component, which is honest residual per-op variation (see weaknesses). The Kernel's `get_folded_*_shape` delegate through this one interface, so SWG's `use_parallel_window_output` (base calling a leaf-only method, `convolutioninputgenerator.py:124`) becomes a property of the composed WindowFolding object — no base→subclass reach. v1 does NOT attempt to unify the exp_cycles cost math (the shuffle `_NestSim`/layernorm magic-constant fusion, `outer_shuffle.py:20`, `layernorm_rtl.py:134`); those live in FoldingSpec implementations, deferred as scoped.

### ports
Axis 5 is `PortSet`, a typed collection of `Port` objects covering BOTH tensor streams and non-tensor ports as first-class peers: `TensorStreamPort`, `WeightPort` (delegates width to WeightDelivery), `AxiMmPort` (iodma/mlo/HBM), `AxiLitePort` (runtime-writeable, checksum, fmpadding registers), `SidebandPort` (TLAST/TKEEP). Each Port answers `stream_width()` and `intf_decl()` — a typed replacement for the stringly-built `get_verilog_top_module_intf_names` (which today branches on mem_mode in the agnostic base, `thresholding.py:292`, and which iodma has to un-do with `intf_names['m_axis']=[]`, `iodma_hls.py:394`). iodma's folded-shape `raise` (`iodma_hls.py:116/131`) is gone: you ask the AxiMmPort for its memory-mapped shape and the TensorStreamPort for its folded stream shape — different ports, honest answers, no method raises. fmpadding's hand-mirrored AXI-Lite register byte offsets (`fmpadding_rtl.py:101`) become fields on the AxiLitePort.

### execution
Axis 6 splits into two distinct concerns living on two objects, dissolving the diamond. 6a: `ReferenceModel` on `kernel.reference`, method `evaluate` (never `execute_node`) — pure golden numpy/qonnx, injectable, no tool state; iodma's `pass` becomes an honest `IdentityMover`. 6b: cppsim/rtlsim `Runner` obtained via `kernel.compute.runner(mode, env)` — the RtlsimRunner is ONE shared object (killing the VVAU indent bug `vectorvectoractivation_rtl.py:89` and the MVAU HLS/RTL rtlsim copy-paste `_rtl.py:100`≈`_hls.py:569`). The `finnxsi` singleton becomes an injected `Simulator` in BuildContext, so rtlsim can be faked (criterion #10). All three share the Kernel's PortSet as the common port contract, so a reference run and an rtlsim run agree on stream layout by construction.

## Conformance sketches

### mvau
`make_mvau(attrs)` builds ONE `Kernel`: `contract=MVAUContract` (owns the 8 accessors + the golden matmul+multithreshold `ReferenceModel`, the genuinely-clean part per census `_census_digest.json:185`); `compute ∈ {HlsCompute(mvau.hpp), RtlCompute(mvu_vvu_axi), StaticIpCompute}`; `weights ∈ {Embedded, DecoupledStream, External, DynamicLoad, LoopFetched}`; `memory ∈ {LutRom, Bram, Uram, OffChipDma}`; `folding=PeSimdFolding`; `ports=[TensorStreamPort(in0), WeightPort(in1→weights), TensorStreamPort(out0), AxiLitePort?, AxiMmPort?(mlo)]`. The 236-line `code_generation_ipi` (`matrixvectoractivation.py:920`) becomes `Stitcher.stitch(kernel)`, a GENERIC driver that concatenates `compute.contribute_ipi + weights.contribute_ipi + memory.contribute_ipi`. `self.instantiate_ip()` (base calling a leaf-only method — the uninstantiable-by-contract defect) becomes `compute.instantiate(bd)` on the composed strategy. `pumpedCompute` (RTL-only attr the base sniffs via try/except, `:892`) is a field on `RtlCompute`, never read elsewhere. Compute-vs-weight-delivery are now two swappable objects: this is the exact separation criterion #1 demands, expressed as composition of declared parts.

### thresholding
`Kernel` with `contract=ThresholdingContract` (2-input, ind=1 is threshold tensor; owns golden multithreshold reference reused for cppsim by all compute kinds). The embedded-vs-decoupled axis that today smears `mem_mode` into the agnostic base (`thresholding.py:179/292`) is now the orthogonal `MemoryStrategy` (`LutRom` vs `Bram`+`DecoupledStream`) — the `ThresholdingContract` has zero knowledge of it. The three incompatible `make_weight_file` copies (base `:617`, hls `:299`, rtl `:452`) collapse to ONE `ParamSerializer.layout(thresholds, constraint)`; `RtlCompute.layout_constraints()` returns `SortedForBinarySearch`, honored by the single serializer, so the sorted-threshold assumption (`thresholding_rtl.py:457`) is a declared constraint, not a hidden correctness dependency. The HLS-emits-RTL-streamer leak (`thresholding_hls.py:181` via the base allowlist) is gone: `DecoupledStream.emit_artifacts` emits the streamer regardless of compute kind.

### finn_loop
A ContainerKernel is just a `Kernel` whose `compute=SubgraphCompute` (holding a child `KernelGraph`). It honestly satisfies the SAME `ComputeStrategy` interface: `SubgraphCompute.artifacts()` returns the assembled block design, `contribute_ipi` does the BD assembly — so `get_rtl_file_list→None` (`finn_loop.py:1175`, a required-abstract stubbed to pass) simply does not exist; there is no abstract to stub. The 8 shape accessors delegate to boundary children's `PortSet`s through `contract=SubgraphContract`, an HONEST delegation (a container legitimately has no intrinsic shape) rather than iodma-style raises. The sibling-op coupling — hard-coded MVAU/Thresholding/Elementwise `.dat` naming and stream-tap dispatch (`finn_loop.py:421/712/558`) — is replaced by querying each child's `weights.artifacts()` and `weights.extra_ports()` through the interface: the container asks 'give me your param artifacts and taps', never 'are you an MVAU'. This is the first-class container concept the census demands, obtained with no new base class — a Kernel with a Subgraph compute part.

### iodma
`Kernel` with `contract=IODMAContract`, `compute=DwcChainCompute`, and a `PortSet` that makes non-tensor ports first-class: `[AxiMmPort(direction), TensorStreamPort, AxiLitePort(s_axi_control)]`. The folded-shape `raise` on the AXI-MM side (`iodma_hls.py:116/131`) is dissolved: the AxiMmPort answers a memory-mapped-shape query and the TensorStreamPort answers a folded-stream query — you ask the right port, nothing raises, so a generic pass iterating nodes never crashes on an iodma (the criterion #7 'honest shape/port contract'). `direction` becomes which Port variant is composed in, not an attribute every method bifurcates on. `execute_node=pass` becomes `reference=IdentityMover` so cppsim data flows and correctness is observable. No agnostic base is needed — composition never required one; iodma is a normal Kernel with infra ports, a first-class infra kernel rather than a taxonomy violation.

### static_ip_gemm
`StaticIpCompute` implements the ComputeStrategy Protocol identically to HLS/RTL. It is registered as a peer candidate for the MVAU contract kind. `can_realize(spec)` self-advertises: INT8 in/out, MW/MH within the hand-tuned range, PE/SIMD matching the hard IP's config — this is the SAME mechanism `_mvu_rtl_possible` uses, just co-located on the strategy. `Selector.resolve` auto-selects it for its common case with zero special-casing (it is one more entry in `registry.compute_for(MVAU)`). `emit` returns the pre-synthesized `.xci`/IP artifact; `contribute_ipi` instantiates the vendor IP cell; `runner` drives rtlsim against the hard IP; `negotiate` declares it wants `Embedded` weights + `Bram` memory. The proof of criterion #2: because `Stitcher`, `Selector`, and the Kernel's 8-method delegation call ONLY ComputeStrategy interface methods and never `isinstance`/`op_type`, adding StaticIpCompute is a new file in `compute/` plus a registry entry — ZERO edits to kernel.py, stitcher.py, selector.py, or any shared code. This is the forcing function passing.

## Criteria self-assessment (must-fix 1-10)

| # | verdict | how |
|--|--|--|
| 1 | **yes** | WeightDelivery is a first-class composed axis (Embedded/DecoupledStream/External/DynamicLoad/LoopFetched), orthogonal to ComputeStrategy. The 236-line IPI (matrixvectoractivation.py:920) becomes generic Stitcher composing each part's contribute_ipi(); weight streamer artifacts move to WeightDelivery.emit_artifacts. Expressible once, per delivery kind. |
| 2 | **yes** | There are no subclasses to branch on. Substrate (Kernel delegation, Stitcher, Selector) calls only the six Protocol interfaces; adding/removing an op = new DataContract, adding a kind = new ComputeStrategy — proven by StaticIpCompute needing zero shared-code edits. The op_type allowlist (hwcustomop.py:310) has no analogue. |
| 3 | **yes** | specialize_layers god-switch (specialize_layers.py:40-211) replaced by a ~30-line generic Selector querying each candidate ComputeStrategy.can_realize. _mvu_rtl_possible/_vvu_rtl_possible move onto the RTL strategy components; one feasibility method per strategy makes the pre-gate drift (:60/:69 vs :275) impossible. |
| 4 | **yes** | Reference semantics (kernel.reference.evaluate) and per-impl execution (kernel.compute.runner(mode)) are two named slots on two objects — no MRO, no shims, no execute_node name at all in the impl path. Single shared RtlsimRunner eliminates the VVAU indent bug (vectorvectoractivation_rtl.py:89) and the MVAU HLS/RTL rtlsim copy-paste. |
| 5 | **yes** | ComputeStrategy.emit is a pure function EmitInputs->Artifacts. HlsCompute builds a typed CppTranslationUnit value (includes/defines/docompute/blackbox as fields) instead of four ordered methods mutating self.code_gen_dict (hlsbackend.py:136). Each part is unit-testable in isolation; requant/thresholding _scale_is_one side-channels become fields on the returned value. |
| 6 | **yes** | MemoryStrategy (LutRom/Bram/Uram/OffChipDma) is one shared concept composed per Kernel. calc_wmem/calc_tmem live here. The unified ParamSerializer.layout replaces the three divergent thresholding make_weight_file copies; RTL's sorted requirement is a declared LayoutConstraint, not a re-branch. |
| 7 | **yes** | finn_loop is a Kernel with SubgraphCompute honestly implementing ComputeStrategy (no get_rtl_file_list stub because no abstract to stub). iodma is a Kernel with AxiMmPort/AxiLitePort as first-class ports; folded-shape raises are gone because you query the correct Port. Both satisfy the contract truthfully, not nominally. |
| 8 | **yes** | Parameterization (typed Token->TypedValue) + declarative ArtifactManifest (single source of truth) replace 26 untyped $KEY$ str.replace and the triplicated manifests (fmpadding_rtl.py:137/153/165). Token binding validates presence and type, so layernorm [31:0] and requant 6-decimal truncation become declared validated params. RtllibLocator injected via BuildContext evicts FINN_ROOT reads and os.listdir globbing. |
| 9 | **yes** | The hidden 'vivado' backend inside streamingfifo_rtl (streamingfifo_rtl.py:141), selected by runtime impl_style with base try/except reach-through (streamingfifo.py:90/104), becomes an explicit peer VivadoInfraCompute ComputeStrategy chosen by the same Selector. All implementation kinds are explicit composed peers; no runtime attr selects a hidden path. |
| 10 | **yes** | BuildContext injects Simulator (replacing the finnxsi import-time singleton at hwcustomop.py:39/hlsbackend.py:44/rtlbackend.py:37), RtllibLocator, fpgapart, build_dir, tool paths. Components read none of these from os.environ/module globals. A Kernel can be built/tested/simulated with a FakeSimulator and a fake locator in isolation. SWG↔VVAU layout coupling becomes an explicit FoldingSpec shared by both compositions, not a hidden reach. |

## Weaknesses (self-declared)

Where this design pays: (1) COORDINATION HAS TO LIVE SOMEWHERE. Some constraints span components — RtlCompute requires sorted thresholds from the serializer; StaticIpCompute requires a specific PE/SIMD folding; DecoupledStream needs a compatible MemoryStrategy. I route these through a `negotiate(spec)` handshake + `LayoutConstraint` objects, but this is the composition-specific hazard: if under-designed, the Selector or Stitcher quietly becomes a new god-object re-encoding cross-part knowledge — the very thing I evicted from the base. The design's integrity depends on keeping negotiation declarative (constraints as data the components publish), and I have not fully specified the constraint algebra. (2) DELEGATION DEPTH. `get_instream_width` now hops Kernel→Port→WeightDelivery; a stack trace is three frames deeper than a flat override, and 'where does this number come from' requires knowing the composition. Cognitive cost, not runtime cost. (3) RESIDUAL PER-OP MATH. FoldingSpec is a shared INTERFACE, but the fold/exp_cycles math still lives in per-family FoldingSpec objects (PeSimdFolding, WindowFolding, ...) — a judge will correctly note this is duplication the design did not eliminate (deliberately, per §9.3, but still a cost). Same for DataContract: op semantics is irreducibly per-family, so there are still ~27 DataContract implementations. I claim the sharp line 'math semantics is per-op, everything else is composed' is defensible, but the DataContract is the one place inheritance-like per-op code survives. (4) SUBGRAPH-AS-COMPUTE IS A STRETCH. Modeling a container's block-design assembly as a ComputeStrategy is elegant for uniformity but semantically strained — a subgraph is arguably a different category than 'how the math is realized'. It works because the interface (emit/contribute_ipi/artifacts) fits, but the naming lies slightly. (5) CONSTRUCTION CEREMONY. Assembling six components per Kernel is more verbose than a two-class mixin; the KernelSpec/builder/registry machinery is new surface area that must be learned, and mis-wiring (a valid-typed but semantically-wrong composition) is a new failure mode the type system won't catch without the negotiate handshake.

## Prior fidelity

This design commits maximally to composition-over-inheritance rather than drifting to a generic middle. The decisive, distinctive choice: ONE final Kernel class for all 27 families — the inheritance lattice of ~80 `(OpBase, Backend)` classes collapses to a single concrete type parameterized entirely by composed strategy objects. There is no shared base holding behavior (so the base-class leak hwcustomop.py:307/355/407 has nowhere to live), no mixin (so HLSBackend/RTLBackend asymmetry and the undeclared host-object interface, core-interfaces.md:228, cannot exist), and no MRO (so the execute_node diamond and the ElementwiseBitShift attr-schema diamond cannot arise). The only surviving inheritance is nominal Protocol conformance — structural, obligation-free. Every axis of variation is an object you swap, not a class you subclass: implementation kind, weight delivery, memory, folding, ports, execution are all reference-held components invoked through declared interfaces. The substrate is pure generic drivers (Kernel delegation, Stitcher, Selector, BuildContext) that never see a concrete op or kind. I resisted the tempting hedge of keeping a thin per-op base class 'just for shape math' — instead shape math is a composed DataContract component, so even op semantics is a plugged-in part, not an inherited one. This is the fully-committed reference-semantics-is-one-object, per-impl-execution-is-another decomposition the assigned angle calls for.
