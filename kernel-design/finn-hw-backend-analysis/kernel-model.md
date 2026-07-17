# The Kernel Backend System — To-Be Design

*The synthesized target design for FINN's HW-op backend, replacing the current
2-axis `HWCustomOp` + `{HLSBackend, RTLBackend}` abstraction with a single
unified Kernel model where HLS, RTL, and static IP are peer implementation kinds.*

## How this design was produced

A structured design arc, each phase gated and its artifact on disk:

1. **As-is model** (`hw-backend-model.md`) — verified 3-phase analysis of the current system: the base-class leak, the `execute_node` diamond, the selection god-switch, 331 hacks, top-10 pressure list.
2. **Design brief** (`design/00-design-brief.md`) — the fixed rubric: invariants, 10 must-fix criteria (1-4 disqualifiers), static-IP forcing function, 6 orthogonal axes, 5 conformance ops.
3. **Design panel** (`design/01`–`05`) — 5 independent designs from assigned priors (composition, capability-trait, multi-axis, ir-lowering, evolutionary).
4. **Score + adversarial stress-test** (`design/06-scoring-and-ranking.md`) — 10 independent evaluators + ranker. **Winner: composition** (only top-scorer whose adversary didn't collapse). ir-lowering tied on score but collapsed at the anchor test → 4th.
5. **This document** — synthesis: composition backbone + grafts from the runners-up + fresh design of the stitch layer (the one tension no design solved) + rulings on 6 tensions + migration path. Completeness-critic verified.

## The design in one paragraph

There is exactly **one** `@final`, never-subclassed `Kernel` class. It *is* a
`DataContract` (op semantics) and *has* six Protocol-typed parts (compute,
weight-delivery, memory, folding, ports, reference/runner execution). The
substrate is generic drivers that read only Protocols, `Role` tags, and
`CellKind` — never `isinstance`, never `op_type`. This makes criterion #2 (no
substrate branch on op identity) hold **by construction** — there is no base to
leak from and no subclass to branch on. The `execute_node` diamond (criterion #4)
becomes **unnameable**: reference semantics live on `kernel.reference.evaluate`,
per-impl execution on `kernel.compute.runner(mode)` — two names, no MRO
collision. Selection is a solver over declared precondition-data with `explain()`.
Static IP is a peer `ComputeStrategy` that auto-selects for its common case with
zero substrate edits. Block-design stitching — the 236-line `code_generation_ipi`
god-method — dissolves into role-tagged cells + a region that owns every net + one
op-agnostic resolver.

## Contents

- **§1 Core Kernel model + the six axes** — the backbone and how each axis is represented.
- **§2 Selection** — constraint-solved realization binding (rules T1, T2).
- **§3 The Stitch Layer** — the typed coordination model that dissolves the IPI god-method (T4, fresh work).
- **§4 Conformance walkthroughs + migration path** — all 5 ops in the new model + the 8-PR strangler-fig migration (T5).
- **§5 Completeness critique** — independent verification + the 2 spec-cleanup items before implementation.

## Tension rulings (summary; detail in the sections)

| Tension | Ruling |
|---|---|
| **T1** feasibility as code vs data | **Data** — precondition-AST + one solver with `explain()`; kills the `:60/:275` drift structurally |
| **T2** where cross-axis constraints live | **Three tiers** — distributed per-realization preconditions / small pairwise `LEGAL` / bounded n-ary `KNOWN_COUPLINGS` for MLO & pumpedMemory |
| **T3** delivery: part vs graph-node | **Composed part** with peer role-tagged ports; compute never touches delivery privates |
| **T4** stitch/IPI coordination | **Dissolved** — role-tagged cells + region-owned nets + op-agnostic resolver; IP-packager surgery contained as scoped opaque residual |
| **T5** purity vs migration | **Clean-slate target, incremental delivery** — 8-PR strangler-fig, green against 27-family regression at every step |
| **T6** memory: peer axis vs sub-choice | **Independent axis** owning `pumpedMemory`/`calc_wmem`/estimation; MLO coupling handled in T2 tier-3 |

> **Two spec-cleanup items before implementation** (from the critic, §5, neither structural):
> **C1** — the MLO constraint is n-ary (tier-3 `MloCoupling`, which subsumes the `LoopFetched⇒OffChipDma` pairwise forcing); §1 wording that calls it "one `Legal` row" should defer to the tier-3 classification.
> **C2** — assign provenance of the clock/reset/DATA boundary ports to `PortSet`/`contract` construction, not to any compute/weight/memory part.

---




---

# § 1. Core Kernel Model + The Six Axes

# Core Kernel Model + The Six Axes (D3 Synthesis)

> **Thesis.** There is exactly **one** `@final`, never-subclassed `Kernel` class. It *is* a `DataContract` (op semantics) and it *has* six Protocol-typed parts — one per orthogonal axis of the brief §4. Every one of the eight downstream accessors is **pure delegation** into those parts; the Kernel owns no compute, no codegen, no simulation, no branch. Criterion #2 (substrate never branches on subclass) holds because *there is no subclass and no behavior-bearing base to leak from*; criterion #4 (the `execute_node` diamond) holds because reference-semantics and per-impl execution are **two differently-named slots on two different objects**, so no method name can collide under an MRO that no longer exists. The parts connect through the **Stitch Layer** (T4, provided) — each part publishes only role-tagged `Cell`s/`Port`s and never names another part.

This section assembles: composition's `Kernel`+delegation backbone; multi-axis's typed `Port` sum with **shape-by-role** and Memory as a **genuine peer axis** (T6); ir-lowering's **param-delivery-as-peer-producer port model** without the IR framework (T3); and the **reference/runner** execution split.

---

## 1. The one final Kernel class

```python
@final
class Kernel:
    # ── the semantic anchor: the ONLY irreducibly per-family object ──
    contract:  DataContract     # op SEMANTICS (MVAUContract, ThresholdingContract, IODMAContract, …)

    # ── the six orthogonal axis parts (brief §4), each a structural Protocol ──
    compute:   ComputeStrategy  # axis 1 — how the math is realized {HLS, RTL, StaticIP, Subgraph}
    weights:   WeightDelivery   # axis 2 — how coefficients reach compute (a PEER PRODUCER, T3)
    memory:    MemoryStrategy    # axis 3 — where params live + read discipline (INDEPENDENT axis, T6)
    folding:   FoldingSpec       # axis 4 — PE/SIMD vocabulary (shared interface, math per-op)
    ports:     PortSet            # axis 5 — typed Port sum; only DATA/WEIGHT ports carry a shape
    reference: ReferenceModel     # axis 6a — pure golden semantics (NEVER named execute_node)
    #  axis 6b (cppsim/rtlsim runners) is owned BY `compute` — see §6
```

`Kernel` has **no abstract methods** and is **never subclassed**. All 27 op families are `Kernel` *instances*. "MVAU" is not a class; it is `make_mvau(attrs)` returning a `Kernel` whose `contract` is an `MVAUContract`. The ~80-class `(OpBase, Backend)` lattice collapses to **one concrete type + a library of interchangeable parts**.

The `DataContract` is the seventh field but is *not* one of the six axes — it is the op's identity (what makes an MVAU an MVAU regardless of realization), the one place per-family code irreducibly survives. The six *axes* are the six pluggable parts; hardware variation lives entirely in them.

---

## 2. The six Protocol interfaces (narrow, structural, post-graft)

Each axis is a `typing.Protocol` — structural conformance, zero inheritance obligation. The substrate holds a `Kernel` and calls **only** these methods.

```python
class DataContract(Protocol):        # semantic anchor — one per family
    def input_dtype(self, ind) -> DataType;  def output_dtype(self, ind) -> DataType
    def normal_in(self, ind) -> Shape;        def normal_out(self, ind) -> Shape
    def attr_schema(self) -> AttrSchema       # typed; replaces stringly get_nodeattr_types

class ComputeStrategy(Protocol):     # axis 1
    def can_realize(self, spec, tgt) -> Precondition   # DATA, not opaque method (T1 graft, §7)
    def emit(self, e: EmitInputs) -> Artifacts          # PURE: typed in → artifacts out (crit #5)
    def emit_cell(self, env: BuildContext) -> Cell      # role-tagged Cell for the stitch layer (S1.2)
    def ports(self) -> tuple[Port, ...]                 # compute's OWN ports (incl. WEIGHT_SINK)
    def runner(self, mode: ExecMode, env) -> Runner     # axis 6b — cppsim/rtlsim
    def clock_plan(self) -> ClockPlan                   # PUMP consumer iff pumpedCompute (S3)
    def layout_constraints(self) -> list[LayoutConstraint]

class WeightDelivery(Protocol):      # axis 2 — a PEER PRODUCER (T3)
    def emit_cell(self, env) -> Cell | None             # streamer Cell, or None for Embedded/External
    def ports(self) -> tuple[Port, ...]                 # its OWN ports (WEIGHT_SOURCE / CONFIG / OFFCHIP)
    def address_plan(self) -> AddressPlan               # OFFCHIP/CONFIG masters (stitch S8)
    def clock_plan(self) -> ClockPlan                   # PUMP consumer iff pumpedMemory reaches it (S3)

class MemoryStrategy(Protocol):      # axis 3 — INDEPENDENT axis (T6)
    def primitive(self) -> Primitive                    # {LUTROM, BRAM, URAM, DMA}
    def layout(self, params: ParamTensor, req: LayoutConstraint) -> MemImage   # ONE serializer
    def resource_estimate(self, geom) -> ResRow
    def clock_plan(self) -> ClockPlan                   # owns the pumpedMemory fuse decision (S3)

class FoldingSpec(Protocol):         # axis 4 — SHARED INTERFACE only; math per-op (brief §9.3)
    def parallelism_axes(self) -> dict                  # {"PE":pe, "SIMD":simd, …} common vocabulary
    def fold(self, normal: Shape, port: Port) -> Shape
    def stream_width(self, dt: DataType, port: Port) -> int
    def exp_cycles(self) -> int

class ReferenceModel(Protocol):      # axis 6a
    def evaluate(self, inputs: dict) -> dict            # pure golden numpy/qonnx; NEVER execute_node
```

Axis 5 (`Port`/`PortSet`) is not a behavioral Protocol but a **typed value model** — it is exactly the Stitch Layer's `Port` atom (S1.1). See §5.

**Adding an op family** = add one `DataContract`, reuse the strategy library. **Adding an implementation kind** = add one `ComputeStrategy`. Neither edits shared code — the definition of criterion #2 passing (§7).

---

## 3. The 8-method contract as pure delegation

The externally-observable op contract (brief §1) is eight accessors. The `Kernel` implements every one by delegating into `contract`, `folding`, and `ports` — it computes nothing itself:

```python
def get_input_datatype(self, ind=0):    return self.contract.input_dtype(ind)
def get_output_datatype(self, ind=0):   return self.contract.output_dtype(ind)
def get_normal_input_shape(self, ind=0): return self.contract.normal_in(ind)
def get_normal_output_shape(self, ind=0):return self.contract.normal_out(ind)

def get_folded_input_shape(self, ind=0):
    return self.folding.fold(self.contract.normal_in(ind),  self.ports.data_in(ind))
def get_folded_output_shape(self, ind=0):
    return self.folding.fold(self.contract.normal_out(ind), self.ports.data_out(ind))

def get_instream_width(self, ind=0):
    return self.folding.stream_width(self.contract.input_dtype(ind),  self.ports.data_in(ind))
def get_outstream_width(self, ind=0):
    return self.folding.stream_width(self.contract.output_dtype(ind), self.ports.data_out(ind))
```

Three structural wins visible here:

- **Folded shape is `folding × port`, not per-op math sprinkled through backends.** SWG's `use_parallel_window_output` (base calling a leaf-only method, `convolutioninputgenerator.py:124`) becomes a property of the composed `WindowFolding` object — the Kernel never reaches into a subclass.
- **Stream width is a question you ask a `Port`, delegated to `folding`.** MVAU's `get_instream_width` branching on `dynamic_input/mem_mode/mlo_max_iter` (`matrixvectoractivation.py:256`), Thresholding's `mem_mode` try/except (`thresholding.py:173/179`), and Requant returning `0` for `ind!=0` are all the **same** question. It is answered by *which ports exist*: an `Embedded` `WeightDelivery` publishes **no** `WEIGHT_SOURCE` port, so the weight stream simply isn't in the `PortSet` — width-0 with **no branch and no `try/except AttributeError`**.
- **Non-tensor ports never raise.** `data_in(ind)`/`data_out(ind)` return only `DATA_*`-role ports; an `OFFCHIP` (AXI-MM) port has `shape=None` and is never handed to `folding.fold`. iodma's raising getters (`iodma_hls.py:116/131`) are a category error the type structurally forbids (§5, crit #7).

---

## 4. Axis 1 — Compute, and Axis 2 — Weight delivery as a **peer producer** (T3 ruling)

### Axis 1: `ComputeStrategy`
Peers `HlsCompute`, `RtlCompute`, `StaticIpCompute`, `SubgraphCompute` — a **composed reference**, not an inherited mixin. This is the move that makes HLS/RTL/static-IP genuine peers (killing the 638-vs-146-LOC asymmetry). `emit` is a **pure** `EmitInputs → Artifacts` function (no `code_gen_dict` side-channel, `hlsbackend.py:136`; crit #5). RTL-only concerns (`pumpedCompute`, `matrixvectoractivation.py:939`) are **fields on `RtlCompute`**, read by no one else, surfaced only as a `PUMP` clock consumer in `emit_cell` (Stitch S3).

### Axis 2: `WeightDelivery` — the T3 ruling, concretely

**Ruling (T3):** weight delivery stays a **composed part** (`kernel.weights`) — no IR graph, no folding-propagation pass — **but it is modeled as a peer producer with its own ports**, exactly as ir-lowering's `ParamSource`. The compute strategy **never reaches into the delivery object's privates**; the *only* channel between them is the shared `Role` vocabulary on ports.

Concretely:

- `ComputeStrategy.ports()` publishes a `Port(role=WEIGHT_SINK)` (the coefficient input, e.g. MVAU `in1_V`).
- `WeightDelivery.ports()` publishes a `Port(role=WEIGHT_SOURCE)` (the streamer output, e.g. `m_axis_0`).
- **Neither names the other.** There is no `self.weights` access inside a `ComputeStrategy` and no `self.compute` access inside a `WeightDelivery`. The net between them is *derived* by the Stitch Layer's op-agnostic `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS)` (S2) — owned by the enclosing `Region`, not by either part.

Peers and what they publish:

| `WeightDelivery` peer | emits a `Cell`? | ports published | as-is site dissolved |
|---|---|---|---|
| `Embedded` | no | *(none — weights compiled in)* | params.h array; width-0 weight stream by absence |
| `DecoupledStream` | yes (streamer) | `WEIGHT_SOURCE`, `CONFIG?` | `generate_hdl_memstream` allowlist (`hwcustomop.py:307`) |
| `External` | no | *(compute's `WEIGHT_SINK` marked `boundary=True`)* | `elif mem_mode=="external"` (`:1149`) |
| `DynamicLoad` | yes | `DATA_IN` (dynamic weight stream) | dynload branch (`:973`) |
| `LoopFetched` (MLO) | yes | `WEIGHT_SOURCE`, `INDEX_SINK`, `OFFCHIP` | `fetch_weights`, `mlo_max_iter` (`hwcustomop.py:100`) |

The MVAU common-case matrix `mem_mode × dynamic_input × mlo` becomes: **pick one `WeightDelivery` part.** There is no matrix in code. The `mlo_max_iter` flag threaded through ~12 sites is present only when `LoopFetched` is composed in, invisible otherwise; a `ThresholdingContract` has no reference to weight delivery at all, so `thresholding.py:133`'s entanglement cannot occur.

**Why the port model, not the full IR node (T3):** modeling delivery as a *graph node* (ir-lowering) drags in a folding-propagation pass to keep `param_in` fold consistent with compute `PE×SIMD` and a region/IR dialect. We take the *port model only*: delivery is a peer producer whose `WEIGHT_SOURCE` fold is set by the same `FoldingSpec` the compute uses (both read `kernel.folding`), so consistency is a shared reference, not a propagation pass. Lighter, and the orthogonality is identical.

---

## 5. Axis 5 — Ports: the typed sum with **shape-by-role** (multi-axis graft), realized as the Stitch `Port`

The port atom **is** the Stitch Layer's `Port` (S1.1) — I do not redefine it. Its load-bearing property is multi-axis's discipline, stated in the stitch layer as **"`shape` is present by role, not by op"**: only `DATA_IN/DATA_OUT/WEIGHT_SINK/WEIGHT_SOURCE` carry a `FoldedShape`; `OFFCHIP`/`CONFIG`/`CLOCK` ports carry a `width` or nothing and have `shape=None`.

`PortSet` (axis 5) is the assembled collection, and it is **not** a monolithic getter — it is the **union of the ports each part publishes** (multi-axis's composition + ir-lowering's peer-producer):

```python
@final
class PortSet:
    ports: tuple[Port, ...]                     # Port = the Stitch Layer atom (S1.1)

    def data_in(self, ind)  -> Port: ...        # the ind-th DATA_IN port
    def data_out(self, ind) -> Port: ...        # the ind-th DATA_OUT port
    def tensor_ports(self)  -> Iterable[Port]:  # the ONLY ports with a shape — for generic folded-shape passes
        return (p for p in self.ports if p.shape is not None)
    def boundary_ports(self) -> tuple[Port, ...]:  # din/dout/clk/rst(+idx/aximm) — consumed by stitch S4

def ports_for(compute, weights, memory) -> PortSet:      # assembled, never hand-built
    return PortSet(compute.ports() + weights.ports() + memory.ports())
```

Consequences (crit #7, by type):

- A generic folded-shape pass iterates `port_set.tensor_ports()` and **never touches** iodma's AXI-MM side — you cannot ask a `shape=None` port for its shape. `iodma_hls.py:116/131` is deleted, not reimplemented.
- iodma's `direction` bifurcation (`iodma_hls.py:391-399`) becomes **which ports the `IODMAContract`'s parts publish**: `direction=="in"` → `{OFFCHIP, DATA_OUT, CONFIG}`; `"out"` → `{DATA_IN, OFFCHIP, CONFIG}`. Chosen once at construction, not bifurcated across eight methods.
- checksum's AXI-Lite reg, tlastmarker's `SIDEBAND`, fmpadding's regmap are first-class `Port` variants with typed roles, not `intf_names` dict keys. clk2x is a `CLOCK` consumer with a `ClockDomain` label — a detail inside axes 1/3, **not a seventh axis** (brief §9.4).

---

## 6. Axis 3 — Memory (T6), Axis 4 — Folding, Axis 6 — Execution

### Axis 3: `MemoryStrategy` — a **genuinely independent** peer axis (T6 ruling)

**Ruling (T6):** Memory is a fully independent axis — `LutRom | Bram(ram_style) | Uram | OffChipDma` — chosen **independently** of `Embedded/DecoupledStream/External/LoopFetched`. This raises the winner's weakest sub-score (axis orthogonality 4→5) and evicts the residual memory-into-delivery coupling that capability-trait admitted. `calc_wmem`/`calc_tmem`/`bram_estimation`/`uram_estimation` (leaked onto the substrate via `hwcustomop.py:307`; `matrixvectoractivation.py:387/365`) live **here**, where the geometry actually is. The three divergent threshold serializers (`thresholding.py:617` + `_hls.py:299` + `_rtl.py:452`) collapse to **one** `layout(params, req)`; RTL's binary-search "sorted" requirement is a `LayoutConstraint` the compute strategy *requests*, honored by the one serializer — not a fourth copy.

The **only** cross-axis coupling is a **small declared `Legal` pairing** for the genuinely-coupled MLO case: `LoopFetched ⇒ OffChipDma`. This pairing is **not** in `Memory` and **not** in `WeightDelivery` — it lives as one row in the selection layer's bounded pairwise relation (T2's answer), reviewed once. `Memory` and `WeightDelivery` themselves remain oblivious to each other; `Embedded×LutRom`, `DecoupledStream×Bram`, `DecoupledStream×Uram(ram_style=ultra)` compose freely. This is the clean "motion vs substrate" split, thin only exactly at MLO — which is why MLO gets the one explicit `Legal` row.

`MemoryStrategy` also **owns the `pumpedMemory` decision** as data: it produces the `ClockPlan` whose `fuse` map collapses `PUMP→BASE` when not pumped (Stitch S3). The god-method's `clk2x` branch (`matrixvectoractivation.py:1098-1112`) becomes a table lookup contributed by Memory (the plan) × WeightDelivery (the `PUMP` consumer) — **zero branches in the stitch path**.

### Axis 4: `FoldingSpec` — shared interface, math per-op (brief §9.3, unchanged by graft)
`PeSimdFolding`, `WindowFolding`, `PassthroughFolding` implement one interface — `parallelism_axes()`, `fold()`, `stream_width()`, `exp_cycles()` — but the per-family fold math stays in each object (honest residual, deliberately not unified in v1). Both `compute` and `weights` read the **same** `kernel.folding`, so the `WEIGHT_SINK` and `WEIGHT_SOURCE` folds agree by shared reference (this is what lets T3 skip a propagation pass, §4). The SWG↔VVAU layout coupling becomes one shared `FoldingSpec` object both compositions hold — an explicit dependency, not a hidden reach (crit #10).

### Axis 6: Execution — the **reference/runner split** (kills the diamond, crit #4)
Two distinct concerns on two distinct objects:

- **6a `ReferenceModel`** on `kernel.reference`, method **`evaluate`** — pure golden numpy/qonnx, injectable, zero tool state. iodma's `execute_node = pass` (`iodma_hls.py:388`) becomes an honest `IdentityMover` so cppsim data actually flows and correctness is observable.
- **6b `Runner`** via `kernel.compute.runner(mode, env)` — cppsim or rtlsim. The `RtlsimRunner` is **one shared object** parameterized by the strategy's artifacts + an injected `Simulator` (from `BuildContext`). It is written once, so the VVAU indentation bug (`vectorvectoractivation_rtl.py:89`) is structurally impossible and the MVAU HLS/RTL rtlsim copy-paste (`_rtl.py:100` ≈ `_hls.py:569`) collapses to one runner. The `finnxsi` import-time singleton (`hwcustomop.py:39`, `hlsbackend.py:44`, `rtlbackend.py:37`) is a constructor argument — a `FakeSimulator` injects for unit tests (crit #10).

All three interpretations (reference, cppsim, rtlsim) share the Kernel's `PortSet`, so a reference run and an rtlsim run agree on stream layout **by construction**.

---

## 7. How the parts connect — via the Stitch Layer (T4), not redesigned here

Assembly is generic and owned by the composition builder, **not** by any part. Each part emits exactly one role-tagged `Cell` (S1.2) and nothing about any other part; the enclosing `Region` owns every net between two parts (S1.3); one op-agnostic `RegionResolver` derives nets from role-tags (S2); one `TclRenderer` prints them (S4):

```python
def assemble_region(kernel: Kernel, env: BuildContext) -> Region:      # Stitch S4 — ~15 lines, generic
    cells = [c for c in (kernel.compute.emit_cell(env),
                         kernel.weights.emit_cell(env)) if c is not None]  # None ⇒ Embedded/External
    return Region(
        name         = kernel.contract.node_name,
        cells        = tuple(cells),
        boundary     = kernel.ports.boundary_ports(),
        clock_plan   = kernel.memory.clock_plan() | kernel.compute.clock_plan(),   # S3
        address_plan = kernel.weights.address_plan(),                              # S8
        bindings     = STANDARD_BINDINGS,     # the op-AGNOSTIC role-match table (S2)
    )

def stitch(kernel, env) -> TclScript:
    return TclRenderer().render(resolve(assemble_region(kernel, env)))
```

The 236-line `code_generation_ipi` (`matrixvectoractivation.py:920`) is entirely this. The weight-streamer→compute net (`:1085-1089`) is derived by `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS)`. External-weights and the static-IP GEMM **edit nothing** in the stitch layer (Stitch S5). See the Stitch Layer section for the full ruling — this core model *consumes* it and does not re-specify it.

---

## 8. Criteria #2 and #4 — how they hold **by construction**

### Criterion #2 — the substrate never branches on subclass identity
- **There are no subclasses.** `Kernel` is `@final`. The taxonomy that was ~80 `(OpBase, Backend)` classes is **one** concrete class + a part library. There is *nothing to branch on* and *no behavior-bearing base to leak from* — the `op_type` allowlist at `hwcustomop.py:310` has **no home** in this shape.
- **The substrate (Kernel's 8-method delegation, `assemble_region`/`resolve`/`render`, the selector) calls only the six Protocols + the Stitch value types (`Role`, `Protocol`, `CellKind`, `ClockDomain`).** It reads no pin name and no op type.
- **Proof — the forcing function.** `StaticIpCompute` is registered as a peer candidate for the MVAU contract kind. Its `emit_cell` returns a `Cell(kind=VENDOR_IP)` whose ports are tagged `DATA_IN/DATA_OUT/WEIGHT_SINK/CLOCK(BASE)` exactly like the RTL MVAU. Adding it is **one new file in `compute/` + one registry entry — zero edits** to `kernel.py`, the stitch layer, the selector, `STANDARD_BINDINGS`, or any other op. It stitches by construction because it *speaks the role vocabulary*, not because any driver learned about it.

### Criterion #4 — `execute_node` is not an inheritance diamond
- Reference semantics (`kernel.reference.evaluate`) and per-impl execution (`kernel.compute.runner(mode)`) are **two differently-named slots on two different objects**. The name `execute_node` **does not exist** anywhere in the impl path.
- There is **no MRO** (no multiple inheritance, no shared base), so no two meanings of one method name can resolve to each other. The collision is not refactored — it is **unnameable**.
- The single shared `RtlsimRunner` makes the VVAU indent bug (`vectorvectoractivation_rtl.py:89`) and the MVAU rtlsim duplication structurally impossible. The attr-schema face of the same diamond (`ElementwiseBitShift_hls.get_nodeattr_types` dropping `direction`, blocker #15) is gone too: `attr_schema()` is a `DataContract` method returning a typed merged schema, not an MRO-order-dependent dict merge.

---

## 9. Module DAG (no back-edges)

```
                          ┌───────────────────────────────────────────────┐
                          │  Kernel  (@final, one class, never subclassed) │
                          │  implements the 8-method contract by DELEGATION│
                          └──┬────┬────┬────┬────┬────┬────┬───────────────┘
     contract (semantics) ◄─┘    │    │    │    │    │    └─► reference  (axis 6a)
                                 │    │    │    │    │           ReferenceModel.evaluate
        ┌────────────────────────┘    │    │    │    └──────────► ports  (axis 5)
        ▼                             │    │    │                   PortSet of Port
   compute (axis 1)                   │    │    └───────► folding (axis 4)
   ComputeStrategy                    │    │              FoldingSpec (shared iface)
   {Hls,Rtl,StaticIp,Subgraph}       │    └───► memory (axis 3)  ◄─ INDEPENDENT (T6)
     • runner(mode)  (axis 6b)        │          MemoryStrategy {LutRom,Bram,Uram,Dma}
     • can_realize → Precondition     │            • layout() = ONE serializer
     • emit(pure) / emit_cell         └───► weights (axis 2)  ◄─ PEER PRODUCER (T3)
                                              WeightDelivery {Embedded,Decoupled,
                                              External,Dynamic,LoopFetched}
                                                • ports(): WEIGHT_SOURCE / OFFCHIP / CONFIG

   parts publish role-tagged Cells + Ports; they NEVER name each other
        │ compute.emit_cell   │ weights.emit_cell   │ *.clock_plan / address_plan
        └──────────┬──────────┴──────────┬──────────┘
                   ▼                      ▼
        ══════════════ STITCH LAYER (T4, provided — not redesigned) ══════════════
          assemble_region → RegionResolver(STANDARD_BINDINGS) → TclRenderer
          Region OWNS every inter-part net; resolver reads Role, never an op/pin
                   ▲
        BuildContext (INJECTED: Simulator, RtllibLocator, fpgapart, build_dir, tool paths)

   Only cross-axis coupling in the whole model:  one Legal row  LoopFetched ⇒ OffChipDma
   (lives in the selection layer's bounded pairwise relation — T2 — not in any part)
```

**Every edge points strictly downward:** `Kernel → parts → Stitch Layer → BuildContext`. No substrate→subclass edge exists because no subclass exists. The as-is cyclic graph (substrate↔backend, agnostic-MVAU↔leaves, finn_loop→siblings) is acyclic by construction.

---

## 10. Tension rulings settled in this section

- **T3 (weight delivery: composed part vs graph node) → composed part with peer-producer ports.** `kernel.weights` is a component (no IR framework), but it publishes its own `WEIGHT_SOURCE`/`OFFCHIP`/`CONFIG` ports and the compute publishes `WEIGHT_SINK`. Compute never touches delivery privates; the sole channel is the shared `Role` vocabulary, resolved by the Stitch Layer's `RoleBinding`. Fold consistency is a shared `kernel.folding` reference, not a propagation pass.
- **T6 (memory: peer axis vs sub-choice) → genuinely independent peer axis.** `MemoryStrategy` is chosen independently of `WeightDelivery` (orthogonality 5/5), with exactly **one** declared `Legal` pairing (`LoopFetched ⇒ OffChipDma`) for the sole genuinely-coupled case, held in the selection layer — not inside either part.

Deferred to their owning sections (referenced, not decided here): **T1** feasibility-as-DATA (`can_realize → Precondition`, capability-trait graft) and **T2** the bounded pairwise `Legal` relation both live in the **selection engine**; **T4** the stitch/IPI coordination is the **provided Stitch Layer**; **T5** the 8-PR migration path is the delivery plan. This section commits the core Kernel model and the six axes that all four consume.


---

# § 2. Selection — Constraint-Solved Realization Binding

# The Selection Model — Constraint-Solved Realization Binding (Tensions T1 & T2)

> **Thesis:** Variant selection is not a per-op ladder and it is not a method the substrate calls and branches on. It is **constraint satisfaction over declared DATA**: each candidate part publishes a `Precondition` (a tree of typed primitives) plus a `priority`; one op-agnostic `Solver` evaluates preconditions against an injected `SelectionContext`, ranks the survivors, and binds one part per axis-slot into the single `final` `Kernel`. Cross-component legality — the thing composition's `negotiate` hole left blank — is a **three-tier** structure: distributed per-realization preconditions for *local* feasibility, one small declared pairwise `Legal` relation for *genuinely cross-component* constraints, and a **closed, enumerated** set of n-ary `Coupling` objects for the 2 known cases (MLO, pumpedMemory). `specialize_layers.py:40-211` deletes into `Solver.select`; the `:60`-vs-`:275` drift bug becomes unrepresentable because there is exactly one bitwidth threshold object and feasibility is separated from preference; the static-IP GEMM auto-selects for its common case as one new file with zero edits to the solver, the Legal relation, or any substrate.

The failure mode named in ranking §5-T2 is explicit: composition routes cross-part constraints through an under-specified `negotiate(spec)` that "could quietly become a new god-object," and multi-axis's central `Legal` table "grows with each new kind" — a data-driven echo of the god-switch. The discipline below is the resolution: **feasibility is local and distributed by default; cross-component knowledge is admitted only in a tier structure where each tier has a hard, reviewable growth bound, and no tier is ever keyed on an op type.**

This model plugs into composition's `Selector.resolve` (`01-composition.md §8`) — it *is* the body of that pass — and hands a fully-bound `Kernel` to the stitch layer's `assemble_region` (Stitch §S4). The two layers share one fact from two sides: the selection-time `Legal` rule `StaticIP ⇒ Embedded|External` is *why* the stitch-time region has no internal weight net (Stitch §S5). Selection proves the composition is legal; stitch wires it.

---

## Ruling on T1 — feasibility is DATA, not a method

**Decision: adopt capability-trait's precondition-as-DATA form. Reject the opaque `can_realize` method (composition's hybrid) and the self-advertising `feasible()` classmethod (evolutionary).**

The brief (§9.2) permits both a self-advertising method and a separate-pass-over-data. They are not equivalent under pressure. A method (`def can_realize(self, spec, tgt) -> bool`) is a black box: when the solver reports "no feasible realization," a developer cannot ask *which clause failed and why* without stepping a debugger through op-specific code. A `Precondition` value can answer that structurally — every primitive carries its own reason string, and `explain(select)` returns the full ranked candidate list with per-clause verdicts. This is the debuggability the current god-switch never had, and it is the mechanism that makes the drift bug (T1's motivating defect, §Sel9) *structurally* impossible rather than merely relocated.

Composition's `ComputeStrategy.can_realize(spec, tgt) -> Feasibility` (`01-composition.md §2`) is trivially reskinned: the method body **is** a `Precondition` object, and the "call" is `precondition.eval(ctx)`. Nothing about composition's one-final-Kernel structure changes — the six parts are still composed objects; we have only changed how each part *advertises* the contexts it can run in, from opaque code to inspectable data.

The honest cost, named by capability-trait's own weakness #3 and the ranking (§1, "~2 of 6 predicates need a `Predicate(named_fn)` escape hatch"): two predicates (elementwise broadcast, MVU narrow-from-initializer) resist trivial expression. **Ruling: those two become first-class composite primitives (`BroadcastCompatible`, `ParamIsNarrow`), not `Predicate(named_fn)` escape hatches.** An escape hatch reopens opaque per-op logic; a first-class primitive keeps everything as inspectable, serializable data at the cost of one governed vocabulary entry each. The primitive set is closed and reviewed (§Sel12). This is the difference between "preconditions are DATA" being true and being 90%-true.

---

## Sel1. The primitives — typed constraint DATA

A `Precondition` is a tree of primitives over an injected `SelectionContext`. Every primitive is a frozen dataclass with two methods: `eval(ctx) -> bool` and `reason(ctx) -> str` (the human-readable failure explanation). Nothing in a primitive reads an op type or a class name.

```python
class SelectionContext:                       # INJECTED — no module globals (criterion #10)
    in_dtypes:   tuple[DataType, ...]         # from graph
    out_dtypes:  tuple[DataType, ...]
    attrs:       AttrView                     # typed read-only node attributes
    fpgapart:    FpgaPart                     # .family, .dsp_block, .is_versal — was is_versal(global) at specialize_layers.py:37
    param_stats: ParamStats | None            # min/max/is_narrow of initializer; None ⇒ dynamic weights
    folding:     FoldingView                  # .pe(), .simd(), .scheme

# Leaf primitives (data, serializable, explain()-able)
@dataclass(frozen=True)
class DtypeBits:   port: PortId; op: Literal["<=",">=","==","<",">"]; n: int
@dataclass(frozen=True)
class DtypeSigned: port: PortId; want: bool
@dataclass(frozen=True)
class DtypeIs:     port: PortId; dt: DataType
@dataclass(frozen=True)
class DtypeIsInteger: port: PortId
@dataclass(frozen=True)
class AttrEquals:  name: str; val: object
@dataclass(frozen=True)
class AttrDivides: a: str; b: str
@dataclass(frozen=True)
class DeviceFamily: any_of: tuple[str, ...]
@dataclass(frozen=True)
class DspBlockIn:  any_of: tuple[str, ...]
# the two that would otherwise be escape hatches — promoted to first-class primitives:
@dataclass(frozen=True)
class ParamIsNarrow:        port: PortId       # reads ctx.param_stats; None ⇒ treated as wdt.min()
@dataclass(frozen=True)
class BroadcastCompatible:  lhs: PortId; rhs: PortId; out: PortId

# Combinators
@dataclass(frozen=True)
class AllOf:   items: tuple[Precondition, ...]
@dataclass(frozen=True)
class AnyOf:   items: tuple[Precondition, ...]
@dataclass(frozen=True)
class Not:     item: Precondition
@dataclass(frozen=True)
class Implies: cond: Precondition; then: Precondition
```

`PortId` is a *role*, not a pin name (`WEIGHTS`, `ACTIVATIONS`, `IN`, `OUT`, `LHS`, `RHS`) — the same semantic vocabulary the stitch layer's `Role` uses (Stitch §S1.1). A precondition never names `in1_V`; it names `WEIGHTS`. This keeps selection op-agnostic for the identical reason stitch is: the coordination key is the semantic role.

---

## Sel2. The Realization — a candidate for one axis-slot

A `Realization` is a candidate to fill **one** composed slot of the `Kernel` (compute, weight-delivery, or memory). It carries the part object plus its selection metadata. This is the union of composition's part-Protocols and capability-trait's `Capabilities`:

```python
@dataclass(frozen=True)
class Realization:
    slot:        Literal["compute", "weights", "memory"]
    kind:        str                    # "rtl" | "hls" | "staticip" | "embedded" | "decoupled" | ...
    provides:    frozenset[Trait]       # e.g. {Compute(MatMul)}, {Delivers(WEIGHTS, decoupled)}
    precondition: Precondition          # LOCAL feasibility — the tier-1 DATA (T2)
    priority:    int                    # staticip=30 > rtl=20 > hls=10  (was the "prefer RTL" ladder, now data)
    part:        object                 # the ComputeStrategy | WeightDelivery | MemoryStrategy to compose in
```

`provides` is a set of `Trait` values (the six-axis vocabulary from capability-trait §"Trait vocabulary"); it is used for **semantic matching** (does this realization even apply to a MatMul op) and for the tier-2 `Legal` relation (§Sel5). `precondition` is the **local feasibility** (§Sel4). `priority` is the **preference** — the single most important separation in this whole model, because it is what makes the drift bug unrepresentable (§Sel9): *feasibility (can it run) and preference (should we prefer it) are two different fields, never fused into one threshold.*

Registration is decorator-based, appending to a module-level pool — no shared edit:

```python
@realizes(slot="compute", op=MatMul)
class MvuRtl:  # the RtlCompute part
    provides     = frozenset({Compute(MatMul)})
    priority     = 20
    precondition = AllOf((...))          # §Sel7 — the data-ized _mvu_rtl_possible
```

---

## Sel3. The Solver — one op-agnostic pass with `explain()`

`Solver.select` is the entire body of composition's `Selector.resolve`. It is ~40 lines, reads no op type, no class name, no pin name, and fills every axis-slot by the same mechanism:

```python
def resolve(self, spec: KernelSpec, env: BuildContext) -> Kernel:
    ctx = SelectionContext.from_spec(spec, env)        # injected world, hermetic
    compute = self._bind("compute", spec, ctx)         # one solve
    weights = self._bind("weights", spec, ctx, given={"compute": compute})   # tier-2 aware
    memory  = self._bind("memory",  spec, ctx, given={"compute": compute, "weights": weights})
    self._check_couplings(spec, compute, weights, memory, ctx)   # tier-3 (§Sel6)
    return Kernel(spec.contract, compute.part, weights.part, memory.part,
                  folding_for(spec), ports_for(spec, weights, memory), reference_for(spec))

def _bind(self, slot, spec, ctx, given=()) -> Realization:
    cands   = self.pool.for_slot(slot, spec.contract.semantic())    # no if-ladder, no optype
    legal   = [c for c in cands if self._legal_given(c, given)]     # tier-2 pairwise filter (§Sel5)
    feasible, rejected = [], []
    for c in legal:
        fails = c.precondition.explain(ctx)            # [] or [(primitive, reason), ...]
        (feasible if not fails else rejected).append((c, fails))
    pref = spec.contract.attr("preferred_impl_style")
    feasible = self._apply_preference(feasible, pref)  # HARD filter or SOFT boost — NEVER a feasibility gate
    if not feasible:
        raise Infeasible(self._trace(slot, feasible, rejected))    # explain(select): full ranked list + reasons
    feasible.sort(key=lambda c: (c.pref_rank, -c.priority, c.est_cost(ctx), c.kind))  # deterministic
    return feasible[0]
```

Tie-break is fully specified and deterministic (reproducible builds): (1) user preference honored; (2) integer `priority`; (3) `est_cost`; (4) `kind` lexicographic. `explain(select)` surfaces every candidate's verdict — the answer to "why HLS here and not RTL?" that the god-switch's silent `warn_str` duplication (`specialize_layers.py:109-211`) could never give.

The reference/runner split (composition's diamond-kill) is **orthogonal to selection**: `reference_for(spec)` is chosen by the op's `contract`, not by the solver — a container has a recursive reference, an infra op has `IdentityMover`, MVAU has matmul+multithreshold. The solver only ranks *implementation* candidates. Golden semantics are never a solve.

---

## Ruling on T2 — three tiers, each with a hard growth bound

**Decision: adopt a three-tier constraint structure. Reject composition's unbounded `negotiate`, reject multi-axis's single central `Legal` table as the *only* mechanism, reject ir-lowering's *nothing* (which collapsed at the anchor).**

The disagreement among the panel is really about *where* cross-component knowledge lives and *how fast that place grows*. The ruling makes growth the governing criterion:

| Tier | What it constrains | Where it lives | Grows when… | Bound |
|---|---|---|---|---|
| **1 — distributed precondition** | LOCAL feasibility: can THIS realization run in THIS context, alone | on each `Realization` (§Sel4) | a new realization is added | O(realizations); **adding an op adds zero shared rows** |
| **2 — pairwise `Legal`** | cross-component: one slot's choice forbids/requires another's | one small module-level relation (§Sel5) | a genuinely new *kind of pairwise coupling* appears | closed, ~3 rows today; static-IP adds **zero** |
| **3 — n-ary `Coupling`** | ≥3-way / parent-relative interactions | a closed, enumerated tuple (§Sel6) | one of the 2 known n-ary cases changes, or a 3rd is discovered | **hard-enumerated: 2 objects (MLO, pumpedMemory)** |

The discipline that keeps this from becoming the god-switch: **the vast majority of all constraints are tier 1 (local, distributed, zero shared surface).** Tier 2 is entered *only* when a constraint genuinely spans two slots and cannot be expressed as either slot's local precondition. Tier 3 is a hard-capped list of named exceptions. Every tier is keyed on *semantic role/trait*, never on op type — a `Legal` row says `StaticIP ⇒ Embedded|External`, never `if op == "MVAU"`.

---

## Sel4. Tier 1 — distributed local preconditions (the common case)

All six `_*_rtl_possible` ladders are tier 1: each is a local feasibility question ("can RTL-MVU run given these dtypes/device") with **no** cross-component dependency, so each lives as the `precondition` field on its own `Realization`. This is where `specialize_layers.py:235-374` goes (§Sel7). Adding a 6th op adds its realizations with their local preconditions and touches no shared surface — the definition of criterion #2 passing at the selection layer.

The key property: **tier 1 has no central registry of constraints.** There is no table that grows. `MvuRtl.precondition` is co-located with `MvuRtl`. This is capability-trait's distributed answer, and it is strictly more principled than multi-axis's central `Legal`-for-everything, because a local feasibility fact has no business being in a shared relation.

---

## Sel5. Tier 2 — the small declared pairwise `Legal` relation

Some constraints genuinely span two slots and *cannot* be a local precondition, because neither slot can see the other's choice from its own context. The canonical one, the anchor's weight-ownership constraint that **collapsed ir-lowering** (ranking §1, §4): a static-IP compute owns its weight path, so it is legal only with a weight-delivery that does not also drive coefficients.

This is one row of a closed, module-level relation over *traits* (not op types):

```python
@dataclass(frozen=True)
class LegalRule:
    when:   Trait                          # a trait on one bound slot
    then:   frozenset[Trait]               # the allowed traits on another slot (AnyOf)
    reason: str

LEGAL = (
    LegalRule(Compute(StaticIP),
              frozenset({Delivers(WEIGHTS, embedded), Delivers(WEIGHTS, external)}),
              "hard IP owns its coefficient path; a streamer would double-drive weights"),
    LegalRule(Compute(StaticIP),
              frozenset({Store(rom), Store(offchip)}),
              "hard IP embeds params or reads them external; no on-chip streamer store"),
    # ... the entire relation is ~3 rows. It is NOT per-op. It is per genuinely-new-coupling.
)
```

`_legal_given` in the solver (§Sel3) filters a slot's candidates against the already-bound slots using `LEGAL`. It reads only traits. **The relation grows only when a genuinely new *kind* of pairwise coupling is invented — reviewed once, then never touched.** A new *op* adds zero rows. A new *compute kind that owns its weights* reuses the existing `Compute(StaticIP)`-style rows by tagging itself with the ownership trait — it does not add a row (§Sel10). This is the bounded discipline that answers multi-axis's "the table grows with each new kind": it grows with new *couplings*, of which there are a handful, not new *kinds*, of which there are many.

**This is exactly the fact the stitch layer consumes from the other side.** `StaticIP ⇒ Embedded` means the bound weight-delivery emits *no* streamer cell and publishes *no* `WEIGHT_SOURCE` (Stitch §S5); so the region has one cell and no internal weight net. Selection guarantees the composition is legal before stitch ever runs; stitch's `RoleBinding` then wires whatever ports actually got published. Two layers, one fact.

---

## Sel6. Tier 3 — the bounded n-ary escape hatch (exactly 2 objects)

Two known constraints are irreducibly n-ary and one is parent-relative — flagged by every adversary (multi-axis strains on both; the digest calls MLO "a cross-KERNEL coupling, not merely a thin axis boundary"). Pairwise `Legal` cannot express them. **Ruling: model them as a *closed, enumerated* tuple of `Coupling` objects — not a general n-ary predicate language (which would re-become the god-switch), but a hard-capped list of named exceptions.**

```python
class Coupling(Protocol):
    def check(self, compute, weights, memory, ports, ctx) -> list[str]: ...   # [] = ok

KNOWN_COUPLINGS: tuple[Coupling, ...] = (PumpedClockCoupling(), MloCoupling())
# ^ This tuple is the ENTIRE n-ary surface of the system. Adding to it is a governed,
#   reviewed, central edit expected to happen ~never. It has 2 entries.
```

**`PumpedClockCoupling`** — the `WeightDelivery × Memory × Clock` triple that composition's adversary caught at `matrixvectoractivation.py:1098-1112` and multi-axis strained on. It validates: if `MemoryStrategy.pumped`, then (a) the bound compute/streamer must support a 2x clock, and (b) the region will carry a `PUMP` clock domain. Selection *sets* `pumpedMemory` on the `MemoryStrategy` as a data field; the stitch layer's `ClockPlan.fuse` (Stitch §S3) *consumes* it. The coupling object only checks consistency at bind time — it does not wire anything. The decision is made once, as data, at composition; there is zero `if` in the stitch path because selection already resolved the triple.

**`MloCoupling`** — `LoopFetched ⇒ OffChipDma ∧ AxiMmPort(OFFCHIP) ∧ IndexPort(INDEX_SINK) ∧ enclosing Container`. This is the parent-relative one: `mlo_max_iter`, `LAYER_OFFS`, `N_LAYERS` are provided by the finn_loop container, not the MVAU (digest: "MLO is a parent-child protocol"). The coupling validates that a `LoopFetched` weight-delivery is only bound when the memory is `OffChipDma`, the port set includes the AXI-MM and index ports (which the stitch layer promotes to the region boundary, Stitch §S6 "MLO"), and the kernel is inside a container that will supply the loop-control context. It forces the memory choice and the port set as a unit — the one place the clean "motion vs substrate" split is genuinely thin, made explicit and bounded rather than smeared across ~12 sites (`hwcustomop.py:100`).

The honest residual (§Sel12): this tuple is the bounded remainder of composition's `negotiate` hole. It is not zero cross-component code. But it is **2 named, reviewable objects** enumerated in one place, each covering a real hardware coupling, versus per-op `if mem_mode`/`if mlo` scattered through the substrate. The reviewer's job is precisely to keep this list at ~2 and reject any `Coupling` that is really an op-specific special case in disguise.

---

## Sel7. The six ladders re-expressed as tier-1 DATA (file:line)

Grounded in `specialize_layers.py:235-374`. Each `_*_rtl_possible` becomes the `precondition` of its RTL realization. (Correspondences verified against the source read above.)

**`_mvu_rtl_possible` (`:235-278`) → `MvuRtl.precondition`:**
```python
AllOf((
    AttrEquals("noActivation", 1),               # :246-249  activation off
    AttrEquals("binaryXnorMode", 0),             # :247      not binary-xnor
    DtypeSigned(WEIGHTS, True),                   # :252-253  signed weights
    DtypeBits(ACTIVATIONS, ">=", 2),             # :274-275  SINGLE source of truth for the min
    DtypeBits(WEIGHTS,     ">=", 2),             # :276
    DtypeBits(WEIGHTS,     "<=", 8),             # header rule: DSP48 8sx8u
    Implies(Not(ParamIsNarrow(WEIGHTS)),         # :266-269  non-narrow ⇒ not DSP48E1
            Not(DspBlockIn(("DSP48E1",)))),
))
```
`ParamIsNarrow` reads `ctx.param_stats` (None ⇒ dynamic ⇒ `wdt.min()`, reproducing `:262-266`) — a first-class primitive, not an escape hatch.

**`_vvu_rtl_possible` (`:281-298`) → `VvuRtl.precondition`:**
```python
AllOf((
    AttrEquals("noActivation", 1),                                   # :287
    DeviceFamily(("versal",)),                                       # :289
    AnyOf(( DtypeBits(ACTIVATIONS, "<=", 8),                         # :294
            AllOf((DtypeBits(ACTIVATIONS, "==", 9), DtypeSigned(ACTIVATIONS, True))) )),
    DtypeBits(WEIGHTS, "<=", 8),                                     # :295
    DtypeSigned(WEIGHTS, True),                                      # :296
))
```

**`_dwc_determine_impl_style` (`:221-232`) → `DwcRtl.precondition`:** `AnyOf((AttrDivides("inWidth","outWidth"), AttrDivides("outWidth","inWidth")))` — the `:227-229` ratio check.
**`_requant_rtl_possible` (`:363-374`) → `RequantRtl.precondition`:** `AllOf((DtypeIsInteger(IN), Not(DtypeSigned(OUT, True)), AttrEquals("narrow", 0)))`.
**`_layernorm_rtl_possible` (`:350-360`) → `LayerNormRtl.precondition`:** `AllOf((DeviceFamily(("versal",)), DtypeIs(IN, FLOAT32)))`.
**`_elementwise_rtl_possible` (`:301-347`) → `ElementwiseRtl.precondition`:**
```python
AllOf((
    DeviceFamily(("versal",)),                                       # :305
    AnyOf((DtypeIs(LHS, FLOAT32), DtypeIs(RHS, FLOAT32))),           # :312-323  at least one float32
    Not(AllOf((AttrEquals("lhs_style","const"), AttrEquals("rhs_style","const")))),  # :330-331
    BroadcastCompatible(RHS, LHS, OUT),                              # :336-345  the whole broadcast loop, one primitive
))
```
The `:340-345` broadcast loop folds into one `BroadcastCompatible` primitive — the second predicate that would otherwise be an escape hatch, promoted to first-class data.

All six now live *with their realization*, evaluated by one uniform pass. `specialize_layers.py` collapses to `Solver.resolve`.

---

## Sel8. What deletes: `specialize_layers.py:40-211`

The entire `_determine_impl_style` function (`:40-218`) — the `if optype == "MVAU"/"VVAU"/"LayerNorm"/"Requant"/elementwise` ladder, duplicated across the `impl_style == ""` auto branch (`:53-101`) and the `impl_style == "rtl"` explicit branch (`:122-211`), plus the string-concat instantiation `optype + "_" + impl_style` (`:399`) — is **deleted**. It is replaced by:

- the per-realization `precondition` DATA (§Sel7) — tier 1;
- the `LEGAL` relation (§Sel5) — tier 2;
- `KNOWN_COUPLINGS` (§Sel6) — tier 3;
- `Solver.resolve` (§Sel3) — the one op-agnostic pass;
- `priority` fields — the "prefer RTL for simple layers" heuristic (`:51-52` comment, `:90` fall-through `return "rtl"`) expressed as data, not control flow.

The `_*_rtl_possible` helpers (`:235-374`) move onto realizations. The `warn_str` duplication (`:109-211`, one bespoke string per arm) is replaced by `explain(select)`'s per-clause reasons. The `SpecializeLayers` transformation (`:377-416`) becomes: for each hw node, `kernel = Solver.resolve(spec, env)` — no `helper.make_node(optype + "_" + impl_style)`, because there is no `_hls`/`_rtl` class to name; there is one `Kernel` with a bound compute part.

---

## Sel9. The `:60/:275` drift bug, made structurally impossible (worked)

**The bug, precisely.** For MVAU, "minimum bitwidth for RTL" is encoded in **two** places that **disagree**:

- The auto path (`impl_style == ""`, `:57-65`) pre-gates on `idt.bitwidth() >= 4 and wdt.bitwidth() >= 4` (`:60-61`) *before* calling `_mvu_rtl_possible`.
- The explicit path (`impl_style == "rtl"`, `:137-139`) calls `_mvu_rtl_possible` directly, with **no** `>= 4` pre-gate.
- `_mvu_rtl_possible` itself (`:274-276`) gates on `2 <= idt.bitwidth()` and `2 <= wdt.bitwidth()`.

**Consequence:** a 3-bit×3-bit MVAU resolves to **HLS** when the user leaves `preferred_impl_style` unset (fails the `>= 4` pre-gate) but to **RTL** when the user pins `"rtl"` (passes `_mvu_rtl_possible`'s `>= 2`). Same node, same hardware capability, two different backends — a silent correctness/perf divergence driven only by whether an unrelated attribute was set. The root cause is **two sources of truth for one threshold**, and a **fusion of feasibility with preference** (the `>= 4` is a "prefer RTL only if wide enough" heuristic wearing a feasibility gate's clothing).

**Why the DATA model makes it unrepresentable.** Two structural facts, not discipline:

1. **One threshold object.** The minimum bitwidth is `DtypeBits(ACTIVATIONS, ">=", 2)` — it exists exactly once, in `MvuRtl.precondition` (§Sel7). There is no auto-vs-explicit *branch*: `Solver.resolve` evaluates the same precondition object regardless of whether the user set a preference (preference is applied as a `pref_rank` sort key / hard filter in `_apply_preference`, §Sel3 — it never re-asks feasibility). There is no second place for a `>= 4` to live and drift from a `>= 2`.

2. **Feasibility and preference are different fields.** "Can RTL-MVU run?" is `precondition` (feasibility). "Should we prefer RTL over HLS when both are feasible?" is `priority` / `est_cost` (preference). The `:60` `>= 4` was a preference heuristic ("RTL isn't worth it below 4 bits") miscoded as a feasibility gate. In this model it can only be expressed as preference — e.g. an `est_cost` that ranks RTL below HLS when `bitwidth < 4` — which is a *ranking* among feasible candidates, applied uniformly to both the auto and explicit paths because there is only one path. A preference cannot silently flip a node to a *different feasibility verdict* depending on an unrelated attribute, because feasibility is not where preference lives.

The bug is not "fixed"; it is **unrepresentable**. To reintroduce it a developer would have to add a *second* bitwidth primitive to a *different* object and wire a *second* evaluation path keyed on `preferred_impl_style` — i.e. re-build the god-switch by hand, which review rejects on sight.

---

## Sel10. The static-IP GEMM auto-selects with ZERO substrate edits (worked)

The forcing function (brief §3, §6.5). One new file in `compute/`:

```python
@realizes(slot="compute", op=MatMul)
class MvuInt8StaticIp:                          # the StaticIpCompute part
    provides     = frozenset({Compute(StaticIP)})      # NOTE: the ownership trait, reused by LEGAL
    priority     = 30                                   # > MvuRtl(20) > MvuHls(10)
    precondition = AllOf((
        DtypeIs(ACTIVATIONS, INT8),
        DtypeIs(WEIGHTS,     INT8),
        AttrEquals("noActivation", 1),
        DtypeBits("MW", "<=", 512),  DtypeBits("MH", "<=", 512),   # hand-tuned range
        # (folding scheme compatibility checked via provides/Trait match)
    ))
    def emit(self, e):     return e.sink.instantiate_vlnv("finn.int8gemm:1.0", e.port_map)
    def runner(self, mode, env): return RtlsimRunner(self.artifacts, env.simulator)
```

**Auto-selection for the common case.** For an INT8 512×512 no-activation MatMul, `Solver.resolve` (§Sel3): the pool's compute candidates for `MatMul` are `{MvuHls, MvuRtl, MvuInt8StaticIp}` — gathered by `pool.for_slot("compute", MatMul)`, no if-ladder. All three preconditions pass; `_apply_preference` (no user pin) keeps all three; the sort key `-priority` puts `MvuInt8StaticIp` (30) first. It is bound. **Transparent fallback:** for a 4-bit or 1024×1024 or activation-fused MatMul, `MvuInt8StaticIp.precondition` fails (e.g. `DtypeIs(ACTIVATIONS, INT8)` is false), it is dropped into `rejected` with a reason, and the solver falls through to `MvuRtl` (20) or `MvuHls` (10) — exactly the "prefer static-IP, else RTL, else HLS" behavior, as *data*, with `explain(select)` showing precisely why the IP was rejected.

**Zero substrate edits, enumerated:**
- `Solver.resolve` / `_bind`: untouched — they read `slot`, `provides`, `precondition`, `priority`, never a kind string. `MvuInt8StaticIp` is one more entry in `pool.for_slot("compute", MatMul)`.
- `LEGAL` (§Sel5): **untouched.** The weight-ownership constraint is already `Compute(StaticIP) ⇒ Embedded|External`. `MvuInt8StaticIp` tags itself `Compute(StaticIP)` and the *existing* rows apply. If a *second* static-IP kind is added tomorrow, it also tags `Compute(StaticIP)` (or a shared ownership trait) and adds **zero** rows. This is the bounded-growth discipline in action: the relation grows with new *couplings*, and "hard IP owns its weights" is not a new coupling — it is the one already there.
- `KNOWN_COUPLINGS` (§Sel6): untouched — static-IP embedded weights are pairwise (tier 2), not n-ary.
- The `Kernel` class, the stitch layer's `resolve`/`TclRenderer`/`STANDARD_BINDINGS`: untouched (Stitch §S5 — the IP's cell speaks the role vocabulary, so it stitches by construction).

The static-IP GEMM is a peer to HLS/RTL under one contract, selected by the identical declared-capability mechanism as everything else. That single property — new file, zero shared edits, auto-selected by priority, transparent fallback by precondition — is the proof the abstraction is real and not a rename of the {HLS, RTL} split (brief §3). It is criterion #2 and criterion #3 holding *together* at the selection layer, which is precisely where ir-lowering collapsed (ranking §4).

---

## Sel11. Handoff to the stitch layer — one fact, two sides

Selection produces a fully-bound `Kernel`; stitch consumes it via `assemble_region(kernel, env)` (Stitch §S4). The two layers meet at exactly three points, and in each the selection tier *is* the stitch precondition:

| Selection decides (this model) | Stitch consumes (Stitch layer) |
|---|---|
| Tier-2 `LEGAL`: `StaticIP ⇒ Embedded` ⇒ weight-delivery emits no streamer | Region has one cell, no `WEIGHT_SOURCE`, no internal weight net (§S5) |
| Tier-3 `PumpedClockCoupling`: sets `MemoryStrategy.pumped` as data | `ClockPlan.fuse` routes `PUMP`→`ap_clk2x` or fuses to `ap_clk`, zero `if` (§S3) |
| Tier-3 `MloCoupling`: forces `OffChipDma` + AXI-MM/index ports | Ports promote to region boundary → up to finn_loop → HBM (§S6, §S7) |

Selection never wires a net; stitch never re-decides feasibility. The `Legal` relation and the `RoleBinding` table are the same knowledge viewed from selection (is this composition legal?) and from stitch (which ports connect?). Neither is a god-object because each is a small declared table read by a generic pass.

---

## Sel12. Honest residual and governance

A responsible ruling names what stays a shared surface; each is bounded and declared so it cannot leak into per-op code:

1. **The primitive vocabulary is closed and governed.** `DtypeBits`, `ParamIsNarrow`, `BroadcastCompatible`, etc. are a reviewed set. A genuinely novel feasibility predicate requires adding a primitive — a governed central edit. This is capability-trait's weakness #2/#3 accepted honestly: coupling moves from op-type strings to a *bounded* typed vocabulary. The two predicates that would have been `Predicate(named_fn)` escape hatches are promoted to first-class primitives (§Sel1) so the "preconditions are DATA" claim holds with no opaque holes — at the cost of two vocabulary entries. **The danger a reviewer must guard:** a primitive like `MvuWeightsOk` creeping in. Primitives must stay *semantic and op-agnostic* (`DtypeBits`, `ParamIsNarrow`), never op-named, or the vocabulary becomes the god-switch reincarnated — the exact mirror of the stitch layer's `Role`-governance warning (Stitch §S8.4).

2. **The `LEGAL` relation is a shared surface, but bounded by coupling-kind, not op-count.** ~3 rows today; a new *op* adds zero, a new *compute-that-owns-weights* adds zero (reuses the ownership trait), only a genuinely new *kind of pairwise coupling* adds a row. Reviewed once. This is the bounded remainder of multi-axis's "the table grows" — real, but growth is O(coupling-kinds) ≈ constant, not O(ops).

3. **`KNOWN_COUPLINGS` is hard-enumerated at 2.** `PumpedClockCoupling`, `MloCoupling`. This is the bounded residual of composition's `negotiate` hole — cross-component knowledge is not zero, but it is 2 named objects in one place, each a real hardware n-ary coupling, versus `if mem_mode`/`if mlo` across ~12 sites. The reviewer's standing job: keep it at ~2 and reject any `Coupling` that is an op special-case in disguise. MLO's parent-relative nature (the container supplies `LAYER_OFFS`/`N_LAYERS`) is the genuinely hardest case and is contained here, made explicit rather than smeared.

4. **Preference math (`est_cost`) is declared-but-fallible.** The "prefer RTL below 4 bits" heuristic, recast as ranking (§Sel9), is a real cost function that can still be *wrong* (mis-rank), but it can no longer flip a *feasibility verdict* — the drift class is closed even if the preference class is not perfect. Typing surfaces it as a first-class `priority`/`est_cost` field instead of a bitwidth threshold buried in control flow.

---

### Summary of the ruling

- **T1 — feasibility is DATA.** Each realization carries a `Precondition` (typed primitive tree); one op-agnostic `Solver` evaluates it against an injected `SelectionContext`; `explain(select)` returns the ranked candidate list with per-clause reasons. The two would-be escape-hatch predicates become first-class primitives (`ParamIsNarrow`, `BroadcastCompatible`), keeping the DATA claim whole.
- **T2 — three tiers with hard growth bounds.** Tier 1: distributed per-realization preconditions (local feasibility, zero shared surface, where all six `_*_rtl_possible` go). Tier 2: a small declared pairwise `Legal` relation (~3 rows) for genuinely cross-component constraints like `StaticIP ⇒ Embedded|External`, growing per *coupling-kind* not per *op*. Tier 3: a closed, enumerated tuple of exactly 2 n-ary `Coupling` objects (MLO, pumpedMemory).
- **Deletion.** `specialize_layers.py:40-211` (`_determine_impl_style` + the duplicated per-op ladder + `optype+"_"+impl_style`) is replaced by `Solver.resolve` + per-realization precondition DATA + the `LEGAL` relation + `KNOWN_COUPLINGS`.
- **Drift structurally impossible.** One `DtypeBits(ACTIVATIONS,">=",2)` object, one evaluation path, and a hard separation of feasibility (`precondition`) from preference (`priority`/`est_cost`) — so the `:60`(`>=4`)-vs-`:275`(`>=2`) two-threshold divergence has no place to exist.
- **Static-IP auto-selects, zero substrate edits.** One new file (`MvuInt8StaticIp`, `priority=30`, INT8 precondition), auto-selected for its common case by priority, transparent precondition-fallback to RTL/HLS elsewhere, adding zero rows to `LEGAL`/`KNOWN_COUPLINGS` and zero edits to the solver — the forcing function passing at the exact seam where ir-lowering collapsed.
- **Handoff.** The selection tiers *are* the stitch preconditions: `Legal` ⇒ no `WEIGHT_SOURCE`; `PumpedClockCoupling` ⇒ `ClockPlan.fuse`; `MloCoupling` ⇒ boundary-promoted AXI-MM/index ports. Selection proves legality; stitch wires it.


---

# § 3. The Stitch Layer — Typed Coordination (Tension T4)

# The Stitch Layer — A Typed Coordination Model (Tension T4)

> **Thesis:** Block-design stitching is not fragment concatenation and it is not owned by any one composed part. It is a **placement-and-resolution** problem over a typed graph of cells whose ports carry *roles*, *clock domains*, and *protocols*. The god-method dissolves not by moving into a `Stitcher`, but by splitting into three things that were fused inside it: (1) parts emit **role-tagged `Cell`s** and nothing else; (2) a **`Region`** is the container that *owns every net between two parts*; (3) one generic **`RegionResolver`** turns role-tags + a clock plan + an address plan into nets, and one generic **`TclRenderer`** prints them. No part names another part. No resolver names an op. The clk2x branch, the weight net, and the MLO fan-out all become table lookups.

The failure mode we must avoid is explicit and named in the ranking §5-T4: every panel design produced a `Stitcher.compose(fragments)` that is a god-method wearing a new hat, because a *fragment* is still a blob of TCL that one part emitted with knowledge of another part's pin names. The discipline below is the opposite: **a part may only describe itself.** The knowledge that "a weight source connects to a weight sink" is not in any part and not in the resolver — it is one row of a declared, op-agnostic `RoleBinding` table.

---

## S1. The typed model

### S1.1 `Port` — the atom (replaces the `intf_names` dict)

Today a port is an untyped tuple buried in `get_verilog_top_module_intf_names()` (`hwcustomop.py:116-142`): `intf_names["s_axis"] = [("in0_V", width), ...]`, `intf_names["aximm"] = [("m_axi_hbm", addr_bits)]`, `intf_names["clk2x"] = [...]`. Protocol is encoded by *which dict key* the tuple sits under; role and clock domain are nowhere. iodma bifurcates this dict by `direction` (`iodma_hls.py:391-399`); MVAU rebuilds it inline six times inside the god-method (`:930-933`).

Replace with one closed, typed value:

```python
class Protocol(Enum):        # what wire discipline this port speaks (was the dict KEY)
    AXIS       = auto()      # xilinx.com:interface:axis_rtl:1.0
    AXIMM      = auto()      # xilinx.com:interface:aximm_rtl:1.0
    AXILITE    = auto()      # xilinx.com:interface:aximm_rtl:1.0 (slave, regmap)
    CLK        = auto()      # -type clk
    RST        = auto()      # -type rst
    SCALAR     = auto()      # ap_none control bit (done_if, sim_finish)

class Role(Enum):            # the SEMANTIC job of the port — the load-bearing tag
    DATA_IN       = auto()   # a tensor stream the kernel consumes
    DATA_OUT      = auto()   # a tensor stream the kernel produces
    WEIGHT_SINK   = auto()   # compute's coefficient input   (MVAU in1_V)
    WEIGHT_SOURCE = auto()   # delivery's coefficient output  (streamer m_axis_0 / out0_V)
    INDEX_SINK    = auto()   # MLO iteration index input      (in_idx0_V on streamer)
    INDEX_SOURCE  = auto()   # MLO iteration index from parent (in_idx0_V on hier)
    OFFCHIP       = auto()   # AXI-MM to DRAM/HBM             (axi_mm, m_axi_gmem, m_axi_hbm)
    CONFIG        = auto()   # runtime-writeable axilite      (s_axi_control)
    SIDEBAND      = auto()   # TLAST/TKEEP
    CLOCK         = auto()   # a clock consumer (domain in .clock, see S3)
    RESET         = auto()
    CONTROL       = auto()   # scalar control (done_if, sim_finish)

@dataclass(frozen=True)
class Port:
    name:     str                      # the concrete HDL pin name, e.g. "in1_V"
    role:     Role
    proto:    Protocol
    dir:      Literal["I", "O"]        # master/slave resolved from role+proto at render
    clock:    ClockDomain | None = None    # ONLY for role==CLOCK or clocked interfaces (S3)
    shape:    FoldedShape | None = None     # ONLY populated for DATA_IN/DATA_OUT/WEIGHT_* streams
    width:    int | None = None            # bit width for non-tensor streams (aximm addr, config)
    boundary: bool = False             # True ⇒ promote to region boundary, don't match internally
```

Two rules make this the fix for criterion #7 and the iodma/tlastmarker raises:

- **`shape` is present *by role*, not by op.** Only `DATA_*`/`WEIGHT_*` ports carry a `FoldedShape`. An `OFFCHIP` port has a `width` (address/data bits) and *no* `shape` — so the generic folded-shape pass iterates `[p for p in cell.ports if p.shape is not None]` and **never touches** iodma's AXI-MM side. `iodma_hls.py:116/131` (`get_folded_*_shape` raising for the memory-mapped port) is a category error the type structurally forbids — you cannot ask a shape-less port for its shape.
- **`role` is the coordination key, `proto`/`name` are rendering detail.** No matcher ever reads `name`. A part is free to call its weight input `in1_V`, `weights_V`, or `param_axis` — as long as it is tagged `WEIGHT_SINK` it wires correctly. This is what makes the layer op-agnostic (S5).

### S1.2 `Cell` — an independently-emitted component (what a part produces)

Each composed part emits exactly one `Cell` and nothing about any other part. This is the entirety of a part's stitch obligation — there is no `contribute_ipi(bd)` that mutates a shared command list.

```python
class CellKind(Enum):
    HIER      = auto()   # create_bd_cell -type hier         (a nested Region, S7)
    MODULE    = auto()   # create_bd_cell -type module -reference <rtl>
    VENDOR_IP = auto()   # create_bd_cell -type ip -vlnv <vlnv>   (static-IP, finn_loop body, IODMA)
    BD_REF    = auto()   # create_bd_cell -type hier -reference <packaged_bd>  (weight streamer)

@dataclass(frozen=True)
class Cell:
    inst:   str                         # instance name inside the region (node_name + "_wstrm")
    kind:   CellKind
    ref:    str                         # rtl module name / vlnv / packaged-bd name
    ports:  tuple[Port, ...]            # the cell's OWN ports, role-tagged
    sources: SourceManifest            # typed file list (S8) — replaces os.listdir last-match
    packaging: tuple[PackagingDirective, ...] = ()   # opaque-but-scoped IP surgery (S8)
```

A `ComputeStrategy.emit_cell()` returns the MVAU compute IP as a `Cell` whose `in1_V` port is tagged `WEIGHT_SINK`. A `WeightDelivery.emit_cell()` returns the streamer as a `Cell` whose output is tagged `WEIGHT_SOURCE`. **Neither cell references the other.** That is the whole point.

### S1.3 `Net` and `Region` — where connections live

```python
@dataclass(frozen=True)
class Net:                              # a resolved connection (the output of the resolver)
    proto:  Protocol
    src:    PortRef                     # (cell_inst | BOUNDARY, port_name)
    dst:    PortRef

@dataclass(frozen=True)
class Region:
    name:     str                       # the enclosing hier cell (node_name, or bd_design)
    cells:    tuple[Cell, ...]          # independently-emitted parts placed side by side
    boundary: tuple[Port, ...]          # the region's OWN external pins (the hier's clk/din/dout/…)
    clock_plan:   ClockPlan             # S3
    address_plan: AddressPlan           # S8
    bindings: tuple[RoleBinding, ...]   # the op-AGNOSTIC role-match rules (a constant table, S2)
    nets:     tuple[Net, ...] = ()      # filled by RegionResolver — parts never write here
```

**This is the answer to "who owns the weight-streamer → compute net":** the `Region` owns it. The net is not in the compute `Cell`, not in the weight `Cell`, and not hand-written in a resolver branch. It is *derived* by the resolver from two facts each part published independently (one `WEIGHT_SOURCE`, one `WEIGHT_SINK`) plus one op-agnostic binding rule that lives on the region. Ownership is structural: a net between cell A and cell B belongs to the smallest region containing both, exactly as scope works in a programming language.

---

## S2. The ownership ruling and the role-matcher

A `RoleBinding` is a pure data rule — a closed, op-independent table shared by *every* region in the system:

```python
@dataclass(frozen=True)
class RoleBinding:
    source: Role
    sink:   Role
    proto:  Protocol
    arity:  Literal["1:1", "1:N"] = "1:1"

STANDARD_BINDINGS = (
    RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS),     # streamer.out → compute.in1_V
    RoleBinding(INDEX_SOURCE,  INDEX_SINK,  AXIS),     # MLO index fan-in
    RoleBinding(DATA_OUT,      DATA_IN,     AXIS),     # inter-kernel tensor edge (graph region, S7)
)
```

The resolver is ~40 lines, fully generic, and reads no op type and no cell name:

```python
def resolve(region: Region) -> Region:
    nets = []
    # 1. INTERNAL role matching: bind producer roles to consumer roles within this region
    for rule in region.bindings:
        srcs = ports_with(region, rule.source, rule.proto, boundary=False)
        snks = ports_with(region, rule.sink,   rule.proto, boundary=False)
        nets += match(srcs, snks, rule.arity)          # 1:1 asserts |srcs|==|snks|==1
    # 2. BOUNDARY promotion: expose region-external ports and wire cell→boundary
    for p in ports_with(region, boundary=True):
        bpin = region.boundary_pin_for(p)              # the hier's din/dout/axi_mm
        nets.append(Net(p.proto, ref(p), ref(bpin)))
    # 3. CLOCK/RESET distribution: every clock consumer → its domain's source pin
    nets += region.clock_plan.wire(region.all_clock_consumers())   # S3
    # 4. ADDRESS assignment for OFFCHIP/CONFIG masters
    nets += region.address_plan.assign(region.mem_mapped_ports())  # S8
    return replace(region, nets=tuple(nets))
```

The god-method's lines `1085-1089` (the `strm_out_name → in1_V` connect) are produced by step 1, rule row 1. Its lines `1136-1145` (externalize `din`/`dout` to the hier) are step 2. Its lines `1090-1112` (clocks/resets) are step 3. Its `assign_bd_address` (`:1127`) is step 4. Nothing in `resolve` mentions MVAU, `mem_mode`, `pumpedMemory`, or a pin name.

**Why this is not the god-method relocated:** the god-method knew *both* endpoints' pin names and the condition under which the net exists. Here, each endpoint is described only by its own emitter; the *existence* of the net is entailed by a rule that is the same for MVAU, static-IP GEMM, and a future op that has not been written. Adding an op adds zero rows. Adding a genuinely new *kind of coupling* (rare) adds one row to `STANDARD_BINDINGS`, reviewed once, never touched again.

---

## S3. Cross-clock as data — `ClockDomain` / `ClockPlan`

The sharpest branch in the god-method is `:1098-1112`:

```python
if self.get_nodeattr("pumpedMemory"):
    cmd.append("connect ... clk2x_name ... /wstrm/ap_clk2x")   # 2x pin → clk2x source
else:
    cmd.append("connect ... clk_name   ... /wstrm/ap_clk2x")   # 2x pin → regular clk
```

This is a `WeightDelivery × Clock × Memory` interaction expressed as an `if`. Model the clock as a **domain-valued label on the consumer port** and the routing as a **plan on the region**:

```python
class ClockDomain(Enum):
    BASE = auto()      # ap_clk           — the 1x clock
    PUMP = auto()      # ap_clk2x         — the 2x clock

@dataclass(frozen=True)
class ClockPlan:
    # which physical boundary pin drives each domain in THIS region
    source_pin: dict[ClockDomain, str]         # {BASE: "ap_clk", PUMP: "ap_clk2x"} — or …
    # a FUSION map: when the region has no 2x clock, PUMP collapses onto BASE's pin
    fuse:       dict[ClockDomain, ClockDomain] # {} normally; {PUMP: BASE} when not pumped

    def wire(self, consumers: list[Port]) -> list[Net]:
        nets = []
        for c in consumers:                        # c.clock is the domain the port WANTS
            dom = self.fuse.get(c.clock, c.clock)  # collapse if fused — no branch
            nets.append(Net(CLK, ref(self.source_pin[dom]), ref(c)))
        return nets
```

Now the pumped decision is made **once, as data, at composition time**, not at stitch time:

- The memstreamer `Cell` unconditionally publishes its clk2x consumer as `Port(name="ap_clk2x", role=CLOCK, clock=ClockDomain.PUMP)`. The streamer does not know or care whether a 2x clock exists.
- The `MemoryStrategy` (which owns the `pumpedMemory` geometry, per composition §memory_strategy) sets one field when the region is assembled:
  - `pumpedMemory == True`  → `ClockPlan(source_pin={BASE:"ap_clk", PUMP:"ap_clk2x"}, fuse={})`, and the region declares a `PUMP` boundary pin (the god-method's `:944-945` `create_bd_pin clk2x`).
  - `pumpedMemory == False` → `ClockPlan(source_pin={BASE:"ap_clk"}, fuse={PUMP: BASE})`, no `PUMP` boundary pin.

`ClockPlan.wire` then routes the streamer's `PUMP` consumer to `ap_clk2x` or, via `fuse`, back to `ap_clk` — **the exact two arms of `:1103/:1108`, with zero `if` in the stitch path.** `pumpedCompute` (`:939-942`, read via try/except today) is the identical mechanism: the compute `Cell` publishes a `PUMP` clock consumer iff `RtlCompute.pumpedCompute`, and the same plan routes it. The whole `clk2x` special case is: *does a cell publish a `PUMP` consumer, and does the plan carry a `PUMP` source or fuse it.* Two data questions, no branch, and it composes for any future part that wants a 2x clock without editing the resolver.

---

## S4. Assembly: parts → `Region` → TCL

The composition builder (not the resolver) assembles a `Region` from a `Kernel`'s parts. This is generic and ~15 lines:

```python
def assemble_region(kernel: Kernel, env: BuildContext) -> Region:
    cells = [p.emit_cell(env) for p in kernel.stitch_parts()]   # compute, weights, memory
    return Region(
        name        = kernel.node_name,
        cells       = tuple(cells),
        boundary    = kernel.ports.boundary_ports(),            # din/dout/clk/rst (+idx/aximm if MLO)
        clock_plan  = kernel.memory.clock_plan() | kernel.compute.clock_plan(),  # S3
        address_plan= kernel.weights.address_plan(),            # S8
        bindings    = STANDARD_BINDINGS,
    )

def stitch(kernel, env) -> TclScript:
    return TclRenderer().render(resolve(assemble_region(kernel, env)))
```

`TclRenderer.render(region)` is the *only* place TCL strings are produced, and it is uniform over the typed model:

```python
def render(self, r: Region) -> TclScript:
    out = [f"create_bd_cell -type hier {r.name}"]
    for p in r.boundary:  out += self._pin(r.name, p)          # create_bd_[intf_]pin
    for c in r.cells:
        out += c.sources.add_files_cmds()                       # typed manifest, no os.listdir
        out += [self._instantiate(c)]                           # kind→create_bd_cell dispatch
    for n in r.nets:      out += [self._connect(n)]             # proto→connect_bd_[intf_]net
    for d in r.all_packaging(): out += d.emit()                 # scoped surgery (S8)
    return TclScript("\n".join(out))
```

`_connect` dispatches on `Net.proto` (AXIS/AXIMM → `connect_bd_intf_net`; CLK/RST/SCALAR → `connect_bd_net`) — the two forms visible at `:1085` vs `:1090`. `_instantiate` dispatches on `CellKind`. Neither reads a role or an op. Every line the god-method emitted is now one of: a boundary pin, a typed `add_files`, a `CellKind` instantiation, a resolved `Net`, or a `PackagingDirective`.

---

## S5. Op-agnosticism: the two forcing tests edit nothing

**Add MVAU-with-external-weights.** `External` `WeightDelivery` emits *no streamer cell at all* and publishes no `WEIGHT_SOURCE`. Instead the compute's `WEIGHT_SINK` port is marked `boundary=True` (the coefficients arrive from outside the kernel). Step 1 of `resolve` finds zero `WEIGHT_SOURCE`/`WEIGHT_SINK` pairs; step 2 promotes the sink to a boundary pin. **No resolver edit, no renderer edit, no new binding row.** This is the god-method's `elif mem_mode == "external": self.instantiate_ip(cmd)` arm (`:1149-1153`) — but expressed as "the part emitted fewer ports," not as a branch the base takes.

**Add the static-IP INT8 GEMM.** `StaticIpCompute.emit_cell()` returns a `Cell(kind=VENDOR_IP, ref=<vlnv>)` whose ports are tagged `DATA_IN`/`DATA_OUT`/`WEIGHT_SINK`/`CLOCK(BASE)` exactly like the RTL MVAU's ports. If the hard IP embeds its weights, its cell publishes no `WEIGHT_SINK` and the region has one cell and no internal nets. If it wants a streamer, it publishes a `WEIGHT_SINK` and composes with `DecoupledStream` through the *same* binding row. Either way: **one new file in `compute/`, zero edits to `stitch.py`, `resolve`, `TclRenderer`, or `STANDARD_BINDINGS`.** The static-IP GEMM stitches by construction because it speaks the role vocabulary, not because the resolver learned about it. This is criterion #2 holding at the stitch layer, which is precisely where ir-lowering's `RenderTcl` collapsed.

---

## S6. The hard cases, worked

**MVAU decoupled (the canonical 236 lines).** Region `MVAU_0` with two cells: compute (`in0_V`:DATA_IN, `out0_V`:DATA_OUT, `in1_V`:WEIGHT_SINK, `ap_clk`:CLOCK/BASE, optional `ap_clk2x`:CLOCK/PUMP) and streamer `MVAU_0_wstrm` (`m_axis_0`:WEIGHT_SOURCE, `ap_clk`:CLOCK/BASE, `ap_clk2x`:CLOCK/PUMP, `s_axilite`:CONFIG if runtime-writeable). Boundary = `{din:DATA_IN, dout:DATA_OUT, ap_clk, ap_rst_n, (ap_clk2x if pumped)}`. `resolve`: step 1 emits the one weight net (`:1085-1089`); step 2 externalizes din/dout (`:1136-1145`); step 3 the `ClockPlan` routes both cells' BASE consumers to `ap_clk`, and the streamer's PUMP consumer to `ap_clk2x` or fused `ap_clk` (`:1100-1112`); step 4 assigns the CONFIG axilite address (`:1114-1127`). Every original line is reproduced with no branch.

**MVAU MLO / loop-fetched.** `LoopFetched` streamer cell additionally publishes `in_idx0_V`:INDEX_SINK and `axi_mm`:OFFCHIP. The region declares matching boundary ports `in_idx0_V`:INDEX_SOURCE(boundary) and `axi_mm`:OFFCHIP(boundary) — these are the god-method's `:958-966` extra bd-intf-pins. Step 1 matches nothing new *internally* for the index (the source is on the boundary), so INDEX is handled by step 2 boundary promotion, reproducing `:1066-1077`; the `axi_mm` OFFCHIP promotes to the region boundary and continues *up* to the finn_loop/graph region (S7), where it reaches HBM. The `mem_mode`/`mlo`/`dyn_input` three-way `if` at `:971/996/1040` is gone: each delivery variant simply emits a different `Cell` with different role-tagged ports; the resolver is identical for all three.

**MVAU dynamic-load.** `DynamicLoad` streamer publishes `s_axis_0`:DATA_IN(the dynamic weight stream). The region marks the compute-side dynamic weight path as a boundary DATA_IN (`win_name`, `:973-977`) and step 2 wires hier→streamer (`:1079-1084`). Same resolver.

**pumpedMemory clk2x.** Fully covered in S3: a `PUMP`-domain consumer on the streamer + a `ClockPlan` whose `fuse` map is set by `MemoryStrategy.pumpedMemory`. The `WeightDelivery × Clock × Memory` triple interaction is expressed as (delivery publishes the consumer) × (memory owns the plan) — two independent data contributions the resolver merges, never a nested `if`.

**finn_loop container HBM/aximm.** The finn_loop node is itself a `Region` (S7) whose cells are: the packaged loop-body BD (`VENDOR_IP`, `:968`), the fetch/store control cells, and the IODMA. The loop body publishes `m_axi_hbm`:OFFCHIP (its aximm, `:1137-1152`); the region's boundary carries `m_axi_hbm`:OFFCHIP(boundary), promoted to the top. The sibling-op `.dat`/tap dispatch (`finn_loop.py:421/712`) is replaced by reading each child region's published `OFFCHIP`/`WEIGHT_SOURCE` ports — the container asks "give me your role-tagged boundary ports," never "are you an MVAU." The address-space `set_property name m_axi_hbm` (`:1028`) and per-aximm renaming (`:1031-1032`) become `AddressPlan` entries (S8).

**iodma AXI-MM.** iodma is a single-cell `Region` (`VENDOR_IP`). `direction=="in"` → ports `{m_axi_gmem:OFFCHIP, out0_V:DATA_OUT, s_axi_control:CONFIG}`; `direction=="out"` → `{in0_V:DATA_IN, m_axi_gmem:OFFCHIP, s_axi_control:CONFIG}`. The `direction` bifurcation of `iodma_hls.py:391-399` becomes **which ports the cell publishes**, chosen once at composition. The AXI-MM port has `shape=None`, so it never enters a folded-shape pass. Nothing raises; the resolver treats it as any other OFFCHIP master to be address-assigned.

---

## S7. One recursion: kernel-region = graph-region = container-region

The single most important structural consequence: **the same `Region` type models three things the as-is treats as unrelated code.**

| Level | Region name | Cells | Internal nets (role rule) | Boundary |
|---|---|---|---|---|
| Intra-kernel | `MVAU_0` | compute + streamer | WEIGHT_SOURCE→WEIGHT_SINK | din/dout/clk/rst |
| Container | `finn_loop_0` | body-BD + IODMA + ctrl | INDEX, OFFCHIP fan-out | in0/out0/HBM/done_if |
| Graph (CreateStitchedIP) | `top` | kernel cells | DATA_OUT→DATA_IN | external I/O + HBM |

A `Cell` of `kind=HIER`/`BD_REF` *is* a nested resolved `Region`. `resolve` recurses: a region's cells may themselves be regions, and a boundary `OFFCHIP` port propagates outward level by level (MVAU `axi_mm` → finn_loop `m_axi_hbm` → top `m_axi_hbm`) purely by boundary promotion. The inter-kernel tensor edge (`DATA_OUT→DATA_IN`, the third `STANDARD_BINDINGS` row) is the *same* rule as the intra-kernel weight edge, one level up. This is why the layer does not fragment into "a Stitcher, a Composer, and a CreateStitchedIP" — there is one resolver and one renderer, applied recursively.

---

## S8. Honest residual — the escape hatches

A responsible design names what stays ugly. Four things resist full typing; each is **scoped and declared** so it cannot leak into the generic path:

1. **`PackagingDirective` — IP-packaging surgery.** The aperture removal (`finn_loop.py:1071`), `set_property ipi_drc {ignore_freq_hz true}` (`:1067`), `value_resolve_type user` (`:1078`), and the `mdd-data` copytree (`:1082`) are operations on `component.xml` *after* the netlist exists — they are not nets and cannot be role-matched. Model them as a typed, opaque-bodied value that **travels with the part that needs it**:
   ```python
   @dataclass(frozen=True)
   class PackagingDirective:
       phase: Literal["pre_package", "post_package"]
       scope: str                 # the cell/core it applies to — NOT global
       body:  TclFragment         # still a TCL string — the honest wart
   ```
   The IODMA part emits the aperture directive; the resolver and renderer never read its body. This is strictly better than the god-method (the surgery is *scoped to and owned by* the part, declared in its `Cell`, and invisible to every other op) but it is not *dissolved* — the body is still vendor TCL. I claim this is irreducible: it is Vivado IP-packager behavior with no netlist-level meaning. The win is containment, not elimination.

2. **`AddressPlan` — segment sizing.** `assign_bd_address` with the `TODO calculate and pass in segment size` (`:1126-1127`) and the DDR_LOW/DDR_HIGH aperture interaction are a real unsolved arithmetic problem in the as-is. The `AddressPlan` types the *intent* (which OFFCHIP/CONFIG masters map where) but the segment-size computation remains a declared function that can still be wrong; typing surfaces it as a first-class field instead of a buried TODO, but does not compute it for free.

3. **`SourceManifest` vs `os.listdir` last-match.** The god-method's `for fname in os.listdir(code_gen_dir): if fname.endswith(suffix): strm_tmpl = fname` (no `break`, `:983/1005/1046`, and `finn_loop.py:704`) is replaced by the part *returning* the exact wrapper name it emitted, so the manifest is a typed list, not a filesystem scan. This is fully fixable and fixed — listed here only because it means the `WeightDelivery.emit_cell` must thread the generated wrapper name through as data (a small ceremony the as-is avoided by scanning).

4. **Role-vocabulary governance.** `Role` is a closed enum. A genuinely novel coupling (some future op needs a source→sink pairing not in the seven roles) requires adding a role + a `RoleBinding` row — a governed, reviewed, central edit. This is the *bounded* residual of composition's `negotiate` hole (ranking §5-T2): cross-part knowledge is not zero, but it is a single small table of pairwise role rules, not per-op TCL, and every existing conformance op (MVAU all-modes, thresholding, finn_loop, iodma, static-IP GEMM) is expressible with the seven roles and three binding rows above. The danger to watch — and the one a reviewer must guard — is a role like `MVAU_WEIGHT` creeping in; roles must stay *semantic* (WEIGHT_SOURCE), never *op-named*, or the table becomes the god-switch reincarnated.

---

### Summary of the ruling

- **The weight-streamer → compute net belongs to the `Region`**, derived by an op-agnostic `RoleBinding` from two independently-published ports (`WEIGHT_SOURCE`, `WEIGHT_SINK`). Neither part owns it; neither part names the other.
- **Cross-clock is data:** a `ClockDomain` label on the consumer port + a `ClockPlan` with a `fuse` map. `pumpedMemory`/`clk2x` is a table lookup, contributed independently by `WeightDelivery` (the consumer) and `MemoryStrategy` (the plan). Zero branches in the stitch path.
- **Op-agnosticism** holds because the resolver and renderer read only `Role`, `Protocol`, `CellKind`, `ClockDomain` — never a pin name or op type. Adding external-weights MVAU or the static-IP GEMM edits nothing in the layer.
- **One recursive `Region`** unifies the 236-line MVAU method, `CreateStitchedIP`, and the finn_loop container.
- **Residual, contained not eliminated:** `PackagingDirective` (aperture/IP-packager surgery) stays as scoped opaque TCL owned by the part; `AddressPlan` segment sizing stays a declared-but-fallible computation; the `Role` enum is a governed central vocabulary that must remain semantic, never op-named.


---

# § 4. Conformance Walkthroughs + Migration Path

# Part A — Conformance walkthroughs in the synthesized model

**The synthesized backbone (recap of what every walkthrough assumes).** One `final` `Kernel` = `contract` (per-op `DataContract`) + six Protocol parts (`compute`, `weights`, `memory`, `folding`, `ports`, `reference`; the cppsim/rtlsim `Runner` hangs off `compute`). The substrate is generic drivers — the `Selector`, the `assemble_region`/`resolve`/`TclRenderer` stitch pipeline, and the 8-method delegation on `Kernel` — none of which read an op type or an `isinstance`. Grafted in: capability-trait's **preconditions-as-DATA** solver with `explain()` (ruling **T1 = DATA**); multi-axis's **typed `Port` sum where only `DATA_*`/`WEIGHT_*` ports carry a shape** and **Memory as an independent axis** (ruling **T6 = peer axis + one `Legal` pair for MLO**); ir-lowering's **weight delivery publishes its own role-tagged ports** but stays a composed part, not a graph node (ruling **T3**); ir-lowering's **Region** as the container model (**T4**, per the stitch layer). Cross-component constraints live as **distributed per-realization preconditions + a small pairwise `Legal` relation + a bounded n-ary escape hatch** (ruling **T2**). The stitch layer is the typed `Region`/`Cell`/`Port`/`Role`/`RoleBinding`/`ClockPlan`/`RegionResolver`/`TclRenderer` model given in the brief; each walkthrough drives it, does not redesign it.

Each op is walked across the six axes and the stitch `Region`, and every claim cites the as-is `file:line` it retires.

---

## A1. MVAU — the worst case, all delivery modes

**Compute.** `make_mvau(attrs)` returns one `Kernel` whose `compute ∈ {HlsCompute(mvau.hpp), RtlCompute(mvu_vvu_axi), StaticIpCompute}`. The base-uninstantiable inversion — `code_generation_ipi` living on the agnostic MVAU base yet calling `self.instantiate_ip()` defined only on the leaves (`matrixvectoractivation.py:920` calling `matrixvectoractivation_hls.py:681` / `_rtl.py:163`) — is gone: instantiation is `compute.emit_cell(env)` returning a `Cell`, an ordinary method on the concrete strategy, never reached through a base. `pumpedCompute`, an RTL-only attr the base sniffs by `try/except AttributeError` (`matrixvectoractivation.py:892`, `:939-942`; defined only at `MVAU_rtl:53`), is a plain field on `RtlCompute`; the substrate never reads it. The `$ACCU_WIDTH$`-from-output-dtype bug (`matrixvectoractivation_rtl.py:349`, which silently uses `get_output_datatype().bitwidth()` instead of `accDataType`) is fixed by the typed `RtlModule` param schema grafted from ir-lowering: `ACCU_WIDTH` binds from `accDataType` by declaration, and a mismatch is a bind-time type error, not silent wrongness.

**Weight delivery (axis 2, the criterion-#1 fix).** The `mem_mode × dynamic_input × mlo` matrix that today is emitted from *both* backends and stitched by the 236-line base method becomes: pick one `WeightDelivery` part — `Embedded | DecoupledStream | External | DynamicLoad | LoopFetched`. There is no matrix in code. The cross-backend leak where an HLS MVAU emits finn-rtllib streamer Verilog (`matrixvectoractivation_hls.py:144-155`) is dissolved structurally: `DecoupledStream.emit_cell()` emits the streamer regardless of which `ComputeStrategy` is composed, because compute and delivery are two separate objects. The op-type allowlist that gates the base streamer emit (`hwcustomop.py:310` for memstream, `:358-359` for fetch_weights) has no analogue — there is no base method and no string list. Per ruling **T3**, the delivery part **publishes its own role-tagged ports** (`WEIGHT_SOURCE`, plus `INDEX_SINK`/`OFFCHIP` for MLO, `DATA_IN` for dynamic-load) instead of the compute reaching into its privates; the `mlo_max_iter` flag threaded through ~12 MVAU sites (`hwcustomop.py:100`) is simply the presence of a `LoopFetched` part — invisible when not composed in.

**Memory (axis 3, independent per ruling T6).** `memory ∈ {LutRom, Bram, Uram, OffChipDma}` is chosen *independently* of the delivery motion, resolving multi-axis's and capability-trait's residual "storage rides on delivery" coupling. `calc_wmem`/`ram_style` — leaked onto the substrate via the allowlisted base method — live on the `MemoryStrategy` that owns the geometry. `bram_estimation`/`uram_estimation` (`matrixvectoractivation_hls.py:416-427`) move here too. The one genuinely coupled case, `(LoopFetched, OffChipDma)`, is expressed as a single row in the small `Legal` relation, not a branch. `pumpedMemory`'s hex-word interleave during serialization — which today reaches across axes and `raise`s when `pe==simd==1` (`matrixvectoractivation.py:715`) — is a `LayoutConstraint` the `MemoryStrategy` requests of the one `ParamSerializer`, with the `pe==simd==1` degenerate case surfaced as a declared precondition failure (with an `explain()` string) rather than a mid-serialization exception.

**Ports (axis 5).** `ports = [TensorStreamPort(in0)=DATA_IN, WeightPort(in1)=WEIGHT_SINK, TensorStreamPort(out0)=DATA_OUT, AxiLitePort?=CONFIG, AxiMmPort?(mlo)=OFFCHIP]`. `get_instream_width` branching on `dynamic_input/mem_mode/mlo_max_iter` (`matrixvectoractivation.py:256/268`) collapses: width is a question you ask a `Port`, and the `WeightPort` delegates to `self.weights.is_streamed()`, yielding 0 for embedded with no `try/except`.

**Stitch — the `Region` (T4).** `MVAU_0` is a `Region` with two `Cell`s: the compute cell (`in0_V`:DATA_IN, `out0_V`:DATA_OUT, `in1_V`:WEIGHT_SINK, `ap_clk`:CLOCK/BASE, optional `ap_clk2x`:CLOCK/PUMP) and `MVAU_0_wstrm` (`m_axis_0`:WEIGHT_SOURCE, clocks, `s_axilite`:CONFIG if runtime-writeable). The 236 lines of `code_generation_ipi` (`matrixvectoractivation.py:920`) reproduce as table-driven `resolve` steps with **zero op branches**:
- The `strm/m_axis_0 → node/in1_V` net (`:1085-1089`) — the connection the composition adversary flagged as un-emittable by either fragment alone — is produced by `resolve` step 1 from the single `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS)`. Neither cell names the other.
- The clk2x branch (`:1098-1112`), which routes the memstreamer's `ap_clk2x` to the 2x or the 1x clock on `pumpedMemory`, and the always-present dead 2x pin (`matrixvectoractivation_rtl.py:202-205`), become a `ClockPlan`: the streamer unconditionally publishes a `PUMP` consumer; `MemoryStrategy.pumpedMemory` sets `fuse={PUMP:BASE}` or declares the `PUMP` boundary pin. Two data contributions, no `if` in the stitch path.
- `din`/`dout` externalization (`:1136-1145`) is step 2; `assign_bd_address` for the runtime-writeable axilite with its `TODO calculate segment size` (`:1126`) is step 4's `AddressPlan` (now a first-class field, honestly still a fallible computation per stitch-layer S8).
- `os.listdir(code_gen_dir)` last-match for `*_memstream_wrapper.v`/`_dynamic_load_/_fetch_weights_` (`:983/1005/1046`) is replaced by the `Cell.sources` `SourceManifest` — the delivery part returns the exact wrapper name it emitted (stitch-layer S8 residual #3). `FINN_ROOT` path assembly (`:979-1002`) goes through the injected `RtllibLocator`.

The three delivery-mode `if`s (`mem_mode`/`mlo`/`dyn_input` at `:971/996/1040`) vanish: each variant emits a different `Cell` with different role-tagged ports; the resolver is identical for all three. **MLO** additionally publishes `in_idx0_V`:INDEX_SINK and `axi_mm`:OFFCHIP, promoted to the region boundary and continued up into the `finn_loop` region (A3); the `n_max_layers=64` hard-code (`hwcustomop.py:379`) becomes a declared `LoopFetched` parameter.

---

## A2. Thresholding — memory as orthogonal axis, unified serialization

**Compute / contract.** `contract = ThresholdingContract` (2-input; `ind=1` is the threshold tensor), owning the golden multithreshold `ReferenceModel` reused for cppsim by *all* compute kinds — killing the RTL/HLS rtlsim copy-paste (`thresholding_rtl.py:347` re-implementing `Thresholding_hls.execute_node`) via the one shared `CppsimRunner`/`RtlsimRunner`.

**Memory is the orthogonal axis (criterion #6, ruling T6).** Embedded-vs-decoupled is `memory ∈ {LutRom, Bram}` composed with `weights ∈ {Embedded, DecoupledStream}` — the `ThresholdingContract` has *zero* reference to either. This deletes: the `get_instream_width` `try/except mem_mode` sniff in the agnostic base (`thresholding.py:179`); the `get_verilog_top_module_intf_names` `mem_mode`-plus-`mlo_max_iter` topology branch (`thresholding.py:310`); and the agnostic threshold class's entanglement with the MLO flag it does not implement (`thresholding.py:133`). The HLS-emits-an-RTL-streamer leak via the base allowlist (`thresholding_hls.py:181`, riding `hwcustomop.py:310`) is gone because `DecoupledStream.emit_cell` owns the streamer irrespective of compute kind.

**Unified serialization (criterion #6, the triplication fix).** The three divergent `make_weight_file` copies — base `thresholding.py:617`, HLS `thresholding_hls.py:299`, RTL `thresholding_rtl.py:452` (plus the three divergent `minimize_weight_bit_width` at `thresholding.py:151`/`_hls.py:781`/`_rtl.py:566`) — collapse to **one** `ParamSerializer.layout(thresholds, constraint)`. The RTL binary-search requirement that thresholds be pre-sorted ascending — today a *silent correctness dependency* (`thresholding_rtl.py:457-462`) that the HLS variant does not share — becomes `RtlCompute.layout_constraints()` returning `SortedForBinarySearch`, a **declared** `LayoutConstraint` honored by the single serializer. This is the honest reading of the capability-trait/multi-axis "layout is dictated by the compute core" strain: the coupling is real, so it is made a *declared constraint the compute publishes*, not a hidden fourth copy.

**Stitch.** `Region` identical in shape to MVAU's: compute cell + optional `Thresholding_0_wstrm`, one `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS)` net. The `os.listdir` memstream-wrapper discovery that can leave `strm_tmpl` undefined and `NameError` (`thresholding_hls.py:675`) is replaced by the `SourceManifest`. The "2x clock not used, wire to 1x for now" placeholder (`thresholding_hls.py:704`) is the `ClockPlan` `fuse={PUMP:BASE}` default — the same mechanism as MVAU, now uniform.

---

## A3. finn_loop — the Region container model (grafted from ir-lowering)

**The container is a `Region`, recursively (stitch-layer S7).** `finn_loop` is a `Kernel` with `compute = SubgraphCompute` holding a child `KernelGraph`; its stitch obligation is to emit a `Region` whose `Cell`s are the packaged loop-body BD (`kind=BD_REF`), the fetch/store control cells, and the IODMA. The `get_rtl_file_list → None` stub that only nominally satisfies a required `@abstractmethod` (`finn_loop.py:1175`, dodged by overriding `prepare_rtlsim` at `:277`) does not exist — there is no abstract to stub; a container legitimately emits a `Region`, not a leaf RTL file list.

**Sibling-op coupling dissolved.** The three worst `op_type`-string dispatches are replaced by reading each child region's published role-tagged boundary ports:
- `generate_params` dispatching on `'MVAU'/'Elementwise'/'Thresholding'` prefixes with `else: raise` (`finn_loop.py:421/450`) → the container reads each child's `WEIGHT_SOURCE`/`OFFCHIP` boundary ports; it asks "give me your role-tagged ports," never "are you an MVAU."
- `adjacency_list` hard-coding the same three families gated on `mlo_max_iter>0` (`finn_loop.py:712`) → wiring is the `RoleBinding(INDEX_SOURCE, INDEX_SINK, AXIS)` and `DATA_OUT→DATA_IN` rules applied at container level (the *same* rules as intra-kernel, one level up — S7).
- `generate_hdl_stream_tap` special-casing `Thresholding_rtl` for `TAP_REP` (`finn_loop.py:558`) → `TAP_REP = prod(child.folding.folded_shape[:-1])`, an interface call, not an op branch.
- The two `replace_verilog_relpaths` copies walking child ipgen dirs and string-replacing `memblock.dat`/`./<node.name>` (`finn_loop.py:468`, `:521`) → the child `Cell.sources` manifest carries the concatenated path as data.

**Boundary promotion for infra ports.** The HBM/aximm interface — hard-coded `ext_intf_signals = ['in0_V','out0_V','m_axi_hbm']` (`finn_loop.py:651`), `addr_bits=64` (`:1121`), and the `set_property name` re-rename of Vivado's `_0`-suffixed ports (`:1024-1034`) — is the loop body's `m_axi_hbm`:OFFCHIP(boundary) port promoted outward, level by level (MVAU `axi_mm` → finn_loop `m_axi_hbm` → top), by `resolve` step 2. This directly answers the ir-lowering strain (a deep interior AXI-MM port is "neither a first-node input nor a last-node output"): promotion is by **role tag**, not by boundary-position projection, so interior infra ports surface correctly. The `LAYER_OFFS_INT = 2**ceil(log2(...))` admitted placeholder (`finn_loop.py:385-388`) lands in the `AddressPlan` (stitch-layer S8 residual #2: typed as intent, still fallible).

**Honest residuals.** The `ipx::remove_segment m_axi_gmem0:APERTURE_0` aperture hotfix (`finn_loop.py:1071`), the `ipi_drc {ignore_freq_hz true}` / `value_resolve_type user`, and the `mdd-data` copytree (`:1082`) are `PackagingDirective`s **scoped to and owned by** the IODMA/body cell (stitch-layer S8 residual #1) — strictly better than the god-method (invisible to every other op), but still opaque vendor TCL, not dissolved. The `dtype=='g'` nodeattr smuggling of a whole `ModelWrapper` (`finn_loop.py:99`) becomes the typed `SubgraphContract` holding the child `KernelGraph`; the "second input is always the param" assumption (`finn_loop.py:161`) is replaced by the WEIGHT_SINK role tag on whichever port carries coefficients.

---

## A4. iodma — typed AxiMM port that does not raise

**The category error is forbidden by type (criterion #7).** In the stitch model, `shape` is present on a `Port` **by role, not by op**: only `DATA_*`/`WEIGHT_*` ports carry a `FoldedShape`; an `OFFCHIP` port carries a `width` and `shape=None`. The generic folded-shape pass iterates `[p for p in cell.ports if p.shape is not None]` and **structurally cannot** touch the AXI-MM side. The two raising getters — `get_folded_input_shape` raising when `direction=='in'` and `get_folded_output_shape` raising when `direction=='out'` (`iodma_hls.py:116/131`), which crash any generic node pass — simply have nothing to raise from: you cannot ask a shape-less port for a shape. This retires the copy-paste bug too (the OUTPUT method carrying the INPUT method's wrong assertion message, `iodma_hls.py:137`).

**Direction is which ports the cell publishes, not a bifurcation.** iodma is a single-cell `Region` (`kind=VENDOR_IP`). `direction=="in"` → `{m_axi_gmem:OFFCHIP, out0_V:DATA_OUT, s_axi_control:CONFIG}`; `direction=="out"` → `{in0_V:DATA_IN, m_axi_gmem:OFFCHIP, s_axi_control:CONFIG}`. The `intf_names` surgery that calls `super()` then *overwrites* `intf_names['m_axis']=[]` to un-do the streaming port the base auto-populated (`iodma_hls.py:394`) is gone — the cell publishes exactly its real ports, chosen once at composition. The hard-coded `'m_axi_gmem'`/`'s_axi_control'` string contract (`iodma_hls.py:397-398`) is now the `name` field on role-tagged `Port`s, and **no matcher ever reads the name** — only the `Role` — so a rename cannot break stitching.

**Honest reference (criterion #4/#7).** `execute_node = pass` (`iodma_hls.py:388`), which makes any cppsim/rtlsim data-verification flow through iodma a silent no-op, becomes `reference = IdentityMover` so data actually flows and correctness is observable. The 100-line `direction × width-divisibility` template branch tree (`iodma_hls.py:239-318`) is contained in `StaticIpCompute`-style emission of one cell; the DWC-chain arithmetic remains a per-op detail on the compute part, not smeared into the substrate.

---

## A5. Static-IP INT8 GEMM — the peer 3rd kind, auto-selected, zero substrate edits

**The forcing function passes by construction.** `StaticIpCompute` implements the `ComputeStrategy` Protocol identically to HLS/RTL and is registered as a peer candidate for the MVAU contract kind. Adding it is **one new file in `compute/` plus a registry entry** — zero edits to `kernel.py`, the `Selector`, `assemble_region`, `resolve`, `TclRenderer`, or `STANDARD_BINDINGS`, because every substrate driver reads only `Role`/`Protocol`/`CellKind`/`ClockDomain`, never a pin name or op type. This is precisely the seam where ir-lowering's `RenderTcl` collapsed (its adversary returned `collapses=true` at this exact test); the composition backbone + role-typed stitch holds.

**Selection is contract-driven DATA (criterion #3, ruling T1).** `StaticIpCompute`'s preconditions are inspectable constraint data — `AllOf(DtypeBits(in)==8, DtypeBits(out)==8, MW∈range, MH∈range, PE/SIMD matches the hard-IP config)` — resolved by the one op-agnostic solver with `explain()`, at `priority=30` (beating `RtlCompute=20`) with transparent fallback when a precondition fails. This deletes the entire `specialize_layers.py:40-211` per-op RTL-feasibility ladder, the `optype+"_"+impl_style` string-concat resolution (`specialize_layers.py:399`), and — because there is now one source of truth per strategy — makes the `:60/:69`-pre-gate-vs-`:275`-predicate drift **structurally impossible**. The hidden runtime "vivado" third backend inside `streamingfifo_rtl.py:141` (criterion #9), selected by a runtime attr with base `try/except` reach-through, is the same fix: it becomes an explicit `VivadoInfraCompute` peer chosen by the identical solver.

**Stitch by construction (ruling T2/T3).** `StaticIpCompute.emit_cell()` returns a `Cell(kind=VENDOR_IP, ref=<vlnv>)` whose ports are role-tagged `DATA_IN`/`DATA_OUT`/`CLOCK(BASE)` exactly like the RTL MVAU's. The weight-ownership constraint that broke ir-lowering — the hard IP owns its weight path — is expressed two honest ways: if weights are embedded, the cell publishes **no** `WEIGHT_SINK`, so `resolve` finds zero source/sink pairs and emits no weight net (one cell, no internal nets); the pairwise `Legal` rule `StaticIP ⇒ WeightDelivery ∈ {Embedded, External}` (ruling T2's small declared relation) forbids an illegal upstream streamer at selection time — the constraint ir-lowering had no home for. If instead the IP wants a streamer, it publishes a `WEIGHT_SINK` and composes with `DecoupledStream` through the *same* binding row. Either way the stitch layer edits nothing.

**Conformance matrix (as-is site each op retires):**

| Op | Primary as-is defect retired | Axis / mechanism |
|---|---|---|
| MVAU | `matrixvectoractivation.py:920` (236-line IPI); `_hls.py:144` cross-backend; `hwcustomop.py:310` allowlist | weight-delivery part + `Region`/`RoleBinding` |
| Thresholding | `thresholding.py:179/310` mem_mode sniff; three `make_weight_file` (`:617`/`_hls:299`/`_rtl:452`) | Memory peer axis + one `ParamSerializer` |
| finn_loop | `finn_loop.py:1175` stub; `:421/712/558` op_type dispatch | recursive `Region` + role-tagged boundary promotion |
| iodma | `iodma_hls.py:116/131` raising getters; `:394` intf un-do; `:388` `pass` | shape-by-role `Port` + `IdentityMover` reference |
| Static-IP GEMM | `specialize_layers.py:40-211` ladder; would-be `hwcustomop.py:310` edit | DATA-precondition solver + zero substrate edits |

---

# Part B — Migration path: clean-slate target, incremental delivery

**Ruling T5 in force:** adopt the composition core (clean-slate ceiling, zero-substrate-edit purity) but execute via evolutionary's 8-PR discipline — each PR independently landable and **green against the 27-family regression at every step**. The mechanism that makes a clean-slate *target* reachable incrementally is a **strangler-fig with a single adapter seam**: a `KernelAdapter(HWCustomOp)` wraps a new `Kernel` and re-presents the legacy `HWCustomOp` surface (the 8 getters, `execute_node`, `code_generation_ipi`, `get_rtl_file_list`) so that FINN's transforms and builder never change during migration. Families cut over from legacy classes to `make_<op>(attrs)` factories one group at a time, behind that one seam. This maps evolutionary's R1–R8 onto the composition target: R-order is preserved where the fix is identical, but each PR now *lands a slice of the final composed model* rather than patching the mixin in place.

| PR | Delivers (composition target) | Mirrors evo | Criteria | As-is retired | Blast radius / green-keeping |
|----|-------------------------------|-------------|----------|---------------|------------------------------|
| **P1** | New `kernel/` package: `Port`/`Role`/`Protocol`/`Cell`/`Net`/`Region` types, `RegionResolver`, `TclRenderer`, `ReferenceModel` Protocol, and the `KernelAdapter` seam. **Dormant** — no live op uses it yet; ships with unit tests on the typed model. | (new spine) | scaffolding | — | Nothing live changes → 27-family green by construction. |
| **P2** | **Execution axis first** (highest value, lowest risk): route the substrate `execute_node` dispatcher through `reference.evaluate` + one shared `RtlsimRunner`/`CppsimRunner`; delete the 8 MRO shims. | R1 | 4 | diamond; VVAU indent bug `vectorvectoractivation_rtl.py:89`; MVAU rtlsim copy-paste `matrixvectoractivation_rtl.py:100`≈`_hls.py:569` | ~11 files mechanical; adapter delegates to shared runner so legacy + composed ops share one harness. |
| **P3** | **Selection axis:** `Selector` + preconditions-as-DATA solver with `explain()`; per-op `_*_rtl_possible` predicates move to co-located constraint data; `preference_order`/`priority` table. | R5 | 3, 9 | `specialize_layers.py:40-211` ladder; `:60/:69`-vs-`:275` drift; `:399` string-concat; `streamingfifo_rtl.py:141` hidden backend | `specialize_layers` becomes a ~15-line generic pass; both legacy classes and `KernelAdapter`s expose `feasible`/precondition data. |
| **P4** | **Ports axis:** typed `PortSet` with shape-by-role; `KernelAdapter` implements the 8 legacy getters from the `PortSet`. | R6 (infra half) | 7 | `iodma_hls.py:116/131` raises; `:394` intf un-do; `tlastmarker` raising getters | iodma/tlastmarker/checksum cut first (smallest, they *only* have infra ports) — proves the shape-by-role fix end-to-end. |
| **P5** | **Weight-delivery + Memory axes:** `WeightDelivery` parts (`Embedded`/`DecoupledStream`/`External`/`DynamicLoad`/`LoopFetched`) publishing role-tagged ports; independent `MemoryStrategy` (T6) + one `ParamSerializer`. Delete the three `generate_hdl_*` base methods and the op-type allowlist. | R2 | 1, 2, 6 | `hwcustomop.py:307/310/355/358/407`; three `make_weight_file`; `matrixvectoractivation_hls.py:144` | 6 weight-bearing ops + substrate; migrate MVAU/Thresholding/VVAU factories together so delivery is exercised across compute kinds. |
| **P6** | **Stitch layer:** `assemble_region`/`resolve`/`TclRenderer`/`ClockPlan`/`AddressPlan`; the adapter's `code_generation_ipi` now calls `stitch(kernel, env)`. | (T4 — fresh) | 1, 5 | `matrixvectoractivation.py:920` full IPI incl. clk2x `:1098-1112`, weight net `:1085-1089`, `os.listdir` `:983`, `assign_bd_address` `:1126` | The 236-line method deletes for migrated ops; legacy ops keep their own IPI until their factory lands. Ratchet (below) enforces monotone shrink. |
| **P7** | **Containers + remaining families:** `SubgraphCompute` + recursive `Region` for finn_loop/SDP; migrate the tail of the 27 families to factories; typed `Parameterization`/`ArtifactManifest` for the rtllib adapter; injected `BuildContext`/`RtllibLocator`/`Simulator`. | R4+R6+R7 | 5, 8, 10 | `finn_loop.py:1175/421/712/558/1071`; 26 `$KEY$` replaces; `finnxsi` singleton `hwcustomop.py:39` | Containers last (they depend on child regions from P4–P6). |
| **P8** | **Static-IP GEMM proof + demolition:** add `StaticIpCompute` (one file, zero substrate edits); then **delete the legacy `HWCustomOp`/`HLSBackend`/`RTLBackend` path and the `KernelAdapter`**. Interregnum ends. | R8 | forcing fn, 2 | mixin diamond itself; the adapter seam | +1 file for the proof; the deletion is large but pure removal — 27-family green means the composed path already covers everything. |

**Why each PR is independently landable.** The `KernelAdapter` presents the exact legacy `HWCustomOp` surface, so at every PR boundary the mix of (legacy classes) + (composed `Kernel`s behind the adapter) satisfies the same external contract the FINN transforms and builder consume. A PR lands, runs the full 27-family op test suite, and merges before the next starts — the defining property of an evolutionary path, now aimed at a clean-slate target. P2–P4 touch the substrate seam but leave *op behavior* identical; P5–P7 cut families over in groups chosen by dependency order (infra → weight-bearing → containers); P8 is the only PR that removes the old world, and it does so only after the composed path demonstrably covers every family.

**The interregnum risk — "two ways to do everything" — and how it is bounded.** During P2–P7 the codebase genuinely has both a legacy path and a composed path. Evolutionary flagged this as the price of safe landability; four disciplines bound it:

1. **Single seam.** `KernelAdapter` is the *only* bridge between the two worlds. No op ever sees both models; there is exactly one file where legacy and composed meet, so the interregnum surface is one class, not N ops. (This is the direct fix for evolutionary's "back-compat shims scattered everywhere" weakness — one shim, not a shim per axis.)
2. **Per-family cutover registry, not per-call flags.** A single `COMPOSED_FAMILIES` set records which of the 27 families have migrated. There is no runtime `if new_path else old_path` inside op logic — a family is wholly legacy or wholly composed. This prevents the half-migrated per-method state (the `code_gen_dict`-fallback / `finnxsi`-default duplication) that evolutionary admits as its long tail.
3. **A monotone ratchet in CI.** A test asserts the legacy-path line count (and the count of ops *not* in `COMPOSED_FAMILIES`) only ever decreases across PRs. This makes the interregnum provably converge — you cannot land a PR that grows the old world, so "two ways" is strictly shrinking, not drifting.
4. **Time-box by construction.** P8 *deletes* the legacy path and the adapter. The interregnum is bounded to the P2→P8 window; the design does not ship with a permanent dual path. The clean-slate target's ceiling (zero-substrate-edit purity, no mixin diamond) is only actually realized at P8 — which is the whole point of T5: the incremental path is delivery, not the destination.

**Rollback posture.** Because every intermediate PR keeps both paths green behind the adapter, any PR is independently revertible without touching the others — the safe-rollback property evolutionary has and a big-bang rewrite lacks. The only irreversible step is P8's demolition, gated on the ratchet reaching zero legacy families; by then the composed path has been the sole implementation for every family through a full regression cycle.

---

**Key file:line anchors used:** MVAU `matrixvectoractivation.py:920` / `:892` / `:715` / `:983` / `:1085-1112` / `:1126`, `_hls.py:144/681`, `_rtl.py:100/163/202/349`; `hwcustomop.py:39/100/310/358/379/407`; Thresholding `thresholding.py:133/179/310/617`, `_hls.py:181/299/675/704`, `_rtl.py:452/457/566`; iodma `iodma_hls.py:116/131/137/388/394/397`; finn_loop `finn_loop.py:99/161/385/421/468/521/558/651/712/1024/1071/1121/1175`; selection `specialize_layers.py:40-211/60/69/275/399`; `streamingfifo_rtl.py:141`; `vectorvectoractivation_rtl.py:89`.


---

# § 5. Completeness Critique

*Independent critic pass verifying the synthesis is whole and internally consistent — scrutinizing hardest whether the stitch layer dissolves the god-method or merely relocates it.*

# 1. Consistency / gap checklist

| Item | Status | Evidence |
|---|---|---|
| **T1** feasibility as DATA | **PASS** | Selection §"Ruling T1" adopts precondition-AST; reflected in `Precondition` primitives (Sel1), `explain()`, and the two escape-hatch predicates promoted to first-class primitives (`ParamIsNarrow`, `BroadcastCompatible`). Not just asserted — the six `_*_rtl_possible` ladders are re-expressed as data at `specialize_layers.py:235-374` (verified those lines exist). |
| **T2** where cross-axis constraints live | **PASS** | Three-tier structure (distributed preconditions / pairwise `LEGAL` / bounded `KNOWN_COUPLINGS`) with explicit growth bounds. Directly replaces composition's `negotiate` (`01-composition.md:137,182` — verified the hole exists). |
| **T3** delivery composed-part vs graph-node | **PASS** | Core §4/§10: composed part + peer role-tagged ports; compute never touches delivery privates; fold consistency via shared `kernel.folding` reference, not a propagation pass. Consistent across core/selection/stitch. |
| **T4** stitch layer | **PASS (with contained residual)** | See §3 below — scrutinized hardest. |
| **T5** migration | **PASS** | 8-PR strangler-fig, `KernelAdapter` seam, monotone CI ratchet, time-boxed to P8. |
| **T6** memory peer axis | **PASS** | Core §6/§10: independent `MemoryStrategy`, owns `pumpedMemory`/`calc_wmem`/estimations. But tier-assignment of the MLO pairing conflicts across sections — see §2. |
| 5 conformance ops each fix as-is line | **PASS** | MVAU (`matrixvectoractivation.py:920` — verified 236-line method `def code_generation_ipi` at 920, `get_verilog_top_module_intf_names` at 879), thresholding, finn_loop (`:1175` stub verified, `:421/558/712` op_type dispatch verified), iodma (`iodma_hls.py:116/131` raises verified, `:388` `execute_node`, `:391-399` intf), static-IP. Matrix at end of Part A. |
| Crit #1 weight-delivery axis | **PASS** | Core §4; MVAU matrix → pick-one-part. |
| Crit #2 no substrate branch on op | **PASS at destination** | `@final` Kernel, no subclass, substrate reads only Protocols/`Role`/`CellKind`. Caveat: `KernelAdapter(HWCustomOp)` + `COMPOSED_FAMILIES` set exist during P2–P7 interregnum — purity is only literally true post-P8. T5 accepts this explicitly. |
| Crit #3 contract-driven selection | **PASS** | Solver over declared data; `specialize_layers.py:40-211` deleted. |
| Crit #4 no execute_node diamond | **PASS** | `reference.evaluate` vs `compute.runner` — two names, no MRO. Adapter delegates, does not re-diamond. |
| Crit #5 pure emit, no `code_gen_dict` | **PASS** | `ComputeStrategy.emit: EmitInputs→Artifacts`, pure. |
| Crit #6 memory-strategy modeled | **PASS** | T6; one `ParamSerializer` collapses the 3 `make_weight_file` copies. |
| Crit #7 infra/container ports honest | **PASS** | shape-by-role: `shape=None` on `OFFCHIP` structurally forbids iodma's raise (`:116/131`). |
| Crit #8 typed adapter not `str.replace` | **PASS (light)** | Addressed via typed `RtlModule` param schema (ACCU_WIDTH-from-`accDataType` fix) + `Parameterization`/`ArtifactManifest` in P7. Weakest-covered criterion — no dedicated axis treatment in the core model; folded into compute artifacts. Sufficient but thin. |
| Crit #9 no hidden third backend | **PASS** | `streamingfifo_rtl.py:141` "vivado" → explicit `VivadoInfraCompute` peer via solver. |
| Crit #10 no singletons, injected deps | **PASS** | `BuildContext`/`SelectionContext` injected; `FakeSimulator`; `finnxsi` singleton → constructor arg. |

# 2. Contradictions to resolve

**C1 (real, minor) — MLO constraint tier is inconsistent across sections.** Core §6/§10 states `LoopFetched ⇒ OffChipDma` is "**one row** in the selection layer's bounded **pairwise** relation (T2's answer)" / "one declared `Legal` pairing." But the Selection section (Sel5/Sel6) puts MLO in **tier-3** `KNOWN_COUPLINGS` as the n-ary `MloCoupling` (forcing OffChipDma + AxiMM + INDEX ports + enclosing container), and explicitly says "pairwise `Legal` cannot express them." Both cannot be the authoritative classification.
*Resolution:* split the fact — the memory-forcing sub-constraint (`LoopFetched⇒OffChipDma`) is genuinely pairwise (tier 2); the full port/container coupling is n-ary (tier 3 `MloCoupling`). Core §6/§10 should say "tier-3 `MloCoupling` (which subsumes the OffChipDma forcing)," not "one `Legal` row." Purely a spec-wording fix; no structural rework.

**C2 (gap, minor) — port provenance underspecified.** `ports_for = compute.ports() + weights.ports() + memory.ports()` (core §5), but `PortSet.boundary_ports()` returns `din/dout/clk/rst`. Which part emits the CLOCK/RESET boundary ports and the kernel-level DATA boundary pins is not assigned to any of the three emitters. `assemble_region` also reads `kernel.ports.boundary_ports()` as if pre-existing. Needs one sentence: boundary/clock/reset ports are contributed by the `contract`/`PortSet` construction, not by a compute/weights/memory part. Does not threaten #2/#4.

No contradiction found between selection and core on T3 (both treat delivery as composed part with peer ports — selection's `Realization.part` is a `WeightDelivery`, never a graph node). The `negotiate` handoff is cleanly overridden, not left dangling.

# 3. Does the stitch layer dissolve the god-method, or relocate it? (highest-risk item)

**Verdict: the coordination god-method genuinely dissolves; the IP-packager surgery is contained, not dissolved — and the design says so honestly.**

The load-bearing test is whether `resolve`/`TclRenderer` can reproduce all 236 lines with **zero op/pin branches**. Checked against the mapped line ranges:
- Weight net `:1085-1089` → derived by `RoleBinding(WEIGHT_SOURCE, WEIGHT_SINK, AXIS)`, step 1. Neither cell names the other. **Dissolved.**
- clk2x branch `:1098-1112` (verified: `if pumpedMemory` routing memstreamer 2x pin) → `ClockPlan.fuse` data + a `PUMP` consumer published unconditionally. **Dissolved into two independent data contributions** (memory owns plan, delivery owns consumer). This is the sharpest branch and it does become a table lookup.
- din/dout externalization `:1136-1145` → boundary promotion, step 2. **Dissolved.**
- `assign_bd_address` `:1126` → `AddressPlan`, step 4. **Dissolved as coordination; segment-size arithmetic honestly flagged as still-fallible (S8 residual #2).**
- `os.listdir` last-match `:983` → `SourceManifest`. **Dissolved.**
- Three delivery-mode `if`s (`mem_mode`/`mlo`/`dyn_input`) → different `Cell`s, identical resolver. **Dissolved** — this is the crux and it holds because the resolver keys on `Role`, never op.

The one thing that does **not** dissolve is IP-packager surgery (`finn_loop.py:1071` aperture removal — verified, `ipi_drc`, `mdd-data` copytree). The design does not pretend otherwise: `PackagingDirective` keeps a `body: TclFragment` of opaque vendor TCL, **scoped to and owned by the emitting cell**, invisible to `resolve`/`render`. This is the correct call — that surgery was never coordination, it does not grow with op count, and it cannot be role-matched. Ranking §5-T4 warned "the god-method simply relocates"; the discriminating property that prevents relocation is that **no part names another part and the resolver reads no pin name** — which is structurally enforced (a part emits one `Cell` of its own ports; the net's *existence* is entailed by a shared `RoleBinding`, not written by either endpoint). This is a genuine dissolution of the coordination logic, not `ir-lowering`'s `RenderTcl` (which centralized emission and re-grew branches). The residual is bounded and named, satisfying the T4 mandate.

The governance risk the design itself flags (S8.4 / Sel12): a role like `MVAU_WEIGHT` or a primitive like `MvuWeightsOk` creeping in would re-grow the god-switch as a data table. This is correctly identified as the standing reviewer duty, not a latent defect.

# 4. Shippable-as-a-design verdict

**Shippable.** All six tensions are ruled with the ruling reflected in mechanism (not merely asserted); all five conformance ops fix concrete verified as-is lines; criteria #2 and #4 hold by construction in the assembled model; and the highest-risk item (T4) genuinely dissolves the coordination god-method rather than relocating it, with an honestly-bounded opaque residual.

Two spec-cleanup items before implementation, neither structural: **(C1)** reconcile the MLO constraint's tier (core calls it pairwise tier-2; selection makes it n-ary tier-3 — pick tier-3-subsumes-tier-2 wording); **(C2)** assign provenance of clock/reset/DATA boundary ports in `PortSet` assembly. Criterion #8 is the thinnest-covered criterion (typed op↔rtllib adapter is folded into compute artifacts + P7 rather than given first-class axis treatment) — adequate but the place a reviewer should push for more detail. Criterion #2 "by construction" is literally true only at P8; the interregnum's `KernelAdapter`+`COMPOSED_FAMILIES` is a deliberate, ratchet-bounded, time-boxed exception per T5, not a leak.

No blocking gaps. The synthesis is whole and internally consistent modulo the two wording fixes.