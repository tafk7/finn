# KernelOp over TENSOR/BLOCK/STREAM — the folding model, op/impl split, and the derived-tiling primitive

*Design record. Establishes the KernelOp as the WHAT-owning op node whose folding is
declared as per-interface TENSOR→BLOCK→STREAM tiling (Brainsmith's dataflow model),
with the analytic (cost/shape/dtype) surface DERIVED from that declaration and a
per-implementation override seam for microarchitecture-specific precision. Grounded in a
three-op stress test — MVU, LayerNorm, Elementwise — read from both FINN
(`src/finn/custom_op/fpgadataflow/`) and Brainsmith (`brainsmith/kernels/`). Companion to
[design-space-model.md](design-space-model.md) (the engine), [port-taxonomy.md](port-taxonomy.md)
(roles/stitch), [param-delivery-design-space.md](param-delivery-design-space.md) (composition),
and [../finn-hw-backend-analysis/consumer-surface-model.md](../finn-hw-backend-analysis/consumer-surface-model.md)
(the FINN-facing contract KernelOp must satisfy).*

---

## 0. Thesis

A KernelOp is the FINN-facing op node (the HWCustomOp equivalent) that owns **WHAT** is
computed; enrolled Implementations own **HOW** it is realized. The `specialize_layers`
seam and the `HLSBackend`/`RTLBackend` split are erased: backend selection is the
`implementation` axis resolved as design-space data, and codegen is each Implementation's
hermetic `emit`. This document fixes what "WHAT" concretely contains.

The central move: **folding is not a set of op-specific axes (PE/SIMD/TH) — it is a
general law (TENSOR→BLOCK→STREAM) that each op instantiates by declaring, per interface, a
`block_tiling` and a `stream_tiling`.** PE/SIMD/TH become *derived labels* on stream dials,
not first-class op axes. From that one declaration the engine derives the entire Tier-0..3
consumer surface (the 8 port-indexed getters, stream widths, divisibility predicates, and
a rough throughput cost). Each Implementation may override the rough analytic model with a
microarchitecture-accurate function.

We adopt TENSOR/BLOCK/STREAM in full (per Brainsmith `docs/developer-guide/dataflow-modeling.md`).

---

## 1. The general law (why PE/SIMD are not op-owned axes)

Brainsmith's data hierarchy:

```
tensor_blocks = ceil(tensor_dim / block_dim)   # spatial decomposition (TENSOR -> BLOCK)
stream_cycles = ceil(block_dim  / stream_dim)  # temporal execution   (BLOCK  -> STREAM)
total_cycles  = prod(tensor_blocks) x prod(stream_cycles)
```

- **TENSOR** — complete ONNX dims (from the graph).
- **BLOCK** — the kernel's atomic computation unit; for a reduction op, "one quantum of
  the calculation state" (a dot-product's input vector). STREAM cannot exceed BLOCK.
- **STREAM** — elements per clock; the parallelism dial resolved by DSE (SIMD/PE).

Every FINN op today hand-codes this same reshape in `get_folded_input_shape`
(`normal[:-1] + [fold, simd]`) and re-derives cost in `get_exp_cycles`. Brainsmith's
`KernelSchema`/`InputSchema`/`OutputSchema` lift that per-op reshape into a *declaration*.
PE/SIMD/TH are MVU-specific names for slots in this law; they are not properties of a
realization, so they are op-level — but as **derived stream dials on interfaces**, not as
flat op axes.

---

## 2. The three-op stress test (ground truth)

Read from FINN abstract op classes + Brainsmith kernels.

| Op | inputs | block_tiling | stream_tiling | Stress it applies |
|---|---|---|---|---|
| **MVU** | 2 (act + weights) + opt thresh | `[FULL_DIM, MW]` act; `[FULL_DIM, MH]` out | act `[SIMD]`, out `[PE]`, weight `[PE·SIMD/TH]` | two dials on **different** interfaces; a reduction the block must hold; a temporal param TH; a derived cross-interface stream width |
| **LayerNorm** | 1 | `[FULL_DIM]` (=`(1,1,channels)`) | `["SIMD"]` last dim only | reduction domain = block; spatial dims stream whole, cannot fold; pipeline-fill latency is impl-specific |
| **Elementwise** | **1 or 2** | `[FULL_DIM]` both | lhs `["PE"]`; out `derive_dim("lhs", STREAM, -1)` | arbitrary input ports; broadcast (size-1) axis that must NOT fold; per-port static/dynamic style; polymorphic output dtype |

### 2.1 What passes cleanly

- **act interface**: `stream_tiling=[SIMD]` on reduction dim MW; trip count `sf=MW/SIMD`.
- **out interface**: `stream_tiling=[PE]` on output dim MH; trip count `nf=MH/PE`.

**Key finding:** PE and SIMD are *not* two dials on one interface — they are **one dial
each on two different interfaces** (SIMD on activation-in, PE on output). The interface-list
model separates them correctly; FINN carries both as flat node attrs. This is cleaner than
FINN.

### 2.2 Where the naive last-axis model breaks — MVU weight port

FINN MVU `get_folded_input_shape(ind=1)` (decoupled) = `(n_vecs, sf·nf, (SIMD·PE)//TH)`.
The last term `WSIMD = PE·SIMD/TH` is exactly `mvu_tiled/mvu_tiled_axi_wrapper.v:20`. It is
**not** a fold of any weight *tensor* axis — it is a function of the OTHER interfaces'
stream dials and a temporal param. Three consequences:

1. **Derived stream_tiling is mandatory, not optional.** The weight port =
   `derive_dim(act, STREAM) · derive_dim(out, STREAM) / param(TH)`. A pure-literal
   `["SIMD"]` model cannot express MVU. (Brainsmith already uses `derive_dim(...)` for
   elementwise output — the mechanism exists; MVU makes it load-bearing.)
2. **TH is a temporal param, not a tiling entry.** It reshapes streams (width `PE·SIMD/TH`,
   cycles `×TH`) without folding a tensor axis. It belongs in `kernel_params`, read by the
   weight interface's derived tiling and by the cost model. So folding ≠ stream_tiling
   alone; there is a third kind: a temporal param that reshapes streams.
3. **Reduction structure is invisible to tiling shapes.** act (`sf` trips) and out (`nf`
   trips) stream independently, but the computation couples them (accumulate all `sf`
   reduction steps before emitting each of `nf` outputs). That coupling lives in the
   cost/block model ("BLOCK = one quantum of calculation state"), not in stream_tiling.

### 2.3 Cost-model check (rough op-level formula reproduces FINN)

`prod(tensor_blocks) · prod(stream_cycles) · temporal_params`:
- **MVU**: `nf · sf · n_vecs · TH` = FINN `(MH/PE)(MW/SIMD)·prod(numInputVectors)·TH`. **Exact.**
- **LayerNorm**: leading term `prod(idim)//SIMD`. **Matches** (the `+val_queue_len` fill
  terms are impl-specific → override).
- **Elementwise**: `prod(get_folded_output_shape()[:-1])` — FINN already writes it in the
  general form. **Exact.**

The rough op-level cost is `prod(all_stream_trips) · prod(tensor_blocks) · temporal_params`,
monotone in the stream dims — which is exactly what `consumer-surface-model.md` requires of
`get_exp_cycles` for the `SetFolding` search. Rough-first is the correct floor, not a
compromise.

### 2.4 Elementwise: the two additional breaks

- **Arbitrary input ports.** FINN indexes everything by `ind` and returns per-port lists
  (`get_input_datatype(ind) -> [lhs,rhs][ind]`, `inFIFODepths=[2,2]`). The **schema** must be
  a *list* of `InterfaceSchema`, not the MVU-biased "one activation + one weight" triple in
  today's `ops/mvau/names.py`.
- **Broadcast folding asymmetry.** Elementwise `get_folded_input_shape` branches: a
  broadcast (size-1) axis does not fold by PE; it stays width-1 and replicates. `stream_dim │
  block_dim` fails naively (can't divide size-1 by PE=16). Handled by a **`broadcast_aware`
  stream-tiling expression** (engine, not author), so the size-1 exception is engine-level.
- **Per-port polymorphic dtype.** Brainsmith `_elementwise_binary_output_datatype()`
  dispatches on `func` across 17 ops. This is a `Derived` reading `p.func`, but it belongs
  **on the OutputSchema**, co-located with its tiling — not in a flat op-level bucket.
- **Role reassigned by a param.** `rhs` role flips DATA_IN (dynamic_dynamic) vs
  WEIGHT_SINK/const (dynamic_static) on `input_pattern` — the port-taxonomy's "role names the
  binding, not the pin," driven by a kernel_param.

---

## 3. The KernelOp object

```
KernelOp                                     # the OP: WHAT is computed + its legal space
  op_type / domain / backend markers         # identity — makes getCustomOp resolve (R1 compat)
  interfaces: list[InterfaceSchema]          # ← LIST, port-indexed. THE core declaration.
  kernel_params: {name: spec}                # structural, fixed at inference (func, epsilon, TH)
  dse_axes:     tuple[Axis]                   # explorable → engine Axes (free stream dials)
  constraints:  tuple[Predicate]             # op-level legality
  cost_model:   (point,ctx) -> cycles        # DEFAULT rough throughput; impl may override
  resource_model: (point,ctx) -> {LUT,...}   # DEFAULT rough shape; impl fills coefficients
  implementations: pool[Implementation]      # HOW — feasibility, sources, emit, cost OVERRIDE
  parameters: composed pool | None           # weight delivery (only if an interface is a param)
  capabilities: {owns_runtime_weights, ...}  # explicit flags; replace R1–R3 duck-typing

InterfaceSchema                              # one port (or port group)
  name, index, direction                     # lhs/rhs/weights/output; port index
  block_tiling:  list[dim-expr]              # TENSOR→BLOCK  (FULL_DIM or a callable)
  stream_tiling: list[dim-expr | derived]    # BLOCK→STREAM  (literal axis, OR derive_dim(...))
  datatype:      resolver                     # per-port dtype (may dispatch on kernel_param)
  role: Role                                  # port-taxonomy role (DATA_IN/WEIGHT_SINK/DATA_OUT)
  required_layout: str | None
```

Everything in Tier-0..3 of `consumer-surface-model.md` — the 8 getters, stream widths,
divisibility predicates, `get_exp_cycles` — is **derived from `interfaces` + `cost_model`**,
never hand-written. The author writes the declaration; the engine emits the FINN surface.

---

## 4. The three ops expressed

### LayerNorm (clean reduction; 1 in / 1 out)
```
interfaces = [
  InterfaceSchema("input",  IN,  block=[FULL_DIM], stream=["SIMD"], role=DATA_IN,
                  datatype=graph_dtype),
  InterfaceSchema("output", OUT, block=[FULL_DIM], stream=[derive_dim("input", STREAM, -1)],
                  role=DATA_OUT, datatype=const("FLOAT32")),
]
kernel_params = { epsilon: (f, required, 1e-5) }
dse_axes      = [ divisor_axis("SIMD", from=input.last_dim) ]
constraints   = [ epsilon > 0 ]
cost_model    = default   # prod(idim)//SIMD leading term;
                          # +val_queue_len pipeline-fill is an IMPL (rtl) override
```
LayerNorm proves the rough-op / precise-impl seam: `layernorm_rtl.get_exp_cycles` adds
`val_queue_len` fill terms that are microarchitecture-specific → the override.

### Elementwise (multi-port + broadcast; N in / 1 out)
```
interfaces = [
  InterfaceSchema("lhs", IN, index=0, block=[FULL_DIM], stream=["PE"], role=DATA_IN,
                  datatype=graph_dtype),
  InterfaceSchema("rhs", IN, index=1, block=[FULL_DIM], stream=[broadcast_aware("PE")],
                  role=DATA_IN_or_WEIGHT_SINK,             # flips on input_pattern
                  datatype=VALUE_OPTIMIZED),
  InterfaceSchema("output", OUT, block=[FULL_DIM], stream=[derive_dim("lhs", STREAM, -1)],
                  role=DATA_OUT, datatype=polymorphic_on("func")),   # Add widens / Div / bitwise
]
kernel_params = { func: (...17 ops), input_pattern: {dynamic_static, dynamic_dynamic} }
dse_axes      = [ divisor_axis("PE", ...), ram_style, mem_mode ]
capabilities  = { owns_runtime_weights: (input_pattern uses a const port) }
```
`interfaces` as a list is what makes `get_*_datatype(ind)`, `get_folded_*_shape(ind)`,
`inFIFODepths=[2,2]` all index correctly. The broadcast size-1 exception is a
`broadcast_aware` stream expr handled by the engine.

### MVU (the hard case — forces derived tiling + temporal param)
```
interfaces = [
  InterfaceSchema("act",     IN,  index=0, block=[FULL_DIM, "MW"], stream=["SIMD"],
                  role=DATA_IN,  datatype=graph_dtype),
  InterfaceSchema("weights", IN,  index=1, block=[...param...],
                  stream=[derive("SIMD") * derive("PE") / param("TH")],   # WSIMD = PE·SIMD/TH
                  role=WEIGHT_SINK, datatype=value_optimized_weight),
  InterfaceSchema("output",  OUT, block=[FULL_DIM, "MH"], stream=["PE"],
                  role=DATA_OUT, datatype=acc_datatype),                  # data-dependent Derived
]
kernel_params = { TH: (i, 1), noActivation, binaryXnorMode }
dse_axes      = [ divisor_axis("SIMD", from=MW), divisor_axis("PE", from=MH) ]
constraints   = [ MW%SIMD==0, MH%PE==0 ]
cost_model    = default   # nf·sf·n_vecs·TH — reproduces FINN exactly
parameters    = compose(parameters_pool)   # weights interface is a param
implementations = [
  mvau_hls        (feasible: TH==1),                       # HLS can't tile
  mvau_rtl_untiled(feasible: TH==1, sources: mvu/*),
  mvau_rtl_tiled  (feasible: TH>1,  sources: mvu_tiled/*),  # tiled = POOL PEER, not a generate fork
]
```
The weight port's cross-interface derived tiling and TH-as-kernel-param are unrepresentable
in the naive model. `mvu_tiled` becomes a flat Implementation peer with `feasible=(TH>1)`,
dissolving FINN's internal `generate` fork — the `implementation` axis IS the tiled/untiled
choice.

---

## 5. The op/impl split (final, grounded)

| Concern | Owner | Proved by |
|---|---|---|
| Interface **list** + per-port block/stream tiling | **Op (schema)** | Elementwise (N ports); MVU (PE/SIMD on different interfaces) |
| Derived / cross-interface stream_tiling | **Op (schema, `derive_dim`)** | MVU weight port = f(act, out, TH) |
| Temporal params (TH) | **Op (kernel_params)**, read by tiling + cost | MVU |
| Broadcast / size-1 folding exception | **Engine** (`broadcast_aware`) | Elementwise rhs |
| 8 getters, stream widths, divisibility preds | **Engine-derived from interfaces** | all three hand-code these today |
| Per-port polymorphic datatype resolver | **Op, on the interface** | Elementwise output (17-op); MVU acc |
| **Rough** throughput cost | **Op (default)** — `prod(trips)·temporal` | all three; MVU verified exact |
| **Precise** cost (pipeline fill, queue latency, memory bound) | **Impl (override)** | LayerNorm rtl queue terms; CIG-rtl overrides `get_exp_cycles` |
| Feasibility of a point, `.sv` sources, `emit` | **Impl** | MVU tiled(TH>1) vs hls(TH==1) |
| Weight delivery | **Composed parameters pool** | MVU / elementwise-const; absent for LayerNorm |

Retired by this model: PE/SIMD as flat op axes (now derived stream dials on interfaces);
`mvu_tiled` as an internal fork (now a pool peer).

---

## 5.1 REVISION (2026-07-20): stream tiling is Implementation-owned, not schema-owned

### The generating principle: BLOCK = Op (math), STREAM = Backend (RTL)

The whole op/backend split follows from ONE conceptual line, mapped onto TENSOR/BLOCK/STREAM:

| Tier | Answers | Determined by | Owner |
|---|---|---|---|
| **TENSOR** | what the data is | the ONNX graph | **Op** |
| **BLOCK** | how the *mathematics* segments the tensor — the calculation-state quantum | the operator's math | **Op** |
| **STREAM** | how many elements/clock — the parallelization | the RTL realization | **Backend** |

Everything **at or above BLOCK** is "how the math segments the data" (ONNX-semantic,
op-owned); **STREAM is the RTL translation** of that segmentation into cycles and wires
(backend-owned). Tiling (PE/SIMD/TH, folded shapes, divisibility predicates) is *by
definition* the BLOCK→STREAM lowering, so it is backend-owned **by construction, not by
convention.** This is why there is no op-level tiling default: the op has no nominal tiling
to refine — it has a *block structure*, and each backend independently decides how to stream
it. The REDUCTION/FREE/PARAM tags ARE the block definition (tagging MW REDUCTION = "the block
must span MW to hold one dot-product's state" — a statement about the math, silent about
hardware). This principle is a **test for every future op/backend split**: does the concern
describe how the operator's mathematics segments the tensor (→ Op), or how a circuit realizes
that segmentation (→ Backend)? Datatype narrowing (realization → backend), reduction domains
(math → op), pipeline-fill latency (circuit → backend), rough trip-count cost (math → op
default) all independently satisfy it.

**One seam named honestly — BLOCK is where the two worlds meet, so split it one level finer:**
- **Block *structure*** (which dims form the calculation state — the reduction topology):
  pure ONNX-math → **Op**. A matmul contracts MW; no backend disagrees.
- **Block *extent*** (how much of that state the hardware materializes per step — partial
  accumulation, N-vectors/block, a systolic tile): a genuine RTL degree of freedom → may
  drift to **Backend**. This is the op↔backend handoff point (see the "Open" note below), not
  a stray edge case.

### The corrected ownership split

§3–§5 above show tiling declared on the op-level `KernelSchema` (mirroring Brainsmith,
whose backends are subclasses that inherit one schema). **This is superseded.** In our pool
model each `Implementation` contributes its own axes guarded on selection, so tiling-on-a-
shared-schema is the wrong default: it forces one realization's PE/SIMD/TH onto every
backend. The corrected split — consistent with the datatype-narrowing and cost decisions
(op owns the ONNX invariant; Implementation owns realization detail):

**No op-level tiling default.** An op that has multiple backends folding identically still
declares tiling per-backend; we do NOT add an inherited nominal default (decided 2026-07-20).
The duplication of `stream=[SIMD]` across peers is accepted as the price of a clean,
single-owner rule.

| Concern | Owner (revised) |
|---|---|
| interface **list** (name, index, direction, role) — the arity | **Op** |
| per-dim **semantic tags** REDUCTION / FREE / PARAM (which dims are contracted) | **Op** — the ONNX computational contract |
| **normal** (TENSOR) shapes + shape inference (Tier-0, pre-specialize) | **Op** |
| boundary/semantic datatype **rules** (e.g. elementwise output dtype dispatch on `func`) | **Op** |
| op-invariant constraints (initializer present, epsilon>0) | **Op** |
| rough cost model | **Op (default)** |
| each interface's **stream_tiling** expression | **Implementation** |
| the folding **params** (PE, SIMD, TH, …) as axes | **Implementation** |
| the folding **divisibility predicate** (`MW%SIMD==0`) + stream-width derived | **Implementation** |
| **folded** (STREAM) shapes (Tier-2+, post-specialize) | **Implementation** |
| value-optimized datatype **narrowing**; realization dtypes | **Implementation** |
| cost **override**; sources; emit; feasibility | **Implementation** |

**Why the split lands here — three independent decompositions agree:**
- the **param + its predicate + its stream-width are one unit** (`MW%SIMD==0` is meaningless
  until SIMD exists; SIMD exists only on a backend that folds act by SIMD). That unit is
  exactly what `pool_schema` already contributes per Implementation — no new mechanism.
- **normal shape = op / folded shape = Implementation** is the same seam FINN already has
  (`get_normal_*` needs no backend; `get_folded_*` needs PE) and the same
  **pre/pos-specialize tier boundary** in `consumer-surface-model.md`.
- FINN's own MVU already varies stream tiling by realization: `get_folded_input_shape(ind=1)`
  branches on `mem_mode` (weight port `(SIMD·PE)/TH` decoupled vs `PE` dynamic).

**MVU re-expressed under this split:**
```
# OP (KernelSchema) — no PE/SIMD/TH, no stream_tiling:
interfaces = [
  Interface("act",     IN,  0, role=DATA_IN,    dims=[FREE(vecs…), REDUCTION(MW)]),
  Interface("weights", IN,  1, role=WEIGHT_SINK, dims=[REDUCTION(MW), FREE(MH)], kind=PARAM),
  Interface("output",  OUT, 0, role=DATA_OUT,    dims=[FREE(vecs…), FREE(MH)]),
]
kernel_params = { noActivation, binaryXnorMode }
constraints   = [ weights initializer present unless params dynamic ]

# IMPLEMENTATIONS (tiling + params + predicates live here):
mvau_hls          act=[SIMD] out=[PE] weights=[PE*SIMD]      axes=[SIMD,PE,resType]
                  preds=[MW%SIMD==0, MH%PE==0, SIMD≥MW/1024, no-true-binary]
mvau_rtl_untiled  act=[SIMD] out=[PE] weights=[PE*SIMD]      axes=[SIMD,PE,dsp_version]  sources=mvu/*
mvau_rtl_tiled    act=[SIMD] out=[PE] weights=[(PE*SIMD)/TH] axes=[SIMD,PE,TH]           sources=mvu_tiled/*
                  preds=[MW%SIMD==0, MH%PE==0, (PE*SIMD)%TH==0]
```
`TH` exists ONLY on `mvau_rtl_tiled`, so the weight-port expression `(PE·SIMD)/TH` is only
writable where its operands are in scope — **all its deps are backend-local**, dissolving the
cross-interface un-introspectable-dep worry (§6.2). LayerNorm: `channels` tagged REDUCTION
(block spans it) even though output preserves it — REDUCTION means "block spans," not "dim
disappears"; `layernorm_rtl` carries the cost override beside its tiling. Elementwise: no
REDUCTION dim → BLOCK=TENSOR is itself an op-level fact; the N-input arity is the op-level
interface list; the broadcast `broadcast_aware` and const-narrowing live on the impl.

**Open (defer to empirics) — the block-structure/extent seam:** block *structure* (reduction
topology) is settled op-level; block *extent* (a backend that materializes 2 input-vectors/
block, or a systolic tile) is the genuine op↔backend handoff and may be Implementation-owned —
which would shrink the op contract toward "interface list + reduction domains + boundary
dtypes." Let MVU-tiled + a systolic sketch decide; do not settle now.

**Consequence for the evaluator (§6.2):** this DIVERGES from the vendored `schemas.py`
(which bundles `name+stream_tiling+datatype` on one `InputSchema` because schema-owns-
everything). We split it: an **op-side interface** (identity + semantic dims + role) and an
**impl-side tiling** (stream expression + params). The expression evaluator therefore
resolves expressions that live on `Implementation` interfaces; TH-referencing ones are scoped
to `mvau_rtl_tiled`.

---

## 5.2 The unified ownership principle: GIVEN / op-BLOCK-default / backend-STREAM-override

Tiling, cost, and datatype are not three ad-hoc ownership calls — they are **one pattern**,
and their differences fall out of *where each sits on the BLOCK↔STREAM axis*. Every coordinate
decomposes into three provenances:

- **GIVEN** — fixed by the ONNX graph / Context; neither op nor backend owns it.
- **op-STRUCTURAL default** — a rule whose *shape* is dictated by the op's BLOCK structure
  (the reduction topology / the math), which *reads backend-owned STREAM dials as inputs*.
- **backend-STREAM override** — the microarchitecture's realization, deviating from the
  op-structural ideal when the hardware clamps or deviates from steady state.

| Coordinate | GIVEN | op-STRUCTURAL default (BLOCK) | backend-STREAM override | Why it lands here |
|---|---|---|---|---|
| **Tiling** | — | **none** | **fully backend** | *pure STREAM* — no BLOCK content, so there is nothing for the op to default (this is *why* §5.1 has no op-level tiling default) |
| **Cost** | — | yes — reduction-aware formula | yes — fill / stall / drain | formula *shape* is BLOCK (reduction), *inputs* are backend STREAM dials |
| **Datatype** | input dtypes; param dtype **ceiling** | yes — envelope from reduction length + givens | yes — hardware clamp (e.g. 48-bit DSP accumulator) | envelope is BLOCK-derivable; realization may clamp it |

**This is the deep answer to "why no op-level tiling default but an op-level cost default":**
tiling is the *pure STREAM* case, so it has no op content; cost and datatype are *mixed* —
their shape is BLOCK-structural (reduction-dependent) even though they read STREAM dials as
inputs. The op-level cost model earns its place through **reduction ownership**, NOT parallelism
ownership: MVU's cost is a *product across the reduction* (`nf·sf·n_vecs`) that the engine's
generic max-over-interfaces floor under-counts because MW is a reduction domain that must be
fully traversed per output — an op-owned math fact; elementwise has no reduction ⇒ BLOCK=TENSOR
⇒ the generic floor is exact ⇒ `cost_model is None`. (Both validated:
`test_kernelop_mvu.py` / `test_kernelop_elementwise.py`.)

### Datatype: the three-provenance split, precisely

"Datatypes are Derived with one source of truth" (MOTIVATION §1.5) is right *per fact* but hides
a gradient of provenance:

1. **Input / activation dtypes — GIVEN.** Read from the graph; we do not control the activation
   data handed to us. The op only *names which tensor* carries them (interface identity). Zero
   ownership.
2. **Param dtypes — GIVEN ceiling + gated narrowing.** The graph declares the dtype (a ceiling).
   The kernel may narrow it value-optimized from the actual param values — but ONLY when the
   params are **static**, and staticness is a *parameter-delivery* coordinate (the composed
   sub-pool). Graph sets the ceiling; the delivery topology decides whether narrowing is legal;
   the narrowing is math on the values.
3. **Internal (accumulator) + output dtypes — op-DERIVED envelope, backend-overridable.**
   `accDataType = f(input_dtype, param_dtype, reduction_length)` — a function of GIVENs and the
   BLOCK reduction length (op-owned math), so the *semantic envelope* is op-derivable and lives
   as an op-level `Derived`. The backend enters ONLY when its microarchitecture **constrains**
   the envelope (a hard DSP accumulator width, a fixed output packing): then it *overrides* the
   op's ideal with its realizable dtype. Output = accumulator when no activation, else GIVEN.

**Clean statement:** the graph owns the GIVENs; the op owns the datatype **contract** (which
ports carry dtypes + the semantic rules relating them — one source per fact, MOTIVATION §1.5
preserved); the backend owns the **realization** (the envelope its hardware actually provides).
**Honest current state:** in FINN *and* our MVAU, all three derived dtypes are op-level and **no
backend overrides one yet** — every backend accumulates full-precision. The backend override is
the design *affordance* (same slot-shape as the cost override), not exercised today. "Usually
determined by the backend" is more precisely: *derived by an op-level rule that a backend may
realize differently* — and today they always agree.

---

## 6. Engine features this requires BEFORE KernelOp code

The model survives MVU (the hardest case) only with three features the naive last-axis
version lacks:

1. **An op-side interface list** (identity + semantic dims REDUCTION/FREE/PARAM + role) —
   replaces the MVU-biased INPUT/WEIGHTS/OUTPUT triple in `ops/mvau/names.py`. The vendored
   `InterfaceSchema`/`KernelSchema` provide the container; per §5.1 the **stream_tiling is
   NOT on this op-side object** — it moves to the Implementation.
2. **A derived stream-tiling expression language, resolved on Implementation interfaces** —
   `derive_dim(iface, hierarchy, axis)`, arithmetic (`* / `), `param(...)`, and
   `broadcast_aware(...)`, evaluated against a resolved `Point`. This is the one genuinely new
   engine primitive; MVU's `mvau_rtl_tiled` weight port `(PE·SIMD)/TH` is its acid test (all
   deps backend-local). **Load-bearing, not sugar.**
3. **Cost/resource model as a defaultable op-level element with a per-impl override slot** —
   mirroring how `emit` already dispatches per Implementation.

---

## 7. Build order

Tier-3 (estimate-only) first, per `consumer-surface-model.md` ("the cheapest useful
migration target"). Everything Tier-3 needs — the 8 getters, widths, rough cost — is pure
derivation from `interfaces` + `cost_model`; no Vivado, no emit, no stitch.

1. **LayerNorm** — ideal first vertical slice: 1-in/1-out, clean tiling, and its
   cost-override cleanly demonstrates the op/impl seam.
2. **MVU** — proves derived stream-tiling + the tiled Implementation peer.
3. **Elementwise** — proves the interface list + broadcast + polymorphic dtype.

Tier-4 (codegen dispatch to `emit`, `get_verilog_top_module_intf_names` structural
sub-contract, `make_weight_file` capability) reuses the emit/stitch machinery already
validated in the Docker harnesses.

The load-bearing prerequisite for everything right of `specialize_layers`:
**`feasible_impls()` as a `resolve` over the `implementation` axis's feasibility gates**
(consumer-surface R11). The gates already live on each bundle's `feasible`; MVU's
`hls(TH==1)` vs `rtl_tiled(TH>1)` is the acid test that it maps cleanly.

---

## 8. Provenance

Grounded against, and verified line-by-line from:
- FINN: `src/finn/custom_op/fpgadataflow/{matrixvectoractivation,layernorm,elementwise_binary}.py`
  and their `rtl/`,`hls/` leaves; `finn-rtllib/mvu/`, `finn-rtllib/mvu_tiled/`.
- Brainsmith: `brainsmith/kernels/{layernorm,elementwise_binary,channelwise}/`;
  `docs/developer-guide/dataflow-modeling.md`; `brainsmith.dataflow` (`KernelSchema`,
  `InputSchema`/`OutputSchema`, `FULL_DIM`, `derive_dim`, `KernelOp`).
- MVU cost identity `nf·sf·n_vecs·TH` and weight `WSIMD=PE·SIMD/TH` cross-checked against
  `mvu_tiled_axi_wrapper.v` params and `matrixvectoractivation.get_folded_input_shape`.
