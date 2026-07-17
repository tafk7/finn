# KernelOp vs Kernel Implementation — Roles & Interfaces

*Status: proposed spec for review, 2026-07-14. Defines the roles and interface
contracts of the dataflow-kernel backend; leaves optimization/lowering
algorithms open (interfaces only). Grounded in the `finn-hw-backend-analysis/`
deep dive (existing HWCustomOp split, the F1–F6 fusions, a 27-op census) and the
existing implementation under `finn/src/finn/kernels/`.*

---

## 0. Why this doc exists

We kept patching symptoms — where does `ram_style` live, is it `knob_specs` or
`dse_parameters`, is feasibility a `precondition` or a `constraint`, *when* do
backend knobs join the design space — because the **roles** of the two central
objects were never crisply defined. This spec fixes the roles first; the
symptom-questions then resolve as consequences (§6).

Scope of this pass: **two roles** (`KernelOp`, `Implementation`), the **composed
design space**, a **three-phase feasibility model**, and **pass interfaces**. A
third role — memory / weight-delivery (`mem_mode`) — is real (it is the largest
historical fusion, F2) but **explicitly deferred**: it needs analysis of the new
MLO system and the prototype `kernel-flow` work. It appears here only as a named
extension point (§7), not a defined interface.

---

## 1. The two roles, in one sentence each

- **`KernelOp` — the op identity.** *What* is computed and *what shapes/types/
  folding* the computation admits — **device-independent**, one per op family.
- **`Implementation` — a realization.** *How* a chosen design point becomes
  hardware on a *specific device* — **device-dependent**, one per backend or
  microarchitecture; an op has many.

The dividing question for any responsibility is: **does it change if we swap the
FPGA part or the backend?** If no → `KernelOp`. If yes → `Implementation`.

This is the same clean line FINN's own `HWCustomOp` draws for the 8
shape/width/datatype methods (the part everyone honors) — we keep that line and
stop it from leaking (§4).

---

## 2. `KernelOp` — the op identity

Owns everything intrinsic to the op, independent of any backend or device.

| Responsibility | Contract | Notes |
|---|---|---|
| Structure | `build_schema(node, model) -> KernelSchema` | interfaces, semantic params, constraints, op-intrinsic DSE params |
| Golden model | `reference(inputs, attrs) -> outputs` | **authoritative oracle** — see §2.1 |
| Inference | `can_infer_from` / `infer_from` | ONNX-op → kernel-node lowering |
| Derivation | builds `KernelDesignSpace`, `configure()`s a `KernelDesignPoint` | inherited, not per-op |
| Params | `extract_params(node, model) -> ParamBundle` | initializer tensors only |

**The op owns three kinds of parameter** (all declared on its `KernelSchema`):
1. **Semantic** (`num_steps`, `act_val`) — part of the op's meaning. Fixed values.
2. **Tiling / folding** (`PE`, `SIMD`) — op-intrinsic *design-space* params. The
   op owns the *values*; a backend may *gate their validity* (§3, §5) but does
   not own them.
3. **Structural + optimization constraints** — device-free validity (is this
   design point legal *at all*).

The op never sees `fpgapart`, never generates code, never names a backend.

### 2.1 The golden reference is authoritative

The census found backends that are **numerically divergent** behind one op name
(requant: HLS float vs RTL truncated fixed-point; DWC: two different converter
algorithms). Consequence: the golden reference is the **oracle both
Implementations validate against**, so it must live on the op, and Implementations
are *not* assumed bit-identical to each other — only faithful to the reference
within their declared tolerance.

**Escape valve:** infra/shell ops (iodma, tlastmarker) have *no* meaningful
golden model. They are **not** `KernelOp`s in the full sense — see §7.

---

## 3. `Implementation` — a realization

Owns how a *valid* design point becomes hardware on a *specific* device.

| Responsibility | Contract | Notes |
|---|---|---|
| Contributed DSE params | `dse_parameters() -> dict[str, ParameterSpec]` | **the composed-space extension** — §5 |
| Feasibility | `realizability(ctx) -> None \| Reason` | device-aware, §5.2; strictly can-build |
| Preference | `priority: int` (+ future cost hook) | **separate** from feasibility |
| Codegen | `emit(design_point, params, config) -> Artifacts` | hermetic (unchanged) |

An `Implementation` is a **stateless strategy**, resolved **by name** from the
`implementation` nodeattr (nothing pickled). It reads config as data and never
touches the graph (the hermeticity contract, already enforced and tested).

**Feasibility and preference are different fields.** The census's named drift bug
(`_mvu_rtl_possible`) was a *preference* heuristic ("RTL not worth it below 4
bits") miscoded as a *feasibility* gate. `realizability` answers only "can this
backend build this point on this part"; "should we prefer it" lives in `priority`
/ ranking. Never mix them.

---

## 4. What must NOT leak — the F1–F6 checklist

The existing `HWCustomOp` split fails at six fusion points. Our roles must avoid
each; this is the acceptance checklist:

- **F1 — agnostic base calls backend-only methods.** *Ours:* `KernelOp` has no
  method that dispatches into an `Implementation`. The one boundary that knows
  both is the `KernelCustomOp` adapter.
- **F2 — hidden third axis (weight/memory delivery).** *Ours:* named and
  **deferred** (§7), not smeared into `KernelOp` or the compute `Implementation`.
- **F3 — IPI/block-design has no contract slot.** *Ours:* `Artifacts.ipi` is a
  first-class typed slot. (Decoupled-mode stitch still open, but it has a *home*.)
- **F4 — divergent numeric algorithm behind one op name.** *Ours:* the golden
  reference is authoritative (§2.1); backends validate against it.
- **F5 — agnostic base fused to backend microarch via cost models.** *Ours:*
  resource/timing cost is an `Implementation` concern (it depends on the backend),
  never on `KernelOp`.
- **F6 — backend-only ops with no agnostic base.** *Ours:* infra/shell ops are a
  separate category (§7), not forced into the `KernelOp` contract.

---

## 5. The composed design space (the central resolution)

The question "*when/where do backend knobs join the design space*" is answered by
making the space **composed**: op-owned params, *extended* by the selected
Implementation's contributed params.

```
KernelOp.build_schema()  ─►  op design_space            (PE, SIMD, semantic params)
                                     │
                                     ▼   §5.2 feasibility filter (device-aware)
        select Implementation  ◄─────────────────  realizability(ctx) over op-space
                                     │
                                     ▼   the chosen Implementation
   effective design_space  =  op params  ⊕  Implementation.dse_parameters()
                                     │           (ram_style @ HLS; depth_trigger_* @ RTL)
                                     ▼
        configure / explore  ─►  design_point  ─►  emit()
```

**Pipeline:** derive op-space → select backend by device feasibility over the
op-space → **extend** the space with the backend's contributed params → configure
/ explore the full point → emit. A knob like `ram_style` simply *does not exist*
until HLS is chosen — which is honest, not awkward.

This is a **genuine extension past the prior "ideal" corpus**, which kept all
explorable params op-level and let backends only *gate*. The census supports it:
DSP microarch (`$VERSION$`), `depth_trigger_*`, `ram_style`, `resType` are real
backend-only explorable axes. The borrowed framing is ir-lowering's "peer producer
with its own published surface" — extended from *ports* to *parameters*.

### 5.1 Parameter ownership table

| Kind | Owner | Mechanism | DSE-visible? |
|---|---|---|---|
| Semantic (`num_steps`, `act_val`) | KernelOp | `KernelSchema.kernel_params` | no (fixed) |
| Tiling / folding (`PE`, `SIMD`) | KernelOp | schema (auto from `stream_tiling`) | yes |
| Op resource knob | KernelOp | `KernelSchema.dse_parameters` | yes |
| **Backend resource knob** (`ram_style`, `depth_trigger_*`) | **Implementation** | `Implementation.dse_parameters()` | **yes** ← was dead |

### 5.2 Three-phase feasibility

Extend Brainsmith's `Constraint` (currently `evaluation_phase ∈ {structural,
optimization}`, device-free) with a **third phase**:

| Phase | When | Owner | Context | Answers |
|---|---|---|---|---|
| `structural` | design-space build | KernelOp | shapes/dtypes/params | is the *space* well-formed? |
| `optimization` | per `configure()` | KernelOp | + stream shapes | is *this point* legal? |
| **`realization`** | per candidate at selection | **Implementation** | **+ `fpgapart`, toolchain version** | can *this backend* build it *here*? |

`realization` constraints return a **reason string** (`None` = feasible), not a
silent bool — feeding an `explain()`-style "why was this backend rejected" report
(cheap, worth it). The context must carry toolchain version too: the census shows
Lookup URAM needs Versal **and** Vivado 2024.2.

---

## 6. The two original questions, resolved

- **`knob_specs` → gone.** Backend tuning params become **`dse_parameters`
  contributed by the `Implementation`** (§5). Uses Brainsmith's existing
  `ParameterSpec` machinery: auto-registered as nodeattrs, DSE-visible,
  round-tripping. Closes the dead-code gap structurally — the same gap real
  Brainsmith has (its backend knobs sit in `get_nodeattr_types`, invisible to DSE).
- **`precondition` → a `realization`-phase `Constraint`.** Same vocabulary as the
  op's constraints, one phase later, device-aware, reason-string result (§5.2).
  Feasibility only; preference stays in `priority`.

---

## 7. Named extension points (out of scope this pass)

- **Memory / weight-delivery role (the third axis, F2).** Deferred: requires
  analysis of the new **MLO** system and the prototype `kernel-flow` work. Will be
  a composed part the op holds, orthogonal to the compute `Implementation`. *Not*
  designed here. The census + prototype confirm this is not a side-axis: for MVAU
  and Thresholding, `mem_mode` (embedded / decoupled / external) is the *primary*
  multiplicity mechanism — the prototype expresses it as intra-kernel config
  branching, with *no* same-kind registry variants at all.
- **Composite / stitched Implementation (the prototype's "SIP").** The prototype's
  default Thresholding and only MVAU are `sip` kernels: an Implementation that owns
  no compute and *stitches sub-Implementations* (an HLS compute core + an RTL
  memstream) into one block via a `subkernels` tuple. This is neither `KernelOp`
  nor a leaf `Implementation` — it is an Implementation *composed of* Implementations,
  and it is tightly coupled to the deferred memory role (SIP exists largely to stitch
  a compute core to a weight streamer). **Deferred, and possibly avoided entirely:**
  MVAU-as-SIP is a monster kernel, and the cleaner answer may be to *decompose* it
  into several distinct kernels rather than bless the composite pattern. Named here
  so the role model does not silently assume every Implementation is a leaf; the
  decision (compose vs. decompose) is future work alongside the memory role.
- **Infra / shell primitive category.** iodma, tlastmarker, checksum have no
  golden reference and only a partial shape contract. They need a category *outside*
  the full `KernelOp` contract (a no-reference "mover"/shell op). Named, not defined.
- **Container / hierarchical op.** FINNLoop / StreamingDataflowPartition compose
  child subgraphs; they are neither `KernelOp` nor `Implementation`. Separate concept.
- **Cost / preference model.** The ranking algorithm behind `priority` — an open
  seam. Interface fixed (a candidate-set ranker), algorithm open.

---

## 8. Pass interfaces (shapes fixed, algorithms open)

Passes read **declared contracts only** — never `isinstance`/`op_type`. Adding a
backend or op is a registry entry, not a pass edit. Three interface shapes:

```
# 1. Lowering / codegen — one op + injected target -> artifact
emit(design_point, params, config) -> Artifacts          # already built, hermetic

# 2. Selection — a candidate set -> a binding (feasibility ⊥ preference)
select(op_kind, ctx, cost_fn?) -> Implementation | Infeasible(reasons)
    feasible = [i for i in pool if i.realizability(ctx) is None]   # phase-3
    rank feasible by (priority, cost_fn?, name)                     # preference

# 3. Exploration — a design space -> a chosen point (OPEN algorithm)
explore(design_space, objective) -> design_point
    # interface only: sweep/search strategy is future work.
    # today: configure() a single point. The engine is vendored but unused —
    # this is the seam that turns it on.
```

The **objective/cost function** and the **search strategy** are deliberately left
open — that is the space for future exploration. This spec fixes only the
*interfaces* through which such passes plug in.

---

## 9. Acceptance criteria — SATISFIED (refactor complete 2026-07-14)

1. ✅ `knob_specs` deleted; `ram_style` (HLS) and `depth_trigger_*`/`deep_pipeline`
   (RTL) are `Implementation`-contributed `dse_parameters`, DSE-visible and
   round-tripping. `Implementation.dse_parameters() -> dict[str, ParameterSpec]`;
   merged into the op schema at selection via `KernelCore._composed_schema()`.
2. ✅ `precondition` replaced by `realization`-phase constraints
   (`Implementation.realization_constraints()`), evaluated by
   `Implementation.realizability(ctx)` against a new `RealizationValidationContext`
   carrying `fpgapart` (+ toolchain). Returns `None` or a reason string; the
   registry surfaces reasons (explain-style). The RTL URAM rule is now
   `_UramRequiresUltraScale` (a real `Constraint`, phase `"realization"`).
3. ✅ Composed space at selection: op-only space = `{PE}`; +HLS = `{PE, ram_style}`;
   +RTL = `{PE, depth_trigger_uram, depth_trigger_bram, deep_pipeline}`. A
   different design space per backend — the honest model.
4. ✅ No F1–F6 regression: hermeticity source-scan + graph-destroyed gate still
   pass for both backends.
5. ✅ 31 pass / 5 container-gated skip. New tests:
   `test_backend_dse_param_composes_into_design_space`,
   `test_realizability_gates_on_device`,
   `test_infeasible_reason_surfaces_on_selection_failure`.

**Vendored footprint (as predicted by the dissection):** 3 small edits —
`builder.py` 3-way constraint split, `constraints.py` docstring for the third
phase, `validation.py` new `RealizationValidationContext`. Everything else landed
in our own code (`implementation.py`, `core.py`, `registry.py`, `kernel_op.py`,
the two backends).
