# Derivation-Layer Dissection — for the spec refactor

*Direct dissection of the vendored `finn/src/finn/kernels/derivation/` modules the
spec refactor touches, before editing them. Produced 2026-07-14. Goal: edit the
two largest modules (`builder.py` 708, `constraints.py` 1080) with a full
behavioral model, not from fragments. Grounds the §9 acceptance criteria.*

---

## 1. The two-phase pipeline (behavioral model)

```
KernelSchema  ──build()──►  KernelDesignSpace  ──configure(params)──►  KernelDesignPoint
 (structure)                 (shapes+ranges,           (stream shapes,
                              validated once)           re-validated each call)
```

**Phase 1 — `DesignSpaceBuilder.build(ctx)` (builder.py:170).** Reads ONNX shapes,
resolves datatypes (graph fallback / VALUE_OPTIMIZED / derive), resolves block
shapes from `block_tiling`, normalizes `stream_tiling` to a template, then:
- **splits constraints by phase** (builder.py:269-275): `structural` vs everything
  else (`optimization`).
- **validates `structural` constraints once, then DISCARDS them** (builder.py:283-297,
  and the note at :310 "validated above but not stored — never re-validated").
- **stores only `optimization_constraints` on the space** (builder.py:316).
- **computes the design-space parameters** via `_compute_dimension_ranges`
  (builder.py:567) → `KernelDesignSpace.parameters`.

**Phase 2 — `KernelDesignSpace.configure(config)` (dse_models.py).** Resolves each
interface's `stream_tiling` template against the config → concrete `stream_shape`,
builds the `KernelDesignPoint`, then re-runs the stored `optimization_constraints`.

**Key consequence for the refactor:** the builder has exactly TWO constraint homes —
validated-once-and-discarded (`structural`) and stored-and-re-run-per-config
(`optimization`). A **`realization` constraint fits NEITHER** — it is device-aware
and evaluated at *selection* time, which is outside the builder/space entirely. So
the realization phase is **not a builder edit** (see §3.2).

---

## 2. The composed design space ALREADY EXISTS (the big finding)

`_compute_dimension_ranges` (builder.py:567-702) already fuses two sources into one
`all_dimensions` dict that becomes `KernelDesignSpace.parameters`:

1. **Tiling dims** (PE, SIMD) — auto-extracted from `stream_tiling` strings, valued
   as divisors of the GCD of block dims, wrapped in `OrderedParameter` (:646-654).
2. **DSE dims** — from `schema.dse_parameters`, auto-detected ordered (list/tuple →
   `OrderedParameter`) vs discrete (set/frozenset → `frozenset`), callable values
   evaluated against the build ctx (:662-700).

And `KernelSchema.build_nodeattr_registry()` (schemas.py:429-471) already emits
`dse_parameters` as nodeattrs (:464-466), with `_validate_transformation_fields`
guarding name-uniqueness vs `kernel_params` (schemas.py:421-427).

**So `dse_parameters` is a complete, integrated, working system.** `ParameterSpec`
(schemas.py:50) supports exactly what the spec needs: `ram_style={"distributed",
"block"}` is the *documented example* (schemas.py:96, 361, 601). Our `knob_specs`
was reinventing a subset of this, worse (not DSE-visible, not auto-registered).

**Refactor consequence:** the "composed design space" §9 item is *mostly already
built at the op level*. What's missing is only the **backend contribution** —
letting the selected `Implementation` add its `dse_parameters` to the op schema's
before the space is built (or extend the space after). See §3.1.

---

## 3. Integration seams for the two spec changes

### 3.1 `knob_specs` → `Implementation`-contributed `dse_parameters`

The mechanism exists; only the *plumbing of backend params into the schema* is new.
Two clean options, both small:

- **(A) Extend the schema before build.** At selection time (in `KernelCore`), once
  the `Implementation` is chosen, merge `impl.dse_parameters()` into a copy of the
  op's `KernelSchema.dse_parameters`, then build the space from the merged schema.
  The space is then genuinely composed (op ⊕ backend), matching the spec's pipeline.
  *Touches:* `implementation.py` (add `dse_parameters()` returning
  `dict[str, ParameterSpec]`), `core.py` (merge at build), delete `knob_specs`.
  *Does NOT touch vendored code* — `KernelSchema` already accepts `dse_parameters`.
- **(B) Extend the space after build.** Build op-space, then add backend dims to
  `KernelDesignSpace.parameters`. Rejected: `parameters` is assembled immutably in
  the builder; post-hoc mutation fights the two-phase design. (A) is cleaner.

**Chosen: (A).** No vendored edit for this change — it's schema composition at the
adapter layer. `ParameterSpec` is the type `Implementation.dse_parameters()` returns.

### 3.2 `precondition(ctx) -> bool` → a `realization`-phase constraint

`Constraint` is a **`Protocol`** (constraints.py:58, `@runtime_checkable`), duck-typed
— NOT a base class. A realization constraint needs only: `check(ctx) -> str|None`,
`describe() -> str`, `evaluation_phase == "realization"`. No inheritance.

The `evaluation_phase` heuristic (constraints.py:85-118) returns `structural` by
default, `optimization` for STREAM-hierarchy. It needs a **third allowed value**
`"realization"` — but critically, the builder's split (builder.py:269-275) currently
does `structural` vs `!= structural`, which would wrongly bucket `realization` into
`optimization` (stored on the space, re-run per configure without device ctx). So:

**Two small vendored edits, plus the real work outside the vendored layer:**
1. **builder.py:269-275** — make the split three-way: `structural`, `optimization`,
   and *drop* `realization` from both (the builder must NOT try to evaluate device
   constraints — it has no `fpgapart`). Realization constraints pass through the
   schema untouched by the builder.
2. **A new `RealizationValidationContext`** (validation.py) carrying `fpgapart` (+
   toolchain version) alongside the design_point accessors — the device-aware context
   §5.2 of the spec calls for.
3. **The evaluation itself lives in the registry/selection** (registry.py), NOT the
   builder: `realizability` = run the op's + backend's `realization`-phase constraints
   against a `RealizationValidationContext`, collect reason strings. This *replaces*
   `Implementation.precondition(ctx) -> bool` with
   `Implementation.realization_constraints() -> list[Constraint]` (or the impl still
   owns a `realizability` method that internally evaluates them — TBD in refactor).

**Net vendored footprint of the whole refactor: tiny.** One 3-way split in builder.py,
one new context dataclass in validation.py, one extra allowed `evaluation_phase`
string. The bulk of the change is in OUR code (implementation.py, core.py,
registry.py) — the vendored derivation layer is well-factored enough that the spec
changes mostly compose *on top of* it, not *into* it.

---

## 4. Used-vs-vestigial ledger (for the later trim pass)

What our design actually exercises vs. carried-along Brainsmith generality:

| Module | LOC | Status | Notes |
|---|---|---|---|
| `schemas.py` | 545 | **USED (core)** | KernelSchema/Input/Output/ParameterSpec — all load-bearing |
| `builder.py` | 708 | **USED (core)** | build() + _compute_dimension_ranges — the pipeline |
| `dse_models.py` | 1030 | **USED (partial)** | design_space/point + interface props USED; the ~15 navigation methods (sweep_dimension, with_step_up, with_percentage, …) are UNEXERCISED — the "DSE engine" that's dead until `explore` is built |
| `constraints.py` | 1080 | **USED (partial)** | Constraint protocol + the ~6 constraint classes we use (IsDynamic, IsStatic, DimensionDivisible, DatatypeInteger) USED; the other ~10 (ShapesEqual, DimensionInRange, TensorDimMatches, AttrCompare, CustomConstraint, …) UNEXERCISED |
| `validation.py` | 376 | **USED (core)** | the two contexts; will gain a third (realization) |
| `ordered_parameter.py` | 336 | **USED** | OrderedParameter backs tiling + ordered DSE dims |
| `types.py` | 202 | **USED** | FULL_DIM/FULL_SHAPE/ShapeHierarchy/VALUE_OPTIMIZED |
| `spec_helpers.py` | 512 | **USED (partial)** | derive_dim, value_optimized_datatype, derive_datatype USED; other helpers latent |
| `template_resolution.py` | 256 | **USED** | resolve_template/normalize_template — block/stream resolution |
| `broadcast_helpers.py` | 335 | **VESTIGIAL (today)** | no current op uses broadcasting; AddStreams/elementwise *would* (future) |
| `inference_helpers.py` | 510 | **VESTIGIAL (today)** | lift_scalar_to_rank1 etc.; ONNX-inference preprocessing, unused by our infer_from |
| `transformation.py` | 32 | **USED (thin)** | TransformationResult — returned by infer_from |
| `_math.py` | 27 | **USED** | divisors (vendored helper) |

**Trim candidates (later, separate pass):** `broadcast_helpers.py` (335) and
`inference_helpers.py` (510) are fully unexercised today = ~845 lines. BUT both are
plausibly needed for the *second op* (AddStreams uses broadcast). **Recommendation:
do NOT trim before the second op** — decide after we know whether AddStreams pulls
them in. The DSE-navigation methods in `dse_models.py` and the unused constraint
classes are "latent, not vestigial" — they're the surface `explore` will use; keep.

---

## 5. Verdict

**A direct black-box refactor would have been unsafe** — both changes touch the two
largest vendored modules. But the dissection shows the derivation layer is **well
factored**: `dse_parameters` is already a complete composed-space mechanism (we were
reinventing it), and `Constraint` is a duck-typed Protocol that admits a third phase
without inheritance. The refactor's vendored footprint is **~3 small edits**; the bulk
lands in our own code. The "extract smallest coherent system and rebuild" path is
**not warranted** — it would re-author proven, working derivation math for no gain.

**Proceed with the spec refactor**, in this order (all venv-testable):
1. `Implementation.dse_parameters()` + delete `knob_specs`; compose schema in `core.py` (§3.1).
2. Third `evaluation_phase="realization"` + 3-way builder split + `RealizationValidationContext` (§3.2).
3. Move feasibility eval into registry selection; `realizability` returns reasons.
4. Tests per §9; trim pass deferred until after the second op decides broadcast/inference.
