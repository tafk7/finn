# Generic Design-Space Model

*Status: proposed spec for review, 2026-07-15. Defines the generic design-space
model for the dataflow-kernel backend, designed against the full MVAU map
(`mvau-design-space.md`) and unwinding the prior three-phase / composed-space
assumptions. Validated line-by-line against MVAU's four stress-tests (§6 of the
map). Supersedes the parameter/constraint parts of `roles-and-interfaces.md` §5;
leaves optimization/search algorithms open (interfaces only).*

**§8 holds the worked MVAU decision tree — the concrete acid test.** Read it
alongside the primitives; every construct in §1 earns its place there.

---

## 0. What this replaces and why

The prior model (as built) had three assumptions MVAU falsifies:

1. **Axes are a static dict with fixed domains, computed once at build.** MVAU:
   whether `ram_style` even *exists* depends on `mem_mode`'s value; querying an
   absent axis **raises** (base:315). Axis existence and domain are *functions of
   the point-so-far*.
2. **Constraints come in three phases (structural / optimization / realization).**
   MVAU: the URAM rule reads config **and** device in one condition (hls:147) —
   irreducibly cross-phase. Phase is not a category; it is *what context a
   predicate happens to read*.
3. **The design space is composed per selected backend (op ⊕ backend params).**
   MVAU: `mem_mode` is an *op-level* axis that guards more downstream axes than
   backend choice does. "Backend" is just one guarding axis among many, not a
   privileged composition boundary.

The generic model replaces all three with **one** structure: a **dependent design
space** of guarded axes + derived quantities, over which a **single predicate**
decides legality against an all-available **context**.

---

## 1. The four primitives

```
Context     — everything known that is NOT a design choice:
              tensor shapes, initializer VALUES, fpgapart, toolchain version.
Axis        — a free design choice, with GUARDED existence and a
              point-dependent DOMAIN.
Derived     — a computed quantity (never a choice); may read other axes,
              the Context, and initializer data.
Predicate   — check(point, context) -> reason | None. The ONLY legality kind.
```

A **Point** is a partial-or-total assignment of values to the axes that *exist*
given the assignments made so far. The space is walked in dependency order.

### 1.1 Context (validates MVAU stress-test #2, #3)

```
Context:
    tensor_shape(name)        -> tuple[int, ...]      # from ONNX graph
    tensor_datatype(name)     -> DataType             # graph dtype
    initializer(name)         -> ndarray | None       # the actual weight/threshold VALUES
    fpgapart                  -> str
    toolchain_version         -> str | None
```

Carries **everything available at t=0** (fpgapart is a build-config global —
map §"where fpgapart enters"). No phase gating: a predicate or derivation reads
whatever it needs. `initializer()` is what makes data-dependent derivation
(`accDataType` from weight values, base:498) and data-dependent feasibility
(narrow-weight DSP packing, rtl:281) expressible. Input **datatypes come from the
graph** via `tensor_datatype` — a given, not a choice; all other dtypes are
`Derived` (§1.3).

### 1.2 Axis — guarded existence + point-dependent domain (validates #1, #4)

```
Axis:
    name        : str
    exists(point)            -> bool           # the GUARD (default: always)
    domain(point, context)   -> Domain         # valid values, may depend on point+context
    default(point, context)  -> value          # chosen when unset
```

- **`exists`** is the guard. `ram_style.exists(p) = (p.mem_mode == "internal_decoupled")`
  (map §2). If `exists` is false, the axis is *not in the point at all* — reading
  it is an error, not a default (matches base:315's hard raise).
- **`domain`** is point- and context-dependent. `PE.domain(p, ctx) = divisors(MH)`
  where `MH` comes from `ctx.tensor_shape` (base:351); `SEGMENTLEN` is not an axis
  (it is `Derived`). `Domain` is either an `OrderedParameter` (ordered, navigable —
  the existing type) or a discrete `frozenset`.

### 1.2.1 The `implementation` axis (NOT `backend`) — audit correction

An earlier draft had `backend.domain = {hls, rtl}`. **That is wrong** — it groups
designs by *language*, which is not a meaningful domain (audit 2026-07-15). FINN's
own stated intent (HANDOFF §3): "HLS/RTL is a language *field*, not the identity."
Evidence: FINN ships *multiple RTL source-sets for one op* (finn-rtllib/mvu/:
`mvu_8sx8u_dsp48` systolic, `mvu_vvu_8sx9_dsp58` GEMM, `mvu_4sx4u` DSP-packed), and
hls-vs-rtl of one op can be numerically divergent (requant). So "rtl" is not one
design — it is several.

The real axis is **`implementation`**: its domain is the op's **realization pool**
— the set of concrete source-template realizations, one per distinct compute design.
For MVAU today, grounded in the actual source files: `{mvau_hls, mvau_dsp_softvec,
mvau_dsp_packed}` (`mvau_dsp_softvec` = `mvu.sv`; `mvau_dsp_packed` =
`mvu_vvu_8sx9_dsp58.sv`). Then:

- **`language` and `dsp_primitive` (`$VERSION$`)** are `Derived` — `language` is a
  property of the chosen realization; `dsp_primitive` is a silicon consequence forced
  by `fpgapart` with no alternative design (see §1.2.2 for why this is `Derived`, not
  a pool axis).
- **The device constrains the pool via a `Predicate`**, it does not *determine* one:
  `mvau_dsp_packed` has a predicate "requires DSP58 ∧ w≤8 ∧ a≤9" (rtl:313); on a
  7-series it is illegal, `mvau_dsp_softvec` and `mvau_hls` remain.
- **`implementation` guards backend-specific axes** — `pumpedCompute` exists only for
  realizations whose `language`/core supports it. This is the unification: the old
  "composed-space, backend contributes params" collapses into "the `implementation`
  axis guards those axes," no special composition machinery.
- **`auto`/`resType=auto` is a deferral marker, not a value** — it means "no
  committed choice"; it is the *absence* of an assignment, resolved by `default`, not
  a domain member.

This matches the prototype registry, which already keys the pool by op and selects a
*class* (a realization), not a language (`kernel_registry.py`).

### 1.2.2 What defines the pool — buildable designs, not current feasibility

The single most error-prone question in this whole model is *"is this a pool member
or a `Derived`?"* — we got it wrong twice on MVAU's softvec-vs-packed cores. The
sharp rule (established 2026-07-15, grounded in `mvu_vvu_axi.sv:313`):

**The pool is the set of designs that COULD be built. It is never contracted because
the CURRENT device makes some infeasible.** A distinct source/compute template is a
pool member — permanently. Test by counting mutually-exclusive source templates: two
compute source files (`mvu.sv` soft-vectorized, `mvu_vvu_8sx9_dsp58.sv` INT8-packed)
= two pool members, full stop.

**"Forced" is a resolve-time OUTCOME, not a design property.** A selection looks
"forced" only when `resolve` + preference over *this* context happens to leave one
survivor. FINN's `generate` branch (`if (packed fits) use packed; else softvec`) is
not evidence that packed is the only design — it is FINN **hardcoding a preference**
("prefer packed when feasible") into silicon-generation code. Both `mvu.sv` and the
packed core can build on a DSP58+INT8 config; packed is merely better. Decomposed:

```
pool        = { mvau_hls, mvau_rtl_softvec, mvau_rtl_packed }   ← 3 buildable designs
feasibility = packed needs DSP58 ∧ w≤8 ∧ a≤9 ∧ lanes≤3 (Predicate over device+dtype+data)
              softvec needs any DSP part;  hls needs nothing
preference  = packed > softvec when both feasible   (the buried `generate` if, made explicit)
```

Feasibility **narrows** the pool at resolve time; it never **defines** it. The absence
of a user lever means only that FINN pre-decided the preference — the design
alternative still exists in the space. **Model for the wider menu; let predicates
narrow it.** This is future-proof by construction: a new device with more DSP options,
or a second design targeting the same primitive, turns a "forced" branch into a real
multi-candidate choice with **zero model change** — the preference simply starts
having more than one candidate to rank. A collapse-to-`Derived` would bake today's
narrow device menu into the model permanently and force a re-modeling when the menu
widens.

**`Derived` is reserved for a physical/logical consequence with NO alternative design
you could build.** `dsp_primitive` (DSP48E1 vs DSP58) is `Derived`: you *cannot*
instantiate DSP58 silicon on a 7-series — that is physics, one answer, no template
choice. But even this is context-relative: if a future part offered *both* DSP58 and
DSP48 modes selectably, `dsp_primitive` would graduate from `Derived` to a pool axis.
(This corrects §1.2.1's earlier phrasing that `$VERSION$` is "implementation identity
gated by a device predicate" — it is a `Derived` silicon consequence, not a pool axis,
*today*.)

**Sub-option vs pool member.** A resource/tuning knob *within* one design is an
`Axis` guarded by `implementation`, not a pool member. HLS `resType ∈ {lut, dsp}` is a
real user lever that stays inside `mvau_hls` (a guarded sub-axis). RTL has no
LUT-based MVU today (the `LUT6_2` in `mvu.sv:251` is a primitive *inside* the DSP
design, not a separate design), so no LUT pool member exists on the RTL side — a
future LUT-based RTL MVU would enter as a *new pool member* with zero model change.
The test: distinct buildable design → pool member; knob within a design → guarded sub-axis.

### 1.3 Derived — computed, never chosen (validates #3)

```
Derived:
    name        : str
    compute(point, context) -> value    # reads axes + context (incl. initializer data)
```

A `Derived` is **not** an axis and never enters the search space. `accDataType`,
`WMEM`, `TMEM`, stream widths, `$VERSION$`, `$SEGMENTLEN$`, `narrow_weights`,
`language` and DSP-microarch (properties of `implementation`) are all `Derived`
(map §4). `compute` may read `context.initializer(...)` — `accDataType` uses actual
weight values unless weights are external/dynamic (base:482-498). The model must
never enumerate a derived value as a dimension.

**Datatypes are the canonical `Derived`, and Brainsmith already models them —
reuse it, do not reinvent (audit 2026-07-15).** A datatype is *never a free axis*.
Brainsmith's **`DatatypeSpec` union** (`types.py:139`) is exactly this `Derived`
mechanism specialized to dtypes — a dtype is one of:

- `None` — take from the ONNX graph (a *given*, i.e. `Context`, not a choice)
- a fixed `DataType` — pinned by the op
- `str` (an interface name) — derive by copying another interface's dtype
- `VALUE_OPTIMIZED` — narrow from the actual tensor **values** (data-dependent derive)
- a `Callable(point, context) -> DataType` — arbitrary computed derivation

MVAU's dtype derivations map directly: `weightDataType` narrowing = `VALUE_OPTIMIZED`;
`outputDataType = accDataType when noActivation` = a guarded `str`-derive;
**`accDataType`** = a range-builder derivation. Brainsmith's `_binary_op_datatype`
(`spec_helpers.py:470`) already generalizes MVAU's `minimize_accumulator_width`: it
computes an output range from operand bounds using **worst-case type bounds when an
operand is dynamic, actual values when static** — *exactly* base:482-498. The
accumulator lives in Brainsmith's `internal_datatypes` slot (`schemas.py:351`), a
first-class home for derived internal dtypes.

**Consequence:** the model's `Derived` primitive **is** Brainsmith's resolver
mechanism (`DatatypeSpec` for dtypes, the same `compute(point, context)` shape for
everything else). There is no separate "declared vs realized dtype" split — there is
the *given* input dtype (`Context`) and *derived* dtypes (`DatatypeSpec`). Choosing a
*derivation policy* (VALUE_OPTIMIZED vs a fixed width) is the schema author's choice,
not a runtime axis.

### 1.4 Predicate — the one legality kind (validates the phase-collapse)

```
Predicate:
    check(point, context) -> reason: str | None    # None = legal
    describe() -> str
```

**One kind.** No `evaluation_phase`. A predicate reads whatever it needs from
`point` + `context`; what it *happens* to read (config only / +device / +toolchain
/ +data) is **provenance the evaluator can inspect**, not a category the author
declares. Examples, all one shape:

- `MH % PE == 0` — reads point (base:351)
- `pumpedCompute ⇒ SIMD ≠ 1` — reads point (rtl:334)
- `ram_style==ultra ∧ ¬is_versal(ctx.fpgapart) ⇒ runtime_writeable==1` — reads
  point **and** context.fpgapart (hls:147) — the case that killed three phases
- `SEGMENTLEN feasible only if ctx.clk > 0.741ns` — reads context.toolchain (rtl:242)
- narrow-weight packing needs DSP58 ∧ actual weights narrow — reads context.fpgapart
  **and** context.initializer (rtl:281)

The `reason` string powers explain-style diagnostics ("why is this point illegal").

---

## 2. How the space is walked (build + configure, reframed)

The two-phase build/configure split **survives as a performance structure, not a
semantic one** — exactly the "tiering is an engine concern" conclusion:

```
resolve(schema, context, assignment) -> Point | Illegal(reasons):
    point = {}
    for axis in schema.axes in dependency order:
        if not axis.exists(point):        # guard
            continue                       # axis absent — correct, not defaulted
        dom = axis.domain(point, context)
        val = assignment.get(axis.name, axis.default(point, context))
        if val not in dom:  return Illegal([f"{axis} = {val} not in {dom}"])
        point[axis.name] = val
    for d in schema.derived:              # derived computed after axes fixed
        point[d.name] = d.compute(point, context)
    reasons = [p.check(point, context) for p in schema.predicates if p.check(...)]
    if reasons: return Illegal(reasons)
    return point
```

- **Dependency order** is a topological sort over "axis A's guard/domain reads axis
  B." (MVAU: `ram_style` after `mem_mode`; `SEGMENTLEN`-domain after `SIMD`.) A
  cycle is a schema authoring error, caught once.
- **Tiered evaluation (optional optimization):** predicates that read only `point`
  (config) run before those reading `context.fpgapart` before those reading
  `context.initializer`; short-circuit on first reason. This *reconstructs* the old
  structural→optimization→realization ordering as a pure caching strategy the
  evaluator infers from provenance — never declared.
- **`build` vs `configure` collapse:** what used to be "build the space once, derive
  design points many times" becomes "resolve is pure over (schema, context,
  assignment)"; memoize the context-only prefix if it pays. No separate constraint
  buckets, no validated-and-discarded set.

---

## 3. Selection reframed — realization phase dissolves

Selection is no longer a distinct feasibility phase. `implementation` is an axis; a
realization-specific device rule is just a `Predicate` that reads
`point.implementation` + `context.fpgapart`. So:

```
select(schema, context, assignment, cost_fn?) -> Point:
    # enumerate legal points over the implementation axis (and any unfixed axes)
    legal = [resolve(schema, context, a) for a in expand(assignment)] filtered to Point
    if not legal: raise NoFeasible(collected reasons)   # explain-style
    return cost_fn(legal) or lowest-priority-first(legal)
```

- The old `realization_constraints` / `realizability` / `RealizationValidationContext`
  **fold away**: a device rule is a `Predicate`; `Context` already carries
  `fpgapart`. `_UramRequiresUltraScale` becomes a plain predicate reading
  `point.ram_style` + `context.fpgapart`. "Requires DSP58" on the systolic
  realization is a predicate reading `point.implementation` + `context.fpgapart`.
- Feasibility ⊥ preference **survives**: `resolve` decides legal; `cost_fn`/priority
  ranks. Preference never enters a predicate (the map's drift-bug lesson holds).
- "Implementation as guarding axis" gives **partial exploration for free**: leave
  `implementation` and `fpgapart` unfixed → explore the device-agnostic sub-space;
  fix them → the masked sub-space. No separate composed-vs-full-space machinery — it
  is one space, optionally sliced.

---

## 4. Validation against MVAU's four stress-tests

| MVAU stress-test (map §6) | Model construct that carries it |
|---|---|
| **#1 conditional axis existence** (ram_style exists iff decoupled; absence raises) | `Axis.exists(point)` guard; absent axis is not in the point |
| **#2 device-dependent feasibility** (URAM+Ultrascale; DSP microarch) | `Predicate.check(point, context)` reads `context.fpgapart`; `$VERSION$` is `Derived` from device |
| **#3 derived, data-dependent** (accDataType from weight values) | `Derived.compute(point, context)` reads `context.initializer(...)` — for dtypes, Brainsmith's `DatatypeSpec`/`_binary_op_datatype` (the accumulator-range builder that IS minimize_accumulator_width) |
| **#4 weight-delivery third axis** (mem_mode × … owned by neither backend) | delivery axes are ordinary guarded axes at the op level; the delivery *component* (SIP/memstream) is a composed sub-kernel — **still deferred** (§5) |

The unified `check(point, context)` predicate absorbs **every** entry in the map's
feasibility catalogue (§3) — config-only, device, toolchain, and the combination
cases — with no phase distinction. That is the phase-collapse, validated.

---

## 5. Scope boundaries (what this spec does and does not do)

**In scope (design now):** the four primitives, dependent-space `resolve`, the
single predicate, selection-as-enumeration, backend-as-guarding-axis.

**Deferred, unchanged from `roles-and-interfaces.md` §7:**
- **The weight-delivery / memory component (MVAU stress-test #4).** The *axes*
  (`mem_mode`, `ram_style`, `pumpedMemory`, …) are ordinary guarded axes this model
  holds fine. The *composed sub-kernel* that realizes decoupled delivery (memstream
  as its own kernel with its own space + couplings) needs the MLO analysis and is
  NOT designed here. The model must not *preclude* it — a sub-kernel is a `Derived`
  component whose own sub-space composes — but the composition mechanism is future.
- **SIP / composite Implementation** — as before; possibly dissolved by decomposing
  MVAU rather than blessing composition.
- **The search/exploration algorithm** (`explore`) — interface only: it consumes a
  schema + context and walks the dependent space; strategy open.
- **The cost/preference model** behind ranking — open seam.

**Explicitly NOT deferred (this is the point of the pass):** the guarded-axis space,
the derived-quantity category, and the one-predicate collapse — the three things
MVAU proves the current model lacks.

---

## 6. Migration from the current code (net simplification)

The refactor this implies is *smaller* than what exists — it removes machinery:

- `KernelSchema.dse_parameters` (fixed-domain dict) → axes with `exists`/`domain`.
  The `_compute_dimension_ranges` divisor logic becomes an `Axis.domain` impl.
- `Constraint.evaluation_phase` + the 3-way builder split → **removed**; one
  predicate kind, provenance inferred.
- `RealizationValidationContext` + `realization_constraints` + `realizability` →
  **removed**; device rules are predicates over the one `Context`.
- `_composed_schema()` (merge backend params at selection) → **removed**;
  `implementation` is an axis, its guarded axes are always in the schema.
- `backend ∈ {hls, rtl}` framing → **removed**; one `implementation` axis over the
  realization pool, with `language`/microarch/`resType` as `Derived` properties.
- Survives: `OrderedParameter` (an ordered `Domain`), Brainsmith's `DatatypeSpec` /
  `internal_datatypes` / range-builders (they ARE the `Derived` mechanism for
  dtypes), the golden-reference/identity split, `emit` hermeticity,
  feasibility ⊥ preference, by-name resolution.

## 7. Acceptance criteria for the refactor that follows

1. An op declares a schema of `Axis` (with `exists`/`domain`), `Derived`, and
   `Predicate` — no `evaluation_phase`, no `dse_parameters` dict, no
   realization-specific context.
2. `resolve(schema, context, assignment)` walks the dependent space, skips
   guarded-out axes, computes derived, returns `Point | Illegal(reasons)`.
3. Thresholding re-expressed in the new model: `PE` (domain=divisors), `ram_style`
   (guarded by backend=hls), `depth_trigger_*` (guarded by backend=rtl), the URAM
   rule as a `Predicate` reading `context.fpgapart`. All current tests pass.
4. A *guarded-axis* test: an axis absent under one backend, present under another,
   proven by `resolve`; querying an absent axis errors.
5. A *data-dependent derived* test: a `Derived` reading `context.initializer`.
6. The unified predicate surfaces the combination case (config+device) that the
   three-phase model could not file.

---

## 8. Worked example — the MVAU decision tree

The concrete acid test for §1's primitives. Every node is exactly one of:
**◆ CHOICE** (an axis with a domain), **▸ FORCED** (a `Derived`, no choice), or
**✗ GATE** (a `Predicate`, feasibility over point+context). Walked in dependency
order. `PE`/`SIMD` shown as single nodes — they are the real `PE×SIMD` combinatorial
mass, identical under every implementation.

**Confirmed (2026-07-15): multiple distinct MVAU implementations are in the
pipeline.** The `implementation` pool is genuinely rich; the device-prunes-a-real-
menu structure is load-bearing, not hypothetical. Microarchitectures (systolic /
GEMM / LUT / AIE) are **distinct `implementation` variants** (different source-set,
codegen path, cost profile — a designer chooses and cost-compares them). The
silicon-primitive generation *within* a DSP-family impl (DSP48E1/E2/DSP58) stays
**▸ FORCED** from `fpgapart` — the part physically has one; it is not a choice.
This is the forced⇒derived discipline that prevents fake combinatorial width.

```
CONTEXT (givens — not decisions)
  MW, MH ............... weight tensor shape           (base:63-64)
  inputDataType ....... ONNX graph                     (base:232)
  weight values ....... initializer(input[1])          (base:482)
  fpgapart, clk ....... build-config globals

◆ implementation                                    ← THE ROOT CHOICE
  domain = { mvau_hls, mvau_dsp_softvec, mvau_dsp_packed }   (open pool; +mvau_aie etc. later)
           one member per distinct compute SOURCE TEMPLATE (§1.2.2); menu, NOT a product
  ✗ device+dtype prune the pool (they narrow, they don't define — §1.2.2):
      mvau_dsp_packed  requires DSP58 ∧ w≤8 ∧ a≤9 ∧ lanes≤3   (mvu_vvu_8sx9_dsp58.sv, rtl:313)
      mvau_dsp_softvec requires a DSP part                     (mvu.sv, any VERSION)
      mvau_hls         requires nothing
  ⓟ preference: mvau_dsp_packed > mvau_dsp_softvec when both feasible
     (FINN's buried `generate if`, made an explicit ranking — rtl:313)
  │
  ├─ branch mvau_hls
  │    ◆ resType   domain = {lut, dsp}     (real HLS user lever, guarded sub-axis; hls:58)
  │    ▸ language = "hls"
  │    ✗ SIMD ≥ MW/1024   (HLS array-partition synth limit, hls:216)
  │
  ├─ branch mvau_dsp_softvec   (mvu.sv — VERSION-parameterized, any DSP part)
  │    ◆ resType   domain = {dsp}          (RTL lut path not exposed, rtl:256)
  │    ▸ language = "rtl"
  │    ▸ dsp_primitive = f(fpgapart) → DSP48E1|DSP48E2|DSP58   FORCED (silicon, §1.2.2)
  │    ◆ pumpedCompute ∈ {0,1}             (clk2x DSP; exists only on DSP RTL cores; rtl:53)
  │    ▸ SEGMENTLEN   = f(SIMD, clk, pumpedCompute)   (rtl:230)
  │    ▸ narrow_weights = f(weight values)            (rtl:281)
  │    ✗ SEGMENTLEN feasible only if clk > 0.741ns    (toolchain, rtl:242)
  │    ✗ pumpedCompute ⇒ SIMD ≠ 1                     (rtl:334)
  │
  └─ branch mvau_dsp_packed    (mvu_vvu_8sx9_dsp58.sv — DSP58 INT8-packed)
       ◆ resType {dsp} ▸ language=rtl ▸ dsp_primitive=DSP58 (forced)
       shares the pumpedCompute/SEGMENTLEN sub-structure of the DSP RTL cores

SHARED AXES (exist under every implementation — not guarded by it)
  ◆ PE     domain = divisors(MH)        (base:351)  ┐ the real
  ◆ SIMD   domain = divisors(MW)        (base:352)  ┘ PE×SIMD product
  ✗ MH % PE == 0,  MW % SIMD == 0                   (config)

  ◆ noActivation ∈ {0,1}                            ← guards threshold cluster
    ├─ =0  ◆ ActVal ∈ int
    │      ◆ ram_style_thresholds ∈ {auto,block,distributed}
    │      ▸ TMEM = MH/PE
    │      ✗ bipolar×bipolar ⇒ nonneg int thresholds   (config+data, base:576)
    └─ =1  ▸ outputDataType = accDataType              (base:517)

  ◆ numInputVectors ∈ list[int]

  ◆ mem_mode ∈ {internal_embedded, internal_decoupled, external}  ← guards delivery cluster
    ├─ internal_embedded   ▸ weight instream width = 0 (no port, no ram_style)
    ├─ internal_decoupled
    │    ◆ ram_style ∈ {auto,block,distributed,ultra}
    │    ◆ runtime_writeable_weights ∈ {0,1}
    │    ◆ pumpedMemory ∈ {0,1}
    │    ◆ dynamic_input ∈ {0,1}
    │    ▸ memstream sub-kernel EXISTS   (deferred composed component — §5)
    │    ✗ ram_style=ultra ∧ ¬is_versal(fpgapart) ⇒ runtime_writeable=1
    │         ← THE combination gate: reads point AND device in one (hls:147)
    │    ✗ pumpedMemory ⇒ ¬(PE==SIMD==1)   (config, base:717)
    └─ external            ◆ dynamic_input relevant

FORCED everywhere (Derived — computed after axes fixed, never dimensions)
  ▸ WMEM = MW·MH/(PE·SIMD)
  ▸ accDataType = range-builder(inputDT, weightDT, MW)   (DatatypeSpec / internal_datatypes;
       data-dependent: actual weight values unless external/dynamic → worst-case, base:482-498)
  ▸ weightDataType(realized) = VALUE_OPTIMIZED narrow    (base:529)
  ▸ stream widths = f(PE, SIMD, dtypes)
  ▸ language, dsp_primitive = properties of implementation / device

GATES reading data/config (on the assembled point)
  ✗ weight initializer must exist unless mem_mode=external ∨ dynamic_input  (base:782)
  ✗ true-binary non-bipolar unsupported   (dtype config, hls:167)
```

**What the tree proves about §1:**
- The **root is one `◆ implementation` axis** over an open pool; the device is a
  **✗ pruning predicate**, not a forced selector. Adding a variant is additive to a
  menu, not multiplicative — no combinatorial explosion from a rich pool.
- **`dsp_primitive` is ▸ FORCED inside the DSP branch** — the discipline that keeps
  DSP48-vs-DSP58 out of the axis set (silicon forces it; a fake axis would inflate
  the space with illegal points). Contrast: *microarch* (systolic vs GEMM) IS a
  choice → an implementation variant.
- The single **✗ combination gate** (`ram_style=ultra ∧ ¬is_versal ⇒ …`) reads
  point AND `context.fpgapart` in one condition — the case that dissolves the
  three-phase model, sitting naturally in the decoupled branch with no phase tag.
- **Guards compress**: each `mem_mode` branch opens a *disjoint* sub-set (a sum
  across 3 branches, not a product of all their axes). The dependent space is
  strictly smaller than the naive cartesian product.
- The genuine combinatorial mass is **`◆ PE × ◆ SIMD`** — intrinsic to the hardware,
  identical under every implementation, and the `explore` search's problem to manage
  (deferred, open — §5). Space *size* is a hardware fact; the model's job is faithful
  compact representation (guards) — not shrinking the space.
