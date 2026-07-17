# Handoff: Build the Generic `resolve` Engine

You are building a generic design-space **`resolve` engine** and proving it against
MVAU as an executable fixture. The design is fully specified in
`kernel-final-design/design-space-model.md`; this document tells you what to build,
where, and how you'll know it's done.

Working root: `/home/tkeller/prj-kernels/`. All paths from there.

---

## 0. Goal

Represent a hardware-kernel **design space** as data and resolve it:

```
resolve(schema, context, assignment) -> Point | Illegal(reasons)
```

where a schema is **Axes** (guarded choices) + **Derived** (computed quantities) +
**Predicates** (legality), resolved against a **Context** (givens). Prove it by
running the **MVAU decision tree** (`design-space-model.md` §8) as a fixture.

---

## 1. Where you build

Everything lands in the git-tracked package **`finn/src/finn/design_space/`**.

```
finn/src/finn/design_space/
  primitives/     ← ALREADY BUILT — reuse these, import them, don't re-derive them
    ordered_parameter.py    OrderedParameter                (the ordered Domain type)
    types.py                ShapeHierarchy, prod, TilingSpec, FULL_DIM,
                            FULL_SHAPE, VALUE_OPTIMIZED, DatatypeSpec
    spec_helpers.py         derive_dim, derive_datatype, value_optimized_datatype,
                            compute_{add,sub,mul,min,max}_range,
                            smallest_datatype_for_range, {add,sub,mul,…}_datatype
    template_resolution.py  resolve_template, normalize_template
    interface.py            InterfaceDesignSpace, InterfaceDesignPoint
                            (tensor/block/stream shapes, folding factors,
                             stream_width_bits)

  space/          ← YOU BUILD THIS
    context.py      Context
    axis.py         Axis, Domain
    derived.py      Derived
    predicate.py    Predicate
    resolve.py      resolve()

  fixtures/       ← YOU BUILD THIS
    mvau.py         the MVAU decision tree as data

  tests/          ← YOU BUILD THIS
```

The primitives are proven value objects (shape/width math, datatype range-builders,
ordered-parameter navigation). Build `space/` to consume them. Import path is
`finn.design_space.primitives.<module>`.

---

## 2. What to build (`space/`)

Full definitions, rationale, and the three disciplines are in
`design-space-model.md` §1. In brief:

```
Context     tensor_shape(name), tensor_datatype(name), initializer(name),
            fpgapart, toolchain_version.  Everything known at build time.

Axis        name; exists(point)->bool (the guard); domain(point,context)->Domain;
            default(point,context)->value.
            Domain is an OrderedParameter (ordered) or a frozenset (discrete).

Derived     name; compute(point,context)->value. Reads axes + context +
            initializer data. Datatypes use the primitives' DatatypeSpec /
            *_datatype range-builders directly.

Predicate   check(point,context)->reason:str|None (None == legal); describe()->str.
            One kind. What it reads is provenance to inspect, not a declared phase.

resolve(schema, context, assignment) -> Point | Illegal(reasons):
    walk axes in dependency order;
    skip an axis whose guard is false (it is ABSENT — reading it errors, not defaults);
    take value from assignment or default; reject if outside domain;
    compute Derived after axes are fixed;
    run Predicates on the assembled point; collect reasons.
```

Three disciplines the design turns on (`design-space-model.md` §2, §1.2.1, §1.3):
1. **Forced ⇒ Derived, choice ⇒ Axis.** Anything a device or the data forces
   (DSP48-vs-DSP58 from `fpgapart`; accumulator width from weight values) is
   `Derived`. Only a genuine alternative a designer cost-compares is an axis value.
2. **`implementation` is one Axis over a realization pool.** Language, microarch,
   primary resType are `Derived` properties of the chosen realization. The device is
   a **pruning Predicate** over the pool.
3. **Guards compress.** A guarded-out axis is absent from the point; the dependent
   space is a strict subset of the naive cartesian product.

---

## 3. What to build (`fixtures/mvau.py`)

Encode the MVAU decision tree (`design-space-model.md` §8) as a schema of `Axis` /
`Derived` / `Predicate` instances. **Write this before the engine** — it is the
target the engine must satisfy, and writing it first forces the primitive shapes to
fit the real case. Source of truth for every axis/dependency/predicate with
file:line is `kernel-final-design/mvau-design-space.md`.

---

## 4. Done when — the MVAU fixture proves (acceptance)

1. A guarded axis (`ram_style`) is **absent** when its guard is false
   (`mem_mode ≠ internal_decoupled`); reading it **errors**.
2. The device **prunes the implementation pool**: on a 7-series part a DSP58-only
   realization resolves to `Illegal` with a reason; other realizations remain.
3. A forced-derived value (`dsp_primitive` from `fpgapart`; `accDataType` from weight
   values) is **computed**, never enumerated as an axis.
4. The **combination predicate** fires:
   `ram_style=ultra ∧ ¬is_versal(fpgapart) ⇒ runtime_writeable=1` — reads point AND
   context in one `check`.
5. `resolve` on a fully-specified legal assignment returns a `Point` carrying all
   derived quantities; on an illegal one returns `Illegal([reasons])`.
6. Enumerating a small MVAU slice yields the **dependent-space count**, not the naive
   cartesian-product count (guards compress).

---

## 5. Environment

Venv at `prj-kernels/.kernel-venv/` (numpy, onnx, qonnx-from-source; the engine is
pure Python + data — no Docker). Run:

```
PYTHONPATH="qonnx/src:finn/src" .kernel-venv/bin/python -m pytest \
  finn/src/finn/design_space/tests/ -q
```

The primitives already import and pass a shape-math smoke test under this recipe.

---

## 6. Reading order

1. `kernel-final-design/design-space-model.md` — **the spec.** Four primitives,
   `resolve`, the three disciplines, and §8 the worked MVAU tree (your fixture target
   and acceptance test).
2. `kernel-final-design/mvau-design-space.md` — MVAU's full space as data
   (every axis/dependency/predicate, file:line). The fixture's source of truth.
3. `finn/src/finn/design_space/primitives/` — the value objects you build on. Read
   `interface.py` (shape/width) and `spec_helpers.py` (datatype derivation) closely;
   they are the Derived layer's building blocks.

Real MVAU code, if you need to check a detail:
`finn/src/finn/custom_op/fpgadataflow/{matrixvectoractivation.py,hls/,rtl/}` and
`finn-rtllib/mvu/` (`mvu.sv` `case(VERSION)` — the DSP-primitive selection that
stays Derived; the source-sets that are distinct implementations).
