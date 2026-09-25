# Kernel composition pass

This pass uses two real consumers to test Space/Subspace authoring: typed
stream interfaces over scalar contracts (experiment 1), and MVAU weight delivery
as a heterogeneous structural choice (experiment 2). Both experiments are
integrated in production kernels. **The generic engine (`src/finn/core/space`)
is unchanged.** Every behavior below uses existing Space APIs.

The base is `b2d01750a` on `feature/kernel-package-extraction`. FinnLib is
unchanged at `b17eae6a`, and its untracked `input_gen_lift*` files are untouched.
Revisions are recorded in [BASELINE.json](BASELINE.json), and the full commands
and results in [VALIDATION.txt](VALIDATION.txt).

## Experiment 1: scalars and typed ports

### What changed

| Before | After |
|---|---|
| `Scalar(dtype, policy)`: a `Subspace` subclass that runs a `ScopeBuilder`, calls `policy.rebind()` to create local bound Params, then injects `policy.constraints()` | Handwritten `Scalar` → `IntegerScalar` → `BoundedIntegerScalar` Spaces. `integer_scalar(dtype, Integer(...))` is a 20-line factory that returns an ordinary placement |
| `AxiStreamInterface`: builder-generated scope that owns its dtype Param and injects the policy constraints into the stream scope | `AxiStreamPort(TypedStream)`: a handwritten Space bound to a **separately owned** scalar (raw `dtype` and accepted `encoding`). `axi_stream(...)` is the factory |
| No typed native port; eltwise constructed `ReadyValidStream`s by hand and repeated the operand check | `NativeStreamPort(TypedStream)` / `native_stream(...)`; eltwise operands are `EltwiseOperand(Scalar)` |
| `Integer.constraints()`, `Integer.rebind()`, `DatatypeDomain`, `type_constraints` | Removed. The shared predicates are public (`check_integer_family`, `check_bit_bound`). `check()` and `domain()` are unchanged |
| `self.input.view(IntToFp32Kernel.input.view())()` | `self.input.encoding()` |
| `DotpAxiKernel.activation.dtype`, nested `bindings={...}` in MVAU | `DotpAxiKernel.activation_dtype`, bound flat: `activation_dtype=activation_dtype` |

Dotp now declares its operands as follows:

```python
activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
activation_type = integer_scalar(activation_dtype, Integer(min_bits=2))
activation = axi_stream("s_axis_input", simd, Endpoint.TARGET, activation_type, last=True)
```

Before, this was eight lines of `AxiStreamInterface(...)` per port, including a
fresh `dtype=Param(...)` and an `error_code`. Opaque word streams (FIFO, input
generator, replay, cyclic) still use detached `ReadyValidStream` values and get
no scalar contract.

### How the alternatives compare

I compared three forms against the handoff's criteria:

- **Builder**: the previous ScopeBuilder form, in which the port contains its
  admission.
- **Contained**: a handwritten port that owns a `Scalar` child.
- **Bound** (adopted): a handwritten port bound to a separately owned scalar.

[compare_port_forms.py](compare_port_forms.py) runs the contained and bound
forms on identical facts. Both give identical rule-by-rule results, including a
family refusal with an unresolved dynamic bound.

| Criterion | Builder (before) | Contained, handwritten | Bound, handwritten (adopted) |
|---|---|---|---|
| One owner per fact; explicit sharing | dtype owned by the port; the kernel reaches it through a nested ref | dtype forwarded into the port and then into its child | dtype is a kernel Param; scalar and port bind to it |
| Independently inspectable dtype, width, packing and interface facts | yes | yes | yes: raw facts come from `dtype`, and only `stream` needs acceptance |
| Accepted child product consumed where required | builder view reads its own injected constraints | port reads `self.element.encoding()` | port binds `scalar.accepted(Scalar.encoding)`; `stream` cannot exist without it |
| Static typing of child and view access | `Subspace[AxiStreamScope]` plus a `.view()` handle; `self.x.view(K.x.view())()` | typed | typed: `self.activation.stream()` → `AxiStream`, `self.input.encoding()` → `ScalarEncoding` |
| Known refusal visible while a bound is unresolved | yes (per-rule constraints) | yes | yes (tests `test_known_family_refusal_*`, `test_integer_admission_retains_*`) |
| Plumbing | builder, bindings, exports, forwarding properties, and a rebind protocol on the policy | one port subclass per policy shape, and each dynamic bound forwarded as another Param | two short declarations per operand; no per-policy classes |

The contained form works. Subclasses override the `element` placement by name,
but every policy shape then needs a port subclass. The bound form keeps
interface structure independent of admission-policy structure. That
independence is the actual reduction.

### Findings

- Ordinary Spaces were sufficient. The structure varies only between an unbounded
  and a bounded integer, and subclassing handles that. `BoundedIntegerScalar`
  redefines `admission = ConstraintGroup(...)`, and the inherited `encoding` view
  resolves the override by name. ScopeBuilder is no longer used by any kernel. It
  remains appropriate when a template's structure is computed.
- A policy object is useful as plain data, but not as a scope-aware object. A
  referenced bound in `Integer(1, limit)` becomes a normal child binding. The
  rebind protocol duplicated what bindings already do.
- Moving operand dtypes to kernel Params reverses an earlier dotp test assertion.
  A dtype is an operand fact, admission belongs to the scalar, and transport
  belongs to the port.
- Cost: separate scalar scopes enlarge the prepared model. Dotp grows from 69 to
  90 authored declarations and from 4 to 7 scopes. Eltwise grows from 11 to 82
  declarations because it now models typed operands and ports. One
  bind+choose+`build_requirements()` of dotp takes 14.2 ms instead of 12.9 ms
  (300 iterations, two runs). Kernel infrastructure goes from 669 to 618 lines,
  including the new native port.

## Experiment 2: MVAU weight delivery as a structural choice

### Structure

```text
MVAU
  facts: geometry, dtypes, target, segment_length, weights (optional)
  decisions: pe, simd
  traversal (derived, folding refusal)   compute = DotpAxiKernel (owns compute_pumping)
  implementation = SubspaceChoice
      external: ExternalWeights(traversal, dtypes, compute=compute.accepted(build_requirements))
      cyclic:   CyclicWeights(..., weights=weights)
                  rom_style = Decision(auto | distributed | block)
                  image (derived) -> weight_source = cyclic_stream(image, rom_style)
      exports: ASSEMBLY_VIEW (MVAUAssembly), BUILD_VIEW (ModuleBuildRequirements)
  assembly = View(implementation.accepted(ASSEMBLY_VIEW), constraints=(dimensions,))
  build_requirements = View(implementation.accepted(BUILD_VIEW), constraints=(dimensions,))
```

The families are genuinely different. External delivery has an `in1_V` top-level
stream, three instances and no initializer. Cyclic delivery has no weight port,
a `u_weights` ROM instance, an embedded image and the `rom_style` choice.
Asserting the common export type does not make their ABIs interchangeable; the
tests assert the differing port sets. The dimension constraint group is now
attached to the accepted `assembly` and `build_requirements` views. Previously
`assemble()` was procedural.

**The case-local choice is real.** It is not an invented knob. The ROM now
takes `ROM_STYLE` and passes `distributed` or `block` to synthesis as its
`rom_style` attribute; `auto` leaves inference to the tool. Out-of-context
synthesis of `cyclic_stream` (Vivado 2025.2, xczu3eg-sbva484-1-e, 16-bit words)
gives:

| Depth | auto | distributed | block |
|---|---|---|---|
| 32 | 12 LUT, 0 RAMB18 | 12 LUT, 0 RAMB18 | 4 LUT, 1 RAMB18 |
| 256 | 15 LUT, 1 RAMB18 | 73 LUT, 0 RAMB18 | 15 LUT, 1 RAMB18 |

UltraRAM is deliberately not offered, because bitstream initialization of
UltraRAM is not available on every supported device family. External delivery
has no comparable local choice. I did not add one to make the example symmetric.

### Behavior demonstrated (`tests/kernels/test_mvau_delivery_choice.py`)

- **Laziness.** With external delivery selected, `inspection.explain` of
  `assembly` visits no `implementation.cyclic.*` node and does not read
  `weights`. The inactive family's members are `Inapplicable`.
- **Ownership and applicability.** `implementation.cyclic.rom_style` is owned by
  the cyclic scope. Until the selector is committed, its candidates are
  `Unresolved`, with the blocker owned by `implementation`.
- **Optional facts.** With cyclic delivery selected but no `weights` supplied,
  `image` and `assembly` are `Unresolved`, and the evidence reports the omitted
  `weights` input. The compute product and folding still settle, and external
  delivery never needs weights. No defaults or callback-input modes were
  introduced. Bad values or shapes are refused with `mvau-weights`, owned by
  `implementation.cyclic.image`.
- **Persistence.** Capture contains `compute.compute_pumping`, `implementation`,
  `implementation.cyclic.rom_style`, `pe` and `simd`. The selection encodes and
  decodes through a `SelectionSchema` and restores on an empty root to an
  identical assembly. Replay with different weights gives a new image. Replay
  without weights restores the choices, but the assembly stays unresolved. Replay
  under facts that invalidate `pe` is refused atomically. A configured receiver
  is rejected.
- **Atomic switching.** Changing the selector from cyclic to external while
  `rom_style` is committed is refused, and the receiver is returned unchanged.
  The refusal reports `implementation.cyclic.rom_style` as `Inapplicable`.
  Clearing the choice in the same batch succeeds. Switching back can commit a
  new `rom_style` in the same batch. Committing `rom_style` while its family is
  unselected is refused.
- **Adapter parity.** `mvau_assembly(...)` equals the Space path for both modes
  and keeps its errors. It gains `rom_style="auto"`, which reproduces the old
  ROM's inference behavior.

### Initializer facts and artifacts

The weight matrix is an immutable, optional configuration fact. Only the cyclic
family consumes it, through its own required Param. The packed image is a
derived value, and the requirements embed it in `INIT_DATA`. As a result:

- the prepared build has no data slots;
- Space evaluation performs no file I/O;
- a different image changes the concrete build identity.

`MVAUAssembly.initializer` exposes the same image to callers.

## What the experiments taught us

1. **Handwritten Spaces plus small factories replace the builder for kernel
   interfaces.** Binding a port to a separately owned scalar is simpler than
   nesting the scalar inside the port. Structural variation belongs in
   subclasses, and override-by-name of constraint groups makes that clean.
2. **Policies are values, not scope participants.** Bindings, not a policy
   protocol, place their references.
3. **`SubspaceChoice` needed no new API for a real heterogeneous consumer.**
   Typed exports, lazy cases, case-owned decisions, an optional fact bound into
   one case, selector persistence and explicit stale-choice clearing all work as
   documented.
4. **Candidate simplifications (not implemented; a second consumer should decide):**
   - The selector handle is reachable only through `inspection.choices(point)`.
     Atomic switching therefore needs a metadata lookup, and `select(case)` is a
     separate publication. A bound selector on `ChoiceView` would make atomic
     switching ordinary.
   - `alternative(case)` returns an untyped `Space`. Case members stay typed
     through their references, as in `family.field(CyclicWeights.rom_style)`.
   - The five bindings shared by both families are written twice. Splatting a
     dict of references is rejected by strict typing.
5. **Accepted child products compose across siblings.**
   `compute.accepted(DotpAxiKernel.build_requirements)` bound into each family
   means a family cannot wire a refused compute core. The refusal propagates to
   the parent's `assembly` view. `assemble()` previously enforced this with
   explicit exception handling.

## Remaining limitations

- The selector ergonomics above are recorded, not changed.
- External delivery has no local choices. Weight-stream buffering or FIFOs
  belong to graph-level insertion, not to this family.
- Typed ports carry no markers, because no typed consumer needs them. Marker
  streams (replay, input generator) stay opaque.
- The scratchpad `space/MVAU-EXAMPLE.md` and its script are pinned to
  `b2d01750a`, which still has `MVAU.weight_delivery`. They are valid for that
  revision, not for this one. Against this revision the script fails at
  preparation with `kernel: missing child parameter bindings ['weights']`,
  because a parent that places `MVAU` must bind every child formal, including
  optional ones. The fix is to bind `weights` to a literal, a parent fact, or a
  fresh optional `Param(INTEGER_MATRIX, required=False)`. The example is left
  pinned rather than edited.
- ROM-style evidence is component-level out-of-context synthesis. The full MVAU
  was simulated with XSI (see VALIDATION.txt), not synthesized or placed.
- DataflowOp, Regions, graph transformations, scheduling, the dotp architecture,
  thresholding arithmetic and HLS remain parked, as the handoff required.
