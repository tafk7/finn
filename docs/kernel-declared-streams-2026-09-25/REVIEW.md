# Declared streams and the FIFO slot

This increment makes connections first-class Space declarations. It builds on
the [stream contract](../kernel-stream-contract-2026-09-25/REVIEW.md) and its
traversal revision. The generic Space engine and FinnLib are unchanged.
Validation is in [VALIDATION.txt](VALIDATION.txt).

## Model

```text
class MVAU(Space):
    activations   = Stream(activation_spec)
    replayed      = Stream(replayed_spec)
    weight_stream = Stream(weight_spec, buffered=True)
    results       = Stream(result_spec)

    source  = Subspace(TopInput, name="in0_V", output_stream=activations.spec)
    replay  = Subspace(ReplayBuffer, input_stream=activations.spec, output_stream=replayed.spec, ...)
    compute = Subspace(DotpAxiKernel, activation_stream=replayed.spec,
                       weights_stream=weight_stream.spec, result_stream=results.spec, ...)
    implementation = SubspaceChoice({
        "external": Subspace(TopInput, name="in1_V", output_stream=weight_stream.spec),
        "cyclic":   Subspace(CyclicDelivery, ..., output_stream=weight_stream.spec)})
    sink    = Subspace(TopOutput, name="out0_V", input_stream=results.spec)
```

The pieces of `finn.kernels.streams`:

- **`StreamSpec`** is the logical sequence a stream carries per consumer pass:
  element, traversal, repetition and marker rules.
- **`Stream(spec)`** declares a connection. It is a placement of a
  `StreamLink` Space, and kernels bind to it through `stream.spec`.
- **`Port(direction)`** is a kernel's endpoint formal. It is an optional Param,
  so kernels still configure on their own.
- **Components.** Each stream-capable kernel exports a `component` view: its
  module (none for a boundary) and one `StreamContract` per port. The kernels
  that do so now are `DotpAxiKernel`, `CyclicDelivery`, the new `ReplayBuffer`,
  `TopInput` and `TopOutput`.
- **Producers derive their own contracts.** Replay computes its output
  traversal and markers from its input and parameters. Dotp checks that each
  stream's element matches its port encoding, and requires a frame-marker rule
  on its activation stream.
- **`assemble_streams(point, ...)`** runs inside a parent view:
  - it finds each stream's producer and consumer from the declared bindings;
  - it checks and wires every connection with `Composition.connect`;
  - it routes clocks and resets by role;
  - it turns boundary placements into AXIS top ports.

  It needs no engine change: an in-callback read of a choice's exported view
  resolves the selected case.
- **Choices on either side of a stream.** A stream endpoint may be a
  `SubspaceChoice`; every case must bind the same ports to the same streams.
  `ExternalWeights`, `CyclicWeights` and `WeightDeliveryFamily` are gone. The
  choice's cases are now the boundary port and the reusable delivery kernel
  themselves.

## The FIFO slot

**Enabling a FIFO.** A stream declared `buffered=True` owns a `transport`
choice between `direct` and `fifo`.

- **Owned choices.** The `fifo` case (`StreamFifo`) places a `FifoKernel` as
  `buffer`. It owns two decisions: `depth` (any native value, 2 or more) and
  the kernel's `ram_style`. Both are demanded only when `fifo` is selected.
- **Identity adapter.** The FIFO's two contracts are the stream's own spec. The
  assembler splits the connection into producer → FIFO → consumer and checks
  both halves.
- **Author control.** Unbuffered streams have no transport choice at all, so
  no selector commitment is needed. The kernel author decides where a FIFO may
  go. MVAU allows one on `weight_stream`, following baseline's decoupled weight
  path (roster V44).
- **Compiler control.** Detection and sizing belong to the compiler. A planner
  commits `transport` and `depth` with ordinary `with_choices`. Selections
  persist them and restore them on an empty root. Removing a FIFO must clear
  its stale depth and memory style atomically; this reuses the existing
  structural-choice rule.

The adapter gains `weight_fifo_depth`, and the MVAU numeric harness gains
`--weight-fifo-depth`.

## Changed keys and API

| Before | After |
|---|---|
| `implementation.cyclic.source.rom_style` | `implementation.cyclic.rom_style` |
| (none) | `weight_stream.transport`, plus `weight_stream.transport.fifo.buffer.{depth,ram_style}` |
| `ExternalWeights`, `CyclicWeights`, `WeightDeliveryFamily`, `_wire_mvau` | removed; `MVAU` declares streams |
| `MVAU._Traversal` / `traversal` | `_Folding` / `folding`, not to be confused with `forms.Traversal` |
| dotp, delivery | gain `*_stream` ports and a `component` view |

MVAU now also has a required `weight_stream.transport` decision.
`mvau_assembly` commits it to `direct` unless a FIFO depth is given.

## Limits and follow-ups

- **One producer, one consumer per stream.** The first physical profile
  forbids data fan-out. A duplicating adapter would be a kernel.
- **Padding between children.** A padded AXIS child still cannot feed another
  child through a FIFO or directly, because child padding cannot be partially
  disposed. That doesn't arise in MVAU today.
- **Ports are ordinary formals.** A parent that places a stream-capable kernel
  outside any stream must still bind its ports, because Space requires every
  child formal to be bound, including optional ones. Binding them to fresh
  optional `Param`s is enough, as `tests/kernels/test_dotp.py` does. A
  standalone root simply leaves them unsupplied.
- **Topology errors are programming errors.** Unbound ports, inconsistent case
  bindings and fan-out raise during evaluation. Contract mismatches are
  refusals (`mvau-stream`).
- **Forms are declared, not negotiated.** Stream traversals are still stated
  by the parent. Ports declaring supported traversal families, and adapter
  insertion on a stream, remain the next steps (see the previous REVIEW's
  proposal).
- **Placement discovery is class-level.** Streams and bindings are found by
  inspecting class declarations, so inherited placements are included.
  Dynamically built ScopeBuilder templates are not covered.

## Revision: Space-native streams

This revision implements the direction in [STATUS.md](../kernel-status-2026-09-25/STATUS.md)
§4. It replaces `assemble_streams` and the `MVAU.assembly` view.

- **Explicit endpoints.** `Stream(spec, source=(instance, port_view),
  sink=(instance, port_view), buffered=...)` binds its producer and consumer
  through ordinary accepted refs, so the linker owns the topology. Kernels
  publish one accepted view per port: dotp's `activation_port`, `weights_port`
  and `result_port`; replay's `input_port` and `output_port`;
  `CyclicDelivery.output`; and `TopInput.port`/`TopOutput.port`. Choice cases
  export `OUTPUT_PORT` and `MODULE`. Streams are declared after the
  placements, and kernel ports bind to the parent's spec values directly, so
  there is no declaration cycle.
- **Checks are constraints; connections are views.** `StreamLink.compatible`
  owns each stream's refusal, and it checks both halves when a FIFO is
  selected. `StreamLink.connection` is the accepted, detached `Connection`.
  `connected(stream)` gives the parent one constraint per stream, and
  `ConstraintGroup` attaches them to its views. Independent refusals are all
  visible at once, each owned by `<stream>.compatible`.
- **A thin reduction.** `compose(module, producer, instances, connections)` is
  a pure function. MVAU's `structure` view passes it the accepted instances and
  connections by name, and `build_requirements` returns its lowering.
- **`MVAUAssembly` is adapter-only.** Beat counts come from `folding`. Delivery
  follows the selected case. The initializer is `CyclicDelivery.image`.
  `mvau_assembly()` returns the same record as before.
- **Removed code.** Class-attribute discovery, `Component`/`COMPONENT`, the
  `component` views, `CyclicDelivery`'s port formal and `ReplayBuffer`'s
  output port formal are all removed.
- **Helpers.**
  - `finn.kernels.configure.configure(space, facts, choices)` commits choices
    by their inspection keys in one batch and raises readable refusals; the
    adapter and README use it.
  - `describe()` formats findings.
  - `ScalarEncoding.admit(dtype)` replaces MVAU's private encoding wrapper.

**Evidence.**

- **Unchanged build identity.** `module_build_fingerprint` is identical to
  `ee5e8852e` for six MVAU configurations: external; cyclic with a block ROM;
  a FIFO on each delivery; padded output; and pumped DSP58.
- **Independent refusals.** `tests/kernels/test_declared_streams.py` shows two
  streams refusing independently, with their own owners.
- **Per-stream evidence.** `inspection.explain` of the parent reports
  per-stream `connection` and `compatible` nodes.
- **Validation.** XSim and gate results are in VALIDATION.txt.

**Remaining limits.**

- Each stream still requires literal instance names for its endpoints. They are
  explicit and checked by the physical validator, but repeated between the
  `instances` mapping and the `Stream` declarations.
- `compose` can still raise, for example on a clock-domain conflict. That is a
  defect no per-stream constraint can see. The parent converts it to an
  `mvau-composition` refusal.
