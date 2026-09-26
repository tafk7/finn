# Stream contract and reusable cyclic delivery

This increment implements option S1 and a reusable cyclic delivery kernel from
the [kernel roster map](../../../scratchpad/open/kernel-roster-map/SUMMARY.md)
(`STREAMS.md` §5, `DELIVERY.md` §3). It builds on the
[composition pass](../kernel-composition-2026-09-25/REVIEW.md). The generic
Space engine and FinnLib are unchanged. Validation is in [VALIDATION.txt](VALIDATION.txt).

## The problem

Kernel streams described their transport (pins, widths, markers) and, for typed
ports, their element encoding. Nothing described **which operand positions each
beat carries, in what order, how often the pass repeats, or what a marker
means**. `_wire_mvau` derived those facts by hand and wrote every pin wire
itself. The cyclic ROM was hard-wired into MVAU's weight family, and its image
function belonged to MVAU's traversal. As a result, eltwise, VVAU and
thresholding could not reuse it, even though the roster shows that the cyclic
on-chip source is the one delivery mechanism that recurs across families.

## What was built

| Module | Contents |
|---|---|
| `physical/forms.py` | Logical order (first as `Fold`/`Tile`/`Repeat`/`Batch`; now the canonical `Traversal`, see the revision below), a `Repetition` (`ONCE` or `CYCLIC`), the periodic marker rule `Every(k)`, and `pack(form, values, bits)`. |
| `physical/contract.py` | `StreamContract` combines the transport with the element, form, repetition and marker rules. `compatibility(source, sink)` returns every mismatch, each labelled logical, physical or protocol. |
| `physical/composition.py` | `Composition` places instances and routes clocks and resets. `drive` derives reset inversion from the declared polarities. `connect` checks two stream ends, including their clock domain, then emits every data, padding, handshake and marker wire. |
| `delivery.py` | `CyclicDelivery`, a standalone kernel. It takes a dtype, the consumer's `form` and the operand `values`, and owns the `rom_style` choice. Its views are `output` (a cyclic stream contract) and `build_requirements`. |
| `streaming.replay_buffer_contracts` | The replay's contract transform: `Batch(seq, n)` becomes `Batch(Repeat(seq, REP), n)`, with `olast = Every(len)` and `ofin = Every(len·REP)`. |
| `DotpAxiKernel.interfaces` | dotp's accepted ports, published as a view so that parents do not rebuild its pin names. |

**MVAU now uses the contract end to end:**

- `_wire_mvau` states the MVAU traversal as forms:
  - activation `Batch(Fold(MW, SIMD), R)`, replayed into `Batch(Repeat(·, NF), R)`;
  - weights `Repeat(Tile(MH, MW, PE, SIMD), R)`;
  - results `Batch(Fold(MH, PE), R)`.

  It then calls `connect` four times. There is no hand-written stream wiring left.
- `CyclicWeights` places `CyclicDelivery` as its `source` and supplies it the
  `Tile` form. The unused replay `ofin` marker is disposed of automatically,
  because no consumer requires it.

## What the contract checks

- **Logical level** (a mismatch here needs a kernel to repair it):
  - element encoding;
  - lanes;
  - form;
  - repetition (a cyclic producer of `F` feeds a consumer of `F` or `Repeat(F, k)`);
  - required marker rules.
- **Physical level:** payload fits the word; child padding cannot be silently
  dropped; a top word must be wide enough to carry a child's padding. Padding is
  zero-filled toward children and forwarded to the top.
- **Protocol level:**
  - endpoint direction as seen from each side;
  - clock domain: both ends need a clock driven from the top, and the same domain;
  - marker pins on the top boundary.

Equal widths are not accepted as evidence. `Tile(4,4,1,4)` and `Tile(4,4,4,1)`
have identical lanes and word widths but carry different positions. The test
suite shows that they are refused.

## Evidence

- **Behavior preserved.** Every structural assertion in `test_mvau_assembly.py`
  passes unchanged with generated wiring. These cover the replay `olast → tlast`
  wire, zero-filled child padding, ignored top input padding, forwarded result
  padding, the cyclic image order and parameters, and portable builds.
- **Hardware.** The full MVAU XSI matrix and a block-ROM case pass (see
  VALIDATION.txt).
- **Reuse by a second consumer.** `test_stream_contract.py` composes
  `CyclicDelivery` with `EltwiseKernel` (ADD): the delivery is a channel vector
  `Fold(4, 2)`, and eltwise's rhs reads `Repeat(Fold(4, 2), 3)`. The only glue is
  `connect`. XSim confirms the broadcast sums under input and output stalls.
  A delivery in a different form with identical widths (`Batch(Fold(4,2),1)`) is
  refused before any wire is emitted.

## Design decisions and limits

- **Placement stays open, as intended.** `CyclicDelivery` is a standalone kernel.
  MVAU places it inside its cyclic family, and the eltwise test places it beside
  the consumer. The contract is the same either way.
- **Only enough of the canon is used.** Forms are named constructions with value
  equality, which corresponds to the canon `BeatSequence`. They are not
  enumerated beat maps. Schedules, requirements and availability stay parked. A
  consumer's contract states the form the parent intends; for framing-agnostic
  cores such as dotp, the parent supplies it.
- **Cross-port pairing is not checked.** dotp pairs activation and weight beats
  one to one. That relation belongs to the kernel's semantics and is not part of
  a single stream's contract.
- **Not yet covered:**
  - `Level(d)` loop-end markers, as used by the input generator;
  - valid-only cores;
  - clock-domain crossing;
  - multi-set delivery with a set-index sideband;
  - AXI-Lite writable delivery;
  - dropping a child's padding when one AXIS child feeds another. The first
    physical profile cannot express partial disposal, so `connect` refuses it.

  These are the next increments indicated by the roster.
- **Persistence key changed.** The key is now
  `implementation.cyclic.source.rom_style`, because the choice belongs to the
  reusable kernel. Recorded in scratchpad `space/MIGRATION.md`.

## Revision: traversals replace the ad hoc forms

`Fold`/`Tile`/`Repeat`/`Batch` are replaced by one canonical `Traversal`: a
loop nest over the row-major operand. `beat_loops` step between beats and
`lane_loops` between fields, with field zero least significant. Each loop is an
`(extent, stride)` in flat elements, and stride 0 means replay. Construction
drops unit loops and merges contiguous loops. A property test over 4,000
random pairs confirms that two traversals are equal exactly when they present
the same sequence. `vector_major`, `tile`, `.repeated()` and `.replayed()` build
the common orders.

`classify(source, sink)` names what a mismatch needs. It refines both loop nests
at their shared stride boundaries before comparing them:

| Adaptation | Test | Realized by |
|---|---|---|
| identity | equal | nothing |
| lane_permutation | equal beat loops; lane offsets permuted | wires; `connect` crosses fields |
| reorder | equal lane loops; beat loops permuted, plus replay loops | `input_gen` / OuterShuffle, with `frame_beats`, `DIMS` and `COEFS` derived |
| width_conversion | same element-level order, different lane count | data-width converter |
| lane_regroup | same positions under another lane axis | InnerShuffle |
| incompatible | different positions | nothing |

**Stress cases.** Each result below is derived from the two traversals alone:

- **Tiled MVU input:** `DIMS=(NF,SF,T)`, `COEFS=(0,1,SF)`, frame `SF·T`. These
  are exactly the activation `input_gen` parameters in `mvu_tiled_axi.sv`.
- **Tiled MVU output:** `DIMS=(T,NF)`, `COEFS=(1,T)`, frame `NF·T`. These are
  exactly its `genReorder` parameters.
- **Tiled MVU weights:** the chunked weight stream classifies as a width
  conversion from full tiles. `CyclicDelivery` can also produce the chunked
  order directly, so no adapter is needed.
- **OuterShuffle:** coefficients equal FINN's
  `shuffle_perfect_loopnest_coeffs / SIMD`.
- **InnerShuffle:** classifies as a lane regroup.

Baseline hard-codes those two tiled-MVU stages because every FINN stream is
assumed to be vector-major. With traversals, they become derivable adapters, or
disappear when a neighbour produces the core's order.

**Limit.** A split that is not a loop nest cannot be expressed. The tiled MVU's
weight chunks are a loop nest when `WSIMD` and `SIMD` nest (one divides the
other). In general they need a flattened-axis view.

## Proposal (not implemented): declared streams, adapters, block formats

**Declared streams.** A connection becomes a Space declaration, and kernels bind
their ports to it:

```text
class MVAU(Kernel):
    x  = Stream(boundary=INPUT)                    # top-level port
    xr = Stream()                                  # internal
    w  = Stream()
    y  = Stream(boundary=OUTPUT)
    replay  = Subspace(ReplayBuffer, input=x, output=xr, ...)
    compute = Subspace(DotpAxiKernel, activation=xr, weights=w, result=y, ...)
    weights = SubspaceChoice({"external": Boundary(w), "cyclic": Subspace(CyclicDelivery, output=w, ...)})
```

- Every kernel publishes a contract view per port.
- The parent's assembly is generic: enumerate the streams, find each one's
  producer and consumers, and `connect`.
- A stream's check becomes a Space constraint, so a mismatch is an ordinary
  attributable refusal rather than an exception.
- A choice of kernels on either side is an ordinary `SubspaceChoice`.
- Negotiation fits naturally. Each port declares the traversals it supports as a
  domain, and the stream owns a `form` Decision over their intersection. Adapter
  insertion later becomes a choice of adapter kernels on the stream.
- Open question: whether this needs engine support to enumerate stream bindings,
  or whether inspection over placements is enough.

**Adapter (infrastructure) kernels.** An adapter maps one operand to itself
across traversals (reorder, width conversion, lane regroup) or across encodings
(int to float, MX pack/unpack). It declares a transform signature, so `classify`
results can be matched to available adapters. The FinnLib candidates are
`input_gen` (reorder/replay), `inner_shuffle` (lane regroup) and a width
converter.

**Block formats (MX).** Standardize these at the element-encoding level:

- a `BlockEncoding` of element dtype, block size and scale dtype, next to
  `ScalarEncoding`;
- the physical standard: a block's elements are packed LSB-first, and its shared
  scale travels as a second operand whose traversal is derived from the values'
  traversal by dividing the block axis;
- lanes must hold whole blocks.

This keeps traversals and adapters unchanged. The MX pack and unpack kernels are
encoding adapters.
