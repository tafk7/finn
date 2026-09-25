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
| `physical/forms.py` | Logical beat forms (`Fold`, `Tile`, `Repeat`, `Batch`), a `Repetition` (`ONCE` or `CYCLIC`), the periodic marker rule `Every(k)`, and `pack(form, values, bits)`. Forms compare by value; `positions()` enumerates them. |
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
