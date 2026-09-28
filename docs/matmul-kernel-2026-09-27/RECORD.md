# MatMulKernel: implementation record

The [spec](SPEC.md) and its [M0 analysis](M0.md) define the work; this record
states what each increment changed and the evidence for it. Evidence files are
under [`evidence/`](evidence/). XSim runs use the local Vivado 2025.2
libraries; each simulation runs in a fresh process.

## M1: `MVAU` becomes `MatMulKernel` (dense)

**Change.** `finn.kernels.mvau` is replaced by `finn.kernels.matmul`:

| Before | After |
|---|---|
| `MVAU`, `MVAUAssembly`, `mvau_assembly` | `MatMulKernel`, `MatMulAssembly`, `matmul_assembly` |
| facts `repetitions`, `matrix_width`, `matrix_height` | `rows`, `reduction`, `outputs` |
| Decision `implementation` (`implementation.cyclic.rom_style`) | `delivery` (`delivery.cyclic.rom_style`); `selected(delivery)` is `delivered` |
| instance `u_implementation_cyclic` | `u_delivery_cyclic` |
| module `finn_mvau_<delivery>`, producer `finn.mvau.<delivery>` | `finn_matmul_<delivery>`, `finn.matmul.<delivery>` |
| refusal codes `mvau-arithmetic`, `mvau-folding` | `matmul-arithmetic`, `matmul-folding` |

Tests, the numeric harness (`rtlsim/matmul_numeric.py`), the benchmark
workload (which also drops its stale `segment_length` argument) and the README
follow. MVAU had no thresholding references to remove.

**Evidence** (`evidence/m1/`).

- Structures: the six fingerprinted configurations
  ([`structure_dump.py`](structure_dump.py)) are identical to before once the
  module and cyclic instance names are normalized. All six fingerprints change,
  by those names alone.
- Keys: `delivery` and `delivery.cyclic.rom_style` replace `implementation` and
  `implementation.cyclic.rom_style`; the rest are unchanged.
- Numeric XSI: 28/28 (direct, and a weight FIFO for `packed` and
  `int8_pumped`; external and cyclic; free and stalled output).
- Gates: Space 427, kernels 787 passed; ruff, mypy clean; dataflow gate clean.

## M2: each dot-product core is its own kernel

**FinnLib** (`kernels/matmul-20260927`, `b9262df`, pinned): `dotp_axi` gains a
`CORE` parameter naming the core module, `"dotp"` or `"dotp_8sx9_dsp58"`. The
empty default keeps its own selection, so FinnLib's other users are
unaffected. An explicit INT8 core refuses non-DSP58 targets, weights wider
than 8 bits and activations that do not fit 9 signed bits; the packed core
refuses unbroadcast activations. The output-queue bound is unchanged (the
conservative one of both cores). The testbench gains one forced case per core;
FinnLib's `dotp_axi` and `dotp_axi_backpressure` simulations pass (11 of 11
cases, 217 checks each).

**Kernels.** `DotpAxiKernel` becomes the shared declaration and places no core
(it refuses with `dotp-core`). Two kernels subclass it:

| Kernel | Core | Accepts |
|---|---|---|
| `PackedDotpKernel` | `dotp` | DSP48E1, DSP48E2, DSP58; broadcast activations; `narrow_weights` (default off) |
| `Int8Dsp58DotpKernel` | `dotp_8sx9_dsp58` | DSP58; weights ≤ 8 bits, activations fitting 9 signed bits |

Each names only its own core's sources. The RTL checker now reads string parameters
(`CORE`), which it used to decline.

**MatMulKernel.** `compute` is a Decision over the two core kernels
(`packed`, `int8_dsp58`); both reference the same streams and share the
`compute_pumping` Decision, now a key of the composite. Instances are
`u_compute_packed` / `u_compute_int8_dsp58`. `configure.compatible(point,
choice, requirement)` commits each candidate over a configuration and keeps
those that satisfy a requirement; `matmul_assembly(core=None)` takes the one
compatible core, reports every core's refusals when none is, and asks for a
choice when several are (the choice among them is future DSE work).

**Port relation.** B1's 2-D walks become per-axis walks (`axis_walk`,
`walk_loops`, `split_walk` over operands of any rank), and dotp checks one
labelled relation for both contractions (M0 Q7): lanes per port, beat-by-beat
agreement on shared indices, frames moving only along `k`, and results
following each frame. The dense refusals and their attribution are unchanged.

**Evidence** (`evidence/m2/`).

- Structures: the six fingerprinted configurations are identical to M1 once
  the compute instance name (`u_compute` → `u_compute_<core>`), the `CORE`
  parameter, the implementation identity and the source lists are normalized.
  `pumped-dsp58` takes the INT8 core, the one FinnLib selected for it; the
  others take the packed core, as before.
- Numeric XSI: the dense sweep gains `int8_narrow` (the INT8 core on 3-bit
  operands, which FinnLib's own selection never chose): 32/32. Pure dotp, each
  case on the core FinnLib selected: 13/13 cases (26 runs); stress 8/8 (16).
- FinnLib: `dotp_axi` 11/11 cases, `dotp_axi_backpressure` pass.

## M3: the per-channel mode of the INT8 core

`Contraction` (`DENSE`, `PER_CHANNEL`) is a dotp Param (M0 Q1). Per channel,
the activation port carries PE × SIMD lanes, `ACTIVATION_BROADCASTING` is 0,
and only `Int8Dsp58DotpKernel` accepts it (`dotp-contraction` otherwise).
`forms.channel_tile(rows, window, channels, pe, simd)` is FinnLib's order:
beats walk rows, channel folds, window folds; field `s·PE + p` is window
position `s` of channel `p` (M0 Q5). The labelled relation checks it with no
per-channel code of its own.

**Evidence.** Port tests: the per-channel streams are accepted; window-fastest
lanes (the hlslib order), a dense operand, frames crossing channel groups,
weights walking window folds first, and results walking rows first are each
refused on their own stream. dotp tests: lanes, broadcasting, and the packed
core's refusal.

## M4: one composite for both contractions

`MatMulKernel` takes `contraction` (default `DENSE`). For per-channel, the
activation operand is `(rows, window, channels)` in `channel_tile` order,
`outputs` are the channels and `reduction` the window, and the weights are
`(channels, window)` in the same `tile` walk as dense. `reuse` is derived:
the output folds when dense (rows replayed), 1 per channel (the replay buffer
only adds frame markers, `REP = 1`). `matmul_assembly(contraction=...)`.

**Evidence** (`evidence/m4/`).

- The six fingerprinted dense configurations are identical to M2's; the
  dense sweep is rerun after M5, which changes dense cyclic modules.
- Per-channel numeric XSI, the INT8 DSP58 core, external and cyclic, free and
  stalled output: 20/20. Cases: PE 1 to 4, SIMD 2 to 9 (chains of one to three
  DSP58s), one-beat reductions, INT4 to INT9 activations, and pumped compute,
  which baseline FINN's RTL VVAU never offered.
- Unit tests (`test_matmul_per_channel.py`): derived reuse and markers,
  widths, the packed core's refusal, and no compatible core on DSP48E2.

## M5: the dense realization, and derived NARROW_WEIGHTS

**Dense realization (X8).** `realization` (`native`, `dense`) is a Decision
present only for a per-channel contraction. Densely realized, the datapath is
dense: rows of `window × channels` activations (the operand `(R, K, C)` read
row-major) against block-diagonal weights `W'[c, k·C + c'] = W[c, k]` if
`c' = c`, else 0 (`datapath_weights`). SIMD divides `K·C`; the result
precision stays the operation's (a window of K products). It needs the
weights, so it is refused under external delivery (`matmul-realization`). On
targets without the INT8 DSP58 core it is the only realization: per-channel
operations now run on DSP48E1/E2, at C times the MACs and weight memory.

**NARROW_WEIGHTS (X3, provisional).** Derived by the composite: 1 when the
weights are known (cyclic, later read-only memstream) and none is its type's
most negative value, else 0; passed to the packed core, whose lane packing it
changes. One consequence is visible in the tests: under cyclic delivery the
packed core now waits for the weights, and before the delivery is chosen it
waits for that choice.

**The adapter.** Per channel, `matmul_assembly` commits the realization
together with the folding (the SIMD domain depends on it). Left out, the one
compatible realization is taken; on DSP58 with known weights both are, and
it asks for a choice.

**Evidence** (`evidence/m5/`).

- Dense sweep, rerun because cyclic modules may now carry NARROW_WEIGHTS:
  34/34 (the 7 cases × 2 deliveries × 2 stalls, less the external
  `narrow_weights`, plus the weight FIFO runs). `narrow_weights` exercises
  `NARROW_WEIGHTS = 1` on the packed core (DSP48E2, INT4 weights in −7..7).
- Per-channel sweep: 22/22, adding `ch_dense_e2`, the dense realization on
  DSP48E2 with cyclic weights.
- Unit tests: the block-diagonal image hand-packed, the realization refusals
  and the DSP58 ambiguity, NARROW_WEIGHTS for narrow, full-range and external
  weights.

## C3: replay as a choice

Dense rows are read once per output fold. `MatMulKernel.replay` is now a
Decision over two nodes, present only on a dense datapath:

- `buffer`: FinnLib `replay_buffer` (as before, now the instance
  `u_replay_buffer`);
- `input_gen`: FinnLib `input_gen` with one frame per row (`FM_SIZE` = the
  reduction folds), `DIMS = (reuse, folds)` and `COEFS = (0, 1)`, which owns
  its `ram_style` (`replay.input_gen.ram_style`).

A per-channel datapath has no replay choice: `markers`, a one-repetition
replay buffer, adds the frame markers (M0 X5). `matmul_assembly(replay=...)`
defaults to `buffer`: both replays are always compatible, so the adapter's
default is the caller's choice made explicit, not a filter.

**Kernel layer.**

- `InputGeneratorKernel` joins the stream idiom: optional `input_stream` and
  `output_stream`, with the output contract derived from the input's. A
  frame's beats must be one run; each nest loop steps `stride` beats of it.
  `forms.split_beats` names the frame split.
- A marker rule may name one bit of a wider loop-completion marker,
  `olst[1]`. `Composition.connect` wires that bit as a slice and disposes the
  unread bits (`UnusedOutput` with a bit range, from B3). dotp's TLAST is
  `olst[1]`; `olst[0]` is disposed.

**Evidence** (`evidence/c3/`).

- Numeric XSI, every dense case and delivery through the input generator,
  free and stalled, with the replayed words and frame markers observed at the
  generator's output: 26/26.
- Unit tests (`test_matmul_replay.py`): the keys, the `input_gen` parameters,
  `olst[1]` wired as a one-bit slice and `olst[0]` disposed, marker-bit
  validation.

## C4: an RTL memstream delivery, and runtime-writable weights

**`MemStreamKernel`** (`memstream.py`, FinnLib `memstream_axi`, ported in A2)
is a delivery kernel like `CyclicDelivery`: the consumer gives the operand
`values` and the `form` it reads them in, and the kernel packs the image in
that order. It owns `ram_style` (`auto`, `distributed`, `block`, `ultra`) and
`pumped_memory` (the memory at `ap_clk2x` on half-width words; its 2x clock is
driven by role and tied low when unpumped). `writable` presents the AXI-Lite
port through a `control` bus; otherwise the port is tied off.

**INIT_FILE.** The artifact layer had no way to ship generated contents: a
`DataSlot` is a declared hole that no build stage fills yet. A new
contribution, `GeneratedData(path, data)`, carries generated bytes; the
kernel names the file by its contents (`memstream_<sha256[:16]>.dat`), sets
`INIT_FILE` to that name, and the build stages it as a data source
(`Language.DATA`, `Role.DATA`) next to the RTL. Composition flattens it like a
copied source. A pumped memory's image holds each word as its low half, then
its high half, as FINN writes it. **Open:** the artifact layer's documented
intent is that parameter images are holes bound late (`DataSlot`,
`DataBinding`), so one structural component serves every weight value. This
change puts the image in the build identity instead, as the cyclic ROM's
`INIT_DATA` parameter already does. Moving both to bound slots belongs to the
artifact-integration work.

**MatMulKernel.** `delivery` gains the `memstream` candidate over the same
`weight_period` and `datapath_weights`. `writable_weights` (a Param) makes it
runtime-writable; the composite places a `ControlBus` named `s_axilite` and
`netlist` exports it (B3). Writable weights with another delivery are refused
(`matmul-writable`). NARROW_WEIGHTS is derived for read-only memstream
weights as for cyclic ones. The adapter gains `ram_style`, `pumped_memory`
and `writable_weights`. The kernel-local `cyclic_stream.sv` stays a separate
candidate (the user's answer; consolidating them is flagged future work).

**Harness.** `rtl_transport` places data files where XSim resolves
`INIT_FILE`, and carries AXI-Lite writes (address, then data, then the
response) after reset and before any stream starts.

**Evidence** (`evidence/c4/`; C5's is here too).

- Every dense case through a read-only memstream, free and stalled: 14/14.
  Per-channel cases, including the dense realization: 12/12.
- Pumped memory (the image as half-words, `ap_clk2x`): 14/14.
- Runtime-writable: each case writes a second weight matrix through
  `s_axilite` before streaming, 20 rows. The memstream has already fetched up
  to `FULL_CREDIT` = 8 words of the stored image (`memstream.sv:131`), so the
  harness finds where the consumed words become the written image (at most
  8), and checks every row read wholly after it against the written weights:
  14/14. The first rows may use stored weights; that is the memory's
  read-ahead, not a fault, and a consumer that writes weights at run time
  must allow for it.
- Several sets (C5): three sets, each row selecting set `(3r + 1) mod 3`
  through `in2_V`; results and the consumed weight words both checked per
  row: 14/14.
- Unit tests (`test_memstream.py`): the image and its INIT file, pumped
  half-words, tie-offs, the materialized data file, `s_axilite` export, the
  set index port, the refusals.

## C5: multi-set delivery

`weight_sets` > 1 stores one weight operand per set in the memstream
(`SETS`), and each row selects its set by an index on `in2_V`, an ordinary
stream of one `UINT(⌈log2 SETS⌉)` index per row (`set_index`, present only
with several sets; FinnLib's `SET_BITS` rule). The memstream reads it through
its `set_stream` reference input and presents one pass per index. Other
deliveries refuse several sets (`matmul-sets`). The index producer (MLO) is
instance wiring outside the module (D10, D11).

A known limit, recorded for D10: the weight stream's form stays the
per-pass tile, so the stream does not say *which* set each pass carries. A
data-dependent gather is not a traversal; the stream model has no value for
it yet.

## C6: stream adapters

**Built.** Two FinnLib shape components join the stream idiom as adapter
kernels (`adapters.py`). Each sits between two streams and derives its output
contract from its input's, so the adaptation it performs is the one
`classify` names between the two forms:

| Kernel | FinnLib | Adaptation | Notes |
|---|---|---|---|
| `WidthConverterKernel` | `vpc` | `WIDTH_CONVERSION`: the same element sequence, `lanes` a beat | vectors of `N = lcm(PI, PO)`, so neither side pads; a stream that is not whole vectors is refused (`vpc-geometry`) |
| `TransposeKernel` | `inner_shuffle` | `LANE_REGROUP`: row-major `(I, J)` rows in, columns out | SIMD divides I and J; the input must be row-major (`transpose-form`); `ram_style` is its choice |

`forms.regrouped(form, lanes)` presents a form's element sequence at another
lane count. Neither adapter emits markers, so a consumer that needs frame
markers refuses them.

**Evidence** (`evidence/c6/`). Each adapter composed into a module between
`in0_V` and `out0_V`, expected words packed from the element values, free and
stalled (`rtlsim/adapter_numeric.py`): `vpc` 2→3, 4→2, 3→12 and 6→4, and
`inner_shuffle` 4×6/2, 6×6/3, 4×4/2 and 6×9/3: 16/16. Unit tests
(`test_adapters.py`): each output form is the adaptation `classify` names; the
refusals.

**Not built: the adapter as a Decision inside a stream.** Phase C planned an
`adapter` Decision over these nodes inside each stream, chosen by `classify`.
The D10 design round (`../stream-model-2026-09-27/DESIGN.md` §0, §5.4, §8.2)
found why that cannot be done well yet: today a composite writes one form per
stream and both ends adopt it, so a stream never holds two forms to adapt
between. It needs D10's first increment, S1 ("presentation at the ends"),
which is design work awaiting review. Inside one MatMulKernel every stream
agrees by construction, so there is also no intra-module mismatch to adapt
today; adapters earn their place at a module's boundary (a boundary
presented differently from its consumer, D10 decision 3) and between modules
(instance wiring, D10). Until then an adapter is an ordinary node a composite
places between two of its streams, as the tests and the XSim harness do.

**Found: a FinnLib `inner_shuffle` defect.** With SIMD 4 and a matrix side of
4 or 8 (4×4, 8×4, 4×8), the output carries undefined lanes when the input
arrives in bursts with idle cycles between them. Output backpressure alone
does not trigger it. FinnLib's own testbench passes these shapes with its
random stimulus, and fails the same way once only its input timing is changed
to bursts of two beats and three idle cycles, with the output always ready
(`evidence/c6/finnlib-inner_shuffle-probe.{diff,txt}`: beat 0 of the second
round reads `xxxxxxxx00040000` for `000c000800040000`). The condition is not
characterized, so `TransposeKernel` refuses nothing yet and says so; the
harness keeps these cases behind `--known-defects`. It needs a FinnLib fix
before the transpose adapter is used.

## Where this leaves things (2026-09-28)

**The composite.** `MatMulKernel` is one design space for dense and
per-channel matrix multiplication:

| Slot | Kind | Present when |
|---|---|---|
| `contraction`, extents, dtypes, `target_dsp`, `target_period_ns`, `weights`, `writable_weights`, `weight_sets` | Params | always |
| `realization` (`native`, `dense`) | Decision | per-channel |
| `pe`, `simd` | Decisions (divisors; SIMD of the datapath's reduction) | always |
| `replay` (`buffer`, `input_gen`) | Decision over nodes | dense datapath |
| `markers` (replay buffer, one repetition) | derived node | per-channel datapath |
| `compute` (`packed`, `int8_dsp58`), `compute_pumping` | Decision over nodes, shared Decision | always |
| `delivery` (`external`, `cyclic`, `memstream`) | Decision over nodes | always |
| `weight_stream.transport` (`direct`, `fifo`) | Decision over nodes | always |
| `set_index` (`in2_V`) | stream | several sets |
| `s_axilite` | exported control bus | writable weights |

**Evidence at the end.** All XSim runs above pass: dense 34, input-generator
replay 26, per-channel 22, memstream 14 + 12 per-channel, pumped memory 14,
writable 14, several sets 14, adapters 16. Gates: see the final gate lines in
`evidence/final/`.

**For the user to decide or review.**

1. The D10 design round and its decisions (`../stream-model-2026-09-27/DESIGN.md`
   §0, §11), including S1, which unblocks C6's in-stream adapter Decision.
2. `NARROW_WEIGHTS` derivation (flagged provisional), and its consequence
   that the packed core waits for known weights.
3. The FinnLib `inner_shuffle` defect (C6): fix upstream on the fork, or
   refuse the affected shapes once characterized.
4. Parameter images in the build identity (C4) versus late-bound slots.
5. The adapter defaults `replay="buffer"` and `core` by compatibility, and
   the requirement to choose when several cores or realizations fit.
6. FinnLib branch `kernels/matmul-20260927` (`b9262df`) is pushed to the fork
   and pinned; this repository's branch is not pushed.
