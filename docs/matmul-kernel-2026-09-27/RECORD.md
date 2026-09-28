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
