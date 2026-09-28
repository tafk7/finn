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
