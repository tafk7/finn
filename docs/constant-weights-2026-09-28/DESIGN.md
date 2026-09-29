# Design note: a constant-weight RTL dot product

Date: 2026-09-28. Status: **proposed**, for review. Follows the removal of the
cyclic ROM (`docs/kernel-composition-2026-09-28/RECORD.md`, "ROM removed").

## 1. The goal

Weights known at build time, embedded as constants that synthesis can
simplify: a zero weight disappears, a weight of ±1 becomes an add or a
subtract, a power of two becomes a shift, and a low-precision product becomes
a few LUTs instead of a DSP. This is what FINN's `internal_embedded` offers,
but only in its HLS MVAU.

Nothing in the kernel layer does this today:

- FinnLib's `dotp_axi` takes weights only as a stream (`s_axis_weights`).
- Its cores (`dotp`, `dotp_8sx9_dsp58`) instantiate `DSP48E2`/`DSP58`
  primitives. Vivado does not turn a primitive with a constant input into
  fabric logic.
- With folding, each multiplier lane sees NF × SF different weights (NF = N /
  PE, SF = K / SIMD), one per beat. They are data, not constants.

The removed ROM streamed its words through a registered handshake into those
DSPs, so it eliminated nothing; a read-only memstream does the same job.

## 2. When constants can be eliminated

Two conditions, both necessary:

1. **Each multiplier sees one weight forever.** A MatMul's weights do not
   depend on the row, so folding over the rows (M) is free. Folding over N or
   K is not: that is **full unfolding in N and K**, PE = N and SIMD = K, with
   N × K multipliers, each tied to one weight.
2. **Multipliers are inferred in fabric**: behavioural `x * W` with `W` a
   parameter, never a DSP primitive.

It pays for small, low-precision layers: binary, ternary and 2- to 4-bit
weights, 2- to 8-bit activations, N × K in the hundreds to low thousands. It
grows as N × K × (activation bits × weight bits), so it is a choice among
cores, never the default.

**A middle ground: products as tables.** With some folding, a lane sees a
fixed, small table of weights indexed by the fold counter. `x * W[i]`, with
`W` constant, is then a LUT function of (i, x): about log2(NF × SF) +
activation-bits inputs per product bit, which fits 6-input LUTs when both are
small (e.g. 4 folds of 2-bit activations). Synthesis still folds zeros and
signs inside each table. Whether this beats a DSP needs measuring, so it is
an option (Q2), not part of the first version.

## 3. The core

A new RTL module, `dotp_const` (name to settle), in FinnLib preferably (Q1):

- **Parameters.** `N`, `K`, `ACTIVATION_WIDTH`, `SIGNED_ACTIVATIONS`,
  `WEIGHT_WIDTH`, `ACCU_WIDTH`, and `WEIGHTS`: the N × K weights packed into
  one parameter, `n` major, `k` minor, each `WEIGHT_WIDTH` bits, two's
  complement.
- **Ports.** `s_axis_input` (K activations a beat, no `TLAST`: each beat is
  a whole row), `m_axis_output` (N results a beat). No weight port. `ap_clk`, `ap_rst_n`.
- **Datapath.** For each output `n`: K behavioural products
  `x[k] * WEIGHTS[n][k]`, generated with `generate` loops over parameters, so
  every product sees a constant; a pipelined adder tree (registers every
  `log2` level or every few levels, a parameter); the result registered.
  Products with a zero weight are never generated; ±1 are sign-selected
  activations. Synthesis does the rest.
- **Throughput.** One row a cycle; latency = tree depth + 2. Backpressure
  holds the pipeline (an enable), or a skid buffer at the output; the same
  credit scheme as `dotp_axi`'s output queue would do.
- **Accumulator.** Exact from the weights actually present:
  `max over n of sum over k of |W[n][k]| × max|x|`, often far below the
  worst case `exact_result_dtype` gives, since the weights are known. The
  kernel derives it; the MatMul's result type becomes the core's (Q4).

## 4. The kernel

`ConstantDotpKernel` on the Kernel protocol, a third candidate of MatMul's
`compute` Decision (`"constant"`):

- **Ports.** A `ScheduledPort` `x` (lanes: all of `k`) and `y` (lanes: all of
  `n`) from the same `Schedule`, with folds fixed at SF = NF = 1. There is no
  `w` port.
- **Facts.** It takes the weights themselves (`weights: IntegerTensor`,
  stored `(k, n)`), `target_dsp` only to admit the device.
- **Admission.** Refuses unknown or runtime-writable weights and several
  sets (constants are neither), a depthwise form (first version), and a
  size bound: `N × K × ACTIVATION_WIDTH × WEIGHT_WIDTH` above a budget set as
  a Param with a default taken from the synthesis measurements (§6).
- **Choices.** None of its own in the first version: no PE, SIMD or pumping.
  The pipeline depth might become one (Q3).

**Fitting it into MatMul.** Two things change:

1. `w_stream` is a shared binding of `compute` today, and shared bindings are
   strict (every candidate declares them). It moves to the entries of the
   streaming cores (`packed`, `int8_dsp58`), and `weights` becomes a binding
   of the constant core only.
2. With the constant core, nothing uses `weight_stream`, and a stream without
   users is refused (`stream-unused`). `weight_stream`, the `memory`
   Decision and `set_index` are guarded by a derived `streams_weights`
   (`selected(compute) != "constant"`), as `set_index` is already guarded by
   `multi_set`. `settle` then treats the three cores as compatible where their
   admissions allow; with known, read-only weights that fit, the choice among
   them stays the caller's (DSE later).

`finn_attributes` maps the constant core to `mem_mode = "internal_embedded"`,
PE = MH and SIMD = MW. As everywhere in `finn.graph`, FINN's own generator is
not what gets built.

## 5. Alternatives considered

- **Constant ROM into `dotp_axi`** (the removed `RomKernel`): eliminates
  nothing, as §1 explains.
- **`dotp_axi` with `FORCE_BEHAVIORAL` and a depth-1 weight stream**: when
  fully unfolded, the weight word never changes, and Vivado can propagate a
  constant flop output. But the lanes are packed for DSP arithmetic (lane
  offsets, guard bits), so the constants reach the multipliers through the
  packing logic, and the result depends on how far `opt_design` propagates
  through it. That is fragile and unmeasured; a dedicated core is clearer.
- **FINN's HLS MVAU with `internal_embedded`**: out of scope; the kernel layer
  builds RTL and FinnLib cores only.

## 6. Verification

- **Numeric.** The MatMul numeric harness gains a constant-core case:
  random weights with forced zeros, ±1 and powers of two, free and stalled,
  against the NumPy reference, in XSim.
- **Constant elimination.** An out-of-context synthesis (a non-Versal part,
  or skipped and reported as skipped) of the same small layers with (a) the
  constant core, (b) `dotp` with a read-only memstream: LUTs, FFs, DSPs,
  BRAM. Then the constant core alone at 0 %, 50 % and 90 % zero weights: LUTs
  should fall with the zeros, which is the evidence that constants are
  eliminated. The budget in §4 comes from these runs.
- **Structure.** Kernel tests: the core admits only full unfolding, refuses
  writable weights and sets, MatMul without a weight stream or memory, and
  `settle` leaving three compatible cores open.

## 7. Questions for review

1. **Where the RTL lives**: a FinnLib module (`rtl/linalg/dotp_const.sv`,
   pinned like the others) or a kernels-side resource, as `cyclic_stream.sv`
   was?
2. **Scope**: full unfolding only, or also the table mode (§2) for small
   folds?
3. **Pipelining**: a fixed adder-tree register cadence, or a Decision traded
   against `target_period_ns`?
4. **Result type**: the exact accumulator from the known weights (narrower,
   so downstream kernels get narrower types), or the worst case that
   `exact_result_dtype` gives from the types alone?
5. **Depthwise**: a per-channel variant (each channel's K weights on its own
   lane) is natural for the constant core too; first version or later?
