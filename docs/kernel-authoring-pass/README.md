# Flat FinnLib kernel authoring pass

> Historical baseline: this document records the earlier flat authoring pass.
> Its Space/Problem syntax and validation results describe that source state.
> Use the [current Space API](../../../scratchpad/space/AUTHORING.md) and
> [physical-kernel guide](../../src/finn/kernels/README.md) for the delivered
> runtime. The experimental dataflow port remains deferred.

This pass supplies concrete examples before designing another authoring API.
Each class owns its identity, supplied inputs, implementation decisions,
derivations, constraints, native interfaces and source requirements. The
declarations have no child kernels, Regions, or compiler-node dependencies.
An HDL implementation may still instantiate its own internal modules.

## The seven examples

Paths below are relative to `src/finn/kernels`.

| Class | Source | Supplied facts and decisions | Accepted result |
|---|---|---|---|
| DotpAxiKernel | `dotp.py` | PE/SIMD, operand/result dtypes, DSP target, segment length; pumping decision | Native AXI-stream RTL module requirements; existing reference unchanged |
| FifoKernel | `fifo.py` | Opaque word width, requested depth; RAM-style decision | Native unpadded ready/valid pins |
| InputGeneratorKernel | `input_generator.py` | Word width, input-frame length, immutable loop extents/strides; RAM style | Native ready/valid pins and one marker bit per loop level |
| ThresholdingAxiKernel | `thresholding.py` | Input/threshold dtypes, immutable `(sets, channels, thresholds)` integer table, PE, bias, memory triggers; AXI-Lite and pipeline decisions | AXI data/configuration/set-selection interfaces and fully specified initialization |
| EltwiseKernel | `eltwise.py` | Operation, PE, two dtypes, scale and target | Native unpadded two-input arithmetic interface with derived result dtype |
| IntToFp32Kernel | `int_to_fp32.py` | Integer input dtype | Combinational `ival`/`fval` pins, FLOAT32 output, round toward zero |
| MemStreamHlsKernel | `memstream_hls.py` | Scalar element dtype and memory depth | Explicit HLS function interfaces and a complete C++ source bundle |

All classes are exported from `finn.kernels`. Bind with the existing
Space/Problem/Subspace machinery and consume `point.physical.accepted_answer`.
`tests/kernels/test_flat_kernels.py` contains small binding examples for every
new declaration. Unresolved inputs stay unresolved; raw `codegen` values do not
claim acceptance. Supported cases must pass the declared physical View.

## Deliberate profile boundaries

- FIFO keeps words opaque and exposes the native requested-depth and RAM-style
  behavior. FinnLib can round capacity up and overrides shallow memory styles.
- InputGenerator accepts positive loop extents and nonnegative strides with all
  selected words inside the current input frame. Zero strides support replay.
  The vectors are immutable tuples; they are not parsed expressions or a second
  schedule language. `olst[i]` marks completion of level i and all inner levels.
- Thresholding initially supports ordinary integer inputs and thresholds with
  matching signedness. The native extension/saturating narrowing to threshold
  precision is part of the computation. Tables must be rectangular and sorted;
  their shape derives SETS, C and N. Initialization is an explicit SV parameter,
  so changing a value changes the build requirements. Runtime writes must keep
  each row sorted. Float threshold comparison remains outside this first profile.
- Thresholding retains all native configuration/set pins when those services
  are disabled, including unspecified disabled outputs. Multiple static sets
  are supported, but multi-set AXI-Lite configuration is rejected: the pinned
  wrapper's configuration address width omits the set bits.
- Thresholding's negative-bias width derivation preserves native unsigned
  32-bit expression evaluation. For example, three thresholds with BIAS=-10
  have a native 33-bit output rather than the mathematically minimal five bits.
  Simulation also shows the native result addition zero-extending the negative
  32-bit BIAS into that output. This profile therefore refuses BIAS < -N-1;
  matching the pin width alone would not establish the claimed numerical result.
- Eltwise supports ADD, SUB, SBR and MUL. Integer pairs must match widths and
  signedness. Unsigned subtraction produces a signed result. Mixed/float paths
  require DSP58 and use the native integer-to-FLOAT32 conversion. Scaling is
  rounded to binary32 first, then constrained exactly as the native operation
  requires. Integer widths are bounded to 1..128 in this profile.
- IntToFp32 admits ordinary integers of 1..128 bits, keeping outputs finite.
  It deliberately introduces no clock, reset, stream, or byte-padding contract.
- MemStreamHLS admits ordinary integers up to the default ap_int limit of 1024
  bits and QONNX IEEE FLOAT32 values. Native depth one would instantiate
  a zero-width pointer, so depth must be at least two. Memory is initialized
  through runtime configuration; no memory contents are silently invented.

## HLS source boundary

`artifacts/hls.py` adds a small source-level handoff and rendering function.
It reuses existing source contributions, dependency closure, and template
rendering. It carries C++ argument types/shapes and HLS interface modes, without
inventing an RTL ComponentABI before synthesis. There is no new build runner.

The top in `resources/memstream_hls.cpp.j2` follows FinnLib's native MemStream
wrapper: memory and return/control are on the AXI-Lite `control` bundle; output
is AXI stream. Continuous streaming requires start/auto-restart. The supplied
top function is `memstream_hls`. Stage all returned source paths unchanged and
add the declared relative include directories when invoking HLS.

## Native source correction

Vivado and slang both reject `rtl/infra/axilite.sv` because `snk_re` is used
before its declaration. `resources/axilite.sv` is a pinned local copy that moves
the declaration before use and expresses the same initializer as a continuous
assignment. The shared FinnLib checkout is untouched. Revision and upstream
SHA-256 are recorded in the copied file. Existing dotp RTL is unchanged.

## What to compare before refining authorship

1. Repeated native pin declarations versus the existing byte-aligned AxiStream
   declaration. Opaque words, numeric fields and plain pins must stay distinct.
2. Datatype admission versus numerical result derivation versus realization
   restrictions. Eltwise and thresholding provide different examples from dotp.
3. Canonical immutable array inputs and final literal emission. InputGenerator
   vectors and threshold tables expose what scalar-only helpers miss.
4. Named readiness/constraint-group registration and raw versus accepted values.
5. HLS function-interface declarations versus known RTL port declarations.

Keep these examples explicit during this pass. Shared constructs should replace
repeated authorship demonstrated here. WeightFetch, DynamicWeightLoader, tiled
MVAU, new Op composition, and graph/compiler integration remain future work.
BurstRead is the proposed next example for memory-mapped interfaces.

## Verification

The focused suite checks native pin names/directions/widths by elaborating the
real source modules with the selected scalar and array parameters. It also
checks acceptance/refusal, output types, source closure, partial evaluation,
immutable input recognition, and generated HLS C++ with actual vendor headers.
Vivado simulations cover FIFO, replay traversal and loop markers, saturating
threshold input conversion, signed interpretation of unsigned subtraction,
mixed integer/float arithmetic, and combinational round-toward-zero conversion.
Stream simulations include backpressure, stability checks and a drain interval.

Run `scripts/check-kernels.sh` for the independent package gate. Vitis headers
for the C++ check can be selected with `VITIS_HLS_INCLUDE`. Detailed outcomes and
the representative HLS synthesis invocation are recorded in `RESULTS.md`.
