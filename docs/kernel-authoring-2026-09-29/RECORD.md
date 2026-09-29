# Record: kernel authoring

Plan: [PLAN.md](PLAN.md). P0: [p0/REPORT.md](p0/REPORT.md). Branch
`feature/kernel-authoring`, worktree `finn-kernel-authoring`, from `6278ac53a`
(P0 committed); `deps/` fetched (FinnLib `d03f2fc`).

Each increment passes the fast gates (both gate scripts, Vivado off `PATH`, so
XSim tests skip) and is committed; XSim runs from a snapshot of the commit
(`a1/xsim.sh`: `git archive`, its own `deps/finnlib` and `deps/qonnx`).

Baseline at `6278ac53a`: Space 448, kernels 805 with XSim, graph 6, dataflow 40.

## A1: the conformance harness

### What landed

Commits: `59299e0a3` (the harness and its cases), `a0e881750` (fixes from the
first XSim sweep, below). Fast gates at both (Vivado off `PATH`), as observed;
identical counts:

| Gate | Result |
|---|---|
| `check-space.sh` | 448 passed; ruff, mypy clean |
| `tests/kernels` | 804 passed, 23 skipped (was 791 + 14: +13 Python tests, +9 XSim tests skipped) |
| `tests/graph` | 4 passed, 2 skipped |
| ruff format, ruff check, mypy (`finn.kernels`, `finn.graph`, typed tests) | clean |
| `check-dataflow-design.sh` | 40 passed; ruff, mypy clean |

- **`tests/kernels/conformance.py`.** `conformance(family, *, inputs, outputs,
  reference, folds=SAMPLED, choices={}, facts={}, xsim=None)` returns the
  samples it checked. Per sample:
  1. **Place.** A `Design` generated with `type(...)`: one `Stream` per
     reference input, named by it and presented at the boundary under the same
     port name; the kernel as node `kernel`; the sample's Decisions and
     `choices` committed by key (`kernel.<name>`), its Params given; then
     `helpers.settled` (the adapter chains, their `input_gen` memories `auto`).
  2. **ABI.** The kernel's own `build_requirements` materialized
     (`xsim.materialize`), then `check_abi` against the sources with the
     declared binding (`abi.parameters`). A refusal fails; `Declined` is a
     `RtlDeclined` warning and fails under `--strict-rtl` (a new option,
     `tests/kernels/conftest.py`).
  3. **Model.** The design composes (`structure`); on every stream, the
     kernel's end covers its tensor (every position at least once); a
     `ScheduledPort` presents `schedule.beat_count` over the steps of its
     `reduces` and `holds`; each boundary presents the kernel's traversal (an
     input's `unreplayed`), which is what the stimulus is packed from;
     `parameters()` keys equal the non-local parameters `extract` finds (only
     when the checker did not decline).
  4. **XSim** (when `xsim=` is given): random integers in each input's range
     (seeded by kernel id and sample label), the reference outputs, inputs
     packed with `traversal.pack` in `unreplayed(port form)` order, outputs in
     the port's order; `xsim.stream_through`, free and stalled. Failures are
     collected over all samples and raised together (`NonConformance`, with
     `failures` per sample and mode). A design fed by a cyclic source (the
     adapter sample, memstream) repeats, so each output is compared over its
     first pass (`stream_through(..., repeating=True)`: ready low after the
     expected words).
  5. **Adapter sample.** On the middle plain configuration, the first input's
     stream is fed by a read-only `MemStreamKernel` (`ram_style` `auto`,
     unpumped) presenting `vector_major` at the first lane count dividing the
     innermost extent, other than the port's, whose stream plan contains
     `Step.WIDTH`. That input is not driven in XSim; its values are the
     memory's contents.
- **Sampling.** `SAMPLED`: the kernel's open scalar Decisions under `kernel.*`
  (not selectors, not pinned by `choices`), each read with
  `point.field(ref).candidates()` after committing the ones before it:
  first candidate (smallest), middle (interior), last (largest); deduplicated;
  then the adapter sample. `ALL`: every combination by the same walk.
  Explicit: a sequence of configurations; a key naming a `Param` of the family
  is given at construction, any other is committed as a Decision.
- **Element of an output.** A shape takes its element from the kernel: a
  probe design with the inputs placed and that output unplaced, reading the
  port's `element`. A `Tensor` bypasses the probe. This is not D5's check (it
  asserts nothing when the kernel cannot say; it asks for a Tensor).
- **`tests/kernels/test_conformance.py`.** Eight cases, each a Python test
  (fast gate) and an XSim test (`requires_xsim`):

  | Case | Folds | Samples |
  |---|---|---|
  | `dotp-packed` (DSP48E2, INT4, M 2, K 6, N 4) | `SAMPLED` over `pe`, `simd` | (1,1), (2,3), (4,6), adapter at (2,3) |
  | `dotp-int8` (DSP58, INT8, same shape) | `SAMPLED` | same |
  | `dotp-int8-depthwise` (x (2,3,4)) | `SAMPLED` | (1,1), (2,3), (4,3), adapter at (2,3) |
  | `thresholding` (3 x 6, INT4, three thresholds per channel) | `pe` 1, 3, 6 (a Param) | 3 + adapter |
  | `thresholding-rows-first` (the proof's control) | `pe` 1, 2, 3 | 3 + adapter |
  | `eltwise` (ADD, lhs 3 x 6, rhs (6,) broadcast) | `pe` 1, 3, 6 (a Param) | 3 + adapter |
  | `transpose` (2 x 6 x 6) | `input_form` SIMD 1, 3, 6 (a Param) | 3 + adapter |
  | `memstream` (INT4 4 x 6, identity) | `form`: `vector_major` 1, 3, 6, `tile(4,6,2,3)` | 4, no adapter (no input) |

  dotp's references are `x @ w` and `(x * w).sum(axis=1)`; weights `(k, n)`.
- **The proof.** `RowsFirst` is a test kernel: `ThresholdingAxiKernel` with
  its input and output replaced by `ScheduledPort`s on a schedule
  `{r, c}`, `c` folded by `pe`, beats `(r, c)`: the RTL's order.
  `ChannelsFirst` declares beats `(c, r)`. Folds `pe` 1, 2, 3 on 6 channels,
  so the two orders differ in every sample. Tests: `ChannelsFirst` passes
  every Python check; in XSim it must fail in every sample and both modes;
  `RowsFirst` passes both.
  Why thresholding and not dotp (reasoned, not simulated): `dotp_axi` pairs its
  activation and weight frames beat by beat, so a dotp declaring `n` outside
  `m` would likely still compute every result correctly, in the order it
  declared. thresholding's RTL counts channel folds itself, so a declared order
  it does not walk applies the wrong thresholds.
- **The checks refuse.** Two more test kernels: `Misnamed` (memstream with its
  output bus named `m_axis_1`) is refused by `check_abi`; `Unbound`
  (memstream without `RAM_STYLE`, which `memstream_axi` declares with a
  default) fails the parameter-names check.
- **Runner.** `a1/xsim.sh <commit>`: snapshot, one pytest process per
  conformance XSim test in parallel, and the rest of `tests/kernels` with
  Vivado on `PATH`.
- **`tests/kernels/xsim.py`.** `stream_through` gains `repeating` (default
  False: every existing caller unchanged).

No `src/` change. No decision key or name changes. New test-side names: `conformance`, `samples`, `place`, `SAMPLED`, `ALL`, `Sample`, `NonConformance`, `RtlDeclined`, `--strict-rtl`, `stream_through(repeating=)`.

### The RTL checker's decline rate, as measured

Python run, every sample (warnings, `-W always`):

| Kernel | Checked | Declined, and why |
|---|---|---|
| dotp, INT8 core (dense, depthwise) | all 8 samples | none |
| memstream | all 4 | none |
| dotp, packed core | smallest (SIMD 1) | interior, largest, adapter: `add_multi.sv:45` "cannot call a function declared inside a generate block in a constant expression" (slang; Vivado accepts it) |
| thresholding (and `RowsFirst`, `ChannelsFirst`) | none | `THRESHOLDS` is an array parameter: `extract` takes integer and string values only |
| eltwise | none | `B_SCALE` is a real parameter, same rule |
| transpose | none | `inner_shuffle.sv:294` uses `read_addr` before its declaration (slang refuses; `xvlog --relax` accepts it with a warning) |

So the ABI and parameter-name checks bind for 13 of the 32 case samples. Under
`--strict-rtl` six Python tests fail (packed dotp, eltwise, thresholding,
`RowsFirst`, transpose, and the wrong-order Python test).

### XSim from the commits

**First sweep, `59299e0a3`** (snapshot `/tmp/a1-xsim-59299e0a3`):

- Passed: dotp packed, dotp INT8, dotp INT8 depthwise, eltwise (4 samples x
  free and stalled each), and the wrong-order test.
- **memstream** failed every sample at word N (24, 8, 4, 4: one past the last
  expected word), and **thresholding** / **`RowsFirst`** failed their adapter
  sample at word 6 / 9 (one past the last): every expected word matched, then
  the cyclic source kept the design producing and `stream_through` compared a
  word beyond its table. A harness defect, fixed in `a0e881750` (`repeating`).
- **transpose** failed every sample before simulating: strict `xvlog --sv`
  refuses FinnLib `d03f2fc`'s `rtl/shape/inner_shuffle.sv` ("[VRFC 10-3380]
  identifier 'read_addr' is used before its declaration", line 294; declared
  at 309). **Corrected in review:** that is the harness being stricter than
  FINN's flow (`finn_xsi` elaborates with `xelab -relax`; `xvlog --relax`
  accepts the file with a warning), and `inner_shuffle` *had* been simulated
  (`tests/kernels/rtlsim/adapter_numeric.py`, `TRANSPOSES`, the "adapters 26"
  sweep). The `xfail` added here ("does not compile") was wrong; see the
  follow-up below.
- The rest of `tests/kernels` with Vivado on `PATH`: 805 passed (the baseline
  count).

**Second sweep, `a0e881750`** (snapshot `/tmp/a1-xsim-a0e881750`), as
observed:

| Test | Result |
|---|---|
| `dotp-packed`, `dotp-int8`, `dotp-int8-depthwise` | passed (4 samples x 2 modes each) |
| `thresholding`, `thresholding-rows-first`, `eltwise` | passed (4 x 2 each) |
| `memstream` | passed (4 x 2) |
| `transpose` | xfailed (for a wrong reason; see the follow-up) |
| wrong loop order (`ChannelsFirst`) | passed: all 4 samples fail in both modes, each at an output word: `pe=1` word 2 (`00` for `1`), `pe=2` word 1 (`09` for `d`), `pe=3` word 1 (`01` for `13`), adapter `pe=2` word 1 (`08` for `d`) |
| rest of `tests/kernels`, Vivado on `PATH` | 805 passed (baseline count) |
| `tests/graph`, Vivado on `PATH` | 6 passed |

The same module with the RTL's order (`RowsFirst`) passes, so the failure is
the declared order and nothing else in the test kernel.

### Deviations

- **`outputs` accepts a `Tensor`.** D1 has shapes only, element from the
  kernel. Today dotp's `y` (a `ScheduledPort`) takes its stream's element,
  and transpose's `output_stream` is a required Param, so neither can be
  built with its output unplaced. Both cases give a `Tensor`; A5 (transpose)
  and A6 (dotp's `result_dtype`) remove the need. The proof's test kernel
  gives one for the same reason as dotp.
- **`xsim=` parameter, and a return value.** D1's signature has neither. The
  Python checks run in the fast gate; the XSim test of each case passes its
  `tmp_path`. `conformance` returns the samples so the sampling tests can read
  them (`samples()` is also public).
- **Beat counts are checked for `ScheduledPort`s only.** A `GivenPort` has
  no schedule to compare with; coverage and the boundary check apply to both.
- **D1 step 3's parameter-name check depends on step 2.** When the checker
  declines, the names are not established and are not compared.
- **The adapter sample reuses the middle configuration**, not a fourth fold
  configuration: it is distinguished by what feeds the first input.
- **memstream has no adapter sample** (no input); its fourth sample is a
  `tile` form.
- **Not added (as instructed):** D5's unplaced-output check (A6).

### Follow-ups this surfaced

- The checker declines three of the five modules for reasons unrelated to
  ports: non-integer parameter values (arrays, reals) and two constructs
  slang refuses and Vivado accepts. Establishing names without values would
  let the parameter-name check bind for thresholding and eltwise; that is a
  change to `artifacts/rtl.py`, outside A1.
- FinnLib `inner_shuffle.sv` (`d03f2fc`) declares `read_addr` after its first
  use. Strict tools refuse it (slang, `xvlog --sv`); relaxed ones accept it. A
  non-blocking FinnLib cleanliness report, separate from its bursty-input
  defect (A1 follow-up).
- A stream fed by a cyclic source makes the whole design repeat; the boundary
  presents a single pass by rule, but nothing stops the design after it. The
  harness compares the first pass; whether a composite should say so is open.

### A1 review and follow-up

Review (2026-09-29): A1 accepted; the spec deviations above accepted; one
correction (transpose), done in this follow-up.

- **`xsim.simulate` relaxes xvlog** (`--sv --relax`), as FINN's own flow does
  (`finn_xsi`: `xelab -relax`). Strict mode refused `inner_shuffle.sv` only.
- **transpose simulates, and finds the bursty-input defect.** Observed at the
  working tree before the commit (`/tmp/a1f-transpose`), matching the review's
  scratch run:

  | Sample | free | stalled |
  |---|---|---|
  | SIMD 1 (`input_form=72x1`) | pass | pass |
  | SIMD 3 (`24x3`) | pass | fail: `output_stream word 13: 0xab != 3ab` |
  | SIMD 6 (`12x6`) | pass | fail: `word 6: x569a8 != 7569a8` |
  | adapter (a `vpc` feeding SIMD 3) | fail: `word 12: 0x6a != 76a` | fail: same word |

  Undefined upper lanes: the defect `transpose.py` documented at SIMD 4 with
  a side of 4 or 8, here at SIMD 3 and 6 under the harness's stall pattern.
- **Known failures, strict and per simulation.** `conformance(..., known=)`
  maps (sample label, mode) to a reason: any other failure raises
  `NonConformance`; a known one that passes fails ("known failures now pass");
  a key naming no simulation is a `ValueError`. The transpose case names the
  four failing simulations above (`TRANSPOSE_BURSTY`); SIMD 1 and the free
  SIMD 3 and 6 runs must pass. The `UNCOMPILED` xfail is gone. A unit test
  covers the strictness.
- **`transpose.py`'s defect note** now records SIMD 3 and 6 at `d03f2fc`.
- **A second planted error, lane order.** `LanesInOrder`: thresholding with
  PE = C = 6, the channel index split `c = 3 co + ci`, fields `(co, ci)` (the
  RTL's, contiguous). `LanesReversed`: fields `(ci, co)`, a lane permutation.
  Both pass every Python check (coverage, boundary, beats), including an
  adapter sample; `LanesInOrder` is a conformance case, and `LanesReversed`
  must fail in XSim in every sample and mode on an output word. The wrong-order
  tests are parametrized over `loop-order` and `lane-order`.
- **Planned:** the RTL checker establishing parameter names without values, a
  small increment before A4 (PLAN, `A3b`).

Fast gates (Vivado off `PATH`), as observed: Space 448; kernels 807 passed,
25 skipped (+3 Python tests: the `thresholding-lanes-in-order` case, the
lane-order Python test, the strictness test; +2 XSim skipped); graph 4 + 2
skipped; dataflow 40; ruff and mypy clean.

FOLLOWUP_XSIM
