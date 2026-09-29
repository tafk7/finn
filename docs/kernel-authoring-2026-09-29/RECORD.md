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

Fast gates (Vivado off `PATH`), as observed:

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
     `failures` per sample and mode).
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
- **The checks refuse.** Two more test kernels: `Misnamed` (memstream with its
  output bus named `m_axis_1`) is refused by `check_abi`; `Unbound`
  (memstream without `RAM_STYLE`, which `memstream_axi` declares with a
  default) fails the parameter-names check.
- **Runner.** `a1/xsim.sh <commit>`: snapshot, one pytest process per
  conformance XSim test in parallel, and the rest of `tests/kernels` with
  Vivado on `PATH`.

No `src/` change. No key or name changes.

### The RTL checker's decline rate, as measured

Python run, every sample (warnings, `-W always`):

| Kernel | Checked | Declined, and why |
|---|---|---|
| dotp, INT8 core (dense, depthwise) | all 8 samples | none |
| memstream | all 4 | none |
| dotp, packed core | smallest (SIMD 1) | interior, largest, adapter: `add_multi.sv:45` "cannot call a function declared inside a generate block in a constant expression" (slang; Vivado accepts it) |
| thresholding (and `RowsFirst`, `ChannelsFirst`) | none | `THRESHOLDS` is an array parameter: `extract` takes integer and string values only |
| eltwise | none | `B_SCALE` is a real parameter, same rule |
| transpose | none | `inner_shuffle.sv:294` uses `read_addr` before its declaration (slang refuses; Vivado accepts) |

So the ABI and parameter-name checks bind for 13 of the 32 case samples. Under
`--strict-rtl` six Python tests fail (packed dotp, eltwise, thresholding,
`RowsFirst`, transpose, and the wrong-order Python test).

XSIM_RESULTS

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
