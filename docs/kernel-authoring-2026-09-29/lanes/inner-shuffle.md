# Lane C: FinnLib `inner_shuffle` under bursty input

Issue: scratchpad `issues/inner-shuffle-bursty-input.md`. FINN worktree
`finn-inner-shuffle`, branch `fix/inner-shuffle`, from `a361697be`. FinnLib
worktree `finnlib-inner-shuffle`, branch `fix/inner-shuffle`, from the pin
`d03f2fc`. Nothing pushed; the pin is unchanged.

## Root cause

`rtl/shape/inner_shuffle.sv` writes matrices alternately into two pages of
its banks, tracked by `WrJobsDone[1:0]` (page written, not yet read) and
`CurrentPageRd`. Two defects in that page management, both in `d03f2fc`:

1. **The read guard only guards page A.**
   `rd_guard = !CurrentPageRd && !WrJobsDone[0] && !WrJobsDone[1]` is 0
   whenever `CurrentPageRd` is 1: after page A the reader reads page B whether
   or not it is written. When the output drains faster than the input arrives,
   the reader overtakes the writer in page B. The first column needs the
   matrix's last rows, which arrive last, so the lanes holding the last rows
   come out undefined (`x`), or stale once page B has held an earlier matrix.
   Fix: `rd_guard = !WrJobsDone[CurrentPageRd]`.
2. **A page counts as written one beat early.**
   `if(WrAddr == PAGE_OFFSET-1) WrJobsDone[0] <= 1` (and the page B twin) sets
   the flag when the write address *reaches* the page's last address, whether
   or not that beat is written. An input pausing there releases the page
   before its last beat exists, and the flag is set again after the reader
   clears it, so the page is read a second time (a phantom matrix). Fix: set
   the flag on the write, `wr_en && WrAddr == ...`.

Trace of (1), SIMD 3, 6x6, the conformance stall pattern (per-cycle
`$display` of DUT internals): at cycle 35 the reader finishes page A
(`CurrentPageRd` 0 -> 1) with `WrJobsDone=00`, `rd_guard=0`, and the writer
at `WrAddr=21` (9 of page B's 12 beats written). At cycle 36 the reader
issues column 0 rows 3..5, whose row 5 is at address 22, written at cycle 37.
Output word 13 then has lane 2 `x`: `got xxx0f036 exp 0420f036`, the
conformance harness's `output_stream word 13: 0xab != 3ab`.

Trace of (2), SIMD 4, 4x4, last beat of every matrix held back 200 cycles:
`WrJobsDone[0]` is set at cycle 8 with `WrAddr=3` and 3 beats written; the
reader reads page A, clears the flag at cycle 12, and it is set again at
cycle 13 (`WrAddr` still 3). The DUT emits 8 words, two matrices' worth,
before the fourth input beat is written.

Each fix alone is not enough (characterization bench, all 14 configurations
x 8 timings = 112 runs): guard fix only, 22 runs fail (all of timing 5, some
of 4 and 7); flag fix only, 58 fail (as the pin); both, 112 pass.

Also ruled out: `rd_pattern_skid` is pushed on `rd_req_en` rather than on the
bank handshake, which would misalign patterns and data if `rd_req_en` were
ever high while the banks lack credit. The bench counts those cycles: 0 in
every run, at the pin and fixed.

## Characterization (at `d03f2fc`)

Bench: `inner-shuffle-char_tb.sv` (here). 4 matrices per run, distinct values
per matrix, BITS 10. Configurations as SIMD (I x J). Timings (input valid /
output ready):

| # | Input valid | Output ready |
|---|---|---|
| 0 | always | always |
| 1 | `cycle % 3 != 0` | `cycle % 4 != 1` (the conformance "stalled") |
| 2 | `cycle % 3 != 0` | always |
| 3 | always | `cycle % 4 != 1` |
| 4 | bursts, `cycle % 8 < 3` | always |
| 5 | always, but each matrix's last beat held 200 cycles | always |
| 6 | random gap 1/37 (FinnLib tb) | random 6/7 (FinnLib tb) |
| 7 | bursts, `cycle % 8 < 3` | `cycle % 4 != 1` |

Pass/fail at the pin (F = fail, with the number of wrong words):

| Config | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 1 (6x6) | pass | pass | pass | pass | F 77 | F 109 | pass | F 76 |
| 2 (2x3) | pass | pass | pass | pass | F 9 | F 11 | pass | F 9 |
| 2 (4x6) | pass | pass | pass | pass | F 30 | F 38 | pass | F 30 |
| 2 (6x6) | pass | pass | pass | pass | F 43 | F 56 | pass | F 43 |
| 3 (6x4) | pass | pass | pass | pass | F 20 | F 27 | pass | F 20 |
| 3 (6x6) | pass | F 2 | F 2 | pass | F 30 | F 39 | pass | F 30 |
| 3 (6x7) | pass | pass | pass | pass | F 35 | F 45 | pass | F 35 |
| 4 (4x4) | pass | F 2 | F 2 | pass | F 16 | F 16 | pass | F 16 |
| 4 (4x8) | pass | F 2 | F 2 | pass | F 24 | F 28 | pass | F 24 |
| 4 (8x4) | pass | F 2 | F 2 | pass | F 20 | F 28 | pass | F 20 |
| 4 (8x8) | pass | F 2 | F 2 | pass | F 41 | F 52 | pass | F 41 |
| 4 (8x10) | pass | F 2 | F 2 | pass | F 50 | F 64 | pass | F 50 |
| 5 (10x4) | pass | F 2 | F 2 | pass | F 20 | F 28 | pass | F 20 |
| 6 (6x6) | pass | F 2 | F 2 | pass | F 18 | F 24 | pass | F 18 |

With the fix: 112 of 112 pass.

- Only the input timing matters: output stalls alone (3) never fail, and
  timing 2 fails exactly where 1 does. It fails when reading overtakes
  writing, so whether a mild stall (1, 2) fails depends on SIMD, I and J
  (SIMD 1 and 2 got away with it); slow bursts (4, 7) and a held last beat
  (5) fail every configuration, SIMD 1 included.
- Timings 1/2: 2 wrong words per run, the first in the second matrix (page B)
  at word `BEATS + I/SIMD - 1`, the column-0 word holding the last row, with
  only the top lane (the matrix's last row) `x`. This is the conformance
  harness's failure: SIMD 3 word 13, SIMD 6 word 6.
- Timing 5 fails in the first matrix (page A), from defect 2.
- The FinnLib testbench's own timing (6) never fails: its input is nearly
  continuous and its output slower, so the reader never catches up. That is
  why its testbench passed at the pin.

## Changes

**FinnLib `99d75e8`** (`fix/inner-shuffle`, parent `d03f2fc`), "inner_shuffle:
read a page only once all of it is written":

- `rtl/shape/inner_shuffle.sv`: the two fixes above; `read_addr` and
  `ReadAddrReg` declared before first use (strict `xvlog --sv` refused line
  294, VRFC 10-3380; relaxed `xvlog` warns on both, lines 294 and 314).
- `rtl/shape/inner_shuffle_tb.sv`: every configuration also runs from a slow
  feed (a gap before a beat with probability 1/2, beside the existing 1/37),
  plus SIMD 4 at 4x4 and 8x8; messages name the feed (`GAP:`).

**FINN** (`fix/inner-shuffle`):

- `tests/kernels/test_conformance.py`: `BURSTY`/`TRANSPOSE_BURSTY` removed;
  the transpose case has no known failures.
- `src/finn/kernels/transpose.py`: the "Known defect" note replaced by what
  was established; the module note says the fix reopens the adapter option.
- `tests/kernels/rtlsim/adapter_numeric.py`: `KNOWN_DEFECTS` ((4,4,4),
  (8,4,4), (4,8,4)) folded into `TRANSPOSES`, `--known-defects` removed.

## Verification, as observed

| Check | FinnLib | Result |
|---|---|---|
| transpose conformance, `known` removed (driver on `conformance()`) | pin | 4 failures, exactly the 4 in `TRANSPOSE_BURSTY` (SIMD 3 stalled word 13, SIMD 6 stalled word 6, adapter free and stalled word 12) |
| same | fix | all samples, both modes, pass |
| `test_the_kernel_conforms_in_xsim[transpose]`, before this change (with `TRANSPOSE_BURSTY`) | fix | fails: "known failures now pass" (the expected signal) |
| same, after this change | fix | 1 passed |
| same, after this change | pin (`deps/finnlib`) | fails, NonConformance on the 4 simulations: expected until the pin moves |
| FinnLib `inner_shuffle_tb`, original, direct xsim | pin | passes (185 PASS lines; it never exercises the defect) |
| FinnLib `inner_shuffle_tb`, extended | pin | fails: `[SIMD:4, I:4, J:4, GAP:2] Mismatch at beat 0: got xxxx000800040000, expected 000c000800040000` |
| FinnLib `inner_shuffle_tb`, extended, `rtl/run_tests.sh sim shape/inner_shuffle` | fix | PASS (44 instances, 410 PASS lines) |
| `rtl/run_tests.sh synth shape/inner_shuffle` (xcvc1902, 128x384 SIMD 4) | fix | PASS, `synth_design completed successfully` |
| strict `xvlog --sv` of `inner_shuffle.sv` | pin / fix | error VRFC 10-3380 / clean |
| characterization bench (112 runs) | pin / fix | 58 fail / 112 pass |
| `adapter_numeric` (`TRANSPOSES` as before) | fix | 26 PASS |
| `adapter_numeric --known-defects` (before folding) | fix | 24 PASS (SIMD 4 cases pass free and stalled) |
| same | pin | 4x4/4 passes free, fails stalled (the XSI child exits 1; the harness does not relay its error) |
| `adapter_numeric`, folded `TRANSPOSES` (this branch) | fix | 32 PASS, exit 0 |
| `tests/kernels/test_adapters.py` | fix | 12 passed (Python only) |
| `check-kernels.sh` steps after `check-space.sh`, Vivado off `PATH` | pin | kernels 809 passed, 25 skipped; graph 4 passed, 2 skipped; ruff, mypy clean |
| `check-dataflow-design.sh`, Vivado off `PATH` | - | 40 passed; ruff, mypy clean |
| `check-space.sh` | - | 5 mypy-fixture typing tests fail, identically at the base `a361697be` (not this change) |

## Still open

- **RTL checker still declines transpose**, for a new reason: elaboration now
  succeeds, then `extract` (`artifacts/rtl.py`) refuses the unpacked-array
  localparam `RD_INIT_PAT` ("neither an integer nor a string parameter"). A
  checker question (it could skip localparams it cannot represent), for the
  RTL-checker lane; nothing here changes it.
- **Adoption:** push FinnLib `99d75e8` to the remote, then bump
  `FINNLIB_COMMIT` in `fetch-repos.sh`. Until then this branch's transpose
  XSim test and `adapter_numeric` fail against the pin, by design.
- `inner_shuffle` as a stream adapter candidate (lane regroup) is reopened, not
  done.
