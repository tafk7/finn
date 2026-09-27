# Phases A and B: record

Date: 2026-09-27. Plan of record: [STATUS.md](../kernel-status-2026-09-27/STATUS.md)
§4, phases A and B. Transcripts are in [`evidence/`](evidence/). Every run is
local (no Docker): the kernel venv, ruff and mypy from `PATH`, Vivado 2025.2
`xvlog`/`xelab`/`xsim` on `PATH`, so the XSim-backed tests execute.

## A1. Per-checkout `deps/`

**Change (`56475970c`).**

- `fetch-repos.sh` refuses a symlinked `deps/` or dependency. Fetching a pin
  through a link would check it out in a clone that other checkouts use.
- The FinnLib pin moves from `dfeafac8` to `b17eae6a`, the commit every gate
  and numeric run had used. `dfeafac8` has the reorganized layout, which the
  kernels' flat source paths do not match; the pin was never usable. A2
  replaces it.
- `CLAUDE.md` states the rule and the `FINNLIB_ROOT` override for a working
  clone.
- This checkout: the `deps/finnlib` and `deps/qonnx` symlinks (to the shared
  `prj-kernels/finnlib` and `prj-kernels/qonnx` clones) are replaced by clones
  from `fetch-repos.sh` (`FINN_SKIP_BOARD_FILES=1`), at the pins: FinnLib
  `b17eae6a`, qonnx `21d4c1a7`. The shared qonnx clone was at `342dffb`, on
  another branch than the pin; the gates now run against the pin.
- The spike worktree `finn-space-declarative`: its `deps` symlink (to this
  checkout's `deps`) is replaced by its own clones at the commits it was
  validated with (FinnLib `b17eae6a`, qonnx `21d4c1a7`).

**Gate (on `26e224ca9` + the working-tree change, committed unchanged).**

| Evidence | Result | Transcript |
|---|---|---|
| Guard | `fetch-repos.sh` with the symlinks present: refuses, shared clones untouched | — |
| `scripts/check-kernels.sh` | Space **423 passed**; kernels **763 passed, 0 skipped** (the 9 XSim tests executed); format, lint, strict mypy clean; exit 0 | `a1/gate-check-kernels.txt` |
| `scripts/check-dataflow-design.sh` | **16 passed**; clean; exit 0 | `a1/gate-check-dataflow-design.txt` |
| MVAU fingerprints | **identical** to the landing (`../space-declarative-2026-09-26/evidence/landing/fingerprints.txt`) | `a1/fingerprints.txt` |
| MVAU numeric XSI | **28/28 PASS**: direct 20 (5 cases x external/cyclic x free/stalled); depth-2 weight FIFO 8 (`packed`, `int8_pumped`); each run exit 0 | `a1/mvau-numeric-*.txt` |

## A2. FinnLib on one pinned commit (J0)

**FinnLib (`tkeller/finnlib`, branch `kernels/consolidated-20260927`, pushed).**
Base: upstream `tpreusse/finnlib` `dev` at `5306111`, the latest upstream,
rather than `dfeafac8`'s older `dev`; the newer base adds upstream work only
(`dotp` parameter checks, `fmaf`, `requantf`, `vpc`). On top:

| Commit | Content |
|---|---|
| `7653a47`, `62f8baf` | `replay_buffer` (cherry-picked from `dfeafac8`'s branch, unchanged; byte-identical to `b17eae6a`'s copy) |
| `0d0be24` | Carried from `b17eae6a`: the `dotp_axi` output reservation (both cores' pipeline depth, plus one entry for the registered output lock) and `axilite` declaring `snk_re` before use; `dotp_axi_backpressure_tb` |
| `c266da8` | Carried from `f774412`: the `input_gen` live-window span monitor |
| `11b5c64` | `memstream` and `memstream_axi` ported from `finn-rtllib/memstream` into `rtl/infra/`, with ABC files, synthesis tops and testbenches. FinnLib's `axilite` is the adapter it instantiates (same interface, so it is not ported twice). Two changes: a single-set `memstream` drives `srdy` (it was undriven), and the `memstream_axi` testbench sets `SETS`, ties the set stream, and runs a pumped and an unpumped instance |

`b17eae6a`'s `KERNEL_CONTRACT_REFINEMENT.md` is not carried: it documents the
flat layout this replaces.

**Kernels (`34d734de8`).**

- `fetch-repos.sh` pins `11b5c64b`.
- Every FinnLib source path moves to the grouped layout. The HLS memstream
  includes `hls/infra` and `hls/util`.
- Eltwise's closure takes the consolidated `fifo`, which replaced `queue`.
- The FIFO kernel's storage model follows the new `fifo.sv`, found by the gate's
  native capacity regression (depth 65, `auto`: capacity 65, model said 66).
  `auto` now selects LUTRAM for 34–257 words, except up to 64 words narrower
  than 12 bits. `distributed` is a real LUTRAM FIFO holding exactly `DEPTH`
  words, no longer an alias for the shift register. `FifoKernel` is version 2.
  The regression gains `(257, auto)` and `(40, distributed)`.
- The kernel README states the new baseline.

**Gates.**

| Evidence | Result | Transcript |
|---|---|---|
| FinnLib behavioural simulation (Vivado 2025.2 `xsim`, ABC flow) | 7/7 PASS: `infra/memstream`, `infra/memstream_axi` (pumped and unpumped), `infra/axilite`, `infra/replay_buffer` (5 cases), `linalg/dotp_axi`, `linalg/dotp_axi_backpressure`, `shape/input_gen_span` (12 nests) | `a2/finnlib-sim.txt` |
| `memstream`, `memstream_axi` synthesis | Both synthesize on `xczu3eg-sbva484-1-i` (licensed): `synth_design completed successfully`; `memstream_axi` (SETS 2, 912x73, pumped) infers 4 RAMB36 + 1 RAMB18. The default Versal part is unlicensed here and was not run | `a2/memstream-synth.txt` |
| `scripts/check-kernels.sh` | Space **423**; kernels **763 passed, 0 skipped**; clean; exit 0 | `a2/gate-check-kernels.txt` |
| `scripts/check-dataflow-design.sh` | **16 passed**; clean; exit 0 | `a2/gate-check-dataflow-design.txt` |
| MVAU numeric XSI | **28/28 PASS** (20 direct, 8 FIFO), FinnLib `11b5c64b` | `a2/mvau-numeric-*.txt` |
| Decision and node keys (`../space-declarative-2026-09-26/mvau_keys.py`) | identical | `a2/keys-diff.txt` |

**Recorded fingerprint change.** All six MVAU fingerprints change: every
FinnLib source identity changes (paths and contents). `structure_dump.py` prints
each fingerprinted configuration without source identities (top ABI, instances,
parameters, child ABIs, wires, dispositions, source file names); run on
`c2b6c927d` and on `34d734de8`, the outputs differ only in the FIFO kernel's
version (`a2/structure-diff.txt`).

| Configuration | Before (A1) | After (A2) |
|---|---|---|
| external | `9973c6fb601f4fdc…` | `de1b5f07904f0563…` |
| cyclic-block | `36d7f9e02cfae22e…` | `858a267f72b617bd…` |
| fifo-external | `09d35f61e605caa9…` | `f432a2c4e7581597…` |
| fifo-cyclic | `24f47e030dd0d8de…` | `c2d630e072143441…` |
| padded-output | `58ef50a5c141fdc1…` | `b5a34e3b339cd266…` |
| pumped-dsp58 | `68ff13429c461b49…` | `0669b3c963c01471…` |

Full values: `a1/fingerprints.txt`, `a2/fingerprints.txt`.

**Upstream.** Nothing is proposed upstream yet. Candidates for PRs to
`tpreusse/finnlib`: `replay_buffer`, the `dotp_axi`/`axilite` corrections with
their regression, and the memstream port. `replay_buffer`, its testbench and
the backpressure testbench carry BSD-3-Clause headers while the library is
MIT; relicensing them is the author's call before a PR.

## B1. Ports that check their streams, with per-port attribution (J2)

**Engine: per-input exports** ([proposal](PROPOSAL-per-input-exports.md), STATUS D8).
An export may map a key to one view per reference input:
`exports = {KEY: {input: view, ...}}`. `Users(KEY)` on a referenced node then
yields only the view presented through the input that references it, and
`Members(KEY)` one entry per input, located by the input's name. Definition-time
checks: each inner key is a reference input of the family, each value a view
with compatible semantics. A plain export keeps its meaning. Changes:
`collection.py` (`EffectiveSpace.input_exports`), `_linker.py` (per-input
member map, `Users`/`Members` gathering, semantics check), the widened
`Space.exports` annotation, and the `Members`/`Users` docstrings. The
canonical Space documentation (`scratchpad/space/{AUTHORING,DESIGN}.md`) gains
the rule and one example.

**Kernels.**

- `streams.py`: the per-kernel `Ports` record, `Port`, `Flow`, `produces` and
  `consumes` are removed. `PORT` is a key of `StreamContract`, exported per
  stream input; a stream reads its users' contracts through `Users(PORT)` and
  takes direction from the contract's transport endpoint (initiator produces).
- dotp owns what it reads instead of adopting the stream's form:
  - activation: SIMD lanes of consecutive columns, one frame marker, and a
    frame (the marker period) within one activation row;
  - weights: a matrix over the activation's columns, PE rows of SIMD columns per
    beat (SIMD fastest), whose column walk equals the activation's beat by beat,
    and one group of rows per frame;
  - results: PE consecutive columns of the weights' rows, each beat holding its
    frame's activation row and weight rows.

  The checks compare row/column walks of the traversals (`forms.beat_walk`,
  `walk_axis`, `split_walk`, new), not enumerated positions, so they cost the
  size of the loop nests. Refusal codes: `dotp-stream-lanes`, `dotp-stream-form`.
- Duplicate dotp checks are removed: the support group checks only the DSP's
  own bounds (activation width against the B input, the INT8 special case,
  weight width against the A input as `dotp-weight-width`, accumulator
  capacity); the port scalars own the encodings (family, signedness, two bits).
  `codegen` no longer repeats the support group's geometry and width checks.
- `_port` takes its AXIS declaration instead of a tuple index.
- `ReplayBuffer`'s contracts are a named `ReplayContracts(input, output)`, not a
  tuple read through `cast`.
- The kernel README's composition paragraph describes the landed model
  (it still showed `TopInput`, `compose` and `Present`, audit D14).

**Evidence.**

| Evidence | Result | Transcript |
|---|---|---|
| `scripts/check-kernels.sh` on `d7a64b0be` (clean) | Space **427 passed** (+4 per-input export tests); kernels **772 passed, 0 skipped** (+7 port-contract tests, +2 walk tests); clean; exit 0 | `b1/gate-check-kernels.txt` |
| `scripts/check-dataflow-design.sh` | **16 passed**; clean; exit 0 | `b1/gate-check-dataflow-design.txt` |
| MVAU numeric XSI | **28/28 PASS** on the B1 tree before one annotation-only mypy fix (`list[Step]()` in `split_walk`), which the kernel gate above covers | `b1/mvau-numeric-*.txt` |
| MVAU fingerprints | **identical to A2**: no build requirement changes | `b1/fingerprints.txt` |
| Keys | decision keys identical; node keys lose `compute.ports`, `replay.ports`, `implementation.cyclic.ports` | `b1/keys-diff.txt` |
| Space documentation examples (`scratchpad/space/check-examples.py`) | 25/25 pass (one new: per-input exports) | — |

**Regressions for the audit probes** (`tests/kernels/test_port_contracts.py`):

- P2: an MVAU with unsigned weights refuses `weight_stream` only
  (`dtype-family`); `replayed` and `results` connect.
- P5: a one-lane activation stream is refused at dotp's activation port
  (`dotp-stream-lanes`); a transposed weight tile at the weight port
  (`dotp-stream-form`).
- Further form cases: weights whose column walk does not follow the
  activations; results that swap the order of frames; a frame crossing
  activation rows.

## B2. Clock and reset as a Space (J3)

**Model** (`src/finn/kernels/clocks.py`, new).

- A `ClockDomain` is a node in the composite, named by its top pins
  (`clock`, `reset`). A `DerivedClock` runs at twice its `base` domain's rate,
  phase aligned, and has no reset of its own.
- A kernel has one reference input per domain it runs in and exports, per
  input, the `Clocking` (clock pin, reset pin) that domain drives. dotp:
  `clock` drives `ap_clk`/`ap_rst_n`; `fast_clock` drives `ap_clk2x` only when
  compute is pumped. `ReplayBuffer` and `CyclicDelivery` name `clk`/`rst`.
- A domain sees its kernels through `Users(CLOCKING)` and exports a `Domain`
  (`DOMAIN`). A domain no kernel runs in is absent from the top.
- A stream references the domain it lives in (`Stream.clock`): its AXIS
  boundary is associated with the domain's pins, and a transport stage (the
  FIFO) is clocked by it.
- `Tieoffs` (`TIEOFFS`): inputs a kernel holds constant, and outputs it leaves
  unconnected, in its configuration. Unpumped dotp ties `ap_clk2x` low.

**Netlist.** `netlist(modules, streams, domains, tieoffs, ...)` builds the top
clocks and resets from the used domains (a base reset is synchronous to its
clock and every used derived clock; derived clocks bring their alignment) and
drives each child clock and reset pin from the domain that names it. The
`"clk2x" in name` routing and the scan of child pin names for
`ap_clk`/`ap_clk2x`/`ap_rst_n` are gone. An input that no stream, domain or
tie-off drives is refused with its name. `Composition` gains `tie` and
`dispose`.

**MVAU.** `clock = ClockDomain(clock="ap_clk", reset="ap_rst_n")` and
`fast_clock = DerivedClock(clock="ap_clk2x", base=clock)`, referenced by every
stream and kernel. New node keys `clock` and `fast_clock`; decision keys
unchanged.

**Recorded ABI change (D7).** An unpumped MVAU no longer has a top `ap_clk2x`
input; dotp's `ap_clk2x` pin is tied to 0. The five unpumped fingerprints
change; `structure_dump.py` shows exactly two differences per unpumped
configuration, the top ABI without `ap_clk2x` and the tie wire. The pumped
configuration's fingerprint is unchanged: its top ABI, wires and their order are
identical. The XSI engine drives `ap_clk2x` only when the top has it.

| Evidence | Result | Transcript |
|---|---|---|
| `scripts/check-kernels.sh` on the B2 tree (`d7a64b0be` + the change committed unchanged as `b5b57e1e1`, cherry-picked as `bf9dfe0c9`) | Space **427**; kernels **777 passed, 0 skipped** (+5 clock-domain tests); clean; exit 0 | `b2/gate-check-kernels.txt` |
| `scripts/check-dataflow-design.sh` | **16 passed**; clean; exit 0 | `b2/gate-check-dataflow-design.txt` |
| MVAU numeric XSI | **28/28 PASS**; the unpumped cases run without a top `ap_clk2x` | `b2/mvau-numeric-*.txt` |
| Structures against A2 (`structure_dump.py`) | 20 differing lines: for each of the five unpumped configurations, the top ABI without `ap_clk2x` and the tie wire; pumped identical | `b2/structure-diff.txt` |
| Fingerprints | five unpumped changed, `pumped-dsp58` identical (`0669b3c963c01471…`) | `b2/fingerprints.txt` |
| Keys | decision keys identical; nodes `clock`, `fast_clock` and the kernels' clocking and tie-off views added | `b2/keys-diff.txt` |

Tests (`tests/kernels/test_clock_domains.py`): an unpumped MVAU has one domain
and ties `ap_clk2x` (probe P1's `Data`-role top pin is gone); a pumped MVAU adds
the derived domain, its alignment and the two-clock reset; a kernel whose pins
follow no naming convention is driven from the domains that name its pins; a
derived clock refuses a reset pin; an input nothing drives is refused by name.

## B3. Non-stream interfaces (J4)

**Control buses** (`src/finn/kernels/control.py`, new). A `ControlBus` node
is named by the top port it presents and clocked by a `ClockDomain`. A kernel
with a control interface references it and exports, per input, the `Control`
it presents there (its bus, or none in this configuration). The node sees its
kernel through `Users(CONTROL)` and exports the bus renamed to its port
(`<port>_<MEMBER>`, target, associated with the domain's pins); `netlist` adds
it to the top ABI after the streams and wires it member by member
(`Composition.export`). One kernel per control bus.

**Tie-offs.** A kernel's `Tieoffs` hold inputs constant and leave outputs
unconnected (from B2). Thresholding uses them for AXI-Lite when its thresholds
are not runtime-writable, and for the set selector of a single set.
Runtime-writable thresholds without a control bus are refused
(`threshold-control`).

**Sidebands.** A set-index stream is an ordinary `Stream`. Thresholding with
several sets sits on a `set_stream` whose port requires one index per input
beat (`threshold-set-stream`); a single set takes none.

**Child padding.** A padded AXIS child may now feed a child: `compatibility`
no longer refuses child-to-child padding, and `Composition.connect` leaves the
producer's padding bits unconnected and drives the consumer's padding with
zeros. `UnusedOutput` gains a bit range (`offset`, `width`); validation counts
disposed bits, and lowering leaves a pin unconnected only when all its bits are
disposed.

**Thresholding in the stream idiom (the part B3 needs of C2).**
`ThresholdingAxiKernel` gains `clock`, `input_stream`, `output_stream`,
`set_stream` and `control` reference inputs and exports per-input ports,
clocking, control and tie-offs. The input port requires channels innermost
with PE per beat (`vector_major`), the output keeps the input's order. The
AXI-Lite bus and the AXIS ports become derived values shared by the module and
the ports; the module's build requirements are unchanged. Fusing it into MVAU
(C2) is not part of B3.

| Evidence | Result | Transcript |
|---|---|---|
| `scripts/check-kernels.sh` on `75595eedb` | Space **427**; kernels **784 passed, 0 skipped** (+7 interface tests, two of them XSim); clean; exit 0 | `b3/gate-check-kernels.txt` |
| `scripts/check-dataflow-design.sh` | **16 passed**; clean; exit 0 | `b3/gate-check-dataflow-design.txt` |
| dotp → thresholding in XSim (`test_interfaces.py`) | The composed module's four levels equal the thresholded dot products, with the AXI-Lite bus tied and with it exported (held idle by the testbench) | in the gate |
| MVAU numeric XSI | **28/28 PASS**, run on the B3 tree before a guard for malformed threshold tables that the first gate run found (MVAU does not import thresholding) | `b3/mvau-numeric-*.txt` |
| MVAU fingerprints and keys | identical to B2 | `b3/fingerprints.txt` |

Tests (`tests/kernels/test_interfaces.py`): probe P3 (dotp's padded INT9
result feeds thresholding; bits 9–15 unconnected, the consumer's padding
zero); probe P4 (read-only thresholds: every input driven, AXI-Lite and the
set selector tied, their outputs unconnected, nothing exported); writable
thresholds export `s_axilite` through the control node, and are refused without
one; a two-set table takes a set-selector stream and refuses a short one.

## Outcome

Phases A and B are complete; each ended green on both gates with XSim
executed, and on the 28-case MVAU numeric sweep.

| | A1 | A2 | B1 | B2 | B3 |
|---|---|---|---|---|---|
| Space tests | 423 | 423 | 427 | 427 | 427 |
| Kernel tests (0 skipped) | 763 | 763 | 772 | 777 | 784 |
| Dataflow tests | 16 | 16 | 16 | 16 | 16 |
| MVAU numeric XSI | 28/28 | 28/28 | 28/28 | 28/28 | 28/28 |
| MVAU fingerprints | = landing | all six change (sources) | = A2 | five unpumped change (`ap_clk2x`) | = B2 |

**Left for later.** FinnLib upstream PRs (and the licence headers); the
remaining stream migrations (eltwise, input generator); the query pass items
the work touched: the `when=` guards an optional stream still needs, and
`Users` through forwarding composites (a FIFO stage is clocked from its
stream's domain rather than as a user of it).
