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
rather than `dfeafac8`'s older `dev`; the newer base adds only upstream
hardening (`dotp` parameter checks, `requantf`, `vpc`). On top:

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
