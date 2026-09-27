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
