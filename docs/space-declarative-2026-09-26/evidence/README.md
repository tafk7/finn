# Evidence for the declarative-space spike (iteration 4)

Evidence for the landing on `feature/kernel-package-extraction` (gates, XSim,
fingerprints and the MVAU numeric XSI sweep) is in [`landing/`](landing/),
described in [`../LANDING.md`](../LANDING.md) section 2.

All runs are on `spike/space-declarative-4`. The gates used
`PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`, ruff and mypy
from `PATH`, and the normal `PATH`, which includes the Xilinx 2025.2 tools
(`xvlog`, `xelab`, `xsim`), so the 9 XSim-backed kernel tests **executed**
(no Docker). Python scripts run from the FINN checkout with
`PYTHONPATH=src:tests:deps/qonnx/src`. The iteration-3 baseline is
`426882411`, extracted with `git archive 426882411 src docs tests` and put
first on `PYTHONPATH`. Transcripts are `.txt` because `*.log` is gitignored.
Earlier iterations' transcripts are in git history (iteration 3 at
`426882411`, iteration 2 at `43d4576a6`).

| File | What it records |
|---|---|
| `gate-check-kernels.txt` | `scripts/check-kernels.sh`: Space and kernel pytest (XSim included), format, lint, strict mypy; exit status last |
| `gate-check-dataflow-design.txt` | `scripts/check-dataflow-design.sh`; exit status last |
| `xsim-tests.txt` | the 9 XSim-backed kernel tests run on their own with `-v -rA`, and the simulator version |
| `fingerprints.txt` | `../../space-graph-composition-2026-09-25/fingerprints.py`: six MVAU configurations through `mvau_assembly`; identical to iteration 3 |
| `fingerprints-renamed.txt` | `../fingerprints_renamed.py`: the same, with the cyclic instance renamed from `u_implementation_cyclic` to `u_weights`; identical to `../../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` |
| `mvau-keys.diff` | `../mvau_keys.py` at `426882411` (iteration 3) and on this branch: empty, so every decision key, node key and node kind is unchanged |
| `mvau-abi-ports.txt` | the top-level ABI port names of the six fingerprinted configurations, identical at `426882411` and on this branch |
| `collapse-probe.txt` | `../collapse_probe.py`: node counts and evaluated work with and without collapsed forwarding, for MVAU, the scale probe's pipeline and a pipeline of forwarding composites |
| `scale-probe.txt` | `../scale_probe.py`: the cost of one local edit in an N-node graph, with the iteration-3 probe run on iteration-3 sources in the same session |
