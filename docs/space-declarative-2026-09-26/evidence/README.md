# Evidence for the declarative-space spike (iteration 3)

All runs are on `spike/space-declarative-3`. The gates used
`PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`, ruff and mypy
from `PATH`, and the normal `PATH`, which includes the Xilinx 2025.2 tools
(`xvlog`, `xelab`, `xsim`), so the 9 XSim-backed kernel tests **executed**
(no Docker). Python scripts run from the FINN checkout with
`PYTHONPATH=src:tests:deps/qonnx/src`. Transcripts are `.txt` because `*.log`
is gitignored. Iteration 2's transcripts are in git history (`43d4576a6`).

| File | What it records |
|---|---|
| `gate-check-kernels.txt` | `scripts/check-kernels.sh`: Space and kernel pytest (XSim included), format, lint, strict mypy; exit status last |
| `gate-check-dataflow-design.txt` | `scripts/check-dataflow-design.sh`; exit status last |
| `xsim-tests.txt` | the 9 XSim-backed kernel tests run on their own with `-v -rA`, and the simulator version |
| `fingerprints.txt` | `../../space-graph-composition-2026-09-25/fingerprints.py`: six MVAU configurations through `mvau_assembly`; identical to iteration 2 |
| `fingerprints-renamed.txt` | `../fingerprints_renamed.py`: the same, with the cyclic instance renamed from `u_implementation_cyclic` to `u_weights`; identical to `../../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` |
| `mvau-keys.diff` | `../mvau_keys.py` at `43d4576a6` (iteration 2) and on this branch: empty, so every decision key, node key and node kind is unchanged |
| `collapse-probe.txt` | `../collapse_probe.py`: node counts and evaluated work with and without collapsed forwarding, for MVAU, the scale probe's pipeline and a pipeline of forwarding composites |
| `scale-probe.txt` | `../scale_probe.py`: the cost of one local edit in an N-node graph, with the iteration-2 probe run on iteration-2 sources in the same session |
