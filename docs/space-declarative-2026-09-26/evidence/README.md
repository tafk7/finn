# Evidence for the declarative-space spike (iteration 2)

All runs are on `spike/space-declarative-2`. The gates used
`PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`, ruff and mypy
from `PATH`, and the normal `PATH`, which includes the Xilinx 2025.2 tools
(`xvlog`, `xelab`, `xsim`), so the 9 XSim-backed kernel tests **executed**
(no Docker). Python scripts run from the FINN checkout with
`PYTHONPATH=src:tests:deps/qonnx/src`. Transcripts are `.txt` because `*.log`
is gitignored.

| File | What it records |
|---|---|
| `gate-check-kernels.txt` | `scripts/check-kernels.sh`: Space and kernel pytest (XSim included), format, lint, strict mypy; exit status last |
| `gate-check-dataflow-design.txt` | `scripts/check-dataflow-design.sh`; exit status last |
| `xsim-tests.txt` | the 9 XSim-backed kernel tests run on their own with `-v -rA`, and the simulator version |
| `fingerprints.txt` | `../../space-graph-composition-2026-09-25/fingerprints.py`: six MVAU configurations through `mvau_assembly`; identical to iteration 1 |
| `fingerprints-renamed.txt` | `../fingerprints_renamed.py`: the same, with the cyclic instance renamed from `u_implementation_cyclic` to `u_weights`; identical to `../../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` |
| `mvau-keys.diff` | `../mvau_keys.py` at `35e59b442` (iteration 1) and on this branch: decision keys identical, node-key changes listed |
| `scale-probe.txt` | `../scale_probe.py`: the cost of one local edit in an N-node graph, with the iteration-1 probe run in the same session |

The XSim tests are the native stream, FIFO, flat-kernel and eltwise+cyclic
constant RTL checks (`test_flat_kernels.py` x6, `test_native_streams.py`,
`test_stream_contract.py`, `test_streaming_components.py`). The eltwise test
drives a `CyclicDelivery`, whose family gained a stream input and a `PORTS`
export in this iteration.
