# Evidence for the design-graph spike

All runs are on `spike/space-design-graph` at `0acc51e1e`.

| File | What it records |
|---|---|
| `gate-check-kernels-spike.txt` | `scripts/check-kernels.sh`: Space 314, kernels 758; format, lint and strict mypy clean; exit 0 |
| `gate-check-dataflow-design-spike.txt` | `scripts/check-dataflow-design.sh`: 16 passed, exit 0 |
| `doc-examples-spike.txt` | `scratchpad/space/check-examples.py --finn-root <checkout>`: 18 of 18 |
| `fingerprints-spike.txt` | `../../space-graph-composition-2026-09-25/fingerprints.py` |
| `fingerprints-spike-instance-renamed.txt` | `../fingerprints_renamed.py`: identical to `../../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` |
| `scale-probe.txt` | `../scale_probe.py`: the cost of one local edit in an N-node graph |

The gates ran with `PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`,
and with ruff and mypy taken from `PATH`. The Python scripts run from the FINN
checkout with `PYTHONPATH=src:tests:deps/qonnx/src`.
