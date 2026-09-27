# Evidence for the graph-composition design

| File | What it records |
|---|---|
| `gate-check-kernels-baseline-0d700b1ab.txt` | `scripts/check-kernels.sh` at the base revision |
| `gate-check-dataflow-design-baseline-0d700b1ab.txt` | `scripts/check-dataflow-design.sh` at the base revision |
| `gate-check-kernels-spike.txt` | `scripts/check-kernels.sh` on `spike/space-graph-composition` at `759ee0e17` |
| `gate-check-dataflow-design-spike.txt` | `scripts/check-dataflow-design.sh` on the spike |
| `fingerprints-baseline-0d700b1ab.txt` | `../fingerprints.py` at the base revision |
| `fingerprints-spike.txt` | `../fingerprints.py` on the spike |
| `fingerprints-spike-instance-renamed.txt` | `../fingerprints_renamed.py` on the spike: the netlist with node `implementation` named `weights` again |

Both gates ran with
`PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`, and with ruff
and mypy taken from `PATH`. Run the fingerprint scripts from the FINN checkout
with `PYTHONPATH=src:tests:deps/qonnx/src`.
