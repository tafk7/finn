# Evidence for the declarative-space spike

All runs are on `spike/space-declarative` with the code at `335535a6c`. The
gates used `PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python`,
ruff and mypy from `PATH`, and a `PATH` with the Xilinx tools removed, so that
no XSim run starts (see DESIGN.md §9). Python scripts run from the FINN
checkout with `PYTHONPATH=src:tests:deps/qonnx/src`.

| File | What it records |
|---|---|
| `gate-check-kernels.txt` | `scripts/check-kernels.sh`: Space and kernel pytest, format, lint, strict mypy; exit status last |
| `gate-check-dataflow-design.txt` | `scripts/check-dataflow-design.sh`; exit status last |
| `fingerprints.txt` | `../../space-graph-composition-2026-09-25/fingerprints.py`: six MVAU configurations through `mvau_assembly` |
| `fingerprints-renamed.txt` | `../fingerprints_renamed.py`: the same, with the cyclic instance renamed from `u_implementation_cyclic` to `u_weights`; identical to `../../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` |
| `scale-probe.txt` | `../scale_probe.py`: the cost of one local edit in an N-node graph |
