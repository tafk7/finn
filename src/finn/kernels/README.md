# `finn.kernels`

Physical kernels: each binds one FinnLib RTL module, or places kernel children,
in a design space in `finn.core.space`, with the stream order it presents on each
port. The package never reads an ONNX graph; the KernelOps, which do, live in
`finn.custom_op.kernels`. Experimental: its design and authoring guide are maintained
outside this repository until the package is final.

| Module | Holds |
|---|---|
| `base.py`, `port.py`, `channels.py`, `adapters.py` | the kernel base (a leaf, or a kernel with children; its clocking), `AxiStreamPort`, the `Channel` (one Space per edge: its plan, adapter, transport and source) and the adapters planned between them |
| `dotp.py`, `matmul.py`, `thresholding.py`, `eltwise.py`, `fifo.py`, `memstream.py`, `input_generator.py`, `transpose.py`, `vpc.py` | the kernels |
| `configure.py` | committing and settling choices by key |
| `control.py`, `target.py` | control buses, and target DSP blocks |
| `datatypes/` | scalar datatype domains and semantics |
| `transport.py` | ready/valid and AXI-Stream transports, and stream contracts |
| `artifacts/` | module values (`Leaf`, a flat `Composed` netlist), their emission and the RTL checker |

FinnLib is the `finnlib` resource (`src/finn/resources.toml`); set
`FINN_RESOURCES_FINNLIB` to work against a clone. Checks:
`bash scripts/check-kernels.sh`.
