# `finn.kernels`

Physical kernels: each binds one RTL or HLS module (mostly from FinnLib) to a
design space in `finn.core.space`, with the stream order it presents on each
port. The package never reads an ONNX graph; graph adapters live in
`finn.graph`. Experimental: its design and authoring guide are maintained
outside this repository until the package is final.

| Module | Holds |
|---|---|
| `base.py`, `composite.py`, `port.py`, `streams.py`, `adapters.py` | the kernel base, composite kernels, `AxiStreamPort`, streams and the adapters planned between them |
| `dotp.py`, `matmul.py`, `thresholding.py`, `eltwise.py`, `fifo.py`, `memstream.py`, `input_generator.py`, `transpose.py`, `vpc.py` | the kernels |
| `configure.py` | committing and settling choices by key |
| `control.py`, `target.py` | control buses, clocks and target devices |
| `datatypes/` | scalar datatype domains and semantics |
| `physical/` | physical structure, AXI-Stream contracts, composition and lowering |
| `artifacts/` | build requirements, sources, rendering, packaging and the artifact store |
| `resources/` | templates shipped with the package |

FinnLib is the `finnlib` resource (`src/finn/resources.toml`); set
`FINN_RESOURCES_FINNLIB` to work against a clone. Checks:
`bash scripts/check-kernels.sh`.
