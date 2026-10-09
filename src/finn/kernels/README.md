# `finn.kernels`

Physical kernels: each binds one FinnLib RTL module (or an HLS top FINN writes over a
FinnLib HLS component), or places kernel children,
in a design space in `finn.core.space`, with the order it presents on each port.
The package never reads an ONNX graph; the KernelOps, which do, live in
`finn.custom_op.kernels`. Experimental: its design and authoring guide are maintained
outside this repository until the package is final.

| Module | Holds |
|---|---|
| `base.py`, `port.py`, `channels.py`, `adapters.py` | the kernel base (a leaf, or a kernel with children; its clocking), the ports (`AxiStreamPort`, `WordPort`), the `Channel` (one Space per edge: its plan, adapter, transport and source) and the adapters planned between its ends |
| `dotp.py`, `matmul.py`, `thresholding.py`, `eltwise.py`, `fifo.py`, `memstream.py`, `input_generator.py`, `transpose.py`, `vpc.py`, `pool.py` | the kernels (`pool.py`: the first HLS kernel, over `pooling_stream`) |
| `configure.py` | committing and settling choices by key |
| `explore.py`, `fifo_sizing.py` | the DSE seam (`Seam`), its strategies (`Pinned`, `TargetThroughput`, `MaxThroughput`, `SizeFifos`) and its completion policies (`Baseline`, the debug `Placeholder`); a channel's FIFO depth from both ends' beat patterns |
| `control.py`, `target.py`, `utilization.py` | control buses, and the build target kernels see (`Platform`: capabilities, clock and the part's `Resources`; `DspBlock`, `Fabric`); `finn.platform` resolves it |
| `values/` | the values kernels hold and ports admit: the semantics of datatypes, integer vectors and tensors, and threshold tables; the `Integer` port policy |
| `transport.py` | ready/valid and AXI-Stream transports, and the contract of one channel end (`StreamContract`, with its `pace`) |
| `artifacts/` | module values (`Leaf`, a flat `Composed` netlist) and their ABI, their sources (`contributions.py`: copied files, generated data, HLS build requests), their emission, an HLS request's staging (`hls.py`), the RTL checker (and `evaluate`, the constants a module derives at elaboration) and the IP-XACT packaging Tcl (`ipxact.py`) |

FinnLib is the `finnlib` resource (`src/finn/resources.toml`); set
`FINN_RESOURCES_FINNLIB` to work against a clone. Checks:
`bash scripts/check-kernels.sh`.
