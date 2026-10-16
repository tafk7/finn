# `finn.kernels`

Physical kernels: each binds one FinnLib RTL module (or an HLS top FINN writes over a
FinnLib HLS component), or places kernel children,
in a design space in `finn.core.space`, with the order it presents on each port.
The package never reads an ONNX graph; the KernelOps, which do, live in
`finn.custom_op.kernels`. Experimental: its design and authoring guide are maintained
outside this repository until the package is final.

| Module | Holds |
|---|---|
| `base.py`, `port.py`, `channels.py`, `adapters.py` | the kernel's two bases and what each must state (`LeafKernel`: one module, its cycles its ports' beats; `ComposedKernel`: a kernel with children; the clocking; the end each port presents, `PortEnd`), the ports (`AxiStreamPort`; `native_stream`, a channel stage's on FinnLib's native pins, presenting a given sequence), the `Channel` (one kernel with children per edge, taking its two ends as facts: its plan, adapters, transport and source; its netlist, cycles, buffering and resources its members') and the adapters planned between its ends |
| `dotp.py`, `matmul.py`, `thresholding.py`, `eltwise.py`, `fifo.py`, `memstream.py`, `input_generator.py`, `transpose.py`, `vpc.py`, `pool.py`, `pwpolyf.py` | the kernels (`pool.py`: the first HLS kernel, over `pooling_stream`; `pwpolyf.py`: GELU, SiLU, Sigmoid or Tanh by piecewise polynomials on DSP58, FLOAT32 and approximate) |
| `configure.py` | committing choices by key, and naming the open and the committed ones |
| `network.py` | the network: kernels and channels by name, each its own space bound from plain facts (a kernel given bound already, by its owner's cache), wired by the ends each member presents (`Interface`) and what a kernel states for its parameter channel (`Fact`); its members' stored choices replayed as each is bound; a change (`trial`) atomic over the members it reaches, naming what replay drops; its composed module (`compose`; a kernel with children merges its netlist with the same `merged`), each member under its node. The graph binds one from a selection of KernelOp nodes (`finn.custom_op.kernels.network`) |
| `explore.py`, `fifo_sizing.py` | the DSE seam (`Seam`) over a network, its keys `(owner, key)` pairs: its one change (`set`, which drops what a change makes stale and names it), its strategies (`Pinned`, `TargetThroughput`, `MaxThroughput`, `SizeFifos`, `Rebaseline`, the debug `DebugPlaceholder`) and the operations that keep a network complete (`baseline`, `reopen`, `settle`); a channel's FIFO depth from both ends' beat patterns |
| `control.py`, `ends.py`, `utilization.py` | control buses, a boundary's end (`IodmaEnd`: what a shell offers at a free side, its contract and resources), and the kernels' cost model: a memory's primitives on the platform's fabric, its styles, the fitted counts (`utilization.py`, over `finn.platform.capabilities`' `Fabric` and `Resources`; `uram_requirements`) |
| `values/` | the values kernels hold and ports admit: the semantics of datatypes, integer vectors and tensors, and threshold tables; the `Integer` port policy |
| `transport.py` | ready/valid and AXI-Stream transports, and the contract of one channel end (`StreamContract`, with its `pace`) |
| `artifacts/` | module values (`Leaf`, a flat `Composed` netlist) and their ABI, their sources (`contributions.py`: copied files, generated data, HLS build requests), their emission, an HLS request's staging (`hls.py`), the RTL checker (and `evaluate`, the constants a module derives at elaboration), the IP-XACT packaging Tcl (`ipxact.py`), a packaged module's `interface.json` (`interface.py`) and the canonical digest (`projection.py`) |

FinnLib is the `finnlib` resource (`src/finn/resources.toml`); set
`FINN_RESOURCES_FINNLIB` to work against a clone. Checks:
`bash scripts/check-kernels.sh`.
