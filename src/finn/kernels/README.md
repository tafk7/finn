# Physical kernels

Start with `dotp.py`: `DotpAxiKernel` declares inputs, the pumping decision,
interfaces, constraints and a physical `View` in one class. Bind it through
`Space`/`Subspace` from `finn.kernels.space`, select decisions, and consume
`point.physical.accepted_answer`. A raw `codegen` answer does not establish that
the selected implementation is accepted.

```text
dotp.py                  flat DotpAxiKernel and physical View
mvau.py                  MVAU Space, accepted dotp child, explicit wiring
streaming.py             replay and initialized cyclic word delivery
target.py                DSP targets and port capacities
base.py                  neutral Kernel base
space/ + _engine/        declarations, binding, decisions and assessment
physical/                AXI declarations, packing, wiring and lowering
datatypes/               QONNX scalar values and their constraints/codecs
artifacts/               requirements, source resolution, rendering and builds
resources/               corrected dotp RTL, cyclic RTL, assembly template

Space -> accepted physical View -> ModuleBuildRequirements
                                         |
                                         v
                         explicit roots + artifact store -> RTL sources
```

The flat authoring comparison now also includes:

| Declaration | File | Native interface and authoring concern |
|---|---|---|
| `FifoKernel` | `fifo.py` | Opaque unpadded words; depth and RAM-style choice |
| `InputGeneratorKernel` | `input_generator.py` | Immutable extent/stride vectors; native multi-bit loop markers |
| `ThresholdingAxiKernel` | `thresholding.py` | Integer threshold tables, derived output encoding, AXI-Lite and set selection |
| `EltwiseKernel` | `eltwise.py` | Two integer/float operands, dependent type constraints, unpadded ready/valid words |
| `IntToFp32Kernel` | `int_to_fp32.py` | Combinational pins and a fixed FLOAT32 result; no clock or stream |
| `MemStreamHlsKernel` | `memstream_hls.py` | C++ type and memory/interface declarations before HLS synthesis |

Together with `DotpAxiKernel`, these are seven independent, flat declarations.
ReplayBuffer remains a companion example. No new interface-authoring DSL or
kernel-composition hierarchy is introduced by this pass. See
[`docs/kernel-authoring-pass/README.md`](../../../docs/kernel-authoring-pass/README.md)
for the supported profiles, native-source findings, and the comparison to use
when refining the authoring API.

The six RTL examples return `ModuleBuildRequirements` from their accepted
physical View. MemStreamHLS returns `HlsSourceRequirements`: C++ interfaces and
source generation are known, while RTL pins are established by synthesis.
Render its source bundle with `finn.kernels.artifacts.hls.render_hls_sources`,
supplying the same explicit FinnLib and template roots. The returned paths are
relative to a staging directory; retain that layout and use the declared
`include_directories`. Its AXI-Lite memory and `ap_ctrl_hs` registers share the
`control` bundle; software must enable start/auto-restart for continuous output.

`MVAU` owns matrix geometry, PE/SIMD folding, result precision and weight
delivery. Its `compute` child is a bound `DotpAxiKernel`; assembly requires that
child's accepted physical View. `mvau_assembly` binds this same Space for callers
with a complete configuration:

```python
from qonnx.core.datatype import DataType
from finn.kernels import DspBlock, WeightDelivery, mvau_assembly

built = mvau_assembly(
    repetitions=2,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP48E2,
    weight_delivery=WeightDelivery.CYCLIC,
    weights=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
)
```

`built.structure` exposes wiring, `built.initializer` contains packed cyclic
weights, and `built.requirements` is the artifact handoff. External weight
delivery uses `WeightDelivery.EXTERNAL` and omits `weights`.

Pass source roots explicitly to `finn.kernels.artifacts.build.prepare_module_build`:
`roots={"kernels": resource_root(), "finnlib": finnlib_root}` and
`template_roots=(template_root(),)`, importing both helpers from
`finn.kernels.resources`. `finnlib_root` is a `Path` to the pinned FinnLib checkout;
`blobs` is an `ArtifactStore`. `materialize_module_sources(prepared, store)` then
produces the complete source set. Both resource helpers work from an installed
package. Local source paths are relative to the resource directory; no source
checkout layout is assumed.

Run `scripts/check-kernels.sh` from the repository root for the independent
kernel checks. Explicit XSI checks live in `tests/kernels/rtlsim`; run, for
example, `python -m kernels.rtlsim.mvau_assembly_numeric --case packed` with
`PYTHONPATH=src:tests:deps/qonnx/src`, `FINN_ROOT`, `FINNLIB_ROOT` and the Vivado
library path configured. `pure_dot_product_numeric --stress` exercises sustained
one-beat reductions with long output stalls.

The older code under `finn.dataflow.kernels` serves modeling and compiler
integration experiments. The canonical physical definitions and shared support
belong here. Keep the local RTL byte-preserved during package changes; review
the pinned upstream source and correction recorded in `resources/dotp_axi.sv`
when updating FinnLib.

`resources/axilite.sv` contains one additional pinned correction: `snk_re` is
declared before its first use, preserving the original continuous assignment.
Thresholding uses this source because Vivado rejects the upstream declaration
order. Both local corrections record their upstream revision and hash.
