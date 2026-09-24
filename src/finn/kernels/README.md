# Physical kernels

The supported authoring API is `finn.kernels.space`. A kernel declares the
facts it consumes, its implementation decisions, and the typed views it can
answer. Bind Params directly, commit choices, and call the view:

```python
from finn.kernels import FifoKernel
from finn.kernels.space import Available, compile_space

fifo_model = compile_space(FifoKernel)
fifo_base = fifo_model.bind(word_bits=16, depth=32)
fifo_configuration = fifo_base.with_choices(ram_style="block")
assessment = fifo_configuration.build_requirements()
assert isinstance(assessment.accepted_result, Available)
requirements = assessment.accepted_result.value
```

The model can bind many independent configurations; each one freezes its own inputs.
Choice replacement returns immutable successors. The raw output in an assessment does
not establish that its constraints and readiness obligations are accepted.
See the [Space API guide](../../../docs/design-space.md) for signature binding,
scopes, guarded choices, atomic refinement, inspection, and sparse selections.

```text
base.py                  neutral Kernel identity and capability metadata
space/                   one language, compiler, runtime and public services
dotp.py                  activation/weights/result scopes and build requirements
mvau.py                  parent-owned folding/delivery and accepted dotp child
streaming.py             replay and initialized cyclic word delivery
target.py                DSP targets and port capacities
physical/                AXI scopes, detached packing, wiring and lowering
datatypes/               QONNX scalar semantics, admission constraints and codecs
artifacts/               requirements, source resolution, rendering and builds
resources/               corrected dotp RTL, cyclic RTL, assembly template

Space -> accepted build_requirements view -> ModuleBuildRequirements
                                         |
                                         v
                         explicit roots + artifact store -> RTL sources
```

| Declaration | File | Native interface and authoring concern |
|---|---|---|
| `FifoKernel` | `fifo.py` | Opaque unpadded words; depth and RAM-style choice |
| `InputGeneratorKernel` | `input_generator.py` | Immutable extent/stride vectors; native multi-bit loop markers |
| `ThresholdingAxiKernel` | `thresholding.py` | Threshold tables, output encoding, AXI-Lite and set selection |
| `EltwiseKernel` | `eltwise.py` | Integer/float operands, dependent type constraints, unpadded ready/valid words |
| `IntToFp32Kernel` | `int_to_fp32.py` | Combinational pins and a fixed FLOAT32 result; no clock or stream |
| `MemStreamHlsKernel` | `memstream_hls.py` | C++ type and memory/interface declarations before HLS synthesis |
| `DotpAxiKernel` | `dotp.py` | Typed AXIS interface scopes; target, pumping, segmentation and accumulator admission |

Dotp's supplied dtype handles are `DotpAxiKernel.activation.dtype`,
`DotpAxiKernel.weights.dtype`, and `DotpAxiKernel.result.dtype`. Each child
exposes narrow dtype/width fields independently of its accepted complete stream.
The interface aggregate expands through `ScopeBuilder` and ordinary
`Subspace` placement; it does not inject members into its parent class.

The [original authoring-pass notes](../../../docs/kernel-authoring-pass/README.md)
retain the adopted physical profiles and native-source findings. Their old
binding syntax and validation counts are historical evidence, not current API
instructions or a new dataflow compatibility claim.

The six RTL examples return `ModuleBuildRequirements` from their accepted
`build_requirements` View. MemStreamHLS returns `HlsSourceRequirements`: C++ interfaces and
source generation are known, while RTL pins are established by synthesis.
Render its source bundle with `finn.kernels.artifacts.hls.render_hls_sources`,
supplying the same explicit FinnLib and template roots. The returned paths are
relative to a staging directory; retain that layout and use the declared
`include_directories`. Its AXI-Lite memory and `ap_ctrl_hs` registers share the
`control` bundle; software must enable start/auto-restart for continuous output.

`MVAU` owns matrix geometry, PE/SIMD folding, result precision and weight
delivery. Its `compute` child is a bound `DotpAxiKernel`; assembly requires that
child's accepted `build_requirements` View. `mvau_assembly` binds this same Space for callers
with a complete configuration:

```python
from finn.kernels import DspBlock, WeightDelivery, mvau_assembly
from finn.kernels.datatypes.values import resolve_qonnx_datatype_name

built = mvau_assembly(
    repetitions=2,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=resolve_qonnx_datatype_name("INT3"),
    weights_dtype=resolve_qonnx_datatype_name("INT3"),
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

The experimental code under `finn.dataflow` still consumes retired runtime
interfaces and requires a separate port. Graph inference, nodeattr persistence,
source reconstruction and graph transactions are outside this kernel API.
The independent kernel gate does not claim those consumers work. The canonical
physical definitions and shared support belong here. Keep local RTL unchanged
during package changes; review
the pinned upstream source and correction recorded in `resources/dotp_axi.sv`
when updating FinnLib.

`resources/axilite.sv` contains one additional pinned correction: `snk_re` is
declared before its first use, preserving the original continuous assignment.
Thresholding uses this source because Vivado rejects the upstream declaration
order. Both local corrections record their upstream revision and hash.
