# Physical kernels

The supported authoring API is `finn.core.space`. A kernel declares the
facts it consumes, its implementation decisions, and the typed views it can
answer. Calling a kernel family with its facts declares a node;
`design_space(node)` is the one compile step and returns its initial
configuration. Commit choices on that configuration and read the view: a view
reads as its accepted value, like any member, and its assessment is an explicit
`inspect` of the view declaration:

```python
from finn.kernels import FifoKernel
from finn.core.space import Available, design_space

fifo_base = design_space(FifoKernel(word_bits=16, depth=32))
fifo_configuration = fifo_base.with_choices(ram_style="block")
requirements = fifo_configuration.build_requirements
assessment = fifo_configuration.inspect(FifoKernel.build_requirements)
assert assessment.accepted_result == Available(requirements)
assert fifo_configuration.query(FifoKernel.build_requirements) == Available(requirements)
assert fifo_configuration.field(FifoKernel.ram_style).get() == "block"
```

Each `design_space` call returns an independent configuration that freezes its own
inputs; configurations of the same family share its compiled model. Choice
replacement returns immutable successors. Reading a view that is not accepted
raises `ValueUnavailableError` carrying its result (`query` returns that result
instead). The raw output in an assessment does not establish that its
constraints and readiness obligations are accepted.
See the [Space API guide](../../../../scratchpad/space/AUTHORING.md) for ordinary self methods,
node declarations, Decisions over nodes, atomic refinement, inspection, and sparse selections.
The [Space design](../../../../scratchpad/space/DESIGN.md),
[internals](../../../../scratchpad/space/INTERNALS.md), and
[migration notes](../../../../scratchpad/space/MIGRATION.md) are maintained in
that separate repository while experimental.

The [kernel and artifact integration refactoring spec](../../../docs/kernel-artifact-integration-2026-09-25/SPEC.md)
records the planned declaration, source-preparation, composition, and provenance
work while preserving the independent Space and artifact systems. It describes
planned changes, not additional APIs already delivered here.

```text
base.py                  neutral Kernel identity and capability metadata
dotp.py                  operand scalars, AXIS ports and build requirements
mvau.py                  folding, dotp child node, and a weight-delivery Decision over nodes
streaming.py             replay and initialized cyclic word delivery
target.py                DSP targets and port capacities
physical/                typed native/AXIS ports, detached packing, wiring and lowering
datatypes/               QONNX identity, integer policies, scalar Spaces and codecs
artifacts/               requirements, source resolution, rendering and builds
resources/               cyclic RTL and source-generation templates

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
| `EltwiseKernel` | `eltwise.py` | Integer/float operand scalars, dependent type constraints, typed unpadded ports |
| `IntToFp32Kernel` | `int_to_fp32.py` | Combinational pins and a fixed FLOAT32 result; no clock or stream |
| `MemStreamHlsKernel` | `memstream_hls.py` | C++ type and memory/interface declarations before HLS synthesis |
| `DotpAxiKernel` | `dotp.py` | Typed AXIS ports; target, pumping, segmentation and accumulator admission |

Operand datatypes are ordinary kernel Params, such as
`DotpAxiKernel.activation_dtype`. Each operand has its own scalar node
(`activation_type = integer_scalar(activation_dtype, ...)`) that owns its
admission, and each port binds to that scalar's raw dtype and accepted encoding. Ports expose narrow dtype, width and packing fields
independently of their accepted `stream` view, which requires the scalar:

```python
from finn.kernels import DotpAxiKernel, DspBlock
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype

dotp = design_space(
    DotpAxiKernel(
        activation_dtype=dtype("INT3"),
        weights_dtype=dtype("INT3"),
        result_dtype=dtype("INT8"),
        pe=2,
        simd=2,
        target_dsp=DspBlock.DSP48E2,
        segment_length=0,
    )
)
assert dotp.activation.carrier_bits == 8
assert dotp.activation_type.encoding.bits == 3
assert dotp.activation.stream.payload_bits == 6
```

The same policy constrains input and caller-selected output encodings.

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

`MVAU` owns matrix geometry, PE/SIMD folding and result precision, and declares
its connections as streams (`finn.kernels.streams`). It derives a `StreamSpec`
(element, traversal, repetition, markers) for each stream; kernels reference
the streams they sit on through reference inputs and export one port contract
per input (`exports = {PORT: {activation_stream: activation_port, ...}}`):

```text
in0_V ─activations─► replay ─replayed─► compute (dotp) ─results─► out0_V
                                           ▲
       implementation ─── weight_stream ───┘   (buffered: direct | fifo)
       external: boundary in1_V  |  cyclic: CyclicDelivery (rom_style, weights)
```

A stream sees each user's port on that stream only (`Users(PORT)`), so a port's
refusal names its own stream and independent streams settle independently.
Ports check what they read: dotp refuses a stream whose lanes, lane order,
column walk or frame rows differ from its PE/SIMD reading, rather than adopting
the stream's form. Every stream owns a `compatible` constraint, and its accepted
`connection` feeds the parent's `structure` view: `netlist` wires
`Members(MODULE)` through `Members(CONNECTION)`, routes clocks and resets, and
turns boundary streams into AXIS. `build_requirements` lowers that structure.
The `implementation` Decision places either nothing (`external`: the weight
stream has one user and is the boundary `in1_V`) or its `cyclic`
CyclicDelivery candidate, named `implementation.cyclic`; only the selected
candidate is evaluated. A `BufferedStream` owns a `transport`
Decision over nodes: `direct`, or a `fifo` candidate whose depth and memory style
are its own decisions. Whether a FIFO is needed and how deep is a compiler
decision; the stream only provides the slot. `commit` (from
`finn.kernels.configure`) commits choices by their inspection keys in one atomic
batch on a configured point:

```python
from finn.core.space import Unresolved, selections
from finn.kernels import MVAU
from finn.kernels.configure import commit

facts = dict(
    repetitions=2,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=dtype("INT3"),
    weights_dtype=dtype("INT3"),
    target_dsp=DspBlock.DSP48E2,
    segment_length=0,
)
identity = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
point = commit(
    design_space(MVAU(**facts, weights=identity)),
    {
        "implementation": "cyclic",
        "implementation.cyclic.rom_style": "block",
        "weight_stream.transport": "fifo",
        "weight_stream.transport.fifo.buffer.depth": 16,
        "weight_stream.transport.fifo.buffer.ram_style": "auto",
        "compute.compute_pumping": False,
        "pe": 2,
        "simd": 2,
    },
)
structure = point.structure.structure
assert [item.instance_id for item in structure.instances] == [
    "u_replay",
    "u_compute",
    "u_implementation_cyclic",
    "u_weight_stream_fifo",
]
assert point.build_requirements == point.structure.requirements
saved = selections.capture(point)
replayed = selections.restore(design_space(MVAU(**facts)), saved).instance  # weights omitted
assert isinstance(replayed.query(MVAU.structure), Unresolved)
```

Changing a selector does not discard the old case's choices: clear
`rom_style` (or a FIFO's depth and memory style) in the same batch when
switching away from that case. The
`mvau_assembly` adapter configures this same family and commits every choice
for callers with a complete configuration:

```python
from finn.kernels import WeightDelivery, mvau_assembly

built = mvau_assembly(
    repetitions=2,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=dtype("INT3"),
    weights_dtype=dtype("INT3"),
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP48E2,
    weight_delivery=WeightDelivery.CYCLIC,
    weights=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
)
```

`built.structure` exposes wiring, `built.initializer` contains packed cyclic
weights, and `built.requirements` is the artifact handoff. External weight
delivery uses `WeightDelivery.EXTERNAL` and omits `weights`; `rom_style`
(default `auto`) applies only to cyclic delivery, and `weight_fifo_depth`
places a FIFO on the weight stream.

Pass source roots explicitly to `finn.kernels.artifacts.build.prepare_module_build`:
`roots={"kernels": resource_root(), "finnlib": finnlib_root}` and
`template_roots=(template_root(),)`, importing both helpers from
`finn.kernels.resources`. `finnlib_root` is a `Path` to the pinned FinnLib checkout;
`blobs` is an `ArtifactStore`. `materialize_module_sources(prepared, store)` then
produces the complete source set. Both resource helpers work from an installed
package. Local source paths are relative to the resource directory; no source
checkout layout is assumed.

Run `scripts/check-kernels.sh` from the repository root for the independent
generic Space and kernel code checks. From scratchpad, run
`python space/check-examples.py --finn-root /path/to/finn-checkout`
for executable documentation examples. The generic package has its own
`py.typed` marker and uses the declared
`greenlet==3.2.4` runtime dependency. Explicit XSI checks live in `tests/kernels/rtlsim`; run, for
example, `python -m kernels.rtlsim.mvau_assembly_numeric --case packed` with
`PYTHONPATH=src:tests:deps/qonnx/src`, `FINN_ROOT`, `FINNLIB_ROOT` and the Vivado
library path configured. `pure_dot_product_numeric --stress` exercises sustained
one-beat reductions with long output stalls.

`finn.kernels` builds on the canonical logical values in `finn.dataflow`
(Regions, Networks, maps, validation and the QONNX datatype boundary), which in
turn depend only on `finn.core.space`. The retired dataflow implementation lives
in `finn.parked`, outside every gate; nothing live imports it. Graph inference, nodeattr persistence,
source reconstruction and graph transactions are outside this kernel API.
The independent kernel gate does not claim those consumers work. The canonical
physical definitions and shared support belong here.

The source baseline is FinnLib's grouped layout (`rtl/{arith,infra,linalg,
nonlin,shape}/`, `hls/{infra,util}/`) at the `fetch-repos.sh` pin
`11b5c64b6ddb2c89895cc539eb059e49ecf80630` (branch
`kernels/consolidated-20260927` on the `tkeller/finnlib` fork): upstream `dev`
plus `replay_buffer`, the dotp output-buffer and AXI-Lite declaration-order
corrections, and `memstream`/`memstream_axi` ported from `finn-rtllib`. There
are no private copies in `resources`. Eltwise's source closure includes the
consolidated `fifo`, which replaced `queue`. Record and validate source
revisions when updating this dependency; matching filenames do not establish
compatibility.

`Integer(...).domain()` and `integer_scalar(dtype, Integer(...))` share one
policy for owned dtype choices and supplied facts. `integer_scalar` returns an
`IntegerScalar` (or `BoundedIntegerScalar`) node declaration whose policy bounds
are ordinary bindings, including references to parent fields. Each admission rule is a
separate constraint, so type-family refusals remain visible while a dynamic bit
bound is unresolved. `Scalar` itself admits any positive-width encoding; a
kernel extends it by subclassing, as `EltwiseOperand` does for FLOAT32 or
integer operands. Conversion, thresholding and every typed port consume the
accepted `encoding` view before building pins.

`ReadyValidStream` describes native transfer pins, markers and clock/reset
associations. `pins()` preserves their exact widths. `axis_bus()` requires
byte-aligned data and at most one LAST marker; it does not pad words or relabel
loop/replay completion markers. `AxiStream` uses this same transport lowering
while retaining its typed packing. Typed ports (`native_stream`, `axi_stream`)
produce these records from lanes of an accepted scalar. Opaque words need no
scalar: FIFO, input generation and replay construct `ReadyValidStream` values
directly rather than publishing unpadded words as AXI buses.

A `StreamContract` (`physical/contract.py`) adds the logical sequence to a
transport: the element encoding, a `Traversal` (`physical/forms.py`), a
`Repetition` (`ONCE`, or `CYCLIC` for a free-running source) and periodic marker
rules (`Every(k)`). A traversal is a loop nest over the row-major operand:
`beat_loops` step from beat to beat, `lane_loops` from field to field (field
zero is least significant), and a stride of zero replays positions. Tiles,
chunked tiles, transposes and replay are all loop nests; `vector_major` is
FINN's default order. Traversals are canonical, so equal values present equal
sequences.

`classify(source, sink)` names what a mismatch needs:

| Adaptation | Meaning | Realized by |
|---|---|---|
| `identity` | same sequence | nothing |
| `lane_permutation` | same positions per beat, other field order | wires (free) |
| `reorder` | same lanes, other beat order or replay | `input_gen` / outer shuffle, with derived `DIMS`/`COEFS` |
| `width_conversion` | same element order, other lane count | data-width converter |
| `lane_regroup` | the lane axis changes | inner shuffle (banked transpose) |
| `incompatible` | different positions | nothing |

`compatibility` accepts the first two and refuses the rest, naming the adapter.
`Composition.connect` checks clock domains too, then emits every data (with any
lane permutation), padding, handshake and marker wire. The derived reorders
reproduce the tiled MVU's two hard-coded `input_gen` stages and FINN's
OuterShuffle coefficients exactly (see `tests/kernels/test_stream_contract.py`).

`CyclicDelivery` streams a constant integer operand in whatever traversal its
consumer reads, from an initialized ROM, so the consumer's order needs no
adapter. Its `output` view is a cyclic stream contract; the same kernel feeds
MVAU weight tiles or an eltwise channel vector, inside an operation kernel or
beside one:

```python
from finn.kernels import CyclicDelivery
from finn.kernels.physical.forms import Adaptation, Repetition, classify, vector_major

channels = vector_major((4,), 2)
vector = design_space(CyclicDelivery(dtype=dtype("INT4"), form=channels, values=(1, -2, 7, -8)))
rhs = vector.with_choices(rom_style="distributed")
assert rhs.output.repetition is Repetition.CYCLIC
assert rhs.image == (0xE1, 0x87)
pixels = vector_major((3, 4), 2)  # three pixels of four channels, two lanes
assert classify(channels.repeated(3), channels.repeated(3)).adaptation is Adaptation.IDENTITY
assert classify(vector_major((3, 4), 4), pixels).adaptation is Adaptation.WIDTH_CONVERSION
```

FIFO's `ram_style` remains a native preference. Its accepted `storage` view
reports both the effective backing and capacity, including output storage.
For example, `design_space(FifoKernel(word_bits=13, depth=2)).with_choices(ram_style="ultra")`
reports shift storage and capacity five. This is native implementation
information, not a synthesis resource measurement.

See the [refinement review](../../../docs/kernel-refinement-2026-09-25/REVIEW.md)
for the scalar/stream constructs and source baseline, and the
[composition review](../../../docs/kernel-composition-2026-09-25/REVIEW.md) for
the port authoring comparison, the delivery choice, and remaining work.
