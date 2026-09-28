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
base.py                  the Kernel protocol: module, ports' buses, parameters, clocking, admission
port.py                  Port nodes: one stream interface each (element admission, sequence, pins)
dotp.py                  dotp_axi on three ports, one kernel per compute core, its own folds
matmul.py                MatMulKernel: facts m, n, k and form; compute and memory Decisions
rom.py                   RomKernel: a stored operand streamed cyclically from a ROM
streaming.py             initialized cyclic word delivery
memstream.py             FinnLib memstream_axi as a weight memory (writable, sets, INIT_FILE)
streams.py               Stream: tensor, ends, plan, adapter and transport Decisions; netlist
adapters.py              a stream's adapter candidates: input_gen / vpc chains carrying out a plan
transpose.py             FinnLib inner_shuffle, placed explicitly between two streams
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
| `InputGeneratorKernel` | `input_generator.py` | Immutable extent/stride vectors; native multi-bit loop markers (the flat module; on a stream it is an adapter stage) |
| `MemStreamKernel` | `memstream.py` | A stored operand in its consumer's order; RAM style, pumped memory, AXI-Lite, set selection |
| `TransposeKernel` | `transpose.py` | `inner_shuffle` between two streams (not a stream adapter candidate: FinnLib defect under bursty input) |
| `ThresholdingAxiKernel` | `thresholding.py` | Threshold tables, output encoding, AXI-Lite and set selection |
| `EltwiseKernel` | `eltwise.py` | Integer/float operand scalars, dependent type constraints, typed unpadded ports |
| `IntToFp32Kernel` | `int_to_fp32.py` | Combinational pins and a fixed FLOAT32 result; no clock or stream |
| `MemStreamHlsKernel` | `memstream_hls.py` | C++ type and memory/interface declarations before HLS synthesis |
| `PackedDotpKernel`, `Int8Dsp58DotpKernel` | `dotp.py` | One kernel per `dotp_axi` compute core over the shared `DotpAxiKernel`: three ports, its own PE/SIMD/pumping; target, form, segmentation and accumulator admission |
| `RomKernel` | `rom.py` | A stored operand in its consumer's order, streamed cyclically; ROM style |

A kernel on the protocol (`base.py`) declares its RTL `module`, `sources()`,
`parameters()`, its `clocking` when not plain `ap_clk`/`ap_rst_n`, one `Port`
node per stream interface, and an `admission` group. The base derives the
module's ABI (clocking, then every port's bus), `build_requirements` (accepted
under `admission`), `tieoffs` and the `MODULE`/`TIEOFFS` exports. A port
(`port.py`) takes its element from the stream it sits on and admits it by an
integer policy; a `ScheduledPort` presents its kernel's schedule through the
indices it reads. A core therefore sits between streams, whose tensors give
its elements and extents; its folds are its own Decisions:

```python
from finn.core.space import Space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels import DspBlock, PackedDotpKernel
from finn.kernels.configure import commit
from finn.kernels.streams import Stream


class Placed(Space):
    x = Stream(tensor=Tensor((1, 2), ScalarEncoding(dtype("INT3"))), port="in0_V")
    w = Stream(tensor=Tensor((2, 2), ScalarEncoding(dtype("INT3"))), port="in1_V")
    y = Stream(tensor=Tensor((1, 2), ScalarEncoding(dtype("INT8"))), port="out0_V")
    dotp = PackedDotpKernel(
        target_dsp=DspBlock.DSP48E2, target_period_ns=5.0, x_stream=x, w_stream=w, y_stream=y
    )


dotp = commit(
    design_space(Placed()), {"dotp.pe": 2, "dotp.simd": 2, "dotp.compute_pumping": False}
).dotp
assert dotp.x.axis.carrier_bits == 8
assert dotp.x.element.bits == 3
assert dotp.x.axis.payload_bits == 6
```

The same policy constrains input and result encodings.

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

`MatMulKernel` owns the operation's facts (the extents `m`, `n` and `k`; the
`form`, `DENSE` or `DEPTHWISE`; weights stored `(k, n)`) and result precision,
and declares its connections as streams (`finn.kernels.streams`). Each stream
carries a `Tensor` (shape and element encoding, `finn.dataflow.tensor`);
kernels reference the streams they sit on through reference inputs, and each
of their ports exports its contract for its stream (`exports = {PORT:
{stream: contract}}`), presenting the end's own traversal of the tensor (a
`BeatSequence`: traversal, repetition, markers) derived from its kernel's
schedule (`finn.dataflow.schedule`: `n` folded by PE, `k` by SIMD). A boundary
stream presents what its internal end presents, without the replay the
receiver realizes and without markers. Each slot is a node, a Decision over
kernels, or a derived node, and each Decision is present only where its case
applies:

```text
in0_V ─activations─[adapter: input_gen]─► compute ─results─► out0_V
          dense: replay + frame                 ▲
          depthwise: frame only                 └── weight_stream ── memory
   compute: packed (dotp) | int8_dsp58 (dotp_8sx9_dsp58); each owns pe, simd,
            compute_pumping
   depthwise: realization: native | dense (block-diagonal weights)
   memory (optional): none (in1_V) | rom (rom_style) | memstream (RAM: ram_style,
             pumped_memory; writable_weights → s_axilite; weight_sets > 1 → in2_V)
   weight_stream: transport: direct | fifo
```

A stream compares what its source presents with what its sink requires and
derives a `plan` (`finn.dataflow.plan`): empty when the two connect directly
(a field permutation is wires), otherwise reorders (replay included), width
conversions and marker synthesis. A non-empty plan opens the stream's
`adapter` Decision over seven fixed chains of FinnLib `input_gen` and `vpc`
(`finn.kernels.adapters`); each refuses a plan it does not carry out, so one
survives, and `settle` (`finn.kernels.configure`) commits it. `adapter_ram_style` chooses the
`input_gen`'s memory. A stream constructed with `adaptable=False` admits no
adapter and refuses a non-empty plan (`stream-plan`). Between two kernels the
same stream joins independently folded ends: `tests/kernels/test_two_kernels.py`
joins a PE = 4 producer to a SIMD = 2 consumer through `vpc` and `input_gen`.

A stream sees each user's port on that stream only (`Users(PORT)`), so a port's
refusal names its own stream and independent streams settle independently.
Ports present what their kernel's schedule derives: dotp takes its extents
from its streams and folds them by its own PE and SIMD, so a form it cannot
read is never handed to it. A stream end presented by a kernel's port belongs
to that kernel's instance in `netlist`.
Every stream owns `well_formed`, `realizable` and `compatible` constraints, and its accepted
`connection` feeds the parent's `structure` view: `netlist` wires
`Members(MODULE)` through `Members(CONNECTION)`, drives every child clock and
reset pin by its declared role (`ap_clk`, `ap_clk2x` for a clock at twice
`ap_clk`, `ap_rst_n`), holds tied-off inputs (`Members(TIEOFFS)`), exports
control buses (`Members(EXPORTED)`), and turns boundary streams into AXIS. The
module has `ap_clk2x` only when a child needs it: an unpumped MatMulKernel has none. A
kernel generates one module and exposes its interface; wiring that module's
instance into a design is the consumer's. `build_requirements` lowers that structure.
The optional `memory` Decision places either nothing (`none`: the weight
stream has one user and is the boundary `in1_V`) or one of its candidates,
the `rom` RomKernel (`memory.rom`) or the `memstream` MemStreamKernel; only
the selected candidate is evaluated. Compatibility filters the candidates
(a depthwise form is read natively only by the INT8 DSP58 core; a ROM refuses
runtime-writable weights or several sets) and `settle` commits a Decision
with one compatible candidate; where several remain, the choice is the
caller's. A `BufferedStream` owns a `transport`
Decision over nodes: `direct`, or a `fifo` candidate whose depth and memory style
are its own decisions. Whether a FIFO is needed and how deep is a compiler
decision; the stream only provides the slot. `commit` (from
`finn.kernels.configure`) commits choices by their inspection keys in one atomic
batch on a configured point:

```python
from finn.core.space import Unresolved, selections
from finn.kernels import MatMulKernel
from finn.kernels.configure import commit, settle

facts = dict(
    m=2,
    k=4,
    n=4,
    activation_dtype=dtype("INT3"),
    weights_dtype=dtype("INT3"),
    target_dsp=DspBlock.DSP48E2,
    target_period_ns=5.0,
)
identity = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
point = commit(
    design_space(MatMulKernel(**facts, weights=identity)),
    {
        "memory": "rom",
        "memory.rom.rom_style": "block",
        "weight_stream.transport": "fifo",
        "weight_stream.transport.fifo.buffer.depth": 16,
        "weight_stream.transport.fifo.buffer.ram_style": "auto",
        "compute": "packed",
        "compute.packed.compute_pumping": False,
        "compute.packed.pe": 2,
        "compute.packed.simd": 2,
    },
)
# The activation stream's plan (replay and frame) needs an adapter: settle the one,
# then choose its memory.
point = commit(settle(point).point, {"activations.adapter_ram_style": "auto"})
structure = point.structure.structure
assert [item.instance_id for item in structure.instances] == [
    "u_compute_packed",
    "u_memory_rom",
    "u_activations_input_gen",
    "u_weight_stream_fifo",
]
assert point.build_requirements == point.structure.requirements
saved = selections.capture(point)
replayed = selections.restore(
    design_space(MatMulKernel(**facts)), saved
).instance  # weights omitted
assert isinstance(replayed.query(MatMulKernel.structure), Unresolved)
```

Changing a selector does not discard the old case's choices: clear
`rom_style` (or a FIFO's depth and memory style) in the same batch when
switching away from that case. The
`matmul_assembly` adapter configures this same family and commits every choice
for callers with a complete configuration:

```python
from finn.kernels import WeightDelivery, matmul_assembly

built = matmul_assembly(
    m=2,
    k=4,
    n=4,
    activation_dtype=dtype("INT3"),
    weights_dtype=dtype("INT3"),
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP48E2,
    weight_delivery=WeightDelivery.CYCLIC,
    weights=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
)
```

`built.structure` exposes wiring, `built.initializer` contains packed stored
weights, and `built.requirements` is the artifact handoff. External weights
use `WeightDelivery.EXTERNAL` (the `memory` Decision's `none`) and omit
`weights`; `rom_style` (default `auto`) applies only to the ROM, and
`weight_fifo_depth` places a FIFO on the weight stream.

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
example, `python -m kernels.rtlsim.matmul_numeric --case packed` with
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
`b9262df1ba4ee7623f0bbd996e2c7566c411bc5f` (branch
`kernels/matmul-20260927` on the `tkeller/finnlib` fork): upstream `dev`
plus `replay_buffer` (no longer wrapped), the dotp output-buffer and AXI-Lite declaration-order
corrections, `memstream`/`memstream_axi` ported from `finn-rtllib`, and the
`dotp_axi` `CORE` parameter that lets each core be its own kernel. There
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
scalar: FIFO, input generation and width conversion construct `ReadyValidStream` values
directly rather than publishing unpadded words as AXI buses.

A `StreamContract` (`physical/contract.py`) adds the logical sequence to a
transport: the element encoding, a `Traversal` (`finn.dataflow.traversal`), a
`Repetition` (`ONCE`, or `CYCLIC` for a free-running source) and periodic marker
rules (`LevelEnd(k)`). A traversal is a loop nest over the row-major operand:
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

`compatibility` accepts the first two on one hop and refuses the rest, naming
the adapter; a stream's plan (above) chains the rest through its adapter.
`Composition.connect` checks clock domains too, then emits every data (with any
lane permutation), padding, handshake and marker wire. The derived reorders
reproduce the tiled MVU's two hard-coded `input_gen` stages and FINN's
OuterShuffle coefficients exactly (see `tests/kernels/test_stream_contract.py`).

`CyclicDelivery` streams a constant integer operand in whatever traversal its
consumer reads, from an initialized ROM, so the consumer's order needs no
adapter. Its `output` view is a cyclic stream contract; the same kernel feeds
matmul weight tiles or an eltwise channel vector, inside an operation kernel or
beside one:

```python
from finn.kernels import RomKernel
from finn.dataflow.traversal import Adaptation, Repetition, classify, vector_major

channels = vector_major((4,), 2)
vector = design_space(RomKernel(dtype=dtype("INT4"), form=channels, contents=(1, -2, 7, -8)))
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
