# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix-multiply unit on streams, with external or cyclic on-chip weights.

The ``form`` (``finn.dataflow.gemm``) fixes how activations meet the weights,
in canonical GEMM indices: ``m`` the rows, ``n`` the outputs, ``k`` the
reduction. Weights are stored ``(k, n)``, as ONNX ``MatMul`` stores them.

- ``DENSE``: Y[m, n] = sum over k of X[m, k] * W[k, n]. Every output reads the
  whole activation row, so the row is replayed once per output fold.
  Activation beats traverse (m, k fold) with SIMD low-first fields.
- ``DEPTHWISE``: Y[m, n] = sum over k of X[m, k, n] * W[k, n]. Each output
  channel reads its own activations, so nothing is replayed. Activation beats
  traverse (m, n fold, k fold) with PE channels of SIMD window positions,
  channel fastest.

PE folds ``n``, SIMD folds ``k``, and the ``schedule`` walks ``m``, then
``n``, then the reduction (``finn.dataflow.schedule``). Every stream's beat
sequence derives from that one schedule through dotp's port conventions
(``dotp_sequences``): compute and external weights traverse (m, n fold, k
fold) with (PE, SIMD) fields, results (m, n fold) with PE fields, and the
frame marker closes every reduction. A densely realized depthwise operation
reads its (M, K, N) activations as an (M, K * N) view.
There is no top-level last: the declared extents determine all stream lengths.
Input high padding is ignored; output high padding is unspecified.

No Region, logical operand mapping, or dataflow graph is required.
``MatMulKernel`` is a graph of design spaces: ``Stream`` nodes (activations,
weights, results, and the set index with several weight sets), and kernel
nodes that reference them. Each stream sees its users; one with a single user
is a boundary of the kernel and presents its ``port`` name (``in0_V``,
``in1_V``, ``out0_V``, ``in2_V``). The activation stream's plan replays each
dense row and frames each reduction, and its adapter carries that out.
``structure`` wires ``Members(MODULE)`` through ``Members(CONNECTION)``; the
module has ``ap_clk2x`` only when compute is pumped. ``matmul_assembly`` is a
convenience adapter: it configures concrete facts, commits the choices (each
stream's one compatible adapter included) and packs the views into a
``MatMulAssembly``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from finn.kernels.artifacts.build import (
    ModuleBuildRequirements,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.configure import commit, compatible, describe
from finn.kernels.control import EXPORTED, ControlBus
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.dotp import (
    DOTP_SEQUENCES,
    DotpAxiKernel,
    DotpSequences,
    Int8Dsp58DotpKernel,
    PackedDotpKernel,
    dotp_sequences,
)
from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import SCHEDULE, Schedule
from finn.dataflow.tensor import TENSOR, Tensor
from finn.dataflow.traversal import TRAVERSAL, Traversal, period
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.streams import (
    COMPOSED,
    CONNECTION,
    MODULE,
    TIEOFFS,
    BufferedStream,
    Composed,
    Stream,
    commit_adapters,
    netlist,
)
from finn.kernels.target import DspBlock
from finn.core.space import (
    Available,
    ConstraintGroup,
    QueryResult,
    Decision,
    Members,
    Param,
    Rejected,
    Space,
    design_space,
    constraint,
    default_semantics,
    derived,
    divisors_of,
    reject,
    selected,
    view,
)


class WeightDelivery(Enum):
    EXTERNAL = "external"
    CYCLIC = "cyclic"
    MEMSTREAM = "memstream"


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def exact_result_dtype(
    vector_length: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
) -> QONNXDataType:
    """Smallest signed INT covering every full-range integer dot product."""
    _positive(vector_length, "vector_length")
    activation = ordinary_integer_bounds(canonical_qonnx_datatype(activation_dtype))
    weights = ordinary_integer_bounds(canonical_qonnx_datatype(weights_dtype))
    products = tuple(a * w for a in activation for w in weights)
    lower, upper = vector_length * min(products), vector_length * max(products)
    bits = max(1, upper.bit_length() + 1, (~lower).bit_length() + 1 if lower < 0 else 1)
    return resolve_qonnx_datatype_name(f"INT{bits}")


@dataclass(frozen=True, slots=True)
class _Folding:
    """Concrete stream extents and the native PE/SIMD packing order above."""

    rows: int
    reduction: int
    outputs: int
    pe: int
    simd: int
    depthwise: bool = False

    def __post_init__(self) -> None:
        for name in ("rows", "reduction", "outputs", "pe", "simd"):
            _positive(getattr(self, name), name)
        if self.reduction % self.simd or self.outputs % self.pe:
            raise ValueError("SIMD must divide the reduction and PE must divide the outputs")

    @property
    def reduction_folds(self) -> int:
        return self.reduction // self.simd

    @property
    def output_folds(self) -> int:
        return self.outputs // self.pe

    @property
    def activation_beats(self) -> int:
        channel_folds = self.output_folds if self.depthwise else 1
        return self.rows * channel_folds * self.reduction_folds

    @property
    def weight_beats(self) -> int:
        return self.rows * self.output_folds * self.reduction_folds

    @property
    def result_beats(self) -> int:
        return self.rows * self.output_folds


@dataclass(frozen=True, slots=True)
class MatMulAssembly:
    activation_beats: int
    weight_beats: int
    result_beats: int
    result_dtype: QONNXDataType
    weight_delivery: WeightDelivery
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements
    initializer: tuple[int, ...]


FOLDING = default_semantics(_Folding)


class MatMulKernel(Space):
    """Operation facts, folding choices, and kernels that reference declared streams.

    ``activations`` enters at ``in0_V`` and feeds dotp directly: the stream's
    plan replays each dense row once per output fold and closes each reduction
    with a frame marker (markers only, per channel), and its ``adapter``
    realizes that plan. dotp consumes it with ``weight_stream`` and produces
    ``results`` for ``out0_V``. The
    ``compute`` Decision places one core kernel; a depthwise form is read
    natively only by the INT8 DSP58 core. The ``delivery`` Decision decides what drives
    ``weight_stream``: ``external`` places nothing, so the stream has only its
    consumer and is the boundary ``in1_V``; ``cyclic`` places the ``cyclic``
    CyclicDelivery node, which references the stream as its producer and owns
    ``rom_style`` and the optional ``weights``. ``weight_stream`` is buffered:
    ``direct`` or a ``fifo`` with a committed depth.
    """

    rows: int = Param()
    reduction: int = Param()
    outputs: int = Param()
    form: Form = Param(default=Form.DENSE)
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp: DspBlock = Param()
    target_period_ns: float = Param()
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    # Software rewrites the weights at run time through AXI-Lite (memstream only).
    writable_weights: bool = Param(default=False)
    # Several weight sets, one selected per row by an index on ``in2_V`` (memstream
    # only); ``weights`` then holds one operand per set.
    weight_sets: int = Param(default=1)

    @derived
    def depthwise(self) -> bool:
        return self.form is Form.DEPTHWISE

    @derived
    def multi_set(self) -> bool:
        return self.weight_sets > 1

    # A depthwise operation runs natively (one channel per PE lane, INT8 DSP58
    # only) or on the dense datapath with block-diagonal weights, on any core.
    realization: str = Decision(values=("native", "dense"), when=depthwise)

    @derived
    def datapath(self) -> Form:
        """The form the datapath computes: densely realized, a depthwise one is dense."""
        if self.form is Form.DENSE or self.realization == "dense":
            return Form.DENSE
        return Form.DEPTHWISE

    @derived
    def dense_view(self) -> bool:
        """Whether the datapath reads depthwise activations (M, K, N) as (M, K * N)."""
        return self.depthwise and self.datapath is Form.DENSE

    @derived
    def datapath_reduction(self) -> int:
        """The datapath's K: the window times the channels when densely realized."""
        return self.reduction * self.outputs if self.dense_view else self.reduction

    @derived(semantics=INTEGER_TENSOR)
    def datapath_weights(self) -> IntegerTensor:
        """The weights the datapath reads, ``(k, n)``: block-diagonal when densely realized.

        W'[k * N + c, n] = W[k, n] when c = n, and 0 otherwise: the densely read
        activation row (k, c) meets only its own channel's weights.
        """
        weights = self.weights
        if not self.dense_view:
            return weights
        channels = self.outputs

        def blocks(operand: object) -> IntegerTensor:
            return tuple(
                tuple(value if channel == output else 0 for output, value in enumerate(row))
                for row in cast("tuple[tuple[int, ...], ...]", operand)
                for channel in range(channels)
            )

        return tuple(blocks(item) for item in weights) if self.multi_set else blocks(weights)

    pe: int = Decision(domain=divisors_of(outputs))
    simd: int = Decision(domain=divisors_of(datapath_reduction))
    compute_pumping: bool = Decision(values=(False, True))

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_type(self) -> QONNXDataType | Rejected:
        try:
            return exact_result_dtype(self.reduction, self.activation_dtype, self.weights_dtype)
        except ValueError as error:
            return reject("matmul-arithmetic", str(error))

    @derived(semantics=FOLDING)
    def folding(self) -> _Folding | Rejected:
        try:
            return _Folding(
                self.rows,
                self.datapath_reduction,
                self.outputs,
                self.pe,
                self.simd,
                self.datapath is Form.DEPTHWISE,
            )
        except ValueError as error:
            return reject("matmul-folding", str(error))

    @constraint
    def dimensions_supported(self) -> bool:
        # Reading the folding propagates its refusal to this constraint.
        return isinstance(self.folding, _Folding)

    @constraint
    def delivery_supported(self) -> bool | Rejected:
        memstream = self.delivered == WeightDelivery.MEMSTREAM.value
        if self.writable_weights and not memstream:
            return reject("matmul-writable", "runtime-writable weights need the memstream delivery")
        if self.multi_set and not memstream:
            return reject("matmul-sets", "several weight sets need the memstream delivery")
        return True

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.depthwise and self.realization == "dense" and self.delivered == "external":
            return reject(
                "matmul-realization",
                "a dense realization builds block-diagonal weights, so it needs known weights",
            )
        return True

    dimensions = ConstraintGroup(dimensions_supported, realization_supported, delivery_supported)

    # The schedule the datapath computes, the tensors the streams carry, and
    # what each end presents of them.

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        """The datapath's indices folded by PE (``n``) and SIMD (``k``), reduction innermost."""
        f = self.folding
        return matmul_schedule(
            rows=f.rows, reduction=f.reduction, outputs=f.outputs, pe=f.pe, simd=f.simd
        )

    @derived(semantics=default_semantics(tuple))
    def activation_shape(self) -> tuple[int, ...]:
        """The activation tensor: (M, K), or (M, K, N) depthwise, however it is read."""
        f = self.folding  # reading it propagates its refusal of the extents
        if self.depthwise:
            return (f.rows, self.reduction, f.outputs)
        return (f.rows, f.reduction)

    @derived(semantics=DOTP_SEQUENCES)
    def sequences(self) -> DotpSequences | Rejected:
        """What the compute core's ports present, whichever core computes."""
        return dotp_sequences(
            self.schedule,
            self.datapath,
            pe=self.pe,
            simd=self.simd,
            activations=self.activation_shape if self.dense_view else (),
        )

    @derived(semantics=TENSOR)
    def activation_tensor(self) -> Tensor | Rejected:
        element = ScalarEncoding.admit(self.activation_dtype)
        if isinstance(element, Rejected):
            return element
        return Tensor(self.activation_shape, element)

    @derived(semantics=TRAVERSAL)
    def weight_period(self) -> Traversal:
        """One pass of the weights: what a stored delivery repeats."""
        return period(self.sequences.weights.form)

    @derived(semantics=TENSOR)
    def weight_tensor(self) -> Tensor | Rejected:
        element = ScalarEncoding.admit(self.weights_dtype)
        if isinstance(element, Rejected):
            return element
        f = self.folding
        return Tensor((f.reduction, f.outputs), element)

    @derived(semantics=TENSOR)
    def set_tensor(self) -> Tensor:
        """One set index per row, as wide as the memory's selector."""
        sets = self.weight_sets
        bits = (sets - 1).bit_length() if sets > 2 else 1
        index = ScalarEncoding(resolve_qonnx_datatype_name(f"UINT{bits}"))
        return Tensor((self.folding.rows,), index)

    @derived(semantics=TENSOR)
    def result_tensor(self) -> Tensor | Rejected:
        element = ScalarEncoding.admit(self.result_type)
        if isinstance(element, Rejected):
            return element
        f = self.folding
        return Tensor((f.rows, f.outputs), element)

    # Streams: relations between the kernels that reference them. A stream with a
    # single user is a boundary of the kernel and presents its ABI port name.
    # The activations enter at in0_V, each row once; the stream's adapter
    # replays them for dotp and closes each reduction with a frame marker.
    activations = Stream(tensor=activation_tensor, port="in0_V")
    weight_stream = BufferedStream(tensor=weight_tensor, port="in1_V")
    results = Stream(tensor=result_tensor, port="out0_V")
    set_index = Stream(tensor=set_tensor, port="in2_V", when=multi_set)

    @derived
    def narrow_weights(self) -> bool:
        """Known weights that avoid their type's most negative value let the packed core
        pack more lanes (NARROW_WEIGHTS). Provisional: the user means to revisit it."""
        read_only = self.delivered == WeightDelivery.CYCLIC.value or (
            self.delivered == WeightDelivery.MEMSTREAM.value and not self.writable_weights
        )
        if not read_only:
            return False  # weights arriving or rewritten at run time promise nothing
        low, _ = ordinary_integer_bounds(self.weights_dtype)
        return all(value > low for value in _leaves(self.weights))

    @derived(semantics=default_semantics(tuple))
    def dotp_activation_shape(self) -> tuple[int, ...]:
        """The activation tensor dotp reads through a view, when densely realized."""
        return self.activation_shape if self.dense_view else ()

    # The compute cores: handles naming the candidates of ``compute``. Each
    # refuses what its core cannot build; both share the pumping choice.
    packed = PackedDotpKernel(
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_type,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        compute_pumping=compute_pumping,
        form=datapath,
        narrow_weights=narrow_weights,
        activation_stream=activations,
        weights_stream=weight_stream,
        result_stream=results,
        schedule=schedule,
        activation_shape=dotp_activation_shape,
    )
    int8_dsp58 = Int8Dsp58DotpKernel(
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_type,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        compute_pumping=compute_pumping,
        form=datapath,
        activation_stream=activations,
        weights_stream=weight_stream,
        result_stream=results,
        schedule=schedule,
        activation_shape=dotp_activation_shape,
    )
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        values={"packed": packed, "int8_dsp58": int8_dsp58}
    )
    # A handle naming the cyclic candidate; the Decision places it. It references
    # weight_stream as its producer, so only when selected is the stream internal.
    cyclic = CyclicDelivery(
        dtype=weights_dtype,
        form=weight_period,
        values=datapath_weights,
        output_stream=weight_stream,
    )
    # The memstream candidate: a RAM image in the same order, optionally
    # rewritable through the ``config`` control bus, exported as ``s_axilite``.
    config = ControlBus(port="s_axilite")
    memstream = MemStreamKernel(
        dtype=weights_dtype,
        form=weight_period,
        values=datapath_weights,
        writable=writable_weights,
        sets=weight_sets,
        output_stream=weight_stream,
        set_stream=set_index,
        control=config,
    )
    delivery: CyclicDelivery | MemStreamKernel | None = Decision(
        values={
            WeightDelivery.EXTERNAL.value: None,
            WeightDelivery.CYCLIC.value: cyclic,
            WeightDelivery.MEMSTREAM.value: memstream,
        }
    )
    delivered = selected(delivery)
    modules = Members(MODULE)
    streams = Members(CONNECTION)
    tieoffs = Members(TIEOFFS)
    controls = Members(EXPORTED)

    @view(semantics=COMPOSED, requires=(dimensions, modules, streams, tieoffs, controls))
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.tieoffs,
            self.controls,
            module="finn_matmul_" + self.delivered,
            producer=ProducerIdentity("finn.matmul." + self.delivered, "1"),
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure.requirements


def matmul_schedule(*, rows: int, reduction: int, outputs: int, pe: int, simd: int) -> Schedule:
    """MatMul's schedule: ``n`` folded by PE, ``k`` by SIMD; ``m``, then ``n``, then ``k``."""
    return Schedule({m: rows, n: outputs, k: reduction}, folds={n: pe, k: simd}, beats=(m, n, k))


def _frozen(values: object) -> object:
    """Nested sequences as nested tuples."""
    if isinstance(values, Sequence):
        return tuple(_frozen(item) for item in values)
    return values


def _leaves(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _leaves(item))


ROM_STYLE = CyclicDelivery.rom_style


def _realizes(base: MatMulKernel, choices: dict[str, object]) -> QueryResult[bool]:
    """Accepted when the choices commit, the realization's own rule holds, and some
    core can compute it."""
    try:
        point = commit(base, choices)
    except ValueError as error:
        return reject("matmul-realization", str(error))
    rule = point.inspect(MatMulKernel.realization_supported).result
    if not isinstance(rule, Available):
        return rule
    cores = compatible(
        point, "compute", lambda item: item.compute.inspect(DotpAxiKernel.support).result
    )
    return Available(True) if cores else reject("matmul-realization", "no core computes it")


def matmul_assembly(
    *,
    rows: int,
    reduction: int,
    outputs: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    pe: int,
    simd: int,
    target_dsp: DspBlock,
    form: Form = Form.DENSE,
    target_period_ns: float = 5.0,
    compute_pumping: bool = False,
    core: str | None = None,
    realization: str | None = None,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[object] | None = None,
    rom_style: str = "auto",
    ram_style: str = "auto",
    pumped_memory: bool = False,
    writable_weights: bool = False,
    weight_sets: int = 1,
    weight_fifo_depth: int | None = None,
) -> MatMulAssembly:
    """Bind operation facts, commit every choice, then assemble.

    ``reduction`` is K and ``outputs`` N; for a depthwise ``form`` they are the
    window and the channels. ``weights`` is stored (K, N) either way.

    Weights are required by, and only accepted with, cyclic delivery. ``rom_style``
    applies to cyclic delivery; the ``auto`` default leaves memory inference to
    synthesis, as the ROM did before the choice existed. ``weight_fifo_depth``
    places a FIFO on the weight stream; ``None`` connects it directly.
    ``target_period_ns`` is the clock the module must meet (5 ns: 200 MHz); it
    sets dotp's DSP58 chain segmentation. ``core`` names the compute core
    (``packed`` or ``int8_dsp58``); left out, the one core compatible with the
    configuration is taken, and several compatible cores must be chosen from.
    """
    if not isinstance(weight_delivery, WeightDelivery):
        raise ValueError("weight_delivery must be a WeightDelivery value")
    cyclic = weight_delivery is WeightDelivery.CYCLIC
    known = weight_delivery is not WeightDelivery.EXTERNAL
    if known != (weights is not None):
        raise ValueError("stored delivery requires weights; external delivery has no initializer")
    facts: dict[str, Any] = dict(
        rows=rows,
        reduction=reduction,
        outputs=outputs,
        form=form,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        writable_weights=writable_weights,
        weight_sets=weight_sets,
    )
    if weights is not None:
        facts["weights"] = _frozen(weights)
    case = weight_delivery.value
    buffered = weight_fifo_depth is not None
    choices: dict[str, object] = {
        "delivery": case,
        "weight_stream.transport": "fifo" if buffered else "direct",
        "compute_pumping": compute_pumping,
        "pe": pe,
        "simd": simd,
    }
    if cyclic:
        choices["delivery.cyclic.rom_style"] = rom_style
    if weight_delivery is WeightDelivery.MEMSTREAM:
        choices["delivery.memstream.ram_style"] = ram_style
        choices["delivery.memstream.pumped_memory"] = pumped_memory
    if buffered:
        choices["weight_stream.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["weight_stream.transport.fifo.buffer.ram_style"] = "auto"
    base = design_space(MatMulKernel(**facts))
    if form is Form.DEPTHWISE:
        # The realization sets the datapath's reduction, and so the SIMD domain:
        # each is committed together with the folding.
        if realization is None:
            viable = [
                case
                for case in ("native", "dense")
                if isinstance(_realizes(base, {**choices, "realization": case}), Available)
            ]
            if len(viable) != 1:
                named = ", ".join(viable) or "none"
                raise ValueError(f"realizations compatible with this configuration: {named}")
            realization = viable[0]
        choices["realization"] = realization
    point = commit(base, choices)
    if core is None:
        cores = compatible(
            point, "compute", lambda item: item.compute.inspect(DotpAxiKernel.support).result
        )
        if not cores:
            cases = point.field(MatMulKernel.compute).candidates()
            refusals = (
                commit(point, {"compute": case}).compute.inspect(DotpAxiKernel.support).result
                for case in (cases.value if isinstance(cases, Available) else ())
            )
            raise ValueError(f"no compute core is compatible: {describe(refusals)}")
        if len(cores) > 1:
            named = ", ".join(map(str, cores))
            raise ValueError(f"compute cores {named} are all compatible; choose one")
        core = str(cores[0])
    point = commit_adapters(commit(point, {"compute": core}))
    composed = point.query(MatMulKernel.structure)
    if not isinstance(composed, Available):
        raise ValueError(f"MatMul assembly is not accepted: {describe([composed])}")
    folding = point.folding
    return MatMulAssembly(
        folding.activation_beats,
        folding.weight_beats,
        folding.result_beats,
        point.result_type,
        weight_delivery,
        composed.value.structure,
        composed.value.requirements,
        point.cyclic.image
        if cyclic
        else point.memstream.image
        if weight_delivery is WeightDelivery.MEMSTREAM
        else (),
    )


__all__ = [
    "MatMulAssembly",
    "MatMulKernel",
    "ROM_STYLE",
    "WeightDelivery",
    "exact_result_dtype",
    "matmul_assembly",
    "matmul_schedule",
]
