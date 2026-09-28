# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix-multiply unit on streams, with external or cyclic on-chip weights.

The ``contraction`` fixes how activations meet the weights.

- ``DENSE``: for activations X[row, k] and weights W[output, k], the result is
  Y[row, output] = sum over k of X[row, k] * W[output, k]. Every output reads
  the whole activation row, so the row is replayed once per output fold.
  Activation beats traverse (row, reduction fold) with SIMD low-first fields.
- ``PER_CHANNEL`` (depthwise): for activations X[row, k, c] and weights
  W[c, k], Y[row, c] = sum over k of X[row, k, c] * W[c, k]. Each output
  channel reads its own activations, so nothing is replayed. Activation beats
  traverse (row, channel fold, window fold) with PE channels of SIMD window
  positions, channel fastest (``channel_tile``).

Compute and external weights traverse (row, output fold, reduction fold);
weight fields are (PE, SIMD), SIMD fastest and low-first. Results traverse
(row, output fold), with PE low-first fields. Internal last closes every
reduction-fold group. There is no top-level last: the declared extents
determine all stream lengths. Input high padding is ignored; output high
padding is unspecified.

No Region, logical operand mapping, or dataflow graph is required.
``MatMulKernel`` is a graph of design spaces: four ``Stream`` nodes, and
kernel nodes that reference them. Each stream sees its users; one with a single
user is a boundary of the kernel and presents its ``port`` name (``in0_V``,
``in1_V``, ``out0_V``). ``structure`` wires ``Members(MODULE)`` through
``Members(CONNECTION)``; the module has ``ap_clk2x`` only when compute is
pumped. ``matmul_assembly`` is a convenience adapter: it configures concrete
facts, commits the choices and packs the views into a ``MatMulAssembly``.
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
from finn.kernels.datatypes.scalar import ScalarEncoding
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
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import Contraction, DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.physical.forms import (
    TRAVERSAL,
    Every,
    Traversal,
    channel_tile,
    tile,
    vector_major,
)
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.streaming import ReplayBuffer
from finn.kernels.streams import (
    COMPOSED,
    CONNECTION,
    MODULE,
    STREAM_SPEC,
    TIEOFFS,
    BufferedStream,
    Composed,
    Stream,
    StreamSpec,
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
    per_channel: bool = False

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
    def reuse(self) -> int:
        """How many times each activation row is read: once per output fold when dense."""
        return 1 if self.per_channel else self.output_folds

    @property
    def activation_beats(self) -> int:
        channel_folds = self.output_folds if self.per_channel else 1
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

    ``activations`` enters at ``in0_V``; the replay node presents each row
    ``reuse`` times into ``replayed``, with a frame marker per reduction: once
    per output fold when dense, once (markers only) per channel. dotp consumes
    it with ``weight_stream`` and produces ``results`` for ``out0_V``. The
    ``compute`` Decision places one core kernel; a per-channel contraction is
    read only by the INT8 DSP58 core. The ``delivery`` Decision decides what drives
    ``weight_stream``: ``external`` places nothing, so the stream has only its
    consumer and is the boundary ``in1_V``; ``cyclic`` places the ``cyclic``
    CyclicDelivery node, which references the stream as its producer and owns
    ``rom_style`` and the optional ``weights``. ``weight_stream`` is buffered:
    ``direct`` or a ``fifo`` with a committed depth.
    """

    rows: int = Param()
    reduction: int = Param()
    outputs: int = Param()
    contraction: Contraction = Param(default=Contraction.DENSE)
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp: DspBlock = Param()
    target_period_ns: float = Param()
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)

    @derived
    def per_channel(self) -> bool:
        return self.contraction is Contraction.PER_CHANNEL

    # A per-channel operation runs natively (one channel per PE lane, INT8 DSP58
    # only) or on the dense datapath with block-diagonal weights, on any core.
    realization: str = Decision(values=("native", "dense"), when=per_channel)

    @derived
    def datapath(self) -> Contraction:
        """The contraction the datapath computes: densely realized, a per-channel one is dense."""
        if self.contraction is Contraction.DENSE or self.realization == "dense":
            return Contraction.DENSE
        return Contraction.PER_CHANNEL

    @derived
    def datapath_reduction(self) -> int:
        """The datapath's K: the window times the channels when densely realized."""
        if self.per_channel and self.datapath is Contraction.DENSE:
            return self.reduction * self.outputs
        return self.reduction

    @derived(semantics=INTEGER_TENSOR)
    def datapath_weights(self) -> IntegerTensor:
        """The weights the datapath reads: block-diagonal when densely realized.

        W'[c, k * C + c'] = W[c, k] when c' = c, and 0 otherwise: the densely read
        activation row (k, c') meets only its own channel's weights.
        """
        weights = self.weights
        if not (self.per_channel and self.datapath is Contraction.DENSE):
            return weights
        channels = self.outputs
        return tuple(
            tuple(value if other == channel else 0 for value in row for other in range(channels))
            for channel, row in enumerate(cast("tuple[tuple[int, ...], ...]", weights))
        )

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
                self.datapath is Contraction.PER_CHANNEL,
            )
        except ValueError as error:
            return reject("matmul-folding", str(error))

    @constraint
    def dimensions_supported(self) -> bool:
        # Reading the folding propagates its refusal to this constraint.
        return isinstance(self.folding, _Folding)

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.per_channel and self.realization == "dense" and self.delivered == "external":
            return reject(
                "matmul-realization",
                "a dense realization builds block-diagonal weights, so it needs known weights",
            )
        return True

    dimensions = ConstraintGroup(dimensions_supported, realization_supported)

    @derived
    def reduction_folds(self) -> int:
        return self.folding.reduction_folds

    @derived
    def reuse(self) -> int:
        return self.folding.reuse

    @derived(semantics=STREAM_SPEC)
    def activation_spec(self) -> StreamSpec | Rejected:
        f, element = self.folding, ScalarEncoding.admit(self.activation_dtype)
        if isinstance(element, Rejected):
            return element
        if f.per_channel:
            return StreamSpec(element, channel_tile(f.rows, f.reduction, f.outputs, f.pe, f.simd))
        return StreamSpec(element, vector_major((f.rows, f.reduction), f.simd))

    @derived(semantics=STREAM_SPEC)
    def replayed_spec(self) -> StreamSpec:
        f, spec = self.folding, self.activation_spec
        form = spec.form.replayed(f.reuse, inner_beats=f.reduction_folds)
        return StreamSpec(spec.element, form, markers=(Every(f.reduction_folds),))

    @derived(semantics=TRAVERSAL)
    def weight_period(self) -> Traversal:
        f = self.folding
        return tile(f.outputs, f.reduction, f.pe, f.simd)

    @derived(semantics=STREAM_SPEC)
    def weight_spec(self) -> StreamSpec | Rejected:
        element = ScalarEncoding.admit(self.weights_dtype)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, self.weight_period.repeated(self.folding.rows))

    @derived(semantics=STREAM_SPEC)
    def result_spec(self) -> StreamSpec | Rejected:
        f, element = self.folding, ScalarEncoding.admit(self.result_type)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, vector_major((f.rows, f.outputs), f.pe))

    # Streams: relations between the kernels that reference them. A stream with a
    # single user is a boundary of the kernel and presents its ABI port name.
    activations = Stream(spec=activation_spec, port="in0_V")
    replayed = Stream(spec=replayed_spec)
    weight_stream = BufferedStream(spec=weight_spec, port="in1_V")
    results = Stream(spec=result_spec, port="out0_V")

    replay = ReplayBuffer(
        input_stream=activations,
        output_stream=replayed,
        sequence_length=reduction_folds,
        replay_count=reuse,
    )

    @derived
    def narrow_weights(self) -> bool:
        """Known weights that avoid their type's most negative value let the packed core
        pack more lanes (NARROW_WEIGHTS). Provisional: the user means to revisit it."""
        if self.delivered != WeightDelivery.CYCLIC.value:
            return False  # weights arriving at run time promise nothing
        low, _ = ordinary_integer_bounds(self.weights_dtype)
        return all(value > low for value in _leaves(self.weights))

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
        contraction=datapath,
        narrow_weights=narrow_weights,
        activation_stream=replayed,
        weights_stream=weight_stream,
        result_stream=results,
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
        contraction=datapath,
        activation_stream=replayed,
        weights_stream=weight_stream,
        result_stream=results,
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
    delivery: CyclicDelivery | None = Decision(
        values={WeightDelivery.EXTERNAL.value: None, WeightDelivery.CYCLIC.value: cyclic}
    )
    delivered = selected(delivery)
    modules = Members(MODULE)
    streams = Members(CONNECTION)
    tieoffs = Members(TIEOFFS)

    @view(semantics=COMPOSED, requires=(dimensions, modules, streams, tieoffs))
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.tieoffs,
            module="finn_matmul_" + self.delivered,
            producer=ProducerIdentity("finn.matmul." + self.delivered, "1"),
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure.requirements


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
    contraction: Contraction = Contraction.DENSE,
    target_period_ns: float = 5.0,
    compute_pumping: bool = False,
    core: str | None = None,
    realization: str | None = None,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[Sequence[int]] | None = None,
    rom_style: str = "auto",
    weight_fifo_depth: int | None = None,
) -> MatMulAssembly:
    """Bind operation facts, commit every choice, then assemble.

    ``reduction`` is K and ``outputs`` N; for a per-channel ``contraction`` they
    are the window and the channels, and ``weights`` is (channels, window).

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
    if cyclic != (weights is not None):
        raise ValueError("cyclic delivery requires weights; external delivery has no initializer")
    facts: dict[str, Any] = dict(
        rows=rows,
        reduction=reduction,
        outputs=outputs,
        contraction=contraction,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
    )
    if weights is not None:
        facts["weights"] = tuple(tuple(row) for row in weights)
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
    if buffered:
        choices["weight_stream.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["weight_stream.transport.fifo.buffer.ram_style"] = "auto"
    base = design_space(MatMulKernel(**facts))
    if contraction is Contraction.PER_CHANNEL:
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
                commit(point, {"compute": case}).query(MatMulKernel.structure)
                for case in (cases.value if isinstance(cases, Available) else ())
            )
            raise ValueError(f"no compute core is compatible: {describe(refusals)}")
        if len(cores) > 1:
            named = ", ".join(map(str, cores))
            raise ValueError(f"compute cores {named} are all compatible; choose one")
        core = str(cores[0])
    point = commit(point, {"compute": core})
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
        point.cyclic.image if cyclic else (),
    )


__all__ = [
    "Contraction",
    "MatMulAssembly",
    "MatMulKernel",
    "ROM_STYLE",
    "WeightDelivery",
    "exact_result_dtype",
    "matmul_assembly",
]
