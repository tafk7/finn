# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded physical MVAU assembly, with external or cyclic on-chip weights.

For X[repetition, column], W[row, column], the output is X @ W.T. Input
activation beats traverse (repetition, synapse fold), with SIMD low-first
fields. Replay repeats each vector for every neuron fold. Compute and external
weights traverse (repetition, neuron fold, synapse fold); weight fields are
(PE, SIMD), SIMD fastest and low-first. Results traverse (repetition, neuron
fold), with PE low-first fields. Internal last closes every synapse-fold group.
There is no top-level last: the declared extents determine all stream lengths.
Input high padding is ignored; output high padding is unspecified.

No Region, logical operand mapping, or dataflow graph is required. ``MVAU`` is a
graph of design spaces: four ``Stream`` nodes, and kernel nodes that reference
them. Each stream sees its users; one with a single user is a boundary of MVAU
and presents its ``port`` name (``in0_V``, ``in1_V``, ``out0_V``). ``structure``
wires ``Members(MODULE)`` through ``Members(CONNECTION)``, clocked by
``Members(DOMAIN)``: ``clock`` (``ap_clk``/``ap_rst_n``) and, only when compute
is pumped, ``fast_clock`` (``ap_clk2x``). ``mvau_assembly`` is
a convenience adapter: it configures concrete facts, commits the choices and
packs the views into an ``MVAUAssembly``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
from finn.kernels.configure import commit, describe
from finn.kernels.clocks import DOMAIN, ClockDomain, DerivedClock
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.physical.forms import TRAVERSAL, Every, Traversal, tile, vector_major
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

    repetitions: int
    matrix_width: int
    matrix_height: int
    pe: int
    simd: int

    def __post_init__(self) -> None:
        for name in ("repetitions", "matrix_width", "matrix_height", "pe", "simd"):
            _positive(getattr(self, name), name)
        if self.matrix_width % self.simd or self.matrix_height % self.pe:
            raise ValueError("SIMD must divide matrix_width and PE must divide matrix_height")

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe

    @property
    def activation_beats(self) -> int:
        return self.repetitions * self.synapse_folds

    @property
    def weight_beats(self) -> int:
        return self.repetitions * self.neuron_folds * self.synapse_folds

    @property
    def result_beats(self) -> int:
        return self.repetitions * self.neuron_folds


@dataclass(frozen=True, slots=True)
class MVAUAssembly:
    activation_beats: int
    weight_beats: int
    result_beats: int
    result_dtype: QONNXDataType
    weight_delivery: WeightDelivery
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements
    initializer: tuple[int, ...]


FOLDING = default_semantics(_Folding)


class MVAU(Space):
    """Workload facts, folding choices, and kernels that reference declared streams.

    ``activations`` enters at ``in0_V`` and is replayed once per neuron fold into
    ``replayed``; dotp consumes it with ``weight_stream`` and produces ``results``
    for ``out0_V``. The ``implementation`` Decision decides what drives
    ``weight_stream``: ``external`` places nothing, so the stream has only its
    consumer and is the boundary ``in1_V``; ``cyclic`` places the ``cyclic``
    CyclicDelivery node, which references the stream as its producer and owns
    ``rom_style`` and the optional ``weights``. ``weight_stream`` is buffered:
    ``direct`` or a ``fifo`` with a committed depth.
    """

    repetitions: int = Param()
    matrix_width: int = Param()
    matrix_height: int = Param()
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp: DspBlock = Param()
    segment_length: int = Param()
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    pe: int = Decision(domain=divisors_of(matrix_height))
    simd: int = Decision(domain=divisors_of(matrix_width))

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_type(self) -> QONNXDataType | Rejected:
        try:
            return exact_result_dtype(self.matrix_width, self.activation_dtype, self.weights_dtype)
        except ValueError as error:
            return reject("mvau-arithmetic", str(error))

    @derived(semantics=FOLDING)
    def folding(self) -> _Folding | Rejected:
        try:
            return _Folding(
                self.repetitions, self.matrix_width, self.matrix_height, self.pe, self.simd
            )
        except ValueError as error:
            return reject("mvau-folding", str(error))

    @constraint
    def dimensions_supported(self) -> bool:
        # Reading the folding propagates its refusal to this constraint.
        return isinstance(self.folding, _Folding)

    dimensions = ConstraintGroup(dimensions_supported)

    @derived
    def synapse_folds(self) -> int:
        return self.folding.synapse_folds

    @derived
    def neuron_folds(self) -> int:
        return self.folding.neuron_folds

    @derived(semantics=STREAM_SPEC)
    def activation_spec(self) -> StreamSpec | Rejected:
        f, element = self.folding, ScalarEncoding.admit(self.activation_dtype)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, vector_major((f.repetitions, f.matrix_width), f.simd))

    @derived(semantics=STREAM_SPEC)
    def replayed_spec(self) -> StreamSpec:
        f, spec = self.folding, self.activation_spec
        form = spec.form.replayed(f.neuron_folds, inner_beats=f.synapse_folds)
        return StreamSpec(spec.element, form, markers=(Every(f.synapse_folds),))

    @derived(semantics=TRAVERSAL)
    def weight_period(self) -> Traversal:
        f = self.folding
        return tile(f.matrix_height, f.matrix_width, f.pe, f.simd)

    @derived(semantics=STREAM_SPEC)
    def weight_spec(self) -> StreamSpec | Rejected:
        element = ScalarEncoding.admit(self.weights_dtype)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, self.weight_period.repeated(self.folding.repetitions))

    @derived(semantics=STREAM_SPEC)
    def result_spec(self) -> StreamSpec | Rejected:
        f, element = self.folding, ScalarEncoding.admit(self.result_type)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, vector_major((f.repetitions, f.matrix_height), f.pe))

    # Clock domains, named by their top pins; the 2x domain is present only when
    # the compute kernel runs in it.
    clock = ClockDomain(clock="ap_clk", reset="ap_rst_n")
    fast_clock = DerivedClock(clock="ap_clk2x", base=clock)

    # Streams: relations between the kernels that reference them. A stream with a
    # single user is a boundary of MVAU and presents its ABI port name.
    activations = Stream(spec=activation_spec, port="in0_V", clock=clock)
    replayed = Stream(spec=replayed_spec, clock=clock)
    weight_stream = BufferedStream(spec=weight_spec, port="in1_V", clock=clock)
    results = Stream(spec=result_spec, port="out0_V", clock=clock)

    replay = ReplayBuffer(
        clock=clock,
        input_stream=activations,
        output_stream=replayed,
        sequence_length=synapse_folds,
        replay_count=neuron_folds,
    )
    compute = DotpAxiKernel(
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_type,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
        clock=clock,
        fast_clock=fast_clock,
        activation_stream=replayed,
        weights_stream=weight_stream,
        result_stream=results,
    )
    # A handle naming the cyclic candidate; the Decision places it. It references
    # weight_stream as its producer, so only when selected is the stream internal.
    cyclic = CyclicDelivery(
        dtype=weights_dtype,
        form=weight_period,
        values=weights,
        clock=clock,
        output_stream=weight_stream,
    )
    implementation: CyclicDelivery | None = Decision(
        values={WeightDelivery.EXTERNAL.value: None, WeightDelivery.CYCLIC.value: cyclic}
    )
    delivery = selected(implementation)
    modules = Members(MODULE)
    streams = Members(CONNECTION)
    domains = Members(DOMAIN)
    tieoffs = Members(TIEOFFS)

    @view(semantics=COMPOSED, requires=(dimensions, modules, streams, domains, tieoffs))
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.domains,
            self.tieoffs,
            module="finn_mvau_" + self.delivery,
            producer=ProducerIdentity("finn.mvau." + self.delivery, "1"),
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure.requirements


ROM_STYLE = CyclicDelivery.rom_style


def mvau_assembly(
    *,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    pe: int,
    simd: int,
    target_dsp: DspBlock,
    segment_length: int = 0,
    compute_pumping: bool = False,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[Sequence[int]] | None = None,
    rom_style: str = "auto",
    weight_fifo_depth: int | None = None,
) -> MVAUAssembly:
    """Bind workload facts, commit every choice, then assemble.

    Weights are required by, and only accepted with, cyclic delivery. ``rom_style``
    applies to cyclic delivery; the ``auto`` default leaves memory inference to
    synthesis, as the ROM did before the choice existed. ``weight_fifo_depth``
    places a FIFO on the weight stream; ``None`` connects it directly.
    """
    if not isinstance(weight_delivery, WeightDelivery):
        raise ValueError("weight_delivery must be a WeightDelivery value")
    cyclic = weight_delivery is WeightDelivery.CYCLIC
    if cyclic != (weights is not None):
        raise ValueError("cyclic delivery requires weights; external delivery has no initializer")
    facts: dict[str, Any] = dict(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        target_dsp=target_dsp,
        segment_length=segment_length,
    )
    if weights is not None:
        facts["weights"] = tuple(tuple(row) for row in weights)
    case = weight_delivery.value
    buffered = weight_fifo_depth is not None
    choices: dict[str, object] = {
        "implementation": case,
        "weight_stream.transport": "fifo" if buffered else "direct",
        "compute.compute_pumping": compute_pumping,
        "pe": pe,
        "simd": simd,
    }
    if cyclic:
        choices["implementation.cyclic.rom_style"] = rom_style
    if buffered:
        choices["weight_stream.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["weight_stream.transport.fifo.buffer.ram_style"] = "auto"
    point = commit(design_space(MVAU(**facts)), choices)
    composed = point.query(MVAU.structure)
    if not isinstance(composed, Available):
        raise ValueError(f"MVAU assembly is not accepted: {describe([composed])}")
    folding = point.folding
    return MVAUAssembly(
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
    "MVAU",
    "MVAUAssembly",
    "ROM_STYLE",
    "WeightDelivery",
    "exact_result_dtype",
    "mvau_assembly",
]
