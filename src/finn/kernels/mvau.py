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

No Region, logical operand mapping, or graph is required. ``MVAU`` declares its
connections as ``Stream``s between its placements' port views; each stream
checks its own contract, and ``structure`` composes the accepted connections.
``mvau_assembly`` is a convenience adapter: it commits concrete facts and
choices and packs the resulting views into an ``MVAUAssembly`` record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from finn.kernels.artifacts.build import (
    ModuleBuildRequirements,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.datatypes.semantics import INTEGER_TENSOR, QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.configure import configure, describe
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.physical.forms import TRAVERSAL, Every, Traversal, tile, vector_major
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.streaming import ReplayBuffer
from finn.kernels.streams import (
    COMPOSED,
    MODULE,
    OUTPUT_PORT,
    STREAM_SPEC,
    Composed,
    Stream,
    StreamSpec,
    TopInput,
    TopOutput,
    compose,
    connected,
)
from finn.kernels.target import DspBlock
from finn.core.space import (
    Available,
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    Subspace,
    SubspaceChoice,
    constraint,
    default_semantics,
    derived,
    divisors_of,
    reject,
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
    """Workload facts, folding choices, and kernels connected by declared streams.

    ``activations`` enters at ``in0_V`` and is replayed once per neuron fold into
    ``replayed``; dotp consumes it with ``weight_stream`` and produces ``results``
    for ``out0_V``. The ``implementation`` choice produces ``weight_stream``:
    ``external`` is a top-level port and ``cyclic`` a ``CyclicDelivery`` owning
    ``rom_style`` and the optional ``weights``. ``weight_stream`` is buffered: its
    transport is ``direct`` or a ``fifo`` whose depth is a committed decision.
    """

    repetitions = Param(int)
    matrix_width = Param(int)
    matrix_height = Param(int)
    activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp = Param(DspBlock)
    segment_length = Param(int)
    weights = Param(INTEGER_TENSOR, required=False)
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

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

    source = Subspace(TopInput, name="in0_V", output_stream=activation_spec)
    replay = Subspace(
        ReplayBuffer,
        input_stream=activation_spec,
        sequence_length=synapse_folds,
        replay_count=neuron_folds,
    )
    compute = Subspace(
        DotpAxiKernel,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_type,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
        activation_stream=replayed_spec,
        weights_stream=weight_spec,
        result_stream=result_spec,
    )
    implementation = SubspaceChoice(
        {
            WeightDelivery.EXTERNAL.value: Subspace(
                TopInput, name="in1_V", output_stream=weight_spec
            ),
            WeightDelivery.CYCLIC.value: Subspace(
                CyclicDelivery, dtype=weights_dtype, form=weight_period, values=weights
            ),
        },
        exports=(OUTPUT_PORT, MODULE),
    )
    sink = Subspace(TopOutput, name="out0_V", input_stream=result_spec)

    activations = Stream(
        activation_spec,
        source=("in0_V", source.accepted(TopInput.port)),
        sink=("u_replay", replay.accepted(ReplayBuffer.input_port)),
    )
    replayed = Stream(
        replayed_spec,
        source=("u_replay", replay.accepted(ReplayBuffer.output_port)),
        sink=("u_compute", compute.accepted(DotpAxiKernel.activation_port)),
    )
    weight_stream = Stream(
        weight_spec,
        source=("u_weights", implementation.accepted(OUTPUT_PORT)),
        sink=("u_compute", compute.accepted(DotpAxiKernel.weights_port)),
        buffered=True,
    )
    results = Stream(
        result_spec,
        source=("u_compute", compute.accepted(DotpAxiKernel.result_port)),
        sink=("out0_V", sink.accepted(TopOutput.port)),
    )
    activations_connected = connected(activations)
    replayed_connected = connected(replayed)
    weight_stream_connected = connected(weight_stream)
    results_connected = connected(results)
    streams = ConstraintGroup(
        activations_connected, replayed_connected, weight_stream_connected, results_connected
    )

    @view(semantics=COMPOSED, constraints=(dimensions, streams))
    def structure(self) -> Composed | Rejected:
        weights = self.field(MVAU.implementation.accepted(MODULE)).get()
        delivery = "external" if weights.requirements is None else "cyclic"
        try:
            return compose(
                module="finn_mvau_" + delivery,
                producer=ProducerIdentity("finn.mvau." + delivery, "1"),
                instances={
                    "u_replay": self.replay.build_requirements(),
                    "u_compute": self.compute.build_requirements(),
                    "u_weights": weights.requirements,
                },
                connections=(
                    self.activations.connection(),
                    self.replayed.connection(),
                    self.weight_stream.connection(),
                    self.results.connection(),
                ),
            )
        except ValueError as error:
            return reject("mvau-composition", str(error))

    @view(semantics=default_semantics(ModuleBuildRequirements), constraints=(dimensions, streams))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure().requirements


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
    facts: dict[str, object] = dict(
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
    point = configure(MVAU, facts, choices)
    composed = point.structure.query()
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
        point.implementation.alternative(case).field(CyclicDelivery.image).get() if cyclic else (),
    )


__all__ = [
    "MVAU",
    "MVAUAssembly",
    "ROM_STYLE",
    "WeightDelivery",
    "exact_result_dtype",
    "mvau_assembly",
]
