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
connections as ``Stream``s and binds each kernel's ports to them; the replay,
dotp, the weight source and the boundary ports are wired by
``assemble_streams`` through checked stream contracts. ``mvau_assembly``
supplies concrete facts and choices to this same path.
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
from finn.kernels.datatypes.semantics import INTEGER_TENSOR, QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.physical.forms import TRAVERSAL, Every, Traversal, tile, vector_major
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.streaming import ReplayBuffer
from finn.kernels.streams import (
    COMPONENT,
    STREAM_SPEC,
    Stream,
    StreamSpec,
    TopInput,
    TopOutput,
    assemble_streams,
)
from finn.kernels.target import DspBlock
from finn.core.space import (
    Available,
    ChangeRequest,
    ConstraintGroup,
    Decision,
    Param,
    QueryResult,
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
from finn.core.space import DecisionHandle, inspection
from finn.core.space.errors import ConfigurationError, RequestError


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
ASSEMBLY = default_semantics(MVAUAssembly)


def _encoding(dtype: QONNXDataType) -> ScalarEncoding | Rejected:
    try:
        return ScalarEncoding(dtype)
    except ValueError as error:
        return reject("mvau-encoding", str(error))


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
        f, element = self.folding, _encoding(self.activation_dtype)
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
        element = _encoding(self.weights_dtype)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, self.weight_period.repeated(self.folding.repetitions))

    @derived(semantics=STREAM_SPEC)
    def result_spec(self) -> StreamSpec | Rejected:
        f, element = self.folding, _encoding(self.result_type)
        if isinstance(element, Rejected):
            return element
        return StreamSpec(element, vector_major((f.repetitions, f.matrix_height), f.pe))

    activations = Stream(activation_spec)
    replayed = Stream(replayed_spec)
    weight_stream = Stream(weight_spec, buffered=True)
    results = Stream(result_spec)

    source = Subspace(TopInput, name="in0_V", output_stream=activations.spec)
    replay = Subspace(
        ReplayBuffer,
        input_stream=activations.spec,
        output_stream=replayed.spec,
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
        activation_stream=replayed.spec,
        weights_stream=weight_stream.spec,
        result_stream=results.spec,
    )
    implementation = SubspaceChoice(
        {
            WeightDelivery.EXTERNAL.value: Subspace(
                TopInput, name="in1_V", output_stream=weight_stream.spec
            ),
            WeightDelivery.CYCLIC.value: Subspace(
                CyclicDelivery,
                dtype=weights_dtype,
                form=weight_period,
                values=weights,
                output_stream=weight_stream.spec,
            ),
        },
        exports=(COMPONENT,),
    )
    sink = Subspace(TopOutput, name="out0_V", input_stream=results.spec)

    @view(semantics=ASSEMBLY, constraints=(dimensions,))
    def assembly(self) -> MVAUAssembly | Rejected:
        source = self.field(MVAU.implementation.accepted(COMPONENT)).get()
        delivery = WeightDelivery.EXTERNAL if source.requirements is None else WeightDelivery.CYCLIC
        try:
            built = assemble_streams(
                self,
                module="finn_mvau_" + delivery.value,
                producer=ProducerIdentity("finn.mvau." + delivery.value, "1"),
                instance_names={"implementation": "u_weights"},
            )
        except ValueError as error:
            return reject("mvau-stream", str(error))
        f = self.folding
        return MVAUAssembly(
            f.activation_beats,
            f.weight_beats,
            f.result_beats,
            self.result_type,
            delivery,
            built.structure,
            built.requirements,
            source.initializer,
        )

    @view(semantics=default_semantics(ModuleBuildRequirements), constraints=(dimensions,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.assembly().requirements


ROM_STYLE = CyclicDelivery.rom_style


def _findings(results: Sequence[QueryResult[Any]]) -> str:
    return "; ".join(
        f"{finding.owner}: {finding.code}: {finding.message}"
        for result in results
        if not isinstance(result, Available)
        for finding in result.findings
    )


def _selector(point: Space, key: str) -> DecisionHandle[str]:
    for choice in inspection.choices(point):
        if choice.key == key and choice.selector is not None:
            return choice.selector
    raise LookupError(key)


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
    try:
        point = MVAU(**facts)
        changes: list[ChangeRequest] = [
            point.field(_selector(point, "implementation")).change(case),
            point.field(_selector(point, "weight_stream.transport")).change(
                "fifo" if buffered else "direct"
            ),
            point.compute.field(DotpAxiKernel.compute_pumping).change(compute_pumping),
        ]
        if cyclic:
            family = point.implementation.alternative(case)
            changes.append(family.field(ROM_STYLE).change(rom_style))
        if buffered:
            owned = {item.key: item.reference for item in inspection.decisions(point)}
            stage = "weight_stream.transport.fifo.buffer."
            changes.append(point.field(owned[stage + "depth"]).change(weight_fifo_depth))
            changes.append(point.field(owned[stage + "ram_style"]).change("auto"))
        report = point.try_with_choices(*changes, pe=pe, simd=simd)
    except (RequestError, ConfigurationError) as error:
        raise ValueError(str(error)) from error
    if not report.accepted:
        raise ValueError(
            "MVAU choices are not accepted: "
            + _findings([outcome.result for outcome in report.outcomes])
        )
    result = report.instance.assembly.query()
    if not isinstance(result, Available):
        raise ValueError(f"MVAU assembly is not accepted: {_findings([result])}")
    return result.value


__all__ = [
    "MVAU",
    "MVAUAssembly",
    "ROM_STYLE",
    "WeightDelivery",
    "exact_result_dtype",
    "mvau_assembly",
]
