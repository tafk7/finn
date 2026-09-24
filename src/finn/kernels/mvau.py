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

No Region, logical operand mapping, or graph is required. ``MVAU`` binds its
dotp child Space and consumes that child's accepted physical
View. ``mvau_assembly`` supplies concrete facts and choices to this same path;
its wiring code only receives the accepted component requirements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.streaming import cyclic_stream_requirements, replay_buffer_requirements
from finn.kernels.target import DspBlock
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.physical.lowering import lower_module_structure
from finn.kernels.physical.structure import (
    ConstantBits,
    ModuleInstance,
    PhysicalPin,
    PhysicalStructure,
    PhysicalWire,
    PinSlice,
    UnusedOutput,
)
from finn.kernels.space import (
    Answer,
    ConstraintGroup,
    Decided,
    Decision,
    Param,
    Space,
    Subspace,
    constraint,
    derived,
    divisors_of,
    reject,
)
from finn.kernels.space.errors import RequestError


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
class _Traversal:
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

    def weight_image(
        self, weights: Sequence[Sequence[int]], dtype: QONNXDataType
    ) -> tuple[int, ...]:
        """Pack one row-major W image; word order is (neuron fold, synapse fold)."""
        if len(weights) != self.matrix_height or any(
            len(row) != self.matrix_width for row in weights
        ):
            raise ValueError("weights must have shape (matrix_height, matrix_width)")
        minimum, maximum = ordinary_integer_bounds(dtype)
        if any(
            type(value) is not int or not minimum <= value <= maximum
            for row in weights
            for value in row
        ):
            raise ValueError("every weight must be an integer admitted by weights_dtype")
        bits = dtype.bitwidth()
        mask = (1 << bits) - 1
        return tuple(
            sum(
                (weights[nf * self.pe + p][sf * self.simd + s] & mask)
                << ((p * self.simd + s) * bits)
                for p in range(self.pe)
                for s in range(self.simd)
            )
            for nf in range(self.neuron_folds)
            for sf in range(self.synapse_folds)
        )


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


def _slice(owner: str | None, name: str, bits: int = 1, offset: int = 0) -> PinSlice:
    return PinSlice(PhysicalPin(owner, name), offset, bits)


def _axis(name: str, bits: int, endpoint: Endpoint) -> Bus:
    return Bus(
        name,
        StandardProtocol.AXIS,
        (
            Member("tdata", name + "_tdata", bits),
            Member("tvalid", name + "_tvalid"),
            Member("tready", name + "_tready"),
        ),
        endpoint=endpoint,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


def _wire_mvau(
    *,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    pe: int,
    simd: int,
    compute: ModuleBuildRequirements,
    result_dtype: QONNXDataType,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[Sequence[int]] | None = None,
) -> MVAUAssembly:
    """Wire accepted component requirements into the supported MVAU assembly."""
    traversal = _Traversal(repetitions, matrix_width, matrix_height, pe, simd)
    activation_dtype = canonical_qonnx_datatype(activation_dtype)
    weights_dtype = canonical_qonnx_datatype(weights_dtype)
    if not isinstance(weight_delivery, WeightDelivery):
        raise ValueError("weight_delivery must be a WeightDelivery value")
    if (weight_delivery is WeightDelivery.CYCLIC) != (weights is not None):
        raise ValueError("cyclic delivery requires weights; external delivery has no initializer")
    a_bits, w_bits, y_bits = (
        activation_dtype.bitwidth(),
        weights_dtype.bitwidth(),
        result_dtype.bitwidth(),
    )
    a_payload, w_payload, y_payload = simd * a_bits, pe * simd * w_bits, pe * y_bits
    a_carrier, w_carrier, y_carrier = (
        (bits + 7) // 8 * 8 for bits in (a_payload, w_payload, y_payload)
    )
    replay = replay_buffer_requirements(
        word_bits=a_payload,
        sequence_length=traversal.synapse_folds,
        replay_count=traversal.neuron_folds,
    )
    instances = [ModuleInstance("u_replay", replay), ModuleInstance("u_compute", compute)]
    initializer = () if weights is None else traversal.weight_image(weights, weights_dtype)
    if weight_delivery is WeightDelivery.CYCLIC:
        instances.append(
            ModuleInstance(
                "u_weights",
                cyclic_stream_requirements(
                    word_bits=w_payload, depth=len(initializer), image=initializer
                ),
            )
        )
    top_abi = ModuleABIRequirements(
        GeneratedModuleName("finn_mvau_" + weight_delivery.value),
        (
            *(port for port in compute.abi.ports if not isinstance(port, Bus)),
            _axis("in0_V", a_carrier, Endpoint.TARGET),
            *(
                (_axis("in1_V", w_carrier, Endpoint.TARGET),)
                if weight_delivery is WeightDelivery.EXTERNAL
                else ()
            ),
            _axis("out0_V", y_carrier, Endpoint.INITIATOR),
        ),
        (),
        compute.abi.clock_alignments,
    )
    wires: list[PhysicalWire] = []
    ignored: list[PinSlice] = []

    def connect(destination: PinSlice, source: PinSlice, *, invert: bool = False) -> None:
        wires.append(PhysicalWire(destination, source, invert=invert))

    def fields(
        destination: tuple[str | None, str], source: tuple[str | None, str], count: int, bits: int
    ) -> None:
        for index in range(count):
            connect(_slice(*destination, bits, index * bits), _slice(*source, bits, index * bits))

    def zero_padding(owner: str | None, name: str, payload: int, carrier: int) -> None:
        if carrier > payload:
            wires.append(
                PhysicalWire(
                    _slice(owner, name, carrier - payload, payload),
                    ConstantBits(carrier - payload, 0),
                )
            )

    def controls(
        destination: tuple[str | None, str, str], source: tuple[str | None, str, str]
    ) -> None:
        connect(_slice(destination[0], destination[1]), _slice(source[0], source[1]))
        connect(_slice(source[0], source[2]), _slice(destination[0], destination[2]))

    for name in ("ap_clk", "ap_clk2x", "ap_rst_n"):
        connect(_slice("u_compute", name), _slice(None, name))
    for owner in ("u_replay",) + (("u_weights",) if initializer else ()):
        connect(_slice(owner, "clk"), _slice(None, "ap_clk"))
        connect(_slice(owner, "rst"), _slice(None, "ap_rst_n"), invert=True)
    fields(("u_replay", "idat"), (None, "in0_V_tdata"), simd, a_bits)
    controls(("u_replay", "ivld", "irdy"), (None, "in0_V_tvalid", "in0_V_tready"))
    fields(("u_compute", "s_axis_input_tdata"), ("u_replay", "odat"), simd, a_bits)
    controls(
        ("u_compute", "s_axis_input_tvalid", "s_axis_input_tready"), ("u_replay", "ovld", "ordy")
    )
    connect(_slice("u_compute", "s_axis_input_tlast"), _slice("u_replay", "olast"))
    zero_padding("u_compute", "s_axis_input_tdata", a_payload, a_carrier)
    source = (None, "in1_V_tdata") if not initializer else ("u_weights", "odat")
    fields(("u_compute", "s_axis_weights_tdata"), source, pe * simd, w_bits)
    controls(
        ("u_compute", "s_axis_weights_tvalid", "s_axis_weights_tready"),
        (None, "in1_V_tvalid", "in1_V_tready")
        if not initializer
        else ("u_weights", "ovld", "ordy"),
    )
    zero_padding("u_compute", "s_axis_weights_tdata", w_payload, w_carrier)
    fields((None, "out0_V_tdata"), ("u_compute", "m_axis_output_tdata"), pe, y_bits)
    if y_carrier > y_payload:
        connect(
            _slice(None, "out0_V_tdata", y_carrier - y_payload, y_payload),
            _slice("u_compute", "m_axis_output_tdata", y_carrier - y_payload, y_payload),
        )
    controls(
        (None, "out0_V_tvalid", "out0_V_tready"),
        ("u_compute", "m_axis_output_tvalid", "m_axis_output_tready"),
    )
    for name, payload, carrier in (("in0_V_tdata", a_payload, a_carrier),) + (
        (("in1_V_tdata", w_payload, w_carrier),) if not initializer else ()
    ):
        if carrier > payload:
            ignored.append(_slice(None, name, carrier - payload, payload))
    structure = PhysicalStructure(
        top_abi,
        tuple(instances),
        tuple(wires),
        (
            UnusedOutput(
                PhysicalPin("u_replay", "ofin"),
                "olast closes each accumulation; top stream lengths follow the workload extents",
            ),
        ),
        tuple(ignored),
    )
    wrapper = RenderedSourceRequirement(
        EntryPointSourceName(),
        "decomposed_wrapper.sv.j2",
        ("PORT_DECLARATIONS", "NET_DECLARATIONS", "ASSIGNMENTS", "INSTANCES"),
        SELF_CONTAINED_JINJA_RENDERER,
        requires=tuple(
            "module:" + cast(FixedModuleName, instance.requirements.abi.entry_point).value
            for instance in instances
        ),
        provides_entry_point=True,
    )
    requirements = lower_module_structure(
        structure,
        producer=ProducerIdentity("finn.mvau." + weight_delivery.value, "1"),
        wrapper_template=wrapper,
    )
    return MVAUAssembly(
        traversal.activation_beats,
        traversal.weight_beats,
        traversal.result_beats,
        result_dtype,
        weight_delivery,
        structure,
        requirements,
        initializer,
    )


class MVAU(Space):
    """Workload parameters, folding/delivery choices, and a bound dotp child Space."""

    repetitions = Param(int)
    matrix_width = Param(int)
    matrix_height = Param(int)
    activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp = Param(DspBlock)
    segment_length = Param(int)
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))
    weight_delivery = Decision(WeightDelivery, values=tuple(WeightDelivery))

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_type(
        *, matrix_width: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
    ) -> Answer[QONNXDataType]:
        try:
            return Decided(exact_result_dtype(matrix_width, activation_dtype, weights_dtype))
        except ValueError as error:
            return reject("mvau-arithmetic", str(error))

    @constraint
    def dimensions_supported(
        *, repetitions: int, matrix_width: int, matrix_height: int, pe: int, simd: int
    ) -> Answer[bool]:
        try:
            _Traversal(repetitions, matrix_width, matrix_height, pe, simd)
        except ValueError as error:
            return reject("mvau-folding", str(error))
        return Decided(True)

    compute = Subspace(
        DotpAxiKernel,
        bindings={
            DotpAxiKernel.activation.dtype: activation_dtype,
            DotpAxiKernel.weights.dtype: weights_dtype,
            DotpAxiKernel.result.dtype: result_type,
        },
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
    )

    dimensions = ConstraintGroup(dimensions_supported)

    def assemble(self, weights: Sequence[Sequence[int]] | None = None) -> MVAUAssembly:
        accepted = self.compute.physical().accepted_answer
        if not isinstance(accepted, Decided):
            details = "; ".join(
                f"{finding.code}: {finding.message}" for finding in accepted.findings
            )
            raise ValueError(f"dotp physical View is not accepted: {details}")
        return _wire_mvau(
            repetitions=self.repetitions,
            matrix_width=self.matrix_width,
            matrix_height=self.matrix_height,
            activation_dtype=self.activation_dtype,
            weights_dtype=self.weights_dtype,
            pe=self.pe,
            simd=self.simd,
            compute=accepted.value,
            result_dtype=self.result_type,
            weight_delivery=self.weight_delivery,
            weights=weights,
        )


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
) -> MVAUAssembly:
    """Bind workload and folding choices, then wire the accepted dotp implementation."""
    try:
        point = MVAU.start(
            {
                MVAU.repetitions: repetitions,
                MVAU.matrix_width: matrix_width,
                MVAU.matrix_height: matrix_height,
                MVAU.activation_dtype: activation_dtype,
                MVAU.weights_dtype: weights_dtype,
                MVAU.target_dsp: target_dsp,
                MVAU.segment_length: segment_length,
            }
        )
        report = point.refine(
            point.edit(MVAU.pe, pe),
            point.edit(MVAU.simd, simd),
            point.edit(MVAU.weight_delivery, weight_delivery),
            point.compute.edit(DotpAxiKernel.compute_pumping, compute_pumping),
        )
    except RequestError as error:
        raise ValueError(str(error)) from error
    if not report.accepted:
        details = "; ".join(
            f"{finding.code}: {finding.message}"
            for outcome in report.outcomes
            if not isinstance(outcome.answer, Decided)
            for finding in outcome.answer.findings
        )
        raise ValueError(f"MVAU choices are not accepted: {details}")
    return report.point.assemble(weights)


__all__ = ["MVAU", "MVAUAssembly", "WeightDelivery", "exact_result_dtype", "mvau_assembly"]
