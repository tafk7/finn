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
dotp child Space and passes that child's accepted physical View to one of two
weight-delivery families, selected by the ``implementation`` structural choice.
``ExternalWeights`` adds a top-level weight stream; ``CyclicWeights`` owns an
initialized ROM, its ``rom_style`` and the optional ``weights`` fact. Both export
typed assembly and requirements views. ``mvau_assembly`` supplies concrete facts
and choices to this same path; its wiring code only receives accepted component
requirements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from finn.kernels.artifacts.abi import Bus, Endpoint
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
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import StreamContract, StreamMismatch
from finn.kernels.physical.forms import TRAVERSAL, Every, Traversal, tile, vector_major
from finn.kernels.physical.lowering import lower_module_structure
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.streaming import replay_buffer_contracts, replay_buffer_requirements
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
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    divisors_of,
    reject,
    view,
)
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


def _wire_mvau(
    *,
    folding: _Folding,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    result_dtype: QONNXDataType,
    compute: ModuleBuildRequirements,
    ports: tuple[AxiStream, ...],
    weight_source: tuple[ModuleBuildRequirements, StreamContract] | None,
    initializer: tuple[int, ...] = (),
) -> MVAUAssembly:
    """Compose replay, compute and an optional weight source through checked streams.

    The traversals state the MVAU order: vector-major activations are replayed
    once per neuron fold, and weight tiles repeat once per vector. Without a weight source,
    the weight stream is a top-level port. ``ports`` are dotp's accepted streams.
    """
    x, w, y = (ScalarEncoding(dtype) for dtype in (activation_dtype, weights_dtype, result_dtype))
    t = folding
    external = weight_source is None
    weight_delivery = WeightDelivery.EXTERNAL if external else WeightDelivery.CYCLIC
    clocking = dict(clock="ap_clk", reset="ap_rst_n")

    def top(
        name: str, element: ScalarEncoding, form: Traversal, endpoint: Endpoint
    ) -> StreamContract:
        stream = AxiStream(name, element.dtype, form.lanes, endpoint=endpoint)
        return StreamContract(stream.native(**clocking), element, form)

    x_form = vector_major((t.repetitions, t.matrix_width), t.simd)
    w_form = tile(t.matrix_height, t.matrix_width, t.pe, t.simd).repeated(t.repetitions)
    y_form = vector_major((t.repetitions, t.matrix_height), t.pe)
    x_top = top("in0_V", x, x_form, Endpoint.TARGET)
    w_top = top("in1_V", w, w_form, Endpoint.TARGET)
    y_top = top("out0_V", y, y_form, Endpoint.INITIATOR)
    replay_in, replay_out = replay_buffer_contracts(
        x, x_form, sequence_length=t.synapse_folds, replay_count=t.neuron_folds
    )
    activation, weights, result = (port.native(**clocking) for port in ports)

    top_abi = ModuleABIRequirements(
        GeneratedModuleName("finn_mvau_" + weight_delivery.value),
        (
            *(port for port in compute.abi.ports if not isinstance(port, Bus)),
            *(c.transport.axis_bus() for c in (x_top, *((w_top,) if external else ()), y_top)),
        ),
        (),
        compute.abi.clock_alignments,
    )
    composition = Composition(top_abi)
    composition.add("u_replay", _replay(x, t))
    composition.add("u_compute", compute)
    for pin in ("ap_clk", "ap_clk2x", "ap_rst_n"):
        composition.drive("u_compute", pin, pin)
    children = ("u_replay",) if external else ("u_replay", "u_weights")
    if weight_source is not None:
        composition.add("u_weights", weight_source[0])
    for owner in children:
        composition.drive(owner, "clk", "ap_clk")
        composition.drive(owner, "rst", "ap_rst_n")

    composition.connect(StreamEnd(None, x_top), StreamEnd("u_replay", replay_in))
    composition.connect(
        StreamEnd("u_replay", replay_out),
        StreamEnd(
            "u_compute",
            StreamContract(
                activation,
                x,
                replay_out.form,
                markers={activation.markers[0].signal: Every(t.synapse_folds)},
            ),
        ),
    )
    composition.connect(
        StreamEnd(None, w_top)
        if weight_source is None
        else StreamEnd("u_weights", weight_source[1]),
        StreamEnd("u_compute", StreamContract(weights, w, w_form)),
    )
    composition.connect(
        StreamEnd("u_compute", StreamContract(result, y, y_form)), StreamEnd(None, y_top)
    )
    structure = composition.finish()
    wrapper = RenderedSourceRequirement(
        EntryPointSourceName(),
        "decomposed_wrapper.sv.j2",
        ("PORT_DECLARATIONS", "NET_DECLARATIONS", "ASSIGNMENTS", "INSTANCES"),
        SELF_CONTAINED_JINJA_RENDERER,
        requires=tuple(
            "module:" + cast(FixedModuleName, instance.requirements.abi.entry_point).value
            for instance in structure.instances
        ),
        provides_entry_point=True,
    )
    requirements = lower_module_structure(
        structure,
        producer=ProducerIdentity("finn.mvau." + weight_delivery.value, "1"),
        wrapper_template=wrapper,
    )
    return MVAUAssembly(
        t.activation_beats,
        t.weight_beats,
        t.result_beats,
        result_dtype,
        weight_delivery,
        structure,
        requirements,
        initializer,
    )


def _replay(element: ScalarEncoding, folding: _Folding) -> ModuleBuildRequirements:
    return replay_buffer_requirements(
        word_bits=folding.simd * element.bits,
        sequence_length=folding.synapse_folds,
        replay_count=folding.neuron_folds,
    )


FOLDING = default_semantics(_Folding)
MODULE_BUILD = default_semantics(ModuleBuildRequirements)
ASSEMBLY = default_semantics(MVAUAssembly)
AXI_PORTS = default_semantics(tuple)
ASSEMBLY_VIEW = ViewKey("assembly", ASSEMBLY)
BUILD_VIEW = ViewKey("build_requirements", MODULE_BUILD)


class WeightDeliveryFamily(Space):
    """Facts shared by the weight-delivery families: accepted folding and compute.

    ``compute`` and ``ports`` are the parent's accepted dotp requirements and
    streams, so a family cannot wire a compute core whose physical View was refused.
    """

    folding = Param(FOLDING)
    activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    compute = Param(MODULE_BUILD)
    ports = Param(AXI_PORTS)

    def _assemble(
        self,
        weight_source: tuple[ModuleBuildRequirements, StreamContract] | None,
        initializer: tuple[int, ...] = (),
    ) -> MVAUAssembly | Rejected:
        try:
            return _wire_mvau(
                folding=self.folding,
                activation_dtype=self.activation_dtype,
                weights_dtype=self.weights_dtype,
                result_dtype=self.result_dtype,
                compute=self.compute,
                ports=self.ports,
                weight_source=weight_source,
                initializer=initializer,
            )
        except StreamMismatch as error:
            return reject("mvau-stream", str(error))


class ExternalWeights(WeightDeliveryFamily):
    """Weights arrive on the top-level ``in1_V`` stream, once per repetition."""

    @view(semantics=ASSEMBLY)
    def assembly(self) -> MVAUAssembly | Rejected:
        return self._assemble(None)

    @view(semantics=MODULE_BUILD)
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.assembly().requirements

    exports = {ASSEMBLY_VIEW: assembly, BUILD_VIEW: build_requirements}


class CyclicWeights(WeightDeliveryFamily):
    """A reusable ``CyclicDelivery`` kernel streams the weight tiles; there is no weight port.

    The family supplies the weight values and the tile form dotp reads; the
    delivery kernel owns packing, its ``rom_style`` choice and its build. The
    connection to dotp is checked by stream contract, not wired by hand. Without
    weights the source's image and this family's views stay unresolved.
    """

    weights = Param(INTEGER_TENSOR)

    @derived(semantics=TRAVERSAL)
    def weight_form(self) -> Traversal:
        t = self.folding
        return tile(t.matrix_height, t.matrix_width, t.pe, t.simd)

    source = Subspace(
        CyclicDelivery, dtype=WeightDeliveryFamily.weights_dtype, form=weight_form, values=weights
    )

    @view(semantics=ASSEMBLY)
    def assembly(self) -> MVAUAssembly | Rejected:
        source = self.source
        return self._assemble(
            (source.build_requirements(), source.output()), initializer=source.image
        )

    @view(semantics=MODULE_BUILD)
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.assembly().requirements

    exports = {ASSEMBLY_VIEW: assembly, BUILD_VIEW: build_requirements}


class MVAU(Space):
    """Workload facts, folding choices, a dotp child, and a weight-delivery family.

    ``implementation`` selects ``external`` or ``cyclic`` delivery. The families
    share typed assembly and requirements exports but keep their own ports,
    initialization facts and local choices; only the selected one is evaluated.
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

    compute = Subspace(
        DotpAxiKernel,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_type,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
    )

    compute_requirements = compute.accepted(DotpAxiKernel.build_requirements)
    compute_ports = compute.accepted(DotpAxiKernel.interfaces)
    implementation = SubspaceChoice(
        {
            WeightDelivery.EXTERNAL.value: Subspace(
                ExternalWeights,
                folding=folding,
                activation_dtype=activation_dtype,
                weights_dtype=weights_dtype,
                result_dtype=result_type,
                compute=compute_requirements,
                ports=compute_ports,
            ),
            WeightDelivery.CYCLIC.value: Subspace(
                CyclicWeights,
                folding=folding,
                activation_dtype=activation_dtype,
                weights_dtype=weights_dtype,
                result_dtype=result_type,
                compute=compute_requirements,
                ports=compute_ports,
                weights=weights,
            ),
        },
        exports=(ASSEMBLY_VIEW, BUILD_VIEW),
    )

    assembly = View(implementation.accepted(ASSEMBLY_VIEW), constraints=(dimensions,))
    build_requirements = View(implementation.accepted(BUILD_VIEW), constraints=(dimensions,))


CYCLIC_ROM_STYLE = CyclicWeights.source.decision_ref(CyclicDelivery.rom_style)


def _findings(results: Sequence[QueryResult[Any]]) -> str:
    return "; ".join(
        f"{finding.owner}: {finding.code}: {finding.message}"
        for result in results
        if not isinstance(result, Available)
        for finding in result.findings
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
    rom_style: str = "auto",
) -> MVAUAssembly:
    """Bind workload facts, select one delivery family and its choices, then assemble.

    Weights are required by, and only accepted with, cyclic delivery. ``rom_style``
    applies to cyclic delivery; the ``auto`` default leaves memory inference to
    synthesis, as the ROM did before the choice existed.
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
    try:
        point = cast(MVAU, MVAU(**facts).implementation.select(case).instance)
        changes: list[ChangeRequest] = [
            point.compute.field(DotpAxiKernel.compute_pumping).change(compute_pumping)
        ]
        if cyclic:
            family = point.implementation.alternative(case)
            changes.append(family.field(CYCLIC_ROM_STYLE).change(rom_style))
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
    "ASSEMBLY_VIEW",
    "BUILD_VIEW",
    "CyclicWeights",
    "ExternalWeights",
    "MVAU",
    "MVAUAssembly",
    "WeightDelivery",
    "WeightDeliveryFamily",
    "exact_result_dtype",
    "mvau_assembly",
]
