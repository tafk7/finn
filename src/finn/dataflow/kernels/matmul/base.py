# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What every matrix-multiplication Kernel agrees about before supply differs.

The folding and operand facts are authored once and reused by each concrete
Kernel. Compilation still gives every occurrence its own root-relative
coordinates: ``kernel.dot_product.pe`` and ``kernel.batch_interleaved.pe`` are
distinct persisted Decisions even though both come from the same Python
declaration object. The shared definition prevents duplicated authoring; it
does not merge choices across alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from finn.dataflow.model.kernel import Kernel
from finn.dataflow.analysis.integer_dot import (
    DotProductBounds,
    IntegerRange,
    IntegerSupportReport,
    analyze_integer_dot_ranges,
)
from finn.dataflow.model.logical.datatypes import QONNXDataType
from finn.dataflow.model.logical.composition import LogicalResult, logical_network
from finn.dataflow.model.logical.interface import (
    PublicOperand,
    OperandExport,
    OperandTarget,
    body_operand,
)
from finn.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.model.logical.network import PositionMap, RegionEndpoint
from finn.dataflow.model.logical.refs import DataflowOperandRef, RegionInputRef, RegionOutputRef
from finn.dataflow.model.logical.region import InputInterface
from finn.dataflow.kernels.typing import operand_types_supported, operand_widths_supported
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Space,
    Projection,
    Readiness,
    constraint,
    derived,
    allow_absent,
    divisors_of,
    reject,
)
from finn.dataflow.model.logical.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


class AccumulationMode(str, Enum):
    """How matrix products are accumulated."""

    INTEGER = "integer"
    XNOR_POPCOUNT = "xnor_popcount"
    BIPOLAR_POPCOUNT = "bipolar_popcount"


class ActivationMode(str, Enum):
    """What happens to the accumulator afterwards."""

    NONE = "none"
    MULTITHRESHOLD = "multithreshold"


@dataclass(frozen=True, slots=True)
class MvauComputationProfile:
    """The independent accumulation and post-accumulation semantics."""

    accumulation: AccumulationMode
    activation: ActivationMode

    @property
    def fuses_activation(self) -> bool:
        return self.activation is ActivationMode.MULTITHRESHOLD

    @property
    def name(self) -> str:
        return f"{self.accumulation.value}+{self.activation.value}"


def computation_profile(
    *,
    no_activation: bool,
    binary_xnor: bool,
    activation_type: QONNXDataType | None = None,
    weight_type: QONNXDataType | None = None,
) -> MvauComputationProfile:
    """Derive the reusable mathematical profile from source-level facts."""

    from qonnx.core.datatype import DataType  # type: ignore[import-not-found]  # noqa: PLC0415

    bipolar = DataType["BIPOLAR"]
    if binary_xnor:
        accumulation = AccumulationMode.XNOR_POPCOUNT
    elif activation_type == bipolar and weight_type == bipolar:
        accumulation = AccumulationMode.BIPOLAR_POPCOUNT
    else:
        accumulation = AccumulationMode.INTEGER
    return MvauComputationProfile(
        accumulation,
        ActivationMode.NONE if no_activation else ActivationMode.MULTITHRESHOLD,
    )


class MatmulInterface(Space):
    """Common matrix facts, independent of implementation selection or folding."""

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    computation_profile = Input(MvauComputationProfile)
    integer_bounds = Input(DotProductBounds, allow_absent=True)

    @derived(
        QONNX_DATATYPE_VALUE_SEMANTICS,
        accumulator=accumulator_type,
        output=output_type,
        profile=computation_profile,
    )
    def result_type(
        *, accumulator: QONNXDataType, output: QONNXDataType, profile: MvauComputationProfile
    ) -> QONNXDataType:
        # A fused threshold's declared output precision is semantic quantization.
        # This common type facet does not admit a concrete fused implementation;
        # each concrete Kernel still checks its own computation profile.
        return output if profile.fuses_activation else accumulator

    @constraint(accumulator=accumulator_type, output=output_type, profile=computation_profile)
    def result_requirement(
        *, accumulator: QONNXDataType, output: QONNXDataType, profile: MvauComputationProfile
    ) -> object:
        if profile.fuses_activation:
            return True
        if output != accumulator:
            return reject(
                "output-accumulator-mismatch",
                "bare matrix output must equal the accumulator requirement",
            )
        return True

    @constraint(
        activation=activation_type,
        weight=weight_type,
        accumulator=accumulator_type,
        width=matrix_width,
        bounds=allow_absent(integer_bounds),
        profile=computation_profile,
    )
    def accumulator_precision(
        *,
        activation: QONNXDataType,
        weight: QONNXDataType,
        accumulator: QONNXDataType,
        width: int,
        bounds: object,
        profile: MvauComputationProfile,
    ) -> object:
        if profile.accumulation is not AccumulationMode.INTEGER:
            return True
        if not all(datatype.is_integer() for datatype in (activation, weight, accumulator)):
            return reject(
                "mvau-integer-types", "integer accumulation requires integer element types"
            )
        supported = operand_types_supported(activation, weight, accumulator, accumulator)
        if supported is not True:
            return supported
        low: int | float
        high: int | float
        if isinstance(bounds, DotProductBounds):
            low, high = bounds.every_intermediate.minimum, bounds.every_intermediate.maximum
        else:
            # A standalone point without authenticated value facts uses complete
            # datatype bounds, never a delivery-mode-based fixed-weight guess.
            conservative = analyze_integer_dot_ranges(
                IntegerRange(int(activation.min()), int(activation.max())),
                ((IntegerRange(int(weight.min()), int(weight.max())),) * width,),
            )
            low = conservative.every_intermediate.minimum
            high = conservative.every_intermediate.maximum
        if accumulator.min() > low or accumulator.max() < high:
            return reject(
                "mvau-accumulator-precision",
                "accumulator cannot contain every justified intermediate",
                values={"minimum": low, "maximum": high, "accumulator": accumulator.name},
            )
        return True

    @constraint(
        activation=activation_type,
        weight=weight_type,
        accumulator=accumulator_type,
        output=output_type,
        profile=computation_profile,
    )
    def element_types_supported(
        *,
        activation: QONNXDataType,
        weight: QONNXDataType,
        accumulator: QONNXDataType,
        output: QONNXDataType,
        profile: MvauComputationProfile,
    ) -> object:
        supported = operand_types_supported(
            activation, weight, accumulator, accumulator if profile.fuses_activation else output
        )
        return operand_widths_supported(activation, weight) if supported is True else supported

    type_support = ConstraintGroup(
        result_requirement, accumulator_precision, element_types_supported
    )
    activation_type_ready = Readiness()
    weight_type_ready = Readiness()
    result_type_ready = Readiness(properties=(result_type,), constraints=type_support)
    public_activation_type = Projection(activation_type, readiness=activation_type_ready)
    public_weight_type = Projection(weight_type, readiness=weight_type_ready)
    public_result_type = Projection(
        result_type, readiness=result_type_ready, constraints=type_support
    )

    @derived(RectangularDomain, rows=repetitions, width=matrix_width)
    def activation_domain(*, rows: int, width: int) -> RectangularDomain:
        return RectangularDomain((rows, width))

    @derived(RectangularDomain, width=matrix_width, height=matrix_height)
    def weight_domain(*, width: int, height: int) -> RectangularDomain:
        return RectangularDomain((width, height))

    @derived(RectangularDomain, rows=repetitions, height=matrix_height)
    def result_domain(*, rows: int, height: int) -> RectangularDomain:
        return RectangularDomain((rows, height))

    activation_domain_ready = Readiness(properties=(activation_domain,))
    weight_domain_ready = Readiness(properties=(weight_domain,))
    result_domain_ready = Readiness(properties=(result_domain,))
    public_activation_domain = Projection(activation_domain, readiness=activation_domain_ready)
    public_weight_domain = Projection(weight_domain, readiness=weight_domain_ready)
    public_result_domain = Projection(result_domain, readiness=result_domain_ready)

    public_operands: ClassVar[tuple[PublicOperandDeclaration, ...]] = (
        PublicOperandDeclaration(
            "activation", "input", public_activation_type, public_activation_domain
        ),
        PublicOperandDeclaration("weights", "input", public_weight_type, public_weight_domain),
        PublicOperandDeclaration("result", "output", public_result_type, public_result_domain),
    )


def matrix_operand_export(public: PublicOperand, logical: LogicalResult) -> OperandExport:
    """The family owns private roles and its W[K,N] -> W[N,K] view."""
    network = logical_network(logical)
    refs: tuple[DataflowOperandRef, ...]
    if public.key == "activation":
        # The actual external endpoint distinguishes replay from direct compute.
        boundary = next(item for item in network.boundaries if item.id == "activation")
        region = network.node(boundary.endpoint.node_id).region
        operand = region.input_interface(boundary.endpoint.port_id).operand
        refs = (RegionInputRef(boundary.endpoint.node_id, operand.id),)
    elif public.key == "weights":
        # The public requirement is the supplier's input. The canonical edge
        # carries it onward; exporting compute's downstream use again would
        # incorrectly turn an unported required value into a stream binding.
        supplier = "memory" if any(node.id == "memory" for node in network.nodes) else "compute"
        refs = tuple(RegionInputRef(node.id, "W") for node in network.nodes if node.id == supplier)
    else:
        boundary = next(item for item in network.boundaries if item.id == "output")
        region = network.node(boundary.endpoint.node_id).region
        operand = region.output_interface(boundary.endpoint.port_id).port.operand
        refs = (RegionOutputRef(boundary.endpoint.node_id, operand.id),)
    targets = []
    for ref in refs:
        operand = body_operand(network, ref)
        if public.key == "weights":
            width, height = public.domain.extents
            mapping = PositionMap.affine(
                public.domain,
                view_extents=(width, height),
                sink=operand.position_domain,
                offset=0,
                coefficients=(1, width),
            )
        else:
            mapping = PositionMap.row_major_reshape(public.domain, operand.position_domain)
        region = network.node(ref.node_id).region
        ports = (
            tuple(item.port for item in region.inputs if isinstance(item, InputInterface))
            if isinstance(ref, RegionInputRef)
            else tuple(item.port for item in region.outputs)
        )
        exposed = {boundary.endpoint for boundary in network.boundaries}
        presentations = tuple(
            RegionEndpoint(ref.node_id, port.id)
            for port in ports
            if port.operand.id == ref.operand_id and RegionEndpoint(ref.node_id, port.id) in exposed
        )
        targets.append(OperandTarget(ref, mapping, presentations))
    return OperandExport(public, tuple(targets))


class WeightedDotProductKernel(Kernel, MatmulInterface):
    """The shared operand facts and the two folding choices MVAU owns."""

    narrow_weights = Input(bool)
    target_dsp = Input(DspBlock)
    clock_period_ns = Input(float)
    numerical_support = Input(IntegerSupportReport, allow_absent=True)

    repetitions = MatmulInterface.repetitions
    matrix_width = MatmulInterface.matrix_width
    matrix_height = MatmulInterface.matrix_height
    computation_profile = MatmulInterface.computation_profile
    public_operands = tuple(
        PublicOperandDeclaration(
            item.key, item.direction, item.datatype, item.domain, matrix_operand_export
        )
        for item in MatmulInterface.public_operands
    )

    #: Owned here because each of them changes both Regions and their edge.
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    @constraint(profile=computation_profile)
    def computes_a_bare_accumulator(*, profile: MvauComputationProfile) -> object:
        """These Kernels build a dot product and stop; they fuse no activation.

        A Kernel limitation, argued from the Kernel's own structure rather than
        from the mathematics: every alternative below this class declares a
        Network with activation, weight and output boundaries and a compute
        Region that produces the accumulator directly.  There is no threshold
        boundary for the fourth operand to cross and no stage to apply it in,
        so a fused-threshold node has no *composition* here -- while remaining
        a perfectly valid problem that a later Kernel may build.

        Refusing it here rather than in the operation is what keeps that true:
        the node still binds, still projects its source facts, and reports an
        inapplicable Kernel instead of an unreadable node.
        """

        if not profile.fuses_activation:
            return True
        return reject(
            "mvau-kernel-fuses-no-activation",
            "this Kernel emits its accumulator directly and has no stage for a fused "
            f"threshold; this node computes {profile.name}",
            values={"computation_profile": profile.name},
        )

    logical_support = ConstraintGroup(computes_a_bare_accumulator)


#: Every Input the shared base consumes, for a caller assembling bindings.
SHARED_INPUTS = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "weight_type",
    "accumulator_type",
    "output_type",
    "narrow_weights",
    "target_dsp",
    "clock_period_ns",
    "computation_profile",
    "numerical_support",
    "integer_bounds",
)

__all__ = [
    "AccumulationMode",
    "ActivationMode",
    "DspBlock",
    "MvauComputationProfile",
    "MatmulInterface",
    "SHARED_INPUTS",
    "WeightedDotProductKernel",
    "computation_profile",
]
