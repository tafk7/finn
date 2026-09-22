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
from finn.dataflow.model.logical.datatypes import QONNXDataType, resolve_qonnx_datatype_name
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


def matrix_result_requirement(
    *, no_activation: bool, output: QONNXDataType, accumulator: QONNXDataType
) -> object:
    """The single bare-result precision requirement consumed by source and Kernel."""

    if no_activation and output != accumulator:
        return reject(
            "output-accumulator-mismatch",
            "noActivation requires outputDataType == accDataType",
        )
    return True


def accumulator_type_for_bounds(bounds: DotProductBounds) -> QONNXDataType:
    """Smallest exact signed requirement covering all justified partial sums.

    This is a semantic precision derivation for an admitted source conversion,
    not a folding proposal or a promise that any particular DSP can realize it.
    """

    return resolve_qonnx_datatype_name(f"INT{bounds.minimum_signed_accumulator_bits}")


def _popcount_types_supported(
    profile: MvauComputationProfile,
    activation: QONNXDataType,
    weight: QONNXDataType,
    accumulator: QONNXDataType,
) -> object:
    """Operand meaning and count representation, independent of any Dotp generator."""
    expected = "BINARY" if profile.accumulation is AccumulationMode.XNOR_POPCOUNT else "BIPOLAR"
    if activation.name != expected or weight.name != expected:
        return reject(
            "mvau-popcount-operands",
            f"{profile.accumulation.value} requires two {expected} operands",
            values={"activation": activation.name, "weight": weight.name},
        )
    # BIPOLAR has an integer range but cannot represent the zero count.
    if not accumulator.is_integer() or accumulator.name == "BIPOLAR":
        return reject(
            "mvau-popcount-accumulator-type",
            "a popcount accumulator must represent integer counts including zero",
            values={"accumulator": accumulator.name},
        )
    return True


def _integer_types_supported(
    activation: QONNXDataType, weight: QONNXDataType, accumulator: QONNXDataType
) -> object:
    """Semantic integer arithmetic, without generator encoding or lane limits."""
    if not all(datatype.is_integer() for datatype in (activation, weight, accumulator)):
        return reject("mvau-integer-types", "integer accumulation requires integer element types")
    if accumulator.name == "BIPOLAR":
        return reject(
            "mvau-integer-accumulator-type",
            "an integer accumulator must represent zero and all bounded intermediate sums",
        )
    return True


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
        return matrix_result_requirement(
            no_activation=not profile.fuses_activation, output=output, accumulator=accumulator
        )

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
            supported = _popcount_types_supported(profile, activation, weight, accumulator)
            if supported is not True:
                return supported
            if accumulator.min() > 0 or accumulator.max() < width:
                return reject(
                    "mvau-accumulator-precision",
                    "accumulator cannot contain every count from zero through the matrix width",
                    values={"minimum": 0, "maximum": width, "accumulator": accumulator.name},
                )
            return True
        supported = _integer_types_supported(activation, weight, accumulator)
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
        if profile.accumulation is not AccumulationMode.INTEGER:
            return _popcount_types_supported(profile, activation, weight, accumulator)
        return _integer_types_supported(activation, weight, accumulator)

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

    @constraint(
        activation=activation_type,
        weight=weight_type,
        accumulator=accumulator_type,
        output=output_type,
    )
    def primitive_integer_types_supported(
        *,
        activation: QONNXDataType,
        weight: QONNXDataType,
        accumulator: QONNXDataType,
        output: QONNXDataType,
    ) -> object:
        supported = operand_types_supported(activation, weight, accumulator, output)
        return operand_widths_supported(activation, weight) if supported is True else supported

    @constraint(profile=computation_profile)
    def bare_integer_profile(*, profile: MvauComputationProfile) -> object:
        if profile.accumulation is AccumulationMode.INTEGER and not profile.fuses_activation:
            return True
        return reject(
            "mvau-integer-type-profile", "this type profile requires bare integer products"
        )

    primitive_integer_support = ConstraintGroup(
        primitive_integer_types_supported, bare_integer_profile
    )
    integer_type_profile = Projection(
        result_type,
        readiness=result_type_ready,
        constraints=(type_support, primitive_integer_support),
    )
    # This projection establishes only the shared generator type profile. It
    # neither selects a Kernel nor establishes target, folding or codegen readiness.

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
        """These Kernels multiply integers and stop; they fuse no activation.

        Equality-count profiles likewise need a different implementation, even
        when their independently assessed family result type is available.

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

        if profile.fuses_activation:
            return reject(
                "mvau-kernel-fuses-no-activation",
                "this Kernel emits its accumulator directly and has no stage for a fused "
                f"threshold; this node computes {profile.name}",
                values={"computation_profile": profile.name},
            )
        if profile.accumulation is not AccumulationMode.INTEGER:
            return reject(
                "mvau-kernel-accumulation-unsupported",
                "this Kernel computes integer products, not equality counts",
                values={"computation_profile": profile.name},
            )
        return True

    logical_support = ConstraintGroup(
        computes_a_bare_accumulator,
        *MatmulInterface.primitive_integer_support.constraints,
    )


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
    "matrix_result_requirement",
    "accumulator_type_for_bounds",
]
