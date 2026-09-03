# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU as one source node over a closed set of Designs.

The operation is thin on purpose.  Everything below it -- the folding, the
Regions, the topology, the weight path -- belongs to the Designs and the
Kernels, and everything above it belongs to the graph.  What lives here is the
part only this operation can say: which two Designs are its alternatives, how a
matrix-vector node's tensors become the facts they read, and where each of
those tensors ends up in whichever Network is selected.

The operation *is* the root Space.  Its Problem members are its declared
tensors and attributes, lowered from the source schema, and its ``design``
Variant is an ordinary structural choice on the same class.  There is no
separate source Space and no wrapper between the node and the point.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.declarations import (
    ConstraintGroup,
    Space,
    Subspace,
    Variant,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.occurrence import ProjectionAssessment, VariantView
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import (
    BoundaryDestination,
    CoordinateMapping,
    OperandAssociation,
    RegionStateDestination,
    SourceAssociation,
    StreamDestination,
)
from finn.dataflow.ops.base import (
    BOOL_CODEC,
    INT_CODEC,
    AttributeCodec,
    DataflowOp,
    DataflowOpError,
    DecisionAttribute,
    SelectorAttribute,
    unresolved_reason,
)
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.supplied_dot_product import (
    SuppliedDotProductDesign,
    WeightSupply,
)
from finn.dataflow.ops.schema import Attribute, BuildFact, InputTensor, OutputTensor


def _decode_supply(value: object) -> WeightSupply:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return WeightSupply(text)


#: The weight-supply mode crosses the node boundary as its own stable string.
#: Never the enum's ordinal: reordering the members would silently repoint every
#: saved graph at a different mode.
SUPPLY_CODEC = AttributeCodec(lambda value: WeightSupply(value).value, _decode_supply, kind="s")


def _target_dsp(build: Any) -> DspBlock:
    value = getattr(build, "target_dsp", DspBlock.DSP58)
    return value if isinstance(value, DspBlock) else DspBlock(str(value))


def _design_view(root: Space) -> VariantView:
    return cast(VariantView, root.design)  # type: ignore[attr-defined]


def _selected_design(root: Space) -> WeightedDotProductDesign:
    """The live Design occurrence, or a refusal that says the choice is open."""

    view = _design_view(root)
    chosen = view.selected()
    if not isinstance(chosen, Decided):
        raise DataflowOpError(
            "the Design alternative is not chosen yet, so nothing beneath it can be "
            f"named ({unresolved_reason(chosen)})"
        )
    return cast(WeightedDotProductDesign, view.alternative(chosen.value))


def _compute_segment(root: Space) -> VariantView:
    return cast(VariantView, _selected_design(root).compute)  # type: ignore[attr-defined]


def _compute_kernel(root: Space) -> Space:
    design = _selected_design(root)
    kernel = design.kernel("compute")
    if not isinstance(kernel, Decided):
        raise DataflowOpError(
            f"the compute candidate is not chosen yet ({unresolved_reason(kernel)})"
        )
    return cast(Space, kernel.value)


class MvauDataflowOp(DataflowOp):
    """One matrix-vector node, projected onto the unified Space stack.

    **Ownership.**  Each restriction below is written where something can argue
    for it, rather than copied down from an earlier implementation:

    ``weight`` is rank 2
        Matrix-vector multiplication is defined on a matrix.  An operation
        constraint, because it is the operation's own definition.

    the output shape follows the activation and the matrix
        Likewise the operation's own contract, and the thing
        ``make_shape_compatible_op`` and the graph effects both derive from.

    the activation may be constant
        Deliberately *not* restricted.  A constant activation is
        mathematically valid, and the previous ``NoInitializer`` was inherited
        rather than argued for.

    rank
        A rank-1 activation is a valid *problem*; what it has no applicable
        composition for is a Design.  The rejection therefore lives on
        ``WeightedDotProductDesign``, so such a node yields a valid occurrence
        with no applicable Design rather than a refused source reading.
    """

    family: ClassVar[str] = "finn.dataflow.mvau"
    family_version: ClassVar[str] = "1"

    # -- the source schema ----------------------------------------------------

    activation = InputTensor(index=0)
    weight = InputTensor(index=1, fingerprint=True)
    output = OutputTensor(index=0)

    narrow_weights = Attribute(bool, default=False)

    target_dsp = BuildFact(DspBlock, accessor=_target_dsp)
    clock_period_ns = BuildFact(float, accessor=lambda build: float(build.synth_clk_period_ns))

    # -- what the composition below reads -------------------------------------

    @derived(int, shape=weight.shape)
    def matrix_width(*, shape: tuple[int, ...]) -> object:
        return shape[0]

    @derived(int, shape=weight.shape)
    def matrix_height(*, shape: tuple[int, ...]) -> object:
        return shape[1]

    @derived(int, shape=activation.shape, width=matrix_width)
    def repetitions(*, shape: tuple[int, ...], width: int) -> object:
        total = 1
        for extent in shape:
            total *= extent
        return total // width

    @constraint(shape=weight.shape)
    def weight_is_a_matrix(*, shape: tuple[int, ...]) -> object:
        if len(shape) == 2:
            return True
        return reject(
            "mvau-weight-not-a-matrix",
            f"matrix-vector multiplication needs a rank-2 matrix; this one is {shape}",
            values={"shape": list(shape)},
        )

    @constraint(activation=activation.shape, width=matrix_width)
    def activation_matches_the_matrix(*, activation: tuple[int, ...], width: int) -> object:
        if activation and activation[-1] == width:
            return True
        return reject(
            "mvau-activation-width-mismatch",
            f"the activation's last dimension {activation[-1:]} does not match the "
            f"matrix width {width}",
            values={"activation": list(activation), "width": width},
        )

    @constraint(observed=output.shape, activation=activation.shape, height=matrix_height)
    def output_shape_is_consistent(
        *, observed: tuple[int, ...], activation: tuple[int, ...], height: int
    ) -> object:
        expected = (*activation[:-1], height)
        if tuple(observed) == expected:
            return True
        return reject(
            "mvau-output-shape-mismatch",
            f"the graph annotates the output as {tuple(observed)}; this operation "
            f"produces {expected}",
            values={"observed": list(observed), "expected": list(expected)},
        )

    source_accepts = ConstraintGroup(
        weight_is_a_matrix, activation_matches_the_matrix, output_shape_is_consistent
    )

    # -- the composition ------------------------------------------------------

    design = Variant(
        {
            "dot_product": Subspace(
                DotProductDesign,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation.datatype,
                weight_type=weight.datatype,
                # The accumulator is the output's own type: this core drives it
                # straight out, and inventing a wider one here would be the
                # design space deciding a numeric fact the graph already states.
                accumulator_type=output.datatype,
                output_type=output.datatype,
                narrow_weights=narrow_weights,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
            ),
            "supplied": Subspace(
                SuppliedDotProductDesign,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation.datatype,
                weight_type=weight.datatype,
                accumulator_type=output.datatype,
                output_type=output.datatype,
                narrow_weights=narrow_weights,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
                initializer_present=weight.initializer_present,
            ),
        },
    )

    #: In application order.  Two selectors precede every Decision, because
    #: which alternative is live decides which Decisions exist to be assigned.
    attributes = (
        SelectorAttribute("dataflow_design", _design_view),
        SelectorAttribute("dataflow_compute", _compute_segment),
        DecisionAttribute("PE", _selected_design, WeightedDotProductDesign.pe, INT_CODEC),
        DecisionAttribute("SIMD", _selected_design, WeightedDotProductDesign.simd, INT_CODEC),
        DecisionAttribute(
            "weight_supply",
            _selected_design,
            SuppliedDotProductDesign.weight_supply,
            SUPPLY_CODEC,
        ),
        DecisionAttribute(
            "pumpedCompute", _compute_kernel, DotpAxiKernel.compute_pumping, BOOL_CODEC
        ),
    )

    # -- the projections ------------------------------------------------------

    @property
    def dataflow(self) -> ProjectionAssessment[DataflowNetwork]:
        return _selected_design(self).dataflow

    @property
    def association(self) -> Answer[SourceAssociation]:
        """Where each tensor crosses, given whichever Network was selected.

        Read off the resolved Network rather than off the Design's
        declarations, so that a mode which produces the matrix internally
        reports what actually happens to it instead of the boundary it would
        have had.
        """

        answer = self.network
        if not isinstance(answer, Decided):
            return cast("Answer[SourceAssociation]", answer)
        network = answer.value
        boundaries = {item.id: item for item in network.boundaries}

        operands: list[OperandAssociation] = []
        for operand_id, boundary_id, correspondence in (
            ("activation", "activation", CoordinateMapping.FLATTEN_LEADING),
            ("weight", "weight", CoordinateMapping.TRANSPOSE_2D),
            ("output", "output", CoordinateMapping.FLATTEN_LEADING),
        ):
            operand = self.source.operand(operand_id)
            boundary = boundaries.get(boundary_id)
            destination = (
                BoundaryDestination(
                    boundary_id, boundary.endpoint.node_id, boundary.endpoint.port_id
                )
                if boundary is not None
                # No boundary is a fact about this Network, not a gap: an
                # embedded or decoupled matrix never crosses the Design's edge.
                else _internal_destination(network, operand_id)
            )
            operands.append(
                OperandAssociation(
                    operand_id,
                    operand.tensor,
                    destination,
                    correspondence,
                    operand.shape,
                    _selected_shape(network, destination),
                )
            )
        binding = self.binding
        return Decided(
            SourceAssociation(
                binding.node_identity,
                self.source.node_name,
                type(self).family,
                type(self).family_version,
                tuple(operands),
            )
        )

    # -- what this operation is authoritative for -----------------------------

    def graph_shapes(self) -> dict[str, tuple[int, ...]]:
        activation = self.source.operand("activation")
        weight = self.source.operand("weight")
        output = self.source.operand("output")
        return {output.tensor: (*activation.shape[:-1], weight.shape[1])}


def _internal_destination(
    network: DataflowNetwork, operand_id: str
) -> StreamDestination | RegionStateDestination:
    """Where an operand that never crosses a boundary actually goes.

    Two genuinely different answers, and the record says which.  A decoupled
    matrix is *traffic*: it reaches a real port on a real node, reached without
    crossing the Design's edge.  An embedded matrix is *state*: it is baked into
    the node and there is no port at all.  The previous version reported the
    second case as a port named ``"embedded"``, which is a port that does not
    exist -- a consumer looking it up finds nothing, and the empty shape that
    came with it reads as a zero-element tensor.
    """

    for node in network.nodes:
        for item in node.region.inputs:
            if item.port.id == operand_id:
                return StreamDestination(node.id, item.port.id)
    for node in network.nodes:
        if node.id == "compute":
            return RegionStateDestination(node.id, operand_id)
    raise DataflowOpError(f"the selected Network has nowhere for {operand_id!r} to go")


def _selected_shape(
    network: DataflowNetwork,
    destination: BoundaryDestination | StreamDestination | RegionStateDestination,
) -> tuple[int, ...] | None:
    """The shape at the destination port, or ``None`` when there is no port.

    ``None`` rather than ``()``: an empty tuple is the shape of a scalar, and a
    consumer that compared it against the operand's own shape would report a
    mismatch instead of "this operand has no port to have a shape at".
    """

    if isinstance(destination, RegionStateDestination):
        return None
    node = network.node(destination.node_id)
    for interface in node.region.inputs:
        if interface.port.id == destination.port_id:
            return tuple(interface.port.operand.shape)
    for output in node.region.outputs:
        if output.port.id == destination.port_id:
            return tuple(output.port.operand.shape)
    return None


__all__ = ["MvauDataflowOp"]
