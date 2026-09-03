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
from finn.dataflow.ops.state import DecisionCodec
from finn.dataflow.ops.schema import (
    Attribute,
    BuildFact,
    DatatypeAttribute,
    InputTensor,
    OutputTensor,
)


def _decode_supply(value: object) -> WeightSupply:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return WeightSupply(text)


#: The weight-supply mode crosses the persistence boundary as its own stable
#: string.  Never the enum's ordinal: reordering the members would silently
#: repoint every saved graph at a different mode.  Versioned, so a future change
#: to the spelling is a refusal rather than a reinterpretation.
SUPPLY_CODEC = DecisionCodec(
    "finn.dataflow.weight_supply", 1, lambda value: WeightSupply(value).value, _decode_supply
)


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

    #: The accumulator width, and therefore the output's datatype.  A
    #: source-semantic *attribute*, not a reading of the output annotation: the
    #: value reaches Region construction and the physical realization, so it
    #: must be part of the problem's identity.  Reading it back off the tensor
    #: the operation itself writes would make a fact the design space depends on
    #: something the design space is authoritative for -- and, kept out of the
    #: fingerprint to make repair possible, would let recorded choices be
    #: silently wrong.
    accumulator_type = DatatypeAttribute(default="INT32")

    target_dsp = BuildFact(DspBlock, accessor=_target_dsp)
    clock_period_ns = BuildFact(float, accessor=lambda build: float(build.synth_clk_period_ns))

    # -- what the composition below reads -------------------------------------

    @derived(tuple, shape=weight.shape)
    def matrix(*, shape: tuple[int, ...]) -> object:
        """The validated matrix shape every other derivation reads.

        The one place the rank is checked, and everything downstream depends on
        *this* rather than on the raw shape.  A ``Derived`` cannot rely on a
        sibling constraint having run first -- the engine offers no such
        ordering -- so ``shape[1]`` on a rank-1 weight would raise an
        EvaluationError before the rank constraint ever produced its finding.
        """

        if len(shape) != 2:
            return reject(
                "mvau-weight-not-a-matrix",
                f"matrix-vector multiplication needs a rank-2 matrix; this one is {shape}",
                values={"shape": list(shape)},
            )
        return shape

    @derived(int, shape=matrix)
    def matrix_width(*, shape: tuple[int, ...]) -> object:
        return shape[0]

    @derived(int, shape=matrix)
    def matrix_height(*, shape: tuple[int, ...]) -> object:
        return shape[1]

    @derived(int, shape=activation.shape, width=matrix_width)
    def repetitions(*, shape: tuple[int, ...], width: int) -> object:
        if not shape:
            return reject(
                "mvau-activation-rank",
                "an activation with no dimensions has nothing to multiply",
                values={"shape": []},
            )
        total = 1
        for extent in shape:
            total *= extent
        return total // width

    @constraint(shape=matrix)
    def weight_is_a_matrix(*, shape: tuple[int, ...]) -> object:
        """Restates the refusal ``matrix`` already made, as a *verdict*.

        The derivation refuses so nothing downstream computes on a malformed
        shape; the constraint exists so ``op.dataflow`` reports it as a
        rejection rather than only as an unresolved Network.
        """

        del shape
        return True

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

    #: The output annotation is deliberately *not* here.  A stale annotation is
    #: what ``graph_effects`` repairs, and a constraint that refused it would
    #: refuse the very projection whose acceptance is required to commit the
    #: repair -- so the node could never be fixed.  It is a reconciliation
    #: difference; see ``expected_outputs``.
    source_accepts = ConstraintGroup(weight_is_a_matrix, activation_matches_the_matrix)

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
                accumulator_type=accumulator_type,
                # This core drives the accumulator straight out, so the output
                # type *is* the accumulator type.  Derived onto the tensor, not
                # read back off it.
                output_type=accumulator_type,
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
                accumulator_type=accumulator_type,
                output_type=accumulator_type,
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

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        view = _design_view(self)
        chosen = view.selected()
        if not isinstance(chosen, Decided):
            return None
        return cast(WeightedDotProductDesign, view.alternative(chosen.value)).dataflow

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

    def expected_outputs(self) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        """What this operation produces, derived once and used everywhere.

        The graph effects, the QONNX compatibility methods and the
        reconciliation report all read this, so the formula has one home.
        """

        activation = self.source.operand("activation")
        weight = self.source.operand("weight")
        if len(weight.shape) != 2 or not activation.shape:
            return {}
        accumulator = self.answer(type(self).accumulator_type)
        return {
            "output": (
                (*activation.shape[:-1], weight.shape[1]),
                accumulator.value if isinstance(accumulator, Decided) else None,
            )
        }

    def make_shape_compatible_op(self, model: Any) -> Any:
        """A shape-compatible stand-in, from the same derivation, applying nothing."""

        del model
        from onnx import helper  # type: ignore[import-not-found] # noqa: PLC0415

        expected = self.expected_outputs().get("output")
        if expected is None or expected[0] is None:
            raise DataflowOpError(
                f"{self.source.node_name} cannot state a shape-compatible op: its operands "
                "are not a matrix and a compatible activation"
            )
        return helper.make_node(
            "RandomNormal",
            [],
            [self.onnx_node.output[0]],
            shape=list(expected[0]),
        )

    def infer_node_datatype(self, model: Any) -> None:
        """QONNX's in-place API, taking its value from the same derivation."""

        expected = self.expected_outputs().get("output")
        if expected is None or expected[1] is None:
            return
        model.set_tensor_datatype(self.onnx_node.output[0], expected[1])


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
