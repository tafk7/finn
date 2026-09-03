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

from finn.dataflow._engine import ABSENT, Answer, Decided
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.model.declarations import (
    ConstraintGroup,
    Space,
    Subspace,
    Variant,
    allow_absent,
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
from finn.dataflow.ops.base import DataflowOp, DataflowOpError, unresolved_reason
from finn.dataflow.ops.mvau.computation import (
    MvauComputationProfile,
    computation_profile,
    execute_mvau,
    initializer_excludes_minimum,
)
from finn.dataflow.ops.source import SourceNode, SourceOperand
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.supplied_dot_product import (
    SuppliedDotProductDesign,
)
from finn.dataflow.ops.schema import (
    Attribute,
    BuildFact,
    DatatypeAttribute,
    InitializerAnalysis,
    InputTensor,
    OutputTensor,
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

    a rank-1 activation is one repetition
        Accepted, and deliberately not restricted anywhere.  A shape ``(W,)``
        whose extent matches the matrix width is a single vector through the
        matrix -- ``repetitions`` evaluates to 1 by the same formula every
        other rank uses, and the Designs need nothing special to build it.

        This paragraph previously claimed the opposite: that a rank-1
        activation had no applicable Design and that
        ``WeightedDotProductDesign`` rejected it.  No such rejection existed,
        and none was added to make the sentence true -- a restriction has to be
        argued from the mathematics or from a Design's structure, and neither
        argues for one here.
    """

    family: ClassVar[str] = "finn.dataflow.mvau"
    family_version: ClassVar[str] = "1"

    # -- the source schema ----------------------------------------------------

    activation = InputTensor(index=0)
    weight = InputTensor(index=1, fingerprint=True)
    #: Present exactly when the node fuses an activation.  Optional rather than
    #: conditional-by-declaration: presence is *emergent* -- it is read from the
    #: graph -- and the agreement between it and ``no_activation`` is a
    #: constraint that can be reported, not a schema rule that makes the node
    #: unreadable.
    threshold = InputTensor(index=2, optional=True, fingerprint=True)
    output = OutputTensor(index=0)

    #: The three attributes FINN's graphs already carry, under the names they
    #: already have.  ``no_activation`` defaults to ``True`` because a plain
    #: matrix-vector node is the common case and because that is what every
    #: existing graph means by omitting it.
    no_activation = Attribute(bool, default=True, onnx="noActivation")
    binary_xnor = Attribute(bool, default=False, onnx="binaryXnorMode")
    activation_bias = Attribute(int, default=0, onnx="ActVal")

    #: The accumulator width, and therefore the output's datatype when nothing
    #: is fused.  A source-semantic *attribute*, not a reading of the output
    #: annotation: the value reaches Region construction and the physical
    #: realization, so it must be part of the problem's identity.  Reading it
    #: back off the tensor the operation itself writes would make a fact the
    #: design space depends on something the design space is authoritative for
    #: -- and, kept out of the fingerprint to make repair possible, would let
    #: recorded choices be silently wrong.
    accumulator_type = DatatypeAttribute(default="INT32", onnx="accDataType")

    #: Narrowness is *derived from the weights*, never asserted about them.  An
    #: attribute saying "these weights are narrow" is a claim a caller can get
    #: wrong and nothing can check; whether the matrix uses the most negative
    #: value its datatype allows is a property of the matrix, read once at
    #: binding time and reduced to one boolean before anything enters the point.
    weight_excludes_minimum = InitializerAnalysis(
        weight, bool, evaluate=initializer_excludes_minimum
    )

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
        if any(extent <= 0 for extent in shape):
            # Checked here for the same reason the rank is: everything
            # downstream divides by the width, and a zero would raise before
            # any constraint got to say what was wrong.
            return reject(
                "mvau-degenerate-extent",
                f"a matrix needs positive extents; this one is {shape}",
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
        if not shape or any(extent <= 0 for extent in shape):
            return reject(
                "mvau-degenerate-extent",
                f"an activation needs positive extents in every dimension; got {shape}",
                values={"shape": list(shape)},
            )
        total = 1
        for extent in shape:
            total *= extent
        return total // width

    @derived(MvauComputationProfile, activated=no_activation, xnor=binary_xnor)
    def profile(*, activated: bool, xnor: bool) -> object:
        """Which of the three MVAU computations this node describes.

        Derived once and read by everything -- the execution, the output
        datatype, and the Designs' applicability -- so a consumer that asked
        ``noActivation`` directly could not come to disagree with it.
        """

        return computation_profile(no_activation=activated, binary_xnor=xnor)

    @derived(bool, excludes_minimum=allow_absent(weight_excludes_minimum))
    def effective_narrow_weights(*, excludes_minimum: object) -> object:
        """Whether the weights can be stored one bit narrower.

        Absent -- an operand with no initializer, or values the analysis could
        not judge -- is ``False``: a supplied matrix whose contents are not
        known at build time cannot be promised to avoid its minimum, and
        assuming otherwise would build hardware that cannot represent a weight
        the graph is entitled to deliver later.
        """

        return excludes_minimum is not ABSENT and bool(excludes_minimum)

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
    @constraint(present=threshold.present, activated=no_activation)
    def threshold_present_iff_activated(*, present: bool, activated: bool) -> object:
        """The operand list and the attribute must agree about what this node is.

        Either disagreement is a real fault and neither is repairable here: a
        node that says it fuses an activation and supplies no thresholds cannot
        be executed, and one that supplies thresholds while claiming it does
        not fuse would silently ignore them.  The check is on the operation
        because it is the operation's own mathematics -- no Design has an
        opinion about it.
        """

        if present is not activated:
            return True
        return reject(
            "mvau-threshold-presence-mismatch",
            "a fused-activation MVAU needs a threshold operand and a plain one must not "
            f"have it; noActivation={activated} with threshold present={present}",
            values={"no_activation": activated, "threshold_present": present},
        )

    @constraint(operand=allow_absent(threshold), height=matrix_height)
    def threshold_shape_supported(*, operand: object, height: int) -> object:
        """One threshold row per output channel, when there are thresholds at all.

        Absence-tolerant rather than conditional: a plain MVAU has nothing to
        check here, and that is an ordinary "yes" rather than a constraint that
        does not exist.
        """

        if operand is ABSENT:
            return True
        shape = tuple(cast(SourceOperand, operand).shape)
        if len(shape) == 2 and shape[0] == height:
            return True
        return reject(
            "mvau-threshold-shape",
            f"a threshold operand is one row per output channel: expected {height} rows in a "
            f"rank-2 tensor, got {shape}",
            values={"shape": list(shape), "matrix_height": height},
        )

    source_accepts = ConstraintGroup(
        weight_is_a_matrix,
        activation_matches_the_matrix,
        threshold_present_iff_activated,
        threshold_shape_supported,
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
                accumulator_type=accumulator_type,
                # This core drives the accumulator straight out, so the output
                # type *is* the accumulator type.  Derived onto the tensor, not
                # read back off it.
                output_type=accumulator_type,
                narrow_weights=effective_narrow_weights,
                computation_profile=profile,
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
                narrow_weights=effective_narrow_weights,
                computation_profile=profile,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
                initializer_present=weight.initializer_present,
            ),
        },
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

    def expected_for(self, source: SourceNode) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        """The output contract, from the reading alone.

        Takes the reading rather than ``self`` so QONNX's shape and datatype
        passes -- which run on an unbound wrapper and have no synthesis
        configuration -- reach the same formula the graph effects do.
        """

        activation = source.operand("activation")
        weight = source.operand("weight")
        if len(weight.shape) != 2 or not activation.shape:
            return {}
        fused = not bool(source.attributes["no_activation"])
        return {
            "output": (
                (*activation.shape[:-1], weight.shape[1]),
                # A fused threshold's output type is chosen by whoever wrote
                # the thresholds, and this operation is not authoritative for
                # it.  ``None`` says exactly that: the graph's annotation
                # stands, and nothing here repairs or contradicts it.
                None if fused else cast(Any, source.attributes["accumulator_type"]),
            )
        }

    # -- executing the source semantics ----------------------------------------

    def execute_node(self, context: Any, graph: Any) -> None:
        """Compute this node in ONNX, for the three profiles it can describe."""

        del graph
        source = self.attached_source()
        node = self.onnx_node
        thresholds = (
            context[node.input[2]] if source.has("threshold") and len(node.input) > 2 else None
        )
        result = execute_mvau(
            activation=context[node.input[0]],
            weight=context[node.input[1]],
            thresholds=thresholds,
            profile=computation_profile(
                no_activation=bool(source.attributes["no_activation"]),
                binary_xnor=bool(source.attributes["binary_xnor"]),
            ),
            activation_type=source.operand("activation").datatype,
            weight_type=source.operand("weight").datatype,
            output_type=source.operand("output").datatype,
            activation_bias=int(cast(int, source.attributes["activation_bias"])),
        )
        expected = self.expected_for(source)["output"][0]
        if expected is None:
            raise DataflowOpError(f"{node.name} cannot state the shape of its own output")
        context[node.output[0]] = result.reshape(expected)

    def verify_node(self) -> list[str]:
        """Every way this node's own semantics are inconsistent, as messages.

        The same constraints the projection uses, reported the way QONNX's
        verification expects.  Written as a reading of ``source_accepts``
        rather than as a second set of checks, so a node cannot pass
        verification and then be refused by the design space for a reason
        verification never mentioned.
        """

        assessment = self.assess(type(self).source_accepts)
        if assessment.verdict is True:
            return []
        return [
            finding.message
            for answer in assessment.answers.values()
            for finding in getattr(answer, "findings", ())
        ]


def _internal_destination(
    network: DataflowNetwork, operand_id: str
) -> StreamDestination | RegionStateDestination:
    """Where an operand that never crosses a boundary actually goes.

    Two genuinely different answers, and the record says which.  A decoupled
    matrix is *traffic*: it reaches a real port on a real node, reached without
    crossing the Design's edge.  An embedded matrix is *state*: it is baked into
    the node and there is no port at all.
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
