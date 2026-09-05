# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU as one source node over a closed set of Designs.

The operation is thin on purpose.  Everything below it -- the folding, the
Regions, the topology, the weight path -- belongs to the Designs and the
Kernels, and everything above it belongs to the graph.  What lives here is the
part only this operation can say: which Designs are its alternatives, how a
matrix-vector node's tensors become the facts they read, and where each of
those tensors ends up in whichever Network is selected.

The operation *is* the root Space.  Its Problem members are its declared
tensors and attributes, lowered from the source schema, and its ``design``
SubspaceChoice is an ordinary structural choice on the same class.  There is no
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
    SubspaceChoice,
    allow_absent,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.occurrence import ChoiceView, ProjectionAssessment
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
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
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
    TensorDatatypeFact,
)


def _target_dsp(build: Any) -> DspBlock:
    value = getattr(build, "target_dsp", DspBlock.DSP58)
    return value if isinstance(value, DspBlock) else DspBlock(str(value))


def mvau_profile(source: SourceNode) -> MvauComputationProfile:
    """One reading's computation profile, without an occurrence.

    The formula the derived member uses, reachable from the unbound side --
    ``execute_node`` runs on a wrapper QONNX built and has no design space, and
    two spellings of "what does this node compute" is exactly the drift the
    derivation exists to prevent.
    """

    return computation_profile(
        no_activation=bool(source.attributes["no_activation"]),
        binary_xnor=bool(source.attributes["binary_xnor"]),
        activation_type=source.operand("activation").datatype,
        weight_type=source.operand("weight").datatype,
    )


def origin_nodes(source: SourceNode) -> tuple[str, ...]:
    """The original nodes fused into this one, from the comma-separated list.

    The spelling FINN's transformations already write.  Parsed here rather than
    at every reader, and empty entries are dropped so a trailing comma is not a
    node called ``""``.
    """

    raw = str(source.attributes.get("source_nodes", ""))
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _runtime_writable(build: Any) -> bool:
    """Whether this build writes the matrix at runtime.  Absent means no."""

    return bool(getattr(build, "runtime_writable_weights", False))


def _runtime_weight_range_contract(build: Any) -> bool | None:
    """The caller's promise about a runtime matrix's range, if they made one.

    ``None`` is not ``False``: "no contract" and "a contract that says the
    minimum is used" both prevent narrowing, but only one of them is a
    statement, and a later reader that wants to warn about the first must be
    able to tell them apart.
    """

    value = getattr(build, "runtime_weight_range_contract", None)
    return None if value is None else bool(value)


def _design_view(root: Space) -> ChoiceView:
    return cast(ChoiceView, root.design)  # type: ignore[attr-defined]


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


def _compute_segment(root: Space) -> ChoiceView:
    return cast(ChoiceView, _selected_design(root).compute)  # type: ignore[attr-defined]


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

    #: The output annotation, as a *source fact*, for the nodes whose
    #: mathematics reads it.  A thresholded MVAU scales and biases by the type
    #: it was asked to produce, so that annotation changes the numbers -- and a
    #: value that changes the numbers belongs in the problem and its
    #: fingerprint, or a recorded choice can outlive the value it was made
    #: against.  A plain node derives its own output type from ``accDataType``
    #: and *repairs* the annotation, so for it the annotation is an observation
    #: and this fact is absent.  The distinction is the ``when``.
    output_type = TensorDatatypeFact(
        output, when=lambda source: not bool(source.attributes["no_activation"])
    )

    #: Which original graph nodes were fused into this one.  Carried because it
    #: is provenance nothing else records: once several nodes become one
    #: logical MVAU, the lineage exists only here.
    source_nodes = Attribute(str, default="", onnx="dataflow_source_nodes")

    target_dsp = BuildFact(DspBlock, accessor=_target_dsp)
    #: Whether the matrix is written at runtime, and what range the caller
    #: promises it will hold.  Build facts rather than node attributes: both
    #: are decisions of the surrounding build, not properties of the graph.
    runtime_writable_weights = BuildFact(
        bool, accessor=_runtime_writable, default=False, required=False
    )
    runtime_weight_range_contract = BuildFact(
        bool, accessor=_runtime_weight_range_contract, required=False
    )
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

    @derived(
        MvauComputationProfile,
        activated=no_activation,
        xnor=binary_xnor,
        activation_type=activation.datatype,
        weight_type=weight.datatype,
    )
    def profile(
        *, activated: bool, xnor: bool, activation_type: object, weight_type: object
    ) -> object:
        """What this node computes, on both of the axes that decide it.

        Derived once and read by everything -- the execution, the Designs'
        applicability, any later parity record -- so a consumer that asked
        ``noActivation`` directly could not come to disagree with it.  The
        operand datatypes are dependencies because the accumulation genuinely
        depends on them: two BIPOLAR operands are a popcount whatever the
        attributes say.
        """

        return computation_profile(
            no_activation=activated,
            binary_xnor=xnor,
            activation_type=cast(Any, activation_type),
            weight_type=cast(Any, weight_type),
        )

    @derived(
        bool,
        excludes_minimum=allow_absent(weight_excludes_minimum),
        runtime_writable=allow_absent(runtime_writable_weights),
        runtime_range=allow_absent(runtime_weight_range_contract),
    )
    def effective_narrow_weights(
        *, excludes_minimum: object, runtime_writable: object, runtime_range: object
    ) -> object:
        """Whether the weights can be stored one bit narrower.

        Two sources, and which one applies is a property of the *weights*, not
        a preference.  A matrix baked in at build time is judged by its own
        values.  A runtime-writable matrix has no values yet, so the only thing
        that can promise anything about the range it will hold is the caller's
        explicit contract -- and reading the initializer analysis for such a
        node would narrow the hardware on the strength of weights that are
        about to be overwritten.

        Absent, on either path, is ``False``: no initializer, values the
        analysis could not judge, or a runtime-writable matrix with no contract
        all mean nobody has promised anything, and hardware must not be built
        on a promise nobody made.
        """

        writable = runtime_writable is not ABSENT and bool(runtime_writable)
        promised = runtime_range if writable else excludes_minimum
        return promised is not ABSENT and bool(promised)

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

    design = SubspaceChoice(
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
            "batch_interleaved": Subspace(
                BatchInterleavedDesign,
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
                origin_nodes(self.source),
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
        """Compute this node in ONNX, on both of its semantic axes."""

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
            profile=mvau_profile(source),
            # The annotation, which for a thresholded node is the declared
            # source fact ``output_type`` -- the same value, read the same way,
            # and in the problem's fingerprint because it changes the numbers.
            output_type=source.operand("output").datatype,
            activation_bias=int(cast(int, source.attributes["activation_bias"])),
        )
        expected = self.expected_for(source)["output"][0]
        if expected is None:
            raise DataflowOpError(f"{node.name} cannot state the shape of its own output")
        context[node.output[0]] = result.reshape(expected)


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


__all__ = ["MvauDataflowOp", "mvau_profile", "origin_nodes"]
