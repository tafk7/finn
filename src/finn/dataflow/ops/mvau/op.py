# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU as one source node over a closed set of Designs.

The operation is thin on purpose.  Everything below it -- the folding, the
Regions, the topology, the weight path -- belongs to the Designs and the
Kernels, and everything above it belongs to the graph.  What lives here is the
part only this operation can say: which two Designs are its alternatives, how a
matrix-vector node's tensors become the facts they read, and where each of
those tensors ends up in whichever Network is selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.declarations import Problem, Space, Subspace, Variant
from finn.dataflow.model.occurrence import VariantView
from finn.dataflow.model.semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import (
    CoordinateMapping,
    OperandAssociation,
    SourceAssociation,
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
from finn.dataflow.ops.source import SourceError, SourceNode, read_source_node


class MvauSource(Space):
    """The graph-side facts of one matrix-vector node, and its Design choice.

    Every member is a ``Problem``: these are read from the model once, frozen,
    and fingerprinted.  A Decision would mean the design space could change one
    of them, and a design space that can change a tensor's shape is not
    describing that tensor.

    The two alternatives are a closed set.  ``dot_product`` is the reference
    composition with the matrix always arriving from outside; ``supplied`` is
    the same composition with the weight path as its own dial.  Both are
    ``WeightedDotProductDesign`` subclasses, which is what lets ``pe`` and
    ``simd`` be one Decision object and therefore one persisted attribute
    whichever alternative is live.
    """

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    narrow_weights = Problem(bool)
    target_dsp = Problem(DspBlock)
    clock_period_ns = Problem(float)
    initializer_present = Problem(bool)

    design = Variant(
        {
            "dot_product": Subspace(
                DotProductDesign,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation_type,
                weight_type=weight_type,
                accumulator_type=accumulator_type,
                output_type=output_type,
                narrow_weights=narrow_weights,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
            ),
            "supplied": Subspace(
                SuppliedDotProductDesign,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation_type,
                weight_type=weight_type,
                accumulator_type=accumulator_type,
                output_type=output_type,
                narrow_weights=narrow_weights,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
                initializer_present=initializer_present,
            ),
        },
    )


def _decode_supply(value: object) -> WeightSupply:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return WeightSupply(text)


#: The weight-supply mode crosses the node boundary as its own stable string.
#: Never the enum's ordinal: reordering the members would silently repoint every
#: saved graph at a different mode.
SUPPLY_CODEC = AttributeCodec(lambda value: WeightSupply(value).value, _decode_supply, kind="s")


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
    """One matrix-vector node, projected onto the unified Space stack."""

    family: ClassVar[str] = "finn.dataflow.mvau"
    family_version: ClassVar[str] = "1"
    source_space: ClassVar[type[Space]] = MvauSource

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

    def source_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        return {"narrow_weights": ("i", False, 0)}

    def read_source(self) -> SourceNode:
        model = self._attached()
        return read_source_node(
            model,
            self.onnx_node,
            inputs=("activation", "weight"),
            outputs=("output",),
            attributes={"narrow_weights": bool(self.get_nodeattr("narrow_weights"))},
        )

    def source_facts(self, source: SourceNode, build: Any) -> Mapping[Problem[Any], object]:
        activation = source.operand("activation")
        weight = source.operand("weight")
        output = source.operand("output")
        if len(activation.shape) < 2 or len(weight.shape) != 2:
            raise SourceError(
                f"{source.node_name} expects a rank-2 matrix and a matrix-shaped activation; "
                f"got {activation.shape} and {weight.shape}"
            )
        matrix_width, matrix_height = weight.shape
        if activation.shape[-1] != matrix_width:
            raise SourceError(
                f"{source.node_name} activation last dimension {activation.shape[-1]} "
                f"does not match matrix width {matrix_width}"
            )
        repetitions = activation.elements // matrix_width
        return {
            MvauSource.repetitions: repetitions,
            MvauSource.matrix_width: matrix_width,
            MvauSource.matrix_height: matrix_height,
            MvauSource.activation_type: activation.datatype,
            MvauSource.weight_type: weight.datatype,
            # The accumulator is the output's own type: this core drives it
            # straight out, and inventing a wider one here would be the design
            # space deciding a numeric fact the graph already states.
            MvauSource.accumulator_type: output.datatype,
            MvauSource.output_type: output.datatype,
            MvauSource.narrow_weights: bool(source.attributes["narrow_weights"]),
            MvauSource.target_dsp: _target_dsp(build),
            MvauSource.clock_period_ns: float(build.synth_clk_period_ns),
            MvauSource.initializer_present: weight.initializer,
        }

    def network(self, build: Any) -> Answer[DataflowNetwork]:
        return _selected_design(self.occurrence(build)).dataflow.accepted_answer

    def association(self, build: Any) -> Answer[SourceAssociation]:
        """Where each tensor crosses, given whichever Network was selected.

        Read off the resolved Network rather than off the Design's declarations,
        so that a mode which produces the matrix internally reports no boundary
        instead of reporting the boundary it would have had.
        """

        source = self.read_source()
        answer = self.network(build)
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
            operand = source.operand(operand_id)
            boundary = boundaries.get(boundary_id)
            if boundary is None:
                # No boundary is a fact about this Network, not a gap: an
                # embedded or decoupled matrix never crosses the Design's edge.
                node_id, port_id = _internal_weight_destination(network)
                operands.append(
                    OperandAssociation(
                        operand_id,
                        operand.tensor,
                        None,
                        node_id,
                        port_id,
                        correspondence,
                        operand.shape,
                        _selected_shape(network, node_id, port_id),
                    )
                )
                continue
            node_id = boundary.endpoint.node_id
            port_id = boundary.endpoint.port_id
            operands.append(
                OperandAssociation(
                    operand_id,
                    operand.tensor,
                    boundary_id,
                    node_id,
                    port_id,
                    correspondence,
                    operand.shape,
                    _selected_shape(network, node_id, port_id),
                )
            )
        return Decided(
            SourceAssociation(
                self.scope_id(),
                source.node_name,
                type(self).family,
                type(self).family_version,
                tuple(operands),
            )
        )


def _internal_weight_destination(network: DataflowNetwork) -> tuple[str, str]:
    """Where a matrix that never crosses a boundary is consumed, or held."""

    for node in network.nodes:
        for item in node.region.inputs:
            if item.port.id == "weight":
                return node.id, item.port.id
    # Embedded: the matrix is state of the compute node rather than traffic.
    for node in network.nodes:
        if node.id == "compute":
            return node.id, "embedded"
    raise DataflowOpError("the selected Network has nowhere for the matrix to go")


def _selected_shape(network: DataflowNetwork, node_id: str, port_id: str) -> tuple[int, ...]:
    node = network.node(node_id)
    for interface in node.region.inputs:
        if interface.port.id == port_id:
            return tuple(interface.port.operand.shape)
    for output in node.region.outputs:
        if output.port.id == port_id:
            return tuple(output.port.operand.shape)
    return ()


def _target_dsp(build: Any) -> DspBlock:
    value = getattr(build, "target_dsp", DspBlock.DSP58)
    return value if isinstance(value, DspBlock) else DspBlock(str(value))


__all__ = ["MvauDataflowOp", "MvauSource"]
