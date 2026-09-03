# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The non-MVAU forcing operation.

Written second and deliberately unlike the first.  It has one input operand and
no optional one, no Design alternatives and therefore no selector attribute, a
one-node Network, and an association with two entries rather than three.  Where
MVAU's ``source_facts`` has to reconcile two operands against a matrix shape,
this one reads a single tensor -- and the layer between them does not change.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.model.declarations import Problem, Space, Subspace
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
    INT_CODEC,
    DataflowOp,
    DecisionAttribute,
)
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.source import SourceError, SourceNode, read_source_node


class ActivationReplaySource(Space):
    """One activation tensor, the fold it is replayed across, and its Design.

    A fixed ``Subspace``, not a ``Variant``.  There is one way to do this, and
    declaring a one-alternative choice to look like MVAU would add a selector
    that no caller could usefully set.
    """

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)

    design = Subspace(
        ActivationReplayDesign,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation_type,
    )


def _design(root: Space) -> ActivationReplayDesign:
    return cast(ActivationReplayDesign, root.design)  # type: ignore[attr-defined]


class ActivationReplayOp(DataflowOp):
    """Repeat each activation row once per neuron fold."""

    family: ClassVar[str] = "finn.dataflow.activation_replay"
    family_version: ClassVar[str] = "1"
    source_space: ClassVar[type[Space]] = ActivationReplaySource

    attributes = (
        DecisionAttribute("PE", _design, ActivationReplayDesign.pe, INT_CODEC),
        DecisionAttribute("SIMD", _design, ActivationReplayDesign.simd, INT_CODEC),
    )

    def source_nodeattr_types(self) -> dict[str, tuple[str, bool, object]]:
        return {"neuron_folds": ("i", True, 1)}

    def read_source(self) -> SourceNode:
        return read_source_node(
            self._attached(),
            self.onnx_node,
            inputs=("activation",),
            outputs=("expanded",),
            attributes={"neuron_folds": int(self.get_nodeattr("neuron_folds"))},
        )

    def source_facts(self, source: SourceNode, build: Any) -> dict[Problem[Any], object]:
        del build
        activation = source.operand("activation")
        if len(activation.shape) < 2:
            raise SourceError(
                f"{source.node_name} replays a matrix-shaped activation; got {activation.shape}"
            )
        matrix_width = activation.shape[-1]
        neuron_folds = int(cast(int, source.attributes["neuron_folds"]))
        if neuron_folds < 1:
            raise SourceError(f"{source.node_name} needs at least one neuron fold")
        return {
            ActivationReplaySource.repetitions: activation.elements // matrix_width,
            ActivationReplaySource.matrix_width: matrix_width,
            # The buffer's fold is stated as a height over one PE lane, which is
            # the vocabulary its Region already speaks.
            ActivationReplaySource.matrix_height: neuron_folds,
            ActivationReplaySource.activation_type: activation.datatype,
        }

    def network(self, build: Any) -> Answer[DataflowNetwork]:
        return _design(self.occurrence(build)).dataflow.accepted_answer

    def association(self, build: Any) -> Answer[SourceAssociation]:
        source = self.read_source()
        answer = self.network(build)
        if not isinstance(answer, Decided):
            return cast("Answer[SourceAssociation]", answer)
        boundaries = {item.id: item for item in answer.value.boundaries}
        operands = tuple(
            OperandAssociation(
                operand_id,
                source.operand(operand_id).tensor,
                operand_id,
                boundaries[operand_id].endpoint.node_id,
                boundaries[operand_id].endpoint.port_id,
                CoordinateMapping.FLATTEN_LEADING,
                source.operand(operand_id).shape,
                tuple(
                    answer.value.node(boundaries[operand_id].endpoint.node_id)
                    .region.input_interface(boundaries[operand_id].endpoint.port_id)
                    .port.operand.shape
                    if operand_id == "activation"
                    else answer.value.node(boundaries[operand_id].endpoint.node_id)
                    .region.output_interface(boundaries[operand_id].endpoint.port_id)
                    .port.operand.shape
                ),
            )
            for operand_id in ("activation", "expanded")
        )
        return Decided(
            SourceAssociation(
                self.scope_id(),
                source.node_name,
                type(self).family,
                type(self).family_version,
                operands,
            )
        )


__all__ = ["ActivationReplayOp", "ActivationReplaySource"]
