# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The non-MVAU forcing operation.

Written second and deliberately unlike the first.  It has one input operand and
no optional one, no Design alternatives and therefore no selector attribute, a
one-node Network, and an association with two entries rather than three.  Where
MVAU reconciles two operands against a matrix shape, this one reads a single
tensor -- and the layer between them does not change.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.model.declarations import (
    ConstraintGroup,
    Space,
    Subspace,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.occurrence import ProjectionAssessment
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import (
    BoundaryDestination,
    CoordinateMapping,
    OperandAssociation,
    SourceAssociation,
)
from finn.dataflow.ops.base import (
    INT_CODEC,
    DataflowOp,
    DataflowOpError,
    DecisionAttribute,
)
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.schema import Attribute, InputTensor, OutputTensor


def _design(root: Space) -> ActivationReplayDesign:
    return cast(ActivationReplayDesign, root.design)  # type: ignore[attr-defined]


class ActivationReplayOp(DataflowOp):
    """Repeat each activation row once per neuron fold."""

    family: ClassVar[str] = "finn.dataflow.activation_replay"
    family_version: ClassVar[str] = "1"

    activation = InputTensor(index=0)
    expanded = OutputTensor(index=0)

    neuron_folds = Attribute(int, default=1)

    @derived(tuple, shape=activation.shape)
    def matrix(*, shape: tuple[int, ...]) -> object:
        """The validated activation shape every other derivation reads.

        A ``Derived`` cannot assume a sibling constraint ran first, so the rank
        requirement lives here rather than leaving ``shape[-1]`` to raise on an
        empty shape.
        """

        if len(shape) < 2:
            return reject(
                "replay-activation-rank",
                f"a replay buffer repeats rows of a matrix-shaped activation; got {shape}",
                values={"shape": list(shape)},
            )
        return shape

    @derived(int, shape=matrix)
    def matrix_width(*, shape: tuple[int, ...]) -> object:
        return shape[-1]

    @derived(int, shape=matrix, width=matrix_width)
    def repetitions(*, shape: tuple[int, ...], width: int) -> object:
        total = 1
        for extent in shape:
            total *= extent
        return total // width

    #: The buffer's fold is stated as a height over one PE lane, which is the
    #: vocabulary its Region already speaks.
    @derived(int, folds=neuron_folds)
    def matrix_height(*, folds: int) -> object:
        return folds

    @constraint(folds=neuron_folds)
    def at_least_one_fold(*, folds: int) -> object:
        if folds >= 1:
            return True
        return reject(
            "replay-no-folds",
            f"a replay buffer repeats each row at least once; this one asks for {folds}",
            values={"neuron_folds": folds},
        )

    @constraint(shape=matrix)
    def activation_is_a_matrix(*, shape: tuple[int, ...]) -> object:
        """Restates the refusal ``matrix`` made, as a verdict on the projection."""

        del shape
        return True

    #: The output annotation is a reconciliation difference, not a rejection:
    #: see ``expected_outputs``.
    source_accepts = ConstraintGroup(at_least_one_fold, activation_is_a_matrix)

    design = Subspace(
        ActivationReplayDesign,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation.datatype,
    )

    attributes = (
        DecisionAttribute("PE", _design, ActivationReplayDesign.pe, INT_CODEC),
        DecisionAttribute("SIMD", _design, ActivationReplayDesign.simd, INT_CODEC),
    )

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        """A fixed Subspace, so the child is always selected."""

        return _design(self).dataflow

    @property
    def association(self) -> Answer[SourceAssociation]:
        answer = self.network
        if not isinstance(answer, Decided):
            return cast("Answer[SourceAssociation]", answer)
        network = answer.value
        boundaries = {item.id: item for item in network.boundaries}
        operands: list[OperandAssociation] = []
        for operand_id in ("activation", "expanded"):
            operand = self.source.operand(operand_id)
            boundary = boundaries[operand_id]
            destination = BoundaryDestination(
                operand_id, boundary.endpoint.node_id, boundary.endpoint.port_id
            )
            node = network.node(destination.node_id)
            interfaces = node.region.inputs if operand_id == "activation" else node.region.outputs
            selected = next(
                (
                    tuple(item.port.operand.shape)
                    for item in interfaces
                    if item.port.id == destination.port_id
                ),
                None,
            )
            operands.append(
                OperandAssociation(
                    operand_id,
                    operand.tensor,
                    destination,
                    CoordinateMapping.FLATTEN_LEADING,
                    operand.shape,
                    selected,
                )
            )
        return Decided(
            SourceAssociation(
                self.binding.node_identity,
                self.source.node_name,
                type(self).family,
                type(self).family_version,
                tuple(operands),
            )
        )

    def expected_outputs(self) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        activation = self.source.operand("activation")
        if len(activation.shape) < 2:
            return {}
        leading = 1
        for extent in activation.shape[:-1]:
            leading *= extent
        folds = int(cast(int, self.source.attributes["neuron_folds"]))
        return {"expanded": ((leading * folds, activation.shape[-1]), activation.datatype)}

    def make_shape_compatible_op(self, model: Any) -> Any:
        del model
        from onnx import helper  # type: ignore[import-not-found] # noqa: PLC0415

        expected = self.expected_outputs().get("expanded")
        if expected is None or expected[0] is None:
            raise DataflowOpError(
                f"{self.source.node_name} cannot state a shape-compatible op: its activation "
                "is not matrix-shaped"
            )
        return helper.make_node(
            "RandomNormal", [], [self.onnx_node.output[0]], shape=list(expected[0])
        )

    def infer_node_datatype(self, model: Any) -> None:
        expected = self.expected_outputs().get("expanded")
        if expected is None or expected[1] is None:
            return
        model.set_tensor_datatype(self.onnx_node.output[0], expected[1])


__all__ = ["ActivationReplayOp"]
