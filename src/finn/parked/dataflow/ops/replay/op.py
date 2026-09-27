# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The non-MVAU forcing operation.

Written second and deliberately unlike the first.  It has one input operand and
no optional one, no Kernel alternatives and therefore no selector attribute, a
one-node Network, and an association with two entries rather than three.  Where
MVAU reconciles two operands against a matrix shape, this one reads a single
tensor -- and the layer between them does not change.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast
from finn.kernels._engine import Decided

from finn.kernels.space.declarations import (
    ConstraintGroup,
    Subspace,
    constraint,
    derived,
    reject,
)
from finn.parked.dataflow.ops.mapping import CoordinateMapping
from finn.parked.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.parked.dataflow.ops.space import DataflowSpace
from finn.parked.dataflow.ops.binding import ChoiceBinding, ImplementationBinding, OperandBinding
from finn.parked.dataflow.kernels.replay import ActivationReplayKernel
from finn.parked.dataflow.ops.schema import Attribute, OpInput, OpOutput


class ReplaySpace(DataflowSpace):
    """Repeat each activation row once per neuron fold."""

    family: ClassVar[str] = "finn.dataflow.activation_replay"
    family_version: ClassVar[str] = "2"
    schema_version: ClassVar[int] = 7
    retired_choice_keys: ClassVar[frozenset[str]] = frozenset({"kernel__pe"})

    implementation_binding = ImplementationBinding(("kernel",))
    operand_bindings = (
        OperandBinding("activation", "activation", 0, adapter=CoordinateMapping.FLATTEN_LEADING),
        OperandBinding("expanded", "result", 0, output=True, adapter=CoordinateMapping.IDENTITY),
    )
    choice_bindings = (ChoiceBinding("kernel__simd", ("kernel",), "simd"),)

    activation = OpInput(index=0, operand="X", correspondence=CoordinateMapping.FLATTEN_LEADING)
    expanded = OpOutput(index=0, operand="XR", correspondence=CoordinateMapping.IDENTITY)

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
        if any(extent <= 0 for extent in shape):
            return reject(
                "replay-degenerate-extent",
                f"an activation needs positive extents in every dimension; got {shape}",
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

    kernel = Subspace(
        ActivationReplayKernel,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation.datatype,
    )

    # QONNX verification belongs to the adapter and consumes this initialized
    # Space's source assessment; the family needs no separate verification hook.

    def expected_outputs(self) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        use = self
        datatype, domain = use.operand_type("result"), use.operand_domain("result")
        return {
            "expanded": (
                domain.value.extents if isinstance(domain, Decided) else None,
                datatype.value if isinstance(datatype, Decided) else None,
            )
        }


class ActivationReplayOp(DataflowOp):
    space_type = ReplaySpace

    def execute_node(self, context: Any, graph: Any) -> None:
        """Repeat each activation row once per neuron fold.

        Consecutively, not tiled: fold *f* of row *r* is what the buffer
        replays before moving to row *r+1*, and a consumer that folded the
        output back would otherwise reassemble the rows in the wrong order.
        """

        del graph
        import numpy  # type: ignore[import-not-found]  # noqa: PLC0415 - heavy import

        source = self.space.source
        node = self.space.node_snapshot()
        activation = numpy.asarray(context[node.input[0]])
        folds = int(cast(int, source.attributes["neuron_folds"]))
        expected = self.space.expected_outputs().get("expanded", (None, None))[0]
        if expected is None:
            raise DataflowOpError(f"{node.name} cannot state the shape of its own output")
        rows = activation.reshape(-1, activation.shape[-1])
        context[node.output[0]] = numpy.repeat(rows, folds, axis=0).reshape(expected)


__all__ = ["ReplaySpace", "ActivationReplayOp"]
