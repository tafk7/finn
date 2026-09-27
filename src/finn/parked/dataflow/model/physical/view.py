# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical capability view."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from finn.kernels._engine import DesignPoint, Engine
from finn.kernels.space.compiler import _CompiledSpace
from finn.kernels.space.declarations import ConstraintGroup, Projection, Readiness, ValueSource
from finn.kernels.space.occurrence import ProjectionAssessment, evaluate_projection

T_co = TypeVar("T_co", covariant=True)


class PhysicalView(Projection[T_co]):
    """Codegen preconditions and detached requirements at the requested scope.

    Acceptance establishes declared generator, target and interface conditions.
    It does not imply logical graph acceptance, synthesis, timing or RTL execution.
    """

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "physical",
    ) -> None:
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


def kernel_physical(
    engine: Engine, compiled: _CompiledSpace[Any], point: DesignPoint
) -> ProjectionAssessment[object]:
    return evaluate_projection(engine, point, compiled.projection("physical"))


__all__ = ["PhysicalView", "kernel_physical"]
