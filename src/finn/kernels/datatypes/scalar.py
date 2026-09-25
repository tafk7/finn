# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar encoding admission independent of pins, streams, and implementation language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from finn.core.space import (
    Constraint,
    ConstraintGroup,
    Param,
    Rejected,
    ScopeBuilder,
    Space,
    Subspace,
    ValueRef,
    View,
    ViewKey,
    default_semantics,
    derived,
    reject,
)
from finn.kernels.datatypes.domains import BitBound, DatatypeDomain
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)

S = TypeVar("S", bound=Space)


def type_constraints(
    builder: ScopeBuilder[S],
    dtype: ValueRef[QONNXDataType],
    policy: DatatypeDomain,
) -> tuple[Constraint, ...]:
    """Localize declared policy dependencies without inspecting the policy's concrete type."""

    def bind(name: str, value: BitBound) -> BitBound:
        if not isinstance(value, ValueRef):
            return value
        parameter = builder.add(name, Param(int))
        builder.bind(parameter, value)
        return parameter

    local = policy.rebind(bind)
    return tuple(
        builder.add(f"dtype_{name}", condition) for name, condition in local.constraints(dtype)
    )


@dataclass(frozen=True, init=False)
class ScalarEncoding:
    """Positive-width QONNX storage encoding, detached by canonical identity."""

    datatype_name: str

    def __init__(self, dtype: QONNXDataType) -> None:
        canonical = canonical_qonnx_datatype(dtype)
        if qonnx_datatype_width(canonical) < 1:
            raise ValueError("a scalar storage encoding must have positive width")
        object.__setattr__(self, "datatype_name", canonical.name)

    @property
    def dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(self.datatype_name)

    @property
    def bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def signed(self) -> bool:
        return self.dtype.signed()


class ScalarScope(Space):
    dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)

    @derived
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @derived(semantics=default_semantics(ScalarEncoding))
    def encoding(self) -> ScalarEncoding | Rejected:
        try:
            return ScalarEncoding(self.dtype)
        except ValueError as error:
            return reject("dtype-storage", str(error))


SCALAR_VIEW = ViewKey("scalar", ScalarEncoding)


class Scalar(Subspace[ScalarScope]):
    """A supplied or chosen dtype with independently inspectable encoding admission."""

    def __init__(
        self,
        dtype: ValueRef[QONNXDataType],
        valid_types: DatatypeDomain,
    ) -> None:
        builder = ScopeBuilder(ScalarScope, name="ScalarBoundary")
        admission = builder.add(
            "admission", ConstraintGroup(*type_constraints(builder, ScalarScope.dtype, valid_types))
        )
        accepted = builder.add("physical", View(ScalarScope.encoding, constraints=(admission,)))
        builder.export(SCALAR_VIEW).view(accepted)
        builder.bind(ScalarScope.dtype, dtype)
        placement = builder.place()
        super().__init__(
            placement.space_type,
            when=placement.when,
            bindings=placement.parameter_bindings,
            **placement.bindings,
        )
        self._view = accepted

    @property
    def dtype(self) -> ValueRef[QONNXDataType]:
        return self.ref(ScalarScope.dtype)

    @property
    def accepted_encoding(self) -> ValueRef[ScalarEncoding]:
        return self.accepted(SCALAR_VIEW)

    def view(self) -> View[ScalarEncoding]:
        return self._view


__all__ = ["Scalar", "ScalarScope", "ScalarEncoding", "type_constraints"]
