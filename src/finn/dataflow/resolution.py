# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-generic selected-dataflow references and resolution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar
from finn.dataflow.design import Decided, DesignPoint, Engine, ValueSemantics
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.network import DataflowNetwork


@dataclass(frozen=True)
class NetworkRef:
    """Private transitional engine value for selected Network aggregation."""

    network_id: str
    network: DataflowNetwork
    source_association: object


DataflowOpResult = NetworkRef

DATAFLOW_OP_RESULT_SEMANTICS: ValueSemantics[object] = ValueSemantics(
    DataflowOpResult,
    "DataflowOpResult",
    lambda value: isinstance(value, NetworkRef),
    lambda left, right: left == right,
    lambda value: value,
)


AssociationT = TypeVar("AssociationT")


@dataclass(frozen=True)
class ResolvedDataflowOp(Generic[AssociationT]):
    """One re-created design point and its selected logical dataflow."""

    engine: Engine = field(compare=False, repr=False)
    point: DesignPoint = field(compare=False, repr=False)
    source_scope_id: str
    selected_design_id: str
    network: DataflowNetwork
    source_association: AssociationT
    compiled: object | None = field(default=None, compare=False, repr=False)

    @property
    def result(self) -> ResolvedDataflowOp[AssociationT]:
        """Compatibility spelling during removal of the old result wrapper."""

        return self

    @property
    def network_id(self) -> str:
        """Compatibility spelling for the selected Design identity."""

        return self.selected_design_id

    def declared_value(self, member_name: str) -> object:
        """Resolve one compiler-declared member without exposing raw point traversal."""

        declarations = getattr(self.compiled, "declarations", None)
        if declarations is None:
            raise KeyError(member_name)
        ref = declarations.ref(member_name)
        if not isinstance(ref, Ref):
            raise KeyError(member_name)
        if ref.path in self.point.problem:
            return self.point.problem[ref.path]
        answer = self.engine.query_property(self.point, ref.path)
        if not isinstance(answer, Decided):
            raise KeyError(member_name)
        return answer.value


__all__ = [
    "DATAFLOW_OP_RESULT_SEMANTICS",
    "DataflowOpResult",
    "NetworkRef",
    "ResolvedDataflowOp",
]
