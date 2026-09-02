# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private scoped lowering support for one ``DataflowOp``.

``OpDesign`` is used while binding class-local operation declarations. Legacy
tests also exercise it directly, but it is not part of the public contributor
façade.

Provenance is FINN authoring metadata, not an engine problem-field kind.  It
answers one question the engine has no opinion about -- *who is entitled to
supply this value* -- and that question is what keeps the two projections
honest.  A graph projection that writes a build fact, or a build projection
that fabricates tensor geometry, is a mistake the operation can detect at the
moment it happens rather than a value silently overwritten later.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from finn.dataflow.authoring.provenance import (
    BUILD_OWNED,
    GRAPH_OWNED,
    ProblemProvenance,
    Provenance,
)
from finn.dataflow.authoring.scope import AuthoringError, Ref, Scope, T
from finn.dataflow.design import QualifiedPath, ValueSemantics

#: The path root under which target facts live, shared across operations
#: because the target is not owned by any one of them.
TARGET_ROOT = "target"


class OpDesign(Scope):
    """One operation's authoring namespace, with problem ownership.

    The operation owns two path roots that a Kernel scope does not: its
    problem namespace, and (jointly with every other operation) the shared
    target namespace.  Semantic and constraint paths come from ``Scope``.
    """

    def __init__(self, namespace: str, *, problem_namespace: str | None = None) -> None:
        super().__init__(namespace)
        self.problem_namespace = problem_namespace if problem_namespace is not None else namespace
        self._provenance: dict[QualifiedPath, Provenance] = {}

    # -- path allocation ---------------------------------------------------

    def problem_path(self, name: str) -> QualifiedPath:
        return QualifiedPath(f"problem.{self.problem_namespace}.{name}")

    def target_path(self, name: str) -> QualifiedPath:
        return QualifiedPath(f"problem.{TARGET_ROOT}.{name}")

    # -- problem declaration -----------------------------------------------

    def fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        provenance: Provenance,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """Declare one problem field and record who supplies it.

        ``path`` overrides the derived path for a field this operation reads
        but shares with another family, such as a generic parameter-supply
        contract.  The provenance is recorded either way.
        """

        resolved = (
            self.target_path(name)
            if path is None and provenance is Provenance.TARGET
            else self.problem_path(name)
            if path is None
            else QualifiedPath.parse(path)
        )
        if resolved in self._provenance:
            raise AuthoringError(f"problem field {resolved} is declared twice")
        handle: Ref[T] = self.problem_field(
            resolved,
            value_type,
            required=required,
            validate=validate,
            description=description,
        )
        self._provenance[resolved] = provenance
        return handle

    def graph_fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """A fact read directly from the source graph."""

        return self.fact(
            name,
            value_type,
            provenance=Provenance.GRAPH,
            path=path,
            required=required,
            validate=validate,
            description=description,
        )

    def analysis_fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """A fact an analysis recovered from the source graph."""

        return self.fact(
            name,
            value_type,
            provenance=Provenance.GRAPH_ANALYSIS,
            path=path,
            required=required,
            validate=validate,
            description=description,
        )

    def target_fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """A property of the deployment target, under the shared target root."""

        return self.fact(
            name,
            value_type,
            provenance=Provenance.TARGET,
            path=path,
            required=required,
            validate=validate,
            description=description,
        )

    def build_fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """A build-configuration or invocation fact."""

        return self.fact(
            name,
            value_type,
            provenance=Provenance.BUILD,
            path=path,
            required=required,
            validate=validate,
            description=description,
        )

    def upstream_fact(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        path: QualifiedPath | str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """A fact an enclosing assembly supplies rather than a projection."""

        return self.fact(
            name,
            value_type,
            provenance=Provenance.UPSTREAM,
            path=path,
            required=required,
            validate=validate,
            description=description,
        )

    # -- output ------------------------------------------------------------

    def provenance(self) -> ProblemProvenance:
        return ProblemProvenance(MappingProxyType(dict(self._provenance)))


__all__ = [
    "BUILD_OWNED",
    "GRAPH_OWNED",
    "OpDesign",
    "ProblemProvenance",
    "Provenance",
    "TARGET_ROOT",
]
