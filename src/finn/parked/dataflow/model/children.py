# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel child conventions over generic Subspace choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from finn.dataflow.model.logical.semantics import DATAFLOW_LOGICAL_RESULT_SEMANTICS
from finn.kernels.space.capabilities import (
    AssessedCapabilityOutput,
    canonicalize_assessed_capability_outputs,
)
from finn.kernels.space.compiler import _CompiledSpace
from finn.kernels.space.declarations import (
    AuthoringError,
    Projection,
    Space,
    Subspace,
    SubspaceChoice,
    ValueSource,
    declared_members,
    exported_members,
)

if TYPE_CHECKING:
    from finn.parked.dataflow.model.logical.authoring import KernelEndpoint


def _atomic(what: str, value: str | None) -> None:
    if value is None:
        return
    if not value or "." in value or "/" in value:
        raise AuthoringError(f"{what} must be one non-empty path segment")


class KernelChoice(SubspaceChoice):
    """A parent-owned lazy choice between Kernel-capable child Spaces."""

    selector_name = "kernel"
    capability_outputs: Mapping[str, AssessedCapabilityOutput] = MappingProxyType(
        {
            "logical_result": AssessedCapabilityOutput("logical"),
            "physical_result": AssessedCapabilityOutput("physical"),
            "physical_streams": AssessedCapabilityOutput(
                "physical",
                forwards_accepted_value=False,
                accepted_output_name="physical-streams",
            ),
        }
    )
    node_id: str | None

    __slots__ = ("node_id",)

    def __init__(
        self,
        *alternatives: Subspace[Space],
        role: str | None = None,
        node_id: str | None = None,
        when: ValueSource[bool] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        _atomic("a Kernel child role", role)
        _atomic("a Kernel child node id", node_id)
        object.__setattr__(self, "node_id", node_id)
        self._from_candidates(alternatives, outputs=outputs, when=when, name=role)

    def candidate_id(self, subspace: Subspace[Space]) -> str:
        if subspace.stable_name is not None:
            _atomic("a Kernel candidate id", subspace.stable_name)
            return subspace.stable_name
        candidate = getattr(subspace.space_type, "id", "")
        if not isinstance(candidate, str) or not candidate:
            raise AuthoringError(
                f"{subspace.space_type.__name__} needs a stable id or an explicit Subspace name"
            )
        _atomic("a Kernel candidate id", candidate)
        return candidate

    def validate_candidate(self, owner: type[Space], subspace: Subspace[Space]) -> None:
        del owner
        declarations = dict(declared_members(subspace.space_type))
        if not isinstance(declarations.get("logical"), Projection):
            raise AuthoringError(
                f"{subspace.space_type.__name__} does not declare a logical capability"
            )
        logical = exported_members(subspace.space_type).get("logical_result")
        if (
            logical is None
            or logical.value_semantics.type_token
            is not DATAFLOW_LOGICAL_RESULT_SEMANTICS.type_token
        ):
            raise AuthoringError(
                f"{subspace.space_type.__name__} does not export the logical Kernel capability"
            )

    def default_outputs(self) -> tuple[str, ...]:
        return ("logical_result",)

    @property
    def logical_result(self) -> ValueSource[object]:
        return self.__getattr__("logical_result")

    def input(self, boundary_id: str) -> KernelEndpoint:
        from finn.parked.dataflow.model.logical.authoring import KernelEndpoint  # noqa: PLC0415

        return KernelEndpoint(self, boundary_id, output=False)

    def output(self, boundary_id: str) -> KernelEndpoint:
        from finn.parked.dataflow.model.logical.authoring import KernelEndpoint  # noqa: PLC0415

        return KernelEndpoint(self, boundary_id, output=True)


def kernel_choice_members(kernel_type: type[Space]) -> tuple[tuple[str, KernelChoice], ...]:
    return tuple(
        (name, declaration)
        for name, declaration in declared_members(kernel_type)
        if isinstance(declaration, KernelChoice)
    )


def choice_role(member_name: str, declaration: KernelChoice) -> str:
    return member_name if declaration.stable_name is None else declaration.stable_name


def canonicalize_kernel_child_capabilities(
    compiled: _CompiledSpace[Space],
) -> _CompiledSpace[Space]:
    choices = dict(kernel_choice_members(compiled.owner))
    bindings = {
        member_name: declaration.capability_outputs for member_name, declaration in choices.items()
    }
    return canonicalize_assessed_capability_outputs(compiled, bindings)


__all__ = ["KernelChoice", "kernel_choice_members"]
