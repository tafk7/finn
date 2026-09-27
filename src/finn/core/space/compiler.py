# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile authored declarations into an immutable, validated linked model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

from ._configuration import Space
from ._linker import link_space
from .errors import DefinitionError, RequestError
from .ir import LinkedModel
from .references import ValueHandle, resolve_reference

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class SpaceModel(Generic[S]):
    """A prepared definition with frozen indexes and a typed configuration factory."""

    space_type: type[S]
    linked: LinkedModel

    def resolve(self, scope: int, reference: object) -> int:
        """Resolve a declaration in this exact scope without consulting its class."""

        if isinstance(reference, ValueHandle):
            if type(scope) is not int or not 0 <= scope < len(self.linked.scopes):
                raise RequestError("reference scope does not belong to this model")
            return reference._resolve(self.linked)
        return resolve_reference(
            self.linked.nodes, self.linked.scopes, self.linked.choices, scope, reference
        )

    def bind(
        self,
        parameters: Mapping[object, object] | None = None,
        /,
        **keyword_parameters: object,
    ) -> S:
        from .occurrence import bind  # noqa: PLC0415 - keep compilation evaluator-independent

        return bind(self, parameters if parameters is not None else {}, keyword_parameters)


_PREPARATION_LOCK = RLock()


def _constructor_families(space_type: type[Space]) -> tuple[type[object], ...]:
    """Return every class whose constructor protocol affects this family."""

    result: list[type[object]] = []
    for base in space_type.__mro__:
        if base is Space:
            break
        result.append(base)
    return tuple(result)


def _validate_constructors(space_types: tuple[type[Space], ...]) -> None:
    checked: set[type[object]] = set()
    for space_type in space_types:
        for base in _constructor_families(space_type):
            if base in checked:
                continue
            checked.add(base)
            if "__init__" in base.__dict__:
                raise DefinitionError(
                    f"{base.__qualname__}: custom instance __init__ is unsupported; "
                    "use declarations and ordinary helper methods"
                )
            if "__new__" in base.__dict__:
                raise DefinitionError(
                    f"{base.__qualname__}: custom instance __new__ is unsupported"
                )


def _definition_families(space_types: tuple[type[Space], ...]) -> tuple[type[Space], ...]:
    """Include concrete families and Space bases contributing effective declarations."""

    result: list[type[Space]] = []
    seen: set[type[Space]] = set()
    for space_type in space_types:
        for base in space_type.__mro__:
            if base is Space:
                break
            if issubclass(base, Space) and base not in seen:
                seen.add(base)
                result.append(base)
    return tuple(result)


def compile_space(space_type: type[S]) -> SpaceModel[S]:
    """Return the canonical prepared definition for one root family."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("compile_space requires a Space subclass")
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_prepared_model")
        if cached is not None:
            if not isinstance(cached, SpaceModel) or cached.space_type is not space_type:
                raise DefinitionError("invalid prepared-model cache on Space family")
            return cached
        linked = link_space(space_type)
        scope_types = tuple(dict.fromkeys(scope.space_type for scope in linked.scopes))
        _validate_constructors(scope_types)
        model = SpaceModel(space_type, linked)
        # Publication happens only after the complete definition linked successfully.
        for family in _definition_families(scope_types):
            type.__setattr__(family, "_space_definition_finalized", True)
        type.__setattr__(space_type, "_space_prepared_model", model)
        return model


__all__ = ["SpaceModel", "compile_space"]
