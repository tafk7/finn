# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile declarations into models, and open a declaration's design space.

Vocabulary: a *declaration* (``Room(area=12)``) is a template; a *model* is its
compiled structure, with no inputs; a *design space* is a model with its inputs
bound and every choice open; a *configuration* is a design space after some
choices, and a *design point* one complete for the question asked.

``design_space(node)`` is the one step from a declaration to a design space. A
Space class is compiled once, into the model every design space of it shares; a
root supplies plain values for its value formals, bound as runtime inputs. A
root that supplies structure (a reference, a fresh Decision) is refused:
structure belongs to a Space class. Preparing a model freezes every node
declaration it instantiates, and ``design_space`` freezes its root: from then
on, assignment is refused.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeVar, cast

from ._configuration import Space
from ._linker import link_space
from ._nodes import (
    NodeDecl,
    missing_formal,
    node_record,
    space_formals,
    structural_settings,
    unsupplied_formals,
)
from .declarations import MISSING, UNSUPPLIED
from .errors import DefinitionError, RequestError
from .ir import LinkedModel
from .references import ValueHandle, resolve_decision, resolve_reference

_OWNER = "space.compiler"

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class Model(Generic[S]):
    """A compiled declaration: its structure, with no inputs or choices. What
    inspection reads of its structure alone (its Decisions below a scope) is kept
    with it (``read``)."""

    space_type: type[S]
    linked: LinkedModel
    _read: dict[tuple[str, int], object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def resolve(self, scope: int, reference: object) -> int:
        """Resolve a declaration in this exact scope without consulting its class."""

        if isinstance(reference, ValueHandle):
            if type(scope) is not int or not 0 <= scope < len(self.linked.scopes):
                raise RequestError("reference scope does not belong to this model")
            return reference._resolve(self.linked)
        return resolve_reference(
            self.linked.nodes, self.linked.scopes, self.linked.choices, scope, reference
        )

    def decision(self, scope: int, reference: object) -> int:
        """The owning decision ``reference`` may edit in this scope."""

        if isinstance(reference, ValueHandle):
            index = self.resolve(scope, reference)
            if self.linked.nodes[index].kind != "decision":
                raise RequestError("parameter aliases are not independently editable")
            return index
        return resolve_decision(
            self.linked.nodes,
            self.linked.scopes,
            self.linked.choices,
            scope,
            reference,
            self.linked.editable_aliases,
        )


_PREPARATION_LOCK = RLock()


def _constructor_classes(space_type: type[Space]) -> tuple[type[object], ...]:
    """Return every class whose constructor protocol affects this Space class."""

    result: list[type[object]] = []
    for base in space_type.__mro__:
        if base is Space:
            break
        result.append(base)
    return tuple(result)


def _validate_constructors(space_types: tuple[type[Space], ...]) -> None:
    checked: set[type[object]] = set()
    for space_type in space_types:
        for base in _constructor_classes(space_type):
            if base in checked:
                continue
            checked.add(base)
            if "__init__" in base.__dict__:
                raise DefinitionError.broken(
                    "instance-constructor-custom",
                    _OWNER,
                    f"{base.__qualname__}: custom instance __init__ is unsupported; "
                    "use declarations and ordinary helper methods",
                    declaration=base.__qualname__,
                )
            if "__new__" in base.__dict__:
                raise DefinitionError.broken(
                    "instance-constructor-custom",
                    _OWNER,
                    f"{base.__qualname__}: custom instance __new__ is unsupported",
                    declaration=base.__qualname__,
                )


def _definition_classes(space_types: tuple[type[Space], ...]) -> tuple[type[Space], ...]:
    """Include concrete Space classes and Space bases contributing effective declarations."""

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


def _publish(space_type: type[S], linked: LinkedModel) -> Model[S]:
    scope_types = tuple(dict.fromkeys(scope.space_type for scope in linked.scopes))
    _validate_constructors(scope_types)
    model = Model(space_type, linked)
    # Publication happens only after the complete definition linked successfully.
    for definition in _definition_classes(scope_types):
        type.__setattr__(definition, "_space_definition_finalized", True)
    reason = f"{space_type.__name__} was prepared"
    for scope in linked.scopes:
        if isinstance(scope.record, NodeDecl):
            scope.record.freeze(reason)
    return model


def compile_model(space_type: type[S]) -> Model[S]:
    """The canonical model of a Space class whose formals are all runtime inputs."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError.broken(
            "not-a-space", _OWNER, "compile_model requires a Space subclass"
        )
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_prepared_model")
        if cached is not None:
            if not isinstance(cached, Model) or cached.space_type is not space_type:
                raise DefinitionError.broken(
                    "model-cache-invalid", _OWNER, "invalid prepared-model cache on Space class"
                )
            return cached
        model = _publish(space_type, link_space(space_type))
        type.__setattr__(space_type, "_space_prepared_model", model)
        return model


def root_record(node: object) -> NodeDecl:
    record = node_record(node)
    if record is None:
        raise RequestError(
            "design_space() takes a node declaration, as in design_space(House(budget=100))"
        )
    if record.placement is not None:
        raise RequestError(
            f"design_space() takes a root: {record.describe()} is already placed "
            f"({record.placement}); "
            "open the design space of the node that contains it"
        )
    structural = structural_settings(record)
    if structural:
        raise DefinitionError.broken(
            "root-structural-formal",
            _OWNER,
            f"design_space() of {record.describe()}: a root supplies plain values for its "
            f"value formals, and {', '.join(structural)} supplies structure (a reference "
            "or a Decision); write a subclass, or build one with composite()",
            declaration=record.describe(),
        )
    return record


def root_parameters(model: Model[Space], record: NodeDecl) -> dict[int, object]:
    """The root's plain values and value defaults, by parameter node."""

    values: dict[int, object] = {}
    root = model.linked.scopes[0]
    own = record.bindings
    for name, formal in space_formals(record.space_type).items():
        index = root.named_members.get(name)
        if index is None or model.linked.nodes[index].kind != "param":
            continue
        if name in own:
            value = own[name]
            values[index] = getattr(value, "value", value) if _is_const(value) else value
        else:
            default = formal.default
            if default is not MISSING and default is not UNSUPPLIED:
                values[index] = default
    return values


def _is_const(value: object) -> bool:
    from .declarations import Const

    return isinstance(value, Const)


def design_space(node: S) -> S:
    """Open the design space of a root declaration: its inputs bound, every choice open.

    ``design_space(House(budget=100))`` is typed as ``House``: the empty
    configuration of the space, narrowed with ``with_choices``.
    """

    from .occurrence import bind  # noqa: PLC0415 - keep compilation evaluator-independent

    record = root_record(node)
    formals = space_formals(record.space_type)
    missing = [
        missing_formal(name, formals[name], None, record.space_type.__qualname__)
        for name in unsupplied_formals(record)
    ]
    if missing:
        raise DefinitionError.broken(
            "formal-unsupplied",
            _OWNER,
            f"{'; '.join(missing)}; while opening the design space of {record.describe()}",
            declaration=record.describe(),
        )
    try:
        model = cast(Model[Space], compile_model(record.space_type))
    except DefinitionError as error:
        raise DefinitionError(
            f"{error}; while opening the design space of {record.describe()}",
            findings=error.findings,
        ) from error
    record.freeze("design_space() took it as a root")
    return cast(S, bind(model, root_parameters(model, record)))


__all__ = ["Model", "compile_model", "design_space"]
