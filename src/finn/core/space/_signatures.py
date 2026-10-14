# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Python signature binding and annotation policies for authored callbacks."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, Union, cast, get_args, get_origin, get_type_hints

from ._configuration import Space
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Constraint,
    Derived,
    MemberRef,
    Projection,
    Supplied,
    ValueRef,
    View,
)
from .errors import DefinitionError
from .expressions import Expr
from .results import Available, Inapplicable, Rejected, Unresolved, marked_value_type
from .semantics import ValueSemantics, default_semantics

if TYPE_CHECKING:
    from .collection import EffectiveSpace

_OWNER = "space.signatures"


@dataclass(frozen=True, slots=True)
class BoundArgument:
    name: str
    source: ValueRef[object]
    annotation: object = object


@dataclass(frozen=True, slots=True)
class BoundFunction:
    function: Callable[..., object]
    dependencies: tuple[BoundArgument, ...]
    semantics: ValueSemantics[object]
    call_style: Literal["explicit", "self"] = "explicit"


def resolve_annotations(
    function: Callable[..., object], namespace: Mapping[str, object], owner: str
) -> dict[str, object]:
    try:
        return get_type_hints(function, localns=namespace)
    except (NameError, TypeError, ValueError) as exc:
        raise DefinitionError.broken(
            "annotation-unresolved",
            _OWNER,
            f"{owner}: cannot resolve annotations: {exc}",
            declaration=owner,
        ) from exc


def _answer_value_type(annotation: object) -> object | None:
    origin = get_origin(annotation)
    if origin not in (Union, types.UnionType):
        return None
    arguments = get_args(annotation)
    value_types = [get_args(arg)[0] for arg in arguments if get_origin(arg) is Available]
    if len(value_types) == 1 and all(
        get_origin(arg) is Available or arg in (Inapplicable, Rejected, Unresolved)
        for arg in arguments
    ):
        return cast(object, value_types[0])
    return None


def output_semantics(
    declaration: Derived[object] | View[object] | Constraint,
    hints: Mapping[str, object],
    owner: str,
) -> ValueSemantics[object]:
    annotation = hints.get("return")
    if annotation is None:
        raise DefinitionError.broken(
            "return-annotation-missing",
            _OWNER,
            f"{owner}: a return annotation is required",
            declaration=owner,
        )
    answer_type = _answer_value_type(annotation)
    semantics = cast("ValueSemantics[object] | None", declaration.semantics)
    if answer_type is not None and semantics is None:
        raise DefinitionError.broken(
            "semantics-unstated",
            _OWNER,
            f"{owner}: QueryResult[T] returns require explicit semantics=",
            declaration=owner,
        )
    value_type = answer_type if answer_type is not None else annotation
    marked_type = marked_value_type(value_type)
    value_type = marked_type if marked_type is not None else value_type
    origin = get_origin(value_type)
    nominal_type = origin if origin is not None else value_type
    if semantics is None:
        if getattr(nominal_type, "_is_protocol", False):
            raise DefinitionError.broken(
                "semantics-unstated",
                _OWNER,
                f"{owner}: Protocol outputs require explicit semantics=",
                declaration=owner,
            )
        if origin in (Union, types.UnionType) or not isinstance(nominal_type, type):
            raise DefinitionError.broken(
                "semantics-unstated",
                _OWNER,
                f"{owner}: output annotation needs explicit semantics=",
                declaration=owner,
            )
        try:
            semantics = default_semantics(nominal_type)
        except (TypeError, ValueError) as exc:
            raise DefinitionError.broken(
                "semantics-unstated", _OWNER, f"{owner}: {exc}", declaration=owner
            ) from exc
    token_origin = get_origin(semantics.type_token)
    nominal_token = semantics.type_token if token_origin is None else token_origin
    if (
        origin not in (Union, types.UnionType)
        and isinstance(nominal_type, type)
        and isinstance(nominal_token, type)
        and not getattr(nominal_type, "_is_protocol", False)
    ):
        if nominal_type is not nominal_token:
            raise DefinitionError.broken(
                "annotation-semantics-incompatible",
                _OWNER,
                f"{owner}: output annotation {nominal_type.__name__} is incompatible with "
                f"{semantics.name} semantics",
                declaration=owner,
            )
    return semantics


def resolve_source(source: object, effective: EffectiveSpace, owner: str) -> ValueRef[object]:
    if not isinstance(source, (ValueRef, View)):
        raise DefinitionError.broken(
            "dependency-not-value",
            _OWNER,
            f"{owner}: dependency must be a value reference",
            declaration=owner,
        )
    name = effective.aliases.get(source)
    if name is not None:
        replacement = effective.members[name]
        if not isinstance(replacement, (ValueRef, View)):
            raise DefinitionError.broken(
                "dependency-not-value",
                _OWNER,
                f"{owner}: overridden dependency {name} is not a value",
                declaration=owner,
            )
        return cast(ValueRef[object], replacement)
    if isinstance(source, (MemberRef, ChoiceMemberRef)):
        return source
    if isinstance(source, (Expr, Supplied, CaseRef, Projection)) and source.owner is None:
        return cast(ValueRef[object], source)
    raise DefinitionError.broken(
        "dependency-undeclared",
        _OWNER,
        f"{owner}: dependency is not declared in this effective scope",
        declaration=owner,
    )


def source_semantics(
    source: object,
    effective: EffectiveSpace,
) -> ValueSemantics[object] | None:
    """Infer local value/view types; cross-scope types are checked after linking."""
    seen: set[int] = set()
    while isinstance(source, (ValueRef, View)) and id(source) not in seen:
        seen.add(id(source))
        declared = cast("ValueSemantics[object] | None", source.semantics)
        semantics = effective.semantics.get(source, declared)
        if semantics is not None:
            return semantics
        if not isinstance(source, View) or source.source is None:
            return None
        source = source.source
    return None


def _annotation_accepts(annotation: object, value_type: type[object]) -> bool:
    if isinstance(annotation, tuple):
        return any(_annotation_accepts(option, value_type) for option in annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts(option, value_type) for option in get_args(annotation))
    if isinstance(origin, type):
        if getattr(origin, "_is_protocol", False):
            return True
        return issubclass(value_type, origin)
    if isinstance(annotation, type):
        if getattr(annotation, "_is_protocol", False):
            # Non-runtime Protocols cannot be checked with issubclass. The
            # explicit adapter semantics determines admissible runtime values.
            return True
        return issubclass(value_type, annotation)
    return True


def validate_argument(
    argument: BoundArgument, semantics: ValueSemantics[object], *, owner: str
) -> None:
    """Check one linked supplier using the same dependency-mode policy as collection."""
    value_type = argument.annotation
    token_origin = get_origin(semantics.type_token)
    nominal_token = semantics.type_token if token_origin is None else token_origin
    if isinstance(nominal_token, type) and not _annotation_accepts(value_type, nominal_token):
        raise DefinitionError.broken(
            "argument-annotation-incompatible",
            _OWNER,
            f"{owner}: annotation {argument.annotation!r} cannot consume {semantics.name}",
            declaration=owner,
        )


def bind_function(
    declaration: Derived[object] | Constraint | View[object],
    effective: EffectiveSpace,
    *,
    name: str,
    annotations: Mapping[str, object],
    semantics: ValueSemantics[object],
) -> BoundFunction:
    """Bind all arguments against one complete table, respecting overrides."""
    owner = f"{effective.space_type.__qualname__}.{name}"
    function = declaration.function
    if function is None:
        raise DefinitionError.broken(
            "view-function-absent",
            _OWNER,
            f"{owner}: value-authored view has no function to bind",
            declaration=owner,
        )
    hints = annotations
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if (
        inspect.isgeneratorfunction(function)
        or inspect.iscoroutinefunction(function)
        or inspect.isasyncgenfunction(function)
    ):
        raise DefinitionError.broken(
            "computation-not-function",
            _OWNER,
            f"{owner}: computations must be ordinary synchronous functions",
            declaration=owner,
        )
    if parameters and parameters[0].name == "self":
        parameter = parameters[0]
        if (
            len(parameters) != 1
            or declaration.aliases
            or parameter.kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            or parameter.default is not inspect.Parameter.empty
        ):
            raise DefinitionError.broken(
                "self-method-dependencies",
                _OWNER,
                f"{owner}: self methods cannot declare dependency arguments or aliases",
                declaration=owner,
            )
        receiver = hints.get("self", Self)
        if receiver is not Self and not (
            isinstance(receiver, type)
            and issubclass(receiver, Space)
            and issubclass(effective.space_type, receiver)
        ):
            raise DefinitionError.broken(
                "receiver-annotation-incompatible",
                _OWNER,
                f"{owner}: receiver annotation {receiver!r} cannot accept "
                f"{effective.space_type.__qualname__}",
                declaration=owner,
            )
        return BoundFunction(function, (), semantics, "self")
    extra = declaration.aliases.keys() - signature.parameters.keys()
    if extra:
        raise DefinitionError.broken(
            "alias-unknown",
            _OWNER,
            f"{owner}: aliases name unknown arguments {sorted(extra)}",
            declaration=owner,
        )
    arguments: list[BoundArgument] = []
    for parameter in signature.parameters.values():
        label = f"{owner} argument {parameter.name}"
        if parameter.name in ("self", "cls"):
            raise DefinitionError.broken(
                "implicit-self",
                _OWNER,
                f"{label}: implicit self/cls is unsupported",
                declaration=label,
            )
        if parameter.name == "when":
            raise DefinitionError.broken(
                "when-reserved",
                _OWNER,
                f"{label}: when is a reserved authoring control argument",
                declaration=label,
            )
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise DefinitionError.broken(
                "parameter-kind-unsupported",
                _OWNER,
                f"{label}: positional-only and variadic parameters are unsupported",
                declaration=label,
            )
        if parameter.default is not inspect.Parameter.empty:
            raise DefinitionError.broken(
                "dependency-default",
                _OWNER,
                f"{label}: dependency parameters cannot have defaults",
                declaration=label,
            )
        if parameter.name not in hints:
            raise DefinitionError.broken(
                "dependency-annotation-missing",
                _OWNER,
                f"{label}: a dependency annotation is required",
                declaration=label,
            )
        source = declaration.aliases.get(parameter.name)
        if parameter.name not in declaration.aliases:
            record = effective.members.get(parameter.name)
            if record is None:
                raise DefinitionError.broken(
                    "declaration-unknown",
                    _OWNER,
                    f"{label}: no declaration with this name",
                    declaration=label,
                )
            source = record
        resolved = resolve_source(source, effective, label)
        supplier_semantics = source_semantics(resolved, effective)
        input_type = hints[parameter.name]
        argument = BoundArgument(parameter.name, resolved, input_type)
        if supplier_semantics is not None:
            validate_argument(argument, supplier_semantics, owner=label)
        arguments.append(argument)
    return BoundFunction(function, tuple(arguments), semantics)
