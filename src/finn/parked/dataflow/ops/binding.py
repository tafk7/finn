# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authored binding helpers for a DataflowOp at its ordinary Space point."""

# Source/native imports are delayed to keep the source authoring boundary acyclic.
# ruff: noqa: PLC0415

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, cast

from finn.kernels._engine import Answer, Decided
from finn.kernels.space.declarations import AuthoringError, Space, Subspace, SubspaceChoice
from finn.kernels.space.occurrence import (
    _occurrence_state,
    occurrence_child,
    occurrence_choice,
    occurrence_commit_paths,
)


@dataclass(frozen=True, slots=True)
class ImplementationBinding:
    """An authored child-member route, with explicit choice alternatives en route.

    A final choice member resolves its selected alternative lazily. Intermediate
    choice members consume the next route element as an exact alternative id.
    """

    members: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if any(not isinstance(name, str) or not name for name in self.members):
            raise AuthoringError("implementation members must be nonempty names")

    def resolve(self, root: Space) -> Answer[Space]:
        current = root
        remaining = iter(self.members)
        for name in remaining:
            declaration = getattr(type(current), name, None)
            if isinstance(declaration, Subspace):
                if declaration.when is not None:
                    active = current.answer(declaration.when)
                    if not isinstance(active, Decided):
                        return cast("Answer[Space]", active)
                    if not active.value:
                        from finn.kernels._engine import Absent

                        return Absent()
                current = occurrence_child(current, declaration)
            elif isinstance(declaration, SubspaceChoice):
                choice = occurrence_choice(current, declaration)
                selected = choice.selected()
                if not isinstance(selected, Decided):
                    return cast("Answer[Space]", selected)
                explicit = next(remaining, None)
                if explicit is not None and explicit != selected.value:
                    from finn.kernels._engine import Absent

                    return Absent()
                alternative = dict(declaration.alternatives)[selected.value]
                if alternative.when is not None:
                    active = current.answer(alternative.when)
                    if not isinstance(active, Decided):
                        return cast("Answer[Space]", active)
                    if not active.value:
                        from finn.kernels._engine import Absent

                        return Absent()
                current = choice.alternative(selected.value)
            else:
                raise AuthoringError(f"{type(current).__name__}.{name} is not a child member")
        return Decided(current)

    def authored_targets(
        self, root_type: type[Space]
    ) -> tuple[tuple[type[Space], dict[int, object]], ...]:
        """Trace bindings through authored nesting, independently of selection."""

        def visit(
            cls: type[Space], route: tuple[str, ...], inputs: dict[int, object]
        ) -> list[tuple[type[Space], dict[int, object]]]:
            if not route:
                return [(cls, inputs)]
            declaration = getattr(cls, route[0], None)
            tail = route[1:]
            children: tuple[Subspace[Any], ...]
            if isinstance(declaration, Subspace):
                children = (declaration,)
            elif isinstance(declaration, SubspaceChoice):
                alternatives = dict(declaration.alternatives)
                if tail:
                    if tail[0] not in alternatives:
                        raise AuthoringError("implementation route names an unknown alternative")
                    children = (alternatives[tail[0]],)
                    tail = tail[1:]
                else:
                    children = tuple(alternatives.values())
            else:
                raise AuthoringError("implementation route must name authored child members")
            results = []
            for child in children:
                child_inputs = {
                    id(getattr(child.space_type, name)): inputs.get(id(value), value)
                    for name, value in child.bindings
                }
                results.extend(visit(child.space_type, tail, child_inputs))
            return results

        return tuple(visit(root_type, self.members, {}))


@dataclass(frozen=True, slots=True)
class ChoiceBinding:
    """A durable ONNX key and an exact authored occurrence/choice locator."""

    key: str
    members: tuple[str, ...]
    member: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if (
            not isinstance(self.key, str)
            or not self.key
            or not isinstance(self.member, str)
            or not self.member
            or any(not isinstance(item, str) or not item for item in self.members)
        ):
            raise AuthoringError("a choice binding needs a key and exact member locator")

    @property
    def path(self) -> str:
        return ".".join((*self.members, self.member))


@dataclass(frozen=True, slots=True)
class OperandBinding:
    """One source declaration supplies value identity and all tensor facets."""

    source: str
    role: str
    index: int
    output: bool = False
    adapter: object = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str)
            or not self.source
            or not isinstance(self.role, str)
            or not self.role
            or type(self.index) is not int
            or self.index < 0
            or type(self.output) is not bool
        ):
            raise AuthoringError("operand binding requires source, public role and index")


def same_occurrence(left: Space, right: Space) -> bool:
    """Value equality/root containment alone is not occurrence association."""

    a, b = _occurrence_state(left), _occurrence_state(right)
    return a.runtime is b.runtime and a.compiled is b.compiled and a.scope == b.scope


def validate_bindings(operation: Any) -> None:
    """Validate the Op's authored bindings, without copying its point or schema."""
    operation._bound_node()
    implementation = getattr(type(operation), "implementation_binding", None)
    if not isinstance(implementation, ImplementationBinding):
        raise AuthoringError("DataflowOp must declare its exact implementation binding")
    operands = tuple(getattr(type(operation), "operand_bindings", ()))
    interface = getattr(type(operation), "interface_binding", None)
    if interface is not None and not isinstance(interface, ImplementationBinding):
        raise AuthoringError("interface binding must name an authored implementation route")
    if any(not isinstance(binding, OperandBinding) for binding in operands):
        raise AuthoringError("operand bindings must be authored OperandBinding declarations")
    from finn.parked.dataflow.ops.base import source_declarations
    from finn.parked.dataflow.ops.schema import OpInput, OpOutput

    declarations = dict(source_declarations(type(operation)))
    sources, roles = set(), set()
    for binding in operands:
        declaration = declarations.get(binding.source)
        if not isinstance(declaration, (OpInput, OpOutput)):
            raise AuthoringError(f"unknown source operand {binding.source!r}")
        if declaration.index != binding.index or declaration.output != binding.output:
            raise AuthoringError("source operand binding disagrees with its declaration")
        if binding.source in sources or binding.role in roles:
            raise AuthoringError("source and public operand bindings must be unique")
        sources.add(binding.source)
        roles.add(binding.role)
    from finn.parked.dataflow.model.logical.interface_authoring import public_operand_declarations

    targets = implementation.authored_targets(type(operation))
    if interface is not None:
        targets += interface.authored_targets(type(operation))
    for target_type, inputs in targets:
        public = {item.key: item for item in public_operand_declarations(target_type)}
        for binding in operands:
            if binding.role not in public:
                raise AuthoringError(f"implementation has no public role {binding.role!r}")
            export = public[binding.role]
            if (export.direction == "output") != binding.output:
                raise AuthoringError("public operand binding direction disagrees with source")
            if not binding.output:
                source = cast(OpInput, declarations[binding.source])
                # Direct tensor datatype bindings have one canonical source.
                # Repeated equal datatypes cannot hide a wrong source operand.
                supplied = inputs.get(id(export.datatype.output))
                if supplied is not None and supplied is not source.datatype:
                    raise AuthoringError(
                        "operand facts and public value binding name different sources"
                    )


def resolve_implementation(operation: Any) -> Answer[Space]:
    validate_bindings(operation)
    binding = cast(ImplementationBinding, type(operation).implementation_binding)
    return binding.resolve(operation)


def require_implementation(operation: Any, occurrence: Space | None = None) -> Space:
    from finn.parked.dataflow.ops.base import DataflowOpError

    answer = resolve_implementation(operation)
    if not isinstance(answer, Decided):
        raise DataflowOpError("implementation is not selected", answer.findings)
    if occurrence is not None and not same_occurrence(answer.value, occurrence):
        raise DataflowOpError("implementation is not this exact node occurrence and point")
    return answer.value


def operand_facet(operation: Any, key: str, name: str) -> Answer[Any]:
    from finn.parked.dataflow.model.logical import interface_authoring

    validate_bindings(operation)
    operands = tuple(getattr(type(operation), "operand_bindings", ()))
    implementation = type(operation).implementation_binding
    interface = getattr(type(operation), "interface_binding", None)
    key = next((binding.role for binding in operands if binding.source == key), key)
    if key not in {binding.role for binding in operands}:
        raise AuthoringError(f"unknown bound public operand {key!r}")
    target: Answer[Space] = (interface or implementation).resolve(operation)
    if not isinstance(target, Decided):
        return target
    binding = next(item for item in operands if item.role == key)
    answer: Answer[Any]
    if name == "operand_type" and binding.output:
        from finn.kernels.space.occurrence import combine_assessments

        declaration = interface_authoring.operand_declaration(target.value, key)
        facet = target.value.assess_view(declaration.datatype)
        semantic = operation.assess(type(operation).type_source_accepts)
        answer = combine_assessments(
            "type_source", (semantic, *facet.constraints), facet
        ).accepted_answer
    else:
        answer = cast("Answer[Any]", getattr(interface_authoring, name)(target.value, key))
    if name == "operand_domain" and isinstance(answer, Decided):
        from finn.parked.dataflow.ops.mapping import CoordinateMapping, checked_boundary_map
        from finn.dataflow.model.logical.maps import (
            IdentityCoordinateMap,
            AffineRankMap,
            ExplicitCoordinateMap,
        )
        from finn.dataflow.model.logical.network import PositionMap
        from finn.kernels._engine import Absent, Finding, FindingKind, QualifiedPath

        explicit = isinstance(
            binding.adapter, (IdentityCoordinateMap, AffineRankMap, ExplicitCoordinateMap)
        )
        if not binding.output or explicit:
            shape = (
                PositionMap.from_coordinate_map(binding.adapter).source_set.ambient.extents
                if binding.output and explicit
                else operation.source.operand(binding.source).shape
            )
            try:
                checked_boundary_map(
                    binding.adapter or CoordinateMapping.IDENTITY,
                    shape,
                    answer.value.extents,
                    output=binding.output,
                )
            except (TypeError, ValueError) as error:
                return Absent(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "operand-domain-binding",
                            QualifiedPath(key),
                            str(error),
                        ),
                    )
                )
    return answer


def bound_operand_export(operation: Any, key: str) -> Answer[Any]:
    from finn.parked.dataflow.model.logical.interface_authoring import operand_export

    key = next(
        (
            binding.role
            for binding in getattr(type(operation), "operand_bindings", ())
            if binding.source == key
        ),
        key,
    )
    target = resolve_implementation(operation)
    if not isinstance(target, Decided):
        return target
    return operand_export(target.value, key)


def commit_choices(operation: Any, values: Mapping[str, object]) -> Any:
    """Commit stable native keys using the compiled schema and engine batch."""
    from finn.parked.dataflow.ops.native import choice_schema, encode_choice_value

    operation._bound_node()
    schema = {entry.name: entry for entry in choice_schema(operation)}
    unknown = values.keys() - schema.keys()
    if unknown:
        raise AuthoringError(f"unknown native choice keys: {sorted(unknown)}")
    assignments = {}
    for key, value in values.items():
        entry = schema[key]
        encode_choice_value(entry, value)  # exact nominal/selector codec checks
        assignments[entry.choice.reference.path] = value
    root = occurrence_commit_paths(operation, assignments)
    from finn.parked.dataflow.ops.native import serialize_choices

    reachable = serialize_choices(root)
    if any(key not in reachable for key in values):
        raise AuthoringError("choice is not reachable in this implementation point")
    return root
