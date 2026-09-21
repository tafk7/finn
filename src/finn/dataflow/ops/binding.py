# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact immutable node uses over the existing Space occurrence runtime."""

# Source/native imports are delayed to keep the source authoring boundary acyclic.
# ruff: noqa: PLC0415

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.space.declarations import AuthoringError, Space, Subspace, SubspaceChoice
from finn.dataflow.space.occurrence import (
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
                        from finn.dataflow._engine import Absent

                        return Absent()
                current = occurrence_child(current, declaration)
            elif isinstance(declaration, SubspaceChoice):
                choice = occurrence_choice(current, declaration)
                selected = choice.selected()
                if not isinstance(selected, Decided):
                    return cast("Answer[Space]", selected)
                explicit = next(remaining, None)
                if explicit is not None and explicit != selected.value:
                    from finn.dataflow._engine import Absent

                    return Absent()
                alternative = dict(declaration.alternatives)[selected.value]
                if alternative.when is not None:
                    active = current.answer(alternative.when)
                    if not isinstance(active, Decided):
                        return cast("Answer[Space]", active)
                    if not active.value:
                        from finn.dataflow._engine import Absent

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


@dataclass(frozen=True, slots=True)
class HydratedUse:
    root: Any
    implementation: ImplementationBinding
    operands: tuple[OperandBinding, ...]
    choices: tuple[ChoiceBinding, ...]
    interface: ImplementationBinding | None = None

    def __post_init__(self) -> None:
        # Freeze association to the actual source occurrence, not an arbitrary
        # externally supplied SourceNode with coincidentally equal fields.
        self.root._bound_node()
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "choices", tuple(self.choices))
        self._validate_association()
        from finn.dataflow.ops.base import source_declarations
        from finn.dataflow.ops.schema import OpInput, OpOutput

        declarations = dict(source_declarations(type(self.root)))
        sources, roles = set(), set()
        for binding in self.operands:
            declaration = declarations.get(binding.source)
            if not isinstance(declaration, (OpInput, OpOutput)):
                raise AuthoringError(f"unknown source operand {binding.source!r}")
            if declaration.index != binding.index or declaration.output != binding.output:
                raise AuthoringError("source operand binding disagrees with its declaration")
            if binding.source in sources or binding.role in roles:
                raise AuthoringError("source and public operand bindings must be unique")
            sources.add(binding.source)
            roles.add(binding.role)
        from finn.dataflow.model.logical.interface_authoring import public_operand_declarations

        targets = self.implementation.authored_targets(type(self.root))
        if self.interface is not None:
            targets += self.interface.authored_targets(type(self.root))
        for target_type, inputs in targets:
            public = {item.key: item for item in public_operand_declarations(target_type)}
            for binding in self.operands:
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

    @property
    def source(self) -> Any:
        return self.root.source

    def _validate_association(self) -> None:
        from finn.dataflow.ops.base import DataflowOpError

        definition = type(self.root)
        if (
            self.implementation != getattr(definition, "implementation_binding", None)
            or self.interface != getattr(definition, "interface_binding", None)
            or self.operands != tuple(getattr(definition, "operand_bindings", ()))
            or self.choices != tuple(getattr(definition, "choice_bindings", ()))
        ):
            raise DataflowOpError("use binding contradicts its source definition")

    def resolve(self) -> Answer[Space]:
        self._validate_association()
        return self.implementation.resolve(self.root)

    def require_implementation(self, occurrence: Space | None = None) -> Space:
        from finn.dataflow.ops.base import DataflowOpError

        answer = self.resolve()
        if not isinstance(answer, Decided):
            raise DataflowOpError("implementation is not selected", answer.findings)
        if occurrence is not None and not same_occurrence(answer.value, occurrence):
            raise DataflowOpError("implementation is not this exact node occurrence and point")
        return answer.value

    def _facet(self, key: str, name: str) -> Answer[Any]:
        from finn.dataflow.model.logical import interface_authoring

        self._validate_association()
        key = next((binding.role for binding in self.operands if binding.source == key), key)
        if key not in {binding.role for binding in self.operands}:
            raise AuthoringError(f"unknown bound public operand {key!r}")
        target = (self.interface or self.implementation).resolve(self.root)
        if not isinstance(target, Decided):
            return target
        binding = next(item for item in self.operands if item.role == key)
        answer: Answer[Any]
        if name == "operand_type" and binding.output:
            from finn.dataflow.space.occurrence import combine_assessments

            declaration = interface_authoring.operand_declaration(target.value, key)
            facet = target.value.assess_view(declaration.datatype)
            semantic = self.root.assess(type(self.root).type_source_accepts)
            answer = combine_assessments(
                "type_source", (semantic, *facet.constraints), facet
            ).accepted_answer
        else:
            answer = cast("Answer[Any]", getattr(interface_authoring, name)(target.value, key))
        if name == "operand_domain" and isinstance(answer, Decided):
            from math import prod
            from finn.dataflow.ops.mapping import CoordinateMapping
            from finn.dataflow._engine import Absent, Finding, FindingKind, QualifiedPath

            if not binding.output:
                shape = self.source.operand(binding.source).shape
                expected = shape
                if binding.adapter == CoordinateMapping.FLATTEN_LEADING and shape:
                    expected = (prod(shape[:-1]), shape[-1])
                elif binding.adapter == CoordinateMapping.TRANSPOSE_2D and len(shape) == 2:
                    expected = (shape[1], shape[0])
                if answer.value.extents != expected:
                    return Absent(
                        (
                            Finding(
                                FindingKind.REJECTION,
                                "operand-domain-binding",
                                QualifiedPath(key),
                                "source boundary adapter does not match the public operand domain",
                            ),
                        )
                    )
        return answer

    def operand_type(self, key: str) -> Answer[Any]:
        return self._facet(key, "operand_type")

    def operand_domain(self, key: str) -> Answer[Any]:
        return self._facet(key, "operand_domain")

    def operand_export(self, key: str) -> Answer[Any]:
        from finn.dataflow.model.logical.interface_authoring import operand_export

        key = next((binding.role for binding in self.operands if binding.source == key), key)
        target = self.resolve()
        if not isinstance(target, Decided):
            return target
        return operand_export(target.value, key)

    def successor(self, root: Any) -> HydratedUse:
        from finn.dataflow.ops.base import DataflowOpError

        old, new = _occurrence_state(self.root), _occurrence_state(root)
        if (
            old.runtime.lineage is not new.runtime.lineage
            or old.compiled is not new.compiled
            or old.scope != new.scope
            or root._bound_node() is not self.root._bound_node()
        ):
            raise DataflowOpError("successor does not share this source occurrence association")
        return HydratedUse(root, self.implementation, self.operands, self.choices, self.interface)

    def commit(self, values: dict[str, object]) -> HydratedUse:
        """Commit stable native keys using the compiled schema and engine batch."""
        from finn.dataflow.ops.native import choice_schema, encode_choice_value

        schema = {entry.name: entry for entry in choice_schema(self.root)}
        unknown = values.keys() - schema.keys()
        if unknown:
            raise AuthoringError(f"unknown native choice keys: {sorted(unknown)}")
        assignments = {}
        for key, value in values.items():
            entry = schema[key]
            encode_choice_value(entry, value)  # exact nominal/selector codec checks
            assignments[entry.choice.reference.path] = value
        root = occurrence_commit_paths(self.root, assignments)
        result = self.successor(root)
        from finn.dataflow.ops.native import serialize_choices

        reachable = serialize_choices(root)
        if any(key not in reachable for key in values):
            raise AuthoringError("choice is not reachable in this implementation point")
        return result

    def rebind(self, model: Any, build: Any = None, *, graph_context: Any = None) -> HydratedUse:
        return cast(
            "HydratedUse",
            self.root.rebind(model, build, graph_context=graph_context).hydrated_use(),
        )
