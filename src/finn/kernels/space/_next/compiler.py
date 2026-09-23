# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile authored declarations into an immutable, validated linked model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import inspect
from typing import Generic, TypeVar

from .collection import BoundFunction, EffectiveSpace, collect_space
from .declarations import (
    Const,
    Constraint,
    ConstraintGroup,
    Declaration,
    Decision,
    Derived,
    Param,
    Readiness,
    Space,
    ValueRef,
    View,
)
from .domains import Domain
from .errors import DefinitionError, RequestError
from .ir import Argument, LinkedModel, Node, Scope
from .results import Finding, FindingKind
from .semantics import ValueSemantics

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class SpaceModel(Generic[S]):
    """A reusable family. Its frozen indexes remain authoritative after compile."""

    space_type: type[S]
    linked: LinkedModel

    def resolve(self, scope: int, reference: object) -> int:
        """Resolve a declaration in this exact scope without consulting its class."""

        if type(scope) is not int or not 0 <= scope < len(self.linked.scopes):
            raise RequestError("reference scope does not belong to this model")
        try:
            return self.linked.scopes[scope].members[reference]
        except (KeyError, TypeError) as cause:
            raise RequestError("reference is not a member of this compiled scope") from cause

    def start(self, parameters: Mapping[object, object] | None = None) -> S:
        from .occurrence import start  # noqa: PLC0415 - keep compilation evaluator-independent

        return start(self, parameters if parameters is not None else {})


def _validated_order(nodes: tuple[Node, ...]) -> tuple[int, ...]:
    """Iterative Kosaraju validation and dependency-first order in O(V + E)."""

    dependencies = tuple(node.dependencies for node in nodes)
    reverse: list[list[int]] = [[] for _ in nodes]
    for node in nodes:
        for dependency in dependencies[node.index]:
            reverse[dependency].append(node.index)

    # Finish dependencies before their users. Iterator frames avoid Python's
    # recursion limit even when the dependency depth equals the model size.
    visited: set[int] = set()
    finished: list[int] = []
    for root in range(len(nodes)):
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(dependencies[root]))]
        while stack:
            current, adjacent = stack[-1]
            child = next(adjacent, None)
            if child is None:
                finished.append(current)
                stack.pop()
            elif child not in visited:
                visited.add(child)
                stack.append((child, iter(dependencies[child])))

    assigned: set[int] = set()
    cycles: list[tuple[str, ...]] = []
    for root in reversed(finished):
        if root in assigned:
            continue
        component: list[int] = []
        pending = [root]
        assigned.add(root)
        while pending:
            current = pending.pop()
            component.append(current)
            for child in reverse[current]:
                if child not in assigned:
                    assigned.add(child)
                    pending.append(child)
        if len(component) > 1 or root in dependencies[root]:
            cycles.append(tuple(sorted(nodes[index].key for index in component)))
    if cycles:
        findings = tuple(
            Finding(
                FindingKind.AUTHORING,
                "cyclic-dependency",
                members[0],
                "cyclic dependencies: " + ", ".join(members),
                details=(("members", members),),
            )
            for members in sorted(cycles)
        )
        raise DefinitionError("model contains cyclic dependencies", findings=findings)
    return tuple(finished)


class _FlatLinker:
    def __init__(self, effective: EffectiveSpace) -> None:
        self.effective = effective
        self.by_name = {name: index for index, name in enumerate(effective.members)}
        self.declarations = tuple(record.declaration for record in effective.members.values())
        self.by_declaration: dict[object, int] = {
            declaration: self.by_name[name] for declaration, name in effective.aliases.items()
        }
        self.nodes: list[Node] = []
        self.generated: list[Node] = []

    def reference(self, source: object, *, owner: str) -> int:
        try:
            return self.by_declaration[source]
        except (KeyError, TypeError) as cause:
            raise DefinitionError(f"{owner}: reference is not a member of this scope") from cause

    def semantics(self, declaration: Declaration, *, owner: str) -> ValueSemantics[object]:
        semantics = self.effective.semantics.get(declaration)
        if semantics is None:
            # A value-authored view can follow an inferred Derived. Resolve its
            # source through the effective table instead of mutating the view.
            if isinstance(declaration, View) and declaration.source is not None:
                index = self.reference(declaration.source, owner=owner)
                source = self.declarations[index]
                semantics = self.effective.semantics.get(source)
        if semantics is None:
            raise DefinitionError(f"{owner}: no value semantics for this declaration")
        return semantics

    def arguments(self, function: BoundFunction, *, owner: str) -> tuple[Argument, ...]:
        result: list[Argument] = []
        for dependency in function.dependencies:
            if dependency.mode not in ("required", "optional", "answer"):
                raise DefinitionError(f"{owner}: unsupported dependency mode")
            result.append(
                Argument(
                    dependency.name,
                    self.reference(dependency.source, owner=owner),
                    dependency.mode,
                )
            )
        return tuple(result)

    def obligations(
        self,
        sources: Iterable[object],
        allowed: tuple[type[object], ...],
        *,
        owner: str,
    ) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        for source in sources:
            if not isinstance(source, allowed):
                raise DefinitionError(f"{owner}: unsupported obligation declaration")
            index = self.reference(source, owner=owner)
            if index in seen:
                raise DefinitionError(f"{owner}: duplicate obligation")
            result.append(index)
            seen.add(index)
        return tuple(result)

    def domain(
        self, declaration: Decision[object], *, owner: str
    ) -> tuple[Domain[object], tuple[Argument, ...]]:
        semantics = self.semantics(declaration, owner=owner)
        # Domain records and finite values are frozen independently of the
        # declaration. Pure user callbacks themselves are deliberately shared.
        try:
            domain = declaration.domain.with_semantics(semantics)
        except Exception as cause:
            raise DefinitionError(f"{owner}: invalid decision domain: {cause}") from cause
        arguments: list[Argument] = []
        for name, source in domain.dependencies:
            if not isinstance(source, ValueRef):
                raise DefinitionError(
                    f"{owner}: domain dependency {name} must be a value reference"
                )
            arguments.append(Argument(name, self.reference(source, owner=owner)))
        supplied = dict.fromkeys((argument.name for argument in arguments))
        for role, function, values in (
            ("membership", domain.accepts, {"candidate": None, **supplied}),
            ("enumeration", domain.candidates, supplied),
        ):
            if function is not None:
                try:
                    inspect.signature(function).bind(**values)
                except (TypeError, ValueError) as cause:
                    raise DefinitionError(
                        f"{owner}: incompatible domain {role} signature"
                    ) from cause
        return domain, tuple(arguments)

    def link(self, name: str, declaration: Declaration) -> Node:
        index = self.by_name[name]
        # Explicit constructors keep the IR fields typed, without a dictionary
        # of Any values crossing the compile/runtime boundary.
        if isinstance(declaration, Param):
            return Node(
                index,
                0,
                name,
                "param",
                self.semantics(declaration, owner=name),
                required=declaration.required,
            )
        if isinstance(declaration, Const):
            semantics = self.semantics(declaration, owner=name)
            try:
                value = semantics.freeze(declaration.value)
            except Exception as cause:
                raise DefinitionError(f"{name}: constant snapshot failed") from cause
            return Node(index, 0, name, "const", semantics, value=value)
        if isinstance(declaration, Decision):
            domain, arguments = self.domain(declaration, owner=name)
            return Node(
                index,
                0,
                name,
                "decision",
                self.semantics(declaration, owner=name),
                domain=domain,
                domain_arguments=arguments,
            )
        if isinstance(declaration, (Derived, Constraint)):
            function = self.effective.functions[name]
            return Node(
                index,
                0,
                name,
                "constraint" if isinstance(declaration, Constraint) else "derived",
                function.semantics,
                arguments=self.arguments(function, owner=name),
                function=function.function,
            )
        if isinstance(declaration, ConstraintGroup):
            return Node(
                index,
                0,
                name,
                "group",
                constraints=self.obligations(declaration.constraints, (Constraint,), owner=name),
            )
        if isinstance(declaration, Readiness):
            return Node(
                index,
                0,
                name,
                "readiness",
                requires=self.obligations(
                    declaration.requires,
                    (ValueRef, Constraint, ConstraintGroup),
                    owner=name,
                ),
            )
        if isinstance(declaration, View):
            semantics = self.semantics(declaration, owner=name)
            if declaration.source is not None:
                if not isinstance(declaration.source, ValueRef):
                    raise DefinitionError(f"{name}: view output must be a value reference")
                output = self.reference(declaration.source, owner=name)
            else:
                function = self.effective.functions[name]
                output = len(self.effective.members) + len(self.generated)
                self.generated.append(
                    Node(
                        output,
                        0,
                        name + ".$output",
                        "derived",
                        semantics,
                        arguments=self.arguments(function, owner=name),
                        function=function.function,
                    )
                )
            return Node(
                index,
                0,
                name,
                "view",
                semantics,
                output=output,
                constraints=self.obligations(
                    declaration.constraints,
                    (Constraint, ConstraintGroup),
                    owner=name,
                ),
                requires=self.obligations(
                    declaration.requires,
                    (ValueRef, Constraint, ConstraintGroup, Readiness),
                    owner=name,
                ),
            )
        raise DefinitionError(f"{name}: nested or unsupported declaration in the flat compiler")

    def build(self) -> LinkedModel:
        for name, record in self.effective.members.items():
            self.nodes.append(self.link(name, record.declaration))
        nodes = tuple((*self.nodes, *self.generated))
        order = _validated_order(nodes)
        root = Scope(
            0,
            None,
            "",
            self.effective.space_type,
            self.by_declaration,
            named_members=self.by_name,
        )
        return LinkedModel(
            nodes,
            (root,),
            order,
            tuple(node.index for node in nodes if node.kind == "param"),
            tuple(node.index for node in nodes if node.kind == "decision"),
            {node.key: node.index for node in nodes},
        )


def compile_space(space_type: type[S]) -> SpaceModel[S]:
    """Validate the family once. No user evaluator runs while linking it."""

    effective = collect_space(space_type)
    return SpaceModel(space_type, _FlatLinker(effective).build())


__all__ = ["SpaceModel", "compile_space"]
