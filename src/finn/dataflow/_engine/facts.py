# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private semantic graph analysis and runtime evaluation planning."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from itertools import pairwise
from types import MappingProxyType

from .declarations import DependencyKind, DependencyRef, DesignSpaceSpec
from .primitives import QualifiedPath, stable_topological_order


class FactKind(str, Enum):
    APPLIES = "applies"
    DOMAIN_READY = "domain_ready"
    VALUE = "value"
    PROPOSAL = "proposal"
    PROPERTY = "property"
    CONSTRAINT = "constraint"


_FACT_KIND_ORDER = {
    FactKind.APPLIES: 0,
    FactKind.DOMAIN_READY: 1,
    FactKind.VALUE: 2,
    FactKind.PROPOSAL: 3,
    FactKind.PROPERTY: 4,
    FactKind.CONSTRAINT: 5,
}


@total_ordering
@dataclass(frozen=True, slots=True)
class FactKey:
    kind: FactKind
    path: QualifiedPath

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FactKey):
            return NotImplemented
        return (self.path, _FACT_KIND_ORDER[self.kind]) < (
            other.path,
            _FACT_KIND_ORDER[other.kind],
        )

    def __str__(self) -> str:
        return f"{self.kind.value}({self.path})"


@dataclass(frozen=True, slots=True)
class EdgeProvenance:
    owner: QualifiedPath
    role: str
    dependency_name: str | None = None
    dependency_path: QualifiedPath | None = None
    dependency_kind: DependencyKind | None = None


@dataclass(frozen=True, slots=True)
class DeadProposal:
    path: QualifiedPath
    fact_trace: tuple[FactKey, ...]
    provenance: tuple[EdgeProvenance, ...]


@dataclass(frozen=True, slots=True)
class GraphAnalysis:
    """Validation/debug information discarded by an accepted design space."""

    prerequisites: Mapping[FactKey, frozenset[FactKey]]
    edge_provenance: Mapping[tuple[FactKey, FactKey], tuple[EdgeProvenance, ...]]
    topological_order: tuple[FactKey, ...]
    core_cycles: tuple[tuple[FactKey, ...], ...]
    dead_proposals: Mapping[QualifiedPath, DeadProposal]
    proposal_dependencies: Mapping[QualifiedPath, frozenset[QualifiedPath]]
    bootstrap_components: tuple[tuple[QualifiedPath, ...], ...]


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Only graph data used while evaluating accepted design spaces."""

    prerequisites: Mapping[FactKey, frozenset[FactKey]]
    topological_rank: Mapping[FactKey, int]
    dead_proposals: Mapping[QualifiedPath, DeadProposal]
    proposal_order: tuple[QualifiedPath, ...]
    value_dependencies: Mapping[QualifiedPath, frozenset[QualifiedPath]]

    def prerequisite_closure(self, targets: Iterable[FactKey]) -> frozenset[FactKey]:
        pending = list(targets)
        closure: set[FactKey] = set()
        while pending:
            node = pending.pop()
            if node in closure:
                continue
            closure.add(node)
            pending.extend(self.prerequisites.get(node, ()))
        return frozenset(closure)

    def evaluation_order(self, targets: Iterable[FactKey]) -> tuple[FactKey, ...]:
        closure = self.prerequisite_closure(targets)
        return tuple(sorted(closure, key=self.topological_rank.__getitem__))

    def assignment_order(self, submitted: frozenset[QualifiedPath]) -> tuple[QualifiedPath, ...]:
        graph = {
            path: self.value_dependencies.get(path, frozenset()) & submitted for path in submitted
        }
        return stable_topological_order(graph)

    def adoption_order(self, targets: frozenset[QualifiedPath]) -> tuple[QualifiedPath, ...]:
        retained = tuple(path for path in self.proposal_order if path in targets)
        return (*retained, *sorted(targets - frozenset(retained)))


def _fact_for_dependency(ref: DependencyRef) -> FactKey | None:
    if ref.kind is DependencyKind.DECISION:
        return FactKey(FactKind.VALUE, ref.path)
    if ref.kind is DependencyKind.PROPERTY:
        return FactKey(FactKind.PROPERTY, ref.path)
    return None


def _strong_components(
    graph: Mapping[FactKey, frozenset[FactKey]],
) -> tuple[tuple[FactKey, ...], ...]:
    """Return SCCs using iterative Kosaraju with a true DFS postorder."""

    visited: set[FactKey] = set()
    finish: list[FactKey] = []
    for start in sorted(graph):
        if start in visited:
            continue
        stack: list[tuple[FactKey, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for dependency in sorted(graph.get(node, ()), reverse=True):
                if dependency in graph and dependency not in visited:
                    stack.append((dependency, False))

    reverse: dict[FactKey, set[FactKey]] = {node: set() for node in graph}
    for node, dependencies in graph.items():
        for dependency in dependencies:
            if dependency in reverse:
                reverse[dependency].add(node)

    assigned: set[FactKey] = set()
    components: list[tuple[FactKey, ...]] = []
    for start in reversed(finish):
        if start in assigned:
            continue
        assigned.add(start)
        component: list[FactKey] = []
        pending = [start]
        while pending:
            node = pending.pop()
            component.append(node)
            for successor in sorted(reverse[node], reverse=True):
                if successor not in assigned:
                    assigned.add(successor)
                    pending.append(successor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _cyclic_components(
    graph: Mapping[FactKey, frozenset[FactKey]],
) -> tuple[tuple[FactKey, ...], ...]:
    return tuple(
        component
        for component in _strong_components(graph)
        if len(component) > 1 or component[0] in graph.get(component[0], ())
    )


def _proposal_schedule(
    graph: Mapping[QualifiedPath, frozenset[QualifiedPath]],
) -> tuple[tuple[QualifiedPath, ...], tuple[tuple[QualifiedPath, ...], ...]]:
    fact_nodes = {
        FactKey(FactKind.PROPOSAL, path): frozenset(
            FactKey(FactKind.PROPOSAL, dependency) for dependency in dependencies
        )
        for path, dependencies in graph.items()
    }
    components = _strong_components(fact_nodes)
    component_for = {node: component for component in components for node in component}
    condensation = {
        component: frozenset(
            dependency_component
            for node in component
            for dependency in fact_nodes[node]
            if (dependency_component := component_for[dependency]) != component
        )
        for component in components
    }
    ordered_components = stable_topological_order(condensation)
    order = tuple(node.path for component in ordered_components for node in component)
    bootstrap = tuple(
        tuple(node.path for node in component)
        for component in components
        if len(component) > 1 or component[0] in fact_nodes[component[0]]
    )
    return order, bootstrap


def _shortest_path(
    graph: Mapping[FactKey, frozenset[FactKey]], start: FactKey, target: FactKey
) -> tuple[FactKey, ...] | None:
    pending: deque[FactKey] = deque([start])
    previous: dict[FactKey, FactKey | None] = {start: None}
    while pending:
        node = pending.popleft()
        if node == target:
            path: list[FactKey] = []
            cursor: FactKey | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            return tuple(reversed(path))
        for dependency in sorted(graph.get(node, ())):
            if dependency not in previous:
                previous[dependency] = node
                pending.append(dependency)
    return None


def _first_value_frontier(
    graph: Mapping[FactKey, frozenset[FactKey]], start: FactKey
) -> frozenset[QualifiedPath]:
    pending = list(graph.get(start, ()))
    visited: set[FactKey] = set()
    values: set[QualifiedPath] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.kind is FactKind.VALUE:
            values.add(node.path)
            continue
        pending.extend(graph.get(node, ()))
    return frozenset(values)


def _provenance_sort_key(record: EdgeProvenance) -> tuple[QualifiedPath, str, str, str, str]:
    return (
        record.owner,
        record.role,
        record.dependency_name or "",
        "" if record.dependency_path is None else record.dependency_path.value,
        "" if record.dependency_kind is None else record.dependency_kind.value,
    )


def render_provenance(records: Iterable[EdgeProvenance]) -> tuple[str, ...]:
    return tuple(
        f"{record.owner}:{record.role}:{record.dependency_name or '-'}:"
        f"{record.dependency_path or '-'}"
        for record in records
    )


def compile_facts(specification: DesignSpaceSpec) -> tuple[GraphAnalysis, EvaluationPlan]:
    graph: dict[FactKey, set[FactKey]] = {}
    provenance: dict[tuple[FactKey, FactKey], list[EdgeProvenance]] = {}

    def node(kind: FactKind, path: QualifiedPath) -> FactKey:
        key = FactKey(kind, path)
        graph.setdefault(key, set())
        return key

    def edge(
        source: FactKey,
        target: FactKey,
        owner: QualifiedPath,
        role: str,
        ref: DependencyRef | None = None,
    ) -> None:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())
        provenance.setdefault((source, target), []).append(
            EdgeProvenance(
                owner,
                role,
                None if ref is None else ref.name,
                None if ref is None else ref.path,
                None if ref is None else ref.kind,
            )
        )

    def dependency_edges(
        source: FactKey,
        owner: QualifiedPath,
        role: str,
        dependencies: tuple[DependencyRef, ...],
    ) -> None:
        for ref in dependencies:
            target = _fact_for_dependency(ref)
            if target is not None:
                edge(source, target, owner, role, ref)

    for decision in specification.decisions:
        applies = node(FactKind.APPLIES, decision.path)
        ready = node(FactKind.DOMAIN_READY, decision.path)
        value = node(FactKind.VALUE, decision.path)
        if decision.applies_if is not None:
            dependency_edges(
                applies, decision.path, "applicability", decision.applies_if.dependencies
            )
        edge(ready, applies, decision.path, "domain_applicability")
        dependency_edges(ready, decision.path, "domain", decision.domain.dependencies)
        edge(value, applies, decision.path, "value_applicability")
        edge(value, ready, decision.path, "value_domain_ready")
        if decision.proposal is not None:
            proposal = node(FactKind.PROPOSAL, decision.path)
            edge(proposal, applies, decision.path, "proposal_applicability")
            edge(proposal, ready, decision.path, "proposal_domain_ready")
            dependency_edges(proposal, decision.path, "proposal", decision.proposal.dependencies)

    for derived_property in specification.properties:
        applies = node(FactKind.APPLIES, derived_property.path)
        value = node(FactKind.PROPERTY, derived_property.path)
        if derived_property.applies_if is not None:
            dependency_edges(
                applies,
                derived_property.path,
                "applicability",
                derived_property.applies_if.dependencies,
            )
        edge(value, applies, derived_property.path, "property_applicability")
        dependency_edges(
            value,
            derived_property.path,
            "property",
            derived_property.evaluator.dependencies,
        )

    for constraint in specification.constraints:
        applies = node(FactKind.APPLIES, constraint.path)
        value = node(FactKind.CONSTRAINT, constraint.path)
        if constraint.applies_if is not None:
            dependency_edges(
                applies, constraint.path, "applicability", constraint.applies_if.dependencies
            )
        edge(value, applies, constraint.path, "constraint_applicability")
        dependency_edges(value, constraint.path, "constraint", constraint.evaluator.dependencies)

    frozen_graph = {key: frozenset(value) for key, value in graph.items()}
    core_graph = {
        key: frozenset(
            dependency for dependency in dependencies if dependency.kind is not FactKind.PROPOSAL
        )
        for key, dependencies in frozen_graph.items()
        if key.kind is not FactKind.PROPOSAL
    }
    core_cycles = _cyclic_components(core_graph)

    proposal_paths = frozenset(
        decision.path for decision in specification.decisions if decision.proposal is not None
    )
    dead: dict[QualifiedPath, DeadProposal] = {}
    proposal_dependencies: dict[QualifiedPath, frozenset[QualifiedPath]] = {}
    for path in sorted(proposal_paths):
        proposal = FactKey(FactKind.PROPOSAL, path)
        trace = _shortest_path(frozen_graph, proposal, FactKey(FactKind.VALUE, path))
        if trace is not None:
            records: list[EdgeProvenance] = []
            for source, dependency in pairwise(trace):
                records.extend(
                    sorted(provenance.get((source, dependency), ()), key=_provenance_sort_key)
                )
            dead[path] = DeadProposal(path, trace, tuple(records))
        frontier = _first_value_frontier(frozen_graph, proposal)
        proposal_dependencies[path] = frozenset(frontier & proposal_paths) - {path}

    value_dependencies = {
        decision.path: frozenset(
            path
            for path in _first_value_frontier(frozen_graph, FactKey(FactKind.VALUE, decision.path))
            if path != decision.path
        )
        for decision in specification.decisions
    }
    frozen_proposal_dependencies = {
        path: frozenset(dependencies) for path, dependencies in proposal_dependencies.items()
    }
    proposal_order, bootstrap_components = _proposal_schedule(frozen_proposal_dependencies)
    topological_order = stable_topological_order(frozen_graph)
    frozen_prerequisites = MappingProxyType(frozen_graph)
    frozen_provenance = MappingProxyType(
        {
            key: tuple(sorted(records, key=_provenance_sort_key))
            for key, records in provenance.items()
        }
    )
    frozen_dead = MappingProxyType(dead)
    analysis = GraphAnalysis(
        frozen_prerequisites,
        frozen_provenance,
        topological_order,
        core_cycles,
        frozen_dead,
        MappingProxyType(frozen_proposal_dependencies),
        bootstrap_components,
    )
    plan = EvaluationPlan(
        frozen_prerequisites,
        MappingProxyType({key: rank for rank, key in enumerate(topological_order)}),
        frozen_dead,
        proposal_order,
        MappingProxyType(value_dependencies),
    )
    return analysis, plan


__all__ = [
    "DeadProposal",
    "EdgeProvenance",
    "EvaluationPlan",
    "FactKey",
    "FactKind",
    "GraphAnalysis",
    "compile_facts",
    "render_provenance",
]
