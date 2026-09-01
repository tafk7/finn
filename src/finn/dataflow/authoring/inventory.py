# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authoritative operation assembly over a ``DataflowDesign`` inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignScope,
    DesignInput,
    DesignNode,
    KernelPlacement,
    PlacementSelection,
)
from finn.dataflow.authoring.input_supply import InputSupplyDeclaration
from finn.dataflow.authoring.realization import (
    DESIGN_REALIZATION_PATH,
    DesignRealization,
    validate_realization,
)
from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Ref,
    Scope,
    finite,
    predicate,
)
from finn.dataflow.design import (
    ABSENT,
    Absent,
    Answer,
    Decided,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.kernels._declaration import CompiledKernelDeclaration
from finn.dataflow.kernels.kernel import BoundRegion, Kernel, bind_kernel, check_declared_references
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import validate_network
from finn.dataflow.resolution import NetworkRef
from finn.dataflow.spec_algebra import assemble_specs, duplicate_values, gate_spec


def _finding(code: str, message: str, **values: object) -> Finding:
    return Finding(
        FindingKind.LIMITATION,
        code,
        DESIGN_REALIZATION_PATH,
        message,
        tuple(values.items()),
    )


def _design_selected(choice: Ref[str], design_id: str) -> EvaluatorSpec[Answer[bool]]:
    def selected(design: str) -> bool:
        return design == design_id

    return predicate(choice.path, {"design": choice}, selected)


@dataclass(frozen=True)
class DataflowDesignDeclaration:
    """The immutable compiled declarations for one authored design."""

    id: str
    version: str
    namespace: str
    spec: DesignSpaceSpec
    network: Ref[DataflowNetwork]
    nodes: tuple[DesignNode, ...]
    placements: tuple[KernelPlacement, ...]
    input_mappings: tuple[DesignInput, ...]
    hardware: tuple[CompiledKernelDeclaration, ...]
    owner: type[DataflowDesign]
    decision_handles: tuple[Ref[object], ...] = field(default=(), repr=False, compare=False)
    constraint_handles: tuple[ConstraintRef, ...] = field(default=(), repr=False, compare=False)

    def placement(self, name: str) -> KernelPlacement:
        for placement in self.placements:
            if placement.name == name:
                return placement
        raise KeyError(f"design {self.id!r} has no placement {name!r}")

    def realize(self, engine: Engine, point: DesignPoint) -> Answer[DesignRealization]:
        """Bind every active placement and validate exact resolved coverage."""

        network_answer = engine.query_property(point, self.network.path)
        if not isinstance(network_answer, Decided):
            return cast("Answer[DesignRealization]", network_answer)
        network = cast(DataflowNetwork, network_answer.value)
        network_issues = validate_network(network).issues
        if network_issues:
            return Unresolved(
                tuple(
                    _finding(
                        "design-network-invalid",
                        issue.message,
                        issue_code=issue.code,
                        issue_path=issue.path,
                    )
                    for issue in network_issues
                )
            )

        kernels: dict[str, Kernel] = {}
        active: list[str] = []
        findings: list[Finding] = []
        for placement in self.placements:
            answer = engine.query_property(point, placement.selected_kernel.path)
            if isinstance(answer, Absent):
                if answer.is_rejection:
                    findings.extend(answer.findings)
                continue
            active.append(placement.name)
            if not isinstance(answer, Decided):
                findings.extend(answer.findings)
                continue
            selected = cast(PlacementSelection, answer.value)
            if selected.kernel_id is None:
                findings.append(
                    _finding(
                        "design-placement-has-no-kernel",
                        f"active placement {placement.name!r} has no physical Kernel candidate",
                        placement=placement.name,
                    )
                )
                continue
            candidate = placement.candidate(selected.kernel_id)
            regions: dict[str, BoundRegion] = {}
            missing = False
            for node in placement.nodes:
                try:
                    resolved_node = network.node(node.node_id)
                except KeyError:
                    findings.append(
                        _finding(
                            "design-placement-node-absent",
                            f"active placement {placement.name!r} covers node "
                            f"{node.node_id!r}, which is absent from the resolved Network",
                            placement=placement.name,
                            node=node.node_id,
                        )
                    )
                    missing = True
                    continue
                regions[node.role] = BoundRegion(node.role, node.node_id, resolved_node.region)
            edge_ids = {item.role: item.edge_id for item in placement.edges}
            if missing:
                continue
            bound = bind_kernel(engine, candidate, point, regions, edge_ids)
            if isinstance(bound, Decided):
                kernels[placement.name] = bound.value
            else:
                findings.extend(bound.findings)
        if findings:
            return Unresolved(tuple(findings))
        return self.validate_realization(network, kernels, active_placements=tuple(active))

    def validate_realization(
        self,
        network: DataflowNetwork,
        kernels: Mapping[str, Kernel],
        *,
        active_placements: Sequence[str] | None = None,
    ) -> Answer[DesignRealization]:
        """Validate one already-configured Kernel collection against this design."""

        active_names = (
            tuple(item.name for item in self.placements)
            if active_placements is None
            else tuple(active_placements)
        )
        by_name = {item.name: item for item in self.placements}
        findings: list[Finding] = []
        for name in sorted(set(active_names) - set(by_name)):
            findings.append(
                _finding(
                    "design-active-placement-unknown",
                    f"active placement {name!r} is not declared by design {self.id!r}",
                    placement=name,
                )
            )
        for name in sorted(set(active_names) - set(kernels)):
            findings.append(
                _finding(
                    "design-placement-kernel-missing",
                    f"active placement {name!r} has no configured Kernel",
                    placement=name,
                )
            )
        for name in sorted(set(kernels) - set(active_names)):
            findings.append(
                _finding(
                    "design-placement-kernel-foreign",
                    f"configured Kernel was supplied for inactive or unknown placement {name!r}",
                    placement=name,
                )
            )
        placed: dict[str, Kernel] = {}
        for name in active_names:
            placement = by_name.get(name)
            kernel = kernels.get(name)
            if placement is None or kernel is None:
                continue
            candidate_ids = {item.id for item in placement.candidates}
            if kernel.id not in candidate_ids:
                findings.append(
                    _finding(
                        "design-placement-kernel-not-a-candidate",
                        f"Kernel {kernel.id!r} is not a candidate for placement {name!r}",
                        placement=name,
                        kernel=kernel.id,
                    )
                )
            expected_nodes = {item.node_id for item in placement.nodes}
            if set(kernel.node_ids) != expected_nodes:
                findings.append(
                    _finding(
                        "design-placement-node-coverage-mismatch",
                        f"Kernel {kernel.id!r} does not cover placement {name!r} exactly",
                        placement=name,
                        expected=tuple(sorted(expected_nodes)),
                        actual=kernel.node_ids,
                    )
                )
            expected_edges = {item.edge_id for item in placement.edges}
            if set(kernel.edge_ids) != expected_edges:
                findings.append(
                    _finding(
                        "design-placement-edge-coverage-mismatch",
                        f"Kernel {kernel.id!r} does not absorb placement {name!r} exactly",
                        placement=name,
                        expected=tuple(sorted(expected_edges)),
                        actual=kernel.edge_ids,
                    )
                )
            placed[name] = kernel
        coverage = validate_realization(self.id, network, placed)
        if isinstance(coverage, Unresolved):
            findings.extend(coverage.findings)
        if findings:
            return Unresolved(tuple(findings))
        return coverage


def declare_dataflow_design(
    design: type[DataflowDesign],
    namespace: str,
    inputs: object,
    *,
    input_supplies: Sequence[InputSupplyDeclaration] = (),
) -> tuple[DataflowDesignDeclaration, DataflowDesignScope[object]]:
    """Run one ``DataflowDesign`` subclass under a collision-free namespace."""

    if not design.id or not design.version:
        raise AuthoringError(f"{design.__name__} must set a design id and version")
    scope: DataflowDesignScope[object] = DataflowDesignScope(namespace, inputs)
    design.define(scope)
    for supply in input_supplies:
        supply.apply(scope)
    declaration = DataflowDesignDeclaration(
        design.id,
        design.version,
        namespace,
        scope.spec(),
        scope.network_ref,
        scope.nodes,
        scope.placements,
        scope.input_mappings,
        scope.hardware_declarations,
        design,
        scope.decision_handles,
        (
            *scope.constraint_handles,
            *(
                constraint
                for declaration in scope.hardware_declarations
                for constraint in declaration.constraint_handles
            ),
        ),
    )
    return declaration, scope


@dataclass(frozen=True)
class DataflowDesignEntry:
    """One design class and the typed inputs its declaration receives."""

    design: type[DataflowDesign]
    inputs: object
    specs: tuple[DesignSpaceSpec, ...] = ()
    decision_handles: tuple[Ref[object], ...] = ()
    constraint_handles: tuple[ConstraintRef, ...] = ()


@dataclass(frozen=True)
class DataflowDesignInventory:
    """One Operation's closed inventory of selectable designs."""

    namespace: str
    declarations: tuple[DataflowDesignDeclaration, ...]
    input_supplies: tuple[InputSupplyDeclaration, ...]
    design_path: QualifiedPath | None
    specification: DesignSpaceSpec
    design_selection: Ref[str] | None = field(default=None, repr=False, compare=False)

    @property
    def design_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.declarations)

    @property
    def constraint_handles(self) -> tuple[ConstraintRef, ...]:
        return tuple(
            dict.fromkeys(
                constraint
                for declaration in self.declarations
                for constraint in declaration.constraint_handles
            )
        )

    @property
    def structural_decisions(self) -> tuple[Ref[object], ...]:
        return (
            *((self.design_selection,) if self.design_selection is not None else ()),
            *(
                handle
                for declaration in self.declarations
                for handle in declaration.decision_handles
            ),
            *(supply.choice for supply in self.input_supplies),
        )

    @property
    def artifact_decisions(self) -> tuple[Ref[object], ...]:
        return (
            *self.structural_decisions,
            *(
                handle
                for declaration in self.declarations
                for placement in declaration.placements
                for candidate in placement.candidates
                for handle in candidate.decision_handles
            ),
            *(
                handle
                for supply in self.input_supplies
                for handle in supply.decision_handles
                if handle.path != supply.choice.path
            ),
        )

    def declaration(self, design_id: str) -> DataflowDesignDeclaration:
        for declaration in self.declarations:
            if declaration.id == design_id:
                return declaration
        raise KeyError(f"{design_id!r} is not in the {self.namespace!r} design inventory")

    def selected(self, point: DesignPoint) -> Answer[DataflowDesignDeclaration]:
        if self.design_path is None:
            return Decided(self.declarations[0])
        if self.design_path not in point.assignments:
            return Unresolved(
                (
                    _finding(
                        "dataflow-design-unselected",
                        f"no DataflowDesign is committed for {self.namespace}",
                    ),
                )
            )
        return Decided(self.declaration(cast(str, point.assignments[self.design_path])))

    def realize(self, engine: Engine, point: DesignPoint) -> Answer[DesignRealization]:
        selected = self.selected(point)
        if not isinstance(selected, Decided):
            return cast("Answer[DesignRealization]", selected)
        return selected.value.realize(engine, point)


def declare_dataflow_design_inventory(
    namespace: str,
    entries: Sequence[DataflowDesignEntry],
    *,
    input_supplies: Sequence[InputSupplyDeclaration] = (),
    shared_specs: Sequence[DesignSpaceSpec] = (),
) -> DataflowDesignInventory:
    """Compile a closed Operation inventory into one ordinary flat spec."""

    declared_entries = tuple(entries)
    if not namespace or not declared_entries:
        raise AuthoringError("a design inventory needs a namespace and at least one design")
    ids = tuple(item.design.id for item in declared_entries)
    if any(not item for item in ids):
        raise AuthoringError("every DataflowDesign in an inventory must have an id")
    duplicates = duplicate_values(ids)
    if duplicates:
        raise AuthoringError(f"DataflowDesign ids are duplicated: {list(duplicates)}")
    supplies = tuple(input_supplies)
    source_duplicates = duplicate_values(tuple(item.source_operand for item in supplies))
    if source_duplicates:
        raise AuthoringError(
            f"input-supply policies overlap source operands: {list(source_duplicates)}"
        )

    declarations = tuple(
        replace(
            declaration,
            spec=assemble_specs((*entry.specs, declaration.spec)),
            decision_handles=(*entry.decision_handles, *declaration.decision_handles),
            constraint_handles=(
                *entry.constraint_handles,
                *declaration.constraint_handles,
            ),
        )
        for entry in declared_entries
        for declaration in (
            declare_dataflow_design(
                entry.design,
                f"{namespace}.design.{entry.design.id}",
                entry.inputs,
                input_supplies=supplies,
            )[0],
        )
    )
    design_path: QualifiedPath | None = None
    design_selection: Ref[str] | None = None
    own_spec = DesignSpaceSpec()
    design_specs: tuple[DesignSpaceSpec, ...]
    if len(declarations) == 1:
        design_specs = (declarations[0].spec,)
    else:
        selector = Scope(namespace)
        choice = selector.decision("design", str, domain=finite(ids))
        design_path = choice.path
        design_selection = choice
        own_spec = selector.spec()
        design_specs = tuple(
            gate_spec(
                declaration.spec,
                _design_selected(choice, declaration.id),
            )
            for declaration in declarations
        )

    specification = assemble_specs(
        (
            *tuple(shared_specs),
            own_spec,
            *(item.spec for item in supplies),
            *design_specs,
        )
    )
    check_declared_references(
        specification,
        tuple(kernel for declaration in declarations for kernel in declaration.hardware),
    )
    return DataflowDesignInventory(
        namespace,
        declarations,
        supplies,
        design_path,
        specification,
        design_selection,
    )


@dataclass(frozen=True)
class DesignSelectionMetadata:
    """Structural identity of one selected design and its active placements."""

    design_id: str
    decision_paths: tuple[QualifiedPath, ...]
    placements: tuple[tuple[str, str], ...]

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(kernel_id for _placement, kernel_id in self.placements)

    def kernel_id(self, placement: str) -> str | None:
        return next(
            (kernel_id for name, kernel_id in self.placements if name == placement),
            None,
        )


def selected_design_metadata(
    inventory: DataflowDesignInventory,
    design_id: str,
    *,
    supply_modes: Mapping[str, str],
    placement_selections: Mapping[tuple[str, str], object],
) -> DesignSelectionMetadata:
    """Describe selected declaration identity without inspecting compiled specs."""

    declaration = inventory.declaration(design_id)
    paths: list[QualifiedPath] = []
    if inventory.design_selection is not None:
        paths.append(inventory.design_selection.path)
    paths.extend(supply.choice.path for supply in inventory.input_supplies)
    paths.extend(handle.path for handle in declaration.decision_handles)

    placements: list[tuple[str, str]] = []
    for placement in declaration.placements:
        selected = placement_selections.get((design_id, placement.name), ABSENT)
        if selected is ABSENT:
            continue
        if not isinstance(selected, PlacementSelection):
            raise TypeError(
                f"placement {design_id}.{placement.name} produced "
                f"{type(selected).__name__}, not PlacementSelection"
            )
        if selected.kernel_id is None:
            continue
        candidate = placement.candidate(selected.kernel_id)
        placements.append((placement.name, candidate.id))
        paths.extend(handle.path for handle in candidate.decision_handles)

    for supply in inventory.input_supplies:
        mode = supply_modes[supply.source_operand]
        if mode == supply.external_id:
            continue
        paths.extend(
            handle.path for handle in supply.decision_handles if handle.path != supply.choice.path
        )

    return DesignSelectionMetadata(
        design_id,
        tuple(dict.fromkeys(paths)),
        tuple(placements),
    )


@dataclass(frozen=True)
class DataflowOpAuthoring:
    """Compiled operation declarations plus their typed runtime handles."""

    specification: DesignSpaceSpec
    result: Ref[NetworkRef]
    source_association: Ref[object]
    selection_constraint_set: str | None
    structural_readiness_profile: str | None
    artifact_readiness_profile: str | None
    feasibility_constraint_sets: tuple[str, ...]
    inventory: DataflowDesignInventory


def declare_dataflow_op_authoring(
    inventory: DataflowDesignInventory,
    operation: Scope,
    *,
    result: Ref[NetworkRef],
    source_association: Ref[object],
    structural_properties: Sequence[Ref[object]],
    structural_constraints: Sequence[ConstraintRef],
    structural_constraint_set: str,
    feasibility_constraint_set: str,
    structural_readiness_profile: str,
    artifact_readiness_profile: str,
    additional_structural_decisions: Sequence[Ref[object]] = (),
    additional_artifact_decisions: Sequence[Ref[object]] = (),
    additional_feasibility_constraints: Sequence[ConstraintRef] = (),
) -> DataflowOpAuthoring:
    """Compile the uniform operation boundary owned by an inventory.

    Operation code supplies only its genuinely local result properties and
    constraints.  Design decisions, Kernel choices, inventory constraints,
    and readiness aggregation remain authoritative here.
    """

    feasibility = (
        *inventory.constraint_handles,
        *additional_feasibility_constraints,
        *structural_constraints,
    )
    operation.include_in(structural_constraint_set, *structural_constraints)
    operation.include_in(feasibility_constraint_set, *feasibility)
    operation.readiness_profile(
        structural_readiness_profile,
        decisions=(*inventory.structural_decisions, *additional_structural_decisions),
        properties=structural_properties,
        constraints=structural_constraints,
    )
    operation.readiness_profile(
        artifact_readiness_profile,
        decisions=(
            *inventory.artifact_decisions,
            *additional_structural_decisions,
            *additional_artifact_decisions,
        ),
        properties=structural_properties,
        constraints=feasibility,
    )
    return DataflowOpAuthoring(
        assemble_specs((inventory.specification, operation.spec())),
        result,
        source_association,
        feasibility_constraint_set,
        structural_readiness_profile,
        artifact_readiness_profile,
        (feasibility_constraint_set,),
        inventory,
    )


__all__ = [
    "DataflowOpAuthoring",
    "DataflowDesignDeclaration",
    "DataflowDesignEntry",
    "DataflowDesignInventory",
    "DesignSelectionMetadata",
    "declare_dataflow_design",
    "declare_dataflow_op_authoring",
    "declare_dataflow_design_inventory",
    "selected_design_metadata",
]
