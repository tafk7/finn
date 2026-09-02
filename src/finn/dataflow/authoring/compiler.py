# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private compiler for class-authored dataflow operation declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import cast

from finn.dataflow.authoring.declarations import (
    CompiledClassDeclarations,
    DeclarationGroup,
    DeclarationLayer,
    DeclarationTemplate,
    compile_class_declarations,
)
from finn.dataflow.authoring.design import DataflowDesign
from finn.dataflow.authoring.inventory import (
    DataflowDesignEntry,
    DataflowDesignInventory,
    declare_dataflow_design_inventory,
    declare_dataflow_op_authoring,
)
from finn.dataflow.authoring.input_supply import InputSupplyDeclaration
from finn.dataflow.authoring.op_design import OpDesign, ProblemProvenance
from finn.dataflow.authoring.persistence import (
    DecisionStorageCodec,
    Persist,
    compile_persistence,
)
from finn.dataflow.authoring.projection import ProjectionPlan
from finn.dataflow.authoring.scope import AuthoringError, Ref
from finn.dataflow.design import (
    DATAFLOW_NETWORK_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    Answer,
    Decided,
    DependencyView,
    DesignSpaceSpec,
    EvaluatorSpec,
    QualifiedPath,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network


@dataclass(frozen=True, slots=True)
class BoundTensor:
    """Typed handles exported by one class-local tensor declaration."""

    present: Ref[bool]
    tensor_id: Ref[str]
    shape: Ref[tuple[int, ...]]
    datatype: Ref[object]
    initializer_present: Ref[bool]
    initializer_fingerprint: Ref[str]


@dataclass(frozen=True, slots=True)
class DesignExport:
    """Unbound reference to a named declaration exported by one Design use."""

    use: UsesDesign
    member_path: str

    def __getattr__(self, name: str) -> DesignExport:
        if name.startswith("_"):
            raise AttributeError(name)
        return DesignExport(self.use, f"{self.member_path}.{name}")


@dataclass(frozen=True, slots=True)
class UsesDesign:
    """One class-local use of a reusable ``DataflowDesign`` declaration."""

    design: type[DataflowDesign]
    inputs: object
    association_member: str = "source_association"

    def __getattr__(self, name: str) -> DesignExport:
        if name.startswith("_"):
            raise AttributeError(name)
        return DesignExport(self, name)


@dataclass(frozen=True, slots=True)
class DesignChoice:
    owner: ClosedDesigns


@dataclass(frozen=True, slots=True)
class ClosedDesigns:
    """Closed operation-owned inventory of Design uses."""

    uses: tuple[UsesDesign, ...]
    choice: DesignChoice

    def __init__(self, *uses: UsesDesign) -> None:
        if not uses:
            raise ValueError("ClosedDesigns requires at least one Design")
        if len({id(item) for item in uses}) != len(uses):
            raise ValueError("ClosedDesigns cannot contain the same use twice")
        object.__setattr__(self, "uses", tuple(uses))
        object.__setattr__(self, "choice", DesignChoice(self))


@dataclass(frozen=True, slots=True)
class SupplyExport:
    use: UsesInputSupply
    member_path: str

    def __getattr__(self, name: str) -> SupplyExport:
        if name.startswith("_"):
            raise AttributeError(name)
        return SupplyExport(self.use, f"{self.member_path}.{name}")


@dataclass(frozen=True, slots=True)
class UsesInputSupply:
    """One operation-owned conditional source-supply declaration."""

    factory: Callable[..., object]
    inputs: object
    choice: SupplyExport

    def __init__(self, factory: Callable[..., object], inputs: object) -> None:
        object.__setattr__(self, "factory", factory)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "choice", SupplyExport(self, "declaration.choice"))

    def __getattr__(self, name: str) -> SupplyExport:
        if name.startswith("_"):
            raise AttributeError(name)
        return SupplyExport(self, name)


@dataclass(frozen=True, slots=True)
class CompiledDataflowOperation:
    """Complete private adapter product for one class-authored operation."""

    owner: type[object]
    namespace: str
    declarations: CompiledClassDeclarations
    specification: DesignSpaceSpec
    problem_provenance: ProblemProvenance
    projection: ProjectionPlan
    result: Ref[object]
    source_association: Ref[object]
    persistence: Mapping[QualifiedPath, DecisionStorageCodec]
    selection_constraint_set: str | None
    structural_readiness_profile: str | None
    artifact_readiness_profile: str | None
    feasibility_constraint_sets: tuple[str, ...]
    inventory: DataflowDesignInventory | None = None


def _string_setting(owner: type[object], name: str) -> str:
    value = getattr(owner, name, "")
    if not isinstance(value, str) or not value:
        raise AuthoringError(f"{owner.__name__}.{name} must be a non-empty string")
    return value


def _resolve_template_value(
    value: object,
    declarations: CompiledClassDeclarations,
    external: Mapping[int, Ref[object]] = MappingProxyType({}),
) -> object:
    if id(value) in external:
        return external[id(value)]
    if isinstance(value, DeclarationTemplate):
        member = declarations.template_members.get(id(value))
        if member is None:
            raise AuthoringError("a Design input references a declaration outside the operation")
        return declarations.ref(member)
    if isinstance(value, DeclarationGroup):
        member = next(
            (name for name, group in declarations.groups.items() if group is value),
            None,
        )
        if member is None:
            raise AuthoringError(
                "a Design input references a declaration group outside the operation"
            )
        return BoundTensor(
            cast("Ref[bool]", declarations.ref(f"{member}.present")),
            cast("Ref[str]", declarations.ref(f"{member}.tensor_id")),
            cast("Ref[tuple[int, ...]]", declarations.ref(f"{member}.shape")),
            declarations.ref(f"{member}.datatype"),
            cast("Ref[bool]", declarations.ref(f"{member}.initializer_present")),
            cast("Ref[str]", declarations.ref(f"{member}.initializer_fingerprint")),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _resolve_template_value(
                    getattr(value, item.name), declarations, external
                )
                for item in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_resolve_template_value(item, declarations, external) for item in value)
    if isinstance(value, list):
        return [_resolve_template_value(item, declarations, external) for item in value]
    if isinstance(value, Mapping):
        return {
            _resolve_template_value(key, declarations, external): _resolve_template_value(
                item, declarations, external
            )
            for key, item in value.items()
        }
    return value


def _selection_evaluator(
    selector: Ref[str] | None,
    values: Sequence[tuple[str, Ref[object]]],
) -> EvaluatorSpec[Answer[object]]:
    dependencies = []
    if selector is not None:
        dependencies.append(selector.dependency("selected_design"))
    dependencies.extend(
        ref.allow_absent().dependency(f"value_{index}")
        for index, (_design_id, ref) in enumerate(values)
    )

    def select(view: DependencyView) -> Answer[object]:
        selected = values[0][0] if selector is None else cast(str, view["selected_design"])
        for index, (design_id, _ref) in enumerate(values):
            if design_id == selected:
                return Decided(view[f"value_{index}"])
        raise KeyError(f"selected Design {selected!r} is absent from the closed inventory")

    return EvaluatorSpec(tuple(dependencies), select)


def _compile_designs(
    owner: type[object],
    declarations: CompiledClassDeclarations,
    supplies: Sequence[InputSupplyDeclaration] = (),
    supplied_refs: Mapping[int, Ref[object]] = MappingProxyType({}),
) -> tuple[
    DataflowDesignInventory | None,
    Ref[object] | None,
    Ref[object] | None,
    Mapping[int, Ref[object]],
]:
    closed = getattr(owner, "designs", None)
    if not isinstance(closed, ClosedDesigns):
        return None, None, None, MappingProxyType({})
    inventory = declare_dataflow_design_inventory(
        declarations.namespace,
        tuple(
            DataflowDesignEntry(
                use.design,
                _resolve_template_value(use.inputs, declarations, supplied_refs),
            )
            for use in closed.uses
        ),
        input_supplies=supplies,
        reference_specs=(declarations.scope.spec(),),
    )
    by_use = {id(use): inventory.declaration(use.design.id) for use in closed.uses}

    def exported_ref(use: UsesDesign, member_path: str) -> Ref[object] | None:
        declaration = by_use[id(use)]
        direct = declaration.exports.get(member_path)
        if direct is not None:
            return direct
        suffix = f".{member_path}"
        candidates = tuple(
            handle
            for placement in declaration.placements
            for candidate in placement.candidates
            for handle in candidate.decision_handles
            if handle.path.value.endswith(suffix)
        )
        return candidates[0] if len(candidates) == 1 else None

    associations: list[tuple[str, Ref[object]]] = []
    for use in closed.uses:
        declaration = by_use[id(use)]
        association = exported_ref(use, use.association_member)
        if association is None:
            raise AuthoringError(
                f"{use.design.__name__} does not export {use.association_member!r}"
            )
        associations.append((declaration.id, association))
    first_semantics = associations[0][1].semantics
    if any(
        not first_semantics.is_compatible_with(item.semantics)
        for _design_id, item in associations[1:]
    ):
        raise AuthoringError("all Design source-association exports need compatible semantics")

    scope = declarations.scope
    association_ref = scope.derived_evaluator(
        "op.source_association",
        first_semantics,
        evaluate=_selection_evaluator(inventory.design_selection, associations),
    )
    networks = tuple(
        (declaration.id, cast("Ref[object]", declaration.network))
        for declaration in inventory.declarations
    )
    result_ref = scope.derived_evaluator(
        "op.network",
        DATAFLOW_NETWORK_SEMANTICS,
        evaluate=_selection_evaluator(inventory.design_selection, networks),
    )
    network_validation = scope.derived(
        "op.network_validation",
        NETWORK_VALIDATION_REPORT_SEMANTICS,
        dependencies={"network": result_ref},
        evaluate=lambda network: validate_network(cast(DataflowNetwork, network)),
    )
    scope.constraint(
        "op.network_structurally_well_formed",
        dependencies={"report": network_validation},
        evaluate=lambda report: not cast(NetworkValidationReport, report),
    )

    external: dict[int, Ref[object]] = dict(supplied_refs)
    if inventory.design_selection is not None:
        external[id(closed.choice)] = cast("Ref[object]", inventory.design_selection)
    for item in getattr(owner, "persistence", ()):
        if not isinstance(item, Persist) or not isinstance(item.decision, DesignExport):
            continue
        selected_declaration = by_use.get(id(item.decision.use))
        if selected_declaration is None:
            raise AuthoringError("a persisted Design export belongs to an unregistered Design use")
        exported = exported_ref(item.decision.use, item.decision.member_path)
        if exported is None:
            raise AuthoringError(
                f"{selected_declaration.owner.__name__} does not export "
                f"{item.decision.member_path!r}"
            )
        external[id(item.decision)] = exported
    return inventory, result_ref, association_ref, MappingProxyType(external)


def _lookup_path(value: object, member_path: str) -> object:
    current = value
    for name in member_path.split("."):
        current = getattr(current, name)
    return current


def _compile_input_supplies(
    owner: type[object],
    declarations: CompiledClassDeclarations,
) -> tuple[tuple[InputSupplyDeclaration, ...], Mapping[int, Ref[object]]]:
    uses = tuple(
        value
        for declaring_class in reversed(owner.__mro__)
        for value in declaring_class.__dict__.values()
        if isinstance(value, UsesInputSupply)
    )
    compiled_supplies: list[InputSupplyDeclaration] = []
    refs: dict[int, Ref[object]] = {}
    for use in uses:
        inputs = _resolve_template_value(use.inputs, declarations, refs)
        result = use.factory(inputs)
        declaration = result.declaration if hasattr(result, "declaration") else result
        if not isinstance(declaration, InputSupplyDeclaration):
            raise AuthoringError("an input-supply factory returned an invalid declaration")
        compiled_supplies.append(declaration)
        for item in getattr(owner, "persistence", ()):
            if not isinstance(item, Persist) or not isinstance(item.decision, SupplyExport):
                continue
            if item.decision.use is not use:
                continue
            exported = _lookup_path(result, item.decision.member_path)
            if not isinstance(exported, Ref):
                raise AuthoringError(
                    f"input supply does not export Ref {item.decision.member_path!r}"
                )
            refs[id(item.decision)] = exported
        refs[id(use.choice)] = cast("Ref[object]", declaration.choice)
    return tuple(compiled_supplies), MappingProxyType(refs)


def compile_dataflow_operation(owner: type[object]) -> CompiledDataflowOperation:
    """Compile one direct ``DataflowOp`` class without mutating it."""

    namespace = _string_setting(owner, "declaration_namespace")
    declarations = compile_class_declarations(
        owner,
        layer=DeclarationLayer.OP,
        namespace=namespace,
    )
    scope = declarations.scope
    if not isinstance(scope, OpDesign):
        raise AssertionError("operation declarations must compile through OpDesign")
    persistence = getattr(owner, "persistence", ())
    if not isinstance(persistence, tuple) or not all(
        isinstance(item, Persist) for item in persistence
    ):
        raise AuthoringError(f"{owner.__name__}.persistence must be a tuple of Persist values")

    supplies, supplied_refs = _compile_input_supplies(owner, declarations)
    inventory, generated_result, generated_association, external = _compile_designs(
        owner, declarations, supplies, supplied_refs
    )
    if inventory is None:
        result_member = getattr(owner, "result_member", "result")
        association_member = getattr(owner, "source_association_member", "source_association")
        if not isinstance(result_member, str) or not isinstance(association_member, str):
            raise AuthoringError("operation result member names must be strings")
        result = declarations.ref(result_member)
        association = declarations.ref(association_member)
        specification = scope.spec()
        selection_constraints = getattr(owner, "selection_constraints", None)
        structural_readiness = getattr(owner, "structural_readiness", None)
        artifact_readiness = getattr(owner, "artifact_readiness", None)
        feasibility_constraints = tuple(getattr(owner, "feasibility_constraints", ()))
    else:
        assert generated_result is not None and generated_association is not None
        selection_constraints = (
            getattr(owner, "selection_constraints", None) or f"{namespace}.selection"
        )
        structural_constraint_set = (
            getattr(owner, "structural_constraint_set", None)
            or f"{namespace}.structural_constraints"
        )
        structural_readiness = (
            getattr(owner, "structural_readiness", None) or f"{namespace}.structural"
        )
        artifact_readiness = getattr(owner, "artifact_readiness", None) or f"{namespace}.artifact"
        authored = declare_dataflow_op_authoring(
            inventory,
            scope,
            result=cast("Ref[DataflowNetwork]", generated_result),
            source_association=generated_association,
            structural_properties=(
                generated_association,
                generated_result,
                scope.handle("op.network_validation", NetworkValidationReport),
            ),
            structural_constraints=scope.constraint_handles,
            structural_constraint_set=structural_constraint_set,
            feasibility_constraint_set=selection_constraints,
            structural_readiness_profile=structural_readiness,
            artifact_readiness_profile=artifact_readiness,
            additional_structural_decisions=scope.decision_handles,
        )
        result = generated_result
        association = generated_association
        specification = authored.specification
        feasibility_constraints = authored.feasibility_constraint_sets

    compiled_persistence = compile_persistence(
        declarations,
        persistence,
        external=external,
    )
    return CompiledDataflowOperation(
        owner=owner,
        namespace=namespace,
        declarations=declarations,
        specification=specification,
        problem_provenance=scope.provenance(),
        projection=ProjectionPlan.compile(declarations),
        result=result,
        source_association=association,
        persistence=compiled_persistence,
        selection_constraint_set=selection_constraints,
        structural_readiness_profile=structural_readiness,
        artifact_readiness_profile=artifact_readiness,
        feasibility_constraint_sets=feasibility_constraints,
        inventory=inventory,
    )


__all__ = [
    "BoundTensor",
    "ClosedDesigns",
    "CompiledDataflowOperation",
    "DesignChoice",
    "DesignExport",
    "UsesDesign",
    "UsesInputSupply",
    "compile_dataflow_operation",
]
