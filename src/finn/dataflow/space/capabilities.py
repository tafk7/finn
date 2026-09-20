# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed capability vocabulary and generic assessed-output binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeAlias, TypeVar, cast

from finn.dataflow._engine import (
    AbsenceMode,
    Absent,
    Answer,
    Decided,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.space.compiler import _CompiledSpace, _Ref
from finn.dataflow.space.declarations import AuthoringError, Projection, Space, semantics_for


View: TypeAlias = Projection
S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class AssessedCapabilityOutput:
    """Bind one selected output to a candidate's assessed capability.

    Most outputs forward the capability's accepted value. Some companion values
    instead retain their own output and merely require that the named capability
    was accepted at the same point.
    """

    capability: str
    forwards_accepted_value: bool = True
    accepted_output_name: str | None = None

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("an assessed capability binding needs a capability name")
        if self.accepted_output_name == "":
            raise ValueError("an assessed capability output name must be non-empty")


def canonicalize_assessed_capability_outputs(
    compiled: _CompiledSpace[S],
    bindings: Mapping[str, Mapping[str, AssessedCapabilityOutput]],
) -> _CompiledSpace[S]:
    """Replace selected raw outputs with their assessed capability values.

    ``bindings`` is keyed first by the owning branch member and then by selected
    output name. The mechanism is layer-neutral: callers choose which output
    represents which compatible Projection and whether the accepted Projection
    value replaces or only gates that output.
    """

    properties = list(compiled.spec.properties)
    properties_by_path = {item.path: item for item in compiled.spec.properties}
    decisions_by_path = {item.path: item for item in compiled.spec.decisions}
    accepted: dict[tuple[str, str], _Ref[object]] = {}

    def accepted_view(case: object, view_name: str) -> _Ref[object]:
        compiled_case = cast(Any, case).compiled
        key = (compiled_case.namespace, view_name)
        if key in accepted:
            return accepted[key]
        try:
            view = compiled_case.projection(view_name)
        except AuthoringError:
            raise AuthoringError(
                f"{compiled_case.owner.__name__} does not declare consumed capability {view_name!r}"
            ) from None
        path = QualifiedPath(f"semantic.{compiled_case.namespace}.accepted-{view_name}")
        dependencies: list[DependencyRef] = [
            replace(view.output, absence=AbsenceMode.PRESERVES_ANSWER).dependency("output")
        ]
        readiness_names: list[str] = []
        acceptance_witnesses: list[tuple[str, QualifiedPath]] = []
        profiles = {item.name: item for item in compiled_case.spec.readiness_profiles}
        readiness = profiles[view.readiness_profile]
        for index, decision_path in enumerate(readiness.decisions):
            decision = decisions_by_path[decision_path]
            dependencies.append(
                DependencyRef(
                    f"ready_decision_{index}",
                    decision_path,
                    DependencyKind.DECISION,
                    decision.value_semantics,
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(f"ready_decision_{index}")
        for index, property_path in enumerate(readiness.properties):
            declaration = properties_by_path[property_path]
            dependencies.append(
                DependencyRef(
                    f"ready_property_{index}",
                    property_path,
                    DependencyKind.PROPERTY,
                    declaration.value_semantics,
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(f"ready_property_{index}")
        for index, constraint_path in enumerate(readiness.constraints):
            name = f"ready_constraint_{index}"
            dependencies.append(
                DependencyRef(
                    name,
                    constraint_path,
                    DependencyKind.CONSTRAINT,
                    semantics_for(bool),
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            readiness_names.append(name)
        groups = {item.name: item for item in compiled_case.spec.constraint_sets}
        acceptance_paths = tuple(
            dict.fromkeys(
                constraint_path
                for group_name in view.constraint_sets
                for constraint_path in groups[group_name].constraints
            )
        )
        for index, constraint_path in enumerate(acceptance_paths):
            name = f"accept_constraint_{index}"
            dependencies.append(
                DependencyRef(
                    name,
                    constraint_path,
                    DependencyKind.CONSTRAINT,
                    semantics_for(bool),
                    AbsenceMode.PRESERVES_ANSWER,
                )
            )
            acceptance_witnesses.append((name, constraint_path))

        def evaluate(values: DependencyView) -> Answer[object]:
            output = cast("Answer[object]", values["output"])
            readiness_answers = [cast("Answer[object]", values[name]) for name in readiness_names]
            acceptance_answers = [
                (cast("Answer[bool]", values[name]), constraint_path)
                for name, constraint_path in acceptance_witnesses
            ]
            unresolved = tuple(
                finding
                for answer in (
                    output,
                    *readiness_answers,
                    *(item[0] for item in acceptance_answers),
                )
                if isinstance(answer, Unresolved)
                for finding in answer.findings
            )
            if unresolved:
                return Unresolved(unresolved)
            if isinstance(output, Absent):
                return output
            refusals: list[Finding] = []
            for answer, constraint_path in acceptance_answers:
                if isinstance(answer, Absent):
                    if answer.is_rejection:
                        refusals.extend(answer.findings)
                    continue
                if isinstance(answer, Decided) and answer.value is False:
                    refusals.append(
                        Finding(
                            FindingKind.REJECTION,
                            "projection-constraint-refused",
                            path,
                            "a consumed capability constraint refused this point",
                            (("constraint", constraint_path),),
                            (constraint_path,),
                        )
                    )
            if refusals:
                return Absent(tuple(refusals))
            if not isinstance(output, Decided):
                raise AuthoringError("a consumed capability returned an invalid output answer")
            return Decided(output.value)

        properties.append(
            DerivedProperty(
                path,
                view.output.semantics,
                EvaluatorSpec(tuple(dependencies), evaluate),
                view.applicability,
            )
        )
        reference: _Ref[object] = _Ref(path, DependencyKind.PROPERTY, view.output.semantics)
        accepted[key] = reference
        return reference

    for member_name, branch in compiled.branches:
        output_bindings = bindings.get(member_name)
        if output_bindings is None:
            continue
        for output_name, selected_output in branch.outputs:
            binding = output_bindings.get(output_name)
            if binding is None:
                continue
            selected_property = properties_by_path[selected_output.path]
            replacements: dict[str, _Ref[object]] = {}
            for case in branch.cases:
                raw = case.compiled.exported(output_name)
                assessed = accepted_view(case, binding.capability)
                if binding.forwards_accepted_value:
                    if not raw.semantics.is_compatible_with(assessed.semantics):
                        raise AuthoringError(
                            f"{case.compiled.owner.__name__}.{binding.capability} output "
                            f"is incompatible with selected output {output_name!r}"
                        )
                    replacements[f"case@{case.case_id}"] = assessed
                    continue
                accepted_output_name = binding.accepted_output_name or output_name
                path = QualifiedPath(
                    f"semantic.{case.compiled.namespace}.accepted-{accepted_output_name}"
                )

                def gate(values: DependencyView) -> Answer[object]:
                    accepted_answer = cast("Answer[object]", values["capability"])
                    if not isinstance(accepted_answer, Decided):
                        return accepted_answer
                    return cast("Answer[object]", values["value"])

                properties.append(
                    DerivedProperty(
                        path,
                        raw.semantics,
                        EvaluatorSpec(
                            (
                                replace(assessed, absence=AbsenceMode.PRESERVES_ANSWER).dependency(
                                    "capability"
                                ),
                                replace(raw, absence=AbsenceMode.PRESERVES_ANSWER).dependency(
                                    "value"
                                ),
                            ),
                            gate,
                        ),
                    )
                )
                replacements[f"case@{case.case_id}"] = _Ref(
                    path, DependencyKind.PROPERTY, raw.semantics
                )
            dependencies = tuple(
                replace(replacements[item.name], absence=AbsenceMode.PRESERVES_ANSWER).dependency(
                    item.name
                )
                if item.name in replacements
                else item
                for item in selected_property.evaluator.dependencies
            )
            selector = branch.selector
            only_case = branch.cases[0].case_id

            def select(
                values: DependencyView,
                selector: object = selector,
                only_case: str = only_case,
            ) -> Answer[object]:
                if selector is not None:
                    chosen = cast(str, values["selector"])
                    return cast("Answer[object]", values[f"case@{chosen}"])
                return cast("Answer[object]", values[f"case@{only_case}"])

            replacement = replace(
                selected_property,
                evaluator=EvaluatorSpec(dependencies, select),
            )
            properties[properties.index(selected_property)] = replacement
            properties_by_path[selected_output.path] = replacement

    return replace(compiled, spec=replace(compiled.spec, properties=tuple(properties)))


__all__ = ["AssessedCapabilityOutput", "View", "canonicalize_assessed_capability_outputs"]
