# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic ONNX/QONNX and build projection for class-authored operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np  # type: ignore[import-not-found]

from finn.dataflow.authoring.declarations import (
    CompiledClassDeclarations,
    Condition,
    DeclarationGroup,
    DeclarationLayer,
    DeclarationTemplate,
    Problem,
    collect_class_declarations,
)
from finn.dataflow.authoring.provenance import Provenance
from finn.dataflow.authoring.scope import Ref, semantics_for
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.design import (
    Finding,
    FindingKind,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    QualifiedPath,
)
from finn.dataflow.op_contracts import NodeAttributeType


class GraphModelView(Protocol):
    def get_tensor_shape(self, tensor_name: str) -> list[int] | None: ...

    def get_tensor_datatype(self, tensor_name: str) -> object: ...

    def get_initializer(self, tensor_name: str) -> object | None: ...


@dataclass(frozen=True, slots=True)
class TensorShape:
    """Required concrete tensor-shape contract."""

    rank: int | None = None
    min_rank: int | None = None

    def __post_init__(self) -> None:
        if self.rank is not None and self.rank < 0:
            raise ValueError("tensor rank must not be negative")
        if self.min_rank is not None and self.min_rank < 0:
            raise ValueError("minimum tensor rank must not be negative")
        if self.rank is not None and self.min_rank is not None and self.rank < self.min_rank:
            raise ValueError("fixed tensor rank cannot be smaller than its minimum rank")

    def accepts(self, shape: tuple[int, ...]) -> bool:
        return (
            (self.rank is None or len(shape) == self.rank)
            and (self.min_rank is None or len(shape) >= self.min_rank)
            and all(type(extent) is int and extent > 0 for extent in shape)
        )


@dataclass(frozen=True, slots=True)
class InitializerPolicy:
    required: bool | None
    fingerprint: bool = False


def NoInitializer() -> InitializerPolicy:
    return InitializerPolicy(False)


def OptionalInitializer(*, fingerprint: bool = False) -> InitializerPolicy:
    return InitializerPolicy(None, fingerprint)


def RequiredInitializer(*, fingerprint: bool = False) -> InitializerPolicy:
    return InitializerPolicy(True, fingerprint)


def _attribute_kind(value_type: type[object]) -> str:
    if value_type is bool or value_type is int:
        return "i"
    if value_type is float:
        return "f"
    if value_type is str or issubclass(value_type, Enum):
        return "s"
    raise TypeError(f"unsupported source attribute type {value_type.__name__}")


@dataclass(frozen=True, slots=True, eq=False)
class Attribute(Problem[Any]):
    """One QONNX node attribute projected into the operation problem."""

    attribute_name: str = ""
    default: object = None
    allowed_values: frozenset[object] | None = None
    required_on_node: bool = False

    def __init__(
        self,
        attribute_name: str,
        value_type: type[Any],
        *,
        default: object = None,
        required: bool = False,
        allowed_values: frozenset[object] | set[object] | None = None,
        stable_name: str | None = None,
    ) -> None:
        Problem.__init__(
            self,
            value_type,
            provenance=Provenance.GRAPH,
            stable_name=stable_name,
            required=True,
        )
        if not attribute_name:
            raise ValueError("a source attribute name must not be empty")
        if required and default is not None:
            raise ValueError("a required source attribute cannot also have a default")
        _attribute_kind(cast("type[object]", value_type))
        object.__setattr__(self, "attribute_name", attribute_name)
        object.__setattr__(self, "default", default)
        object.__setattr__(
            self,
            "allowed_values",
            None if allowed_values is None else frozenset(allowed_values),
        )
        object.__setattr__(self, "required_on_node", required)

    @property
    def nodeattr_definition(self) -> NodeAttributeType:
        value_type = cast("type[object]", self.value_type)
        allowed = self.allowed_values
        if value_type is bool:
            default = int(bool(self.default))
            encoded_allowed: set[object] = (
                {0, 1} if allowed is None else {int(bool(item)) for item in allowed}
            )
            return ("i", self.required_on_node, default, encoded_allowed)
        if isinstance(value_type, type) and issubclass(value_type, Enum):
            encoded_default = self.default.value if isinstance(self.default, Enum) else self.default
            enum_allowed: set[object] | None = (
                {cast(Enum, item).value for item in allowed} if allowed is not None else None
            )
            return ("s", self.required_on_node, cast(str, encoded_default or ""), enum_allowed)
        kind = _attribute_kind(value_type)
        fallback: int | float | str | bool
        if self.default is None:
            fallback = cast("int | float | str | bool", {"i": 0, "f": 0.0, "s": ""}[kind])
        else:
            fallback = cast("int | float | str | bool", self.default)
        return (
            kind,
            self.required_on_node,
            fallback,
            None if allowed is None else set(allowed),
        )

    def decode(self, value: object) -> object:
        value_type = cast("type[object]", self.value_type)
        if value_type is bool and type(value) is int and value in {0, 1}:
            return bool(value)
        if isinstance(value_type, type) and issubclass(value_type, Enum):
            if isinstance(value, str):
                for member in value_type:
                    if member.value == value or member.name == value:
                        return member
        if type(value) is value_type:
            return value
        raise ValueError(f"attribute {self.attribute_name!r} is not a valid {value_type.__name__}")


@dataclass(frozen=True, slots=True, eq=False)
class DatatypeAttribute(Problem[QONNXDataType]):
    """Legacy/source datatype spelling normalized to one QONNX datatype value."""

    attribute_name: str = ""
    default: str = ""
    required_on_node: bool = False

    def __init__(
        self,
        attribute_name: str,
        *,
        default: str,
        required: bool = False,
        stable_name: str | None = None,
    ) -> None:
        Problem.__init__(
            self,
            QONNX_DATATYPE_VALUE_SEMANTICS,
            provenance=Provenance.GRAPH,
            stable_name=stable_name,
        )
        object.__setattr__(self, "attribute_name", attribute_name)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "required_on_node", required)

    @property
    def nodeattr_definition(self) -> NodeAttributeType:
        return ("s", self.required_on_node, self.default, None)

    @staticmethod
    def decode(value: object) -> QONNXDataType:
        if not isinstance(value, str):
            raise ValueError("datatype attributes must be strings")
        return resolve_qonnx_datatype_name(value)


@dataclass(frozen=True, slots=True, eq=False)
class BuildFact(Problem[Any]):
    """One target/build field projected from a restricted configuration view."""

    key: str = ""
    default: object = None
    accessor: Callable[[object], object] | None = None

    def __init__(
        self,
        key: str,
        value_type: type[Any] | Any,
        *,
        provenance: Provenance = Provenance.BUILD,
        default: object = None,
        required: bool = True,
        accessor: Callable[[object], object] | None = None,
        stable_name: str | None = None,
        path: QualifiedPath | str | None = None,
    ) -> None:
        Problem.__init__(
            self,
            value_type,
            provenance=provenance,
            stable_name=stable_name,
            required=required,
            path=path,
        )
        if not key:
            raise ValueError("a build fact key must not be empty")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "accessor", accessor)

    def read(self, config: object) -> object:
        if self.accessor is not None:
            return self.accessor(config)
        return getattr(config, self.key, self.default)


@dataclass(frozen=True, slots=True, eq=False)
class InitializerAnalysis(Problem[Any]):
    """One pure graph analysis over a declared tensor initializer."""

    tensor: TensorOperand | None = None
    evaluate: Callable[[object, object], object] = lambda _initializer, _datatype: None

    def __init__(
        self,
        tensor: TensorOperand,
        value_type: type[Any],
        *,
        evaluate: Callable[[object, object], object],
        required: bool = False,
        stable_name: str | None = None,
    ) -> None:
        Problem.__init__(
            self,
            value_type,
            provenance=Provenance.GRAPH_ANALYSIS,
            stable_name=stable_name,
            required=required,
        )
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "evaluate", evaluate)


def TargetFpgaPart(*, required: bool = False) -> BuildFact:
    def read(config: object) -> object:
        resolver = getattr(config, "_resolve_fpga_part", None)
        if callable(resolver):
            return resolver()
        return getattr(config, "fpga_part", None)

    return BuildFact(
        "fpga_part",
        str,
        provenance=Provenance.TARGET,
        required=required,
        accessor=read,
        stable_name="fpga_part",
    )


def TargetClockPeriod(*, required: bool = True) -> BuildFact:
    return BuildFact(
        "synth_clk_period_ns",
        float,
        provenance=Provenance.TARGET,
        required=required,
        stable_name="clock_period_ns",
    )


def BuildFlag(key: str, *, default: bool = False) -> BuildFact:
    return BuildFact(key, bool, default=default)


def BuildString(key: str, *, default: str = "", required: bool = True) -> BuildFact:
    return BuildFact(key, str, default=default, required=required)


def SourceScope(*, stable_name: str = "source_scope_id") -> Problem[str]:
    return Problem(str, provenance=Provenance.GRAPH, stable_name=stable_name)


@dataclass(frozen=True, slots=True, eq=False)
class TensorOperand(DeclarationGroup):
    """One input or output tensor and its compiler-owned projected facts."""

    role: str
    index: int
    direction: str
    shape_contract: TensorShape
    initializer_policy: InitializerPolicy | None
    when: DeclarationTemplate[bool] | Condition | None
    present: Problem[bool]
    tensor_id: Problem[str]
    shape: Problem[tuple[Any, ...]]
    datatype: Problem[object]
    initializer_present: Problem[bool]
    initializer_fingerprint: Problem[str]
    layers = frozenset({DeclarationLayer.OP})

    def __init__(
        self,
        role: str,
        *,
        index: int,
        direction: str,
        shape: TensorShape,
        initializer: InitializerPolicy | None,
        when: DeclarationTemplate[bool] | Condition | None,
    ) -> None:
        if not role or index < 0 or direction not in {"input", "output"}:
            raise ValueError("a tensor operand needs a role, non-negative index, and direction")
        conditional = when is not None
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "shape_contract", shape)
        object.__setattr__(self, "initializer_policy", initializer)
        object.__setattr__(self, "when", when)
        object.__setattr__(
            self,
            "present",
            Problem(bool, provenance=Provenance.GRAPH, stable_name=f"{role}.present"),
        )
        object.__setattr__(
            self,
            "tensor_id",
            Problem(
                str,
                provenance=Provenance.GRAPH,
                stable_name=f"{role}.tensor_id",
                required=not conditional,
            ),
        )
        object.__setattr__(
            self,
            "shape",
            Problem(
                tuple,
                provenance=Provenance.GRAPH,
                stable_name=f"{role}.shape",
                required=not conditional,
            ),
        )
        object.__setattr__(
            self,
            "datatype",
            Problem(
                QONNX_DATATYPE_VALUE_SEMANTICS,
                provenance=Provenance.GRAPH,
                stable_name=f"{role}.datatype",
                required=not conditional,
            ),
        )
        object.__setattr__(
            self,
            "initializer_present",
            Problem(
                bool,
                provenance=Provenance.GRAPH,
                stable_name=f"{role}.initializer.present",
                required=not conditional,
            ),
        )
        object.__setattr__(
            self,
            "initializer_fingerprint",
            Problem(
                str,
                provenance=Provenance.GRAPH,
                stable_name=f"{role}.initializer.fingerprint",
                required=False,
            ),
        )

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        return (
            (f"{member_name}.present", self.present),
            (f"{member_name}.tensor_id", self.tensor_id),
            (f"{member_name}.shape", self.shape),
            (f"{member_name}.datatype", self.datatype),
            (f"{member_name}.initializer_present", self.initializer_present),
            (f"{member_name}.initializer_fingerprint", self.initializer_fingerprint),
        )


def InputTensor(
    role: str,
    *,
    index: int,
    shape: TensorShape,
    initializer: InitializerPolicy,
    when: DeclarationTemplate[bool] | Condition | None = None,
) -> TensorOperand:
    return TensorOperand(
        role,
        index=index,
        direction="input",
        shape=shape,
        initializer=initializer,
        when=when,
    )


def OutputTensor(
    role: str,
    *,
    index: int,
    shape: TensorShape,
    when: DeclarationTemplate[bool] | Condition | None = None,
) -> TensorOperand:
    return TensorOperand(
        role,
        index=index,
        direction="output",
        shape=shape,
        initializer=None,
        when=when,
    )


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    values: Mapping[QualifiedPath, object]
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectionPlan:
    attributes: tuple[tuple[str, Attribute | DatatypeAttribute, Ref[object]], ...]
    tensors: tuple[tuple[str, TensorOperand, Mapping[str, Ref[object]]], ...]
    graph_facts: tuple[tuple[str, Problem[Any], Ref[object]], ...]
    initializer_analyses: tuple[tuple[str, InitializerAnalysis, Ref[object]], ...]
    build_facts: tuple[tuple[str, BuildFact, Ref[object]], ...]

    @classmethod
    def compile(cls, declarations: CompiledClassDeclarations) -> ProjectionPlan:
        attributes: list[tuple[str, Attribute | DatatypeAttribute, Ref[object]]] = []
        graph_facts: list[tuple[str, Problem[Any], Ref[object]]] = []
        initializer_analyses: list[tuple[str, InitializerAnalysis, Ref[object]]] = []
        build_facts: list[tuple[str, BuildFact, Ref[object]]] = []
        for item in declarations.members:
            value = declarations.members[item]
            if not isinstance(value, Ref):
                continue
            template = next(
                entry.template
                for entry in declarations_for(declarations.owner)
                if entry.member_name == item
            )
            if isinstance(template, (Attribute, DatatypeAttribute)):
                attributes.append((item, template, value))
            elif isinstance(template, BuildFact):
                build_facts.append((item, template, value))
            elif isinstance(template, InitializerAnalysis):
                initializer_analyses.append((item, template, value))
            elif isinstance(template, Problem) and template.provenance in {
                Provenance.GRAPH,
                Provenance.GRAPH_ANALYSIS,
            }:
                graph_facts.append((item, template, value))
        tensors: list[tuple[str, TensorOperand, Mapping[str, Ref[object]]]] = []
        for name, group in declarations.groups.items():
            if not isinstance(group, TensorOperand):
                continue
            refs = {
                suffix: declarations.ref(f"{name}.{suffix}")
                for suffix in (
                    "present",
                    "tensor_id",
                    "shape",
                    "datatype",
                    "initializer_present",
                    "initializer_fingerprint",
                )
            }
            tensors.append((name, group, MappingProxyType(refs)))
        tensor_paths = {ref.path for _name, _group, refs in tensors for ref in refs.values()}
        graph_facts = [item for item in graph_facts if item[2].path not in tensor_paths]
        return cls(
            tuple(attributes),
            tuple(tensors),
            tuple(graph_facts),
            tuple(initializer_analyses),
            tuple(build_facts),
        )

    @property
    def source_nodeattrs(self) -> Mapping[str, NodeAttributeType]:
        return MappingProxyType(
            {
                template.attribute_name: template.nodeattr_definition
                for _, template, _ in self.attributes
            }
        )


def declarations_for(owner: type[object]) -> tuple[Any, ...]:
    return collect_class_declarations(owner, DeclarationLayer.OP)


def _condition_value(
    condition: DeclarationTemplate[bool] | Condition | None,
    values: Mapping[int, object],
) -> bool:
    if condition is None:
        return True
    if isinstance(condition, DeclarationTemplate):
        return bool(values[id(condition)])
    return bool(condition.evaluate(*(values[id(item)] for item in condition.dependencies)))


def _fingerprint_initializer(value: object) -> str:
    array = np.asarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finding(path: QualifiedPath, code: str, message: str, **values: object) -> Finding:
    return Finding(FindingKind.REJECTION, code, path, message, tuple(values.items()))


def project_graph(
    plan: ProjectionPlan,
    operation: object,
    model: GraphModelView,
    node: object,
    source_scope_id: str,
) -> ProjectionResult:
    """Execute a compiled graph projection without exposing ``ModelWrapper`` to an Op."""

    projected: dict[QualifiedPath, object] = {}
    template_values: dict[int, object] = {}
    findings: list[Finding] = []
    for _name, template, ref in plan.attributes:
        try:
            raw = getattr(operation, "get_nodeattr")(template.attribute_name)
            value = template.decode(raw)
            projected[ref.path] = value
            template_values[id(template)] = value
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            findings.append(
                _finding(
                    ref.path,
                    "dataflow-source-attribute-invalid",
                    str(exc),
                    attribute=template.attribute_name,
                )
            )

    for name, graph_template, ref in plan.graph_facts:
        if name == "source_scope_id":
            projected[ref.path] = source_scope_id
            template_values[id(graph_template)] = source_scope_id

    inputs = cast(Any, node).input
    outputs = cast(Any, node).output
    for _name, tensor, refs in plan.tensors:
        try:
            active = _condition_value(tensor.when, template_values)
        except KeyError:
            # A prerequisite already contributed its own precise finding.
            # Do not invent a dependent presence value or duplicate the error.
            continue
        projected[refs["present"].path] = active
        template_values[id(tensor.present)] = active
        names = inputs if tensor.direction == "input" else outputs
        has_slot = tensor.index < len(names) and bool(names[tensor.index])
        if not active:
            if has_slot:
                findings.append(
                    _finding(
                        refs["tensor_id"].path,
                        "dataflow-source-conditional-operand-present",
                        f"inactive {tensor.direction} operand {tensor.role!r} is present",
                    )
                )
            continue
        if not has_slot:
            findings.append(
                _finding(
                    refs["tensor_id"].path,
                    "dataflow-source-operand-missing",
                    f"required {tensor.direction} operand {tensor.role!r} is absent",
                )
            )
            continue
        tensor_id = cast(str, names[tensor.index])
        projected[refs["tensor_id"].path] = tensor_id
        template_values[id(tensor.tensor_id)] = tensor_id
        raw_shape = model.get_tensor_shape(tensor_id)
        shape = None if raw_shape is None else tuple(int(item) for item in raw_shape)
        if shape is None or not tensor.shape_contract.accepts(shape):
            findings.append(
                _finding(
                    refs["shape"].path,
                    "dataflow-source-shape-invalid",
                    f"operand {tensor.role!r} has no accepted concrete positive shape",
                    shape=shape,
                )
            )
        else:
            projected[refs["shape"].path] = shape
            template_values[id(tensor.shape)] = shape
        try:
            datatype = canonical_qonnx_datatype(model.get_tensor_datatype(tensor_id))
        except (TypeError, ValueError) as exc:
            findings.append(
                _finding(
                    refs["datatype"].path,
                    "dataflow-source-datatype-invalid",
                    str(exc),
                )
            )
        else:
            projected[refs["datatype"].path] = datatype
            template_values[id(tensor.datatype)] = datatype
        initializer = model.get_initializer(tensor_id) if tensor.direction == "input" else None
        initialized = initializer is not None
        projected[refs["initializer_present"].path] = initialized
        template_values[id(tensor.initializer_present)] = initialized
        policy = tensor.initializer_policy
        if policy is not None:
            if policy.required is True and not initialized:
                findings.append(
                    _finding(
                        refs["initializer_present"].path,
                        "dataflow-source-initializer-required",
                        f"operand {tensor.role!r} requires an initializer",
                    )
                )
            elif policy.required is False and initialized:
                findings.append(
                    _finding(
                        refs["initializer_present"].path,
                        "dataflow-source-initializer-forbidden",
                        f"operand {tensor.role!r} must not have an initializer",
                    )
                )
            if initialized and policy.fingerprint:
                fingerprint = _fingerprint_initializer(initializer)
                projected[refs["initializer_fingerprint"].path] = fingerprint
                template_values[id(tensor.initializer_fingerprint)] = fingerprint

    for _name, analysis, ref in plan.initializer_analyses:
        analysis_tensor = analysis.tensor
        if analysis_tensor is None:
            raise AssertionError("initializer analysis has no tensor declaration")
        tensor_name = template_values.get(id(analysis_tensor.tensor_id))
        analysis_datatype = template_values.get(id(analysis_tensor.datatype))
        if not isinstance(tensor_name, str) or analysis_datatype is None:
            if analysis.required:
                findings.append(
                    _finding(
                        ref.path,
                        "dataflow-source-analysis-input-unavailable",
                        "initializer analysis inputs are unavailable",
                    )
                )
            continue
        initializer = model.get_initializer(tensor_name)
        if initializer is None:
            if analysis.required:
                findings.append(
                    _finding(
                        ref.path,
                        "dataflow-source-analysis-initializer-unavailable",
                        "initializer analysis requires a concrete initializer",
                        tensor=tensor_name,
                    )
                )
            continue
        try:
            analyzed = analysis.evaluate(initializer, analysis_datatype)
            if analyzed is None:
                if analysis.required:
                    raise ValueError("initializer analysis did not produce a value")
                continue
            if not semantics_for(analysis.value_type).accepts(analyzed):
                raise TypeError("initializer analysis produced an incompatible value")
            projected[ref.path] = analyzed
            template_values[id(analysis)] = analyzed
        except (TypeError, ValueError) as exc:
            findings.append(_finding(ref.path, "dataflow-source-analysis-failed", str(exc)))

    return ProjectionResult(MappingProxyType(projected), tuple(findings))


def project_build(plan: ProjectionPlan, config: object) -> ProjectionResult:
    projected: dict[QualifiedPath, object] = {}
    findings: list[Finding] = []
    for _name, template, ref in plan.build_facts:
        try:
            value = template.read(config)
            if value is None:
                if template.required:
                    raise ValueError(f"required build fact {template.key!r} is unavailable")
                continue
            expected = semantics_for(template.value_type)
            if not expected.accepts(value):
                raise TypeError(
                    f"build fact {template.key!r} is {type(value).__name__}, not {expected.name}"
                )
            projected[ref.path] = value
        except (AttributeError, TypeError, ValueError) as exc:
            findings.append(
                _finding(ref.path, "dataflow-build-fact-invalid", str(exc), key=template.key)
            )
    return ProjectionResult(MappingProxyType(projected), tuple(findings))


__all__ = [
    "Attribute",
    "BuildFact",
    "BuildFlag",
    "BuildString",
    "DatatypeAttribute",
    "GraphModelView",
    "InitializerPolicy",
    "InitializerAnalysis",
    "InputTensor",
    "NoInitializer",
    "OptionalInitializer",
    "OutputTensor",
    "ProjectionPlan",
    "ProjectionResult",
    "RequiredInitializer",
    "SourceScope",
    "TargetClockPeriod",
    "TargetFpgaPart",
    "TensorOperand",
    "TensorShape",
    "project_build",
    "project_graph",
]
