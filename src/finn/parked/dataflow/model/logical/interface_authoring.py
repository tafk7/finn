# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Ordinary assessed Space projections for public operand facets and exports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Literal, cast

from finn.kernels._engine import Answer, Decided
from finn.parked.dataflow.model._authoring import GENERATED_MEMBERS, generated_member
from finn.dataflow.model.logical.composition import LogicalResult, logical_network
from finn.dataflow.datatypes import QONNXDataType, QONNX_DATATYPE_TOKEN
from finn.dataflow.model.logical.interface import (
    OperandExport,
    PublicOperand,
    validate_operand_export,
)
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.kernels.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Projection,
    Space,
    ValueSource,
    reject,
    semantics_for,
)


@dataclass(frozen=True)
class PublicOperandDeclaration:
    key: str
    direction: Literal["input", "output"]
    datatype: Projection[QONNXDataType]
    domain: Projection[RectangularDomain]
    export_builder: Callable[[PublicOperand, LogicalResult], OperandExport] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.key):
            raise AuthoringError("public operand keys must be stable identifier segments")
        if self.direction not in ("input", "output"):
            raise AuthoringError("public operand direction must be input or output")
        if (
            not isinstance(self.datatype, Projection)
            or self.datatype.output.value_semantics.type_token is not QONNX_DATATYPE_TOKEN
        ):
            raise AuthoringError("public datatype facets require a canonical datatype Projection")
        if (
            not isinstance(self.domain, Projection)
            or self.domain.output.value_semantics.type_token is not RectangularDomain
        ):
            raise AuthoringError("public domain facets require a RectangularDomain Projection")


def public_operand_declarations(space: Space | type[Space]) -> tuple[PublicOperandDeclaration, ...]:
    cls = space if isinstance(space, type) else type(space)
    declarations = tuple(getattr(cls, "public_operands", ()))
    if any(not isinstance(item, PublicOperandDeclaration) for item in declarations):
        raise AuthoringError("public_operands must contain PublicOperandDeclaration values")
    if len({item.key for item in declarations}) != len(declarations):
        raise AuthoringError("public operand keys must be unique")
    return declarations


def operand_declaration(space: Space, key: str) -> PublicOperandDeclaration:
    for declaration in public_operand_declarations(space):
        if declaration.key == key:
            return declaration
    raise AuthoringError(f"{type(space).__name__} exports no public operand {key!r}")


def operand_type(space: Space, key: str) -> Answer[QONNXDataType]:
    return space.assess_view(operand_declaration(space, key).datatype).accepted_answer


def operand_domain(space: Space, key: str) -> Answer[RectangularDomain]:
    return space.assess_view(operand_declaration(space, key).domain).accepted_answer


def operand_export(space: Space, key: str) -> Answer[OperandExport]:
    declaration = operand_declaration(space, key)
    if declaration.export_builder is None:
        raise AuthoringError("this family interface declares facets, not a logical body")
    logical: Answer[LogicalResult] = space.assess_view("logical").accepted_answer
    if not isinstance(logical, Decided):
        return cast("Answer[OperandExport]", logical)
    return cast("Answer[OperandExport]", space.answer(getattr(type(space), f"public_{key}_export")))


def _export_evaluator(declaration: PublicOperandDeclaration) -> Callable[..., object]:
    def build(*, datatype: QONNXDataType, domain: RectangularDomain, body: LogicalResult) -> object:
        assert declaration.export_builder is not None
        try:
            public = PublicOperand(declaration.key, declaration.direction, datatype, domain)
            export = declaration.export_builder(public, body)
            if export.operand != public:
                raise ValueError("export builder changed its shared public operand")
            validate_operand_export(logical_network(body), export)
            return export
        except (KeyError, ValueError) as error:
            return reject("kernel-public-export", str(error))

    return build


def _export_available(*, export: OperandExport) -> bool:
    return isinstance(export, OperandExport)


def _fact_available(*, value: object) -> bool:
    del value
    return True


def _facet_applicable(*, value: bool) -> object:
    if value:
        return True
    return reject("kernel-public-inapplicable", "a required public operand facet is inapplicable")


def attach_public_interface(kernel_type: type[Space]) -> None:
    """Add export obligations to full logical acceptance, sharing facet sources."""
    declarations = public_operand_declarations(kernel_type)
    logical = getattr(kernel_type, "logical", None)
    if not isinstance(logical, Projection):
        return
    generated = set(
        cast("frozenset[str]", kernel_type.__dict__.get(GENERATED_MEMBERS, frozenset()))
    )
    groups: list[ConstraintGroup] = []
    for declaration in declarations:
        if declaration.export_builder is None:
            raise AuthoringError(
                "a full logical capability must bind every public operand to its body"
            )

        value: Derived[OperandExport] = Derived(
            semantics_for(OperandExport),
            None,
            (
                ("datatype", cast("ValueSource[object]", declaration.datatype.output)),
                ("domain", cast("ValueSource[object]", declaration.domain.output)),
                ("body", cast("ValueSource[object]", logical.output)),
            ),
            _export_evaluator(declaration),
        )
        prefix = f"public_{declaration.key}"
        generated_member(kernel_type, f"{prefix}_export", value, generated)
        valid = Constraint((("export", value),), _export_available)
        generated_member(kernel_type, f"{prefix}_valid", valid, generated)
        group = ConstraintGroup(valid, name=f"{prefix}_accepts")
        generated_member(kernel_type, f"{prefix}_accepts", group, generated)
        groups.extend((*declaration.datatype.constraints, *declaration.domain.constraints, group))
        for kind, facet in (("type", declaration.datatype), ("domain", declaration.domain)):
            requirements = list(facet.readiness.constraints)
            for index, source in enumerate(
                (*facet.readiness.properties, *facet.readiness.decisions)
            ):
                ready = Constraint((("value", source),), _fact_available)
                generated_member(kernel_type, f"{prefix}_{kind}_ready_{index}", ready, generated)
                requirements.append(ready)
            if facet.applicable_if is not None:
                applies = Constraint((("value", facet.applicable_if),), _facet_applicable)
                generated_member(kernel_type, f"{prefix}_{kind}_applicable", applies, generated)
                requirements.append(applies)
            if requirements:
                required = ConstraintGroup(*requirements, name=f"{prefix}_{kind}_requirements")
                generated_member(kernel_type, f"{prefix}_{kind}_requirements", required, generated)
                groups.append(required)
    if groups:
        # Reuse the same logical value and readiness. Constraint availability and
        # refusal already propagate through the canonical Projection assessment.
        setattr(
            kernel_type,
            "logical",
            type(logical)(
                logical.output,
                applicable_if=logical.applicable_if,
                readiness=logical.readiness,
                constraints=(*logical.constraints, *groups),
                name=logical.stable_name,
            ),
        )
    setattr(kernel_type, GENERATED_MEMBERS, frozenset(generated))


__all__ = [
    "PublicOperandDeclaration",
    "public_operand_declarations",
    "operand_declaration",
    "operand_type",
    "operand_domain",
    "operand_export",
    "attach_public_interface",
]
