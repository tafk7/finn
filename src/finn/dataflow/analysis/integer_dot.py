# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact bounded integer analysis and execution for matrix-vector products.

This module deliberately imports no design-space, operation, Kernel, or
compiler type.  Source and implementation layers translate their facts into
these immutable values and consume the resulting witness; the arithmetic proof
has one implementation and no knowledge of the hierarchy around it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from math import prod
from typing import Any


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class IntegerRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _integer(self.minimum, "range minimum")
        _integer(self.maximum, "range maximum")
        if self.minimum > self.maximum:
            raise ValueError("an integer range minimum must not exceed its maximum")

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum

    def contains_range(self, other: IntegerRange) -> bool:
        return self.minimum <= other.minimum and other.maximum <= self.maximum


@dataclass(frozen=True, slots=True)
class IntegerType:
    """The exact integer-domain facts needed from one logical datatype."""

    name: str
    value_range: IntegerRange
    bit_width: int
    signed: bool

    def __post_init__(self) -> None:
        _nonempty(self.name, "integer type name")
        if not isinstance(self.value_range, IntegerRange):
            raise TypeError("integer type value_range must be IntegerRange")
        _integer(self.bit_width, "integer type bit width")
        if self.bit_width < 1:
            raise ValueError("integer type bit width must be positive")
        if type(self.signed) is not bool:
            raise TypeError("integer type signed must be bool")


@dataclass(frozen=True, slots=True)
class OperandIdentity:
    operand_id: str
    direction: str
    index: int

    def __post_init__(self) -> None:
        _nonempty(self.operand_id, "operand id")
        if self.direction not in {"input", "output"}:
            raise ValueError("operand direction must be 'input' or 'output'")
        _integer(self.index, "operand index")
        if self.index < 0:
            raise ValueError("operand index must be nonnegative")


@dataclass(frozen=True, slots=True)
class InvocationScope:
    """One stable operation occurrence authorized to consume a promise."""

    scope_id: str

    def __post_init__(self) -> None:
        _nonempty(self.scope_id, "invocation scope id")


@dataclass(frozen=True, slots=True)
class FixedWeightPremise:
    values: tuple[int, ...]
    content_digest: str
    source_carrier: str

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if any(type(value) is not int for value in values):
            raise TypeError("fixed weights must contain integers and never Boolean values")
        _nonempty(self.content_digest, "fixed-weight content digest")
        _nonempty(self.source_carrier, "fixed-weight source carrier")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class RuntimeWeightPromise:
    """A range promise with an explicit applicability domain.

    ``covered_invocations`` may name several occurrences.  That is the only
    broader scope in this profile: there is no wildcard whose applicability
    could be inferred from a similar-looking hash or name.
    """

    value_range: IntegerRange
    count: int
    source_operand: OperandIdentity
    covered_invocations: tuple[InvocationScope, ...]
    promise_id: str
    source_carrier: str

    def __post_init__(self) -> None:
        if not isinstance(self.value_range, IntegerRange):
            raise TypeError("runtime promise value_range must be IntegerRange")
        _integer(self.count, "runtime promise count")
        if self.count < 1:
            raise ValueError("runtime promise count must be positive")
        if not isinstance(self.source_operand, OperandIdentity):
            raise TypeError("runtime promise source_operand must be OperandIdentity")
        scopes = tuple(self.covered_invocations)
        if not scopes or any(not isinstance(scope, InvocationScope) for scope in scopes):
            raise TypeError("runtime promise must cover one or more InvocationScope values")
        if len(scopes) != len(set(scopes)):
            raise ValueError("runtime promise invocation scopes must be unique")
        _nonempty(self.promise_id, "runtime promise id")
        _nonempty(self.source_carrier, "runtime promise source carrier")
        object.__setattr__(
            self, "covered_invocations", tuple(sorted(scopes, key=lambda x: x.scope_id))
        )

    def covers(self, invocation: InvocationScope) -> bool:
        return invocation in self.covered_invocations


WeightPremise = FixedWeightPremise | RuntimeWeightPromise


@dataclass(frozen=True, slots=True)
class DotProductPremise:
    activation_range: IntegerRange
    activation_logical_type: IntegerType
    activation_source_carrier: str
    activation_shape: tuple[int, ...]
    activation_source: OperandIdentity
    weights: WeightPremise
    weight_logical_type: IntegerType
    weight_source_carrier: str
    weight_shape: tuple[int, ...]
    weight_source: OperandIdentity
    accumulator_type: IntegerType
    result_type: IntegerType
    output_shape: tuple[int, ...]
    invocation_scope: InvocationScope

    def __post_init__(self) -> None:
        if not isinstance(self.activation_range, IntegerRange):
            raise TypeError("activation_range must be IntegerRange")
        if not isinstance(self.activation_logical_type, IntegerType):
            raise TypeError("activation_logical_type must be IntegerType")
        if not isinstance(self.weight_logical_type, IntegerType):
            raise TypeError("weight_logical_type must be IntegerType")
        if not isinstance(self.accumulator_type, IntegerType):
            raise TypeError("accumulator_type must be IntegerType")
        if not isinstance(self.result_type, IntegerType):
            raise TypeError("result_type must be IntegerType")
        if not isinstance(self.activation_source, OperandIdentity):
            raise TypeError("activation_source must be OperandIdentity")
        if not isinstance(self.weight_source, OperandIdentity):
            raise TypeError("weight_source must be OperandIdentity")
        if not isinstance(self.invocation_scope, InvocationScope):
            raise TypeError("invocation_scope must be InvocationScope")
        for name, shape in (
            ("activation_shape", self.activation_shape),
            ("weight_shape", self.weight_shape),
            ("output_shape", self.output_shape),
        ):
            normalized = tuple(shape)
            if not normalized or any(
                type(extent) is not int or extent <= 0 for extent in normalized
            ):
                raise ValueError(f"{name} must contain positive integer extents")
            object.__setattr__(self, name, normalized)
        _nonempty(self.activation_source_carrier, "activation source carrier")
        _nonempty(self.weight_source_carrier, "weight source carrier")


@dataclass(frozen=True, slots=True)
class DotProductBounds:
    product: IntegerRange
    every_intermediate: IntegerRange
    result: IntegerRange
    output_ranges: tuple[IntegerRange, ...]
    minimum_signed_accumulator_bits: int


@dataclass(frozen=True, slots=True)
class NumericalFinding:
    code: str
    message: str
    values: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.code, "numerical finding code")
        _nonempty(self.message, "numerical finding message")
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True, slots=True)
class IntegerSupport:
    premise: DotProductPremise
    premise_fingerprint: str
    bounds: DotProductBounds
    operand_carrier: str
    matmul_result_carrier: str
    output_carrier: str
    internal_bits: int
    target_max_accumulator_bits: int


@dataclass(frozen=True, slots=True)
class IntegerSupportReport:
    support: IntegerSupport | None
    findings: tuple[NumericalFinding, ...]

    @property
    def supported(self) -> bool:
        return self.support is not None and not self.findings


@dataclass(frozen=True, slots=True)
class RuntimeIntegerTensor:
    values: Any
    carrier: str
    shape: tuple[int, ...]
    source: OperandIdentity
    invocation_scope: InvocationScope
    content_digest: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.carrier, "runtime tensor carrier")
        shape = tuple(self.shape)
        if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
            raise ValueError("runtime tensor shape must contain positive integer extents")
        if not isinstance(self.source, OperandIdentity):
            raise TypeError("runtime tensor source must be OperandIdentity")
        if not isinstance(self.invocation_scope, InvocationScope):
            raise TypeError("runtime tensor invocation_scope must be InvocationScope")
        if self.content_digest is not None:
            _nonempty(self.content_digest, "runtime tensor content digest")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class ValidatedDotProductOperands:
    activation: tuple[int, ...]
    weights: tuple[int, ...]
    activation_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    premise_fingerprint: str


def encode_dot_product_premise(premise: DotProductPremise) -> dict[str, object]:
    """Canonical JSON-shaped form of a complete detached numerical premise."""

    def integer_range(value: IntegerRange) -> list[int]:
        return [value.minimum, value.maximum]

    def integer_type(value: IntegerType) -> dict[str, object]:
        return {
            "name": value.name,
            "range": integer_range(value.value_range),
            "bit_width": value.bit_width,
            "signed": value.signed,
        }

    def operand(value: OperandIdentity) -> dict[str, object]:
        return {
            "operand_id": value.operand_id,
            "direction": value.direction,
            "index": value.index,
        }

    weights: dict[str, object]
    if isinstance(premise.weights, FixedWeightPremise):
        weights = {
            "kind": "fixed",
            "values": list(premise.weights.values),
            "content_digest": premise.weights.content_digest,
            "source_carrier": premise.weights.source_carrier,
        }
    else:
        weights = {
            "kind": "runtime",
            "value_range": integer_range(premise.weights.value_range),
            "count": premise.weights.count,
            "source_operand": operand(premise.weights.source_operand),
            "covered_invocations": [
                scope.scope_id for scope in premise.weights.covered_invocations
            ],
            "promise_id": premise.weights.promise_id,
            "source_carrier": premise.weights.source_carrier,
        }
    return {
        "activation_range": integer_range(premise.activation_range),
        "activation_logical_type": integer_type(premise.activation_logical_type),
        "activation_source_carrier": premise.activation_source_carrier,
        "activation_shape": list(premise.activation_shape),
        "activation_source": operand(premise.activation_source),
        "weights": weights,
        "weight_logical_type": integer_type(premise.weight_logical_type),
        "weight_source_carrier": premise.weight_source_carrier,
        "weight_shape": list(premise.weight_shape),
        "weight_source": operand(premise.weight_source),
        "accumulator_type": integer_type(premise.accumulator_type),
        "result_type": integer_type(premise.result_type),
        "output_shape": list(premise.output_shape),
        "invocation_scope": premise.invocation_scope.scope_id,
    }


def decode_dot_product_premise(value: object) -> DotProductPremise:
    """Decode the exact canonical premise form, refusing unknown fields."""

    from collections.abc import Mapping, Sequence  # noqa: PLC0415

    def mapping(raw: object, fields: set[str], name: str) -> Mapping[str, object]:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError(f"{name} has unsupported fields")
        if any(type(key) is not str for key in raw):
            raise TypeError(f"{name} field names must be strings")
        return raw

    def integer(raw: object, name: str) -> int:
        if type(raw) is not int:
            raise TypeError(f"{name} must be an integer")
        return raw

    def string(raw: object, name: str) -> str:
        if type(raw) is not str or not raw:
            raise TypeError(f"{name} must be a non-empty string")
        return raw

    def sequence(raw: object, name: str) -> Sequence[object]:
        if not isinstance(raw, (tuple, list)):
            raise TypeError(f"{name} must be a sequence")
        return raw

    def integer_range(raw: object, name: str) -> IntegerRange:
        items = sequence(raw, name)
        if len(items) != 2:
            raise ValueError(f"{name} must contain two endpoints")
        return IntegerRange(integer(items[0], name), integer(items[1], name))

    def integer_type(raw: object, name: str) -> IntegerType:
        item = mapping(raw, {"name", "range", "bit_width", "signed"}, name)
        signed = item["signed"]
        if type(signed) is not bool:
            raise TypeError(f"{name}.signed must be bool")
        return IntegerType(
            string(item["name"], f"{name}.name"),
            integer_range(item["range"], f"{name}.range"),
            integer(item["bit_width"], f"{name}.bit_width"),
            signed,
        )

    def operand(raw: object, name: str) -> OperandIdentity:
        item = mapping(raw, {"operand_id", "direction", "index"}, name)
        return OperandIdentity(
            string(item["operand_id"], f"{name}.operand_id"),
            string(item["direction"], f"{name}.direction"),
            integer(item["index"], f"{name}.index"),
        )

    root = mapping(
        value,
        {
            "activation_range",
            "activation_logical_type",
            "activation_source_carrier",
            "activation_shape",
            "activation_source",
            "weights",
            "weight_logical_type",
            "weight_source_carrier",
            "weight_shape",
            "weight_source",
            "accumulator_type",
            "result_type",
            "output_shape",
            "invocation_scope",
        },
        "dot-product premise",
    )
    raw_weights = root["weights"]
    if not isinstance(raw_weights, Mapping):
        raise TypeError("dot-product premise weights must be a mapping")
    kind = raw_weights.get("kind")
    if kind == "fixed":
        fixed = mapping(
            raw_weights,
            {"kind", "values", "content_digest", "source_carrier"},
            "fixed weights",
        )
        weights: WeightPremise = FixedWeightPremise(
            tuple(integer(item, "fixed weight") for item in sequence(fixed["values"], "values")),
            string(fixed["content_digest"], "fixed content digest"),
            string(fixed["source_carrier"], "fixed source carrier"),
        )
    elif kind == "runtime":
        runtime = mapping(
            raw_weights,
            {
                "kind",
                "value_range",
                "count",
                "source_operand",
                "covered_invocations",
                "promise_id",
                "source_carrier",
            },
            "runtime weights",
        )
        weights = RuntimeWeightPromise(
            integer_range(runtime["value_range"], "runtime value range"),
            integer(runtime["count"], "runtime count"),
            operand(runtime["source_operand"], "runtime source operand"),
            tuple(
                InvocationScope(string(item, "covered invocation"))
                for item in sequence(runtime["covered_invocations"], "covered invocations")
            ),
            string(runtime["promise_id"], "runtime promise id"),
            string(runtime["source_carrier"], "runtime source carrier"),
        )
    else:
        raise ValueError("dot-product premise has an unsupported weight kind")
    return DotProductPremise(
        integer_range(root["activation_range"], "activation range"),
        integer_type(root["activation_logical_type"], "activation logical type"),
        string(root["activation_source_carrier"], "activation source carrier"),
        tuple(
            integer(item, "activation shape")
            for item in sequence(root["activation_shape"], "activation shape")
        ),
        operand(root["activation_source"], "activation source"),
        weights,
        integer_type(root["weight_logical_type"], "weight logical type"),
        string(root["weight_source_carrier"], "weight source carrier"),
        tuple(
            integer(item, "weight shape") for item in sequence(root["weight_shape"], "weight shape")
        ),
        operand(root["weight_source"], "weight source"),
        integer_type(root["accumulator_type"], "accumulator type"),
        integer_type(root["result_type"], "result type"),
        tuple(
            integer(item, "output shape") for item in sequence(root["output_shape"], "output shape")
        ),
        InvocationScope(string(root["invocation_scope"], "invocation scope")),
    )


def _term_range(left: IntegerRange, right: IntegerRange) -> IntegerRange:
    products = (
        left.minimum * right.minimum,
        left.minimum * right.maximum,
        left.maximum * right.minimum,
        left.maximum * right.maximum,
    )
    return IntegerRange(min(products), max(products))


def _signed_bits(value_range: IntegerRange) -> int:
    bits = 1
    while value_range.minimum < -(1 << (bits - 1)) or value_range.maximum > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def _weight_ranges(premise: DotProductPremise) -> tuple[tuple[IntegerRange, ...], ...]:
    width, height = premise.weight_shape
    if isinstance(premise.weights, FixedWeightPremise):
        values = premise.weights.values
        if len(values) != width * height:
            raise ValueError("fixed weight count differs from the declared matrix shape")
        return tuple(
            tuple(
                IntegerRange(values[row * height + column], values[row * height + column])
                for row in range(width)
            )
            for column in range(height)
        )
    # Runtime promises apply uniformly to every matrix element, so every output
    # column has the same bound. Retain one distinct column range rather than
    # allocating O(matrix_height) identical proof objects.
    return (tuple(premise.weights.value_range for _row in range(width)),)


def analyze_integer_dot_product(premise: DotProductPremise) -> DotProductBounds:
    if len(premise.weight_shape) != 2:
        raise ValueError("dot-product weights must have rank two")
    width, height = premise.weight_shape
    if premise.activation_shape[-1] != width:
        raise ValueError("activation width and weight rows differ")
    if premise.output_shape != (*premise.activation_shape[:-1], height):
        raise ValueError("output shape does not match the matrix product")
    columns = _weight_ranges(premise)
    output_ranges: list[IntegerRange] = []
    product_min: int | None = None
    product_max: int | None = None
    intermediate_min: int | None = None
    intermediate_max: int | None = None
    for column in columns:
        terms = tuple(_term_range(premise.activation_range, weight) for weight in column)
        result = IntegerRange(
            sum(term.minimum for term in terms),
            sum(term.maximum for term in terms),
        )
        every = IntegerRange(
            sum(min(0, term.minimum) for term in terms),
            sum(max(0, term.maximum) for term in terms),
        )
        output_ranges.append(result)
        product_min = (
            min(term.minimum for term in terms)
            if product_min is None
            else min(product_min, *(term.minimum for term in terms))
        )
        product_max = (
            max(term.maximum for term in terms)
            if product_max is None
            else max(product_max, *(term.maximum for term in terms))
        )
        intermediate_min = (
            every.minimum if intermediate_min is None else min(intermediate_min, every.minimum)
        )
        intermediate_max = (
            every.maximum if intermediate_max is None else max(intermediate_max, every.maximum)
        )
    assert product_min is not None and product_max is not None
    assert intermediate_min is not None and intermediate_max is not None
    result_range = IntegerRange(
        min(item.minimum for item in output_ranges),
        max(item.maximum for item in output_ranges),
    )
    intermediate = IntegerRange(intermediate_min, intermediate_max)
    return DotProductBounds(
        IntegerRange(product_min, product_max),
        intermediate,
        result_range,
        tuple(output_ranges),
        _signed_bits(intermediate),
    )


def _carrier_range(name: str) -> IntegerRange | None:
    fixed = {
        "BOOL": IntegerRange(0, 1),
        "INT8": IntegerRange(-(1 << 7), (1 << 7) - 1),
        "UINT8": IntegerRange(0, (1 << 8) - 1),
        "INT16": IntegerRange(-(1 << 15), (1 << 15) - 1),
        "UINT16": IntegerRange(0, (1 << 16) - 1),
        "INT32": IntegerRange(-(1 << 31), (1 << 31) - 1),
        "UINT32": IntegerRange(0, (1 << 32) - 1),
        "INT64": IntegerRange(INT64_MIN, INT64_MAX),
        "UINT64": IntegerRange(0, (1 << 64) - 1),
        "FLOAT16": IntegerRange(-(1 << 11), 1 << 11),
        "FLOAT32": IntegerRange(-(1 << 24), 1 << 24),
        "FLOAT64": IntegerRange(-(1 << 53), 1 << 53),
    }
    return fixed.get(name)


def _fingerprint(premise: DotProductPremise) -> str:
    weights: dict[str, object]
    if isinstance(premise.weights, FixedWeightPremise):
        weights = {
            "kind": "fixed",
            "values": list(premise.weights.values),
            "content_digest": premise.weights.content_digest,
            "source_carrier": premise.weights.source_carrier,
        }
    else:
        weights = {
            "kind": "runtime",
            "range": [premise.weights.value_range.minimum, premise.weights.value_range.maximum],
            "count": premise.weights.count,
            "source": [
                premise.weights.source_operand.operand_id,
                premise.weights.source_operand.direction,
                premise.weights.source_operand.index,
            ],
            "covered_invocations": [
                scope.scope_id for scope in premise.weights.covered_invocations
            ],
            "promise_id": premise.weights.promise_id,
            "source_carrier": premise.weights.source_carrier,
        }
    value = {
        "activation_range": [premise.activation_range.minimum, premise.activation_range.maximum],
        "activation_type": premise.activation_logical_type.name,
        "activation_carrier": premise.activation_source_carrier,
        "activation_shape": list(premise.activation_shape),
        "activation_source": [
            premise.activation_source.operand_id,
            premise.activation_source.direction,
            premise.activation_source.index,
        ],
        "weights": weights,
        "weight_type": premise.weight_logical_type.name,
        "weight_carrier": premise.weight_source_carrier,
        "weight_shape": list(premise.weight_shape),
        "weight_source": [
            premise.weight_source.operand_id,
            premise.weight_source.direction,
            premise.weight_source.index,
        ],
        "accumulator_type": premise.accumulator_type.name,
        "result_type": premise.result_type.name,
        "output_shape": list(premise.output_shape),
        "invocation_scope": premise.invocation_scope.scope_id,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finding(code: str, message: str, **values: object) -> NumericalFinding:
    return NumericalFinding(code, message, tuple(sorted(values.items())))


def check_integer_dot_product_support(
    *,
    premise: DotProductPremise,
    selected_internal_bits: int,
    selected_output_carrier: str,
    target_max_accumulator_bits: int,
) -> IntegerSupportReport:
    """Prove every independent obligation before issuing an execution witness."""

    _integer(selected_internal_bits, "selected internal bits")
    _integer(target_max_accumulator_bits, "target accumulator limit")
    findings: list[NumericalFinding] = []
    expected_count = prod(premise.weight_shape)
    if len(premise.weight_shape) != 2:
        findings.append(_finding("integer-weight-shape", "weights must have rank two"))
    if not premise.activation_logical_type.value_range.contains_range(premise.activation_range):
        findings.append(
            _finding(
                "integer-activation-range",
                "activation premise exceeds its logical datatype",
                admitted=(premise.activation_range.minimum, premise.activation_range.maximum),
                logical=(
                    premise.activation_logical_type.value_range.minimum,
                    premise.activation_logical_type.value_range.maximum,
                ),
            )
        )
    if isinstance(premise.weights, FixedWeightPremise):
        if len(premise.weights.values) != expected_count:
            findings.append(
                _finding(
                    "integer-fixed-weight-count",
                    "fixed weight count differs from the matrix shape",
                    expected=expected_count,
                    actual=len(premise.weights.values),
                )
            )
        if premise.weights.source_carrier != premise.weight_source_carrier:
            findings.append(
                _finding(
                    "integer-fixed-weight-carrier",
                    "fixed-weight carrier differs from the requested source operand",
                )
            )
        bad = tuple(
            value
            for value in premise.weights.values
            if not premise.weight_logical_type.value_range.contains(value)
        )
        if bad:
            findings.append(
                _finding(
                    "integer-fixed-weight-range",
                    "fixed weights exceed their logical datatype",
                    first=bad[0],
                )
            )
    else:
        promise = premise.weights
        # Applicability is established before range analysis can issue a witness.
        if promise.source_operand != premise.weight_source:
            findings.append(
                _finding(
                    "integer-runtime-promise-source",
                    "runtime-weight promise names a different source operand",
                    promised=promise.source_operand.operand_id,
                    requested=premise.weight_source.operand_id,
                )
            )
        if not promise.covers(premise.invocation_scope):
            findings.append(
                _finding(
                    "integer-runtime-promise-scope",
                    "runtime-weight promise does not cover this invocation",
                    requested=premise.invocation_scope.scope_id,
                    covered=tuple(scope.scope_id for scope in promise.covered_invocations),
                )
            )
        if promise.count != expected_count:
            findings.append(
                _finding(
                    "integer-runtime-promise-count",
                    "runtime-weight promise count differs from the matrix shape",
                    promised=promise.count,
                    requested=expected_count,
                )
            )
        if promise.source_carrier != premise.weight_source_carrier:
            findings.append(
                _finding(
                    "integer-runtime-promise-carrier",
                    "runtime-weight promise carrier differs from the requested source carrier",
                    promised=promise.source_carrier,
                    requested=premise.weight_source_carrier,
                )
            )
        if not premise.weight_logical_type.value_range.contains_range(promise.value_range):
            findings.append(
                _finding(
                    "integer-runtime-weight-range",
                    "runtime-weight promise exceeds the logical weight datatype",
                )
            )
    activation_carrier = _carrier_range(premise.activation_source_carrier)
    if activation_carrier is None or not activation_carrier.contains_range(
        premise.activation_range
    ):
        findings.append(
            _finding(
                "integer-activation-carrier",
                "the activation source carrier cannot represent every admitted integer exactly",
                carrier=premise.activation_source_carrier,
            )
        )
    weight_admitted = (
        IntegerRange(min(premise.weights.values), max(premise.weights.values))
        if isinstance(premise.weights, FixedWeightPremise) and premise.weights.values
        else premise.weights.value_range
        if isinstance(premise.weights, RuntimeWeightPromise)
        else None
    )
    weight_carrier = _carrier_range(premise.weight_source_carrier)
    if weight_admitted is not None and (
        weight_carrier is None or not weight_carrier.contains_range(weight_admitted)
    ):
        findings.append(
            _finding(
                "integer-weight-carrier",
                "the weight source carrier cannot represent every admitted integer exactly",
                carrier=premise.weight_source_carrier,
            )
        )
    if not IntegerRange(INT64_MIN, INT64_MAX).contains_range(premise.activation_range):
        findings.append(
            _finding("integer-operand-int64", "activation premise does not fit signed INT64")
        )
    if weight_admitted is not None and not IntegerRange(INT64_MIN, INT64_MAX).contains_range(
        weight_admitted
    ):
        findings.append(
            _finding("integer-operand-int64", "weight premise does not fit signed INT64")
        )
    bounds: DotProductBounds | None = None
    if not findings:
        try:
            bounds = analyze_integer_dot_product(premise)
        except ValueError as error:
            findings.append(_finding("integer-shape", str(error)))
    if bounds is not None:
        signed64 = IntegerRange(INT64_MIN, INT64_MAX)
        if not signed64.contains_range(bounds.product):
            findings.append(
                _finding(
                    "integer-product-int64",
                    "a product can exceed signed INT64",
                    product=(bounds.product.minimum, bounds.product.maximum),
                )
            )
        if not premise.accumulator_type.value_range.contains_range(bounds.result):
            findings.append(
                _finding(
                    "integer-source-accumulator-range",
                    "the exact result can exceed accDataType",
                    result=(bounds.result.minimum, bounds.result.maximum),
                    accumulator=premise.accumulator_type.name,
                )
            )
        if not premise.result_type.value_range.contains_range(bounds.result):
            findings.append(
                _finding(
                    "integer-source-output-range",
                    "the exact result can exceed outputDataType",
                    result=(bounds.result.minimum, bounds.result.maximum),
                    output=premise.result_type.name,
                )
            )
        internal = (
            IntegerRange(
                -(1 << (selected_internal_bits - 1)), (1 << (selected_internal_bits - 1)) - 1
            )
            if selected_internal_bits > 0
            else IntegerRange(0, 0)
        )
        if selected_internal_bits < 1 or not internal.contains_range(bounds.every_intermediate):
            findings.append(
                _finding(
                    "integer-intermediate-range",
                    "an intermediate can exceed the selected implementation accumulator",
                    intermediate=(
                        bounds.every_intermediate.minimum,
                        bounds.every_intermediate.maximum,
                    ),
                    internal_bits=selected_internal_bits,
                )
            )
        if not signed64.contains_range(bounds.every_intermediate):
            findings.append(
                _finding(
                    "integer-intermediate-int64",
                    "an intermediate can exceed signed INT64",
                )
            )
        if selected_internal_bits > target_max_accumulator_bits:
            findings.append(
                _finding(
                    "integer-target-accumulator-limit",
                    "the selected accumulator width exceeds the target limit",
                    internal_bits=selected_internal_bits,
                    target_max=target_max_accumulator_bits,
                )
            )
        output_range = _carrier_range(selected_output_carrier)
        if selected_output_carrier not in {"INT32", "INT64"} or output_range is None:
            findings.append(
                _finding(
                    "integer-output-carrier",
                    "the bounded integer path uses an INT32 or INT64 output carrier",
                    carrier=selected_output_carrier,
                )
            )
        elif not output_range.contains_range(bounds.result):
            findings.append(
                _finding(
                    "integer-output-carrier",
                    "the selected output carrier cannot represent every result",
                    carrier=selected_output_carrier,
                )
            )
    if findings or bounds is None:
        return IntegerSupportReport(None, tuple(findings))
    return IntegerSupportReport(
        IntegerSupport(
            premise,
            _fingerprint(premise),
            bounds,
            "INT64",
            "INT64",
            selected_output_carrier,
            selected_internal_bits,
            target_max_accumulator_bits,
        ),
        (),
    )


def _runtime_values(
    tensor: RuntimeIntegerTensor, admitted: IntegerRange, role: str
) -> tuple[int, ...]:
    import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415

    array = np.asarray(tensor.values)
    if tuple(int(extent) for extent in array.shape) != tensor.shape:
        raise ValueError(f"runtime {role} array shape differs from its metadata")
    values: list[int] = []
    for raw in array.reshape(-1):
        item = raw.item() if hasattr(raw, "item") else raw
        if isinstance(item, bool):
            raise ValueError(f"runtime {role} contains a Boolean, not an integer")
        if isinstance(item, int):
            value = item
        elif isinstance(item, float) and math.isfinite(item) and item.is_integer():
            value = int(item)
        else:
            raise ValueError(f"runtime {role} contains a non-integral value")
        if not admitted.contains(value):
            raise ValueError(f"runtime {role} exceeds its admitted premise")
        if value < INT64_MIN or value > INT64_MAX:
            raise ValueError(f"runtime {role} cannot enter signed INT64")
        values.append(value)
    return tuple(values)


def validate_integer_dot_product_operands(
    activation: RuntimeIntegerTensor,
    weights: RuntimeIntegerTensor,
    support: IntegerSupport,
) -> ValidatedDotProductOperands:
    premise = support.premise
    for tensor, source, shape, carrier, role in (
        (
            activation,
            premise.activation_source,
            premise.activation_shape,
            premise.activation_source_carrier,
            "activation",
        ),
        (
            weights,
            premise.weight_source,
            premise.weight_shape,
            premise.weight_source_carrier,
            "weights",
        ),
    ):
        if tensor.source != source:
            raise ValueError(f"runtime {role} source identity differs from the support premise")
        if tensor.invocation_scope != premise.invocation_scope:
            raise ValueError(f"runtime {role} invocation differs from the support premise")
        if tensor.shape != shape:
            raise ValueError(f"runtime {role} shape differs from the support premise")
        if tensor.carrier != carrier:
            raise ValueError(f"runtime {role} carrier differs from the support premise")
    activation_values = _runtime_values(activation, premise.activation_range, "activation")
    if isinstance(premise.weights, FixedWeightPremise):
        weight_range = IntegerRange(min(premise.weights.values), max(premise.weights.values))
    else:
        promise = premise.weights
        if promise.source_operand != weights.source or not promise.covers(weights.invocation_scope):
            raise ValueError("runtime-weight promise is not applicable to these operands")
        if promise.count != prod(weights.shape):
            raise ValueError("runtime-weight promise count differs from the supplied tensor")
        if promise.source_carrier != weights.carrier:
            raise ValueError("runtime-weight promise carrier differs from the supplied tensor")
        weight_range = promise.value_range
    weight_values = _runtime_values(weights, weight_range, "weights")
    if isinstance(premise.weights, FixedWeightPremise):
        if weight_values != premise.weights.values:
            raise ValueError("runtime fixed weights differ from the immutable initializer")
        if weights.content_digest != premise.weights.content_digest:
            raise ValueError("runtime fixed-weight digest differs from the immutable initializer")
    elif len(weight_values) != premise.weights.count:
        raise ValueError("runtime-weight tensor count differs from its promise")
    return ValidatedDotProductOperands(
        activation_values,
        weight_values,
        premise.activation_shape,
        premise.weight_shape,
        premise.output_shape,
        support.premise_fingerprint,
    )


def execute_integer_dot_product(
    operands: ValidatedDotProductOperands,
    support: IntegerSupport,
) -> Any:
    """Execute the validated witness with checked signed-64 intermediates."""

    import numpy as np  # noqa: PLC0415

    if operands.premise_fingerprint != support.premise_fingerprint:
        raise ValueError("validated operands belong to a different numerical premise")
    width, height = operands.weight_shape
    rows = prod(operands.activation_shape[:-1])
    if len(operands.activation) != rows * width:
        raise ValueError("validated activation payload length differs from its shape")
    if len(operands.weights) != width * height:
        raise ValueError("validated weight payload length differs from its shape")
    result: list[int] = []
    for row in range(rows):
        for column in range(height):
            total = 0
            for index in range(width):
                product = (
                    operands.activation[row * width + index]
                    * operands.weights[index * height + column]
                )
                if product < INT64_MIN or product > INT64_MAX:
                    raise OverflowError("validated product exceeded signed INT64")
                total += product
                if total < INT64_MIN or total > INT64_MAX:
                    raise OverflowError("validated partial sum exceeded signed INT64")
            result.append(total)
    dtype = np.int32 if support.output_carrier == "INT32" else np.int64
    return np.asarray(result, dtype=dtype).reshape(operands.output_shape)


__all__ = [
    "DotProductBounds",
    "DotProductPremise",
    "FixedWeightPremise",
    "IntegerRange",
    "IntegerSupport",
    "IntegerSupportReport",
    "IntegerType",
    "InvocationScope",
    "NumericalFinding",
    "OperandIdentity",
    "RuntimeIntegerTensor",
    "RuntimeWeightPromise",
    "ValidatedDotProductOperands",
    "analyze_integer_dot_product",
    "check_integer_dot_product_support",
    "decode_dot_product_premise",
    "encode_dot_product_premise",
    "execute_integer_dot_product",
    "validate_integer_dot_product_operands",
]
