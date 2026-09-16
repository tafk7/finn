# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from finn.dataflow.analysis.integer_dot import (
    DatatypeWeightPremise,
    DotProductPremise,
    FixedWeightPremise,
    IntegerRange,
    IntegerType,
    InvocationScope,
    OperandIdentity,
    RuntimeIntegerTensor,
    RuntimeWeightPromise,
    check_integer_dot_product_support,
    execute_integer_dot_product,
    validate_integer_dot_product_operands,
)


X = OperandIdentity("activation", "input", 0)
W = OperandIdentity("weight", "input", 1)
SCOPE = InvocationScope("invocation-B")
I8 = IntegerType("INT8", IntegerRange(-128, 127), 8, True)
I11 = IntegerType("INT11", IntegerRange(-1024, 1023), 11, True)
I32 = IntegerType("INT32", IntegerRange(-(1 << 31), (1 << 31) - 1), 32, True)
I64 = IntegerType("INT64", IntegerRange(-(1 << 63), (1 << 63) - 1), 64, True)
U64 = IntegerType("UINT64", IntegerRange(0, (1 << 64) - 1), 64, False)


def _fixed(
    activation_range: IntegerRange,
    weights: tuple[int, ...],
    *,
    activation_type: IntegerType = I8,
    weight_type: IntegerType = I8,
    accumulator: IntegerType = I32,
    output: IntegerType = I32,
    activation_carrier: str = "FLOAT32",
    weight_carrier: str = "FLOAT32",
) -> DotProductPremise:
    return DotProductPremise(
        activation_range,
        activation_type,
        activation_carrier,
        (1, len(weights)),
        X,
        FixedWeightPremise(weights, "fixed-digest", weight_carrier),
        weight_type,
        weight_carrier,
        (len(weights), 1),
        W,
        accumulator,
        output,
        (1, 1),
        SCOPE,
    )


def _runtime(
    *,
    promise_scope: tuple[InvocationScope, ...] = (SCOPE,),
    promise_source: OperandIdentity = W,
    promise_count: int = 1,
    promise_carrier: str = "FLOAT32",
) -> DotProductPremise:
    return DotProductPremise(
        IntegerRange(0, 3),
        I32,
        "FLOAT32",
        (1, 1),
        X,
        RuntimeWeightPromise(
            IntegerRange(-1, 1),
            promise_count,
            promise_source,
            promise_scope,
            "runtime-promise",
            promise_carrier,
        ),
        I8,
        "FLOAT32",
        (1, 1),
        W,
        I32,
        I32,
        (1, 1),
        SCOPE,
    )


def _support(premise: DotProductPremise, *, internal_bits: int = 32, target_bits: int = 58):
    carrier = "INT32"
    return check_integer_dot_product_support(
        premise=premise,
        selected_internal_bits=internal_bits,
        selected_output_carrier=carrier,
        target_max_accumulator_bits=target_bits,
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def _execute(
    premise: DotProductPremise,
    activation: np.ndarray,
    weights: np.ndarray,
    *,
    weight_digest: str | None = "fixed-digest",
):
    report = _support(premise, internal_bits=premise.accumulator_type.bit_width)
    assert report.supported and report.support is not None
    validated = validate_integer_dot_product_operands(
        RuntimeIntegerTensor(
            activation,
            "FLOAT32",
            tuple(activation.shape),
            X,
            SCOPE,
        ),
        RuntimeIntegerTensor(
            weights,
            "FLOAT32",
            tuple(weights.shape),
            W,
            SCOPE,
            weight_digest,
        ),
        report.support,
    )
    return execute_integer_dot_product(validated, report.support)


def test_n1_and_n2_source_result_and_wider_exact_path() -> None:
    narrow = _fixed(IntegerRange(-128, 127), (1,) * 8, accumulator=I8, output=I8)
    report = _support(narrow, internal_bits=8)
    assert {"integer-source-accumulator-range", "integer-source-output-range"} <= _codes(report)
    assert report.support is None

    wider = replace(narrow, accumulator_type=I11, result_type=I11)
    result = _execute(
        wider,
        np.asarray([[127] * 8], dtype=np.float32),
        np.asarray([[1], [1], [1], [1], [1], [1], [1], [1]], dtype=np.float32),
    )
    assert result.dtype == np.int32
    assert result.item() == 1016


def test_n3_integer_product_is_not_lost_through_float32_matmul() -> None:
    premise = _fixed(
        IntegerRange(4097, 4097),
        (4097,),
        activation_type=I32,
        weight_type=I32,
        accumulator=I32,
        output=I32,
    )
    result = _execute(
        premise,
        np.asarray([[4097]], dtype=np.float32),
        np.asarray([[4097]], dtype=np.float32),
    )
    assert result.item() == 16_785_409
    assert (
        np.matmul(
            np.asarray([[4097]], dtype=np.float32), np.asarray([[4097]], dtype=np.float32)
        ).item()
        == 16_785_408
    )


def test_n4_partial_range_is_independent_of_the_final_range() -> None:
    premise = _fixed(IntegerRange(127, 127), (1, 1, -1, -1), accumulator=I8, output=I8)
    report = _support(premise, internal_bits=8)
    assert "integer-intermediate-range" in _codes(report)
    assert report.support is None


def test_n5_cancellation_executes_in_integer_order() -> None:
    wide = IntegerType("INT26", IntegerRange(-(1 << 25), (1 << 25) - 1), 26, True)
    premise = _fixed(
        IntegerRange(-(1 << 24), 1 << 24),
        (1, 1, 1),
        activation_type=wide,
        accumulator=I32,
        output=I32,
    )
    activation = np.asarray([[16_777_216, 1, -16_777_216]], dtype=np.float32)
    result = _execute(premise, activation, np.asarray([[1], [1], [1]], dtype=np.float32))
    assert result.item() == 1
    assert np.float32(np.float32(activation[0, 0] + activation[0, 1]) + activation[0, 2]) == 0


def test_n6_and_n8_fixed_identity_uses_values_and_digest() -> None:
    left = _fixed(IntegerRange(0, 3), (0,))
    right = replace(left, weights=FixedWeightPremise((1,), "other-digest", "FLOAT32"))
    left_report, right_report = _support(left), _support(right)
    assert left_report.supported and right_report.supported
    assert left_report.support is not None and right_report.support is not None
    assert left_report.support.premise_fingerprint != right_report.support.premise_fingerprint


def test_n7_explicit_runtime_refinement_validates_and_executes() -> None:
    premise = _runtime()
    report = _support(premise)
    assert report.supported and report.support is not None
    validated = validate_integer_dot_product_operands(
        RuntimeIntegerTensor(np.asarray([[2]], dtype=np.float32), "FLOAT32", (1, 1), X, SCOPE),
        RuntimeIntegerTensor(np.asarray([[1]], dtype=np.float32), "FLOAT32", (1, 1), W, SCOPE),
        report.support,
    )
    assert execute_integer_dot_product(validated, report.support).item() == 2


def test_unknown_weights_use_full_logical_datatype_facts_without_a_promise() -> None:
    premise = _runtime()
    premise = replace(
        premise,
        weights=DatatypeWeightPremise(I8.value_range, 1, W, "FLOAT32"),
    )
    report = _support(premise)
    assert report.supported and report.support is not None
    assert report.support.bounds.result == IntegerRange(-384, 381)


def test_carried_runtime_promise_applicability_source_scope_count_and_carrier() -> None:
    unrelated = InvocationScope("unrelated-invocation-A")
    cases = (
        (_runtime(promise_scope=(unrelated,)), "integer-runtime-promise-scope"),
        (
            _runtime(promise_source=OperandIdentity("other-weight", "input", 1)),
            "integer-runtime-promise-source",
        ),
        (_runtime(promise_count=2), "integer-runtime-promise-count"),
        (_runtime(promise_carrier="INT32"), "integer-runtime-promise-carrier"),
    )
    for premise, code in cases:
        report = _support(premise)
        assert report.support is None
        assert code in _codes(report)

    broader = _runtime(promise_scope=(unrelated, SCOPE))
    assert _support(broader).supported


def test_n9_target_limit_is_not_clamped() -> None:
    premise = _fixed(IntegerRange(-128, 127), (1,) * 8, accumulator=I11, output=I11)
    report = _support(premise, internal_bits=11, target_bits=10)
    assert "integer-target-accumulator-limit" in _codes(report)


def test_n10_and_n15_float32_carrier_requires_universal_exactness() -> None:
    premise = _fixed(
        I32.value_range,
        (0,),
        activation_type=I32,
        activation_carrier="FLOAT32",
    )
    assert "integer-activation-carrier" in _codes(_support(premise))


def test_n11_support_does_not_select_folding_or_family() -> None:
    report = _support(_fixed(IntegerRange(-3, 3), (1, -1)))
    assert report.supported and report.support is not None
    assert not hasattr(report.support, "pe")
    assert not hasattr(report.support, "simd")
    assert not hasattr(report.support, "implementation")


def test_n12_uint64_operand_rejects_even_when_multiplied_by_zero() -> None:
    premise = _fixed(
        U64.value_range,
        (0,),
        activation_type=U64,
        activation_carrier="UINT64",
        weight_carrier="INT8",
    )
    assert "integer-operand-int64" in _codes(_support(premise))


def test_n13_out_of_int64_product_rejects_before_cancellation() -> None:
    premise = _fixed(
        IntegerRange(1 << 62, 1 << 62),
        (2, -2),
        activation_type=I64,
        accumulator=I64,
        output=I64,
        activation_carrier="INT64",
        weight_carrier="INT8",
    )
    report = check_integer_dot_product_support(
        premise=premise,
        selected_internal_bits=64,
        selected_output_carrier="INT64",
        target_max_accumulator_bits=64,
    )
    assert "integer-product-int64" in _codes(report)


def test_n14_fixed_premises_do_not_coerce_float_or_bool() -> None:
    with pytest.raises(TypeError, match="integers"):
        FixedWeightPremise((1.0,), "digest", "FLOAT32")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integers"):
        FixedWeightPremise((True,), "digest", "BOOL")  # type: ignore[arg-type]


def test_n16_equal_bounds_do_not_make_premises_interchangeable() -> None:
    small = _fixed(IntegerRange(0, 3), (0,), activation_type=I32)
    large = replace(small, activation_range=IntegerRange(100, 200))
    small_report, large_report = _support(small), _support(large)
    assert small_report.support is not None and large_report.support is not None
    assert small_report.support.bounds == large_report.support.bounds
    assert small_report.support.premise_fingerprint != large_report.support.premise_fingerprint
    with pytest.raises(ValueError, match="activation exceeds"):
        _execute(
            small,
            np.asarray([[150]], dtype=np.float32),
            np.asarray([[0]], dtype=np.float32),
        )
    assert (
        _execute(
            large,
            np.asarray([[150]], dtype=np.float32),
            np.asarray([[0]], dtype=np.float32),
        ).item()
        == 0
    )
