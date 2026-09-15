# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure analyses shared by dataflow source and implementation adapters."""

from finn.dataflow.analysis.integer_dot import (
    DotProductBounds,
    DotProductPremise,
    FixedWeightPremise,
    IntegerRange,
    IntegerSupport,
    IntegerSupportReport,
    IntegerType,
    InvocationScope,
    NumericalFinding,
    OperandIdentity,
    RuntimeIntegerTensor,
    RuntimeWeightPromise,
    ValidatedDotProductOperands,
    analyze_integer_dot_product,
    check_integer_dot_product_support,
    execute_integer_dot_product,
    validate_integer_dot_product_operands,
)

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
    "execute_integer_dot_product",
    "validate_integer_dot_product_operands",
]
