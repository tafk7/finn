# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure analyses shared by dataflow source and implementation adapters."""

from finn.dataflow.analysis.integer_dot import (
    DotProductBounds,
    DotProductPremise,
    DatatypeWeightPremise,
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
    decode_dot_product_premise,
    encode_dot_product_premise,
    execute_integer_dot_product,
    validate_integer_dot_product_operands,
)

__all__ = [
    "DotProductBounds",
    "DotProductPremise",
    "DatatypeWeightPremise",
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
