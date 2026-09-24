# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Combinational integer-to-IEEE-FLOAT32 conversion, rounding toward zero.

There are no clocks, resets, streams, or handshake signals. This profile admits
ordinary integer encodings up to 128 bits, whose converted values remain finite.
"""

from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.space import Const, Param, Rejected, constraint, reject, view


class IntToFp32Kernel(Kernel):
    id = "finnlib.int_to_fp32"
    version = "1"

    input_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype = Const(
        resolve_qonnx_datatype_name("FLOAT32"), semantics=QONNX_DATATYPE_VALUE_SEMANTICS
    )

    @constraint(dtype=input_dtype)
    def input_supported(*, dtype: QONNXDataType) -> bool | Rejected:
        try:
            ordinary_integer_bounds(dtype)
        except DatatypeError as error:
            return reject("int-to-fp32-type", str(error))
        if not 1 <= dtype.bitwidth() <= 128:
            return reject(
                "int-to-fp32-width", "the finite-result profile supports 1 through 128 input bits"
            )
        return True

    @view(constraints=(input_supported,), dtype=input_dtype, result=result_dtype)
    def physical(*, dtype: QONNXDataType, result: QONNXDataType) -> ModuleBuildRequirements:
        parameters = (("SIGNED", int(dtype.signed())), ("WIDTH", dtype.bitwidth()))
        return ModuleBuildRequirements(
            IntToFp32Kernel.id,
            IntToFp32Kernel.version,
            parameters,
            ModuleABIRequirements(
                FixedModuleName("int_to_fp32"),
                (
                    Signal("ival", Direction.IN, dtype.bitwidth()),
                    Signal("fval", Direction.OUT, result.bitwidth()),
                ),
                tuple((key, str(value)) for key, value in parameters),
            ),
            (
                CopiedSource(
                    "finnlib", "rtl/arith/int_to_fp32.sv", provides=("module:int_to_fp32",)
                ),
            ),
        )


__all__ = ["IntToFp32Kernel"]
