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
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.dataflow.datatypes import (
    resolve_qonnx_datatype_name,
)
from finn.core.space import Const, Param, view


class IntToFp32Kernel(Kernel):
    id = "finnlib.int_to_fp32"
    version = "1"

    input_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    input = integer_scalar(input_dtype, Integer(1, 128))
    result_dtype = Const(
        resolve_qonnx_datatype_name("FLOAT32"), semantics=QONNX_DATATYPE_VALUE_SEMANTICS
    )

    @view
    def build_requirements(self) -> ModuleBuildRequirements:
        encoding = self.input.encoding()
        result = self.result_dtype
        parameters = (("SIGNED", int(encoding.signed)), ("WIDTH", encoding.bits))
        return ModuleBuildRequirements(
            IntToFp32Kernel.id,
            IntToFp32Kernel.version,
            parameters,
            ModuleABIRequirements(
                FixedModuleName("int_to_fp32"),
                (
                    Signal("ival", Direction.IN, encoding.bits),
                    Signal("fval", Direction.OUT, result.bitwidth()),
                ),
                tuple((key, str(value)) for key, value in parameters),
            ),
            (CopiedSource("finnlib", "rtl/int_to_fp32.sv", provides=("module:int_to_fp32",)),),
        )


__all__ = ["IntToFp32Kernel"]
