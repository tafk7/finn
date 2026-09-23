# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib elementwise arithmetic on two native, unpadded ready/valid streams.

Integer pairs require equal widths and signedness. Integer results grow by one
bit for add/subtract and double for multiply; unsigned subtraction has a signed
result. Mixed inputs convert integers to FLOAT32 toward zero before arithmetic.
Float arithmetic uses the native DSP58 implementation. b_scale is rounded to
binary32 before checking its native restrictions and emitting the parameter.
Operation and scale describe the computation; they are supplied inputs, not
interchangeable implementation choices.
"""

import math
import struct

from finn.kernels.artifacts.abi import Clock, Direction, Reset, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.derivation import Scalar
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    ScalarTable,
)
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.space import ConstraintGroup, Input, Readiness, View, constraint, derived, reject
from finn.kernels.target import DspBlock


class EltwiseKernel(Kernel):
    id = "finnlib.eltwise"
    version = "1"

    operation = Input(str)
    pe = Input(int)
    lhs_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    rhs_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    b_scale = Input(float)
    target_dsp = Input(DspBlock)

    @derived(float, scale=b_scale)
    def native_scale(*, scale: float) -> object:
        try:
            rounded = float(struct.unpack("!f", struct.pack("!f", scale))[0])
        except (OverflowError, struct.error) as error:
            return reject("eltwise-scale", str(error))
        if not math.isfinite(rounded):
            return reject("eltwise-scale", "B_SCALE must be finite binary32")
        return rounded

    @constraint(
        operation=operation, pe=pe, a=lhs_dtype, b=rhs_dtype, scale=native_scale, target=target_dsp
    )
    def implementation_supported(
        *,
        operation: str,
        pe: int,
        a: QONNXDataType,
        b: QONNXDataType,
        scale: float,
        target: DspBlock,
    ) -> object:
        if operation not in ("ADD", "SUB", "SBR", "MUL") or not 1 <= pe <= 0xFFFFFFFF:
            return reject("eltwise-operation", "positive PE and ADD, SUB, SBR or MUL are required")
        for dtype in (a, b):
            if dtype.name != "FLOAT32":
                try:
                    ordinary_integer_bounds(dtype)
                except DatatypeError as error:
                    return reject("eltwise-type", str(error))
                if not 1 <= dtype.bitwidth() <= 128:
                    return reject(
                        "eltwise-width", "this profile supports integer widths from 1 to 128"
                    )
        both_int = a.name != "FLOAT32" and b.name != "FLOAT32"
        if both_int and (a.bitwidth() != b.bitwidth() or a.signed() != b.signed()):
            return reject(
                "eltwise-integer-pair", "integer inputs must match in width and signedness"
            )
        if scale != 1.0 and (both_int or operation == "MUL"):
            return reject(
                "eltwise-scale", "scaling requires float arithmetic and an add/subtract operation"
            )
        if not both_int and target is not DspBlock.DSP58:
            return reject("eltwise-target", "native floating-point arithmetic requires DSP58")
        return True

    @derived(QONNX_DATATYPE_VALUE_SEMANTICS, operation=operation, a=lhs_dtype, b=rhs_dtype)
    def result_dtype(*, operation: str, a: QONNXDataType, b: QONNXDataType) -> QONNXDataType:
        if a.name == "FLOAT32" or b.name == "FLOAT32":
            return resolve_qonnx_datatype_name("FLOAT32")
        bits = 2 * a.bitwidth() if operation == "MUL" else a.bitwidth() + 1
        signed = a.signed() or operation in ("SUB", "SBR")
        return resolve_qonnx_datatype_name(f"{'INT' if signed else 'UINT'}{bits}")

    @derived(
        ModuleBuildRequirements,
        operation=operation,
        pe=pe,
        a=lhs_dtype,
        b=rhs_dtype,
        result=result_dtype,
        scale=native_scale,
    )
    def codegen(
        *,
        operation: str,
        pe: int,
        a: QONNXDataType,
        b: QONNXDataType,
        result: QONNXDataType,
        scale: float,
    ) -> object:
        if pe < 1:
            return reject("eltwise-interface", "PE must be positive")
        parameter_values: dict[str, Scalar] = {
            "OP": f'"{operation}"',
            "PE": pe,
            "B_SCALE": repr(scale),
            "A_FLOAT": int(a.name == "FLOAT32"),
            "B_FLOAT": int(b.name == "FLOAT32"),
            "A_WIDTH": a.bitwidth(),
            "A_SIGNED": int(a.signed()),
            "B_WIDTH": b.bitwidth(),
            "B_SIGNED": int(b.signed()),
            "FORCE_BEHAVIORAL": 0,
        }
        parameters: ScalarTable = tuple(sorted(parameter_values.items()))
        abi = ModuleABIRequirements(
            FixedModuleName("eltwise"),
            (
                Signal("clk", Direction.IN, 1, Clock()),
                Signal(
                    "rst",
                    Direction.IN,
                    1,
                    Reset(active_low=False, synchronous=True, synchronous_to=("clk",)),
                ),
                Signal("adat", Direction.IN, pe * a.bitwidth()),
                Signal("avld", Direction.IN, 1),
                Signal("ardy", Direction.OUT, 1),
                Signal("bdat", Direction.IN, pe * b.bitwidth()),
                Signal("bvld", Direction.IN, 1),
                Signal("brdy", Direction.OUT, 1),
                Signal("odat", Direction.OUT, pe * result.bitwidth()),
                Signal("ovld", Direction.OUT, 1),
                Signal("ordy", Direction.IN, 1),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        sources = tuple(
            CopiedSource("finnlib", path, provides=(f"module:{name}",), requires=requires)
            for path, name, requires in (
                ("rtl/arith/binopi.sv", "binopi", ()),
                ("rtl/arith/binopf.sv", "binopf", ()),
                ("rtl/arith/int_to_fp32.sv", "int_to_fp32", ()),
                ("rtl/infra/fifo.sv", "fifo", ()),
                (
                    "rtl/arith/eltwise.sv",
                    "eltwise",
                    ("module:binopi", "module:binopf", "module:int_to_fp32", "module:fifo"),
                ),
            )
        )
        return ModuleBuildRequirements(
            EltwiseKernel.id, EltwiseKernel.version, parameters, abi, sources
        )

    support = ConstraintGroup(implementation_supported)
    physical_ready = Readiness()
    physical = View(codegen, readiness=physical_ready, constraints=support)


__all__ = ["EltwiseKernel"]
