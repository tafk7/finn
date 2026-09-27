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

from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.derivation import Scalar as BuildScalar
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    ScalarTable,
)
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import Scalar
from finn.kernels.physical.ports import native_stream
from finn.kernels.physical.stream import STREAM_INTERFACES, ReadyValidStream
from finn.dataflow.datatypes import (
    QONNXDataType,
    resolve_qonnx_datatype_name,
)
from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    Subspace,
    constraint,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.kernels.target import DspBlock


class EltwiseOperand(Scalar):
    """FLOAT32, or an ordinary integer of at most 128 bits."""

    @constraint
    def supported(self) -> bool | Rejected:
        return True if self.dtype.name == "FLOAT32" else Integer(1, 128).check(self.dtype)

    admission = ConstraintGroup(supported)


class EltwiseKernel(Kernel):
    id = "finnlib.eltwise"
    version = "1"

    operation = Param(str)
    pe = Param(int)
    lhs_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    rhs_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_dtype(self) -> QONNXDataType:
        operation = self.operation
        a = self.lhs_dtype
        b = self.rhs_dtype
        if a.name == "FLOAT32" or b.name == "FLOAT32":
            return resolve_qonnx_datatype_name("FLOAT32")
        bits = 2 * a.bitwidth() if operation == "MUL" else a.bitwidth() + 1
        signed = a.signed() or operation in ("SUB", "SBR")
        return resolve_qonnx_datatype_name(f"{'INT' if signed else 'UINT'}{bits}")

    lhs_type = Subspace(EltwiseOperand, dtype=lhs_dtype)
    rhs_type = Subspace(EltwiseOperand, dtype=rhs_dtype)
    result_type = Subspace(Scalar, dtype=result_dtype)
    lhs = native_stream("lhs", pe, Endpoint.TARGET, lhs_type, pins=("adat", "avld", "ardy"))
    rhs = native_stream("rhs", pe, Endpoint.TARGET, rhs_type, pins=("bdat", "bvld", "brdy"))
    result = native_stream(
        "result", pe, Endpoint.INITIATOR, result_type, pins=("odat", "ovld", "ordy")
    )

    b_scale = Param(float)

    @derived(semantics=default_semantics(float))
    def native_scale(self) -> float | Rejected:
        scale = self.b_scale
        try:
            rounded = float(struct.unpack("!f", struct.pack("!f", scale))[0])
        except (OverflowError, struct.error) as error:
            return reject("eltwise-scale", str(error))
        if not math.isfinite(rounded):
            return reject("eltwise-scale", "B_SCALE must be finite binary32")
        return rounded

    target_dsp = Param(DspBlock)

    @constraint
    def implementation_supported(self) -> bool | Rejected:
        operation = self.operation
        pe = self.pe
        a = self.lhs_dtype
        b = self.rhs_dtype
        scale = self.native_scale
        target = self.target_dsp
        if operation not in ("ADD", "SUB", "SBR", "MUL") or not 1 <= pe <= 0xFFFFFFFF:
            return reject("eltwise-operation", "positive PE and ADD, SUB, SBR or MUL are required")
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

    @view(semantics=STREAM_INTERFACES)
    def interfaces(self) -> tuple[ReadyValidStream, ...] | Rejected:
        if not 1 <= self.pe <= 0xFFFFFFFF:
            return reject("eltwise-interface", "PE must be positive and fit native unsigned int")
        return (self.lhs.stream(), self.rhs.stream(), self.result.stream())

    @view(
        semantics=default_semantics(ModuleBuildRequirements),
        constraints=(implementation_supported,),
    )
    def build_requirements(self) -> ModuleBuildRequirements | Rejected:
        operation = self.operation
        pe = self.pe
        a = self.lhs_dtype
        b = self.rhs_dtype
        scale = self.native_scale
        streams = self.interfaces()
        parameter_values: dict[str, BuildScalar] = {
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
                *(pin for stream in streams for pin in stream.pins()),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        sources = tuple(
            CopiedSource("finnlib", path, provides=(f"module:{name}",), requires=requires)
            for path, name, requires in (
                ("rtl/binopi.sv", "binopi", ()),
                ("rtl/binopf.sv", "binopf", ()),
                ("rtl/int_to_fp32.sv", "int_to_fp32", ()),
                ("rtl/queue.sv", "queue", ()),
                (
                    "rtl/eltwise.sv",
                    "eltwise",
                    ("module:binopi", "module:binopf", "module:int_to_fp32", "module:queue"),
                ),
            )
        )
        return ModuleBuildRequirements(
            EltwiseKernel.id, EltwiseKernel.version, parameters, abi, sources
        )


__all__ = ["EltwiseKernel"]
