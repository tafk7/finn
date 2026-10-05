# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib elementwise arithmetic on two native, unpadded ready/valid streams.

Integer pairs require equal widths and signedness. Integer results grow by one
bit for add/subtract and double for multiply; unsigned subtraction has a signed
result. Mixed inputs convert integers to FLOAT32 toward zero before arithmetic.
Float arithmetic uses the native DSP58 implementation, so it needs a ``platform``
whose DSP block is DSP58. b_scale is rounded to binary32 before checking its
native restrictions and emitting the parameter.
Operation and scale describe the computation; they are supplied inputs, not
interchangeable implementation choices.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
)
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow.schedule import Index, Schedule
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel, factor_domain
from finn.kernels.channels import Channel
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import DspBlock, Platform

c = Index("c")


class EltwiseKernel(Kernel):
    """PE results a beat of ``lhs`` and ``rhs``, element by element.

    Placed on channels, each operand's tensor is walked row-major, PE elements
    of its innermost axis a beat, on one schedule over ``lhs``'s axes. An
    ``rhs`` whose shape is a trailing part of ``lhs``'s (a channel vector, say)
    reads the trailing indices, so it is broadcast: the port presents it once
    per ``lhs`` element it meets, and its channel's adapter replays it. An
    ``rhs`` of another shape disagrees on an extent (``kernel-extents``).
    """

    id = "finnlib.eltwise"
    version = 1
    rtl_module = "eltwise"

    operation: str = Param()
    # PE elements of the innermost axis a beat: its divisors placed, any the RTL takes flat.
    pe: int = Decision(domain=factor_domain(c))
    lhs_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    rhs_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    # The channels it sits on, when a parent places it.
    lhs_channel: Channel = Param(required=False)
    rhs_channel: Channel = Param(required=False)
    result_channel: Channel = Param(required=False)

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

    b_scale: float = Param()

    @derived
    def native_scale(self) -> float | Rejected:
        scale = self.b_scale
        try:
            rounded = float(struct.unpack("!f", struct.pack("!f", scale))[0])
        except (OverflowError, struct.error) as error:
            return reject("eltwise-scale", str(error))
        if not math.isfinite(rounded):
            return reject("eltwise-scale", "B_SCALE must be finite binary32")
        return rounded

    platform: Platform = Param()

    @constraint
    def implementation_supported(self) -> bool | Rejected:
        operation = self.operation
        pe = self.pe
        a = self.lhs_dtype
        b = self.rhs_dtype
        scale = self.native_scale
        target = self.platform.dsp
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

    @constraint
    def operands_supported(self) -> bool | Rejected:
        """Each operand is FLOAT32 or an ordinary integer of at most 128 bits."""
        for dtype in (self.lhs_dtype, self.rhs_dtype):
            if dtype.name != "FLOAT32":
                admitted = Integer(1, 128).check(dtype)
                if isinstance(admitted, Rejected):
                    return admitted
        return True

    admission = ConstraintGroup(implementation_supported, operands_supported)

    @derived
    def indices(self) -> tuple[Index, ...]:
        """lhs's axes, ``c`` innermost; the result keeps lhs's shape."""
        rank = len(self.lhs_channel.tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 1)), c)

    @derived
    def rhs_indices(self) -> tuple[Index, ...]:
        """rhs reads the trailing axes of lhs: a broadcast it presents once per element it meets."""
        indices, rank = self.indices, len(self.rhs_channel.tensor.shape)
        return indices[max(0, len(indices) - rank) :]  # a longer rhs is refused by rank

    @derived
    def factors(self) -> dict[Index, int]:
        return {c: self.pe}

    @derived
    def schedule(self) -> Schedule | Rejected:
        """Row-major over lhs's axes, ``c`` split by PE innermost."""
        return self.bound_schedule(self.indices, self.factors)

    lhs = AxiStreamPort(
        name="lhs",
        endpoint=Endpoint.TARGET,
        channel=lhs_channel,
        schedule=schedule,
        factors=factors,
        index=indices,
        lanes=(c,),
        dtype=lhs_dtype,
        signals=("adat", "avld", "ardy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    rhs = AxiStreamPort(
        name="rhs",
        endpoint=Endpoint.TARGET,
        channel=rhs_channel,
        schedule=schedule,
        factors=factors,
        index=rhs_indices,
        lanes=(c,),
        dtype=rhs_dtype,
        signals=("bdat", "bvld", "brdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    result = AxiStreamPort(
        name="result",
        endpoint=Endpoint.INITIATOR,
        channel=result_channel,
        schedule=schedule,
        factors=factors,
        index=indices,
        lanes=(c,),
        dtype=result_dtype,
        signals=("odat", "ovld", "ordy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        a, b = self.lhs_dtype, self.rhs_dtype
        return {
            "OP": f'"{self.operation}"',
            "PE": self.pe,
            "B_SCALE": repr(self.native_scale),
            "A_FLOAT": int(a.name == "FLOAT32"),
            "B_FLOAT": int(b.name == "FLOAT32"),
            "A_WIDTH": a.bitwidth(),
            "A_SIGNED": int(a.signed()),
            "B_WIDTH": b.bitwidth(),
            "B_SIGNED": int(b.signed()),
            "FORCE_BEHAVIORAL": 0,
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return tuple(
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


__all__ = ["EltwiseKernel"]
