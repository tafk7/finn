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

from __future__ import annotations

import math
import struct
from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    constraint,
    default_semantics,
    derived,
    reject,
)
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import CLOCKING, NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import Scalar
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import GivenPort
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock


class EltwiseOperand(Scalar):
    """FLOAT32, or an ordinary integer of at most 128 bits."""

    @constraint
    def supported(self) -> bool | Rejected:
        return True if self.dtype.name == "FLOAT32" else Integer(1, 128).check(self.dtype)

    admission = ConstraintGroup(supported)


class EltwiseKernel(Kernel):
    """PE results a beat of ``lhs`` and ``rhs``, element by element.

    Placed on streams, each operand's tensor is walked row-major, PE elements
    of its innermost axis a beat. An ``rhs`` whose shape is a trailing part of
    ``lhs``'s (a channel vector, say) is broadcast: the port presents it once
    per ``lhs`` element it meets, and its stream's adapter replays it.
    """

    id = "finnlib.eltwise"
    version = "1"
    module = "eltwise"

    operation: str = Param()
    pe: int = Param()
    lhs_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    rhs_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    # The streams it sits on, when a parent places it.
    lhs_stream: Stream = Param(required=False)
    rhs_stream: Stream = Param(required=False)
    result_stream: Stream = Param(required=False)

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

    lhs_type = EltwiseOperand(dtype=lhs_dtype)
    rhs_type = EltwiseOperand(dtype=rhs_dtype)
    result_type = Scalar(dtype=result_dtype)

    b_scale: float = Param()

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

    target_dsp: DspBlock = Param()

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

    @constraint
    def operands_supported(self) -> bool | Rejected:
        """Each operand's encoding is one the arithmetic takes."""
        _ = (self.lhs_type.encoding, self.rhs_type.encoding, self.result_type.encoding)
        return True

    @constraint
    def carried(self) -> bool | Rejected:
        """Each placed stream carries its operand's element."""
        placed: list[tuple[str, str, QONNXDataType]] = []
        if self.present(EltwiseKernel.lhs_stream):
            placed.append(("lhs", self.lhs_stream.tensor.element.datatype_name, self.lhs_dtype))
        if self.present(EltwiseKernel.rhs_stream):
            placed.append(("rhs", self.rhs_stream.tensor.element.datatype_name, self.rhs_dtype))
        if self.present(EltwiseKernel.result_stream):
            placed.append(
                ("result", self.result_stream.tensor.element.datatype_name, self.result_dtype)
            )
        for name, carried, dtype in placed:
            if carried != dtype.name:
                return reject(
                    "eltwise-stream-element",
                    f"the {name} stream carries {carried}, the operand {dtype.name}",
                )
        return True

    admission = ConstraintGroup(implementation_supported, operands_supported, carried)

    def _walk(self, shape: tuple[int, ...]) -> BeatSequence | Rejected:
        try:
            return BeatSequence(vector_major(shape, self.pe))
        except ValueError as error:
            return reject("eltwise-stream-form", f"PE={self.pe}: {error}")

    @derived(semantics=BEAT_SEQUENCE)
    def lhs_sequence(self) -> BeatSequence | Rejected:
        return self._walk(self.lhs_stream.tensor.shape)

    @derived(semantics=BEAT_SEQUENCE)
    def rhs_sequence(self) -> BeatSequence | Rejected:
        """``lhs``'s shape, or a trailing part of it presented once per element it meets."""
        shape, full = self.rhs_stream.tensor.shape, self.lhs_stream.tensor.shape
        if full[len(full) - len(shape) :] != shape:
            return reject(
                "eltwise-stream-form", f"rhs {shape} is not a trailing part of lhs {full}"
            )
        sequence = self._walk(shape)
        if isinstance(sequence, Rejected):
            return sequence
        count = self.lhs_stream.tensor.size // self.rhs_stream.tensor.size
        return sequence if count == 1 else BeatSequence(sequence.form.repeated(count))

    @derived(semantics=BEAT_SEQUENCE)
    def result_sequence(self) -> BeatSequence | Rejected:
        shape = self.result_stream.tensor.shape
        if shape != self.lhs_stream.tensor.shape:
            return reject("eltwise-stream-form", "the result keeps lhs's shape")
        return self._walk(shape)

    lhs = GivenPort(
        name="lhs",
        endpoint=Endpoint.TARGET,
        stream=lhs_stream,
        sequence=lhs_sequence,
        idle_dtype=lhs_dtype,
        idle_lanes=pe,
        signals=("adat", "avld", "ardy"),
        clock="clk",
        reset="rst",
    )
    rhs = GivenPort(
        name="rhs",
        endpoint=Endpoint.TARGET,
        stream=rhs_stream,
        sequence=rhs_sequence,
        idle_dtype=rhs_dtype,
        idle_lanes=pe,
        signals=("bdat", "bvld", "brdy"),
        clock="clk",
        reset="rst",
    )
    result = GivenPort(
        name="result",
        endpoint=Endpoint.INITIATOR,
        stream=result_stream,
        sequence=result_sequence,
        idle_dtype=result_dtype,
        idle_lanes=pe,
        signals=("odat", "ovld", "ordy"),
        clock="clk",
        reset="rst",
    )

    @derived(semantics=CLOCKING)
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


__all__ = ["EltwiseKernel", "EltwiseOperand"]
