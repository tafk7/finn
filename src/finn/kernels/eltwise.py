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
interchangeable implementation choices. The operands' types are their tensors'.
Its resources are not characterised: it states so (``kernel-resources``).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import ClassVar, cast

from finn.core.space import (
    Decision,
    Domain,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    domain,
    reject,
    view,
)
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow.schedule import Access, Index, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Clocking, Endpoint, Pin
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import Kernel, extent_of
from finn.kernels.laws import NATIVE, NATIVE_CLOCKING, Native
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.port import held, pins
from finn.kernels.port_model import IndexedPort, ScheduledPort
from finn.kernels.values.domains import Integer, admit_element
from finn.platform.capabilities import DspBlock, Platform, Resources

c = Index("c")

#: The largest value of the RTL's PE parameter, a 32-bit integer.
_PE_MAX = 0xFFFFFFFF


def _pe_values(extent: object) -> Domain[int]:
    """The positive divisors of ``extent`` the RTL's PE parameter holds (up to
    ``_PE_MAX``), ascending: ``divisors_of``'s, bounded."""
    divisors = cast(Callable[..., tuple[int, ...]], divisors_of(extent).candidates)

    def accepts(*, candidate: int, extent: int) -> bool:
        return type(extent) is int and 0 < candidate <= _PE_MAX and extent % candidate == 0

    def candidates(*, extent: int) -> tuple[int, ...]:
        return tuple(value for value in divisors(extent=extent) if value <= _PE_MAX)

    return domain(
        accepts=accepts,
        candidates=candidates,
        ordered=True,
        extent=extent,
    )


class EltwiseKernel(Kernel):
    """PE results a beat of ``lhs`` and ``rhs``, element by element.

    Each operand's tensor is walked row-major, PE elements of its innermost axis a
    beat, on one schedule over ``lhs``'s axes; the result keeps ``lhs``'s shape. An
    ``rhs`` whose shape is a trailing part of ``lhs``'s (a channel vector, say)
    reads the trailing indices, so it is broadcast: the port presents it once
    per ``lhs`` element it meets, and its channel's adapter replays it. An
    ``rhs`` of another shape disagrees on an extent (``kernel-extents``).
    """

    id = "finnlib.eltwise"
    clock_domain = NATIVE_CLOCKING

    operation: str = Param()
    # The tensors its operand ports read: facts it is given.
    lhs_tensor: Tensor = Param()
    rhs_tensor: Tensor = Param()
    channels = extent_of(c)  # the innermost axis
    # PE elements of the innermost axis a beat: a divisor of the channels the RTL's PE
    # parameter, a 32-bit integer, holds.
    pe: int = Decision(domain=_pe_values(channels))

    @derived
    def lhs_dtype(self) -> QONNXDataType:
        """``lhs``'s element type: its tensor's."""
        return self.lhs.element.dtype

    @derived
    def rhs_dtype(self) -> QONNXDataType:
        """``rhs``'s element type: its tensor's."""
        return self.rhs.element.dtype

    @derived
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
        """The operation, the operands' types, the scale and the platform's DSP block are
        ones the RTL implements: facts alone, so a kernel wrong on them is refused before
        its PE is set."""
        operation = self.operation
        a = self.lhs_dtype
        b = self.rhs_dtype
        scale = self.native_scale
        target = self.platform.dsp
        if operation not in ("ADD", "SUB", "SBR", "MUL"):
            return reject("eltwise-operation", "ADD, SUB, SBR or MUL is required")
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

    @derived
    def indices(self) -> tuple[Index, ...]:
        """lhs's axes, ``c`` innermost; the result keeps lhs's shape."""
        rank = len(self.lhs_tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 1)), c)

    @derived
    def rhs_indices(self) -> tuple[Index, ...]:
        """rhs reads the trailing axes of lhs: a broadcast it presents once per element it meets."""
        indices, rank = self.indices, len(self.rhs_tensor.shape)
        return indices[max(0, len(indices) - rank) :]  # a longer rhs is refused by rank

    @derived
    def schedule(self) -> Schedule | Rejected:
        """Row-major over lhs's axes, ``c`` split by PE innermost."""
        return self.bound_schedule(self.indices, {c: self.pe})

    @derived
    def result_tensor(self) -> Tensor | Rejected:
        """Its results: ``lhs``'s shape, in ``result_dtype``."""
        shape = self.spanned(self.indices)
        if isinstance(shape, Rejected):
            return shape
        element = admit_element(self.result_dtype, None)
        if isinstance(element, Rejected):
            return element
        return Tensor(shape, element)

    @derived
    def port_clocking(self) -> Clocking:
        return type(self).clock_domain

    lhs = IndexedPort(
        label="lhs",
        transport=Native(prefix="a"),
        endpoint=Endpoint.TARGET,
        tensor=lhs_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        clocking=port_clocking,
    )
    rhs = IndexedPort(
        label="rhs",
        transport=Native(prefix="b"),
        endpoint=Endpoint.TARGET,
        tensor=rhs_tensor,
        schedule=schedule,
        index=rhs_indices,
        lanes=(c,),
        clocking=port_clocking,
    )
    result = ScheduledPort(
        label="result",
        transport=NATIVE,
        endpoint=Endpoint.INITIATOR,
        tensor=result_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        clocking=port_clocking,
    )

    def _accesses(self) -> list[Access]:
        found: list[Access] = []
        if self.lhs.binds:
            found.append(self.lhs.access)
        if self.rhs.binds:
            found.append(self.rhs.access)
        return found

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "the eltwise module's resources are not characterised")

    @view
    def parameters(self) -> dict[str, int | str]:
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

    module_spec: ClassVar[ModuleSpec] = ModuleSpec(
        "eltwise",
        tuple(
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
        ),
    )

    def _lhs_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.lhs.presenting()
        return pins(
            self.lhs.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.lhs.clocking,
        )

    def _lhs_held(self) -> Held | Rejected:
        if self.lhs.placed:
            return Held()
        return held(self.lhs.physical_spec(idle=True), self.lhs.element, self.lhs.clocking)

    def _rhs_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.rhs.presenting()
        return pins(
            self.rhs.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.rhs.clocking,
        )

    def _rhs_held(self) -> Held | Rejected:
        if self.rhs.placed:
            return Held()
        return held(self.rhs.physical_spec(idle=True), self.rhs.element, self.rhs.clocking)

    def _result_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.result.presenting()
        return pins(
            self.result.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.result.clocking,
        )

    def _result_held(self) -> Held | Rejected:
        if self.result.placed:
            return Held()
        return held(self.result.physical_spec(idle=True), self.result.element, self.result.clocking)

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pin_parts = _each((self._lhs_pins(), self._rhs_pins(), self._result_pins()))
        if isinstance(pin_parts, Rejected):
            return pin_parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        assert sources is not None
        refused = module_unprovided(spec, type(self).__qualname__, sources)
        if isinstance(refused, Rejected):
            return refused
        data = ()
        held_parts = _each((self._lhs_held(), self._rhs_held(), self._result_held()))
        if isinstance(held_parts, Rejected):
            return held_parts
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            tuple(pin_parts),
            sources,
            data,
            tuple(held_parts),
            (True, True, True),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return max(
            self.lhs.stream.sequence.form.beats,
            self.rhs.stream.sequence.form.beats,
            self.result.stream.sequence.form.beats,
        )


__all__ = ["EltwiseKernel"]
