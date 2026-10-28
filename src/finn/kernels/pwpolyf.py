# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An activation function, elementwise in FLOAT32, on FinnLib's ``pwpolyf``.

``pwpolyf`` (FinnLib ``rtl/nonlin/pwpolyf.sv``, its coefficients in ``pwpolyf_pkg.sv``)
computes GELU, SiLU, Sigmoid or Tanh (its ``FUNC``) of each binary32 lane, PE lanes a
beat. It approximates the function by a polynomial chosen by the input's sign, exponent
and leading fraction bits, evaluated by Horner's rule on a chain of DSPFP32 multiply-adds;
for a magnitude of 8 and above it gives the function's limit in place of the polynomial
(GELU and SiLU pass a positive input through). Its coefficients are the RTL's own: nothing
here states or changes them. Its native ports take a beat a cycle, with a fixed latency
that no port states.

The function is a semantic fact (``function``), as eltwise's ``operation``: what the
kernel computes, not an implementation choice. PE is a Decision over the divisors of the
innermost axis. The input port reads a tensor of any rank, row-major, the innermost axis
(``c``) split by PE; the output, the tensor it states (``output_tensor``), has the
input's shape in FLOAT32 and is presented on the same schedule, so its pace is the
input's, exactly.

Its domain is what that hardware takes: a FLOAT32 input, on a platform whose DSP block
is DSP58 (DSPFP32 is Versal's), and one of the four functions. Its results approximate
the function, so their range is FLOAT32's own. Its resources are not characterised: it
states so (``kernel-resources``). Its computation's reference is test-side until a
KernelOp reaches it: ``tests/support/kernel_specs/pwpolyf.py``.
"""

from __future__ import annotations

from typing import ClassVar

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    reject,
    view,
)
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.schedule import Access, Index, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Clocking, Endpoint, Pin
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace, extent_of
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.port import held, pins
from finn.kernels.port_model import IndexedPort, ScheduledPort
from finn.kernels.values.domains import admit_element
from finn.platform.capabilities import DspBlock, Platform, Resources

c = Index("c")

#: The functions ``pwpolyf`` computes, as its ``FUNC`` names them.
FUNCTIONS = ("gelu", "silu", "sigmoid", "tanh")
FLOAT32 = resolve_qonnx_datatype_name("FLOAT32")
#: Its FinnLib sources, providers first.
SOURCES = (
    CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),
    CopiedSource("finnlib", "rtl/nonlin/pwpolyf_pkg.sv", provides=("package:pwpolyf_pkg",)),
    CopiedSource(
        "finnlib",
        "rtl/nonlin/pwpolyf.sv",
        provides=("module:pwpolyf", "module:pwpolyf_dspfp32"),
        requires=("package:pwpolyf_pkg", "module:fifo"),
    ),
)


class PwPolyKernel(KernelSpace):
    """``function`` of each element of its input, PE elements a beat; see the module
    docstring."""

    id = "finnlib.pwpolyf"
    clock_domain = NATIVE_CLOCKING

    # What it computes: a supplied input, not an interchangeable implementation choice.
    function: str = Param()
    platform: Platform = Param()
    # The tensor its input port reads: a fact it is given.
    input_tensor: Tensor = Param()

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes, ``c`` innermost; the output keeps them."""
        rank = len(self.input_tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 1)), c)

    channels = extent_of(c)  # the innermost axis
    # PE elements of the innermost axis a beat.
    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def schedule(self) -> Schedule | Rejected:
        """Row-major over the input's axes, ``c`` split by PE innermost."""
        return self.bound_schedule(self.indices, {c: self.pe})

    @derived
    def output_tensor(self) -> Tensor | Rejected:
        """Its results: the input's shape, in FLOAT32 over its whole range."""
        shape = self.spanned(self.indices)
        if isinstance(shape, Rejected):
            return shape
        element = admit_element(FLOAT32, None)
        if isinstance(element, Rejected):
            return element
        return Tensor(shape, element)

    def _accesses(self) -> list[Access]:
        found: list[Access] = []
        if self.input.binds:
            found.append(self.input.access)
        return found

    @derived
    def port_clocking(self) -> Clocking:
        return type(self).clock_domain

    input = IndexedPort(
        name="x",
        label="input",
        clocking=port_clocking,
        endpoint=Endpoint.TARGET,
        tensor=input_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        signals=("xdat", "xvld", "xrdy"),
    )
    output = ScheduledPort(
        name="y",
        label="output",
        clocking=port_clocking,
        endpoint=Endpoint.INITIATOR,
        tensor=output_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        signals=("ydat", "yvld", "yrdy"),
    )

    @constraint
    def function_supported(self) -> bool | Rejected:
        """One of the functions ``pwpolyf`` computes."""
        if self.function not in FUNCTIONS:
            return reject(
                "pwpolyf-function",
                f"{self.function!r} is not one of {', '.join(FUNCTIONS)}",
                values={"function": self.function},
            )
        return True

    @constraint
    def element_supported(self) -> bool | Rejected:
        """A FLOAT32 input: the polynomial is evaluated in binary32."""
        dtype = self.input_tensor.element.dtype
        if dtype.name != FLOAT32.name:
            return reject(
                "dtype-family",
                f"{dtype.name} is outside the FLOAT32 datatype domain",
                values={"datatype": dtype.name, "expected": FLOAT32.name},
            )
        return True

    @constraint
    def target_supported(self) -> bool | Rejected:
        """A DSP58 platform: the polynomial is evaluated on DSPFP32."""
        if self.platform.dsp is not DspBlock.DSP58:
            return reject("pwpolyf-target", "the polynomial's DSPFP32 chain requires DSP58")
        return True

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "the pwpolyf module's resources are not characterised")

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"PE": self.pe, "FUNC": f'"{self.function}"'}

    module_spec: ClassVar[ModuleSpec] = ModuleSpec("pwpolyf", SOURCES)

    def _input_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.input.presenting()
        return pins(
            self.input.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.input.clocking,
        )

    def _input_held(self) -> Held | Rejected:
        if self.input.placed:
            return Held()
        return held(self.input.physical_spec(idle=True), self.input.element, self.input.clocking)

    def _output_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.output.presenting()
        return pins(
            self.output.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.output.clocking,
        )

    def _output_held(self) -> Held | Rejected:
        if self.output.placed:
            return Held()
        return held(self.output.physical_spec(idle=True), self.output.element, self.output.clocking)

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pin_parts = _each((self._input_pins(), self._output_pins()))
        if isinstance(pin_parts, Rejected):
            return pin_parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        if sources is None:
            sources = self.module_sources
        refused = module_unprovided(spec, type(self).__qualname__, sources)
        if isinstance(refused, Rejected):
            return refused
        data = ()
        held_parts = _each((self._input_held(), self._output_held()))
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
            (self.input.placed, self.output.placed),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int | Rejected:
        beats: list[int] = []
        if self.input.placed:
            beats.append(self.input.stream.sequence.form.beats)
        if self.output.placed:
            beats.append(self.output.stream.sequence.form.beats)
        return (
            max(beats)
            if beats
            else reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        )


__all__ = ["FLOAT32", "FUNCTIONS", "SOURCES", "PwPolyKernel"]
