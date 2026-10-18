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

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    reject,
)
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import Contribution, CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, LeafKernel, extent_of, interface
from finn.kernels.port import AxiStreamPort
from finn.kernels.utilization import RESOURCES_SEMANTICS
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


class PwPolyKernel(LeafKernel):
    """``function`` of each element of its input, PE elements a beat; see the module
    docstring."""

    id = "finnlib.pwpolyf"
    version = 1
    rtl_module = "pwpolyf"

    # What it computes: a supplied input, not an interchangeable implementation choice.
    function: str = Param()
    platform: Platform = Param()
    # The tensor its input port reads: a fact it is given.
    input_tensor: Tensor = Param(required=False)

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

    input = AxiStreamPort(
        name="x",
        endpoint=Endpoint.TARGET,
        tensor=input_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        signals=("xdat", "xvld", "xrdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    output = AxiStreamPort(
        name="y",
        endpoint=Endpoint.INITIATOR,
        tensor=output_tensor,
        schedule=schedule,
        index=indices,
        lanes=(c,),
        signals=("ydat", "yvld", "yrdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    input_end = interface("input")
    output_end = interface("output")

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

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "the pwpolyf module's resources are not characterised")

    @derived
    def parameters(self) -> dict[str, int | str]:
        return {"PE": self.pe, "FUNC": f'"{self.function}"'}

    @derived
    def sources(self) -> tuple[Contribution, ...]:
        return SOURCES


__all__ = ["FLOAT32", "FUNCTIONS", "SOURCES", "PwPolyKernel"]
