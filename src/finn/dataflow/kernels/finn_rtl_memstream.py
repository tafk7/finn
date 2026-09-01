# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FINN RTL memstream Kernel for an exact cyclic input demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.scope import Ref, unresolved
from finn.dataflow.design import ABSENT
from finn.dataflow.kernels import (
    KernelScope,
    Kernel,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.computation import ComputationContract
from finn.dataflow.parameters.cyclic.computation import CYCLIC_PARAMETER_DELIVERY
from finn.dataflow.parameters.cyclic.definition import (
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.region import DataflowRegion, Port

FINN_MEMSTREAM_ROOT = "finn"
FINN_MEMSTREAM_SOURCES = (
    "finn-rtllib/memstream/hdl/memstream_wrapper_template.v",
    "finn-rtllib/memstream/hdl/memstream.sv",
    "finn-rtllib/memstream/hdl/memstream_axi.sv",
    "finn-rtllib/axi/hdl/axilite.sv",
)
FINN_MEMSTREAM_MODULE = "finn.rtl.memstream.generated_wrapper"


@dataclass(frozen=True)
class FinnRtlMemstreamInputs:
    """Operation-neutral contracts consumed by the physical memstream."""

    role: str
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    output_port: Ref[Port]
    initializer_available: Ref[bool]
    runtime_writable: Ref[bool]
    target_memory_capabilities: Ref[CyclicTargetMemoryCapabilities]
    ram_style: Ref[CyclicRamStyle]
    pumped_memory: Ref[bool]
    sets: Ref[int]


def _depth(output_port: Port) -> int:
    return output_port.operand.position_count // output_port.beat_sequence.elements_per_beat


def _width(output_port: Port) -> int:
    logical = output_port.logical_beat_bits
    return ((logical + 7) // 8) * 8


def _ram_style_name(ram_style: CyclicRamStyle) -> str:
    return ram_style.value


def _initializer_file(
    initializer_available: bool,
    ram_style: CyclicRamStyle,
    runtime_writable: bool,
    target_memory_capabilities: object,
) -> str:
    if not initializer_available:
        return ""
    if ram_style is not CyclicRamStyle.URAM:
        return "memblock.dat"
    if (
        target_memory_capabilities is not ABSENT
        and cast(
            CyclicTargetMemoryCapabilities, target_memory_capabilities
        ).supports_initialized_uram
    ):
        return "memblock.dat"
    return "" if runtime_writable else "memblock.dat"


def _initializer_backed(initializer_available: bool) -> bool:
    """Limit this implementation to its currently proven initializer-backed path."""

    return initializer_available


def _pumping_supported(output_port: Port, pumped_memory: bool) -> bool:
    return not pumped_memory or output_port.beat_sequence.elements_per_beat > 1


def _uram_initialization_supported(
    initializer_available: bool,
    ram_style: CyclicRamStyle,
    runtime_writable: bool,
    target_memory_capabilities: object,
) -> object:
    if ram_style is not CyclicRamStyle.URAM or runtime_writable or not initializer_available:
        return True
    if target_memory_capabilities is ABSENT:
        return unresolved(
            "cyclic-target-memory-capabilities-missing",
            "initialized URAM requires target memory capabilities",
        )
    return cast(
        CyclicTargetMemoryCapabilities, target_memory_capabilities
    ).supports_initialized_uram


class FinnRtlMemstreamKernel(Kernel):
    """FINN's generated, optionally writable cyclic memory streamer."""

    id = "finn_rtl_memstream"
    version = "1"

    @classmethod
    def define_design(cls, design: KernelScope[FinnRtlMemstreamInputs]) -> None:
        inputs = design.inputs
        design.covers_region(
            inputs.role,
            region=inputs.region,
            computation=inputs.computation,
            implements=CYCLIC_PARAMETER_DELIVERY,
        )
        depth = design.derived(
            "depth", int, dependencies={"output_port": inputs.output_port}, evaluate=_depth
        )
        width = design.derived(
            "width", int, dependencies={"output_port": inputs.output_port}, evaluate=_width
        )
        ram_style_name = design.derived(
            "ram_style_name",
            str,
            dependencies={"ram_style": inputs.ram_style},
            evaluate=_ram_style_name,
        )
        init_file = design.derived(
            "init_file",
            str,
            dependencies={
                "initializer_available": inputs.initializer_available,
                "ram_style": inputs.ram_style,
                "runtime_writable": inputs.runtime_writable,
                "target_memory_capabilities": inputs.target_memory_capabilities.allow_absent(),
            },
            evaluate=_initializer_file,
        )
        design.coverage_constraint(
            "initializer_backed",
            dependencies={"initializer_available": inputs.initializer_available},
            evaluate=_initializer_backed,
        )
        design.coverage_constraint(
            "pumping_supported",
            dependencies={
                "output_port": inputs.output_port,
                "pumped_memory": inputs.pumped_memory,
            },
            evaluate=_pumping_supported,
        )
        design.coverage_constraint(
            "uram_initialization_supported",
            dependencies={
                "initializer_available": inputs.initializer_available,
                "ram_style": inputs.ram_style,
                "runtime_writable": inputs.runtime_writable,
                "target_memory_capabilities": inputs.target_memory_capabilities.allow_absent(),
            },
            evaluate=_uram_initialization_supported,
        )
        design.parameter("DEPTH", cast("Ref[object]", depth))
        design.parameter("SETS", cast("Ref[object]", inputs.sets))
        design.parameter("WIDTH", cast("Ref[object]", width))
        design.parameter("INIT_FILE", cast("Ref[object]", init_file))
        design.parameter("RAM_STYLE", cast("Ref[object]", ram_style_name))
        design.parameter("PUMPED_MEMORY", cast("Ref[object]", inputs.pumped_memory))
        design.parameter("INITIALIZER_AVAILABLE", cast("Ref[object]", inputs.initializer_available))
        design.parameter("RUNTIME_WRITABLE", cast("Ref[object]", inputs.runtime_writable))
        design.source(FINN_MEMSTREAM_ROOT, *FINN_MEMSTREAM_SOURCES)

    @classmethod
    def elaborate(cls, kernel: Kernel) -> tuple[PhysicalComponent, ...]:
        return (
            PhysicalComponent(
                "memstream",
                FINN_MEMSTREAM_MODULE,
                scalar_parameters(dict(kernel.parameters)),
            ),
        )


__all__ = [
    "FINN_MEMSTREAM_MODULE",
    "FINN_MEMSTREAM_ROOT",
    "FINN_MEMSTREAM_SOURCES",
    "FinnRtlMemstreamInputs",
    "FinnRtlMemstreamKernel",
]
