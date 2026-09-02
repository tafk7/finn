# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FINN RTL memstream Kernel for an exact cyclic input demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring import (
    Covers,
    Imported,
    KernelInput,
    Parameter,
    RegionClaim,
    Sources,
    constraint,
    derived,
)
from finn.dataflow.authoring.scope import Ref, unresolved
from finn.dataflow.design import ABSENT, DATAFLOW_REGION_SEMANTICS
from finn.dataflow.kernels import (
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


def _sets_supported(sets: int) -> bool:
    """The shared Kernel currently proves one initializer-backed set."""

    return sets == 1


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
    uses_class_authoring = True

    covered_region = Imported(DATAFLOW_REGION_SEMANTICS, stable_name="region")
    computation = Imported(ComputationContract)
    output_port = Imported(Port)
    initializer_available = Imported(bool)
    runtime_writable = Imported(bool)
    target_memory_capabilities = Imported(CyclicTargetMemoryCapabilities)
    ram_style = Imported(CyclicRamStyle)
    pumped_memory = Imported(bool)
    sets = Imported(int)
    coverage = Covers(
        RegionClaim(
            KernelInput("role"),
            covered_region,
            computation,
            CYCLIC_PARAMETER_DELIVERY,
        )
    )
    depth = derived(output_port, value_type=int)(_depth)
    width = derived(output_port, value_type=int)(_width)
    ram_style_name = derived(ram_style, value_type=str)(_ram_style_name)
    init_file = derived(
        initializer_available,
        ram_style,
        runtime_writable,
        target_memory_capabilities.allow_absent(),
        value_type=str,
    )(_initializer_file)
    initializer_backed = constraint(initializer_available, sets=("coverage",))(_initializer_backed)
    pumping_supported = constraint(output_port, pumped_memory, sets=("coverage",))(
        _pumping_supported
    )
    sets_supported = constraint(sets, sets=("coverage",))(_sets_supported)
    uram_initialization_supported = constraint(
        initializer_available,
        ram_style,
        runtime_writable,
        target_memory_capabilities.allow_absent(),
        sets=("coverage",),
    )(_uram_initialization_supported)
    depth_parameter = Parameter("DEPTH", depth)
    sets_parameter = Parameter("SETS", sets)
    width_parameter = Parameter("WIDTH", width)
    init_file_parameter = Parameter("INIT_FILE", init_file)
    ram_style_parameter = Parameter("RAM_STYLE", ram_style_name)
    pumped_parameter = Parameter("PUMPED_MEMORY", pumped_memory)
    initializer_parameter = Parameter("INITIALIZER_AVAILABLE", initializer_available)
    writable_parameter = Parameter("RUNTIME_WRITABLE", runtime_writable)
    source_files = Sources(FINN_MEMSTREAM_ROOT, *FINN_MEMSTREAM_SOURCES)

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
