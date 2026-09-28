# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower detached physical structures to portable module requirements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
)
from finn.kernels.artifacts.contributions import CopiedSource, DataSlot, GeneratedData
from finn.kernels.artifacts.derivation import ProducerIdentity, Scalar
from finn.kernels.physical.structure import (
    ConstantBits,
    ModuleInstance,
    PhysicalPin,
    PhysicalStructure,
    PhysicalStructureError,
    PinSlice,
)
from finn.kernels.physical.validation import abi_pins, pin_info, validate_physical_structure

PORT_DECLARATIONS = "PORT_DECLARATIONS"
NET_DECLARATIONS = "NET_DECLARATIONS"
ASSIGNMENTS = "ASSIGNMENTS"
INSTANCES = "INSTANCES"


def _sv_width(width: int) -> str:
    return "" if width == 1 else f" [{width - 1}:0]"


def _sv_port_declarations(structure: PhysicalStructure) -> str:
    declarations: list[str] = []
    for port in structure.top_abi.ports:
        if isinstance(port, Signal):
            declarations.append(
                f"    {port.direction.value} logic{_sv_width(port.width)} {port.name}"
            )
            continue
        directions = dict(port.member_directions())
        for member in port.signals:
            declarations.append(
                f"    {directions[member.physical].value} logic"
                f"{_sv_width(member.width)} {member.physical}"
            )
    return ",\n".join(declarations)


def _net_name(pin: PhysicalPin) -> str:
    return pin.signal_id if pin.instance_id is None else f"n__{pin.instance_id}__{pin.signal_id}"


def _sv_slice(value: PinSlice, *, pin_width: int) -> str:
    base = _net_name(value.pin)
    if value.bit_width == 1:
        return base if pin_width == 1 else f"{base}[{value.bit_offset}]"
    return f"{base}[{value.bit_offset + value.bit_width - 1}:{value.bit_offset}]"


def _sv_constant(value: ConstantBits) -> str:
    if value.value == 0:
        return f"{value.bit_width}'b" + "0" * value.bit_width
    return f"{value.bit_width}'h{value.value:x}"


def _unconnected(structure: PhysicalStructure) -> set[PhysicalPin]:
    """Child outputs disposed whole; a pin with only some bits disposed keeps its net."""
    widths = {
        PhysicalPin(instance.instance_id, name): info.width
        for instance in structure.instances
        for name, info in abi_pins(instance.requirements.abi).items()
    }
    bits: dict[PhysicalPin, int] = {}
    for item in structure.unused_outputs:
        width = widths[item.pin] - item.offset if item.width is None else item.width
        bits[item.pin] = bits.get(item.pin, 0) + width
    return {pin for pin, count in bits.items() if count == widths[pin]}


def _sv_net_declarations(structure: PhysicalStructure) -> str:
    unused = _unconnected(structure)
    lines = []
    for instance in structure.instances:
        for name, info in abi_pins(instance.requirements.abi).items():
            pin = PhysicalPin(instance.instance_id, name)
            if pin not in unused:
                lines.append(f"    logic{_sv_width(info.width)} {_net_name(pin)};")
    return "\n".join(lines)


def _sv_assignments(structure: PhysicalStructure) -> str:
    top = abi_pins(structure.top_abi)
    children = {
        instance.instance_id: abi_pins(instance.requirements.abi)
        for instance in structure.instances
    }

    def render_slice(value: PinSlice) -> str:
        return _sv_slice(value, pin_width=pin_info(value.pin, top=top, children=children).width)

    lines = []
    for wire in structure.wires:
        source = (
            render_slice(wire.source)
            if isinstance(wire.source, PinSlice)
            else _sv_constant(wire.source)
        )
        if wire.invert:
            source = f"!{source}"
        lines.append(f"    assign {render_slice(wire.destination)} = {source};")
    return "\n".join(lines)


def _sv_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if hasattr(value, "value"):
        return str(cast(object, value).value)  # type: ignore[attr-defined]
    return str(value)


def _sv_instances(structure: PhysicalStructure) -> str:
    unused = _unconnected(structure)
    blocks = []
    for instance in structure.instances:
        name = cast(FixedModuleName, instance.requirements.abi.entry_point).value
        parameters = instance.requirements.parameters
        parameter_block = ""
        if parameters:
            parameter_block = (
                " #(\n"
                + ",\n".join(f"        .{key}({_sv_scalar(value)})" for key, value in parameters)
                + "\n    )"
            )
        connections = []
        for signal in abi_pins(instance.requirements.abi):
            pin = PhysicalPin(instance.instance_id, signal)
            connection = "" if pin in unused else _net_name(pin)
            connections.append(f"        .{signal}({connection})")
        blocks.append(
            f"    {name}{parameter_block} {instance.instance_id} (\n"
            + ",\n".join(connections)
            + "\n    );"
        )
    return "\n\n".join(blocks)


def _flatten_contributions(
    instances: Sequence[ModuleInstance],
) -> tuple[CopiedSource | GeneratedData, ...]:
    result: list[CopiedSource | GeneratedData] = []
    destinations: dict[tuple[str, str], CopiedSource | GeneratedData] = {}
    for instance in instances:
        if instance.requirements.render_inputs:
            raise PhysicalStructureError("a composed structure does not nest a rendered child")
        for contribution in instance.requirements.contributions:
            if isinstance(contribution, DataSlot):
                raise PhysicalStructureError("a composed structure cannot flatten a data slot")
            if not isinstance(contribution, (CopiedSource, GeneratedData)):
                raise PhysicalStructureError(
                    "a composed structure flattens copied sources and generated data only"
                )
            coordinate = (
                (contribution.library, contribution.path)
                if isinstance(contribution, CopiedSource)
                else ("", contribution.path)
            )
            previous = destinations.get(coordinate)
            if previous is not None:
                if previous != contribution:
                    raise PhysicalStructureError(
                        f"child sources stage incompatible declarations at {coordinate!r}"
                    )
                continue
            destinations[coordinate] = contribution
            result.append(contribution)
    return tuple(result)


def lower_module_structure(
    structure: PhysicalStructure,
    *,
    producer: ProducerIdentity,
    wrapper_template: RenderedSourceRequirement,
) -> ModuleBuildRequirements:
    """Lower one validated structure to model-free generated-module inputs."""

    validate_physical_structure(structure)
    if not isinstance(structure.top_abi.entry_point, GeneratedModuleName):
        raise PhysicalStructureError("a composed module requires a generated top name")
    if wrapper_template.renderer != SELF_CONTAINED_JINJA_RENDERER:
        raise PhysicalStructureError("the composed wrapper uses the self-contained renderer")
    if not isinstance(wrapper_template.output, EntryPointSourceName):
        raise PhysicalStructureError("the composed wrapper output follows its generated name")
    if not wrapper_template.provides_entry_point:
        raise PhysicalStructureError("the composed wrapper provides the generated entry point")
    expected_arguments = {PORT_DECLARATIONS, NET_DECLARATIONS, ASSIGNMENTS, INSTANCES}
    if set(wrapper_template.arguments) != expected_arguments:
        raise PhysicalStructureError("the composed wrapper declares the canonical fragment inputs")
    render_inputs: tuple[tuple[str, Scalar], ...] = (
        (PORT_DECLARATIONS, _sv_port_declarations(structure)),
        (NET_DECLARATIONS, _sv_net_declarations(structure)),
        (ASSIGNMENTS, _sv_assignments(structure)),
        (INSTANCES, _sv_instances(structure)),
    )
    return ModuleBuildRequirements(
        producer.producer_id,
        producer.contract_version,
        (),
        structure.top_abi,
        (*_flatten_contributions(structure.instances), wrapper_template),
        render_inputs,
    )


__all__ = ["lower_module_structure"]
