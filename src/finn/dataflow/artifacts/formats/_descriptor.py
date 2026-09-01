# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# msgspec is declared in requirements.txt but is absent from the interpreter
# the mypy gate runs under, so `Struct` resolves to Any and every subclass and
# keyword reads as an error.  The same situation qonnx is in throughout this
# tree.  Waived for this module only.
# mypy: disable-error-code="call-arg, misc, no-any-return"

"""The ABI, written down and read back.

Shared by both in-tree formats, and the reason the round-trip kit is cheap: a
format that can only emit is a format checked against itself.  When IP-XACT
arrives it parses ``component.xml`` instead; the *shape* of the conformance
test does not change, which is the point of putting the codec here.
"""

from __future__ import annotations

import msgspec  # type: ignore[import-not-found]

from finn.dataflow.artifacts.abi import (
    AbiError,
    Bus,
    Clock,
    ComponentABI,
    Config,
    CustomProtocol,
    Data,
    Derived,
    Direction,
    Endpoint,
    Free,
    Interrupt,
    Port,
    Protocol,
    Reset,
    Role,
    Signal,
    StandardProtocol,
    Status,
)

DESCRIPTOR_NAME = "component.json"
DESCRIPTOR_SCHEMA = "component-descriptor-v1"


class _Role(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    kind: str
    derived_of: str = ""
    ratio: int = 0
    active_low: bool = True
    synchronous: bool = False


class _Port(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    kind: str
    name: str
    role: _Role
    direction: str = ""
    width: int = 0
    protocol: str = ""
    protocol_spec: str = ""
    endpoint: str = ""
    signals: tuple[tuple[str, str], ...] = ()
    associated_clock: str | None = None
    associated_reset: str | None = None


class _Descriptor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: str
    entry_point: str
    ports: tuple[_Port, ...]
    parameters: tuple[tuple[str, str], ...] = ()


_ENCODER = msgspec.json.Encoder()
_DECODER = msgspec.json.Decoder(_Descriptor)


def _role_out(role: Role) -> _Role:
    if isinstance(role, Clock):
        if isinstance(role.rate, Derived):
            return _Role("clock", derived_of=role.rate.of, ratio=role.rate.ratio)
        return _Role("clock")
    if isinstance(role, Reset):
        return _Role("reset", active_low=role.active_low, synchronous=role.synchronous)
    for kind, marker in (("config", Config), ("status", Status), ("interrupt", Interrupt)):
        if isinstance(role, marker):
            return _Role(kind)
    return _Role("data")


def _role_in(record: _Role) -> Role:
    if record.kind == "clock":
        if record.derived_of:
            return Clock(Derived(record.derived_of, record.ratio))
        return Clock(Free())
    if record.kind == "reset":
        return Reset(active_low=record.active_low, synchronous=record.synchronous)
    named: dict[str, Role] = {"config": Config(), "status": Status(), "interrupt": Interrupt()}
    return named.get(record.kind, Data())


def _protocol_out(protocol: Protocol) -> tuple[str, str]:
    if isinstance(protocol, CustomProtocol):
        return (protocol.protocol_id, protocol.spec_ref)
    return (protocol.value, "")


def _protocol_in(name: str, spec: str) -> Protocol:
    for standard in StandardProtocol:
        if standard.value == name:
            return standard
    return CustomProtocol(name, spec)


def encode(abi: ComponentABI) -> bytes:
    """The ABI as the descriptor a package carries."""

    ports: list[_Port] = []
    for port in abi.ports:
        if isinstance(port, Signal):
            ports.append(
                _Port(
                    kind="signal",
                    name=port.name,
                    role=_role_out(port.role),
                    direction=port.direction.value,
                    width=port.width,
                )
            )
            continue
        protocol, spec = _protocol_out(port.protocol)
        ports.append(
            _Port(
                kind="bus",
                name=port.name,
                role=_role_out(port.role),
                protocol=protocol,
                protocol_spec=spec,
                endpoint=port.endpoint.value,
                signals=port.signals,
                associated_clock=port.associated_clock,
                associated_reset=port.associated_reset,
            )
        )
    return (
        _ENCODER.encode(
            _Descriptor(DESCRIPTOR_SCHEMA, abi.entry_point, tuple(ports), abi.parameters)
        )
        + b"\n"
    )


def decode(data: bytes) -> ComponentABI:
    """Read an ABI back out of a package.  Refuses anything that is not one."""

    try:
        record = _DECODER.decode(data)
    except (msgspec.ValidationError, msgspec.DecodeError) as error:
        raise AbiError(f"not a component descriptor: {error}") from error
    if record.schema_version != DESCRIPTOR_SCHEMA:
        raise AbiError(f"descriptor schema {record.schema_version!r} is not {DESCRIPTOR_SCHEMA!r}")

    ports: list[Port] = []
    for port in record.ports:
        if port.kind == "signal":
            ports.append(
                Signal(port.name, Direction(port.direction), port.width, _role_in(port.role))
            )
            continue
        ports.append(
            Bus(
                port.name,
                _protocol_in(port.protocol, port.protocol_spec),
                port.signals,
                endpoint=Endpoint(port.endpoint),
                role=_role_in(port.role),
                associated_clock=port.associated_clock,
                associated_reset=port.associated_reset,
            )
        )
    return ComponentABI(record.entry_point, tuple(ports), record.parameters)


__all__ = ["DESCRIPTOR_NAME", "DESCRIPTOR_SCHEMA", "decode", "encode"]
