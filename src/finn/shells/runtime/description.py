# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The runtime file as the driver reads it (``read``): the build's cores, each edge port
with its tensor, shape, element and mover, the units, the control endpoints with their
nodes and registers, the memory bindings, each placed unit's base and each clock as
delivered.

A file of a schema the driver does not list (``SCHEMAS``) is refused
(``schema-unsupported``), as is one whose mover has no one control endpoint
(``endpoint-absent``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import Refused

SCHEMAS = (1,)
"""The runtime file's schemas this driver reads."""


@dataclass(frozen=True)
class Edge:
    """An edge port of a core: its ``port``, its ``tensor``, its ``shape`` and ``element``
    (a FINN data type's name), the offer's ``kind``, the mover's ``unit`` and kind
    (``mover``), the memory ``target`` it binds, and the mover's stream: ``bits`` a beat,
    ``beats`` a frame, ``lanes`` a beat."""

    port: str
    tensor: str
    shape: tuple[int, ...]
    element: str
    kind: str
    unit: str
    mover: str
    target: str
    bits: int
    beats: int
    lanes: int


@dataclass(frozen=True)
class Core:
    """A core, by its node's ``name``, and its edge ports in port order."""

    name: str
    inputs: tuple[Edge, ...]
    outputs: tuple[Edge, ...]


@dataclass(frozen=True)
class Unit:
    """A unit the platform places: its ``name``, its ``kind`` and its control ``window``
    (``None``: none)."""

    name: str
    kind: str
    window: int | None


@dataclass(frozen=True)
class Endpoint:
    """A control bus: its ``unit`` and ``bus``, the graph ``node`` it configures (a core's
    bus), at ``offset`` in its unit, of ``range`` bytes, and its named ``registers``, each
    an offset and its bits (a mover's)."""

    unit: str
    bus: str
    node: str | None
    offset: int
    range: int
    registers: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class Description:
    """The runtime file (the module docstring): its ``schema``, the row it was made for,
    its ``linker`` and ``runtime``, the ``part``, the period asked (``period_ns``), the
    ``cores``, ``units`` and ``endpoints``, each memory port's target by its unit
    (``bindings``: the port and the target), each placed unit's base (``placement``) and
    each clock as delivered, in MHz (``clocks_mhz``)."""

    schema: int
    shell: str
    board: str | None
    linker: str
    runtime: str | None
    part: str
    period_ns: float
    cores: tuple[Core, ...]
    units: tuple[Unit, ...]
    endpoints: tuple[Endpoint, ...]
    bindings: Mapping[str, tuple[str, str]]
    placement: Mapping[str, int]
    clocks_mhz: Mapping[str, float]

    def core(self, name: str) -> Core:
        """The core named ``name``; another is refused (``core-unknown``)."""
        for each in self.cores:
            if each.name == name:
                return each
        known = [each.name for each in self.cores]
        raise Refused("core-unknown", f"{name}: no core of that name (the cores: {known})")

    def endpoint(self, unit: str) -> Endpoint:
        """The control endpoint of the mover ``unit``: its one endpoint."""
        found = [each for each in self.endpoints if each.unit == unit]
        if len(found) != 1:
            raise Refused(
                "endpoint-absent", f"{unit}: {len(found)} control endpoints, a mover has one"
            )
        return found[0]

    def node(self, node: str) -> Endpoint:
        """The endpoint configuring graph node ``node``; another is refused
        (``node-unknown``)."""
        for each in self.endpoints:
            if each.node == node:
                return each
        known = sorted(each.node for each in self.endpoints if each.node is not None)
        raise Refused(
            "node-unknown", f"{node}: no control endpoint configures it (the nodes: {known})"
        )


def _edge(each: Mapping[str, Any]) -> Edge:
    return Edge(
        port=each["port"],
        tensor=each["tensor"],
        shape=tuple(each["shape"]),
        element=each["element"],
        kind=each["kind"],
        unit=each["unit"],
        mover=each["mover"],
        target=each["target"],
        bits=each["bits"],
        beats=each["beats"],
        lanes=each["lanes"],
    )


def read(source: Path | str | Mapping[str, Any]) -> Description:
    """The runtime file at ``source`` (a path, or its JSON already read), as the driver reads
    it (the module docstring)."""
    found: Mapping[str, Any] = (
        source if isinstance(source, Mapping) else json.loads(Path(source).read_text())
    )
    schema = found.get("schema")
    if schema not in SCHEMAS:
        raise Refused(
            "schema-unsupported",
            f"the runtime file's schema is {schema!r}; this driver reads {list(SCHEMAS)}: "
            "deploy the driver the build wrote beside it",
        )
    cores = tuple(
        Core(
            name=core["name"],
            inputs=tuple(map(_edge, core["inputs"])),
            outputs=tuple(map(_edge, core["outputs"])),
        )
        for core in found["cores"]
    )
    described = Description(
        schema=schema,
        shell=found["row"]["shell"],
        board=found["row"]["board"],
        linker=found["linker"],
        runtime=found["runtime"],
        part=found["part"],
        period_ns=found["period_ns"],
        cores=cores,
        units=tuple(
            Unit(name=unit["name"], kind=unit["kind"], window=unit["window"])
            for unit in found["units"]
        ),
        endpoints=tuple(
            Endpoint(
                unit=each["unit"],
                bus=each["bus"],
                node=each["node"],
                offset=each["offset"],
                range=each["range"],
                registers={
                    name: (register["offset"], register["bits"])
                    for name, register in each["registers"].items()
                },
            )
            for each in found["endpoints"]
        ),
        bindings={
            each["port"].split("/")[0]: (each["port"].split("/")[1], each["target"])
            for each in found["bindings"]
        },
        placement=dict(found["placement"]),
        clocks_mhz=dict(found["clocks_mhz"]),
    )
    for core in cores:
        for edge in (*core.inputs, *core.outputs):
            described.endpoint(edge.unit)
    return described


__all__ = ["SCHEMAS", "Core", "Description", "Edge", "Endpoint", "Unit", "read"]
