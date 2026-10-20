# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The integration export: what a shell's integration builds around the build's
partition, read from its network's movers.

``integration(model)`` takes the build's parent graph (its one partition node,
``finn.core.graph.partition.partition_body``), refuses a body whose design its check did
not verify as it is now (``refuse_unverified``), configures the partition's network as
hardware generation does (``configured_network``) and reads the movers its boundary channels
place
(``Channel.mover_contract``) and its shell's row. For the Zynq block design
(``vivado-block-design``, the ``pynq`` shell) it states:

- the partition's IP: its instance, named as its node, and its VLNV;
- each mover as an ``IODMA_hls`` (``IodmaConfiguration``), its node attributes from the
  mover's facts: vectors ``(1, beats)``, the stream's bytes a beat, memory and stream
  widths, as ``InsertIODMA`` sized them. ``IODMA_hls`` moves bytes: its ``dataType``
  stays the ``UINT8`` container and its generated code is the one it always was; the
  stream's element is the mover's (``MoverContract.element``);
- the connections, each from its source pin to its sink pin: each mover to its
  partition port over AXI-Stream, each mover's memory port to the memory interconnect's
  next slave port over AXI-MM, each AXI-Lite bus (in the order: input movers, the
  partition's, output movers) from the control interconnect's next master port, and every
  instance's clock and reset from the shell's;
- the clock: the period asked (the target's), which the shell's processor is asked for;
- the addresses: each AXI-Lite bus from the row's ``control_base``, in that order,
  aligned to its aperture, at least the row's ``control_aperture`` (an ``IODMA_hls``
  control map, three scalar registers, fits the least; a partition's bus takes
  ``2**awaddr`` bytes, as its IP's register map states).

It wires one AXI-Lite bus and one memory port a mover: a mover whose contract states
other counts is refused, by name (``wire_one_bus_each``), as is a partition bus with
no ``awaddr`` (no aperture).

The names are the block design's: movers ``idma<i>`` and ``odma<j>`` by direction in port
order, the static region's instances ``<ip>_0`` (``finn.platform.shells.StaticRegion``), and an
mover's pins those of its ``IODMA_hls`` IP as the pynq shell's runner packages it
(``IODMA_PINS``).

It is a function of the configured network: the export is the integration's (a block
design's names, addresses and Vivado IPs), needed once a build and not at every point
an exploration costs, and the partition's name is its graph node's. A shell without an
integration (``ip``) has no export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from finn.core.graph.annotations.platform import read_target
from finn.core.graph.facts import known_shape
from finn.core.graph.partition import partition_body
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.custom_op.kernels.shell import configured_network, refuse_unverified
from finn.kernels.artifacts.abi import Bus, StandardProtocol
from finn.kernels.artifacts.ipxact import vlnv
from finn.kernels.movers import IODMA_HLS, MoverContract
from finn.platform.shells import VIVADO_BLOCK_DESIGN

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

MOVER_INSTANCES = {"in": "idma", "out": "odma"}
"""A mover's instance name in the block design by its direction, numbered in port order."""

IODMA_PINS = {
    "stream_in": "s_axis_0",
    "stream_out": "m_axis_0",
    "memory": "m_axi_gmem0",
    "control": "s_axi_control_0",
}
"""The pins of an ``IODMA_hls`` mover's IP, its one node packaged as a stitched IP (the pynq
shell's runner, ``finn.shells.pynq.runner``)."""

CLOCK, RESET = "ap_clk", "ap_rst_n"
"""Every instance's clock and reset pins."""


@dataclass(frozen=True, kw_only=True)
class IodmaConfiguration:
    """An ``IODMA_hls`` node for a mover: its ``direction``; ``vectors``, the frame's beats
    as ``numInputVectors``; ``channels``, the stream's bytes a beat (``NumChannels``)
    in the byte ``container`` (``dataType``); and its ``memory_width``
    (``intfWidth``) and ``stream_width`` (``streamWidth``), bits."""

    direction: str
    vectors: tuple[int, ...]
    channels: int
    container: str
    memory_width: int
    stream_width: int

    @property
    def attributes(self) -> dict[str, Any]:
        """The ``IODMA_hls`` node's attributes."""
        return {
            "numInputVectors": list(self.vectors),
            "NumChannels": self.channels,
            "dataType": self.container,
            "intfWidth": self.memory_width,
            "streamWidth": self.stream_width,
            "direction": self.direction,
        }


@dataclass(frozen=True, kw_only=True)
class IntegratedMover:
    """A mover in the integration: its ``instance``, the boundary ``tensor`` and partition
    ``port`` it meets, the tensor's ``shape`` (one inference, as the graph states it),
    its facts (``contract``) and its ``IODMA_hls`` node (``iodma``)."""

    instance: str
    tensor: str
    port: str
    shape: tuple[int, ...]
    contract: MoverContract
    iodma: IodmaConfiguration


@dataclass(frozen=True)
class Connection:
    """One connection: its ``kind`` (``axis``, ``aximm``, ``axilite``, ``clock`` or
    ``reset``), from ``source`` to ``sink``, each ``<instance>/<pin>``."""

    kind: str
    source: str
    sink: str


@dataclass(frozen=True)
class Address:
    """An AXI-Lite bus's address: its ``interface`` (``<instance>/<pin>``), its
    ``offset`` and ``range`` (bytes) in the processor's space."""

    interface: str
    offset: int
    range: int


@dataclass(frozen=True, kw_only=True)
class Integration:
    """The integration export of a partition in its shell (see the module docstring)."""

    shell: str
    board: str | None
    part: str
    integration: str
    host_runtime: str | None
    period_ns: float
    partition: str
    vlnv: str
    movers: tuple[IntegratedMover, ...]
    connections: tuple[Connection, ...]
    addresses: tuple[Address, ...]

    def mover(self, instance: str) -> IntegratedMover:
        """The mover named ``instance``."""
        (found,) = (mover for mover in self.movers if mover.instance == instance)
        return found

    def report(self) -> dict[str, Any]:
        """The export as JSON (``report/integration.json``)."""
        return {
            "shell": self.shell,
            "board": self.board,
            "part": self.part,
            "integration": self.integration,
            "host_runtime": self.host_runtime,
            "period_ns": self.period_ns,
            "partition": {"instance": self.partition, "vlnv": self.vlnv},
            "movers": [
                {
                    "instance": mover.instance,
                    "tensor": mover.tensor,
                    "port": mover.port,
                    "shape": list(mover.shape),
                    "kind": mover.contract.kind,
                    "element": mover.contract.element.dtype.name,
                    "iodma": mover.iodma.attributes,
                }
                for mover in self.movers
            ],
            "connections": [
                {"kind": item.kind, "source": item.source, "sink": item.sink}
                for item in self.connections
            ],
            "addresses": [
                {"interface": item.interface, "offset": hex(item.offset), "range": item.range}
                for item in self.addresses
            ],
        }


class IntegrationError(Refusal):
    """A partition its shell does not integrate, or not by an integration exported here:
    one rejection, owned by who refused (the integration, or the target's shell), its
    message the finding's line."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),))


def iodma_configuration(contract: MoverContract) -> IodmaConfiguration:
    """An ``IODMA_hls`` mover's node, from its facts, in its byte container."""
    if contract.kind != IODMA_HLS:
        raise IntegrationError(
            "mover-kind-unsupported",
            "iodma_configuration",
            f"a mover of kind {contract.kind!r} is not an {IODMA_HLS} mover",
        )
    return IodmaConfiguration(
        direction=contract.direction,
        vectors=(1, contract.beats),
        channels=contract.tdata // 8,
        container="UINT8",
        memory_width=contract.memory_width,
        stream_width=contract.tdata,
    )


def wire_one_bus_each(contract: MoverContract, where: str) -> None:
    """Refuse a mover (at ``where``, for the message) the block design cannot wire: it
    connects one AXI-Lite bus and one memory port a mover, while the shell's budgets
    and costing read the counts the mover's contract states."""
    if (contract.control_buses, contract.memory_ports) != (1, 1):
        raise IntegrationError(
            "mover-buses-unwired",
            VIVADO_BLOCK_DESIGN,
            f"{where}: its {contract.kind} mover states {contract.control_buses} AXI-Lite "
            f"buses and {contract.memory_ports} memory ports; the {VIVADO_BLOCK_DESIGN!r} "
            "integration wires one of each a mover",
        )


def _aperture(bus: Bus, least: int) -> int:
    """The bytes an AXI-Lite ``bus`` takes in the processor's map: ``2**awaddr``, and
    ``least`` at least. A bus with no ``awaddr`` is refused, naming it."""
    widths = [signal.width for signal in bus.signals if signal.logical == "awaddr"]
    if not widths:
        raise IntegrationError(
            "bus-aperture-unknown",
            VIVADO_BLOCK_DESIGN,
            f"{bus.name}: an AXI-Lite bus with no awaddr has no aperture",
        )
    return max(1 << widths[0], least)


def integration(model: ModelWrapper) -> Integration:
    """The integration export of the build's parent graph ``model``: its partition's
    network, as emission builds it (``configured_network``), the design its check verified
    (``refuse_unverified``), in its shell's integration (see the module docstring)."""
    node, body, _ = partition_body(model)
    refuse_unverified(body, node.name)
    point, boundary = configured_network(body, node.name)
    row = point.row
    if row.integration != VIVADO_BLOCK_DESIGN or row.static_region is None:
        raise IntegrationError(
            "integration-unexported",
            "configuration.target.shell",
            f"the {row.shell!r} shell's integration is {row.integration!r}: no export "
            f"(the {VIVADO_BLOCK_DESIGN!r} integration has one)",
        )
    region = row.static_region
    memory, control = f"{region.memory_interconnect}_0", f"{region.control_interconnect}_0"
    counts = {"in": 0, "out": 0}
    movers = []
    for tensor, port in boundary:
        channel = point.channel(tensor)
        if not channel.offered:
            raise IntegrationError(
                "mover-absent",
                VIVADO_BLOCK_DESIGN,
                f"{tensor} ({port}): the {row.shell!r} shell places no mover on it; its "
                "integration connects every boundary port to a mover",
            )
        shape = known_shape(body, tensor)
        if shape is None:
            raise IntegrationError(
                "shape-unstated",
                VIVADO_BLOCK_DESIGN,
                f"{tensor} ({port}): the partition states no shape for it",
            )
        contract: MoverContract = channel.mover_contract
        wire_one_bus_each(contract, f"{tensor} ({port})")
        prefix = MOVER_INSTANCES[contract.direction]
        instance = f"{prefix}{counts[contract.direction]}"
        counts[contract.direction] += 1
        movers.append(
            IntegratedMover(
                instance=instance,
                tensor=tensor,
                port=port,
                shape=shape,
                contract=contract,
                iodma=iodma_configuration(contract),
            )
        )
    inputs = [mover for mover in movers if mover.contract.direction == "in"]
    outputs = [mover for mover in movers if mover.contract.direction == "out"]
    partition = node.name
    buses = [
        pin
        for pin in point.module.abi.pins
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXILITE
    ]
    # The block design's order: the input movers, the partition, the output movers.
    control_buses: list[tuple[str, int]] = [
        (f"{mover.instance}/{IODMA_PINS['control']}", region.control_aperture) for mover in inputs
    ]
    control_buses += [
        (f"{partition}/{bus.name}", _aperture(bus, region.control_aperture)) for bus in buses
    ]
    control_buses += [
        (f"{mover.instance}/{IODMA_PINS['control']}", region.control_aperture) for mover in outputs
    ]
    connections: list[Connection] = []
    for mover in movers:
        if mover.contract.direction == "in":
            stream = Connection(
                "axis", f"{mover.instance}/{IODMA_PINS['stream_out']}", f"{partition}/{mover.port}"
            )
        else:
            stream = Connection(
                "axis", f"{partition}/{mover.port}", f"{mover.instance}/{IODMA_PINS['stream_in']}"
            )
        connections.append(stream)
    for index, mover in enumerate([*inputs, *outputs]):
        connections.append(
            Connection(
                "aximm", f"{mover.instance}/{IODMA_PINS['memory']}", f"{memory}/S{index:02d}_AXI"
            )
        )
    addresses = []
    base = region.control_base
    for index, (interface, aperture) in enumerate(control_buses):
        connections.append(Connection("axilite", f"{control}/M{index:02d}_AXI", interface))
        offset = -(-base // aperture) * aperture
        addresses.append(Address(interface, offset, aperture))
        base = offset + aperture
    for instance in [
        *(mover.instance for mover in inputs),
        partition,
        *(e.instance for e in outputs),
    ]:
        connections.append(Connection("clock", f"{memory}/aclk", f"{instance}/{CLOCK}"))
        connections.append(Connection("reset", f"{memory}/aresetn", f"{instance}/{RESET}"))
    target = read_target(body)
    return Integration(
        shell=row.shell,
        board=row.board,
        part=target.part,
        integration=row.integration,
        host_runtime=row.host_runtime,
        period_ns=target.platform.period_ns,
        partition=partition,
        vlnv=vlnv(partition),
        movers=tuple(movers),
        connections=tuple(connections),
        addresses=tuple(addresses),
    )


__all__ = [
    "MOVER_INSTANCES",
    "IODMA_PINS",
    "VIVADO_BLOCK_DESIGN",
    "Address",
    "Connection",
    "IntegratedMover",
    "Integration",
    "IntegrationError",
    "IodmaConfiguration",
    "integration",
    "iodma_configuration",
    "wire_one_bus_each",
]
