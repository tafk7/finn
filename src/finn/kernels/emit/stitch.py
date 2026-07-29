############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The composition stitch — an op-agnostic resolver that wires role-tagged cells.

This is the piece FINN implements as the ~270-line imperative ``code_generation_ipi``
Tcl god-method (hardcoded pin names, branches on ``mem_mode``, knows its sub-blocks).
Here it is a trivial, declarative resolver: given a set of CELLS (each a block with a
declared :class:`~finn.kernels.model.ports.Port` surface), bind complementary
ROLE pairs into nets, broadcast clock/reset, and export whatever is left as region
boundary pins.

The whole point of the port taxonomy is that this code is trivial and NEVER special-
cases an op: it reads ONLY ``(direction, kind, role, index, width)`` to DECIDE a
binding (a port's concrete ``pin`` name appears only inside the emitted command
strings, never in a conditional). No pin-name literal, no op type, no ``mem_mode``.
That op-agnosticism is what makes it handle N-memory / compute⊗compute / zero-param
kernel shapes for free — the cardinality question is answered structurally by
``(role, index)``, not by a hardcoded "one weight stream" assumption. See
``kernel-design/kernel-final-design/port-taxonomy.md``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .artifacts import IPICommands
from ..model.ports import Kind, Port, Role, STANDARD_BINDINGS


class StitchError(ValueError):
    """Raised for an unstitchable cell set — an ambiguous (role, index) binding or a
    width/kind mismatch between complementary ports the resolver was asked to wire."""


@dataclass(frozen=True)
class Cell:
    """One instantiable block in a composed region: a bd-cell ``instance`` name, the
    ``module`` it references (for ``create_bd_cell``), and its declared port surface.
    The resolver binds cells by their ports' roles — it does not inspect ``module``
    for wiring (that is only the instantiation reference)."""

    instance: str
    module: str
    ports: tuple[Port, ...]


# Roles handled by broadcast (a single shared region net), not by pairwise binding.
_BROADCAST = {Role.CLOCK: "ap_clk", Role.RESET: "ap_rst_n"}


def stitch(cells: tuple[Cell, ...], region_name: str = "region") -> IPICommands:
    """Resolve a set of cells into block-design (IPI) commands.

    1. Instantiate each cell (``create_bd_cell``).
    2. For each complementary role pair in ``STANDARD_BINDINGS``, bind a SOURCE output
       to the SINK input sharing its ``index`` (across different cells) into an
       interface net — requiring a compatible ``kind`` and equal ``width``.
    3. Broadcast clock/reset to every cell.
    4. Export every still-unbound port (data edges, config surfaces, an unmatched
       weight port) up as a region boundary pin — this is where an embedded compute's
       unbound weight sink becomes a top-level port with no delivery sibling.
    """
    commands: list[str] = []

    for cell in cells:
        # A cell is a leaf RTL module added to the project by the adapter, so it is
        # referenced as `-type module` (FINN's rtlbackend form). An HLS-packaged cell
        # would instead be `-type ip -vlnv` — a per-cell realization kind we can carry
        # on Cell later; the built MVAU path is all RTL modules.
        commands.append(
            f"create_bd_cell -type module -reference {cell.module} {cell.instance}"
        )

    consumed: set[tuple[str, str]] = set()
    for src_role, sink_role in STANDARD_BINDINGS:
        commands.extend(
            _bind_pair(cells, src_role, sink_role, consumed)
        )

    commands.extend(_broadcast(cells, region_name))
    commands.extend(_export_unbound(cells, region_name, consumed))

    return IPICommands(tuple(commands))


def _bind_pair(cells, src_role, sink_role, consumed) -> list[str]:
    """Bind SOURCE→SINK ports of a complementary role pair by matching ``index``.
    Only INTERNAL (non-boundary) ports bind — a boundary port exports to the enclosing
    region by definition and never wires to a sibling (this is what lets a compute⊗
    compute chain keep its region-input DATA_IN and region-output DATA_OUT boundary
    while binding only the A.out→B.in internal edge). Exactly one source and one sink
    per index; more than one on either side is an ambiguous stitch (a real error, not a
    silent pick)."""
    src_by_index: dict[int, list[tuple[Cell, Port]]] = defaultdict(list)
    sink_by_index: dict[int, list[tuple[Cell, Port]]] = defaultdict(list)
    for cell in cells:
        for port in cell.ports:
            if port.boundary:
                continue
            if port.role == src_role:
                src_by_index[port.index].append((cell, port))
            elif port.role == sink_role:
                sink_by_index[port.index].append((cell, port))

    lines: list[str] = []
    for index, sources in src_by_index.items():
        sinks = sink_by_index.get(index, [])
        if not sinks:
            continue  # unbound source — exported later
        if len(sources) > 1 or len(sinks) > 1:
            raise StitchError(
                f"ambiguous binding for {src_role.value}->{sink_role.value} at index "
                f"{index}: {len(sources)} sources, {len(sinks)} sinks"
            )
        (sc, sp), (kc, kp) = sources[0], sinks[0]
        _check_compatible(src_role, sink_role, sc, sp, kc, kp)
        lines.append(
            f"connect_bd_intf_net "
            f"[get_bd_intf_pins {sc.instance}/{sp.pin}] "
            f"[get_bd_intf_pins {kc.instance}/{kp.pin}]"
        )
        consumed.add((sc.instance, sp.pin))
        consumed.add((kc.instance, kp.pin))
    return lines


def _check_compatible(src_role, sink_role, sc, sp, kc, kp) -> None:
    if sp.kind != kp.kind:
        raise StitchError(
            f"kind mismatch binding {sc.instance}/{sp.pin} ({sp.kind.value}) to "
            f"{kc.instance}/{kp.pin} ({kp.kind.value})"
        )
    if sp.width != kp.width:
        raise StitchError(
            f"width mismatch binding {sc.instance}/{sp.pin} ({sp.width}) to "
            f"{kc.instance}/{kp.pin} ({kp.width})"
        )


def _broadcast(cells, region_name) -> list[str]:
    """Connect every cell's clock/reset pin to the shared region-level port. The region
    top is the composed kernel itself, so its clk/rst are external ``bd_ports`` (created
    by the adapter), not hier pins — every cell's pin fans out from that one port."""
    lines: list[str] = []
    for role, region_port in _BROADCAST.items():
        for cell in cells:
            for port in cell.ports:
                if port.role == role:
                    lines.append(
                        f"connect_bd_net "
                        f"[get_bd_ports {region_port}] "
                        f"[get_bd_pins {cell.instance}/{port.pin}]"
                    )
    return lines


def _export_unbound(cells, region_name, consumed) -> list[str]:
    """Export every port not bound to a sibling and not a broadcast pin as a region
    boundary pin. Interface kinds (AXIS/AXIMM/AXILITE) export as interface pins; a bare
    SIGNAL exports as a net pin."""
    lines: list[str] = []
    for cell in cells:
        for port in cell.ports:
            if port.role in _BROADCAST:
                continue
            if (cell.instance, port.pin) in consumed:
                continue
            if port.kind in (Kind.AXIS, Kind.AXIMM, Kind.AXILITE):
                lines.append(
                    f"make_bd_intf_pins_external "
                    f"[get_bd_intf_pins {cell.instance}/{port.pin}]"
                )
            else:
                lines.append(
                    f"make_bd_pins_external [get_bd_pins {cell.instance}/{port.pin}]"
                )
    return lines
