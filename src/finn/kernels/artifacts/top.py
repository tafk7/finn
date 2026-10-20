# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The top: a flat netlist wrapped as one composed module, built only at emission.

A netlist (a ``Fragment``: leaves under their nodes, and the control buses they present)
becomes a top with FINN's top conventions (``compose``): a free-running ``ap_clk``,
``ap_clk2x`` at twice it when an instance takes a doubled clock (``doubled``), the
synchronous ``ap_rst_n``, each boundary bus (inputs, then outputs), and each presented
bus as a target port of its own name (``top_bus``). A top that is not well formed raises
the artifacts' error (``ArtifactError``): a caller that refuses what cannot be built
catches it.
"""

from __future__ import annotations

from collections.abc import Iterable

from finn.kernels.artifacts.abi import (
    CLOCK2X,
    Bus,
    Clock,
    Clocking,
    Derived,
    Endpoint,
    Member,
    abi_pins,
)
from finn.kernels.artifacts.module import Abi, Composed, Fragment, ProducerIdentity


def doubled(fragment: Fragment) -> bool:
    """Whether an instance of ``fragment`` takes a doubled clock: a clock pin whose rate is
    derived from another's."""
    return any(
        isinstance(info.role, Clock) and isinstance(info.role.rate, Derived)
        for _, leaf in fragment.instances
        for info in abi_pins(leaf.abi.pins).values()
    )


def top_bus(child: Bus, port: str, clock: str, reset: str) -> Bus:
    """The child's bus as a top target port: members renamed ``<port>_<MEMBER>``."""
    return Bus(
        port,
        child.protocol,
        tuple(
            Member(item.logical, f"{port}_{item.logical.upper()}", item.width)
            for item in child.signals
        ),
        Endpoint.TARGET,
        child.role,
        clock,
        reset,
    )


def module_abi(fragment: Fragment, boundary: Iterable[Bus]) -> Abi:
    """A composed module's pins: its clocks and reset (``ap_clk2x`` when an instance takes
    a doubled clock), each boundary bus (inputs, then outputs), then each presented bus."""
    takes = doubled(fragment)
    clocking = Clocking(doubled=CLOCK2X if takes else None, doubling=takes)
    buses = list(boundary)
    return Abi(
        (
            *clocking.signals(),
            *(bus for bus in buses if bus.endpoint is Endpoint.TARGET),
            *(bus for bus in buses if bus.endpoint is not Endpoint.TARGET),
            *(
                top_bus(item.bus, item.port, clocking.clock, clocking.reset)
                for item in fragment.exports
            ),
        ),
        (),
        clocking.alignments(),
    )


def compose(
    producer: ProducerIdentity, stem: str, fragment: Fragment, boundary: Iterable[Bus]
) -> Composed:
    """The top of ``fragment``, named ``stem`` and derived by ``producer``: the pins of
    ``module_abi``. Raises ``ArtifactError`` where it is not well formed."""
    return Composed(
        producer.producer_id,
        producer.contract_version,
        stem,
        module_abi(fragment, boundary),
        fragment,
    )


__all__ = ["compose", "doubled", "module_abi", "top_bus"]
