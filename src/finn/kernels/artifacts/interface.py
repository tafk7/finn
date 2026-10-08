# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The interface description of a packaged module: what a user integrating the IP by
hand needs to know of its pins, as JSON (``interface.json`` beside the IP).

``describe_interface`` states, from the module's ABI pins and the facts of its
streams (the partition's boundary facts, ``finn.partition``):

- ``part`` and ``period_ns``: what the module was explored and packaged for;
- ``clocks``: each clock pin with its ``FREQ_HZ`` at the period, as the IP-XACT
  states it (``ipxact.frequency_hz``); a derived clock (``ap_clk2x``) at its multiple
  of its reference, which the integrator must supply phase-aligned;
- ``resets``: each reset pin, its polarity and the clocks it is synchronous to;
- ``streams``: each AXI-Stream port, its direction, ``tdata`` width and clock, and
  what it carries: the tensor, its element (the channel's, by name) and its value
  range, the lanes a beat, the beats a frame, the frame's shape and the ``order``
  its beats and lanes present the frame in (whether row-major, and the traversal);
- ``axilite``: each AXI-Lite bus, its address and data widths, its clock and its
  register map, the IP-XACT's one ``Reg0`` block (``ipxact.register_window``), its
  registers as wide as the bus's ``RegisterMap`` writes them (32 bits for a bus that
  states none);
- ``aximm``: each AXI-MM port with its map. The ABI has no AXI-MM protocol yet, so
  the list is empty.

A pin outside these, or stream facts that contradict the pins (a port with no facts,
facts with no port, another ``tdata``), are refused: the description is the pins'.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Endpoint,
    Pin,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.ipxact import address_width, frequency_hz, register_window
from finn.kernels.artifacts.module import RegisterMap

#: The file the description is written to, beside the packaged IP.
INTERFACE_FILE = "interface.json"

#: Each stream fact the description states, from the boundary facts.
STREAM_FACTS = ("tensor", "element", "range", "lanes", "beats", "shape", "order")


class InterfaceError(Exception):
    """Pins and facts that describe no interface."""


def _width(bus: Bus, logical: str) -> int:
    return next(member.width for member in bus.signals if member.logical == logical)


def describe_interface(
    pins: Sequence[Pin],
    streams: Mapping[str, Mapping[str, Any]],
    *,
    part: str,
    period_ns: float,
    registers: Mapping[str, RegisterMap] | None = None,
) -> dict[str, Any]:
    """The interface description of a module with ``pins``; ``streams`` holds each
    AXI-Stream port's facts by port name (``STREAM_FACTS`` and its ``tdata``), and
    ``registers`` each AXI-Lite bus's ``RegisterMap`` by port name (none: the default
    map, 32-bit words)."""
    registers = registers or {}
    described: dict[str, Any] = {
        "part": part,
        "period_ns": period_ns,
        "clocks": [],
        "resets": [],
        "streams": [],
        "axilite": [],
        "aximm": [],
    }
    unstated = set(streams)
    for pin in pins:
        if isinstance(pin, Signal) and isinstance(pin.role, Clock):
            clock: dict[str, Any] = {
                "name": pin.name,
                "freq_hz": frequency_hz(pin, period_ns, pins),
            }
            if isinstance(pin.role.rate, Derived):
                clock["derived"] = {"of": pin.role.rate.of, "ratio": pin.role.rate.ratio}
            described["clocks"].append(clock)
        elif isinstance(pin, Signal) and isinstance(pin.role, Reset):
            described["resets"].append(
                {
                    "name": pin.name,
                    "polarity": "ACTIVE_LOW" if pin.role.active_low else "ACTIVE_HIGH",
                    "synchronous_to": list(pin.role.synchronous_to or ()),
                }
            )
        elif isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXIS:
            facts = streams.get(pin.name)
            if facts is None:
                raise InterfaceError(f"{pin.name}: a stream with no facts")
            tdata = _width(pin, "tdata")
            if facts.get("tdata") != tdata:
                raise InterfaceError(
                    f"{pin.name}: its facts state tdata {facts.get('tdata')}, its pin {tdata}"
                )
            unstated.discard(pin.name)
            described["streams"].append(
                {
                    "name": pin.name,
                    "direction": "in" if pin.endpoint is Endpoint.TARGET else "out",
                    "tdata": tdata,
                    "clock": pin.associated_clock,
                    **{fact: facts[fact] for fact in STREAM_FACTS},
                }
            )
        elif isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXILITE:
            described["axilite"].append(
                {
                    "name": pin.name,
                    "address_width": address_width(pin),
                    "data_width": _width(pin, "wdata"),
                    "clock": pin.associated_clock,
                    "register_map": [
                        {
                            "name": "Reg0",
                            "base": 0,
                            "range": register_window(pin),
                            "width": registers.get(pin.name, RegisterMap()).word_bits,
                            "usage": "register",
                        }
                    ],
                }
            )
        else:
            raise InterfaceError(f"{pin.name}: a pin outside every interface the IP declares")
    if unstated:
        raise InterfaceError(f"facts for streams the module does not have: {sorted(unstated)}")
    return described


__all__ = ["INTERFACE_FILE", "STREAM_FACTS", "InterfaceError", "describe_interface"]
