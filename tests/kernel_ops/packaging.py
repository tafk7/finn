# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging without Vivado: a toolchain double that stops PackagePartition where Vivado
would start, so a test reads what packaging checks before then, and one that stands in
for Vivado's packaging, so a test reads what PackagePartition writes beside the IP.
``read_back`` checks an interface description against a module's ABI pins."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper

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
from finn.transformation.kernels import PackagePartition
from finn.util.toolchain import Toolchain


class ReachedVivado(Exception):
    """Raised by ``NoVivado`` when packaging gets as far as running a tool."""


class NoVivado:
    """A toolchain double (packaging calls ``run`` only): it records the first tool
    packaging runs, and stops there."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, tool: str, *args: object, **options: object) -> None:
        self.ran.append(tool)
        raise ReachedVivado(tool)


def reaches_vivado(model: ModelWrapper, name: str, project: Path) -> None:
    """Package ``model`` against ``NoVivado``: its emitted top elaborates, and packaging
    gets as far as running Vivado."""
    toolchain = NoVivado()
    with pytest.raises(ReachedVivado):
        model.transform(
            PackagePartition(name, directory=project, toolchain=cast(Toolchain, toolchain))
        )
    assert toolchain.ran == ["vivado"]


class PackagedByStub:
    """A toolchain double for Vivado's packaging: it writes ``ip/component.xml``, as the
    emitted-text tool's stub does, so PackagePartition completes without Vivado."""

    def run(self, tool: str, args: object, *, cwd: Path, **options: object) -> None:
        (Path(cwd) / "ip").mkdir(parents=True, exist_ok=True)
        (Path(cwd) / "ip" / "component.xml").write_text("<stub/>")


def read_back(described: dict[str, Any], pins: Sequence[Pin], period_ns: float) -> None:
    """Check an interface description against the module's ABI pins: every pin is stated
    once, in its section, with its direction, widths, clock and rate, and nothing else."""
    base = round(1e9 / period_ns)
    clocks = {clock["name"]: clock for clock in described["clocks"]}
    resets = {reset["name"]: reset for reset in described["resets"]}
    streams = {stream["name"]: stream for stream in described["streams"]}
    buses = {bus["name"]: bus for bus in described["axilite"]}
    seen = []
    for pin in pins:
        seen.append(pin.name)
        if isinstance(pin, Signal) and isinstance(pin.role, Clock):
            rate = pin.role.rate
            ratio = rate.ratio if isinstance(rate, Derived) else 1
            assert clocks[pin.name]["freq_hz"] == ratio * base, pin.name
        elif isinstance(pin, Signal) and isinstance(pin.role, Reset):
            polarity = "ACTIVE_LOW" if pin.role.active_low else "ACTIVE_HIGH"
            assert resets[pin.name]["polarity"] == polarity
        elif isinstance(pin, Bus):
            widths = {member.logical: member.width for member in pin.signals}
            if pin.protocol is StandardProtocol.AXIS:
                stream = streams[pin.name]
                direction = "in" if pin.endpoint is Endpoint.TARGET else "out"
                assert (stream["direction"], stream["tdata"]) == (direction, widths["tdata"])
                assert stream["clock"] == pin.associated_clock
                assert stream["lanes"] * stream["element_bits"] <= stream["tdata"]
            else:
                assert buses[pin.name]["address_width"] == widths["awaddr"]
        else:
            raise AssertionError(f"{pin.name}: a pin no section describes")
    described_names = [*clocks, *resets, *streams, *buses]
    described_names += [port["name"] for port in described["aximm"]]
    assert sorted(described_names) == sorted(seen)


__all__ = ["NoVivado", "PackagedByStub", "ReachedVivado", "reaches_vivado", "read_back"]
