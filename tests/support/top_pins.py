# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's bus counts checked against the top's pins, on every network a test reads.

A shell counts a partition's AXI-Lite buses, memory initiators and doubled clock from its
netlist and boundary buses (``finn.platform.shells.bus_counts``), without building its top.
The oracle counts the same from the top's pins (``bus_counts_of_abi``, of
``finn.kernels.artifacts.top.module_abi``). The autouse ``_top_pins`` checks the two equal on
every netlist a network reads (``Network.netlist``, which ``Network.module`` and the shell's
counts read) while a test runs, and fails the test at its end where one differs. A netlist
whose top's pins cannot be formed has no oracle, and is passed by. A conftest imports the
fixture; every conftest that a gate's ``--confcutdir`` stops at does so itself."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest

from finn.core.space import Available
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.module import Abi, Fragment
from finn.kernels.artifacts.top import module_abi
from finn.kernels.network import Network
from finn.platform.shells import bus_counts


def bus_counts_of_abi(abi: Abi) -> tuple[int, int, bool]:
    """What a module's pins ask of a shell: the AXI-Lite buses it presents, the AXI memory
    ports it initiates (a bus beside its streams that it initiates), and whether it takes
    an aligned doubled clock. The oracle: the counts as the shell read them from the top."""
    buses = [pin for pin in abi.pins if isinstance(pin, Bus)]
    control = sum(bus.protocol is StandardProtocol.AXILITE for bus in buses)
    memory = sum(
        bus.protocol is not StandardProtocol.AXIS and bus.endpoint is Endpoint.INITIATOR
        for bus in buses
    )
    return control, memory, bool(abi.clock_alignments)


def mismatch(fragment: Fragment, boundary: Iterable[Bus]) -> str | None:
    """How ``bus_counts`` of ``fragment`` and ``boundary`` differs from its top's pins'
    (``bus_counts_of_abi``); None where they agree, or where the top's pins cannot be
    formed."""
    buses = tuple(boundary)
    try:
        abi = module_abi(fragment, buses)
    except ArtifactError:
        return None
    counted, expected = bus_counts(fragment, buses), bus_counts_of_abi(abi)
    if counted == expected:
        return None
    return f"bus_counts {counted} != the top's pins' {expected}"


@pytest.fixture(autouse=True)
def _top_pins(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each netlist this test's networks read checked (``mismatch``); the test fails at its
    end, naming each difference, where one differs."""
    read = Network.netlist
    found: list[str] = []

    def netlist(network: Network) -> object:
        answer = read(network)
        if isinstance(answer, Available):
            differs = mismatch(*answer.value)
            if differs is not None:
                found.append(f"{network!r}: {differs}")
        return answer

    monkeypatch.setattr(Network, "netlist", netlist)
    yield
    assert not found, "; ".join(found)
