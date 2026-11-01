# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's netlist counts checked against the top's pins, on every partition a test reads.

A shell reads a partition's AXI-Lite buses' windows, AXI4 memory initiators and doubled
clock from its netlist and boundary buses (``finn.kernels.shell_rules.netlist_counts``),
without building its top. The oracle reads the same from the top's pins
(``netlist_counts_of_abi``, of ``finn.kernels.artifacts.top.module_abi``). The autouse
``_top_pins`` checks the two equal on every netlist a core's composition reads
(``CoreComposition.netlist``, which a test's module, ``support.helpers.module_of``, and the
shell's counts read: a graph's and one built by hand) while a test runs, and fails the
test at its end where one differs. A netlist whose top cannot be formed (its pins, or a pin
of an instance nothing drives: a memory port no top presents) has no oracle, and is passed
by: emission refuses it. A conftest imports the fixture; every conftest that a gate's
``--confcutdir`` stops at does so itself."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest

from finn.core.space import Available
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.module import Abi, Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose
from finn.kernels.explore import CoreComposition
from finn.kernels.shell_rules import netlist_counts


def netlist_counts_of_abi(abi: Abi) -> tuple[tuple[int | None, ...], int, bool]:
    """What a module's pins ask of a shell: the window each AXI-Lite bus it presents takes
    (``2**awaddr``, None without an ``awaddr``), the AXI4 memory ports it initiates, and
    whether it takes an aligned doubled clock. The oracle: the counts as the shell read
    them from the top."""
    buses = [pin for pin in abi.pins if isinstance(pin, Bus)]
    windows = []
    for bus in buses:
        if bus.protocol is StandardProtocol.AXILITE and bus.endpoint is Endpoint.TARGET:
            widths = [member.width for member in bus.signals if member.logical == "awaddr"]
            windows.append(2 ** widths[0] if widths else None)
    memory = sum(
        bus.protocol is StandardProtocol.AXIMM and bus.endpoint is Endpoint.INITIATOR
        for bus in buses
    )
    return tuple(windows), memory, bool(abi.clock_alignments)


def mismatch(fragment: Fragment, boundary: Iterable[Bus]) -> str | None:
    """How ``netlist_counts`` of ``fragment`` and ``boundary`` differs from its top's pins'
    (``netlist_counts_of_abi``); None where they agree, or where its top cannot be
    formed."""
    buses = tuple(boundary)
    try:
        abi = compose(ProducerIdentity("test", "1"), "top", fragment, buses).abi
    except ArtifactError:
        return None
    counted, expected = netlist_counts(fragment, buses), netlist_counts_of_abi(abi)
    if counted == expected:
        return None
    return f"netlist_counts {counted} != the top's pins' {expected}"


@pytest.fixture(autouse=True)
def _top_pins(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each netlist this test's partitions read checked (``mismatch``); the test fails at
    its end, naming each difference, where one differs."""
    read = CoreComposition.netlist
    found: list[str] = []

    def netlist(network: CoreComposition) -> object:
        answer = read(network)
        if isinstance(answer, Available):
            differs = mismatch(*answer.value)
            if differs is not None:
                found.append(f"{network!r}: {differs}")
        return answer

    monkeypatch.setattr(CoreComposition, "netlist", netlist)
    yield
    assert not found, "; ".join(found)
