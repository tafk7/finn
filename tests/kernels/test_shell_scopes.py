# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a core gives its shell's rules, read at every scope its netlist places, in the
order it places them (``finn.kernels.shell_rules``), on kernels declared here.

A core's control windows are in its instance order (``Kernel.instance_order``), as its
netlist exports its buses: in any order of unequal windows, the windows the rules place are
the netlist's, and so is the span they take, a member's buses at its place whether it
presents them or its child does. A kernel's memory ports are its present members' too
(``Kernel.memory_initiators``), a selected case's or a guarded child's where it is present,
and a subclass's over its own members: a core whose member's child initiates one is refused
on every row. Emission's
count reads every instance's memory ports, presented or not (``netlist_counts``), so a leaf
stating none while its module initiates one is counted.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import TEST_CORE, by_hand, keyed

from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    Select,
    derived,
    reject,
    view,
)
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import RegisterMap
from finn.kernels.base import Kernel
from finn.kernels.composer import DataflowCoreDeclaration, Facts
from finn.kernels.control import ControlModel
from finn.kernels.dataflow_core import bound
from finn.kernels.explore import CoreComposition
from finn.kernels.laws import Axis
from finn.kernels.library import ModuleSpec
from finn.kernels.port_model import InPort, MemoryPort
from finn.kernels.shell import shell
from finn.kernels.shell_rules import SHELL, ShellRules, counts, netlist_counts, placed_windows
from finn.platform.capabilities import Resources

ROW: Mapping[str, object] = {
    "shell": "narrow",
    "units": None,
    "least_window": 4096,
    "aperture": None,
    "targets": (),
    "doubled": True,
    "fixed": None,
}
"""A shell's row as a partition in it holds it, its control's aperture unbounded."""

MEMORY = Bus(
    "m_axi_mem",
    StandardProtocol.AXIMM,
    tuple(
        Member(name, f"m_axi_mem_{name}", 32 if name in ("araddr", "rdata") else 1)
        for name in ("araddr", "arvalid", "arready", "rdata", "rvalid", "rready")
    ),
    Endpoint.INITIATOR,
)
"""An AXI4 initiator: a memory port."""


class Stub(Kernel):
    """A model-only leaf: one input port, idle without the tensor it is not given (no
    holder here connects it)."""

    id = "test.scopes.stub"

    tensor: Tensor | None = Param(default=None)

    x = InPort(
        tensor=tensor,
        dtype=DataType["UINT8"],
        transport=Axis(bus="s_axis"),
        sequence=BeatSequence(vector_major((4,), 1)),
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module_spec = ModuleSpec(
        "stub", (CopiedSource("finnlib", "rtl/stub.sv", provides=("module:stub",)),)
    )


class Windowed(Stub):
    """A leaf presenting its AXI-Lite bus of ``bits`` address bits: a ``2**bits`` window."""

    id = "test.scopes.windowed"

    control: str = Param()
    bits: int = Param()

    @derived
    def bus(self) -> Bus:
        return Bus(
            "cfg",
            StandardProtocol.AXILITE,
            (
                Member("awvalid", "cfg_awvalid"),
                Member("awready", "cfg_awready"),
                Member("awaddr", "cfg_awaddr", self.bits),
            ),
            Endpoint.TARGET,
        )

    config = ControlModel(bus=bus, presented=True)

    @view
    def register_map(self) -> RegisterMap:
        return RegisterMap()


class Reader(Stub):
    """A leaf initiating its memory port, as it states."""

    id = "test.scopes.reader"

    memory = MemoryPort(bus=MEMORY)

    @derived
    def memory_initiators(self) -> int:
        return 1


class Unstated(Stub):
    """A leaf whose module initiates a memory port it does not state."""

    id = "test.scopes.unstated"

    memory = MemoryPort(bus=MEMORY)


class Wrapped(Kernel):
    """A reader as its child."""

    id = "test.scopes.wrapped"

    child = Reader()
    x = child.x


class Chosen(Kernel):
    """A reader or a stub, as chosen."""

    id = "test.scopes.chosen"

    child = Select(reader=Reader(), stub=Stub())
    x = child.x


class Guarded(Kernel):
    """A reader beside a stub, present where it is ``on``."""

    id = "test.scopes.guarded"

    on: bool = Decision(domain=(True, False))
    child = Stub()
    reader = Reader(when=on)
    x = child.x


class Extended(Wrapped):
    """A wrapped reader and a second reader its subclass adds."""

    id = "test.scopes.extended"

    second = Reader()
    x2 = second.x


class ExtendedGuarded(Guarded):
    """A guarded reader and a second reader, always present, its subclass adds."""

    id = "test.scopes.extended_guarded"

    second = Reader()
    x2 = second.x


class StatedExtended(Extended):
    """An extended reader stating its memory ports itself."""

    id = "test.scopes.stated_extended"

    @derived
    def memory_initiators(self) -> int:
        return 2


class Nested(Kernel):
    """A windowed leaf as its child, its bus presented through it."""

    id = "test.scopes.nested"

    bits: int = Param()
    child = Windowed(bits=bits, control="inner_cfg")
    x = child.x


class SelectedFirst(Kernel):
    """A selector placed before its sibling: its selected case a leaf presenting an 8 KiB
    window or a composite whose child presents 16 KiB; its sibling 4 KiB."""

    id = "test.scopes.selected-first"

    selected = Select(leaf=Windowed(bits=13, control="selected_cfg"), nested=Nested(bits=14))
    after = Windowed(bits=12, control="after_cfg")
    x = selected.x
    x2 = after.x
    instance_order = ("selected", "after")


BITS = {"small": 12, "large": 13, "small2": 12}
"""Each windowed leaf's address bits: 4, 8 and 4 KiB."""


def windowed(
    order: tuple[str, ...],
    aperture: int | None,
    kernels: Mapping[str, Facts] | None = None,
    chosen: Mapping[tuple[str, str], object] | None = None,
) -> CoreComposition:
    """A core of ``kernels``, by default the windowed leaves (``BITS``), placed in
    ``order``, in a shell of ``ROW`` but ``aperture``, its edge ports its own, ``chosen``
    set."""
    declared = DataflowCoreDeclaration(
        kernels
        or {
            name: Facts(Windowed, {"control": f"{name}_cfg", "bits": bits})
            for name, bits in BITS.items()
        },
        {},
        name=TEST_CORE,
        order=order,
    )
    made = shell((declared,), row={**ROW, "aperture": aperture})
    return CoreComposition(bound(made, dict(chosen or {})).point)


def agree(
    order: tuple[str, ...],
    windows: tuple[int, ...],
    kernels: Mapping[str, Facts] | None = None,
    chosen: Mapping[tuple[str, str], object] | None = None,
) -> None:
    """The rules place ``windows`` for a core of ``kernels`` in ``order``, ``chosen`` set
    (``windowed``), as its netlist exports them, and admit the span they take and refuse a
    byte less."""
    network = windowed(order, None, kernels, chosen)
    netlist = network.netlist()
    assert isinstance(netlist, Available)
    assert counts(network.member(SHELL)) == Available((windows, 0, False))
    assert netlist_counts(*netlist.value) == (windows, 0, False)
    span = placed_windows(windows)[-1] + windows[-1]
    for aperture, status in ((span, "valid"), (span - 1, "refused")):
        assert windowed(order, aperture, kernels, chosen).validity(SHELL).status == status


@pytest.mark.parametrize("order", list(permutations(BITS)), ids="-".join)
def test_a_core_s_windows_are_in_the_order_its_netlist_exports_its_buses(
    order: tuple[str, ...],
) -> None:
    """In any order of the unequal windows, the windows the rules place are the netlist's,
    so is their span: the rule admits the span the netlist takes and refuses a byte less."""
    agree(order, tuple(1 << BITS[name] for name in order))


@pytest.mark.parametrize(
    "order", [("direct", "nested", "last"), ("nested", "direct", "last")], ids="-".join
)
@pytest.mark.parametrize(("direct", "nested"), [(12, 13), (13, 12)], ids=["4-8", "8-4"])
def test_a_child_s_bus_is_at_its_member_s_place_among_the_core_s_windows(
    order: tuple[str, ...], direct: int, nested: int
) -> None:
    """A bus a member's child presents is at the member's place, beside the buses the core's
    leaves present: the rules' windows are the netlist's, whichever is larger."""
    bits = {"direct": direct, "nested": nested, "last": 12}
    kernels = {
        "direct": Facts(Windowed, {"control": "direct_cfg", "bits": direct}),
        "nested": Facts(Nested, {"bits": nested}),
        "last": Facts(Windowed, {"control": "last_cfg", "bits": 12}),
    }
    agree(order, tuple(1 << bits[name] for name in order), kernels)


@pytest.mark.parametrize(
    ("case", "windows"), [("leaf", (8192, 4096)), ("nested", (16384, 4096))], ids=str
)
def test_a_selected_case_s_bus_is_at_its_selector_s_place(
    case: str, windows: tuple[int, ...]
) -> None:
    """A selector before its sibling: its selected case's bus, a leaf's or its child's, is
    at the selector's place, before the sibling's, in the netlist as in the rules."""
    kernels = {"k": Facts(SelectedFirst, {})}
    agree(("k",), windows, kernels, {("k", "selected"): case})


def core_of(
    kind: type[Kernel], stored: Mapping[str, object], row: Mapping[str, object] | None
) -> CoreComposition:
    """A core of one kernel of ``kind`` (``k0``), ``stored`` (``member.key``) bound, in a
    shell of ``row`` (alone where None: a refused shell answers no netlist)."""
    return by_hand({"k0": Facts(kind, {})}, keyed(stored), shell=row)


def counted(kind: type[Kernel], stored: Mapping[str, object]) -> tuple[object, object]:
    """The counts of a core of one kernel of ``kind`` (``core_of``): its shell's rules', and
    emission's of its netlist."""
    netlist = core_of(kind, stored, None).netlist()
    assert isinstance(netlist, Available)
    return counts(core_of(kind, stored, ROW).member(SHELL)), netlist_counts(*netlist.value)


@pytest.mark.parametrize(
    ("kind", "stored", "ports"),
    [
        (Reader, {}, 1),
        (Wrapped, {}, 1),
        (Chosen, {"k0.child": "reader"}, 1),
        (Chosen, {"k0.child": "stub"}, 0),
        (Guarded, {"k0.on": True}, 1),
        (Guarded, {"k0.on": False}, 0),
        (Extended, {}, 2),
        (ExtendedGuarded, {"k0.on": True}, 2),
        (ExtendedGuarded, {"k0.on": False}, 1),
        (StatedExtended, {}, 2),
    ],
    ids=[
        "leaf",
        "child",
        "selected",
        "not-selected",
        "guarded",
        "absent",
        "subclass",
        "subclass-guarded",
        "subclass-absent",
        "subclass-stated",
    ],
)
def test_a_core_s_memory_ports_are_its_members_at_every_present_scope(
    kind: type[Kernel], stored: Mapping[str, object], ports: int
) -> None:
    """A kernel's memory ports are its present members': a child's, a selected case's, a
    guarded child's where it is present, and a subclass's over the members it adds, unless
    it states its own. The rules count them as the netlist does, and a core initiating one
    is refused."""
    network = core_of(kind, stored, ROW)
    assert network.member("k0").memory_initiators == ports
    assert counted(kind, stored) == (Available(((), ports, False)), ((), ports, False))
    found = network.query(SHELL, ShellRules.core_stream_only)
    codes = [each.code for each in found.findings] if isinstance(found, Rejected) else []
    assert codes == ["core-memory-port"] * bool(ports)
    assert network.validity(SHELL).status == ("refused" if ports else "valid")


def test_a_memory_port_a_leaf_does_not_state_is_counted_from_its_module() -> None:
    """A leaf whose module initiates a memory port it does not state passes the rules, which
    read its statement; emission's count reads its module's pins, and differs."""
    assert counted(Unstated, {}) == (Available(((), 0, False)), ((), 1, False))
