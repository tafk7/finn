# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The connection rules where a holder's member ports are a selector's or a Group's.

A channel on a selector member's port (``consumer=compute.input.side``) connects that port
of every case, and its hops meet the case selected (``compute.a``). An optional port a member
exposes through a selector (MemStream's ``set``, ``set = source.set``) is idle by its
structure where every case leaves its fact out. A kernel an included Group holds
(``parts.fifo``) is its holder's member, each port connected, exposed or refused.
"""

from __future__ import annotations

from typing import Any

from support.helpers import FULL_DSP48E2, in_order, stage_tensor
from support.refusals import refused

from finn.core.space import (
    Available,
    DefinitionError,
    Group,
    Select,
    design_space,
    inspection,
)
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.module import Fragment
from finn.kernels.base import Kernel, neighbours
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.fifo import FifoKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.sides import Side
from finn.kernels.values.semantics import IntegerTensorValue

TENSOR = stage_tensor(4, words=4)
FIFO: dict[str, Any] = dict(
    input_tensor=TENSOR, arriving=in_order(TENSOR), depth=8, platform=FULL_DSP48E2
)
DIRECT = {"incoming.transport": "direct", "outgoing.transport": "direct"}


class Plain(Kernel):
    id = "test.holder_connections.plain"
    compute = FifoKernel(**FIFO)
    incoming = Channel(
        tensor=TENSOR,
        platform=FULL_DSP48E2,
        producer=Side.boundary("s_axis_0"),
        consumer=compute.input.side,
    )
    outgoing = Channel(
        tensor=TENSOR,
        platform=FULL_DSP48E2,
        producer=compute.output.side,
        consumer=Side.boundary("m_axis_0"),
    )
    x = incoming.outer_in
    y = outgoing.outer_out


class Selecting(Kernel):
    id = "test.holder_connections.selecting"
    compute: FifoKernel = Select(a=FifoKernel(**FIFO), b=FifoKernel(**FIFO))
    incoming = Channel(
        tensor=TENSOR,
        platform=FULL_DSP48E2,
        producer=Side.boundary("s_axis_0"),
        consumer=compute.input.side,
    )
    outgoing = Channel(
        tensor=TENSOR,
        platform=FULL_DSP48E2,
        producer=compute.output.side,
        consumer=Side.boundary("m_axis_0"),
    )
    x = incoming.outer_in
    y = outgoing.outer_out


def _netlist(point: Any) -> Fragment:
    found = point.query(Kernel.netlist)
    assert isinstance(found, Available), found
    assert isinstance(found.value, Fragment)
    return found.value


def test_a_channel_on_a_selector_s_port_connects_that_port_of_each_case() -> None:
    """The channels name the selector's ports, one side whichever case is selected, and
    the class is made; at a point, each case's netlist is the plain member's, its instance
    the case selected, and the channel's neighbour is its port there."""
    assert {name: tuple(map(str, sides)) for name, sides in Selecting.connections.items()} == {
        "incoming": ("None", "compute.input"),
        "outgoing": ("compute.output", "None"),
    }
    plain = _netlist(commit(design_space(Plain()), {"compute.ram_style": "auto", **DIRECT}))
    for case in ("a", "b"):
        point = commit(
            design_space(Selecting()),
            {"compute": case, f"compute.{case}.ram_style": "auto", **DIRECT},
        )
        assert inspection.validity(point).status == "valid"
        netlist = _netlist(point)
        assert [label for label, _ in netlist.instances] == [f"compute.{case}"]
        assert repr(netlist).replace(f"compute.{case}", "compute") == repr(plain)
        around = point.query(neighbours(Selecting, "incoming"))
        assert isinstance(around, Available)
        assert around.value.consumer.instance == f"compute.{case}"


def test_a_selector_s_port_on_no_channel_is_refused_by_each_case() -> None:
    with refused("port-unconnected", error=DefinitionError, member="compute", port="b.output"):

        class Unread(Kernel):
            id = "test.holder_connections.unread"
            compute: FifoKernel = Select(a=FifoKernel(**FIFO), b=FifoKernel(**FIFO))
            incoming = Channel(
                tensor=TENSOR,
                platform=FULL_DSP48E2,
                producer=Side.boundary("s_axis_0"),
                consumer=compute.input.side,
            )
            x = incoming.outer_in


MEMORY: dict[str, Any] = dict(
    dtype=TENSOR.element.dtype,
    form=vector_major((4,), 1),
    contents=IntegerTensorValue.of((1, 2, 3, 4)),
    platform=FULL_DSP48E2,
)


class Memory(Kernel):
    """A memory of one set, either of two: its set port, exposed, idle in each."""

    id = "test.holder_connections.memory"
    source: MemStreamKernel = Select(a=MemStreamKernel(**MEMORY), b=MemStreamKernel(**MEMORY))
    output = source.output
    set = source.set


def test_a_port_exposed_through_a_selector_idle_in_every_case_needs_no_channel() -> None:
    """MemStream's ``set`` takes its tensor from an optional fact (``set_tensor``) neither
    case is given: the holder of the memory leaves it unexposed, and is made."""

    class Holder(Kernel):
        id = "test.holder_connections.idle"
        memory = Memory()
        output = memory.output

    point = commit(
        design_space(Holder()),
        {
            "memory.source": "a",
            "memory.source.a.ram_style": "auto",
            "memory.source.a.pumped_memory": False,
        },
    )
    assert inspection.validity(point).status == "valid"
    assert point.query(Holder.memory.set.placed) == Available(False)


def test_a_port_exposed_through_a_selector_one_case_places_is_refused() -> None:
    """Idle only where every case leaves its fact out: a memory one case of which is given
    its set indices places its set port there, which its holder must connect or expose."""

    class Indexed(Kernel):
        id = "test.holder_connections.indexed"
        source: MemStreamKernel = Select(
            a=MemStreamKernel(**MEMORY),
            b=MemStreamKernel(
                **{**MEMORY, "contents": IntegerTensorValue.of(((1, 2, 3, 4), (4, 3, 2, 1)))},
                set_tensor=stage_tensor(2, words=1),
            ),
        )
        output = source.output
        set = source.set

    with refused("port-unconnected", error=DefinitionError, member="memory", port="set"):

        class Holder(Kernel):
            id = "test.holder_connections.indexed_holder"
            memory = Indexed()
            output = memory.output


class Parts(Group):
    fifo = FifoKernel(**FIFO)


def test_a_kernel_an_included_group_holds_is_its_holder_s_member() -> None:
    """Its ports are connected, exposed or refused, each named by its path
    (``parts.fifo.input``)."""
    with refused(
        "port-unconnected", error=DefinitionError, member="parts.fifo", port="input"
    ) as caught:

        class Hidden(Kernel):
            id = "test.holder_connections.hidden"
            parts = Parts()

    assert [dict(each.details)["port"] for each in caught.error.findings] == ["input", "output"]

    class Exposed(Kernel):
        id = "test.holder_connections.exposed"
        parts = Parts()
        x = parts.fifo.input
        y = parts.fifo.output

    assert not Exposed.connections


def test_a_channel_on_a_port_of_a_group_s_kernel_meets_it_by_its_path() -> None:
    class Grouped(Kernel):
        id = "test.holder_connections.grouped"
        parts = Parts()
        incoming = Channel(
            tensor=TENSOR,
            platform=FULL_DSP48E2,
            producer=Side.boundary("s_axis_0"),
            consumer=parts.fifo.input.side,
        )
        x = incoming.outer_in
        y = parts.fifo.output

    assert {name: tuple(map(str, sides)) for name, sides in Grouped.connections.items()} == {
        "incoming": ("None", "parts.fifo.input")
    }
    point = commit(
        design_space(Grouped()),
        {"parts.fifo.ram_style": "auto", "incoming.transport": "direct"},
    )
    assert inspection.validity(point).status == "valid"
    around = point.query(neighbours(Grouped, "incoming"))
    assert isinstance(around, Available)
    assert around.value.consumer.instance == "parts.fifo"
    assert [label for label, _ in _netlist(point).instances] == ["parts.fifo"]
