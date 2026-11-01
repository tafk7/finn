# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A transparent wrapper: a kernel exposing a MatMul's ports, on a partition's channels.

``WrappedMatMul`` places a ``MatMulKernel`` and exposes its ports (``x = mm.x``), which are
MatMul's core's in turn: one port each, owned by the core, read by three names. On the
channels a partition gives a MatMul (``support.helpers.matmul_network``) it is MatMul: the
same choices (under ``mm``, their owner), the same leaves and links one name deeper, its
ports presented under its own names with their pins on the core (``PortPins.instance``),
its doubled clock and intake MatMul's, through two levels of members.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    built,
    choose,
    labels,
    matmul_network,
    with_adapter_memories,
    with_channel_defaults,
    with_direct_transports,
)
from support.refusals import refused

from finn.core.space import (
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    derived,
    inspection,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.explore import CoreComposition
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import exposed_name, port_direction
from finn.kernels.port_model import exposed_ports
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Platform

FACTS: dict[str, Any] = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)


class WrappedMatMul(Kernel):
    """A MatMul, and nothing else: its facts given down, its ports and the facts a network
    reads of it exposed."""

    id = "test.wrapped_matmul"
    form: Form = Param(default=Form.DENSE)
    platform: Platform = Param()
    x_tensor: Tensor = Param()
    weights: Tensor | KnownWeights = Param()

    mm = MatMulKernel(form=form, platform=platform, x_tensor=x_tensor, weights=weights)
    x = mm.x
    w = mm.w
    y = mm.y

    # The facts a network reads of a MatMul, MatMul's. (A class-body value alias,
    # ``result_tensor = mm.result_tensor``, is not read from outside its class.)

    @derived
    def result_tensor(self) -> Tensor | Rejected:
        found: Tensor | Rejected = self.mm.result_tensor
        return found

    @derived
    def carried_weights(self) -> Tensor | Rejected:
        found: Tensor | Rejected = self.mm.carried_weights
        return found

    @derived
    def stored_weights(self) -> IntegerTensorValue | Inapplicable:
        found: IntegerTensorValue | Inapplicable = self.mm.stored_weights
        return found

    @derived
    def required_weights(self) -> Tensor | Rejected:
        found: Tensor | Rejected = self.mm.required_weights
        return found


def assembled(space_type: type[Kernel], at: str, pumped: bool = False) -> CoreComposition:
    """MatMul (or its wrapper, its MatMul ``at`` below it) on its partition's channels,
    its packed core chosen, the channels connected directly."""
    network = matmul_network(cast(type[MatMulKernel], space_type), **FACTS)
    network = choose(
        network,
        {
            f"matmul.{at}compute": "packed",
            f"matmul.{at}pe": 2,
            f"matmul.{at}simd": 2,
            f"matmul.{at}compute.packed.compute_pumping": pumped,
            f"matmul.{at}compute.packed.reducer": "tree",
        },
    )
    network = choose(with_channel_defaults(network), {"w.transport": "direct"})
    return with_direct_transports(with_adapter_memories(network))


def test_the_wrapper_exposes_matmul_s_ports_as_its_own() -> None:
    assert list(exposed_ports(WrappedMatMul)) == ["x", "w", "y"]
    assert [port_direction(WrappedMatMul, port) for port in "xwy"] == [
        port_direction(MatMulKernel, port) for port in "xwy"
    ]
    assert exposed_name(WrappedMatMul, "x") == "x"


def test_on_its_channels_the_wrapper_is_matmul_one_name_deeper() -> None:
    plain, wrapped = assembled(MatMulKernel, ""), assembled(WrappedMatMul, "mm.")
    # Its choices are MatMul's, owned where they are declared.
    assert [key for _, key in wrapped.chosen(["matmul"])] == [
        f"mm.{key}" for _, key in plain.chosen(["matmul"])
    ]
    point = wrapped.member("matmul")
    assert inspection.path(point.x) == "matmul.mm.compute.packed.x"
    # The same leaves and links, the core one name deeper.
    module, reference = built(wrapped, "matmul"), built(plain, "matmul")

    def deeper(label: str | None) -> str | None:
        return None if label is None else label.replace("matmul.compute", "matmul.mm.compute")

    assert labels(module) == [deeper(label) for label in labels(reference)]
    assert [leaf for _, leaf in module.fragment.instances] == [
        leaf for _, leaf in reference.fragment.instances
    ]
    assert {
        (deeper(link.source.instance), link.source.data, deeper(link.sink.instance), link.sink.data)
        for link in reference.fragment.links
    } == {
        (link.source.instance, link.source.data, link.sink.instance, link.sink.data)
        for link in module.fragment.links
    }
    assert module.abi == reference.abi
    # Each channel meets its port under the wrapper's name, its pins on the core.
    for channel, side, port in (("x", "consumer", "x"), ("w", "consumer", "w")):
        found = getattr(wrapped.neighbours(channel), side)
        assert (found.label, found.instance) == (f"matmul.{port}", "matmul.mm.compute.packed")
        assert found.pins == getattr(plain.neighbours(channel), side).pins
    # Its intake is MatMul's, under its ports' names.
    assert point.query(Kernel.intake) == plain.member("matmul").query(Kernel.intake)


@pytest.mark.parametrize("left", ["x", "w", "y"])
def test_a_member_s_port_the_wrapper_neither_exposes_nor_connects_is_refused(left: str) -> None:
    """A member's port is connected inside its holder, by a channel, or presented outside it,
    by an alias, and no holder ties a port off: a wrapper holding no channel that leaves one
    of MatMul's ports unexposed, an input or an output, is refused when its class is made,
    the port named (``port-unconnected``)."""

    def partial() -> type[Kernel]:
        class Partial(Kernel):
            id = "test.partial_matmul"
            platform: Platform = Param()
            x_tensor: Tensor = Param()
            weights: Tensor | KnownWeights = Param()

            mm = MatMulKernel(platform=platform, x_tensor=x_tensor, weights=weights)
            if left != "x":
                x = mm.x
            if left != "w":
                w = mm.w
            if left != "y":
                y = mm.y

        return Partial

    with refused("port-unconnected", error=DefinitionError, member="mm", port=left):
        partial()


def test_a_port_of_a_selector_s_case_its_holder_does_not_expose_is_refused() -> None:
    """A selector's member ports are each case's: MatMul without its alias of its core's
    weights (``w = compute.w``) leaves every core's ``w`` open, each refused by its case."""
    with pytest.raises(DefinitionError) as raised:

        class Unweighted(MatMulKernel):
            id = "test.unweighted_matmul"
            w = None  # type: ignore[assignment]

    assert [
        (each.code, dict(each.details)["member"], dict(each.details)["port"])
        for each in raised.value.findings
    ] == [
        ("port-unconnected", "compute", "packed.w"),
        ("port-unconnected", "compute", "int8_dsp58.w"),
    ]


@pytest.mark.parametrize("pumped", [False, True])
def test_the_wrapper_takes_the_doubled_clock_its_core_takes(pumped: bool) -> None:
    point = assembled(WrappedMatMul, "mm.", pumped=pumped).member("matmul")
    assert point.takes_doubled_clock is pumped
    assert point.mm.takes_doubled_clock is pumped
    assert point.clocking.doubling is False


def test_a_member_s_guard_is_read_only_where_it_is_its_holder_s_own_member() -> None:
    """Whether a member that may take the doubled clock or present a bus is placed is
    read from its guard (``when``) where its holder derives its own: the holder's own
    member, never another's (``member-guard-unread``)."""

    class Own(Kernel):
        id = "test.exposed.guard.own"
        flag: bool = Param(default=True)
        memory = MemStreamKernel(when=flag)
        output = memory.output

    # Its own flag is read: it derives whether it takes the doubled clock from the memory's.
    assert "takes_doubled_clock" in vars(Own)

    class Flags(Space):
        flag: bool = Param(default=True)

    with refused("member-guard-unread", error=DefinitionError):

        class Foreign(Kernel):
            id = "test.exposed.guard.foreign"
            flags = Flags()
            memory = MemStreamKernel(when=flags.flag)
