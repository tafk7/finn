# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One port class, and extents bound from the tensors a kernel's input ports read.

``AxiStreamPort`` presents its kernel's schedule through the indices it reads,
or a given sequence (exactly one). An input placed with a schedule exports its
read of the tensor, and its kernel binds each index's extent from those reads
(``LeafKernel.extents``): a kernel writes no extent getter, and its folding factors'
domains read ``extent_of`` members. A producer's tensor is its kernel's statement,
derived from those extents, and its element is that tensor's. An input without its
tensor is idle: an interface not used. A port's constraint counts toward its holder's
validity, while what the holder reads to bind its extents and check its clock are values:
a refused port still answers them.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP58, choose, dotp_extents, with_direct_transports
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Members,
    Param,
    Rejected,
    Space,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import Contribution
from finn.kernels.base import ACCESS, CLOCKED, PINS, LeafKernel, extent_of, interface
from finn.kernels.channels import Channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.port import INTEGER_POLICY, AxiStreamPort
from finn.kernels.utilization import RESOURCES_SEMANTICS
from finn.kernels.values.domains import Integer
from finn.platform.capabilities import Resources

b, s, c = Index("b"), Index("s"), Index("c")


class Pool(LeafKernel):
    """Global sum pooling, model only: Y[b, c] = sum_s X[b, s, c], PE channels a beat."""

    id = "test.accpool"
    rtl_module = "accpool_axi"
    x_tensor: Tensor = Param(required=False)

    channels = extent_of(c)
    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def schedule(self) -> Schedule | Rejected:
        return self.bound_schedule(order=(b, s, c), factors={c: self.pe})

    @view
    def y_tensor(self) -> Tensor | Rejected:
        shape = self.spanned((b, c))
        if isinstance(shape, Rejected):
            return shape
        return Tensor(shape, ScalarEncoding(DataType["INT8"]))

    x = AxiStreamPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        tensor=x_tensor,
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
    )
    y = AxiStreamPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        tensor=y_tensor,
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
    )
    x_end = interface("x")
    y_end = interface("y")

    @derived
    def parameters(self) -> dict[str, int | str]:
        return {"CHANNELS": self.channels, "PE": self.pe}

    @derived
    def sources(self) -> tuple[Contribution, ...]:
        return ()

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")


def tensor(shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType[dtype]))


X, Y = tensor((1, 4, 8), "INT4"), tensor((1, 8), "INT8")


def placing(kernel: Space, x_tensor: Tensor = X, y_tensor: Tensor | None = Y) -> Network:
    """``kernel`` (``kernel``) between two boundary channels, ``x`` and ``y`` (none without
    ``y_tensor``), of the tensors ``x_tensor`` and ``y_tensor``."""
    members: dict[str, Any] = {
        "x": wire(
            tensor=x_tensor,
            port="in0_V",
            consumer=Interface("kernel", "x_end"),
            platform=FULL_DSP58,
        )
    }
    if y_tensor is not None:
        members["y"] = wire(
            tensor=y_tensor,
            port="out0_V",
            producer=Interface("kernel", "y_end"),
            platform=FULL_DSP58,
        )
    return Network({**members, "kernel": kernel})


def pool(x_shape: tuple[int, ...] = (1, 4, 8), y_shape: tuple[int, ...] = (1, 8)) -> Network:
    x_tensor, y_tensor = tensor(x_shape, "INT4"), tensor(y_shape, "INT8")
    return placing(Pool(x_tensor=x_tensor), x_tensor, y_tensor)


def coded_messages(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected), result
    return {(finding.code, finding.message) for finding in result.findings}


def test_a_kernel_binds_its_extents_from_its_inputs_and_its_folding_factors_divide_them() -> None:
    kernel = pool().member("kernel")
    assert kernel.extents == {b: 1, s: 4, c: 8}
    assert kernel.field(Pool.pe).candidates().value == (1, 2, 4, 8)
    kernel = choose(pool(), {"kernel.pe": 4}).member("kernel")
    x, y = kernel.x.presented.form, kernel.y.presented.form
    assert (x.beats, x.lanes, y.beats, y.lanes) == (8, 4, 2, 4)
    assert dict(kernel.module.parameters) == {"CHANNELS": 8, "PE": 4}
    assert kernel.cycles == 8  # the most beats a port carries: x's


def test_a_producers_tensor_is_its_kernels_and_its_channel_refuses_another() -> None:
    point = with_direct_transports(choose(pool(x_shape=(1, 4, 6)), {"kernel.pe": 2}))
    kernel = point.member("kernel")
    assert kernel.y_tensor == tensor((1, 6), "INT8")
    assert kernel.y.element == ScalarEncoding(DataType["INT8"])
    refused = point.member("y").query(Channel.netlist)
    assert "channel-tensor" in {code for code, _ in coded_messages(refused)}


def test_an_input_without_its_tensor_is_idle_and_carries_one_lane() -> None:
    point = design_space(Pool())
    assert point.x.idle
    ((code, message),) = coded_messages(point.x.query(AxiStreamPort.lane_count))
    assert code == "port-idle" and message.startswith("s_axis_input: an idle port carries one")


def test_extent_of_is_a_named_member() -> None:
    class Inline(Pool):
        id = "test.accpool.inline"
        pe: int = Decision(domain=divisors_of(extent_of(c)))

    with refused("reference-not-in-scope", error=DefinitionError, declaration="pe"):
        placing(Inline(x_tensor=X))


def test_a_port_presents_a_schedule_or_a_sequence_not_both() -> None:
    class Both(Pool):
        id = "test.accpool.both"
        x = AxiStreamPort(
            name="s_axis_input",
            endpoint=Endpoint.TARGET,
            tensor=Pool.x_tensor,
            schedule=Pool.schedule,
            sequence=BeatSequence(vector_major((1, 4, 8), 8)),
            index=(b, s, c),
            lanes=(c,),
        )

    point = choose(placing(Both(x_tensor=X)), {"kernel.pe": 8})
    ((code, message),) = coded_messages(point.member("kernel").x.query(AxiStreamPort.presented))
    assert code == "port-presentation"
    assert message.startswith("s_axis_input: exactly one of a schedule or a sequence")


def test_a_producers_element_is_its_tensors() -> None:
    class Stated(Pool):
        id = "test.accpool.stated"
        y = AxiStreamPort(
            name="m_axis_output",
            endpoint=Endpoint.INITIATOR,
            tensor=Pool.y_tensor,
            schedule=Pool.schedule,
            index=(b, c),
            lanes=(c,),
            reduces=(s,),
            dtype=DataType["INT9"],
        )

    point = choose(placing(Stated(x_tensor=X)), {"kernel.pe": 4})
    ((code, message),) = coded_messages(point.member("kernel").y.query(AxiStreamPort.element))
    assert code == "port-element"
    assert message.startswith("m_axis_output: a producer's element is its tensor's")


def test_dotp_binds_from_its_inputs_through_the_dense_view() -> None:
    x, w = tensor((2, 3, 4), "INT3"), tensor((12, 4), "INT3")
    point = design_space(
        PackedDotpKernel(
            form=Form.DENSE,
            reshape_activations=True,
            platform=FULL_DSP58,
            result_range=ordinary_integer_bounds(DataType["INT9"]),
            x_tensor=x,
            w_tensor=w,
            **dotp_extents(w),
        )
    )
    assert (point.rows, point.outputs, point.reduction) == (2, 4, 12)
    assert point.y_tensor == tensor((2, 4), "INT9")

    # A too-wide activation tensor is refused, not left partly unread.
    wide = design_space(
        PackedDotpKernel(
            platform=FULL_DSP58,
            result_range=ordinary_integer_bounds(DataType["INT9"]),
            x_tensor=tensor((2, 14), "INT3"),
            w_tensor=w,
            **dotp_extents(w),
        )
    )
    assert coded_messages(wide.query(PackedDotpKernel.extents)) == {
        ("kernel-extents", "k is 14 (x axis 1) and 12 (w axis 0)")
    }


class Holding(Space):
    """A holder of one input port, reading what a leaf reads of its ports: their reads of
    their tensors (``ACCESS``), their clocks (``CLOCKED``) and their pins (``PINS``)."""

    admits: Integer | None = Param(default=None, semantics=INTEGER_POLICY)
    x = AxiStreamPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        tensor=X,
        sequence=BeatSequence(vector_major((1, 4, 8), 8)),
        index=(b, s, c),
        lanes=(c,),
        admits=admits,
    )
    accesses = Members(ACCESS)
    clocks = Members(CLOCKED)
    pins = Members(PINS)


def test_a_refused_port_still_exports_its_access_and_its_clock_as_values() -> None:
    taken = design_space(Holding())
    assert inspection.validity(taken).status == "valid"
    assert isinstance(taken.query(Holding.pins), Available)

    point = design_space(Holding(admits=Integer(max_bits=2)))
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["x.element_taken"]
    # The values its holder reads answer all the same: its extents and clock stand.
    ((access,),) = [[item.value for item in point.accesses]]
    assert (access.name, access.shape) == ("s_axis_input", (1, 4, 8))
    assert [(item.node, item.value) for item in point.clocks] == [("x", "ap_clk")]
    # Its pins are a product, refused with it.
    assert "dtype-maximum-bits" in {code for code, _ in coded_messages(point.query(Holding.pins))}
