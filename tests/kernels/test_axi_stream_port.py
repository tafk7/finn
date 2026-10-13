# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One port class, and extents bound from the tensors a kernel's ports read.

``AxiStreamPort`` presents its kernel's schedule through the indices it reads,
or a given sequence (exactly one). Placed with a schedule it exports its read
of the tensor, and its kernel binds each index's extent from those reads
(``Kernel.extents``): a kernel writes no extent getter, its folding factors'
domains read ``extent_of`` members, and ports disagreeing on an extent are
refused (``kernel-extents``). An idle port carries the lanes of its folding
factors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Space,
    derived,
    design_space,
    divisors_of,
)
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel, extent_of, interface
from finn.kernels.channels import Channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.port import AxiStreamPort
from kernels.helpers import FULL_DSP58, choose, with_direct_transports

b, s, c = Index("b"), Index("s"), Index("c")


class Pool(Kernel):
    """Global sum pooling, model only: Y[b, c] = sum_s X[b, s, c], PE channels a beat."""

    id = "test.accpool"
    rtl_module = "accpool_axi"
    x_tensor: Tensor = Param(required=False)
    y_tensor: Tensor = Param(required=False)

    channels = extent_of(c)
    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def factors(self) -> dict[Index, int]:
        return {c: self.pe}

    @derived
    def schedule(self) -> Schedule | Rejected:
        return self.bound_schedule(order=(b, s, c), factors=self.factors)

    x = AxiStreamPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        tensor=x_tensor,
        schedule=schedule,
        factors=factors,
        index=(b, s, c),
        lanes=(c,),
    )
    y = AxiStreamPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        tensor=y_tensor,
        schedule=schedule,
        factors=factors,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
        dtype=DataType["INT8"],
    )
    x_end = interface("x")
    y_end = interface("y")

    def parameters(self) -> Mapping[str, int | str]:
        return {"CHANNELS": self.channels, "PE": self.pe}


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
    return placing(Pool(x_tensor=x_tensor, y_tensor=y_tensor), x_tensor, y_tensor)


def coded_messages(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected), result
    return {(finding.code, finding.message) for finding in result.findings}


def test_a_kernel_binds_its_extents_from_its_ports_and_its_folding_factors_divide_them() -> None:
    kernel = pool().member("kernel")
    assert kernel.extents == {b: 1, s: 4, c: 8}
    assert kernel.field(Pool.pe).candidates().value == (1, 2, 4, 8)
    kernel = choose(pool(), {"kernel.pe": 4}).member("kernel")
    x, y = kernel.x.presented.form, kernel.y.presented.form
    assert (x.beats, x.lanes, y.beats, y.lanes) == (8, 4, 2, 4)
    assert dict(kernel.module.parameters) == {"CHANNELS": 8, "PE": 4}


def test_ports_that_disagree_on_an_extent_are_refused() -> None:
    kernel = pool(x_shape=(1, 4, 6)).member("kernel")
    expected = {("kernel-extents", "c is 6 (x axis 2) and 8 (y axis 1)")}
    assert coded_messages(kernel.query(Pool.extents)) == expected
    assert coded_messages(kernel.query(Pool.module)) == expected


def test_an_idle_port_carries_the_lanes_of_its_folding_factors() -> None:
    kernel = choose(placing(Pool(x_tensor=X), y_tensor=None), {"kernel.pe": 2}).member("kernel")
    assert kernel.y.idle and kernel.y.lane_count == 2
    assert kernel.y.axis.elements_per_beat == 2


def test_an_unplaced_kernel_binds_no_extent_and_says_which() -> None:
    point = design_space(Pool())
    assert point.extents == {}
    refused = point.field(Pool.pe).candidates()
    assert coded_messages(refused) == {("kernel-extents", "c is bound by no placed port")}


def test_extent_of_is_a_named_member() -> None:
    class Inline(Pool):
        id = "test.accpool.inline"
        pe: int = Decision(domain=divisors_of(extent_of(c)))

    with pytest.raises(DefinitionError, match="is not a member of this scope"):
        placing(Inline(x_tensor=X, y_tensor=Y))


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

    point = choose(placing(Both(x_tensor=X, y_tensor=Y)), {"kernel.pe": 8})
    ((code, message),) = coded_messages(point.member("kernel").x.query(AxiStreamPort.presented))
    assert code == "port-presentation"
    assert message.startswith("s_axis_input: exactly one of a schedule or a sequence")


def test_a_stated_element_is_the_ports_and_its_stream_refuses_another() -> None:
    kernel = Pool(x_tensor=X, y_tensor=Y)
    kernel.y.dtype = DataType["INT9"]  # a declaration may pin what the producer states
    point = with_direct_transports(choose(placing(kernel), {"kernel.pe": 4}))
    assert point.member("kernel").y.element == ScalarEncoding(DataType["INT9"])
    refused = point.member("y").query(Channel.netlist)
    assert "channel-tensor" in {code for code, _ in coded_messages(refused)}


def test_dotp_binds_from_its_ports_through_the_dense_view() -> None:
    x, w, y = tensor((2, 3, 4), "INT3"), tensor((12, 4), "INT3"), tensor((2, 4), "INT9")
    point = design_space(
        PackedDotpKernel(
            form=Form.DENSE,
            reshape_activations=True,
            platform=FULL_DSP58,
            result_range=ordinary_integer_bounds(DataType["INT9"]),
            x_tensor=x,
            w_tensor=w,
            y_tensor=y,
        )
    )
    assert (point.rows, point.outputs, point.reduction) == (2, 4, 12)

    # A too-wide activation tensor is refused, not left partly unread.
    wide = design_space(
        PackedDotpKernel(
            platform=FULL_DSP58,
            result_range=ordinary_integer_bounds(DataType["INT9"]),
            x_tensor=tensor((2, 14), "INT3"),
            w_tensor=w,
            y_tensor=y,
        )
    )
    assert coded_messages(wide.query(PackedDotpKernel.extents)) == {
        ("kernel-extents", "k is 14 (x axis 1) and 12 (w axis 0)")
    }


def test_a_producer_states_its_element() -> None:
    class Silent(Pool):
        id = "test.accpool.silent"
        y = AxiStreamPort(
            name="m_axis_output",
            endpoint=Endpoint.INITIATOR,
            tensor=Pool.y_tensor,
            schedule=Pool.schedule,
            index=(b, c),
            lanes=(c,),
            reduces=(s,),
        )

    point = choose(placing(Silent(x_tensor=X, y_tensor=Y)), {"kernel.pe": 4})
    ((code, message),) = coded_messages(point.member("kernel").y.query(AxiStreamPort.element))
    assert code == "port-element" and message.startswith("m_axis_output: a producer states")
