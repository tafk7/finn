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
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel, extent_of
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.port import AxiStreamPort
from kernels.helpers import FULL_DSP58, with_direct_transports

b, s, c = Index("b"), Index("s"), Index("c")


class Pool(Kernel):
    """Global sum pooling, model only: Y[b, c] = sum_s X[b, s, c], PE channels a beat."""

    id = "test.accpool"
    rtl_module = "accpool_axi"
    x_channel: Channel = Param(required=False)
    y_channel: Channel = Param(required=False)

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
        channel=x_channel,
        schedule=schedule,
        factors=factors,
        index=(b, s, c),
        lanes=(c,),
    )
    y = AxiStreamPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        channel=y_channel,
        schedule=schedule,
        factors=factors,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
        dtype=DataType["INT8"],
    )

    def parameters(self) -> Mapping[str, int | str]:
        return {"CHANNELS": self.channels, "PE": self.pe}


def stream(shape: tuple[int, ...], dtype: str, port: str) -> Channel:
    return Channel(
        tensor=Tensor(shape, ScalarEncoding(DataType[dtype])), port=port, platform=FULL_DSP58
    )


def pool(x_shape: tuple[int, ...] = (1, 4, 8), y_shape: tuple[int, ...] = (1, 8)) -> Space:
    class Placed(Space):
        x = stream(x_shape, "INT4", "in0_V")
        y = stream(y_shape, "INT8", "out0_V")
        kernel = Pool(x_channel=x, y_channel=y)

    return design_space(Placed())


def codes(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected), result
    return {(finding.code, finding.message) for finding in result.findings}


def test_a_kernel_binds_its_extents_from_its_ports_and_its_folding_factors_divide_them() -> None:
    point = pool()
    assert point.kernel.extents == {b: 1, s: 4, c: 8}
    assert point.kernel.field(Pool.pe).candidates().value == (1, 2, 4, 8)
    point = commit(point, {"kernel.pe": 4})
    x, y = point.kernel.x.presented.form, point.kernel.y.presented.form
    assert (x.beats, x.lanes, y.beats, y.lanes) == (8, 4, 2, 4)
    assert dict(point.kernel.module.parameters) == {"CHANNELS": 8, "PE": 4}


def test_ports_that_disagree_on_an_extent_are_refused() -> None:
    point = pool(x_shape=(1, 4, 6))
    expected = {("kernel-extents", "c is 6 (x axis 2) and 8 (y axis 1)")}
    assert codes(point.kernel.query(Pool.extents)) == expected
    assert codes(point.kernel.query(Pool.module)) == expected


def test_an_idle_port_carries_the_lanes_of_its_folding_factors() -> None:
    class Half(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        kernel = Pool(x_channel=x)

    point = commit(design_space(Half()), {"kernel.pe": 2})
    assert point.kernel.y.idle and point.kernel.y.lane_count == 2
    assert point.kernel.y.axis.elements_per_beat == 2


def test_an_unplaced_kernel_binds_no_extent_and_says_which() -> None:
    point = design_space(Pool())
    assert point.extents == {}
    refused = point.field(Pool.pe).candidates()
    assert codes(refused) == {("kernel-extents", "c is bound by no placed port")}


def test_extent_of_is_a_named_member() -> None:
    class Inline(Pool):
        id = "test.accpool.inline"
        pe: int = Decision(domain=divisors_of(extent_of(c)))

    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        kernel = Inline(x_channel=x, y_channel=y)

    with pytest.raises(DefinitionError, match="is not a member of this scope"):
        design_space(Placed())


def test_a_port_presents_a_schedule_or_a_sequence_not_both() -> None:
    class Both(Pool):
        id = "test.accpool.both"
        x = AxiStreamPort(
            name="s_axis_input",
            endpoint=Endpoint.TARGET,
            channel=Pool.x_channel,
            schedule=Pool.schedule,
            sequence=BeatSequence(vector_major((1, 4, 8), 8)),
            index=(b, s, c),
            lanes=(c,),
        )

    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        kernel = Both(x_channel=x, y_channel=y)

    point = commit(design_space(Placed()), {"kernel.pe": 8})
    ((code, message),) = codes(point.kernel.x.query(AxiStreamPort.presented))
    assert code == "port-presentation"
    assert message.startswith("s_axis_input: a schedule or a sequence, not both")


def test_a_stated_element_is_the_ports_and_its_stream_refuses_another() -> None:
    class Stated(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        kernel = Pool(x_channel=x, y_channel=y)
        kernel.y.dtype = DataType["INT9"]  # a parent may pin what the producer states

    point = commit(with_direct_transports(design_space(Stated())), {"kernel.pe": 4})
    assert point.kernel.y.element == ScalarEncoding(DataType["INT9"])
    refused = point.y.query(Channel.netlist)
    assert "channel-tensor" in {code for code, _ in codes(refused)}


def test_dotp_binds_from_its_ports_through_the_dense_view() -> None:
    class Placed(Space):
        x = stream((2, 3, 4), "INT3", "in0_V")
        w = stream((12, 4), "INT3", "in1_V")
        y = stream((2, 4), "INT9", "out0_V")
        compute = PackedDotpKernel(
            form=Form.DENSE,
            reshape_activations=True,
            platform=FULL_DSP58,
            result_dtype=DataType["INT9"],
            x_channel=x,
            w_channel=w,
            y_channel=y,
        )

    point = design_space(Placed()).compute
    assert (point.rows, point.outputs, point.reduction) == (2, 4, 12)

    # A too-wide activation tensor is refused, not left partly unread.
    class Wide(Placed):
        x = stream((2, 14), "INT3", "in0_V")
        compute = PackedDotpKernel(
            platform=FULL_DSP58,
            result_dtype=DataType["INT9"],
            x_channel=x,
            w_channel=Placed.w,
            y_channel=Placed.y,
        )

    wide = design_space(Wide()).compute
    assert codes(wide.query(PackedDotpKernel.extents)) == {
        ("kernel-extents", "k is 14 (x axis 1) and 12 (w axis 0)")
    }


def test_a_producer_states_its_element() -> None:
    class Silent(Pool):
        id = "test.accpool.silent"
        y = AxiStreamPort(
            name="m_axis_output",
            endpoint=Endpoint.INITIATOR,
            channel=Pool.y_channel,
            schedule=Pool.schedule,
            index=(b, c),
            lanes=(c,),
            reduces=(s,),
        )

    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        kernel = Silent(x_channel=x, y_channel=y)

    point = commit(design_space(Placed()), {"kernel.pe": 4})
    ((code, message),) = codes(point.kernel.y.query(AxiStreamPort.element))
    assert code == "port-element" and message.startswith("m_axis_output: a producer states")
