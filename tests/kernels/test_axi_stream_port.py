# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One port declaration, and extents bound from the tensors a kernel's input ports read.

``AxiStreamPort`` presents its kernel's schedule through the indices it reads,
or a given sequence (exactly one). An input placed with a schedule states its read of
the tensor (``x_access``), and its kernel binds each index's extent from those reads
(``KernelSpace.extents``): a kernel writes no extent getter, and its folding factors'
domains read ``extent_of`` members. A producer's tensor is its kernel's statement,
derived from those extents, and its element is that tensor's. An input without its
tensor is idle: a port not used. A port's rule (``x_element_taken``) is its
kernel's constraint, counted in its validity, while what the kernel reads to bind its
extents are values: a refused port still answers them.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP58, by_hand, choose, dotp_extents, keyed
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Param,
    Rejected,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import Access, Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import KernelSpace, extent_of
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.library import ModuleSpec, _each, assemble_module, library_module
from finn.kernels.partition import Partition
from finn.kernels.port import lane_count, pins
from finn.kernels.port_model import IndexedPort, ScheduledPort
from finn.kernels.sides import Side
from finn.kernels.values.domains import Integer
from finn.platform.capabilities import Resources

b, s, c = Index("b"), Index("s"), Index("c")


class Pool(KernelSpace):
    """Global sum pooling, model only: Y[b, c] = sum_s X[b, s, c], PE channels a beat."""

    id = "test.accpool"
    x_tensor: Tensor | None = Param(default=None)

    channels = extent_of(c)
    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def schedule(self) -> Schedule | Rejected:
        return self.bound_schedule(order=(b, s, c), factors={c: self.pe})

    @derived
    def y_tensor(self) -> Tensor | Rejected:
        shape = self.spanned((b, c))
        if isinstance(shape, Rejected):
            return shape
        return Tensor(shape, ScalarEncoding(DataType["INT8"]))

    x = IndexedPort(
        name="s_axis_input",
        label="x",
        endpoint=Endpoint.TARGET,
        tensor=x_tensor,
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
    )
    y = ScheduledPort(
        name="m_axis_output",
        label="y",
        endpoint=Endpoint.INITIATOR,
        tensor=y_tensor,
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"CHANNELS": self.channels, "PE": self.pe}

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module_spec = ModuleSpec(
        "accpool_axi",
        constant_sources=(
            CopiedSource("finnlib", "rtl/accpool.sv", provides=("module:accpool_axi",)),
        ),
    )

    def _accesses(self) -> list[Access]:
        return [self.x.access] if self.x.binds else []

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        parts = _each(
            (
                pins(self.x.physical_spec(idle=False), self.x.stream, self.x.clocking),
                pins(self.y.physical_spec(idle=False), self.y.stream, self.y.clocking),
            )
        )
        if isinstance(parts, Rejected):
            return parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        assert sources is not None
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            tuple(parts),
            sources,
            (),
            (Held(), Held()),
            (True, True),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return max(self.x.stream.sequence.form.beats, self.y.stream.sequence.form.beats)


def tensor(shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType[dtype]))


X, Y = tensor((1, 4, 8), "INT4"), tensor((1, 8), "INT8")


def placing(kernel: Facts, x_tensor: Tensor = X, y_tensor: Tensor | None = Y) -> Partition:
    """``kernel`` (``kernel``) between two boundary channels, ``x`` and ``y`` (none without
    ``y_tensor``), of the tensors ``x_tensor`` and ``y_tensor``."""
    members: dict[str, Any] = {
        "x": channel(
            Side.boundary("in0_V"), Port("kernel", "x"), tensor=x_tensor, platform=FULL_DSP58
        )
    }
    if y_tensor is not None:
        members["y"] = channel(
            Port("kernel", "y"), Side.boundary("out0_V"), tensor=y_tensor, platform=FULL_DSP58
        )
    return by_hand({**members, "kernel": kernel})


def pool(x_shape: tuple[int, ...] = (1, 4, 8), y_shape: tuple[int, ...] = (1, 8)) -> Partition:
    x_tensor, y_tensor = tensor(x_shape, "INT4"), tensor(y_shape, "INT8")
    return placing(Facts(Pool, {"x_tensor": x_tensor}), x_tensor, y_tensor)


def coded_messages(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected), result
    return {(finding.code, finding.message) for finding in result.findings}


def test_a_kernel_binds_its_extents_from_its_inputs_and_its_folding_factors_divide_them() -> None:
    kernel = pool().member("kernel")
    assert kernel.extents == {b: 1, s: 4, c: 8}
    assert kernel.field(Pool.pe).candidates().value == (1, 2, 4, 8)
    kernel = choose(pool(), {"kernel.pe": 4}).member("kernel")
    x, y = kernel.x.stream.sequence.form, kernel.y.stream.sequence.form
    assert (x.beats, x.lanes, y.beats, y.lanes) == (8, 4, 2, 4)
    assert dict(library_module(kernel).parameters) == {"CHANNELS": 8, "PE": 4}
    assert kernel.cycles == 8  # the most beats a port carries: x's


def test_a_producers_tensor_is_its_kernels_and_its_channel_refuses_another() -> None:
    """The kernel's output states its tensor, ``(1, 6)``; the channel on it carries
    another, ``(1, 8)``, so the fold that gives the output its stream is refused by the
    channel's rule (``channel-tensor``), as on a graph's partition."""
    network = pool(x_shape=(1, 4, 6))
    kernel = network.member("kernel")
    assert kernel.y_tensor == tensor((1, 6), "INT8")
    assert kernel.y.element == ScalarEncoding(DataType["INT8"])
    trial = network.trial(keyed({"kernel.pe": 2}))
    assert list(trial.refused) == [("y", "well_formed")]
    assert "channel-tensor" in {
        code for code, _ in coded_messages(Rejected(trial.refused[("y", "well_formed")]))
    }


def test_an_input_without_its_tensor_is_idle_and_carries_one_lane() -> None:
    point = design_space(Pool())
    assert not Pool.x.placed(point, "x")
    ((code, message),) = coded_messages(point.query(Pool.x.element))
    assert code == "port-element" and message == "s_axis_input: an idle port states its dtype"
    ((code, message),) = coded_messages(lane_count(Pool.x, ScalarEncoding(DataType["INT4"])))
    assert code == "port-idle" and message.startswith("s_axis_input: an idle port carries one")


def test_extent_of_is_a_named_member() -> None:
    class Inline(Pool):
        id = "test.accpool.inline"
        pe: int = Decision(domain=divisors_of(extent_of(c)))

    with refused("reference-not-in-scope", error=DefinitionError, declaration="pe"):
        placing(Facts(Inline, {"x_tensor": X}))


def test_a_port_presents_a_schedule_or_a_sequence_not_both() -> None:
    class Both(Pool):
        id = "test.accpool.both"
        x = IndexedPort(
            name="s_axis_input",
            label="x",
            endpoint=Endpoint.TARGET,
            tensor=Pool.x_tensor,
            schedule=Pool.schedule,
            sequence=BeatSequence(vector_major((1, 4, 8), 8)),
            index=(b, s, c),
            lanes=(c,),
        )

    # The fold that places the port is refused by it, its stream's rule.
    trial = placing(Facts(Both, {"x_tensor": X})).trial(keyed({"kernel.pe": 8}))
    ((code, message),) = coded_messages(Rejected(trial.refused[("kernel", "x.stream")]))
    assert code == "port-presentation"
    assert message.startswith("s_axis_input: exactly one of a schedule or a sequence")


def test_a_producers_element_is_its_tensors() -> None:
    class Stated(Pool):
        id = "test.accpool.stated"
        y = ScheduledPort(
            name="m_axis_output",
            label="y",
            endpoint=Endpoint.INITIATOR,
            tensor=Pool.y_tensor,
            schedule=Pool.schedule,
            index=(b, c),
            lanes=(c,),
            reduces=(s,),
            dtype=DataType["INT9"],
        )

    trial = placing(Facts(Stated, {"x_tensor": X})).trial(keyed({"kernel.pe": 4}))
    ((code, message),) = coded_messages(Rejected(trial.refused[("kernel", "y.element")]))
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
            pe=1,
            simd=1,
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
            pe=1,
            simd=1,
        )
    )
    assert coded_messages(wide.query(PackedDotpKernel.extents)) == {
        ("kernel-extents", "k is 14 (x axis 1) and 12 (w axis 0)")
    }


class Holding(KernelSpace):
    """A kernel of one input port, given the integer policy its hardware takes."""

    id = "test.holding"
    admits: Integer | None = Param(default=None)
    x = IndexedPort(
        name="s_axis_input",
        label="x",
        endpoint=Endpoint.TARGET,
        tensor=X,
        sequence=BeatSequence(vector_major((1, 4, 8), 8)),
        index=(b, s, c),
        lanes=(c,),
        admits=admits,
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module_spec = ModuleSpec(
        "holding",
        constant_sources=(CopiedSource("finnlib", "rtl/holding.sv", provides=("module:holding",)),),
    )

    def _accesses(self) -> list[Access]:
        return [self.x.access] if self.x.binds else []

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        found = pins(self.x.physical_spec(idle=False), self.x.stream, self.x.clocking)
        if isinstance(found, Rejected):
            return found
        spec = type(self).module_spec
        sources = spec.constant_sources
        assert sources is not None
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            (found,),
            sources,
            (),
            (Held(),),
            (True,),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return self.x.stream.sequence.form.beats


def test_a_refused_port_still_states_its_access_and_its_kernel_its_extents() -> None:
    taken = design_space(Holding())
    assert inspection.validity(taken).status == "valid"
    assert isinstance(taken.query(Holding.netlist), Available)

    point = design_space(Holding(admits=Integer(max_bits=2)))
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["x.element_taken"]
    # The values its kernel reads answer all the same: its read of its tensor, its extents.
    access = point.x.access
    assert (access.name, access.shape) == ("x", (1, 4, 8))
    assert point.extents == {b: 1, s: 4, c: 8}
    # Its module is a product, refused with it.
    assert "dtype-maximum-bits" in {
        code for code, _ in coded_messages(point.query(Holding.netlist))
    }
