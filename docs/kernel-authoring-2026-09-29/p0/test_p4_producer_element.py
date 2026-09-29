"""P0.4: a producer states its element without its output stream (plan D5, G0.5).

`AccPool.y` states `dtype=sum_dtype`, derived from the kernel's facts and its input
element only. With `y_stream` absent (the port idle) the kernel still reports the
element. `ProducerPort` is the D5 port on today's `ScheduledPort`: its element is
its `dtype` placed or idle, so a stream carrying another element refuses it
(`stream-tensor`), as code-quality C does for `GivenPort`.
"""

from __future__ import annotations

from math import ceil, log2

from _bind import BoundKernel, extent_of
from _pool import b, c, s, stream
from qonnx.core.datatype import DataType

from finn.core.space import Param, Rejected, Space, derived, design_space
from finn.core.space.results import Available, Unresolved
from finn.dataflow.datatypes import QONNXDataType, ordinary_integer_bounds
from finn.dataflow.schedule import SCHEDULE, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import SCALAR_ENCODING, ScalarEncoding
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import ScheduledPort
from finn.kernels.streams import Stream as KernelStream


class ProducerPort(ScheduledPort):
    """D5: a producing port's element is its stated `dtype`, placed or idle."""

    @derived(semantics=SCALAR_ENCODING)
    def element(self) -> ScalarEncoding | Rejected:
        return ScalarEncoding.admit(self.dtype)


class AccPool(BoundKernel):
    id = "probe.accpool"
    module = "accpool_axi"
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    pixels = extent_of(s)
    channels = extent_of(c)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def sum_dtype(self) -> QONNXDataType:
        """What the RTL emits: the input element's range times the pixels, signed."""
        low, high = ordinary_integer_bounds(self.x.element.dtype)
        need = max(abs(low), high) * self.pixels
        return DataType[f"INT{ceil(log2(need + 1)) + 1}"]

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        return Schedule(self.extents, folds={c: 2}, beats=(b, s, c))

    x = ScheduledPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        stream=x_stream,
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
    )
    y = ProducerPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        stream=y_stream,
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
        dtype=sum_dtype,
    )


def with_output(y_dtype: str | None) -> Space:
    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        if y_dtype is None:
            pool = AccPool(x_stream=x)  # the output unplaced: y idle
        else:
            y = stream((1, 8), y_dtype, "out0_V")
            pool = AccPool(x_stream=x, y_stream=y)

    return design_space(Placed())


def test_the_output_element_is_stated_with_the_output_stream_absent() -> None:
    point = with_output(None)
    assert point.pool.y.idle is True
    assert point.pool.extents == {b: 1, s: 4, c: 8}  # bound from x alone
    # INT4 is [-8, 7]; 4 pixels sum to [-32, 28]: INT7.
    assert point.pool.sum_dtype == DataType["INT7"]
    assert point.pool.y.element == ScalarEncoding(DataType["INT7"])


def test_a_stream_of_that_element_accepts_the_producer() -> None:
    point = with_output("INT7")
    assert isinstance(point.y.query(KernelStream.connection), Available)
    assert point.pool.y.sequence.form.beats == 4


def test_a_stream_of_another_element_refuses_the_producer() -> None:
    point = with_output("INT8")
    refused = point.y.query(KernelStream.connection)
    assert isinstance(refused, Rejected)
    # The stream's well_formed (stream-tensor) and the producer-to-boundary element
    # check (stream-element) both refuse.
    assert {(f.code, f.message) for f in refused.findings} == {
        ("stream-tensor", "pool.y carries INT7; the stream carries INT8"),
        ("stream-element", "stream-element: INT7 cannot feed INT8"),
    }


def test_today_a_placed_scheduled_port_takes_the_streams_element() -> None:
    """The contrast: without D5 the stream's element wins silently (no refusal)."""

    class Today(AccPool):
        id = "probe.accpool_today"
        y = ScheduledPort(
            name="m_axis_output",
            endpoint=Endpoint.INITIATOR,
            stream=AccPool.y_stream,
            schedule=AccPool.schedule,
            index=(b, c),
            lanes=(c,),
            reduces=(s,),
            dtype=AccPool.sum_dtype,
        )

    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        pool = Today(x_stream=x, y_stream=y)

    point = design_space(Placed())
    assert point.pool.y.element == ScalarEncoding(DataType["INT8"])
    assert isinstance(point.y.query(KernelStream.connection), Available)


def test_a_dtype_reading_its_own_output_stream_has_no_element_unplaced() -> None:
    """Why the rule: a producer dtype read from its own output stream is unknown until placed."""

    class ReadsOwn(AccPool):
        id = "probe.accpool_reads_own"

        @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        def sum_dtype(self) -> QONNXDataType:
            return self.y_stream.tensor.element.dtype

    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        pool = ReadsOwn(x_stream=x)

    answer = design_space(Placed()).pool.y.query(ProducerPort.element)
    assert isinstance(answer, Unresolved)
    assert [(f.code, f.owner) for f in answer.findings] == [("input-unsupplied", "pool.y_stream")]
