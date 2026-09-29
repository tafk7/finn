"""P0 probe kernels: accpool_axi on the D4 helpers (`extents` bound from its ports)."""

from __future__ import annotations

from _bind import BoundKernel, extent_of
from qonnx.core.datatype import DataType

from finn.core.space import Decision, Param, Space, derived, design_space, divisors_of
from finn.dataflow.schedule import SCHEDULE, Index, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.port import ScheduledPort
from finn.kernels.streams import Stream as KernelStream

b, s, c = Index("b"), Index("s"), Index("c")


class Pool(BoundKernel):
    id = "probe.pool"
    module = "accpool_axi"
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    channels = extent_of(c)

    pe: int = Decision(domain=divisors_of(channels))

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:  # bound_schedule(beats=(b, s, c), folds={c: pe})
        return Schedule(self.extents, folds={c: self.pe}, beats=(b, s, c))

    x = ScheduledPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        stream=x_stream,
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
    )
    y = ScheduledPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        stream=y_stream,
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
    )


def stream(shape: tuple[int, ...], dtype: str, port: str) -> KernelStream:
    return KernelStream(tensor=Tensor(shape, ScalarEncoding(DataType[dtype])), port=port)


def placed_pool(xshape: tuple[int, ...], yshape: tuple[int, ...] = (1, 8)) -> Space:
    class Placed(Space):
        x = stream(xshape, "INT4", "in0_V")
        y = stream(yshape, "INT8", "out0_V")
        pool = Pool(x_stream=x, y_stream=y)

    return design_space(Placed())
