"""Probe: can a kernel's schedule read its own ports' static index declarations (no cycle)?"""
from qonnx.core.datatype import DataType
from finn.core.space import Decision, Param, Space, derived, design_space, divisors_of, Rejected, reject
from finn.dataflow.schedule import SCHEDULE, Index, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.port import ScheduledPort
from finn.kernels.streams import Stream as KStream

b, s, c = Index("b"), Index("s"), Index("c")

def bind(accesses):
    """Extents of every index that alone addresses a tensor axis; refused on disagreement."""
    extents = {}
    for shape, index in accesses:
        if len(shape) != len(index):
            raise ValueError(f"{len(index)} indices for a rank-{len(shape)} tensor")
        for extent, i in zip(shape, index):
            if isinstance(i, Index):
                if extents.setdefault(i, extent) != extent:
                    raise ValueError(f"{i!r} is {extents[i]} on one port and {extent} on another")
    return extents

class Pool(Kernel):
    id = "probe.pool"; module = "accpool_axi"
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    @derived
    def extents(self) -> dict:
        return bind([(self.x_stream.tensor.shape, self.x.index), (self.y_stream.tensor.shape, self.y.index)])

    @derived
    def channels(self) -> int:
        return self.extents[c]

    pe: int = Decision(domain=divisors_of(channels))

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        return Schedule(self.extents, folds={c: self.pe}, beats=(b, s, c))

    x = ScheduledPort(name="s_axis_input", endpoint=Endpoint.TARGET, stream=x_stream,
                      schedule=schedule, index=(b, s, c), lanes=(c,))
    y = ScheduledPort(name="m_axis_output", endpoint=Endpoint.INITIATOR, stream=y_stream,
                      schedule=schedule, index=(b, c), lanes=(c,), reduces=(s,))

def placed(xshape):
    class P(Space):
        x = KStream(tensor=Tensor(xshape, ScalarEncoding(DataType["INT4"])), port="in0_V")
        y = KStream(tensor=Tensor((1, 8), ScalarEncoding(DataType["INT8"])), port="out0_V")
        pool = Pool(x_stream=x, y_stream=y)
    return design_space(P())

p = placed((1, 4, 8))
print("pe domain:", p.pool.field(type(p.pool).pe).state if False else "open")
q = commit(p, {"pool.pe": 4})
print("extents:", q.pool.extents, "| x:", q.pool.x.sequence.form.beats, "beats x", q.pool.x.sequence.form.lanes)
try:
    bad = commit(placed((1, 4, 6)), {"pool.pe": 2})
    print("mismatched x:", bad.pool.x.sequence)
except Exception as e:
    print("mismatched x ->", type(e).__name__, str(e)[:160])
