"""A padded AXIS child (dotp result) feeding another child (a replay buffer) is refused."""
from finn.core.space import design_space, Space, Param, Members, Rejected, view, default_semantics
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt
from finn.kernels import DotpAxiKernel, DspBlock
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.physical.forms import vector_major, tile, Every
from finn.kernels.streaming import ReplayBuffer
from finn.kernels.streams import Stream, StreamSpec, CONNECTION, MODULE

A, W, R = dt("INT3"), dt("INT3"), dt("INT9")  # PE=1 x 9 bits: AXIS carrier 16, padded
act = StreamSpec(ScalarEncoding(A), vector_major((1, 4), 2), markers=(Every(2),))
wgt = StreamSpec(ScalarEncoding(W), tile(1, 4, 1, 2))
res = StreamSpec(ScalarEncoding(R), vector_major((1, 1), 1))

class Chain(Space):
    a = Stream(spec=act, port="in0_V")
    w = Stream(spec=wgt, port="in1_V")
    r = Stream(spec=res)
    o = Stream(spec=res, port="out0_V")
    compute = DotpAxiKernel(activation_dtype=A, weights_dtype=W, result_dtype=R, pe=1, simd=2,
                            target_dsp=DspBlock.DSP48E2, segment_length=0,
                            activation_stream=a, weights_stream=w, result_stream=r)
    after = ReplayBuffer(input_stream=r, output_stream=o, sequence_length=1, replay_count=1)

p = design_space(Chain()).with_choices({Chain.compute.compute_pumping: False})
for name in "awro":
    r = getattr(p, name).query(Stream.connection)
    print(name, type(r).__name__, [(f.owner, f.code, f.message[:90]) for f in getattr(r, "findings", ())])
