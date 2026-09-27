"""P5: dotp's ports adopt whatever form their streams carry.

A transposed weight walk, and a one-lane activation stream feeding a SIMD=2 dotp,
are both accepted by every stream and by the netlist.
"""

from finn.core.space import Located, Space, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt
from finn.kernels import DotpAxiKernel, DspBlock
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.physical.forms import Every, Traversal, tile, vector_major
from finn.kernels.streams import Stream, StreamSpec, netlist

A, W, R = dt("INT3"), dt("INT3"), dt("INT8")


def chain(act_form: Traversal, wgt_form: Traversal) -> Space:
    class Chain(Space):
        a = Stream(spec=StreamSpec(ScalarEncoding(A), act_form, markers=(Every(2),)), port="in0_V")
        w = Stream(spec=StreamSpec(ScalarEncoding(W), wgt_form), port="in1_V")
        r = Stream(spec=StreamSpec(ScalarEncoding(R), vector_major((1, 2), 2)), port="out0_V")
        compute = DotpAxiKernel(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=2,
            simd=2,
            target_dsp=DspBlock.DSP48E2,
            segment_length=0,
            activation_stream=a,
            weights_stream=w,
            result_stream=r,
        )

    return design_space(Chain()).with_choices({Chain.compute.compute_pumping: False})


good = chain(vector_major((1, 4), 2), tile(2, 4, 2, 2))
# A transposed weight walk: SIMD folds outer, PE folds inner; still PE*SIMD lanes.
transposed = Traversal.over((2, 4), ((0, 1, 2), (1, 2, 2)), ((1, 2, 1), (0, 2, 1)))
bad_order = chain(vector_major((1, 4), 2), transposed)
narrow = chain(vector_major((1, 4), 1), tile(2, 4, 2, 2))  # one lane where dotp reads SIMD=2
for label, point in (("good", good), ("transposed weights", bad_order), ("1-lane acts", narrow)):
    print(label, {n: type(getattr(point, n).query(Stream.connection)).__name__ for n in "awr"})

modules = [Located("compute", "build_requirements", narrow.compute.build_requirements)]
streams = [Located(n, "connection", getattr(narrow, n).connection) for n in "awr"]
composed = netlist(modules, streams, module="probe", producer=ProducerIdentity("probe", "1"))
print("netlist for 1-lane activations:", type(composed).__name__)
