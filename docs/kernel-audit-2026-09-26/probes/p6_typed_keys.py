"""P6: MVAU choices can be keyed by typed references, not only by strings.

``mvau_assembly`` and ``configure.commit`` use string keys. Typed handles work for
class-attribute nodes and candidate handles; a candidate declared inline in a
Decision (BufferedStream's ``fifo``) is reached only as a ``ChoiceMemberRef``.
"""

from finn.core.space import design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt
from finn.kernels import MVAU, DspBlock

identity = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
base = design_space(
    MVAU(
        repetitions=2,
        matrix_width=4,
        matrix_height=4,
        activation_dtype=dt("INT3"),
        weights_dtype=dt("INT3"),
        target_dsp=DspBlock.DSP48E2,
        segment_length=0,
        weights=identity,
    )
)
point = base.with_choices(
    {
        MVAU.implementation: "cyclic",
        MVAU.cyclic.rom_style: "block",
        MVAU.weight_stream.transport: "fifo",
        MVAU.pe: 2,
        MVAU.simd: 2,
        MVAU.compute.compute_pumping: False,
    }
)
print("typed keys accepted; rom_style =", point.cyclic.rom_style)
print("inline candidate reference:", type(MVAU.weight_stream.transport.fifo).__name__)
