"""R16: a refusal in one dotp port reaches every stream dotp sits on."""
from finn.core.space import design_space, Rejected, Available
from finn.kernels import MVAU, DspBlock
from finn.kernels.configure import commit
from finn.kernels.streams import Stream
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt

facts = dict(repetitions=2, matrix_width=4, matrix_height=4, activation_dtype=dt("INT3"),
             weights_dtype=dt("UINT3"),  # dotp requires signed weights
             target_dsp=DspBlock.DSP48E2, segment_length=0)
p = commit(design_space(MVAU(**facts)), {"implementation": "external", "weight_stream.transport": "direct",
            "compute.compute_pumping": False, "pe": 2, "simd": 2})
for name in ("activations", "replayed", "weight_stream", "results"):
    r = getattr(p, name).query(Stream.connection)
    print(name, type(r).__name__, sorted({(f.owner, f.code) for f in getattr(r, "findings", ())}))
s = p.query(MVAU.structure)
print("structure", type(s).__name__, len(s.findings) if hasattr(s, "findings") else "")
for f in s.findings: print("   ", f.owner, f.code)
