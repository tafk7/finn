"""P1: how the MVAU netlist finds and routes clocks and resets.

Clocks are found by pin name (``ap_clk``, ``ap_clk2x``, ``ap_rst_n``) and routed by
role or by the substring ``clk2x``. Reset inversion is derived from polarity.
"""

from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt
from finn.kernels import DspBlock, mvau_assembly

for pumped in (False, True):
    built = mvau_assembly(
        repetitions=2,
        matrix_width=4,
        matrix_height=4,
        activation_dtype=dt("INT3"),
        weights_dtype=dt("INT3"),
        pe=2,
        simd=2,
        target_dsp=DspBlock.DSP58,
        compute_pumping=pumped,
    )
    top = built.structure.top_abi
    print(f"pumped={pumped} top ports:", [(p.name, type(p.role).__name__) for p in top.ports])
    for wire in built.structure.wires:
        source = getattr(wire.source, "pin", None)
        if source is not None and source.instance_id is None and source.signal_id.startswith("ap_"):
            if source.signal_id in ("ap_clk", "ap_clk2x", "ap_rst_n"):
                pin = wire.destination.pin
                print(f"   {pin.instance_id}.{pin.signal_id} <- {source.signal_id} invert={wire.invert}")
