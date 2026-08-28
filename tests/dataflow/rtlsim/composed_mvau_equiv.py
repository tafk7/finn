############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 5: composed replay + dot product is bit-equivalent to the fused MVAU.

Runs INSIDE the FINN Docker container; needs Vivado's ``xelab`` and ``finn_xsi``.

The decomposition claims that ``mvu_vvu_axi`` is an activation replay followed
by a dot-product core, and that naming the two halves separately changes
nothing observable.  The RTL says the same thing: ``dotp_axi`` is
``mvu_vvu_axi`` with the ``replay_buffer`` lifted out and ``alast`` taken from
``s_axis_input_tlast`` instead of the replay's ``olast``.  This test drives both
with the same random stream and requires bit-identical output.

Both DUTs are compared to *each other*, not to a numeric golden, so the
stimulus is raw random integers packed to the stream widths.  Any value reaches
both identically; a mismatch can only mean the composition altered behaviour.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_equiv.sh
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from finn.xsi import (
    close_rtlsim,
    compile_sim_obj,
    load_sim_obj,
    reset_rtlsim,
    rtlsim_multi_io,
)

LIVENESS = 200000

#: What the fused golden compiles against: FINN's own ship-everything list.
FUSED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
)

#: What the composed DUT compiles against: the same cores, reached through
#: FinnLib's dot-product wrapper, plus FINN's replay buffer.
COMPOSED_FINN_SOURCES = ("mvu_pkg.sv", "replay_buffer.sv")
COMPOSED_FINNLIB_SOURCES = (
    "add_multi_pkg.sv",
    "add_multi.sv",
    "dotp_axi.sv",
    "dotp.sv",
    "dotp_8sx9_dsp58.sv",
)

#: ``(label, VERSION, PE, SIMD, MW, MH, activation bits, weight bits)``.
#: VERSION 2 exercises the soft-vector core, 3 the packed DSP58 core.
CONFIGS = [
    ("softvec", 2, 2, 2, 4, 4, 8, 8),
    ("packed", 3, 2, 2, 4, 4, 8, 8),
    ("one_neuron_fold", 3, 4, 2, 4, 4, 8, 8),
    ("one_synapse_fold", 3, 2, 4, 4, 4, 8, 8),
]

ACCU_WIDTH = 16


def _segment_length(simd: int, version: int) -> int:
    """Match what the declared ``segment_length`` property would produce.

    The property derives this from the target clock; the harness fixes a 5 ns
    clock, which covers the full chain, so the chain length is the binding term.
    """

    if version != 3:
        return 0
    return -(-simd // 3)


def _fused_top(name: str, config: tuple) -> str:
    _label, version, pe, simd, mw, mh, act_w, w_w = config
    return f"""
module {name} #(
    parameter WSTREAM = {(pe * simd * w_w + 7) // 8 * 8},
    parameter ISTREAM = {(simd * act_w + 7) // 8 * 8},
    parameter OSTREAM = {(pe * ACCU_WIDTH + 7) // 8 * 8}
)(
    input  logic ap_clk,
    input  logic ap_clk2x,
    input  logic ap_rst_n,
    input  logic [WSTREAM-1:0] in1_V_tdata,
    input  logic in1_V_tvalid,
    output logic in1_V_tready,
    input  logic [ISTREAM-1:0] in0_V_tdata,
    input  logic in0_V_tvalid,
    output logic in0_V_tready,
    output logic [OSTREAM-1:0] out0_V_tdata,
    output logic out0_V_tvalid,
    input  logic out0_V_tready
);
    mvu_vvu_axi #(
        .IS_MVU(1), .VERSION({version}),
        .MW({mw}), .MH({mh}), .PE({pe}), .SIMD({simd}),
        .SEGMENTLEN({_segment_length(simd, version)}),
        .ACTIVATION_WIDTH({act_w}), .WEIGHT_WIDTH({w_w}), .ACCU_WIDTH({ACCU_WIDTH}),
        .NARROW_WEIGHTS(0), .SIGNED_ACTIVATIONS(1),
        .PUMPED_COMPUTE(0), .FORCE_BEHAVIORAL(1)
    ) core (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(in0_V_tdata),
        .s_axis_input_tvalid(in0_V_tvalid),
        .s_axis_input_tready(in0_V_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


def _composed_top(name: str, config: tuple) -> str:
    """Replay feeding a dot product, wired exactly as the fused core wires them.

    ``replay_buffer`` is instantiated with the parameters the declared derived
    properties produce -- ``LEN = SF``, ``REP = NF``, ``W = SIMD * width`` --
    and its ``olast`` becomes the dot product's ``tlast``.  ``ofin`` is left
    unconnected because the fused core never reads it either.
    """

    _label, version, pe, simd, mw, mh, act_w, w_w = config
    synapse_folds, neuron_folds = mw // simd, mh // pe
    return f"""
module {name} #(
    parameter WSTREAM = {(pe * simd * w_w + 7) // 8 * 8},
    parameter ISTREAM = {(simd * act_w + 7) // 8 * 8},
    parameter OSTREAM = {(pe * ACCU_WIDTH + 7) // 8 * 8}
)(
    input  logic ap_clk,
    input  logic ap_clk2x,
    input  logic ap_rst_n,
    input  logic [WSTREAM-1:0] in1_V_tdata,
    input  logic in1_V_tvalid,
    output logic in1_V_tready,
    input  logic [ISTREAM-1:0] in0_V_tdata,
    input  logic in0_V_tvalid,
    output logic in0_V_tready,
    output logic [OSTREAM-1:0] out0_V_tdata,
    output logic out0_V_tvalid,
    input  logic out0_V_tready
);
    localparam int unsigned REPLAY_W = {simd * act_w};

    uwire rst = !ap_rst_n;
    uwire [REPLAY_W-1:0] replayed_tdata;
    uwire replayed_tvalid;
    uwire replayed_tlast;
    uwire replayed_tready;

    replay_buffer #(
        .LEN({synapse_folds}), .REP({neuron_folds}), .W(REPLAY_W)
    ) activation_replay (
        .clk(ap_clk), .rst(rst),
        .idat(in0_V_tdata[REPLAY_W-1:0]),
        .ivld(in0_V_tvalid),
        .irdy(in0_V_tready),
        .odat(replayed_tdata),
        .olast(replayed_tlast),
        .ofin(),
        .ovld(replayed_tvalid),
        .ordy(replayed_tready)
    );

    dotp_axi #(
        .VERSION({version}), .ACTIVATION_BROADCASTING(1),
        .PE({pe}), .SIMD({simd}),
        .SEGMENTLEN({_segment_length(simd, version)}),
        .ACTIVATION_WIDTH({act_w}), .WEIGHT_WIDTH({w_w}), .ACCU_WIDTH({ACCU_WIDTH}),
        .NARROW_WEIGHTS(0), .SIGNED_ACTIVATIONS(1),
        .PUMPED_COMPUTE(0), .FORCE_BEHAVIORAL(1)
    ) dot_product (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(replayed_tdata),
        .s_axis_input_tvalid(replayed_tvalid),
        .s_axis_input_tlast(replayed_tlast),
        .s_axis_input_tready(replayed_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


def _random_word(generator: np.random.RandomState, bits: int) -> int:
    """A random integer of exactly ``bits`` bits.

    Built from bytes rather than ``randint`` because a packed weight beat is
    ``PE * SIMD * WEIGHT_WIDTH`` wide and reaches 64 bits at modest folding,
    which ``randint`` cannot represent.
    """

    raw = int.from_bytes(bytes(generator.randint(0, 256, size=(bits + 7) // 8)), "little")
    return raw & ((1 << bits) - 1)


def _write(directory: str, name: str, text: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _sources(root: str, subdirectory: str, names: tuple[str, ...]) -> list[str]:
    resolved = []
    for name in names:
        path = os.path.join(root, subdirectory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing RTL source: {path}")
        resolved.append(path)
    return resolved


def _drive(
    top_module: str, sources: list[str], top_path: str, stimulus: dict, expected: int
) -> list[int]:
    with tempfile.TemporaryDirectory() as scratch:
        sim_dir, so_rel = compile_sim_obj(top_module, [*sources, top_path], scratch, behav=True)
        sim = load_sim_obj(sim_dir, so_rel)
        reset_rtlsim(sim)
        local = {
            "inputs": {key: list(value) for key, value in stimulus["inputs"].items()},
            "outputs": {"out0": []},
        }
        rtlsim_multi_io(sim, local, expected, sname="_V", liveness_threshold=LIVENESS)
        close_rtlsim(sim)
        return local["outputs"]["out0"]


def _run(config: tuple, finn_root: str, finnlib_root: str) -> bool:
    label, _version, pe, simd, mw, mh, act_w, w_w = config
    print(f"\n========== fixture 5: {label} ==========")
    synapse_folds, neuron_folds = mw // simd, mh // pe

    generator = np.random.RandomState(0)
    activation = [_random_word(generator, simd * act_w) for _ in range(synapse_folds)]
    weight = [_random_word(generator, pe * simd * w_w) for _ in range(synapse_folds * neuron_folds)]
    stimulus = {"inputs": {"in0": activation, "in1": weight}, "outputs": {"out0": []}}
    print(
        f"  geometry: SF={synapse_folds} NF={neuron_folds} "
        f"in0={len(activation)} in1={len(weight)} expect {neuron_folds} outputs"
    )

    with tempfile.TemporaryDirectory() as scratch:
        fused_path = _write(scratch, "mvau_fused.sv", _fused_top("mvau_fused", config))
        composed_path = _write(scratch, "mvau_composed.sv", _composed_top("mvau_composed", config))
        fused_sources = _sources(finn_root, "finn-rtllib/mvu", FUSED_SOURCES)
        composed_sources = [
            *_sources(finn_root, "finn-rtllib/mvu", COMPOSED_FINN_SOURCES),
            *_sources(finnlib_root, "rtl", COMPOSED_FINNLIB_SOURCES),
        ]

        fused = _drive("mvau_fused", fused_sources, fused_path, stimulus, neuron_folds)
        composed = _drive("mvau_composed", composed_sources, composed_path, stimulus, neuron_folds)

    print(f"  fused   : {fused}")
    print(f"  composed: {composed}")
    if fused == composed and len(fused) == neuron_folds:
        print(f"  {label.upper()}: PASS (bit-identical, {neuron_folds} outputs)")
        return True
    print(f"  {label.upper()}: FAIL")
    return False


def main() -> int:
    finn_root = os.environ["FINN_ROOT"]
    finnlib_root = os.environ.get("FINNLIB_ROOT", os.path.join(finn_root, "..", "finnlib"))
    if not os.path.isdir(os.path.join(finnlib_root, "rtl")):
        print(f"FinnLib RTL not found under {finnlib_root}; set FINNLIB_ROOT")
        return 2
    ok = True
    for config in CONFIGS:
        ok &= _run(config, finn_root, finnlib_root)
    print("\nRESULT:", "FIXTURE 5 PASS" if ok else "FIXTURE 5 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
