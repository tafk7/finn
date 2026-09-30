# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared case geometry, packing and observation wrapper for dotp RTL checks.

These helpers contain no kernel construction or arithmetic policy. Both the
physical kernel suite and retained dataflow harness use this single copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.target import DspBlock


@dataclass(frozen=True)
class Case:
    label: str
    target: DspBlock
    repetitions: int
    matrix_width: int
    matrix_height: int
    pe: int
    simd: int
    activation: str
    weight: str
    accumulator: str
    activation_values: str = "full"
    weight_values: str = "full"
    pumping: bool = False

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd


def _encode(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _decode(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _activation_beats(case: Case, values: np.ndarray, width: int) -> list[int]:
    beats: list[int] = []
    for repetition in range(case.repetitions):
        for _neuron_fold in range(case.neuron_folds):
            for synapse_fold in range(case.synapse_folds):
                beat = 0
                for lane in range(case.simd):
                    value = int(values[repetition, synapse_fold * case.simd + lane])
                    beat |= _encode(value, width) << (lane * width)
                beats.append(beat)
    return beats


def _weight_beats(case: Case, values: np.ndarray, width: int) -> list[int]:
    beats: list[int] = []
    for _repetition in range(case.repetitions):
        for neuron_fold in range(case.neuron_folds):
            for synapse_fold in range(case.synapse_folds):
                beat = 0
                for lane in range(case.pe):
                    for element in range(case.simd):
                        value = int(
                            values[
                                synapse_fold * case.simd + element,
                                neuron_fold * case.pe + lane,
                            ]
                        )
                        shift = (lane * case.simd + element) * width
                        beat |= _encode(value, width) << shift
                beats.append(beat)
    return beats


def _unpack(case: Case, beats: list[int], width: int) -> np.ndarray:
    result = np.zeros((case.repetitions, case.matrix_height), dtype=np.int64)
    index = 0
    for repetition in range(case.repetitions):
        for neuron_fold in range(case.neuron_folds):
            beat = beats[index]
            index += 1
            for lane in range(case.pe):
                raw = (beat >> (lane * width)) & ((1 << width) - 1)
                result[repetition, neuron_fold * case.pe + lane] = _decode(raw, width)
    return result


def _rtl(value: object) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def _wrapper(name: str, case: Case, parameters: dict[str, object]) -> str:
    parameter_text = ",\n        ".join(
        f".{key}({_rtl(parameters[key])})" for key in sorted(parameters)
    )
    input_width = (case.simd * DataType[case.activation].bitwidth() + 7) // 8 * 8
    weight_width = (case.pe * case.simd * DataType[case.weight].bitwidth() + 7) // 8 * 8
    output_width = (case.pe * DataType[case.accumulator].bitwidth() + 7) // 8 * 8
    return f"""module {name}(
    input logic ap_clk, input logic ap_clk2x, input logic ap_rst_n,
    input logic [{input_width - 1}:0] in0_V_tdata,
    input logic in0_V_tvalid, output logic in0_V_tready,
    input logic [{weight_width - 1}:0] in1_V_tdata,
    input logic in1_V_tvalid, output logic in1_V_tready,
    output logic [{output_width - 1}:0] out0_V_tdata,
    output logic out0_V_tvalid, input logic out0_V_tready
);
    integer unsigned synapse_fold = 0;
    wire input_last = synapse_fold == {case.synapse_folds - 1};
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) synapse_fold <= 0;
        else if (in0_V_tvalid && in0_V_tready) begin
            if (input_last) synapse_fold <= 0;
            else synapse_fold <= synapse_fold + 1;
        end
    end
    dotp_axi #(
        {parameter_text}
    ) dut (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(in0_V_tdata),
        .s_axis_input_tvalid(in0_V_tvalid),
        .s_axis_input_tlast(input_last),
        .s_axis_input_tready(in0_V_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""
