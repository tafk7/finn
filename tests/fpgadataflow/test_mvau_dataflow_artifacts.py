# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.dataflow.mvau.artifacts import (
    build_mvau_rtl_artifact,
    build_mvau_rtl_artifact_requirements,
    simulate_mvau_rtl_artifact,
)
from finn.dataflow.mvau.elaboration import elaborate_mvau_rtl_softvec
from finn.dataflow.mvau.source import (
    MVAULegacyImportMode,
    MVAUProjectionContext,
    project_mvau_source,
    start_mvau_projection,
)

NODE_ID = "mvau_artifact"
PART = "xczu3eg-sbva484-1-e"
CLOCK_NS = 5.0


def _model(mem_mode: str) -> ModelWrapper:
    activation = helper.make_tensor_value_info("activation", TensorProto.FLOAT, [2, 4])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "MVAU_rtl",
        ["activation", "weights"],
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        PE=2,
        SIMD=2,
        MW=4,
        MH=4,
        TH=1,
        numInputVectors=[2],
        inputDataType="INT8",
        weightDataType="INT8",
        accDataType="INT16",
        outputDataType="INT16",
        noActivation=1,
        binaryXnorMode=0,
        mem_mode=mem_mode,
        resType="dsp",
        ram_style="block",
        runtime_writeable_weights=0,
        pumpedMemory=0,
        pumpedCompute=0,
    )
    graph = helper.make_graph([node], "mvau-artifact", [activation, weights], [output])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-artifact-test"))
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    model.set_initializer(
        "weights",
        np.asarray(
            [
                [-128, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            dtype=np.float32,
        ),
    )
    return model


def _selected(model: ModelWrapper):
    return start_mvau_projection(
        project_mvau_source(
            model,
            NODE_ID,
            MVAUProjectionContext(
                "finn.MinimizeAccumulatorWidth",
                fpga_part=PART,
                clock_period_ns=CLOCK_NS,
            ),
            import_mode=MVAULegacyImportMode.PRESERVE_SPECIALIZATION,
        )
    )


@pytest.mark.vivado
@pytest.mark.parametrize("mem_mode", ["external", "internal_decoupled"])
def test_requirements_backed_mvau_rtl_simulation(tmp_path: Path, mem_mode: str) -> None:
    model = _model(mem_mode)
    selected = _selected(model)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    requirements = build_mvau_rtl_artifact_requirements(selected, elaboration, model, Path.cwd())
    artifact = build_mvau_rtl_artifact(
        requirements,
        tmp_path / mem_mode,
        prepare_rtlsim=True,
    )
    activation = np.asarray([[1, 2, 3, 4], [-2, 1, 0, 3]], dtype=np.float32)
    expected = np.matmul(activation, requirements.weight_initializer.as_array())

    cppsim = simulate_mvau_rtl_artifact(artifact, activation, mode="cppsim")
    rtlsim = simulate_mvau_rtl_artifact(artifact, activation, mode="rtlsim")

    assert np.array_equal(cppsim, expected)
    assert np.array_equal(rtlsim, expected)
