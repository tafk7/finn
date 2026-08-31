# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.dataflow.mvau.compat.artifacts import (
    build_mvau_rtl_artifact,
    build_mvau_rtl_artifact_requirements,
    observe_mvau_cyclic_stitched_artifact,
    observe_mvau_rtl_artifact,
    simulate_mvau_rtl_artifact,
)
from finn.dataflow.mvau.elaboration import elaborate_mvau_rtl_softvec
from finn.dataflow.mvau.compat.evidence import collect_mvau_rtl_softvec_evidence
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import SOFT_VECTOR_PATHS, MVAUComputeKernelId
from finn.dataflow.mvau.source import (
    MVAUProjectionContext,
)
from finn.dataflow.mvau.compat.source import (
    project_legacy_mvau_source,
    start_legacy_mvau_projection,
)
from finn.dataflow.mvau.compat.operation import (
    MVAU_COMPUTE_SELECTION,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    CyclicRamStyle,
    MVAUWeightSupplyKernelId,
    WeightOrganization,
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
    return start_legacy_mvau_projection(
        project_legacy_mvau_source(
            model,
            NODE_ID,
            MVAUProjectionContext(
                "finn.MinimizeAccumulatorWidth",
                fpga_part=PART,
                clock_period_ns=CLOCK_NS,
            ),
        )
    )


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float = CLOCK_NS
    fpga_part: str = PART

    def _resolve_fpga_part(self) -> str:
        return self.fpga_part


def _logical_selected(model: ModelWrapper, mem_mode: str):
    node = model.graph.node[0]
    node.op_type = "MvauDataflowOp"
    node.domain = "finn.custom_op.dataflow"
    del node.attribute[:]
    node.attribute.extend(
        [
            helper.make_attribute("noActivation", 1),
            helper.make_attribute("binaryXnorMode", 0),
            helper.make_attribute("accDataType", "INT16"),
            helper.make_attribute("dataflow_scope_id", f"{NODE_ID}_scope"),
        ]
    )
    model.model.opset_import.append(helper.make_opsetid("finn.custom_op.dataflow", 1))
    operation = model.get_customop_wrapper(node)
    assert isinstance(operation, MvauDataflowOp)
    assignments = {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.SOFT_VECTOR.value,
        SOFT_VECTOR_PATHS.pe: 2,
        SOFT_VECTOR_PATHS.simd: 2,
        SOFT_VECTOR_PATHS.compute_pumping: False,
    }
    if mem_mode == "internal_decoupled":
        assignments.update(
            {
                MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: (
                    MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
                ),
                FINN_RTL_MEMSTREAM_PATHS.organization: WeightOrganization.AS_DEMANDED,
                FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
                FINN_RTL_MEMSTREAM_PATHS.pumped_memory: False,
                MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel: NO_KERNEL,
            }
        )
    else:
        assignments[MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel] = NO_KERNEL
    context = MVAUDataflowBuildContext(_BuildConfig())
    operation.commit_dataflow_assignments(context, assignments)
    return operation.resolve_dataflow(context)


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
    assert requirements.weight_initializer is not None
    expected = np.matmul(activation, requirements.weight_initializer.as_array())

    cppsim = simulate_mvau_rtl_artifact(artifact, activation, mode="cppsim")
    observation = observe_mvau_rtl_artifact(artifact, activation)
    rtlsim = np.asarray(observation.output_values, dtype=np.float32).reshape(expected.shape)

    assert np.array_equal(cppsim, expected)
    assert np.array_equal(rtlsim, expected)
    stitched_observation = None
    if mem_mode == "internal_decoupled":
        without_stitched = collect_mvau_rtl_softvec_evidence(
            selected,
            elaboration,
            artifact,
            simulation=observation,
        )
        assert not without_stitched.emitted_realization_observed
        stitched_observation = observe_mvau_cyclic_stitched_artifact(
            artifact,
            activation,
            tmp_path / "stitched",
        )
        assert stitched_observation.numerical_match
    evidence = collect_mvau_rtl_softvec_evidence(
        selected,
        elaboration,
        artifact,
        simulation=observation,
        stitched_simulation=stitched_observation,
    )
    assert evidence.emitted_realization_observed
    assert evidence.cycles is not None
    assert evidence.cycles.oracle == "finn.xsi:MVAU_rtl:cycles_rtlsim"
    assert evidence.cycles.measured_cycles > 0
    assert dict(evidence.cycles.configuration) == {
        "clock_period_ns": CLOCK_NS,
        "fpga_part": PART,
        "matrix_height": 4,
        "matrix_width": 4,
        "mem_mode": mem_mode,
        "pe": 2,
        "repetitions": 2,
        "simd": 2,
    }
    with pytest.raises(ValueError, match="another artifact"):
        collect_mvau_rtl_softvec_evidence(
            selected,
            elaboration,
            artifact,
            simulation=replace(observation, artifact_identity="different"),
            stitched_simulation=stitched_observation,
        )


@pytest.mark.vivado
@pytest.mark.parametrize("mem_mode", ["external", "internal_decoupled"])
def test_logical_dataflow_op_drives_mvau_rtl_simulation(tmp_path: Path, mem_mode: str) -> None:
    model = _model(mem_mode)
    selected = _logical_selected(model, mem_mode)
    elaboration = elaborate_mvau_rtl_softvec(selected)
    requirements = build_mvau_rtl_artifact_requirements(selected, elaboration, model, Path.cwd())
    artifact = build_mvau_rtl_artifact(
        requirements,
        tmp_path / f"logical-{mem_mode}",
        prepare_rtlsim=True,
    )
    activation = np.asarray([[1, 2, 3, 4], [-2, 1, 0, 3]], dtype=np.float32)
    assert requirements.weight_initializer is not None
    expected = np.matmul(activation, requirements.weight_initializer.as_array())
    observation = observe_mvau_rtl_artifact(artifact, activation)
    output = np.asarray(observation.output_values, dtype=np.float32).reshape(expected.shape)
    assert np.array_equal(output, expected)
    if mem_mode == "internal_decoupled":
        stitched = observe_mvau_cyclic_stitched_artifact(
            artifact,
            activation,
            tmp_path / "logical-stitched",
        )
        assert stitched.numerical_match
