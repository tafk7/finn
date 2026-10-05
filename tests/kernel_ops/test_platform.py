# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform a KernelOp binds: its model's target (``target(model).platform``).

The node roots bind it to their kernels and to a stream with a source, so a value
case that requires a capability (``requires``) is refused, by name, on a device
without it and viable on one with it; the DSP block is the platform's. A bare
kernel, with no model, keeps the default platform, which refuses nothing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import Rejected, design_space, inspection
from finn.custom_op.kernels.base import KernelOpError, write_target
from finn.custom_op.kernels.partition import partition_root
from finn.custom_op.kernels.roots import StoredMatMulNode
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.target import DspBlock, Platform, Target, resolve_target
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import matmul_model, thresholding_model

ZYNQ = resolve_target("xczu3eg-sbva484-1-e", 5.0, "vivado_zynq")  # Ultra96 in its shell
ALVEO = resolve_target("xcu55c-fsvh2892-2L-e", 5.0, "vitis_alveo")
URAM = Target("a part with UltraRAM it initializes", 5.0, Platform(dsp=DspBlock.DSP58))


def targeted(model: ModelWrapper, target: Target) -> ModelWrapper:
    write_target(model, target)
    return model


def op(model: ModelWrapper) -> Any:
    return model.get_customop_wrapper(model.graph.node[0])


def test_the_node_root_binds_the_models_platform_and_its_dsp_block() -> None:
    point = op(targeted(matmul_model(), URAM)).point()
    assert point.platform == URAM.platform
    assert point.matmul.platform.dsp is DspBlock.DSP58  # the platform's, read by the cores
    assert commit(point, {"matmul.compute": "packed"}).matmul.compute.dsp is DspBlock.DSP58
    assert point.w.platform == URAM.platform and point.matmul.platform == URAM.platform


def test_a_memory_case_the_device_cannot_build_is_refused_by_name() -> None:
    ultra96 = op(matmul_model())  # TARGET: Ultra96, no UltraRAM
    with pytest.raises(KernelOpError, match="w.source.memstream.ram_style.*uram-absent"):
        ultra96.save({"w.source.memstream.ram_style": "ultra"})
    uram = op(targeted(matmul_model(), URAM))
    uram.save({"w.source.memstream.ram_style": "ultra"})
    assert uram.point().w.source.ram_style == "ultra"


def test_a_doubled_clock_is_the_shells_and_forced_off_without_it() -> None:
    zynq = op(targeted(matmul_model(), ZYNQ))
    with pytest.raises(KernelOpError, match="pumped_memory.*clk2x-absent"):
        zynq.save({"w.source.memstream.pumped_memory": True})
    forced = {item.key: item for item in inspection.forced(zynq.point())}
    pumped = forced["w.source.memstream.pumped_memory"]
    assert pumped.value is False and "clk2x-absent" in pumped.refused["True"]
    # Without a shell (a stitched IP), nothing states the clock away.
    alone = op(matmul_model())
    alone.save({"w.source.memstream.pumped_memory": True})
    assert alone.point().w.source.pumped_memory is True


def test_runtime_writable_thresholds_need_a_control_port() -> None:
    with pytest.raises(KernelOpError, match="use_axilite.*control-absent"):
        op(targeted(thresholding_model(), ALVEO)).save({"use_axilite": True})
    ultra96 = op(thresholding_model())
    ultra96.save({"use_axilite": True})
    assert ultra96.point().activate.use_axilite is True


def test_a_partitions_weight_stream_reads_the_platform() -> None:
    model = targeted(matmul_model(), ZYNQ).transform(InferKernelTensors())
    root = partition_root(model, model.graph.node)
    stream = root.point.w
    assert stream.platform == ZYNQ.platform
    assert isinstance(stream.source, MemStreamKernel)
    with pytest.raises(ValueError, match="clk2x-absent"):
        commit(root.point, {"w.source.memstream.pumped_memory": True})


def test_a_bare_kernel_without_a_platform_has_no_dsp_block() -> None:
    formals = op(matmul_model()).facts().formals()
    facts = {name: formals[name] for name in ("m", "n", "k", "activation_dtype", "weights_dtype")}
    bare = design_space(MatMulKernel(**facts, target_period_ns=5.0, weights=formals["weights"]))
    assert bare.platform == Platform()
    answer = commit(bare, {"compute": "packed"}).compute.query(PackedDotpKernel.dsp)
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["dotp-dsp"]


def test_a_platform_without_a_dsp_block_is_refused_by_the_cores() -> None:
    platform = replace(URAM.platform, dsp=None)
    formals = {**op(matmul_model()).facts().formals(), "platform": platform}
    answer = design_space(StoredMatMulNode(**formals)).matmul.query(MatMulKernel.compute)
    assert isinstance(answer, Rejected)
    (finding,) = answer.findings  # no core is viable, each for the same reason
    for core in ("packed", "int8_dsp58"):
        assert f"{core}: matmul.compute.{core}.dsp: dotp-dsp" in finding.message
