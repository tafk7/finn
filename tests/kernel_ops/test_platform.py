# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform a KernelOp binds: its model's target (``read_target(model).platform``).

A KernelOp binds it to its kernel, and the network to every channel, so a value
case that requires a capability (``requires``) is refused, by name, on a device
without it and viable on one with it; the DSP block is the platform's. A bare
kernel, with no model, has no platform: its caller must state one.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.helpers import FULL_DSP58, choose

from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.space import DefinitionError, Rejected, design_space
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import InShell, in_shell, save_channels
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.platform.capabilities import DspBlock, Platform
from finn.platform.resolve import resolve_target
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import MATMUL, TARGET, matmul_model, thresholding_model

ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")  # Ultra96 in its shell
URAM = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")


def targeted(model: ModelWrapper, target: Target) -> ModelWrapper:
    write_target(model, target)
    return model


def op(model: ModelWrapper) -> Any:
    return kernel_op(model, model.graph.node[0])


def folded(model: ModelWrapper) -> InShell:
    """The MatMul folded as the Chain's first, its network bound in its shell: its
    channels' ends known."""
    op(model).save(MATMUL)
    return in_shell(model, model.graph.node)


def test_a_node_binds_the_models_platform_and_its_dsp_block() -> None:
    model = targeted(matmul_model(), URAM)
    point = op(model).point()
    assert point.platform == URAM.platform
    assert point.platform.dsp is DspBlock.DSP58  # the platform's, read by the cores
    assert commit(point, {"compute": "packed"}).compute.dsp is DspBlock.DSP58
    assert folded(model).member("w").platform == URAM.platform


def test_a_memory_case_the_device_cannot_build_is_refused_by_name() -> None:
    ultra = {"w": {"source.memstream.ram_style": "ultra"}}
    ultra96 = matmul_model()  # TARGET: Ultra96, no UltraRAM
    # Its weight channel's ends wait on the MatMul's folding: nothing to check yet.
    with pytest.raises(KernelOpError, match="w: its channel's ends wait on first's open"):
        save_channels(ultra96, ultra)
    op(ultra96).save(MATMUL)
    with pytest.raises(KernelOpError, match="w: .*source.memstream.ram_style.*uram-absent"):
        save_channels(ultra96, ultra)
    uram = targeted(matmul_model(), URAM)
    op(uram).save(MATMUL)
    save_channels(uram, ultra)
    assert folded(uram).member("w").source.ram_style == "ultra"


def test_a_doubled_clock_is_the_shells_not_the_platforms() -> None:
    """The platform carries no doubled clock (the shell row owns it, and the shell
    admits it): a KernelOp offers a pumped memory on any target, the Zynq shell's
    included, whose admission then refuses it (``test_shell_admission``)."""
    assert "clk2x" not in {item.name for item in fields(Platform)}
    for target in (TARGET, ZYNQ):
        model = targeted(matmul_model(), target)
        op(model).save(MATMUL)
        save_channels(model, {"w": {"source.memstream.pumped_memory": True}})
        assert folded(model).member("w").source.pumped_memory is True


def test_runtime_writable_thresholds_need_a_control_bus() -> None:
    # A KernelOp places no control bus to present its AXI-Lite interface through.
    with pytest.raises(KernelOpError, match="use_axilite.*threshold-control"):
        op(thresholding_model()).save({"use_axilite": True})


def test_a_partitions_weight_stream_reads_the_platform() -> None:
    model = targeted(matmul_model(), ZYNQ).transform(InferKernelTensors())
    bound = folded(model)
    stream = bound.member("w")
    assert stream.platform == ZYNQ.platform
    assert isinstance(stream.source, MemStreamKernel)
    # Its transport FIFO reads it too: Ultra96 has no UltraRAM.
    fifo = {"w.transport": "fifo", "w.transport.fifo.buffer.depth": 4096}
    with pytest.raises(ValueError, match="uram-absent"):
        choose(bound, {**fifo, "w.transport.fifo.buffer.ram_style": "ultra"})
    assert choose(bound, {**fifo, "w.transport.fifo.buffer.ram_style": "block"})


def test_a_bare_kernel_states_its_platform() -> None:
    tensors = {
        f"{port}_tensor": tensor for port, tensor in op(matmul_model()).input_edges().items()
    }
    with pytest.raises(DefinitionError, match="platform is not supplied"):
        design_space(MatMulKernel(**tensors))


def test_a_platform_without_a_dsp_block_is_refused_by_the_cores() -> None:
    platform = replace(URAM.platform, dsp=None)
    node = op(matmul_model())
    arguments = {**node.facts().arguments, "platform": platform}
    answer = design_space(MatMulKernel(**arguments)).query(MatMulKernel.compute)
    assert isinstance(answer, Rejected)
    (finding,) = answer.findings  # no core is viable, each for the same reason
    for core in ("packed", "int8_dsp58"):
        assert f"{core}: compute.{core}.dsp: dotp-dsp" in finding.message
