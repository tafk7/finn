# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform a KernelOp binds: its model's target (``read_target(model).platform``).

A KernelOp binds it to its kernel, and the network to every channel, so a value
that requires a capability is refused, by the rule that says so, on a device without it
and possible on one with it; the DSP block is the platform's. A bare kernel, with no
model, has no platform: its caller must state one.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured
from support.helpers import FULL_DSP58, choose, refused_codes
from support.models import MATMUL, TARGET, matmul_model, thresholding_model
from support.refusals import refused

from finn.core.graph.annotations.platform import write_target
from finn.core.space import DefinitionError, design_space, inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import InShell, in_shell
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.platform.capabilities import DspBlock, Platform
from finn.platform.resolve import resolve_target
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors

ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")  # Ultra96 in its shell
URAM = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")


def targeted(model: ModelWrapper, target: Target) -> ModelWrapper:
    write_target(model, target)
    return model


def op(model: ModelWrapper) -> Any:
    return kernel_op(model, model.graph.node[0])


def folded(model: ModelWrapper) -> InShell:
    """The MatMul folded as the Chain's first, its network bound in its shell: its
    channels' contracts known."""
    configured(model, {"first": MATMUL})
    return in_shell(model, model.graph.node)


def test_a_node_binds_the_models_platform_and_its_dsp_block() -> None:
    model = targeted(matmul_model(), URAM)
    point = op(model).point()
    assert point.platform == URAM.platform
    assert point.platform.dsp is DspBlock.DSP58  # the platform's, read by the cores
    assert commit(point, {"compute": "packed"}).compute.dsp is DspBlock.DSP58
    assert folded(model).member("w").platform == URAM.platform


def test_a_memory_case_the_device_cannot_build_is_refused_by_name() -> None:
    ultra = {"w": {"source.ram_style": "ultra"}}
    ultra96 = matmul_model()  # TARGET: Ultra96, no UltraRAM
    # Its weight channel's contracts wait on the MatMul's core, its memory's rule on
    # UltraRAM does not: the channel's own rule refuses it already.
    with refused("uram-absent", owner="source.uram_present"):
        configured(ultra96, ultra)
    configured(ultra96, {"first": MATMUL})
    with refused("uram-absent", owner="source.uram_present"):
        configured(ultra96, ultra)
    uram = targeted(matmul_model(), URAM)
    configured(uram, {"first": MATMUL})
    configured(uram, ultra)
    assert folded(uram).member("w").source.ram_style == "ultra"


def test_a_doubled_clock_is_the_shells_not_the_platforms() -> None:
    """The platform carries no doubled clock (the shell row owns it, and the shell
    supplies it): a KernelOp offers a pumped memory on any target, and its partition in
    the Zynq shell, which supplies none, refuses it by the shell's budget
    (``clock-unavailable``, the weight channel's clock rule; ``test_shell_budgets``)."""
    assert "clk2x" not in {item.name for item in fields(Platform)}
    model = targeted(matmul_model(), TARGET)
    configured(model, {"first": MATMUL})
    configured(model, {"w": {"source.pumped_memory": True}})
    assert folded(model).member("w").source.pumped_memory is True
    zynq = targeted(matmul_model(), ZYNQ)
    configured(zynq, {"first": MATMUL})
    with refused("clock-unavailable", owner="w-clock"):
        configured(zynq, {"w": {"source.pumped_memory": True}})
    assert folded(zynq).member("w").source.pumped_memory is False


def test_runtime_writable_thresholds_need_a_control_bus() -> None:
    # A KernelOp places no control bus to present its AXI-Lite interface through: of the
    # choice's values only False is possible, True refused by ``threshold-control``, which
    # refuses a change setting it.
    model = thresholding_model()
    (viable,) = inspection.viable(op(model).point(), [ThresholdingAxiKernel.use_axilite])
    assert viable.cases == (False,)
    ((value, broken),) = [
        (item.value, item.broken)
        for item in viable.values or ()
        if isinstance(item, inspection.Refused)
    ]
    assert value is True and [rule.result.findings[0].code for rule in broken] == [
        "threshold-control"
    ]
    with refused("threshold-control", owner="control_for_writes"):
        configured(model, {"activate": {"use_axilite": True}})


def test_a_partitions_weight_stream_reads_the_platform() -> None:
    model = targeted(matmul_model(), ZYNQ).transform(InferKernelTensors())
    bound = folded(model)
    stream = bound.member("w")
    assert stream.platform == ZYNQ.platform
    assert isinstance(stream.source, MemStreamKernel)
    # Its transport FIFO reads it too: Ultra96 has no UltraRAM.
    fifo = {"w.transport": "fifo", "w.transport.fifo.depth": 4096}
    # Committed, ultra leaves the buffer refused by its rule on the platform.
    assert "uram-absent" in refused_codes(
        bound, {**fifo, "w.transport.fifo.buffer.ram_style": "ultra"}
    )
    assert choose(bound, {**fifo, "w.transport.fifo.buffer.ram_style": "block"})


def test_a_bare_kernel_states_its_platform() -> None:
    node = op(matmul_model())
    tensors = {node.wiring[port].tensor: tensor for port, tensor in node.input_edges().items()}
    with refused("formal-unsupplied", error=DefinitionError, message="^platform is not supplied"):
        design_space(MatMulKernel(**tensors))


def test_a_platform_without_a_dsp_block_is_refused_by_the_cores() -> None:
    platform = replace(URAM.platform, dsp=None)
    node = op(matmul_model())
    arguments = {**node.facts().arguments, "platform": platform}
    point = design_space(MatMulKernel(**arguments))
    (viable,) = inspection.viable(point, [MatMulKernel.compute])
    # No core is possible, each refused by its own rule for the same reason.
    assert viable.cases == ()
    assert {
        item.value: [(rule.rule, rule.result.findings[0].code) for rule in item.broken]
        for item in viable.values or ()
        if isinstance(item, inspection.Refused)
    } == {core: [(f"compute.{core}.dsp", "dotp-dsp")] for core in ("packed", "int8_dsp58")}
