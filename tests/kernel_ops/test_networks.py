# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX entry's T2: a generated single-op network through the kernel path into XSim.

A positive graph of an op's spec (``kernel_ops.specs``), which every platform row admits
(``tests/kernel_ops/test_reference.py``), is a network of one op. It goes through the
builder's kernel-path phase (``phase_kernel_path``: ``ToKernelOps``, inferring as it
converts, the choices, the cut and its verification) for Ultra96's part on the ``ip`` shell, the
target the specs are checked on, to the partition's body: the whole network, its
boundary the network's. Its root, its choices completed as the phase's verification completes them,
runs in XSim (``finn.harness.ops.check_parity``) on seeded inputs drawn from the
graph's annotations, random with their extremes, and every output word is checked
against the KernelOps' ``execute_node`` (the oracle) on the same inputs.

Lean (KT10): two graphs an op, one simulation each. MatMul: unsigned activations over
k = 64 with weights stored, and weights streamed as a second graph input. Thresholding:
a row a channel, and NHWC images. WindowedMatMul: an overlapping 3 x 3 window over a
6 x 6 x 4 image, and a strided, dilated one that passes rows and columns: the image
enters once, and the window is its channel's ``input_gen``
(``tests/kernel_ops/test_windowed_matmul.py`` folds CNV's first layer).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest
from kernels.xsim import requires_xsim
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

from finn.builder.kernel_build_config import KernelBuildConfig
from finn.builder.kernel_build_steps import phase_kernel_path
from finn.custom_op.kernels.base import read_target
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.harness.ops import boundary_inputs, check_parity, graph_inputs
from finn.harness.reference import OpSpec
from finn.platform import TargetRequest
from finn.transformation.kernels.choose import completion
from kernel_ops.models import TARGET
from kernel_ops.specs import matmul, thresholding, windowed_matmul

NETWORKS = [
    pytest.param(spec, name, id=f"{spec.op.op_type}-{name}")
    for spec, name in (
        (matmul.SPEC, "int4-uint4"),
        (matmul.SPEC, "streamed-weights"),
        (thresholding.SPEC, "rows"),
        (thresholding.SPEC, "nhwc"),
        (windowed_matmul.SPEC, "3x3-stride-1"),
        (windowed_matmul.SPEC, "strided-dilated"),
    )
]


def config(directory: Path) -> KernelBuildConfig:
    """The kernel path for ``TARGET`` (Ultra96's part at 5 ns on the ``ip`` shell), its
    choices the default exploration's (none) and completion's (the baseline)."""
    return KernelBuildConfig(
        output_dir=str(directory),
        target=TargetRequest(part=TARGET.part, period_ns=TARGET.platform.period_ns),
        generate_outputs=[],
        steps=["phase_kernel_path"],
        enable_build_pdb_debug=False,
    )


def network(spec: OpSpec, name: str, directory: Path) -> tuple[ModelWrapper, ModelWrapper]:
    """The positive graph ``name`` as a network, and the partition's body the builder's
    kernel-path phase cuts of it."""
    source = spec.positive[name]().transform(InferShapes())
    _, body, _ = partition_body(phase_kernel_path(source, config(directory)))
    return source, body


@pytest.mark.parametrize(("spec", "name"), NETWORKS)
def test_the_kernel_path_takes_the_network_whole(spec: OpSpec, name: str, tmp_path: Path) -> None:
    """The phase's partition is the whole network: one KernelOp, the network's inputs
    and outputs its boundary, the target stated."""
    source, body = network(spec, name, tmp_path)
    assert [node.op_type for node in body.graph.node] == [spec.op.op_type]
    assert graph_inputs(body) == graph_inputs(source)
    assert [item.name for item in body.graph.output] == [item.name for item in source.graph.output]
    assert read_target(body) == TARGET


@requires_xsim
@pytest.mark.parametrize(("spec", "name"), NETWORKS)
def test_the_network_computes_in_xsim_what_its_kernel_op_computes(
    spec: OpSpec, name: str, tmp_path: Path
) -> None:
    cfg = config(tmp_path / "build")
    _, body = network(spec, name, tmp_path / "build")
    label = f"{spec.op.op_type}-{name}"
    check_parity(
        body,
        tmp_path / "xsim",
        inputs=boundary_inputs(body, zlib.crc32(label.encode())),
        completion=completion(cfg.kernel_completion),
        label=label,
    )
