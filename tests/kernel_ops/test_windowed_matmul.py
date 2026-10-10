# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""WindowedMatMul beyond its spec (``kernel_ops.specs.windowed_matmul``): the window
reaches the hardware as its image channel's reorder, and CNV's convolutions convert.

- A single convolution through the builder's kernel-path phase: the image enters once,
  in row-major order (the boundary presents the frame, not the patches: ``unreplayed``),
  and the image channel's adapter is an ``input_gen`` whose nest is the window's
  (``FM_SIZE = H * W * C'``, ``DIMS = (OH, OW, N / PE, KH, KW * C')``, ``COEFS = (W *
  C', C', 0, W * C', 1)``, ``C' = C / SIMD``), its TLAST closing each window.
- In XSim (``finn.harness.ops.check_parity``, the ``XSim`` executor), the folded
  single layers compute what the op's ``execute_node`` computes, every output word:
  CNV's first layer at SIMD 3 and PE 16 among them.
- CNV_W2A2 (the BNN-PYNQ CNV at two bits, its weights from the torch hub cache, as
  ``kernel_ops.tfc`` takes TFC) through graph preparation and ``ToKernelOps``: each of
  its six ``Im2Col`` and ``MatMul`` pairs converts to one WindowedMatMul, and the host
  nodes between KernelOps are the two max pools, the flatten's Transpose and its
  Reshape.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest
from kernels.xsim import requires_xsim
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

from finn.builder.kernel_build_config import KernelBuildConfig
from finn.builder.kernel_build_steps import phase_graph_preparation, phase_kernel_path
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.dataflow.traversal import vector_major
from finn.harness.ops import boundary_inputs, check_parity
from finn.platform import TargetRequest
from finn.transformation.kernels import ToKernelOps
from finn.transformation.kernels.convert import between_kernel_ops
from finn.transformation.prepare import GraphPreparation
from kernel_ops.specs import windowed_matmul
from kernel_ops.test_networks import config
from kernel_ops.tfc import PREPROCESSING, ULTRA96

CNV_SHAPE = (1, 3, 32, 32)
CNV_EXPORT = "cnv_w2a2.onnx"


def single_layer(name: str, directory: Path) -> ModelWrapper:
    """The positive graph ``name`` through the kernel-path phase: the partition's body."""
    source = windowed_matmul.SPEC.positive[name]().transform(InferShapes())
    _, body, _ = partition_body(phase_kernel_path(source, config(directory)))
    return body


FOLDED = [
    # 6 x 6 x 4, 3 x 3 at stride 1, 8 outputs: C' = 2 at SIMD 2.
    ("3x3-stride-1", 2, 2, (72, "'{4, 4, 4, 3, 6}", "'{12, 2, 0, 12, 1}")),
    # CNV's first layer at SIMD 3 (C' = 1) and PE 16: the nest the overlap note gives.
    ("cnv-conv0", 3, 16, (1024, "'{30, 30, 4, 3, 3}", "'{32, 1, 0, 32, 1}")),
    # Stride 2, dilation 2 over 8 x 8 x 2 at PE 4 (one output fold): rows and columns
    # 1, 3, 5 and 7 unread.
    ("strided-dilated", 1, 4, (128, "'{2, 2, 3, 3, 2}", "'{32, 4, 32, 4, 1}")),
]


def folded(name: str, simd: int, pe: int, directory: Path) -> ModelWrapper:
    """The single layer ``name``'s body, its SIMD and PE chosen on its node."""
    body = single_layer(name, directory)
    (node,) = body.graph.node
    kernel_op(body, node).save({"compute.packed.simd": simd, "compute.packed.pe": pe})
    return body


@pytest.mark.parametrize(("name", "simd", "pe", "nest"), FOLDED)
def test_the_image_enters_once_and_its_channel_reads_the_windows(
    name: str, simd: int, pe: int, nest: tuple[int, str, str], tmp_path: Path
) -> None:
    body = folded(name, simd, pe, tmp_path)
    (node,) = body.graph.node
    point, _ = configured_root(body, name)
    channel = getattr(point, node.input[0])
    ends = channel.ends
    assert ends.source.sequence.form == vector_major(channel.tensor.shape, simd)
    assert ends.source.sequence.markers == ()
    (marker,) = ends.sink.sequence.markers
    kernel = getattr(point, node.name).compute
    assert marker.beats == kernel.schedule.beat_count // kernel.y.presented.form.beats
    instances = dict(point.module.fragment.instances)
    parameters = dict(instances[f"{node.input[0]}.adapter.input_gen.input_gen"].parameters)
    assert (parameters["FM_SIZE"], parameters["DIMS"], parameters["COEFS"]) == nest


@requires_xsim
@pytest.mark.parametrize(("name", "simd", "pe"), [case[:3] for case in FOLDED])
def test_the_folded_layer_computes_in_xsim_what_its_kernel_op_computes(
    name: str, simd: int, pe: int, tmp_path: Path
) -> None:
    body = folded(name, simd, pe, tmp_path / "build")
    label = f"WindowedMatMul-{name}"
    check_parity(
        body,
        tmp_path / "xsim",
        inputs=boundary_inputs(body, zlib.crc32(label.encode())),
        label=label,
    )


def cnv_exported(directory: Path) -> ModelWrapper:
    """CNV_W2A2 as Brevitas exports it (QONNX, opset 13) into ``directory``, and beside it
    its preprocessing (``kernel_ops.tfc.PREPROCESSING``), as ``kernel_ops.tfc.exported``
    exports TFC."""
    import torch  # noqa: PLC0415
    from brevitas.export import export_qonnx  # type: ignore[import-untyped]  # noqa: PLC0415

    from finn.util.pytorch import ToTensor  # noqa: PLC0415
    from finn.util.test import get_test_model_trained  # noqa: PLC0415

    network = directory / CNV_EXPORT
    directory.mkdir(parents=True, exist_ok=True)
    trained = get_test_model_trained("CNV", 2, 2)  # type: ignore[no-untyped-call]
    export_qonnx(trained, torch.randn(CNV_SHAPE), network, opset_version=13)
    preprocessing = ToTensor()  # type: ignore[no-untyped-call]
    export_qonnx(preprocessing, torch.randn(CNV_SHAPE), directory / PREPROCESSING, opset_version=13)
    return ModelWrapper(str(network))


@pytest.mark.slow
def test_cnvs_six_convolutions_each_convert_to_one_windowed_matmul(tmp_path: Path) -> None:
    """CNV_W2A2 prepared as its build states it (the preprocessing merged, the input
    UINT8, the top label selected; the equivalence check off, as the conversion is what
    is checked), then ``ToKernelOps`` for Ultra96 at 5 ns in the Zynq shell."""
    settings = KernelBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(board="Ultra96", period_ns=5.0, shell="pynq"),
        preparation=GraphPreparation(
            preprocessing=str(tmp_path / PREPROCESSING), input_datatype="UINT8", topk=1
        ),
        verify_steps=[],
        enable_build_pdb_debug=False,
        save_intermediate_models=False,
    )
    prepared = phase_graph_preparation(cnv_exported(tmp_path), settings)
    conversion = ToKernelOps(ULTRA96)
    model = prepared.transform(conversion)
    windowed = [outcome for outcome in conversion.outcomes if outcome.op == "WindowedMatMul"]
    assert [outcome.nodes for outcome in windowed] == [
        (f"Im2Col_{layer}", f"MatMul_{layer}") for layer in range(6)
    ]
    assert not [node for node in model.graph.node if node.op_type == "Im2Col"]
    assert between_kernel_ops(model) == (
        "MaxPoolNHWC_0",
        "MaxPoolNHWC_1",
        "Transpose_1",
        "Reshape_0",
    )
