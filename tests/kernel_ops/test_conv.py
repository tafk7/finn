# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conv beyond its spec (``support.op_specs.conv``): the window
reaches the hardware as its image channel's reorder, and CNV's convolutions convert.

- A single convolution through the builder's kernel-path phases: the image enters once,
  in row-major order (the boundary presents the frame, not the patches: ``unreplayed``),
  and the image channel's adapter is an ``input_gen`` whose nest is the window's
  (``FM_SIZE = H * W * C'``, ``DIMS = (OH, OW, N / PE, KH, KW * C')``, ``COEFS = (W *
  C', C', 0, W * C', 1)``, ``C' = C / SIMD``), its TLAST closing each window.
- In XSim (``finn.harness.ops.check_parity``, the ``XSim`` executor), the folded
  single layers compute what the op's ``execute_node`` computes, every output word:
  CNV's first layer at SIMD 3 and PE 16 among them.
- Inside a partition: a Thresholding (NHWC, PE channels a beat) feeding a Conv
  (SIMD channels a beat) is one edge whose steps are the window's reorder, after a width
  conversion where PE and SIMD differ; in XSim the two compute what their
  ``execute_node`` do.
- CNV_W2A2 (``support.cnv``, prepared once a run: ``conftest.py``'s
  ``cnv_streamlined``) through ``ToKernelOps``: each of
  its six ``Im2Col`` and ``MatMul`` pairs converts to one Conv, and the host
  nodes between KernelOps are the two max pools, the flatten's Transpose and its
  Reshape.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from support.configure import configured
from support.kernel_path import config
from support.models import convert
from support.op_specs import conv
from support.op_specs.base import GENERAL, source
from support.tfc import ULTRA96, run_steps
from support.xsim import requires_xsim

from finn.core.graph.partition import KERNEL_OPS_DOMAIN, partition_body
from finn.custom_op.kernels import Conv
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import configured_network
from finn.dataflow.traversal import vector_major
from finn.harness.ops import check_parity
from finn.transformation.kernels import ToKernelOps
from finn.transformation.kernels.convert import between_kernel_ops, kernel_ops_by_anchor
from finn.transformation.kernels.cut import MakeCutChoices
from finn.util.draws import boundary_inputs


def test_the_kernel_op_conv_and_onnxs_conv_never_resolve_to_each_other() -> None:
    """The op type is ONNX's name in the KernelOps' domain: qonnx resolves a node by its
    domain, an ONNX ``Conv`` (the default domain) is no custom op, and conversion finds
    KernelOps by their anchor (``Im2Col``), never by a ``Conv``."""
    model, conversion = convert(conv.lowered().transform(InferShapes()))
    (node,) = model.graph.node
    assert (node.domain, node.op_type) == (KERNEL_OPS_DOMAIN, "Conv")
    assert isinstance(model.get_customop_wrapper(node), Conv)
    assert is_custom_op(KERNEL_OPS_DOMAIN, "Conv") and not is_custom_op("", "Conv")
    assert not is_custom_op(GENERAL, "Conv")
    assert ("", "Conv") not in kernel_ops_by_anchor()
    assert [op.op_type for op in kernel_ops_by_anchor()[GENERAL, "Im2Col"]] == ["Conv"]
    # qonnx's inference reads the node through its domain: its output stays the kernel's.
    stated = model.get_tensor_datatype("y")
    inference = InferDataTypes()  # type: ignore[no-untyped-call]
    again = model.transform(InferShapes()).transform(inference)
    assert again.get_tensor_datatype("y") == stated
    assert [o.op for o in conversion.outcomes] == ["Conv"]


def single_layer(name: str, directory: Path) -> ModelWrapper:
    """The positive graph ``name`` through the kernel-path phases: the partition's body."""
    source = conv.SPEC.positive[name]().transform(InferShapes())
    _, body, _ = partition_body(run_steps(source, config(directory)))
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
    """The single layer ``name``'s body, its SIMD and PE chosen on its node by hand, and
    what that opens (pumping, at SIMD 2 and more) set as the cut sets it (``MakeCutChoices``)."""
    body = single_layer(name, directory)
    (node,) = body.graph.node
    configured(body, {node.name: {"simd": simd, "pe": pe}})
    return body.transform(MakeCutChoices())


@pytest.mark.parametrize(("name", "simd", "pe", "nest"), FOLDED)
def test_the_image_enters_once_and_its_channel_reads_the_windows(
    name: str, simd: int, pe: int, nest: tuple[int, str, str], tmp_path: Path
) -> None:
    body = folded(name, simd, pe, tmp_path)
    (node,) = body.graph.node
    point, _ = configured_network(body, name)
    channel = point.channel(node.input[0])
    streams = channel.streams
    assert streams.source.sequence.form == vector_major(channel.tensor.shape, simd)
    assert streams.source.sequence.markers == ()
    (marker,) = streams.sink.sequence.markers
    kernel = point.network.member(node.name).compute
    assert marker.beats == kernel.schedule.iterations // kernel.y.stream.sequence.form.beats
    instances = dict(point.module.fragment.instances)
    parameters = dict(instances[f"{node.input[0]}.adapter.input_gen.input_gen"].parameters)
    assert (parameters["FM_SIZE"], parameters["DIMS"], parameters["COEFS"]) == nest


@requires_xsim
@pytest.mark.parametrize(("name", "simd", "pe"), [case[:3] for case in FOLDED])
def test_the_folded_layer_computes_in_xsim_what_its_kernel_op_computes(
    name: str, simd: int, pe: int, tmp_path: Path
) -> None:
    body = folded(name, simd, pe, tmp_path / "build")
    label = f"Conv-{name}"
    check_parity(
        body,
        tmp_path / "xsim",
        inputs=boundary_inputs(body, zlib.crc32(label.encode())),
        label=label,
    )


def thresholded(pe: int, simd: int) -> ModelWrapper:
    """A Thresholding of a 6 x 6 x 4 INT4 image (NHWC, UINT2 levels) at ``pe``, and a 3 x 3
    Conv of its levels to 8 outputs at ``simd`` and PE 2: converted for
    ``TARGET`` (Ultra96's part, the ``ip`` shell), its choices on its nodes."""
    threshold = helper.make_node(
        "MultiThreshold",
        ["x", "t"],
        ["a"],
        name="mt",
        domain=GENERAL,
        out_dtype="UINT2",
        data_layout="NHWC",
    )
    patches = conv.im2col((1, 6, 6, 4), (3, 3))
    patches.input[0] = "a"
    nodes = [threshold, patches, helper.make_node("MatMul", ["p", "w"], ["y"], name="mm")]
    stored = {
        "t": (np.array([[-2.0, 0.0, 2.0]] * 4), "INT4"),
        "w": (conv.weights(36, 8, -8, 7), "INT4"),
    }
    graph = source(nodes, {"x": ([1, 6, 6, 4], "INT4")}, stored).transform(InferShapes())
    model, conversion = convert(graph)
    assert [outcome.op for outcome in conversion.outcomes] == ["Thresholding", "Conv"]
    first, second = model.graph.node
    configured(
        model,
        {
            first.name: {"pe": pe},
            second.name: {"simd": simd, "pe": 2},
        },
    )
    return model


INSIDE = [(2, 2, ("reorder", "markers")), (4, 2, ("width_conversion", "reorder", "markers"))]


@pytest.mark.parametrize(("pe", "simd", "steps"), INSIDE)
def test_a_window_inside_the_partition_is_its_edges_reorder(
    pe: int, simd: int, steps: tuple[str, ...]
) -> None:
    # Converted, not cut: its channels' choices created as the cut creates them.
    point, _ = configured_network(thresholded(pe, simd).transform(MakeCutChoices()), "inside")
    assert tuple(hop.step.value for hop in point.channel("a").steps.hops) == steps
    labels = [label for label, _ in point.module.fragment.instances if label.startswith("a.")]
    assert "a.adapter.input_gen.input_gen" in labels


@requires_xsim
@pytest.mark.parametrize(("pe", "simd"), [case[:2] for case in INSIDE])
def test_a_window_inside_the_partition_computes_in_xsim_what_its_kernel_ops_compute(
    pe: int, simd: int, tmp_path: Path
) -> None:
    model = thresholded(pe, simd)
    label = f"inside-{pe}-{simd}"
    check_parity(
        model, tmp_path, inputs=boundary_inputs(model, zlib.crc32(label.encode())), label=label
    )


@pytest.mark.slow
def test_cnvs_six_convolutions_each_convert_to_one_conv(cnv_streamlined: Path) -> None:
    """CNV_W2A2 prepared as its build states it (the preprocessing merged, the input
    UINT8, the top label selected), then ``ToKernelOps`` for Ultra96 at 5 ns in the Zynq
    shell."""
    prepared = ModelWrapper(str(cnv_streamlined))
    conversion = ToKernelOps(ULTRA96)
    model = prepared.transform(conversion)
    windowed = [outcome for outcome in conversion.outcomes if outcome.op == "Conv"]
    assert [outcome.nodes for outcome in windowed] == [
        (f"Im2Col_{layer}", f"MatMul_{layer}") for layer in range(6)
    ]
    assert not [node for node in model.graph.node if node.op_type == "Im2Col"]
    # Each KernelOp's own choices created as it converts, each set by its default policy:
    # 93 values on CNV's 17 KernelOps, every choice in play (each MatMul and Conv's core and
    # its pumping, each Thresholding's AXI-Lite and UltraRAM stages among them).
    converted = [
        kernel_op(model, node) for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN
    ]
    assert len(converted) == 17
    assert sum(len(op.choices()) for op in converted) == 93
    assert between_kernel_ops(model) == (
        "MaxPoolNHWC_0",
        "MaxPoolNHWC_1",
        "Transpose_1",
        "Reshape_0",
    )
