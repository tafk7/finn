# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Conv KernelOp: a lowered convolution, qonnx ``Im2Col`` followed by the ONNX
``MatMul`` that reads its patches, on ``MatMulKernel`` read through a sliding window.

The build lowers each ONNX ``Conv`` to this pair before conversion
(``LowerConvsToMatMul``, in ``step_streamline``). qonnx's ``Im2Col`` is NHWC and orders
a patch ``(kh, kw, c)``, ``c`` innermost, and ``LowerConvsToMatMul`` stores the (K, N)
weights in that order. The op covers both nodes, anchored at the ``Im2Col``, so the
patches never become a tensor in the graph or in hardware: the node reads the image
``(1, H, W, C)`` and the weights, and writes ``(1, OH, OW, N)``, which the kernel states
at the image's rank. The window is a reorder
on the image's input channel (``finn.dataflow.gemm.Window``), not a kernel of its own.

The op type is ``Conv``, in the KernelOps' domain. qonnx looks an op up by its domain
first, so it never collides with ONNX's ``Conv``.

The pattern refuses padding (``window-pads``), a depthwise ``Im2Col``
(``window-depthwise``), a window, stride or dilation that is not two positive integers
(``window-geometry``), patches that no ``MatMul`` reads as A (``window-unread``), and
weights that are not 2-D (``matmul-batched``). Conversion also refuses patches that
another node reads (``match-interior-exposed``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qonnx.custom_op.general.im2col import im2col_indices_nchw

from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import known_shape, shape
from finn.core.reporting.findings import Finding
from finn.core.space import Rejected
from finn.custom_op.kernels.base import Match, node_attributes, refused, unstated
from finn.custom_op.kernels.matmul import MatMul
from finn.dataflow.gemm import Window
from finn.kernels.matmul import MatMulKernel

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

IM2COL = ("qonnx.custom_op.general", "Im2Col")

#: The semantic attributes and the ``Im2Col`` attribute each is read from.
GEOMETRY = {"window": "kernel_size", "stride": "stride", "dilation": "dilations"}


def _pair(value: object) -> tuple[int, int] | None:
    """``value`` as (rows, columns) of positive integers, or None."""
    found = tuple(value) if isinstance(value, (list, tuple)) else ()
    if len(found) != 2 or any(type(each) is not int or each < 1 for each in found):
        return None
    return found[0], found[1]


class Conv(MatMul):
    """qonnx Im2Col then ONNX MatMul, bound to ``MatMulKernel`` read through a window."""

    op_type = "Conv"
    op_version = MatMulKernel.version
    kernel = MatMulKernel
    formals = (*MatMul.formals, "window")
    semantic = {name: ("ints", True, []) for name in GEOMETRY}
    anchor = IM2COL

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        owner, patches = cls.op_type, node.output[0]
        attributes = node_attributes(node)
        findings: list[Finding] = []
        pads = [int(each) for each in attributes.get("pad_amount", (0, 0, 0, 0))]
        if any(pads):
            message = f"pads {pads}: a window reads only the image; a pad is its own stage"
            findings.append(refused(owner, "window-pads", message, pads=tuple(pads)))
        if int(attributes.get("depthwise", 0)):
            message = "a depthwise Im2Col: its MatMul is per channel, not the window's dense one"
            findings.append(refused(owner, "window-depthwise", message))
        geometry: dict[str, object] = {}
        for name, attribute in GEOMETRY.items():
            stated = attributes.get(attribute, [1, 1] if name == "dilation" else None)
            pair = _pair(stated)
            if pair is None:
                message = f"{attribute} {stated} is not two positive integers (rows, columns)"
                findings.append(refused(owner, "window-geometry", message, attribute=attribute))
            else:
                geometry[name] = list(pair)
        readers = [
            reader
            for reader in model.find_consumers(patches)
            if (reader.domain, reader.op_type) == ("", "MatMul") and reader.input[0] == patches
        ]
        if not readers:
            message = f"no MatMul reads the patches {patches} as its activations"
            findings.append(refused(owner, "window-unread", message, tensor=patches))
        else:
            b = readers[0].input[1]
            dims = known_shape(model, b)
            if dims is None:
                findings.append(unstated(owner, f"the shape of the weights {b} is not known"))
            elif len(dims) != 2:
                message = f"the weights {b} are {len(dims)}-D {dims}, not a (k, n) matrix"
                findings.append(refused(owner, "matmul-batched", message, shape=dims))
        if findings:
            return Rejected(tuple(findings))
        return Match((node, readers[0]), geometry)

    def window(self) -> Window:
        """The window its semantic attributes give over its image's (H, W)."""
        label, image = self.label, self.image()
        found: dict[str, tuple[int, int]] = {}
        for name in GEOMETRY:
            pair = _pair(self.get_nodeattr(name))
            if pair is None:
                raise KernelOpError.refused(
                    "geometry-invalid",
                    self.op_type,
                    f"{label}: {name} is not two positive integers",
                    node=label,
                )
            found[name] = pair
        try:
            return Window((image[1], image[2]), found["window"], found["stride"], found["dilation"])
        except ValueError as error:
            raise KernelOpError.refused(
                "window-invalid", self.op_type, f"{label}: {error}", node=label
            ) from error

    def image(self) -> tuple[int, int, int, int]:
        """The image's (1, H, W, C)."""
        x, label = self.onnx_node.input[0], self.label
        dims = shape(self.model(), x, label)
        if len(dims) != 4 or dims[0] != 1:
            raise KernelOpError.refused(
                "image-not-nhwc",
                self.op_type,
                f"{label}: the image {x} is {dims}, not one (1, H, W, C)",
                node=label,
                tensor=x,
            )
        return dims[0], dims[1], dims[2], dims[3]

    def shaped(self) -> dict[str, object]:
        """The ``window`` over the image; the weights' rows must be the window's taps times
        the image's channels."""
        label, (x, w) = self.label, self.onnx_node.input
        channels, window = self.image()[3], self.window()
        k, _ = self.weights()
        if k != window.tap_count * channels:
            raise KernelOpError.refused(
                "weights-not-window",
                self.op_type,
                f"{label}: the weights {w} have {k} rows, not the {window.size} window's "
                f"{window.tap_count} taps of the {channels} channels of {x}",
                node=label,
                tensor=w,
            )
        return {"window": window}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        """The reference: qonnx's Im2Col of the image, then MatMul's product with the
        weights."""
        x, w = self.onnx_node.input
        window, image = self.window(), np.asarray(context[x])
        batch, height, width, channels = image.shape
        (taps_h, taps_w), (rows, columns) = window.size, window.output
        cols = im2col_indices_nchw(  # type: ignore[no-untyped-call]
            image.transpose(0, 3, 1, 2),
            height,
            width,
            taps_h,
            taps_w,
            [0, 0, 0, 0],
            window.stride[0],
            window.stride[1],
            dilation_h=window.dilation[0],
            dilation_w=window.dilation[1],
        )
        # As qonnx's Im2Col: (C, KH, KW, OH, OW) to (OH, OW, KH, KW, C).
        patches = (
            cols.reshape(batch, channels, taps_h, taps_w, rows, columns)
            .transpose(0, 4, 5, 2, 3, 1)
            .reshape(batch, rows, columns, taps_h * taps_w * channels)
        )
        context[self.onnx_node.output[0]] = self.product(patches, context[w])


__all__ = ["Conv"]
