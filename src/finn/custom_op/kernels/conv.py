# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Conv KernelOp: a lowered convolution, qonnx ``Im2Col`` then ONNX
``MatMul``, bound to ``MatMulKernel`` read through a window.

A convolution reaches graph preparation's output as ``Im2Col`` (qonnx's: NHWC by
definition, the patches' K ordered ``(kh, kw, c)``, ``c`` innermost) and the
``MatMul`` reading its patches as A against the (K, N) weights
``LowerConvsToMatMul`` stores in that order. This op covers the pair (a nested
pattern, anchored at the ``Im2Col``): the patches are never a tensor of the graph or
of the hardware. The node reads the image ``(1, H, W, C)`` and the weights, and
writes ``(1, OH, OW, N)``; its kernel is MatMul's on a dense form read through a
``Window`` (``finn.dataflow.gemm``), so the image's channel carries the window as
its reorder (an ``input_gen`` adapter), as FINN's sliding-window generator fed its
matrix-vector unit, with no kernel of its own.

Its op type is ONNX's name, ``Conv``, in the KernelOps' domain
(``finn.custom_op.kernels``): qonnx resolves a node's op by its domain first, so an
ONNX ``Conv`` (the default domain) is never this op, and this op never ONNX's. It
anchors at ``Im2Col``, not at a ``Conv``: graph preparation lowers each ONNX ``Conv``
before conversion.

Its pattern: an ``Im2Col`` whose patches a ``MatMul`` reads as A, the weights a
matrix (``matmul-batched`` otherwise). Refused by code: padding
(``window-pads``; a pad is a constant the window does not read, its own stage),
a depthwise ``Im2Col`` (``window-depthwise``: its MatMul is per channel, which the
window does not read), a geometry that is not two positive integers a field
(``window-geometry``), and patches no MatMul reads as A (``window-unread``). Patches
another node reads too are conversion's to refuse (``match-interior-exposed``).
Whether the kernels take a stride or a dilation is theirs to say (any the channel's
reorder carries). Its semantic attributes are ``kernel``, ``stride`` and
``dilation`` (rows, columns), as the ``Im2Col`` states them (``kernel_size``,
``stride``, ``dilations``).

Its reference, ``execute_node``: qonnx's ``Im2Col`` of the image, then MatMul's
(exact integer arithmetic for integer operands, carried in the image's container),
the ONNX the pattern covers. Its domain step is MatMul's on the image and the
weights: an ``Im2Col`` copies values, so the patches' partial sums are the image's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qonnx.custom_op.general.im2col import im2col_indices_nchw

from finn.core.space import Finding, Rejected
from finn.custom_op.kernels.base import (
    KernelOpError,
    Match,
    Shapes,
    known_shape,
    node_attributes,
    refused,
    shape,
    unstated,
)
from finn.custom_op.kernels.matmul import MatMul
from finn.dataflow.gemm import Window
from finn.kernels.matmul import MatMulKernel

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

IM2COL = ("qonnx.custom_op.general", "Im2Col")

#: The semantic attributes and the ``Im2Col`` attribute each is read from.
GEOMETRY = {"kernel": "kernel_size", "stride": "stride", "dilation": "dilations"}


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
        """The window its semantic attributes state over its image's (H, W)."""
        label, image = self.label, self.image()
        found: dict[str, tuple[int, int]] = {}
        for name in GEOMETRY:
            pair = _pair(self.get_nodeattr(name))
            if pair is None:
                raise KernelOpError(f"{label}: {name} is not two positive integers")
            found[name] = pair
        try:
            return Window((image[1], image[2]), **found)
        except ValueError as error:
            raise KernelOpError(f"{label}: {error}") from error

    def image(self) -> tuple[int, int, int, int]:
        """The image's (1, H, W, C)."""
        x, label = self.onnx_node.input[0], self.label
        dims = shape(self.model(), x, label)
        if len(dims) != 4 or dims[0] != 1:
            raise KernelOpError(f"{label}: the image {x} is {dims}, not one (1, H, W, C)")
        return dims[0], dims[1], dims[2], dims[3]

    def extents(self) -> tuple[int, int, int, dict[str, object]]:
        """(m, n, k): the output pixels, the weights' columns, the window's taps times the
        image's channels; and the ``window``."""
        label, (x, w) = self.label, self.onnx_node.input
        channels, window = self.image()[3], self.window()
        k, n = self.weights()
        if k != window.taps * channels:
            raise KernelOpError(
                f"{label}: the weights {w} have {k} rows, not the {window.kernel} window's "
                f"{window.taps} taps of the {channels} channels of {x}"
            )
        rows, columns = window.output
        return rows * columns, n, k, {"window": window}

    def output_tensors(self) -> Shapes:
        result = self.view("result_tensor")
        rows, columns = self.window().output
        return {
            self.onnx_node.output[0]: ((1, rows, columns, result.shape[-1]), result.element.dtype)
        }

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        """qonnx's Im2Col of the image, then MatMul's product with the weights (module
        docstring)."""
        x, w = self.onnx_node.input
        window, image = self.window(), np.asarray(context[x])
        batch, height, width, channels = image.shape
        (taps_h, taps_w), (rows, columns) = window.kernel, window.output
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
