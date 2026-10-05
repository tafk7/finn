# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX operators to KernelOps (``finn.custom_op.kernels``).

``resolve_target`` maps a build's part, clock period and shell to its
``Target`` (the part and the platform, its clock period and capabilities) from
two tables: a part's device capabilities (``DEVICES``) and a shell's interface
capabilities (``SHELLS``). Part and shell names stop here: kernels see
capabilities only (``finn.kernels.target``).

``ToKernelOps`` states the build target once, in the one place
``target(model)`` reads it, the model's ``finn.platform`` metadata, imports the
domain at its ``opset_version`` when the model does not import it yet
(inserting a node never raises a model's import), and rewrites each node it
can bind:

- ``MatMul`` (the ONNX operator) into a ``MatMul`` KernelOp;
- ``MultiThreshold`` with ``out_scale`` 1, an integral ``out_bias`` and its
  channels on the input's last axis (``NHWC``, or a 2-D input) into a
  ``Thresholding`` with that ``bias``.

Other nodes are left alone. The nodes keep their names, inputs and outputs.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from onnx import helper
from qonnx.transformation.base import Transformation

import finn.custom_op.kernels as domain
from finn.custom_op.kernels.base import write_target
from finn.kernels.target import DspBlock, Platform, Target

DOMAIN = domain.__name__

# Device capabilities by part pattern (fnmatch on the lower-case part), first match
# wins: (pattern, dsp, uram, uram_init, aie). UltraScale+ ignores an UltraRAM's INIT
# (packaging probe 4); Versal is unverified and refused until a synthesis run says
# otherwise (refusing is the side to reverse). A part matching no row is refused.
DEVICES: tuple[tuple[str, DspBlock, bool, bool, bool], ...] = (
    ("xc7*", DspBlock.DSP48E1, False, False, False),  # 7 series: no UltraRAM
    ("xczu7ev-*", DspBlock.DSP48E2, True, False, False),  # ZCU104
    ("xczu28dr-*", DspBlock.DSP48E2, True, False, False),  # ZCU111, RFSoC2x2
    ("xczu48dr-*", DspBlock.DSP48E2, True, False, False),  # RFSoC4x2
    ("xck26-*", DspBlock.DSP48E2, True, False, False),  # KV260
    (
        "xczu*",
        DspBlock.DSP48E2,
        False,
        False,
        False,
    ),  # other Zynq UltraScale+ (ZU3EG, ZU9EG): none stated
    ("xcu*", DspBlock.DSP48E2, True, False, False),  # Alveo (Virtex UltraScale+)
    ("xcvc*", DspBlock.DSP58, True, False, True),  # Versal AI Core (VCK190)
    ("xcve*", DspBlock.DSP58, True, False, True),  # Versal AI Edge (VEK280)
    ("xcv80-*", DspBlock.DSP58, True, False, False),  # V80 (Versal HBM)
)

# Interface capabilities by shell (the builder's ``ShellFlowType`` values):
# (clk2x, control_ports, memory_ports). No shell drives ap_clk2x yet; Vitis and SLASH
# take no AXI-Lite on a compute partition (packaging P7, P8); the shells' memory
# ports are their IODMAs', none a compute partition's. The Zynq shell's AXI
# interconnect has at most 64 masters, two of them the IODMAs'. Without a shell (a
# stitched IP, a harness: ``None``) nothing is stated away: a doubled clock, one
# AXI-Lite port, no memory port.
SHELLS: dict[str | None, tuple[bool, int, int]] = {
    None: (True, 1, 0),
    "vivado_zynq": (False, 62, 0),
    "vitis_alveo": (False, 0, 0),
    "slash_alveo": (False, 0, 0),
}


def resolve_target(part: str, period_ns: float, shell: str | None = None) -> Target:
    """The target of a build for ``part`` at ``period_ns``, integrated by ``shell``
    (none: a stitched IP), from the capability tables."""
    for pattern, dsp, uram, uram_init, aie in DEVICES:
        if fnmatch(part.lower(), pattern):
            break
    else:
        raise ValueError(
            f"no capability row for part {part!r} (finn.transformation.kernels.convert.DEVICES)"
        )
    if shell not in SHELLS:
        named = sorted(name for name in SHELLS if name is not None)
        raise ValueError(f"no capability row for shell {shell!r} (one of {named})")
    clk2x, control_ports, memory_ports = SHELLS[shell]
    platform = Platform(
        period_ns=float(period_ns),
        dsp=dsp,
        uram=uram,
        uram_init=uram_init,
        clk2x=clk2x,
        control_ports=control_ports,
        memory_ports=memory_ports,
        aie=aie,
    )
    return Target(part, platform)


def _attributes(node: Any) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _thresholding(model: Any, node: Any) -> Any | None:
    """A Thresholding for an integer MultiThreshold over the input's last axis, if it is one."""
    attributes = _attributes(node)
    if float(attributes.get("out_scale", 1.0)) != 1.0:
        return None
    bias = float(attributes.get("out_bias", 0.0))
    if bias != int(bias):
        return None
    layout = attributes.get("data_layout", b"NCHW")
    layout = layout.decode() if isinstance(layout, bytes) else layout
    dims = model.get_tensor_shape(node.input[0])
    if layout != "NHWC" and (dims is None or len(dims) > 2):
        return None  # channels on axis 1 of a wider tensor, or a tensor not known yet
    return helper.make_node(
        "Thresholding",
        list(node.input),
        list(node.output),
        name=node.name,
        domain=DOMAIN,
        bias=int(bias),
    )


class ToKernelOps(Transformation):  # type: ignore[misc]
    """Each node a KernelOp binds rewritten as one, the target stated in the model."""

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.target = target

    def apply(self, model: Any) -> tuple[Any, bool]:
        write_target(model, self.target)
        if DOMAIN not in model.get_opset_imports():
            model.set_opset_import(DOMAIN, domain.opset_version)
        graph = model.graph
        for index, node in enumerate(list(graph.node)):
            new: Any
            if node.op_type == "MatMul" and node.domain == "":
                new = helper.make_node(
                    "MatMul", list(node.input), list(node.output), name=node.name, domain=DOMAIN
                )
            elif node.op_type == "MultiThreshold":
                new = _thresholding(model, node)
                if new is None:
                    continue
            else:
                continue
            graph.node.remove(node)
            graph.node.insert(index, new)
        return model, False


__all__ = ["DEVICES", "SHELLS", "ToKernelOps", "resolve_target"]
