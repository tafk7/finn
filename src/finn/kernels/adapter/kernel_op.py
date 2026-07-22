############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``KernelOp`` — the FINN adapter that lets a model-free :class:`Kernel` back a real
ONNX node and answer FINN's ``HWCustomOp`` contract.

The engine (``space/``) is pure and graph-free: its getters take a :class:`Context`
(shapes/datatypes as data) and a resolved :class:`Point`. FINN, by contrast, builds a
fresh transient op instance per ``getCustomOp(node)`` call **with no model**, yet calls
model-free getters (``get_exp_cycles()``, ``get_folded_input_shape(ind)``). This adapter
bridges the two by the discipline FINN's own MVAU already uses: **geometry is baked into
nodeattrs**, and the getters rebuild a ``Context`` *from nodeattrs* — never from the
graph. Three jobs (kernelop-tensor-block-stream.md §7; consumer-surface-model.md Tier 0-3):

  1. **Bridge** — ``_context()`` builds a model-free Context from the baked per-interface
     shape/dtype nodeattrs, keyed to the kernel's literal interface tensor names.
  2. **Configure** — ``_point()`` reads the design axes (``implementation``/PE/SIMD/…)
     off nodeattrs and ``kernel.configure``s them into a Point (or raises on Illegal).
  3. **Project** — each FINN getter delegates to the matching Kernel getter, adapting the
     ``(ind)`` FINN signature to the engine's ``(point, context, ind)``.

Value-dependent work (datatype narrowing that needs weight VALUES) is NOT done here —
it belongs to the model-bearing ABC calls FINN already makes (``infer_node_datatype``),
which read the live graph and bake results into tensor annotations. The getter path uses
a shape/dtype-faithful **placeholder** initializer for weight interfaces so the schema
resolves; the Tier-3 getters read only shapes/widths/cycles, never value-derived dtypes.

Specialization is by the ``implementation`` nodeattr (mapping onto the engine's
``implementation`` selection axis), NOT by FINN's domain mutation — one KernelOp class
serves every compute impl (consumer-surface-model.md R11).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.kernels.space import Context, Illegal, Role
from .nodeattr_registry import axis_nodeattr_types


@dataclass(frozen=True)
class PortSpec:
    """Binds one of the kernel's interfaces to a FINN port index + node tensor slot.

    Attributes:
        iface: the Kernel interface name (== its Context tensor key, e.g. "inp").
        direction: "in" or "out" — which side of the node this port is on.
        index: the FINN port index within that direction (0=activation, 1=weights).
        role: the port-taxonomy Role, so the adapter knows which ports are params
            (needing a placeholder initializer at getter time).
    """

    iface: str
    direction: str
    index: int
    role: Role


class KernelOp(HWCustomOp):
    """Base FINN adapter over a :class:`~finn.kernels.space.kernel.Kernel`.

    Subclasses implement :meth:`kernel` (the design-space object) and :meth:`ports`
    (the interface↔port binding). Everything else — the nodeattr registry, the
    Context bridge, and all the HWCustomOp getters — is derived generically here.
    """

    # -- subclass hooks -----------------------------------------------------

    @abstractmethod
    def kernel(self):
        """Return this op's :class:`Kernel` (the design space). Zero-arg factory
        result, e.g. ``mvau_kernel()``."""
        raise NotImplementedError(f"{type(self).__name__}.kernel()")

    @abstractmethod
    def ports(self) -> tuple[PortSpec, ...]:
        """The interface↔FINN-port binding for this op."""
        raise NotImplementedError(f"{type(self).__name__}.ports()")

    # -- nodeattr schema ----------------------------------------------------

    def get_nodeattr_types(self):
        """HWCustomOp base attrs + the schema-derived design axes + baked geometry.

        The design axes come straight from the kernel schema (the R12 dissolution —
        ``nodeattr_registry``); the geometry attrs (``<iface>_shape``/``<iface>_dtype``)
        carry the per-interface tensor shape + datatype baked at convert/infer time so
        the getters are model-free."""
        attrs = super().get_nodeattr_types()
        attrs.update(axis_nodeattr_types(self.kernel().schema()))
        for port in self.ports():
            attrs[self._shape_key(port.iface)] = ("ints", False, [])
            attrs[self._dtype_key(port.iface)] = ("s", False, "")
        return attrs

    # -- the bridge: nodeattrs -> Context -----------------------------------

    def _context(self) -> Context:
        """Build a model-free Context from the baked geometry nodeattrs, keyed to the
        kernel's literal interface tensor names. A WEIGHT-role interface gets a
        shape/dtype-faithful placeholder initializer so the schema resolves; its VALUES
        are never read by the Tier-3 getters (they drive only value-derived dtypes,
        which are baked separately at infer time)."""
        shapes: dict[str, tuple[int, ...]] = {}
        datatypes: dict = {}
        initializers: dict[str, np.ndarray] = {}
        for port in self.ports():
            shape = tuple(int(d) for d in self.get_nodeattr(self._shape_key(port.iface)))
            dtype = DataType[self.get_nodeattr(self._dtype_key(port.iface))]
            shapes[port.iface] = shape
            datatypes[port.iface] = dtype
            if port.role in (Role.WEIGHT_SINK, Role.INDEX_SINK):
                initializers[port.iface] = np.zeros(shape, dtype=np.float32)
        return Context(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            fpgapart=self._fpgapart(),
        )

    def _fpgapart(self) -> str:
        # Placement/part is harness-owned; absent on a bare estimate node is fine.
        try:
            return self.get_nodeattr("fpgapart")
        except AttributeError:
            return ""

    # -- configure: nodeattrs -> Point --------------------------------------

    def _assignment(self) -> dict:
        """The design-axis values currently set on the node (the ones the user/flow
        pinned). Unset axes are omitted so the schema supplies their defaults."""
        from qonnx.util.basic import get_by_name

        out: dict = {}
        for name in self.kernel().schema().axis_names:
            if get_by_name(self.onnx_node.attribute, name) is not None:
                out[name] = self.get_nodeattr(name)
        return out

    def _point(self):
        """Resolve the current node config into a Point, raising on Illegal."""
        kernel = self.kernel()
        ctx = self._context()
        result = kernel.configure(ctx, self._assignment())
        if isinstance(result, Illegal):
            raise ValueError(
                f"{self.onnx_node.name}: configuration is illegal: "
                f"{'; '.join(result.reasons)}"
            )
        return kernel, ctx, result

    # -- Tier-3 getters: shapes / widths / datatypes ------------------------

    def get_normal_input_shape(self, ind=0):
        kernel, ctx, _ = self._point()
        return kernel.get_normal_input_shape(ctx, ind)

    def get_normal_output_shape(self, ind=0):
        kernel, ctx, _ = self._point()
        return kernel.get_normal_output_shape(ctx, ind)

    def get_input_datatype(self, ind=0):
        # Datatype is a pure Context read; no resolved point needed.
        kernel = self.kernel()
        return kernel.get_input_datatype(self._context(), ind)

    def get_output_datatype(self, ind=0):
        kernel = self.kernel()
        return kernel.get_output_datatype(self._context(), ind)

    def get_folded_input_shape(self, ind=0):
        kernel, ctx, point = self._point()
        return kernel.get_folded_input_shape(point, ctx, ind)

    def get_folded_output_shape(self, ind=0):
        kernel, ctx, point = self._point()
        return kernel.get_folded_output_shape(point, ctx, ind)

    def get_instream_width(self, ind=0):
        kernel, ctx, point = self._point()
        return kernel.get_instream_width(point, ctx, ind)

    def get_outstream_width(self, ind=0):
        kernel, ctx, point = self._point()
        return kernel.get_outstream_width(point, ctx, ind)

    # -- cost / iteration ---------------------------------------------------

    def get_exp_cycles(self):
        kernel, ctx, point = self._point()
        return int(kernel.get_exp_cycles(point, ctx))

    def get_number_output_values(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    # -- base ABC (model-bearing) -------------------------------------------

    def infer_node_datatype(self, model):
        """Derive and propagate output datatypes into the graph. This is the one
        model-bearing hook where value-dependent derivation happens: we resolve a Point
        against a Context built from the *live graph* (real weight VALUES available), so
        the ``outputDataType`` derived is exact — the graph dtype when the op just
        forwards it, or the weight-derived accumulator type under ``noActivation``.
        Annotates the node's output tensors; idempotent."""
        kernel = self.kernel()
        ctx = self._remap_context(Context.from_model(model, self._fpgapart()))
        result = kernel.configure(ctx, self._assignment())
        if isinstance(result, Illegal):
            raise ValueError(
                f"{self.onnx_node.name}: configuration is illegal during infer: "
                f"{'; '.join(result.reasons)}"
            )
        for port in self.ports():
            if port.direction == "out":
                odt = self._output_datatype_from_point(kernel, ctx, result, port.index)
                out_name = self.onnx_node.output[port.index]
                if out_name:
                    model.set_tensor_datatype(out_name, odt)

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        """The datatype to annotate on output port ``index`` after resolving against the
        live-graph context. Default: the plain Context read (output forwards the graph
        dtype). An op with a value-derived output type (e.g. MVAU's accumulator under
        ``noActivation``) overrides to read the derived off the ``point``."""
        return kernel.get_output_datatype(ctx, index)

    def _remap_context(self, ctx: Context) -> Context:
        """Re-key a graph-built Context from real tensor names to the kernel's literal
        interface names, so the kernel getters (which reference "inp"/"weights"/"out")
        line up with the node's actual tensors."""
        shapes, datatypes, initializers = {}, {}, {}
        for port in self.ports():
            tname = self._tensor_name(port)
            if tname in ctx.shapes:
                shapes[port.iface] = ctx.shapes[tname]
            if tname in ctx.datatypes:
                datatypes[port.iface] = ctx.datatypes[tname]
            init = ctx.initializer(tname)
            if init is not None:
                initializers[port.iface] = init
        return Context(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            fpgapart=ctx.fpgapart,
            toolchain_version=ctx.toolchain_version,
            clk_ns=ctx.clk_ns,
        )

    def make_shape_compatible_op(self, model):
        return super().make_const_shape_op(self.get_normal_output_shape())

    def execute_node(self, context, graph):
        raise NotImplementedError(
            f"{type(self).__name__}: execute_node is not part of the estimate-only "
            f"surface (Tier 0-3); implement it when functional exec is needed."
        )

    # -- geometry baking (called at convert/infer time, model in hand) ------

    def bake_geometry(self, model):
        """Snapshot per-interface shape + datatype from the live graph into nodeattrs,
        so the model-free getters can rebuild a Context. Call from a model-bearing
        context (convert/infer). Matches how FINN's MVAU stores MW/MH/dtypes."""
        for port in self.ports():
            tname = self._tensor_name(port)
            shape = model.get_tensor_shape(tname)
            if shape is not None:
                self.set_nodeattr(self._shape_key(port.iface), [int(d) for d in shape])
            dt = model.get_tensor_datatype(tname)
            self.set_nodeattr(self._dtype_key(port.iface), dt.name)

    # -- helpers ------------------------------------------------------------

    def _tensor_name(self, port: PortSpec) -> str:
        slot = self.onnx_node.input if port.direction == "in" else self.onnx_node.output
        return slot[port.index]

    @staticmethod
    def _shape_key(iface: str) -> str:
        return f"{iface}_shape"

    @staticmethod
    def _dtype_key(iface: str) -> str:
        return f"{iface}_dtype"
