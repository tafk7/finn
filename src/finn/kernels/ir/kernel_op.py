############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``KernelOp`` — the FINN adapter that lets a :class:`_LegacyKernel` back a real ONNX node and
answer FINN's ``HWCustomOp`` contract.

The engine (``engine/``) is pure and graph-free: its getters take a :class:`Context`
(shapes/datatypes/VALUES as data) and a resolved :class:`Point`. This adapter sources that
Context from the **live model**, following the ONNX ownership rule: a node owns ONLY its
own internals (the design axes ``backend``/PE/SIMD/…), never graph-owned facts
(tensor shapes, datatypes, weight values). Those come from the model at construction, not
from baked ``<iface>_shape``/``<iface>_dtype`` nodeattrs (which would be a second, staling
source of truth). The op is built model-bearing via ``model.get_customop_wrapper(node)``
(mirroring the brainsmith ``_ensure_ready``/``invalidate`` pattern) — it opts into the qonnx
model-aware contract with ``wants_model = True``, so ``get_customop_wrapper`` attaches the
model automatically; the model-built Context is cached and regenerated when a new model is
attached. Three jobs (kernelop-tensor-block-stream.md §7;
consumer-surface-model.md Tier 0-3):

  1. **Bridge** — ``_context()`` builds the Context from the ATTACHED model (shapes,
     dtypes, REAL initializer values), re-keyed to the kernel's literal interface names.
  2. **Configure** — ``_point()`` reads the design axes off nodeattrs and
     ``kernel.configure``s them into a Point (or raises on Illegal).
  3. **Project** — each FINN getter delegates to the matching _LegacyKernel getter, adapting the
     ``(ind)`` FINN signature to the engine's ``(point, context, ind)``.

Because the Context carries REAL weight values, value-derived dtypes (MVAU's accumulator
under ``noActivation``) are exact for the getters too — not correct-only-by-luck under a
placeholder. Full-fleet ``getCustomOp(node)`` → ``get_customop_wrapper(node)`` migration
(threading the model into every transient op) is deferred; the estimate-tier sites and tests
construct via ``model.get_customop_wrapper(node)``.

Specialization is by the ``backend`` nodeattr (mapping onto the engine's ``backend``
selection axis — the root axis whose value names the selected :class:`Backend` pool member),
NOT by FINN's domain mutation — one KernelOp class serves every compute impl
(consumer-surface-model.md R11).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from onnx import NodeProto

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.kernels.engine.context import Context
from finn.kernels.engine.point import Illegal
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from .nodeattr_registry import axis_nodeattr_types

# Model metadata prop carrying the phase-0 runtime-writability mandate. A metadata prop
# (run-level) rather than a nodeattr (per-node) because it constrains the whole design
# space, not one node's configuration. Written by ``step_configure_kernels`` from
# ``DataflowBuildConfig.runtime_writeable_weights``.
RUNTIME_WRITEABLE_PROP = "finn_kernels_runtime_writeable_weights"


@dataclass(frozen=True)
class TransformationResult:
    """The result of one kernel's ``infer_from`` — the graph edit to apply (the infer-seam
    CONTRACT, co-located with the :class:`KernelOp` base that declares ``infer_from``, so a
    kernel op never has to import the ``InferKernels`` transformation to name its own return
    type).

    Attributes:
        nodes_to_insert: kernel node(s) to insert into the graph.
        nodes_to_remove: frontend node(s) the kernel absorbed and replaces.
        metadata: optional, free-form transformation notes (unused by the driver).
    """

    nodes_to_insert: list[NodeProto]
    nodes_to_remove: list[NodeProto]
    metadata: dict[str, Any] = field(default_factory=dict)


class KernelOp(HWCustomOp):
    """Base FINN adapter over a :class:`~finn.kernels.model.kernel._LegacyKernel`.

    Subclasses implement only :meth:`kernel` (the design-space object). The
    interface↔node-slot binding is the kernel's own
    :class:`~finn.kernels.model.kernel.InterfaceSchema` list (name + direction + index +
    optional) — the adapter iterates ``self.kernel().interfaces`` directly, so there is no
    second binding object to keep in sync (F9). Everything else — the nodeattr registry, the
    Context bridge, and all the HWCustomOp getters — is derived generically here.
    """

    # Every KernelOp derives its shapes/dtypes/widths from live graph context, so it
    # opts into the qonnx model-aware contract (CustomOp.wants_model). This makes
    # ``model.get_customop_wrapper(node)`` attach the model automatically — the
    # graph-derived getters below would otherwise raise (see ``_context``).
    wants_model = True

    # -- subclass hooks -----------------------------------------------------

    @classmethod
    @abstractmethod
    def kernel(cls):
        """Return this op's :class:`_LegacyKernel` (the design space). Zero-arg factory
        result, e.g. ``mvau_kernel()``. A ``classmethod`` — the _LegacyKernel is op-class
        identity, independent of any node/model — so bare-node routing
        (``kernel_hw_language``) can reach the pool via the op class without
        instantiating the op."""
        raise NotImplementedError(f"{cls.__name__}.kernel()")

    # -- nodeattr schema ----------------------------------------------------

    def get_nodeattr_types(self):
        """HWCustomOp base attrs + the schema-derived design axes ONLY.

        The design axes come straight from the kernel schema (the R12 dissolution —
        ``nodeattr_registry``). Graph-owned geometry (tensor shapes/datatypes/values) is
        NOT baked onto the node — it is sourced from the live model at construction, per
        ONNX ownership. So there are no ``<iface>_shape``/``<iface>_dtype`` nodeattrs."""
        attrs = super().get_nodeattr_types()
        attrs.update(axis_nodeattr_types(self.kernel().compile()))
        return attrs

    # -- model attach + Context cache (brainsmith _ensure_ready pattern) ----

    def attach_model(self, model) -> "KernelOp":
        """Attach the live model this op belongs to, so the getters can source their
        Context (shapes/dtypes/REAL values) from it. Caches the built Context; a later
        attach with a different model regenerates it (``invalidate``). Returns ``self``
        so ``model.get_customop_wrapper(node)`` yields the attached op in one expression.

        This overrides the qonnx :meth:`CustomOp.attach_model` hook — a KernelOp opts
        into the model-aware contract via ``wants_model = True``, so this runs whenever
        the op is built through ``get_customop_wrapper``."""
        self._model = model
        self._context_cache = None
        return self

    def _context(self) -> Context:
        """Build (and cache) the Context from the ATTACHED model: per-interface shapes,
        datatypes, and REAL initializer values, re-keyed from the node's actual tensor
        names to the kernel's literal interface names ("inp"/"weights"/"out"). An OPTIONAL
        port the node did not wire is skipped, so the kernel sees the interface as absent
        (emergent existence). Raises if no model was attached — the getters are
        model-bearing by construction now, not model-free."""
        cached = getattr(self, "_context_cache", None)
        if cached is not None:
            return cached
        model = getattr(self, "_model", None)
        if model is None:
            raise ValueError(
                f"{type(self).__name__}: no model attached — construct via "
                f"model.get_customop_wrapper(node) so the Context can be built from the "
                f"live graph"
            )
        graph_ctx = Context.from_model(model, self._fpgapart_from(model))
        shapes: dict[str, tuple[int, ...]] = {}
        datatypes: dict = {}
        initializers: dict[str, np.ndarray] = {}
        for iface in self.kernel().interfaces:
            tname = self._tensor_name(iface)
            if tname is None:
                continue  # optional operand not wired on this node
            if tname in graph_ctx.shapes:
                shapes[iface.name] = graph_ctx.shapes[tname]
            if tname in graph_ctx.datatypes:
                datatypes[iface.name] = graph_ctx.datatypes[tname]
            init = graph_ctx.initializer(tname)
            if init is not None:
                initializers[iface.name] = init
        ctx = Context(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            fpgapart=graph_ctx.fpgapart,
            toolchain_version=graph_ctx.toolchain_version,
            clk_ns=graph_ctx.clk_ns,
            runtime_writeable=self._runtime_writeable_from(model),
        )
        self._context_cache = ctx
        return ctx

    def _runtime_writeable_from(self, model) -> dict:
        """Which delivered parameters this build MANDATES be driver-rewritable (phase 0).

        Sourced from a model METADATA PROP, not a nodeattr, because it is a run-level mandate
        on the design space rather than a per-node design choice — see ``decisions.md``,
        *"An owned initializer is DOWNSTREAM of its kernel"*.
        ``DataflowBuildConfig.runtime_writeable_weights`` is the user-facing surface and
        ``step_configure_kernels`` stamps it onto the model.

        The config surface is a single GLOBAL flag today while :class:`Context` keys it per
        interface. That asymmetry is deliberate: the engine models the general case, so the
        API can grow per-interface control later without touching anything below it."""
        try:
            flag = model.get_metadata_prop(RUNTIME_WRITEABLE_PROP)
        except AttributeError:
            return {}
        if flag is None or str(flag).lower() in ("", "0", "false"):
            return {}
        return {dp.iface: True for dp in self.kernel().delivered_parameters}

    def _fpgapart_from(self, model) -> str:
        # Placement/part is harness-owned; prefer the node's own attr, else empty.
        try:
            return self.get_nodeattr("fpgapart")
        except AttributeError:
            return ""

    # -- configure: nodeattrs -> Point --------------------------------------

    def _assignment(self) -> dict:
        """The node-owned values currently set on the node (the ones the user/flow pinned).
        Unset names are omitted so the schema supplies their defaults.

        Covers BOTH design axes and :class:`~finn.kernels.engine.attr.Attr` constants. The
        attr half is load-bearing and easy to lose: this is the gate deciding WHICH NODEATTRS
        REACH RESOLVE, so narrowing it to ``axis_names`` alone would silently drop the
        frontend-baked ``ActVal`` (infer bakes it; emit reads ``point.ActVal``) and bake a
        default 0 into the hardware — no exception, no Illegal. ``test_attr_round_trip.py``
        is the regression."""
        from qonnx.util.basic import get_by_name

        schema = self.kernel().compile()
        out: dict = {}
        for name in schema.axis_names | schema.attr_names:
            if get_by_name(self.onnx_node.attribute, name) is not None:
                out[name] = self.get_nodeattr(name)
        return out

    def _point(self):
        """Resolve the current node config into a Point, raising on Illegal.

        Impl-DEPENDENT: requires a committed backend. An unspecialized node
        (``backend`` unset — the "" sentinel) has no legal full point, so raise a
        LEGIBLE "unspecialized" error rather than a bare Illegal-ValueError (F1). Callers
        needing only impl-INDEPENDENT facts (normal shape/dtype, output-dtype publication)
        use the context-only getters."""
        from .routing import is_specialized

        if not is_specialized(self.onnx_node):
            raise ValueError(
                f"{self.onnx_node.name}: impl-dependent getter needs a committed backend, "
                f"but `backend` is unset (node is unspecialized). Specialize it "
                f"(Seam B) before folded-shape/width/cycle queries."
            )
        kernel = self.kernel()
        ctx = self._context()
        result = kernel.configure(ctx, self._assignment())
        if isinstance(result, Illegal):
            raise ValueError(
                f"{self.onnx_node.name}: configuration is illegal: "
                f"{'; '.join(result.reasons)}"
            )
        return kernel, ctx, result

    def first_feasible_backend(self) -> str | None:
        """The name of the first pool member feasible for this node's live Context, or ``None``
        — the model-aware bridge behind ``SpecializeKernels``' ``PerNodePolicy(first_feasible)``
        (Seam B). Sources the live Context (real weight VALUES, so value-derived feasibility is
        exact) and delegates to :meth:`_LegacyKernel.first_feasible_backend`.

        The resolve-time counterpart to infer-time ``can_infer_from`` (``compute/mvau/op.py``),
        which trials a bare-node ``_trial_context`` because the node is still a frontend op;
        here the node IS a kernel node, so the model-attached ``_context()`` is in hand."""
        return self.kernel().first_feasible_backend(self._context())

    # -- Tier-3 getters: shapes / widths / datatypes ------------------------

    def get_normal_input_shape(self, ind=0):
        # Normal (tensor) shape is impl-INDEPENDENT — a pure Context read; no committed
        # backend needed, so an unspecialized node answers it (the Seam A verify gate).
        return self.kernel().get_normal_input_shape(self._context(), ind)

    def get_normal_output_shape(self, ind=0):
        return self.kernel().get_normal_output_shape(self._context(), ind)

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
        """Derive and propagate output datatypes into the graph. Attaches ``model`` (so the
        Context carries real weight VALUES) and annotates the node's output tensors;
        idempotent.

        The realized output dtype is BACKEND-SCOPED (which datatypes exist / how they are
        value-optimized is a realization fact), so it is DEFERRED until a backend is
        committed: an UNSPECIALIZED node publishes the raw graph output dtype (a pure Context
        read); a SPECIALIZED node resolves the full schema and publishes the backend-derived
        ``outputDataType`` (e.g. the weight-derived accumulator under ``noActivation``)."""
        from .routing import is_specialized

        self.attach_model(model)
        if is_specialized(self.onnx_node):
            kernel, ctx, result = self._point()
            for iface in kernel.interfaces:
                if iface.direction == Direction.OUT:
                    odt = self._output_datatype_from_point(kernel, ctx, result, iface.index)
                    out_name = self.onnx_node.output[iface.index]
                    if out_name:
                        model.set_tensor_datatype(out_name, odt)
        else:
            # Unspecialized: no committed backend ⇒ no realized output dtype yet. Publish the
            # raw graph dtype (pure Context read) so downstream shape/dtype inference has a
            # value; specialization later refines it.
            kernel = self.kernel()
            ctx = self._context()
            for iface in kernel.interfaces:
                if iface.direction == Direction.OUT:
                    odt = kernel.get_output_datatype(ctx, iface.index)
                    out_name = self.onnx_node.output[iface.index]
                    if out_name:
                        model.set_tensor_datatype(out_name, odt)

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        """The datatype to annotate on output port ``index`` after resolving against the
        live-graph context. Default: the plain Context read (output forwards the graph
        dtype). An op with a value-derived output type (e.g. MVAU's accumulator under
        ``noActivation``) overrides to read the derived off the ``point``."""
        return kernel.get_output_datatype(ctx, index)

    def make_shape_compatible_op(self, model):
        # InferShapes hands us the model — attach it so the output-shape getter can source
        # its Context from the live graph.
        self.attach_model(model)
        return super().make_const_shape_op(self.get_normal_output_shape())

    def execute_node(self, context, graph):
        raise NotImplementedError(
            f"{type(self).__name__}: execute_node is not part of the estimate-only "
            f"surface (Tier 0-3); implement it when functional exec is needed."
        )

    # -- helpers ------------------------------------------------------------

    def _tensor_name(self, iface: InterfaceSchema) -> str | None:
        """The node tensor name bound to ``iface``, or None when the slot is not wired — an
        OPTIONAL operand the node omitted (a shorter input list, or an empty-string slot).
        Callers skip a None-tensor interface so an absent optional operand contributes nothing.
        Reads ``node.input[index]``/``node.output[index]`` — the interface's own direction +
        index (F9: the binding lives on the interface, not a separate PortSpec)."""
        slot = self.onnx_node.input if iface.direction == Direction.IN else self.onnx_node.output
        if iface.index >= len(slot):
            return None
        name = slot[iface.index]
        return name if name else None
