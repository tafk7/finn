############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DataflowOp`` — the FINN adapter that lets a :class:`DataflowKernel` back a real ONNX node and
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
  3. **Project** — each FINN getter delegates to the matching DataflowKernel getter, adapting the
     ``(ind)`` FINN signature to the engine's ``(point, context, ind)``.

Because the Context carries REAL weight values, value-derived dtypes (MVAU's accumulator
under ``noActivation``) are exact for the getters too — not correct-only-by-luck under a
placeholder. Full-fleet ``getCustomOp(node)`` → ``get_customop_wrapper(node)`` migration
(threading the model into every transient op) is deferred; the estimate-tier sites and tests
construct via ``model.get_customop_wrapper(node)``.

Specialization is by the ``backend`` nodeattr (mapping onto the engine's ``backend``
selection axis — the root axis whose value names the selected :class:`Backend` pool member),
NOT by FINN's domain mutation — one DataflowOp class serves every compute backend
(consumer-surface-model.md R11).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from onnx import NodeProto, helper

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.kernels.engine.context import Context
from finn.kernels.engine.device import DeviceFacts
from finn.kernels.engine.point import Illegal
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from .nodeattr_registry import axis_nodeattr_types_for
from .routing import KERNEL_DOMAIN

# Model metadata prop carrying the phase-0 runtime-writability mandate. A metadata prop
# (run-level) rather than a nodeattr (per-node) because it constrains the whole design
# space, not one node's configuration. Written by ``step_configure_kernels`` from
# ``DataflowBuildConfig.runtime_writeable_weights``.
RUNTIME_WRITEABLE_PROP = "finn_kernels_runtime_writeable_weights"


@dataclass(frozen=True)
class TransformationResult:
    """The result of one kernel's ``infer_from`` — the graph edit to apply (the infer-seam
    CONTRACT, co-located with the :class:`DataflowOp` base that declares ``infer_from``, so a
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


class DataflowOp(HWCustomOp):
    """Base FINN adapter over a :class:`~finn.kernels.model.kernel.DataflowKernel`.

    Subclasses implement only :meth:`kernel` (the design-space object). The
    interface↔node-slot binding is the kernel's own
    :class:`~finn.kernels.model.kernel.InterfaceSchema` list (name + direction + index +
    optional) — the adapter iterates ``self.kernel().interfaces`` directly, so there is no
    second binding object to keep in sync (F9). Everything else — the nodeattr registry, the
    Context bridge, and all the HWCustomOp getters — is derived generically here.
    """

    # Every DataflowOp derives its shapes/dtypes/widths from live graph context, so it
    # opts into the qonnx model-aware contract (CustomOp.wants_model). This makes
    # ``model.get_customop_wrapper(node)`` attach the model automatically — the
    # graph-derived getters below would otherwise raise (see ``_context``).
    wants_model = True

    # -- subclass hooks -----------------------------------------------------

    @classmethod
    @abstractmethod
    def kernel(cls):
        """Return this op's :class:`DataflowKernel` (the design space). Zero-arg factory
        result, e.g. ``mvau_kernel()``. A ``classmethod`` — the DataflowKernel is op-class
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
        ONNX ownership. So there are no ``<iface>_shape``/``<iface>_dtype`` nodeattrs.

        Names are the pool UNION; each DOMAIN comes from the SELECTED realization once the
        node is specialized (F3). See
        :func:`~finn.kernels.ir.nodeattr_registry.axis_nodeattr_types_for` for why those are
        different questions."""
        attrs = super().get_nodeattr_types()
        attrs.update(
            axis_nodeattr_types_for(self.kernel().compile(), self._realized_space())
        )
        return attrs

    def _realized_space(self):
        """This node's per-realization design space, or ``None`` when it is unspecialized.

        The selection is read BARE-NODE (``routing.selected_backend_name``), not via
        ``self.get_nodeattr``. That is not an optimization: ``get_nodeattr`` calls
        ``get_nodeattr_def`` → ``get_nodeattr_types``, which is the method this feeds, so
        going through the op recurses until the stack ends.

        Deliberately failure-tolerant. ``get_nodeattr_types`` is on the qonnx attribute
        protocol and is called constantly, including on nodes mid-edit whose ``backend`` names
        a member that is not (yet) in the pool. A node that cannot produce a realized space
        falls back to the union — the pre-F3 behaviour — rather than making an attribute read
        raise."""
        from finn.kernels.model.backend import PoolError

        from .routing import selected_backend_name

        member = selected_backend_name(self.onnx_node)
        if member is None:
            return None
        try:
            return self.kernel().realized_space(member)
        except (PoolError, ValueError, KeyError):
            return None

    # -- model attach + Context cache (brainsmith _ensure_ready pattern) ----

    def attach_model(self, model) -> "DataflowOp":
        """Attach the live model this op belongs to, so the getters can source their
        Context (shapes/dtypes/REAL values) from it. Caches the built Context; a later
        attach with a different model regenerates it (``invalidate``). Returns ``self``
        so ``model.get_customop_wrapper(node)`` yields the attached op in one expression.

        This overrides the qonnx :meth:`CustomOp.attach_model` hook — a DataflowOp opts
        into the model-aware contract via ``wants_model = True``, so this runs whenever
        the op is built through ``get_customop_wrapper``."""
        self._model = model
        self._context_cache = None
        return self

    def attach_device(self, device: DeviceFacts | None) -> "DataflowOp":
        """Attach the build's :class:`~finn.kernels.engine.device.DeviceFacts` — the part,
        clock and toolchain this design targets.

        SEPARATE from :meth:`attach_model` because the two arrive in different phases and
        from different owners (F11). Graph givens come with the node; device facts come from
        whichever step holds ``cfg``, and there is no honest way to recover them from the
        graph — which is what ``_fpgapart_from`` tried to do, via a nodeattr no op declares.

        Invalidates the Context cache, since the facts are baked into it."""
        self._device = device
        self._context_cache = None
        return self

    def _device_facts(self) -> DeviceFacts:
        """This op's device facts, or the explicit unknown when none was attached.

        Unknown is a legitimate state, not a failure: a bare-node query outside a build has
        no part, and a backend whose feasibility needs one correctly declines to answer. What
        F11 got wrong was making unknown UNIVERSAL and silent."""
        return getattr(self, "_device", None) or DeviceFacts.unknown()

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
        device = self._device_facts()
        graph_ctx = Context.from_model(
            model,
            device.fpgapart,
            toolchain_version=device.toolchain_version,
            clk_ns=device.clk_ns,
        )
        shapes: dict[str, tuple[int, ...]] = {}
        datatypes: dict = {}
        initializers: dict[str, np.ndarray] = {}
        sparsity: dict = {}
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
            # SPARSITY. Every given a re-keying drops is a declared rule that silently
            # cannot fire — the weights port declares SparsityFree, and this builder omitted
            # sparsity while the (now deleted) `_trial_context` carried it. So the constraint
            # held at the CLAIM and was inert on the resulting kernel NODE. Two builders, one
            # of them incomplete: exactly the F9 shape, found by collapsing them into one.
            sp = graph_ctx.tensor_sparsity(tname)
            if sp is not None:
                sparsity[iface.name] = sp
        ctx = Context(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            sparsity=sparsity,
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

    # -- Seam A: the frontend claim, asked of a CANDIDATE node ---------------

    @classmethod
    def candidate_op(
        cls, model, inputs, outputs, device: "DeviceFacts | None" = None, **attrs
    ) -> "DataflowOp":
        """A wrapped kernel op over a candidate node that is NOT in the graph.

        The generic replacement for a per-op ``_trial_context``. Asking "could this kernel
        claim these tensors?" used to mean hand-building a Context from a
        ``{iface -> tensor}`` map the op declared separately (``_operand_map``) — a duplicate
        of ``InterfaceSchema.index``, and the docstring claimed a single-source guarantee the
        code did not provide (F9): the claim path read the map while the build path read
        ``node.input[index]``, so the two could disagree silently.

        Here there is nothing to duplicate. Build the node the op would insert, wrap it, and
        let the SAME ``_tensor_name``/``_context`` the build path uses do the re-keying. A
        divergence is now unrepresentable rather than merely tested for.

        The graph is UNMODIFIED — verified: the candidate is never appended, and the model
        serializes byte-identically across the call. ``model.get_customop_wrapper`` needs the
        node only to read its op_type/domain/attributes.
        """
        node = NodeProto()
        node.op_type = cls.op_type_name()
        node.domain = KERNEL_DOMAIN
        node.name = f"{cls.op_type_name()}_candidate"
        node.input.extend(inputs)
        node.output.extend(outputs)
        for name, value in attrs.items():
            node.attribute.append(helper.make_attribute(name, value))
        return model.get_customop_wrapper(node).attach_device(device)

    @classmethod
    def op_type_name(cls) -> str:
        """The ONNX op_type this class backs. Defaults to the kernel's own name, so an op
        whose class, kernel and node all agree declares it once."""
        return cls.kernel().name

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
        needing only backend-INDEPENDENT facts (normal shape/dtype, output-dtype publication)
        use the context-only getters."""
        from .routing import is_specialized

        if not is_specialized(self.onnx_node):
            raise ValueError(
                f"{self.onnx_node.name}: backend-dependent getter needs a committed backend, "
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
        exact) and delegates to :meth:`DataflowKernel.first_feasible_backend`.

        The resolve-time counterpart to infer-time ``can_infer_from`` (``compute/mvau/op.py``),
        which trials a bare-node ``_trial_context`` because the node is still a frontend op;
        here the node IS a kernel node, so the model-attached ``_context()`` is in hand."""
        return self.kernel().first_feasible_backend(self._context())

    # -- Tier-3 getters: shapes / widths / datatypes ------------------------

    def get_normal_input_shape(self, ind=0):
        # Normal (tensor) shape is backend-INDEPENDENT — a pure Context read; no committed
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
