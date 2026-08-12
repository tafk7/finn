############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DataflowOp`` — a hardware kernel AS a real ONNX node.

The op class IS the kernel. Its body declares the design space (interfaces, backend pool,
cell-level axes/derived/predicates); the instance answers FINN's ``HWCustomOp`` contract for
one graph node. There is no separate container object and no ``.kernel()`` hop — that split
was F6, and its visible symptom was ten getters existing twice.

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
  3. **Project** — each FINN getter answers from the node's own point + context. ONE method
     per question: the public ``(ind)`` form is the HWCustomOp contract, and the
     ``(point, context)`` projection math lives in the private helpers it calls.

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
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
from onnx import NodeProto, helper

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.kernels.engine.context import Context
from finn.kernels.engine.design_space import DesignSpace
from finn.kernels.engine.point import AbsentAxisError, Illegal, Point
from finn.kernels.engine.resolve import resolve
from finn.kernels.model._util import prod
from finn.kernels.model.backend import BACKEND_AXIS, Backend, pool_space
from finn.kernels.model.kernel import (
    InterfaceSchema,
    KernelError,
    _resolve_interface_indices,
)
from finn.kernels.model.parameter_source import parameter_source_for
from finn.kernels.model.ports import Direction, Fixed, Protocol, Variadic
from finn.kernels.model.tiling import TileError, generate_tiling
from finn.kernels.model.tiling import stream_width_key as _stream_width_key
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
    """Base FINN adapter over a :class:`~finn.kernels.ir.DataflowOp`.

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

    # -- op-class IDENTITY: the subclass declares these in its body ----------
    #
    # These WERE the fields of a separate frozen `DataflowKernel` dataclass, reached through
    # a `.kernel()` classmethod. They are op-CLASS facts — true of every MVAU node, not of
    # any one — so a class body is their home, and `MvauDataflowOp` simply IS the MVAU
    # kernel. That collapse is F6; the pass-through getters it removes were the symptom.

    name: ClassVar[str] = ""
    interfaces: ClassVar[tuple] = ()
    pool: ClassVar[tuple] = ()
    op_axes: ClassVar[tuple] = ()
    op_derived: ClassVar[tuple] = ()
    op_predicates: ClassVar[tuple] = ()
    kernel_attrs: ClassVar[tuple] = ()  # Attr constants — on the Point, never explored
    constraints: ClassVar[tuple] = ()   # kernel-level relational/value rules
    delivered_parameters: ClassVar[tuple] = ()  # DERIVED from the pool's mem_modes

    def __init_subclass__(cls, **kwargs):
        """Derive and validate the op's identity at CLASS-DEFINITION time.

        This is what the old dataclass `__post_init__` did, relocated. Three things happen
        here and all three are pure functions of ``(interfaces, pool)``:

        1. ``interfaces`` gets its ``index`` sentinels resolved to concrete node slots;
        2. ``_check_port_direction`` validates the backends' per-port facts against the
           op's declared directions, raising :class:`KernelError`;
        3. ``delivered_parameters`` is derived from the pool's ``mem_modes``.

        Firing at class definition rather than at construction makes an authoring mistake an
        IMPORT error instead of a per-node one — strictly better, and the one behavioural
        change in the merge.

        An abstract intermediate (no pool) opts out, so `DataflowOp` itself and any shared
        base can exist without declaring a design space. Each concrete subclass also gets its
        OWN space caches — inheriting them would have every op share one dict."""
        super().__init_subclass__(**kwargs)
        cls._tiling_cache = {}
        cls._space_cache = []
        cls._realized_cache = {}
        if not cls.pool:
            return  # an abstract intermediate; nothing to derive or validate
        cls.interfaces = _resolve_interface_indices(cls.interfaces)
        cls.pool = tuple(cls.pool)
        cls.kernel_attrs = tuple(cls.kernel_attrs)
        cls.constraints = tuple(cls.constraints)
        cls._check_port_direction()
        cls.delivered_parameters = cls._build_delivered()

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
            axis_nodeattr_types_for(self.compile(), self._realized_space())
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
            return self.realized_space(member)
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

    def attach_device(
        self, fpgapart: str = "", clk_ns: float | None = None, toolchain_version=None
    ) -> "DataflowOp":
        """Attach the build's target device — part, clock, toolchain.

        SEPARATE from :meth:`attach_model` because the two arrive in different phases and
        from different owners (F11). Graph givens come with the node; device facts come from
        whichever step holds ``cfg``, and there is no honest way to recover them from the
        graph — which is what ``_fpgapart_from`` tried to do, via a nodeattr no op declares.

        An EMPTY part is a legitimate state, not a failure: a bare-node query outside a build
        genuinely has no device, and a backend whose feasibility needs one correctly declines
        to answer. What F11 got wrong was making that state UNIVERSAL and SILENT.

        Invalidates the Context cache, since these are baked into it."""
        self._device = (fpgapart, clk_ns, toolchain_version)
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
        fpgapart, clk_ns, toolchain_version = getattr(self, "_device", ("", None, None))
        graph_ctx = Context.from_model(
            model, fpgapart, toolchain_version=toolchain_version, clk_ns=clk_ns
        )
        shapes: dict[str, tuple[int, ...]] = {}
        datatypes: dict = {}
        initializers: dict[str, np.ndarray] = {}
        sparsity: dict = {}
        for iface in self.interfaces:
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
        return {dp.iface: True for dp in self.delivered_parameters}

    # -- Seam A: the frontend claim, asked of a CANDIDATE node ---------------

    @classmethod
    def candidate_op(
        cls, model, inputs, outputs, device=None, **attrs
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
        op = model.get_customop_wrapper(node)
        return op.attach_device(*device) if device else op

    @classmethod
    def op_type_name(cls) -> str:
        """The ONNX op_type this class backs. Defaults to the kernel's own name, so an op
        whose class, kernel and node all agree declares it once."""
        return cls.name

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

        schema = self.compile()
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
        result = self.configure(self._context(), self._assignment())
        if isinstance(result, Illegal):
            raise ValueError(
                f"{self.onnx_node.name}: configuration is illegal: "
                f"{'; '.join(result.reasons)}"
            )
        return result

    def selected_backend_for_this_node(self) -> str | None:
        """This node's first feasible backend, sourcing its own Context.

        The instance-side companion to the class-level :meth:`first_feasible_backend`. Kept
        as a distinct NAME rather than an optional-argument overload: with the container
        merged in, a bare `op.first_feasible_backend(ctx)` on the CLASS and on an INSTANCE
        would differ only in whether the first positional argument binds to `self` — which
        is exactly the silent mis-binding that surfaced as `'Context' object has no
        attribute '_context'`."""
        return self._first_feasible_backend(self._context())

    # -- Tier-3 getters: ONE method per question ----------------------------
    #
    # F6. Each of these existed TWICE before the merge: once on the container taking
    # (point, context, ind), once here taking (ind) and sourcing both from the node. That
    # pass-through pair was the container/adapter split made visible. The public signature
    # is the FINN one — `(ind)` — because that is the HWCustomOp contract the host calls;
    # the (point, context) forms survive as the private `_folded_shape`/`_stream_width`
    # helpers below, which is where they were always doing the work.

    def get_normal_input_shape(self, ind=0):
        # Normal (tensor) shape is backend-INDEPENDENT — a pure Context read; no committed
        # backend needed, so an unspecialized node answers it (the Seam A verify gate).
        return tuple(self._context().tensor_shape(self._input(ind).tensor))

    def get_normal_output_shape(self, ind=0):
        return tuple(self._context().tensor_shape(self._output(ind).tensor))

    def get_input_datatype(self, ind=0):
        # Datatype is a pure Context read; no resolved point needed.
        return self._context().tensor_datatype(self._input(ind).tensor)

    def get_output_datatype(self, ind=0):
        return self._context().tensor_datatype(self._output(ind).tensor)

    def get_folded_input_shape(self, ind=0):
        return self._folded_shape(self._stream_input(ind), self._point(), self._context())

    def get_folded_output_shape(self, ind=0):
        return self._folded_shape(self._stream_output(ind), self._point(), self._context())

    def get_instream_width(self, ind=0):
        return self._stream_width(self._stream_input(ind), self._point(), self._context())

    def get_outstream_width(self, ind=0):
        return self._stream_width(self._stream_output(ind), self._point(), self._context())

    def get_exp_cycles(self):
        """PLACEHOLDER cost — NOT accurate. The max over interfaces of each interface's
        stream-cycle count (``prod(tensor) / stream_elems``): monotone in the fold dials (so
        SetFolding still converges), but it does NOT model the nested cross-interface coupling
        real cost needs — e.g. MVU re-traverses the weight block once per input vector, so
        true cost is ``nf·sf·n_vecs`` while this floor gives only ``max(nf·sf, ...)``
        (undercounts whenever n_vecs>1, i.e. conv-as-matmul). A proper cost model (nested
        block traversal, pipeline fill/drain, per-backend overrides) is a FUTURE PASS."""
        return self._exp_cycles(self._point(), self._context())

    @classmethod
    def _exp_cycles(cls, point, context) -> int:
        """The cost floor for an explicit (point, context) — the projection math, without a
        node. The public :meth:`get_exp_cycles` is this sourced from the node's own state;
        this form is what a graph-free caller (and the tiling tests) wants."""
        cycles = 1
        for iface in cls.present_interfaces(context):
            if iface.protocol != Protocol.Stream:
                continue  # only Stream ports contribute a stream-cycle count (T1.4); an
                # MM/Config port has no tensor-axis stream, so it would give a bogus count.
            n = prod(context.tensor_shape(iface.tensor))
            elems = cls._stream_elems(iface, point)
            if elems <= 0 or n % elems != 0:
                # A partial last stream is a real cycle; round up.
                cycles = max(cycles, -(-n // max(elems, 1)))
            else:
                cycles = max(cycles, n // elems)
        return int(cycles)

    @classmethod
    def first_feasible_backend(cls, context: Context) -> str | None:
        """The NAME of the first pool member (declaration order) that yields a legal
        :class:`Point` for this node's Context, or ``None`` if none does — the SELECTION
        query behind ``PerNodePolicy(first_feasible)`` (Seam B).

        Pool order IS selection precedence, and this is the SAME per-backend trial
        :meth:`has_feasible_point` runs, so infer's claim check (the boolean) and resolve's
        selection (the name) converge on ONE query. A node whose datatypes disqualify it from
        EVERY backend (e.g. float32 where only integer is feasible) returns ``None``.

        Each member is trialled against its OWN realized space rather than a merged one.

        ``context`` defaults to this node's own — the normal case now that the op IS the
        kernel. It is passed explicitly only by a caller holding a context for a node that
        does not exist yet (the claim path's candidate)."""
        return cls._first_feasible_backend(context)

    @classmethod
    def _first_feasible_backend(cls, context: Context) -> str | None:
        """The Context-taking form. Separate because the CLAIM path asks this of a candidate
        node's context before any node exists to hang a getter on (``has_feasible_point``)."""
        for backend in cls.pool:
            try:
                result = resolve(
                    cls.realized_space(backend.name), context, {BACKEND_AXIS: backend.name}
                )
            except (ValueError, KeyError, AbsentAxisError):
                # A backend feasibility check that raises on THIS context (e.g. a
                # device-family probe that needs an fpgapart the trial context omits) is not
                # feasible here — treat it as "no point for this backend", not a hard error.
                # These three are the legitimate "can't resolve for this probe" signals
                # (AbsentAxisError is a KeyError subclass); any OTHER exception is a kernel
                # bug that must PROPAGATE (INV5 — the silent-skip class the migration
                # eliminates). A backend that IS feasible resolves cleanly; the pool needs
                # only ONE.
                continue
            if isinstance(result, Point):
                return backend.name
        return None

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
        ctx = self._context()
        specialized = is_specialized(self.onnx_node)
        point = self._point() if specialized else None
        for iface in self.interfaces:
            if iface.direction != Direction.OUT:
                continue
            if specialized:
                odt = self._output_datatype_from_point(ctx, point, iface.index)
            else:
                # Unspecialized: no committed backend ⇒ no realized output dtype yet.
                # Publish the raw graph dtype (pure Context read) so downstream shape/dtype
                # inference has a value; specialization later refines it.
                odt = ctx.tensor_datatype(self._output(iface.index).tensor)
            out_name = self.onnx_node.output[iface.index]
            if out_name:
                model.set_tensor_datatype(out_name, odt)

    def _output_datatype_from_point(self, ctx, point, index):
        """The datatype to annotate on output port ``index`` after resolving against the
        live-graph context. Default: the plain Context read (output forwards the graph
        dtype). An op with a value-derived output type (e.g. MVAU's accumulator under
        ``noActivation``) overrides to read the derived off the ``point``.

        The ``kernel`` parameter this used to take is gone: the op IS the kernel now."""
        return ctx.tensor_datatype(self._output(index).tensor)

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

    # -- the design space (moved from the container) ---------------------
    @classmethod
    def _check_port_direction(cls) -> None:
        """Enforce the datatype/memory direction-exclusivity of each backend's
        :class:`~finn.kernels.model.backend.Interface` against the op schema's declared
        direction — the ONE place both are in hand (an ``Interface`` has no local direction).

        A GATE fact (``accepted_dtypes``) or a memory-realization fact (``mem_modes``) is
        exogenous and INPUT-only; a DERIVATION fact (``derived_dtype``) is endogenous and
        OUTPUT-only. Declaring one on the wrong-direction port is a construction error caught
        here, not a silent no-op at resolve."""
        by_name = {i.name: i for i in cls.interfaces}
        for backend in cls.pool:
            for iface, port in backend.ports.items():
                schema = by_name.get(iface)
                if schema is None:
                    continue  # a port with no op interface — nothing to check direction on
                is_input = schema.direction == Direction.IN
                if is_input and port.derived_dtype is not None:
                    raise KernelError(
                        f"backend {backend.name!r}: derived_dtype set on INPUT port "
                        f"{iface!r} (a derivation is OUTPUT-only)"
                    )
                if not is_input and port.accepted_dtypes is not None:
                    raise KernelError(
                        f"backend {backend.name!r}: accepted_dtypes set on OUTPUT port "
                        f"{iface!r} (a dtype gate is INPUT-only)"
                    )
                if not is_input and port.mem_modes is not None:
                    raise KernelError(
                        f"backend {backend.name!r}: mem_modes set on OUTPUT port "
                        f"{iface!r} (memory realization is INPUT-only)"
                    )

    @classmethod
    def _build_delivered(cls) -> tuple:
        """The delivered-parameter list, DERIVED from the pool: an interface is a delivered
        parameter iff SOME backend declares it in its ``mem_modes`` (the authoritative
        "param port" signal). Each such interface yields one :class:`DeliveredParam` bound to
        ``parameters_pool(name)``. This is the union move — a new core that consumes a new
        param interface gets delivery machinery with zero op edits; delivery-ness is no longer
        re-declared on the op (F1/F2). Iterated over the identity interfaces for deterministic
        order."""
        from finn.kernels.dataflow.parameters import parameters_pool
        from finn.kernels.model.param_contract import DeliveredParam

        declared = {iface for b in cls.pool for iface in b.mem_modes}
        return tuple(
            DeliveredParam(i.name, pool=parameters_pool(i.name))
            for i in cls.interfaces
            if i.name in declared
        )

    @classmethod
    def _constraint_predicates(cls) -> tuple:
        """Every declared constraint compiled to a guard-skipping predicate: the kernel-level
        relational/value rules plus each interface's per-port rules. Each predicate auto-
        noops when its constrained port's tensor is absent (optional-port skip baked in by
        :func:`~finn.kernels.engine.constraints.compile_constraint`)."""
        from finn.kernels.engine.constraints import compile_constraint

        out = [compile_constraint(c) for c in cls.constraints]
        for iface in cls.interfaces:
            out.extend(compile_constraint(c) for c in iface.constraints)
        return tuple(out)

    # -- schema / resolve ---------------------------------------------------

    @classmethod
    def _generated(cls, backend: Backend):
        """The tiling engine's generated fragments + fold map for one Backend,
        derived from its ``stream`` map joined against the op interfaces' ``block``.
        Memoized per op class by backend name."""
        cache = cls._tiling_cache
        got = cache.get(backend.name)
        if got is None:
            dtypes = {n: p.derived_dtype for n, p in backend.ports.items()}
            got = generate_tiling(cls.interfaces, dict(backend.stream), dtypes)
            cache[backend.name] = got
        return got

    @classmethod
    def _augmented_pool(cls) -> tuple[Backend, ...]:
        """Each Backend with the tiling-engine-generated axes/divisibility
        predicates appended to its OWN axes/predicates (so ``pool_space`` dispatches
        them on selection), and the generated stream-width deriveds. The backend's declared
        tiling map is the single source; the fold dials, their ranges, the divisibility,
        and the widths are all derived here — not hand-written on the op."""
        from dataclasses import replace

        out = []
        for backend in cls.pool:
            gen = cls._generated(backend)
            out.append(
                replace(
                    backend,
                    axes=tuple(backend.axes) + gen.axes,
                    derived=tuple(backend.derived) + gen.derived,
                    predicates=tuple(backend.predicates) + gen.predicates,
                )
            )
        return tuple(out)

    @classmethod
    def compile(cls) -> DesignSpace:
        """The UNSPECIALIZED design space — every pool member merged behind a dispatch on
        the ``backend`` root.

        **Scope after T7.** Resolution of a specialized node goes through
        :meth:`realized_space`; this survives for the one thing it alone can express, a node
        with NO committed backend. The ``""`` sentinel is not a pool member, so there is no
        realization to build a space for. Two live callers:

        * ``configure`` with no backend in the assignment — which can only ever return
          ``Illegal("backend = '' not in {...}")``. That verdict is the point: it is what the
          getters turn into a legible "unspecialized" raise.
        * ``get_nodeattr_types`` on an unspecialized node, which needs the NAME SET
          (pitch §5.4 — the node must declare every name it could carry).

        So the merge's *dispatch* is no longer evaluated on any compute path; only its name
        set and its one rejection are. It is not deleted because ``pool_space`` is SHARED
        with the storage pool, which still needs the whole assembler — see the note there.

        MEMOIZED per op class: assembly is pure over the (frozen) kernel.
        Sharing one instance is safe because the result is frozen and its lazily-built
        caches (order, strata) are idempotent — recomputing them yields the same values, so
        a shared space cannot carry state between queries."""
        if cls._space_cache:
            return cls._space_cache[0]
        space = cls._compile()
        cls._space_cache.append(space)
        return space

    @classmethod
    def _compile(cls) -> DesignSpace:
        op = pool_space(
            BACKEND_AXIS,
            tuple(cls.op_axes),
            tuple(cls.op_derived),
            tuple(cls.op_predicates) + cls._constraint_predicates(),
            cls._augmented_pool(),
            unspecialized_sentinel=True,  # compute root: "" = no backend committed (F1)
        )
        # kernel_attrs are CONSTANTS, not choices — they ride `attrs`, outside `axes`, so
        # `stratum_of` does not read them as something that must be pinned. They still reach
        # the Point (resolve writes them first), the backends and the nodeattr bridge; the
        # only thing that changes is that nothing treats them as explorable. Attached AFTER
        # pool_space because they are op-level identity, not a pool contribution: no backend
        # declares one, and none is dispatched on selection.
        op = replace(op, attrs=tuple(cls.kernel_attrs))
        # DELIVERED PARAMETERS: the generic compute→source wiring, OWNED by a
        # ParameterSource per delivered interface — the seam object
        # that holds the DEMAND derived + guarded source sub-schema (design pitch §2). Reads
        # the compute pool's `mem_modes` + each topology's `mem_mode` — no op-specific logic
        # here. Namespaced keys (`parameters.*`) + distinct sources_key mean the union never
        # collides, so resolve walks it unchanged. The fold ORDER below carries no meaning:
        # the supply waterfall COMPUTE→DEMAND→SOURCE is declared on the nodes themselves
        # (deps + optional_deps) and realized by the topo-sort.
        op = DesignSpace.merge(
            op,
            *(
                parameter_source_for(dp, cls.pool).subspace()
                for dp in cls.delivered_parameters
            ),
        )
        # Validate + order the COMPLETE space now (all pools folded in): a cross-pool derived
        # dep (accDataType -> parameters.<iface>.datatype) resolves here, where its
        # target is present, and a genuine typo still fails fast — at compile, before resolve.
        return op.finalize()

    @classmethod
    def compute_cell(cls):
        """This op's compute :class:`~finn.kernels.model.cell.Kernel` — the CELL tier, holding
        the pool plus the cell-level entries currently spelled ``op_axes``/``op_derived``/
        ``op_predicates``.

        Built on demand rather than stored, so this commit stays additive: the container is
        still the authoring surface and its fields are still the source of truth. T7 routes
        resolution through the cell; T9 moves the fields onto it and this becomes the plain
        accessor for a real one."""
        from finn.kernels.model.cell import Kernel as Cell

        return Cell(
            name="compute",
            pool=cls.pool,
            root_axis=BACKEND_AXIS,
            axes=tuple(cls.op_axes),
            derived=tuple(cls.op_derived),
            predicates=tuple(cls.op_predicates) + cls._constraint_predicates(),
        )

    @classmethod
    def realized_space(cls, member_name: str) -> DesignSpace:
        """The op's full design space with ONE compute member realized — the per-realization
        counterpart to :meth:`compile`.

        Same composition as :meth:`_compile`: the compute space, the ``kernel_attrs``, then one
        parameters sub-schema per delivered parameter. ONLY the compute half differs, so any
        divergence between the two can only come from the merge this pass deletes.

        MEMOIZED per member name for the reason ``compile`` is: assembly is pure over the
        frozen kernel, and ``get_nodeattr_types`` sits on a hot protocol path."""
        cached = cls._realized_cache.get(member_name)
        if cached is None:
            space = cls.compute_cell().space_for(member_name, interfaces=cls.interfaces)
            space = replace(space, attrs=tuple(cls.kernel_attrs))
            for dp in cls.delivered_parameters:
                space = space + parameter_source_for(dp, cls.pool).subspace()
            cached = space.finalize()
            cls._realized_cache[member_name] = cached
        return cached

    @classmethod
    def configure(cls, context: Context, assignment: Mapping | None = None):
        """Resolve a design point (or an Illegal).

        Routes through the PER-REALIZATION space when the assignment names a backend, which
        is every backend-dependent query: one member selected means no merge, no dispatch
        closures, no selection guards (F1). An assignment with no backend — or naming one
        outside the pool — still goes through the merged space, which is the only one that
        can represent "unspecialized" (the ``""`` sentinel is not a pool member, so
        ``space_for`` has nothing to build)."""
        assignment = dict(assignment or {})
        member = assignment.get(BACKEND_AXIS)
        if member and any(b.name == member for b in cls.pool):
            return resolve(cls.realized_space(member), context, assignment)
        return resolve(cls.compile(), context, assignment)

    @classmethod
    def has_feasible_point(cls, context: Context) -> bool:
        """Whether ANY pool member yields a legal :class:`Point` for this Context — the
        POOL-FEASIBILITY query (F2). A thin boolean over
        :meth:`first_feasible_backend` (same trial loop; ``can_infer_from`` wants only the
        yes/no, Seam B's selector wants the name). A node whose datatypes disqualify it from
        EVERY backend has no feasible point, so ``can_infer_from`` can delegate to this rather
        than encoding a backend fact in the frontend.

        ``context`` defaults to this node's own — the normal case now that the op IS the
        kernel. The CLAIM path passes one explicitly only when it has already built a
        candidate's context by hand."""
        return cls._first_feasible_backend(context) is not None

    # NOTE — there is exactly ONE schema-assembly path (:meth:`compile`). A second,
    # backend-INDEPENDENT one (``op_space``/``configure_op``) was deleted: it excluded the pool
    # and the whole parameters subspace, so once an op declared a derived with a cross-pool
    # dep (``thresholdDataType`` -> ``parameters.thresholds.datatype``, true for BOTH live
    # ops) it was unresolvable by construction. Its only caller had no callers, so nothing
    # ever noticed. Its purpose — publishing output dtypes on an unspecialized node — is
    # served by the context-only getters today, and by demand-driven ``resolve(want=...)``
    # over the one space thereafter, which cannot diverge from it the way a parallel
    # assembly path silently did.

    # -- interface lookup ---------------------------------------------------

    @classmethod
    def expanded_interfaces(cls, context: Context) -> tuple[InterfaceSchema, ...]:
        """The declared ``interfaces`` with every :class:`~finn.kernels.model.ports.Variadic`
        interface expanded to N concrete peers (N = ``context.arity(count_from)``), each a
        ``Fixed(1)`` at consecutive slot indices ``[base, base+1, …]`` within its direction.
        A ``Fixed`` interface passes through unchanged. This is the context-aware
        generalization of :meth:`present_interfaces`' emergent-presence rule: the concrete
        node-slot interfaces come from Context, not the declared list.

        NO PRODUCTION CALLER — T13 lists it as dead surface and it is KEPT, because deleting
        it would remove a declared capability rather than dead weight. It is the only thing
        that gives :class:`~finn.kernels.model.ports.Variadic` meaning: the field, the
        ``Multiplicity`` union, ``Context.arity`` and a whole test file exist to express
        0-to-N operands, and this is where that expression is cashed. Deleting the method
        alone would leave the declaration inert (worse than either end state); deleting the
        whole cluster is a scope this pass did not take. Both live ops are fixed-arity, so
        the fast path returns ``cls.interfaces`` verbatim and nothing pays for it.

        The all-fixed-arity common case (MVAU/Thresholding) returns the declared list verbatim
        — the field ``interfaces`` is authoritative and no ctx read happens. Only a variadic
        op pays the expansion."""
        if all(isinstance(i.multiplicity, Fixed) and i.multiplicity.n == 1 for i in cls.interfaces):
            return cls.interfaces
        from dataclasses import replace

        out: list[InterfaceSchema] = []
        counters: dict = {}
        for i in cls.interfaces:
            base = counters.get(i.direction, 0)
            if isinstance(i.multiplicity, Variadic):
                count = context.arity(i.multiplicity.count_from)
            else:
                count = i.multiplicity.n
            for k in range(count):
                suffix = f"_{k}" if count > 1 else ""
                out.append(
                    replace(
                        i,
                        name=f"{i.name}{suffix}",
                        index=base + k,
                        multiplicity=Fixed(1),
                    )
                )
            counters[i.direction] = base + count
        return tuple(out)

    @classmethod
    def present_interfaces(cls, context: Context) -> tuple[InterfaceSchema, ...]:
        """The interfaces PRESENT for this node: all required ones, plus each optional one
        whose Context tensor exists. Presence is emergent (phase-3) — an ``optional=True``
        interface with no tensor in ``context`` is absent (the node did not wire that slot),
        so Context-reading loops (cost, any all-interface projection) skip it rather than
        ``KeyError``-ing on its shape. Required interfaces are always present."""
        out = []
        for i in cls.interfaces:
            if i.optional and i.name not in context.shapes:
                continue
            out.append(i)
        return tuple(out)

    @classmethod
    def inputs(cls) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in cls.interfaces if i.direction == Direction.IN)

    @classmethod
    def outputs(cls) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in cls.interfaces if i.direction == Direction.OUT)

    @classmethod
    def _input(cls, ind: int) -> InterfaceSchema:
        ins = cls.inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"input index {ind} out of range (have {len(ins)})")
        return ins[ind]

    @classmethod
    def _output(cls, ind: int) -> InterfaceSchema:
        outs = cls.outputs()
        if ind < 0 or ind >= len(outs):
            raise KernelError(f"output index {ind} out of range (have {len(outs)})")
        return outs[ind]

    @classmethod
    def _stream_inputs(cls) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in cls.inputs() if i.protocol == Protocol.Stream)

    @classmethod
    def _stream_outputs(cls) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in cls.outputs() if i.protocol == Protocol.Stream)

    @classmethod
    def _stream_input(cls, ind: int) -> InterfaceSchema:
        """The ``ind``-th STREAM input — the domain of the folded-shape / stream-width
        projections (T1.4). A ``MemoryMapped``/``Config`` input is outside this projection,
        so it is simply not counted here: a query for its index raises a clean "no stream
        port at index N" rather than a tiling error (IODMA's forcing function)."""
        ins = cls._stream_inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"no stream input port at index {ind} (have {len(ins)})")
        return ins[ind]

    @classmethod
    def _stream_output(cls, ind: int) -> InterfaceSchema:
        outs = cls._stream_outputs()
        if ind < 0 or ind >= len(outs):
            raise KernelError(f"no stream output port at index {ind} (have {len(outs)})")
        return outs[ind]

    # -- Tier-3 getters: normal (TENSOR) shapes — no backend needed ---------

    @classmethod
    def selected_backend(cls, point: Point) -> Backend:
        """The :class:`Backend` pool member the resolved ``point`` selected. Public accessor
        for reading a backend's STATIC identity fields (``language``/``rtl_core_module``) off
        the selection — those fields live on the ``Backend``, not re-projected onto the point
        (F5). Emit reads ``kernel.selected_backend(point).rtl_core_module``."""
        return cls._selected(point)

    @classmethod
    def _selected(cls, point: Point) -> Backend:
        """The pool member named by the resolved ``backend`` axis."""
        backend_name = point[BACKEND_AXIS]
        by_name = {b.name: b for b in cls.pool}
        backend = by_name.get(backend_name)
        if backend is None:
            raise KernelError(
                f"resolved implementation {backend_name!r} is not in the pool "
                f"(have {sorted(by_name)})"
            )
        return backend

    @classmethod
    def _stream_elems(cls, iface: InterfaceSchema, point: Point) -> int:
        """Elements/cycle for this interface = the generated stream-width expression for
        the selected backend (1 if the backend declares no tiling for it)."""
        gen = cls._generated(cls._selected(point))
        expr = gen.width_exprs.get(iface.name)
        if expr is None:
            return 1
        try:
            return int(expr.eval(point))
        except TileError as exc:
            raise KernelError(f"interface {iface.name!r} tiling: {exc}") from exc

    @classmethod
    def _stream_width(cls, iface: InterfaceSchema, point: Point, context: Context) -> int:
        """Stream width in bits for this interface. Reads the per-interface
        ``stream_width.<iface>`` derived the tiling engine produced on the point — the
        SAME value emit reads, so the getter (FINN's contract) and emit share one produced
        quantity instead of a recompute-vs-precompute pair. Falls back to recomputing from
        ``width_exprs`` when the interface has no generated width key (a backend that declares
        no stream for it — the key is absent from the point)."""
        key = _stream_width_key(iface.name)
        if key in point:
            return int(point[key])
        # No generated width derived (backend declares no stream for this interface): the
        # stream is one element/cycle at the interface's realized dtype — the selected
        # backend's declared derived_dtype spec (absent ⇒ the raw graph tensor dtype).
        from finn.kernels.engine.datatype_spec import resolve_datatype_spec

        elems = cls._stream_elems(iface, point)
        spec = cls._selected(point).ports.get(iface.name)
        dt = resolve_datatype_spec(
            spec.derived_dtype if spec is not None else None,
            iface=iface.tensor,
            point=point,
            context=context,
        )
        return elems * dt.bitwidth()

    @classmethod
    def _folds_reshape(cls, iface: InterfaceSchema, point: Point) -> bool:
        """Whether a folded SHAPE is a plain reshape for this interface under the selected
        backend. Derived from the stream folds (a fold whose width is a cross-interface expr
        ⇒ not a plain reshape). True when the backend declares no stream for the interface."""
        gen = cls._generated(cls._selected(point))
        return gen.reshapes.get(iface.name, True)

    @classmethod
    def _folded_shape(cls, iface: InterfaceSchema, point: Point, context: Context):
        """Fold each dim the backend's stream map folds: for each position ``(dim_index,
        elems_expr)``, split that tensor dim into ``(extent // elems, elems)``. The engine's
        ``fold_map`` names WHICH dims fold (any dim, not just the last), so a 2-D weight
        block ``(MW, MH)`` streamed ``[SIMD, PE]`` folds to ``(MW/SIMD, MH/PE, SIMD, PE)``.
        Raises when a fold width is a cross-interface expr (not a plain tensor-axis
        reshape)."""
        if not cls._folds_reshape(iface, point):
            raise KernelError(
                f"interface {iface.name!r} does not fold a tensor axis (its stream WIDTH "
                f"resolves via get_*stream_width, but a folded SHAPE is not a plain "
                f"reshape for this port, e.g. a cross-interface weight width)"
            )
        normal = tuple(context.tensor_shape(iface.tensor))
        if not normal:
            raise KernelError(f"interface {iface.name!r} has no shape to fold")
        gen = cls._generated(cls._selected(point))
        fmap = gen.fold_map.get(iface.name)
        if fmap is None:
            return normal  # no stream for this interface ⇒ unfolded
        # Build the folded shape by expanding each folded position into (fold, elems).
        out: list[int] = []
        for dim_idx, elems_expr in fmap:
            extent = int(normal[dim_idx])
            if elems_expr is None:
                out.append(extent)
                continue
            elems = int(elems_expr.eval(point))
            if elems <= 0 or extent % elems != 0:
                raise KernelError(
                    f"interface {iface.name!r}: stream {elems} does not divide dim "
                    f"{dim_idx} = {extent} (illegal fold)"
                )
            out.extend((extent // elems, elems))
        return tuple(out)

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
