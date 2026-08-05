############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Kernel`` — the WHAT-owning op node, and the Tier-3 (estimate-only) surface
it projects from a resolved :class:`~finn.kernels.engine.point.Point`
(kernelop-tensor-block-stream.md §3, §7; consumer-surface-model.md Tier 0-3).

The op declares its **interfaces** (identity + role — the ONNX-facing arity, the
BLOCK/reduction structure) and a **pool** of Backends. Each Backend owns
its **stream tiling** (the BLOCK->STREAM lowering) — so a *normal* shape resolves with
no backend, while a *folded* shape needs a resolved point that names the selected impl
and its fold dials.

This module builds ONLY the estimate-only surface: the port-indexed normal/folded
shapes + stream widths, and a rough ``get_exp_cycles`` (``prod(stream_cycles)`` — the
monotone throughput floor that drives folding search). No emit, no codegen, no Vivado
(those are Tier-4). Cost is a defaultable op-level derived; a Backend may
override it (not exercised here).

SCOPE (increment 1): folding is a LAST-AXIS reshape of a DATA interface — the common
case shared by LayerNorm/elementwise/MVU activation+output. A PARAM interface whose
stream width is not a tensor-axis fold (MVU's weight port ``WSIMD=PE*SIMD/TH``) resolves
its *width* fine via the tiling evaluator, but its folded *shape* is not a plain
last-axis reshape; asking for that shape raises loudly rather than faking a reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..engine.context import Context
from ._util import prod
from .backend import BACKEND_AXIS, Backend, pool_space
from .parameter_source import parameter_source_for
from ..engine.point import AbsentAxisError, Illegal, Point
from .ports import Direction, Fixed, Multiplicity, Protocol, Variadic
from ..engine.resolve import resolve
from ..engine.design_space import DesignSpace
from .tiling import TileError, generate_tiling, stream_width_key as _stream_width_key


class KernelError(ValueError):
    """Raised for an ill-formed Kernel query (unknown interface index, a folded-shape
    request on a non-last-axis PARAM port, or a stream dial that does not divide)."""


# The three DATAFLOW protocols an InterfaceSchema may declare (pitch §7.0); Sideband/
# Clock/Reset are emit-only pins, not phase-1 interfaces.
_DATAFLOW_PROTOCOLS = frozenset({Protocol.Stream, Protocol.MemoryMapped, Protocol.Config})


def _resolve_interface_indices(interfaces):
    """Resolve each interface's ``index`` sentinel (``-1``) to its declaration-order position
    among same-direction peers, leaving any explicitly-set index untouched. Called once at
    ``Kernel`` construction so ``iface.index`` is always a concrete node-slot index the
    adapter can read (the index-authoritative-at-the-ONNX-boundary rule, F9)."""
    from dataclasses import replace

    counters: dict = {}
    out = []
    for iface in interfaces:
        pos = counters.get(iface.direction, 0)
        counters[iface.direction] = pos + 1
        out.append(iface if iface.index >= 0 else replace(iface, index=pos))
    return tuple(out)


@dataclass(frozen=True)
class InterfaceSchema:
    """One op-side interface — identity + DIRECTION + BLOCK structure (the math), NOT stream.

    The op declares the arity, the direction (which node slot), and how the block segments
    this tensor (``block`` — which dims a calc-state quantum spans). The selected Backend
    owns how the block is folded into a stream (its ``stream`` map). Block folding is the
    math → op-owned; stream folding is the realization → impl-owned. An impl cannot change
    the block: it has no field to express one (the ownership split, made structural).

    NO SEMANTIC ROLE. An interface is neutral: whether it is a weight/parameter feed or a
    dataflow activation is NOT declared — it EMERGES from graph context at resolve time
    (initializer attached → parameter; a producer → dataflow; else → boundary), read by the
    derived/predicate closures via ``ctx.initializer`` (resolution-phases.md: a
    Context-reading fact is phase-3, never a phase-1 declared field). Only DIRECTION (IN/OUT)
    is declared — a node-slot fact known from the op's math alone.

    Attributes:
        name: the interface key — ALSO the Context tensor name (shape + datatype). Matched
            against a Backend's ``stream`` map.
        direction: IN or OUT — which side of the node this interface sits on. A declare-time
            structural fact (the node slot), stored explicitly (not derived from list
            position — no positional rule survives inp/weights/out vs inp/out/indices).
        block: per-tensor-dim BLOCK extents, positional over the tensor's dims. Each is
            ``FULL`` (the whole dim sits in one block), ``1`` (iterate one at a time), an
            int size, or a ``derive(...)`` expr (a bounded / cross-interface block, e.g. a
            conv window). NO reduce/free tag — reduction is emergent from the math, not
            declared (matches brainsmith's block_tiling). The impl's ``stream[name][i]``
            folds ``block[i]``, positionally.
        optional: whether this is an OPTIONAL node input (0-or-1). An ONNX-invariant
            identity fact (like ``direction``) — the opset says the slot may be absent (MVU
            thresholds, a bias, MaxPool indices). Its PRESENCE for a given node is EMERGENT
            from Context (its tensor exists), read at projection time by
            :meth:`Kernel.present_interfaces` — the same declared-slot / emergent-existence
            split as role emergence. A required interface (default ``False``) is always
            present. (Variadic 0-to-N is a later additive generalization of the same rule:
            present interfaces come from Context, not the declared list.)

        constraints: per-port structural-legality
            :mod:`~finn.kernels.engine.constraints` for THIS port — identity rules a
            backend cannot override (a rank check, a static-initializer requirement). Each
            compiles to a predicate that auto-skips when the port's tensor is absent
            (so an optional port needs no ``has_tensor`` guard). Relational rules that read
            ANOTHER tensor live on :attr:`Kernel.constraints` instead.

        index: the node-slot index WITHIN this interface's direction (0=first input/output).
            The adapter reads ``node.input[index]``/``node.output[index]`` to resolve this
            interface's Context tensor — the index-authoritative-at-the-ONNX-boundary fact
            (F9). Defaults to the sentinel ``-1`` = "declaration order among same-direction
            peers", resolved to a concrete positional index at ``Kernel`` construction
            (:func:`_resolve_interface_indices`). An op
            with a non-declaration-order wiring (an operand at a shifted slot) sets it
            explicitly. Replaces the old ``PortSpec.index`` — one interface object now carries
            name/direction/index/optional, so the op no longer restates the binding.

        protocol: the PHYSICAL dataflow protocol this interface speaks — one of the three
            DATAFLOW :class:`~finn.kernels.model.ports.Protocol` members ``Stream`` (default),
            ``MemoryMapped`` (an IODMA off-chip master), or ``Config`` (a runtime-writable
            register surface). ``Sideband``/``Clock``/``Reset`` are emit-only PINS, not
            phase-1 interfaces, so they are rejected here (pitch §7.0). Default ``Stream`` —
            an author sets it only when the op's PURPOSE is that protocol (the fact-placement
            rule). The folded-shape/stream-width projections apply ONLY to ``Stream``
            interfaces (T1.4).

        multiplicity: how many concrete node slots this interface expands to —
            :class:`~finn.kernels.model.ports.Fixed` ``(n)`` (default ``Fixed(1)``, the exact
            common case) or :class:`~finn.kernels.model.ports.Variadic` ``(count_from)`` (N
            homogeneous repeats, N read from Context via ``ctx.arity(count_from)`` — Concat).
            Distinct from ``optional`` (ONNX ``Optional``, a heterogeneous 0-or-1 operand):
            ``Variadic`` is ONNX ``Variadic``, a homogeneous repeat. :meth:`Kernel.interfaces`
            expands ``Variadic`` to N concrete peers.
    """

    name: str
    direction: "Direction"
    block: tuple = ()
    optional: bool = False
    constraints: tuple = ()
    index: int = -1
    protocol: "Protocol" = Protocol.Stream
    multiplicity: "Multiplicity" = field(default_factory=Fixed)

    def __post_init__(self):
        object.__setattr__(self, "block", tuple(self.block))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if self.protocol not in _DATAFLOW_PROTOCOLS:
            raise ValueError(
                f"interface {self.name!r}: protocol {self.protocol} is not a dataflow "
                f"protocol; an InterfaceSchema may only declare "
                f"{sorted(p.name for p in _DATAFLOW_PROTOCOLS)} "
                f"(Sideband/Clock/Reset are emit-only pins, not phase-1 interfaces)"
            )

    @property
    def tensor(self) -> str:
        """The Context tensor name — equals ``name`` (the two were always the same key)."""
        return self.name


@dataclass(frozen=True)
class Kernel:
    """A hardware kernel op: its immutable, ONNX-invariant IDENTITY (name, interfaces,
    op-level design space, frontend-fixed attrs, rough cost) + a ``pool`` of Backends
    (realizations). Delivered parameters (weight/threshold delivery, memory) are DERIVED
    from the pool's ``mem_modes`` declarations, not passed in.

    The identity fields are held directly (the identity ⊥ realization split is the
    field grouping, not a nested wrapper — F10.1 collapsed the former ``KernelSchema``
    onto ``Kernel``). Every identity field is independently optional: a minimal op declares
    only ``name`` + ``interfaces`` (the whole space then comes from the pool's tiling).

    ``kernel_attrs`` is a THIRD design-space category, distinct from ``op_axes`` (the DSE
    dials — SIMD/PE, tiling-engine-generated) and from delivered parameters: a
    ``kernel_attr`` is a nodeattr-backed scalar that reaches the Point (backends read it)
    but is FRONTEND-FIXED — set once at conversion, never a search dial (MVU's ActVal
    activation bias, mlo_max_iter iteration count). Structurally each entry is an ``Axis``
    (built with ``predicate_axis``/``discrete_axis``); resolve carries it onto the Point at
    its assignment/default with nothing exploring it. We call it ``kernel_attrs`` (not
    brainsmith's ``kernel_params``) deliberately: the ``parameters`` namespace here is
    claimed by the weight/threshold DELIVERY subsystem, so "attribute" names what these are
    (node-owned scalars) rather than what they aren't (delivered tensors).

    ``constraints`` is the KERNEL-LEVEL constraint list: structural-legality
    :mod:`~finn.kernels.engine.constraints` that cannot live on a single port — RELATIONAL
    or value rules that read more than one tensor (unsigned INPUT ⇒ THRESHOLDS >= 0). Per-
    port rules (rank, static-initializer) live on the owning ``InterfaceSchema.constraints``
    instead; the two homes mirror the identity/backend split (design pitch, Finding 3c).
    Both compile to predicates via :func:`~finn.kernels.engine.constraints.compile_constraint`.

    ``pool`` is the flat list of Backends; each owns its tiling, feasibility, sources, emit.
    ``delivered_parameters`` is BUILT in ``__post_init__`` by DERIVING from the pool — one
    :class:`~finn.kernels.model.param_contract.DeliveredParam` (interface +
    ``parameters_pool(name)``) per interface some backend declares in its ``mem_modes``,
    each lowered by a
    :class:`~finn.kernels.model.parameter_source.ParameterSource` into the demand stage +
    guarded source pool. :meth:`compile` assembles all into the flat resolve
    :class:`~finn.kernels.engine.design_space.DesignSpace`; :meth:`configure` resolves a
    point; the getters project from it.
    """

    name: str
    interfaces: tuple[InterfaceSchema, ...]
    pool: tuple[Backend, ...]
    op_axes: tuple = ()
    op_derived: tuple = ()
    op_predicates: tuple = ()
    kernel_attrs: tuple = ()  # frontend-fixed nodeattr scalars — in the Point, never explored
    constraints: tuple = ()  # kernel-level structural constraints (relational/value rules)
    cost_model: Any = None  # (point, context) -> int; None => the rough op-level default
    delivered_parameters: tuple = field(default=(), init=False)  # derived from pool mem_modes
    _tiling_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "pool", tuple(self.pool))
        object.__setattr__(
            self, "interfaces", _resolve_interface_indices(self.interfaces)
        )
        object.__setattr__(self, "kernel_attrs", tuple(self.kernel_attrs))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        self._check_port_direction()
        object.__setattr__(self, "delivered_parameters", self._build_delivered())
        object.__setattr__(self, "_tiling_cache", {})

    def _check_port_direction(self) -> None:
        """Enforce the datatype/memory direction-exclusivity of each backend's
        :class:`~finn.kernels.model.backend.Interface` against the op schema's declared
        direction — the ONE place both are in hand (an ``Interface`` has no local direction).

        A GATE fact (``accepted_dtypes``) or a memory-realization fact (``mem_modes``) is
        exogenous and INPUT-only; a DERIVATION fact (``derived_dtype``) is endogenous and
        OUTPUT-only. Declaring one on the wrong-direction port is a construction error caught
        here, not a silent no-op at resolve."""
        by_name = {i.name: i for i in self.interfaces}
        for backend in self.pool:
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

    def _build_delivered(self) -> tuple:
        """The delivered-parameter list, DERIVED from the pool: an interface is a delivered
        parameter iff SOME backend declares it in its ``mem_modes`` (the authoritative
        "param port" signal). Each such interface yields one :class:`DeliveredParam` bound to
        ``parameters_pool(name)``. This is the union move — a new core that consumes a new
        param interface gets delivery machinery with zero op edits; delivery-ness is no longer
        re-declared on the op (F1/F2). Iterated over the identity interfaces for deterministic
        order."""
        from finn.kernels.dataflow.parameters import parameters_pool
        from .param_contract import DeliveredParam

        declared = {iface for b in self.pool for iface in b.mem_modes}
        return tuple(
            DeliveredParam(i.name, pool=parameters_pool(i.name))
            for i in self.interfaces
            if i.name in declared
        )

    def _constraint_predicates(self) -> tuple:
        """Every declared constraint compiled to a guard-skipping predicate: the kernel-level
        relational/value rules plus each interface's per-port rules. Each predicate auto-
        noops when its constrained port's tensor is absent (optional-port skip baked in by
        :func:`~finn.kernels.engine.constraints.compile_constraint`)."""
        from ..engine.constraints import compile_constraint

        out = [compile_constraint(c) for c in self.constraints]
        for iface in self.interfaces:
            out.extend(compile_constraint(c) for c in iface.constraints)
        return tuple(out)

    # -- schema / resolve ---------------------------------------------------

    def _generated(self, impl: Backend):
        """The tiling engine's generated fragments + fold map for one Backend,
        derived from its ``stream`` map joined against the op interfaces' ``block``.
        Memoized per Kernel by impl name."""
        cache = self._tiling_cache
        got = cache.get(impl.name)
        if got is None:
            dtypes = {n: p.derived_dtype for n, p in impl.ports.items()}
            got = generate_tiling(self.interfaces, dict(impl.stream), dtypes)
            cache[impl.name] = got
        return got

    def _augmented_pool(self) -> tuple[Backend, ...]:
        """Each Backend with the tiling-engine-generated axes/divisibility
        predicates appended to its OWN axes/predicates (so ``pool_space`` dispatches
        them on selection), and the generated stream-width deriveds. The impl's declared
        tiling map is the single source; the fold dials, their ranges, the divisibility,
        and the widths are all derived here — not hand-written on the op."""
        from dataclasses import replace

        out = []
        for impl in self.pool:
            gen = self._generated(impl)
            out.append(
                replace(
                    impl,
                    axes=tuple(impl.axes) + gen.axes,
                    derived=tuple(impl.derived) + gen.derived,
                    predicates=tuple(impl.predicates) + gen.predicates,
                )
            )
        return tuple(out)

    def compile(self) -> DesignSpace:
        """The full design space: the identity's op-level shared elements + the
        backend pool (each backend augmented with its tiling-engine-derived fold dials
        / divisibility / widths), plus the delivered parameters' realization sub-schemas.
        ``op_derived``/``op_predicates`` are PURE identity — the cross-coordinate source
        couplings that once lived here relocated into the parameters pool, and the
        compute→source demand crosses the seam owned by a ``ParameterSource``."""
        op = pool_space(
            BACKEND_AXIS,
            # kernel_attrs join the op-level shared axes: each is an Axis carried onto the
            # Point at its assignment/default, reaching backends + the nodeattr bridge, with
            # nothing exploring it (resolve is pure assignment-or-default — no DSE engine).
            tuple(self.op_axes) + tuple(self.kernel_attrs),
            tuple(self.op_derived),
            tuple(self.op_predicates) + self._constraint_predicates(),
            self._augmented_pool(),
            unspecialized_sentinel=True,  # compute root: "" = no backend committed (F1)
        )
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
                parameter_source_for(dp, self.pool).subspace()
                for dp in self.delivered_parameters
            ),
        )
        # Validate + order the COMPLETE space now (all pools folded in): a cross-pool derived
        # dep (accDataType -> parameters.<iface>.datatype) resolves here, where its
        # target is present, and a genuine typo still fails fast — at compile, before resolve.
        return op.finalize()

    def configure(self, context: Context, assignment: Mapping | None = None):
        """Resolve a design point (or an Illegal). Thin wrapper over ``resolve``."""
        return resolve(self.compile(), context, assignment)

    def _early_verdict(self, schema, context: Context, impl_name: str):
        """Decide ``impl_name``'s feasibility WITHOUT a full resolve, where that is sound.

        Returns ``False`` when some fold-independent rule rejects (definitively infeasible),
        ``True`` when every rule for this member is fold-independent and all pass
        (definitively feasible), and ``None`` when the answer genuinely needs a
        configuration — the caller then runs the full-resolve trial exactly as before.

        The soundness argument, in both directions:

        * A **stratum ≤ 1** predicate's read-closure touches nothing but the selection root.
          Its verdict therefore cannot change with any folding choice, so a rejection here is
          a rejection the full resolve would also produce — no legal point exists for this
          member at ANY fold. This is the case that turns a float MatMul from three full
          resolves into a handful of dtype comparisons.
        * ``True`` is only returned when the member has NO stratum-2 rule left to run. If
          one exists it might reject at default fold, so we must not pre-empt the resolve.

        Anything unexpected — a rule that raises, a closure naming a derived we have not
        computed — yields ``None``. Falling back to the existing path is always safe; being
        clever here is not."""
        trial = {BACKEND_AXIS: impl_name}
        view = Point(trial)
        decided = 0
        for pred in schema.predicates_upto(1):
            # Every name the rule declares must be pinned here, else it would read an absent
            # key. Its stratum says it reads only roots; a DIFFERENT root may be unpinned.
            if not all(dep in trial for dep in pred.deps | pred.optional_deps):
                continue
            try:
                reason = pred.check(view, context)
            except Exception:
                # A rule that cannot run against a root-only point tells us nothing. Note
                # this does NOT swallow a typo-class kernel bug: returning None falls
                # through to the full-resolve trial, which runs the same rule against a
                # complete point and lets the exception propagate (INV5). The early path
                # may only ever make a query CHEAPER, never quieter.
                return None
            if reason is not None:
                return False
            decided += 1

        # Only claim feasibility when NOTHING is left to check; otherwise a stratum-2 rule
        # could still reject at default fold and we must not pre-empt the resolve.
        return True if decided == len(schema.predicates) else None

    def first_feasible_backend(self, context: Context) -> str | None:
        """The NAME of the first pool member (declaration order) that yields a legal
        :class:`Point` for this Context, or ``None`` if none does — the SELECTION query behind
        ``PerNodePolicy(first_feasible)`` (Seam B). Trials each backend with only its
        ``backend`` axis pinned (unpinned folding axes take defaults — the
        specialized-at-default-fold semantics), returning the first that resolves.

        Pool order IS selection precedence, and this is the SAME per-backend trial
        :meth:`has_feasible_point` runs, so infer's claim check (the boolean) and resolve's
        selection (the name) converge on ONE query. A node whose datatypes disqualify it from
        EVERY backend (e.g. float32 where only integer is feasible) returns ``None``.

        Each member is first offered to :meth:`_early_verdict`, which answers from the
        fold-independent rules alone where it soundly can. The precedence order and the
        answer are unchanged by that — only the cost is."""
        schema = self.compile()
        for impl in self.pool:
            early = self._early_verdict(schema, context, impl.name)
            if early is False:
                continue  # fold-independent rejection: no fold could rescue this member
            if early is True:
                return impl.name
            try:
                result = resolve(schema, context, {BACKEND_AXIS: impl.name})
            except (ValueError, KeyError, AbsentAxisError):
                # A backend feasibility check that raises on THIS context (e.g. a
                # device-family probe that needs an fpgapart the trial context omits) is not
                # feasible here — treat it as "no point for this backend", not a hard error.
                # These three are the legitimate "can't resolve for this probe" signals
                # (AbsentAxisError is a KeyError subclass); any OTHER exception is a kernel bug
                # that must PROPAGATE (INV5 — the silent-skip class the migration eliminates).
                # A backend that IS feasible resolves cleanly; the pool needs only ONE.
                continue
            if isinstance(result, Point):
                return impl.name
        return None

    def has_feasible_point(self, context: Context) -> bool:
        """Whether ANY pool member yields a legal :class:`Point` for this Context — the
        POOL-FEASIBILITY query (F2). A thin boolean over
        :meth:`first_feasible_backend` (same trial loop; ``can_infer_from`` wants only the
        yes/no, Seam B's selector wants the name). A node whose datatypes disqualify it from
        EVERY backend has no feasible point, so ``can_infer_from`` can delegate to this rather
        than encoding a backend fact in the frontend."""
        return self.first_feasible_backend(context) is not None

    # NOTE — there is exactly ONE schema-assembly path (:meth:`compile`). A second,
    # impl-INDEPENDENT one (``op_space``/``configure_op``) was deleted: it excluded the pool
    # and the whole parameters subspace, so once an op declared a derived with a cross-pool
    # dep (``thresholdDataType`` -> ``parameters.thresholds.datatype``, true for BOTH live
    # ops) it was unresolvable by construction. Its only caller had no callers, so nothing
    # ever noticed. Its purpose — publishing output dtypes on an unspecialized node — is
    # served by the context-only getters today, and by demand-driven ``resolve(want=...)``
    # over the one space thereafter, which cannot diverge from it the way a parallel
    # assembly path silently did.

    # -- interface lookup ---------------------------------------------------

    def expanded_interfaces(self, context: Context) -> tuple[InterfaceSchema, ...]:
        """The declared ``interfaces`` with every :class:`~finn.kernels.model.ports.Variadic`
        interface expanded to N concrete peers (N = ``context.arity(count_from)``), each a
        ``Fixed(1)`` at consecutive slot indices ``[base, base+1, …]`` within its direction.
        A ``Fixed`` interface passes through unchanged. This is the context-aware
        generalization of :meth:`present_interfaces`' emergent-presence rule: the concrete
        node-slot interfaces come from Context, not the declared list.

        The all-fixed-arity common case (MVAU/Thresholding) returns the declared list verbatim
        — the field ``interfaces`` is authoritative and no ctx read happens. Only a variadic
        op pays the expansion."""
        if all(isinstance(i.multiplicity, Fixed) and i.multiplicity.n == 1 for i in self.interfaces):
            return self.interfaces
        from dataclasses import replace

        out: list[InterfaceSchema] = []
        counters: dict = {}
        for i in self.interfaces:
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

    def present_interfaces(self, context: Context) -> tuple[InterfaceSchema, ...]:
        """The interfaces PRESENT for this node: all required ones, plus each optional one
        whose Context tensor exists. Presence is emergent (phase-3) — an ``optional=True``
        interface with no tensor in ``context`` is absent (the node did not wire that slot),
        so Context-reading loops (cost, any all-interface projection) skip it rather than
        ``KeyError``-ing on its shape. Required interfaces are always present."""
        out = []
        for i in self.interfaces:
            if i.optional and i.name not in context.shapes:
                continue
            out.append(i)
        return tuple(out)

    def inputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.interfaces if i.direction == Direction.IN)

    def outputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.interfaces if i.direction == Direction.OUT)

    def _input(self, ind: int) -> InterfaceSchema:
        ins = self.inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"input index {ind} out of range (have {len(ins)})")
        return ins[ind]

    def _output(self, ind: int) -> InterfaceSchema:
        outs = self.outputs()
        if ind < 0 or ind >= len(outs):
            raise KernelError(f"output index {ind} out of range (have {len(outs)})")
        return outs[ind]

    def _stream_inputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.inputs() if i.protocol == Protocol.Stream)

    def _stream_outputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.outputs() if i.protocol == Protocol.Stream)

    def _stream_input(self, ind: int) -> InterfaceSchema:
        """The ``ind``-th STREAM input — the domain of the folded-shape / stream-width
        projections (T1.4). A ``MemoryMapped``/``Config`` input is outside this projection,
        so it is simply not counted here: a query for its index raises a clean "no stream
        port at index N" rather than a tiling error (IODMA's forcing function)."""
        ins = self._stream_inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"no stream input port at index {ind} (have {len(ins)})")
        return ins[ind]

    def _stream_output(self, ind: int) -> InterfaceSchema:
        outs = self._stream_outputs()
        if ind < 0 or ind >= len(outs):
            raise KernelError(f"no stream output port at index {ind} (have {len(outs)})")
        return outs[ind]

    # -- Tier-3 getters: normal (TENSOR) shapes — no backend needed ---------

    def get_normal_input_shape(self, context: Context, ind: int = 0) -> tuple[int, ...]:
        return tuple(context.tensor_shape(self._input(ind).tensor))

    def get_normal_output_shape(self, context: Context, ind: int = 0) -> tuple[int, ...]:
        return tuple(context.tensor_shape(self._output(ind).tensor))

    def get_input_datatype(self, context: Context, ind: int = 0):
        return context.tensor_datatype(self._input(ind).tensor)

    def get_output_datatype(self, context: Context, ind: int = 0):
        return context.tensor_datatype(self._output(ind).tensor)

    # -- Tier-3 getters: folded (STREAM) shapes — need a resolved point -----

    def get_folded_input_shape(self, point: Point, context: Context, ind: int = 0):
        return self._folded_shape(self._stream_input(ind), point, context)

    def get_folded_output_shape(self, point: Point, context: Context, ind: int = 0):
        return self._folded_shape(self._stream_output(ind), point, context)

    def get_instream_width(self, point: Point, context: Context, ind: int = 0) -> int:
        return self._stream_width(self._stream_input(ind), point, context)

    def get_outstream_width(self, point: Point, context: Context, ind: int = 0) -> int:
        return self._stream_width(self._stream_output(ind), point, context)

    # -- rough cost: prod(stream_cycles) over the interfaces ----------------

    def get_exp_cycles(self, point: Point, context: Context) -> int:
        """PLACEHOLDER cost — NOT accurate. Returns the max over interfaces of each
        interface's stream-cycle count (``prod(tensor) / stream_elems``): monotone in the
        fold dials (so SetFolding still converges), but it does NOT model the nested
        cross-interface coupling real cost needs — e.g. MVU re-traverses the weight block
        once per input vector, so true cost is ``nf·sf·n_vecs`` while this floor gives only
        ``max(nf·sf, ...)`` (undercounts whenever n_vecs>1, i.e. conv-as-matmul). A proper
        cost model (nested block traversal, pipeline fill/drain, per-impl overrides) is a
        FUTURE PASS; cost modelling is deliberately ignored for now. ``cost_model`` remains
        an op-level escape hatch if an op needs a real number before then."""
        if self.cost_model is not None:
            return int(self.cost_model(point, context))
        cycles = 1
        for iface in self.present_interfaces(context):
            if iface.protocol != Protocol.Stream:
                continue  # only Stream ports contribute a stream-cycle count (T1.4); an
                # MM/Config port has no tensor-axis stream, so it would give a bogus count.
            n = prod(context.tensor_shape(iface.tensor))
            elems = self._stream_elems(iface, point)
            if elems <= 0 or n % elems != 0:
                # A partial last stream is a real cycle; round up.
                cycles = max(cycles, -(-n // max(elems, 1)))
            else:
                cycles = max(cycles, n // elems)
        return int(cycles)

    # -- internals ----------------------------------------------------------

    def selected_backend(self, point: Point) -> Backend:
        """The :class:`Backend` pool member the resolved ``point`` selected. Public accessor
        for reading a backend's STATIC identity fields (``language``/``rtl_core_module``) off
        the selection — those fields live on the ``Backend``, not re-projected onto the point
        (F5). Emit reads ``kernel.selected_backend(point).rtl_core_module``."""
        return self._selected(point)

    def _selected(self, point: Point) -> Backend:
        """The pool member named by the resolved ``backend`` axis."""
        impl_name = point[BACKEND_AXIS]
        by_name = {b.name: b for b in self.pool}
        bundle = by_name.get(impl_name)
        if bundle is None:
            raise KernelError(
                f"resolved implementation {impl_name!r} is not in the pool "
                f"(have {sorted(by_name)})"
            )
        return bundle

    def _stream_elems(self, iface: InterfaceSchema, point: Point) -> int:
        """Elements/cycle for this interface = the generated stream-width expression for
        the selected impl (1 if the impl declares no tiling for it)."""
        gen = self._generated(self._selected(point))
        expr = gen.width_exprs.get(iface.name)
        if expr is None:
            return 1
        try:
            return int(expr.eval(point))
        except TileError as exc:
            raise KernelError(f"interface {iface.name!r} tiling: {exc}") from exc

    def _stream_width(self, iface: InterfaceSchema, point: Point, context: Context) -> int:
        """Stream width in bits for this interface. Reads the per-interface
        ``stream_width.<iface>`` derived the tiling engine produced on the point — the
        SAME value emit reads, so the getter (FINN's contract) and emit share one produced
        quantity instead of a recompute-vs-precompute pair. Falls back to recomputing from
        ``width_exprs`` when the interface has no generated width key (an impl that declares
        no stream for it — the key is absent from the point)."""
        key = _stream_width_key(iface.name)
        if key in point:
            return int(point[key])
        # No generated width derived (impl declares no stream for this interface): the
        # stream is one element/cycle at the interface's realized dtype — the selected
        # backend's declared derived_dtype spec (absent ⇒ the raw graph tensor dtype).
        from ..engine.datatype_spec import resolve_datatype_spec

        elems = self._stream_elems(iface, point)
        spec = self._selected(point).ports.get(iface.name)
        dt = resolve_datatype_spec(
            spec.derived_dtype if spec is not None else None,
            iface=iface.tensor,
            point=point,
            context=context,
        )
        return elems * dt.bitwidth()

    def _folds_reshape(self, iface: InterfaceSchema, point: Point) -> bool:
        """Whether a folded SHAPE is a plain reshape for this interface under the selected
        impl. Derived from the stream folds (a fold whose width is a cross-interface expr
        ⇒ not a plain reshape). True when the impl declares no stream for the interface."""
        gen = self._generated(self._selected(point))
        return gen.reshapes.get(iface.name, True)

    def _folded_shape(self, iface: InterfaceSchema, point: Point, context: Context):
        """Fold each dim the impl's stream map folds: for each position ``(dim_index,
        elems_expr)``, split that tensor dim into ``(extent // elems, elems)``. The engine's
        ``fold_map`` names WHICH dims fold (any dim, not just the last), so a 2-D weight
        block ``(MW, MH)`` streamed ``[SIMD, PE]`` folds to ``(MW/SIMD, MH/PE, SIMD, PE)``.
        Raises when a fold width is a cross-interface expr (not a plain tensor-axis
        reshape)."""
        if not self._folds_reshape(iface, point):
            raise KernelError(
                f"interface {iface.name!r} does not fold a tensor axis (its stream WIDTH "
                f"resolves via get_*stream_width, but a folded SHAPE is not a plain "
                f"reshape for this port, e.g. a cross-interface weight width)"
            )
        normal = tuple(context.tensor_shape(iface.tensor))
        if not normal:
            raise KernelError(f"interface {iface.name!r} has no shape to fold")
        gen = self._generated(self._selected(point))
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
