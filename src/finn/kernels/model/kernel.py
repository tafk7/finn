"""``InterfaceSchema`` — one op-side interface: identity + DIRECTION + BLOCK structure.

This module used to also hold ``DataflowOp``, the op container — interfaces + pool +
the Tier-3 projections. That container is GONE: its contents are op-CLASS identity, so they
live in the op class's own body now (``ir/dataflow_op.py``'s ``DataflowOp``, and each
concrete op's subclass). `MvauDataflowOp` IS the MVAU kernel; there is no separate value to
build and no ``.kernel()`` hop to make. See design pitch F6.

What is left here is the interface DECLARATION and its two helpers, which are pure data with
no graph and no ONNX:

* :class:`InterfaceSchema` — name, direction, node-slot index, ``block`` (the math),
  optional-ness, per-port constraints. **No role, no stream** — a backend owns the
  BLOCK→STREAM fold, and cannot change the block because it has no field to express one.
* :func:`_resolve_interface_indices` — resolves the ``-1`` slot sentinel to declaration
  order among same-direction peers. Called from ``DataflowOp.__init_subclass__``, so it
  fires at class definition.
* :class:`KernelError` — the ill-formed-query/ill-formed-declaration error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .ports import Direction, Fixed, Multiplicity, Protocol


class KernelError(ValueError):
    """Raised for an ill-formed op declaration or query — a port-direction violation at class
    definition, an unknown interface index, a folded-shape request on a non-last-axis PARAM
    port, or a stream dial that does not divide."""


# The three DATAFLOW protocols an InterfaceSchema may declare (pitch §7.0); Sideband/
# Clock/Reset are emit-only pins, not phase-1 interfaces.
_DATAFLOW_PROTOCOLS = frozenset({Protocol.Stream, Protocol.MemoryMapped, Protocol.Config})


def _resolve_interface_indices(interfaces):
    """Resolve each interface's ``index`` sentinel (``-1``) to its declaration-order position
    among same-direction peers, leaving any explicitly-set index untouched. Called once at
    ``DataflowOp`` subclass CREATION so ``iface.index`` is always a concrete node-slot index
    the adapter can read (the index-authoritative-at-the-ONNX-boundary rule, F9)."""
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
    math → op-owned; stream folding is the realization → backend-owned. An backend cannot change
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
            declared (matches brainsmith's block_tiling). The backend's ``stream[name][i]``
            folds ``block[i]``, positionally.
        optional: whether this is an OPTIONAL node input (0-or-1). An ONNX-invariant
            identity fact (like ``direction``) — the opset says the slot may be absent (MVU
            thresholds, a bias, MaxPool indices). Its PRESENCE for a given node is EMERGENT
            from Context (its tensor exists), read at projection time by
            :meth:`DataflowOp.present_interfaces` — the same declared-slot / emergent-existence
            split as role emergence. A required interface (default ``False``) is always
            present. (Variadic 0-to-N is a later additive generalization of the same rule:
            present interfaces come from Context, not the declared list.)

        constraints: per-port structural-legality
            :mod:`~finn.kernels.engine.constraints` for THIS port — identity rules a
            backend cannot override (a rank check, a static-initializer requirement). Each
            compiles to a predicate that auto-skips when the port's tensor is absent
            (so an optional port needs no ``has_tensor`` guard). Relational rules that read
            ANOTHER tensor live on :attr:`DataflowOp.constraints` instead.

        index: the node-slot index WITHIN this interface's direction (0=first input/output).
            The adapter reads ``node.input[index]``/``node.output[index]`` to resolve this
            interface's Context tensor — the index-authoritative-at-the-ONNX-boundary fact
            (F9). Defaults to the sentinel ``-1`` = "declaration order among same-direction
            peers", resolved to a concrete positional index at ``DataflowOp`` construction
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
            ``Variadic`` is ONNX ``Variadic``, a homogeneous repeat. :meth:`DataflowOp.interfaces`
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
