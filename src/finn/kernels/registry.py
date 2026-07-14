############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""The op pool and contextful selection.

Selection is ``feasibility ⊥ preference`` — the two are orthogonal:

* **Feasibility** is a hard predicate, :meth:`Implementation.precondition`,
  evaluated against a device-aware :class:`SelectionContext` (carrying
  ``fpgapart`` + the derived design point). This is the seam neither prior
  system had: the prototype's constraint was ``Callable[[Kernel], bool]`` with
  no device info, and Brainsmith ported FINN's centralized device ladder into
  the specialize transform rather than onto the backend.
* **Preference** ranks the feasible survivors — by ``priority`` (lower first),
  then a pluggable ``cost_fn`` (an intentionally OPEN algorithm seam; the
  default is first-by-priority).

Implementations are registered as **classes** and resolved **by name** — the
value stored in the ``implementation`` nodeattr re-looks-up the class, so a
saved graph reconstructs behavior without pickling anything.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .implementation import Implementation, SelectionContext

#: A preference strategy: given the feasible candidates (already priority-sorted)
#: and the context, return the chosen one. The open seam for a real cost model.
CostFn = Callable[[Sequence[Implementation], SelectionContext], Implementation]


class NoFeasibleImplementation(Exception):
    """Raised when no registered implementation is feasible for the context."""


class KernelRegistry:
    """Pool of implementations keyed by op kind."""

    def __init__(self) -> None:
        self._pool: dict[str, list[type[Implementation]]] = {}
        self._by_name: dict[str, type[Implementation]] = {}

    # ------------------------------------------------------------- registration
    def register(self, impl_cls: type[Implementation]) -> type[Implementation]:
        """Register an implementation class. Usable as a decorator."""
        op_kind = impl_cls.op_kind
        name = impl_cls.name
        if name in self._by_name and self._by_name[name] is not impl_cls:
            raise ValueError(f"implementation name collision: {name!r}")
        self._pool.setdefault(op_kind, [])
        if impl_cls not in self._pool[op_kind]:
            self._pool[op_kind].append(impl_cls)
        self._by_name[name] = impl_cls
        return impl_cls

    # ------------------------------------------------------------- introspection
    def implementations_for(self, op_kind: str) -> list[type[Implementation]]:
        return list(self._pool.get(op_kind, []))

    def get_by_name(self, name: str) -> Implementation:
        """Reconstruct an implementation instance by its stored name. This is the
        by-name resolution that makes ONNX round-trip work without pickling."""
        if name not in self._by_name:
            raise KeyError(f"no implementation registered as {name!r}")
        return self._by_name[name]()

    # ------------------------------------------------------------------- select
    def feasible(self, op_kind: str, ctx: SelectionContext) -> list[Implementation]:
        """All feasible candidates, priority-sorted (lower priority first, then
        name for a deterministic tie-break — no import-order dependence)."""
        candidates = [cls() for cls in self._pool.get(op_kind, [])]
        viable = [impl for impl in candidates if impl.precondition(ctx)]
        viable.sort(key=lambda impl: (impl.priority, impl.name))
        return viable

    def select(
        self,
        op_kind: str,
        ctx: SelectionContext,
        cost_fn: CostFn | None = None,
    ) -> Implementation:
        """Choose one implementation: feasibility filter, then preference.

        ``cost_fn`` is the open preference seam; when absent, the
        lowest-``priority`` feasible candidate wins.
        """
        viable = self.feasible(op_kind, ctx)
        if not viable:
            raise NoFeasibleImplementation(
                f"no feasible implementation for {op_kind!r} on {ctx.fpgapart!r}"
            )
        if cost_fn is None:
            return viable[0]
        return cost_fn(viable, ctx)


#: Process-wide default registry. Concrete kernels register their
#: implementations onto this at import time.
registry = KernelRegistry()
