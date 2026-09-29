"""P0 helpers: extent binding (plan D3) and a kernel's `extents` over its own ports (plan D4).

A probe copy only; A3 writes the real `bind_extents` in `finn.dataflow.schedule`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import ClassVar

from finn.core.space import Derived, Rejected, default_semantics, derived, reject
from finn.core.space._nodes import node_record
from finn.dataflow.schedule import Affine, Index, Refused
from finn.kernels.base import Kernel
from finn.kernels.port import ScheduledPort, StreamPort

EXTENTS = default_semantics(dict)


@dataclass(frozen=True)
class Access:
    """One placed port: the tensor it reads and its index; `reshaped` reads a row-major view."""

    name: str
    shape: tuple[int, ...]
    index: tuple[Index | Affine, ...]
    reshaped: bool = False


def _plain(axis: Index | Affine) -> Index | None:
    if isinstance(axis, Index):
        return axis
    terms = Affine.of(axis).terms
    return terms[0][0] if len(terms) == 1 and terms[0][1] == 1 else None


def bind_extents(
    accesses: Sequence[Access], given: Mapping[Index, int] | None = None
) -> dict[Index, int]:
    """Each index's extent, from the axes it alone addresses; raises Refused.

    A reshaped port binds nothing: its view's extents are its indices' extents,
    bound elsewhere, and it is checked (same size). An axis addressed by an
    affine of several indices binds nothing and is checked (reach < extent).
    """
    extents = dict(given or {})
    origin = {index: "given" for index in extents}
    for access in accesses:
        if access.reshaped:
            continue
        if len(access.shape) != len(access.index):
            raise Refused(
                f"{access.name}: {len(access.index)} indices for a rank-{len(access.shape)} tensor"
            )
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            index = _plain(expression)
            if index is None:
                continue
            here = f"{access.name} axis {axis}"
            known = extents.setdefault(index, extent)
            origin.setdefault(index, here)
            if known != extent:
                raise Refused(f"{index!r} is {known} ({origin[index]}) and {extent} ({here})")
    for access in accesses:
        for expression in access.index:
            for index in Affine.of(expression).indices:
                if index not in extents:
                    raise Refused(
                        f"{access.name}: {index!r} has no extent (no axis addresses it alone)"
                    )
        if access.reshaped:
            if not all(isinstance(axis, Index) for axis in access.index):
                raise Refused(f"{access.name}: a reshaped port reads plain indices")
            view = tuple(extents[axis] for axis in access.index)  # type: ignore[index]
            if prod(view) != prod(access.shape):
                raise Refused(f"{access.name}: a {access.shape} tensor cannot be viewed as {view}")
            continue
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            if _plain(expression) is None:
                reach = sum(c * (extents[i] - 1) for i, c in Affine.of(expression).terms)
                if reach >= extent:
                    raise Refused(
                        f"{access.name} axis {axis}: {expression!r} reaches {reach}, "
                        f"beyond extent {extent}"
                    )
    return extents


def port_names(family: type[Kernel]) -> tuple[str, ...]:
    """The kernel's ScheduledPort members, read from its class (never the graph)."""
    names = []
    for klass in reversed(family.__mro__):
        for name, member in vars(klass).items():
            record = node_record(member)
            if record is not None and issubclass(record.family, ScheduledPort):
                if name not in names:
                    names.append(name)
    return tuple(names)


class BoundKernel(Kernel):
    """The D4 base helpers on a probe base: `extents` bound from every placed port."""

    id = "probe.bound"  # every Kernel subclass names itself; the real helpers live on Kernel

    # A class-level stand-in for bound_schedule(extents=...): extents no port gives.
    extents_given: ClassVar[Mapping[Index, int]] = {}

    @derived(semantics=EXTENTS)
    def extents(self) -> dict | Rejected:  # type: ignore[type-arg]
        accesses = []
        for name in port_names(type(self)):
            port = getattr(self, name)
            if not port.present(StreamPort.stream):
                continue  # an idle port binds nothing
            accesses.append(
                Access(name, port.stream.tensor.shape, tuple(port.index), port.reshaped)
            )
        try:
            return bind_extents(accesses, type(self).extents_given)
        except Refused as error:
            return reject("kernel-extents", str(error))


def extent_of(index: Index) -> Derived[int]:
    """A derived member: `index`'s bound extent (refused when no port binds it)."""

    @derived(semantics=default_semantics(int))
    def extent(self: BoundKernel) -> int | Rejected:
        extents = self.extents
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return extent
