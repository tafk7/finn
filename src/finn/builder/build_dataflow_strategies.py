# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The strategies a build lists, each by its JSON form: the entries of ``step_explore``'s
``strategies`` setting.

An entry is a bare name or a one-key object, decoded by its type (``finn.util.typed_json``),
so a build described by its JSON alone lists every strategy that has a JSON form:

- ``"size_fifos"``, ``"defaults"``, ``"debug_placeholder"``: a name, a strategy that takes no
  option (``size_fifos`` also as an object, with its options);
- ``{"hold": {"MatMul_0": {"simd": 8}}}``: the values to hold, by owner and key;
- ``{"target_throughput": {"fps": 1e6, "relax": true}}``;
- ``{"max_throughput": {"within": {"lut": 0.5}}}``;
- ``{"size_fifos": {"margin": 2, "ram_style": "block"}}``.

An entry is named as its strategy reports itself (``Hold.strategy``, ...), one name an entry
and a report. Each entry refuses a bad option when it is made, as a ``ValueError``, so the
build's plan refuses it before any step runs. ``strategy()`` makes a new strategy each call:
a strategy keeps what its run found. A strategy with no JSON form (``Ranked`` takes a ranking
object) is no entry; Python lists the entries' own values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias, TypeVar

from finn.core.search import Composition, ExploreError, Hold, Seam
from finn.kernels.explore import (
    DebugPlaceholder,
    Defaults,
    KernelExplorer,
    MaxThroughput,
    SizeFifos,
    TargetThroughput,
)

_SCALAR = (bool, int, str)

S = TypeVar("S", bound=Composition)


def _as_decided(kind: str | None, value: object, place: str) -> object:
    """``value``, as JSON states it, read as a Decision of value type ``kind`` takes it, the
    way a stored choice is read back (an ONNX ``i`` attribute of a ``bool`` Decision is a
    bool): a bool as a bool, 0 or 1 as the bool, an int as an int (never a bool or a float), a
    str as a str; a value of another type is refused, named. A key whose type is not one of
    these is left to the seam."""
    if kind == "bool":
        fits = isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))
    elif kind == "int":
        fits = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "str":
        fits = isinstance(value, str)
    else:
        return value
    if not fits:
        raise ExploreError(
            "hold-value-invalid",
            Hold.strategy,
            f"{place}: {value!r} is no {kind}, the type of its Decision",
        )
    return bool(value) if kind == "bool" else value


def _type_of(network: Composition, owner: str, key: str) -> str | None:
    """The value type of the Decision ``key`` of member ``owner`` (``int``, ``bool``,
    ``str``), None where the network has no such Decision."""
    info = network.decision((owner, key)) if owner in network.members else None
    semantics = None if info is None else info.reference.semantics
    return None if semantics is None else str(semantics.name)


class JsonHold(Hold):
    """A hold whose values are as JSON states them: each is read as the type of its
    Decision (``_as_decided``) when the hold runs, the network being known then, and set and
    held as ``Hold`` does."""

    def explore(self, seam: Seam, network: S) -> S:
        self.values = {
            owner: {
                key: _as_decided(_type_of(network, owner, key), value, f"{owner}.{key}")
                for key, value in keys.items()
            }
            for owner, keys in self.values.items()
        }
        return super().explore(seam, network)


class Named(Enum):
    """The strategies that take no option, by their name."""

    SIZE_FIFOS = "size_fifos"
    DEFAULTS = "defaults"
    DEBUG_PLACEHOLDER = "debug_placeholder"

    def strategy(self) -> KernelExplorer:
        """A new strategy of this name."""
        if self is Named.SIZE_FIFOS:
            return SizeFifos()
        if self is Named.DEFAULTS:
            return Defaults()
        return DebugPlaceholder()


@dataclass(frozen=True)
class HoldEntry:
    """``{"hold": {owner: {key: value}}}``: the choices to hold at their values, by the graph
    name of their owner (a node or a tensor) and their key there; no strategy after it
    changes them (``Hold``). A value is a bool, an int or a str, read as its Decision's
    type."""

    hold: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        for owner, keys in self.hold.items():
            for key, value in keys.items():
                if not isinstance(value, _SCALAR):
                    raise ValueError(f"hold.{owner}.{key}: a bool, an int or a str, not {value!r}")

    def strategy(self) -> KernelExplorer:
        """A new hold of these values."""
        return JsonHold(self.hold)


@dataclass(frozen=True)
class TargetThroughputOptions:
    """``fps``: the frames a second to meet at the target's clock; ``relax``: a throughput
    the folding cannot reach is relaxed to the fastest it reaches."""

    fps: float
    relax: bool = True


@dataclass(frozen=True)
class TargetThroughputEntry:
    """``{"target_throughput": {"fps": ..., "relax": ...}}`` (``TargetThroughput``)."""

    target_throughput: TargetThroughputOptions

    def __post_init__(self) -> None:
        self.strategy()

    def strategy(self) -> KernelExplorer:
        """A new strategy: the least parallelism that meets ``fps``."""
        options = self.target_throughput
        return TargetThroughput(options.fps, relax=options.relax)


@dataclass(frozen=True)
class MaxThroughputOptions:
    """``within``: a fraction of the part's resources for each resource named
    (``{"lut": 0.5}``)."""

    within: dict[str, float]


@dataclass(frozen=True)
class MaxThroughputEntry:
    """``{"max_throughput": {"within": {...}}}`` (``MaxThroughput``)."""

    max_throughput: MaxThroughputOptions

    def __post_init__(self) -> None:
        self.strategy()

    def strategy(self) -> KernelExplorer:
        """A new strategy: the most throughput within the budget."""
        return MaxThroughput(self.max_throughput.within)


@dataclass(frozen=True)
class SizeFifosOptions:
    """``margin``: words added to each FIFO's least depth; ``ram_style``: the memory style
    committed with each depth (None: the held, the earlier, else ``auto``)."""

    margin: int = 0
    ram_style: str | None = None


@dataclass(frozen=True)
class SizeFifosEntry:
    """``{"size_fifos": {"margin": ..., "ram_style": ...}}`` (``SizeFifos``)."""

    size_fifos: SizeFifosOptions

    def __post_init__(self) -> None:
        self.strategy()

    def strategy(self) -> KernelExplorer:
        """A new strategy: every transport not held sized."""
        options = self.size_fifos
        return SizeFifos(margin=options.margin, ram_style=options.ram_style)


StrategyEntry: TypeAlias = (
    Named | HoldEntry | TargetThroughputEntry | MaxThroughputEntry | SizeFifosEntry
)
"""One entry of ``step_explore``'s ``strategies``: a name or a one-key object."""


__all__ = [
    "HoldEntry",
    "JsonHold",
    "MaxThroughputEntry",
    "MaxThroughputOptions",
    "Named",
    "SizeFifosEntry",
    "SizeFifosOptions",
    "StrategyEntry",
    "TargetThroughputEntry",
    "TargetThroughputOptions",
]
