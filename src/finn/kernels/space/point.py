############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""The two outcomes of ``resolve``: a ``Point`` or an ``Illegal``.

A ``Point`` is a partial-or-total assignment of values to the axes that *exist*
given the assignments made so far, unioned with the computed ``Derived``
quantities (they share one namespace). Reading a name that is not present —
because its axis guarded out — is a hard error, never a silent default. This is
the guarded-axis discipline the whole model turns on (design-space-model.md §1.2,
matching base:315's raise).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class AbsentAxisError(KeyError):
    """Raised when a guarded-out (absent) axis or an unknown name is read."""


class Point(Mapping):
    """An immutable by-name map over resolved axes + derived quantities.

    Access by attribute (``p.mem_mode``) or by key (``p["mem_mode"]``); both
    raise :class:`AbsentAxisError` on a name that is not in the point. Presence
    is testable without raising via ``"mem_mode" in p`` or ``p.get("mem_mode")``
    — guards use these to branch on whether an upstream axis exists.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]):
        object.__setattr__(self, "_values", dict(values))

    def __getitem__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AbsentAxisError(
                f"'{name}' is absent from this point (its axis guarded out, or the "
                f"name is unknown). Present: {sorted(self._values)}"
            )

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal attribute lookup fails, so __slots__
        # ('_values') is safe. Dunder lookups must not be routed through the map.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AbsentAxisError(
                f"'{name}' is absent from this point (its axis guarded out, or the "
                f"name is unknown). Present: {sorted(self._values)}"
            )

    def __setattr__(self, name: str, value: Any):
        raise AttributeError("Point is immutable")

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Point({body})"


class Illegal:
    """A rejected assignment, carrying the reasons it was rejected.

    Returned (instead of a :class:`Point`) by ``resolve`` when a value falls
    outside its domain or when one or more predicates fail. The reason strings
    power explain-style diagnostics (design-space-model.md §1.4).
    """

    __slots__ = ("reasons",)

    def __init__(self, reasons):
        object.__setattr__(self, "reasons", tuple(reasons))

    def __setattr__(self, name, value):
        raise AttributeError("Illegal is immutable")

    def __bool__(self) -> bool:
        # An Illegal is falsy so callers can write `if not result: ...`.
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, Illegal) and other.reasons == self.reasons

    def __hash__(self) -> int:
        return hash(self.reasons)

    def __repr__(self) -> str:
        body = "; ".join(self.reasons)
        return f"Illegal([{body}])"
