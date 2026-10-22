# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The design spaces the oracle compares, by target name (``TARGETS``), each a list of
stations: a design space, opened by today's API, and the change made on it.

- ``toys``: FINN's own toy Spaces (``toys.TOYS``: the house, the pipeline, a chain of
  stages, the boundary of a formal read from outside its scope, and the three shapes of
  an optional fact left out), each opened and changed once;
- ``chain-8``, ``chain-64``: stage 7's chain of 8 or 64 kernels (``support.one_space``),
  configured with what conversion and the cut stored (``commit``), then the middle
  MatMul's ``pe`` set to 2;
- ``tfc``: TFC's partition after the cut and after each of seven strategy sequences, in
  ``tfc_target`` (it reads FINN's builder), joined to these by the tool's entry.

Possible values are asked of every choice up to 8 kernels, and of the middle MatMul's
choices beyond. A target is made in the process that compares it: set
``FINN_SPACE_PERMUTE=1`` before it is made to compare the permuted numbering.

This module reads FINN's kernels and flow; the rest of the package is engine-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from finn.core.space import Space, design_space, inspection
from finn.kernels.configure import commit
from support.one_space import chain_space
from support.space_oracle import Implementation, forms
from support.space_oracle.toys import TOYS

WIDEST = 8
"""Up to this many kernels, every choice's possible values are asked; beyond, the middle
MatMul's."""


@dataclass(frozen=True)
class Station:
    """One design space compared (``make`` opens it) and the change made on it (``change``,
    by key, or ``None``). ``viable``: the choices whose possible values are asked (every
    one: ``forms.ALL``); ``facts``: values a walk may give each root fact, by key;
    ``stored``: the choices it was configured with, by key."""

    label: str
    make: Callable[[], Space]
    change: Mapping[str, object] | None = None
    viable: Sequence[str] | None = forms.ALL
    facts: Mapping[str, Sequence[object]] = field(default_factory=dict)
    stored: Mapping[str, object] = field(default_factory=dict)


def toys() -> list[Station]:
    """FINN's own toy Spaces (``toys.TOYS``), each opened with nothing stored and changed
    once."""
    return [
        Station(name, toy.make, dict(toy.change), facts=toy.facts) for name, toy in TOYS.items()
    ]


def _middle(names: Sequence[str]) -> str:
    """The middle of the MatMuls, in graph order."""
    return names[len(names) // 2]


def chain(kernels: int) -> list[Station]:
    """Stage 7's chain of ``kernels`` kernels, configured as conversion and the cut store
    it, then the middle MatMul's ``pe`` set to 2."""
    whole, stored = chain_space(kernels // 2)
    middle = _middle([f"mm{layer}" for layer in range(kernels // 2)])
    return [_partition(f"chain-{kernels}", whole, stored, kernels, middle, change=True)]


def _partition(
    label: str,
    whole: type[Space],
    stored: Mapping[str, object],
    kernels: int,
    middle: str,
    *,
    change: bool = False,
) -> Station:
    def make() -> Space:
        return commit(design_space(whole()), stored)

    viable: Sequence[str] | None = forms.ALL
    if kernels > WIDEST:
        prefix = f"{middle}."
        viable = [item.key for item in inspection.decisions(make()) if item.key.startswith(prefix)]
    return Station(
        label,
        make,
        {f"{middle}.pe": 2} if change else None,
        viable,
        stored=dict(stored),
    )


TARGETS: Mapping[str, Callable[[], list[Station]]] = {
    "toys": toys,
    "chain-8": lambda: chain(8),
    "chain-64": lambda: chain(64),
}
"""Every target but TFC's, by name: each makes its stations when called. TFC's
(``tfc_target.tfc``) reads FINN's builder, so it is kept out of what the tests import
(the XSim groups are keyed by the code their tests import) and joins these in the
tool's entry (``__main__``)."""


def dumped(station: Station, implementation: Implementation) -> forms.Form:
    """``station`` as ``implementation`` answers it, in its canonical form: the stored
    choices it was configured with, the design space, and its change (the request, the
    report and the successor)."""
    answers = implementation.open(station.make())
    found: dict[str, forms.Form] = {
        "stored": {key: forms.value(value) for key, value in sorted(station.stored.items())},
        "space": forms.space(answers, station.viable),
    }
    if station.change is not None:
        changed = answers.change(station.change, ())
        found["change"] = {
            "request": {key: forms.value(value) for key, value in station.change.items()},
            "report": forms.report(changed.report),
            "after": None if changed.after is None else forms.space(changed.after, station.viable),
        }
    return found


def stations(
    name: str, targets: Mapping[str, Callable[[], list[Station]]] = TARGETS
) -> list[Station]:
    """The stations of target ``name``, one of ``targets``."""
    if name not in targets:
        raise ValueError(f"{name!r}: a target is one of {', '.join(targets)}")
    return targets[name]()


__all__ = [
    "TARGETS",
    "WIDEST",
    "Station",
    "chain",
    "dumped",
    "stations",
    "toys",
]
