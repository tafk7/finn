# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay: persisted choices committed on a kernel point.

A node's choices and its owned channels' are committed atomically on the node's cached
base point (``committed``), each typed as its Decision takes it (``typed_choices``: an
ONNX ``i`` attribute of a ``bool`` Decision becomes a bool). A refusal names every
refused key with why (``refusal``), an inapplicable one too, which carries no finding
of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

from finn.core.graph.annotations.channel import value_type
from finn.core.graph.errors import KernelOpError
from finn.core.space import Inapplicable, Space, inspection
from finn.kernels.configure import describe

S = TypeVar("S", bound=Space)


def typed_choices(
    subjects: Iterable[Space | type[Space]], choices: Mapping[str, object]
) -> dict[str, object]:
    """``choices``, by decision key of ``subjects`` (spaces or space classes), as replay
    takes them: an ONNX ``i`` attribute of a ``bool`` decision becomes a bool."""
    kinds = {item.key: value_type(item) for each in subjects for item in inspection.decisions(each)}
    return {
        key: bool(value) if kinds.get(key) == "bool" else value for key, value in choices.items()
    }


def refusal(label: str, found: Mapping[str, str]) -> KernelOpError:
    """The error for choices replay refused: each one, by name, with why."""
    return KernelOpError(
        f"{label}: refused choices: "
        + "; ".join(f"{name}: {why}" for name, why in sorted(found.items())),
        tuple(sorted(found)),
    )


def committed(point: S, choices: Mapping[str, object]) -> S | dict[str, str]:
    """``choices`` committed on ``point`` atomically, or each refused key with why. An
    inapplicable key carries no finding of its own, so it is named here."""
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    found = {key: "not a choice here" for key in choices if key not in handles}
    if found:
        return found
    report = point.try_with_choices({handles[key]: value for key, value in choices.items()})
    if report.accepted:
        return report.instance
    for outcome in report.outcomes:
        if outcome.status == "refused":
            result = outcome.result
            found[outcome.owner] = (
                "inapplicable" if isinstance(result, Inapplicable) else describe([result])
            )
    return found or dict.fromkeys(choices, "refused together")


__all__ = [
    "committed",
    "refusal",
    "typed_choices",
]
