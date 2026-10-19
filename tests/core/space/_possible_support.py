# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reads of the answers a change, ``inspection.viable(point, keys)`` and
``inspection.states(point)`` give once a change is refused by the rules it breaks: each
value possible or refused, each refusal with its rules and the stored choices standing in
the way, and each choice's state.

The tests that read them are expected to fail until the engine answers them; every read
of the new answers goes through here, so the tests state what they expect in plain
dictionaries and the shapes are read in one place.
"""

from __future__ import annotations

from typing import Any, cast

from finn.core.space import Space, inspection
from finn.core.space.edits import ConfigurationResult


def _viable(point: Space, key: object) -> Any:
    (item,) = cast(Any, inspection).viable(point, [key])
    return item


def possible(point: Space, key: object) -> tuple[object, ...]:
    """``key``'s listed values nothing refuses, in list order."""
    item = _viable(point, key)
    return tuple(value.value for value in item.values if not hasattr(value, "broken"))


def refusals(point: Space, key: object) -> dict[object, dict[str, set[str]]]:
    """``key``'s refused values: for each, its broken rules, each with its standing
    choices (by key)."""
    item = _viable(point, key)
    return {
        value.value: {rule.rule: set(rule.standing) for rule in value.broken}
        for value in item.values
        if hasattr(value, "broken")
    }


def listed(point: Space, key: object) -> tuple[object, ...] | None:
    """``key``'s listed values, possible or refused (``None``: known by membership only)."""
    values = _viable(point, key).values
    return None if values is None else tuple(value.value for value in values)


def open_now(point: Space) -> list[str]:
    """The keys of the choices open and present (an ``Open`` state waiting on nothing), in
    rank order."""
    return [
        key
        for key, state in cast(Any, inspection).states(point)
        if type(state).__name__ == "Open" and not state.waits
    ]


def broken(report: ConfigurationResult[Any]) -> dict[str, set[str]]:
    """A refused change's broken rules, each with its standing choices (by key)."""
    return {rule.rule: set(rule.standing) for rule in cast(Any, report).broken}
