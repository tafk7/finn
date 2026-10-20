# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reads of the answers a change, ``inspection.viable(point, keys)`` and
``inspection.states(point)`` give: each value possible or refused, each refusal with its
rules and the stored choices standing in the way, and each choice's state, as plain
values, so the tests state what they expect in plain dictionaries."""

from __future__ import annotations

from typing import Any

from finn.core.space import Space, inspection
from finn.core.space.edits import ConfigurationResult
from finn.core.space.inspection import Open, Possible, Refused, Viable


def _viable(point: Space, key: object) -> Viable:
    (item,) = inspection.viable(point, [key])
    return item


def possible(point: Space, key: object) -> tuple[object, ...]:
    """``key``'s listed values nothing refuses, in list order."""
    return _viable(point, key).cases


def refusals(point: Space, key: object) -> dict[object, dict[str, set[str]]]:
    """``key``'s refused values: for each, its broken rules, each with its standing
    choices (by key)."""
    return {
        value.value: {rule.rule: set(rule.standing) for rule in value.broken}
        for value in _viable(point, key).values or ()
        if isinstance(value, Refused)
    }


def listed(point: Space, key: object) -> tuple[object, ...] | None:
    """``key``'s listed values, possible or refused (``None``: known by membership only)."""
    values = _viable(point, key).values
    return None if values is None else tuple(value.value for value in values)


def moves(point: Space, key: object) -> dict[object, tuple[tuple[str, ...], ...]]:
    """``key``'s possible values: for each, the choices it would put dormant, bring into
    play and open."""
    return {
        value.value: (value.dormant, value.in_play, value.opened)
        for value in _viable(point, key).values or ()
        if isinstance(value, Possible)
    }


def open_now(point: Space) -> list[str]:
    """The keys of the choices open and present (an ``Open`` state waiting on nothing), in
    rank order."""
    return [
        key
        for key, state in inspection.states(point)
        if isinstance(state, Open) and not state.waits
    ]


def broken(report: ConfigurationResult[Any]) -> dict[str, set[str]]:
    """A refused change's broken rules, each with its standing choices (by key)."""
    return {rule.rule: set(rule.standing) for rule in report.broken}
