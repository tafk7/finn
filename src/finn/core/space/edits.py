# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration changes and checked publication reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from .results import QueryResult, Rejected

if TYPE_CHECKING:
    from ._configuration import Space

T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")


@dataclass(frozen=True, slots=True)
class Change(Generic[T_co]):
    """One typed choice patch tied to the exact configuration it changes."""

    base: object = field(repr=False)
    scope: int
    node: int
    value: T_co | None = field(default=None, repr=False)
    remove: bool = False


# A batch accepts only Change objects, with heterogeneous payloads. Any erases
# the payload here without widening type inference in nested change(...) calls.
# Factories and individual Change[T] objects retain their precise value type.
ChangeRequest: TypeAlias = Change[Any]


@dataclass(frozen=True, slots=True)
class ChangeOutcome:
    """What a change did to one choice or fact.

    ``result``: for a choice set, its type check, the domain over the facts, read as if
    the choice existed (wherever it sits: present, dormant or waiting on its presence):
    true where it holds the value; refused outside it, or through an input of the
    domain refusing or not applying; unresolved while a fact the domain reads is
    missing (the choice is stored and waits on it). For a choice dropped, the check it
    failed. For a fact, or a clear, ``Available(True)``.

    ``status``: ``"changed"``, ``"unchanged"`` or ``"removed"`` in a published change,
    and ``"dropped"`` for a stored value that new facts leave outside its domain (or
    whose domain's input refuses or does not apply there): left out of the published
    design space, its old value still the receiver's. In a refused change, ``"refused"``
    for what refuses it and ``"checked"`` for the rest. ``requested`` is false for a
    stored choice the change did not name (a value new facts dropped); a saved choice a
    bind drops is requested."""

    owner: str
    result: QueryResult[bool]
    status: Literal["unchanged", "changed", "removed", "dropped", "checked", "refused"]
    requested: bool = True


@dataclass(frozen=True, slots=True)
class BrokenRule:
    """A rule a change breaks: a piece of validity it makes false through an origin it
    was not false through before. A refused change is refused by it; an accepted change
    of facts reports it as a rule newly false in the published design space: by the new
    facts, or revealed by a clear beside them (a bind: by the saved choices). The
    published design space reads refused.

    ``rule``: the origin's key, where the failure starts: a constraint false on its own
    values, a derived value or a view refusing on its own values, a choice whose listed
    domain lists no value (or whose list cannot be made: an input of its domain does not
    apply), a formal two sources supply; never a guard (a guard refuses only through an
    input, and the refusal names the origin behind it). ``result``: the origin's own
    refusal. ``standing``: the stored choices standing in its way, by key: those the
    origin reads, less those deciding whether it exists. On a refused change, less those
    the change names: the caller sets them in the same change or clears them first (none
    where the values named and the facts alone break it). On an accepted one, every such
    stored choice of the published design space, which a later change may clear (none
    where the facts alone break it)."""

    rule: str
    result: Rejected
    standing: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigurationResult(Generic[S]):
    """A change's report: the design space it publishes (the receiver where it changes
    nothing, or is refused), whether it was accepted, and each item's outcome (the facts
    given, the choices named, then the stored choices new facts dropped, in the flow order).
    ``broken``: on a refused change, the rules it is refused by; on an accepted one, the
    rules newly false in the published design space: by its new facts, or revealed by a
    clear beside them (a change of facts), or broken by the saved choices (a bind); the
    published design space reads refused. Empty on an accepted change of choices. An
    accepted change names, by key, the stored choices it puts out of play (``dormant``:
    present or waiting before, absent after), those it brings into play (``in_play``: not
    present before, present after) and the open choices it brings into presence
    (``opened``)."""

    instance: S
    accepted: bool
    outcomes: tuple[ChangeOutcome, ...]
    broken: tuple[BrokenRule, ...] = ()
    dormant: tuple[str, ...] = ()
    in_play: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()


__all__ = [
    "BrokenRule",
    "Change",
    "ChangeOutcome",
    "ChangeRequest",
    "ConfigurationResult",
]
