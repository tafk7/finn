# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A network's members, configured, and the per-build member cache that keeps them.

A member of a network (``finn.kernels.network``) is a Space class bound to the arguments
it is given (``Facts``: a kernel's formals and tensors, a channel's facts), with the
choices it stores replayed on it (``configured``): the point, and what replay dropped
(refused or inapplicable, each with why) or, for a change, refused. Every bind of a
network configures each member again, and a new point evaluates again what an equal
point evaluated before, so the configured member is kept.

**The key** is what the computation reads: the Space class, its arguments by name, each
by its exact type and value (``1``, ``1.0`` and ``True`` are three keys, as the engine
tells them apart; a tuple, list, set or dict by each item's), and the choices replayed in
their order: those stored, and for a channel in a trial the change and the clears (replay
falls back to each stored choice alone, in order). A large value is
keyed by its digest: an ``IntegerTensorValue`` is equal by its shape and integers'
sha256, loaded only when read. A value class (a frozen dataclass, a datatype) is keyed by
its type and its own equality. The class's code and the platform catalog are fixed for a
process. Arguments that do not hash are not kept.

**The entry** is immutable: the point, and what replay dropped and refused as read-only
mappings. A member with nothing replayed is the entry with nothing stored, and each
replay on it starts from that point, so what the arguments alone decide is evaluated
once.

**One cache per build** (``member_cache``): the builder opens one when a build starts and
drops it when the build ends; it is unbounded inside. A test or a script opens one
itself. With none open, nothing is kept.

**The verifying mode** (the environment variable ``FINN_VERIFY_MEMBER_CACHE``, any value
but ``0``): every hit is configured again without the cache and compared with it, by
what a network reads of a member (its class, its stored choices, what replay dropped and
refused, its forced, viable and open Decisions, its validity and every export's
answer); a mismatch raises ``MemberCacheMismatch``, naming the key. Where the variable
names a directory, each process writes its counts there when it ends
(``verification_counts``). The graph's network cache
(``finn.custom_op.kernels.network``) verifies its hits the same way.
"""

from __future__ import annotations

import atexit
import json
import os
from collections import Counter
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    ConfigurationError,
    Inapplicable,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    design_space,
    inspection,
)
from finn.kernels.configure import chosen

VERIFY = "FINN_VERIFY_MEMBER_CACHE"
"""The environment variable that switches the verifying mode on (see the module docstring)."""


class Facts:
    """A member as it is bound: its Space class and the arguments it is given, by name (a
    record, not a value class: its arguments are anything a Space takes)."""

    __slots__ = ("arguments", "space_type")

    space_type: type[Space]
    arguments: Mapping[str, object]

    def __init__(self, space_type: type[Space], arguments: Mapping[str, object]) -> None:
        self.space_type = space_type
        self.arguments = MappingProxyType(dict(arguments))

    def __repr__(self) -> str:
        return f"Facts({self.space_type.__name__}, {dict(self.arguments)!r})"

    def bind(self) -> Space:
        """Its design space: the class bound to its arguments, nothing chosen."""
        point: Space = design_space(cast(Any, self.space_type)(**self.arguments))
        return point


class Configured:
    """A member configured: its ``point`` (None where a change is refused), the stored
    choices replay ``dropped`` and the change's choices it ``refused``, each by key with
    why (a record the cache keeps, never changed)."""

    __slots__ = ("dropped", "point", "refused")

    point: Space | None
    dropped: Mapping[str, tuple[Finding, ...]]
    refused: Mapping[str, tuple[Finding, ...]]

    def __init__(
        self,
        point: Space | None,
        dropped: Mapping[str, tuple[Finding, ...]] | None = None,
        refused: Mapping[str, tuple[Finding, ...]] | None = None,
    ) -> None:
        self.point = point
        self.dropped = MappingProxyType(dict(dropped or {}))
        self.refused = MappingProxyType(dict(refused or {}))

    def __repr__(self) -> str:
        return f"Configured({self.point!r}, {dict(self.dropped)!r}, {dict(self.refused)!r})"


class MemberCacheMismatch(AssertionError):
    """A hit of the verifying mode that differs from the member configured again."""


# -- replaying choices on a point ------------------------------------------------------------


def rejection(code: str, owner: str, message: str) -> tuple[Finding, ...]:
    return (Finding(FindingKind.REJECTION, code, owner, message),)


def decision(subject: Space | type[Space], key: str) -> inspection.DecisionInfo[object] | None:
    """The Decision ``key`` of a point or a Space class, or None: found by its key in the
    engine's table of its compiled model."""
    return next((item for item in inspection.decisions(subject) if item.key == key), None)


def _refusals(point: Space, report: Any, keys: Collection[str]) -> dict[str, tuple[Finding, ...]]:
    """Each Decision a refused request names, with why (an inapplicable one has no finding
    of its own, so it is named), or every key of the request refused together."""
    found: dict[str, tuple[Finding, ...]] = {}
    for outcome in report.outcomes:
        if outcome.status != "refused":
            continue
        result = outcome.result
        findings = (
            result.findings if isinstance(result, (Inapplicable, Rejected, Unresolved)) else ()
        )
        found[outcome.owner] = (
            rejection("inapplicable", outcome.owner, "inapplicable")
            if isinstance(result, Inapplicable) or not findings
            else findings
        )
    return found or {key: rejection("refused-together", key, "refused together") for key in keys}


def commit(
    point: Space, values: Mapping[str, object], clear: Collection[str] = ()
) -> Space | dict[str, tuple[Finding, ...]]:
    """``point`` with ``values`` committed and ``clear`` cleared, atomically, or each key
    refused with why."""
    handles = {key: decision(point, key) for key in (*values, *clear)}
    unknown = [key for key, item in handles.items() if item is None]
    if unknown:
        return {
            key: rejection("not-a-decision", key, "not a Decision of this member")
            for key in unknown
        }
    found = cast(dict[str, inspection.DecisionInfo[object]], handles)
    try:
        report = point.try_with_choices(
            {found[key].reference: value for key, value in values.items()},
            *(point.field(found[key].reference).clear() for key in clear),
        )
    except (RequestError, ConfigurationError) as error:
        return {key: rejection("request-refused", key, str(error)) for key in (*values, *clear)}
    if report.accepted:
        instance: Space = report.instance
        return instance
    return _refusals(point, report, (*values, *clear))


def replayed(
    base: Space,
    stored: Mapping[str, object],
    change: Mapping[str, object] | None = None,
    clear: Collection[str] = (),
) -> Configured:
    """``base`` with its ``stored`` choices replayed and ``change`` committed, ``clear``
    cleared of either: the point, the stored choices dropped (refused or inapplicable,
    each with why) and the change's refused, by key. A stored choice ``change`` replaces is
    not dropped. Nothing to replay is ``base`` itself."""
    change = change or {}
    values = {key: value for key, value in stored.items() if key not in clear} | dict(change)
    if not values and not clear:
        return Configured(base)
    dropped: dict[str, tuple[Finding, ...]] = {}
    while True:
        outcome = commit(base, values)
        if isinstance(outcome, Space):
            return Configured(outcome, dropped)
        named = {key: why for key, why in outcome.items() if key in values}
        refused = {key: why for key, why in named.items() if key in change}
        if refused:
            return Configured(None, dropped, refused)
        if named:
            dropped |= named
            values = {key: value for key, value in values.items() if key not in named}
            continue
        # Refused naming none of them: the change first, then each stored choice alone.
        point = commit(base, change)
        if not isinstance(point, Space):
            return Configured(None, dropped, point)
        for key, value in values.items():
            if key in change:
                continue
            alone = commit(point, {key: value})
            if isinstance(alone, Space):
                point = alone
            else:
                dropped[key] = tuple(finding for why in alone.values() for finding in why)
        return Configured(point, dropped)


# -- the key ---------------------------------------------------------------------------------

_ATOMS = frozenset((bool, int, float, complex, str, bytes, type(None)))


def typed(value: object) -> Hashable:
    """``value`` as a key part, by its exact type and value: an immutable builtin as
    ``(type, value)``, a tuple, list, set or dict by each item's, anything else as
    ``(type, value)`` by its own equality."""
    kind = type(value)
    if kind in _ATOMS:
        return (kind, value)
    if kind is tuple or kind is list:
        return (kind, tuple(typed(item) for item in cast("tuple[object, ...]", value)))
    if kind is frozenset or kind is set:
        return (kind, frozenset(typed(item) for item in cast("frozenset[object]", value)))
    if kind is dict or kind is MappingProxyType:
        items = cast("Mapping[object, object]", value).items()
        return (kind, frozenset((typed(name), typed(item)) for name, item in items))
    return (kind, value)


def _items(values: Mapping[str, object]) -> tuple[tuple[str, Hashable], ...]:
    return tuple(sorted(_ordered(values)))


def _ordered(values: Mapping[str, object]) -> tuple[tuple[str, Hashable], ...]:
    return tuple((name, typed(value)) for name, value in values.items())


def key(
    facts: Facts,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str],
) -> Hashable | None:
    """The member's key (the module docstring), or None where an argument does not hash."""
    found = (
        facts.space_type,
        _items(facts.arguments),
        _ordered(stored),
        _ordered(change),
        tuple(sorted(clear)),
    )
    try:
        hash(found)
    except TypeError:
        return None
    return found


# -- the cache ---------------------------------------------------------------------------------


def _verifying() -> bool:
    return os.environ.get(VERIFY, "0") not in ("", "0")


_COUNTS = {"member hits verified": 0, "network hits verified": 0, "mismatches": 0}


class MemberCache:
    """The configured members of one build, by key (the module docstring), unbounded, and
    the graph's networks bound (``finn.custom_op.kernels.network``). In the verifying
    mode, each hit is configured again and compared (``verify``). ``counts``: the hits
    and the entries made, by kind; ``hits`` and ``misses`` are the members'."""

    def __init__(self, *, verify: bool | None = None) -> None:
        self.verify = _verifying() if verify is None else verify
        self.entries: dict[Hashable, object] = {}
        self.counts: Counter[tuple[str, str]] = Counter()

    @property
    def hits(self) -> int:
        return self.counts["member", "hits"]

    @property
    def misses(self) -> int:
        return self.counts["member", "misses"]

    def get(
        self,
        found: Hashable,
        make: Callable[[], Any],
        again: Callable[[], Any],
        observed: Callable[[Any], object],
        kind: str = "member",
    ) -> Any:
        """The entry of key ``found``, made on a miss (``make``); on a hit in the verifying
        mode, made again without the cache (``again``) and compared by ``observed``. A make
        that raises keeps nothing, and is not counted."""
        entry = self.entries.get(found)
        if entry is None:
            entry = make()
            self.entries[found] = entry
            self.counts[kind, "misses"] += 1
            return entry
        self.counts[kind, "hits"] += 1
        if self.verify:
            check(found, observed(entry), observed(again()), kind)
        return entry


def check(found: Hashable, hit: object, again: object, kind: str = "member") -> None:
    """Count a verified hit; raise ``MemberCacheMismatch`` naming the key where ``hit``
    differs from ``again``."""
    _COUNTS[f"{kind} hits verified"] += 1
    if hit != again:
        _COUNTS["mismatches"] += 1
        raise MemberCacheMismatch(
            f"the member cache's {kind} hit differs from the {kind} made again, key {found!r}: "
            f"hit {hit!r}, again {again!r}"
        )


_CACHE: ContextVar[MemberCache | None] = ContextVar("finn_member_cache", default=None)


@contextmanager
def member_cache(*, verify: bool | None = None) -> Iterator[MemberCache]:
    """A member cache open for the block (a build's, a test's), dropped at its end; one
    opened inside another replaces it for the inner block."""
    cache = MemberCache(verify=verify)
    token = _CACHE.set(cache)
    try:
        yield cache
    finally:
        _CACHE.reset(token)


def current() -> MemberCache | None:
    """The member cache open here, or None."""
    return _CACHE.get()


@contextmanager
def uncached() -> Iterator[None]:
    """No member cache open for the block: what the verifying mode computes again."""
    token = _CACHE.set(None)
    try:
        yield
    finally:
        _CACHE.reset(token)


def observed(entry: Configured) -> object:
    """What a network reads of a configured member, to compare in the verifying mode."""
    point = entry.point
    if point is None:
        return (None, dict(entry.dropped), dict(entry.refused))
    exports: dict[str, object] = {}
    for name, handle in sorted(type(point).exports.items(), key=lambda item: str(item[0])):
        try:
            exports[str(name)] = point.query(handle)
        except Exception as error:  # noqa: BLE001 - compared, not handled
            exports[str(name)] = (type(error), str(error))
    return (
        type(point),
        chosen(point),
        dict(entry.dropped),
        dict(entry.refused),
        inspection.forced(point),
        inspection.viable(point),
        inspection.open(point),
        inspection.validity(point),
        exports,
    )


def configured(
    facts: Facts,
    stored: Mapping[str, object] | None = None,
    change: Mapping[str, object] | None = None,
    clear: Collection[str] = (),
) -> Configured:
    """The member ``facts`` binds, its ``stored`` choices replayed with ``change``
    committed and ``clear`` cleared (``replayed``); kept in the open member cache."""
    stored, change = stored or {}, change or {}
    cache = current()
    found = None if cache is None else key(facts, stored, change, clear)
    if cache is None or found is None:
        return _configured(facts, stored, change, clear, None)
    return cast(
        Configured,
        cache.get(
            found,
            lambda: _configured(facts, stored, change, clear, cache),
            lambda: _configured(facts, stored, change, clear, None),
            observed,
        ),
    )


def _configured(
    facts: Facts,
    stored: Mapping[str, object],
    change: Mapping[str, object],
    clear: Collection[str],
    cache: MemberCache | None,
) -> Configured:
    if not stored and not change and not clear:
        return Configured(facts.bind())
    # Replayed on the binding with nothing stored: its own entry where a cache is open.
    base = configured(facts).point if cache is not None else facts.bind()
    assert base is not None
    return replayed(base, stored, change, clear)


def verification_counts() -> dict[str, int]:
    """The verifying mode's counts in this process: the hits verified, by kind, and the
    mismatches."""
    return dict(_COUNTS)


def write_counts() -> None:
    """Write this process's verification counts as ``<pid>.json`` into the directory the
    environment variable names, where it names one."""
    named = os.environ.get(VERIFY, "")
    if _verifying() and os.path.isdir(named):
        (Path(named) / f"{os.getpid()}.json").write_text(json.dumps(verification_counts()))


atexit.register(write_counts)


__all__ = [
    "VERIFY",
    "Configured",
    "Facts",
    "MemberCache",
    "MemberCacheMismatch",
    "check",
    "commit",
    "configured",
    "current",
    "decision",
    "key",
    "member_cache",
    "observed",
    "rejection",
    "replayed",
    "typed",
    "uncached",
    "verification_counts",
    "write_counts",
]
