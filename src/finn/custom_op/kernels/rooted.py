# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A root presented as a network: the seam over a node root or a shell root, until the
graph binds a network itself.

Transitional: the graph still composes its kernels and channels into one Space (a
KernelOp's node root, ``roots``; a partition's shell root, ``shell``), while the seam
(``finn.kernels.explore``) explores networks (``finn.kernels.network``), whose keys are
pairs, ``(owner, key)``. ``Rooted`` presents one configured root through the network's
protocol: its members by owner (a kernel's node, a channel's tensor, by graph name; a
member no graph name owns, by its path), ranked kernels first, then channels; each pair
is the root key below the member's path, translated here and nowhere else. A change is
the engine's trial on the whole root, atomic as before: what it makes invalid is
refused, naming the stored choices, which the seam drops. The binding of a network from
the graph replaces this module, the node root and the shell root, with their key
translation.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeVar

from finn.core.space import (
    ConfigurationError,
    Finding,
    FindingKind,
    Inapplicable,
    QueryResult,
    Rejected,
    RequestError,
    Unresolved,
    ViewKey,
    inspection,
)
from finn.kernels.base import RESOURCES
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, member_of
from finn.kernels.network import Key, Trial, dotted
from finn.platform.capabilities import Resources

T = TypeVar("T")


def _member(point: object, path: str) -> Any:
    """The member at a dotted ``path`` below ``point``."""
    found = point
    for name in path.split("."):
        found = getattr(found, name)
    return found


class Rooted:
    """A configured root (``point``) as a network: its members at ``paths``, each owned by
    ``owners[path]`` (a graph name; a path no owner names owns itself); see the module
    docstring. Immutable: ``trial`` returns another."""

    def __init__(
        self, point: Any, paths: Sequence[str], owners: Mapping[str, str] | None = None
    ) -> None:
        self.point = point
        self._paths = tuple(paths)
        owned = dict(owners or {})
        named = {path: owned.get(path, path) for path in self._paths}
        channels = [path for path in self._paths if isinstance(_member(point, path), Channel)]
        ranked = [path for path in self._paths if path not in channels] + channels
        self._owner = MappingProxyType(named)
        self._path = MappingProxyType({owner: path for path, owner in named.items()})
        self._members = tuple(named[path] for path in ranked)
        self._channels = tuple(named[path] for path in channels)
        self._rank = {name: index for index, name in enumerate(self._members)}
        self._infos: dict[str, inspection.DecisionInfo[object]] | None = None
        self._reads: dict[str, Any] = {}

    def _again(self, point: Any) -> Rooted:
        made = Rooted.__new__(Rooted)
        made.__dict__.update(self.__dict__)
        made.point = point
        made._reads = {}
        return made

    def _read(self, question: str, answer: Any) -> Any:
        """``answer()``, read once of this root's point (it is immutable)."""
        if question not in self._reads:
            self._reads[question] = answer()
        return self._reads[question]

    # -- keys -------------------------------------------------------------------------------

    def key(self, root_key: str) -> Key:
        """A root key as a pair: its longest member path's owner, and the key below it."""
        path = member_of(self._owner, root_key)
        if path is None:
            return (root_key, "")
        return (self._owner[path], root_key[len(path) + 1 :])

    def root_key(self, key: Key) -> str:
        """A pair as the root key it names."""
        owner, below = key
        path = self._path.get(owner, owner)
        return f"{path}.{below}" if below else path

    def _pairs(self, found: Iterable[tuple[str, T]]) -> list[tuple[Key, T]]:
        """``found`` by root key as pairs, ranked by member (stable within one)."""
        paired = [(self.key(root_key), value) for root_key, value in found]
        return sorted(paired, key=lambda item: self._rank.get(item[0][0], len(self._rank)))

    def _wanted(self, names: Iterable[str] | None) -> set[str] | None:
        return None if names is None else set(names)

    # -- the protocol -----------------------------------------------------------------------

    @property
    def members(self) -> tuple[str, ...]:
        return self._members

    @property
    def channels(self) -> tuple[str, ...]:
        return self._channels

    def member(self, name: str) -> Any:
        return _member(self.point, self._path.get(name, name))

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]:
        member = self.member(name)
        exports = type(member).exports
        if key not in exports:
            return Inapplicable()
        found: QueryResult[T] = member.query(exports[key])
        return found

    def waits(self, name: str, answer: object) -> tuple[Key, ...]:
        if not isinstance(answer, Unresolved):
            return ()
        return tuple(
            dict.fromkeys(
                self.key(item.owner)
                for item in answer.findings
                if item.code == "decision-unassigned"
            )
        )

    def _decisions(self) -> dict[str, inspection.DecisionInfo[object]]:
        if self._infos is None:
            self._infos = {item.key: item for item in inspection.decisions(self.point)}
        return self._infos

    def decisions(self) -> dict[Key, inspection.DecisionInfo[object]]:
        return dict(self._pairs(self._decisions().items()))

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        return self._decisions().get(self.root_key(key))

    def viable(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.Viable], ...]:
        wanted = self._wanted(names)
        found = self._read(
            "viable",
            lambda: self._pairs((item.key, item) for item in inspection.viable(self.point)),
        )
        return tuple(item for item in found if wanted is None or item[0][0] in wanted)

    def open(  # noqa: A003 - the inspection's name
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.Open], ...]:
        wanted = self._wanted(names)
        found = self._read(
            "open", lambda: self._pairs((item.key, item) for item in inspection.open(self.point))
        )
        return tuple(item for item in found if wanted is None or item[0][0] in wanted)

    def forced(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        wanted = self._wanted(names)
        found = self._read(
            "forced",
            lambda: self._pairs((item.key, item.value) for item in inspection.forced(self.point)),
        )
        return {key: value for key, value in found if wanted is None or key[0] in wanted}

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        wanted = self._wanted(names)
        found = self._read("chosen", lambda: self._pairs(chosen(self.point).items()))
        return {key: value for key, value in found if wanted is None or key[0] in wanted}

    def admission(self, name: str) -> tuple[Finding, ...]:
        admitted = inspection.admission(self.member(name))
        return admitted.findings if isinstance(admitted, Rejected) else ()

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Rooted]:
        """The engine's trial on the whole root, atomic; with ``check``, then no Decision
        left without a viable case and no member a key of the change belongs to refusing
        itself (what the seam checked of a root before)."""
        infos = self._decisions()
        handles = {key: infos.get(self.root_key(key)) for key in (*changes, *clear)}
        unknown = [key for key, info in handles.items() if info is None]
        if unknown:
            return Trial(self, {}, {key: _refused(key, "not-a-decision") for key in unknown})
        try:
            report = self.point.try_with_choices(
                {handles[key].reference: value for key, value in changes.items()},  # type: ignore[union-attr]
                *(self.point.field(handles[key].reference).clear() for key in clear),  # type: ignore[union-attr]
            )
        except (RequestError, ConfigurationError) as error:
            return Trial(
                self,
                {},
                {
                    key: (
                        Finding(FindingKind.REJECTION, "request-refused", dotted(key), str(error)),
                    )
                    for key in (*changes, *clear)
                },
            )
        if not report.accepted:
            why: dict[Key, tuple[Finding, ...]] = {}
            for outcome in report.outcomes:
                if outcome.status != "refused":
                    continue
                result = outcome.result
                found = (
                    result.findings
                    if isinstance(result, (Inapplicable, Rejected, Unresolved))
                    else ()
                )
                key = self.key(outcome.owner)
                why[key] = (
                    _refused(key, "inapplicable")
                    if isinstance(result, Inapplicable) or not found
                    else found
                )
            return Trial(
                self, {}, why or {key: _refused(key, "refused-together") for key in changes}
            )
        reached = self._again(report.instance)
        if check:
            refused: dict[Key, tuple[Finding, ...]] = {}
            # Read once of the point reached (``viable``), which settling reads next.
            for key, item in reached.viable():
                if not item.cases:
                    detail = "; ".join(f"{case}: {why}" for case, why in item.refused.items())
                    refused[key] = (
                        Finding(
                            FindingKind.REJECTION,
                            "decision-no-viable-case",
                            dotted(key),
                            f"no case is viable: {detail}",
                        ),
                    )
            for name in dict.fromkeys(key[0] for key in changes):
                if name in self._path:
                    admitted = reached.admission(name)
                    if admitted:
                        refused[(name, "")] = admitted
            if refused:
                return Trial(self, {}, refused)
        return Trial(reached, {}, {})

    def own_resources(self) -> QueryResult[Resources] | None:
        """The root's own ``RESOURCES``: a shell root's is its partition's, its ends' and
        its static region's sum."""
        found: QueryResult[Resources] = self.point.query(type(self.point).exports[RESOURCES])
        return found


def _refused(key: Key, code: str) -> tuple[Finding, ...]:
    messages = {
        "not-a-decision": "not a Decision of this root",
        "inapplicable": "inapplicable",
        "refused-together": "refused together",
    }
    return (Finding(FindingKind.REJECTION, code, dotted(key), messages[code]),)


__all__ = ["Rooted"]
