# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""E0 probe: the refined ``Decision`` over kernels, ``required()`` and ``settle``.

Built on today's engine (``finn.core.space``) without changing it. A refined
Decision is expanded, at class-creation time, into the Decision over nodes the
engine already compiles (``node_choice``): each entry becomes a fresh node, the
shared bindings are merged into every entry under the strict rules, and an
``optional`` Decision gains a ``None`` candidate. The class-body face is a
``RefinedChoice``, a ``NodeChoice`` that checks direct reads and adds
qualified reads (``compute["packed"]``).

What cannot be expressed on today's engine (narrowing and pinning a Decision
over nodes by key) is prototyped in ``spike.py``, which patches two engine
functions for the duration of a ``with`` block only.

Nothing here is production code; E1 moves the behaviour into the engine.
"""

# The probe reaches engine internals on purpose: it prototypes engine behaviour.
# ruff: noqa: SLF001

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from finn.core.space import Available, QueryResult, Space
from finn.core.space import Decision as EngineDecision
from finn.core.space._configuration import SpaceMeta
from finn.core.space._nodes import (
    NodeChoice,
    NodeDecision,
    check_supplier,
    family_members,
    node_choice,
    node_record,
    path_proxy,
    slot_kind,
)
from finn.core.space.collection import collect_space
from finn.core.space.declarations import (
    ChoiceMemberRef,
    Constraint,
    ConstraintGroup,
    Declaration,
    View,
    _guard,
    at,
    declared_path,
    local_name,
)
from finn.core.space.errors import DefinitionError
from finn.core.space.occurrence import _attach, selected_candidate, state
from finn.core.space.references import DecisionHandle

S = TypeVar("S", bound=Space)

# Keywords of the Decision itself. ``when`` is also reserved as a member name
# by the engine (collection._RESERVED), so no Param can be called ``when``.
OWN_KEYWORDS = frozenset({"when", "optional"})
# Arguments of the scalar Decision form. A shared binding with one of these
# names is refused: it reads as the scalar form (Q1 in REPORT.md).
RESERVED_SHARED = frozenset({"values", "domain", "semantics", "name"})
NONE_CASE = "none"

_HERE = __name__
_ENGINE = "finn.core.space"


def author_origin() -> str | None:
    """``file:line`` of the nearest caller outside the engine and this probe."""
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        inside = module == _HERE or module == _ENGINE or module.startswith(_ENGINE + ".")
        if not inside:
            return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        frame = frame.f_back  # type: ignore[assignment]
    return None


# -- required() -----------------------------------------------------------------------


class Required:
    """A member every subclass must define (``schedule: Schedule = required()``).

    It is not a Space declaration: the engine ignores it. A subclass defines
    the member with any attribute (a Param, a derived value, a view, a method,
    a plain class attribute); a class whose effective value is still this
    placeholder cannot be placed.
    """

    __slots__ = ("name", "owner")

    def __init__(self) -> None:
        self.name: str | None = None
        self.owner: type[object] | None = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.owner, self.name = owner, name

    def __repr__(self) -> str:
        where = f"{self.owner.__qualname__}." if self.owner is not None else ""
        return f"required({where}{self.name})"


def required() -> Any:
    """Declare a member every subclass must define; typed as its annotation."""
    return Required()


def unmet_required(family: type[object]) -> tuple[str, ...]:
    """Names whose effective value (first in the MRO) is still ``required()``."""
    names: dict[str, None] = {}
    for base in family.__mro__:
        for name, value in vars(base).items():
            if isinstance(value, Required):
                names.setdefault(name)
    unmet = []
    for name in names:
        for base in family.__mro__:
            if name in vars(base):
                if isinstance(vars(base)[name], Required):
                    unmet.append(name)
                break
    return tuple(unmet)


def _unmet_message(family: type[object], unmet: tuple[str, ...], where: str) -> str:
    owners = []
    for name in unmet:
        for base in family.__mro__:
            if isinstance(vars(base).get(name), Required):
                owners.append(f"{base.__qualname__}.{name}")
                break
    return (
        f"{family.__qualname__}{where} leaves required members unmet: {', '.join(owners)}; "
        "a class with an unmet required() member cannot be placed (define them in a subclass)"
    )


class RequiredMeta(SpaceMeta):
    """E0 stand-in for the engine check: calling a family with an unmet member refuses.

    In E1 this check moves into ``declare_node`` (every placement goes through
    a call), so every family gets it, not only those using this metaclass.
    """

    def __call__(cls, *args: object, **keywords: object) -> Any:
        origin = author_origin()
        unmet = unmet_required(cls)
        if unmet:
            raise DefinitionError(_unmet_message(cls, unmet, at(origin)))
        node = super().__call__(*args, **keywords)
        record = node_record(node)
        if record is not None:  # the author's line, not this metaclass's
            record.origin = origin
            for name, (supplier, _) in record.overrides.items():
                record.overrides[name] = (supplier, origin)
        return node


# -- the class-body face --------------------------------------------------------------


class RefinedChoice(NodeChoice):
    """``compute``: direct reads of common members, qualified reads of the rest.

    - ``compute.y`` is a reference to the selected candidate's ``y``. It is
      refused at class creation unless every node candidate declares ``y``,
      with compatible value semantics where they are known before linking.
    - ``compute["packed"]`` is the candidate node (a path through the
      Decision): ``compute["packed"].narrow_weights`` is that candidate's member,
      inapplicable when another candidate is selected; an enclosing body
      assigns through it (``mm.compute["packed"].pe = 4``).
    """

    __slots__ = ()

    def __get__(self, instance: object, owner: type[object] | None = None) -> Any:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return RefinedChoice((*path, self._space_decision()))
        return selected_candidate(cast(Space, instance), self._space_decision())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        decision = self._space_decision()
        having: dict[str, Declaration] = {}
        lacking: list[str] = []
        for case, record in decision.candidates.items():
            if record is None:
                continue  # the None candidate reads as inapplicable, not as a lack
            members, _ = family_members(record.family)
            if name in members:
                having[case] = members[name]
            else:
                lacking.append(case)
        if not having:
            raise AttributeError(name)  # getattr probes stay ordinary
        if lacking:
            raise DefinitionError(
                f"{_describe(self)}.{name}: a direct read needs a member every candidate "
                f"declares; candidates {lacking} do not declare {name!r}. Read it qualified, "
                f'as {_name(self)}["{next(iter(having))}"].{name}'
            )
        _check_common_semantics(self, name, decision, having)
        return ChoiceMemberRef(self._space_path, name)

    def __getitem__(self, case: str) -> Any:
        decision = self._space_decision()
        if case not in decision.candidates:
            raise DefinitionError(
                f"{_describe(self)}: no candidate {case!r}; its candidates are "
                f"{list(decision.candidates)}"
            )
        record = decision.candidates[case]
        if record is None:
            raise DefinitionError(
                f"{_describe(self)}: candidate {case!r} is None; it has no members"
            )
        return path_proxy(record.family, (*self._space_path[:-1], record))


def _name(choice: NodeChoice) -> str:
    return str(choice._space_decision().name or "decision")


def _describe(choice: NodeChoice) -> str:
    decision = choice._space_decision()
    owner = decision.owner.__qualname__ + "." if decision.owner is not None else ""
    return f"{owner}{decision.name or 'Decision'}{at(decision.origin)}"


def _semantics_of(family: type[Space], declaration: Declaration) -> object | None:
    try:
        known = collect_space(family).semantics.get(declaration)
    except DefinitionError:
        return None
    return known if known is not None else getattr(declaration, "semantics", None)


def _check_common_semantics(
    choice: NodeChoice, name: str, decision: NodeDecision, having: Mapping[str, Declaration]
) -> None:
    """Refuse a common member whose value types differ where both are known now.

    Semantics known only after linking (a view over a reference) are left to
    the engine's existing link-time check of the selection node.
    """
    known: list[tuple[str, Any]] = []
    for case, declaration in having.items():
        record = decision.candidates[case]
        assert record is not None
        semantics = _semantics_of(record.family, declaration)
        if semantics is not None:
            known.append((case, semantics))
    for case, semantics in known[1:]:
        first_case, first = known[0]
        if not first.is_compatible_with(semantics):
            raise DefinitionError(
                f"{_describe(choice)}.{name}: candidates declare {name!r} with different "
                f"value types ({first_case}: {first.name}, {case}: {semantics.name}); read "
                "each qualified"
            )


# -- the refined Decision -------------------------------------------------------------


Entry = type[Space] | Space


def Decision(  # noqa: N802 - the probe's stand-in for the engine class
    entries: Mapping[str, Entry] | None = None,
    /,
    *,
    optional: bool | str = False,
    when: object = None,
    **keywords: object,
) -> Any:
    """``Decision({"a": A, "b": B(own=...)}, shared=..., optional=..., when=...)``.

    Without positional entries this is today's engine ``Decision`` (scalar,
    or the ``values={...}`` form over nodes). With entries:

    - an entry is a Space class, or a call on one carrying bindings only that
      candidate takes; it becomes a fresh node, placed when selected;
    - every other keyword is a shared binding merged into every entry. Each
      must be declared by every candidate (else one error naming those that
      lack it); a shared binding also written on an entry is a double
      assignment; names of the scalar form's arguments are refused;
    - ``optional=True`` adds a ``None`` candidate keyed ``"none"``, first (a
      string names that key instead: a probe extension, see REPORT.md);
    - a class with an unmet ``required()`` member is refused as an entry.
    """
    if entries is None:
        if optional is not False:
            raise DefinitionError("optional= applies to a Decision over candidate entries")
        if when is not None:
            keywords["when"] = when
        return EngineDecision(**keywords)  # type: ignore[call-overload]
    origin = author_origin()
    where = at(origin)
    if not isinstance(entries, Mapping) or not entries:
        raise DefinitionError(f"Decision{where}: entries must be a nonempty mapping of keys")
    none_key = _none_key(optional, where)
    reserved = sorted(keywords.keys() & RESERVED_SHARED)
    if reserved:
        raise DefinitionError(
            f"Decision{where}: shared bindings {reserved} are named like the Decision's own "
            "arguments; rename the Param (values -> contents), or write the binding on each "
            "entry that takes it"
        )
    families: dict[str, type[Space]] = {}
    calls: dict[str, Any] = {}
    for key, entry in entries.items():
        local_name(key, "candidate key")
        if key == none_key:
            raise DefinitionError(f"Decision{where}: {key!r} is the key of the None candidate")
        if isinstance(entry, type) and issubclass(entry, Space):
            unmet = unmet_required(entry)
            if unmet:
                raise DefinitionError(
                    _unmet_message(entry, unmet, f" (candidate {key!r} of a Decision{where})")
                )
            families[key] = entry
            continue
        record = node_record(entry)
        if record is None:
            raise DefinitionError(
                f"Decision{where}: candidate {key!r} must be a kernel class or a call on one"
            )
        families[key] = record.family
        calls[key] = entry
    _check_shared(families, calls, keywords, where)
    values: dict[str, Any] = {} if none_key is None else {none_key: None}
    for key, family in families.items():
        node = calls[key] if key in calls else family()
        record = node_record(node)
        assert record is not None
        if key not in calls:
            record.origin = origin
        members, _ = family_members(family)
        for name, value in keywords.items():
            label = f"{family.__qualname__}.{name} (shared binding of candidate {key!r}{where})"
            supplier = check_supplier(family, name, members[name], value, label)
            record.overrides[name] = (supplier, origin)
        values[key] = node
    proxy = node_choice(values, when=_guard(when))
    decision = proxy._space_decision()
    object.__setattr__(decision, "origin", origin)
    refined = RefinedChoice((decision,))
    decision.proxy = refined
    return refined


def _none_key(optional: bool | str, where: str) -> str | None:
    if optional is False:
        return None
    if optional is True:
        return NONE_CASE
    if isinstance(optional, str):
        return local_name(optional, "None candidate key")
    raise DefinitionError(f"Decision{where}: optional= takes True or the None candidate's key")


def _check_shared(
    families: Mapping[str, type[Space]],
    calls: Mapping[str, Any],
    shared: Mapping[str, object],
    where: str,
) -> None:
    """Every candidate declares every shared binding; no entry writes one again."""
    problems: list[str] = []
    for name in shared:
        lacking = []
        for key, family in families.items():
            members, _ = family_members(family)
            member = members.get(name)
            if member is None or slot_kind(member) == "behaviour":
                lacking.append(f"{key} ({family.__qualname__})")
        if lacking:
            problems.append(f"shared binding {name!r} is not declared by candidates {lacking}")
    if problems:
        raise DefinitionError(
            f"Decision{where}: "
            + "; ".join(problems)
            + ". A shared binding goes to every candidate: write it on the entries that "
            "take it instead"
        )
    for key, node in calls.items():
        record = node_record(node)
        assert record is not None
        for name in shared:
            earlier = record.overrides.get(name)
            if earlier is not None:
                raise DefinitionError(
                    f"Decision{where}: candidate {key!r} already binds {name} (at {earlier[1]}), "
                    "and the shared bindings bind it again; a body sets a member once: remove "
                    "it from the entry or from the shared bindings"
                )


# -- settle ---------------------------------------------------------------------------


Admission = Callable[[Space], "QueryResult[object] | None"]


def member_admission(name: str = "admission") -> Admission:
    """A candidate's admission is its member ``name`` (a constraint, group or view)."""

    def admission(candidate: Space) -> QueryResult[object] | None:
        members, _ = family_members(type(candidate))
        declaration = members.get(name)
        if declaration is None:
            return None
        if isinstance(declaration, (Constraint, ConstraintGroup)):
            return candidate.inspect(declaration).result
        if isinstance(declaration, View):
            return candidate.query(declaration)
        raise DefinitionError(f"{type(candidate).__qualname__}.{name} is not an admission")

    return admission


def admission_by_family(table: Mapping[type[Space], object]) -> Admission:
    """Today's kernels name their admission differently: map family -> member."""

    def admission(candidate: Space) -> QueryResult[object] | None:
        for family, declaration in table.items():
            if isinstance(candidate, family):
                if isinstance(declaration, (Constraint, ConstraintGroup)):
                    return candidate.inspect(declaration).result
                return candidate.query(declaration)
        return None

    return admission


@dataclass(frozen=True)
class Settlement(Generic[S]):
    """What ``settle`` committed, and the Decisions it left open with their compatible cases."""

    point: S
    committed: Mapping[str, str]
    open: Mapping[str, tuple[str, ...]]


def compatible_cases(point: S, key: str, admission: Admission) -> tuple[str, ...]:
    """The cases of the Decision over nodes keyed ``key`` whose candidate is admitted."""
    snapshot = state(point)
    linked = snapshot.linked
    (choice,) = [item for item in linked.choices if item.key == key]
    handle = DecisionHandle[str](linked, choice.selector)
    found = point.field(handle).candidates()
    if not isinstance(found, Available):
        return ()
    result: list[str] = []
    for case in cast(tuple[str, ...], found.value):
        report = point.try_with_choices({handle: case})
        if not report.accepted:
            continue
        scope = dict(choice.cases)[case]
        if scope is None:
            result.append(case)
            continue
        verdict = admission(_attach(state(report.instance), scope))
        if verdict is None or isinstance(verdict, Available):
            result.append(case)
    return tuple(result)


def settle(point: S, *, admission: Admission | None = None) -> Settlement[S]:
    """Commit every applicable, undecided Decision over nodes with exactly one compatible case.

    Compatible: committing the case is accepted and the candidate's admission
    (``admission``; by default its ``admission`` member, none meaning
    admitted) is available. Repeats until nothing changes, since a commitment
    can make another Decision applicable (an adapter follows its core). Several
    compatible cases are a design choice and stay open; so does none.
    Scalar Decisions are not settled.
    """
    check = admission or member_admission()
    committed: dict[str, str] = {}
    while True:
        snapshot = state(point)
        linked = snapshot.linked
        open_: dict[str, tuple[str, ...]] = {}
        progressed = False
        for choice in sorted(linked.choices, key=lambda item: item.key):
            if linked.nodes[choice.selector].kind != "decision":
                continue  # pinned by an enclosing body (spike.py)
            handle = DecisionHandle[str](linked, choice.selector)
            current = point.field(handle).state
            if not isinstance(current, Available) or current.value.status == "committed":
                continue  # inapplicable, unresolved guard, or already decided
            cases = compatible_cases(point, choice.key, check)
            if len(cases) == 1:
                point = point.with_choices({handle: cases[0]})
                committed[choice.key] = cases[0]
                progressed = True
                break
            open_[choice.key] = cases
        if not progressed:
            return Settlement(point, committed, open_)


__all__ = [
    "Decision",
    "RefinedChoice",
    "Required",
    "RequiredMeta",
    "Settlement",
    "admission_by_family",
    "compatible_cases",
    "member_admission",
    "required",
    "settle",
    "unmet_required",
]
