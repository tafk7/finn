# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A choice's policy, stated by FINN's compiler, and its application to a design space.

Every choice has one of three policies, which say how the compiler gives it a value once,
where it first sets a design space's choices (a node's kernel at conversion, a channel at
the cut):

- **fixed** (``Fixed(value)``): the author's one value (a selector's case by its key), with
  no claim that it is best;
- **first valid** (``FirstValid(order=None)``): the first of its listed values that the
  design space's own rules do not refuse, given its facts and the choices already set: in
  its list's order, or in the order a ranking view of the Decision's class gives
  (``order``, ``MemStreamKernel.ram_preference``), read out of the design point;
- **required** (``Required()``): no policy gives it a value, and a strategy sets it (a
  FIFO's depth).

**Who states them** (``Owner``): the owner of a design space, never the engine and never
the kernel. A KernelOp states its kernel's (``KernelOp.policies``), the channel its own
(``Channel.policies``). A policy is keyed by a class-level reference to a Decision
(``MemStreamKernel.ram_style``) and applies to every placement of that class below the
owner's root. A choice it does not list is first valid, in listed order. ``stated`` checks
the mapping where it is made (a KernelOp's when its class is made, the channel's when its
class is made): each key a Decision of a class the root reaches
(``default-policy-undeclared``), each value a policy (``default-policy-invalid``), a fixed
value of the Decision's type and in its domain where that can be known without facts
(``default-policy-value``), an order a view of the Decision's class
(``default-policy-order``), and a choice known by membership only (no list to take a first
value from) required or fixed (``default-policy-unlisted``).

**One reader**: ``policy(owner, item)``, a Decision's policy under its owner. Conversion,
the cut and the seam read it, and nothing else: no reader asks the engine for a policy.

The engine knows nothing of it: a policy is applied once, by ``defaulted``, and the values
it sets are ordinary choices, unmarked. Choices are taken in the flow order
(``inspection.states``, the design space's lists: a choice its facts alone rule out is in
none): each after every choice its presence, its list or its Space's facts read, an
enclosing selector before the choices of its cases, ties in declaration order. Only those
in play are defaulted: a choice under a case not selected, or whose presence waits on a
choice still open, is left as it is. A value is valid when setting it alone is accepted
(``inspection.viable``): a rule that waits on a choice still open does not refuse it. A
partition's policies are applied in one pass over its design space's root, in the root's
flow order (which interleaves its members), each choice's policy read from its member's
owner.

**A ranking that cannot be read.** A ranking is a view, which refuses while its kernel is
refused and waits while it waits. First valid then takes the listed order and says so
(``ranked``: ``choice-order-unknown``, a limitation owned by the choice, naming the view
and why), so a fallback is never silent: ``defaulted`` returns it beside the point
(``Defaulted.fallbacks``), and conversion, the cut and the seam report it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import ClassVar, Generic, Protocol, TypeAlias, TypeVar

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    BrokenRule,
    DefinitionError,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    View,
    inspection,
)
from finn.core.space.references import ValueHandle

S = TypeVar("S", bound=Space)

_OWNER = "finn.kernels.policies"


@dataclass(frozen=True)
class Fixed:
    """A fixed policy: the author's one ``value`` for a choice (a selector's case by its
    key), set once, with no claim that it is best."""

    value: bool | int | str


class Ranking(Protocol):
    """What a first valid policy's ``order`` names: a view of the Decision's class, by
    class access (``MemStreamKernel.ram_preference``, a ``finn.core.space.View``), its
    declaring class and name fixed once its class is made."""

    @property
    def name(self) -> str | None: ...

    @property
    def owner(self) -> type[object] | None: ...


@dataclass(frozen=True)
class FirstValid:
    """A first valid policy: the first listed value the rules do not refuse, in listed
    order, or in the order ``order`` gives: a view of the Decision's class
    (``MemStreamKernel.ram_preference``, by class access), read out of the design point.
    Where the view cannot answer, listed order, named (``ranked``)."""

    order: Ranking | None = None


@dataclass(frozen=True)
class Required:
    """No policy gives the choice a value: a strategy sets it (a FIFO's depth)."""


Policy: TypeAlias = Fixed | FirstValid | Required
"""A choice's policy: fixed, first valid, or required."""

FIRST_VALID = FirstValid()
"""The policy of every choice its owner does not list: first valid, in listed order."""


class Owner(Protocol):
    """What states a design space's policies: a KernelOp class, for its kernel's choices,
    or the channel class (``finn.kernels.channels.Channel``), for a channel's. ``policies``
    maps class-level Decision references to policies (``stated``)."""

    @property
    def policies(self) -> Mapping[object, Policy]: ...


class Unstated:
    """The owner of a design space no KernelOp places (a network built by hand): it
    states no policy, so every choice is first valid, in listed order."""

    policies: ClassVar[Mapping[object, Policy]] = MappingProxyType({})


def policy_name(policy: Policy) -> str:
    """A policy as a report names it: ``fixed``, ``first valid`` or ``none`` (required)."""
    if isinstance(policy, Fixed):
        return "fixed"
    return "none" if isinstance(policy, Required) else "first valid"


def _local(item: inspection.DecisionInfo[object]) -> str:
    """``item``'s key in the class that declares it (``DecisionInfo.space_type``)."""
    return item.key[len(item.scope) + 1 :] if item.scope else item.key


def declared(item: inspection.DecisionInfo[object]) -> object:
    """The class-level reference of ``item``'s Decision (``MemStreamKernel.ram_style``),
    as its policy is keyed; None where it has no declaring class."""
    if item.space_type is None:
        return None
    return getattr(item.space_type, _local(item), None)


def _declaration(item: inspection.DecisionInfo[object]) -> str:
    """``item``'s Decision as its declaring class names it (``MemStreamKernel.ram_style``)."""
    owner = getattr(item.space_type, "__qualname__", "")
    return f"{owner}.{_local(item)}" if owner else item.key


def policy(owner: Owner, item: inspection.DecisionInfo[object]) -> Policy:
    """The policy of ``item``'s Decision under ``owner``: the one it states for the
    Decision's class-level reference, or first valid, in listed order, where it states
    none. The one reader of a policy."""
    reference = declared(item)
    found = None if reference is None else owner.policies.get(reference)
    return FIRST_VALID if found is None else found


# -- the checks where policies are stated ----------------------------------------------------


def _refused(code: str, owner: str, message: str) -> DefinitionError:
    return DefinitionError.broken(code, _OWNER, message, declaration=f"{owner}.policies")


def _named(key: object) -> str:
    """A key of a policies mapping, as a message names it."""
    owner, name = getattr(key, "owner", None), getattr(key, "name", None)
    if isinstance(owner, type) and isinstance(name, str):
        return f"{owner.__qualname__}.{name}"
    return type(key).__qualname__


def _fixed_refusal(item: inspection.DecisionInfo[object], key: object, value: object) -> str:
    """Why ``value`` cannot be ``item``'s fixed value as far as is known without facts: not
    one of a selector's cases, not of the Decision's type, or outside a domain that reads
    nothing; empty where it may be."""
    if item.selector:
        return "" if value in item.cases else f"not one of its cases {list(item.cases)}"
    semantics = getattr(key, "semantics", None)
    if semantics is not None and not semantics.recognizes(value):
        return f"not a value of its type ({semantics.name})"
    domain = getattr(key, "domain", None)
    if domain is None or domain.dependencies:
        return ""  # its domain reads facts: the design space refuses the value, named
    if domain.candidates is not None:
        listed = domain.candidates()
        if isinstance(listed, Iterable) and value not in tuple(listed):
            return "outside its domain"
        return ""
    accepted = domain.accepts(candidate=value)
    return "outside its domain" if accepted is False else ""


def stated(owner: str, root: type[Space], policies: object) -> Mapping[object, Policy]:
    """``policies``, the policies ``owner`` states for the choices of ``root`` and every
    class it places, checked (the module docstring), as an immutable mapping. Refused
    (``DefinitionError``) where a key is no Decision of a class ``root`` reaches, a value
    is no policy, a fixed value is not one the Decision can take, an order is no view of the
    Decision's class, or a choice known by membership only is neither required nor fixed."""
    if not isinstance(policies, Mapping) or not all(
        isinstance(each, (Fixed, FirstValid, Required)) for each in policies.values()
    ):
        raise _refused(
            "default-policy-invalid",
            owner,
            f"{owner}.policies maps class-level Decision references (Kernel.choice) to "
            "Fixed(value), FirstValid(order=...) or Required(); a choice it does not list "
            "is first valid",
        )
    placed: dict[object, list[inspection.DecisionInfo[object]]] = {}
    for item in inspection.decisions(root):
        reference = declared(item)
        if reference is not None:
            placed.setdefault(reference, []).append(item)
    for key, each in policies.items():
        items = placed.get(key)
        if items is None:
            raise _refused(
                "default-policy-undeclared",
                owner,
                f"{owner}.policies names {_named(key)}, which is no Decision of a class "
                f"{root.__qualname__} reaches",
            )
        name = _declaration(items[0])
        if isinstance(each, Fixed):
            why = _fixed_refusal(items[0], key, each.value)
            if why:
                raise _refused(
                    "default-policy-value",
                    owner,
                    f"{owner}.policies fixes {name} to {each.value!r}: {why}",
                )
        if isinstance(each, FirstValid) and each.order is not None:
            view = each.order
            if not (
                isinstance(view, View)
                and isinstance(view.owner, type)
                and all(
                    item.space_type is not None
                    and issubclass(item.space_type, view.owner)
                    and getattr(item.space_type, str(view.name), None) is view
                    for item in items
                )
            ):
                raise _refused(
                    "default-policy-order",
                    owner,
                    f"{owner}.policies orders {name} by {_named(view)}, which is no view of "
                    f"{items[0].space_type.__qualname__ if items[0].space_type else name}",
                )
    for reference, items in placed.items():
        if items[0].listed or isinstance(policies.get(reference), (Fixed, Required)):
            continue
        raise _refused(
            "default-policy-unlisted",
            owner,
            f"{_declaration(items[0])} is known by membership only, so it has no first valid "
            f"value: state it Required() or Fixed(value) in {owner}.policies",
        )
    return MappingProxyType(dict(policies))


# -- rankings ----------------------------------------------------------------------------------


@cache
def _class_views(space_type: type[Space]) -> Mapping[str, ValueHandle[object]]:
    """Every view below ``space_type``, by its key, read once per class."""
    return _view_handles(inspection.members(space_type))


def _view_handles(members: Iterable[inspection.NodeInfo]) -> Mapping[str, ValueHandle[object]]:
    return {member.key: member.reference for member in members if member.kind == "view"}


def _views(point: Space) -> Mapping[str, ValueHandle[object]]:
    """Every view below ``point``'s scope, by its key, as its Decisions are keyed: a root's
    (its class's, read once), or a Space placed in a root's (a partition's member: its keys
    are its paths there, and its views the root's model's)."""
    if point.root is point:
        return _class_views(type(point))
    return _view_handles(inspection.members(point))


def _view_name(view: object) -> str:
    return _named(view)


def _fallback(key: str, view: object, answer: QueryResult[object]) -> Finding:
    """The finding of a ranking that cannot be read (``choice-order-unknown``), owned by
    the choice's ``key``: listed order taken, the view and why, its findings the causes."""
    if isinstance(answer, Rejected):
        why = "it is refused"
    elif isinstance(answer, Unresolved):
        why = "it waits on a value not known"
    else:
        why = "it does not apply"
    named = _view_name(view)
    return Finding(
        FindingKind.LIMITATION,
        "choice-order-unknown",
        key,
        f"first valid in listed order: its order is not known ({named}: {why})",
        (("policy", "first valid"), ("order", named)),
        tuple(getattr(answer, "findings", ())),
    )


Values = tuple[inspection.Possible | inspection.Refused, ...]


def ranked(
    point: Space,
    owner: Owner,
    item: inspection.DecisionInfo[object],
    values: Values,
    *,
    key: str | None = None,
) -> tuple[Values, Finding | None]:
    """A choice's listed ``values`` (``inspection.viable``'s, in its list's order) in its
    policy's order: a ``FirstValid(order=view)``'s, the view read out of ``point`` at the
    Decision's placement; listed order otherwise. Where the view cannot answer, listed
    order, and the fallback named (``choice-order-unknown``, owned by ``key``, the choice's
    key by default)."""
    found = policy(owner, item)
    if not isinstance(found, FirstValid) or found.order is None:
        return values, None
    view = found.order
    name = str(getattr(view, "name", ""))
    handle = _views(point).get(f"{item.scope}.{name}" if item.scope else name)
    if handle is None:
        return values, None  # unreachable: ``stated`` checked the view is the class's
    answer = point.query(handle)
    if not isinstance(answer, Available) or not isinstance(answer.value, tuple):
        return values, _fallback(key or item.key, view, answer)
    rest = list(values)
    order: list[inspection.Possible | inspection.Refused] = []
    for value in answer.value:
        match = next((each for each in rest if each.value == value), None)
        if match is not None:
            order.append(match)
            rest.remove(match)
    return (*order, *rest), None


# -- applying them once ------------------------------------------------------------------------


class Placed(Protocol):
    """Where the choices of a design space's root belong, for ``defaulted``: the members it
    is made of, each choice's member (by the choice's key in the root), each member's owner
    of policies (``Owner``) and why a member is refused on its own facts, read of the root as
    it stands. A kernel alone is one member (``Alone``); a partition's are its kernels and
    channels (``finn.kernels.partition.Partition.defaulted``)."""

    @property
    def members(self) -> tuple[str, ...]: ...

    def member(self, key: str) -> str: ...

    def owner(self, member: str) -> Owner: ...

    def refusal(self, point: Space, member: str) -> tuple[Finding, ...]: ...


class Alone:
    """One kernel's design space, whole, as ``Placed``: a single member, ``""``, whose
    owner is ``owner`` (the KernelOp, or the channel class for a channel alone: conversion's
    stream check) and whose own validity is the space's."""

    def __init__(self, owner: Owner) -> None:
        self._owner = owner

    @property
    def members(self) -> tuple[str, ...]:
        return ("",)

    def member(self, key: str) -> str:
        return ""

    def owner(self, member: str) -> Owner:
        return self._owner

    def refusal(self, point: Space, member: str) -> tuple[Finding, ...]:
        result = inspection.validity(point).result
        return result.findings if isinstance(result, Rejected) else ()


@dataclass
class Defaulted(Generic[S]):
    """A design space with its policies applied once (``defaulted``): its ``point``; each
    ranking that could not be read, its choice taken in listed order (``fallbacks``:
    ``choice-order-unknown``); each choice no value could be set for, left open (``unset``:
    ``choice-no-valid-value``); and each member refused on its own, none of its choices set
    (``refused``: its member's name and the findings of its validity). ``refused[""]`` is
    the point itself refused (``Alone``'s one member, ``""``: a kernel or a channel alone,
    whole), as against a member of a partition, keyed by its name. A result, not a value a
    Space holds: a plain dataclass."""

    point: S
    fallbacks: tuple[Finding, ...] = ()
    unset: tuple[Finding, ...] = ()
    refused: Mapping[str, tuple[Finding, ...]] = field(default_factory=dict)


def _flow_order(point: Space) -> tuple[str, ...]:
    """The keys of ``point``'s choices in the flow order (``inspection.states``): each
    after every choice its presence, its list or its Space's facts read, an enclosing
    selector before the choices of its cases, ties in declaration order."""
    return tuple(key for key, _ in inspection.states(point))


def _rules(broken: tuple[BrokenRule, ...]) -> str:
    """Each rule broken, at its origin, with the stored choices standing in its way."""
    return ", ".join(
        rule.rule + (f" (with {', '.join(rule.standing)})" if rule.standing else "")
        for rule in broken
    )


def _causes(broken: Iterable[BrokenRule]) -> tuple[Finding, ...]:
    """The findings of each rule broken, each once."""
    return tuple(dict.fromkeys(finding for rule in broken for finding in rule.result.findings))


def _fixed(point: S, reference: object, key: str, value: object) -> S | Finding:
    """``point`` with ``key`` set to its fixed ``value``, or why it cannot be: the value
    outside its domain, or each rule it breaks."""
    report = point.try_with_choices({reference: value})
    if report.accepted:
        return report.instance
    outside = tuple(
        finding
        for outcome in report.outcomes
        if isinstance(outcome.result, Rejected)
        for finding in outcome.result.findings
    )
    why = "; ".join(map(str, outside)) if outside else f"it breaks {_rules(report.broken)}"
    return unset(key, "fixed", f"{value!r} is refused: {why}", outside + _causes(report.broken))


def _first_valid(
    point: S, owner: Owner, item: inspection.DecisionInfo[object]
) -> tuple[S | Finding, Finding | None]:
    """``point`` with ``item``'s choice set to the first of its listed values, in its
    policy's order (``ranked``), that setting alone is accepted (``inspection.viable``), or
    why none is: each value's rules; and the ranking's fallback, where it could not be
    read."""
    key, name = item.key, policy_name(FIRST_VALID)
    if not item.listed:
        return unset(key, name, "it is known by membership only: no value is listed", ()), None
    (viable,) = inspection.viable(point, [item.reference])
    if viable.values is None:
        waits = ", ".join(viable.waits) or "an input of its domain"
        return unset(key, name, f"its values wait on {waits}", ()), None
    items, fallback = ranked(point, owner, item, viable.values)
    first = next((each for each in items if isinstance(each, inspection.Possible)), None)
    if first is not None:
        return point.with_choices({item.reference: first.value}), fallback
    refused = [each for each in viable.values if isinstance(each, inspection.Refused)]
    why = "; ".join(f"{each.value!r} breaks {_rules(each.broken)}" for each in refused)
    causes = _causes(rule for each in refused for rule in each.broken)
    return unset(key, name, why or "its domain lists no value", causes), fallback


def unset(key: str, policy: str, message: str, causes: tuple[Finding, ...]) -> Finding:
    """The finding for a choice no value can be set for (``choice-no-valid-value``), owned
    by its key: the policy (``policy_name``), why each value it tried is refused, and those
    refusals' findings as its causes."""
    return Finding(
        FindingKind.REJECTION,
        "choice-no-valid-value",
        key,
        f"no value can be set ({policy}): {message}",
        (("policy", policy),),
        causes,
    )


def defaulted(point: S, placed: Placed, *, kept: Collection[str] = ()) -> Defaulted[S]:
    """``point``, a design space's root (a kernel bound to its facts, or a partition's design
    space), with the policies of each choice's owner applied once (the module docstring), in
    one pass in the root's flow order: every choice in play and unset that is not required set
    to its fixed value or its first valid one, each policy read from its choice's member's
    owner (``placed``), with each ranking it could not read named (``Defaulted.fallbacks``).
    Choices it holds are kept, and so are the open choices ``kept`` names, by key: left open.

    **First valid** is the first value, in its policy's order, whose change is accepted:
    setting it alone leaves no rule of the root newly false (``inspection.viable``,
    whichever member the rule belongs to, and one that waits on a choice still open is not
    false). It is not that the root is valid after it: a rule false already does not refuse a
    value that leaves it so.

    **A choice no value can be set for** is left open and the pass goes on, reported
    (``Defaulted.unset``: ``choice-no-valid-value``, owned by its key). **A member refused on
    its own** (``Placed.refusal``) is skipped whole, none of its choices set, and reported
    (``Defaulted.refused``). Its refusal is read when the pass reaches the member's first
    choice, not before: what it reads of another member's choices has been set by then, so a
    rule waiting on a choice still open is not read as false."""
    members = set(placed.members)
    decisions = {item.key: item for item in inspection.decisions(point)}
    states = dict(inspection.states(point))
    fallbacks: list[Finding] = []
    unset: list[Finding] = []
    refused: dict[str, tuple[Finding, ...]] = {}
    read: set[str] = set()

    def refusal(member: str) -> None:
        read.add(member)
        found = placed.refusal(point, member)
        if found:
            refused[member] = found

    for key in _flow_order(point):
        member = placed.member(key)
        if member not in members:
            continue
        if member not in read:
            refusal(member)
        if member in refused:
            continue
        item = decisions[key]
        found = policy(placed.owner(member), item)
        state = states[key]
        if (
            isinstance(found, Required)
            or key in kept
            or not isinstance(state, inspection.Open)
            or state.waits
        ):
            continue
        if isinstance(found, Fixed):
            done = _fixed(point, item.reference, key, found.value)
        else:
            done, fallback = _first_valid(point, placed.owner(member), item)
            if fallback is not None:
                fallbacks.append(fallback)
        if isinstance(done, Finding):
            unset.append(done)
            continue
        point = done
        states = dict(inspection.states(point))
    for member in placed.members:
        if member not in read:
            refusal(member)
    return Defaulted(point, tuple(fallbacks), tuple(unset), refused)


__all__ = [
    "FIRST_VALID",
    "Alone",
    "Defaulted",
    "Fixed",
    "FirstValid",
    "Owner",
    "Placed",
    "Policy",
    "Ranking",
    "Required",
    "Unstated",
    "declared",
    "defaulted",
    "policy",
    "policy_name",
    "ranked",
    "stated",
    "unset",
]
