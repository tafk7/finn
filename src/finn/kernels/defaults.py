# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A choice's default policy, and its application to a kernel alone.

Every choice of a kernel has one of three policies, which say how the compiler gives it a
value once, where a node is converted:

- **fixed**: the author's one value (``Fixed``), with no claim that it is best;
- **first valid**: the first of its listed values, in its domain's order, that the
  kernel's own rules do not refuse, given its facts and the choices already set;
- **none**: no policy gives it a value, and a strategy sets it
  (``Decision(required=True)``, a FIFO's depth).

A kernel class states its policies beside its Decisions, in ``defaults``: its own
choices' keys (``reducer``, not ``compute.packed.reducer``), each a ``Fixed``. Every
choice it does not list, and that is not required, is first valid. A key it does not
declare, a fixed value for a required choice, and a choice known by membership only (no
list to take a first value from) that is neither required nor fixed are definition errors
(``policies``).

The engine knows nothing of it: a policy is applied once, by ``defaulted``, and the values
it sets are ordinary choices, unmarked. Choices are taken in the flow order
(``inspection.states``): each after every choice its presence, its list or its Space's
facts read, an enclosing selector before the choices of its cases, ties in declaration
order. Only those in play are defaulted: a choice under a case not selected, or whose
presence waits on a choice still open, is left as it is. A value is valid when setting it
alone is accepted (``inspection.viable``): a rule that waits on a choice still open does
not refuse it. First valid is the first in the domain's order (``ranked``: the listed
values ranked by the domain's order where it states one, as ``candidates`` lists them),
not in its list's.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    BrokenRule,
    DefinitionError,
    Inapplicable,
    Rejected,
    Space,
    Unresolved,
    inspection,
)

S = TypeVar("S", bound=Space)

_OWNER = "finn.kernels.defaults"


@dataclass(frozen=True)
class Fixed:
    """A fixed default: the author's one ``value`` for a choice (a selector's case by its
    key), set at conversion, with no claim that it is best."""

    value: bool | int | str


FIRST_VALID: Literal["first valid"] = "first valid"
"""The policy of every choice its class does not fix and that is not required."""

Policy: TypeAlias = Fixed | Literal["first valid"] | None
"""A choice's policy: fixed, first valid, or none (a required choice)."""


def _local(item: inspection.DecisionInfo[object]) -> str:
    """``item``'s key in the class that declares it (``DecisionInfo.space_type``)."""
    return item.key[len(item.scope) + 1 :] if item.scope else item.key


def _stated(space_type: type[Space] | None) -> Mapping[str, Fixed]:
    found: Mapping[str, Fixed] = getattr(space_type, "defaults", {})
    return found


def stated_defaults(space_type: type[Space], stated: object) -> None:
    """Check the form of a class's ``defaults`` where it is made: a mapping from names to
    ``Fixed`` (``policies`` checks the names once the class's Decisions are known)."""
    if not isinstance(stated, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Fixed) for key, value in stated.items()
    ):
        raise DefinitionError.broken(
            "default-policy-invalid",
            _OWNER,
            f"{space_type.__qualname__}.defaults maps the class's own choice keys to "
            "Fixed(value); a choice it does not list is first valid, a required one has none",
            declaration=space_type.__qualname__,
        )


def policies(space_type: type[Space]) -> dict[str, Policy]:
    """The policy of every choice of ``space_type`` (its own and its children's), by key:
    a ``Fixed``, ``FIRST_VALID``, or None for a required choice. Refused
    (``DefinitionError``) where a class among them fixes a key it does not declare, fixes
    a required choice, or leaves a choice known by membership only first valid."""
    found: dict[str, Policy] = {}
    checked: set[type[Space]] = set()
    for item in inspection.decisions(space_type):
        owner = item.space_type
        if owner is not None and owner not in checked:
            checked.add(owner)
            _check_keys(owner)
        fixed = _stated(owner).get(_local(item))
        declared = f"{getattr(owner, '__qualname__', space_type.__qualname__)}.{_local(item)}"
        if item.required:
            if fixed is not None:
                raise DefinitionError.broken(
                    "default-policy-required",
                    _OWNER,
                    f"{declared} is required, so no policy gives it a value: it cannot be fixed",
                    declaration=declared,
                )
            found[item.key] = None
        elif fixed is not None:
            found[item.key] = fixed
        elif not item.listed:
            raise DefinitionError.broken(
                "default-policy-unlisted",
                _OWNER,
                f"{declared} is known by membership only, so it has no first valid value: "
                "mark it required (Decision(..., required=True)) or fix its value",
                declaration=declared,
            )
        else:
            found[item.key] = FIRST_VALID
    return found


def _check_keys(space_type: type[Space]) -> None:
    """Every key ``space_type`` fixes is one of its own Decisions (``defaults``)."""
    stated = _stated(space_type)
    if not stated:
        return
    own = {item.key for item in inspection.decisions(space_type) if not item.scope}
    unknown = sorted(set(stated) - own)
    if unknown:
        raise DefinitionError.broken(
            "default-policy-undeclared",
            _OWNER,
            f"{space_type.__qualname__}.defaults fixes {unknown}, which are not its own "
            f"choices ({sorted(own)})",
            declaration=space_type.__qualname__,
        )


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


def ranked(
    point: Space, reference: object, values: tuple[inspection.Possible | inspection.Refused, ...]
) -> tuple[inspection.Possible | inspection.Refused, ...] | Unresolved | Rejected:
    """A choice's listed ``values`` (``inspection.viable``'s, in its list's order) in its
    domain's order: as its ``candidates`` rank them where the domain states an order; or
    why that order is not known (a value it reads is open, or refused)."""
    order = point.field(reference).candidates()
    if order is None or isinstance(order, Inapplicable):
        return values  # known by membership only, or not in play: no order to follow
    if not isinstance(order, Available):
        return order
    rest = list(values)
    found: list[inspection.Possible | inspection.Refused] = []
    for value in order.value:
        match = next((item for item in rest if item.value == value), None)
        if match is not None:
            found.append(match)
            rest.remove(match)
    return (*found, *rest)


def _first_valid(point: S, reference: object, key: str) -> S | Finding:
    """``point`` with ``key`` set to the first of its listed values, in its domain's order
    (``ranked``), that setting alone is accepted (``inspection.viable``), or why none is:
    each value's rules."""
    (viable,) = inspection.viable(point, [reference])
    if viable.values is None:
        waits = ", ".join(viable.waits) or "an input of its domain"
        return unset(key, FIRST_VALID, f"its values wait on {waits}", ())
    items = ranked(point, reference, viable.values)
    if not isinstance(items, tuple):
        return unset(key, FIRST_VALID, "its order is not known", items.findings)
    first = next((item for item in items if isinstance(item, inspection.Possible)), None)
    if first is not None:
        return point.with_choices({reference: first.value})
    refused = [item for item in viable.values if isinstance(item, inspection.Refused)]
    why = "; ".join(f"{item.value!r} breaks {_rules(item.broken)}" for item in refused)
    causes = _causes(rule for item in refused for rule in item.broken)
    return unset(key, FIRST_VALID, why or "its domain lists no value", causes)


def unset(key: str, policy: str, message: str, causes: tuple[Finding, ...]) -> Finding:
    """The finding for a choice no value can be set for (``choice-no-valid-value``), owned
    by its key: the policy, why each value it tried is refused, and those refusals' findings
    as its causes."""
    return Finding(
        FindingKind.REJECTION,
        "choice-no-valid-value",
        key,
        f"no value can be set ({policy}): {message}",
        (("policy", policy),),
        causes,
    )


def defaulted(point: S, *, kept: Collection[str] = ()) -> S | tuple[Finding, ...]:
    """``point``, a design space's root (a kernel or a channel bound to its facts), with
    each of its policies applied once (the module docstring): every choice in play and
    unset that is not required set to its fixed value or its first valid one, in the flow
    order; or why it cannot be: its validity's refusal on what it already holds, or a
    choice no value can be set for (``unset``). Choices it holds are kept, and so are the
    open choices ``kept`` names, by key: left open."""
    validity = inspection.validity(point)
    if isinstance(validity.result, Rejected):
        return validity.result.findings
    found = policies(type(point))
    decisions = {item.key: item for item in inspection.decisions(point)}
    for key in _flow_order(point):
        policy = found[key]
        state = dict(inspection.states(point))[key]
        if policy is None or key in kept or not isinstance(state, inspection.Open) or state.waits:
            continue
        reference = decisions[key].reference
        if isinstance(policy, Fixed):
            done = _fixed(point, reference, key, policy.value)
        else:
            done = _first_valid(point, reference, key)
        if isinstance(done, Finding):
            return (done,)
        point = done
    return point


__all__ = [
    "FIRST_VALID",
    "Fixed",
    "Policy",
    "defaulted",
    "policies",
    "ranked",
    "stated_defaults",
    "unset",
]
