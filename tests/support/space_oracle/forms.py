# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The answers by key in one canonical, JSON-able form, the same in every process, and the
differences of two forms, by key.

A value (``value``):
- a dataclass is its type's qualified name and its fields (those its equality compares);
- a mapping is its items, sorted; a set its elements, sorted;
- a sequence is its elements; an enum its name;
- an array (anything with a ``dtype``, a ``shape`` and ``tolist``) its dtype, shape and
  contents;
- a float its ``repr`` (so ``-0.0`` stays apart from ``0.0``);
- ``None``, a bool, an int and a str themselves;
- anything else its ``repr``, refused (``AddressInRepr``) where that holds an address.

Plain ``repr`` of a ``frozenset`` changes with ``PYTHONHASHSEED``, so it cannot key a
file compared across processes; sorting the rendered elements can.

An answer (``answer``) is its kind (value, unresolved, refused, inapplicable), its value,
and its findings, sorted: the order of findings follows evaluation order, which is hidden.
A finding (``finding``) is rendered whole: its kind (a blocker apart from a rejection of
the same code), code, owner, message, details (its structured subject) and causes, each
cause a finding of its own, in the fixed order ``Finding`` keeps them.

A design space (``space``) is every member's kind and answer, every scope's validity,
every choice's state in the flow order, the stored choices, and the possible values of the
choices asked. A change (``report``) is each outcome, the rules broken (sorted, each with
its standing choices), and what it moved. ``raised_paths`` names every question in a form
that raised in place of answering.

``differences`` walks two forms and names every place they differ by its path of keys
(``members/kitchen.cost/answer``).
"""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from finn.core.reporting.findings import Finding
from finn.core.space import (
    Available,
    BrokenRule,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    inspection,
)
from finn.core.space.edits import ConfigurationResult
from support.space_oracle import Answers

Form = Any
"""A JSON value: None, bool, int, str, a list of forms, or a dict of str to forms."""


class AddressInRepr(ValueError):
    """A value whose only form is a ``repr`` holding an address (`` at 0x``): no form the
    same in every process."""


def _ordered(forms: list[Form]) -> list[Form]:
    return sorted(forms, key=lambda form: json.dumps(form, sort_keys=True))


def value(found: object) -> Form:
    """``found`` in its canonical form (the module's rules)."""
    if found is None or isinstance(found, (bool, str)):
        return found
    if isinstance(found, enum.Enum):
        return {"enum": found.name}
    if isinstance(found, int):
        return found
    if isinstance(found, float):
        return {"float": repr(found)}
    if dataclasses.is_dataclass(found) and not isinstance(found, type):
        return {
            "type": type(found).__qualname__,
            "fields": {
                field.name: value(getattr(found, field.name))
                for field in dataclasses.fields(found)
                if field.compare
            },
        }
    if isinstance(found, Mapping):
        return {"mapping": _ordered([[value(key), value(item)] for key, item in found.items()])}
    if isinstance(found, AbstractSet):
        return {"set": _ordered([value(item) for item in found])}
    if isinstance(found, Sequence) and not isinstance(found, (bytes, bytearray)):
        return [value(item) for item in found]
    if (
        not isinstance(found, type)
        and hasattr(found, "dtype")
        and hasattr(found, "shape")
        and hasattr(found, "tolist")
    ):
        return {
            "array": {
                "dtype": str(found.dtype),
                "shape": list(found.shape),
                "contents": value(found.tolist()),
            }
        }
    text = repr(found)
    if " at 0x" in text:
        raise AddressInRepr(f"{type(found).__qualname__}: its repr holds an address: {text}")
    return {"repr": text}


def _kind(result: object) -> str:
    if isinstance(result, Available):
        return "value"
    if isinstance(result, Unresolved):
        return "unresolved"
    if isinstance(result, Rejected):
        return "refused"
    if isinstance(result, Inapplicable):
        return "inapplicable"
    raise TypeError(f"not an answer: {result!r}")


def finding(found: Finding) -> Form:
    """A finding whole: its kind, code, owner, message, details and causes (each a finding,
    in the order the finding keeps them)."""
    return {
        "kind": found.kind.value,
        "code": found.code,
        "owner": found.owner,
        "message": found.message,
        "details": value(found.details),
        "causes": [finding(cause) for cause in found.causes],
    }


def answer(result: QueryResult[Any]) -> Form:
    """An answer: its kind, its value, and its findings (each whole), sorted."""
    if isinstance(result, Available):
        return {"kind": "value", "value": value(result.value)}
    return {
        "kind": _kind(result),
        "findings": _ordered([finding(found) for found in result.findings]),
    }


def raised(error: BaseException) -> Form:
    """A question that raised instead of answering."""
    return {"kind": "raised", "error": type(error).__name__, "message": str(error)}


def raised_paths(form: Form, path: str = "") -> list[str]:
    """The path of every ``raised`` form within ``form`` (a question that raised)."""
    if isinstance(form, dict):
        if form.get("kind") == "raised" and "error" in form:
            return [path]
        return [
            found
            for key, item in form.items()
            for found in raised_paths(item, f"{path}/{key or ROOT}" if path else key or ROOT)
        ]
    if isinstance(form, list):
        return [
            found
            for index, item in enumerate(form)
            for found in raised_paths(item, f"{path}/{index}")
        ]
    return []


def _asked(question: Any) -> Form:
    try:
        return question()
    except Exception as error:  # noqa: BLE001 - a raise is an answer to compare
        return raised(error)


def validity(found: inspection.Validity) -> Form:
    return {
        "status": found.status,
        "pieces": [[piece.owner, answer(piece.result)] for piece in found.pieces],
        "result": answer(found.result),
    }


def state(found: inspection.ChoiceState) -> Form:
    if isinstance(found, inspection.Set):
        return {"state": "set", "value": value(found.value)}
    if isinstance(found, inspection.Waiting):
        return {"state": "waiting", "value": value(found.value), "waits": list(found.waits)}
    if isinstance(found, inspection.Dormant):
        return {"state": "dormant", "value": value(found.value)}
    if isinstance(found, inspection.Open):
        return {"state": "open", "waits": list(found.waits), "required": found.required}
    if isinstance(found, inspection.Absent):
        return {"state": "absent"}
    raise TypeError(f"not a choice state: {found!r}")


def broken(rules: Sequence[BrokenRule]) -> Form:
    """Rules broken, sorted (the order they are found in is evaluation order)."""
    return _ordered(
        [
            {"rule": rule.rule, "result": answer(rule.result), "standing": list(rule.standing)}
            for rule in rules
        ]
    )


def viable(found: inspection.Viable) -> Form:
    if found.values is None:
        return {"values": None, "waits": list(found.waits)}
    values: list[Form] = []
    for item in found.values:
        if isinstance(item, inspection.Possible):
            values.append(
                {
                    "value": value(item.value),
                    "possible": {
                        "dormant": list(item.dormant),
                        "in_play": list(item.in_play),
                        "opened": list(item.opened),
                    },
                }
            )
        else:
            values.append({"value": value(item.value), "refused": broken(item.broken)})
    return {"values": values, "waits": list(found.waits)}


def report(found: ConfigurationResult[Any]) -> Form:
    """A change's or a bind's report: outcomes in order (the facts, the choices named,
    then the choices dropped), the rules broken, and what moved."""
    return {
        "accepted": found.accepted,
        "outcomes": [
            {
                "owner": outcome.owner,
                "status": outcome.status,
                "requested": outcome.requested,
                "result": answer(outcome.result),
            }
            for outcome in found.outcomes
        ],
        "broken": broken(found.broken),
        "dormant": list(found.dormant),
        "in_play": list(found.in_play),
        "opened": list(found.opened),
    }


ALL = None
"""Possible values for every choice (``space``'s ``viable``)."""


def space(answers: Answers, viable_keys: Sequence[str] | None = ALL) -> Form:
    """A design space: every member's kind and answer, every scope's validity, every
    choice's state in the flow order, the stored choices, and the possible values of
    ``viable_keys`` (every choice by default)."""
    states = list(answers.states())
    keys = [key for key, _ in states] if viable_keys is ALL else list(viable_keys)
    asked = list(answers.viable(keys)) if keys else []
    return {
        "members": {
            key: {"kind": kind, "answer": _asked(lambda key=key: answer(answers.query(key)))}
            for key, kind in answers.members()
        },
        "validity": {
            scope: _asked(lambda scope=scope: validity(answers.validity(scope)))
            for scope in answers.scopes()
        },
        "flow": [key for key, _ in states],
        "states": {key: state(found) for key, found in states},
        "stored": {key: value(item) for key, item in sorted(answers.stored().items())},
        "viable": {item.key: viable(item) for item in asked},
    }


# -- differences, by key -----------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    """Where two forms differ (``path``, by key), and what each holds there (``None``
    where one side lacks it)."""

    path: str
    base: Form
    other: Form

    @property
    def kind(self) -> str:
        """What differs: the path's first part that names a question (``members``,
        ``validity``, ``states``, ``viable``, ``report``, ...)."""
        parts = [part for part in self.path.split("/") if part]
        for part in parts:
            if part in KINDS:
                return part
        return parts[0] if parts else ""

    def __str__(self) -> str:
        def text(form: Form) -> str:
            found = json.dumps(form, sort_keys=True)
            return found if len(found) <= 400 else found[:400] + "..."

        return f"{self.path}: {text(self.base)} vs {text(self.other)}"


KINDS = frozenset(("members", "validity", "flow", "states", "stored", "viable", "report", "raised"))
ROOT = "(root)"
"""The root scope's name in a path (its key in a form is ``""``)."""
MISSING = {"missing": True}
"""Where one form holds a key the other lacks."""


def differences(base: Form, other: Form, path: str = "") -> list[Difference]:
    """Every place ``base`` and ``other`` differ, by path: a dict by its keys, a list of the
    same length by its positions, anything else whole."""
    if base == other:
        return []
    if isinstance(base, dict) and isinstance(other, dict):
        found: list[Difference] = []
        for key in sorted(base.keys() | other.keys()):
            name = str(key) or ROOT
            where = f"{path}/{name}" if path else name
            if key not in base:
                found.append(Difference(where, MISSING, other[key]))
            elif key not in other:
                found.append(Difference(where, base[key], MISSING))
            else:
                found.extend(differences(base[key], other[key], where))
        return found
    if isinstance(base, list) and isinstance(other, list) and len(base) == len(other):
        found = []
        for index, (left, right) in enumerate(zip(base, other)):
            found.extend(differences(left, right, f"{path}/{index}"))
        return found
    return [Difference(path, base, other)]


__all__ = [
    "ALL",
    "MISSING",
    "ROOT",
    "AddressInRepr",
    "Difference",
    "Form",
    "answer",
    "broken",
    "differences",
    "finding",
    "raised",
    "raised_paths",
    "report",
    "space",
    "state",
    "validity",
    "value",
    "viable",
]
