# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Where a Space's checks read: a prototype trace, private, plain data out.

A prototype for the Space trace (design lane ``design/space-trace``): it reads the
engine's private state and changes nothing the engine does. For one configuration,
``trace`` returns, for every Decision below it:

- its **state**: committed, forced, open, refused or inapplicable;
- its **own check's reads**: each domain argument and requirement fact (what
  ``_runtime._membership_frame`` reads before the domain is called), followed through
  the evaluations it demanded down to inputs and Decisions, with the path. A read that
  reaches no other Decision is the choice's own domain over facts; one that reaches a
  Decision reads another choice, and names the lowest scope holding both. A domain
  argument (not a requirement) that reaches another Decision is the **leftover case**:
  the domain's own enumeration or membership follows another choice;
- for a requirement, the cases it names, and for each enumerated case it does not name,
  whether that case's membership read its fact (``_runtime.membership``'s reads);
- its guard's reads of other Decisions;
- for a Decision over nodes, **each case's check**: the candidate's ``admission`` read
  on a copy that does not force, with the case assigned (as ``_forcing._nodes`` reads
  it), each Decision it reached inside the candidate or outside it.

Reads are discovered at evaluation: an unexecuted branch reads nothing, so a caller
traces a Space in several configurations and joins them (``join``), each read naming
the configurations that showed it. ``summary`` counts a joined trace.

Everything returned is plain data (str, int, bool, None, lists and dicts), ready for
JSON.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from . import _forcing, _runtime
from ._configuration import Space
from .occurrence import state as _state
from .references import decision_key
from .results import Available, Inapplicable, QueryResult, Rejected, Unresolved

Data = dict[str, Any]

#: The engine's private names this module reads (for the record).
PRIVATE_READS = (
    "occurrence.state",
    "Space._scope",
    "_runtime.Snapshot (model, parameters, assignments, forcing, cache, linked)",
    "_runtime.Evaluation (result, dependencies, via)",
    "_runtime.evaluate",
    "_runtime.membership",
    "_runtime.enumeration",
    "_runtime.decision_state",
    "_forcing.forced (Found: values, refused)",
    "_forcing.admitted",
    "_forcing._extended",
    "ir.LinkedModel (nodes, scopes, decisions, choices, selector_choices, ranks)",
    "ir.Node (key, kind, scope, guard, domain, domain_arguments, via)",
    "ir.Scope (parent, name, space_type)",
    "ir.Choice (cases)",
    "domains.Domain (requirements, candidates, _finite_values)",
    "domains.Requirement (code, cases, applies)",
    "references.decision_key",
)


def _outcome(result: QueryResult[object]) -> str:
    if isinstance(result, Available):
        return "value"
    if isinstance(result, Rejected):
        return "refused"
    if isinstance(result, Unresolved):
        return "waits"
    if isinstance(result, Inapplicable):
        return "inapplicable"
    return type(result).__name__


def _codes(result: QueryResult[object]) -> list[str]:
    findings = getattr(result, "findings", ())
    return sorted({finding.code for finding in findings})


def _scope_name(linked: Any, scope: int) -> str:
    name: str = linked.scopes[scope].name
    return name or "(root)"


def _space(linked: Any, scope: int) -> Data:
    return {
        "scope": _scope_name(linked, scope),
        "space_type": linked.scopes[scope].space_type.__name__,
    }


def _ancestors(linked: Any, scope: int) -> list[int]:
    found = []
    current: int | None = scope
    while current is not None:
        found.append(current)
        current = linked.scopes[current].parent
    return found


def _lowest_common(linked: Any, first: int, second: int) -> int:
    above = set(_ancestors(linked, first))
    for scope in _ancestors(linked, second):
        if scope in above:
            return scope
    return 0


def _within(linked: Any, scope: int) -> set[int]:
    included = {scope}
    for each in linked.scopes:
        if each.parent in included:
            included.add(each.index)
    return included


def _key(linked: Any, index: int) -> str:
    node = linked.nodes[index]
    return decision_key(linked, index) if node.kind == "decision" else str(node.key)


def _walk(snapshot: _runtime.Snapshot, start: int, own: int) -> dict[int, list[int]]:
    """Every leaf (a Decision, an input or a constant) the evaluation of ``start`` read in
    ``snapshot``, through the evaluations it demanded, with the shortest path to each
    (node indices, forwarding aliases spliced in where a read went straight to the
    alias's source). A Decision is a leaf: its own check is its own. ``own`` is not
    followed."""
    linked, cache = snapshot.linked, snapshot.cache
    _runtime.evaluate(snapshot, start)
    paths: dict[int, list[int]] = {start: [start]}
    leaves: dict[int, list[int]] = {}
    pending = deque([start])
    while pending:
        index = pending.popleft()
        node = linked.nodes[index]
        if node.kind in ("decision", "param", "const") and index != own:
            leaves.setdefault(index, paths[index])
            if node.kind == "decision" or index != start:
                continue
        entry = cache.get(index)
        if entry is None:
            continue
        aliases = {source: alias for alias, source in (*node.via, *entry.via)}
        for read in entry.dependencies:
            if read in paths or read == own:
                continue
            step = [aliases[read], read] if read in aliases else [read]
            paths[read] = [*paths[index], *step]
            pending.append(read)
    return leaves


def _enclosing(linked: Any, scope: int) -> set[int]:
    """The selectors whose candidate is ``scope`` or encloses it: a member below a
    candidate reads its presence (its guard) through them."""
    candidates = {
        place: choice.selector
        for choice in linked.choices
        for _, place in choice.cases
        if place is not None
    }
    return {candidates[each] for each in _ancestors(linked, scope) if each in candidates}


def _reads(
    linked: Any,
    leaves: Mapping[int, list[int]],
    prefix: Sequence[str],
    inside: set[int] | None,
    presence: set[int],
) -> tuple[list[Data], list[Data], list[Data]]:
    """The Decisions, the enclosing selectors (``presence``: read for the reader's own
    presence, not a relation to another choice) and the facts among ``leaves``, each with
    its path (keys), and, where ``inside`` names a candidate's scopes, whether it lies
    inside the candidate."""
    choices: list[Data] = []
    enclosing: list[Data] = []
    facts: list[Data] = []
    for index, path in sorted(leaves.items(), key=lambda item: _key(linked, item[0])):
        node = linked.nodes[index]
        item: Data = {
            "key": _key(linked, index),
            "kind": node.kind,
            "scope": node.scope,
            "path": [*prefix, *(_key(linked, step) for step in path)],
        }
        if inside is not None:
            item["inside"] = node.scope in inside
        if node.kind != "decision":
            facts.append(item)
        elif index in presence:
            item.pop("inside", None)
            enclosing.append(item)
        else:
            choices.append(item)
    return choices, enclosing, facts


def _cases_named(requirement: Any, cases: Sequence[object] | None) -> Data:
    named = requirement.cases
    if named is None:
        return {"form": "every case", "named": None if cases is None else [repr(c) for c in cases]}
    if callable(named):
        return {
            "form": "predicate",
            "named": None
            if cases is None
            else [repr(case) for case in cases if requirement.applies(case)],
        }
    return {"form": "listed", "named": [repr(case) for case in named]}


def _state_of(snapshot: _runtime.Snapshot, found: _forcing.Found, index: int) -> tuple[str, bool]:
    """The Decision's state, and whether its guard waits on an open choice."""
    if index in snapshot.assignments:
        return "committed", False
    applicable = _runtime.decision_state(snapshot, index)
    if isinstance(applicable, Inapplicable):
        return "inapplicable", False
    if index in found.values:
        return "forced", False
    if index in found.refused:
        return "refused", False
    return "open", not isinstance(applicable, Available)


def _own_check(copy: _runtime.Snapshot, index: int, cases: Sequence[object] | None) -> list[Data]:
    """Each domain argument and requirement fact of the Decision ``index``, with what it
    reads (on ``copy``, which does not force)."""
    linked = copy.linked
    node = linked.nodes[index]
    domain = node.domain
    via = {source: alias for alias, source in node.via}
    found: list[Data] = []
    for argument in node.domain_arguments:
        requirement = argument.name.startswith("requires[")
        prefix = [_key(linked, index)]
        leaves = _walk(copy, argument.node, index)
        if argument.node in via:
            # The edge went straight to the alias's source: name the alias.
            leaves = {leaf: [via[argument.node], *path] for leaf, path in leaves.items()}
        choices, enclosing, facts = _reads(
            linked, leaves, prefix, None, _enclosing(linked, node.scope)
        )
        result = _runtime.evaluate(copy, argument.node).result
        item: Data = {
            "argument": argument.name,
            "kind": "requirement" if requirement else "domain",
            "node": _key(linked, argument.node),
            "outcome": _outcome(result),
            "choices": choices,
            "presence": enclosing,
            "facts": facts,
        }
        if requirement and domain is not None:
            position = int(argument.name[len("requires[") : -1])
            rule = domain.requirements[position]
            named = _cases_named(rule, cases)
            item["requirement"] = {"code": rule.code, "message": rule.message, **named}
            if cases is not None:
                unnamed = []
                for case in cases:
                    if rule.applies(case):
                        continue
                    entry = _runtime.membership(copy, index, case)
                    unnamed.append(
                        {
                            "case": repr(case),
                            "fact_read": argument.node in entry.dependencies,
                            "outcome": _outcome(entry.result),
                            "codes": _codes(entry.result),
                        }
                    )
                item["unnamed_cases"] = unnamed
        found.append(item)
    return found


def _case_checks(copy: _runtime.Snapshot, index: int) -> list[Data]:
    """Each case of the Decision over nodes ``index``: its candidate's admission, read on
    ``copy`` with the case assigned, and what it reached inside and outside the
    candidate."""
    linked = copy.linked
    choice = linked.choices[linked.selector_choices[index]]
    found: list[Data] = []
    for case, scope in choice.cases:
        if scope is None:
            found.append({"case": case, "candidate": None, "admission": None})
            continue
        trial = _forcing._extended(copy, {**copy.assignments, index: case}, index)
        verdict, node = _forcing.admitted(trial, scope)
        item: Data = {"case": case, "candidate": _space(linked, scope)}
        if node is None or verdict is None:
            item["admission"] = None
            found.append(item)
            continue
        leaves = _walk(trial, node, index)
        choices, enclosing, facts = _reads(
            linked,
            leaves,
            [_key(linked, index) + f"[{case}]"],
            _within(linked, scope),
            _enclosing(linked, linked.nodes[index].scope),
        )
        item["admission"] = _key(linked, node)
        item["outcome"] = _outcome(verdict)
        item["codes"] = _codes(verdict)
        item["choices"] = choices
        item["presence"] = enclosing
        item["facts"] = facts
        found.append(item)
    return found


def _place(linked: Any, decision: int, choices: list[Data]) -> None:
    """Name, for each Decision read, the lowest scope holding both (where idea B puts
    the rule)."""
    own = linked.nodes[decision].scope
    for item in choices:
        item["holds_both"] = _space(linked, _lowest_common(linked, own, item.pop("scope")))


def trace(point: Space, configuration: str) -> Data:
    """Every Decision below ``point``'s scope, its state and where its checks read, in this
    configuration (named ``configuration``)."""
    snapshot = _state(point)
    linked = snapshot.linked
    found = _forcing.forced(snapshot) if snapshot.forcing else _forcing.NOTHING
    # The checks are read as forcing reads them: on a copy that does not force, every
    # forced value assigned, every other open Decision unassigned.
    copy = _runtime.Snapshot(
        snapshot.model,
        snapshot.parameters,
        {**snapshot.assignments, **found.values},
        forcing=False,
    )
    included = _within(linked, point._scope)
    decisions: list[Data] = []
    for index in sorted(linked.decisions, key=linked.ranks.__getitem__):
        node = linked.nodes[index]
        if node.scope not in included:
            continue
        status, guard_waits = _state_of(snapshot, found, index)
        selector = index in linked.selector_choices
        item: Data = {
            "key": _key(linked, index),
            "rank": linked.ranks[index],
            "space": _space(linked, node.scope),
            "selector": selector,
            "state": status,
            "guard_waits": guard_waits,
        }
        if status == "committed":
            item["value"] = repr(snapshot.assignments[index])
        elif status == "forced":
            item["value"] = repr(found.values[index])
        if node.guard is not None:
            guard_choices, _, _ = _reads(
                linked, _walk(copy, node.guard, index), [], None, _enclosing(linked, node.scope)
            )
            _place(linked, index, guard_choices)
            item["guard_choices"] = guard_choices
        if status == "inapplicable" or node.domain is None:
            decisions.append(item)
            continue
        # The enumeration reads every domain argument, the requirements' facts too
        # (``_runtime._enumeration_frame``), so it may wait on another choice; the
        # declared values stand in for it then.
        cases: Sequence[object] | None = None
        listed = _runtime.enumeration(copy, index).result
        item["enumeration"] = {"outcome": _outcome(listed), "codes": _codes(listed)}
        if isinstance(listed, Available) and listed.value is not None:
            cases = tuple(cast("Iterable[object]", listed.value))
            item["cases"] = [repr(case) for case in cases]
        elif node.domain._finite_values is not None:
            cases = tuple(node.domain._finite_values)
            item["cases"] = [repr(case) for case in cases]
            item["enumeration"]["cases_from"] = "declared values"
        reads = _own_check(copy, index, cases)
        for read in reads:
            _place(linked, index, read["choices"])
            for fact in (*read["facts"], *read["presence"]):
                fact.pop("scope")
        item["own_check"] = reads
        if selector:
            checks = _case_checks(copy, index)
            for check in checks:
                for read in check.get("choices", ()):
                    _place(linked, index, [read])
                for fact in (*check.get("facts", ()), *check.get("presence", ())):
                    fact.pop("scope")
            item["case_checks"] = checks
        decisions.append(item)
    return {"configuration": configuration, "decisions": decisions}


# -- joining configurations -------------------------------------------------------------


def _seen(into: Data, field: str, key: str, item: Data, configuration: str) -> None:
    entries: Data = into.setdefault(field, {})
    entry = entries.get(key)
    if entry is None:
        entry = entries[key] = {k: v for k, v in item.items() if k != "path"}
        entry["path"] = item["path"]
        entry["configurations"] = []
    if configuration not in entry["configurations"]:
        entry["configurations"].append(configuration)


def join(traces: Iterable[Data]) -> Data:
    """The union of several configurations' traces of one Space: each Decision once, its
    state per configuration, and each read once, naming the configurations that showed
    it (the path is the first configuration's)."""
    decisions: dict[str, Data] = {}
    configurations: list[str] = []
    for each in traces:
        name = each["configuration"]
        configurations.append(name)
        for item in each["decisions"]:
            joined = decisions.setdefault(
                item["key"],
                {
                    "key": item["key"],
                    "rank": item["rank"],
                    "space": item["space"],
                    "selector": item["selector"],
                    "states": {},
                },
            )
            state = item["state"] + (" (guard waits)" if item["guard_waits"] else "")
            if "value" in item:
                state += f" = {item['value']}"
            joined["states"][name] = state
            if "cases" in item:
                joined.setdefault("cases", {})[name] = item["cases"]
            if "enumeration" in item:
                listed = item["enumeration"]
                joined.setdefault("enumeration", {})[name] = listed["outcome"] + (
                    f" ({', '.join(listed['codes'])})" if listed["codes"] else ""
                )
            for read in item.get("guard_choices", ()):
                _seen(joined, "guard_choices", read["key"], read, name)
            if "own_check" in item:
                joined.setdefault("own_check", {})
            for read in item.get("own_check", ()):
                arguments = joined["own_check"]
                argument = arguments.setdefault(
                    read["argument"],
                    {"argument": read["argument"], "kind": read["kind"], "node": read["node"]},
                )
                if "requirement" in read:
                    rule = argument.setdefault("requirement", dict(read["requirement"]))
                    if rule.get("named") is None and read["requirement"].get("named") is not None:
                        rule["named"] = read["requirement"]["named"]
                for choice in read["choices"]:
                    _seen(argument, "choices", choice["key"], choice, name)
                for fact in read["facts"]:
                    _seen(argument, "facts", fact["key"], fact, name)
                for selector in read["presence"]:
                    _seen(argument, "presence", selector["key"], selector, name)
                for case in read.get("unnamed_cases", ()):
                    cases = argument.setdefault("unnamed_cases", {})
                    entry = cases.setdefault(
                        case["case"], {"case": case["case"], "read_in": [], "not_read_in": []}
                    )
                    entry["read_in" if case["fact_read"] else "not_read_in"].append(name)
            for check in item.get("case_checks", ()):
                checks = joined.setdefault("case_checks", {})
                case = checks.setdefault(
                    check["case"],
                    {"case": check["case"], "candidate": check["candidate"], "outcomes": {}},
                )
                if check.get("admission") is not None:
                    case["admission"] = check["admission"]
                    case["outcomes"][name] = check["outcome"] + (
                        f" ({', '.join(check['codes'])})" if check["codes"] else ""
                    )
                    for choice in check["choices"]:
                        _seen(case, "choices", choice["key"], choice, name)
                    for fact in check["facts"]:
                        _seen(case, "facts", fact["key"], fact, name)
                    for selector in check["presence"]:
                        _seen(case, "presence", selector["key"], selector, name)
    return {
        "configurations": configurations,
        "decisions": sorted(decisions.values(), key=lambda item: (item["rank"], item["key"])),
    }


# -- relations and counts ---------------------------------------------------------------


def relations(joined: Data) -> list[Data]:
    """Every relation a joined trace shows, one row each: a choice's own check reading
    another choice (a requirement, or a domain: the leftover case), and a case check
    reading a choice outside its candidate."""
    found: list[Data] = []
    for decision in joined["decisions"]:
        for argument in decision.get("own_check", {}).values():
            for other in argument.get("choices", {}).values():
                rule = argument.get("requirement")
                found.append(
                    {
                        "relation": "requirement"
                        if argument["kind"] == "requirement"
                        else "leftover",
                        "choice": decision["key"],
                        "choice_space": decision["space"],
                        "check": argument["argument"],
                        "check_code": None if rule is None else rule["code"],
                        "named_cases": None if rule is None else rule.get("named"),
                        "cases": _all_cases(decision),
                        "other": other["key"],
                        "path": other["path"],
                        "holds_both": other["holds_both"],
                        "configurations": other["configurations"],
                    }
                )
        for case in decision.get("case_checks", {}).values():
            for other in case.get("choices", {}).values():
                if other["inside"]:
                    continue
                found.append(
                    {
                        "relation": "case-outside",
                        "choice": decision["key"],
                        "choice_space": decision["space"],
                        "check": f"{case['case']}: {case.get('admission')}",
                        "case": case["case"],
                        "candidate": case["candidate"],
                        "check_code": None,
                        "named_cases": None,
                        "cases": _all_cases(decision),
                        "other": other["key"],
                        "path": other["path"],
                        "holds_both": other["holds_both"],
                        "configurations": other["configurations"],
                    }
                )
    for row in found:
        row["b_cost"] = _cost(row)
    return found


def _all_cases(decision: Data) -> list[str] | None:
    seen: list[str] = []
    for cases in decision.get("cases", {}).values():
        seen.extend(case for case in cases if case not in seen)
    return seen or None


def _cost(row: Data) -> str:
    """What idea B would cost on this relation, in one line."""
    space = row["holds_both"]
    where = f"{space['scope']} ({space['space_type']})"
    if row["relation"] == "requirement":
        named = row["named_cases"]
        trials = (
            "one trial per named case"
            if named is None
            else f"{len(named)} trials, one per named case ({', '.join(named)})"
        )
        return (
            f"forcing {row['choice']} adds {trials}, reading a constraint of {where}; "
            f"the refusal {row['check_code']} moves from the commit of {row['choice']} to the "
            "configuration"
        )
    if row["relation"] == "leftover":
        return (
            f"B as stated does not move it: {row['choice']}'s domain ({row['check']}) follows "
            f"{row['other']}, so its cases move with that choice"
        )
    return (
        f"already a trial per case today; under B the rule over {row['other']} is a "
        f"constraint of {where}: no new trial, the refusal stays the configuration's"
    )


def summary(joined: Data) -> Data:
    """The counts of a joined trace: choice checks that read only their own domain, that
    read another choice (by a requirement, by a domain: the leftover case), case checks
    reading outside their candidate, and requirement facts read for cases they do not
    name."""
    checked = selectors = own_only = reads_other = by_requirement = leftover = 0
    case_checks = outside = requirement_facts = facts_unnamed = 0
    for decision in joined["decisions"]:
        arguments = decision.get("own_check")
        if arguments is not None:
            checked += 1
            selectors += decision["selector"]
            kinds = {argument["kind"] for argument in arguments.values() if argument.get("choices")}
            if kinds:
                reads_other += 1
            else:
                own_only += 1
            by_requirement += "requirement" in kinds
            leftover += "domain" in kinds
        for argument in (arguments or {}).values():
            if argument["kind"] != "requirement":
                continue
            requirement_facts += 1
            if any(case["read_in"] for case in argument.get("unnamed_cases", {}).values()):
                facts_unnamed += 1
        for case in decision.get("case_checks", {}).values():
            if case.get("admission") is None:
                continue
            case_checks += 1
            outside += any(not other["inside"] for other in case.get("choices", {}).values())
    return {
        "decisions": len(joined["decisions"]),
        "choice_checks": checked,
        "of_which_selectors": selectors,
        "choice_checks_own_domain_only": own_only,
        "choice_checks_reading_another_choice": reads_other,
        "of_which_by_a_requirement": by_requirement,
        "of_which_by_the_domain_leftover": leftover,
        "case_checks": case_checks,
        "case_checks_reading_outside_their_candidate": outside,
        "requirement_facts": requirement_facts,
        "requirement_facts_read_for_unnamed_cases": facts_unnamed,
    }


__all__ = ["PRIVATE_READS", "join", "relations", "summary", "trace"]
