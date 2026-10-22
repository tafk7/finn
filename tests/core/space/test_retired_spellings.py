# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The spellings ``Select``, class-owned Decisions and policies replace: each is refused
where it is written, naming its replacement, or gone from the API.

- ``Decision({...})`` (a Decision over nodes): ``Select``; ``Decision`` keeps value domains;
- ``optional=``: a ``None`` case;
- ``required=``: a policy, the compiler's;
- ``name=`` and a Decision given at a call: a Decision of the class, bound;
- a domain's ``order=``: a ranking view a policy names; ``ordered=`` stays;
- a handle beside a selector (a class attribute naming a case's node): refused in
  ``test_candidate_entries``;
- ``selected()``: gone from the API (a binding to the selector reads the case's name).
"""

from __future__ import annotations

from typing import Any

from support.refusals import refused

import finn.core.space as space
from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Select,
    Space,
    composite,
    design_space,
    domain,
)


class Room(Space):
    area: int = Param()


class Hall(Space):
    area: int = Param()


def test_a_decision_over_nodes_is_refused_pointing_to_select() -> None:
    entries: Any = {"room": Room, "hall": Hall}
    with refused("decision-over-nodes", message=r"keeps value domains.*Select\(name=Class"):
        Decision(entries, area=2)  # type: ignore[call-overload]
    with refused("node-decision-not-entries", message="a choice among Space classes is a selector"):
        Decision(values=entries)


def test_optional_is_refused_pointing_to_a_none_case() -> None:
    retired: Any = {"optional": True}
    with refused("decision-optional", message=r"None case of a selector, Select\(none=None"):
        Decision(values=(1, 2), **retired)
    with refused("select-optional", message=r"Select\(none=None, \.\.\.\)"):
        Select(room=Room, hall=Hall, area=2, **retired)


def test_required_is_refused_pointing_to_a_policy() -> None:
    retired: Any = {"required": True}
    with refused("decision-required", message="the compiler's"):
        Decision(values=(1, 2), **retired)
    with refused("select-required", message="the compiler's"):
        Select(room=Room, hall=Hall, area=2, **retired)


def test_a_named_decision_is_refused_pointing_to_the_class() -> None:
    retired: Any = {"name": "depth"}
    with refused("decision-named", message="declare it as a class attribute"):
        Decision(values=(1, 2), **retired)


def test_an_unknown_keyword_is_a_type_error() -> None:
    retired: Any = {"shared": 1}
    try:
        Decision(values=(1, 2), **retired)
    except TypeError as error:
        assert "unexpected keyword arguments ['shared']" in str(error)
    else:
        raise AssertionError("Decision(shared=...) was accepted")


def test_a_decision_given_at_a_call_is_refused_where_it_is_written() -> None:
    with refused("decision-inline", message=r"Room\.area .*Declare it as a class attribute"):

        class Inline(Space):
            room = Room(area=Decision(values=(1, 2)))

    # A sibling declared on the class is bound, and owns the choice.
    class Owned(Space):
        area: int = Decision(values=(1, 2))
        room = Room(area=area)

    assert design_space(Owned()).with_choices(area=2).room.area == 2


def test_a_decision_given_at_a_call_outside_a_class_body_is_refused_when_linked() -> None:
    built = composite("Built", {"room": Room(area=Decision(values=(1, 2)))})
    with refused("decision-inline", message="room.area"):
        design_space(built())


def test_a_domains_order_is_refused_pointing_to_a_ranking_view() -> None:
    with refused("domain-order", message=r"a domain ranks nothing.*FirstValid\(order="):
        domain(
            accepts=lambda *, candidate: True,
            candidates=lambda: (1, 2),
            order=lambda values: values,
        )
    ordered = domain(accepts=lambda *, candidate: True, candidates=lambda: (1, 2), ordered=True)
    assert ordered.ordered


def test_selected_is_gone_from_the_api() -> None:
    assert "selected" not in space.__all__
    assert not hasattr(space, "selected")


def test_the_refusals_are_definition_errors() -> None:
    try:
        Decision({"room": Room})  # type: ignore[call-overload]
    except DefinitionError:
        pass
    else:
        raise AssertionError("the entries form was accepted")
