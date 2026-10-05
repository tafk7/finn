# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Parents override the data of any descendant; behaviour stays the Space class's.

Params and Decisions are template fields exposed for customization. Any body
may assign one of any descendant, at any depth: the outermost assignment wins,
one body assigning a target twice is a definition error, and the child's own
constraints and domains still check whatever is supplied. A value pins a
Decision (its key disappears); another Decision replaces it under the same
key. A child node may be replaced by a fresh node of its Space class or a subclass.
Every supplied value records who set it. None of this is about hardware.
"""

from __future__ import annotations

import re

import pytest

from finn.core.space import (
    Available,
    ConfigurationError,
    Decision,
    DefinitionError,
    Inapplicable,
    Located,
    Members,
    Param,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    Users,
    ViewKey,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    selections,
    view,
)

COST = ViewKey("cost", int)


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def messages(result: object) -> str:
    assert isinstance(result, (Rejected, Unresolved))
    return " | ".join(finding.message for finding in result.findings)


class Room(Space):
    area: int = Param(default=12)
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.area > 20:
            return reject("too-large", f"area {self.area} exceeds 20")
        return True

    @view(requires=(fits,))
    def cost(self) -> int:
        return self.area * self.finish

    exports = {COST: cost}


class LargeRoom(Room):
    """A subclass keeps every member the enclosing bodies can name."""

    windows: int = Decision(values=(2, 4))


class Garden(Space):
    area: int = Param(default=40)


class Wing(Space):
    kitchen = Room(area=14)
    study = Room()
    costs = Members(COST)


class House(Space):
    budget: int = Param(default=500)
    wing = Wing()
    wing.kitchen.area = 16  # overrides the Wing's 14
    wing.study.finish = 2  # pins a Decision of a grandchild

    @view
    def total(self) -> int:
        return sum(item.value for item in self.wing.costs)


class Estate(Space):
    home = House()
    home.wing.kitchen.area = 18  # the outermost assignment wins


def test_the_outermost_assignment_wins_across_three_layers() -> None:
    wing = design_space(Wing())
    assert wing.kitchen.area == 14
    house = design_space(House())
    assert house.wing.kitchen.area == 16
    estate = design_space(Estate())
    assert estate.home.wing.kitchen.area == 18
    # A formal nothing overrides keeps its declared default.
    assert estate.home.wing.study.area == 12
    provenance = inspection.provenance(estate, Estate.home.wing.kitchen.area)
    assert provenance is not None
    assert [layer.body for layer in provenance.layers] == ["Room", "Wing", "House", "Estate"]
    assert [layer.value for layer in provenance.layers] == ["12", "14", "16", "18"]
    text = provenance.text()
    assert text.startswith("home.wing.kitchen.area = 18 (set by Estate at test_overrides.py:")
    assert "; overrides 16 set by House at test_overrides.py:" in text
    assert "; overrides 14 set by Wing at test_overrides.py:" in text
    assert re.search(r"; declared 12 at test_overrides.py:\d+\)$", text)


def test_a_value_pins_a_decision_and_its_key_disappears() -> None:
    wing_keys = {item.key for item in inspection.decisions(Wing)}
    assert wing_keys == {"kitchen.finish", "study.finish"}
    house_keys = {item.key for item in inspection.decisions(House)}
    assert house_keys == {"wing.kitchen.finish"}
    point = design_space(House())
    assert point.wing.study.finish == 2
    (pinned,) = inspection.pinned(point)
    assert pinned.key == "wing.study.finish"
    assert pinned.text().startswith("wing.study.finish = 2 (set by House at ")
    assert re.search(
        r"; declared Decision\(values=\(1, 2, 3\)\) at test_overrides.py:\d+\)$", pinned.text()
    )
    # A pinned coordinate is not a choice any more; editing it is refused with its provenance.
    with pytest.raises(RequestError, match=r"enclosing body pinned it \(wing\.study\.finish = 2"):
        point.with_choices({House.wing.study.finish: 3})
    # A pin at the call is an assignment by the calling body.
    assert [item.key for item in inspection.decisions(design_space(Room(finish=3)))] == []


def test_a_pinned_key_leaves_the_selection_with_who_pinned_it() -> None:
    wing = design_space(Wing()).with_choices({Wing.study.finish: 3, Wing.kitchen.finish: 1})
    captured = selections.capture(wing)
    assert captured.keys == ("kitchen.finish", "study.finish")

    class Pinned(Wing):
        pass

    class Holder(Space):
        wing = Pinned()
        wing.study.finish = 2

    assert selections.restore(design_space(Wing()), captured).accepted
    # In the pinned placement the key is gone, and who pinned it is reported.
    (pinned,) = inspection.pinned(design_space(Holder()))
    assert pinned.key == "wing.study.finish"
    assert "wing.study.finish = 2 (set by Holder at test_overrides.py:" in pinned.text()
    # A captured selection belongs to its model: restoring it elsewhere is refused.
    with pytest.raises(RequestError, match="different compiled model"):
        selections.restore(design_space(Holder()), captured)


def test_a_narrower_decision_keeps_its_key_and_the_declared_domain_still_checks() -> None:
    class Narrow(Space):
        room = Room()
        room.finish = Decision(values=(1, 2))

    class Widened(Space):
        room = Room()
        room.finish = Decision(values=(1, 2, 5))

    assert [item.key for item in inspection.decisions(Narrow)] == ["room.finish"]
    narrow = design_space(Narrow())
    assert narrow.room.field(Room.finish).candidates() == Available((1, 2))
    assert narrow.with_choices({Narrow.room.finish: 2}).room.cost == 24
    with pytest.raises(ConfigurationError):
        narrow.with_choices({Narrow.room.finish: 3})
    # Widening is refused by the Space class's declared domain, with who supplied the Decision.
    widened = design_space(Widened())
    assert widened.room.field(Room.finish).candidates() == Available((1, 2))
    report = widened.try_with_choices({Widened.room.finish: 5})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert codes(outcome.result) == {"domain-membership"}
    assert messages(outcome.result).startswith(
        "room.finish = Decision(values=(1, 2, 5)) (set by Widened at test_overrides.py:"
    )


def test_a_pinned_value_outside_the_declared_domain_is_refused_with_its_provenance() -> None:
    class Wrong(Space):
        room = Room()
        room.finish = 7

    point = design_space(Wrong())
    finish = point.room.query(Room.finish)
    assert codes(finish) == {"domain-membership"}
    text = messages(finish)
    assert text.startswith("room.finish = 7 (set by Wrong at test_overrides.py:")
    assert re.search(r"declared Decision\(values=\(1, 2, 3\)\) at test_overrides.py:\d+", text)
    assert "7 is outside the declared domain" in text
    assert isinstance(point.room.query(Room.cost), Rejected)


def test_a_refusal_names_who_set_the_overridden_value_it_read() -> None:
    class Mansion(Space):
        home = House()
        home.wing.kitchen.area = 30  # beyond what a Room admits

    point = design_space(Mansion()).with_choices({Mansion.home.wing.kitchen.finish: 1})
    refused = point.home.wing.kitchen.inspect(Room.fits)
    assert isinstance(refused.result, Rejected)
    (finding,) = refused.result.findings
    assert finding.message.startswith("area 30 exceeds 20; home.wing.kitchen.area = 30 (set by ")
    assert "set by Mansion at test_overrides.py:" in finding.message
    assert "overrides 16 set by House" in finding.message
    assert re.search(r"declared 12 at test_overrides.py:\d+", finding.message)
    assert dict(finding.details)["provenance"] == (
        inspection.provenance(point, Mansion.home.wing.kitchen.area).text(),  # type: ignore[union-attr]
    )
    # Inspection exposes the same record on the member's metadata.
    info = {item.key: item for item in inspection.members(point)}
    assert info["home.wing.kitchen.area"].provenance is not None


def test_one_body_assigning_a_target_twice_is_a_definition_error() -> None:
    with pytest.raises(DefinitionError, match="kitchen.area is already assigned") as caught:

        class Twice(Space):
            wing = Wing()
            wing.kitchen.area = 1
            wing.kitchen.area = 2

    assert "assigned at test_overrides.py:" in str(caught.value)
    with pytest.raises(DefinitionError, match="area is already assigned"):

        class AtTheCallAndAfter(Space):
            room = Room(area=3)
            room.area = 4

    # Replacing a child and assigning below it in the same body sets it twice.
    class Both(Space):
        wing = Wing()
        wing.kitchen = Room(area=5)
        wing.kitchen.area = 6

    with pytest.raises(DefinitionError, match="assigned twice by Both"):
        design_space(Both())


@pytest.mark.parametrize("member", ["fits", "cost", "costs"])
def test_behaviour_is_not_overridable(member: str) -> None:
    target = Wing() if member == "costs" else Room()
    with pytest.raises(DefinitionError, match="behaviour belongs to the Space class; subclass it"):
        setattr(target, member, 1)

    class Budget(Space):
        claims = Users(COST)

        @derived
        def claimed(self) -> int:
            return len(self.claims)

    for name in ("claims", "claimed"):
        with pytest.raises(DefinitionError, match="behaviour belongs to the Space class"):
            setattr(Budget(), name, 1)


def test_a_child_node_is_replaced_by_a_node_of_its_class_or_a_subclass() -> None:
    class Renovated(Space):
        home = House()
        home.wing.kitchen = LargeRoom(area=19)  # replaces House's (and Wing's) settings
        home.wing.study = Room(area=9)

    point = design_space(Renovated())
    kitchen = point.home.wing.kitchen
    assert isinstance(kitchen, LargeRoom) and kitchen.area == 19
    # The node keeps its name, and its new decisions are keyed below it. House's
    # pin of wing.study.finish addresses a member by path, so it still applies.
    keys = {item.key for item in inspection.decisions(point)}
    assert keys == {"home.wing.kitchen.finish", "home.wing.kitchen.windows"}
    replaced = inspection.provenance(point, Renovated.home.wing.kitchen)
    assert replaced is not None
    assert replaced.text().startswith("home.wing.kitchen = LargeRoom node (set by Renovated at ")
    # Every reference to the declared node now reaches the replacement.
    chosen = point.with_choices({Renovated.home.wing.kitchen.finish: 1})
    assert [(item.node, item.value) for item in chosen.home.wing.costs] == [
        ("kitchen", 19),
        ("study", 18),
    ]
    assert chosen.home.total == 37

    with pytest.raises(DefinitionError, match="expected a Room node .or a subclass., got Garden"):

        class Paved(Space):
            wing = Wing()
            wing.kitchen = Garden()  # type: ignore[assignment]

    shared = Room()

    class Placed(Space):
        room = shared

    with pytest.raises(DefinitionError, match="placed exactly once"):

        class Stolen(Space):
            wing = Wing()
            wing.kitchen = shared


def test_a_decision_over_nodes_is_narrowed_under_its_key() -> None:
    class Heater(Space):
        kw: int = Param()

    class Boiler(Heater):
        pass

    class Pump(Heater):
        cop: int = Decision(values=(3, 4))

    class Plant(Space):
        heating: Boiler | Pump | None = Decision(
            {"boiler": Boiler(kw=24), "pump": Pump(kw=8)}, optional=True
        )

    class Site(Space):
        plant = Plant()
        plant.heating = Decision({"pump": Pump(kw=10)}, optional=True)

    assert {item.key for item in inspection.decisions(Site)} == {
        "plant.heating",
        "plant.heating.pump.cop",
    }
    point = design_space(Site()).with_choices({Site.plant.heating: "pump"})
    assert isinstance(point.plant.heating, Pump) and point.plant.heating.kw == 10
    with pytest.raises(ConfigurationError):
        point.with_choices({Site.plant.heating: "boiler"})
    with pytest.raises(DefinitionError, match="does not add one"):

        class Added(Space):
            plant = Plant()
            plant.heating = Decision({"solar": Boiler(kw=1)})

    # A key pins the choice: the declared candidate is selected, with its own
    # bindings, and the key disappears (it is listed as pinned).
    class Pinned(Space):
        plant = Plant()
        plant.heating = "pump"  # type: ignore[assignment]

    assert "plant.heating" not in {item.key for item in inspection.decisions(Pinned)}
    assert [item.key for item in inspection.pinned(Pinned)] == ["plant.heating"]
    pinned = design_space(Pinned())
    assert isinstance(pinned.plant.heating, Pump) and pinned.plant.heating.kw == 8


def test_an_outer_body_may_override_a_reference_input_below_it() -> None:
    class Account(Space):
        limit: int = Param()
        users = Users(COST)

        @view
        def spent(self) -> int:
            return sum(item.value for item in self.users)

    class Team(Space):
        account: Account = Param()
        size: int = Param(default=1)

        @view
        def cost(self) -> int:
            return self.size * 10

        exports = {COST: cost}

    class Department(Space):
        local = Account(limit=5)
        team = Team(account=local)

    class Company(Space):
        central = Account(limit=100)
        department = Department()
        department.team.account = central  # rewired from the enclosing body
        department.team.size = 3

    point = design_space(Company())
    assert point.department.team.account.limit == 100
    assert point.central.users == (Located("department.team", "account", 30),)
    assert point.department.local.users == ()
    assert isinstance(point.department.local.query(Account.limit), Available)
    assert isinstance(point.department.team.query(Team.account), Available)
    assert isinstance(point.department.query(Department.team), Available)
    assert not isinstance(point.query(Company.department), Inapplicable)


def test_an_assignment_reaches_a_member_of_a_decision_candidate() -> None:
    class Leaf(Space):
        style: str = Decision(values=("a", "b"))

    class Holder(Space):
        leaf = Leaf()  # a handle naming the candidate
        choice: Leaf | None = Decision({"leaf": leaf}, optional=True)

    class Outer(Space):
        holder = Holder()
        holder.leaf.style = "b"  # keyed by the candidate's path: choice.leaf.style

    point = design_space(Outer())
    assert [item.key for item in inspection.decisions(point)] == ["holder.choice"]
    assert [item.key for item in inspection.pinned(point)] == ["holder.choice.leaf.style"]
    chosen = point.with_choices({Outer.holder.choice: "leaf"})
    assert isinstance(chosen.holder.choice, Leaf) and chosen.holder.choice.style == "b"
