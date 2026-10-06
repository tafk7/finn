# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The house toy: a graph of design spaces in the declarative form.

Calling a Space class declares a node; ``kitchen.finish`` is a reference to that
node's member; a Decision over nodes is the structural choice; ``design_space``
is the one compile step. A view reads as its accepted value (``point.total``);
its assessment is ``point.inspect(House.total)``. Nothing here is about hardware.
The house is declared in ``_toys_support``, which the collapse tests share.
"""

from __future__ import annotations

import pytest
from core.space._toys_support import Boiler, HeatPump, House, Room, Thermostat

from finn.core.space import (
    Available,
    ConfigurationError,
    Decision,
    Inapplicable,
    Located,
    Rejected,
    Space,
    Unresolved,
    design_space,
    inspection,
    selections,
)


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def test_the_house_is_declared_then_configured() -> None:
    house = design_space(House(budget=200))
    assert isinstance(house.query(House.total), Unresolved)  # nothing decided yet
    assert house.hall.area == 12  # supplied by the assignment hall.area = kitchen.area
    point = house.with_choices(
        {
            House.want_garage: False,
            House.heating: "heat_pump",
            House.heat_pump.cop: 3,
            House.hall.finish: 1,
            House.kitchen.finish: 2,
            House.dining.finish: 2,
        }
    )
    assert point.thermostat.kw == 8  # heating.kw: the selected candidate's member
    assert point.costs == (
        Located("hall", "cost", 12),
        Located("kitchen", "cost", 24),
        Located("dining", "cost", 32),
        Located("heating.heat_pump", "cost", 38),
        Located("thermostat", "cost", 2),
    )
    assert point.total == 12 + 24 + 32 + 38 + 2


def test_a_structural_choice_reads_as_the_selected_candidate() -> None:
    house = design_space(House(budget=200))
    # heating.kw is unresolved until the decision is made.
    assert isinstance(house.thermostat.query(Thermostat.kw), Unresolved)
    boiler = house.with_choices(heating="boiler")
    assert boiler.thermostat.kw == 24
    selected = boiler.heating
    assert isinstance(selected, Boiler) and selected.kw == 24
    assert isinstance(boiler.query(House.heat_pump.cop), Inapplicable)
    pump = boiler.with_choices({House.heating: "heat_pump", House.heat_pump.cop: 4})
    assert isinstance(pump.heating, HeatPump) and pump.heating.cop == 4
    # A stale candidate-local choice is refused when switching away.
    assert not pump.try_with_choices(heating="boiler").accepted
    cleared = pump.with_choices(pump.heating.field(HeatPump.cop).clear(), heating="boiler")
    assert cleared.thermostat.kw == 24


def test_the_relation_and_the_budget_own_their_refusals() -> None:
    house = design_space(House(budget=80))
    point = house.with_choices(
        {
            House.want_garage: True,
            House.heating: "boiler",
            House.hall.finish: 1,
            House.kitchen.finish: 1,
            House.dining.finish: 3,
            House.garage.finish: 1,
        }
    )
    refused = point.inspect(House.total)
    results = refused.constraints.results
    assert codes(results["matched.agreed"]) == {"mismatch"}
    assert "kitchen.finish=1, dining.finish=3" in {
        finding.message
        for finding in refused.accepted_result.findings  # type: ignore[union-attr]
    }
    assert codes(results["within_budget"]) == {"over-budget"}
    assert isinstance(results["garage.cost"], Available)


def test_the_garage_is_present_only_when_wanted() -> None:
    house = design_space(House(budget=500))
    without = house.with_choices(want_garage=False)
    assert isinstance(without.garage.query(Room.finish), Inapplicable)
    with pytest.raises(ConfigurationError):
        without.with_choices({House.garage.finish: 2})
    wanted = house.with_choices(want_garage=True)
    assert wanted.with_choices({House.garage.finish: 2}).garage.cost == 40


def test_keys_selections_and_inspection_use_declaration_paths() -> None:
    house = design_space(House(budget=200))
    keys = {item.key for item in inspection.decisions(house)}
    assert keys == {
        "want_garage",
        "heating",
        "heating.heat_pump.cop",
        "hall.finish",
        "kitchen.finish",
        "dining.finish",
        "garage.finish",
    }
    point = house.with_choices({House.heating: "heat_pump", House.heat_pump.cop: 4})
    captured = selections.capture(point)
    assert captured.keys == ("heating", "heating.heat_pump.cop")
    replayed = selections.restore(design_space(House(budget=999)), captured)
    assert replayed.accepted and replayed.instance.thermostat.kw == 8
    (choice,) = inspection.choices(house)
    assert [case.name for case in choice.cases] == ["boiler", "heat_pump"]


class Estate(Space):
    """An enclosing body customizes the house it contains: data, never behaviour."""

    home = House(budget=300)
    home.kitchen.area = 14  # overrides House's 12
    home.kitchen.finish = 2  # pins a Decision: its key disappears
    home.heating = Decision({"heat_pump": HeatPump(kw=6)})  # narrows the choice
    home.garage = Room(area=24, finish=1)  # replaces a child node (same Space class)


def test_an_estate_overrides_the_house_it_contains() -> None:
    estate = design_space(Estate())
    keys = {item.key for item in inspection.decisions(estate)}
    assert "home.kitchen.finish" not in keys and "home.garage.finish" not in keys
    assert {"home.heating", "home.heating.heat_pump.cop"} <= keys
    point = estate.with_choices(
        {
            Estate.home.want_garage: True,
            Estate.home.heating: "heat_pump",
            Estate.home.heat_pump.cop: 3,
            Estate.home.hall.finish: 1,
            Estate.home.dining.finish: 2,
        }
    )
    assert point.home.hall.area == 14  # hall.area = kitchen.area follows the override
    assert point.home.thermostat.kw == 6
    assert point.home.total == 14 + 28 + 32 + 24 + 36 + 2
    provenance = inspection.provenance(point, Estate.home.kitchen.area)
    assert provenance is not None
    assert provenance.text().startswith("home.kitchen.area = 14 (set by Estate at test_house.py:")
