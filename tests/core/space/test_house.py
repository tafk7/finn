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

from core.space._choices_support import committed, replay
from core.space._results_support import codes
from core.space._toys_support import Boiler, HeatPump, House, Room, Thermostat
from finn.core.space import (
    Available,
    ConfigurationError,
    Decision,
    Inapplicable,
    Located,
    Unresolved,
    design_space,
    inspection,
)


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
    assert codes(results["matched"]) == {"mismatch"}
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


def test_keys_commitments_and_inspection_use_declaration_paths() -> None:
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
    chosen = committed(point)
    assert chosen == {"heating": "heat_pump", "heating.heat_pump.cop": 4}
    replayed = replay(design_space(House(budget=999)), chosen)
    assert replayed.thermostat.kw == 8
    (choice,) = inspection.choices(house)
    assert [case.name for case in choice.cases] == ["boiler", "heat_pump"]


class FixedRoom(Room):
    finish: int = Decision(values=(2,))


class Estate(House):
    """A subclass customizes the house it inherits: data and structure, redeclared."""

    kitchen = FixedRoom(area=14)  # a larger kitchen of one finish
    heat_pump = HeatPump(kw=6)
    heating: HeatPump = Decision({"heat_pump": heat_pump})  # the heat pump alone


def test_an_estate_customizes_the_house_it_inherits() -> None:
    estate = design_space(Estate(budget=300))
    keys = {item.key for item in inspection.decisions(estate)}
    assert {"kitchen.finish", "heating", "heating.heat_pump.cop"} <= keys
    point = estate.with_choices(
        {
            Estate.want_garage: True,
            Estate.heating: "heat_pump",
            Estate.heat_pump.cop: 3,
            Estate.hall.finish: 1,
            Estate.dining.finish: 2,
            Estate.garage.finish: 1,
        }
    )
    assert point.hall.area == 14  # hall.area = kitchen.area reads the subclass's kitchen
    assert point.kitchen.finish == 2  # its one case: forced
    assert point.thermostat.kw == 6
    assert point.total == 14 + 28 + 32 + 20 + 36 + 2
