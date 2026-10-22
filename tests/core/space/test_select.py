# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``Select``: a selector, chosen or picked by a derived value (``by=``).

A chosen ``Select`` stores the case's name under its key, and its cases are guarded by the
selector naming them. A picked
``Select(by=d)`` has no key and is never a choice: its value is a derived node reading
``d`` (refused ``select-no-case`` where ``d`` names no case), its cases are guarded by that
value being known and naming them, and a refused picker refuses its Space once, in the
selector's name. The full rows and walks are the reference model's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    bind,
    constraint,
    derived,
    design_space,
    domain,
    inspection,
    reject,
    supplied,
    view,
)
from finn.core.space.inspection import Dormant, Open, Set, Waiting


def _name(case: str) -> str:
    return case


# -- the cases ------------------------------------------------------------------------------


class Boiler(Space):
    kw: int = Param()
    stages: int = Decision(values=(1, 2, 3))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.stages * self.kw > 48:
            return reject("boiler-too-big", "more than 48 kW")
        return True

    @derived
    def output(self) -> int:
        return self.stages * self.kw


class HeatPump(Space):
    kw: int = Param()
    cop: int = Decision(values=(2, 3, 4))

    @derived
    def output(self) -> int:
        return self.cop * self.kw


# -- 1. a chosen Select --------------------------------------------


class BySelect(Space):
    load: int = Decision(values=(8, 16, 24))
    gate: int = Decision(values=(0, 1, 2))

    @derived
    def heated(self) -> bool | Rejected:
        if self.gate == 2:
            return reject("gate-refused", "gate 2 refuses")
        return self.gate == 0

    heating = Select(none=None, boiler=Boiler, heat_pump=HeatPump, kw=load, when=heated)
    heat = View(heating.output)
    kind = view(case=heating)(_name)

    @constraint
    def enough(self) -> bool | Rejected:
        found = self.heating
        if found is not None and found.output < 24:
            return reject("too-little", "less than 24 kW")
        return True


def _bound(space: type[Space], choices: Mapping[str, object]) -> Any:
    base = design_space(space())
    owned = {item.key: item.reference for item in inspection.decisions(base)}
    return bind(base, {owned[key]: value for key, value in choices.items()}).instance


def _pieces(point: Space) -> tuple[object, ...]:
    found = inspection.validity(point)
    return (
        found.status,
        tuple(
            (piece.owner, tuple((f.code, f.owner) for f in piece.result.findings))
            for piece in found.pieces
        ),
    )


def test_a_chosen_select_lists_its_cases_and_has_a_key() -> None:
    point = design_space(BySelect())
    assert "heating" in [d.key for d in inspection.decisions(point)]
    (chosen,) = (c for c in inspection.choices(point) if c.key == "heating")
    assert [case.name for case in chosen.cases] == ["none", "boiler", "heat_pump"]
    assert chosen.selector is not None


def test_a_chosen_select_reads_its_cases() -> None:
    point = _bound(BySelect, {"load": 16, "gate": 0, "heating": "heat_pump"})
    point = point.with_choices({BySelect.heating["heat_pump"].cop: 2})
    assert point.kind == "heat_pump" and point.heat == 32
    assert isinstance(point.heating, HeatPump)
    assert isinstance(point.query(BySelect.heating["boiler"].stages), Inapplicable)


# -- 2. the evaluator rule: a refused guard counts once, at the highest of its chain ------------


class Guarded(Space):
    gate: int = Decision(values=(0, 1))

    @derived
    def heated(self) -> bool | Rejected:
        return reject("gate-refused", "gate 1 refuses") if self.gate == 1 else True

    heating: Boiler | HeatPump = Select(boiler=Boiler, heat_pump=HeatPump, kw=8, when=heated)


def test_a_chosen_selector_under_a_refused_when_is_one_piece() -> None:
    # Reading 8 of the reference model: one guard piece "at the highest such guard (scopes
    # under it add nothing)". Each case used to add a piece of its own under the refused
    # ``when``: 3 pieces here with the selector open (heating, heating.boiler,
    # heating.heat_pump), 2 with it set (heating and the case set); now 1.
    for stored in ({}, {Guarded.heating: "boiler"}, {Guarded.heating: "heat_pump"}):
        point = bind(design_space(Guarded()), {Guarded.gate: 1, **stored}).instance
        assert _pieces(point) == (
            "refused",
            (("heating", (("gate-refused", "heated"), ("guard-refused", "heating"))),),
        )
    # a rule reading into the selector's cases counts on its own, as it reads
    point = _bound(BySelect, {"load": 16, "gate": 2, "heating": "boiler"})
    assert [piece.owner for piece in inspection.validity(point).pieces] == ["enough", "heating"]


# Reading 8's second half (SZ111): a refused ``when`` below a selector that is still open
# leaves its piece unknown, waiting on that selector, never counted and never left out:
# counted once, at the highest such ``when``, keyed as a refused guard's piece. Each wing
# holds a selector whose own ``when`` refuses at gate 1: chosen, picked by a derived value
# (its cases then wait through the picker's guard, a chain of guards above the when), or
# chosen with a case placed under a second refused ``when``. No case holds a rule, so
# nothing else waits: the engine answered valid before.


class _Coil(Space):
    kw: int = Param()
    turns: int = Decision(values=(1, 2))


class _Pump(Space):
    kw: int = Param()
    speed: int = Decision(values=(1, 2))


class _Heated(Space):
    gate: int = Param()

    @derived
    def heated(self) -> bool | Rejected:
        return reject("gate-refused", "gate 1 refuses") if self.gate == 1 else True


class ChosenWing(_Heated):
    heating: _Coil | _Pump = Select(coil=_Coil, pump=_Pump, kw=8, when=_Heated.heated)


class PickedWing(_Heated):
    @derived
    def pick(self) -> str:
        return "coil"

    heating: _Coil | _Pump = Select(by=pick, coil=_Coil, pump=_Pump, kw=8, when=_Heated.heated)


class NestedWing(_Heated):
    heating: _Coil | _Pump = Select(
        coil=_Coil(when=_Heated.heated), pump=_Pump, kw=8, when=_Heated.heated
    )


class House(Space):
    gate: int = Param()
    wing: ChosenWing | PickedWing | None = Select(
        chosen=ChosenWing, picked=PickedWing, none=None, gate=gate
    )


class NestedHouse(Space):
    gate: int = Param()
    wing: NestedWing | None = Select(nested=NestedWing, none=None, gate=gate)


def test_a_refused_when_below_an_open_selector_leaves_validity_open() -> None:
    point = design_space(House(gate=1))
    # one unknown piece per wing's selector, waiting on the wing (it answered valid before)
    assert _pieces(point) == (
        "open",
        (
            ("wing.chosen.heating", (("decision-unassigned", "wing"),)),
            ("wing.picked.heating", (("decision-unassigned", "wing"),)),
        ),
    )
    # each case, once chosen, refuses once, in its selector's name
    assert _pieces(design_space(House(gate=0))) == ("valid", ())
    for case in ("chosen", "picked"):
        chosen = bind(point, {House.wing: case}).instance
        assert _pieces(chosen) == (
            "refused",
            (
                (
                    f"wing.{case}.heating",
                    (
                        ("gate-refused", f"wing.{case}.heated"),
                        ("guard-refused", f"wing.{case}.heating"),
                    ),
                ),
            ),
        )


def test_two_nested_refused_whens_below_an_open_selector_are_one_unknown_piece() -> None:
    # ``heating``'s own when and its case ``coil``'s, both refused, the wing open: one
    # piece, unknown, waiting on the wing, keyed by the outer when's owner
    point = design_space(NestedHouse(gate=1))
    assert _pieces(point) == (
        "open",
        (("wing.nested.heating", (("decision-unassigned", "wing"),)),),
    )
    assert _pieces(design_space(NestedHouse(gate=0))) == ("valid", ())
    # the wing chosen: one refused guard's piece, at the outer when
    chosen = bind(point, {NestedHouse.wing: "nested"}).instance
    assert _pieces(chosen) == (
        "refused",
        (
            (
                "wing.nested.heating",
                (
                    ("gate-refused", "wing.nested.heated"),
                    ("guard-refused", "wing.nested.heating"),
                ),
            ),
        ),
    )


# -- 3. a Select picked by a derived value -------------------------------------------------


class Fifo(Space):
    width: int = Param()
    depth: int = Decision(values=(2, 4, 8))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.depth > 2 * self.width:
            return reject("too-deep", "deeper than twice the width")
        return True

    @derived
    def stages(self) -> int:
        return self.depth


class Reg(Space):
    width: int = Param()
    stages: int = Decision(values=(1, 2))


CASES = ("fifo", "reg", "direct", "zz")


class Channel(Space):
    """An adapter picked by ``chain``, of two choices: ``x + y`` names a case (mod 4,
    ``zz`` naming none), and ``chain`` refuses where ``x`` is 3."""

    w: int = Param()
    x: int = Decision(values=(0, 1, 2, 3))
    y: int = Decision(values=(1, 2))

    @derived
    def chain(self) -> str | Rejected:
        if self.x == 3:
            return reject("chain-refused", "x is 3")
        return CASES[(self.x + self.y) % 4]

    adapter: Fifo | Reg | None = Select(by=chain, fifo=Fifo, reg=Reg, direct=None, width=w)
    adapter_stages = View(adapter.stages)  # type: ignore[union-attr]

    @derived
    def stages_read(self) -> int:
        found = self.adapter
        return -1 if found is None else found.stages


def _channel(choices: Mapping[Any, object]) -> Any:
    return bind(design_space(Channel(w=2)), choices).instance


STORED = {Channel.adapter["fifo"].depth: 8, Channel.adapter["reg"].stages: 2}  # type: ignore[index]


def test_a_picker_naming_a_case_places_it_and_leaves_the_others_dormant() -> None:
    point = _channel({Channel.x: 0, Channel.y: 1, **STORED})  # 1: reg
    assert point.adapter_stages == 2 and point.stages_read == 2
    assert isinstance(point.adapter, Reg)
    assert dict(inspection.states(point)) == {
        "x": Set(0),
        "y": Set(1),
        "adapter.fifo.depth": Dormant(8),
        "adapter.reg.stages": Set(2),
    }
    assert _pieces(point) == ("valid", ())
    fifo = _channel({Channel.x: 0, Channel.y: 2, **STORED})  # 2: direct
    assert fifo.adapter is None and fifo.stages_read == -1
    assert isinstance(fifo.query(Channel.adapter_stages), Inapplicable)
    assert dict(inspection.states(fifo))["adapter.reg.stages"] == Dormant(2)
    # a present case's own rules count: fifo, 8 deep at width 2
    deep = _channel({Channel.x: 2, Channel.y: 2, **STORED})
    assert _pieces(deep) == (
        "refused",
        (("adapter.fifo.fits", (("too-deep", "adapter.fifo.fits"),)),),
    )


def test_a_waiting_picker_leaves_its_cases_waiting_and_only_their_rules_unknown() -> None:
    point = _channel({Channel.y: 2, Channel.adapter["fifo"].depth: 4})  # type: ignore[index]
    assert dict(inspection.states(point)) == {
        "x": Open(()),
        "y": Set(2),
        "adapter.fifo.depth": Waiting(4, ("x",)),
        "adapter.reg.stages": Open(("x",)),
    }
    assert _pieces(point) == (
        "open",
        (("adapter.fifo.fits", (("decision-unassigned", "x"),)),),
    )
    assert isinstance(point.query(Channel.adapter_stages), Unresolved)


def test_a_refused_picker_is_one_piece_in_the_selectors_name() -> None:
    point = _channel({Channel.x: 3, Channel.y: 1, **STORED})
    assert _pieces(point) == (
        "refused",
        (("adapter", (("guard-refused", "adapter"), ("chain-refused", "chain"))),),
    )
    assert dict(inspection.states(point)) == {
        "x": Set(3),
        "y": Set(1),
        "adapter.fifo.depth": Dormant(8),
        "adapter.reg.stages": Dormant(2),
    }
    assert isinstance(point.query(Channel.adapter_stages), Rejected)


def test_a_picker_naming_no_case_refuses_select_no_case_as_one_piece() -> None:
    point = _channel({Channel.x: 1, Channel.y: 2, **STORED})  # 3: zz
    assert _pieces(point) == (
        "refused",
        (("adapter", (("guard-refused", "adapter"), ("select-no-case", "adapter"))),),
    )
    assert dict(inspection.states(point))["adapter.fifo.depth"] == Dormant(8)
    # the change that names no case is refused by the selector, the picker's other read
    # standing
    report = _channel({Channel.x: 2, Channel.y: 2}).try_with_choices(x=1)
    assert not report.accepted
    assert [(b.rule, b.standing) for b in report.broken] == [("adapter", ("y",))]
    assert [f.code for f in report.broken[0].result.findings] == ["select-no-case"]


def test_a_change_flipping_the_picker_moves_its_cases_choices() -> None:
    reg = _channel({Channel.x: 0, Channel.y: 1, **STORED})  # 1: reg
    to_fifo = reg.try_with_choices(x=1, y=2)  # 3: zz
    assert not to_fifo.accepted
    to_fifo = reg.try_with_choices(x=2, y=2)  # 0: fifo, 8 deep at width 2 refused
    assert not to_fifo.accepted
    assert [(b.rule, b.standing) for b in to_fifo.broken] == [
        ("adapter.fifo.fits", ("adapter.fifo.depth",))
    ]
    to_fifo = reg.try_with_choices({Channel.x: 2, Channel.y: 2, Channel.adapter["fifo"].depth: 4})  # type: ignore[index]
    assert to_fifo.accepted
    assert (to_fifo.dormant, to_fifo.in_play, to_fifo.opened) == (
        ("adapter.reg.stages",),
        ("adapter.fifo.depth",),
        (),
    )
    back = to_fifo.instance.try_with_choices(x=0, y=1)
    assert back.accepted
    assert (back.dormant, back.in_play) == (("adapter.fifo.depth",), ("adapter.reg.stages",))


# a picker that may answer None states its semantics: ``str | None`` has none of its own
MAYBE_CASE: ValueSemantics[str | None] = ValueSemantics(
    type_token="maybe-case",
    name="str | None",
    recognizes=lambda value: value is None or type(value) is str,
    equal=lambda left, right: left == right,
    snapshot=lambda value: value,
)


def test_the_picker_never_answers_nothing() -> None:
    # a None case is named by its key; a picker answering None names no case
    class Picked(Space):
        x: int = Decision(values=(0, 1))

        @derived(semantics=MAYBE_CASE)
        def chain(self) -> str | None:
            return "fifo" if self.x == 0 else None

        adapter: Fifo | None = Select(by=chain, fifo=Fifo, direct=None, width=2)

    point = bind(design_space(Picked()), {Picked.x: 1}).instance
    assert _pieces(point) == (
        "refused",
        (("adapter", (("guard-refused", "adapter"), ("select-no-case", "adapter"))),),
    )


def test_a_derived_selector_is_listed_only_as_a_choice_and_its_cases_flow_after_its_reads() -> None:
    class Late(Space):
        early: int = Decision(values=(0, 1))

        @derived
        def chain(self) -> str:
            return ("fifo", "reg", "direct")[(self.late + self.later) % 3]

        adapter: Fifo | Reg | None = Select(by=chain, fifo=Fifo, reg=Reg, direct=None, width=4)
        late: int = Decision(values=(0, 1, 2))
        later: int = Decision(values=(0, 1))

    point = design_space(Late())
    order = ["early", "late", "later", "adapter.fifo.depth", "adapter.reg.stages"]
    assert [item.key for item in inspection.decisions(point)] == order
    assert [key for key, _ in inspection.states(point)] == order
    references = [item.reference for item in inspection.decisions(point)]
    assert [item.key for item in inspection.viable(point, references)] == order
    (choice,) = inspection.choices(point)
    assert (choice.key, choice.selector, choice.guard) == ("adapter", None, None)
    assert [(case.name, case.scope) for case in choice.cases] == [
        ("fifo", "adapter.fifo"),
        ("reg", "adapter.reg"),
        ("direct", None),
    ]
    with pytest.raises(Exception):  # noqa: B017 - no key: never a choice
        point.with_choices(adapter="fifo")
    assert isinstance(point.with_choices(late=1, later=0).adapter, Reg)


def test_a_none_case_places_nothing() -> None:
    class Holder(Space):
        x: int = Decision(values=(0, 1))

        @derived
        def chain(self) -> str:
            return ("direct", "fifo")[self.x]

        adapter: Fifo | None = Select(by=chain, direct=None, fifo=Fifo, width=2)
        stages = View(adapter.stages)  # type: ignore[union-attr]
        kind = view(case=adapter)(_name)

    point = bind(design_space(Holder()), {Holder.x: 0}).instance
    assert point.adapter is None and point.kind == "direct"
    assert isinstance(point.query(Holder.stages), Inapplicable)
    assert _pieces(point) == ("valid", ())
    fifo = point.with_choices({Holder.x: 1, Holder.adapter["fifo"].depth: 4})  # type: ignore[index]
    assert fifo.kind == "fifo" and fifo.stages == 4


# -- 4. what a Select refuses at definition -------------------------------------------------


def test_a_class_object_as_a_shared_binding_is_refused() -> None:
    with refused("select-binding-class"):
        Select(fifo=Fifo, reg=Reg, width=int)


def test_a_space_class_named_as_the_cases_formal_is_refused() -> None:
    with refused("select-binding-class"):
        Select(fifo=Fifo, reg=Reg, width=Boiler)


def test_a_select_placing_nothing_is_refused() -> None:
    with refused("select-no-node"):
        Select(direct=None)
    with refused("select-no-node"):
        Select(width=2)


def test_a_derived_selector_reads_a_value() -> None:
    with refused("select-by-not-value"):
        Select(by="fifo", fifo=Fifo, width=2)


def test_a_derived_selector_takes_no_supplied_guard() -> None:
    # Whether a formal is supplied is settled where its holder is placed, and only a chosen
    # selector reads it there: a picked one is refused it until facts are known values.
    with refused("select-by-supplied"):

        class Guarded(Space):
            w: int = Param(required=False)
            given = supplied(w)

            @derived
            def chain(self) -> str:
                return "fifo"

            adapter = Select(by=chain, when=given, fifo=Fifo, width=w)


# -- 5. a selector given as a binding's argument reads its case name ----------------------------


def _is_fifo(case: str) -> bool:
    return case == "fifo"


def _no_reg(case: str) -> bool | Rejected:
    return reject("no-reg", "reg refused") if case == "reg" else True


class ByArgument(Space):
    x: int = Decision(values=(0, 1, 2))

    @derived
    def chain(self) -> str:
        return ("direct", "fifo", "reg")[self.x]

    chosen: Fifo | Reg | None = Select(direct=None, fifo=Fifo, reg=Reg, width=2)
    picked: Fifo | Reg | None = Select(by=chain, direct=None, fifo=Fifo, reg=Reg, width=2)

    chosen_name: str = derived(case=chosen)(_name)
    picked_name: str = derived(case=picked)(_name)
    chosen_view = view(case=chosen)(_name)
    picked_view = view(case=picked)(_name)
    chosen_rule = constraint(case=chosen)(_no_reg)
    picked_rule = constraint(case=picked)(_no_reg)
    # a guard reading the selector's case through a derived value binding it
    fifo_chosen: bool = derived(case=chosen)(_is_fifo)
    fifo_only: int = Decision(values=(1, 2), when=fifo_chosen)

    @derived
    def body(self) -> str:
        # in a body the selector is still the selected case's design space
        return type(self.chosen).__name__


def _read(point: Space) -> tuple[object, ...]:
    # a view of a refused Space is refused: read as its result's kind
    return tuple(
        getattr(found, "value", type(found).__name__)
        for found in map(
            point.query,
            (
                ByArgument.chosen_name,
                ByArgument.picked_name,
                ByArgument.chosen_view,
                ByArgument.picked_view,
                ByArgument.body,
            ),
        )
    )


@pytest.mark.parametrize(
    ("x", "chosen", "read", "status"),
    [
        (0, "direct", ("direct", "direct", "direct", "direct", "NoneType"), "valid"),
        (1, "fifo", ("fifo", "fifo", "fifo", "fifo", "Fifo"), "open"),
        (1, "direct", ("direct", "fifo", "direct", "fifo", "NoneType"), "open"),
        (0, "reg", ("reg", "direct", "Rejected", "Rejected", "Reg"), "refused"),
    ],
)
def test_a_selector_bound_as_an_argument_reads_its_case_name(
    x: int, chosen: str, read: tuple[object, ...], status: str
) -> None:
    point = bind(design_space(ByArgument()), {ByArgument.x: x, ByArgument.chosen: chosen}).instance
    assert _read(point) == read
    assert inspection.validity(point).status == status
    fifo_only = point.query(ByArgument.fifo_only)
    assert isinstance(fifo_only, Unresolved if chosen == "fifo" else Inapplicable)


def test_a_change_of_the_selector_moves_the_bound_value() -> None:
    point = bind(design_space(ByArgument()), {ByArgument.x: 1, ByArgument.chosen: "fifo"}).instance
    moved = point.with_choices({ByArgument.x: 0, ByArgument.chosen: "direct"})
    assert _read(moved) == ("direct", "direct", "direct", "direct", "NoneType")
    assert isinstance(moved.query(ByArgument.fifo_only), Inapplicable)
    report = point.try_with_choices({ByArgument.x: 2, ByArgument.chosen: "reg"})
    assert not report.accepted
    assert [b.rule for b in report.broken] == ["chosen_rule", "picked_rule"]


def test_an_unchosen_selector_leaves_its_bound_value_unresolved() -> None:
    point = bind(design_space(ByArgument()), {ByArgument.x: 1}).instance
    assert isinstance(point.query(ByArgument.chosen_name), Unresolved)
    assert point.picked_name == "fifo"


def test_a_domain_reads_a_picked_selector_given_as_its_argument() -> None:
    def accepts(*, candidate: int, case: str) -> bool:
        return candidate == (1 if case == "fifo" else 2)

    def candidates(*, case: str) -> tuple[int, ...]:
        return (1, 2, 3)

    class Sized(Space):
        n: int = Param()

        @derived
        def chain(self) -> str:
            return "fifo" if self.n == 1 else "reg"

        picked = Select(by=chain, fifo=Fifo, reg=Reg, width=n)
        lanes: int = Decision(domain=domain(accepts=accepts, candidates=candidates, case=picked))

    for n, lanes in ((1, 1), (2, 2)):
        (item,) = inspection.viable(design_space(Sized(n=n)), [Sized.lanes])
        assert [value.value for value in item.values or ()] == [lanes]
