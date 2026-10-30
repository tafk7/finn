# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Finite shell counts over ordinary model bindings, without a timing query."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Decision,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    bind,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.core.space.errors import DefinitionError
from finn.core.space.extensions import composite
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.budgets import ShellBudgets, contributor, counts, shell_members
from finn.kernels.movers import MoverContract
from finn.platform.capabilities import Resources


def shell_type(
    *,
    presented: Sequence[str] = (),
    movers: Sequence[str] = (),
    initiators: Sequence[str] = (),
    doubled: Sequence[str] = (),
) -> type[ShellBudgets]:
    members, annotations = shell_members(
        presented=presented, movers=movers, initiators=initiators, doubled=doubled
    )
    return composite("FiniteShell", members, annotations=annotations, base=ShellBudgets)


def shell_point(kind: type[ShellBudgets], **values: object) -> ShellBudgets:
    return cast(
        ShellBudgets,
        design_space(
            cast(Any, kind)(
                **{
                    "shell": "literal",
                    "control_budget": None,
                    "memory_ports": None,
                    "clk2x": True,
                    **values,
                }
            )
        ),
    )


def mover(control: int, memory: int) -> MoverContract:
    # Actual immutable mover data. Distinct non-control fields make accidental resource,
    # memory or clock counting visible; no timing property is supplied or evaluated.
    return MoverContract(
        kind="test",
        direction="in",
        port="stream",
        tdata=8,
        beats=5,
        lanes=1,
        element=ScalarEncoding(DataType["INT8"]),
        memory_width=32,
        words=3,
        converter=True,
        call_cycles=17,
        frames_per_call=2,
        control_buses=control,
        memory_ports=memory,
        resources=Resources(lut=43, ff=29, bram18=7, dsp=11),
    )


@pytest.mark.parametrize("size", [0, 1, 4])
def test_actual_shell_counts_match_independent_arithmetic(size: int) -> None:
    names = tuple(f"k{index}" for index in range(size))
    presented = tuple(index % 2 == 0 for index in range(size))
    memory = tuple(index + 2 for index in range(size))
    doubled = tuple(index == size - 1 for index in range(size))
    tuples = tuple((mover(index, 19), mover(2, 23)) for index in range(size))
    values = {
        contributor(name, kind): value
        for kind, items in (
            ("presented", presented),
            ("initiators", memory),
            ("doubled", doubled),
            ("mover", tuples),
        )
        for name, value in zip(names, items)
    }
    point = shell_point(
        shell_type(presented=names, movers=names, initiators=names, doubled=names), **values
    )
    # These expected counts use supplied numbers, independently of Group outputs.
    expected_control = (size + 1) // 2 + size * (size - 1) // 2 + 2 * size
    expected_memory = size * (size - 1) // 2 + 2 * size
    assert counts(point) == Available((expected_control, expected_memory, size > 0))
    assert point.presented_buses == (size + 1) // 2
    assert point.mover_buses == size * (size - 1) // 2 + 2 * size
    assert point.memory_initiators == expected_memory
    assert point.doubled_clock is (size > 0)
    assert inspection.validity(point).status == "valid"
    assert inspection.statistics(point).owning_decisions == 0
    assert dict(inspection.presence(point)) == {"": Available(True)}
    assert all(node.scope == "" for node in inspection.members(point))


@pytest.mark.parametrize("budget", [None, 0, 2])
@pytest.mark.parametrize("contributions", [(False, 0), (True, 3)])
def test_budget_limits_and_original_labels(
    budget: int | None, contributions: tuple[bool, int]
) -> None:
    bus, memory = contributions
    kind = shell_type(presented=("k0",), initiators=("k0",))
    point = shell_point(
        kind,
        **{"k0-presented": bus, "k0-initiators": memory},
        control_budget=budget,
        memory_ports=budget,
    )
    assert point.shell == "literal"
    assert counts(point) == Available((int(bus), memory, False))
    for reference, amount, message in (
        (ShellBudgets.within_control_budget, int(bus), "presents 1 and its movers 0"),
        (ShellBudgets.within_memory_ports, memory, "initiates 3"),
    ):
        result = cast(QueryResult[bool], point.query(reference))
        if budget is not None and amount > budget:
            assert isinstance(result, Rejected)
            assert len(result.findings) == 1
            finding = result.findings[0]
            assert finding.code == "interface-budget-exceeded"
            assert finding.owner in ("within_control_budget", "within_memory_ports")
            assert "'literal' shell" in finding.message and message in finding.message
        else:
            assert isinstance(result, Available) and result.value is True


def test_none_budgets_exit_before_open_counts() -> None:
    kind = shell_type(presented=("open",), initiators=("open",))
    host = bound_shell(
        kind, {"open-presented": Inputs.first_bool, "open-initiators": Inputs.first_int}
    )
    point = cast(Any, design_space(host())).shell_counts
    assert inspection.validity(point).status == "valid"
    assert isinstance(counts(point), Unresolved)
    for reference in (ShellBudgets.within_control_budget, ShellBudgets.within_memory_ports):
        evidence = inspection.explain(point, reference)
        assert evidence.result == Available(True)
        assert not any(
            node.declaration.key.endswith(("presented_buses", "mover_buses", "memory_initiators"))
            for node in evidence.nodes
        )


class Inputs(Space):
    first_int: int = Decision(domain=(2, 5, 11))
    later_int: int = Decision(domain=(2, 5, 11))
    last_int: int = Decision(domain=(2, 5, 11))
    first_bool: bool = Decision(domain=(False, True))
    later_bool: bool = Decision(domain=(False, True))
    moves: tuple[MoverContract, ...] = Param(default=())

    @derived
    def failed_int(self) -> int | Rejected:
        return reject("contributor-refused", "the actual model count is refused")

    @derived
    def failed_bool(self) -> bool | Rejected:
        return reject("contributor-refused", "the actual doubled-clock model is refused")

    @derived
    def direct_sum(self) -> int:
        head = self.first_int
        middle = self.later_int
        last = self.last_int
        return head + middle + last

    @derived
    def direct_bool(self) -> bool:
        head = self.first_bool
        tail = self.later_bool
        return head or tail

    @derived
    def sum_before_failure(self) -> int:
        head = self.first_int
        tail = self.failed_int
        return head + tail

    @derived
    def bool_before_failure(self) -> bool:
        head = self.first_bool
        tail = self.failed_bool
        return head or tail

    @view
    def readout(self) -> int:
        return self.first_int


def bound_shell(kind: type[ShellBudgets], values: Mapping[str, object]) -> type[Inputs]:
    class Host(Inputs):
        shell_counts = cast(Any, kind)(
            shell="bound", control_budget=None, memory_ports=None, clk2x=True, **values
        )

    return Host


def demand(point: Space, reference: object, inputs: set[str]) -> tuple[object, list[str]]:
    """Read ordered evidence edges, keeping only the hand-named source operands."""
    evidence = inspection.explain(point, reference)
    nodes = {node.declaration.key: node for node in evidence.nodes}
    ordered: list[str] = []

    def visit(key: str) -> None:
        if key in inputs:
            ordered.append(key)
        for dependency in nodes[key].dependencies:
            visit(dependency._node.key)

    visit(evidence.query.key)
    return evidence.result, ordered


@pytest.mark.parametrize(
    "values", [{}, {"first_int": 2}, {"first_int": 2, "later_int": 5, "last_int": 11}]
)
def test_sum_demand_matches_hand_authored_expression(values: dict[str, int]) -> None:
    kind = shell_type(initiators=("first", "later", "last"))
    host = bound_shell(
        kind,
        {
            "first-initiators": Inputs.first_int,
            "later-initiators": Inputs.later_int,
            "last-initiators": Inputs.last_int,
        },
    )
    point = design_space(host()).with_choices(
        {getattr(Inputs, name): value for name, value in values.items()}
    )
    inputs = {"first_int", "later_int", "last_int"}
    folded = demand(point, cast(Any, host).shell_counts.memory_initiators, inputs)
    direct = demand(point, Inputs.direct_sum, inputs)
    assert folded == direct
    assert folded[1] == (
        ["first_int"]
        if not values
        else ["first_int", "later_int"]
        if len(values) == 1
        else ["first_int", "later_int", "last_int"]
    )


@pytest.mark.parametrize("later", [None, False, True])
def test_true_boolean_head_still_demands_open_tail(later: bool | None) -> None:
    kind = shell_type(doubled=("first", "later"))
    host = bound_shell(
        kind, {"first-doubled": Inputs.first_bool, "later-doubled": Inputs.later_bool}
    )
    values = {"first_bool": True, **({} if later is None else {"later_bool": later})}
    point = design_space(host()).with_choices(
        {getattr(Inputs, name): value for name, value in values.items()}
    )
    inputs = {"first_bool", "later_bool"}
    folded = demand(point, cast(Any, host).shell_counts.doubled_clock, inputs)
    assert folded == demand(point, Inputs.direct_bool, inputs)
    assert folded[1] == ["first_bool", "later_bool"]
    assert isinstance(folded[0], Unresolved) if later is None else folded[0] == Available(True)


@pytest.mark.parametrize("boolean", [False, True])
def test_real_refused_tail_preserves_eager_boolean_and_sum_owner(boolean: bool) -> None:
    kind = shell_type(doubled=("first", "later"), initiators=("first", "later"))
    host = bound_shell(
        kind,
        {
            "first-doubled": Inputs.first_bool,
            "later-doubled": Inputs.failed_bool,
            "first-initiators": Inputs.first_int,
            "later-initiators": Inputs.failed_int,
        },
    )
    point = design_space(host()).with_choices({Inputs.first_bool: boolean, Inputs.first_int: 2})
    for folded, manual, owner in (
        (cast(Any, host).shell_counts.doubled_clock, Inputs.bool_before_failure, "failed_bool"),
        (cast(Any, host).shell_counts.memory_initiators, Inputs.sum_before_failure, "failed_int"),
    ):
        result = point.query(folded)
        assert result == point.query(manual) and isinstance(result, Rejected)
        assert [(each.code, each.owner) for each in result.findings] == [
            ("contributor-refused", owner)
        ]


def test_open_sum_head_halts_before_later_refusal() -> None:
    kind = shell_type(initiators=("first", "later"))
    host = bound_shell(
        kind, {"first-initiators": Inputs.first_int, "later-initiators": Inputs.failed_int}
    )
    point = design_space(host())
    folded = demand(
        point, cast(Any, host).shell_counts.memory_initiators, {"first_int", "failed_int"}
    )
    assert folded == demand(point, Inputs.sum_before_failure, {"first_int", "failed_int"})
    assert isinstance(folded[0], Unresolved) and folded[1] == ["first_int"]
    assert "first_int" in {finding.owner for finding in folded[0].findings}


def test_movers_are_bound_model_tuples_and_only_project_control_buses() -> None:
    kind = shell_type(movers=("actual",))
    host = bound_shell(kind, {"actual-mover": Inputs.moves})
    point = design_space(host(moves=(mover(2, 47), mover(3, 53))))
    assert counts(cast(Any, point).shell_counts) == Available((5, 0, False))
    evidence = inspection.explain(point, cast(Any, host).shell_counts.mover_buses)
    assert any(node.declaration.key == "moves" for node in evidence.nodes)
    assert not any(
        "cycles" in node.declaration.key or "times" in node.declaration.key
        for node in evidence.nodes
    )


def test_model_binding_does_not_accept_a_view() -> None:
    kind = shell_type(initiators=("bad",))
    host = bound_shell(kind, {"bad-initiators": Inputs.readout})
    with pytest.raises(DefinitionError, match="reads the view readout") as error:
        design_space(host())
    assert [(finding.code, finding.owner) for finding in error.value.findings] == [
        ("view-read-in", "shell_counts.bad-initiators")
    ]


@pytest.mark.parametrize(
    "permuted,eager", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fresh_model_placements_preserve_counts_and_have_no_choices(
    monkeypatch: pytest.MonkeyPatch, permuted: bool, eager: bool
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", str(int(permuted)))
    monkeypatch.setenv("FINN_SPACE_EAGER", str(int(eager)))
    left = shell_type(presented=("k",), doubled=("k",))
    right = shell_type(presented=("k",), doubled=("k",))
    assert left is not right
    point = shell_point(left, **{"k-presented": True, "k-doubled": False})
    other = shell_point(right, **{"k-presented": False, "k-doubled": True})
    assert counts(point) == Available((1, 0, False))
    assert counts(other) == Available((0, 0, True))
    assert counts(point) == Available((1, 0, False))
    assert inspection.decisions(point) == ()
    assert dict(inspection.presence(point)) == {"": Available(True)}
    assert inspection.statistics(point).scopes == 1


def test_clock_bypass_and_original_contributor_rule_names_remain() -> None:
    kind = shell_type(doubled=("first", "later"))
    host = bound_shell(
        kind, {"first-doubled": Inputs.first_bool, "later-doubled": Inputs.later_bool}
    )
    supplied = cast(Any, design_space(host())).shell_counts
    assert inspection.validity(supplied).status == "valid"  # clk2x skips both open inputs
    assert isinstance(supplied.query(kind.doubled_clock), Unresolved)
    point = shell_point(kind, **{"first-doubled": True, "later-doubled": False}, clk2x=False)
    result = point.query(getattr(kind, "first-clock"))
    assert isinstance(result, Rejected)
    assert [(each.code, each.owner) for each in result.findings] == [
        ("clock-unavailable", "first-clock")
    ]
    assert point.query(getattr(kind, "later-clock")) == Available(True)
    members, annotations = shell_members(doubled=("first", "later"))
    assert annotations == {"first-doubled": bool, "later-doubled": bool}
    assert {
        "first-clock",
        "later-clock",
        "first-doubled",
        "later-doubled",
        "presented_buses",
        "mover_buses",
        "memory_initiators",
        "doubled_clock",
    } <= members.keys()


class OffsetInt(int):
    """An admitted immutable int subtype whose raw addition is observably different."""

    __slots__ = ()

    def __add__(self, other: int, /) -> int:
        return int(self) + int(other) + 100


@pytest.mark.parametrize("subtyped", [False, True])
def test_initiators_project_each_int_head_before_adding(subtyped: bool) -> None:
    first = OffsetInt(2) if subtyped else 2
    second = OffsetInt(3) if subtyped else 3
    assert first + 0 == (102 if subtyped else 2)  # independent arithmetic witness
    kind = shell_type(initiators=("first", "second"))
    point = shell_point(
        kind, **{"first-initiators": first, "second-initiators": second}, memory_ports=5
    )
    # Both supplied subtypes must be converted before addition, as the original fold did.
    assert point.memory_initiators == int(first) + int(second) == 5
    assert counts(point) == Available((0, 5, False))
    admitted = cast(QueryResult[bool], point.query(ShellBudgets.within_memory_ports))
    assert admitted == Available(True)
    over = shell_point(
        kind, **{"first-initiators": first, "second-initiators": second}, memory_ports=4
    )
    result = over.query(ShellBudgets.within_memory_ports)
    assert isinstance(result, Rejected)
    assert [(each.code, each.owner) for each in result.findings] == [
        ("interface-budget-exceeded", "within_memory_ports")
    ]


class ClockInputs(Space):
    supplied: bool = Decision(domain=(False, True))
    doubled: bool = Decision(domain=(False, True))
    row: str = Param(default="named")
    want_refusal: bool = Param(default=False)
    row_refusal: bool = Param(default=False)

    @derived
    def refused(self) -> bool | Rejected:
        return reject("clock-input-refused", "the actual clock model input is refused")

    @derived
    def refused_row(self) -> str | Rejected:
        return reject("shell-input-refused", "the actual shell model input is refused")

    @derived
    def actual_doubled(self) -> bool:
        if self.want_refusal:
            return self.refused
        return self.doubled

    @derived
    def actual_row(self) -> str:
        if self.row_refusal:
            return self.refused_row
        return self.row

    @derived
    def direct_clock(self) -> bool | Rejected:
        # Independent literal source expression, with exactly the original branch order.
        if self.supplied or not self.actual_doubled:
            return True
        return reject(
            "clock-unavailable",
            f"the partition takes an aligned doubled clock (ap_clk2x), which the "
            f"{self.actual_row!r} shell does not supply",
        )


@pytest.mark.parametrize(
    "permuted,eager", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "supplied,doubled,refused_input,refused_row,expected_reads,status",
    [
        (True, None, False, True, ["supplied"], Available),
        (True, None, True, True, ["supplied"], Available),
        (False, False, False, True, ["supplied", "doubled"], Available),
        (False, True, False, False, ["supplied", "doubled", "row"], Rejected),
        (None, None, True, True, ["supplied"], Unresolved),
        (False, None, False, True, ["supplied", "doubled"], Unresolved),
        (False, None, True, True, ["supplied", "refused"], Rejected),
        (False, True, False, True, ["supplied", "doubled", "refused_row"], Rejected),
    ],
    ids=[
        "supplied-open",
        "supplied-refused",
        "false-skips-row",
        "required",
        "open-supply",
        "open-doubled",
        "refused-doubled",
        "refused-row",
    ],
)
def test_clock_requirement_preserves_exact_source_order_and_owners(
    supplied: bool | None,
    doubled: bool | None,
    refused_input: bool,
    refused_row: bool,
    expected_reads: list[str],
    status: type[object],
    monkeypatch: pytest.MonkeyPatch,
    permuted: bool,
    eager: bool,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", str(int(permuted)))
    monkeypatch.setenv("FINN_SPACE_EAGER", str(int(eager)))
    made = shell_type(doubled=("k0",))

    class Host(ClockInputs):
        counted_shell = cast(Any, made)(
            shell=ClockInputs.actual_row,
            clk2x=ClockInputs.supplied,
            control_budget=None,
            memory_ports=None,
            **{"k0-doubled": ClockInputs.actual_doubled},
        )

    choices = {}
    if supplied is not None:
        choices[ClockInputs.supplied] = supplied
    if doubled is not None and not refused_input:
        choices[ClockInputs.doubled] = doubled
    initial = design_space(Host(want_refusal=refused_input, row_refusal=refused_row))
    trial = initial.try_with_choices(choices)
    assert trial.accepted is (status is not Rejected)
    # Saved-choice binding retains an invalid configuration for inspection; it does
    # not claim that the ordinary admitted change accepted the failing clock.
    saved = bind(initial, choices)
    assert saved.accepted
    point = saved.instance
    inputs = {"supplied", "doubled", "row", "refused", "refused_row"}
    folded, reads = demand(point, getattr(Host.counted_shell, "k0-clock"), inputs)
    direct, direct_reads = demand(point, ClockInputs.direct_clock, inputs)
    assert reads == direct_reads == expected_reads
    assert isinstance(folded, status) and isinstance(direct, status)
    if isinstance(folded, Rejected) and isinstance(direct, Rejected):
        assert [(f.code, f.message) for f in folded.findings] == [
            (f.code, f.message) for f in direct.findings
        ]
        assert [f.owner for f in folded.findings] == (
            ["counted_shell.k0-clock"]
            if expected_reads[-1] == "row"
            else ["refused_row"]
            if expected_reads[-1] == "refused_row"
            else ["refused"]
        )
    else:
        assert folded == direct
