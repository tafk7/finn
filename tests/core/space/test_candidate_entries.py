# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A selector's cases: shared bindings once, reads, keys, required(), forcing.

Two cores with disjoint choices share one fact (``width``); the packed core
also takes a binding of its own (``narrow_weights``). The ``Select`` names each
case once, as a Space class or a call carrying its own bindings, and supplies
the shared bindings to every case. An enclosing body pins it by a key.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.refusals import refused

from core.space._choices_support import committed
from core.space._collapse_support import answers, open_space
from core.space._possible_support import possible
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    View,
    composite,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    required,
    view,
)
from finn.core.space._nodes import NodeChoice
from finn.core.space.declarations import unmet_required

S = Any


class ProtoKernel(Space):
    """Every kernel defines its schedule; the width is a fact."""

    schedule = required(str)
    width: int = Param()


class PackedCore(ProtoKernel):
    pe: int = Decision(domain=divisors_of(ProtoKernel.width))
    simd: int = Decision(values=(1, 2))
    narrow_weights: bool = Param(default=False)

    @derived
    def schedule(self) -> str:
        return f"pe{self.pe}.simd{self.simd}"

    @view
    def cycles(self) -> int:
        return self.width // self.pe

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
        return True


class StubCore(ProtoKernel):
    rows: int = Decision(values=(1, 2, 4))

    @derived
    def schedule(self) -> str:
        return f"rows{self.rows}"

    @view
    def cycles(self) -> int:
        return self.width * self.rows

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True


class WideCore(ProtoKernel):
    """``cycles`` of another type: not a member two candidates share."""

    @derived
    def schedule(self) -> str:
        return "wide"

    @view
    def cycles(self) -> str:
        return "one"


class Unfinished(ProtoKernel):
    """Leaves ``schedule`` unmet: it cannot be placed."""

    @view
    def cycles(self) -> int:
        return 1


class Narrow(Space):
    """A Space class without ``width``: it cannot take the shared binding."""

    depth: int = Param(default=2)


class Unit(Space):
    width: int = Param()
    compute: PackedCore | StubCore = Select(
        packed=PackedCore(narrow_weights=True), stub=StubCore, width=width
    )
    cycles = View(compute.cycles)  # every candidate declares cycles: int
    narrow = View(compute["packed"].narrow_weights)  # type: ignore[index]

    @derived
    def doubled(self) -> int:
        return 2 * self.compute.cycles  # in a method: the selected candidate


class Earlier(Space):
    """The same choice spelled with ``values=``: every binding per candidate."""

    width: int = Param()
    compute: PackedCore | StubCore = Select(
        packed=PackedCore(width=width, narrow_weights=True), stub=StubCore(width=width)
    )
    cycles = View(compute.cycles)


def keys(point: Any) -> list[str]:
    return sorted(item.key for item in inspection.decisions(point))


def commit(point: S, choices: dict[str, object]) -> S:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


# -- entries and shared bindings -------------------------------------------------------


def test_entries_and_shared_bindings_declare_the_earlier_choice_under_the_same_keys() -> None:
    entries, earlier = design_space(Unit(width=8)), design_space(Earlier(width=8))
    assert isinstance(Unit.compute, NodeChoice)
    assert (
        keys(entries)
        == keys(earlier)
        == [
            "compute",
            "compute.packed.pe",
            "compute.packed.simd",
            "compute.stub.rows",
        ]
    )
    for choices in (
        {"compute": "packed", "compute.packed.pe": 2, "compute.packed.simd": 1},
        {"compute": "stub", "compute.stub.rows": 4},
    ):
        left, right = commit(entries, choices), commit(earlier, choices)
        assert left.cycles == right.cycles
        assert left.compute.schedule == right.compute.schedule


def test_a_shared_binding_a_candidate_lacks_is_refused_naming_it() -> None:
    with pytest.raises(DefinitionError) as raised:

        class Lacking(Space):
            width: int = Param()
            compute: PackedCore | Narrow = Select(packed=PackedCore, narrow=Narrow, width=width)

    message = str(raised.value)
    assert "shared binding 'width' is not declared by cases ['narrow (Narrow)']" in message
    assert "test_candidate_entries.py" in message
    with refused(
        "shared-binding-invalid",
        error=DefinitionError,
        message=r"'cycles' is not declared by cases",
    ):

        class Behaviour(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Select(
                packed=PackedCore, stub=StubCore, width=width, cycles=width
            )


def test_a_shared_binding_also_written_on_an_entry_is_a_double_assignment() -> None:
    with refused(
        "shared-binding-repeated",
        error=DefinitionError,
        message="case 'packed' already binds width",
    ):

        class Twice(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Select(
                packed=PackedCore(width=4), stub=StubCore, width=width
            )


def test_an_unsupplied_required_input_is_an_authoring_error_naming_the_candidate() -> None:
    class Open(Space):
        compute: PackedCore | StubCore = Select(packed=PackedCore, stub=StubCore)

    with pytest.raises(DefinitionError) as raised:
        design_space(Open())
    assert "compute.packed.width is not supplied" in str(raised.value)
    assert "compute.stub.width is not supplied" in str(raised.value)


# -- required() --------------------------------------------------------------------------


def test_an_unmet_required_member_cannot_be_placed_or_named_as_a_case() -> None:
    assert unmet_required(Unfinished) == ("ProtoKernel.schedule",)
    assert unmet_required(PackedCore) == ()
    with refused(
        "required-unmet", error=DefinitionError, message=r"Unfinished .* ProtoKernel\.schedule"
    ):
        Unfinished(width=4)
    with refused("required-unmet", error=DefinitionError, message=r"case 'unfinished'"):
        Select(packed=PackedCore, unfinished=Unfinished)

    class Defined(Unfinished):
        schedule = "fixed"  # any attribute meets it

    Defined(width=4)

    class Formal(Unfinished):
        schedule: str = Param(default="given")  # a formal meets it too

    assert design_space(Formal(width=4)).schedule == "given"
    with refused(
        "required-type-invalid", error=DefinitionError, message="takes the member's value type"
    ):
        required("str")  # type: ignore[arg-type]


# -- reads -------------------------------------------------------------------------------


def test_direct_reads_need_a_member_every_candidate_declares_with_one_type() -> None:
    point = commit(design_space(Unit(width=8)), {"compute": "stub", "compute.stub.rows": 2})
    assert point.cycles == 16 and point.doubled == 32
    with refused(
        "direct-read-undeclared",
        error=DefinitionError,
        message=r"cases \['stub'\] do not declare 'pe'",
    ) as err:
        Unit.compute.pe  # type: ignore[union-attr]  # noqa: B018
    assert 'compute["packed"].pe' in str(err.error)
    with refused(
        "direct-read-types-differ", error=DefinitionError, message="different value types"
    ):

        class Mixed(Space):
            width: int = Param()
            compute: PackedCore | WideCore = Select(packed=PackedCore, wide=WideCore, width=width)
            cycles = View(compute.cycles)

    with pytest.raises(AttributeError):
        Unit.compute.nothing  # type: ignore[union-attr]  # noqa: B018


def test_qualified_reads_are_inapplicable_when_another_candidate_is_selected() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 1})
    assert packed.narrow is True
    assert packed.query(Unit.compute["packed"].pe) == Available(4)  # type: ignore[index]
    stub = commit(base, {"compute": "stub", "compute.stub.rows": 1})
    assert isinstance(stub.query(Unit.narrow), Inapplicable)
    assert isinstance(stub.query(Unit.compute["packed"].narrow_weights), Inapplicable)  # type: ignore[index]
    with refused("candidate-unknown", error=DefinitionError, message="no case 'bogus'"):
        Unit.compute["bogus"]  # type: ignore[index]  # noqa: B018


def test_a_class_body_handle_naming_a_case_is_refused() -> None:
    """A class attribute naming a case's node places it twice: the case's path names it."""
    with refused(
        "select-case-handle",
        error=DefinitionError,
        message=r"Handled\.packed: names case 'packed' of a selector .* placed twice",
    ) as caught:

        class Handled(Space):
            width: int = Param()
            packed = PackedCore(narrow_weights=True)
            compute: PackedCore | StubCore = Select(packed=packed, stub=StubCore, width=width)

    assert '<selector>["packed"].<member>' in str(caught.error)
    with refused("select-case-handle", error=DefinitionError, message="Indexed.packed"):

        class Indexed(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Select(packed=PackedCore, stub=StubCore, width=width)
            packed = compute["packed"]  # type: ignore[index]


# -- keys ----------------------------------------------------------------------------------


def test_switching_candidates_keeps_a_stale_candidate_choice_dormant() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 1})
    switched = packed.with_choices({Unit.compute: "stub"})
    pe = Unit.compute["packed"].pe  # type: ignore[index]
    assert isinstance(switched.query(pe), Inapplicable)  # dormant: kept, out of play
    assert committed(switched)["compute.packed.pe"] == 4
    assert switched.with_choices({Unit.compute: "packed"}).compute.pe == 4  # back in play
    cleared_pe = packed.compute.field(PackedCore.pe).clear()
    cleared_simd = packed.compute.field(PackedCore.simd).clear()
    stub = packed.with_choices(
        cleared_pe,
        cleared_simd,
        {Unit.compute: "stub", Unit.compute["stub"].rows: 2},  # type: ignore[index]
    )
    assert stub.cycles == 16 and "compute.packed.pe" not in committed(stub)


def test_a_choice_answers_the_same_with_and_without_collapse() -> None:
    for choices in (
        {},
        {"compute": "packed", "compute.packed.pe": 2, "compute.packed.simd": 1},
        {"compute": "stub", "compute.stub.rows": 2},
    ):
        collapsed = open_space(Unit(width=8), collapsed=True)
        plain = open_space(Unit(width=8), collapsed=False)
        if choices:
            collapsed, plain = commit(collapsed, dict(choices)), commit(plain, dict(choices))
        assert answers(collapsed) == answers(plain)


# -- a None case ---------------------------------------------------------------------------


def test_a_none_case_places_nothing() -> None:
    class Maybe(Space):
        width: int = Param()
        compute: PackedCore | StubCore | None = Select(
            none=None, packed=PackedCore, stub=StubCore, width=width
        )

    base = design_space(Maybe(width=8))
    (choice,) = inspection.choices(base)
    assert [case.name for case in choice.cases] == ["none", "packed", "stub"]
    assert commit(base, {"compute": "none"}).compute is None


# -- replay ------------------------------------------------------------------------------


def refusal(point: Space, choices: dict[str, object]) -> set[tuple[str, str]]:
    """Each refused key of ``choices`` and its finding codes."""
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    report = point.try_with_choices({owned[key]: value for key, value in choices.items()})
    assert not report.accepted
    return {
        (outcome.owner, finding.code)
        for outcome in report.outcomes
        if outcome.status == "refused"
        for finding in getattr(outcome.result, "findings", ())
    }


def test_commitments_replay_by_key_and_an_unknown_case_is_refused() -> None:
    base = design_space(Unit(width=8))
    point = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 2})
    replayed = commit(design_space(Unit(width=8)), committed(point))
    assert replayed.cycles == 2
    assert refusal(base, {"compute": "bogus"}) == {("compute", "domain-membership")}


# -- programmatic declaration ------------------------------------------------------------


def build_unit(name: str, cores: dict[str, type[ProtoKernel]]) -> type[Space]:
    """As a graph adapter would: a Space class built at run time from facts."""
    width = Param[int]()
    members: dict[str, object] = {"width": width, "compute": Select(**cores, width=width)}
    return composite(name, members, annotations={"width": int})


def test_a_space_class_declared_at_run_time_has_the_class_bodys_stable_keys() -> None:
    first = build_unit("UnitA", {"packed": PackedCore, "stub": StubCore})
    assert keys(design_space(first(width=8))) == keys(design_space(Unit(width=8)))  # type: ignore[call-arg]
    point = commit(design_space(first(width=8)), {"compute": "stub", "compute.stub.rows": 2})  # type: ignore[call-arg]
    assert point.compute.cycles == 16


# -- forced cases ------------------------------------------------------------------------


def test_the_candidates_validity_fills_in_no_case() -> None:
    for width in (128, 7, 8, 129):
        answer = design_space(Unit(width=width)).query(Unit.compute)
        assert isinstance(answer, Unresolved)
        assert [finding.code for finding in answer.findings] == ["decision-unassigned"]


def test_the_candidates_validity_refuses_a_case_and_leaves_several_possible() -> None:
    assert possible(design_space(Unit(width=128)), Unit.compute) == ("stub",)
    assert possible(design_space(Unit(width=7)), Unit.compute) == ("packed",)
    assert possible(design_space(Unit(width=8)), Unit.compute) == ("packed", "stub")
    assert possible(design_space(Unit(width=129)), Unit.compute) == ()
