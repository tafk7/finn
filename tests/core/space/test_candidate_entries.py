# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A Decision over candidate entries: shared bindings once, reads, keys, required(), forcing.

Two cores with disjoint choices share one fact (``width``); the packed core
also takes a binding of its own (``narrow_weights``). The Decision names each
candidate once, as a Space class or a call carrying its own bindings, and supplies
the shared bindings to every candidate. An enclosing body pins it by a key or
narrows it by a Decision over keys, keeping the declared candidates.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from core.space._collapse_support import answers, open_space
from finn.core.space import (
    Available,
    ConfigurationError,
    ConstraintGroup,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    QueryResult,
    Rejected,
    Space,
    Users,
    View,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    required,
    selections,
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

    admission = ConstraintGroup(fits)


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

    admission = ConstraintGroup(even)


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
    compute: PackedCore | StubCore = Decision(
        {"packed": PackedCore(narrow_weights=True), "stub": StubCore}, width=width
    )
    cycles = View(compute.cycles)  # every candidate declares cycles: int
    narrow = View(compute["packed"].narrow_weights)  # type: ignore[index]

    @derived
    def doubled(self) -> int:
        return 2 * self.compute.cycles  # in a method: the selected candidate


class Earlier(Space):
    """The same choice spelled with ``values=``: every binding per candidate."""

    width: int = Param()
    compute: PackedCore | StubCore = Decision(
        {
            "packed": PackedCore(width=width, narrow_weights=True),
            "stub": StubCore(width=width),
        }
    )
    cycles = View(compute.cycles)


def keys(point: Any) -> list[str]:
    return sorted(item.key for item in inspection.decisions(point))


def commit(point: S, choices: dict[str, object]) -> S:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


def admission(candidate: Space) -> QueryResult[object] | None:
    group = getattr(type(candidate), "admission", None)
    if not isinstance(group, ConstraintGroup):
        return None
    return cast("QueryResult[object]", candidate.inspect(group).result)


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
            compute: PackedCore | Narrow = Decision(
                {"packed": PackedCore, "narrow": Narrow}, width=width
            )

    message = str(raised.value)
    assert "shared binding 'width' is not declared by candidates ['narrow (Narrow)']" in message
    assert "test_candidate_entries.py" in message
    with pytest.raises(DefinitionError, match=r"'cycles' is not declared by candidates"):

        class Behaviour(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Decision(
                {"packed": PackedCore, "stub": StubCore}, width=width, cycles=width
            )


def test_a_shared_binding_also_written_on_an_entry_is_a_double_assignment() -> None:
    with pytest.raises(DefinitionError, match="candidate 'packed' already binds width"):

        class Twice(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Decision(
                {"packed": PackedCore(width=4), "stub": StubCore}, width=width
            )


def test_shared_bindings_named_like_the_decisions_own_arguments_are_refused() -> None:
    for name in ("values", "domain", "semantics", "name"):
        with pytest.raises(DefinitionError, match=rf"shared bindings \['{name}'\]"):
            Decision({"packed": PackedCore}, **{name: 1})  # type: ignore[call-overload]
    with pytest.raises(DefinitionError, match="must be a Space class or a call"):
        Decision({"packed": 3})  # type: ignore[dict-item]
    with pytest.raises(DefinitionError, match="apply to a Decision over candidate entries"):
        Decision(values=(1, 2), optional=True)  # type: ignore[call-overload]


def test_an_unsupplied_required_input_is_an_authoring_error_naming_the_candidate() -> None:
    class Open(Space):
        compute: PackedCore | StubCore = Decision({"packed": PackedCore, "stub": StubCore})

    with pytest.raises(DefinitionError) as raised:
        design_space(Open())
    assert "compute.packed.width is not supplied" in str(raised.value)
    assert "compute.stub.width is not supplied" in str(raised.value)

    class Supplying(Space):
        inner = Open()
        inner.compute["packed"].width = 4  # type: ignore[index]
        inner.compute["stub"].width = 6  # type: ignore[index]

    point = commit(
        design_space(Supplying()), {"inner.compute": "stub", "inner.compute.stub.rows": 1}
    )
    assert point.inner.compute.cycles == 6


# -- required() --------------------------------------------------------------------------


def test_an_unmet_required_member_cannot_be_placed_or_named_as_an_entry() -> None:
    assert unmet_required(Unfinished) == ("ProtoKernel.schedule",)
    assert unmet_required(PackedCore) == ()
    with pytest.raises(DefinitionError, match=r"Unfinished .* ProtoKernel\.schedule"):
        Unfinished(width=4)
    with pytest.raises(DefinitionError, match=r"candidate 'unfinished' of a Decision"):
        Decision({"packed": PackedCore, "unfinished": Unfinished})

    class Defined(Unfinished):
        schedule = "fixed"  # any attribute meets it

    Defined(width=4)

    class Formal(Unfinished):
        schedule: str = Param(default="given")  # a formal meets it too

    assert design_space(Formal(width=4)).schedule == "given"
    with pytest.raises(DefinitionError, match="takes the member's value type"):
        required("str")  # type: ignore[arg-type]


# -- reads -------------------------------------------------------------------------------


def test_direct_reads_need_a_member_every_candidate_declares_with_one_type() -> None:
    point = commit(design_space(Unit(width=8)), {"compute": "stub", "compute.stub.rows": 2})
    assert point.cycles == 16 and point.doubled == 32
    with pytest.raises(DefinitionError, match=r"candidates \['stub'\] do not declare 'pe'") as err:
        Unit.compute.pe  # type: ignore[union-attr]  # noqa: B018
    assert 'compute["packed"].pe' in str(err.value)
    with pytest.raises(DefinitionError, match="different value types"):

        class Mixed(Space):
            width: int = Param()
            compute: PackedCore | WideCore = Decision(
                {"packed": PackedCore, "wide": WideCore}, width=width
            )
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
    with pytest.raises(DefinitionError, match="no candidate 'bogus'"):
        Unit.compute["bogus"]  # type: ignore[index]  # noqa: B018


def test_a_class_body_handle_is_the_typed_spelling_of_a_qualified_read() -> None:
    class Handled(Space):
        width: int = Param()
        packed = PackedCore(narrow_weights=True)  # a handle, not a second placement
        compute: PackedCore | StubCore = Decision({"packed": packed, "stub": StubCore}, width=width)
        narrow = View(packed.narrow_weights)
        indexed = View(compute["packed"].narrow_weights)  # type: ignore[index]

    class Design(Space):
        held = Handled(width=8)
        held.packed.pe = 2  # the same key as held.compute["packed"].pe

    base = design_space(Handled(width=8))
    assert keys(base) == keys(design_space(Unit(width=8)))
    assert [item.key for item in inspection.pinned(design_space(Design()))] == [
        "held.compute.packed.pe"
    ]
    packed = commit(base, {"compute": "packed"})
    assert packed.narrow is True and packed.indexed is True


# -- keys and enclosing bodies -----------------------------------------------------------


def test_an_enclosing_body_pins_a_candidates_choice_and_overrides_its_inputs() -> None:
    class Design(Space):
        unit = Unit(width=8)
        unit.compute["packed"].pe = 2  # type: ignore[index]
        unit.compute["packed"].narrow_weights = False  # type: ignore[index]
        unit.compute["stub"].width = 6  # type: ignore[index]

    base = design_space(Design())
    assert keys(base) == ["unit.compute", "unit.compute.packed.simd", "unit.compute.stub.rows"]
    assert [item.key for item in inspection.pinned(base)] == ["unit.compute.packed.pe"]
    packed = commit(base, {"unit.compute": "packed", "unit.compute.packed.simd": 1})
    assert packed.unit.cycles == 4 and packed.unit.narrow is False
    stub = commit(base, {"unit.compute": "stub", "unit.compute.stub.rows": 1})
    assert stub.unit.cycles == 6


def test_switching_candidates_refuses_a_stale_candidate_choice_until_cleared() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 1})
    with pytest.raises(ConfigurationError):
        packed.with_choices({Unit.compute: "stub"})
    pe = packed.compute.field(PackedCore.pe).clear()
    simd = packed.compute.field(PackedCore.simd).clear()
    stub = packed.with_choices(pe, simd, {Unit.compute: "stub", Unit.compute["stub"].rows: 2})  # type: ignore[index]
    assert stub.cycles == 16


def test_an_enclosing_body_pins_or_narrows_the_choice_by_key() -> None:
    class Pinned(Space):
        unit = Unit(width=8)
        unit.compute = "stub"  # type: ignore[assignment]

    class Narrowed(Space):
        unit = Unit(width=8)
        unit.compute = Decision(values=("stub",))  # type: ignore[assignment]

    pinned, narrowed = design_space(Pinned()), design_space(Narrowed())
    assert "unit.compute" not in keys(pinned)
    assert [item.key for item in inspection.pinned(pinned)] == ["unit.compute"]
    (choice,) = inspection.choices(pinned)
    assert choice.selector is None and [case.name for case in choice.cases] == ["packed", "stub"]
    point = commit(pinned, {"unit.compute.stub.rows": 4})
    assert isinstance(point.unit.compute, StubCore) and point.unit.cycles == 32
    assert isinstance(point.query(Pinned.unit.compute["packed"].narrow_weights), Inapplicable)  # type: ignore[index]
    assert "unit.compute" in keys(narrowed)
    handle = inspection.decision_handle(narrowed, Narrowed.unit.compute)
    assert narrowed.field(handle).candidates() == Available(("stub",))
    with pytest.raises(ConfigurationError):
        narrowed.with_choices({handle: "packed"})
    with pytest.raises(DefinitionError, match="does not add one"):

        class Added(Space):
            unit = Unit(width=8)
            unit.compute = "bogus"  # type: ignore[assignment]

    with pytest.raises(DefinitionError, match=r"Decision\(values=\(<keys>"):

        class Scalar(Space):
            unit = Unit(width=8)
            unit.compute = Decision(values=(1,))  # type: ignore[assignment]


def test_a_pinned_or_narrowed_choice_answers_the_same_with_and_without_collapse() -> None:
    class Pinned(Space):
        unit = Unit(width=8)
        unit.compute = "packed"  # type: ignore[assignment]

    class Narrowed(Space):
        unit = Unit(width=8)
        unit.compute = Decision(values=("stub",))  # type: ignore[assignment]

    for space_type, choices in (
        (Unit, {}),
        (Unit, {"compute": "packed", "compute.packed.pe": 2, "compute.packed.simd": 1}),
        (Pinned, {"unit.compute.packed.pe": 4}),
        (Narrowed, {"unit.compute": "stub", "unit.compute.stub.rows": 2}),
    ):
        root = space_type(width=8) if space_type is Unit else space_type()
        collapsed = open_space(root, collapsed=True)
        plain = open_space(
            space_type(width=8) if space_type is Unit else space_type(), collapsed=False
        )
        if choices:
            collapsed, plain = commit(collapsed, dict(choices)), commit(plain, dict(choices))
        assert answers(collapsed) == answers(plain)


# -- optional ----------------------------------------------------------------------------


def test_optional_adds_a_none_candidate_that_places_nothing() -> None:
    class Maybe(Space):
        width: int = Param()
        compute: PackedCore | StubCore | None = Decision(
            {"packed": PackedCore, "stub": StubCore}, optional=True, width=width
        )

    base = design_space(Maybe(width=8))
    (choice,) = inspection.choices(base)
    assert [case.name for case in choice.cases] == ["none", "packed", "stub"]
    assert commit(base, {"compute": "none"}).compute is None
    with pytest.raises(DefinitionError, match="'none' is the key of the None candidate"):
        Decision({"none": PackedCore}, optional=True)


# -- selections --------------------------------------------------------------------------


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


def test_selections_capture_and_restore_by_key_and_refuse_unknown_and_narrowed_cases() -> None:
    base = design_space(Unit(width=8))
    point = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 2})
    restored = selections.restore(design_space(Unit(width=8)), selections.capture(point))
    assert restored.accepted and restored.instance.cycles == 2
    assert refusal(base, {"compute": "bogus"}) == {("compute", "domain-membership")}

    class Narrowed(Space):
        unit = Unit(width=8)
        unit.compute = Decision(values=("stub",))  # type: ignore[assignment]

    narrowed = design_space(Narrowed())
    assert commit(narrowed, {"unit.compute": "stub"}).unit.compute is not None
    assert refusal(narrowed, {"unit.compute": "packed"}) == {("unit.compute", "domain-membership")}


# -- programmatic declaration ------------------------------------------------------------


def build_unit(name: str, cores: dict[str, type[ProtoKernel]]) -> type[Space]:
    """As a graph adapter would: a Space class built at run time from facts."""
    width = Param[int]()
    members: dict[str, object] = {"width": width, "compute": Decision(dict(cores), width=width)}
    return composite(name, members, annotations={"width": int})


def test_a_space_class_declared_at_run_time_has_the_class_bodys_stable_keys() -> None:
    first = build_unit("UnitA", {"packed": PackedCore, "stub": StubCore})
    assert keys(design_space(first(width=8))) == keys(design_space(Unit(width=8)))  # type: ignore[call-arg]
    point = commit(design_space(first(width=8)), {"compute": "stub", "compute.stub.rows": 2})  # type: ignore[call-arg]
    assert point.compute.cycles == 16


# -- forced cases ------------------------------------------------------------------------


def forced(point: Any) -> dict[str, object]:
    return {item.key: item.value for item in inspection.forced(point)}


def test_the_candidates_admissions_force_one_case_and_leave_several_open() -> None:
    assert forced(design_space(Unit(width=128))) == {"compute": "stub"}
    assert forced(design_space(Unit(width=7))) == {"compute": "packed"}
    assert forced(design_space(Unit(width=8))) == {}
    neither = design_space(Unit(width=129)).query(Unit.compute)
    assert isinstance(neither, Rejected)
    assert [finding.code for finding in neither.findings] == ["decision-no-viable-case"]
    # Committed, the choice is no longer forced.
    chosen = commit(design_space(Unit(width=128)), {"compute": "stub"})
    assert "compute" not in forced(chosen)


def test_a_pinned_choice_is_no_decision_and_a_narrowed_one_of_one_case_is_forced() -> None:
    class Pinned(Space):
        unit = Unit(width=8)
        unit.compute = "packed"  # type: ignore[assignment]

    class Narrowed(Space):
        unit = Unit(width=8)
        unit.compute = Decision(values=("stub",))  # type: ignore[assignment]

    assert "unit.compute" not in forced(design_space(Pinned()))
    assert forced(design_space(Narrowed()))["unit.compute"] == "stub"


# -- users through a forwarded input -----------------------------------------------------

REACH = ViewKey("reach", int)


class Target(Space):
    users = Users(REACH)

    @view
    def count(self) -> int:
        return len(self.users)


class Leaf(Space):
    target: Target = Param(required=False)

    @view
    def reach(self) -> int:
        return 1

    exports = {REACH: {target: reach}}


class Wrapper(Space):
    """References the target only to forward it to its leaf; it exports nothing."""

    target: Target = Param(required=False)
    leaf = Leaf(target=target)


def test_a_node_forwarding_an_input_is_a_user_of_the_node_it_reaches() -> None:
    class Top(Space):
        shared = Target()
        wrapper = Wrapper(target=shared)

    point = design_space(Top())
    assert [(user.node, user.member) for user in point.shared.users] == [("wrapper.leaf", "target")]
    assert point.shared.count == 1
