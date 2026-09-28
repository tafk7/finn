# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The refined Decision's rules, over stub kernels (case 5 and the coverage list)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from core.space._collapse_support import answers, open_space
from core.space.test_codecs import INTEGER, STRING
from refined import Decision, RefinedChoice, required, settle, unmet_required
from spike import engine_spike
from stubs import Narrow, PackedCore, ProtoKernel, StubCore, Unfinished, WideCore

from finn.core.space import (
    Available,
    ConfigurationError,
    DefinitionError,
    Inapplicable,
    Param,
    RequestError,
    SelectionSchema,
    Space,
    View,
    codec_for,
    codecs,
    composite,
    derived,
    design_space,
    inspection,
    selections,
)
from finn.kernels.configure import commit


class Unit(Space):
    """Case 5: two cores with disjoint folds; ``width`` shared, ``narrow_weights`` packed's own."""

    width: int = Param()
    compute: PackedCore | StubCore = Decision(
        {"packed": PackedCore(narrow_weights=True), "stub": StubCore}, width=width
    )
    cycles = View(compute.cycles)  # a direct read: both cores declare cycles: int
    narrow = View(compute["packed"].narrow_weights)  # a qualified read

    @derived
    def doubled(self) -> int:
        return 2 * self.compute.cycles  # in a method: the selected candidate

    @derived
    def packed_lanes(self) -> int:
        # A qualified read in a method, through the candidate's configuration.
        packed = inspection.candidate(self, Unit.compute, "packed")
        assert packed is not None
        return cast(PackedCore, packed).pe


class TodayUnit(Space):
    """The same choice in today's form: every binding written per candidate."""

    width: int = Param()
    compute: PackedCore | StubCore = Decision(
        values={
            "packed": PackedCore(width=width, narrow_weights=True),
            "stub": StubCore(width=width),
        }
    )
    cycles = View(compute.cycles)


def keys(point: Any) -> list[str]:
    return sorted(item.key for item in inspection.decisions(point))


def choose(point: Any, **values: object) -> Any:
    return commit(point, {key.replace("__", "."): value for key, value in values.items()})


# -- expansion and equivalence --------------------------------------------------------


def test_the_refined_decision_expands_to_todays_decision_over_nodes() -> None:
    refined, today = design_space(Unit(width=8)), design_space(TodayUnit(width=8))
    assert (
        keys(refined)
        == keys(today)
        == [
            "compute",
            "compute.packed.pe",
            "compute.packed.simd",
            "compute.stub.rows",
        ]
    )
    assert isinstance(Unit.compute, RefinedChoice)
    for choices in (
        {"compute": "packed", "compute.packed.pe": 2, "compute.packed.simd": 1},
        {"compute": "packed", "compute.packed.pe": 8, "compute.packed.simd": 2},
        {"compute": "stub", "compute.stub.rows": 4},
    ):
        left, right = commit(refined, choices), commit(today, choices)
        assert left.cycles == right.cycles
        assert left.compute.schedule == right.compute.schedule
    case = {"compute": "packed", "compute.packed.pe": 3, "compute.packed.simd": 1}
    for base in (refined, today):
        with pytest.raises(ValueError, match="compute.packed.pe"):
            commit(base, case)  # 3 does not divide 8, in both forms


# -- the strict errors ----------------------------------------------------------------


def test_a_shared_binding_a_candidate_lacks_is_refused_naming_it() -> None:
    with pytest.raises(DefinitionError) as raised:

        class Lacking(Space):
            width: int = Param()
            compute: PackedCore | Narrow = Decision(
                {"packed": PackedCore, "narrow": Narrow}, width=width
            )

    message = str(raised.value)
    assert "shared binding 'width' is not declared by candidates ['narrow (Narrow)']" in message
    assert "test_rules.py" in message  # the author's line, not the probe's
    with pytest.raises(DefinitionError, match=r"'cycles' is not declared by candidates \["):

        class Behaviour(Space):
            width: int = Param()
            # cycles is a view on every candidate: behaviour, not a binding.
            compute: PackedCore | StubCore = Decision(
                {"packed": PackedCore, "stub": StubCore}, width=width, cycles=width
            )


def test_a_shared_binding_also_written_on_an_entry_is_a_double_assignment() -> None:
    with pytest.raises(DefinitionError, match="candidate 'packed' already binds width") as raised:

        class Twice(Space):
            width: int = Param()
            compute: PackedCore | StubCore = Decision(
                {"packed": PackedCore(width=4), "stub": StubCore}, width=width
            )

    assert "a body sets a member once" in str(raised.value)


def test_shared_bindings_named_like_the_decisions_own_arguments_are_refused() -> None:
    for name in ("values", "domain", "semantics", "name"):
        with pytest.raises(DefinitionError, match=rf"shared bindings \['{name}'\]"):
            Decision({"packed": PackedCore}, **{name: 1})


def test_an_unsupplied_required_input_is_an_authoring_error_naming_the_candidate() -> None:
    class Open(Space):
        compute: PackedCore | StubCore = Decision({"packed": PackedCore, "stub": StubCore})

    with pytest.raises(DefinitionError) as raised:
        design_space(Open())
    message = str(raised.value)
    assert "compute.packed.width is not supplied" in message
    assert "compute.stub.width is not supplied" in message

    # Not checked at class creation: an enclosing body may still supply it.
    class Supplying(Space):
        inner = Open()
        inner.compute["packed"].width = 4
        inner.compute["stub"].width = 6

    point = commit(
        design_space(Supplying()), {"inner.compute": "stub", "inner.compute.stub.rows": 1}
    )
    assert point.inner.compute.cycles == 6


def test_an_unmet_required_member_cannot_be_placed_or_named_as_an_entry() -> None:
    assert unmet_required(Unfinished) == ("schedule",)
    assert unmet_required(PackedCore) == ()
    with pytest.raises(DefinitionError, match=r"Unfinished .* ProtoKernel\.schedule"):
        Unfinished(width=4)
    with pytest.raises(DefinitionError, match=r"candidate 'unfinished' of a Decision"):
        Decision({"packed": PackedCore, "unfinished": Unfinished})

    class Defined(Unfinished):
        schedule = "fixed"  # any attribute meets it, here a plain class attribute

    Defined(width=4)
    with pytest.raises(DefinitionError, match="ProtoKernel.schedule"):

        class Abstract(ProtoKernel):
            extra: int = required()

        Abstract(width=1)


# -- reads ----------------------------------------------------------------------------


def test_direct_reads_need_a_member_every_candidate_declares_with_one_type() -> None:
    point = commit(design_space(Unit(width=8)), {"compute": "stub", "compute.stub.rows": 2})
    assert point.cycles == 16 and point.doubled == 32
    with pytest.raises(DefinitionError, match=r"candidates \['stub'\] do not declare 'pe'") as err:
        Unit.compute.pe  # noqa: B018
    assert 'compute["packed"].pe' in str(err.value)
    with pytest.raises(DefinitionError, match="different value types"):

        class Mixed(Space):
            width: int = Param()
            compute: PackedCore | WideCore = Decision(
                {"packed": PackedCore, "wide": WideCore}, width=width
            )
            cycles = View(compute.cycles)

    with pytest.raises(AttributeError):
        Unit.compute.nothing  # noqa: B018 - no candidate declares it


def test_qualified_reads_are_inapplicable_when_another_candidate_is_selected() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 1})
    assert packed.narrow is True and packed.packed_lanes == 4
    assert packed.query(Unit.compute["packed"].pe) == Available(4)
    stub = commit(base, {"compute": "stub", "compute.stub.rows": 1})
    assert isinstance(stub.query(Unit.narrow), Inapplicable)
    assert isinstance(stub.query(Unit.compute["packed"].narrow_weights), Inapplicable)
    assert isinstance(stub.query(Unit.packed_lanes), Inapplicable)
    with pytest.raises(DefinitionError, match="no candidate 'bogus'"):
        Unit.compute["bogus"]  # noqa: B018


def test_a_candidate_handle_is_the_statically_typed_spelling_of_a_qualified_read() -> None:
    """``compute["packed"]`` is untyped for mypy (the attribute is the candidate union);
    a class attribute naming the entry is typed ``PackedCore`` and reaches the same node."""

    class Handled(Space):
        width: int = Param()
        packed = PackedCore(narrow_weights=True)  # a handle, not a second placement
        compute: PackedCore | StubCore = Decision({"packed": packed, "stub": StubCore}, width=width)
        narrow = View(packed.narrow_weights)
        indexed = View(compute["packed"].narrow_weights)

    class Design(Space):
        held = Handled(width=8)
        held.packed.pe = 2  # through the handle: the same key as held.compute["packed"].pe

    base = design_space(Handled(width=8))
    assert keys(base) == keys(design_space(Unit(width=8)))
    assert [item.key for item in inspection.pinned(design_space(Design()))] == [
        "held.compute.packed.pe"
    ]
    stub = commit(base, {"compute": "stub", "compute.stub.rows": 1})
    assert isinstance(stub.query(Handled.narrow), Inapplicable)
    packed = commit(base, {"compute": "packed"})
    assert packed.narrow is True and packed.indexed is True


# -- keys, pinning, overrides, switching ------------------------------------------------


def test_candidate_choices_are_keyed_under_the_candidate_and_inapplicable_elsewhere() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed"})
    rows = inspection.decision_handle(base, Unit.compute["stub"].rows)
    assert inspection.decision_info(base, rows).key == "compute.stub.rows"
    assert isinstance(packed.field(rows).state, Inapplicable)
    assert isinstance(
        packed.field(inspection.decision_handle(base, Unit.compute["packed"].pe)).state, Available
    )


def test_an_enclosing_body_pins_a_candidates_choice_and_overrides_its_inputs() -> None:
    class Design(Space):
        unit = Unit(width=8)
        unit.compute["packed"].pe = 2  # pins compute.packed.pe
        unit.compute["packed"].narrow_weights = False  # overrides the entry's own binding
        unit.compute["stub"].width = 6  # overrides the shared binding, for one candidate

    base = design_space(Design())
    assert keys(base) == ["unit.compute", "unit.compute.packed.simd", "unit.compute.stub.rows"]
    assert [item.key for item in inspection.pinned(base)] == ["unit.compute.packed.pe"]
    with pytest.raises(ValueError, match=r"stale choices \['unit.compute.packed.pe'\]"):
        commit(base, {"unit.compute.packed.pe": 4})
    packed = commit(base, {"unit.compute": "packed", "unit.compute.packed.simd": 1})
    assert packed.unit.cycles == 4 and packed.unit.narrow is False
    history = inspection.provenance(base, Design.unit.compute["packed"].narrow_weights)
    assert history is not None and len(history.layers) == 3  # declared, entry, Design
    stub = commit(base, {"unit.compute": "stub", "unit.compute.stub.rows": 1})
    assert stub.unit.cycles == 6


def test_switching_candidates_refuses_a_stale_candidate_choice_until_cleared() -> None:
    base = design_space(Unit(width=8))
    packed = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 1})
    assert not packed.try_with_choices({Unit.compute: "stub"}).accepted
    with pytest.raises(ConfigurationError):
        packed.with_choices({Unit.compute: "stub"})
    pe = packed.compute.field(PackedCore.pe).clear()
    simd = packed.compute.field(PackedCore.simd).clear()
    stub = packed.with_choices(pe, simd, {Unit.compute: "stub", Unit.compute["stub"].rows: 2})
    assert stub.cycles == 16


# -- narrowing and pinning the Decision itself ------------------------------------------


def test_todays_narrowing_restates_every_binding_and_cannot_drop_a_read_candidate() -> None:
    """Today an enclosing body narrows with fresh nodes, restating each binding.

    That works for value bindings it can reach (``unit.width``); for streams it
    cannot (test_real_cases). And a dropped candidate that the inner body reads
    qualified leaves that read dangling.
    """

    class Kept(Space):
        unit = Unit(width=8)
        unit.compute = Decision({"packed": PackedCore(narrow_weights=True)}, width=unit.width)

    kept = design_space(Kept())
    assert keys(kept) == ["unit.compute", "unit.compute.packed.pe", "unit.compute.packed.simd"]
    point = commit(kept, {"unit.compute": "packed", "unit.compute.packed.pe": 2})
    assert point.unit.narrow is True  # the declared candidate's reads reach the replacement

    class Dropped(Space):
        unit = Unit(width=8)
        unit.compute = Decision({"stub": StubCore}, width=unit.width)

    with pytest.raises(DefinitionError, match=r"unit\.narrow: PackedCore node .* is not placed"):
        design_space(Dropped())


def test_the_spike_pins_and_narrows_by_key_keeping_the_declared_candidates() -> None:
    with engine_spike():

        class Pinned(Space):
            unit = Unit(width=8)
            unit.compute = "stub"  # type: ignore[assignment]

        class Narrowed(Space):
            unit = Unit(width=8)
            unit.compute = Decision(values=("stub",))

        pinned, narrowed = design_space(Pinned()), design_space(Narrowed())
    assert "unit.compute" not in keys(pinned)
    assert [item.key for item in inspection.pinned(pinned)] == ["unit.compute"]
    with pytest.raises(ValueError, match=r"stale choices \['unit.compute'\]"):
        commit(pinned, {"unit.compute": "packed"})
    point = commit(pinned, {"unit.compute.stub.rows": 4})
    assert isinstance(point.unit.compute, StubCore) and point.unit.cycles == 32
    assert isinstance(point.query(Pinned.unit.compute["packed"].narrow_weights), Inapplicable)
    assert "unit.compute" in keys(narrowed)
    assert narrowed.field(
        inspection.decision_handle(narrowed, Narrowed.unit.compute)
    ).candidates() == Available(("stub",))
    with pytest.raises(ValueError):
        commit(narrowed, {"unit.compute": "packed"})
    assert settle(narrowed).committed == {"unit.compute": "stub"}
    with engine_spike(), pytest.raises(DefinitionError, match="does not add one"):

        class Added(Space):
            unit = Unit(width=8)
            unit.compute = "bogus"  # type: ignore[assignment]


# -- optional -------------------------------------------------------------------------


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
    named = Decision({"packed": PackedCore}, optional="external", width=1)
    assert list(named._space_decision().candidates) == ["external", "packed"]


# -- selections ------------------------------------------------------------------------


def test_selections_capture_and_restore_by_key_and_refuse_unknown_entries() -> None:
    base = design_space(Unit(width=8))
    point = commit(base, {"compute": "packed", "compute.packed.pe": 4, "compute.packed.simd": 2})
    handles = {item.key: item.reference for item in inspection.decisions(base)}
    schema = SelectionSchema(
        base,
        family="unit",
        version=1,
        bindings=(
            codec_for(handles["compute"], STRING),
            codec_for(handles["compute.packed.pe"], INTEGER),
            codec_for(handles["compute.packed.simd"], INTEGER),
            codec_for(handles["compute.stub.rows"], INTEGER),
        ),
    )
    document = codecs.encode(selections.capture(point), schema)
    assert [entry["key"] for entry in document["entries"]] == [  # type: ignore[index, union-attr]
        "compute",
        "compute.packed.pe",
        "compute.packed.simd",
    ]
    restored = selections.restore(design_space(Unit(width=8)), codecs.decode(document, schema))
    assert restored.accepted and restored.instance.cycles == 2
    unknown = {**document, "entries": [{**document["entries"][0], "key": "compute.bogus.pe"}]}  # type: ignore[index]
    with pytest.raises(RequestError, match="unknown selection key 'compute.bogus.pe'"):
        codecs.decode(unknown, schema)
    bogus_case = {**document, "entries": [{**document["entries"][0], "value": "bogus"}]}  # type: ignore[index]
    with pytest.raises(RequestError, match="compute: unknown structural case 'bogus'"):
        codecs.decode(bogus_case, schema)
    stale = {
        **document,
        "entries": [
            {**entry, "value": "stub"} if entry["key"] == "compute" else entry  # type: ignore[index]
            for entry in document["entries"]  # type: ignore[union-attr]
        ],
    }
    refused = selections.restore(design_space(Unit(width=8)), codecs.decode(stale, schema))
    assert not refused.accepted  # the packed-local choices are stale under "stub"


# -- collapse -------------------------------------------------------------------------


def test_every_node_answers_the_same_with_and_without_collapse() -> None:
    for choices in (
        {},
        {"compute": "packed", "compute.packed.pe": 2, "compute.packed.simd": 1},
        {"compute": "stub", "compute.stub.rows": 4},
    ):
        collapsed = open_space(Unit(width=8), collapsed=True)
        plain = open_space(Unit(width=8), collapsed=False)
        if choices:
            collapsed, plain = commit(collapsed, choices), commit(plain, choices)
        assert answers(collapsed) == answers(plain)


# -- programmatic declaration ----------------------------------------------------------


def build_unit(name: str, cores: dict[str, type[ProtoKernel]]) -> type[Space]:
    """As a graph adapter would: a composite family built at run time from facts."""
    width = Param[int]()
    members: dict[str, object] = {
        "width": width,
        "compute": Decision(dict(cores), width=width),
    }
    return composite(name, members, annotations={"width": int})


def test_a_composite_declared_at_run_time_has_the_class_bodys_stable_keys() -> None:
    first = build_unit("UnitA", {"packed": PackedCore, "stub": StubCore})
    second = build_unit("UnitB", {"packed": PackedCore, "stub": StubCore})
    body = keys(design_space(Unit(width=8)))
    assert keys(design_space(first(width=8))) == body  # type: ignore[call-arg]
    assert keys(design_space(second(width=8))) == body  # type: ignore[call-arg]
    point = commit(design_space(first(width=8)), {"compute": "stub", "compute.stub.rows": 2})  # type: ignore[call-arg]
    assert point.compute.cycles == 16  # type: ignore[attr-defined]


# -- settle ---------------------------------------------------------------------------


def test_settle_commits_exactly_one_compatible_candidate_and_leaves_the_rest_open() -> None:
    assert settle(design_space(Unit(width=128))).committed == {"compute": "stub"}
    assert settle(design_space(Unit(width=7))).committed == {"compute": "packed"}
    both = settle(design_space(Unit(width=8)))
    assert both.committed == {} and both.open == {"compute": ("packed", "stub")}
    neither = settle(design_space(Unit(width=129)))
    assert neither.committed == {} and neither.open == {"compute": ()}
    # A committed Decision is left alone; scalar Decisions are not settled.
    chosen = commit(design_space(Unit(width=8)), {"compute": "stub"})
    settled = settle(chosen)
    assert settled.committed == {} and settled.open == {}
    assert "compute.stub.rows" in keys(settled.point)


def test_settle_repeats_until_a_commitment_opens_nothing_new() -> None:
    class Chain(Space):
        width: int = Param()
        compute: PackedCore | StubCore = Decision(
            {"packed": PackedCore, "stub": StubCore}, width=width
        )

        @derived
        def after_stub(self) -> bool:
            return isinstance(self.compute, StubCore)

        post: PackedCore | StubCore = Decision(
            {"packed": PackedCore, "stub": StubCore}, width=width, when=after_stub
        )

    settled = settle(design_space(Chain(width=128)))
    assert settled.committed == {"compute": "stub", "post": "stub"}
