# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Sparse ownership, detached captures, and atomic replay on empty roots."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from finn.core.space import (
    Decision,
    Param,
    Selection,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    ValueSemantics,
    compile_space,
    derived,
    divisors_of,
    domain,
    inspection,
    selections,
)
from finn.core.space.errors import RequestError


def test_capture_child_keeps_each_committed_root_owner_once_and_includes_first_value() -> None:
    class Child(Space):
        supplied = Param(int)
        local = Decision(str, values=("auto", "block"))

    class Family(Space):
        factor = Decision(int, values=(1, 2))
        left = Subspace(Child, supplied=factor)
        right = Subspace(Child, supplied=factor)

        @derived
        def doubled(*, factor: int) -> int:
            return factor * 2

    model = compile_space(Family)
    base = model.bind()
    first = base.with_choices(factor=1)
    point = first.with_choices(first.field(Family.left.decision_ref(Child.local)).change("auto"))
    assert point.doubled == 2
    captured = selections.capture(point.left)
    assert captured.keys == ("factor", "left.local")
    assert [entry.value for entry in captured.entries] == [1, "auto"]
    assert selections.capture(base).keys == ()
    replay = selections.restore(base, captured)
    assert replay.accepted
    assert selections.capture(replay.instance) == captured
    assert isinstance(replay.instance.right.query(Child.local), Unresolved)


def test_configuration_edits_can_be_captured_without_changing_earlier_points() -> None:
    class Family(Space):
        factor = Decision(int, values=(1, 2))
        style = Decision(str, values=("small", "fast"))

    base = Family()
    point = base.with_choices(factor=1, style="small")
    original = selections.capture(point)
    revised = point.with_choices(point.field(Family.style).clear(), factor=2)
    edited = selections.capture(revised)
    assert original.keys == ("factor", "style")
    assert edited.keys == ("factor",)
    assert original.value(Family.factor) == 1
    assert edited.value(Family.factor) == 2
    report = selections.restore(base, edited)
    assert report.accepted and report.instance.factor == 2
    assert point.factor == 1 and point.style == "small"
    for saved in (original, edited, selections.capture(base)):
        with pytest.raises(RequestError, match="no committed choices"):
            selections.restore(point, saved)
    with pytest.raises(KeyError, match="style"):
        edited.value(Family.style)


def test_rebound_inputs_can_refuse_an_earlier_selection_without_partial_publication() -> None:
    class Family(Space):
        extent = Param(int)
        factor = Decision(int, domain=divisors_of(extent))
        style = Decision(str, values=("auto", "block"))

    model = compile_space(Family)
    first = model.bind({Family.extent: 12}).with_choices(factor=4).with_choices(style="block")
    changed = model.bind({Family.extent: 10})
    report = selections.restore(changed, selections.capture(first))
    assert not report.accepted and report.instance is changed
    assert selections.capture(changed).keys == ()
    assert first.factor == 4 and first.extent == 12


def test_selector_change_requires_explicit_case_clearing_before_capture() -> None:
    class Child(Space):
        lanes = Decision(int, values=(1, 2))

    class Family(Space):
        implementation = SubspaceChoice({"left": Subspace(Child), "right": Subspace(Child)})

    base = Family()
    selector = inspection.choices(base)[0].selector
    assert selector is not None
    child = base.implementation.select("left").alternative("left")
    left_lanes = inspection.decision_handle(child, Child.lanes)
    point = child.with_choices(lanes=1).root
    captured = selections.capture(point)
    assert captured.keys == ("implementation", "implementation.left.lanes")
    refused = point.try_with_choices(point.field(selector).change("right"))
    assert not refused.accepted and refused.instance is point
    assert selections.capture(point) == captured
    revised = point.with_choices(
        point.field(left_lanes).clear(), point.field(selector).change("right")
    )
    cleaned = selections.capture(revised)
    accepted = selections.restore(base, cleaned)
    assert accepted.accepted and cleaned.keys == ("implementation",)


def test_singleton_choices_do_not_create_persisted_selectors() -> None:
    class Child(Space):
        lanes = Decision(int, values=(1,))

    class Family(Space):
        implementation = SubspaceChoice({"only": Subspace(Child)})

    point = Family().implementation.alternative("only").with_choices(lanes=1)
    captured = selections.capture(point)
    assert len(captured.entries) == 1
    assert captured.entries[0].key.endswith("lanes")
    assert not inspection.decision_info(point, Child.lanes).selector


@dataclass
class Payload:
    values: list[int]


PAYLOAD = ValueSemantics(
    Payload,
    "payload",
    lambda value: type(value) is Payload,
    lambda left, right: sorted(left.values) == sorted(right.values),
    lambda value: Payload(list(value.values)),
)


def test_capture_and_public_entries_detach_mutable_payloads() -> None:
    class Family(Space):
        payload = Decision(PAYLOAD, values=(Payload([1, 2]), Payload([3])))

    base = Family()
    source = Payload([2, 1])
    point = base.with_choices(payload=source)
    captured = selections.capture(point)
    source.values.append(99)
    public = captured.entries[0].value
    assert isinstance(public, Payload)
    public.values.append(99)
    captured.value(Family.payload).values.append(99)
    assert captured.value(Family.payload).values == [2, 1]
    replacement = Payload([3])
    edited = selections.capture(point.with_choices(payload=replacement))
    replacement.values.append(99)
    assert edited.value(Family.payload).values == [3]
    assert selections.restore(base, edited).instance.payload.values == [3]
    assert point.payload.values == [2, 1]
    equal = selections.capture(point.with_choices(payload=Payload([1, 2])))
    assert equal == captured


def test_foreign_models_invalid_selections_and_child_restore_are_rejected() -> None:
    class Child(Space):
        value = Decision(int, values=(1, 2))

    class Family(Space):
        first = Subspace(Child)
        second = Subspace(Child)

    class OtherFamily(Space):
        first = Subspace(Child)
        second = Subspace(Child)

    point = Family()
    point = point.with_choices(point.field(Family.first.decision_ref(Child.value)).change(1))
    saved = selections.capture(point)
    other = OtherFamily()
    with pytest.raises(RequestError, match="different compiled model"):
        selections.restore(other, saved)
    with pytest.raises(RequestError, match="root configuration"):
        selections.restore(point.first, saved)
    with pytest.raises(RequestError):
        saved.value(Child.value)
    with pytest.raises(RequestError, match="different compiled model"):
        foreign = inspection.decision_handle(other, OtherFamily.first.decision_ref(Child.value))
        saved.value(foreign)
    with pytest.raises(RequestError, match="Selection"):
        selections.restore(other, cast(Selection, object()))


def test_capture_does_not_evaluate_unrelated_uncommitted_guard_callbacks() -> None:
    class Family(Space):
        committed = Decision(int, values=(1,))

        @derived
        def explosive() -> bool:
            raise AssertionError("unrelated capture must not query this guard")

        unrelated = Decision(int, values=(1,), when=explosive)

    point = Family().with_choices(committed=1)
    assert selections.capture(point).keys == ("committed",)


def test_selection_reads_replay_and_codec_bindings_have_strict_types(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    project = Path(__file__).resolve().parents[3]
    common = """from typing import assert_type
from finn.core.space import (
    Decision, Param, Space, ValueCodec, JSONValue, codec_for, compile_space,
    selections, Selection, ConfigurationResult,
)

class Family(Space):
    extent = Param(int)
    factor = Decision(int, values=(1, 2))
    style = Decision(str, values=("auto", "block"))

def integer(value: JSONValue) -> int:
    if type(value) is not int:
        raise ValueError("integer required")
    return value

integer_codec: ValueCodec[int] = ValueCodec("integer", 1, lambda value: value, integer)
base = compile_space(Family).bind({Family.extent: 4})
selected = selections.capture(base)
"""
    positive = (
        common
        + """assert_type(selections.capture(base), Selection)
assert_type(selected.value(Family.factor), int)
assert_type(selections.restore(base, selected), ConfigurationResult[Family])
codec_for(Family.factor, integer_codec)
"""
    )
    negative = (
        common
        + """selected.value(Family.extent)  # E
selected.edit(Family.factor, 2)  # E
selected.remove(Family.factor)  # E
codec_for(Family.style, integer_codec)  # E
selected.with_changes([])  # E

"""
    )
    environment = dict(os.environ, MYPYPATH=f"{project / 'src'}:{project / 'tests'}")
    command = [
        mypy,
        "--strict",
        "--no-incremental",
        "--explicit-package-bases",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    for name, source in (("positive", positive), ("negative", negative)):
        fixture = tmp_path / f"{name}.py"
        fixture.write_text(source)
        result = subprocess.run(
            [*command, str(fixture)],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if name == "positive":
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            expected = {index for index, line in enumerate(source.splitlines(), 1) if "# E" in line}
            actual = {
                int(line) for line in re.findall(r"negative\.py:(\d+): error:", result.stdout)
            }
            assert result.returncode == 1, result.stdout + result.stderr
            assert actual == expected, result.stdout + result.stderr


def test_restore_preconditions_precede_value_adapters_and_admission() -> None:
    events: list[str] = []

    def snapshot(value: int) -> int:
        events.append("snapshot")
        return value

    def accepts(*, candidate: int) -> bool:
        events.append("admission")
        return candidate > 0

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, int.__eq__, snapshot
    )

    class Family(Space):
        value = Decision(semantics, domain=domain(accepts=accepts))

    base = Family()
    configured = base.with_choices(value=1)
    same = selections.capture(configured)
    different = selections.capture(base.with_choices(value=2))
    empty = selections.capture(base)
    events.clear()
    for saved in (same, different, empty):
        with pytest.raises(RequestError, match="no committed choices"):
            selections.restore(configured, saved)
    assert events == []
    # Querying the base does not make it ineligible, and empty replay is a no-op.
    assert isinstance(base.query(Family.value), Unresolved)
    assert selections.restore(base, empty).instance is base
    assert events == []
    assert selections.restore(base, same).instance.value == 1
    cleared = configured.with_choices(configured.field(Family.value).clear())
    assert selections.restore(cleared, different).instance.value == 2


def test_singleton_structural_selection_does_not_disqualify_empty_root() -> None:
    class Child(Space):
        value = Decision(int, values=(1,))

    class Family(Space):
        choice = SubspaceChoice({"only": Subspace(Child)})

    base = Family()
    assert base.choice.select("only").instance is base
    configured = base.choice.alternative("only").with_choices(value=1)
    result = selections.restore(base, selections.capture(configured))
    assert (
        result.accepted and result.instance.choice.alternative("only").field(Child.value).get() == 1
    )
