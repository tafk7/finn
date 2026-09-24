# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Sparse ownership, detached editing, and atomic selection replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import cast

import pytest

from finn.kernels.space import (
    Decision,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    ValueSemantics,
    compile_space,
    derived,
    divisors_of,
    inspection,
    selections,
)
from finn.kernels.space.errors import RequestError
from finn.kernels.space.selections import SelectionChange


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
    base = model.start()
    point = base.assign(Family.factor, 1).assign(Family.left.decision_ref(Child.local), "auto")
    assert point.doubled == 2
    captured = selections.capture(point.left)
    assert captured.keys == ("factor", "left.local")
    assert [entry.value for entry in captured.entries] == [1, "auto"]
    assert selections.capture(base).keys == ()
    replay = selections.restore(base, captured)
    assert replay.accepted
    assert selections.capture(replay.point) == captured
    assert isinstance(replay.point.right.answer(Child.local), Unresolved)


def test_detached_replacement_and_removal_do_not_overwrite_live_points() -> None:
    class Family(Space):
        factor = Decision(int, values=(1, 2))
        style = Decision(str, values=("small", "fast"))

    base = Family.start()
    point = base.assign(Family.factor, 1).assign(Family.style, "small")
    original = selections.capture(point)
    edited = original.with_changes(
        [
            original.edit(Family.factor, 2),
            original.remove(Family.style),
        ]
    )
    assert original.keys == ("factor", "style")
    assert edited.keys == ("factor",)
    assert original.value(Family.factor) == 1
    assert edited.value(Family.factor) == 2
    report = selections.restore(base, edited)
    assert report.accepted and report.point.factor == 2
    assert point.factor == 1 and point.style == "small"
    conflict = selections.restore(point, edited)
    assert not conflict.accepted and conflict.point is point
    assert conflict.outcomes[0].status == "refused"
    assert selections.restore(point, original).point is point
    with pytest.raises(KeyError, match="style"):
        edited.value(Family.style)


def test_rebound_inputs_can_refuse_an_earlier_selection_without_partial_publication() -> None:
    class Family(Space):
        extent = Param(int)
        factor = Decision(int, domain=divisors_of(extent))
        style = Decision(str, values=("auto", "block"))

    model = compile_space(Family)
    first = model.start({Family.extent: 12}).assign(Family.factor, 4).assign(Family.style, "block")
    changed = model.start({Family.extent: 10})
    report = selections.restore(changed, selections.capture(first))
    assert not report.accepted and report.point is changed
    assert selections.capture(changed).keys == ()
    assert first.factor == 4 and first.extent == 12


def test_selector_change_retains_stale_case_commitments_until_explicit_removal() -> None:
    class Child(Space):
        lanes = Decision(int, values=(1, 2))

    class Family(Space):
        implementation = SubspaceChoice({"left": Subspace(Child), "right": Subspace(Child)})

    model = compile_space(Family)
    base = model.start()
    selector = inspection.choices(model)[0].selector
    assert selector is not None
    chosen = base.implementation.select("left")
    child = chosen.alternative("left")
    left_lanes = inspection.decision_handle(child, Child.lanes)
    point = child.assign(Child.lanes, 1).root
    captured = selections.capture(point)
    assert "implementation" in captured.keys
    assert all("$" not in key for key in captured.keys)
    edited = captured.with_changes([captured.edit(selector, "right")])
    assert edited.keys == captured.keys
    refused = selections.restore(base, edited)
    assert not refused.accepted and refused.point is base
    assert selections.capture(base).keys == ()
    cleaned = edited.with_changes([edited.remove(left_lanes)])
    accepted = selections.restore(base, cleaned)
    assert accepted.accepted
    assert selections.capture(accepted.point).keys == ("implementation",)


def test_singleton_choices_do_not_create_persisted_selectors() -> None:
    class Child(Space):
        lanes = Decision(int, values=(1,))

    class Family(Space):
        implementation = SubspaceChoice({"only": Subspace(Child)})

    point = Family.start().implementation.alternative("only").assign(Child.lanes, 1)
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


def test_capture_edit_and_public_entries_detach_mutable_payloads() -> None:
    class Family(Space):
        payload = Decision(PAYLOAD, values=(Payload([1, 2]), Payload([3])))

    base = Family.start()
    source = Payload([2, 1])
    point = base.assign(Family.payload, source)
    captured = selections.capture(point)
    source.values.append(99)
    public = captured.entries[0].value
    assert isinstance(public, Payload)
    public.values.append(99)
    captured.value(Family.payload).values.append(99)
    assert captured.value(Family.payload).values == [2, 1]
    replacement = Payload([3])
    change = captured.edit(Family.payload, replacement)
    replacement.values.append(99)
    edited = captured.with_changes([change])
    assert edited.value(Family.payload).values == [3]
    assert selections.restore(base, edited).point.payload.values == [3]
    assert point.payload.values == [2, 1]
    equal = captured.with_changes([captured.edit(Family.payload, Payload([1, 2]))])
    assert equal == captured


def test_foreign_models_invalid_changes_and_child_restore_are_rejected() -> None:
    class Child(Space):
        value = Decision(int, values=(1, 2))

    class Family(Space):
        first = Subspace(Child)
        second = Subspace(Child)

    first_model = compile_space(Family)
    point = first_model.start().assign(Family.first.decision_ref(Child.value), 1)
    selection = selections.capture(point)
    other = compile_space(Family).start()
    with pytest.raises(RequestError, match="different compiled model"):
        selections.restore(other, selection)
    with pytest.raises(RequestError, match="root occurrence"):
        selections.restore(point.first, selection)
    with pytest.raises(RequestError):
        selection.edit(Child.value, 2)
    with pytest.raises(RequestError, match="different compiled model"):
        selection.with_changes(
            [selections.capture(other).edit(Family.first.decision_ref(Child.value), 2)]
        )
    with pytest.raises(RequestError, match="duplicate"):
        selection.with_changes(
            [
                selection.remove(Family.first.decision_ref(Child.value)),
                selection.remove(Family.first.decision_ref(Child.value)),
            ]
        )
    with pytest.raises(RequestError, match="edit.*remove"):
        selection.with_changes([cast(SelectionChange, object())])


def test_capture_does_not_evaluate_unrelated_uncommitted_guard_callbacks() -> None:
    class Family(Space):
        committed = Decision(int, values=(1,))

        @derived
        def explosive() -> bool:
            raise AssertionError("unrelated capture must not query this guard")

        unrelated = Decision(int, values=(1,), when=explosive)

    point = Family.start().assign(Family.committed, 1)
    assert selections.capture(point).keys == ("committed",)


def test_selection_edits_and_codec_bindings_have_strict_value_types(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    project = Path(__file__).resolve().parents[3]
    common = """from typing_extensions import assert_type
from finn.kernels.space import (
    Decision, Param, Space, ValueCodec, JSONValue, codec_for, compile_space,
    selections, Selection, SelectionChange, RefinementReport,
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
base = compile_space(Family).start({Family.extent: 4})
selected = selections.capture(base)
"""
    positive = (
        common
        + """assert_type(selected.edit(Family.factor, 2), SelectionChange)
assert_type(selected.with_changes([
    selected.edit(Family.factor, 2), selected.edit(Family.style, "block"),
]), Selection)
assert_type(selected.value(Family.factor), int)
assert_type(selections.restore(base, selected), RefinementReport[Family])
codec_for(Family.factor, integer_codec)
"""
    )
    negative = (
        common
        + """selected.edit(Family.factor, "two")  # E
selected.edit(Family.extent, 2)  # E
selected.remove(Family.extent)  # E
codec_for(Family.style, integer_codec)  # E
selected.with_changes(["not a change"])  # E
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
