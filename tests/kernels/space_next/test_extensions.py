# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Neutral extension bundles use ordinary scoped declarations and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import cast

import pytest

from finn.kernels.space._next import (
    Const,
    Decided,
    Decision,
    DecisionRef,
    DefinitionError,
    Param,
    Rejected,
    ScopeBuilder,
    Space,
    Subspace,
    Unresolved,
    ValueKey,
    ValueRef,
    ValueSemantics,
    ViewKey,
    compile_space,
    constraint,
    derived,
)


@dataclass(frozen=True)
class Encoding:
    bits: int
    integer: bool = True


@dataclass(frozen=True)
class StreamValue:
    dtype: Encoding
    lanes: int
    bits: int


ENCODING = ValueSemantics.immutable_nominal(Encoding)
STREAM = ViewKey("stream", StreamValue)
BITS = ValueKey("bits", int)


class StreamShape(Space):
    dtype = Param(ENCODING)
    lanes = Param(int)

    @derived
    def element_bits(*, dtype: Encoding) -> int:
        return dtype.bits

    @derived
    def bits(*, element_bits: int, lanes: int) -> int:
        return element_bits * lanes


class StreamPlacement(Subspace[StreamShape]):
    @property
    def dtype(self) -> ValueRef[Encoding]:
        return self.ref(StreamShape.dtype)

    @property
    def lanes(self) -> ValueRef[int]:
        return self.ref(StreamShape.lanes)

    @property
    def lane_choice(self) -> DecisionRef[int]:
        return self.decision_ref(StreamShape.lanes)

    @property
    def bits(self) -> ValueRef[int]:
        return self.ref(StreamShape.bits)

    @property
    def stream(self) -> ValueRef[StreamValue]:
        return self.accepted(STREAM)


def _admitted(*, dtype: Encoding, maximum_bits: int) -> bool:
    return dtype.integer and 0 < dtype.bits <= maximum_bits


def _positive_lanes(*, lanes: int, minimum: int) -> bool:
    return lanes >= minimum


def _stream(*, dtype: Encoding, lanes: int, bits: int) -> StreamValue:
    return StreamValue(dtype, lanes, bits)


def stream_shape(
    *,
    dtype: Encoding | ValueRef[Encoding],
    lanes: int | ValueRef[int],
    maximum_bits: int | ValueRef[int],
) -> StreamPlacement:
    builder = ScopeBuilder(StreamShape, name="AdmittedStream")
    limit = builder.param("maximum_bits", int)
    minimum = builder.const("minimum_lanes", 1)
    admission = builder.constraint(
        "admitted",
        _admitted,
        dtype=StreamShape.dtype,
        maximum_bits=limit,
    )
    geometry = builder.constraint(
        "positive_lanes",
        _positive_lanes,
        lanes=StreamShape.lanes,
        minimum=minimum,
    )
    value = builder.derived(
        "stream_value",
        _stream,
        dtype=StreamShape.dtype,
        lanes=StreamShape.lanes,
        bits=StreamShape.bits,
    )
    complete = builder.view("complete", value, constraints=(admission, geometry))
    builder.export(STREAM).view(complete)
    builder.export(BITS).value(StreamShape.bits)
    builder.bind(StreamShape.dtype, dtype)
    builder.bind(StreamShape.lanes, lanes)
    builder.bind(limit, maximum_bits)
    placement = builder.place()
    return StreamPlacement(placement.space_type, when=placement.when, **placement.bindings)


def test_stream_shape_places_independent_choices_and_keeps_narrow_fields_available() -> None:
    class Pair(Space):
        limit = Param(int, required=False)
        left = stream_shape(
            dtype=Param(ENCODING),
            lanes=Decision(int, values=(1, 2, 4)),
            maximum_bits=limit,
        )
        right = stream_shape(
            dtype=Param(ENCODING),
            lanes=Decision(int, values=(1, 2, 4)),
            maximum_bits=limit,
        )

        @constraint(a=left.bits, b=right.bits)
        def balanced(*, a: int, b: int) -> bool:
            return a == b

    assert "left_dtype" not in vars(Pair) and "maximum_bits" not in vars(Pair)
    model = compile_space(Pair)
    missing = model.start({Pair.left.dtype: Encoding(3), Pair.right.dtype: Encoding(6)})
    first = missing.assign(Pair.left.lane_choice, 2)
    assert first.left.bits == 6
    assert first.right.element_bits == 6
    assert isinstance(first.right.answer(StreamShape.lanes), Unresolved)
    assert isinstance(first.answer(Pair.left.stream), Unresolved)
    assert isinstance(missing.left.answer(StreamShape.lanes), Unresolved)

    admitted = model.start(
        {
            Pair.left.dtype: Encoding(3),
            Pair.right.dtype: Encoding(6),
            Pair.limit: 4,
        }
    )
    selected = admitted.assign(Pair.left.lane_choice, 2).assign(Pair.right.lane_choice, 1)
    assert selected.left.bits == selected.right.bits == 6
    assert selected.assess(Pair.balanced).verdict is True
    assert selected.answer(Pair.left.stream) == Decided(StreamValue(Encoding(3), 2, 6))
    refused = selected.answer(Pair.right.stream)
    assert isinstance(refused, Rejected)
    assert any("right" in finding.owner for finding in refused.findings)


def test_builder_finish_is_pure_repeatable_and_placements_share_only_the_template() -> None:
    calls: list[int] = []
    builder = ScopeBuilder(Space)
    value = builder.param("value", int)
    choice = builder.decision("choice", int, values=(1, 2))

    def compute(*, value: int, choice: int) -> int:
        calls.append(value)
        return value * choice

    output = builder.derived("output", compute)
    key = ValueKey("output", int)
    builder.export(key).value(output)
    builder.bind(value, 3)
    template = builder.finish()
    assert calls == [] and builder.sealed
    assert builder.finish() is template

    class Parent(Space):
        first = builder.place()
        second = builder.place()

    assert Parent.first.space_type is Parent.second.space_type is template
    point = Parent.start()
    selected = point.assign(Parent.first.decision_ref(choice), 2)
    assert selected.answer(Parent.first.ref(key)) == Decided(6)
    assert isinstance(selected.second.answer(choice), Unresolved)
    assert isinstance(point.first.answer(choice), Unresolved)
    assert calls == [3]


def test_builder_seals_every_mutation_before_snapshot_adapters_run() -> None:
    calls: list[int] = []

    def snapshot(value: int) -> int:
        calls.append(value)
        return value

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, int.__eq__, snapshot
    )
    builder = ScopeBuilder(Space)
    value = builder.param("value", int)
    deferred_export = builder.export(ValueKey("value", int))
    builder.finish()
    with pytest.raises(DefinitionError, match="sealed"):
        builder.const("too_late", 1, semantics=semantics)
    with pytest.raises(DefinitionError, match="sealed"):
        builder.decision("too_late", semantics, values=(1,))
    with pytest.raises(DefinitionError, match="sealed"):
        builder.bind(value, 1)
    with pytest.raises(DefinitionError, match="sealed"):
        deferred_export.value(value)
    with pytest.raises(DefinitionError, match="sealed"):
        builder.add("too_late", Const(1))
    assert calls == []


def test_duplicate_names_owned_declarations_and_foreign_exports_are_definition_errors() -> None:
    builder = ScopeBuilder(StreamShape)
    with pytest.raises(DefinitionError, match="inherited member"):
        builder.param("dtype", int)
    with pytest.raises(DefinitionError, match="already belongs"):
        builder.add("renamed", StreamShape.dtype)
    local = builder.const("constant", 3)
    with pytest.raises(DefinitionError, match="duplicate"):
        builder.const("constant", 4)
    with pytest.raises(DefinitionError, match="already has a member name"):
        builder.add("other_name", local)
    with pytest.raises(DefinitionError, match="not a local or inherited member"):
        builder.export(ValueKey("foreign", int)).value(Param(int))
    builder.export(ValueKey("local", int)).value(local)
    with pytest.raises(DefinitionError, match="duplicate export"):
        builder.export(ValueKey("local", int)).value(local)
    with pytest.raises(DefinitionError, match="name segment"):
        builder.const("invalid.name", 1)


def test_missing_bindings_and_incompatible_exports_fail_without_descriptor_runtime_errors() -> None:
    builder = ScopeBuilder(StreamShape)
    builder.bind(StreamShape.dtype, Encoding(3))
    with pytest.raises(DefinitionError, match="missing child parameter bindings"):
        builder.place()
    assert builder.finish() is builder.finish()
    wrong = ScopeBuilder(Space)
    integer = wrong.const("value", 3)
    wrong.export(ValueKey("value", str)).value(cast(ValueRef[str], integer))
    with pytest.raises(DefinitionError, match="incompatible semantics"):
        wrong.finish()
    with pytest.raises(DefinitionError, match="previously failed"):
        wrong.finish()


def test_external_bound_references_must_be_explicit_child_bindings() -> None:
    class Parent(Space):
        limit = Param(int)

    builder = ScopeBuilder(StreamShape)
    builder.constraint(
        "hidden_parent", _admitted, dtype=StreamShape.dtype, maximum_bits=Parent.limit
    )
    with pytest.raises(DefinitionError, match="not declared in this effective scope"):
        builder.finish()


def test_builder_literal_bindings_snapshot_before_sealing_and_each_placement() -> None:
    class Vector(Space):
        values: Param[list[int]] = Param(list)

    source = [1, 2]
    builder = ScopeBuilder(Vector)
    builder.bind(Vector.values, source)
    builder.finish()
    source.append(9)
    first = builder.place()
    exposed = first.bindings["values"]
    assert isinstance(exposed, list)
    exposed.append(8)

    class Parent(Space):
        vector = builder.place()

    assert Parent.start().vector.values == [1, 2]


def test_extension_typing_fixture(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    project = Path(__file__).resolve().parents[3]
    fixtures = Path(__file__).with_name("typing")
    environment = dict(os.environ, MYPYPATH=f"{project / 'src'}:{project / 'tests'}")
    command = [
        mypy,
        "--strict",
        "--no-incremental",
        "--explicit-package-bases",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    positive = subprocess.run(
        [*command, str(fixtures / "extensions.py")],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert positive.returncode == 0, positive.stdout + positive.stderr
    source = (fixtures / "extensions_negative.py.txt").read_text()
    negative_path = tmp_path / "negative.py"
    negative_path.write_text(source)
    negative = subprocess.run(
        [*command, str(negative_path)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {index for index, line in enumerate(source.splitlines(), 1) if "# E" in line}
    actual = {int(line) for line in re.findall(r"negative\.py:(\d+): error:", negative.stdout)}
    assert negative.returncode == 1, negative.stdout + negative.stderr
    assert actual == expected, negative.stdout + negative.stderr
