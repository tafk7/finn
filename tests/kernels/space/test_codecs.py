# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit portable schemas, codec validation, and detached mutable documents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import cast

import pytest

from finn.kernels.space import (
    Decision,
    JSONValue,
    Param,
    SelectionSchema,
    Space,
    Subspace,
    SubspaceChoice,
    ValueCodec,
    ValueSemantics,
    codec_for,
    codecs,
    compile_space,
    divisors_of,
    inspection,
    selections,
)
from finn.kernels.space.errors import DefinitionError, RequestError


def _integer(value: JSONValue) -> int:
    if type(value) is not int:
        raise ValueError("an integer is required")
    return value


def _string(value: JSONValue) -> str:
    if type(value) is not str:
        raise ValueError("a string is required")
    return value


INTEGER = ValueCodec[int]("integer", 1, lambda value: value, _integer)
STRING = ValueCodec[str]("string", 1, lambda value: value, _string)


def _entries(document: dict[str, JSONValue]) -> list[dict[str, JSONValue]]:
    return cast(list[dict[str, JSONValue]], document["entries"])


def test_partial_document_replays_on_another_compilation_using_explicit_schema_identity() -> None:
    class Family(Space):
        extent = Param(int)
        factor = Decision(int, domain=divisors_of(extent))
        style = Decision(str, values=("auto", "block"))

    first = compile_space(Family)
    schema = SelectionSchema(
        first,
        family="test.family",
        version=2,
        bindings=(codec_for(Family.factor, INTEGER), codec_for(Family.style, STRING)),
    )
    point = first.bind({Family.extent: 12}).with_choices(factor=3)
    document = codecs.encode(selections.capture(point), schema)
    assert [entry["key"] for entry in _entries(document)] == ["factor"]
    assert _entries(document)[0]["codec"] == "integer"
    serialized = json.dumps(document)
    assert "Family" not in serialized and "$" not in serialized
    second = compile_space(Family)
    second_schema = SelectionSchema(
        second,
        family="test.family",
        version=2,
        bindings=(codec_for(Family.factor, INTEGER), codec_for(Family.style, STRING)),
    )
    restored = selections.restore(
        second.bind({Family.extent: 12}), codecs.decode(json.loads(serialized), second_schema)
    )
    assert restored.accepted and restored.instance.factor == 3
    assert selections.capture(restored.instance).keys == ("factor",)


def test_entry_order_does_not_control_dependent_replay() -> None:
    class Family(Space):
        extent = Decision(int, values=(8, 12))
        factor = Decision(int, domain=divisors_of(extent))

    model = compile_space(Family)
    schema = SelectionSchema(
        model,
        family="ordered",
        version=1,
        bindings=(codec_for(Family.extent, INTEGER), codec_for(Family.factor, INTEGER)),
    )
    base = model.bind()
    point = base.with_choices(extent=12).with_choices(factor=3)
    document = codecs.encode(selections.capture(point), schema)
    _entries(document).reverse()
    report = selections.restore(base, codecs.decode(document, schema))
    assert report.accepted and report.instance.extent == 12 and report.instance.factor == 3


@dataclass
class Bag:
    items: list[int]


BAG = ValueSemantics(
    Bag,
    "bag",
    lambda value: type(value) is Bag,
    lambda left, right: sorted(left.items) == sorted(right.items),
    lambda value: Bag(list(value.items)),
)


def _encode_bag(value: Bag) -> JSONValue:
    return [item for item in sorted(value.items)]


def _decode_bag(value: JSONValue) -> Bag:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("expected integer array")
    return Bag([cast(int, item) for item in value])


def test_custom_unhashable_values_round_trip_by_declared_equality_and_detach_documents() -> None:
    class Family(Space):
        bag = Decision(BAG, values=(Bag([1, 2]),))

    model = compile_space(Family)
    schema = SelectionSchema(
        model,
        family="bags",
        version=1,
        bindings=(codec_for(Family.bag, ValueCodec("bag", 1, _encode_bag, _decode_bag)),),
    )
    point = model.bind().with_choices(bag=Bag([2, 1]))
    selection = selections.capture(point)
    document = codecs.encode(selection, schema)
    decoded = codecs.decode(document, schema)
    assert decoded.value(Family.bag).items == [1, 2]
    assert decoded == selection
    assert selections.restore(point, decoded).instance is point
    payload = _entries(document)[0]["value"]
    assert isinstance(payload, list)
    payload.append(9)
    assert decoded.value(Family.bag).items == [1, 2]
    assert selection.value(Family.bag).items == [2, 1]
    decoded.value(Family.bag).items.append(9)
    assert decoded.value(Family.bag).items == [1, 2]


def test_all_schema_and_key_errors_are_found_before_any_decoder_runs() -> None:
    decoded_values: list[JSONValue] = []

    def decode_integer(value: JSONValue) -> int:
        decoded_values.append(value)
        return _integer(value)

    class Family(Space):
        a = Decision(int, values=(1,))
        b = Decision(int, values=(2,))

    model = compile_space(Family)
    codec = ValueCodec[int]("integer", 1, lambda value: value, decode_integer)
    schema = SelectionSchema(
        model,
        family="two",
        version=1,
        bindings=(codec_for(Family.a, codec), codec_for(Family.b, codec)),
    )
    point = model.bind().with_choices(a=1).with_choices(b=2)
    selection = selections.capture(point)
    mutations: tuple[Callable[[dict[str, JSONValue]], None], ...] = (
        lambda document: document.__setitem__("family", "different"),
        lambda document: document.__setitem__("version", 2),
        lambda document: document.__setitem__("version", True),
        lambda document: _entries(document)[1].__setitem__("key", "unknown"),
        lambda document: _entries(document)[1].__setitem__("codec", "other"),
        lambda document: _entries(document)[1].__setitem__("codec_version", 2),
        lambda document: _entries(document).append(dict(_entries(document)[0])),
        lambda document: _entries(document)[1].__setitem__("extra", 1),
    )
    for mutate in mutations:
        document = codecs.encode(selection, schema)
        decoded_values.clear()
        mutate(document)
        with pytest.raises(RequestError):
            codecs.decode(document, schema)
        assert decoded_values == []
    assert point.a == 1 and point.b == 2


def test_missing_codecs_and_invalid_decoded_values_do_not_silently_drop_entries() -> None:
    class Family(Space):
        factor = Decision(int, values=(1,))

    model = compile_space(Family)
    point = model.bind().with_choices(factor=1)
    incomplete = SelectionSchema(model, family="one", version=1, bindings=())
    assert codecs.encode(selections.capture(model.bind()), incomplete)["entries"] == []
    with pytest.raises(RequestError, match="explicit codec"):
        codecs.encode(selections.capture(point), incomplete)
    schema = SelectionSchema(
        model, family="one", version=1, bindings=(codec_for(Family.factor, INTEGER),)
    )
    document = codecs.encode(selections.capture(point), schema)
    _entries(document)[0]["value"] = "wrong"
    with pytest.raises(RequestError, match="decoding failed"):
        codecs.decode(document, schema)
    broken = ValueCodec[int]("integer", 1, lambda value: value, lambda value: cast(int, "wrong"))
    broken_schema = SelectionSchema(
        model, family="one", version=1, bindings=(codec_for(Family.factor, broken),)
    )
    with pytest.raises(RequestError, match="expected selection value"):
        codecs.decode(codecs.encode(selections.capture(point), schema), broken_schema)


def test_selected_case_identity_is_authored_and_unknown_cases_are_diagnosed() -> None:
    class Child(Space):
        pass

    class Family(Space):
        implementation = SubspaceChoice({"small": Subspace(Child), "fast": Subspace(Child)})

    model = compile_space(Family)
    selector = inspection.choices(model)[0].selector
    assert selector is not None
    schema = SelectionSchema(
        model, family="implementations", version=1, bindings=(codec_for(selector, STRING),)
    )
    base = model.bind()
    point = base.with_choices(base.field(selector).change("fast"))
    document = codecs.encode(selections.capture(point), schema)
    assert _entries(document)[0]["key"] == "implementation"
    assert _entries(document)[0]["value"] == "fast"
    assert selections.restore(base, codecs.decode(document, schema)).accepted
    _entries(document)[0]["value"] = "removed-case"
    with pytest.raises(RequestError, match="unknown structural case"):
        codecs.decode(document, schema)
    captured = selections.capture(point)
    invalid = captured.with_changes([captured.edit(selector, "removed-case")])
    with pytest.raises(RequestError, match="unknown structural case"):
        codecs.encode(invalid, schema)
    assert selections.capture(base).keys == ()


def test_nonportable_payloads_cycles_and_duplicate_schema_bindings_are_rejected() -> None:
    class Family(Space):
        factor = Decision(int, values=(1,))

    model = compile_space(Family)
    with pytest.raises(DefinitionError, match="duplicate codec"):
        SelectionSchema(
            model,
            family="one",
            version=1,
            bindings=(codec_for(Family.factor, INTEGER), codec_for(Family.factor, INTEGER)),
        )
    with pytest.raises(DefinitionError, match="positive integer"):
        SelectionSchema(model, family="one", version=0, bindings=())
    schema = SelectionSchema(
        model, family="one", version=1, bindings=(codec_for(Family.factor, INTEGER),)
    )
    point = model.bind().with_choices(factor=1)
    for malformed in (float("nan"), {"value": object()}, {1: "bad-key"}):
        with pytest.raises(RequestError):
            codecs.decode(malformed, schema)
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(RequestError, match="cycle"):
        codecs.decode(cyclic, schema)

    class Other(Space):
        factor = Decision(int, values=(1,))

    other_schema = SelectionSchema(
        compile_space(Other),
        family="one",
        version=1,
        bindings=(codec_for(Other.factor, INTEGER),),
    )
    with pytest.raises(RequestError, match="different compiled models"):
        codecs.encode(selections.capture(point), other_schema)


def test_lossy_codecs_are_refused_and_encoding_callbacks_get_detached_values() -> None:
    class Family(Space):
        factor = Decision(int, values=(1, 2))
        bag = Decision(BAG, values=(Bag([1, 2]),))

    model = compile_space(Family)
    point = model.bind().with_choices(factor=2)
    lossy = ValueCodec[int]("lossy", 1, lambda value: 1, _integer)
    schema = SelectionSchema(
        model, family="lossy", version=1, bindings=(codec_for(Family.factor, lossy),)
    )
    with pytest.raises(RequestError, match="changes the committed value"):
        codecs.encode(selections.capture(point), schema)

    def mutating_encoder(value: Bag) -> JSONValue:
        result: JSONValue = [item for item in value.items]
        value.items.append(999)
        return result

    bag_codec = ValueCodec("bag", 1, mutating_encoder, _decode_bag)
    bag_schema = SelectionSchema(
        model, family="bags", version=1, bindings=(codec_for(Family.bag, bag_codec),)
    )
    original = selections.capture(model.bind().with_choices(bag=Bag([1, 2])))
    encoded = codecs.encode(original, bag_schema)
    assert _entries(encoded)[0]["value"] == [1, 2]
    assert original.value(Family.bag).items == [1, 2]


def test_encoded_stale_case_entries_are_rejected_by_atomic_replay() -> None:
    class Child(Space):
        lanes = Decision(int, values=(1, 2))

    class Family(Space):
        implementation = SubspaceChoice({"left": Subspace(Child), "right": Subspace(Child)})

    model = compile_space(Family)
    base = model.bind()
    selector = inspection.choices(model)[0].selector
    assert selector is not None
    selected = base.implementation.select("left").alternative("left")
    lanes = inspection.decision_handle(selected, Child.lanes)
    point = selected.with_choices(lanes=1).root
    schema = SelectionSchema(
        model,
        family="cases",
        version=1,
        bindings=(codec_for(selector, STRING), codec_for(lanes, INTEGER)),
    )
    document = codecs.encode(selections.capture(point), schema)
    next(entry for entry in _entries(document) if entry["key"] == "implementation")["value"] = (
        "right"
    )
    refused = selections.restore(base, codecs.decode(document, schema))
    assert not refused.accepted and refused.instance is base
    assert selections.capture(base).keys == ()


def test_roundtrip_decoder_failure_keeps_codec_context_and_original_cause() -> None:
    class Family(Space):
        factor = Decision(int, values=(1,))

    def broken_decoder(value: JSONValue) -> int:
        raise RuntimeError("decoder unavailable")

    model = compile_space(Family)
    codec = ValueCodec[int]("broken", 3, lambda value: value, broken_decoder)
    schema = SelectionSchema(
        model, family="failures", version=1, bindings=(codec_for(Family.factor, codec),)
    )
    selection = selections.capture(model.bind().with_choices(factor=1))
    with pytest.raises(RequestError, match="factor.*broken.*round-trip") as raised:
        codecs.encode(selection, schema)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "decoder unavailable" in str(raised.value.__cause__)
