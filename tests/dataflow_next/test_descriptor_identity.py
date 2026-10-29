# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed durable tags belong to exact production classes, not matching names."""

from __future__ import annotations

import pytest

from dataclasses import field, fields, is_dataclass, make_dataclass

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow_next.descriptors import dump, load
from finn.dataflow_next.logical import Node
from finn.dataflow_next.maps import identity
from finn.dataflow_next.queries import WordScope
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.tensor import ScalarEncoding, Tensor


def test_foreign_frozen_node_cannot_reload_as_production_node() -> None:
    foreign = make_dataclass(
        "Node", [("id", str), ("schedule", object, field(default=None))], frozen=True
    )
    value = foreign("foreign-node")
    assert type(value).__name__ == "Node" and not isinstance(value, Node)
    with pytest.raises(TypeError, match="not a durable logical descriptor"):
        dump(value)
    assert load(dump(Node("production-node"))) == Node("production-node")


@pytest.mark.parametrize(
    "value",
    [
        Tensor((2,), ScalarEncoding(resolve_qonnx_datatype_name("INT8"))),
        identity((2,)),
        WordScope("s", "f", "p"),
    ],
)
def test_same_name_and_fields_do_not_grant_a_registered_value_contract(value: object) -> None:
    assert is_dataclass(value) and not isinstance(value, type)
    foreign = make_dataclass(
        type(value).__name__, [(f.name, object) for f in fields(value)], frozen=True
    )
    imitation = foreign(**{f.name: getattr(value, f.name) for f in fields(value)})
    assert type(imitation).__name__ == type(value).__name__
    with pytest.raises(TypeError, match="not a durable logical descriptor"):
        dump(imitation)
    restored = load(dump(value))
    assert type(restored) is type(value) and restored == value


def test_subclasses_are_not_implicitly_registered_by_special_cases() -> None:
    class OtherNode(Node):
        pass

    class OtherSchedule(Schedule):
        pass

    class OtherEncoding(ScalarEncoding):
        pass

    # Even a subclass renamed to the registered spelling has another identity.
    OtherNode.__name__ = "Node"
    index = Index("i")
    dtype = resolve_qonnx_datatype_name("INT8")
    for value in (OtherNode("node"), OtherSchedule({index: 2}), OtherEncoding(dtype)):
        with pytest.raises(TypeError, match="not a durable logical descriptor"):
            dump(value)
    for production in (Node("node"), Schedule({index: 2}), ScalarEncoding(dtype)):
        assert load(dump(production)) == production
