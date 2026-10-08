# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    View,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    inspection,
    selections,
)
from finn.core.space.errors import DefinitionError, RequestError
from finn.dataflow.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    resolve_qonnx_datatype_name,
)
from finn.kernels.base import Kernel
from finn.kernels.values.domains import Integer
from finn.kernels.values.semantics import (
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
)


def views(point: object) -> list[str]:
    return [item.key for item in inspection.members(point) if item.kind == "view"]


@dataclass(frozen=True)
class Pins:
    inputs: tuple[tuple[str, int], ...]
    outputs: tuple[tuple[str, int], ...]


def test_kernel_identity_is_validated_at_class_creation() -> None:
    with pytest.raises(DefinitionError, match="id"):
        type("Unnamed", (Kernel,), {})
    # The version is a positive int (an op's opset version); artifacts carry str(version).
    for wrong in ("1", 0, True):
        with pytest.raises(DefinitionError, match="version"):
            type("WrongVersion", (Kernel,), {"id": "test.wrong", "version": wrong})

    class Empty(Kernel):
        id = "test.empty"

    assert Empty.version == 1
    # The protocol's views; a kernel that declares no module builds none.
    empty = design_space(Empty())
    assert views(empty) == ["buffering", "cycles", "module", "netlist", "resources"]
    refused = empty.query(Empty.module)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"kernel-module"}


@pytest.mark.parametrize(
    "name",
    (
        "BINARY",
        "BIPOLAR",
        "TERNARY",
        "INT1",
        "INT2",
        "INT8",
        "UINT8",
        "FIXED<8,4>",
        "SCALEDINT<8>",
        "FLOAT16",
        "FLOAT32",
        "FLOAT<5,10,7>",
    ),
)
def test_candidate_dtype_adapter_preserves_canonical_names_and_snapshots(name: str) -> None:
    value = resolve_qonnx_datatype_name(name)
    semantics = QONNX_DATATYPE_VALUE_SEMANTICS
    assert semantics.type_token is QONNX_DATATYPE_TOKEN
    assert semantics.accepts(value)
    # A datatype is a value (one immutable instance per canonical name), so the
    # snapshot is the value itself.
    assert semantics.freeze(value) is value
    assert not semantics.accepts(name)


def test_a_dtype_selection_restores_the_exact_datatype() -> None:
    class DtypeKernel(Kernel):
        id = "test.dtype"
        dtype: QONNXDataType = Decision(
            values=(resolve_qonnx_datatype_name("TERNARY"), resolve_qonnx_datatype_name("INT2")),
            semantics=QONNX_DATATYPE_VALUE_SEMANTICS,
        )

    base = design_space(DtypeKernel())
    point = base.with_choices(dtype=resolve_qonnx_datatype_name("TERNARY"))
    restored = selections.restore(base, selections.capture(point))
    assert restored.accepted
    assert restored.instance.dtype.name == "TERNARY"
    assert not QONNX_DATATYPE_VALUE_SEMANTICS.values_equal(
        restored.instance.dtype, resolve_qonnx_datatype_name("INT2")
    )
    with pytest.raises(RequestError):
        base.with_choices(dtype=cast(QONNXDataType, "TERNARY"))


def test_tuple_adapters_preserve_exact_integer_structure_without_geometry_validation() -> None:
    assert INTEGER_VECTOR.accepts((1, 0, -2))
    assert INTEGER_VECTOR.accepts(())
    assert not INTEGER_VECTOR.accepts((True, 2))
    assert not INTEGER_VECTOR.accepts([1, 2])
    vector = (1, 2)
    assert INTEGER_VECTOR.freeze(vector) is vector
    table = (((1, 2), (3,)), ((4,),))
    assert THRESHOLD_TABLE.accepts(table)  # rectangularity belongs to kernel admission
    assert THRESHOLD_TABLE.accepts(())
    assert not THRESHOLD_TABLE.accepts((((True,),),))
    assert not THRESHOLD_TABLE.accepts([[[1]]])
    assert THRESHOLD_TABLE.freeze(table) is table


def test_explicit_dtype_semantics_support_typed_protocol_results_of_immutable_values() -> None:
    class Typed(Kernel):
        id = "test.typed"
        dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)

        @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        def result(self) -> QONNXDataType:
            return self.dtype

        physical = View(result)

    original = resolve_qonnx_datatype_name("INT8")
    point = design_space(Typed(dtype=original))
    assert point.dtype is original
    # Nothing to detach: a datatype value refuses mutation.
    with pytest.raises(AttributeError, match="immutable datatype value"):
        setattr(point.result, "_bitwidth", 32)
    assert point.result is original and point.physical.name == "INT8"


def test_composite_extends_kernel_with_typed_optional_views_and_independent_nodes() -> None:
    # The declarations are built in plain Python and ``composite`` names them as a
    # Space class on the kernel base. ``dtype`` is a formal each placing node binds
    # by name; each node call is an independent scope.
    class InterfaceKernel(Kernel):
        id = "test.interface"

    def bits(*, dtype: QONNXDataType, lanes: int) -> int:
        return dtype.bitwidth() * lanes

    def plain_pins(*, payload_bits: int) -> Pins:
        return Pins((("word", payload_bits),), ())

    def narrow(*, dtype: QONNXDataType) -> bool | Rejected:
        return Integer(max_bits=8).check(dtype)

    pins = derived(plain_pins)
    admitted = constraint(narrow)
    ports = View(pins, requires=(admitted,))
    key = ViewKey("ports", Pins)
    word = composite(
        "IntegerWord",
        {
            "dtype": Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS),
            "lanes": Param(),
            "payload_bits": derived(bits),
            "pin_values": pins,
            "admitted": admitted,
            "ports": ports,
        },
        base=InterfaceKernel,
        annotations={"dtype": QONNXDataType, "lanes": int},
        exports={key: ports},
    )
    assert issubclass(word, InterfaceKernel) and word.id == InterfaceKernel.id

    class Pair(Kernel):
        id = "test.interface_pair"
        activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        activation = word(dtype=activation_dtype, lanes=2)  # type: ignore[call-arg]
        weights = word(dtype=weights_dtype, lanes=2)  # type: ignore[call-arg]

    point = design_space(
        Pair(
            activation_dtype=resolve_qonnx_datatype_name("INT4"),
            weights_dtype=resolve_qonnx_datatype_name("INT16"),
        )
    )
    activation_ports = getattr(Pair.activation, "ports")
    weights_ports = getattr(Pair.weights, "ports")
    assert point.query(activation_ports) == Available(Pins((("word", 8),), ()))
    assert isinstance(point.query(weights_ports), Rejected)
    assert point.query(getattr(Pair.weights, "payload_bits")) == Available(32)
    authored = [key for key in views(point) if key.endswith("ports")]
    assert authored == ["activation.ports", "weights.ports"]
