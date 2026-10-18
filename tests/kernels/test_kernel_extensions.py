# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from support.refusals import refused

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
)
from finn.core.space.errors import DefinitionError, RequestError
from finn.dataflow.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    resolve_qonnx_datatype_name,
)
from finn.kernels.base import ComposedKernel, Kernel
from finn.kernels.configure import chosen, commit
from finn.kernels.values.domains import Integer
from finn.kernels.values.semantics import INTEGER_VECTOR, QONNX_DATATYPE_VALUE_SEMANTICS


def views(point: object) -> list[str]:
    return [item.key for item in inspection.members(point) if item.kind == "view"]


@dataclass(frozen=True)
class Pins:
    inputs: tuple[tuple[str, int], ...]
    outputs: tuple[tuple[str, int], ...]


def test_kernel_identity_is_validated_at_class_creation() -> None:
    with refused(
        "kernel-id-invalid",
        error=DefinitionError,
        message="nonempty string id",
        declaration="Unnamed",
    ):
        type("Unnamed", (Kernel,), {})
    # The version is a positive int (an op's opset version); artifacts carry str(version).
    for wrong in ("1", 0, True):
        with refused(
            "kernel-version-invalid",
            error=DefinitionError,
            message="positive int version",
            declaration="WrongVersion",
        ):
            type("WrongVersion", (Kernel,), {"id": "test.wrong", "version": wrong})

    class Empty(ComposedKernel):
        id = "test.empty"
        interfaces = ()

    assert Empty.version == 1
    # The protocol's views; a kernel with children that places none has no netlist.
    empty = design_space(Empty())
    assert views(empty) == ["buffering", "cycles", "netlist", "resources"]
    rejection = empty.query(Empty.netlist)
    assert isinstance(rejection, Rejected)
    assert {finding.code for finding in rejection.findings} == {"kernel-module"}


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


def test_a_dtype_choice_replays_the_exact_datatype() -> None:
    class DtypeKernel(ComposedKernel):
        id = "test.dtype"
        interfaces = ()
        dtype: QONNXDataType = Decision(
            values=(resolve_qonnx_datatype_name("TERNARY"), resolve_qonnx_datatype_name("INT2")),
            semantics=QONNX_DATATYPE_VALUE_SEMANTICS,
        )

    base = design_space(DtypeKernel())
    point = base.with_choices(dtype=resolve_qonnx_datatype_name("TERNARY"))
    replayed = commit(base, chosen(point))
    assert replayed.dtype.name == "TERNARY"
    assert not QONNX_DATATYPE_VALUE_SEMANTICS.values_equal(
        replayed.dtype, resolve_qonnx_datatype_name("INT2")
    )
    with pytest.raises(RequestError):
        base.with_choices(dtype=cast(QONNXDataType, "TERNARY"))


def test_the_integer_vector_adapter_preserves_exact_integers() -> None:
    assert INTEGER_VECTOR.accepts((1, 0, -2))
    assert INTEGER_VECTOR.accepts(())
    assert not INTEGER_VECTOR.accepts((True, 2))
    assert not INTEGER_VECTOR.accepts([1, 2])
    vector = (1, 2)
    assert INTEGER_VECTOR.freeze(vector) is vector


def test_explicit_dtype_semantics_support_typed_protocol_results_of_immutable_values() -> None:
    class Typed(ComposedKernel):
        id = "test.typed"
        interfaces = ()
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
    class InterfaceKernel(ComposedKernel):
        id = "test.interface"
        interfaces = ()

    def bits(*, dtype: QONNXDataType, lanes: int) -> int:
        return dtype.bitwidth() * lanes

    def plain_pins(*, payload_bits: int) -> Pins:
        return Pins((("word", payload_bits),), ())

    def narrow(*, dtype: QONNXDataType) -> bool | Rejected:
        return Integer(max_bits=8).check(dtype)

    pins = derived(plain_pins)
    admitted = constraint(narrow)
    ports = View(pins)
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

    class Pair(ComposedKernel):
        id = "test.interface_pair"
        interfaces = ()
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
