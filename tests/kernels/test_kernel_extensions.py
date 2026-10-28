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
    Group,
    Param,
    Rejected,
    Select,
    Space,
    composite,
    constraint,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space.errors import DefinitionError, RequestError
from finn.core.space.semantics import annotation_type, canonical
from finn.dataflow.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.kernels.base import KernelSpace
from finn.kernels.configure import chosen, commit
from finn.kernels.port import _child_types
from finn.kernels.values.domains import Integer


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
        type("Unnamed", (KernelSpace,), {})

    class Empty(KernelSpace):
        id = "test.empty"

    # The base's views; a kernel that binds no module and places none has no netlist.
    empty = design_space(Empty())
    assert views(empty) == [
        "buffering",
        "bus_control",
        "cycles",
        "held_bits",
        "netlist",
        "resource_use",
        "resources",
    ]
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
def test_the_datatype_annotation_takes_each_qonnx_datatype_as_it_is(name: str) -> None:
    """``QONNXDataType`` is the check: every datatype qonnx resolves is an instance of
    ``BaseDataType``, its token; a datatype's canonical name is not one. A datatype is a
    registered value type, so its canonical form is the value itself."""
    value = resolve_qonnx_datatype_name(name)
    check = annotation_type(QONNXDataType, "dtype", owner="test")
    assert check.type_token is QONNXDataType
    assert check.accepts(value)
    assert canonical(value) is value
    assert not check.accepts(name)


def test_a_dtype_choice_replays_the_exact_datatype() -> None:
    class DtypeKernel(KernelSpace):
        id = "test.dtype"
        dtype: QONNXDataType = Decision(
            domain=(resolve_qonnx_datatype_name("TERNARY"), resolve_qonnx_datatype_name("INT2"))
        )

    base = design_space(DtypeKernel())
    point = base.with_choices(dtype=resolve_qonnx_datatype_name("TERNARY"))
    replayed = commit(base, chosen(point))
    assert replayed.dtype.name == "TERNARY"
    assert replayed.dtype != resolve_qonnx_datatype_name("INT2")
    with pytest.raises(RequestError):
        base.with_choices(dtype=cast(QONNXDataType, "TERNARY"))


def test_a_datatype_fact_and_derived_value_are_the_datatype_itself() -> None:
    class Typed(KernelSpace):
        id = "test.typed"
        dtype: QONNXDataType = Param()

        @derived
        def result(self) -> QONNXDataType:
            return self.dtype

        physical = view(result)

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
    class InterfaceKernel(KernelSpace):
        id = "test.interface"

    def bits(*, dtype: QONNXDataType, lanes: int) -> int:
        return dtype.bitwidth() * lanes

    def plain_pins(*, payload_bits: int) -> Pins:
        return Pins((("word", payload_bits),), ())

    def narrow(*, dtype: QONNXDataType) -> bool | Rejected:
        return Integer(max_bits=8).check(dtype)

    pins = derived(plain_pins)
    admitted = constraint(narrow)
    ports = view(pins)
    word = composite(
        "IntegerWord",
        {
            "dtype": Param(),
            "lanes": Param(),
            "payload_bits": derived(bits),
            "pin_values": pins,
            "admitted": admitted,
            "ports": ports,
        },
        base=InterfaceKernel,
        annotations={"dtype": QONNXDataType, "lanes": int},
    )
    assert issubclass(word, InterfaceKernel) and word.id == InterfaceKernel.id

    class Pair(KernelSpace):
        id = "test.interface_pair"
        activation_dtype: QONNXDataType = Param()
        weights_dtype: QONNXDataType = Param()
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
    assert inspection.validity(point.weights).status == "refused"
    assert point.query(getattr(Pair.weights, "payload_bits")) == Available(32)
    authored = [key for key in views(point) if key.endswith("ports")]
    assert authored == ["activation.ports", "weights.ports"]


def test_legacy_served_discovery_accepts_only_real_space_children() -> None:
    class Child(Space):
        pass

    class Included(Group):
        pass

    assert _child_types(Child()) == (Child,)
    assert _child_types(Select(child=Child(), absent=None)) == (Child,)
    assert _child_types(Included()) == ()
