# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Resolved products and Vpc artifact parity with the immutable pre-migration leaf."""

from dataclasses import FrozenInstanceError

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import DefinitionError, Rejected, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    HlsSource,
)
from finn.kernels.artifacts.module import BuildError, Held, Leaf, fingerprint
from finn.kernels.base import NATIVE_CLOCKING
from finn.kernels.library import ModuleSpec, assemble_module, library_module, module_unprovided
from finn.kernels.vpc import VpcKernel

# Captured from immutable FINN0dd7e4ac5; full canonical values/pins are retained
# in the output record. This is the original public module fingerprint, not a new oracle.
VPC_FINGERPRINTS = {
    "INT3": "45f1010f94349b82ada2b4fad92e36004f61c6f695caf3837474d6101fb021e1",
    "UINT7": "84415cda7375d1bebcdcb25332828e349e1efc1ca7f7513e09951cab4014a819",
}

VPC_SOURCE = CopiedSource("finnlib", "rtl/shape/vpc.sv", provides=("module:vpc",))


@pytest.mark.parametrize("dtype,lanes_in,lanes_out", [("INT3", 1, 2), ("UINT7", 4, 2)])
def test_resolved_module_matches_actual_existing_vpc_leaf(
    dtype: str, lanes_in: int, lanes_out: int
) -> None:
    tensor = Tensor((2, 4), ScalarEncoding(DataType[dtype]))
    point = design_space(
        VpcKernel(
            input_tensor=tensor,
            arriving=BeatSequence(vector_major(tensor.shape, lanes_in)),
            leaving=BeatSequence(vector_major(tensor.shape, lanes_out)),
        )
    )
    old = library_module(point)
    spec = ModuleSpec("vpc", (VPC_SOURCE,))
    spec.validate("VpcKernel", "VpcKernel.module_spec")
    pin_parts = (point.input.physical_pins, point.output.physical_pins)
    assert not isinstance(pin_parts[0], Rejected) and not isinstance(pin_parts[1], Rejected)
    held_parts = (point.input.idle_pins, point.output.idle_pins)
    assert not isinstance(held_parts[0], Rejected) and not isinstance(held_parts[1], Rejected)
    assert module_unprovided(spec, "VpcKernel", (VPC_SOURCE,)) is None
    new = assemble_module(
        spec,
        VpcKernel.id,
        "VpcKernel",
        point.clocking,
        tuple(sorted(point.parameters.items())),
        (pin_parts[0], pin_parts[1]),
        (VPC_SOURCE,),
        (),
        (held_parts[0], held_parts[1]),
        (True, True),
    )
    assert new == old
    assert isinstance(new, Leaf)
    assert fingerprint(new) == fingerprint(old) == VPC_FINGERPRINTS[dtype]


def test_constant_and_dynamic_source_provision_have_one_plain_contract() -> None:
    first = CopiedSource("fixture", "a.sv", provides=("module:first", "module:other"))
    second = CopiedSource("fixture", "b.sv", provides=("module:second", "module:first"))
    constant = ModuleSpec("missing", (first, second))
    with pytest.raises(DefinitionError) as error:
        constant.validate("Owner", "Owner.module_spec")
    expected = (
        "Owner: its sources provide ['module:first', 'module:other', 'module:second'], "
        "not the library module missing (module:missing)"
    )
    assert expected in str(error.value)
    dynamic = ModuleSpec("missing")
    dynamic.validate("Owner", "Owner.module_spec")
    refused = module_unprovided(dynamic, "Owner", (first, second))
    assert isinstance(refused, Rejected)
    assert [(f.code, f.message) for f in refused.findings] == [
        ("library-module-unprovided", expected)
    ]
    assert module_unprovided(ModuleSpec("first"), "Owner", (first, second)) is None
    with pytest.raises(DefinitionError, match="provide"):
        ModuleSpec("empty", ()).validate("Owner", "Owner.module_spec")
    with pytest.raises(FrozenInstanceError):
        setattr(constant, "rtl_module", "changed")


def test_resolved_input_accounting_and_order_are_preserved() -> None:
    control = Signal("control", Direction.IN, 1)
    data = Signal("data", Direction.IN, 12)
    ready = Signal("ready", Direction.OUT, 1)
    source = CopiedSource("fixture", "module.sv", provides=("module:module",))
    datum = GeneratedData("init.dat", b"01\n")
    spec = ModuleSpec("module", (source,))
    new = assemble_module(
        spec,
        "fixture.module",
        "Owner",
        NATIVE_CLOCKING,
        (("N", 2), ("W", 12)),
        ((control,), (data, ready)),
        (source,),
        datum,
        (Held((("data", 0),), ("ready",)), Held((("control", 0),))),
        (False, False),
    )
    assert isinstance(new, Leaf)
    assert new.abi.pins == (*NATIVE_CLOCKING.signals(), control, data, ready)
    assert new.data == (datum,)
    assert new.sources == (source,)
    assert new.held.inputs == (*NATIVE_CLOCKING.held(), ("data", 0), ("control", 0))
    assert new.held.unused == ("ready",)
    refused = assemble_module(
        spec,
        "fixture.module",
        "Owner",
        NATIVE_CLOCKING,
        (),
        ((control,), (data, ready)),
        (source,),
        (),
        (),
        (False, False),
    )
    assert isinstance(refused, Rejected)
    assert [f.message for f in refused.findings] == [
        "Owner: the inputs ['control', 'data'] are neither held nor presented"
    ]


def test_hls_top_and_generated_data_remain_actual_contribution_values() -> None:
    request = HlsSource.request("top", lambda name: f"void {name}() {{}}", (), 5.0)
    data = GeneratedData("image.dat", b"02\n")
    sources: tuple[Contribution, ...] = (request, data)
    spec = ModuleSpec("top", sources)
    spec.validate("Owner", "Owner.module_spec")
    new = assemble_module(
        spec, "fixture.hls", "Owner", NATIVE_CLOCKING, (), (), sources, data, (), ()
    )
    assert isinstance(new, Leaf)
    assert new.name == request.function
    assert new.sources == (request,)
    assert new.data == (data, data)  # no new normalization of original concatenation
    with pytest.raises(BuildError, match="not the top"):
        assemble_module(
            ModuleSpec("other"),
            "fixture.hls",
            "Owner",
            NATIVE_CLOCKING,
            (),
            (),
            (request,),
            (),
            (),
            (),
        )
