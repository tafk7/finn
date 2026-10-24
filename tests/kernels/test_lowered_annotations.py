# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's hand-written annotations of the members its ports lower to.

A kernel reads the members its declarations add to it (``input_stream``, ``input_data_width``
for a stream port named ``input``) by the names the lowering fixes, and states each with a
plain ``ClassVar`` annotation so the type checker knows what it is. The class refuses an
annotation that names a lowered member with another type, and one that names nothing the
class declares or lowers, when it is made.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from finn.core.space import DefinitionError, Param, view
from finn.core.space.compiler import compile_model
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace
from finn.kernels.library import LibraryModule
from finn.kernels.port import native_stream
from finn.platform.capabilities import Resources

PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))


def test_an_annotation_matching_the_lowered_member_is_accepted() -> None:
    class Annotated(KernelSpace):
        id = "test.lowered_annotations.matching"
        clock_domain = NATIVE_CLOCKING

        input_tensor: Tensor = Param()
        arriving: BeatSequence = Param()
        input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)
        input_data_width: ClassVar[int]

        @view
        def parameters(self) -> dict[str, int | str]:
            return {"WIDTH": 4}

        @view
        def resource_use(self) -> Resources:
            return Resources(lut=3)

        @view
        def held_bits(self) -> int:
            return 5

        module = LibraryModule("probe", parameters=parameters, sources=(PROBE,))

    assert "input_data_width" in Annotated.__annotations__
    assert "input_data_width" in compile_model(Annotated).scopes.keys


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="flips when the kernel base checks its annotations: a lowered member of another "
    "type is refused when the class is made",
)
def test_an_annotation_of_the_wrong_type_for_a_lowered_member_is_refused() -> None:
    with pytest.raises(DefinitionError, match="input_data_width"):

        class Mistyped(KernelSpace):
            id = "test.lowered_annotations.mistyped"
            clock_domain = NATIVE_CLOCKING

            input_tensor: Tensor = Param()
            arriving: BeatSequence = Param()
            input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)
            input_data_width: ClassVar[str]

            @view
            def parameters(self) -> dict[str, int | str]:
                return {"WIDTH": 4}

            @view
            def resource_use(self) -> Resources:
                return Resources(lut=3)

            @view
            def held_bits(self) -> int:
                return 5

            module = LibraryModule("probe", parameters=parameters, sources=(PROBE,))


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="flips when the kernel base checks its annotations: one naming nothing the class "
    "declares or lowers is refused when the class is made",
)
def test_an_annotation_naming_nothing_the_class_declares_or_lowers_is_refused() -> None:
    with pytest.raises(DefinitionError, match="input_width"):

        class Dangling(KernelSpace):
            id = "test.lowered_annotations.dangling"
            clock_domain = NATIVE_CLOCKING

            input_tensor: Tensor = Param()
            arriving: BeatSequence = Param()
            input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)
            input_width: ClassVar[int]

            @view
            def parameters(self) -> dict[str, int | str]:
                return {"WIDTH": 4}

            @view
            def resource_use(self) -> Resources:
                return Resources(lut=3)

            @view
            def held_bits(self) -> int:
                return 5

            module = LibraryModule("probe", parameters=parameters, sources=(PROBE,))
