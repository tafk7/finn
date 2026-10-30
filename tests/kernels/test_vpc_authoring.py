# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual Vpc module authoring, local port gates and serial physical input demand."""

from __future__ import annotations

from typing import assert_type

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Group,
    Param,
    Rejected,
    Space,
    Unresolved,
    ViewAssessment,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import Contribution, CopiedSource
from finn.kernels.artifacts.module import Fragment
from finn.kernels.base import Kernel
from finn.kernels.laws import Native
from finn.kernels.library import ModuleSpec
from finn.kernels.port import LoopEnds, port_endpoint, port_surface
from finn.kernels.port_model import PortSurface, SequencePort
from finn.kernels.vpc import VpcKernel
from finn.platform.capabilities import Resources

UNNAMED_ENDS = LoopEnds((LevelEnd(1),))
"""A loop-completion vector, which a native port named in full has no pin for: each port
so declared refuses its pins directly."""

TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SEQUENCE = BeatSequence(vector_major(TENSOR.shape, 4))


class SerialVpc(VpcKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    @derived
    def first(self) -> BeatSequence | Rejected:
        if self.first_state == "refused":
            return reject("first-sequence", "first input blocks")
        return SEQUENCE

    @derived
    def second(self) -> BeatSequence:
        _ = self.second_state
        return SEQUENCE

    input = SequencePort(
        label="input",
        tensor=VpcKernel.input_tensor,
        sequence=first,
        transport=Native(names=("idat", "ivld", "irdy")),
        loop_ends=UNNAMED_ENDS,
        clocking=VpcKernel.port_clocking,
    )
    output = SequencePort(
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=VpcKernel.output_tensor,
        sequence=second,
        transport=Native(names=("odat", "ovld", "ordy")),
        loop_ends=UNNAMED_ENDS,
        clocking=VpcKernel.port_clocking,
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"W": 3, "PI": 4, "PO": 4, "N": 4, "PAD_ZEROS": 1, "RELAX_THROUGHPUT": 0}


@pytest.mark.parametrize("permuted,eager", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize(
    "first,second", [(None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready")]
)
def test_actual_vpc_netlist_keeps_serial_body_demand_and_direct_refusal_order(
    permuted: bool,
    eager: bool,
    first: str | None,
    second: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(SerialVpc(input_tensor=TENSOR, arriving=SEQUENCE, leaving=SEQUENCE))
    chosen: dict[object, object] = {}
    if first is not None:
        chosen[SerialVpc.first_state] = first
    if second is not None:
        chosen[SerialVpc.second_state] = second
    point = point.with_choices(chosen)
    evidence = inspection.explain(point, SerialVpc.netlist)
    keys = {n.declaration.key for n in evidence.nodes}
    result = evidence.result
    if first is None:
        assert isinstance(result, Unresolved)
    elif first == "refused":
        assert isinstance(result, Rejected)
        assert [f.code for f in result.findings] == ["first-sequence"]
    elif second is None:
        assert isinstance(result, Unresolved)
    else:
        assert isinstance(result, Rejected)
        assert [f.message for f in result.findings] == [
            "input: a native port named in full has no loop-completion pin",
            "output: a native port named in full has no loop-completion pin",
        ]
    if first is None or first == "refused":
        # Both child admission rules remain real owner-gate dependencies.
        # Later physical/body inputs must still be untouched.
        assert not keys & {"second", "second_state", "output.stream", "output.transport"}


def test_actual_vpc_surfaces_are_typed_real_children_and_not_kernel_instances() -> None:
    point = design_space(VpcKernel(input_tensor=TENSOR, arriving=SEQUENCE, leaving=SEQUENCE))
    assert_type(VpcKernel.input, SequencePort)
    assert isinstance(point.input, Space) and not isinstance(point.input, Kernel)
    assert point.netlists == ()
    assert isinstance(point.netlist, Fragment)
    assert inspection.statistics(point).owning_decisions == 0
    assert port_endpoint(VpcKernel, "input") is Endpoint.TARGET
    assert port_endpoint(VpcKernel, "output") is Endpoint.INITIATOR
    surface = port_surface(VpcKernel, "output")
    assert surface.side is not None and surface.contract_reference() is not None
    assert point.query(surface.side) == Available(point.output.side)
    assert point.query(surface.contract_reference()) == Available(point.output.contract)
    with pytest.raises(KeyError):
        port_surface(VpcKernel, "missing")
    with pytest.raises(TypeError, match="node declaration"):
        port_surface(point, "input")


def test_actual_parent_refusal_blocks_module_body_but_not_valid_child_characterization() -> None:
    class Refused(VpcKernel):
        @constraint
        def parent_composition(self) -> bool:
            return False

    point = design_space(Refused(input_tensor=TENSOR, arriving=SEQUENCE, leaving=SEQUENCE))
    evidence = inspection.explain(point, Refused.netlist)
    assert isinstance(evidence.result, Rejected)
    assert isinstance(evidence.assessment, ViewAssessment)
    assert evidence.assessment.output_result is None
    assert isinstance(point.query(Refused.input.physical_pins), Available)


def test_plain_module_definition_still_requires_one_module_and_its_parameters() -> None:
    with pytest.raises(DefinitionError, match="library modules"):

        class Repeated(VpcKernel):
            another = ModuleSpec("vpc", VpcKernel.module_spec.constant_sources)

    # The base assembles the netlist (no kernel states it); the parameters are the kernel's.
    with pytest.raises(DefinitionError, match="states no parameters"):

        class Unstated(Kernel):
            id = "fixture.unstated"
            module_spec = VpcKernel.module_spec


def test_plain_module_spec_rejects_symbolic_identity_and_sources() -> None:
    class Inputs(Space):
        name: str = Param()
        sources: tuple[Contribution, ...] = Param()

    with pytest.raises(TypeError, match="constant string"):
        ModuleSpec(Inputs.name)
    # Runtime class reference deliberately crosses the plain-value boundary.
    with pytest.raises(TypeError, match="constant contribution"):
        ModuleSpec("vpc", Inputs.sources)


def test_a_module_requires_owned_physical_ports_not_parent_forwarding_metadata() -> None:
    with pytest.raises(DefinitionError, match="declares no port"):

        class Unowned(Kernel):
            id = "fixture.unowned_builder"
            module_spec = ModuleSpec(
                "probe", (CopiedSource("finnlib", "probe.sv", provides=("module:probe",)),)
            )
            input = PortSurface(None, Endpoint.TARGET, None, None)

            @view
            def parameters(self) -> dict[str, int | str]:
                return {}

            @view
            def resource_use(self) -> Resources:
                return Resources()


def test_surface_lookup_supports_discovered_group_paths_at_the_actual_node() -> None:
    class Included(Group):
        port = SequencePort(label="p", tensor=TENSOR, sequence=SEQUENCE)

    class Host(Space):
        group = Included()

    declaration = Host()
    surface = port_surface(declaration, "group.port")

    class Envelope(Space):
        host = declaration

    point = design_space(Envelope())
    assert surface.stream is not None
    assert point.query(surface.stream) == Available(point.host.group.port.stream)
    with pytest.raises(KeyError):
        port_surface(Host, "group.missing")
