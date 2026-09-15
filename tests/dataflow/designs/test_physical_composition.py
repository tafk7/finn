# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The selected Design owns a checked, reusable decomposed physical result."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Mapping
from typing import ClassVar, cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.ops.mvau.test_dot_product_design import Placed, _occurrence, _unconfigured
from finn.dataflow._engine import Absent, Answer, Decided, Unresolved
from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.build import (
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    PreparedGeneratedModuleName,
    ScalarTable,
    materialize_module_sources,
    prepare_module_build,
    render_module_sources,
)
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.artifacts.rtl import check_abi
from finn.dataflow.designs.design import DataflowDesign, KernelChoice, NetworkBoundary
from finn.dataflow.designs.physical import (
    ConstantBits,
    DECOMPOSED_PRODUCER,
    DECOMPOSED_WRAPPER_TEMPLATE,
    DesignPhysicalFacts,
    PhysicalCompositionError,
    PhysicalPin,
    PhysicalStructure,
    PhysicalWire,
    PinSlice,
    SemanticPortBinding,
    compose_decomposed,
    lower_module_structure,
    selected_kernel_realization,
    validate_design_physical_facts,
)
from finn.dataflow.kernels.kernel import Kernel, RegionDeclaration
from finn.dataflow.kernels.physical import (
    KernelRealizationFacts,
    KernelStreamBinding,
    PeriodicLast,
    low_fields_binding,
)
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign, WeightSupply
from finn.dataflow.space.declarations import Decision, Input, Problem, Space, Subspace


def _int3_design() -> DotProductDesign:
    return _occurrence(
        repetitions=1,
        matrix_width=4,
        matrix_height=4,
        activation="INT3",
        weight="INT3",
        accumulator="INT16",
        narrow=True,
        pe=2,
        simd=2,
        pumping=True,
    )


def _realizations(
    design: DotProductDesign,
) -> tuple[KernelRealizationFacts, KernelRealizationFacts]:
    replay = selected_kernel_realization(design, "replay")
    compute = selected_kernel_realization(design, "compute")
    assert isinstance(replay, Decided), replay
    assert isinstance(compute, Decided), compute
    return replay.value, compute.value


def test_external_dot_product_produces_the_complete_checked_facts() -> None:
    design = _int3_design()
    answer = design.physical.accepted_answer
    assert isinstance(answer, Decided), answer
    facts = answer.value
    requirements = facts.requirements

    assert requirements.implementation_id == "finn.mvau.decomposed.external"
    assert requirements.implementation_version == "1"
    assert requirements.parameters == ()
    assert requirements.abi.entry_point == GeneratedModuleName("finn_mvau_external")
    assert tuple(item.node_id for item in facts.port_bindings) == (
        "replay",
        "replay",
        "compute",
        "compute",
        "compute",
    )
    assert tuple(item.boundary_id for item in facts.boundary_bindings) == (
        "activation",
        "weight",
        "output",
    )
    assert facts.edge_bindings[0].edge_id == "activation_replay"

    activation = facts.boundary_bindings[0].payload
    assert tuple(
        (item.field_index, item.bit_offset, item.bit_width) for item in activation.fields
    ) == (
        (0, 0, 3),
        (1, 3, 3),
    )
    assert tuple((item.bit_offset, item.bit_width) for item in activation.unused) == ((6, 2),)


def test_lowering_flattens_sources_and_renders_explicit_padding(
    tmp_path: Path,
) -> None:
    answer = _int3_design().physical.accepted_answer
    assert isinstance(answer, Decided), answer
    requirements = answer.value.requirements
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        requirements,
        roots={"finnlib": Path("deps/finnlib")},
        template_roots=(Path("src/finn/dataflow/designs/templates"),),
        blobs=store,
    )
    assert isinstance(prepared.name, PreparedGeneratedModuleName)
    assert prepared.abi.entry_point == f"finn_mvau_external__{prepared.name.seed}"
    assert tuple(
        source.source.path if hasattr(source, "source") else source.output_path
        for source in prepared.sources
    ) == (
        "rtl/infra/replay_buffer.sv",
        "rtl/arith/add_multi_pkg.sv",
        "rtl/arith/add_multi.sv",
        "rtl/linalg/dotp_8sx9_dsp58.sv",
        "rtl/linalg/dotp.sv",
        "rtl/linalg/dotp_axi.sv",
        f"{prepared.abi.entry_point}.sv",
    )
    rendered = render_module_sources(prepared, store)
    wrapper = dict(rendered.contents)[f"{prepared.abi.entry_point}.sv"].decode()
    assert "s_axis_input_tdata[7:6] = 2'b00" in wrapper
    assert "s_axis_weights_tdata[15:12] = 4'b0000" in wrapper
    assert ".NARROW_WEIGHTS(1)" in wrapper
    assert ".ofin()" in wrapper
    assert "u_replay" in wrapper and "u_compute" in wrapper
    stored = materialize_module_sources(prepared, store)
    assert (
        check_abi(
            prepared.abi,
            tuple(Path(stored.directory) / path for path in stored.files),
            prepared.abi.entry_point,
        )
        == ()
    )


def test_equal_selected_points_have_equal_whole_op_requirements() -> None:
    left = _int3_design().physical.accepted_answer
    right = _int3_design().physical.accepted_answer
    assert isinstance(left, Decided) and isinstance(right, Decided)
    assert left.value.requirements == right.value.requirements


def test_a_physical_only_unassigned_choice_does_not_block_logical_acceptance() -> None:
    design = _unconfigured(
        repetitions=1,
        matrix_width=4,
        matrix_height=4,
        activation="INT3",
        weight="INT3",
        accumulator="INT16",
        narrow=True,
    )
    selected_root = cast(
        Placed,
        design.assign(DotProductDesign.pe, 2)
        .assign(DotProductDesign.simd, 2)
        .assign(DotProductDesign.weight_supply, WeightSupply.EXTERNAL)
        .compute.select("dotp_axi")
        .root,
    )
    design = selected_root.design
    assert isinstance(design.dataflow.accepted_answer, Decided)
    assert isinstance(design.physical.accepted_answer, Unresolved)
    assert design.physical.readiness.ready is None


@pytest.mark.parametrize("supply", [WeightSupply.EMBEDDED, WeightSupply.DECOUPLED])
def test_unimplemented_supply_profiles_refuse_normally(supply: WeightSupply) -> None:
    answer = _occurrence(supply).physical.accepted_answer
    assert isinstance(answer, Absent)
    assert {item.code for item in answer.findings} == {"design-physically-unsupported"}


def test_lane_reversal_and_nonzero_padding_are_rejected() -> None:
    design = _int3_design()
    replay, compute = _realizations(design)
    structure = compose_decomposed(replay=replay, compute=compute)
    accepted = design.physical.accepted_answer
    assert isinstance(accepted, Decided)
    facts = accepted.value
    network = design.dataflow.accepted_answer
    assert isinstance(network, Decided)

    wires = list(structure.wires)
    first = next(
        index
        for index, wire in enumerate(wires)
        if wire.destination.pin == PhysicalPin("u_replay", "idat")
        and wire.destination.bit_offset == 0
    )
    second = next(
        index
        for index, wire in enumerate(wires)
        if wire.destination.pin == PhysicalPin("u_replay", "idat")
        and wire.destination.bit_offset == 3
    )
    first_source = wires[first].source
    second_source = wires[second].source
    wires[first] = replace(wires[first], source=second_source)
    wires[second] = replace(wires[second], source=first_source)
    reversed_structure = replace(structure, wires=tuple(wires))
    reversed_facts = replace(
        facts,
        requirements=lower_module_structure(
            reversed_structure,
            producer=DECOMPOSED_PRODUCER,
            wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
        ),
    )
    with pytest.raises(PhysicalCompositionError, match="does not preserve"):
        validate_design_physical_facts(network.value, reversed_structure, reversed_facts)

    wires = list(structure.wires)
    padding = next(
        index
        for index, wire in enumerate(wires)
        if wire.destination.pin == PhysicalPin("u_compute", "s_axis_input_tdata")
        and isinstance(wire.source, ConstantBits)
    )
    wires[padding] = replace(wires[padding], source=ConstantBits(2, 3))
    nonzero_padding = replace(structure, wires=tuple(wires))
    nonzero_facts = replace(
        facts,
        requirements=lower_module_structure(
            nonzero_padding,
            producer=DECOMPOSED_PRODUCER,
            wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
        ),
    )
    with pytest.raises(PhysicalCompositionError, match="driven to zero"):
        validate_design_physical_facts(network.value, nonzero_padding, nonzero_facts)

    wrong_framing_ports = tuple(
        replace(
            binding,
            local=replace(binding.local, framing=PeriodicLast("tlast", 1, 0)),
        )
        if binding.local.framing is not None
        else binding
        for binding in facts.port_bindings
    )
    with pytest.raises(PhysicalCompositionError, match="synapse-fold"):
        validate_design_physical_facts(
            network.value,
            structure,
            replace(facts, port_bindings=wrong_framing_ports),
        )


def test_missing_or_duplicate_destination_bits_refuse_at_structure_construction() -> None:
    replay, compute = _realizations(_int3_design())
    structure = compose_decomposed(replay=replay, compute=compute)
    with pytest.raises(PhysicalCompositionError, match="coverage"):
        replace(structure, wires=structure.wires[:-1])
    with pytest.raises(PhysicalCompositionError, match="more than one driver"):
        replace(structure, wires=(*structure.wires, structure.wires[0]))


def test_lowering_preserves_bit_zero_slices_of_vector_pins() -> None:
    design = _int3_design()
    replay, compute = _realizations(design)
    structure = compose_decomposed(replay=replay, compute=compute)
    answer = design.physical.accepted_answer
    network = design.dataflow.accepted_answer
    assert isinstance(answer, Decided) and isinstance(network, Decided)

    wires: list[PhysicalWire] = []
    for wire in structure.wires:
        if (
            wire.destination.pin == PhysicalPin("u_compute", "s_axis_input_tdata")
            and wire.destination.bit_offset == 0
        ):
            assert wire.destination.bit_width == 3
            assert isinstance(wire.source, PinSlice)
            wires.extend(
                replace(
                    wire,
                    destination=replace(wire.destination, bit_offset=bit, bit_width=1),
                    source=replace(wire.source, bit_offset=bit, bit_width=1),
                )
                for bit in range(3)
            )
        else:
            wires.append(wire)

    split = replace(structure, wires=tuple(wires))
    requirements = _requirements_for_structure(split)
    validate_design_physical_facts(
        network.value,
        split,
        replace(answer.value, requirements=requirements),
    )

    assignments = dict(requirements.render_inputs)["ASSIGNMENTS"]
    assert isinstance(assignments, str)
    assert "assign n__u_compute__s_axis_input_tdata[0] = n__u_replay__odat[0];" in assignments
    assert "assign n__u_compute__s_axis_input_tdata = n__u_replay__odat;" not in assignments
    assert "assign n__u_replay__clk = ap_clk;" in assignments


def _requirements_for_structure(structure: PhysicalStructure) -> ModuleBuildRequirements:
    return lower_module_structure(
        structure,
        producer=DECOMPOSED_PRODUCER,
        wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
    )


def test_top_clock_alignment_reset_domains_and_bus_domains_are_required() -> None:
    design = _int3_design()
    replay, compute = _realizations(design)
    structure = compose_decomposed(replay=replay, compute=compute)
    answer = design.physical.accepted_answer
    network = design.dataflow.accepted_answer
    assert isinstance(answer, Decided) and isinstance(network, Decided)

    free_2x_ports = tuple(
        replace(port, role=Clock(Free()))
        if isinstance(port, Signal) and port.name == "ap_clk2x"
        else port
        for port in structure.top_abi.ports
    )
    free_2x = replace(
        structure,
        top_abi=replace(structure.top_abi, ports=free_2x_ports, clock_alignments=()),
    )
    with pytest.raises(PhysicalCompositionError, match="derived ap_clk2x"):
        validate_design_physical_facts(
            network.value,
            free_2x,
            replace(answer.value, requirements=_requirements_for_structure(free_2x)),
        )

    single_domain_ports = tuple(
        replace(
            port,
            role=Reset(
                active_low=True,
                synchronous=True,
                synchronous_to=("ap_clk",),
            ),
        )
        if isinstance(port, Signal) and port.name == "ap_rst_n"
        else port
        for port in structure.top_abi.ports
    )
    single_domain = replace(
        structure,
        top_abi=replace(structure.top_abi, ports=single_domain_ports),
    )
    with pytest.raises(PhysicalCompositionError, match="synchronous to both clocks"):
        validate_design_physical_facts(
            network.value,
            single_domain,
            replace(answer.value, requirements=_requirements_for_structure(single_domain)),
        )

    wrong_bus_ports = tuple(
        replace(port, associated_clock="ap_clk2x")
        if isinstance(port, Bus) and port.name == "in0_V"
        else port
        for port in structure.top_abi.ports
    )
    wrong_bus = replace(
        structure,
        top_abi=replace(structure.top_abi, ports=wrong_bus_ports),
    )
    with pytest.raises(PhysicalCompositionError, match="top stream"):
        validate_design_physical_facts(
            network.value,
            wrong_bus,
            replace(answer.value, requirements=_requirements_for_structure(wrong_bus)),
        )

    replay_instance, compute_instance = structure.instances
    extra_compute_abi = replace(
        compute_instance.requirements.abi,
        ports=(
            *compute_instance.requirements.abi.ports,
            Signal("extra_clk", Direction.IN, 1, Clock(Free())),
        ),
    )
    extra_compute = replace(
        compute_instance,
        requirements=replace(compute_instance.requirements, abi=extra_compute_abi),
    )
    extra_pin = replace(
        structure,
        instances=(replay_instance, extra_compute),
        wires=(
            *structure.wires,
            PhysicalWire(
                PinSlice(PhysicalPin("u_compute", "extra_clk"), 0, 1),
                PinSlice(PhysicalPin(None, "ap_clk"), 0, 1),
            ),
        ),
    )
    with pytest.raises(PhysicalCompositionError, match="Dotp pin/interface inventory"):
        validate_design_physical_facts(
            network.value,
            extra_pin,
            replace(answer.value, requirements=_requirements_for_structure(extra_pin)),
        )


def _single_output_region(width: int) -> DataflowRegion:
    schedule = LogicalSchedule((ScheduleLevel("beat", 1),))
    operand = Operand("value", DataType["UINT8"], (1,))
    sequence = BeatSequence(1, (((0,),),))
    availability = ScheduledOutputAvailability((((0,), (0,)),))
    return DataflowRegion(
        schedule, (), (OutputInterface(Port("output", operand, sequence), availability),)
    )


class _GoodKernel(Kernel):
    id = "physical_good"
    width = Input(int)
    region = RegionDeclaration(
        family="test.physical.good", version="1", construct=_single_output_region, width=width
    )
    sources = (CopiedSource("test", "good.sv", provides=("module:physical_good",)),)

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        del parameters
        return ComponentABI(
            "physical_good",
            (
                Bus(
                    "out",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "out_tdata", 8),
                        Member("tvalid", "out_tvalid"),
                        Member("tready", "out_tready"),
                    ),
                    endpoint=Endpoint.INITIATOR,
                ),
            ),
        )

    @classmethod
    def local_stream_bindings(
        cls,
        *,
        region: DataflowRegion,
        parameters: ScalarTable,
        abi: ModuleABIRequirements,
    ) -> tuple[KernelStreamBinding, ...]:
        del parameters
        return (
            low_fields_binding(region=region, abi=abi, region_port_id="output", abi_bus_id="out"),
        )


class _TrapKernel(_GoodKernel):
    id = "physical_trap"
    calls: ClassVar[int] = 0

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        del parameters
        cls.calls += 1
        raise AssertionError("an unselected physical provider ran")


def _choice_dependent_region(width: int, mode: bool) -> DataflowRegion:
    del mode
    return _single_output_region(width)


class _ChoiceDependentKernel(_GoodKernel):
    id = "choice_dependent"
    mode = Input(bool)
    region = RegionDeclaration(
        family="test.physical.choice_dependent",
        version="1",
        construct=_choice_dependent_region,
        width=_GoodKernel.width,
        mode=mode,
    )


class _SelectivePhysicalDesign(DataflowDesign):
    id = "selective_physical"
    width = Input(int)
    only = KernelChoice(Subspace(_GoodKernel, width=width), Subspace(_TrapKernel, width=width))
    result = NetworkBoundary(only.output("output"))

    def physical_implementation(self) -> Answer[DesignPhysicalFacts]:
        child = selected_kernel_realization(self, "only")
        if not isinstance(child, Decided):
            return cast("Answer[DesignPhysicalFacts]", child)
        return Decided(
            DesignPhysicalFacts(
                child.value.requirements,
                (SemanticPortBinding("only", "u_only", child.value.streams[0]),),
                (),
                (),
            )
        )


class _SelectiveRoot(Space):
    width = Problem(int)
    design = Subspace(_SelectivePhysicalDesign, width=width)


class _PhysicalChoiceDesign(DataflowDesign):
    id = "physical_choice"
    width = Input(int)
    physical_mode = Decision(bool, values=(False, True))
    only = KernelChoice(Subspace(_GoodKernel, width=width))
    result = NetworkBoundary(only.output("output"))

    def physical_implementation(self) -> Answer[DesignPhysicalFacts]:
        mode = self.answer(type(self).physical_mode)
        if not isinstance(mode, Decided):
            return cast("Answer[DesignPhysicalFacts]", mode)
        return cast(
            "Answer[DesignPhysicalFacts]",
            super().physical_implementation(),
        )


class _PhysicalChoiceRoot(Space):
    width = Problem(int)
    design = Subspace(_PhysicalChoiceDesign, width=width)


class _UnusedBranchChoiceDesign(_SelectivePhysicalDesign):
    id = "unused_branch_choice"
    mode = Decision(bool, values=(False, True))
    only = KernelChoice(
        Subspace(_GoodKernel, width=_SelectivePhysicalDesign.width),
        Subspace(
            _ChoiceDependentKernel,
            width=_SelectivePhysicalDesign.width,
            mode=mode,
        ),
    )
    result = NetworkBoundary(only.output("output"))


class _UnusedBranchChoiceRoot(Space):
    width = Problem(int)
    design = Subspace(_UnusedBranchChoiceDesign, width=width)


def test_physical_dispatch_does_not_evaluate_an_unselected_candidate() -> None:
    _TrapKernel.calls = 0
    root = _SelectiveRoot.start({_SelectiveRoot.width: 1}, namespace="selected")
    design = root.design
    selected_root = cast(_SelectiveRoot, design.only.select("physical_good").root)
    selected = selected_root.design
    assert isinstance(selected.physical.accepted_answer, Decided)
    assert _TrapKernel.calls == 0


def test_an_unselected_branch_does_not_require_its_design_owned_choice() -> None:
    root = _UnusedBranchChoiceRoot.start({_UnusedBranchChoiceRoot.width: 1})
    selected_root = cast(
        _UnusedBranchChoiceRoot,
        root.design.only.select("physical_good").root,
    )
    design = selected_root.design
    assert isinstance(design.dataflow.accepted_answer, Decided)
    assert isinstance(design.physical.accepted_answer, Decided)


def test_a_design_physical_only_choice_does_not_block_the_network() -> None:
    design = _PhysicalChoiceRoot.start({_PhysicalChoiceRoot.width: 1}).design
    assert isinstance(design.dataflow.accepted_answer, Decided)
    assert isinstance(design.physical.accepted_answer, Unresolved)


def test_base_design_dispatch_reports_an_unsupported_profile() -> None:
    class Unsupported(DataflowDesign):
        id = "unsupported_physical"
        width = Input(int)
        only = KernelChoice(Subspace(_GoodKernel, width=width))
        result = NetworkBoundary(only.output("output"))

    class Root(Space):
        width = Problem(int)
        design = Subspace(Unsupported, width=width)

    answer = Root.start({Root.width: 1}).design.physical.accepted_answer
    assert isinstance(answer, Absent)
    assert {item.code for item in answer.findings} == {"design-physically-unsupported"}


def test_source_op_fixture_derives_narrow_weights_true() -> None:
    design = _int3_design()
    compute = selected_kernel_realization(design, "compute")
    assert isinstance(compute, Decided)
    assert dict(compute.value.requirements.parameters)["NARROW_WEIGHTS"] is True


def test_fixture_source_root_is_still_the_same_selected_point() -> None:
    """Keep the helper tied to the real MVAU input declarations."""

    assert Placed.activation_type.value_semantics.freeze(DataType["INT3"]) == DataType["INT3"]
