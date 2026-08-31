# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D1 forcing cases for the operation-generic ``DataflowDesign`` layer."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

import finn.dataflow.authoring.design as design_authoring
from finn.dataflow.authoring import OpDesign, Ref, divisors_of, finite
from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignEntry,
    DataflowDesignInventory,
    DataflowDesignScope,
    DesignNode,
    InputSupplyAlternative,
    InputSupplyContext,
    SupplierAttachment,
    declare_dataflow_design,
    declare_dataflow_design_inventory,
    declare_input_supply,
)
from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.hardware import (
    ComputationContract,
    HardwareDesign,
    HardwareKernel,
    PhysicalComponent,
)
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.spec_algebra import assemble_specs

INT8 = DataType["INT8"]
COPY = ComputationContract("synthetic_copy")
SUPPLY = ComputationContract("synthetic_parameter_supply")


def _region(extent: int, name: str) -> DataflowRegion:
    schedule = LogicalSchedule((ScheduleLevel("element", extent),))
    positions = tuple((index,) for index in range(extent))
    beats = BeatSequence(1, tuple((position,) for position in positions))
    source = Operand(f"{name}_input", INT8, (extent,))
    result = Operand(f"{name}_output", INT8, (extent,))
    requirements_data: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        ((index,), (index,)): 1 for index in range(extent)
    }
    availability_data: dict[tuple[int, ...], tuple[int, ...]] = {
        (index,): (index,) for index in range(extent)
    }
    requirements = ScheduledInputRequirements(requirements_data)
    availability = ScheduledOutputAvailability(availability_data)
    return DataflowRegion(
        schedule,
        (InputInterface(Port("input", source, beats), requirements),),
        (OutputInterface(Port("output", result, beats), availability),),
    )


def _supplier_region(consumer: InputInterface) -> DataflowRegion:
    port = Port("supplied", consumer.port.operand, consumer.port.beat_sequence)
    return DataflowRegion(
        LogicalSchedule(()),
        (),
        (
            OutputInterface(
                port,
                ScheduledOutputAvailability(
                    {position: () for position in port.beat_sequence.image}
                ),
            ),
        ),
    )


def _chain(producer: DataflowRegion, consumer: DataflowRegion) -> DataflowNetwork:
    beats = producer.output_interface("output").port.beat_sequence
    return DataflowNetwork(
        (NetworkNode("producer", producer), NetworkNode("consumer", consumer)),
        (
            Edge(
                "link",
                RegionEndpoint("producer", "output"),
                (
                    SinkContract(
                        RegionEndpoint("consumer", "input"),
                        PositionMap.identity(beats.image),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "input",
                RegionEndpoint("producer", "input"),
                producer.input_interface("input").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint("consumer", "output"),
                consumer.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def _fanout(
    producer: DataflowRegion, left: DataflowRegion, right: DataflowRegion
) -> DataflowNetwork:
    beats = producer.output_interface("output").port.beat_sequence
    return DataflowNetwork(
        (
            NetworkNode("producer", producer),
            NetworkNode("left", left),
            NetworkNode("right", right),
        ),
        (
            Edge(
                "fanout",
                RegionEndpoint("producer", "output"),
                (
                    SinkContract(
                        RegionEndpoint("left", "input"), PositionMap.identity(beats.image)
                    ),
                    SinkContract(
                        RegionEndpoint("right", "input"), PositionMap.identity(beats.image)
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "input",
                RegionEndpoint("producer", "input"),
                producer.input_interface("input").port.beat_sequence,
            ),
            BoundaryContract(
                "left_output",
                RegionEndpoint("left", "output"),
                left.output_interface("output").port.beat_sequence,
            ),
            BoundaryContract(
                "right_output",
                RegionEndpoint("right", "output"),
                right.output_interface("output").port.beat_sequence,
            ),
        ),
    )


@dataclass(frozen=True)
class DesignInputs:
    extent: Ref[int]


@dataclass(frozen=True)
class KernelInputs:
    nodes: tuple[DesignNode, ...]
    lanes: Ref[int] | None = None
    network: Ref[DataflowNetwork] | None = None
    edge_role: str = "link"


class DirectKernel(HardwareKernel):
    id = "direct"

    @classmethod
    def define_design(cls, design: HardwareDesign[KernelInputs]) -> None:
        facts = design.inputs
        for node in facts.nodes:
            design.covers_region(
                node.role,
                region=node.region,
                computation=node.computation,
                implements=COPY,
            )
        if facts.lanes is not None:
            design.parameter("LANES", cast("Ref[object]", facts.lanes))
        if facts.network is not None:
            design.absorbs_edge(
                facts.edge_role,
                network=facts.network,
                source_role=facts.nodes[0].role,
                sink_role=facts.nodes[1].role,
            )

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("direct", "synthetic.direct"),)


class AlternativeKernel(HardwareKernel):
    id = "alternative"

    @classmethod
    def define_design(cls, design: HardwareDesign[KernelInputs]) -> None:
        node = design.inputs.nodes[0]
        design.covers_region(
            node.role,
            region=node.region,
            computation=node.computation,
            implements=COPY,
        )
        design.choice("pipeline", bool, domain=finite((False, True)))

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("alternative", "synthetic.alternative"),)


class SupplierKernel(HardwareKernel):
    id = "supplier"

    @classmethod
    def define_design(cls, design: HardwareDesign[KernelInputs]) -> None:
        node = design.inputs.nodes[0]
        design.covers_region(
            node.role,
            region=node.region,
            computation=node.computation,
            implements=SUPPLY,
        )

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("supplier", "synthetic.supplier"),)


class SingletonDesign(DataflowDesign):
    id = "singleton"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        lanes = design.choice("lanes", int, domain=divisors_of(design.inputs.extent))
        compute = design.region(
            "compute",
            node_id="compute",
            dependencies={"extent": design.inputs.extent, "lanes": lanes},
            evaluate=lambda extent, lanes: _region(extent // lanes, "compute"),
            computation=COPY,
        )
        design.singleton_network(compute)
        design.kernels(
            "compute",
            covers=(compute,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((compute,), lanes),
        )


class AlternativeDesign(DataflowDesign):
    id = "alternatives"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        compute = design.region(
            "compute",
            node_id="compute",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "compute"),
            computation=COPY,
        )
        design.singleton_network(compute)
        design.kernels(
            "compute",
            covers=(compute,),
            candidates=(DirectKernel, AlternativeKernel),
            inputs=KernelInputs((compute,)),
        )


class TwoPlacementDesign(DataflowDesign):
    id = "separate"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        producer = design.region(
            "producer",
            node_id="producer",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "producer"),
            computation=COPY,
        )
        consumer = design.region(
            "consumer",
            node_id="consumer",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "consumer"),
            computation=COPY,
        )
        design.network(
            dependencies={"producer": producer.region, "consumer": consumer.region},
            evaluate=_chain,
        )
        design.kernels(
            "producer",
            covers=(producer,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((producer,)),
        )
        design.kernels(
            "consumer",
            covers=(consumer,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((consumer,)),
        )


class SemanticOnlyDesign(DataflowDesign):
    id = "semantic_only"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        compute = design.region(
            "compute",
            node_id="compute",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "semantic"),
            computation=COPY,
        )
        design.singleton_network(compute)
        design.kernels("compute", covers=(compute,), candidates=())


@dataclass(frozen=True)
class SharedInputs:
    producer: Ref[DataflowRegion]
    consumer: Ref[DataflowRegion]
    producer_computation: Ref[ComputationContract]
    consumer_computation: Ref[ComputationContract]
    network: Ref[DataflowNetwork]


class SharedSeparateDesign(DataflowDesign):
    id = "shared_separate"

    @classmethod
    def define(cls, design: DataflowDesignScope[SharedInputs]) -> None:
        producer = design.node(
            "producer",
            node_id="producer",
            region=design.inputs.producer,
            computation=design.inputs.producer_computation,
        )
        consumer = design.node(
            "consumer",
            node_id="consumer",
            region=design.inputs.consumer,
            computation=design.inputs.consumer_computation,
        )
        design.use_network(design.inputs.network)
        design.kernels(
            "producer",
            covers=(producer,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((producer,)),
        )
        design.kernels(
            "consumer",
            covers=(consumer,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((consumer,)),
        )


class SharedFusedDesign(DataflowDesign):
    id = "shared_fused"

    @classmethod
    def define(cls, design: DataflowDesignScope[SharedInputs]) -> None:
        producer = design.node(
            "producer",
            node_id="producer",
            region=design.inputs.producer,
            computation=design.inputs.producer_computation,
        )
        consumer = design.node(
            "consumer",
            node_id="consumer",
            region=design.inputs.consumer,
            computation=design.inputs.consumer_computation,
        )
        network = design.use_network(design.inputs.network)
        edge = design.edge("link", edge_id="link", source=producer, sink=consumer, network=network)
        design.kernels(
            "fused",
            covers=(producer, consumer),
            absorbs=(edge,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((producer, consumer), network=network),
        )


class FanoutIncompleteDesign(DataflowDesign):
    id = "fanout_incomplete"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        producer = design.region(
            "producer",
            node_id="producer",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "producer"),
            computation=COPY,
        )
        left = design.region(
            "left",
            node_id="left",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "left"),
            computation=COPY,
        )
        right = design.region(
            "right",
            node_id="right",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, "right"),
            computation=COPY,
        )
        network = design.network(
            dependencies={
                "producer": producer.region,
                "left": left.region,
                "right": right.region,
            },
            evaluate=_fanout,
        )
        edge = design.edge("fanout", edge_id="fanout", source=producer, sink=left, network=network)
        design.kernels(
            "incomplete",
            covers=(producer, left),
            absorbs=(edge,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((producer, left), network=network, edge_role="fanout"),
        )
        design.kernels(
            "right", covers=(right,), candidates=(DirectKernel,), inputs=KernelInputs((right,))
        )


class SupplyDesign(DataflowDesign):
    node_id = "compute"

    @classmethod
    def define(cls, design: DataflowDesignScope[DesignInputs]) -> None:
        compute = design.region(
            "compute",
            node_id=cls.node_id,
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _region(extent, cls.id),
            computation=COPY,
        )
        design.singleton_network(compute)
        consumer = design.input_interface("parameter_consumer", compute, "input")
        design.map_input("parameter", boundary_id="input.input", consumer=consumer)
        design.kernels(
            "compute",
            covers=(compute,),
            candidates=(DirectKernel,),
            inputs=KernelInputs((compute,)),
        )


class SupplyDesignA(SupplyDesign):
    id = "supply_a"
    node_id = "compute_a"


class SupplyDesignB(SupplyDesign):
    id = "supply_b"
    node_id = "compute_b"


class SupplyDesignC(SupplyDesign):
    id = "supply_c"
    node_id = "compute_c"


def _declare_supplier(context: InputSupplyContext) -> SupplierAttachment:
    node = context.design.region(
        context.name("region"),
        node_id="parameter_supplier",
        dependencies={"consumer": context.mapping.consumer},
        evaluate=_supplier_region,
        computation=SUPPLY,
        applies_if=context.applies_if,
    )
    context.design.kernels(
        context.name("placement"),
        covers=(node,),
        candidates=(SupplierKernel,),
        inputs=KernelInputs((node,)),
        applies_if=context.applies_if,
    )
    return SupplierAttachment(node, "supplied", "parameter_supply")


def _problem() -> tuple[OpDesign, DesignInputs]:
    operation = OpDesign("synthetic.op", problem_namespace="synthetic")
    return operation, DesignInputs(operation.graph_fact("extent", int))


def _inventory(
    *designs: type[DataflowDesign], supply: bool = False
) -> tuple[DataflowDesignInventory, Engine, DesignPoint, DesignInputs]:
    operation, inputs = _problem()
    policies = (
        (
            declare_input_supply(
                "parameter_supply",
                "1",
                namespace="synthetic.input.parameter",
                source_operand="parameter",
                alternatives=(InputSupplyAlternative("buffer", _declare_supplier),),
            )
            if supply
            else None
        ),
    )
    inventory = declare_dataflow_design_inventory(
        "synthetic",
        tuple(DataflowDesignEntry(design, inputs) for design in designs),
        input_supplies=tuple(item for item in policies if item is not None),
        shared_specs=(operation.spec(),),
    )
    engine = Engine()
    space = engine.validate(inventory.specification)
    point = engine.start(space, {inputs.extent.path: 4})
    return inventory, engine, point, inputs


def _commit(
    engine: Engine, point: DesignPoint, assignments: dict[QualifiedPath, object]
) -> DesignPoint:
    return engine.commit_assignments(point, assignments).point


def _path(specification: DesignSpaceSpec, suffix: str) -> QualifiedPath:
    matches = tuple(
        item.path for item in specification.decisions if str(item.path).endswith(suffix)
    )
    assert len(matches) == 1
    return matches[0]


def test_one_design_one_region_one_kernel_and_no_gratuitous_kernel_choice() -> None:
    inventory, engine, point, _inputs = _inventory(SingletonDesign)
    lanes = _path(inventory.specification, ".lanes")
    point = _commit(engine, point, {lanes: 2})

    realized = inventory.realize(engine, point)
    assert isinstance(realized, Decided)
    assert tuple(node.id for node in realized.value.network.nodes) == ("compute",)
    assert tuple(binding.kernel_id for binding in realized.value.bindings) == ("direct",)
    assert not any(
        "hardware_kernel" in str(item.path) for item in inventory.specification.decisions
    )


def test_a_selected_design_may_keep_region_affecting_choices_unresolved() -> None:
    inventory, engine, point, _inputs = _inventory(SingletonDesign, SemanticOnlyDesign)
    assert inventory.design_path is not None
    point = _commit(engine, point, {inventory.design_path: "singleton"})

    selected = inventory.selected(point)
    assert isinstance(selected, Decided)
    assert selected.value.id == "singleton"
    assert isinstance(engine.query_property(point, selected.value.network.path), Unresolved)


def test_singleton_lift_exposes_every_interface_with_stable_boundaries() -> None:
    inventory, engine, point, _inputs = _inventory(SingletonDesign)
    point = _commit(engine, point, {_path(inventory.specification, ".lanes"): 1})
    network = engine.query_property(point, inventory.declarations[0].network.path)
    assert isinstance(network, Decided)
    value = cast(DataflowNetwork, network.value)
    assert tuple(item.id for item in value.boundaries) == (
        "input.input",
        "output.output",
    )


def test_one_design_may_have_two_independent_placements() -> None:
    inventory, engine, point, _inputs = _inventory(TwoPlacementDesign)
    realized = inventory.realize(engine, point)
    assert isinstance(realized, Decided)
    assert {binding.node_ids for binding in realized.value.bindings} == {
        ("producer",),
        ("consumer",),
    }
    assert realized.value.unabsorbed_edges == ("link",)


def test_equal_coverage_kernel_alternatives_are_a_separate_coordinate() -> None:
    inventory, engine, point, _inputs = _inventory(AlternativeDesign, SemanticOnlyDesign)
    assert inventory.design_path is not None
    kernel = _path(inventory.specification, ".hardware_kernel")
    pipeline = _path(inventory.specification, ".alternative.pipeline")
    point = _commit(
        engine,
        point,
        {
            inventory.design_path: "alternatives",
            kernel: "alternative",
            pipeline: True,
        },
    )
    realized = inventory.realize(engine, point)

    assert isinstance(realized, Decided)
    assert realized.value.bindings[0].kernel_id == "alternative"
    assert len({inventory.design_path, kernel, pipeline}) == 3


def _shared() -> tuple[OpDesign, SharedInputs]:
    operation = OpDesign("shared.op", problem_namespace="shared")
    extent = operation.graph_fact("extent", int)
    producer = operation.derived(
        "producer",
        DataflowRegion,
        dependencies={"extent": extent},
        evaluate=lambda extent: _region(extent, "producer"),
    )
    consumer = operation.derived(
        "consumer",
        DataflowRegion,
        dependencies={"extent": extent},
        evaluate=lambda extent: _region(extent, "consumer"),
    )
    producer_computation = operation.derived(
        "producer_computation", ComputationContract, dependencies={}, evaluate=lambda: COPY
    )
    consumer_computation = operation.derived(
        "consumer_computation", ComputationContract, dependencies={}, evaluate=lambda: COPY
    )
    network = operation.derived(
        "network",
        DataflowNetwork,
        dependencies={"producer": producer, "consumer": consumer},
        evaluate=_chain,
    )
    return operation, SharedInputs(
        producer, consumer, producer_computation, consumer_computation, network
    )


def test_two_designs_share_one_network_declaration_but_partition_it_differently() -> None:
    operation, inputs = _shared()
    inventory = declare_dataflow_design_inventory(
        "shared",
        (
            DataflowDesignEntry(SharedSeparateDesign, inputs),
            DataflowDesignEntry(SharedFusedDesign, inputs),
        ),
        shared_specs=(operation.spec(),),
    )
    separate, fused = inventory.declarations
    assert separate.network.path == fused.network.path == inputs.network.path
    assert len(separate.placements) == 2
    assert len(fused.placements) == 1
    engine = Engine()
    point = engine.start(
        engine.validate(inventory.specification),
        {QualifiedPath("problem.shared.extent"): 4},
    )
    assert inventory.design_path is not None
    separate_point = _commit(engine, point, {inventory.design_path: "shared_separate"})
    fused_point = _commit(engine, point, {inventory.design_path: "shared_fused"})
    separate_realization = inventory.realize(engine, separate_point)
    fused_realization = inventory.realize(engine, fused_point)
    assert isinstance(separate_realization, Decided)
    assert isinstance(fused_realization, Decided)
    assert len(separate_realization.value.bindings) == 2
    assert len(fused_realization.value.bindings) == 1


def test_one_kernel_may_cover_two_regions_and_absorb_their_edge() -> None:
    operation, inputs = _shared()
    inventory = declare_dataflow_design_inventory(
        "shared",
        (DataflowDesignEntry(SharedFusedDesign, inputs),),
        shared_specs=(operation.spec(),),
    )
    engine = Engine()
    point = engine.start(
        engine.validate(inventory.specification), {QualifiedPath("problem.shared.extent"): 4}
    )
    realized = inventory.realize(engine, point)
    assert isinstance(realized, Decided)
    assert realized.value.bindings[0].node_ids == ("consumer", "producer")
    assert realized.value.bindings[0].edge_ids == ("link",)
    assert realized.value.unabsorbed_edges == ()


def test_absorbing_a_fanout_requires_covering_every_sink() -> None:
    inventory, engine, point, _inputs = _inventory(FanoutIncompleteDesign)
    realized = inventory.realize(engine, point)
    assert isinstance(realized, Unresolved)
    assert "design-absorbed-edge-incomplete-fanout" in {item.code for item in realized.findings}


def test_a_semantic_only_design_is_model_valid_but_not_physically_realizable() -> None:
    inventory, engine, point, _inputs = _inventory(SemanticOnlyDesign)
    network = engine.query_property(point, inventory.declarations[0].network.path)
    realized = inventory.realize(engine, point)
    assert isinstance(network, Decided)
    assert isinstance(realized, Unresolved)
    assert "design-placement-has-no-kernel" in {item.code for item in realized.findings}


def test_repeated_design_placement_has_no_path_collisions() -> None:
    operation, inputs = _problem()
    left = declare_dataflow_design(SingletonDesign, "left.design", inputs)[0]
    right = declare_dataflow_design(SingletonDesign, "right.design", inputs)[0]
    specification = assemble_specs((operation.spec(), left.spec, right.spec))
    paths = tuple(
        item.path
        for group in (specification.decisions, specification.properties, specification.constraints)
        for item in group
    )
    assert len(paths) == len(set(paths))


def test_selected_design_may_leave_common_input_supply_unresolved() -> None:
    inventory, engine, point, _inputs = _inventory(SupplyDesignA, SupplyDesignB, supply=True)
    assert inventory.design_path is not None
    point = _commit(engine, point, {inventory.design_path: "supply_a"})
    selected = inventory.declaration("supply_a")
    assert isinstance(engine.query_property(point, selected.network.path), Unresolved)


def test_external_and_supplied_choices_resolve_different_flat_networks() -> None:
    inventory, engine, point, _inputs = _inventory(SupplyDesignA, supply=True)
    supply = inventory.input_supplies[0]
    external = _commit(engine, point, {supply.choice.path: "external"})
    supplied = _commit(engine, point, {supply.choice.path: "buffer"})
    external_network = engine.query_property(external, inventory.declarations[0].network.path)
    supplied_network = engine.query_property(supplied, inventory.declarations[0].network.path)

    assert isinstance(external_network, Decided) and isinstance(supplied_network, Decided)
    external_value = cast(DataflowNetwork, external_network.value)
    supplied_value = cast(DataflowNetwork, supplied_network.value)
    assert {node.id for node in external_value.nodes} == {"compute_a"}
    assert {node.id for node in supplied_value.nodes} == {
        "compute_a",
        "parameter_supplier",
    }
    assert {item.id for item in external_value.boundaries} == {
        "input.input",
        "output.output",
    }
    assert {item.id for item in supplied_value.boundaries} == {"output.output"}


def test_inactive_supplier_region_and_placement_are_absent() -> None:
    inventory, engine, point, _inputs = _inventory(SupplyDesignA, supply=True)
    supply = inventory.input_supplies[0]
    point = _commit(engine, point, {supply.choice.path: "external"})
    declaration = inventory.declarations[0]
    supplier = next(node for node in declaration.nodes if node.node_id == "parameter_supplier")
    placement = next(item for item in declaration.placements if "input.parameter" in item.name)

    assert isinstance(engine.query_property(point, supplier.region.path), Absent)
    assert isinstance(engine.query_property(point, placement.selected_kernel.path), Absent)


def test_active_supplier_placement_participates_in_exact_realization() -> None:
    inventory, engine, point, _inputs = _inventory(SupplyDesignA, supply=True)
    supply = inventory.input_supplies[0]
    point = _commit(engine, point, {supply.choice.path: "buffer"})
    realized = inventory.realize(engine, point)

    assert isinstance(realized, Decided)
    assert {binding.kernel_id for binding in realized.value.bindings} == {"direct", "supplier"}
    assert {node for binding in realized.value.bindings for node in binding.node_ids} == {
        "compute_a",
        "parameter_supplier",
    }
    assert realized.value.unabsorbed_edges == ("parameter_supply",)


def test_one_operation_supply_declaration_serves_two_designs() -> None:
    inventory, _engine, _point, _inputs = _inventory(SupplyDesignA, SupplyDesignB, supply=True)
    supply = inventory.input_supplies[0]
    assert supply.modes == ("external", "buffer")
    assert sum(item.path == supply.choice.path for item in inventory.specification.decisions) == 1
    assert all(
        tuple(mapping.source_operand for mapping in declaration.input_mappings) == ("parameter",)
        for declaration in inventory.declarations
    )


def test_a_new_design_inherits_the_closed_supply_policy_without_copying_alternatives() -> None:
    inventory, _engine, _point, _inputs = _inventory(
        SupplyDesignA, SupplyDesignB, SupplyDesignC, supply=True
    )
    supply = inventory.input_supplies[0]
    assert len(supply.alternatives) == 1
    assert len(inventory.declarations) == 3
    assert all(
        any("input.parameter.buffer" in placement.name for placement in declaration.placements)
        for declaration in inventory.declarations
    )


def test_generic_design_and_supply_authoring_has_no_mvau_dependency() -> None:
    source = Path(__file__).parents[2] / "src" / "finn" / "dataflow" / "authoring" / "design.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("mvau" in module for module in imported)


def test_public_design_authoring_surface_excludes_compiled_metadata() -> None:
    assert {
        "DataflowDesign",
        "DataflowDesignScope",
        "InputSupplyAlternative",
        "declare_dataflow_design_inventory",
    } <= set(design_authoring.__all__)
    assert {
        "DataflowDesignDeclaration",
        "HardwareKernelDeclaration",
        "KernelPlacement",
        "PlacementSelection",
    }.isdisjoint(design_authoring.__all__)


def test_existing_mvau_import_does_not_load_the_new_design_layer() -> None:
    subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import finn.dataflow.ops.mvau_op; "
            "assert 'finn.dataflow.authoring.design' not in sys.modules",
        ),
        check=True,
    )


def test_region_and_network_models_still_do_not_import_the_engine() -> None:
    root = Path(__file__).parents[2] / "src" / "finn" / "dataflow"
    for filename in ("region.py", "network.py"):
        tree = ast.parse((root / filename).read_text(), filename=filename)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any("_engine" in module or module.endswith(".design") for module in imported)
