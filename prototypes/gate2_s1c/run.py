# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The executable evidence for the revised S1-C comparison.  Asserts, then prints."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alternatives import (  # noqa: E402
    D_THREADING_SITES,
    AnnotatedNetwork,
    DispositionGraph,
    DispositionKind,
    RegionResidency,
    RequirementDisposition,
    SemanticRequirement,
    derive_mappings_d,
    derive_mappings_e,
    disposition_agrees_with_network,
)
from candidates import (  # noqa: E402
    CandidateBLimit,
    LocalStateInput,
    RegionLocalState,
    local_state_issue_codes,
    region_b_from_requirements,
)
from cases import (  # noqa: E402
    ACTIVATION,
    COMPUTE,
    MEMORY,
    REPLAY,
    WEIGHT,
    Folding,
    decoupled_network,
    embedded_network,
    external_network,
    lift,
    partial_service_region,
    plural_mapping_network,
    split_supply_region,
)
from dataflow_model import (  # noqa: E402
    InputRequirement,
    ProtoNetwork,
    ProtoNode,
    RegionInputRef,
    RegionOutputRef,
    derive_input_mappings,
    derive_output_mappings,
    exposing_boundaries,
    exposing_ports,
    validate_region,
)

from finn.dataflow.network import DataflowNetwork, NetworkNode  # noqa: E402
from finn.dataflow.network_validation import validate_network  # noqa: E402
from finn.dataflow.ops.mvau.regions import construct_dot_product_region  # noqa: E402
from finn.dataflow.region import (  # noqa: E402
    DataflowRegion,
    InputInterface,
    Operand,
    ScheduledInputRequirements,
)
from finn.dataflow.region_validation import validate_region as production_validate  # noqa: E402

FOLDING = Folding()
CASES = {
    "external": external_network(FOLDING),
    "embedded": embedded_network(FOLDING),
    "decoupled": decoupled_network(FOLDING),
}


def rule(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 72 - len(title)))


def as_network(proto: ProtoNetwork) -> DataflowNetwork:
    """The canonical value, rebuilt by re-welding requirements onto ports.

    Only Regions in which every requirement has exactly one port survive this
    round trip.  The embedded and decoupled cases do not, which is the point --
    they are the Regions the current value cannot express.
    """

    nodes = []
    for node in proto.nodes:
        interfaces = tuple(
            InputInterface(port, node.region.input_requirement(port.operand.id).requirements)
            for port in node.region.input_ports
        )
        nodes.append(
            NetworkNode(
                node.id, DataflowRegion(node.region.schedule, interfaces, node.region.outputs)
            )
        )
    return DataflowNetwork(tuple(nodes), proto.edges, proto.boundaries)


# -- 1. source mappings, three supply modes ----------------------------------

rule("1  source operand -> dataflow requirement, three supply modes")
expected = {
    "external": (RegionInputRef(COMPUTE, "W"),),
    "embedded": (RegionInputRef(COMPUTE, "W"),),
    "decoupled": (RegionInputRef(MEMORY, "W"),),
}
for name, network in CASES.items():
    activation = derive_input_mappings(network, "X")
    weight = derive_input_mappings(network, "W")
    output = derive_output_mappings(network, "Y")
    assert activation == (RegionInputRef(REPLAY, "X"),), (name, activation)
    assert weight == expected[name], (name, weight)
    assert output == (RegionOutputRef(COMPUTE, "Y"),), (name, output)
    boundaries = exposing_boundaries(network, weight[0])
    ports = exposing_ports(network, weight[0])
    print(f"{name:<10} X -> {activation}")
    print(f"{'':<10} W -> {weight}")
    print(
        f"{'':<10}      exposed by ports {tuple(f'{e.node_id}.{e.port_id}' for e in ports)}"
        f"  boundaries {tuple(b.id for b in boundaries)}"
    )
    print(f"{'':<10} Y -> {output}")

assert exposing_boundaries(CASES["external"], RegionInputRef(COMPUTE, "W"))[0].id == "weight"
assert exposing_ports(CASES["embedded"], RegionInputRef(COMPUTE, "W")) == ()
print(
    "\nThe decoupled compute Region still requires W; its requirement is fed by the\n"
    "weight_supply edge, so the *source tensor* corresponds to the memory Region's\n"
    "requirement.  Nothing above says where any bytes are stored."
)

# -- 2. what production answers today ----------------------------------------

rule("2  the current production answer for the decoupled matrix")
from finn.dataflow.ops.mvau.op import _internal_destination  # noqa: E402

current = _internal_destination(as_network(CASES["decoupled"]), "weight")
print(f"_internal_destination(decoupled, 'weight') = {current}")
print(
    f"derive_input_mappings(decoupled, 'W')      = {derive_input_mappings(CASES['decoupled'], 'W')}"
)
assert str(current) == "StreamDestination(node_id='compute', port_id='weight')"
print(
    "The current answer names the downstream consumer's port.  It also matches on the\n"
    "*port* id -- the ports happen to be called 'weight' and the operand is called 'W'."
)

# -- 3. validation over the lifted production Regions ------------------------

rule("3  validation")
for name, network in CASES.items():
    # The round trip welds requirements back onto ports and *loses* every
    # unported requirement, which is exactly what today's value cannot hold.
    # The resulting old-shape Network is still canonically well formed, so
    # nothing here is a Network regression -- the loss is silent, which is the
    # complaint.
    canonical = validate_network(as_network(network))
    assert not canonical, (name, canonical.issues)
    for node in network.nodes:
        issues = validate_region(node.region)
        assert issues == (), (name, node.id, issues)
    lost = tuple(
        f"{node.id}.{item.operand.id}"
        for node in network.nodes
        for item in node.region.input_requirements
        if not node.region.ports_for(item.operand.id)
    )
    print(
        f"{name:<10} candidate-A validate_region: 0 issues   "
        f"lost in the round trip to today's value: {lost or '()'}"
    )

streamed = construct_dot_product_region(2, 8, 8, ACTIVATION, WEIGHT, ACTIVATION, 2, 2)
broken_production = DataflowRegion(
    streamed.schedule,
    tuple(
        InputInterface(
            interface.port,
            ScheduledInputRequirements({((99, 0, 0), (0, 0)): 1})
            if interface.port.id == "weight"
            else interface.requirements,
        )
        for interface in streamed.inputs
    ),
    streamed.outputs,
)
broken_a = replace(
    lift(streamed),
    input_requirements=tuple(
        InputRequirement(item.operand, ScheduledInputRequirements({((99, 0, 0), (0, 0)): 1}))
        if item.operand.id == "W"
        else item
        for item in lift(streamed).input_requirements
    ),
)
production_codes = {issue.code for issue in production_validate(broken_production)}
candidate_codes = {issue.code for issue in validate_region(broken_a)}
assert production_codes == candidate_codes == {"requirement.iteration_out_of_domain"}, (
    production_codes,
    candidate_codes,
)
print(
    f"same Region broken the same way: production {sorted(production_codes)} == "
    f"candidate A {sorted(candidate_codes)}"
)

# -- 3b. the rules that now reach an unported operand ------------------------

rule("3b operand validation now reaches an operand with no port")
bad_operand = Operand("W", WEIGHT, (0, 8))
embedded_compute = next(node for node in CASES["embedded"].nodes if node.id == COMPUTE).region
unported_bad = replace(
    embedded_compute,
    input_requirements=tuple(
        InputRequirement(bad_operand, ScheduledInputRequirements())
        if item.operand.id == "W"
        else item
        for item in embedded_compute.input_requirements
    ),
)
codes = tuple(issue.code for issue in validate_region(unported_bad))
assert "operand.extent_not_positive" in codes, codes
today = DataflowRegion(
    embedded_compute.schedule,
    tuple(
        InputInterface(port, embedded_compute.input_requirement(port.operand.id).requirements)
        for port in embedded_compute.input_ports
    ),
    embedded_compute.outputs,
)
assert production_validate(today).issues == ()
print(f"candidate A, W declared with extent 0 and no port -> {codes}")
print("the same operand in today's embedded Region -> () : it is not in the value at all")

# -- 3c. the new uniqueness and coverage rules -------------------------------

rule("3c the rules the split adds")
requirement = embedded_compute.input_requirement("W")
doubled = replace(
    embedded_compute, input_requirements=(*embedded_compute.input_requirements, requirement)
)
codes = tuple(issue.code for issue in validate_region(doubled))
assert codes == ("requirement.operand_duplicate",), codes
print(f"two requirements for one operand      -> {codes}")

orphan = replace(
    embedded_compute,
    input_requirements=tuple(
        item for item in embedded_compute.input_requirements if item.operand.id != "X"
    ),
)
codes = tuple(issue.code for issue in validate_region(orphan))
assert codes == ("input_port.requirement_missing",), codes
print(f"a port presenting an undeclared operand -> {codes}")

conflicting = replace(
    embedded_compute,
    input_requirements=tuple(
        InputRequirement(Operand("X", WEIGHT, (1, 1)), item.requirements)
        if item.operand.id == "X"
        else item
        for item in embedded_compute.input_requirements
    ),
)
codes = {issue.code for issue in validate_region(conflicting)}
assert "operand.identity_conflict" in codes, codes
print(f"requirement and port disagree on an operand -> {sorted(codes)}")

# -- 4. partial service ------------------------------------------------------

rule("4  a Region whose ports present part of what it requires")
partial = partial_service_region()
assert validate_region(partial) == (), validate_region(partial)
for operand_id in ("X", "W"):
    required = partial.input_requirement(operand_id)
    presented_positions = partial.presented_positions(operand_id)
    presented_fields = sum(
        port.beat_sequence.delivered_field_count for port in partial.ports_for(operand_id)
    )
    print(
        f"{operand_id}: required occurrences {required.occurrence_count:>3}   "
        f"presented fields {presented_fields:>3}   "
        f"positions presented {len(presented_positions)}/{required.operand.position_count}   "
        f"unpresented positions {len(partial.unpresented_positions(operand_id))}"
    )
assert partial.unpresented_positions("X") == frozenset()
assert partial.input_requirement("X").occurrence_count == 24
assert len(partial.unpresented_positions("W")) == 4

old_form = RegionLocalState(
    partial.schedule,
    tuple(
        (port, partial.input_requirement(port.operand.id).requirements)
        for port in partial.input_ports
    ),
    partial.outputs,
    (LocalStateInput(partial.input_requirement("W").operand),),
)
assert local_state_issue_codes(old_form) == ("local_state.operand_also_streamed",)
print(
    "\nExpressing W's half-presented supply in the first submission's form needs W to be\n"
    "streamed *and* local state, which its own rule rejects: "
    f"{local_state_issue_codes(old_form)}.\n"
    "REGION.md 3.7 permits it per position, so the rule rejected a canonical Region."
)

# -- 5. plural mapping -------------------------------------------------------

rule("5  one source tensor, two dataflow requirements")
plural = plural_mapping_network()
for node in plural.nodes:
    assert validate_region(node.region) == (), validate_region(node.region)
weight_refs = derive_input_mappings(plural, "W")
assert weight_refs == (RegionInputRef("compute_a", "W"), RegionInputRef("compute_b", "W")), (
    weight_refs
)
print(f"derive_input_mappings(plural, 'W') = {weight_refs}")
print("zero mappings is an error, one is ordinary, two are two.  An OpInput whose")
print("operation contract needs singularity can require it; the model does not.")

# -- 6. one operand, two ports -----------------------------------------------

rule("6  one operand presented by two ports")
split = split_supply_region()
assert validate_region(split) == (), validate_region(split)
print(f"candidate A: W requirement + ports {tuple(p.id for p in split.ports_for('W'))} -> 0 issues")
try:
    region_b_from_requirements(
        split.schedule,
        {item.operand.id: item.requirements for item in split.input_requirements},
        {item.operand.id: item.operand for item in split.input_requirements},
        split.input_ports,
        split.outputs,
    )
    raise AssertionError("expected candidate B to fail")
except CandidateBLimit as error:
    print(f"candidate B: CandidateBLimit: {error}")
region_b = region_b_from_requirements(
    partial.schedule,
    {item.operand.id: item.requirements for item in partial.input_requirements},
    {item.operand.id: item.operand for item in partial.input_requirements},
    partial.input_ports,
    partial.outputs,
)
assert len(region_b.inputs) == 2
print("candidate B does hold every one-port-per-operand case, including partial service.")

# -- 7. what is left for the physical binding --------------------------------

rule("7  what the dataflow model states, and what it leaves open")
for label, region, operand_id in (
    ("embedded compute", embedded_compute, "W"),
    ("decoupled memory", CASES["decoupled"].node(MEMORY).region, "W"),
    ("partial service ", partial, "W"),
):
    requirement = region.input_requirement(operand_id)
    print(
        f"{label}  occurrences {requirement.occurrence_count:>6}  "
        f"unpresented positions {len(region.unpresented_positions(operand_id)):>5}  "
        f"element {requirement.operand.element_type.name} shape {requirement.operand.shape}"
    )
print(
    "\nstated by the model : which positions, at which schedule points, how often,\n"
    "                      of which element type and shape, and which ports present\n"
    "                      any of them\n"
    "left to the binding  : which service covers the rest -- embedded ROM, shared\n"
    "                      off-chip, constant generation, replay register, parameter\n"
    "                      memory -- plus addressing, packing, alignment, banking,\n"
    "                      access conflict and any generated external interface"
)

# -- 8. schemas D and E on the same cases ------------------------------------

rule("8  companion metadata (D) and an authored disposition graph (E)")
memory_region_value = CASES["decoupled"].node(MEMORY).region
annotated = AnnotatedNetwork(
    ProtoNetwork(
        tuple(
            ProtoNode(node.id, replace(node.region, input_requirements=()))
            for node in CASES["decoupled"].nodes
        ),
        CASES["decoupled"].edges,
        CASES["decoupled"].boundaries,
    ),
    {
        node.id: RegionResidency(
            tuple(item.operand for item in node.region.input_requirements),
            tuple(item.requirements for item in node.region.input_requirements),
        )
        for node in CASES["decoupled"].nodes
    },
)
assert derive_mappings_d(annotated, "W") == ("RegionInputRef(memory,W)",)
print(f"D reaches the same answer: {derive_mappings_d(annotated, 'W')}")
print(f"   seams that must carry the companion end to end: {len(D_THREADING_SITES)}")
for site in D_THREADING_SITES:
    print(f"     - {site}")
print(
    "   and the companion now holds the requirements, so it is half the Region\n"
    "   rather than a small annotation beside it."
)

truthful = DispositionGraph(
    (SemanticRequirement("W", WEIGHT, (64, 64)),),
    (RequirementDisposition("W", DispositionKind.LOCAL_STATE, MEMORY),),
)
lying = DispositionGraph(
    truthful.requirements,
    (RequirementDisposition("W", DispositionKind.LOCAL_STATE, COMPUTE),),
)
assert disposition_agrees_with_network(truthful, CASES["decoupled"], "W")
assert not disposition_agrees_with_network(lying, CASES["decoupled"], "W")
print(f"\nE looks the answer up in a table: {derive_mappings_e(truthful, 'W')}")
print("   a disposition naming 'compute' is well-formed and passes E's own rules;")
print("   catching it needs candidate A's derivation, which E must then also carry.")

# -- 9. type and authority accounting ----------------------------------------

rule("9  public type accounting")
print("added    InputRequirement, RegionInputRef, RegionOutputRef            3")
print("removed  InputInterface (an input interface is now a Port)            1")
print("         BoundaryDestination, StreamDestination,                      3")
print("         RegionStateDestination, OperandDestination alias             1")
print("                                                              net    -2")
print("stored derived values: none.  exposing_ports returns RegionEndpoint and")
print("exposing_boundaries returns BoundaryContract -- both already canonical.")

print("\nall assertions passed")
