# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The executable evidence for the final S1-C proposal.  Asserts, then prints."""

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
    MIGRATION,
    MULTI_PORT_TRIGGER,
    MULTI_PORT_WIDENING,
    LocalStateInput,
    MultiPortLimit,
    NullableInput,
    RegionLocalState,
    local_state_issue_codes,
    refuse_multi_port,
)
from cases import (  # noqa: E402
    ACTIVATION,
    COMPUTE,
    MEMORY,
    REPLAY,
    WEIGHT,
    Folding,
    colliding_network,
    decoupled_network,
    embedded_network,
    external_network,
    lift,
    partial_internal_network,
    partial_service_region,
    plural_target_network,
)
from dataflow_model import (  # noqa: E402
    InputInterface,
    ProtoNetwork,
    ProtoNode,
    RegionInputRef,
    RegionOutputRef,
    UnportedInput,
    derive_input_mappings,
    derive_output_mappings,
    exposing_boundaries,
    exposing_ports,
    externally_supplied_positions,
    internally_supplied_positions,
    network_operand_issues,
    presented_positions,
    required_positions,
    unsupplied_positions,
    validate_region,
)

from finn.dataflow.network import DataflowNetwork, NetworkNode  # noqa: E402
from finn.dataflow.network_validation import validate_network  # noqa: E402
from finn.dataflow.ops.mvau.regions import construct_dot_product_region  # noqa: E402
from finn.dataflow.region import DataflowRegion, Operand  # noqa: E402
from finn.dataflow.region import InputInterface as ProductionInterface  # noqa: E402
from finn.dataflow.region import ScheduledInputRequirements  # noqa: E402
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
    """The canonical value, rebuilt by dropping every unported input."""

    nodes = []
    for node in proto.nodes:
        interfaces = tuple(
            ProductionInterface(item.port, item.requirements)
            for item in node.region.inputs
            if isinstance(item, InputInterface)
        )
        nodes.append(
            NetworkNode(
                node.id, DataflowRegion(node.region.schedule, interfaces, node.region.outputs)
            )
        )
    return DataflowNetwork(tuple(nodes), proto.edges, proto.boundaries)


def sites(network: ProtoNetwork, ref: RegionInputRef) -> str:
    internal = internally_supplied_positions(network, ref)
    external = externally_supplied_positions(network, ref)
    owed = unsupplied_positions(network, ref)
    boundaries = tuple(item.id for item in exposing_boundaries(network, ref))
    return (
        f"internal {len(internal):>5}  external {len(external):>5}  owed {len(owed):>5}"
        f"  boundaries {boundaries or '()'}"
    )


# -- 1. nullable port versus the sum type ------------------------------------

rule("1  nullable port vs InputInterface | UnportedInput")
print(f"{'':<42}{'sum type':>10}{'nullable':>10}")
for label, counts in MIGRATION.items():
    print(f"{label:<42}{counts['sum type']:>10}{counts['nullable']:>10}")
print(
    "\nMeasured on 546538087, not estimated.  The sum type keeps InputInterface's\n"
    "name, fields and constructor, so 20 constructions and 33 input_interface()\n"
    "lookups are untouched; the nullable form rewrites all 53."
)

streamed = construct_dot_product_region(2, 8, 8, ACTIVATION, WEIGHT, ACTIVATION, 2, 2)
weight = streamed.input_interface("weight")
disagreeing = NullableInput(Operand("Z", WEIGHT, (8, 8)), weight.requirements, weight.port)
assert disagreeing.operand_disagrees
print(
    "\nAnd the state the sum type removes is constructible under the nullable form:\n"
    f"  NullableInput(operand=Z, port.operand=W)  ->  disagrees: "
    f"{disagreeing.operand_disagrees}\n"
    "so the nullable form needs input.port_operand_mismatch and the sum type does not."
)

# -- 2. source mappings, three supply modes ----------------------------------

rule("2  source operand -> dataflow requirement, three supply modes")
expected = {
    "external": (RegionInputRef(COMPUTE, "W"),),
    "embedded": (RegionInputRef(COMPUTE, "W"),),
    "decoupled": (RegionInputRef(COMPUTE, "W"), RegionInputRef(MEMORY, "W")),
}
for name, network in CASES.items():
    activation = derive_input_mappings(network, "X")
    weight_refs = derive_input_mappings(network, "W")
    output = derive_output_mappings(network, "Y")
    assert weight_refs == expected[name], (name, weight_refs)
    assert output == (RegionOutputRef(COMPUTE, "Y"),), (name, output)
    print(f"{name}")
    for ref in activation + weight_refs:
        print(f"   {ref.operand_id} @ {ref.node_id:<8} {sites(network, ref)}")

print(
    "\nEvery Region requiring the operand is a target: that is correspondence, and it\n"
    "is the only meaning derive_input_mappings has.  Which of them the Network itself\n"
    "supplies is asked separately, per position, and never folded into the mapping."
)
decoupled = CASES["decoupled"]
assert unsupplied_positions(decoupled, RegionInputRef(COMPUTE, "W")) == frozenset()
assert len(unsupplied_positions(decoupled, RegionInputRef(MEMORY, "W"))) == 64 * 64
assert exposing_boundaries(CASES["external"], RegionInputRef(COMPUTE, "W"))[0].id == "weight"
assert exposing_ports(CASES["embedded"], RegionInputRef(COMPUTE, "W")) == ()
print(
    "decoupled: compute owes nothing (the edge supplies all of W); memory owes all of\n"
    "it.  Where the memory's owed positions come from is U6's question, not this one."
)

# -- 3. what production answers today ----------------------------------------

rule("3  the current production answer for the decoupled matrix")
from finn.dataflow.ops.mvau.op import _internal_destination  # noqa: E402

current = _internal_destination(as_network(CASES["decoupled"]), "weight")
print(f"_internal_destination(decoupled, 'weight') = {current}")
print(f"derive_input_mappings(decoupled, 'W')      = {derive_input_mappings(decoupled, 'W')}")
assert str(current) == "StreamDestination(node_id='compute', port_id='weight')"
print(
    "One answer, chosen by node scan order, naming the consumer's port.  It also\n"
    "matches on the *port* id -- the ports happen to be called 'weight' and the\n"
    "operand is called 'W'."
)

# -- 4. partial internal service ---------------------------------------------

rule("4  a port an edge feeds that still presents only half")
partial_internal = partial_internal_network()
for node in partial_internal.nodes:
    assert validate_region(node.region) == (), validate_region(node.region)
assert not validate_network(as_network(partial_internal))
refs = derive_input_mappings(partial_internal, "W")
assert refs == (RegionInputRef(COMPUTE, "W"), RegionInputRef(MEMORY, "W")), refs
for ref in refs:
    print(f"   W @ {ref.node_id:<8} {sites(partial_internal, ref)}")
compute_ref = RegionInputRef(COMPUTE, "W")
assert len(internally_supplied_positions(partial_internal, compute_ref)) == 2
assert len(unsupplied_positions(partial_internal, compute_ref)) == 2
print(
    "\ncompute's w_hi IS an edge sink, so a boolean 'edge-fed' test would have dropped\n"
    "this requirement from the mapping entirely -- and with it the two positions the\n"
    "Network does not supply.  Position-granular, the residue is visible and named."
)

# -- 5. partial service within one Region ------------------------------------

rule("5  positions versus occurrences")
partial = partial_service_region()
assert validate_region(partial) == (), validate_region(partial)
for operand_id in ("X", "W"):
    item = partial.input(operand_id)
    print(
        f"{operand_id}: required occurrences {item.requirements.occurrence_count:>3}   "
        f"presented fields {item.port.beat_sequence.delivered_field_count:>3}   "
        f"positions presented {len(presented_positions(item))}/"
        f"{len(required_positions(item))}"
    )
assert required_positions(partial.input("X")) == presented_positions(partial.input("X"))
assert partial.input("X").requirements.occurrence_count == 24
assert len(required_positions(partial.input("W")) - presented_positions(partial.input("W"))) == 4
print(
    "\nX is presented once and required three times: the tensor has entered, and the\n"
    "re-reads are the binding's business.  REGION.md 3.7 refuses any required-versus-\n"
    "presented equality for inputs, so the sets above are position-granular.  W is a\n"
    "different fact: half its positions never arrive."
)

old_form = RegionLocalState(
    partial.schedule,
    tuple(
        (item.port, item.requirements)
        for item in partial.inputs
        if isinstance(item, InputInterface)
    ),
    partial.outputs,
    (LocalStateInput(partial.input("W").operand),),
)
assert local_state_issue_codes(old_form) == ("local_state.operand_also_streamed",)
print(
    "\nThe first pass's form needs W to be streamed *and* local state, which its own\n"
    f"rule rejects: {local_state_issue_codes(old_form)}."
)

# -- 6. plural targets and operand-id collisions -----------------------------

rule("6  plural targets, and operand ids colliding across nodes")
plural = plural_target_network()
for node in plural.nodes:
    assert validate_region(node.region) == (), validate_region(node.region)
assert network_operand_issues(plural) == ()
refs = derive_input_mappings(plural, "W")
assert refs == (RegionInputRef("compute_a", "W"), RegionInputRef("compute_b", "W")), refs
print(f"two Regions, one tensor: {refs}")
print("zero targets is an error, one is ordinary, two are two.")

colliding = colliding_network()
codes = tuple(issue.code for issue in network_operand_issues(colliding))
assert codes == ("network.operand_identity_conflict",), codes
print(f"\ntwo unrelated tensors both called W, disagreeing on shape -> {codes}")
print(
    "Two unrelated tensors agreeing on type and shape are invisible to any structural\n"
    "rule.  The answer to those is declaration-side qualification, not node ordering."
)

# -- 7. validation ------------------------------------------------------------

rule("7  validation")
for name, network in CASES.items():
    canonical = validate_network(as_network(network))
    assert not canonical, (name, canonical.issues)
    for node in network.nodes:
        assert validate_region(node.region) == (), (name, node.id)
    lost = tuple(
        f"{node.id}.{item.operand.id}"
        for node in network.nodes
        for item in node.region.inputs
        if isinstance(item, UnportedInput)
    )
    print(f"{name:<10} 0 issues   dropped by today's value: {lost or '()'}")

broken_production = DataflowRegion(
    streamed.schedule,
    tuple(
        ProductionInterface(
            interface.port,
            ScheduledInputRequirements({((99, 0, 0), (0, 0)): 1})
            if interface.port.id == "weight"
            else interface.requirements,
        )
        for interface in streamed.inputs
    ),
    streamed.outputs,
)
lifted = lift(streamed)
broken_recommended = replace(
    lifted,
    inputs=tuple(
        InputInterface(item.port, ScheduledInputRequirements({((99, 0, 0), (0, 0)): 1}))
        if item.operand.id == "W"
        else item
        for item in lifted.inputs
    ),
)
production_codes = {issue.code for issue in production_validate(broken_production)}
recommended_codes = {issue.code for issue in validate_region(broken_recommended)}
assert production_codes == recommended_codes == {"requirement.iteration_out_of_domain"}
print(f"same Region broken the same way: production == recommended {sorted(production_codes)}")

rule("7b operand validation now reaches an operand with no port")
embedded_compute = CASES["embedded"].node(COMPUTE).region
unported_bad = replace(
    embedded_compute,
    inputs=tuple(
        UnportedInput(Operand("W", WEIGHT, (0, 8)), ScheduledInputRequirements())
        if item.operand.id == "W"
        else item
        for item in embedded_compute.inputs
    ),
)
codes = tuple(issue.code for issue in validate_region(unported_bad))
assert "operand.extent_not_positive" in codes, codes
today = DataflowRegion(
    embedded_compute.schedule,
    tuple(
        ProductionInterface(item.port, item.requirements)
        for item in embedded_compute.inputs
        if isinstance(item, InputInterface)
    ),
    embedded_compute.outputs,
)
assert production_validate(today).issues == ()
print(f"W declared with extent 0 and no port -> {codes}")
print("the same operand in today's embedded Region -> () : it is not in the value")

rule("7c the one rule the new case adds")
held = embedded_compute.input("W")
doubled = replace(embedded_compute, inputs=(*embedded_compute.inputs, held))
codes = tuple(issue.code for issue in validate_region(doubled))
assert codes == ("input.operand_duplicate",), codes
print(f"two inputs for one operand -> {codes}")
print("there is no operand/port agreement rule: the sum type makes it unrepresentable.")

# -- 8. the one-port limitation ----------------------------------------------

rule("8  the one structural refusal, and what widening really costs")
try:
    refuse_multi_port("W", ("w_lo", "w_hi"))
    raise AssertionError("expected the refusal")
except MultiPortLimit as error:
    print(f"MultiPortLimit: {error}")
print(f"\ntrigger: {MULTI_PORT_TRIGGER}")
print(MULTI_PORT_WIDENING)

# -- 9. what the model states, and what it leaves open -----------------------

rule("9  the binding boundary")
for label, network, ref in (
    ("embedded compute", CASES["embedded"], RegionInputRef(COMPUTE, "W")),
    ("decoupled memory", CASES["decoupled"], RegionInputRef(MEMORY, "W")),
    ("partial internal ", partial_internal, RegionInputRef(COMPUTE, "W")),
):
    item = network.node(ref.node_id).region.input(ref.operand_id)
    print(
        f"{label}  occurrences {item.requirements.occurrence_count:>6}  "
        f"owed positions {len(unsupplied_positions(network, ref)):>5}  "
        f"{item.operand.element_type.name} {item.operand.shape}"
    )
print(
    "\nstated  which positions, at which schedule points, how often; the operand's\n"
    "        element type and shape; which positions a port presents and in what beat\n"
    "        order; whether an edge feeds that port or a boundary exposes it\n"
    "open    which service covers the owed positions -- embedded ROM, shared off-chip,\n"
    "        constant generation, replay register, parameter memory; storage datatype,\n"
    "        packing, alignment, addressing, banking; access conflict and bandwidth;\n"
    "        any generated external interface; the REGION.md 5.2 witness itself"
)

# -- 10. schemas D and E ------------------------------------------------------

rule("10 companion metadata (D) and an authored disposition graph (E)")
annotated = AnnotatedNetwork(
    ProtoNetwork(
        tuple(
            ProtoNode(
                node.id,
                replace(
                    node.region,
                    inputs=tuple(
                        item for item in node.region.inputs if isinstance(item, InputInterface)
                    ),
                ),
            )
            for node in decoupled.nodes
        ),
        decoupled.edges,
        decoupled.boundaries,
    ),
    {
        node.id: RegionResidency(
            tuple(item.operand for item in node.region.inputs),
            tuple(item.requirements for item in node.region.inputs),
        )
        for node in decoupled.nodes
    },
)
assert derive_mappings_d(annotated, "W") == ("RegionInputRef(memory,W)",)
print(f"D reaches an answer: {derive_mappings_d(annotated, 'W')}")
print(f"   seams that must carry the companion end to end: {len(D_THREADING_SITES)}")
print("   and the companion now carries the requirements, so it is half the Region.")

truthful = DispositionGraph(
    (SemanticRequirement("W", WEIGHT, (64, 64)),),
    (
        RequirementDisposition("W", DispositionKind.INTERNAL_STREAM, COMPUTE),
        RequirementDisposition("W", DispositionKind.LOCAL_STATE, MEMORY),
    ),
)
lying = DispositionGraph(
    truthful.requirements,
    (
        RequirementDisposition("W", DispositionKind.INTERNAL_STREAM, COMPUTE),
        RequirementDisposition("W", DispositionKind.LOCAL_STATE, REPLAY),
    ),
)
assert disposition_agrees_with_network(truthful, decoupled, "W")
assert not disposition_agrees_with_network(lying, decoupled, "W")
print(f"\nE looks it up in a table: {derive_mappings_e(truthful, 'W')}")
print("   and the table now has to restate the whole correspondence set, because")
print("   correspondence is plural.  A row naming the wrong node is well-formed and")
print("   passes E's own rules; catching it needs the derivation underneath.")

# -- 11. accounting -----------------------------------------------------------

rule("11 public type accounting")
print("added    UnportedInput, RegionInputRef, RegionOutputRef                3")
print("         RegionInput (a union alias, not a class)                      -")
print("removed  BoundaryDestination, StreamDestination,                       3")
print("         RegionStateDestination, OperandDestination alias              1")
print("                                                              net     -1")
print("kept     InputInterface, DataflowRegion's three fields, their names")
print("         and their order")
print("stored derived values: none.  exposing_ports returns RegionEndpoint,")
print("exposing_boundaries returns BoundaryContract, the position queries return")
print("plain frozensets.")

print("\nreplay X and compute X both correspond to the activation tensor:")
for reference in derive_input_mappings(decoupled, "X"):
    print(f"  {reference.node_id:<8} {sites(decoupled, reference)}")
print("  correspondence is plural; provenance tells them apart.")

print("\nall assertions passed")
