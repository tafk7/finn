# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The executable evidence for the S1-C comparison.  Asserts, then prints."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alternatives import (  # noqa: E402
    AnnotatedNetwork,
    B_THREADING_SITES,
    DispositionGraph,
    DispositionKind,
    RegionResidency,
    RequirementDisposition,
    SemanticRequirement,
    derive_placement_b,
    derive_placement_c,
    disposition_agrees_with_network,
)
from cases import (  # noqa: E402
    ACTIVATION,
    COMPUTE,
    MEMORY,
    REPLAY,
    Folding,
    decoupled_network,
    embedded_network,
    external_network,
    weight_local_state,
)
from schema_a import (  # noqa: E402
    InternalStream,
    External,
    PlacementError,
    ProtoNetwork,
    ProtoNode,
    LocalState,
    LocalStateInput,
    added_region_issues,
    derive_input_placement,
    derive_output_placement,
    selected_shape,
)

from finn.dataflow.network import DataflowNetwork, NetworkNode  # noqa: E402
from finn.dataflow.network_validation import validate_network  # noqa: E402
from finn.dataflow.region import ScheduledInputRequirements  # noqa: E402
from finn.dataflow.region_validation import validate_region  # noqa: E402

FOLDING = Folding()
CASES = {
    "external": external_network(FOLDING),
    "embedded": embedded_network(FOLDING),
    "decoupled": decoupled_network(FOLDING),
}


def rule(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 72 - len(title)))


def as_network(proto: ProtoNetwork) -> DataflowNetwork:
    """The canonical value the prototype is a superset of."""

    return DataflowNetwork(
        tuple(NetworkNode(node.id, node.region.streamed_projection) for node in proto.nodes),
        proto.edges,
        proto.boundaries,
    )


# -- 1. the three placements, derived ----------------------------------------

rule("1  derived placement, three supply modes")
expected = {
    "external": (External("weight", COMPUTE, "weight"), External("output", COMPUTE, "output")),
    "embedded": (LocalState(COMPUTE, "W"), External("output", COMPUTE, "output")),
    "decoupled": (LocalState(MEMORY, "W"), External("output", COMPUTE, "output")),
}
for name, network in CASES.items():
    activation = derive_input_placement(network, "X")
    weight = derive_input_placement(network, "W")
    output = derive_output_placement(network, "Y")
    assert activation == External("activation", REPLAY, "activation_in"), (name, activation)
    assert (weight, output) == expected[name], (name, weight, output)
    print(f"{name:<10} X -> {activation}")
    print(f"{'':<10} W -> {weight}   shape at port {selected_shape(network, weight)}")
    print(f"{'':<10} Y -> {output}")

# -- 2. what the production code answers today -------------------------------

rule("2  the current production answer for the decoupled matrix")
from finn.dataflow.ops.mvau.op import _internal_destination  # noqa: E402

current = _internal_destination(as_network(CASES["decoupled"]), "weight")
print(f"_internal_destination(decoupled, 'weight')  = {current}")
print(
    "schema A  derive_input_placement(.., 'W')   = "
    f"{derive_input_placement(CASES['decoupled'], 'W')}"
)
print(
    "The current answer names the *consumer* of the matrix.  A caller asking "
    "where the\ninitializer bytes have to be installed is told 'the compute "
    "node's weight port',\nwhich has no storage and is not where they go."
)
# It also matches on the *port* id, not the operand id: the ports happen to be
# called "weight" and the operand is called "W".
assert str(current) == "StreamDestination(node_id='compute', port_id='weight')"

rule("2b the embedded fallback, with the guess removed")
without_local_state = ProtoNetwork(
    tuple(
        ProtoNode(node.id, replace(node.region, local_state=())) for node in CASES["embedded"].nodes
    ),
    CASES["embedded"].edges,
    CASES["embedded"].boundaries,
)
try:
    derive_input_placement(without_local_state, "W")
    raise AssertionError("expected no entry site")
except PlacementError as error:
    print(f"PlacementError: {error}")
print(
    "The embedded Region as it stands today cannot say where the matrix went, "
    "and the\nproduction code answers only by falling back to the node literally "
    "named 'compute'."
)

# -- 3. canonical validation is unchanged ------------------------------------

rule("3  canonical validation, unchanged and clean")
for name, network in CASES.items():
    canonical = as_network(network)
    assert not validate_network(canonical), (name, validate_network(canonical).issues)
    for node in network.nodes:
        assert not validate_region(node.region.streamed_projection)
        assert added_region_issues(node.region) == ()
    print(f"{name:<10} validate_network: 0 issues   added local-state rules: 0 issues")

# -- 4. the two new rules actually fire --------------------------------------

rule("4  the two rules local-state inputs add")
compute = next(node for node in CASES["embedded"].nodes if node.id == COMPUTE)
held = weight_local_state(FOLDING)
doubled = replace(compute.region, local_state=(held, held))
codes = tuple(issue.code for issue in added_region_issues(doubled))
assert codes == ("local_state.operand_duplicate",), codes
print(f"duplicate local-state operand    -> {codes}")

streamed_compute = next(node for node in CASES["external"].nodes if node.id == COMPUTE)
contradictory = replace(streamed_compute.region, local_state=(held,))
codes = tuple(issue.code for issue in added_region_issues(contradictory))
assert codes == ("local_state.operand_also_streamed",), codes
print(f"streamed and local state at once -> {codes}")

memory = next(node for node in CASES["decoupled"].nodes if node.id == MEMORY)
assert added_region_issues(memory.region) == ()
print("local state W and an output port W -> () (what a parameter source is)")

out_of_domain = replace(
    compute.region,
    local_state=(
        LocalStateInput(held.operand, ScheduledInputRequirements({((9, 9, 9), (0, 0)): 1})),
    ),
)
codes = tuple(issue.code for issue in added_region_issues(out_of_domain))
assert codes == ("local_state.requirement.iteration_out_of_domain",), codes
print(f"local-state requirement domain  -> {codes}")

# -- 5. ambiguity is reported, not resolved ----------------------------------

rule("5  two entry sites are an error with both named")
ambiguous = ProtoNetwork(
    tuple(
        ProtoNode(node.id, replace(node.region, local_state=(held,))) if node.id == REPLAY else node
        for node in CASES["embedded"].nodes
    ),
    CASES["embedded"].edges,
    CASES["embedded"].boundaries,
)
try:
    derive_input_placement(ambiguous, "W")
    raise AssertionError("expected ambiguity")
except PlacementError as error:
    print(f"PlacementError: {error}")

# -- 6. the embedded core reads what the streamed core reads -----------------

rule("6  A2's content: embedded requirements == streamed requirements")
streamed_weight = (
    next(node for node in CASES["external"].nodes if node.id == COMPUTE)
    .region.streamed_projection.input_interface("weight")
    .requirements
)
embedded_weight = compute.region.local_state[0].requirements
assert streamed_weight == embedded_weight
print(
    f"entries: streamed {len(streamed_weight.entries)}  "
    f"local state {len(embedded_weight.entries)}  equal: True"
)
print(
    "A1 (operand only) drops these entries and the claim with them; A2 keeps "
    f"{len(embedded_weight.entries)}\nsparse entries on the embedded Region "
    "that it does not carry today."
)


def as_a1(network: ProtoNetwork) -> ProtoNetwork:
    """The same Networks with requirement-free local state."""

    return ProtoNetwork(
        tuple(
            ProtoNode(
                node.id,
                replace(
                    node.region,
                    local_state=tuple(
                        LocalStateInput(item.operand) for item in node.region.local_state
                    ),
                ),
            )
            for node in network.nodes
        ),
        network.edges,
        network.boundaries,
    )


for name, network in CASES.items():
    for operand_id in ("X", "W"):
        assert derive_input_placement(as_a1(network), operand_id) == derive_input_placement(
            network, operand_id
        )
    assert derive_output_placement(as_a1(network), "Y") == derive_output_placement(network, "Y")
print("every placement above is identical under A1: the derivation never reads requirements.")

# InternalStream is a case with no live producer: over a Network that passes
# canonical validation, every entry site is a boundary or local state.
for name, network in CASES.items():
    for operand_id in ("X", "W"):
        assert not isinstance(derive_input_placement(network, operand_id), InternalStream)
    assert not isinstance(derive_output_placement(network, "Y"), InternalStream)
print("InternalStream never fires over a valid Network -- see the recommendation to drop it.")
for folding in (
    Folding(1, 64, 64, 8, 8),
    Folding(4, 64, 64, 8, 8),
    Folding(1, 1024, 1024, 32, 32),
    Folding(8, 512, 512, 16, 16),
):
    entries = len(weight_local_state(folding).requirements.entries)
    print(
        f"  A2 entry cost  R={folding.repetitions:<2} "
        f"{folding.matrix_width}x{folding.matrix_height} "
        f"PE={folding.pe} SIMD={folding.simd}  {entries:>9,} entries "
        f"(~{entries * 168 / 1e6:.0f} MB)"
    )

# -- 7. MLO: grouping changes no Region value --------------------------------

rule("7  prospective MLO -- one realization over two Networks")
small = decoupled_network(Folding(repetitions=2, matrix_width=32, matrix_height=32, pe=4, simd=4))
group = []
for label, network in (("mvau_a", CASES["decoupled"]), ("mvau_b", small)):
    placement = derive_input_placement(network, "W")
    assert isinstance(placement, LocalState)
    group.append((label, placement))
print("shared realization over:", group)
regrouped = [derive_input_placement(network, "W") for network in (CASES["decoupled"], small)]
assert [item for _label, item in group] == regrouped
assert CASES["decoupled"] == decoupled_network(FOLDING)
print(
    "Grouping is a set of placements held by whoever owns the realization.  The "
    "Region\nvalues are byte-identical before and after, so no memstream Kernel "
    "learns about it."
)

# -- 8. schema B and schema C on the same cases ------------------------------

rule("8  schema B -- same answer, two values")
annotated = AnnotatedNetwork(
    as_network(CASES["decoupled"]),
    {MEMORY: RegionResidency((held.operand,), (held.requirements,))},
)
print("derive_placement_b(decoupled, 'W') =", derive_placement_b(annotated, "W"))
print(f"seams that must carry the companion for this to work end to end: {len(B_THREADING_SITES)}")
for site in B_THREADING_SITES:
    print(f"  - {site}")

rule("9  schema C -- an authored table that can lie")
graph = DispositionGraph(
    (SemanticRequirement("W", ACTIVATION, (64, 64)),),
    (RequirementDisposition("W", DispositionKind.LOCAL_STATE, MEMORY),),
)
print("derive_placement_c(decoupled, 'W') =", derive_placement_c(graph, "W"))
assert disposition_agrees_with_network(graph, CASES["decoupled"], "W")
lying = DispositionGraph(
    graph.requirements,
    (RequirementDisposition("W", DispositionKind.LOCAL_STATE, COMPUTE),),
)
assert not disposition_agrees_with_network(lying, CASES["decoupled"], "W")
print("a disposition naming the wrong node is well-formed and passes its own rules;")
print("catching it needs schema A's derivation, which C must then also carry.")

# -- 10. type and authority accounting ---------------------------------------

rule("10 public type count added by each schema")
print("A: LocalStateInput                                         1 value + 1 Region field")
print("   External / LocalState / InternalStream (derived report)  3, and they replace")
print("   BoundaryDestination / StreamDestination /              3 that exist today")
print("   RegionStateDestination                                 net 0")
print("B: RegionResidency, AnnotatedRegion, AnnotatedNetwork     3 values + 1 semantics")
print("   + a second Region-shaped value at 9 seams              net +3 and a pair everywhere")
print("C: SemanticRequirement, RequirementDisposition,           4 values, one of which")
print("   DispositionKind, DispositionGraph                      duplicates Operand")

print("\nall assertions passed")
