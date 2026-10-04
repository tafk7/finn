# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel's weight memory as an optional Decision over kernels: none, or a memstream.

The MatMul sits in a test root (``kernels.helpers.placed_matmul``) that
declares its streams. The ``none`` case places nothing, so the root's weight
stream ``w`` has only its consumer and is the boundary ``in1_V``;
``memstream`` places a ``MemStreamKernel``, which drives the stream it is
passed and has its own contents and local choices. These tests cover laziness,
optional facts, persistence, atomic switching and diagnostics.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Inapplicable,
    Rejected,
    Unresolved,
    inspection,
    selections,
)
from finn.core.space.errors import ConfigurationError, RequestError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.matmul import MatMulKernel
from kernels.helpers import WeightDelivery, labels, matmul_assembly, matmul_point, placed
from kernels.helpers import with_adapter_memories
from finn.kernels.target import DspBlock

IMAGE = MemStreamKernel.image
RAM_STYLE = MemStreamKernel.ram_style
# The memstream candidate of the MatMul's ``memory`` Decision is ``matmul.memory.memstream``.
STORED_INSTANCE = "matmul.memory.memstream"
ADAPTER_INSTANCE = "x.adapter.input_gen.input_gen"
# Written by output, stored (k, n).
BY_OUTPUT = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
WEIGHTS = tuple(zip(*BY_OUTPUT))
FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
    target_period_ns=5.0,
)

# Each adapter chain's input_gen stages, by the key below the stream's adapter.
INPUT_GEN_STAGES = (
    "input_gen.input_gen",
    "vpc_input_gen.input_gen",
    "input_gen_vpc.input_gen",
    "input_gen_vpc_input_gen.input_gen",
    "input_gen_vpc_input_gen.input_gen_1",
    "vpc_input_gen_vpc.input_gen",
    "vpc_input_gen_vpc_input_gen.input_gen",
    "vpc_input_gen_vpc_input_gen.input_gen_1",
)


def base(**facts):
    return matmul_point(**{**FACTS, **facts})


def selector(point, key="matmul.memory"):
    (choice,) = [item for item in inspection.choices(point) if item.key == key]
    assert choice.selector is not None
    return choice.selector


def transport(point):
    return selector(point, "w.transport")


def stored(point):
    # The candidate's configuration exists whether or not it is selected.
    return inspection.candidate(point.matmul, MatMulKernel.memory, "memstream")


def ram_style(point):
    return inspection.decision_handle(stored(point), MemStreamKernel.ram_style)


def pumped(point):
    return inspection.decision_handle(stored(point), MemStreamKernel.pumped_memory)


def configured(point, case, *, style=None, pe=2, simd=2):
    changes = [
        point.field(selector(point)).change(case),
        point.field(transport(point)).change("direct"),
        point.field(handle(point, "matmul.compute")).change("packed"),
    ]
    if style is not None:
        changes.append(point.field(ram_style(point)).change(style))
        changes.append(point.field(pumped(point)).change(False))
    # The activation stream's adapter follows from the folding: settle the one that fits.
    point = point.with_choices(*changes)
    return with_adapter_memories(
        commit(
            point,
            {
                "matmul.compute.packed.pe": pe,
                "matmul.compute.packed.simd": simd,
                "matmul.compute.packed.compute_pumping": False,
            },
        )
    )


def instance_parameters(point, instance):
    return dict(placed(point.module, instance).parameters)


def delivery(point):
    stored_ = STORED_INSTANCE in labels(point.module)
    return WeightDelivery.MEMSTREAM if stored_ else WeightDelivery.EXTERNAL


def keys(result):
    return {finding.code for finding in result.findings}


def owners(result):
    return {finding.owner for finding in result.findings}


def test_families_share_typed_exports_but_keep_their_own_ports_and_components():
    external = configured(base(), "none")
    stored_ = configured(base(weights=WEIGHTS), "memstream", style="block")
    compute = "matmul.compute.packed"
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, [ADAPTER_INSTANCE, compute]),
        (stored_, {"in0_V", "out0_V"}, [ADAPTER_INSTANCE, compute, STORED_INSTANCE]),
    ):
        module = point.module
        assert isinstance(module, Composed)
        assert {p.name for p in module.pins.ports if isinstance(p, Bus)} == ports
        assert labels(module) == instances
    assert delivery(external) is WeightDelivery.EXTERNAL
    # The Decision reads as the selected candidate's configuration, or None.
    matmul = stored_.matmul
    assert external.matmul.memory is None and external.matmul.supplied == "none"
    assert isinstance(matmul.memory, MemStreamKernel) and matmul.supplied == "memstream"
    assert matmul.memory.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # Instance labels come from the located node names: the candidate is memory.memstream.
    assert [item.node for item in matmul.netlists] == ["compute.packed", "memory.memstream"]
    # The weight stream's producer is the memory's output port, below the MatMul.
    assert stored_.w.endpoints.source_owner == "matmul.memory.memstream.output"
    assert [item.node for item in external.matmul.netlists] == ["compute.packed"]
    memory = instance_parameters(stored_, STORED_INSTANCE)
    assert memory["RAM_STYLE"] == '"block"'
    # What derives a MatMul's netlist depends on its memory.
    assert external.matmul.producer_identity() != matmul.producer_identity()


def test_the_inactive_family_is_never_demanded():
    point = configured(base(), "none")
    evidence = inspection.explain(point, Kernel.module)
    visited = {node.declaration.key for node in evidence.nodes}
    # External delivery places no node: the stream sees only its consumer and is the
    # boundary in1_V. The Decision over nodes is itself the selector.
    assert {"matmul.memory", "w.ends", "w.endpoints"} <= visited
    assert [
        node.selector for node in evidence.nodes if node.declaration.key == "matmul.memory"
    ] == [True]
    # The inactive family is reached only to settle its guard; none of its work runs.
    reached = {
        node.declaration.key: node.result
        for node in evidence.nodes
        if node.declaration.key.startswith("matmul.memory.memstream.")
    }
    assert "matmul.memory.memstream.$selected" in reached
    assert "matmul.memory.memstream.netlist" in reached
    assert all(
        isinstance(result, Inapplicable)
        for key, result in reached.items()
        if key != "matmul.memory.memstream.$selected"
    )
    # Whether weights are known is read (they decide the memory), never their values.
    assert "weights" in visited and "datapath_weights" not in visited
    # The unselected candidate's configuration is still reachable, selected or not.
    inactive = stored(point)
    assert point.matmul.memory is None
    assert inspection.candidate(point.matmul, MatMulKernel.memory, "none") is None
    assert isinstance(inactive, MemStreamKernel)
    assert isinstance(inactive.query(IMAGE), Inapplicable)
    assert isinstance(inactive.field(MemStreamKernel.ram_style).query(), Inapplicable)


def test_case_local_choices_are_owned_by_their_family():
    records = {item.key: item for item in inspection.decisions(base())}
    assert set(records) == {
        "matmul.compute",
        "matmul.compute.packed.pe",
        "matmul.compute.packed.simd",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.int8_dsp58.pe",
        "matmul.compute.int8_dsp58.simd",
        "matmul.compute.int8_dsp58.compute_pumping",
        "matmul.memory",
        "matmul.memory.memstream.pumped_memory",
        "matmul.memory.memstream.ram_style",
        "matmul.realization",
        # Every stream of the root (the set index's with several sets) may need an
        # adapter; each Decision applies only under a plan. Each input_gen stage of each
        # chain owns its memory's ram_style.
        *(
            key
            for stream in ("x", "w", "y", "set")
            for key in (
                f"{stream}.adapter",
                *(f"{stream}.adapter.{chain}.ram_style" for chain in INPUT_GEN_STAGES),
            )
        ),
        "w.transport",
        "w.transport.fifo.buffer.depth",
        "w.transport.fifo.buffer.ram_style",
    }
    assert records["matmul.memory"].selector
    assert records["matmul.memory"].cases == ("none", "memstream")
    local = records["matmul.memory.memstream.ram_style"]
    # The choice belongs to the reusable memory kernel placed by the family.
    assert local.scope == "matmul.memory.memstream" and not local.selector
    (choice,) = [item for item in inspection.choices(base()) if item.key == "matmul.memory"]
    cases = {case.name: (case.scope, case.space_type) for case in choice.cases}
    # The optional Decision's none case places nothing; memstream places the memory.
    assert cases == {
        "none": (None, None),
        "memstream": ("matmul.memory.memstream", MemStreamKernel),
    }
    # Applicability of a case-local choice waits for the selector, and names it.
    unselected = stored(base()).field(RAM_STYLE)
    pending = unselected.candidates()
    assert isinstance(pending, Unresolved) and owners(pending) == {"matmul.memory"}
    chosen = commit(base(), {"matmul.memory": "memstream"}).matmul.memory
    assert chosen.field(RAM_STYLE).candidates() == Available(
        ("auto", "distributed", "block", "ultra")
    )


def test_known_weights_decide_the_memory():
    external = configured(base(), "none")
    assert isinstance(external.query(Kernel.module), Available)
    # A memory without known weights, and known weights without a memory, are refused.
    for point, case in ((base(), "memstream"), (base(weights=WEIGHTS), "none")):
        chosen = commit(point, {"matmul.memory": case}).matmul
        assert keys(chosen.query(MatMulKernel.supply_supported)) == {"matmul-memory"}


def test_the_stored_family_needs_its_own_choices_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS), "memstream")
    assert isinstance(uncommitted.query(Kernel.module), Unresolved)
    # Weights outside their type are refused where MatMul states their range.
    refused = base(weights=((4,) * 4,) * 4).matmul.query(MatMulKernel.weight_tensor)
    assert isinstance(refused, Rejected)
    assert keys(refused) == {"memstream-values"}
    assert owners(refused) == {"matmul.weight_tensor"}
    # A shape error is refused by the memory that packs them.
    wrong = configured(base(weights=((0,),)), "memstream", style="auto").query(Kernel.module)
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message


def test_known_refusals_remain_visible_while_the_family_is_unselected():
    point = commit(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    point = commit(
        point,
        {
            "matmul.compute.packed.pe": 4,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": True,
        },
    )
    assessment = point.inspect(Kernel.module)
    assert isinstance(assessment.accepted_result, Unresolved)
    # The packed core does not wait for the memory: its weights' range is MatMul's own.
    compute = point.matmul.compute.inspect(DotpAxiKernel.module)
    assert isinstance(compute.accepted_result, Available)
    assert point.matmul.compute.parameters()["NARROW_WEIGHTS"] == 0  # the weights hold -4
    assert point.matmul.compute.inspect(DotpAxiKernel.admission).result == Available(True)
    folding = commit(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    folding = commit(folding, {"matmul.compute.packed.simd": 4})
    # Folding refusals settle before a PE, memory or memory style is chosen.
    trial = folding.try_with_choices({handle(folding, "matmul.compute.packed.pe"): 3})
    assert not trial.accepted
    assert "domain-membership" in {
        finding.code for outcome in trial.outcomes for finding in outcome.result.findings
    }


def test_selector_and_case_choices_round_trip_through_an_empty_root():
    point = configured(base(weights=WEIGHTS), "memstream", style="block")
    saved = selections.capture(point)
    assert saved.keys == (
        "matmul.compute",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.packed.pe",
        "matmul.compute.packed.simd",
        "matmul.memory",
        "matmul.memory.memstream.pumped_memory",
        "matmul.memory.memstream.ram_style",
        "w.transport",
        "x.adapter.input_gen.input_gen.ram_style",
    )
    fresh = base(weights=WEIGHTS)
    replayed = selections.restore(fresh, saved)
    assert replayed.accepted
    assert replayed.instance.module == point.module
    # Replay under different supplied facts: the same choices, a new image.
    other = selections.restore(base(weights=tuple(row[::-1] for row in WEIGHTS)), saved)
    assert other.accepted
    assert other.instance.matmul.memory.image != point.matmul.memory.image
    # Without the family's optional fact the choices replay but stay unresolved,
    # and the memory is refused: no known weights to store.
    unresolved = selections.restore(base(), saved)
    assert unresolved.accepted
    assert isinstance(unresolved.instance.query(Kernel.module), Unresolved)
    assert keys(unresolved.instance.matmul.query(MatMulKernel.supply_supported)) == {
        "matmul-memory"
    }
    # Facts that invalidate a saved folding refuse replay atomically.
    changed = base(weights=WEIGHTS, n=5)
    refused = selections.restore(changed, saved)
    assert not refused.accepted and refused.instance is changed
    # A configured receiver is not a replay target.
    with pytest.raises(RequestError):
        selections.restore(point, saved)


def test_switching_families_is_atomic_and_requires_clearing_stale_case_choices():
    memory = configured(base(weights=WEIGHTS), "memstream", style="block")
    stale = memory.try_with_choices(memory.field(selector(memory)).change("none"))
    assert not stale.accepted and stale.instance is memory
    refused = {outcome.owner: outcome for outcome in stale.outcomes if outcome.status == "refused"}
    assert set(refused) == {
        "matmul.memory.memstream.ram_style",
        "matmul.memory.memstream.pumped_memory",
    }
    assert all(isinstance(outcome.result, Inapplicable) for outcome in refused.values())
    assert delivery(memory) is WeightDelivery.MEMSTREAM
    switched = memory.with_choices(
        memory.field(selector(memory)).change("none"),
        memory.field(ram_style(memory)).clear(),
        memory.field(pumped(memory)).clear(),
    )
    # The switch commits; known weights without a memory are then refused.
    assert switched.matmul.supplied == "none"
    assert keys(switched.matmul.query(MatMulKernel.supply_supported)) == {"matmul-memory"}
    assert selections.capture(switched).keys == (
        "matmul.compute",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.packed.pe",
        "matmul.compute.packed.simd",
        "matmul.memory",
        "w.transport",
        "x.adapter.input_gen.input_gen.ram_style",
    )
    # Switching back commits the case-local choices in the same batch.
    back = switched.with_choices(
        switched.field(selector(switched)).change("memstream"),
        switched.field(ram_style(switched)).change("distributed"),
        switched.field(pumped(switched)).change(False),
    )
    assert delivery(back) is WeightDelivery.MEMSTREAM
    assert instance_parameters(back, STORED_INSTANCE)["RAM_STYLE"] == '"distributed"'
    # A case-local choice for an unselected family is refused, not stored.
    early = switched.try_with_choices(switched.field(ram_style(switched)).change("block"))
    assert not early.accepted and early.instance is switched


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_public_adapter_matches_the_space_path(delivery):
    known = delivery is not WeightDelivery.EXTERNAL
    adapted = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=delivery,
        weights=[list(row) for row in WEIGHTS] if known else None,
    )
    point = configured(
        base(weights=WEIGHTS) if known else base(),
        delivery.value,
        style="auto" if known else None,
    )
    assert adapted.module == point.module


@pytest.mark.parametrize("style", ("auto", "distributed", "block", "ultra"))
def test_ram_style_reaches_the_stored_memory(style):
    built = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=WEIGHTS,
        ram_style=style,
    )
    assert ("RAM_STYLE", f'"{style}"') in placed(built.module, STORED_INSTANCE).parameters


def test_adapter_rejects_an_unknown_ram_style():
    with pytest.raises(ValueError, match="domain"):
        matmul_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.MEMSTREAM,
            weights=WEIGHTS,
            ram_style="lutram",
        )


# -- the weight stream's transport slot --------------------------------------------------

FIFO = "w.transport.fifo.buffer"
FIFO_DEPTH = "w.transport.fifo.buffer.depth"
FIFO_RAM_STYLE = "w.transport.fifo.buffer.ram_style"


def handle(point, key):
    return next(item.reference for item in inspection.decisions(point) if item.key == key)


def buffered(point, depth=None):
    changes = [
        point.field(transport(point)).change("fifo"),
        point.field(handle(point, FIFO_RAM_STYLE)).change("auto"),
    ]
    if depth is not None:
        changes.append(point.field(handle(point, FIFO_DEPTH)).change(depth))
    return point.with_choices(*changes)


def test_a_buffered_stream_places_a_fifo_between_its_producer_and_consumer():
    for case, weights in (("none", None), ("memstream", WEIGHTS)):
        point = configured(
            base(weights=weights) if weights else base(), case, style=("auto" if weights else None)
        )
        module = buffered(point, depth=32).module
        assert FIFO in labels(module)
        hops = {(link.source.instance, link.sink.instance) for link in module.fragment.links}
        producer = STORED_INSTANCE if case == "memstream" else None
        assert (producer, FIFO) in hops
        assert (FIFO, "matmul.compute.packed") in hops
        assert dict(placed(module, FIFO).parameters)["DEPTH"] == 32


def test_fifo_depth_is_owned_by_the_stream_and_only_demanded_when_selected():
    point = configured(base(), "none")
    records = {item.key: item for item in inspection.decisions(point)}
    assert records[FIFO_DEPTH].scope == FIFO
    assert isinstance(point.field(handle(point, FIFO_DEPTH)).query(), Inapplicable)
    undecided = buffered(point)
    assert isinstance(undecided.query(Kernel.module), Unresolved)
    with pytest.raises(ConfigurationError):
        buffered(point, depth=1)
    deep = buffered(point, depth=64)
    saved = selections.capture(deep)
    assert FIFO_DEPTH in saved.keys
    assert selections.restore(base(), saved).instance.module == deep.module
    # Removing the FIFO requires clearing its stale depth and memory style atomically.
    stale = deep.try_with_choices(deep.field(transport(deep)).change("direct"))
    assert not stale.accepted
    direct = deep.with_choices(
        deep.field(transport(deep)).change("direct"),
        deep.field(handle(deep, FIFO_DEPTH)).clear(),
        deep.field(handle(deep, FIFO_RAM_STYLE)).clear(),
    )
    assert direct.module == point.module


def test_adapter_places_the_weight_fifo_on_request():
    built = matmul_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert FIFO in labels(built.module)
