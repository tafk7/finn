# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel's weight memory as an optional Decision over kernels: none, or a memstream.

The ``none`` case places nothing, so ``weight_stream`` has only its consumer
and is the boundary ``in1_V``; ``memstream`` places a ``MemStreamKernel``,
which references the stream as its producer and has its own contents and
local choices. These tests cover laziness, optional facts, persistence, atomic
switching and diagnostics.
"""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Inapplicable,
    JSONValue,
    Rejected,
    SelectionSchema,
    Unresolved,
    ValueCodec,
    codec_for,
    codecs,
    design_space,
    inspection,
    selections,
)
from finn.core.space.errors import ConfigurationError, RequestError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import ModuleBuildRequirements, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.matmul import MatMulKernel, WeightDelivery, matmul_assembly
from kernels.helpers import settled
from finn.kernels.resources import resource_root, template_root
from finn.kernels.target import DspBlock

ROOT = Path(__file__).resolve().parents[2]
IMAGE = MemStreamKernel.image
RAM_STYLE = MemStreamKernel.ram_style
# The memstream candidate of the ``memory`` Decision is named ``memory.memstream``.
STORED_INSTANCE = "u_memory_memstream"
ADAPTER_INSTANCE = "u_activations_input_gen"
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
    return design_space(MatMulKernel(**{**FACTS, **facts}))


def selector(point, key="memory"):
    (choice,) = [item for item in inspection.choices(point) if item.key == key]
    assert choice.selector is not None
    return choice.selector


def transport(point):
    return selector(point, "weight_stream.transport")


def stored(point):
    # The candidate's configuration exists whether or not it is selected.
    return inspection.candidate(point, MatMulKernel.memory, "memstream")


def ram_style(point):
    return inspection.decision_handle(stored(point), MemStreamKernel.ram_style)


def pumped(point):
    return inspection.decision_handle(stored(point), MemStreamKernel.pumped_memory)


def configured(point, case, *, style=None, pe=2, simd=2):
    changes = [
        point.field(selector(point)).change(case),
        point.field(transport(point)).change("direct"),
    ]
    if style is not None:
        changes.append(point.field(ram_style(point)).change(style))
        changes.append(point.field(pumped(point)).change(False))
    # The activation stream's adapter follows from the folding: settle the one that fits.
    point = point.with_choices(*changes, compute="packed")
    return settled(
        point.with_choices(
            {
                MatMulKernel.packed.pe: pe,
                MatMulKernel.packed.simd: simd,
                MatMulKernel.packed.compute_pumping: False,
            }
        )
    )


def instance_parameters(point, instance):
    (found,) = (
        dict(item.requirements.parameters)
        for item in point.structure.structure.instances
        if item.instance_id == instance
    )
    return found


def delivery(point):
    names = {item.instance_id for item in point.structure.structure.instances}
    return WeightDelivery.MEMSTREAM if STORED_INSTANCE in names else WeightDelivery.EXTERNAL


def keys(result):
    return {finding.code for finding in result.findings}


def owners(result):
    return {finding.owner for finding in result.findings}


def test_families_share_typed_exports_but_keep_their_own_ports_and_components():
    external = configured(base(), "none")
    stored_ = configured(base(weights=WEIGHTS), "memstream", style="block")
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, ["u_compute_packed", ADAPTER_INSTANCE]),
        (stored_, {"in0_V", "out0_V"}, ["u_compute_packed", STORED_INSTANCE, ADAPTER_INSTANCE]),
    ):
        built = point.structure
        requirements = point.build_requirements
        assert isinstance(requirements, ModuleBuildRequirements)
        assert requirements == built.requirements
        assert {p.name for p in built.structure.top_abi.ports if isinstance(p, Bus)} == ports
        assert [item.instance_id for item in built.structure.instances] == instances
    assert delivery(external) is WeightDelivery.EXTERNAL
    # The Decision reads as the selected candidate's configuration, or None.
    assert external.memory is None and external.supplied == "none"
    assert isinstance(stored_.memory, MemStreamKernel) and stored_.supplied == "memstream"
    assert stored_.memory.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # Instance names come from the located node names: the candidate is memory.memstream.
    assert [item.node for item in stored_.modules] == ["compute.packed", "memory.memstream"]
    # The stream's producer is the memory's output port; netlist names its module's instance.
    assert [(item.node, item.value.source_owner) for item in stored_.streams][1] == (
        "weight_stream",
        "memory.memstream.output",
    )
    assert [item.node for item in external.modules] == ["compute.packed"]
    memory = instance_parameters(stored_, STORED_INSTANCE)
    assert memory["RAM_STYLE"] == '"block"'
    assert external.build_requirements.implementation_id != (
        stored_.build_requirements.implementation_id
    )


def test_the_inactive_family_is_never_demanded():
    point = configured(base(), "none")
    evidence = inspection.explain(point, MatMulKernel.structure)
    visited = {node.declaration.key for node in evidence.nodes}
    # External delivery places no node: the stream sees only its consumer and is the
    # boundary in1_V. The Decision over nodes is itself the selector.
    assert {"memory", "weight_stream.ends", "weight_stream.endpoints"} <= visited
    assert [node.selector for node in evidence.nodes if node.declaration.key == "memory"] == [True]
    # The inactive family is reached only to settle its guard; none of its work runs.
    reached = {
        node.declaration.key: node.result
        for node in evidence.nodes
        if node.declaration.key.startswith("memory.memstream.")
    }
    assert "memory.memstream.$selected" in reached
    assert "memory.memstream.build_requirements" in reached
    assert all(
        isinstance(result, Inapplicable)
        for key, result in reached.items()
        if key != "memory.memstream.$selected"
    )
    assert "weights" not in visited
    # The unselected candidate's configuration is still reachable, selected or not.
    inactive = stored(point)
    assert point.memory is None
    assert inspection.candidate(point, MatMulKernel.memory, "none") is None
    assert isinstance(inactive, MemStreamKernel)
    assert isinstance(inactive.query(IMAGE), Inapplicable)
    assert isinstance(inactive.field(MemStreamKernel.ram_style).query(), Inapplicable)


def test_case_local_choices_are_owned_by_their_family():
    records = {item.key: item for item in inspection.decisions(base())}
    assert set(records) == {
        "compute",
        "compute.packed.pe",
        "compute.packed.simd",
        "compute.packed.compute_pumping",
        "compute.int8_dsp58.pe",
        "compute.int8_dsp58.simd",
        "compute.int8_dsp58.compute_pumping",
        "memory",
        "memory.memstream.pumped_memory",
        "memory.memstream.ram_style",
        "realization",
        "fused",  # applies only where a parent places it
        # Every stream may need an adapter; each Decision applies only under a plan.
        # Each input_gen stage of each chain owns its memory's ram_style.
        *(
            key
            for stream in ("activations", "results", "set_index", "weight_stream")
            for key in (
                f"{stream}.adapter",
                *(f"{stream}.adapter.{chain}.ram_style" for chain in INPUT_GEN_STAGES),
            )
        ),
        "weight_stream.transport",
        "weight_stream.transport.fifo.buffer.depth",
        "weight_stream.transport.fifo.buffer.ram_style",
    }
    assert records["memory"].selector
    assert records["memory"].cases == ("none", "memstream")
    local = records["memory.memstream.ram_style"]
    # The choice belongs to the reusable memory kernel placed by the family.
    assert local.scope == "memory.memstream" and not local.selector
    (choice,) = [item for item in inspection.choices(base()) if item.key == "memory"]
    cases = {case.name: (case.scope, case.space_type) for case in choice.cases}
    # The optional Decision's none case places nothing; memstream places the memory.
    assert cases == {
        "none": (None, None),
        "memstream": ("memory.memstream", MemStreamKernel),
    }
    # Applicability of a case-local choice waits for the selector, and names it.
    unselected = stored(base()).field(RAM_STYLE)
    pending = unselected.candidates()
    assert isinstance(pending, Unresolved) and owners(pending) == {"memory"}
    chosen = base().with_choices(memory="memstream").memory
    assert chosen.field(RAM_STYLE).candidates() == Available(
        ("auto", "distributed", "block", "ultra")
    )


def test_missing_stored_weights_leave_only_the_selected_family_unresolved():
    external = configured(base(), "none")
    assert isinstance(external.query(MatMulKernel.structure), Available)
    stored_ = configured(base(), "memstream", style="distributed")
    family = stored_.memory
    assert family is not None and family.ram_style == "distributed"
    assert isinstance(family.query(IMAGE), Unresolved)
    assessment = stored_.inspect(MatMulKernel.structure)
    assert isinstance(assessment.accepted_result, Unresolved)
    # Only the selected family's module waits; every stream is already accepted.
    waiting = {
        key
        for key, result in assessment.constraints.results.items()
        if not isinstance(result, (Available, Inapplicable))
    }
    # The packed core waits too: known weights decide its NARROW_WEIGHTS.
    assert waiting == {
        "memory.memstream.build_requirements",
        "compute.packed.build_requirements",
    }
    assert owners(stored_.compute.query(DotpAxiKernel.build_requirements)) == {"weights"}
    evidence = inspection.explain(stored_, MatMulKernel.structure)
    omitted = [node for node in evidence.nodes if node.input_presence == "omitted"]
    assert [node.declaration.key for node in omitted] == ["weights"]


def test_the_stored_family_needs_its_own_choices_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS), "memstream")
    assert isinstance(uncommitted.query(MatMulKernel.structure), Unresolved)
    bad = configured(base(weights=((4,) * 4,) * 4), "memstream", style="auto")
    refused = bad.query(MatMulKernel.structure)
    assert isinstance(refused, Rejected)
    assert keys(refused) == {"memstream-values"}
    assert owners(refused) == {"memory.memstream.image"}
    # A shape error is refused the same way, and never demanded by external delivery.
    wrong = configured(base(weights=((0,),)), "memstream", style="auto").query(
        MatMulKernel.structure
    )
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message
    assert isinstance(
        configured(base(weights=((0,),)), "none").query(MatMulKernel.structure), Available
    )


def test_known_refusals_remain_visible_while_the_family_is_unselected():
    point = base(weights=WEIGHTS).with_choices(compute="packed")
    point = point.with_choices(
        {
            MatMulKernel.packed.pe: 4,
            MatMulKernel.packed.simd: 2,
            MatMulKernel.packed.compute_pumping: True,
        }
    )
    assessment = point.inspect(MatMulKernel.structure)
    assert isinstance(assessment.accepted_result, Unresolved)
    # The packed core waits only for the memory, which decides whether its
    # weights are known (NARROW_WEIGHTS); its own refusals settle meanwhile.
    compute = point.compute.inspect(DotpAxiKernel.build_requirements)
    assert owners(compute.accepted_result) == {"memory"}
    assert point.compute.inspect(DotpAxiKernel.admission).result == Available(True)
    folding = base(weights=WEIGHTS).with_choices(compute="packed")
    folding = folding.with_choices({MatMulKernel.packed.simd: 4})
    # Folding refusals settle before a PE, memory or memory style is chosen.
    trial = folding.try_with_choices({MatMulKernel.packed.pe: 3})
    assert not trial.accepted
    assert "domain-membership" in {
        finding.code for outcome in trial.outcomes for finding in outcome.result.findings
    }


STRING: ValueCodec[str] = ValueCodec("string", 1, lambda value: value, str)
INTEGER: ValueCodec[int] = ValueCodec("integer", 1, lambda value: value, int)


def _boolean(value: JSONValue) -> bool:
    if type(value) is not bool:
        raise ValueError("expected a boolean")
    return value


BOOLEAN: ValueCodec[bool] = ValueCodec("boolean", 1, lambda value: value, _boolean)


def schema(point):
    return SelectionSchema(
        MatMulKernel,
        family="finn.matmul",
        version=1,
        bindings=(
            codec_for(selector(point), STRING),
            codec_for(transport(point), STRING),
            # A reference into the candidate is accepted as well as a handle.
            codec_for(MatMulKernel.memory["memstream"].ram_style, STRING),  # type: ignore[index]
            codec_for(MatMulKernel.memory["memstream"].pumped_memory, BOOLEAN),  # type: ignore[index]
            codec_for(MatMulKernel.packed.pe, INTEGER),
            codec_for(MatMulKernel.packed.simd, INTEGER),
            codec_for(MatMulKernel.compute, STRING),
            codec_for(MatMulKernel.activations.adapter, STRING),
            codec_for(MatMulKernel.activations.adapter["input_gen"].input_gen.ram_style, STRING),
            codec_for(MatMulKernel.packed.compute_pumping, BOOLEAN),
        ),
    )


def test_selector_and_case_choices_round_trip_through_an_empty_root():
    point = configured(base(weights=WEIGHTS), "memstream", style="block")
    saved = selections.capture(point)
    assert saved.keys == (
        "activations.adapter",
        "activations.adapter.input_gen.input_gen.ram_style",
        "compute",
        "compute.packed.compute_pumping",
        "compute.packed.pe",
        "compute.packed.simd",
        "memory",
        "memory.memstream.pumped_memory",
        "memory.memstream.ram_style",
        "weight_stream.transport",
    )
    document = codecs.encode(saved, schema(point))
    decoded = codecs.decode(document, schema(point))
    assert decoded == saved
    fresh = base(weights=WEIGHTS)
    replayed = selections.restore(fresh, decoded)
    assert replayed.accepted
    assert replayed.instance.structure == point.structure
    # Replay under different supplied facts: the same choices, a new image.
    other = selections.restore(base(weights=tuple(row[::-1] for row in WEIGHTS)), decoded)
    assert other.accepted
    assert other.instance.memory.image != point.memory.image
    # Without the family's optional fact the choices replay but stay unresolved.
    unresolved = selections.restore(base(), decoded)
    assert unresolved.accepted
    assert isinstance(unresolved.instance.query(MatMulKernel.structure), Unresolved)
    # Facts that invalidate a saved folding refuse replay atomically.
    changed = base(weights=WEIGHTS, n=5)
    refused = selections.restore(changed, decoded)
    assert not refused.accepted and refused.instance is changed
    # A configured receiver is not a replay target.
    with pytest.raises(RequestError):
        selections.restore(point, decoded)


def test_switching_families_is_atomic_and_requires_clearing_stale_case_choices():
    memory = configured(base(weights=WEIGHTS), "memstream", style="block")
    stale = memory.try_with_choices(memory.field(selector(memory)).change("none"))
    assert not stale.accepted and stale.instance is memory
    refused = {outcome.owner: outcome for outcome in stale.outcomes if outcome.status == "refused"}
    assert set(refused) == {"memory.memstream.ram_style", "memory.memstream.pumped_memory"}
    assert all(isinstance(outcome.result, Inapplicable) for outcome in refused.values())
    assert delivery(memory) is WeightDelivery.MEMSTREAM
    switched = memory.with_choices(
        memory.field(selector(memory)).change("none"),
        memory.field(ram_style(memory)).clear(),
        memory.field(pumped(memory)).clear(),
    )
    assert delivery(switched) is WeightDelivery.EXTERNAL
    assert selections.capture(switched).keys == (
        "activations.adapter",
        "activations.adapter.input_gen.input_gen.ram_style",
        "compute",
        "compute.packed.compute_pumping",
        "compute.packed.pe",
        "compute.packed.simd",
        "memory",
        "weight_stream.transport",
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
    composed = point.structure
    assert (adapted.structure, adapted.requirements) == (composed.structure, composed.requirements)


@pytest.mark.parametrize("style", ("auto", "distributed", "block", "ultra"))
def test_ram_style_reaches_the_prepared_build_without_data_slots(tmp_path, style):
    built = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=WEIGHTS,
        ram_style=style,
    )
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        built.requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    assert prepared.slots == ()
    assert ("RAM_STYLE", f'"{style}"') in dict(
        (item.instance_id, item.requirements.parameters) for item in built.structure.instances
    )[STORED_INSTANCE]


def test_adapter_rejects_an_unknown_ram_style():
    with pytest.raises(ValueError, match="domain"):
        matmul_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.MEMSTREAM,
            weights=WEIGHTS,
            ram_style="lutram",
        )


# -- the weight stream's transport slot --------------------------------------------------

FIFO_DEPTH = "weight_stream.transport.fifo.buffer.depth"
FIFO_RAM_STYLE = "weight_stream.transport.fifo.buffer.ram_style"


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
        fifo = buffered(point, depth=32)
        built = fifo.structure
        instances = [item.instance_id for item in built.structure.instances]
        assert instances[-1] == "u_weight_stream_fifo"
        destinations = {
            (wire.destination.pin.instance_id, wire.source.pin.instance_id)
            for wire in built.structure.wires
            if hasattr(wire.source, "pin")
            and wire.destination.pin.signal_id in ("idat", "s_axis_weights_tdata")
        }
        producer = STORED_INSTANCE if case == "memstream" else None
        assert ("u_weight_stream_fifo", producer) in destinations
        assert ("u_compute_packed", "u_weight_stream_fifo") in destinations
        depth = dict(built.structure.instances[-1].requirements.parameters)["DEPTH"]
        assert depth == 32


def test_fifo_depth_is_owned_by_the_stream_and_only_demanded_when_selected():
    point = configured(base(), "none")
    records = {item.key: item for item in inspection.decisions(point)}
    assert records[FIFO_DEPTH].scope == "weight_stream.transport.fifo.buffer"
    assert isinstance(point.field(handle(point, FIFO_DEPTH)).query(), Inapplicable)
    undecided = buffered(point)
    assert isinstance(undecided.query(MatMulKernel.structure), Unresolved)
    with pytest.raises(ConfigurationError):
        buffered(point, depth=1)
    deep = buffered(point, depth=64)
    saved = selections.capture(deep)
    assert FIFO_DEPTH in saved.keys
    assert selections.restore(base(), saved).instance.structure == deep.structure
    # Removing the FIFO requires clearing its stale depth and memory style atomically.
    stale = deep.try_with_choices(deep.field(transport(deep)).change("direct"))
    assert not stale.accepted
    direct = deep.with_choices(
        deep.field(transport(deep)).change("direct"),
        deep.field(handle(deep, FIFO_DEPTH)).clear(),
        deep.field(handle(deep, FIFO_RAM_STYLE)).clear(),
    )
    assert direct.structure == point.structure


def test_adapter_places_the_weight_fifo_on_request():
    built = matmul_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert [item.instance_id for item in built.structure.instances][-1] == "u_weight_stream_fifo"
