# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU weight delivery as a Decision over nodes: nothing, or a cyclic delivery node.

External delivery places no node and presents ``in1_V``; cyclic delivery places
the ``cyclic`` CyclicDelivery node, with its own initialization fact and local
choice. These tests cover laziness, optional facts, persistence, atomic
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
    configure,
    inspection,
    selections,
)
from finn.core.space.errors import ConfigurationError, RequestError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import ModuleBuildRequirements, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.mvau import MVAU, WeightDelivery, mvau_assembly
from finn.kernels.resources import resource_root, template_root
from finn.kernels.target import DspBlock

ROOT = Path(__file__).resolve().parents[2]
IMAGE = CyclicDelivery.image
ROM_STYLE = CyclicDelivery.rom_style
# The cyclic candidate of the ``implementation`` Decision is named ``implementation.cyclic``.
CYCLIC_INSTANCE = "u_implementation_cyclic"
WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
FACTS = dict(
    repetitions=3,
    matrix_width=4,
    matrix_height=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
    segment_length=0,
)


def base(**facts):
    return configure(MVAU(**{**FACTS, **facts}))


def selector(point, key="implementation"):
    (choice,) = [item for item in inspection.choices(point) if item.key == key]
    assert choice.selector is not None
    return choice.selector


def transport(point):
    return selector(point, "weight_stream.transport")


def rom_style(point):
    # The candidate's configuration exists whether or not it is selected.
    return inspection.decision_handle(point.cyclic, CyclicDelivery.rom_style)


def configured(point, case, *, style=None, pe=2, simd=2):
    changes = [
        point.field(selector(point)).change(case),
        point.field(transport(point)).change("direct"),
        point.compute.field(DotpAxiKernel.compute_pumping).change(False),
    ]
    if style is not None:
        changes.append(point.field(rom_style(point)).change(style))
    return point.with_choices(*changes, pe=pe, simd=simd)


def delivery(point):
    names = {item.instance_id for item in point.structure().structure.instances}
    return WeightDelivery.CYCLIC if CYCLIC_INSTANCE in names else WeightDelivery.EXTERNAL


def keys(result):
    return {finding.code for finding in result.findings}


def owners(result):
    return {finding.owner for finding in result.findings}


def test_families_share_typed_exports_but_keep_their_own_ports_and_components():
    external = configured(base(), "external")
    cyclic = configured(base(weights=WEIGHTS), "cyclic", style="block")
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, ["u_replay", "u_compute"]),
        (cyclic, {"in0_V", "out0_V"}, ["u_replay", "u_compute", CYCLIC_INSTANCE]),
    ):
        built = point.structure()
        requirements = point.build_requirements()
        assert isinstance(requirements, ModuleBuildRequirements)
        assert requirements == built.requirements
        assert {p.name for p in built.structure.top_abi.ports if isinstance(p, Bus)} == ports
        assert [item.instance_id for item in built.structure.instances] == instances
    assert delivery(external) is WeightDelivery.EXTERNAL
    # The Decision reads as the selected candidate's configuration, or None.
    assert external.implementation is None and external.delivery == "external"
    assert isinstance(cyclic.implementation, CyclicDelivery) and cyclic.delivery == "cyclic"
    assert cyclic.implementation.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # Instance names come from the located node names: the candidate is implementation.cyclic.
    assert [item.node for item in cyclic.modules] == ["replay", "compute", "implementation.cyclic"]
    assert [(item.node, item.value.source_owner) for item in cyclic.streams][2] == (
        "weight_stream",
        "implementation.cyclic",
    )
    assert [item.node for item in external.modules] == ["replay", "compute"]
    rom = dict(cyclic.structure().structure.instances[2].requirements.parameters)
    assert rom["ROM_STYLE"] == '"block"'
    assert external.build_requirements().implementation_id != (
        cyclic.build_requirements().implementation_id
    )


def test_the_inactive_family_is_never_demanded():
    point = configured(base(), "external")
    evidence = inspection.explain(point, MVAU.structure)
    visited = {node.declaration.key for node in evidence.nodes}
    # External delivery places no node; the boundary member in1_V drives the stream.
    # The Decision over nodes is itself the selector (no generated ``$selector`` node).
    assert {"implementation", "in1_V", "weight_stream.source"} <= visited
    assert [
        node.selector for node in evidence.nodes if node.declaration.key == "implementation"
    ] == [True]
    # The inactive family is reached only to settle its guard; none of its work runs.
    reached = {
        node.declaration.key: node.result
        for node in evidence.nodes
        if node.declaration.key.startswith("implementation.cyclic.")
    }
    assert set(reached) == {
        "implementation.cyclic.$selected",
        "implementation.cyclic.build_requirements",
        "implementation.cyclic.output",
    }
    assert all(
        isinstance(result, Inapplicable)
        for key, result in reached.items()
        if key != "implementation.cyclic.$selected"
    )
    assert "weights" not in visited
    # The unselected candidate's configuration is still reachable, selected or not.
    inactive = point.cyclic
    assert point.implementation is None
    assert inspection.candidate(point, MVAU.implementation, "external") is None
    assert isinstance(inspection.candidate(point, MVAU.implementation, "cyclic"), CyclicDelivery)
    assert isinstance(inactive.query(IMAGE), Inapplicable)
    assert isinstance(inactive.field(CyclicDelivery.rom_style).query(), Inapplicable)


def test_case_local_choices_are_owned_by_their_family():
    records = {item.key: item for item in inspection.decisions(base())}
    assert set(records) == {
        "pe",
        "simd",
        "compute.compute_pumping",
        "implementation",
        "implementation.cyclic.rom_style",
        "weight_stream.transport",
        "weight_stream.transport.fifo.buffer.depth",
        "weight_stream.transport.fifo.buffer.ram_style",
    }
    assert records["implementation"].selector
    assert records["implementation"].cases == ("external", "cyclic")
    local = records["implementation.cyclic.rom_style"]
    # The choice belongs to the reusable delivery kernel placed by the family.
    assert local.scope == "implementation.cyclic" and not local.selector
    cases = {
        case.name: (case.scope, case.space_type) for case in inspection.choices(base())[0].cases
    }
    # External delivery places nothing (a None candidate, formerly the empty External
    # family); cyclic places the reusable delivery kernel.
    assert cases == {
        "external": (None, None),
        "cyclic": ("implementation.cyclic", CyclicDelivery),
    }
    # Applicability of a case-local choice waits for the selector, and names it.
    unselected = base().cyclic.field(ROM_STYLE)
    pending = unselected.candidates()
    assert isinstance(pending, Unresolved) and owners(pending) == {"implementation"}
    chosen = base().with_choices(implementation="cyclic").cyclic
    assert chosen.field(ROM_STYLE).candidates() == Available(("auto", "distributed", "block"))


def test_missing_cyclic_weights_leave_only_the_selected_family_unresolved():
    external = configured(base(), "external")
    assert isinstance(external.structure.query(), Available)
    cyclic = configured(base(), "cyclic", style="distributed")
    family = cyclic.implementation
    assert family is not None and family.rom_style == "distributed"
    assert isinstance(family.query(IMAGE), Unresolved)
    assessment = cyclic.structure.inspect()
    assert isinstance(assessment.accepted_result, Unresolved)
    # Only the selected family's module waits; every stream is already accepted.
    waiting = {
        key
        for key, result in assessment.constraints.results.items()
        if not isinstance(result, Available)
    }
    assert waiting == {"implementation.cyclic.build_requirements"}
    # The compute product and folding do not wait for the family's optional fact.
    assert isinstance(cyclic.compute.build_requirements.query(), Available)
    evidence = inspection.explain(cyclic, MVAU.structure)
    omitted = [node for node in evidence.nodes if node.input_presence == "omitted"]
    assert [node.declaration.key for node in omitted] == ["weights"]


def test_cyclic_family_needs_its_own_rom_choice_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS), "cyclic")
    assert isinstance(uncommitted.structure.query(), Unresolved)
    bad = configured(base(weights=((4,) * 4,) * 4), "cyclic", style="auto")
    refused = bad.structure.query()
    assert isinstance(refused, Rejected)
    assert keys(refused) == {"cyclic-values"}
    assert owners(refused) == {"implementation.cyclic.image"}
    # A shape error is refused the same way, and never demanded by external delivery.
    wrong = configured(base(weights=((0,),)), "cyclic", style="auto").structure.query()
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message
    assert isinstance(configured(base(weights=((0,),)), "external").structure.query(), Available)


def test_known_refusals_remain_visible_while_the_family_is_unselected():
    point = base(weights=WEIGHTS).with_choices(pe=4, simd=2)
    point = point.compute.with_choices(compute_pumping=True).root
    assessment = point.structure.inspect()
    assert isinstance(assessment.accepted_result, Unresolved)
    compute = point.compute.build_requirements.inspect()
    assert isinstance(compute.accepted_result, Available)
    folding = base(weights=WEIGHTS).with_choices(simd=4)
    # Folding refusals settle before a PE, family or ROM style is chosen.
    trial = folding.try_with_choices(pe=3)
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
        MVAU,
        family="finn.mvau",
        version=1,
        bindings=(
            codec_for(selector(point), STRING),
            codec_for(transport(point), STRING),
            # A reference into the candidate is accepted as well as a handle.
            codec_for(MVAU.cyclic.rom_style, STRING),
            codec_for(MVAU.pe, INTEGER),
            codec_for(MVAU.simd, INTEGER),
            codec_for(
                inspection.decision_handle(point.compute, DotpAxiKernel.compute_pumping), BOOLEAN
            ),
        ),
    )


def test_selector_and_case_choices_round_trip_through_an_empty_root():
    point = configured(base(weights=WEIGHTS), "cyclic", style="block")
    saved = selections.capture(point)
    assert saved.keys == (
        "compute.compute_pumping",
        "implementation",
        "implementation.cyclic.rom_style",
        "pe",
        "simd",
        "weight_stream.transport",
    )
    document = codecs.encode(saved, schema(point))
    decoded = codecs.decode(document, schema(point))
    assert decoded == saved
    fresh = base(weights=WEIGHTS)
    replayed = selections.restore(fresh, decoded)
    assert replayed.accepted
    assert replayed.instance.structure() == point.structure()
    # Replay under different supplied facts: the same choices, a new image.
    other = selections.restore(base(weights=tuple(row[::-1] for row in WEIGHTS)), decoded)
    assert other.accepted
    assert other.instance.cyclic.image != point.cyclic.image
    # Without the family's optional fact the choices replay but stay unresolved.
    unresolved = selections.restore(base(), decoded)
    assert unresolved.accepted
    assert isinstance(unresolved.instance.structure.query(), Unresolved)
    # Facts that invalidate a saved folding refuse replay atomically.
    changed = base(weights=WEIGHTS, matrix_height=5)
    refused = selections.restore(changed, decoded)
    assert not refused.accepted and refused.instance is changed
    # A configured receiver is not a replay target.
    with pytest.raises(RequestError):
        selections.restore(point, decoded)


def test_switching_families_is_atomic_and_requires_clearing_stale_case_choices():
    cyclic = configured(base(weights=WEIGHTS), "cyclic", style="block")
    stale = cyclic.try_with_choices(cyclic.field(selector(cyclic)).change("external"))
    assert not stale.accepted and stale.instance is cyclic
    refused = {outcome.owner: outcome for outcome in stale.outcomes if outcome.status == "refused"}
    assert set(refused) == {"implementation.cyclic.rom_style"}
    assert isinstance(refused["implementation.cyclic.rom_style"].result, Inapplicable)
    assert delivery(cyclic) is WeightDelivery.CYCLIC
    switched = cyclic.with_choices(
        cyclic.field(selector(cyclic)).change("external"),
        cyclic.field(rom_style(cyclic)).clear(),
    )
    assert delivery(switched) is WeightDelivery.EXTERNAL
    assert selections.capture(switched).keys == (
        "compute.compute_pumping",
        "implementation",
        "pe",
        "simd",
        "weight_stream.transport",
    )
    # Switching back commits the case-local choice in the same batch.
    back = switched.with_choices(
        switched.field(selector(switched)).change("cyclic"),
        switched.field(rom_style(switched)).change("distributed"),
    )
    assert delivery(back) is WeightDelivery.CYCLIC
    rom = dict(back.structure().structure.instances[2].requirements.parameters)
    assert rom["ROM_STYLE"] == '"distributed"'
    # A case-local choice for an unselected family is refused, not stored.
    early = switched.try_with_choices(switched.field(rom_style(switched)).change("block"))
    assert not early.accepted and early.instance is switched


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_public_adapter_matches_the_space_path(delivery):
    cyclic = delivery is WeightDelivery.CYCLIC
    adapted = mvau_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=delivery,
        weights=[list(row) for row in WEIGHTS] if cyclic else None,
    )
    point = configured(
        base(weights=WEIGHTS) if cyclic else base(),
        delivery.value,
        style="auto" if cyclic else None,
    )
    composed = point.structure()
    assert (adapted.structure, adapted.requirements) == (composed.structure, composed.requirements)


@pytest.mark.parametrize("style", ("auto", "distributed", "block"))
def test_rom_style_reaches_the_prepared_build_without_data_slots(tmp_path, style):
    built = mvau_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=WeightDelivery.CYCLIC,
        weights=WEIGHTS,
        rom_style=style,
    )
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        built.requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    assert prepared.slots == ()
    assert ("ROM_STYLE", f'"{style}"') in dict(
        (item.instance_id, item.requirements.parameters) for item in built.structure.instances
    )[CYCLIC_INSTANCE]


def test_adapter_rejects_an_unknown_rom_style():
    with pytest.raises(ValueError, match="domain"):
        mvau_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.CYCLIC,
            weights=WEIGHTS,
            rom_style="ultra",
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
    for case, weights in (("external", None), ("cyclic", WEIGHTS)):
        point = configured(
            base(weights=weights) if weights else base(), case, style=("auto" if weights else None)
        )
        fifo = buffered(point, depth=32)
        built = fifo.structure()
        instances = [item.instance_id for item in built.structure.instances]
        assert instances[-1] == "u_weight_stream_fifo"
        destinations = {
            (wire.destination.pin.instance_id, wire.source.pin.instance_id)
            for wire in built.structure.wires
            if hasattr(wire.source, "pin")
            and wire.destination.pin.signal_id in ("idat", "s_axis_weights_tdata")
        }
        producer = CYCLIC_INSTANCE if case == "cyclic" else None
        assert ("u_weight_stream_fifo", producer) in destinations
        assert ("u_compute", "u_weight_stream_fifo") in destinations
        depth = dict(built.structure.instances[-1].requirements.parameters)["DEPTH"]
        assert depth == 32


def test_fifo_depth_is_owned_by_the_stream_and_only_demanded_when_selected():
    point = configured(base(), "external")
    records = {item.key: item for item in inspection.decisions(point)}
    assert records[FIFO_DEPTH].scope == "weight_stream.transport.fifo.buffer"
    assert isinstance(point.field(handle(point, FIFO_DEPTH)).query(), Inapplicable)
    undecided = buffered(point)
    assert isinstance(undecided.structure.query(), Unresolved)
    with pytest.raises(ConfigurationError):
        buffered(point, depth=1)
    deep = buffered(point, depth=64)
    saved = selections.capture(deep)
    assert FIFO_DEPTH in saved.keys
    assert selections.restore(base(), saved).instance.structure() == deep.structure()
    # Removing the FIFO requires clearing its stale depth and memory style atomically.
    stale = deep.try_with_choices(deep.field(transport(deep)).change("direct"))
    assert not stale.accepted
    direct = deep.with_choices(
        deep.field(transport(deep)).change("direct"),
        deep.field(handle(deep, FIFO_DEPTH)).clear(),
        deep.field(handle(deep, FIFO_RAM_STYLE)).clear(),
    )
    assert direct.structure() == point.structure()


def test_adapter_places_the_weight_fifo_on_request():
    built = mvau_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert [item.instance_id for item in built.structure.instances][-1] == "u_weight_stream_fifo"
