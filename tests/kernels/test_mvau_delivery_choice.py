# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU weight delivery as a heterogeneous structural choice of two families.

External and cyclic delivery share typed assembly and requirements exports but
keep different ports, initialization facts and local choices. These tests cover
laziness, optional family facts, persistence, atomic switching and diagnostics.
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
    compile_space,
    codecs,
    inspection,
    selections,
)
from finn.core.space.errors import RequestError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import ModuleBuildRequirements, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.mvau import (
    CyclicWeights,
    ExternalWeights,
    MVAU,
    WeightDelivery,
    mvau_assembly,
)
from finn.kernels.resources import resource_root, template_root
from finn.kernels.target import DspBlock

ROOT = Path(__file__).resolve().parents[2]
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
    return MVAU(**{**FACTS, **facts})


def selector(point):
    (choice,) = inspection.choices(point)
    assert choice.key == "implementation" and choice.selector is not None
    return choice.selector


def rom_style(point):
    return inspection.decision_handle(
        point.implementation.alternative("cyclic"), CyclicWeights.rom_style
    )


def configured(point, case, *, style=None, pe=2, simd=2):
    changes = [
        point.field(selector(point)).change(case),
        point.compute.field(DotpAxiKernel.compute_pumping).change(False),
    ]
    if style is not None:
        changes.append(point.field(rom_style(point)).change(style))
    return point.with_choices(*changes, pe=pe, simd=simd)


def keys(result):
    return {finding.code for finding in result.findings}


def owners(result):
    return {finding.owner for finding in result.findings}


def test_families_share_typed_exports_but_keep_their_own_ports_and_components():
    external = configured(base(), "external")
    cyclic = configured(base(weights=WEIGHTS), "cyclic", style="block")
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, ["u_replay", "u_compute"]),
        (cyclic, {"in0_V", "out0_V"}, ["u_replay", "u_compute", "u_weights"]),
    ):
        built = point.assembly()
        requirements = point.build_requirements()
        assert isinstance(requirements, ModuleBuildRequirements)
        assert requirements == built.requirements
        assert {p.name for p in built.structure.top_abi.ports if isinstance(p, Bus)} == ports
        assert [item.instance_id for item in built.structure.instances] == instances
    assert external.assembly().weight_delivery is WeightDelivery.EXTERNAL
    assert external.assembly().initializer == ()
    assert cyclic.assembly().initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
    rom = dict(cyclic.assembly().structure.instances[2].requirements.parameters)
    assert rom["ROM_STYLE"] == '"block"'
    assert external.build_requirements().implementation_id != (
        cyclic.build_requirements().implementation_id
    )


def test_the_inactive_family_is_never_demanded():
    point = configured(base(), "external")
    evidence = inspection.explain(point, MVAU.assembly)
    visited = {node.declaration.key for node in evidence.nodes}
    assert any(key.startswith("implementation.external.") for key in visited)
    assert not any(key.startswith("implementation.cyclic.") for key in visited)
    assert "weights" not in visited
    inactive = point.implementation.alternative("cyclic")
    assert isinstance(inactive.query(CyclicWeights.image), Inapplicable)
    assert isinstance(inactive.field(CyclicWeights.rom_style).query(), Inapplicable)


def test_case_local_choices_are_owned_by_their_family():
    records = {item.key: item for item in inspection.decisions(base())}
    assert set(records) == {
        "pe",
        "simd",
        "compute.compute_pumping",
        "implementation",
        "implementation.cyclic.rom_style",
    }
    assert records["implementation"].selector
    assert records["implementation"].cases == ("external", "cyclic")
    local = records["implementation.cyclic.rom_style"]
    assert local.scope == "implementation.cyclic" and not local.selector
    cases = {case.name: case.space_type for case in inspection.choices(base())[0].cases}
    assert cases == {"external": ExternalWeights, "cyclic": CyclicWeights}
    # Applicability of a case-local choice waits for the selector, and names it.
    unselected = base().implementation.alternative("cyclic").field(CyclicWeights.rom_style)
    pending = unselected.candidates()
    assert isinstance(pending, Unresolved) and owners(pending) == {"implementation"}
    selected = base().implementation.select("cyclic").alternative("cyclic")
    assert selected.field(CyclicWeights.rom_style).candidates() == Available(
        ("auto", "distributed", "block")
    )


def test_missing_cyclic_weights_leave_only_the_selected_family_unresolved():
    external = configured(base(), "external")
    assert isinstance(external.assembly.query(), Available)
    cyclic = configured(base(), "cyclic", style="distributed")
    family = cyclic.implementation.alternative("cyclic")
    assert family.rom_style == "distributed"
    assert isinstance(family.query(CyclicWeights.image), Unresolved)
    assessment = cyclic.assembly.inspect()
    assert isinstance(assessment.accepted_result, Unresolved)
    assert assessment.constraints.verdict is True
    # The compute product and folding do not wait for the family's optional fact.
    assert isinstance(cyclic.compute.build_requirements.query(), Available)
    evidence = inspection.explain(cyclic, MVAU.assembly)
    omitted = [node for node in evidence.nodes if node.input_presence == "omitted"]
    assert [node.declaration.key for node in omitted] == ["weights"]


def test_cyclic_family_needs_its_own_rom_choice_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS), "cyclic")
    assert isinstance(uncommitted.assembly.query(), Unresolved)
    bad = configured(base(weights=((4,) * 4,) * 4), "cyclic", style="auto")
    refused = bad.assembly.query()
    assert isinstance(refused, Rejected)
    assert keys(refused) == {"mvau-weights"}
    assert owners(refused) == {"implementation.cyclic.image"}
    # A shape error is refused the same way, and never demanded by external delivery.
    wrong = configured(base(weights=((0,),)), "cyclic", style="auto").assembly.query()
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message
    assert isinstance(configured(base(weights=((0,),)), "external").assembly.query(), Available)


def test_known_refusals_remain_visible_while_the_family_is_unselected():
    point = base(weights=WEIGHTS).with_choices(pe=4, simd=2)
    point = point.compute.with_choices(compute_pumping=True).root
    assessment = point.assembly.inspect()
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
        compile_space(MVAU),
        family="finn.mvau",
        version=1,
        bindings=(
            codec_for(selector(point), STRING),
            codec_for(rom_style(point), STRING),
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
    )
    document = codecs.encode(saved, schema(point))
    decoded = codecs.decode(document, schema(point))
    assert decoded == saved
    fresh = base(weights=WEIGHTS)
    replayed = selections.restore(fresh, decoded)
    assert replayed.accepted
    assert replayed.instance.assembly() == point.assembly()
    # Replay under different supplied facts: the same choices, a new image.
    other = selections.restore(base(weights=tuple(row[::-1] for row in WEIGHTS)), decoded)
    assert other.accepted
    assert other.instance.assembly().initializer != point.assembly().initializer
    # Without the family's optional fact the choices replay but stay unresolved.
    unresolved = selections.restore(base(), decoded)
    assert unresolved.accepted
    assert isinstance(unresolved.instance.assembly.query(), Unresolved)
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
    assert cyclic.assembly().weight_delivery is WeightDelivery.CYCLIC
    switched = cyclic.with_choices(
        cyclic.field(selector(cyclic)).change("external"),
        cyclic.field(rom_style(cyclic)).clear(),
    )
    assert switched.assembly().weight_delivery is WeightDelivery.EXTERNAL
    assert selections.capture(switched).keys == (
        "compute.compute_pumping",
        "implementation",
        "pe",
        "simd",
    )
    # Switching back commits the case-local choice in the same batch.
    back = switched.with_choices(
        switched.field(selector(switched)).change("cyclic"),
        switched.field(rom_style(switched)).change("distributed"),
    )
    assert back.assembly().weight_delivery is WeightDelivery.CYCLIC
    rom = dict(back.assembly().structure.instances[2].requirements.parameters)
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
    assert adapted == point.assembly()


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
    )["u_weights"]


def test_adapter_rejects_an_unknown_rom_style():
    with pytest.raises(ValueError, match="domain"):
        mvau_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.CYCLIC,
            weights=WEIGHTS,
            rom_style="ultra",
        )
