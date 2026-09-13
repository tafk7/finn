from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from unittest.mock import patch

import numpy as np
import pytest

from dataflow.ops.test_dataflow_op import (
    Build,
    _configure_mvau_point,
    _mvau_model,
    _replay_model,
    _unbound,
)
from dataflow.ops.test_selected_verification import _renamed
from finn.dataflow._engine import Decided
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.model import DataflowNetwork
from finn.dataflow.ops import selected
from finn.dataflow.ops import native
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.mvau.designs.dot_product import WeightSupply
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.designs.design import SelectedGraph
from finn.dataflow.ops.native import NativeAttribute
from finn.dataflow.ops.persistence import (
    apply_graph_effects,
    apply_selected_publication,
    plan_selected_publication,
)
from finn.dataflow.ops.reconstruction import bind_sources_only, rebind_selected_graph
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.selected import SELECTED_METADATA_KEY, SelectedGraphError


@dataclass
class NondefaultBuild:
    synth_clk_period_ns: float = 5.5
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool = True


def _planned_mvau(*, supply: WeightSupply = WeightSupply.EXTERNAL):
    model = _mvau_model()
    build = NondefaultBuild()
    bound = _unbound(model, "mvau0").bind(model, build)
    chosen = _configure_mvau_point(bound, supply=supply, pe=2, simd=2)
    return model, build, chosen, plan_selected_publication(chosen)


def test_publication_plan_is_detached_and_applies_native_and_selected_results() -> None:
    model, build, operation, plan = _planned_mvau(supply=WeightSupply.DECOUPLED)
    expected_context = {
        "target_dsp": DspBlock.DSP48E2,
        "runtime_writable_weights": False,
        "runtime_weight_range_contract": True,
        "clock_period_ns": 5.5,
    }
    assert {item.member_name: item.value for item in plan.context.build_facts} == expected_context
    assert not any(item.key == SELECTED_METADATA_KEY for item in model.graph.metadata_props)
    del build, operation
    gc.collect()

    published = apply_selected_publication(model, plan)
    assert dict(published.operation.recorded()) == {
        item.path: item.value for item in plan.expected_choices
    }
    assert published.selected.snapshot.model_bytes == plan.candidate.model_bytes
    assert published.selected.network.node("memory")
    assert not any(item.key == SELECTED_METADATA_KEY for item in model.graph.metadata_props)


def test_publication_captures_the_native_choice_set_once() -> None:
    model = _mvau_model()
    operation = _configure_mvau_point(
        _unbound(model, "mvau0").bind(model, NondefaultBuild()), pe=2, simd=2
    )
    original = native.occurrence_answer_at
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with patch.object(native, "occurrence_answer_at", counted):
        plan_selected_publication(operation)
    assert calls == len(native.choice_schema(operation))


def test_publication_inference_preserves_stable_ids_for_renamed_recipe_output() -> None:
    model = _mvau_model()
    operation = _configure_mvau_point(
        _unbound(model, "mvau0").bind(model, NondefaultBuild()), pe=2, simd=2
    )
    declaration = DotProductDesign.selected_graph
    assert declaration is not None
    construction = declaration.construction
    original = construction.construct

    def renamed(facts, inputs):
        return _renamed(original(facts, inputs), reorder=False)

    with patch.object(
        DotProductDesign,
        "selected_graph",
        SelectedGraph(replace(construction, construct=renamed)),
    ):
        plan = plan_selected_publication(operation)
    assert plan.candidate.declaration.graph_nodes


def test_publication_derives_missing_output_shape_and_logical_type() -> None:
    model = _mvau_model()
    output = model.graph.output[0]
    output.type.tensor_type.ClearField("shape")
    annotations = [
        item for item in model.graph.quantization_annotation if item.tensor_name != output.name
    ]
    del model.graph.quantization_annotation[:]
    model.graph.quantization_annotation.extend(annotations)
    bound = _unbound(model, "mvau0").bind(model, NondefaultBuild())
    chosen = _configure_mvau_point(bound, pe=2, simd=2)

    published = apply_selected_publication(model, plan_selected_publication(chosen))
    assert model.get_tensor_shape(output.name) == [2, 4]
    assert model.get_tensor_datatype(output.name).name == "INT32"
    assert published.selected.selection_facts.source.operands[-1].shape == (2, 4)


def test_publication_plan_allows_unrelated_edit_and_coherent_input_rename() -> None:
    model, _build, _operation, plan = _planned_mvau()
    model.graph.node[0].doc_string = "unrelated documentation"
    model.rename_tensor("activation", "renamed_activation")
    published = apply_selected_publication(model, plan)
    assert isinstance(published.operation.network, Decided)


def test_publication_stale_source_refuses_without_writes() -> None:
    model, _build, _operation, plan = _planned_mvau()
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="initializer_content|different problem"):
        apply_selected_publication(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_publication_final_decode_failure_rolls_back_source_writes() -> None:
    model = _replay_model()
    bound = _unbound(model, "replay0").bind(model, Build())
    chosen = bound.design.assign(ActivationReplayDesign.pe, 1).root
    chosen = chosen.design.assign(ActivationReplayDesign.simd, 2).root
    plan = plan_selected_publication(chosen)
    before = model.model.SerializeToString(deterministic=True)

    original = selected.decode_selected_graph
    calls = 0

    def fail_after_preflight(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SelectedGraphError("selected.test.finish", "selected", "injected failure")
        return original(*args, **kwargs)

    with patch.object(selected, "decode_selected_graph", fail_after_preflight):
        with pytest.raises(SelectedGraphError, match="injected failure"):
            apply_selected_publication(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_publication_rejects_written_physical_choice_different_from_plan() -> None:
    model, _build, _operation, plan = _planned_mvau()
    changed = tuple(
        (owner, name, NativeAttribute("i", 1))
        if name == "design__dot_product__compute__dotp_axi__compute_pumping"
        else (owner, name, value)
        for owner, name, value in plan.source_effects.set_attributes
    )
    plan = replace(plan, source_effects=replace(plan.source_effects, set_attributes=changed))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="effects differ|written native choices differ"):
        apply_selected_publication(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_native_graph_effects_reject_written_choice_different_from_plan() -> None:
    model, _build, operation, _plan = _planned_mvau()
    effects = operation.graph_effects()
    changed = tuple(
        (owner, name, NativeAttribute("i", 1))
        if name == "design__dot_product__compute__dotp_axi__compute_pumping"
        else (owner, name, value)
        for owner, name, value in effects.model_effects().set_attributes
    )
    model_effects = replace(effects.model_effects(), set_attributes=changed)
    effects = replace(
        effects,
        set_attributes={name: value for _owner, name, value in model_effects.set_attributes},
    )
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="effects differ|written native choices differ"):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before


def test_rebind_accepts_source_rename_and_rejects_changed_choice_or_weight() -> None:
    model, _build, operation, plan = _planned_mvau()
    published = apply_selected_publication(model, plan)
    model.rename_tensor("activation", "renamed_activation")
    renamed = published.operation.rebind(model, NondefaultBuild())
    rebound = rebind_selected_graph(renamed, plan.candidate)
    assert rebound.selection_facts.selection_fingerprint

    changed_choice = _configure_mvau_point(renamed.reconstruct(), pe=2, simd=4)
    with pytest.raises(DataflowOpError, match="logical choices"):
        rebind_selected_graph(changed_choice, plan.candidate)

    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    source_only = bind_sources_only(
        model,
        NondefaultBuild(),
        operations=(_unbound(model, "mvau0"),),
    )[0]
    changed_weight = _configure_mvau_point(source_only, pe=2, simd=2)
    with pytest.raises(DataflowOpError, match="source semantics"):
        rebind_selected_graph(changed_weight, plan.candidate)


def test_rebind_rejects_changed_or_malformed_current_projector() -> None:
    model, _build, _operation, plan = _planned_mvau()
    current = apply_selected_publication(model, plan).operation
    declaration = DotProductDesign.selected_graph
    assert declaration is not None
    construction = declaration.construction

    with patch.object(
        DotProductDesign,
        "selected_graph",
        SelectedGraph(replace(construction, project=lambda _facts: DataflowNetwork((), (), ()))),
    ):
        with pytest.raises(DataflowOpError, match="different artifact Network"):
            rebind_selected_graph(current, plan.candidate)

    node = current.network.value.nodes[0]  # type: ignore[union-attr]
    malformed = DataflowNetwork((node, node), (), ())
    with patch.object(
        DotProductDesign,
        "selected_graph",
        SelectedGraph(replace(construction, project=lambda _facts: malformed)),
    ):
        with pytest.raises(DataflowOpError, match="invalid Network"):
            rebind_selected_graph(current, plan.candidate)
