from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops import selected
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.legacy import inspect_legacy_selection
from finn.dataflow.ops.native import NativeAttribute, SCHEMA_VERSION_ATTRIBUTE, read_attributes
from finn.dataflow.ops.persistence import (
    PublishedSelection,
    apply_selection_migration,
    plan_selection_migration,
)
from finn.dataflow.ops.model_effects import ModelReadKind, ModelReadSet
from finn.dataflow.ops.reconstruction import bind_operations, bind_sources_only
from finn.dataflow.ops.selected import SelectedGraphError


FIXTURES = Path(__file__).parent / "fixtures" / "legacy_selected"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())


@dataclass(frozen=True)
class LegacyBuild:
    synth_clk_period_ns: float = 5.5
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool | None = True


def _load(file: str):
    model = ModelWrapper(str(FIXTURES / file))
    (operation,) = bind_sources_only(model, LegacyBuild())
    legacy = inspect_legacy_selection(operation)
    return model, operation, legacy


@pytest.mark.parametrize("record", MANIFEST["fixtures"], ids=lambda item: item["file"])
def test_ordinary_binding_strictly_refuses_each_legacy_fixture(record) -> None:
    model = ModelWrapper(str(FIXTURES / record["file"]))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="schema version"):
        bind_operations(model, LegacyBuild())
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("record", MANIFEST["fixtures"], ids=lambda item: item["file"])
def test_frozen_legacy_selections_migrate_atomically_and_reload(record) -> None:
    path = FIXTURES / record["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    model, operation, legacy = _load(record["file"])
    assignments = {item.path: item.value for item in legacy.choices}
    plan = plan_selection_migration(operation, legacy, assignments)
    migrated = apply_selection_migration(model, plan)

    expected_schema = 3 if record["operation"] == "ActivationReplayOp" else 5
    assert read_attributes(model.graph.node[0])[SCHEMA_VERSION_ATTRIBUTE].value == expected_schema
    assert dict(migrated.recorded()) == assignments
    (rebound,) = bind_operations(model, LegacyBuild())
    assert dict(rebound.recorded()) == assignments


def test_legacy_migration_and_selected_publication_share_one_transaction() -> None:
    model, operation, legacy = _load("mvau-schema3-dot-product.onnx")
    assignments = {item.path: item.value for item in legacy.choices}
    plan = plan_selection_migration(
        operation,
        legacy,
        assignments,
        publish_selected=True,
    )
    published = apply_selection_migration(model, plan)
    assert isinstance(published, PublishedSelection)
    assert published.selected.declaration.version == 2
    assert dict(published.operation.recorded()) == assignments


def test_partial_legacy_selection_requires_explicit_missing_choices_for_publication() -> None:
    model, operation, legacy = _load("replay-schema2-partial.onnx")
    before = model.model.SerializeToString(deterministic=True)
    assignments = {item.path: item.value for item in legacy.choices}
    with pytest.raises(DataflowOpError, match="selected publication is unavailable"):
        plan_selection_migration(
            operation,
            legacy,
            assignments,
            publish_selected=True,
        )
    assert model.model.SerializeToString(deterministic=True) == before

    assignments["design.simd"] = 3
    plan = plan_selection_migration(
        operation,
        legacy,
        assignments,
        publish_selected=True,
    )
    assert isinstance(apply_selection_migration(model, plan), PublishedSelection)


def test_batch_legacy_selection_refuses_selected_publication_without_writes() -> None:
    model, operation, legacy = _load("mvau-schema3-batch-interleaved.onnx")
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="selected publication is unavailable"):
        plan_selection_migration(
            operation,
            legacy,
            {item.path: item.value for item in legacy.choices},
            publish_selected=True,
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_invalid_or_stale_migration_never_writes() -> None:
    model, operation, legacy = _load("replay-schema2-full.onnx")
    before = model.model.SerializeToString(deterministic=True)
    assignments = {item.path: item.value for item in legacy.choices}
    with pytest.raises(DataflowOpError, match="invalid for current source"):
        plan_selection_migration(operation, legacy, {**assignments, "design.simd": 4})
    assert model.model.SerializeToString(deterministic=True) == before

    plan = plan_selection_migration(operation, legacy, assignments)
    node = model.graph.node[0]
    attribute = next(item for item in node.attribute if item.name == "design__simd")
    attribute.i = 6
    stale = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="legacy source changed|Decision attribute"):
        apply_selection_migration(model, plan)
    assert model.model.SerializeToString(deterministic=True) == stale


def test_migration_accepts_an_explicit_valid_changed_assignment() -> None:
    model, operation, legacy = _load("replay-schema2-full.onnx")
    assignments = {item.path: item.value for item in legacy.choices}
    assignments["design.simd"] = 2
    plan = plan_selection_migration(operation, legacy, assignments)
    migrated = apply_selection_migration(model, plan)
    assert migrated.recorded()["design.simd"] == 2


@pytest.mark.parametrize("tamper", ("foreign_owner", "unrelated_remove", "missing_read"))
def test_migration_reconstructs_its_exact_source_surface_at_apply(tamper: str) -> None:
    model, operation, legacy = _load("replay-schema2-full.onnx")
    if tamper == "foreign_owner":
        model.graph.node.append(
            helper.make_node(
                "Identity",
                ["activation"],
                ["foreign"],
                name="foreign",
                dataflow_scope_id="foreign_scope",
            )
        )
    elif tamper == "unrelated_remove":
        model.graph.node[0].attribute.append(helper.make_attribute("review_unrelated", 1))
    (operation,) = bind_sources_only(model, LegacyBuild())
    legacy = inspect_legacy_selection(operation)
    plan = plan_selection_migration(
        operation,
        legacy,
        {item.path: item.value for item in legacy.choices},
    )
    if tamper == "foreign_owner":
        changed = (
            *plan.source_effects.set_attributes,
            ("foreign_scope", "review_foreign", NativeAttribute("i", 1)),
        )
        plan = replace(plan, source_effects=replace(plan.source_effects, set_attributes=changed))
    elif tamper == "unrelated_remove":
        changed = (
            *plan.source_effects.remove_attributes,
            ("legacy_replay_scope", "review_unrelated"),
        )
        plan = replace(plan, source_effects=replace(plan.source_effects, remove_attributes=changed))
    else:
        reads = tuple(
            item
            for item in plan.source_effects.read_set.expectations
            if not (item.kind is ModelReadKind.ATTRIBUTE and item.field == "design__simd")
        )
        plan = replace(
            plan,
            source_effects=replace(plan.source_effects, read_set=ModelReadSet(reads)),
        )
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="effects differ"):
        apply_selection_migration(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_migration_hydration_and_selected_decode_failures_roll_back() -> None:
    model, operation, legacy = _load("replay-schema2-full.onnx")
    plan = plan_selection_migration(
        operation,
        legacy,
        {item.path: item.value for item in legacy.choices},
    )
    before = model.model.SerializeToString(deterministic=True)
    with patch("finn.dataflow.ops.base.hydrate", side_effect=RuntimeError("hydrate failed")):
        with pytest.raises(RuntimeError, match="hydrate failed"):
            apply_selection_migration(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_migration_rejects_written_physical_choice_different_from_plan() -> None:
    model, operation, legacy = _load("mvau-schema3-dot-product.onnx")
    plan = plan_selection_migration(
        operation,
        legacy,
        {item.path: item.value for item in legacy.choices},
    )
    changed = tuple(
        (owner, name, NativeAttribute("i", 1))
        if name == "design__dot_product__compute__dotp_axi__compute_pumping"
        else (owner, name, value)
        for owner, name, value in plan.source_effects.set_attributes
    )
    plan = replace(plan, source_effects=replace(plan.source_effects, set_attributes=changed))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="effects differ|written native choices differ"):
        apply_selection_migration(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before

    model, operation, legacy = _load("mvau-schema3-dot-product.onnx")
    plan = plan_selection_migration(
        operation,
        legacy,
        {item.path: item.value for item in legacy.choices},
        publish_selected=True,
    )
    before = model.model.SerializeToString(deterministic=True)
    original = selected.decode_selected_graph
    calls = 0

    def fail_after_preflight(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SelectedGraphError("selected.test.finish", "selected", "decode failed")
        return original(*args, **kwargs)

    with patch.object(selected, "decode_selected_graph", fail_after_preflight):
        with pytest.raises(SelectedGraphError, match="decode failed"):
            apply_selection_migration(model, plan)
    assert model.model.SerializeToString(deterministic=True) == before


def test_all_five_migrations_run_in_fresh_processes() -> None:
    script = f"""
from dataclasses import dataclass
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops.legacy import inspect_legacy_selection
from finn.dataflow.ops.persistence import plan_selection_migration, apply_selection_migration
from finn.dataflow.ops.reconstruction import bind_sources_only

@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 5.5
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool | None = True

for path in {tuple(str(FIXTURES / item["file"]) for item in MANIFEST["fixtures"])!r}:
    model = ModelWrapper(path)
    operation, = bind_sources_only(model, Build())
    legacy = inspect_legacy_selection(operation)
    plan = plan_selection_migration(
        operation, legacy, {{item.path: item.value for item in legacy.choices}}
    )
    apply_selection_migration(model, plan)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
