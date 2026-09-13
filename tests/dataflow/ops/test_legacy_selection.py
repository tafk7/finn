from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops.legacy import (
    LegacySelectionError,
    inspect_legacy_selection,
)
from finn.dataflow.ops.native import FINGERPRINT_ATTRIBUTE, SCHEMA_VERSION_ATTRIBUTE
from finn.dataflow.ops.reconstruction import bind_sources_only


FIXTURES = Path(__file__).parent / "fixtures" / "legacy_selected"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())


@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 5.5
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool | None = True


def _bytes(model: ModelWrapper) -> bytes:
    return model.model.SerializeToString(deterministic=True)


def _bound(path: Path, build: Build | None = None):
    model = ModelWrapper(str(path))
    operations = bind_sources_only(model, Build() if build is None else build)
    assert len(operations) == 1
    return model, operations[0]


def _replace_attribute(model: ModelWrapper, name: str, value: object) -> None:
    node = model.graph.node[0]
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend((*kept, helper.make_attribute(name, value)))


@pytest.mark.parametrize("record", MANIFEST["fixtures"], ids=lambda item: item["file"])
def test_all_frozen_legacy_files_inspect_in_fresh_process(record) -> None:
    path = FIXTURES / record["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    script = f"""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from qonnx.core.modelwrapper import ModelWrapper
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.reconstruction import bind_sources_only
from finn.dataflow.ops.legacy import inspect_legacy_selection

@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 5.5
    target_dsp: DspBlock = DspBlock.DSP48E2
    runtime_writable_weights: bool = False
    runtime_weight_range_contract: bool | None = True

def value(item):
    return item.value if isinstance(item, Enum) else item

model = ModelWrapper({str(path)!r})
before = model.model.SerializeToString(deterministic=True)
operation, = bind_sources_only(model, Build())
selection = inspect_legacy_selection(operation)
after = model.model.SerializeToString(deterministic=True)
print(json.dumps({{
    'adapter_id': selection.adapter_id,
    'adapter_version': selection.adapter_version,
    'from_schema_version': selection.from_schema_version,
    'to_schema_version': selection.to_schema_version,
    'problem_fingerprint': selection.problem_fingerprint,
    'choices': {{item.path: value(item.value) for item in selection.choices}},
    'owned_attributes': list(selection.owned_attributes),
    'present_owned_attributes': [name for name, _raw in selection.owned_attribute_bytes],
    'owned_attribute_sha256': {{
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in selection.owned_attribute_bytes
    }},
    'model_byte_identical': before == after,
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = json.loads(result.stdout)
    replay = record["operation"] == "ActivationReplayOp"
    assert decoded["adapter_id"] == (
        "finn.dataflow.legacy.activation_replay" if replay else "finn.dataflow.legacy.mvau"
    )
    assert decoded["adapter_version"] == 1
    assert decoded["from_schema_version"] == record["schema_version"]
    assert decoded["to_schema_version"] == (3 if replay else 4)
    assert decoded["problem_fingerprint"] == record["problem_fingerprint"]
    assert decoded["choices"] == record["recorded_choices"]
    assert decoded["model_byte_identical"] is True
    assert SCHEMA_VERSION_ATTRIBUTE in decoded["present_owned_attributes"]
    assert FINGERPRINT_ATTRIBUTE in decoded["present_owned_attributes"]
    assert "dataflow_scope_id" not in decoded["owned_attributes"]
    assert set(decoded["present_owned_attributes"]) == {
        SCHEMA_VERSION_ATTRIBUTE,
        FINGERPRINT_ATTRIBUTE,
        *(path.replace(".", "__") for path in record["recorded_choices"]),
    }


def test_partial_selection_owns_absent_legacy_paths_and_is_immutable() -> None:
    model, operation = _bound(FIXTURES / "replay-schema2-partial.onnx")
    before = _bytes(model)
    selection = inspect_legacy_selection(operation)
    assert tuple(item.path for item in selection.choices) == ("design.pe",)
    assert selection.owned_attributes == (
        SCHEMA_VERSION_ATTRIBUTE,
        FINGERPRINT_ATTRIBUTE,
        "design__pe",
        "design__simd",
    )
    assert tuple(name for name, _raw in selection.owned_attribute_bytes) == (
        SCHEMA_VERSION_ATTRIBUTE,
        FINGERPRINT_ATTRIBUTE,
        "design__pe",
    )
    assert before == _bytes(model)
    with pytest.raises(FrozenInstanceError):
        selection.adapter_version = 2  # type: ignore[misc]


def test_changed_source_and_required_build_facts_refuse_without_writing() -> None:
    replay = ModelWrapper(str(FIXTURES / "replay-schema2-full.onnx"))
    replay.set_tensor_datatype("activation", DataType["INT4"])
    replay_before = _bytes(replay)
    replay_operation = bind_sources_only(replay, Build())[0]
    with pytest.raises(LegacySelectionError, match="different source/build facts"):
        inspect_legacy_selection(replay_operation)
    assert _bytes(replay) == replay_before

    mvau, operation = _bound(
        FIXTURES / "mvau-schema3-dot-product.onnx",
        replace(Build(), target_dsp=DspBlock.DSP58),
    )
    mvau_before = _bytes(mvau)
    with pytest.raises(LegacySelectionError, match="different source/build facts"):
        inspect_legacy_selection(operation)
    assert _bytes(mvau) == mvau_before


def test_unknown_schema_refuses_without_writing() -> None:
    model = ModelWrapper(str(FIXTURES / "replay-schema2-full.onnx"))
    _replace_attribute(model, SCHEMA_VERSION_ATTRIBUTE, 99)
    operation = bind_sources_only(model, Build())[0]
    before = _bytes(model)
    with pytest.raises(LegacySelectionError, match="expects schema 2"):
        inspect_legacy_selection(operation)
    assert _bytes(model) == before


@pytest.mark.parametrize("case", ("wrong_kind", "duplicate", "unknown"))
def test_malformed_or_unknown_owned_attributes_refuse_without_writing(case) -> None:
    model = ModelWrapper(str(FIXTURES / "replay-schema2-full.onnx"))
    node = model.graph.node[0]
    if case == "wrong_kind":
        _replace_attribute(model, "design__pe", "one")
    elif case == "duplicate":
        original = next(item for item in node.attribute if item.name == "design__pe")
        node.attribute.add().CopyFrom(original)
    else:
        node.attribute.append(helper.make_attribute("design__obsolete", 1))
    operation = bind_sources_only(model, Build())[0]
    before = _bytes(model)
    with pytest.raises(LegacySelectionError):
        inspect_legacy_selection(operation)
    assert _bytes(model) == before


def test_owned_attribute_bytes_are_exact_present_protobufs() -> None:
    model, operation = _bound(FIXTURES / "mvau-schema3-dot-product.onnx")
    selection = inspect_legacy_selection(operation)
    actual = {
        item.name: item.SerializeToString(deterministic=True)
        for item in model.graph.node[0].attribute
        if item.name in selection.owned_attributes
    }
    assert dict(selection.owned_attribute_bytes) == actual
    assert "dataflow_scope_id" not in actual
    assert "outputDataType" not in actual
