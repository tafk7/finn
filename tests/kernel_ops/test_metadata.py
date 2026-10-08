# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's typed graph metadata (qonnx's ``qonnx.core.metadata``): the build target in
``finn.platform``, as ``finn.platform.resolve_target`` resolves it, and a partition's
boundary facts in ``finn.partition``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from onnx import helper
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import (
    PLATFORM,
    PLATFORM_FIELDS,
    PLATFORM_KEYS,
    KernelOpError,
    read_target,
    write_target,
)
from finn.kernels.target import Target
from finn.platform import resolve_target
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION,
    PARTITION_INPUTS,
    PARTITION_OUTPUTS,
)
from finn.transformation.kernels import ToKernelOps
from kernel_ops.models import TARGET, chain_source


def entries(model: ModelWrapper) -> dict[str, str]:
    return {item.key: item.value for item in model.graph.metadata_props}


def holder(body_name: str = "body") -> ModelWrapper:
    """A model with one node holding a (here empty) subgraph body."""
    body = helper.make_graph([], body_name, [], [])
    node = helper.make_node("Holder", [], [], name="holder", body=body)
    return ModelWrapper(qonnx_make_model(helper.make_graph([node], "outer", [], [])))


def body_of(model: ModelWrapper) -> ModelWrapper:
    wrapper: ModelWrapper = model.make_subgraph_modelwrapper(model.graph.node[0].attribute[0].g)
    return wrapper


#: Ultra96 in the Zynq shell: a target with a board and its part's resources.
ULTRA96 = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
#: VCK190's part on the ip shell: no board, its resources not known.
VERSAL = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=4.0)


def test_the_target_is_stated_typed_every_key_and_read_back() -> None:
    model = holder()
    write_target(model, TARGET)
    assert read_target(model) == TARGET
    assert set(PLATFORM_KEYS) == {"part", "shell", "board", *PLATFORM_FIELDS}
    stated = entries(model)
    assert stated["finn.platform/@version"] == "1"
    assert stated["finn.platform/part"] == "xczu3eg-sbva484-1-e"
    assert stated["finn.platform/shell"] == "ip"
    assert stated["finn.platform/board"] == "null"
    assert stated["finn.platform/period_ns"] == "5.0"
    assert stated["finn.platform/dsp"] == "DSP48E2"
    assert stated["finn.platform/fabric"] == "ULTRASCALE"
    assert stated["finn.platform/uram"] == "false"
    assert stated["finn.platform/resources"] == (
        '{"bram18":432,"dsp":360,"ff":141120,"lut":70560,"uram":0}'
    )
    # A body opened through its parent reads the parent's; extracted, it carries a copy.
    body = body_of(model)
    assert read_target(body) == TARGET
    body.inherit_metadata(PLATFORM)
    assert read_target(ModelWrapper(body.model)) == TARGET


@pytest.mark.parametrize("target", [ULTRA96, VERSAL], ids=["board", "unknown-resources"])
def test_a_target_round_trips_with_its_board_and_resources_or_none(target: Target) -> None:
    model = holder()
    write_target(model, target)
    assert read_target(ModelWrapper(model.model)) == target
    stated = entries(model)
    if target.board is None:
        assert (stated["finn.platform/board"], stated["finn.platform/resources"]) == (
            "null",
            "null",
        )
    else:
        assert stated["finn.platform/board"] == '"Ultra96"'


@pytest.mark.parametrize("missing", ["dsp", "period_ns", "clk2x", "shell", "fabric", "resources"])
def test_a_partial_target_is_refused_naming_what_it_misses(missing: str) -> None:
    model = holder()
    write_target(model, TARGET)
    model.delete(PLATFORM_KEYS[missing])
    with pytest.raises(
        KernelOpError, match=f"states no target \\(finn.platform: {missing} missing"
    ):
        read_target(model)
    with pytest.raises(KernelOpError, match="states no target"):
        read_target(holder())


#: The shape before the registry: capabilities a shell budgets (control_ports,
#: memory_ports) and an AI Engine flag, no shell, board, fabric or resources.
OLD_SHAPE = {
    "@version": "1",
    "part": "xczu3eg-sbva484-1-e",
    "period_ns": "5.0",
    "dsp": "DSP48E2",
    "uram": "false",
    "uram_init": "false",
    "clk2x": "true",
    "control_ports": "1",
    "memory_ports": "0",
    "aie": "false",
}


def test_the_old_shape_is_refused_naming_the_keys_it_states() -> None:
    model = holder()
    for name, text in OLD_SHAPE.items():
        model.set_metadata_prop(f"finn.platform/{name}", text)
    with pytest.raises(KernelOpError) as refused:
        read_target(model)
    assert "stored keys ['aie', 'control_ports', 'memory_ports'] are not declared" in str(
        refused.value
    )
    assert "run ToKernelOps to state it again" in str(refused.value)
    # Without those keys, it is a partial one: the new keys are named.
    for name in ("aie", "control_ports", "memory_ports"):
        model.graph.metadata_props.remove(
            next(item for item in model.graph.metadata_props if item.key == f"finn.platform/{name}")
        )
    with pytest.raises(
        KernelOpError, match="finn.platform: shell, board, fabric, resources missing"
    ):
        read_target(model)


@pytest.mark.parametrize(
    "key, text, expected",
    [
        ("period_ns", "fast", "finn.platform/period_ns: stored 'fast' is not a float"),
        ("resources", '{"lut":1}', "finn.platform/resources: stored '{\"lut\":1}' is not"),
        ("board", '""', "finn.platform/board: stored '\"\"' is not"),
        ("shell", "", "finn.platform/shell: stored '' is not"),
    ],
)
def test_a_malformed_key_is_refused(key: str, text: str, expected: str) -> None:
    model = holder()
    write_target(model, TARGET)
    model.set_metadata_prop(f"finn.platform/{key}", text)
    with pytest.raises(KernelOpError) as refused:
        read_target(model)
    assert expected in str(refused.value)


def test_a_target_that_is_not_one_writes_nothing() -> None:
    model = holder()
    with pytest.raises(KernelOpError, match="states its DSP block"):
        write_target(model, replace(TARGET, platform=replace(TARGET.platform, dsp=None)))
    with pytest.raises(KernelOpError, match="period_ns: cannot store 0.0"):
        write_target(model, replace(TARGET, platform=replace(TARGET.platform, period_ns=0.0)))
    with pytest.raises(KernelOpError, match="shell: cannot store ''"):
        write_target(model, replace(TARGET, shell=""))
    assert entries(model) == {}


def test_conversion_states_the_target_it_is_given() -> None:
    zcu104 = resolve_target(board="ZCU104", period_ns=4.0, shell="pynq")
    assert read_target(chain_source().transform(ToKernelOps(zcu104))) == zcu104


def port(name: str, tensor: str, **facts: Any) -> dict[str, Any]:
    return dict(
        port=name,
        tensor=tensor,
        shape=[1, 784],
        datatype="UINT8",
        lanes=16,
        beats=49,
        element_bits=8,
        tdata=128,
        **facts,
    )


def test_a_partitions_boundary_facts_are_typed_and_its_own() -> None:
    model = holder()
    inputs = [port("s_axis_0", "x")]
    outputs = [{**port("m_axis_0", "y"), "shape": [1, 10], "lanes": 10, "beats": 1, "tdata": 104}]
    model.set(PARTITION_INPUTS, inputs)
    model.set(PARTITION_OUTPUTS, outputs)
    assert model.namespace(PARTITION) == {"inputs": inputs, "outputs": outputs}
    assert body_of(model).namespace(PARTITION) == {}  # not inherited: the partition's own
    for bad in (
        [port("s_axis_0", "x", extra=1)],
        [{**port("s_axis_0", "x"), "lanes": 0}],
        [{**port("s_axis_0", "x"), "shape": [1, True]}],
        [{**port("s_axis_0", "x"), "tensor": ""}],
        port("s_axis_0", "x"),
    ):
        with pytest.raises(MetadataError, match="finn.partition/inputs: cannot store"):
            model.set(PARTITION_INPUTS, bad)
