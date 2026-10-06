# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's typed graph metadata (qonnx's ``qonnx.core.metadata``): the build target in
``finn.platform``, resolved from the capability tables, and a partition's boundary
facts in ``finn.partition``."""

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
from finn.kernels.target import DspBlock, Platform, Target
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION,
    PARTITION_INPUTS,
    PARTITION_OUTPUTS,
)
from finn.transformation.kernels import ToKernelOps, resolve_target
from finn.util.basic import get_dsp_block, part_map
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


def test_the_target_is_stated_typed_every_key_and_read_back() -> None:
    model = holder()
    write_target(model, TARGET)
    assert read_target(model) == TARGET
    assert set(PLATFORM_KEYS) == {"part", *PLATFORM_FIELDS}
    stated = entries(model)
    assert stated["finn.platform/@version"] == "1"
    assert stated["finn.platform/part"] == "xczu3eg-sbva484-1-e"
    assert stated["finn.platform/period_ns"] == "5.0"
    assert stated["finn.platform/dsp"] == "DSP48E2"
    assert stated["finn.platform/uram"] == "false"
    assert stated["finn.platform/control_ports"] == "1"
    # A body opened through its parent reads the parent's; extracted, it carries a copy.
    body = body_of(model)
    assert read_target(body) == TARGET
    body.inherit_metadata(PLATFORM)
    assert read_target(ModelWrapper(body.model)) == TARGET


@pytest.mark.parametrize("missing", ["dsp", "period_ns", "clk2x"])
def test_a_missing_key_is_refused_by_name(missing: str) -> None:
    model = holder()
    write_target(model, TARGET)
    model.delete(PLATFORM_KEYS[missing])
    with pytest.raises(
        KernelOpError, match=f"states no target \\(finn.platform: {missing} missing"
    ):
        read_target(model)
    with pytest.raises(KernelOpError, match="states no target"):
        read_target(holder())


def test_a_malformed_key_is_refused() -> None:
    model = holder()
    write_target(model, TARGET)
    model.set_metadata_prop("finn.platform/period_ns", "fast")
    with pytest.raises(
        KernelOpError, match="finn.platform/period_ns: stored 'fast' is not a float"
    ):
        read_target(model)


def test_a_target_that_is_not_one_writes_nothing() -> None:
    model = holder()
    with pytest.raises(KernelOpError, match="states its DSP block"):
        write_target(model, Target("xczu3eg-sbva484-1-e", replace(TARGET.platform, dsp=None)))
    with pytest.raises(KernelOpError, match="period_ns: cannot store 0.0"):
        write_target(model, Target("xczu3eg-sbva484-1-e", replace(TARGET.platform, period_ns=0.0)))
    assert entries(model) == {}


def test_conversion_states_the_target_it_is_given() -> None:
    zcu104 = resolve_target("xczu7ev-ffvc1156-2-e", 4.0, "vivado_zynq")
    assert read_target(chain_source().transform(ToKernelOps(zcu104))) == zcu104


def test_the_capability_tables() -> None:
    ultra96 = resolve_target("xczu3eg-sbva484-1-e", 5.0)
    assert ultra96.platform == Platform(
        period_ns=5.0,
        dsp=DspBlock.DSP48E2,
        uram=False,
        uram_init=False,
        clk2x=True,
        control_ports=1,
        memory_ports=0,
        aie=False,
    )
    zcu104 = resolve_target("xczu7ev-ffvc1156-2-e", 5.0, "vivado_zynq").platform
    # UltraScale+ has UltraRAM here but ignores its INIT; the Zynq shell drives no 2x clock.
    assert (zcu104.uram, zcu104.uram_init, zcu104.clk2x, zcu104.control_ports) == (
        True,
        False,
        False,
        62,
    )
    vck190 = resolve_target("xcvc1902-vsva2197-2MP-e-S", 5.0).platform
    assert (vck190.dsp, vck190.aie, vck190.uram_init) == (DspBlock.DSP58, True, False)
    assert resolve_target("xcu55c-fsvh2892-2L-e", 3.0, "vitis_alveo").platform.control_ports == 0
    with pytest.raises(ValueError, match="no capability row for part 'xcku040'"):
        resolve_target("xcku040", 5.0)
    with pytest.raises(ValueError, match="no capability row for shell 'pynq'"):
        resolve_target("xczu3eg-sbva484-1-e", 5.0, "pynq")


@pytest.mark.parametrize("board", sorted(part_map))
def test_every_board_has_a_row_and_its_dsp_agrees_with_finns(board: str) -> None:
    platform = resolve_target(part_map[board], 5.0).platform
    assert platform.dsp is DspBlock(get_dsp_block(part_map[board]))


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
