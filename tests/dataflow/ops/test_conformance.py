# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""L15: both operations, and every MVAU Design, through one shared harness.

The harness is the deliverable, not these cases.  A third operation added later
gets the whole lifecycle checked by writing a ``DataflowOpConformanceCase`` and
nothing else -- which is the claim ``finn.dataflow.conformance`` exists to make
testable rather than assertable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.conformance import (
    ConformanceFailure,
    DataflowOpConformanceCase,
    assert_dataflow_op_conforms,
)
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.supplied_dot_product import (
    SuppliedDotProductDesign,
    WeightSupply,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow._engine import Decided


@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 4.0
    target_dsp: DspBlock = DspBlock.DSP58


def _tensor(name: str, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _mvau_model(*, repetitions: int = 4, matrix_width: int = 8, matrix_height: int = 4) -> Any:
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weight"],
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
    )
    graph = helper.make_graph(
        [node],
        "mvau",
        [_tensor("activation", (repetitions, matrix_width))],
        [_tensor("output", (repetitions, matrix_height))],
        value_info=[_tensor("weight", (matrix_width, matrix_height))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weight", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT32"])
    model.set_initializer("weight", np.ones((matrix_width, matrix_height), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _replay_model(*, repetitions: int = 2, matrix_width: int = 8, folds: int = 4) -> Any:
    node = helper.make_node(
        "ActivationReplayOp",
        ["activation"],
        ["expanded"],
        domain=DATAFLOW_DOMAIN,
        name="replay0",
        neuron_folds=folds,
    )
    graph = helper.make_graph(
        [node],
        "replay",
        [_tensor("activation", (repetitions, matrix_width))],
        [_tensor("expanded", (repetitions * folds, matrix_width))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("expanded", DataType["INT8"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _widen_the_matrix(model: Any) -> None:
    """Change a fact the recorded choices were made against."""

    model.set_tensor_shape("weight", [8, 8])
    model.set_initializer("weight", np.ones((8, 8), dtype=np.float32))


def _widen_the_activation(model: Any) -> None:
    model.set_tensor_shape("activation", [4, 8])


def _configure_dot_product(bound: Any) -> Any:
    chosen = bound.design.select("dot_product").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
    ):
        chosen = chosen.design.alternative("dot_product").assign(declaration, value).root
    kernel = chosen.design.alternative("dot_product").kernel("compute")
    assert isinstance(kernel, Decided)
    return kernel.value.assign(DotpAxiKernel.compute_pumping, False).root


def _configure_supplied(bound: Any) -> Any:
    chosen = bound.design.select("supplied").root
    chosen = (
        chosen.design.alternative("supplied")
        .assign(SuppliedDotProductDesign.weight_supply, WeightSupply.EXTERNAL)
        .root
    )
    chosen = chosen.design.alternative("supplied").compute.select("dotp_axi").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
    ):
        chosen = chosen.design.alternative("supplied").assign(declaration, value).root
    kernel = chosen.design.alternative("supplied").kernel("compute")
    assert isinstance(kernel, Decided)
    return kernel.value.assign(DotpAxiKernel.compute_pumping, False).root


def _configure_batch_interleaved(bound: Any) -> Any:
    chosen = bound.design.select("batch_interleaved").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
        (BatchInterleavedDesign.interleave, 2),
    ):
        chosen = chosen.design.alternative("batch_interleaved").assign(declaration, value).root
    kernel = chosen.design.alternative("batch_interleaved").kernel("compute")
    assert isinstance(kernel, Decided)
    return kernel.value.assign(DotpAxiKernel.compute_pumping, False).root


def _configure_replay(bound: Any) -> Any:
    chosen = bound
    for declaration, value in (
        (ActivationReplayDesign.pe, 2),
        (ActivationReplayDesign.simd, 4),
    ):
        chosen = chosen.design.assign(declaration, value).root
    return chosen


def _mvau_execution_context() -> dict[str, Any]:
    return {
        "activation": np.ones((4, 8), dtype=np.float32),
        "weight": np.ones((8, 4), dtype=np.float32),
        "output": np.zeros((4, 4), dtype=np.float32),
    }


ALTERNATIVES = (
    ("dot_product", _configure_dot_product),
    ("supplied", _configure_supplied),
    ("batch_interleaved", _configure_batch_interleaved),
)


@pytest.mark.parametrize(("label", "configure"), ALTERNATIVES)
def test_every_mvau_design_alternative_conforms(label: str, configure: Any, tmp_path: Path) -> None:
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_mvau_model(),
            node_name="mvau0",
            operation_type=MvauDataflowOp,
            build=Build(),
            configure=configure,
            reload_path=tmp_path / f"{label}.onnx",
            mutate_problem=_widen_the_matrix,
            other_build=Build(synth_clk_period_ns=2.5),
            execution_context=_mvau_execution_context(),
        )
    )
    assert dict(result.committed.recorded())["design.case"] == label
    assert dict(result.restored.recorded()) == dict(result.committed.recorded())


def test_the_replay_operation_conforms(tmp_path: Path) -> None:
    """The second operation, sharing nothing with MVAU but the layer."""

    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_replay_model(),
            node_name="replay0",
            operation_type=ActivationReplayOp,
            build=Build(),
            configure=_configure_replay,
            reload_path=tmp_path / "replay.onnx",
            mutate_problem=_widen_the_activation,
        )
    )
    assert set(result.committed.recorded()) == {"design.pe", "design.simd"}


def test_the_harness_fails_when_a_promise_is_broken(tmp_path: Path) -> None:
    """Otherwise a harness that asserted nothing would pass everything."""

    class _Silent(MvauDataflowOp):
        """An operation whose ``recorded()`` answers on an unbound wrapper."""

        def recorded(self) -> Any:
            return {}

    case = DataflowOpConformanceCase(
        model=_mvau_model(),
        node_name="mvau0",
        operation_type=MvauDataflowOp,
        build=Build(),
        configure=_configure_dot_product,
        reload_path=tmp_path / "broken.onnx",
    )
    node = case.model.graph.node[0]
    silent = _Silent(node)
    silent.attach_model(case.model)

    original = ModelWrapper.get_customop_wrapper

    def _return_silent(self: Any, target: Any) -> Any:
        del target
        return silent

    ModelWrapper.get_customop_wrapper = _return_silent  # type: ignore[method-assign]
    try:
        with pytest.raises(ConformanceFailure, match="recorded"):
            assert_dataflow_op_conforms(case)
    finally:
        ModelWrapper.get_customop_wrapper = original  # type: ignore[method-assign]


def test_the_harness_runs_verification_through_finns_own_path() -> None:
    """Not an assertion about the harness's text: the real analysis pass runs."""

    from finn.analysis.verify_custom_nodes import verify_nodes  # noqa: PLC0415

    model = _mvau_model()
    report = verify_nodes(model)
    assert set(report) == {"MvauDataflowOp"}
    assert report["MvauDataflowOp"] == []

    # The wrapper that pass used is unbound, and stays that way.
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, DataflowOp)
    assert not operation.is_bound


def test_verification_does_not_replay_recorded_choices(tmp_path: Path) -> None:
    """A document made against a different problem must not make a node invalid."""

    model = _mvau_model()
    bound = _unbound_mvau(model).bind(model, Build())
    _configure_dot_product(bound).commit(model, Build())
    model.save(str(tmp_path / "recorded.onnx"))

    stale = ModelWrapper(str(tmp_path / "recorded.onnx"))
    _widen_the_matrix(stale)

    report = __import__(
        "finn.analysis.verify_custom_nodes", fromlist=["verify_nodes"]
    ).verify_nodes(stale)
    assert report["MvauDataflowOp"] == []


def _unbound_mvau(model: Any) -> Any:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return operation


def test_the_case_is_the_only_operation_specific_input() -> None:
    """The harness names no operation, no Design and no Decision."""

    source = Path("src/finn/dataflow/conformance.py").read_text()
    for forbidden in ("Mvau", "mvau", "ActivationReplay", "dot_product", "weight_supply"):
        assert forbidden not in source, forbidden


def test_replacing_the_case_build_does_not_disturb_the_frozen_one(tmp_path: Path) -> None:
    """``other_build`` is genuinely different, so the refusal it checks is real."""

    case = DataflowOpConformanceCase(
        model=_mvau_model(),
        node_name="mvau0",
        operation_type=MvauDataflowOp,
        build=Build(),
        configure=_configure_dot_product,
        reload_path=tmp_path / "builds.onnx",
        other_build=Build(synth_clk_period_ns=2.5),
    )
    assert case.other_build != case.build
    assert replace(case, other_build=None).other_build is None
    assert_dataflow_op_conforms(case)
