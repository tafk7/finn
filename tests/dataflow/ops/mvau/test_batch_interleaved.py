# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""G7: a third Design, authored against the layer as it stands.

The point of this module is not that batch interleaving works -- the Region
mathematics was retained from U1.5 and is compared against here, not rewritten.
It is that adding a third alternative, with a Decision of its own, needed no new
persistence code, no new Design mechanism, and no change to the operation beyond
one entry in its SubspaceChoice.  Every claim below is a claim about *that*.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Answer, Decided, QualifiedPath, Unresolved
from finn.dataflow.computation import DOT_PRODUCT_COMPUTATION
from finn.dataflow.designs.design import Boundary, Kernels
from finn.dataflow.kernels.dotp_axi import (
    BatchInterleavedDotpAxiKernel,
    DotpAxiKernel,
    DspBlock,
)
from finn.dataflow.model.compiler import _compile_space
from finn.dataflow.model.declarations import Problem, Space, Subspace, ValueSource
from finn.dataflow.model.occurrence import (
    occurrence_commit_paths,
    occurrence_persistable,
)
from finn.dataflow.model.semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import validate_network
from finn.dataflow.ops.association import BoundaryDestination
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp
from finn.dataflow.ops.mvau.computation import (
    AccumulationMode,
    ActivationMode,
    MvauComputationProfile,
)
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.batch_interleaved import (
    DESIGN_INPUTS,
    BatchInterleavedDesign,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.model.declarations import derived
from finn.dataflow.ops.schema import BuildFact, DatatypeAttribute, InputTensor, OutputTensor
from finn.dataflow.ops.state import decode_dataflow_state
from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_mvau_weight_port as baseline_weight_port,
)
from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region as baseline_region,
)
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids

CLOCK_PERIOD_NS = 4.0


@dataclass(frozen=True)
class Build:
    synth_clk_period_ns: float = 4.0
    target_dsp: DspBlock = DspBlock.DSP58


class Problem_(Space):
    """The graph-side facts, exactly as the other Design tests state them."""

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    narrow_weights = Problem(bool)
    target_dsp = Problem(DspBlock)
    clock_period_ns = Problem(float)
    computation_profile = Problem(MvauComputationProfile)


class Placed(Problem_):
    design = Subspace(
        BatchInterleavedDesign,
        name="batch_interleaved",
        **{name: cast("ValueSource[object]", getattr(Problem_, name)) for name in DESIGN_INPUTS},
    )


def _occurrence(
    *,
    repetitions: int = 4,
    matrix_width: int = 8,
    matrix_height: int = 4,
    activation: str = "INT8",
    weight: str = "INT8",
    accumulator: str = "INT32",
    target: DspBlock = DspBlock.DSP58,
    pe: int = 2,
    simd: int = 2,
    interleave: int = 2,
) -> BatchInterleavedDesign:
    root = Placed.start(
        {
            Placed.repetitions: repetitions,
            Placed.matrix_width: matrix_width,
            Placed.matrix_height: matrix_height,
            Placed.activation_type: DataType[activation],
            Placed.weight_type: DataType[weight],
            Placed.accumulator_type: DataType[accumulator],
            Placed.output_type: DataType[accumulator],
            Placed.narrow_weights: False,
            Placed.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.NONE
            ),
            Placed.target_dsp: target,
            Placed.clock_period_ns: CLOCK_PERIOD_NS,
        },
        namespace="mvau",
    )
    design = cast(BatchInterleavedDesign, root.design)
    design = design.assign(BatchInterleavedDesign.pe, pe)
    design = design.assign(BatchInterleavedDesign.simd, simd)
    return cast(
        BatchInterleavedDesign, design.assign(BatchInterleavedDesign.interleave, interleave)
    )


def _network(**kwargs: object) -> Answer[DataflowNetwork]:
    return _occurrence(**kwargs).dataflow.accepted_answer  # type: ignore[arg-type]


#: Interleave against PE, SIMD, the row count and both DSP generations.  Every
#: row satisfies the three divisibility rules; the ones that do not have their
#: own tests below, because "refused" is a different claim from "equal".
MATRIX = (
    ("minimum", 2, 4, 2, 1, 2, 2, DspBlock.DSP58),
    ("pe_and_simd", 4, 8, 4, 2, 2, 2, DspBlock.DSP58),
    ("deep_interleave", 8, 8, 4, 2, 2, 4, DspBlock.DSP58),
    ("full_batch", 4, 8, 4, 2, 2, 4, DspBlock.DSP58),
    ("wide_tile", 4, 8, 8, 4, 2, 2, DspBlock.DSP48E2),
    ("simd_one", 3, 4, 6, 3, 1, 3, DspBlock.DSP48E1),
    ("odd_folds", 6, 12, 6, 3, 4, 3, DspBlock.DSP58),
)


@pytest.mark.parametrize(
    ("label", "repetitions", "matrix_width", "matrix_height", "pe", "simd", "interleave", "target"),
    MATRIX,
)
def test_the_interleaved_region_matches_the_retained_authority(
    label: str,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    pe: int,
    simd: int,
    interleave: int,
    target: DspBlock,
) -> None:
    """The Design places the U1.5 Region, element for element."""

    del label
    design = _occurrence(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        pe=pe,
        simd=simd,
        interleave=interleave,
        target=target,
    )
    expected = baseline_region(
        repetitions,
        matrix_width,
        matrix_height,
        DataType["INT8"],
        DataType["INT8"],
        DataType["INT32"],
        pe,
        simd,
        interleave,
    )
    assert design.region("compute") == Decided(expected)

    network = design.dataflow.accepted_answer
    assert isinstance(network, Decided), network
    assert not validate_network(network.value).issues
    assert network.value.node("compute").region == expected


@pytest.mark.parametrize(
    ("label", "repetitions", "matrix_width", "matrix_height", "pe", "simd", "interleave", "target"),
    MATRIX,
)
def test_the_weight_boundary_is_the_chunked_one(
    label: str,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    pe: int,
    simd: int,
    interleave: int,
    target: DspBlock,
) -> None:
    """Interleaving is visible at the boundary, which is why it is semantic."""

    del label, target
    answer = _network(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        pe=pe,
        simd=simd,
        interleave=interleave,
    )
    assert isinstance(answer, Decided)
    port = answer.value.node("compute").region.input_interface("weight").port
    assert port == baseline_weight_port(
        repetitions, matrix_width, matrix_height, DataType["INT8"], pe, simd, interleave
    )
    assert port.beat_sequence.elements_per_beat == pe * simd // interleave


def test_the_network_places_one_node_with_no_edge_and_three_boundaries() -> None:
    """One Kernel is the composition, not a missing one: see the module docstring."""

    answer = _network()
    assert isinstance(answer, Decided)
    network = answer.value
    assert tuple(node.id for node in network.nodes) == ("compute",)
    assert network.edges == ()
    assert tuple(item.id for item in network.boundaries) == ("activation", "output", "weight")


def test_a_degenerate_interleave_is_refused_rather_than_tied() -> None:
    assessment = _occurrence(interleave=1).dataflow
    assert isinstance(assessment.accepted_answer, Absent)
    codes = {
        finding.code
        for answer in assessment.constraints[0].answers.values()
        if isinstance(answer, Absent)
        for finding in answer.findings
    }
    assert "mvau-interleave-degenerate" in codes


def test_an_interleave_that_does_not_split_the_tile_is_refused() -> None:
    """repetitions=6, interleave=3, PE*SIMD=4: a legal batch, an illegal chunk."""

    assessment = _occurrence(
        repetitions=6, matrix_width=8, matrix_height=4, pe=2, simd=2, interleave=3
    ).dataflow
    assert isinstance(assessment.accepted_answer, Absent)
    codes = {
        finding.code
        for answer in assessment.constraints[0].answers.values()
        if isinstance(answer, Absent)
        for finding in answer.findings
    }
    assert "mvau-interleave-uneven-tile" in codes


def test_a_fused_threshold_node_is_refused_by_the_inherited_constraint() -> None:
    """The base class's constraint is carried forward, not dropped by the override."""

    root = Placed.start(
        {
            Placed.repetitions: 4,
            Placed.matrix_width: 8,
            Placed.matrix_height: 4,
            Placed.activation_type: DataType["INT8"],
            Placed.weight_type: DataType["INT8"],
            Placed.accumulator_type: DataType["INT32"],
            Placed.output_type: DataType["INT32"],
            Placed.narrow_weights: False,
            Placed.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.MULTITHRESHOLD
            ),
            Placed.target_dsp: DspBlock.DSP58,
            Placed.clock_period_ns: CLOCK_PERIOD_NS,
        },
        namespace="mvau",
    )
    design = cast(BatchInterleavedDesign, root.design)
    design = design.assign(BatchInterleavedDesign.pe, 2)
    design = design.assign(BatchInterleavedDesign.simd, 2)
    design = design.assign(BatchInterleavedDesign.interleave, 2)
    codes = {
        finding.code
        for answer in design.dataflow.constraints[0].answers.values()
        if isinstance(answer, Absent)
        for finding in answer.findings
    }
    assert "mvau-design-fuses-no-activation" in codes


def test_the_build_unit_is_honestly_unavailable() -> None:
    """A resolved Region and no RTL: the two are said separately, as they are."""

    design = _occurrence()
    assert isinstance(design.dataflow.accepted_answer, Decided)
    kernel = design.kernel("compute")
    assert isinstance(kernel, Decided)
    built = kernel.value.assign(DotpAxiKernel.compute_pumping, False).physical.accepted_answer
    assert isinstance(built, Absent)
    assert any(finding.code == "kernel-physically-unsupported" for finding in built.findings)


def test_interleave_is_a_design_decision_and_the_kernel_imports_it() -> None:
    """Region-visible, therefore not the Kernel's to own."""

    root = _compile_space(Problem_, "mvau", problem_namespace="problem.mvau")
    design = _compile_space(
        BatchInterleavedDesign,
        "mvau.batch_interleaved",
        {
            name: cast(Any, root.member(name))
            for name in DESIGN_INPUTS  # the Design's Inputs, bound to the problem
        },
        _allow_problem=False,
    )
    paths = {str(item.path) for item in design.spec.decisions}
    assert paths == {
        "mvau.batch_interleaved.pe",
        "mvau.batch_interleaved.simd",
        "mvau.batch_interleaved.interleave",
        "mvau.batch_interleaved.compute.dotp_axi_batch_interleaved.compute_pumping",
    }


# -- the operation: a third alternative, and nothing else --------------------


def _tensor(name: str, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _mvau_model(*, repetitions: int = 4, matrix_width: int = 8, matrix_height: int = 4):
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
    model.set_initializer("weight", np.zeros((matrix_width, matrix_height), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _unbound(model: ModelWrapper) -> DataflowOp:
    node = next(item for item in model.graph.node if item.name == "mvau0")
    operation = model.get_customop_wrapper(node)
    assert isinstance(operation, DataflowOp)
    return operation


def _interleaved_operation(model: ModelWrapper, *, interleave: int = 2) -> MvauDataflowOp:
    chosen = _unbound(model).bind(model, Build()).design.select("batch_interleaved").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
        (BatchInterleavedDesign.interleave, interleave),
    ):
        chosen = chosen.design.alternative("batch_interleaved").assign(declaration, value).root
    committed = chosen.commit(model, Build())
    assert isinstance(committed, MvauDataflowOp)
    return committed


def test_the_operation_offers_three_alternatives() -> None:
    model = _mvau_model()
    view = _unbound(model).bind(model, Build()).design  # type: ignore[attr-defined]
    assert set(view.alternatives) == {"dot_product", "supplied", "batch_interleaved"}


def test_the_interleave_decision_persists_without_any_new_persistence_code() -> None:
    """Discovered by the walk, named by compiled path, stored by the generic codec."""

    model = _mvau_model()
    operation = _interleaved_operation(model, interleave=2)
    recorded = dict(operation.recorded())
    assert recorded["design.case"] == "batch_interleaved"
    assert recorded["design.batch_interleaved.interleave"] == 2
    assert recorded["design.batch_interleaved.pe"] == 2


def test_the_interleaved_choice_survives_a_save_and_reload(tmp_path: Path) -> None:
    model = _mvau_model()
    operation = _interleaved_operation(model, interleave=2)
    before = operation.network
    assert isinstance(before, Decided)

    path = tmp_path / "interleaved.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _unbound(reloaded).bind(reloaded, Build())

    assert dict(restored.recorded()) == dict(operation.recorded())
    after = restored.network
    assert isinstance(after, Decided)
    assert after.value == before.value


def test_switching_among_three_alternatives_leaves_nothing_behind() -> None:
    """The reachability prune, now over three branches rather than two."""

    model = _mvau_model()
    operation = _interleaved_operation(model, interleave=2)
    assert "design.batch_interleaved.interleave" in dict(operation.recorded())

    switched = _unbound(model).bind(model, Build()).reconstruct()
    switched = switched.design.select("dot_product").root
    final = switched.commit(model, Build())
    recorded = dict(final.recorded())
    assert recorded["design.case"] == "dot_product"
    assert not any(name.startswith("design.batch_interleaved.") for name in recorded)

    back = _unbound(model).bind(model, Build()).reconstruct()
    back = back.design.select("batch_interleaved").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
        (BatchInterleavedDesign.interleave, 4),
    ):
        back = back.design.alternative("batch_interleaved").assign(declaration, value).root
    returned = dict(back.commit(model, Build()).recorded())
    assert returned["design.batch_interleaved.interleave"] == 4
    assert not any(name.startswith("design.dot_product.") for name in returned)


def test_the_association_reports_the_interleaved_weight_boundary() -> None:
    """Where the matrix crosses, at the beat contract interleaving produced."""

    model = _mvau_model()
    operation = _interleaved_operation(model, interleave=2)
    answer = operation.association
    assert isinstance(answer, Decided)
    weight = next(item for item in answer.value.operands if item.operand == "weight")
    assert isinstance(weight.destination, BoundaryDestination)
    assert weight.destination.boundary == "weight"
    assert weight.destination.node_id == "compute"

    # And the boundary it names carries the chunked contract, which is the part
    # interleaving actually changes: PE * SIMD / interleave elements per beat.
    network = operation.network
    assert isinstance(network, Decided)
    boundary = next(item for item in network.value.boundaries if item.id == "weight")
    assert boundary.external_beat_sequence.elements_per_beat == 2


def test_an_unresolved_interleave_leaves_the_network_unresolved() -> None:
    model = _mvau_model()
    chosen = _unbound(model).bind(model, Build()).design.select("batch_interleaved").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
    ):
        chosen = chosen.design.alternative("batch_interleaved").assign(declaration, value).root
    assert isinstance(chosen.network, Unresolved)


# -- an aliased candidate: the id is the slot's, not the class's --------------


class _AliasedDesign(WeightedDotProductDesign):
    """One Kernel class filling two candidate slots under names of its own.

    The case generic persistence has to survive and no production Design
    exercises: the recorded selector value is the *candidate id*, so a document
    that stored the Kernel's id instead would replay into the wrong slot -- or,
    with two slots holding the same class, into an ambiguous one.
    """

    id = "aliased"
    version = "1"

    repetitions = WeightedDotProductDesign.repetitions
    matrix_width = WeightedDotProductDesign.matrix_width
    matrix_height = WeightedDotProductDesign.matrix_height
    activation_type = WeightedDotProductDesign.activation_type
    weight_type = WeightedDotProductDesign.weight_type
    accumulator_type = WeightedDotProductDesign.accumulator_type
    output_type = WeightedDotProductDesign.output_type
    narrow_weights = WeightedDotProductDesign.narrow_weights
    target_dsp = WeightedDotProductDesign.target_dsp
    clock_period_ns = WeightedDotProductDesign.clock_period_ns
    pe = WeightedDotProductDesign.pe
    simd = WeightedDotProductDesign.simd
    interleave = BatchInterleavedDesign.interleave

    compute = Kernels(
        Subspace(
            BatchInterleavedDotpAxiKernel,
            name="first_slot",
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
            interleave=interleave,
        ),
        Subspace(
            BatchInterleavedDotpAxiKernel,
            name="second_slot",
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
            interleave=interleave,
        ),
        computation=DOT_PRODUCT_COMPUTATION,
    )

    activation = Boundary(compute.input("activation"))
    weight = Boundary(compute.input("weight"))
    output = Boundary(compute.output("output"))


class _PlacedAliased(Problem_):
    design = Subspace(
        _AliasedDesign,
        name="aliased",
        **{name: cast("ValueSource[object]", getattr(Problem_, name)) for name in DESIGN_INPUTS},
    )


def _aliased_root() -> Space:
    return _PlacedAliased.start(
        {
            _PlacedAliased.repetitions: 4,
            _PlacedAliased.matrix_width: 8,
            _PlacedAliased.matrix_height: 4,
            _PlacedAliased.activation_type: DataType["INT8"],
            _PlacedAliased.weight_type: DataType["INT8"],
            _PlacedAliased.accumulator_type: DataType["INT32"],
            _PlacedAliased.output_type: DataType["INT32"],
            _PlacedAliased.narrow_weights: False,
            _PlacedAliased.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.NONE
            ),
            _PlacedAliased.target_dsp: DspBlock.DSP58,
            _PlacedAliased.clock_period_ns: CLOCK_PERIOD_NS,
        },
        namespace="root",
    )


def test_an_aliased_candidate_is_persisted_by_its_slot_name() -> None:
    root = _aliased_root()
    choices = {item.path: item for item in occurrence_persistable(root)}
    selector = choices["aliased.compute.kernel"]
    assert selector.selector is True

    replayed = occurrence_commit_paths(
        root,
        {
            QualifiedPath("root.aliased.compute.kernel"): "second_slot",
            QualifiedPath("root.aliased.pe"): 2,
            QualifiedPath("root.aliased.simd"): 2,
            QualifiedPath("root.aliased.interleave"): 2,
        },
    )
    design = cast(_AliasedDesign, replayed.design)  # type: ignore[attr-defined]
    assert design.selected("compute") == Decided("second_slot")
    # Both slots hold the same class, so the class name could not have told the
    # two apart; the slot name is what distinguishes them.
    kernel = design.kernel("compute")
    assert isinstance(kernel, Decided)
    assert isinstance(kernel.value, BatchInterleavedDotpAxiKernel)
    assert isinstance(design.dataflow.accepted_answer, Decided)


def test_the_aliased_slots_are_two_distinct_persistable_subtrees() -> None:
    """A path per slot, so a Decision in one cannot be replayed into the other."""

    paths = {item.path for item in occurrence_persistable(_aliased_root())}
    assert "aliased.compute.first_slot.compute_pumping" in paths
    assert "aliased.compute.second_slot.compute_pumping" in paths


# -- and the same aliasing, persisted through a real node --------------------


class _AliasedOp(DataflowOp):
    """A whole operation over the aliased Design, so persistence is end to end.

    The path-level tests above prove the two slots have distinct persistable
    identities.  That is necessary and not sufficient: what a caller relies on
    is that a *document written to a node* replays into the slot it named, and
    the seam between "the walk found this path" and "the file said this" is
    exactly where an aliased candidate could go wrong without any path test
    noticing.
    """

    family = "test.aliased_mvau"
    family_version = "1"

    activation = InputTensor(index=0)
    weight = InputTensor(index=1)
    output = OutputTensor(index=0)

    accumulator_type = DatatypeAttribute(default="INT32", onnx="accDataType")
    target_dsp = BuildFact(DspBlock, accessor=lambda build: build.target_dsp)
    clock_period_ns = BuildFact(float, accessor=lambda build: float(build.synth_clk_period_ns))

    @derived(int, shape=weight.shape)
    def matrix_width(*, shape: tuple[int, ...]) -> object:
        return shape[0]

    @derived(int, shape=weight.shape)
    def matrix_height(*, shape: tuple[int, ...]) -> object:
        return shape[1]

    @derived(int, shape=activation.shape)
    def repetitions(*, shape: tuple[int, ...]) -> object:
        total = 1
        for extent in shape[:-1]:
            total *= extent
        return total

    @derived(bool, shape=weight.shape)
    def narrow_weights(*, shape: tuple[int, ...]) -> object:
        del shape
        return False

    @derived(MvauComputationProfile, shape=weight.shape)
    def profile(*, shape: tuple[int, ...]) -> object:
        del shape
        return MvauComputationProfile(AccumulationMode.INTEGER, ActivationMode.NONE)

    design = Subspace(
        _AliasedDesign,
        name="aliased",
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation.datatype,
        weight_type=weight.datatype,
        accumulator_type=accumulator_type,
        output_type=accumulator_type,
        narrow_weights=narrow_weights,
        target_dsp=target_dsp,
        clock_period_ns=clock_period_ns,
        computation_profile=profile,
    )

    def selected_dataflow(self) -> Any:
        return cast(Any, self.design).dataflow  # type: ignore[attr-defined]


def _aliased_model() -> Any:
    node = helper.make_node(
        "_AliasedOp",
        ["activation", "weight"],
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="aliased0",
        accDataType="INT32",
    )
    graph = helper.make_graph(
        [node],
        "aliased",
        [_tensor("activation", (4, 8))],
        [_tensor("output", (4, 4))],
        value_info=[_tensor("weight", (8, 4))],
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
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _configure_alias(model: Any, slot: str, *, pumped: bool, fresh: bool = False) -> Any:
    """Select one slot and commit a Decision that lives *inside* it.

    ``fresh`` reconstructs first, which is how a *structural* choice is
    changed: an immutable point does not rebase a committed selector, so
    switching slots means a point that never had the first one.
    """

    operation = _AliasedOp(model.graph.node[0], 1).bind(model, Build())
    if fresh:
        operation = operation.reconstruct()
    design = cast(Any, operation.design)
    chosen = design.compute.select(slot).root
    kernel = cast(Any, chosen.design).kernel("compute")
    assert isinstance(kernel, Decided)
    chosen = kernel.value.assign(DotpAxiKernel.compute_pumping, pumped).root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
        (BatchInterleavedDesign.interleave, 2),
    ):
        chosen = cast(Any, chosen.design).assign(declaration, value).root
    return chosen.commit(model, Build())


@pytest.mark.parametrize("slot", ["first_slot", "second_slot"])
def test_an_aliased_slot_survives_a_real_save_and_reload(slot: str, tmp_path: Path) -> None:
    other = "second_slot" if slot == "first_slot" else "first_slot"

    model = _aliased_model()
    committed = _configure_alias(model, slot, pumped=True)
    recorded = dict(committed.recorded())
    assert recorded["aliased.compute.kernel"] == slot
    assert recorded[f"aliased.compute.{slot}.compute_pumping"] is True

    path = tmp_path / f"{slot}.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _AliasedOp(reloaded.graph.node[0], 1).bind(reloaded, Build())
    returned = dict(restored.recorded())

    assert returned == recorded
    assert returned["aliased.compute.kernel"] == slot
    assert returned[f"aliased.compute.{slot}.compute_pumping"] is True

    # And nothing came back under the alias that was not chosen -- neither in
    # the replayed point nor in the document on the node.
    assert not any(name.startswith(f"aliased.compute.{other}.") for name in returned)
    document = decode_dataflow_state(reloaded.graph.node[0])
    assert document is not None
    assert not any(name.startswith(f"aliased.compute.{other}.") for name in document.assignments)
    assert document.assignments["aliased.compute.kernel"].value == slot


def test_switching_the_alias_leaves_the_other_slots_decision_behind() -> None:
    """The prune is by *slot*, not by Kernel class -- which here are not the same."""

    model = _aliased_model()
    _configure_alias(model, "first_slot", pumped=True)
    committed = _configure_alias(model, "second_slot", pumped=False, fresh=True)

    recorded = dict(committed.recorded())
    assert recorded["aliased.compute.kernel"] == "second_slot"
    assert recorded["aliased.compute.second_slot.compute_pumping"] is False
    assert "aliased.compute.first_slot.compute_pumping" not in recorded
