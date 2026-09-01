# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 gate: one semantic Network, two physical readings of it.

The migration's central claim is that physical Kernel coverage is not
one-to-one with Regions.  Everything up to here could still be read as "one
Region, one Kernel, renamed": the decomposed path binds a replay buffer to the
replay node and a dot product to the compute node, which is exactly the shape
the old provider had.  ``MvuVvuAxiKernel`` is what makes the claim falsifiable
-- it covers *both* Regions and the edge between them, from the same selected
semantics, and elaborates to one component instead of two.

So the assertions that matter are comparative.  The same point, the same
Network, the same Region *values*; different bindings, different elaborations.
Anything that differed on the semantic side would mean the physical choice had
changed the design, which is the one thing selecting a Kernel may never do.

The fused Kernel is placed only here.  It is a forcing case, not a
production-selectable alternative, and the last section of this file is what
holds that line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.authoring import Ref, assemble_specs
from finn.dataflow.design import (
    Absent,
    Answer,
    ConstraintAssessment,
    Decided,
    DependencyKind,
    DesignPoint,
    Engine,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.kernels import (
    BoundRegion,
    Kernel,
    bind_kernel,
    bound_regions,
    check_declared_references,
    declare_kernel,
    kernel_namespace,
)
from finn.dataflow.kernels.kernel import CompiledKernelDeclaration
from finn.dataflow.ops.mvau.semantics import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
)
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.kernels.dotp_axi import (
    DotpAxiKernel,
    covers_numeric_types as dotp_axi_covers_numeric_types,
)
from dataflow.mvau.mvu_vvu_axi_kernel import (
    ACTIVATION_EDGE_ROLE,
    COMPUTE_ROLE,
    REPLAY_ROLE,
    FusedMatrixVectorHardwareInputs,
    MvuVvuAxiKernel,
    covers_numeric_types as fused_covers_numeric_types,
)
from dataflow.rtlsim.composed_mvau_equiv import CONFIGS, Config, declared_parameters
from finn.dataflow.ops.mvau.binding import finnlib_root
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.kernels.numeric import DotProductNumericTypes, RoleVerdict
from finn.dataflow.ops.mvau.problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    NetworkRef,
)
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY
from dataflow.mvau.test_decomposed_op import (  # noqa: F401 - the real operation fixture
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    _committed,
    _context,
    _model,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.region import DataflowRegion, NumericElementType

#: The one repetition count these tests use; the claim is structural.
REPETITIONS = 2


class InfeasiblePoint(RuntimeError):
    """Raised when a test tries to bind hardware to a point the operation refuses."""


INT2 = DataType["INT2"]
INT8 = DataType["INT8"]
INT16 = DataType["INT16"]
INT25 = DataType["INT25"]
INT27 = DataType["INT27"]
INT32 = DataType["INT32"]
UINT8 = DataType["UINT8"]
UINT16 = DataType["UINT16"]
TERNARY = DataType["TERNARY"]
BINARY = DataType["BINARY"]
BIPOLAR = DataType["BIPOLAR"]
FLOAT16 = DataType["FLOAT16"]
OTHER_FLOAT16 = DataType["FLOAT<5,10,7>"]

DOT_PRODUCT_DECLARATION = MVAU_DESIGN_INVENTORY.inventory.declaration(DotProductDesign.id)
DOTP_AXI = DOT_PRODUCT_DECLARATION.placement("compute").candidates[0]
REPLAY_BUFFER = DOT_PRODUCT_DECLARATION.placement("replay").candidates[0]


# -- placement ---------------------------------------------------------------
#
# The fused Kernel is declared here rather than in ``build_decomposed_mvau_kernels``
# on purpose.  Declaring it in the production assembly would put its coverage
# constraints into the operation's feasibility set and its paths into the
# production design space, which is precisely the "not production-selectable"
# line this phase draws.
#
# What it *reads*, however, is production's own: the same Region and
# computation declarations the two separate Kernels name one each, and the
# operation's own ``semantic.mvau.op.network`` property. An earlier version of
# this file derived a private Network from the two Regions instead. It produced
# an equal value, which is exactly what made it a bad fixture -- a change to
# the operation's assembly would have left every test here green while the
# fused Kernel bound against a Network nothing in production builds.

#: The operation's Network property, read exactly as the operation declares it.
#: The semantics are the object-widened ones ``ops/mvau.py`` uses; a mismatch
#: here is caught by ``check_declared_references`` rather than at binding.
OP_NETWORK: Ref[DataflowNetwork] = DOT_PRODUCT_DECLARATION.network


def _fused_declaration() -> CompiledKernelDeclaration:
    declaration, _design = declare_kernel(
        MvuVvuAxiKernel,
        kernel_namespace("test.mvau.fused", MvuVvuAxiKernel.id),
        FusedMatrixVectorHardwareInputs(
            replay_region=MVAU_DESIGN_INVENTORY.dot_product.replay_region,
            replay_computation=MVAU_DESIGN_INVENTORY.dot_product.replay_computation,
            compute_region=MVAU_DESIGN_INVENTORY.dot_product.dot_product_region,
            compute_computation=MVAU_DESIGN_INVENTORY.dot_product.dot_product_computation,
            network=OP_NETWORK,
            matrix_width=MVAU_PROBLEM.matrix_width,
            matrix_height=MVAU_PROBLEM.matrix_height,
            pe=MVAU_DESIGN_INVENTORY.dot_product.pe,
            simd=MVAU_DESIGN_INVENTORY.dot_product.simd,
            activation_element_type=MVAU_PROBLEM.activation_element_type,
            weight_element_type=MVAU_PROBLEM.weight_element_type,
            output_element_type=MVAU_PROBLEM.output_element_type,
            accumulator_element_type=MVAU_PROBLEM.accumulator_element_type,
            narrow_weights=MVAU_EFFECTIVE_NARROW_WEIGHTS,
            target_dsp_block=MVAU_PROBLEM.target_dsp_block,
            target_clock_period_ns=MVAU_PROBLEM.target_clock_period_ns,
        ),
    )
    return declaration


FUSED = _fused_declaration()


class _Placed:
    """One resolved point carrying all three physical Kernels at once.

    All three, deliberately.  Binding the fused Kernel in one design space and
    the decomposed pair in another would compare two designs; the claim is
    about one.
    """

    def __init__(self, engine: Engine, point: DesignPoint) -> None:
        self.engine = engine
        self.point = point

    def _value(self, handle: Ref[object]) -> object:
        answer = self.engine.query_property(self.point, handle.path)
        assert isinstance(answer, Decided), answer
        return answer.value

    @property
    def compute_region(self) -> DataflowRegion:
        return cast(
            DataflowRegion, self._value(MVAU_DESIGN_INVENTORY.dot_product.dot_product_region)
        )

    @property
    def replay_region(self) -> DataflowRegion:
        return cast(DataflowRegion, self._value(MVAU_DESIGN_INVENTORY.dot_product.replay_region))

    @property
    def network(self) -> DataflowNetwork:
        """The operation's own Network -- the value both readings bind against."""

        return cast(DataflowNetwork, self._value(OP_NETWORK))

    def feasibility(self) -> ConstraintAssessment:
        """The operation's own verdict on this point.

        Binding asks a Kernel's coverage and nothing else, which is correct --
        a Kernel has no business re-litigating whether the source was
        expressible. But it means a fixture that hands a point straight to
        ``bind_kernel`` has skipped the gate the production path runs
        first, and can therefore bind hardware to a design the operation would
        have refused outright.

        The set is the one ``MvauDataflowOp`` itself selects on, read off the
        operation class rather than named here. An earlier version of this
        guard used the *compute pool's* set instead. That is a strict subset:
        it omits the other three pools, the hardware coverage constraints, and
        every structural constraint -- so a Network that fails validation, or a
        source association whose leading shape contradicts the repetition
        count, passed the guard and bound cleanly.
        """

        constraint_set = MvauDataflowOp.selection_constraint_set()
        assert constraint_set is not None, "the operation must select on a constraint set"
        return self.engine.evaluate_constraint_set(self.point, constraint_set)

    def semantic_refusals(self) -> set[str]:
        """Why the operation refuses, by constraint name. Diagnostic only.

        Three shapes. ``Decided(False)`` is the flat refusal; a *rejecting*
        ``Absent`` is a ``reject(...)`` -- a refusal carrying a reason, which
        is how every coverage constraint says no; and ``Unresolved`` means the
        constraint could not be evaluated at all, which is not permission to
        proceed.

        A non-rejecting ``Absent`` is deliberately excluded. This set spans
        every pool, so a constraint belonging to a Kernel this point did not
        select is legitimately absent, and treating that as a refusal would
        report the whole unselected inventory as broken.

        ``ConstraintAssessment.refused`` draws the first two of those lines, so
        this does not redraw them.
        """

        return {str(path).rsplit(".", 1)[-1] for path in self.feasibility().refused} | {
            str(path).rsplit(".", 1)[-1]
            for path, answer in self.feasibility().answers.items()
            if isinstance(answer, Unresolved)
        }

    def bind(
        self,
        declaration: CompiledKernelDeclaration,
        regions: dict[str, BoundRegion],
        edges: dict[str, str] | None = None,
    ) -> Answer[Kernel]:
        """Bind, but only once the operation has accepted the point.

        ``verdict is True`` and nothing weaker: ``False`` is a refusal and
        ``None`` means the set could not be decided, which is not the same as
        permission.

        Raised rather than asserted. ``python -O`` strips ``assert``, and a
        guard that silently disappears under an optimisation flag is worse than
        no guard -- it would restore the exact bypass this replaced, invisibly.
        """

        assessment = self.feasibility()
        if assessment.verdict is not True:
            raise InfeasiblePoint(
                f"the operation refuses this point before any hardware sees it "
                f"(verdict={assessment.verdict}): {sorted(self.semantic_refusals())}"
            )
        return self.bind_unchecked(declaration, regions, edges)

    def bind_unchecked(
        self,
        declaration: CompiledKernelDeclaration,
        regions: dict[str, BoundRegion],
        edges: dict[str, str] | None = None,
    ) -> Answer[Kernel]:
        """Bind without the semantic gate, to reach a Kernel's own refusal.

        Needed for the datatype cases, and the reason is worth stating: in
        production those points never get this far. Source admission removes
        them first, because ``some_hardware_covers_the_numeric_types``
        quantifies over the declared inventory and refuses a graph no Kernel
        can multiply.

        The Kernel's own coverage is therefore a second line rather than the
        only one — and it is not redundant. The inventory the operation
        quantifies over holds ``DotpAxiKernel`` alone; the fused Kernel is
        deliberately absent from it. Widen that inventory with a Kernel that
        multiplies floats and these points become admissible, at which point
        this Kernel's own refusal is the only thing standing between them and
        a multiplier that cannot produce them.
        """

        return bind_kernel(self.engine, declaration, self.point, regions, edges)

    def both_roles(self) -> dict[str, BoundRegion]:
        return bound_regions(
            (
                (REPLAY_ROLE, REPLAY_NODE, self.replay_region),
                (COMPUTE_ROLE, DOT_PRODUCT_NODE, self.compute_region),
            )
        )

    def fused(self, edges: dict[str, str] | None = None) -> Kernel:
        answer = self.bind(
            FUSED,
            self.both_roles(),
            {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE} if edges is None else edges,
        )
        assert isinstance(answer, Decided), answer
        return answer.value

    def decomposed(self) -> tuple[Kernel, Kernel]:
        replay = self.bind(
            REPLAY_BUFFER, bound_regions((("replay", REPLAY_NODE, self.replay_region),))
        )
        compute = self.bind(
            DOTP_AXI, bound_regions((("compute", DOT_PRODUCT_NODE, self.compute_region),))
        )
        assert isinstance(replay, Decided), replay
        assert isinstance(compute, Decided), compute
        return replay.value, compute.value


def _source_description(repetitions: int) -> MVAUSourceDescription:
    """The tensor identities the operation projects from a graph.

    Supplied here because the fixture assembles the operation's real
    specification, which declares the field and reads it in its own feasibility
    constraints -- ``source_association_valid`` among them, which compares this
    leading shape against the repetition count.
    """

    return MVAUSourceDescription(
        source_node_id="mvau",
        activation_operand_id="activation",
        weight_operand_id="weights",
        output_operand_id="output",
        leading_shape=(repetitions,),
    )


#: Sentinel for "omit the source description entirely", distinct from ``None``
#: because ``None`` is a value the field could plausibly take.
OMITTED = object()


def _place(
    *,
    source_description: object = None,
    repetitions: int = REPETITIONS,
    activation: NumericElementType = INT8,
    weight: NumericElementType = INT8,
    accumulator: NumericElementType = INT16,
    output: NumericElementType = INT16,
    target: DspBlock | object = DspBlock.DSP58,
    matrix_width: int = 8,
    matrix_height: int = 4,
    pe: int = 2,
    simd: int = 2,
    pumping: bool = False,
    narrow: bool = False,
) -> _Placed:
    engine = Engine()
    # The operation's *real* specification, plus the fused Kernel. Not a
    # reassembly of its parts: this is what gives the fixture the operation's
    # own Network property, its feasibility constraint set, and every semantic
    # condition the production path applies before hardware is consulted.
    specification = assemble_specs((MVAU_DATAFLOW_OP_SPEC, FUSED.spec))
    # The same authoring check the production assembly runs: a fused Kernel
    # reading a Region the space does not declare is exactly as broken as a
    # separate one doing so, and covering two of them is no excuse.
    check_declared_references(specification, (DOTP_AXI, REPLAY_BUFFER, FUSED))
    space = engine.validate(specification)
    problem: dict[QualifiedPath, object] = {}
    if source_description is not OMITTED:
        problem[MVAUProblemPaths.SOURCE_DESCRIPTION] = (
            _source_description(repetitions) if source_description is None else source_description
        )
    # Both target facts are optional in the schema, which is what makes leaving
    # one out a way to reach an *unresolved* constraint rather than a refused
    # one. The source description is required, so omitting it is rejected by
    # ``engine.start`` before any constraint is evaluated.
    if target is not OMITTED:
        problem[MVAUProblemPaths.TARGET_DSP_BLOCK] = target
    point = engine.start(
        space,
        {
            **problem,
            MVAUProblemPaths.REPETITIONS: repetitions,
            MVAUProblemPaths.MATRIX_WIDTH: matrix_width,
            MVAUProblemPaths.MATRIX_HEIGHT: matrix_height,
            MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: activation,
            MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: weight,
            MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: accumulator,
            MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: output,
            MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
            MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
            MVAUProblemPaths.RUNTIME_WRITABLE: False,
            MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM: narrow,
            MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 4.0,
        },
    )
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    point = engine.commit_assignments(
        point,
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
            MVAU_DESIGN_INVENTORY.dot_product.pe.path: pe,
            MVAU_DESIGN_INVENTORY.dot_product.simd.path: simd,
            MVAU_DESIGN_INVENTORY.compute_pumping.path: pumping,
            FUSED.spec.decisions[0].path: pumping,
        },
    ).point
    return _Placed(engine, point)


# -- the gate ----------------------------------------------------------------


def test_one_network_binds_either_as_two_kernels_or_as_one_fused_kernel() -> None:
    """The Phase 4 claim, stated as one assertion pair.

    Same point, same Network. Two bindings covering one node each and absorbing
    nothing, or one binding covering both nodes and absorbing the edge between
    them.
    """

    placed = _place()
    replay, compute = placed.decomposed()
    fused = placed.fused()

    assert sorted(replay.node_ids + compute.node_ids) == sorted(fused.node_ids)
    assert replay.edge_ids == () and compute.edge_ids == ()
    assert fused.edge_ids == (ACTIVATION_EDGE,)


def test_the_two_readings_elaborate_differently() -> None:
    """One component against two, from identical semantics.

    The decomposed pair additionally needs a generated top to wire them, which
    is not counted here: even before that, two is not one.
    """

    placed = _place()
    replay, compute = placed.decomposed()

    assert len(replay.components()) == 1
    assert len(compute.components()) == 1
    assert len(placed.fused().components()) == 1
    assert placed.fused().components()[0].module == "finn-rtllib.mvu.mvu_vvu_axi"


def test_the_semantic_values_are_identical_across_both_bindings() -> None:
    """Not "equivalent" -- the same values.

    A physical choice that perturbed a Region would mean the Kernel had
    re-decided the logical dataflow, so this compares the Region values the two
    readings cover rather than any summary of them.
    """

    placed = _place()
    replay, compute = placed.decomposed()
    fused = placed.fused()
    covered = {item.role: item.region for item in fused.regions.values()}

    assert covered[REPLAY_ROLE] == replay.regions[REPLAY_ROLE].region
    assert covered[COMPUTE_ROLE] == compute.regions[COMPUTE_ROLE].region
    assert covered[REPLAY_ROLE] == placed.replay_region
    assert covered[COMPUTE_ROLE] == placed.compute_region


def test_the_absorbed_edge_is_the_one_the_network_declares() -> None:
    placed = _place()
    edge = next(item for item in placed.network.edges if item.id == ACTIVATION_EDGE)

    assert edge.source.node_id == REPLAY_NODE
    assert tuple(item.endpoint.node_id for item in edge.sinks) == (DOT_PRODUCT_NODE,)
    assert placed.fused().edges == {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}


def test_the_fused_binding_leaves_no_node_of_the_network_uncovered() -> None:
    """One Kernel realizes the whole two-node Network, with nothing left over.

    Stated over the Network's own node list rather than over the roles the
    Kernel declares, so it is a claim about the semantics being fully realized
    rather than about the Kernel being internally consistent.
    """

    placed = _place()
    assert {node.id for node in placed.network.nodes} == set(placed.fused().node_ids)
    assert {edge.id for edge in placed.network.edges} == set(placed.fused().edge_ids)


def test_the_fused_kernel_is_not_an_alternative_to_either_separate_one() -> None:
    """Their coverage signatures differ, so a pool would refuse them together.

    This is the mechanical reason the fused Kernel cannot simply be added
    beside ``DotpAxiKernel``: they are not alternatives for one another. One
    covers a role the other does not, which is a different question from
    whether either is a good idea.
    """

    fused_signature = FUSED.coverage.signature
    dotp_signature = DOTP_AXI.coverage.signature

    assert fused_signature != dotp_signature
    assert any(
        "covered by only one of them" in reason
        for reason in fused_signature.difference(dotp_signature)
    )
    assert set(FUSED.coverage.region_roles) == {REPLAY_ROLE, COMPUTE_ROLE}
    assert FUSED.coverage.edge_roles == (ACTIVATION_EDGE_ROLE,)
    assert DOTP_AXI.coverage.edge_roles == ()


def test_a_fused_binding_checks_the_edge_rather_than_trusting_the_id() -> None:
    placed = _place()
    absent = placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: "no_such_edge"})

    assert isinstance(absent, Unresolved)
    assert any(item.code == "hardware-covered-edge-absent" for item in absent.findings)


def test_a_fused_binding_rejects_the_edge_running_the_wrong_way() -> None:
    """Swap the roles and the real edge no longer connects them."""

    placed = _place()
    swapped = bound_regions(
        (
            (REPLAY_ROLE, DOT_PRODUCT_NODE, placed.compute_region),
            (COMPUTE_ROLE, REPLAY_NODE, placed.replay_region),
        )
    )
    answer = placed.bind(FUSED, swapped, {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})

    assert isinstance(answer, Unresolved)
    codes = {item.code for item in answer.findings}
    assert "hardware-coverage-region-not-the-declared-one" in codes
    assert "hardware-covered-edge-misconnected" in codes


def test_a_fused_binding_without_its_edge_is_refused() -> None:
    """Bound without the edge it absorbs, a fused Kernel would silently be two."""

    placed = _place()
    answer = placed.bind(FUSED, placed.both_roles(), {})

    assert isinstance(answer, Unresolved)
    assert any(item.code == "hardware-coverage-edge-roles-mismatch" for item in answer.findings)


# -- the Network is the operation's, not the fixture's ------------------------


def test_the_bound_network_is_the_one_the_operation_builds() -> None:
    """Resolve a real ``MvauDataflowOp`` and compare the value, not the shape.

    This is the test whose absence let the earlier version of this file drift:
    it derived a private Network from the two Regions, which happened to equal
    the operation's. Equal today is not the same as connected, and a change to
    ``_construct_network`` would have left every test here green while the
    fused Kernel bound against a Network production never builds.
    """

    resolved = _committed(_model(repetitions=REPETITIONS)).resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)

    placed = _place(matrix_width=MATRIX_WIDTH, matrix_height=MATRIX_HEIGHT)
    assert placed.network == resolved.result.network


def test_the_fused_kernel_reads_the_operations_own_network_property() -> None:
    """Stated over the declaration, so it cannot regress into a private copy."""

    (edge,) = FUSED.coverage.edges
    assert edge.network.path == DOT_PRODUCT_DECLARATION.network.path
    assert edge.network.kind is DependencyKind.PROPERTY


def test_the_fixture_assembles_the_operations_real_specification() -> None:
    """Everything the operation declares, plus the fused Kernel and nothing else.

    The point is the *semantic* half: assembling a hand-picked subset of pools
    would drop the operation's feasibility constraints, which is how an
    accumulator/output mismatch reached hardware in the first place.
    """

    placed = _place()
    declared = {str(path) for path in placed.point.design_space.constraints}
    operation_owned = {str(item.path) for item in MVAU_DATAFLOW_OP_SPEC.constraints}

    assert operation_owned <= declared
    assert any("accumulator_output_type_supported" in path for path in declared)


# -- semantic feasibility gates the binding -----------------------------------


def test_an_output_that_is_not_the_accumulator_never_reaches_hardware() -> None:
    """The defect this fixture used to hide.

    ``INT16`` accumulator with an ``INT8`` output bound cleanly and produced
    ``ACCU_WIDTH=16`` while the Region's output operand said ``INT8``. The RTL
    output width is always ``PE*ACCU_WIDTH``, so that binding promised an
    eight-bit semantic output while emitting sixteen-bit values.

    The equality is correctly owned by the semantic Kernel rather than by
    hardware -- an output that is not the accumulator is a different
    computation, not a computation this core cannot do -- so what was missing
    was the gate, not the constraint.
    """

    placed = _place(accumulator=INT16, output=INT8)
    assert "accumulator_output_type_supported" in placed.semantic_refusals()

    # And the fixture refuses to bind past it, rather than leaving the
    # discipline to whoever writes the next test.
    with pytest.raises(InfeasiblePoint, match="refuses this point"):
        placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})


def test_the_guard_uses_the_operations_whole_design_feasibility_set() -> None:
    """The test-only binding is gated by the production design's own verdict."""

    assert MvauDataflowOp.selection_constraint_set() == "mvau_op_feasibility"

    (operation_set,) = (
        item for item in MVAU_DATAFLOW_OP_SPEC.constraint_sets if item.name == "mvau_op_feasibility"
    )
    design_constraints = {item.path for item in DOT_PRODUCT_DECLARATION.spec.constraints}
    assert design_constraints < set(operation_set.constraints)


def test_an_unresolved_operation_constraint_stops_the_binding() -> None:
    """Not evaluable is not permission.

    With no target DSP block the coverage constraints cannot be answered at
    all. The old guard collected only ``Decided(False)``, so an unanswerable
    point looked exactly like a clean one and bound -- producing hardware for a
    board nobody had named.

    The target is used rather than the source description because the source
    description is a *required* field: omitting it is refused by
    ``engine.start`` before a constraint is ever evaluated, which is a
    different failure and one the fixture already gets right.
    """

    placed = _place(target=OMITTED)
    assessment = placed.feasibility()
    unresolved = {
        str(path).rsplit(".", 1)[-1]
        for path, answer in assessment.answers.items()
        if isinstance(answer, Unresolved)
    }

    assert "width_supported" in unresolved
    assert not any(
        isinstance(answer, Decided) and answer.value is False
        for answer in assessment.answers.values()
    ), "this point must be unanswerable, not refused, or it proves nothing"
    assert assessment.verdict is None
    with pytest.raises(InfeasiblePoint):
        placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})

    # Honest about the strength of this one: binding would have refused this
    # point anyway, because the Kernel's own parameters cannot resolve without
    # a target. What it pins is the *shape* of the guard -- ``verdict is None``
    # has to stop it, not only ``verdict is False`` -- and that shape is what
    # the structural case above depends on.
    assert isinstance(
        placed.bind_unchecked(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}),
        Unresolved,
    )


def test_a_required_field_left_out_is_refused_before_any_constraint_runs() -> None:
    """The other absence, and it fails earlier and louder.

    ``source_description`` is required, so a point without one never becomes a
    point. Worth pinning because the earlier fixture assembled a context that
    excluded the field entirely, which is how a missing source description
    reached a binding at all.
    """

    with pytest.raises(RequestError) as raised:
        _place(source_description=OMITTED)
    assert any(item.code == "problem-required" for item in raised.value.findings)


def test_the_guard_is_not_an_assert_and_survives_optimised_python() -> None:
    """``python -O`` strips ``assert``; the guard must not be one.

    Run in a subprocess with ``-O`` because that is the only way to observe it:
    the flag is read at compile time, so no in-process check can tell.
    """

    source = (
        "from dataflow.mvau.test_fused_hardware import ("
        "  _place, FUSED, ACTIVATION_EDGE_ROLE, InfeasiblePoint, OMITTED)\n"
        "from finn.dataflow.ops.mvau.semantics import ACTIVATION_EDGE\n"
        "placed = _place(target=OMITTED)\n"
        "try:\n"
        "    placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})\n"
        "except InfeasiblePoint:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('the guard vanished under -O')\n"
    )
    finn_root = Path(__file__).parents[3]
    environment = {
        **os.environ,
        "FINN_ROOT": str(finn_root),
        "PYTHONPATH": os.pathsep.join(
            str(finn_root / part) for part in ("src", "tests", "deps/qonnx/src")
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-O", "-c", source], capture_output=True, text=True, env=environment
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_matching_accumulator_and_output_still_binds() -> None:
    """So the gate above is not merely refusing everything."""

    placed = _place(accumulator=INT16, output=INT16)
    assert placed.semantic_refusals() == set()
    assert dict(placed.fused().parameters)["ACCU_WIDTH"] == 16
    assert placed.compute_region.output_interface("output").port.operand.element_type == INT16


# -- weights have to pack into the DSP A port ---------------------------------


@pytest.mark.parametrize(
    ("target", "weight"),
    [(DspBlock.DSP48E2, INT27), (DspBlock.DSP58, INT27)],
)
def test_weights_that_fill_the_a_port_are_refused_without_the_narrow_promise(
    target: DspBlock, weight: NumericElementType
) -> None:
    """Reproduces a binding that would have died inside the RTL.

    Both of these used to bind cleanly. Both reach the generic ``mvu`` core,
    whose ``sliceLanes()`` computes ``bit_slack = -1`` and terminates with
    "Cannot accommodate 27-bit non-narrow weights" at ``mvu.sv:113``.

    The refusal is asserted at *operation feasibility* first, and that is the
    stronger of the two claims. Binding refusing it means no invalid artifact
    is produced; the operation refusing it means the point is eliminated while
    it is still a design point, which is what asking coverage at feasibility
    was for. Those were briefly not the same thing -- see
    ``test_a_rejecting_constraint_makes_the_set_verdict_false``.
    """

    placed = _place(weight=weight, accumulator=INT32, output=INT32, target=target, narrow=False)

    assert placed.feasibility().verdict is False
    assert "narrow_weights_supported" in placed.semantic_refusals()
    with pytest.raises(InfeasiblePoint, match="narrow_weights_supported"):
        placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})

    # And the Kernel's own refusal still carries the arithmetic that explains
    # it, for a caller who reaches binding by another route.
    answer = placed.bind_unchecked(
        FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}
    )
    assert isinstance(answer, Unresolved)
    finding = next(
        item for item in answer.findings if item.code == "mvu-vvu-axi-weights-do-not-pack"
    )
    assert dict(finding.values)["bit_slack"] == -1


def test_a_rejecting_coverage_constraint_makes_the_operation_refuse() -> None:
    """The guarantee coverage-at-feasibility exists for.

    Physical coverage constraints are asked at operation feasibility so a point
    no hardware can build is eliminated while it is still a design point --
    refused by selection, not discovered at binding.

    That guarantee was broken. A coverage constraint refuses by returning
    ``reject(...)``, which is an ``Absent`` carrying a ``REJECTION``, and the
    engine's verdict ignored every ``Absent``. So this configuration left
    ``mvau_op_feasibility`` reporting ``True``: ``SelectDataflowDesign`` could
    pick it, and only binding would say no.

    Asserted on the *verdict* rather than on a later refusal, because "binding
    eventually refuses it" is a weaker property that held throughout.
    """

    placed = _place(
        weight=INT27,
        accumulator=INT32,
        output=INT32,
        target=DspBlock.DSP48E2,
        narrow=False,
    )
    assessment = placed.feasibility()

    assert assessment.verdict is False
    (refused,) = assessment.refused
    assert str(refused).endswith("narrow_weights_supported")
    # Spelled as a rejection, which is exactly why it was being dropped.
    answer = assessment.answers[refused]
    assert isinstance(answer, Absent) and answer.is_rejection
    assert refused not in assessment.not_applicable


@pytest.mark.parametrize("target", [DspBlock.DSP48E1, DspBlock.DSP48E2, DspBlock.DSP58])
def test_ordinary_weights_pack_without_the_narrow_promise(target: DspBlock) -> None:
    """The over-refusal half.

    The old rule required the narrow promise for every DSP48E1 configuration.
    Eight-bit weights pack into two lanes there with a bit of slack left, so
    that rule removed working designs from coverage.
    """

    placed = _place(weight=INT8, target=target, narrow=False)
    answer = placed.bind(FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE})
    assert isinstance(answer, Decided)


def test_the_lane_calculation_matches_both_rtl_sources() -> None:
    """The model is shared between two Kernels, so pin it against both cores.

    Sharing ``pack_lanes`` is only defensible while the two sources really do
    compute the same thing. Reading the lines is what makes that a checked fact
    rather than an assumption -- the same discipline the duplicated datatype
    predicates get, applied to the one piece that *is* shared.
    """

    expected = (
        "NUM_LANES = A_WIDTH == WEIGHT_WIDTH? 1 : "
        "1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / MIN_LANE_WIDTH"
    )
    slack = "bit_slack -= !NARROW_WEIGHTS + WEIGHT_WIDTH + (NUM_LANES - 1) * MIN_LANE_WIDTH"
    finn_root = Path(__file__).parents[3]
    sources = (
        finn_root / "finn-rtllib" / "mvu" / "mvu.sv",
        finnlib_root(finn_root) / "rtl" / "linalg" / "dotp.sv",
    )
    for source in sources:
        text = " ".join(source.read_text().split())
        assert expected in text, source
        assert slack in text, source


def test_the_two_cores_pack_lanes_identically() -> None:
    """Same demonstration as the datatype predicates get, over the packing.

    They share one implementation here, so this compares the two *Kernels'*
    coverage answers rather than two functions -- which is what would diverge
    first if one Kernel started asking a different question.
    """

    for target in (DspBlock.DSP48E1, DspBlock.DSP48E2, DspBlock.DSP58):
        for weight in (INT8, INT25, INT27):
            for narrow in (False, True):
                placed = _place(
                    weight=weight,
                    accumulator=INT32,
                    output=INT32,
                    target=target,
                    narrow=narrow,
                )
                fused = placed.bind_unchecked(
                    FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}
                )
                dotp = placed.bind_unchecked(
                    DOTP_AXI,
                    bound_regions((("compute", DOT_PRODUCT_NODE, placed.compute_region),)),
                )
                assert isinstance(fused, Decided) == isinstance(dotp, Decided), (
                    target,
                    weight.name,
                    narrow,
                )


# -- what the association records, and what it cannot yet -------------------


def test_the_binding_records_the_regions_and_the_edge_it_covers() -> None:
    """Association at the granularity the physical Kernel surface has."""

    origin = _place().fused().origin()

    assert origin.kernel_id == "mvu_vvu_axi"
    assert origin.covered_nodes == (DOT_PRODUCT_NODE, REPLAY_NODE)
    assert origin.covered_edges == (ACTIVATION_EDGE,)
    assert dict(origin.computations) == {
        REPLAY_ROLE: "mvau.activation_replay:1",
        COMPUTE_ROLE: "mvau.dot_product:1",
    }


def test_port_level_association_is_not_representable_here_yet() -> None:
    """Recorded as a limit rather than left to be discovered.

    Phase 4's action list asks for the fused component to be associated with
    both Regions, their relevant *ports*, and the absorbed edge. Regions and
    the edge are recorded above. Ports are not, and cannot be:
    ``PhysicalComponent`` carries an id, a module, parameters and a parent, and
    deliberately nothing else -- interfaces and semantic-port refs live in the
    richer MVAU physical model, which only the decomposed composition path
    builds.

    Completing that for the fused reading means a fused elaborator producing an
    ``MVAUPhysicalElaboration``, which is Phase 6's business. Asserting the
    absence here keeps the gap visible instead of letting the Phase 4 record
    imply it was done.
    """

    (component,) = _place().fused().components()

    assert not hasattr(component, "interfaces")
    assert not hasattr(component, "semantic_ports")
    assert {field for field in vars(component)} == {"id", "module", "parameters", "parent"}


# -- the source manifest -----------------------------------------------------


def test_the_manifest_is_the_existing_fused_rtl_and_not_a_fork() -> None:
    """The same six files the legacy fused path compiles.

    Compared as a *set*, because the two disagree about order on purpose: the
    legacy list is whatever ``MVAU_rtl`` accumulated, and this one is
    dependency order, which is what a staged compile needs. What must not
    differ is the text -- a fused Kernel built from forked RTL would prove
    nothing about the core FINN actually ships.
    """

    legacy = {
        "mvu_pkg.sv",
        "mvu_vvu_axi.sv",
        "replay_buffer.sv",
        "mvu.sv",
        "mvu_vvu_8sx9_dsp58.sv",
        "add_multi.sv",
    }
    declared = {Path(item.path).name for item in FUSED.sources}

    assert declared == legacy
    assert all(item.root == "finn" for item in FUSED.sources)
    # The replay is inside this manifest and inside the core; in the decomposed
    # reading the same file is a separate Kernel's whole manifest.
    assert "replay_buffer.sv" in declared
    assert {Path(item.path).name for item in REPLAY_BUFFER.sources} == {
        "mvu_pkg.sv",
        "replay_buffer.sv",
    }


def test_the_declared_sources_exist_in_this_checkout() -> None:
    """A manifest naming files that are not there is caught here, not in xelab."""

    root = Path(__file__).parents[3]
    for item in FUSED.sources:
        assert (root / item.path).is_file(), item.path


# -- the parameter table -----------------------------------------------------

#: The module header of ``finn-rtllib/mvu/mvu_vvu_axi.sv``, excluding its
#: ``localparam`` deductions.  Exact, not a subset: a parameter the audit omits
#: is one whose value elaboration would have to invent.
MVU_VVU_AXI_PARAMETERS = {
    "IS_MVU",
    "VERSION",
    "MW",
    "MH",
    "PE",
    "SIMD",
    "SEGMENTLEN",
    "ACTIVATION_WIDTH",
    "WEIGHT_WIDTH",
    "ACCU_WIDTH",
    "NARROW_WEIGHTS",
    "SIGNED_ACTIVATIONS",
    "PUMPED_COMPUTE",
    "FORCE_BEHAVIORAL",
    "M_REG_LUT",
}


def test_the_audit_covers_exactly_the_parameters_the_fused_rtl_takes() -> None:
    assert set(FUSED.parameter_names) == MVU_VVU_AXI_PARAMETERS


def test_the_matrix_geometry_is_the_fusion_visible_in_the_parameter_list() -> None:
    """``MW`` and ``MH`` are here and absent from ``dotp_axi``.

    This core sizes its internal replay from them.  In the decomposed reading
    that same geometry became the replay Kernel's ``LEN`` and ``REP``, which is
    why the dot product needs neither.
    """

    assert {"MW", "MH"} <= set(FUSED.parameter_names)
    assert {"MW", "MH"}.isdisjoint(DOTP_AXI.parameter_names)
    assert {"LEN", "REP"} <= set(REPLAY_BUFFER.parameter_names)


def test_every_fused_parameter_resolves_from_the_point() -> None:
    placed = _place(matrix_width=8, matrix_height=4, pe=2, simd=2)
    values = dict(placed.fused().parameters)

    assert values["MW"] == 8
    assert values["MH"] == 4
    assert values["PE"] == 2
    assert values["SIMD"] == 2
    assert values["ACTIVATION_WIDTH"] == 8
    assert values["ACCU_WIDTH"] == 16
    assert values["IS_MVU"] == 1
    assert values["FORCE_BEHAVIORAL"] == 0
    assert set(values) == MVU_VVU_AXI_PARAMETERS


# -- parity with the RTL evidence that already exists -------------------------

#: The parameters fixture 5 drives into ``mvu_vvu_axi`` for its golden, beyond
#: the geometry it supplies separately.  Read from the fixture rather than
#: restated, so a change there shows up here.
_FIXTURE_SHARED = (
    "VERSION",
    "PE",
    "SIMD",
    "SEGMENTLEN",
    "ACTIVATION_WIDTH",
    "WEIGHT_WIDTH",
    "ACCU_WIDTH",
    "NARROW_WEIGHTS",
    "SIGNED_ACTIVATIONS",
    "PUMPED_COMPUTE",
    "FORCE_BEHAVIORAL",
)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda item: item.label)
def test_this_kernel_declares_what_fixture_five_already_measured(config: Config) -> None:
    """Tie the Kernel's parameters to the configurations that have RTL evidence.

    Fixture 5 compares ``mvu_vvu_axi`` against the composed path across seven
    configurations, so it is the natural evidence to cite for this Kernel --
    but only for the points it actually ran. An earlier draft of the Phase 4
    record claimed it covered "these exact parameter values"; it did not.
    Every fixture configuration resolves ``NARROW_WEIGHTS`` to ``True`` from a
    real initializer, and this file's default point resolved it to ``False``,
    so the two never agreed on the parameter that decides how the weights pack.

    This makes the citation true instead of removing it: the Kernel is placed
    at each fixture configuration and its declared table is compared against
    what the fixture drives.
    """

    placed = _place(
        repetitions=config.repetitions,
        matrix_width=config.matrix_width,
        matrix_height=config.matrix_height,
        pe=config.pe,
        simd=config.simd,
        target=config.target,
        pumping=config.pumping,
        # Taken from the configuration, not defaulted.  ``dsp48e1`` runs 4-bit
        # activations, and a citation placed at a width the fixture does not
        # run would be citing a different point than the one it claims.
        activation=DataType[f"INT{config.activation_bits}"],
        weight=DataType[f"INT{config.weight_bits}"],
        narrow=True,
    )
    declared = dict(placed.fused().parameters)
    fixture = declared_parameters(config)

    for name in _FIXTURE_SHARED:
        assert declared[name] == fixture[name], name
    # The three the fixture supplies to the fused wrapper directly, because the
    # decomposed parameter set does not carry them.
    assert declared["IS_MVU"] == 1
    assert declared["MW"] == config.matrix_width
    assert declared["MH"] == config.matrix_height
    # ...and nothing outside those two groups is left unaccounted for.
    assert set(declared) == set(_FIXTURE_SHARED) | {"IS_MVU", "MW", "MH", "M_REG_LUT"}


def test_the_narrow_weight_parameter_is_what_the_fixture_actually_resolved() -> None:
    """Guard the specific mismatch, so the parity above cannot drift silently.

    Asserted directly because it is the one parameter the earlier claim got
    wrong, and because it is resolved from an initializer rather than chosen --
    the kind of value a fixture and a fixture-free point disagree about without
    either looking wrong.
    """

    assert {declared_parameters(config)["NARROW_WEIGHTS"] for config in CONFIGS} == {True}


# -- the datatype contract ---------------------------------------------------


def _verdicts(
    activation: NumericElementType = INT8,
    weight: NumericElementType = INT8,
    accumulator: NumericElementType = INT16,
    output: NumericElementType = INT16,
) -> dict[str, RoleVerdict]:
    types = DotProductNumericTypes(activation, weight, accumulator, output)
    return {item.role: item for item in fused_covers_numeric_types(types)}


def test_the_signature_is_complete_so_no_role_can_go_unasked() -> None:
    """Four roles in, four verdicts out.

    The shape is the guarantee.  A predicate taking two loose operands is what
    admitted an integer product with a floating-point accumulator, and no
    amount of care in the body fixes a signature that never receives the value.
    """

    assert set(_verdicts()) == {"activation", "weight", "accumulator", "output"}
    assert all(item.supported for item in _verdicts().values())


def test_an_unsigned_activation_is_accepted_and_the_other_roles_are_not() -> None:
    """The role contract is deliberately non-uniform, as ``mvu.sv`` is.

    Only ``a`` is declared without a sign, because ``SIGNED_ACTIVATIONS`` tells
    the core which it is getting.  Applying one rule to all four roles is the
    shape that admitted an unsigned accumulator with no refusal at all.
    """

    assert _verdicts(activation=UINT8)["activation"].supported

    for role, types in (
        ("weight", {"weight": DataType["UINT8"]}),
        ("accumulator", {"accumulator": UINT16}),
        ("output", {"output": UINT16}),
    ):
        verdict = _verdicts(**types)[role]
        assert not verdict.supported
        assert "unsigned" in verdict.detail
        assert "signed" in verdict.detail


def test_floating_point_is_refused_in_every_role() -> None:
    """A float MVAU is a Region this Kernel cannot build, not a Region that
    does not exist -- so the refusal belongs here rather than upstream."""

    for role in ("activation", "weight", "accumulator", "output"):
        verdict = _verdicts(**{role: FLOAT16})[role]
        assert not verdict.supported
        assert "not a two's-complement integer" in verdict.detail


def test_integer_valued_encodings_are_refused_by_canonical_identity() -> None:
    """``BINARY``, ``BIPOLAR`` and ``TERNARY`` all answer ``is_integer()`` true.

    None of them is a two's-complement encoding, so a predicate built on
    ``is_integer()`` admits them and the datapath then computes over a range it
    was never given.  ``TERNARY`` is the one that bit: two bits wide, spanning
    -1..1, and lowered as ``INT2``.
    """

    for encoding in (BINARY, BIPOLAR, TERNARY):
        assert encoding.is_integer()
        assert not _verdicts(weight=encoding)["weight"].supported


def test_equal_width_does_not_imply_equal_identity() -> None:
    """The pairs that separate reading names from reading widths.

    ``TERNARY`` and ``INT2`` are both two bits; ``FLOAT16`` and
    ``FLOAT<5,10,7>`` are both sixteen.  Under the family-and-width pair this
    migration removed, each pair was one value.
    """

    assert TERNARY.bitwidth() == INT2.bitwidth()
    assert not _verdicts(weight=TERNARY)["weight"].supported
    assert _verdicts(weight=INT2)["weight"].supported

    assert FLOAT16.bitwidth() == OTHER_FLOAT16.bitwidth()
    assert FLOAT16 != OTHER_FLOAT16
    for float_type in (FLOAT16, OTHER_FLOAT16):
        assert not _verdicts(accumulator=float_type)["accumulator"].supported


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("unsigned accumulator and output", {"accumulator": UINT16, "output": UINT16}),
        ("ternary weights", {"weight": TERNARY}),
        ("floating accumulator and output", {"accumulator": FLOAT16, "output": FLOAT16}),
    ],
)
def test_the_legacy_compatibility_path_refuses_unsupported_numerics(
    label: str, overrides: dict[str, NumericElementType]
) -> None:
    """The legacy path keeps its local predicate until B7 removes it."""

    placed = _place(**overrides)  # type: ignore[arg-type]
    assert "operand_types_supported" in placed.semantic_refusals(), label


def test_a_refusal_reaches_the_binding_and_names_the_offending_role() -> None:
    """The Kernel's own refusal, reached by stepping past the semantic gate.

    Second line of defence, not dead code: the inventory admission quantifies
    over holds ``DotpAxiKernel`` alone, so a future Kernel that widened it
    would make these points admissible and leave this refusal as the only one.
    """

    placed = _place(accumulator=UINT16, output=UINT16)
    answer = placed.bind_unchecked(
        FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}
    )

    assert isinstance(answer, Unresolved)
    codes = {item.code for item in answer.findings}
    assert "mvu-vvu-axi-numeric-types-unsupported" in codes
    named = {
        key
        for item in answer.findings
        if item.code == "mvu-vvu-axi-numeric-types-unsupported"
        for key, _ in item.values
    }
    assert named == {"accumulator", "output"}


def test_the_fused_kernel_refuses_ternary_by_name_not_by_width() -> None:
    """The end-to-end form of the defect the adoption closed."""

    placed = _place(weight=TERNARY)
    answer = placed.bind_unchecked(
        FUSED, placed.both_roles(), {ACTIVATION_EDGE_ROLE: ACTIVATION_EDGE}
    )

    assert isinstance(answer, Unresolved)
    assert any(
        "TERNARY" in item.message or "TERNARY" in str(item.values) for item in answer.findings
    )


# -- the two cores are separate predicates -----------------------------------


def test_the_fused_kernel_owns_its_predicate_rather_than_delegating() -> None:
    """Two functions, in two modules, over two cores.

    ``mvu_vvu_axi`` and ``dotp_axi`` are different hardware.  Aliasing one
    predicate to the other would make a change to either core silently redefine
    what the other accepts, and the day they diverge nothing would say so.
    """

    assert fused_covers_numeric_types is not dotp_axi_covers_numeric_types
    assert fused_covers_numeric_types.__module__ != dotp_axi_covers_numeric_types.__module__


#: Every signature worth comparing the two cores over: the supported baseline,
#: each role made unsigned, each role made floating, and the special encodings.
_COMPARISON = (
    DotProductNumericTypes(INT8, INT8, INT16, INT16),
    DotProductNumericTypes(UINT8, INT8, INT16, INT16),
    DotProductNumericTypes(INT8, DataType["UINT8"], INT16, INT16),
    DotProductNumericTypes(INT8, INT8, UINT16, UINT16),
    DotProductNumericTypes(FLOAT16, INT8, INT16, INT16),
    DotProductNumericTypes(INT8, FLOAT16, INT16, INT16),
    DotProductNumericTypes(INT8, INT8, FLOAT16, FLOAT16),
    DotProductNumericTypes(INT8, TERNARY, INT16, INT16),
    DotProductNumericTypes(BIPOLAR, INT8, INT16, INT16),
    DotProductNumericTypes(INT8, INT2, INT16, INT16),
)


def test_the_two_cores_agree_today_and_the_agreement_is_demonstrated() -> None:
    """They do agree -- and that is a measurement, not an assumption.

    Both cores declare the same three roles signed and multiply the same two
    families, so every signature below lands the same way in both.  Recording
    it as a test is what makes the agreement a fact about this revision: when
    one core's port list changes, this fails and names the signature, instead
    of one Kernel quietly inheriting the other's new limits.
    """

    for types in _COMPARISON:
        fused = {item.role: item.supported for item in fused_covers_numeric_types(types)}
        dotp = {item.role: item.supported for item in dotp_axi_covers_numeric_types(types)}
        assert fused == dotp, types


# -- test-only, and held there -----------------------------------------------


def test_the_fused_kernel_is_absent_from_the_production_assembly() -> None:
    """It is placed by this test file and nowhere else."""

    candidates = {
        item.id
        for declaration in MVAU_DESIGN_INVENTORY.inventory.declarations
        for placement in declaration.placements
        for item in placement.candidates
    }
    assert {DotpAxiKernel.id, ReplayBufferKernel.id} <= candidates
    assert MvuVvuAxiKernel.id not in candidates


def test_the_fused_kernel_lives_only_in_the_test_tree() -> None:
    assert MvuVvuAxiKernel.__module__ == "dataflow.mvau.mvu_vvu_axi_kernel"
    assert not Path("src/finn/dataflow/mvau/hardware/mvu_vvu_axi.py").exists()


def test_the_fused_coverage_constraints_are_not_in_the_operation_feasibility_set() -> None:
    """Its refusals must not remove points nothing was going to build with it."""

    production = {
        path
        for placement in DOT_PRODUCT_DECLARATION.placements
        for candidate in placement.candidates
        for path in candidate.coverage_constraints
    }
    assert production
    assert production.isdisjoint(FUSED.coverage_constraints)
