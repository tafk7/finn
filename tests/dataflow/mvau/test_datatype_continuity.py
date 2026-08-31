# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6a: one datatype, followed from the graph to the artifact.

``finn.dataflow.datatypes`` claims there is no intermediate datatype
representation: a tensor's type travels from the ``ModelWrapper`` through the
problem instance, the Region operand, Kernel coverage and artifact metadata as
the same QONNX value.  Every stage of that claim is tested somewhere; the claim
that it is *one chain* is tested nowhere, and a chain is exactly what a dropped
and reconstructed value would still look like at each end.

Two things this module refuses to do, and both are the point.

**It never compares names.**  ``DataType["INT8"] == "INT8"`` is true in both
directions and the hashes agree, so a name-to-name assertion passes even when
the value was replaced by its own label.  Comparison is over the whole value
domain -- name, width, signedness, and the representable range -- which is what
distinguishes ``TERNARY`` from ``INT2``, the pair the retired ``(family, width)``
reduction collapsed.

**It says which parameters each datatype became.**  The chain legitimately ends
before the artifact: at the last hop the activation type is no longer a value,
it is ``ACTIVATION_WIDTH`` and ``SIGNED_ACTIVATIONS``.  So the final assertion
is that the effect is *there* -- perturb the type, and the named parameters and
no others move.  A datatype distinction that reaches nothing would otherwise be
a silent collision in an artifact key, which is the failure the Phase 5 review
kept finding in other clothes.

The output type is the interesting case: it reaches no Kernel parameter at all,
and that is only sound because coverage forces it to equal the accumulator.
That rule was documented in ``dotp_axi`` and not asked, and following the chain
here is what found it -- see
:func:`test_an_output_that_is_not_the_accumulator_never_reaches_a_wrapper`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from dataflow.mvau.test_decomposed_op import (
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    _choices,
    _context,
    _wrapped,
)
from finn.dataflow.datatypes import is_qonnx_datatype
from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.elaboration import MVAUElaborationError
from finn.dataflow.mvau.hardware.binding import bind_decomposed
from finn.dataflow.mvau.hardware.composition import (
    build_decomposed_artifact_requirements,
    elaborate_decomposed,
)
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.mvau_problem import MVAUProblemPaths
from finn.dataflow.ops.mvau_op import MvauDataflowOp
from finn.dataflow.region import NumericElementType

FINN_ROOT = Path(__file__).resolve().parents[3]

NODE_ID = "mvau_continuity"
REPETITIONS = 4

#: Small, non-negative, and deliberately excluding the minimum of every weight
#: type used here.  ``EFFECTIVE_NARROW_WEIGHTS`` is derived from whether the
#: initializer reaches its type's minimum, so an initializer that reached it
#: for ``INT8`` and not for ``INT4`` would make the narrow flag move whenever
#: the width did, and the perturbation table below could not tell the two
#: effects apart.
WEIGHTS = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 4], [5, 6, 7, 0]], dtype=np.float32)

INT4 = DataType["INT4"]
INT8 = DataType["INT8"]
UINT8 = DataType["UINT8"]
INT16 = DataType["INT16"]
INT24 = DataType["INT24"]
INT32 = DataType["INT32"]


# -- the roles, and where each one enters ------------------------------------


@dataclass(frozen=True)
class _Role:
    """One numeric role and every place its datatype is supposed to appear."""

    label: str
    #: How it enters the graph.  Three of the four are tensor annotations; the
    #: accumulator is a node attribute, because it is a compilation decision
    #: the source carries rather than a property of any tensor.
    tensor: str | None
    attribute: str | None
    problem_path: QualifiedPath
    #: ``(binding, interface)`` pairs whose Region operand this type is.
    operands: tuple[tuple[str, str], ...]
    #: The RTL parameters whose values this datatype determines, exactly.
    parameters: frozenset[str]
    #: The generated wrapper's port-width parameters it sizes.
    wrapper_parameters: frozenset[str]


ROLES = (
    _Role(
        "activation",
        "activation",
        None,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE,
        (("replay", "activation_in"), ("replay", "activation_out"), ("compute", "activation")),
        # ``W`` is the replay buffer's beat width, which is SIMD activations
        # wide -- the activation type reaches both Kernels, not just the core
        # that multiplies with it.
        frozenset({"ACTIVATION_WIDTH", "SIGNED_ACTIVATIONS", "W"}),
        frozenset({"ISTREAM"}),
    ),
    _Role(
        "weight",
        "weights",
        None,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE,
        (("compute", "weight"),),
        frozenset({"WEIGHT_WIDTH"}),
        frozenset({"WSTREAM"}),
    ),
    _Role(
        "accumulator",
        None,
        "accDataType",
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE,
        # It types no boundary port: the accumulation is internal to the core,
        # and what leaves is the output operand.
        (),
        frozenset({"ACCU_WIDTH"}),
        frozenset(),
    ),
    _Role(
        "output",
        "output",
        None,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE,
        (("compute", "output"),),
        # None, and sound only because coverage pins it to the accumulator.
        frozenset(),
        frozenset({"OSTREAM"}),
    ),
)

ROLES_BY_LABEL = {role.label: role for role in ROLES}


# -- the model -----------------------------------------------------------------


def _model(
    *,
    activation: NumericElementType = INT8,
    weight: NumericElementType = INT8,
    accumulator: NumericElementType = INT16,
    output: NumericElementType = INT16,
) -> ModelWrapper:
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weights"],
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=f"{NODE_ID}_scope",
        accDataType=accumulator.name,
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [node],
                "datatype-continuity",
                [
                    helper.make_tensor_value_info(
                        "activation", TensorProto.FLOAT, [REPETITIONS, MATRIX_WIDTH]
                    ),
                    helper.make_tensor_value_info(
                        "weights", TensorProto.FLOAT, [MATRIX_WIDTH, MATRIX_HEIGHT]
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "output", TensorProto.FLOAT, [REPETITIONS, MATRIX_HEIGHT]
                    )
                ],
            ),
            producer_name="datatype-continuity-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", activation)
    model.set_tensor_datatype("weights", weight)
    model.set_tensor_datatype("output", output)
    model.set_initializer("weights", WEIGHTS)
    return model


def _resolved(model: ModelWrapper) -> MVAUResolvedDesign:
    operation = _wrapped(model)
    assert isinstance(operation, MvauDataflowOp)
    operation.initialize_dataflow_scope_id()
    operation.commit_dataflow_assignments(_context(), _choices())
    return operation.resolve_dataflow(_context())


# -- the chain -----------------------------------------------------------------


@dataclass(frozen=True)
class _Chain:
    """One build, read at every stage a datatype is supposed to survive."""

    annotation: Mapping[str, object]
    problem: Mapping[str, object]
    operands: Mapping[tuple[str, str], object]
    parameters: Mapping[str, object]
    wrapper_parameters: Mapping[str, int]
    artifact_parameters: Mapping[str, object]


def _wrapper_parameters(source: str) -> dict[str, int]:
    """The generated top's declared port widths, read out of the text.

    Read from the text rather than recomputed, because the text is what a
    consumer compiles.  A helper that recomputed the widths would agree with a
    wrapper that had them wrong.
    """

    found = {}
    for line in source.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped.startswith("parameter "):
            continue
        name, _, value = stripped[len("parameter ") :].partition(" = ")
        found[name.strip()] = int(value)
    return found


def _byte_aligned(bits: int) -> int:
    """A ``tdata`` width, rounded up as the generator rounds it.

    Restated here rather than imported: the generator's helper is private, and
    a test that called it would agree with a generator that rounded wrongly.
    """

    return ((bits + 7) // 8) * 8


def _operand_type(region: object, port_id: str) -> object:
    """The element type of one Region port, found by the port's own id."""

    for interface in region.interfaces:  # type: ignore[attr-defined]
        if interface.port.id == port_id:
            return interface.port.operand.element_type
    raise AssertionError(f"no interface {port_id!r} on this Region")


def _chain(model: ModelWrapper) -> _Chain:
    resolved = _resolved(model)
    bindings = bind_decomposed(resolved)
    by_binding = {"replay": bindings.replay, "compute": bindings.compute}
    requirements = build_decomposed_artifact_requirements(
        resolved, elaborate_decomposed(resolved), FINN_ROOT
    )

    annotation: dict[str, object] = {}
    for role in ROLES:
        if role.tensor is not None:
            annotation[role.label] = model.get_tensor_datatype(role.tensor)
        else:
            assert role.attribute is not None
            annotation[role.label] = DataType[_wrapped(model).get_nodeattr(role.attribute)]

    return _Chain(
        annotation,
        {role.label: resolved.projection.problem_data[role.problem_path] for role in ROLES},
        {
            (binding, interface): _operand_type(by_binding[binding].regions[0].region, interface)
            for role in ROLES
            for binding, interface in role.operands
        },
        {name: value for binding in bindings.bindings for name, value in binding.parameters},
        _wrapper_parameters(requirements.wrapper_source),
        {
            name: value
            for kernel in requirements.identity.kernels
            for name, value in kernel.parameters
        },
    )


# -- comparing values, not names -----------------------------------------------


def _signature(value: object) -> tuple[object, ...]:
    """The whole value domain, so nothing weaker can stand in for equality.

    ``==`` alone is not enough here.  A QONNX datatype compares equal to its own
    canonical name in both directions, so an assertion between two stages passes
    when one of them is holding the string.  The recognizer rejects ``str``, and
    the range is what separates ``TERNARY`` from ``INT2``.
    """

    assert is_qonnx_datatype(value), f"{value!r} is not a datatype value"
    datatype = value
    return (
        datatype.name,
        datatype.bitwidth(),
        datatype.signed(),
        datatype.min(),
        datatype.max(),
        datatype.is_integer(),
    )


def test_the_comparison_this_module_uses_is_stronger_than_equality() -> None:
    """The premise, stated before it is relied on four times below.

    Two hazards, both real in the pinned QONNX.  A canonical name is ``==`` its
    own datatype, so a name would pass a value assertion; and ``TERNARY`` and
    ``INT2`` are both two-bit signed integers, so width and family would pass
    one too.  The signature fails both.
    """

    assert DataType["INT8"] == "INT8"
    assert not is_qonnx_datatype("INT8")
    with pytest.raises(AssertionError):
        _signature("INT8")

    ternary, int2 = DataType["TERNARY"], DataType["INT2"]
    assert ternary.bitwidth() == int2.bitwidth()
    assert ternary.signed() == int2.signed()
    assert _signature(ternary) != _signature(int2)


# -- the chain holds -----------------------------------------------------------


@pytest.mark.parametrize("role", ROLES, ids=lambda item: item.label)
def test_the_datatype_is_the_same_value_from_the_graph_to_the_problem(role: _Role) -> None:
    """Hop one: the annotation, and the field projected from it."""

    chain = _chain(_model())
    assert _signature(chain.problem[role.label]) == _signature(chain.annotation[role.label])


@pytest.mark.parametrize("role", ROLES, ids=lambda item: item.label)
def test_the_datatype_is_the_same_value_in_every_region_operand_it_types(role: _Role) -> None:
    """Hop two: the Region operands, which are the semantic boundary.

    The activation appears three times -- both replay ports and the core's
    input -- and all three must be the one value.  A reconstruction anywhere in
    the assembly would show up as one of them disagreeing.
    """

    chain = _chain(_model())
    expected = _signature(chain.annotation[role.label])
    for key in role.operands:
        assert _signature(chain.operands[key]) == expected, key


def test_the_kernel_covers_the_regions_carrying_those_operands() -> None:
    """Hop three: coverage is not a fourth place the type is written down.

    ``bind_hardware_kernel`` checks that the bound Region is the one the
    coverage handle derives, so the Kernel's view of the operand types is the
    Region's by construction rather than by copy.  What is worth asserting is
    that the roles line up -- an operand checked above under ``compute`` that
    the compute Kernel does not actually cover would make the hop vacuous.
    """

    bindings = bind_decomposed(_resolved(_model()))
    covered = {
        "replay": bindings.replay.regions[0].role,
        "compute": bindings.compute.regions[0].role,
    }
    assert covered == {"replay": "replay", "compute": "compute"}
    for role in ROLES:
        for binding, _interface in role.operands:
            assert binding in covered


def test_the_artifact_records_the_same_parameters_the_bindings_hold() -> None:
    """Hop four: nothing is recomputed on the way into the identity."""

    chain = _chain(_model())
    assert dict(chain.artifact_parameters) == dict(chain.parameters)


# -- where the chain ends ------------------------------------------------------

#: Each role, a second datatype for it, and what that change is expected to
#: move.  The accumulator and the output move together because coverage
#: requires them equal -- which is itself the reason the output's own parameter
#: set is empty.
PERTURBATIONS = (
    ("activation width", {"activation": INT4}, frozenset({"ACTIVATION_WIDTH", "W"})),
    ("activation signedness", {"activation": UINT8}, frozenset({"SIGNED_ACTIVATIONS"})),
    ("weight width", {"weight": INT4}, frozenset({"WEIGHT_WIDTH"})),
    (
        "accumulator width",
        {"accumulator": INT32, "output": INT32},
        frozenset({"ACCU_WIDTH"}),
    ),
)


@pytest.mark.parametrize(
    ("label", "change", "expected"), PERTURBATIONS, ids=[item[0] for item in PERTURBATIONS]
)
def test_a_datatype_change_moves_exactly_the_parameters_it_is_declared_to(
    label: str, change: dict[str, NumericElementType], expected: frozenset[str]
) -> None:
    """The end of the chain, asserted in both directions.

    *Every* declared parameter moves, so nothing claimed is decorative, and *no
    other* parameter moves, so nothing else silently depends on the type.  This
    is what makes the artifact key honest about the datatype: a distinction
    reaching no parameter would produce two different graphs sharing one key.
    """

    before = _chain(_model())
    after = _chain(_model(**change))
    moved = {
        name for name in before.parameters if before.parameters[name] != after.parameters[name]
    }
    assert moved == expected


def test_the_declared_parameter_sets_are_disjoint_and_cover_what_moved() -> None:
    """No parameter is claimed by two roles, and the table is not aspirational."""

    claimed: list[str] = []
    for role in ROLES:
        claimed.extend(role.parameters)
    assert len(claimed) == len(set(claimed))
    assert set(claimed) <= set(_chain(_model()).parameters)


def test_the_output_type_reaches_no_kernel_parameter_and_sizes_the_top_instead() -> None:
    """Where the output type actually lands, said out loud.

    It is the only role with an empty parameter set, and an empty set is
    normally the failure this module exists to catch.  It is not one here
    because the type has not vanished: it sizes the generated top's ``OSTREAM``,
    and coverage guarantees that width is the accumulator's.
    """

    assert ROLES_BY_LABEL["output"].parameters == frozenset()
    chain = _chain(_model())
    output = chain.annotation["output"]
    assert is_qonnx_datatype(output)
    pe = chain.parameters["PE"]
    assert isinstance(pe, int)
    width = output.bitwidth()
    assert chain.wrapper_parameters["OSTREAM"] == _byte_aligned(width * pe)
    # ...and that is the accumulator's width, because nothing else would fit
    # what the core drives.
    accumulator = chain.parameters["ACCU_WIDTH"]
    assert isinstance(accumulator, int)
    assert chain.wrapper_parameters["OSTREAM"] == _byte_aligned(accumulator * pe)


@pytest.mark.parametrize("role", ROLES, ids=lambda item: item.label)
def test_every_role_reaches_either_a_parameter_or_a_port_width(role: _Role) -> None:
    """The rule the table encodes: a datatype that reaches nothing is a bug.

    Stated as a property rather than left implicit in four rows, so that adding
    a fifth role cannot quietly add one with no effect anywhere.
    """

    assert role.parameters or role.wrapper_parameters


@pytest.mark.parametrize("role", ROLES, ids=lambda item: item.label)
def test_each_role_sizes_the_wrapper_port_it_is_declared_to(role: _Role) -> None:
    """The generated top's port widths, against the datatype and the folding.

    Computed from the annotation rather than read back from the same text, so
    this says the width is *right* and not merely that it is what it is.  One
    activation beat is ``SIMD`` elements, one weight beat is ``SIMD * PE``, and
    one output beat is ``PE``.

    Byte-aligned, because these are AXI-Stream ``tdata`` widths and the
    generator rounds them up.  Written out rather than left to the arithmetic
    happening to divide: every width here is a multiple of eight already, so an
    unrounded expectation would pass today and mislead the first time a
    non-byte-multiple element type reached it.
    """

    chain = _chain(_model())
    simd, pe = chain.parameters["SIMD"], chain.parameters["PE"]
    assert isinstance(simd, int) and isinstance(pe, int)
    elements = {"ISTREAM": simd, "WSTREAM": simd * pe, "OSTREAM": pe}
    width = _signature(chain.annotation[role.label])[1]
    assert isinstance(width, int)
    for name in role.wrapper_parameters:
        assert chain.wrapper_parameters[name] == _byte_aligned(width * elements[name]), name


# -- the defect this chain found -----------------------------------------------


def test_an_output_that_is_not_the_accumulator_never_reaches_a_wrapper() -> None:
    """``dotp_axi`` documented this rule in its role table and never asked it.

    The core drives ``PE * ACCU_WIDTH`` bits straight out of
    ``m_axis_output_tdata``, and the composed wrapper sizes ``out0_V_tdata``
    from the *output* element type.  With an ``INT32`` accumulator and an
    ``INT24`` output the generated top declared ``OSTREAM = 48`` around a core
    driving 64 bits -- no truncation, no conversion, no finding.

    The semantic Kernels do carry the rule, as a source constraint, which gates
    inference.  Committing a selection directly is a supported path that never
    passes through inference, so the rule had to be a *coverage* question to be
    load-bearing.  It refuses at binding, which is before any elaboration runs.
    """

    resolved = _resolved(_model(accumulator=INT32, output=INT24))
    with pytest.raises(MVAUElaborationError) as refusal:
        bind_decomposed(resolved)
    detail = " ".join(item.message for item in refusal.value.findings) + " ".join(
        str(item.values) for item in refusal.value.findings
    )
    assert "operand_types_supported" in detail


def test_the_refusal_happens_before_anything_is_elaborated() -> None:
    """Item 12's remaining half, for this case.

    A refusal that surfaces out of wrapper generation has already lost the
    diagnostic and done work it should not have begun.  Both entry points into
    the physical path have to stop at the same place.
    """

    resolved = _resolved(_model(accumulator=INT32, output=INT24))
    for entry in (bind_decomposed, elaborate_decomposed):
        with pytest.raises(MVAUElaborationError):
            entry(resolved)
