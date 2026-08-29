# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three facts behind ``NARROW_WEIGHTS`` and who owns each.

``NARROW_WEIGHTS`` is a promise about the values reaching the implementation,
not a property of the weight datatype, so the fact that carries it depends on
who owns those values.  The operation used to project one ``weights_narrow``
field from the graph and then overwrite it from build configuration; these
tests pin the replacement, in which no projection writes another's fact.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.authoring.op_design import Provenance
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM_PROVENANCE,
    MVAUComputationProfile,
    MVAUProblemPaths,
)
from finn.dataflow.spec_algebra import assemble_specs

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


def _narrow(
    *,
    excludes_minimum: bool | None = None,
    runtime_contract: bool | None = None,
    runtime_writable: bool = False,
) -> object:
    P = MVAUProblemPaths
    problem: dict[QualifiedPath, object] = {
        P.REPETITIONS: 2,
        P.MATRIX_WIDTH: 4,
        P.MATRIX_HEIGHT: 4,
        P.ACTIVATION_ELEMENT_TYPE: INT8,
        P.WEIGHT_ELEMENT_TYPE: INT8,
        P.ACCUMULATOR_ELEMENT_TYPE: INT16,
        P.OUTPUT_ELEMENT_TYPE: INT16,
        P.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        P.WEIGHT_INITIALIZER_AVAILABLE: True,
        P.RUNTIME_WRITABLE: runtime_writable,
    }
    if excludes_minimum is not None:
        problem[P.INITIALIZER_EXCLUDES_MINIMUM] = excludes_minimum
    if runtime_contract is not None:
        problem[P.RUNTIME_WEIGHT_RANGE_CONTRACT] = runtime_contract

    engine = Engine()
    space = engine.validate(assemble_specs((compute_pool_context(),)))
    answer = engine.query_property(engine.start(space, problem), P.EFFECTIVE_NARROW_WEIGHTS)
    assert isinstance(answer, Decided)
    return answer.value


def test_the_three_facts_have_three_different_owners() -> None:
    kind_of = MVAU_PROBLEM_PROVENANCE.kind_of
    assert kind_of(MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM) is Provenance.GRAPH_ANALYSIS
    assert kind_of(MVAUProblemPaths.RUNTIME_WEIGHT_RANGE_CONTRACT) is Provenance.BUILD
    assert kind_of(MVAUProblemPaths.RUNTIME_WRITABLE) is Provenance.BUILD


def test_an_immutable_initializer_governs_the_promise() -> None:
    assert _narrow(excludes_minimum=True) is True
    assert _narrow(excludes_minimum=False) is False


def test_runtime_writable_weights_ignore_the_initializer_entirely() -> None:
    """The old overwrite forced False here; now the caller's contract decides."""

    assert _narrow(excludes_minimum=True, runtime_writable=True) is False
    assert _narrow(excludes_minimum=True, runtime_writable=True, runtime_contract=True) is True
    assert _narrow(excludes_minimum=False, runtime_writable=True, runtime_contract=True) is True


def test_no_value_authority_is_conservatively_not_narrow() -> None:
    assert _narrow() is False
    assert _narrow(runtime_writable=True) is False


def test_a_contract_alone_does_not_narrow_initializer_backed_weights() -> None:
    """A runtime promise says nothing about values that are never written."""

    assert _narrow(runtime_contract=True) is False
