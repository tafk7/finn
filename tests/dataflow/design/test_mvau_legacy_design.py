# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The retained standalone MVAU design space is unaffected by the Kernel pool."""

from __future__ import annotations

from finn.dataflow.design import Decided, Engine
from finn.dataflow.mvau.legacy_design import MVAU_DESIGN_SPACE_SPEC, MVAUDesignPaths
from finn.dataflow.mvau.regions import construct_standard_streamed_mvau_region
from finn.dataflow.ops.mvau import MVAU_DATAFLOW_OP_SPEC
from finn.dataflow.region import NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def test_the_legacy_spec_is_not_the_operation_spec() -> None:
    assert str(MVAUDesignPaths.PE) == "mvau.pe"
    assert str(MVAUDesignPaths.REGION) == "semantic.mvau.region"
    assert MVAU_DESIGN_SPACE_SPEC is not MVAU_DATAFLOW_OP_SPEC


def test_the_legacy_spec_accepts_its_original_problem_schema_and_paths() -> None:
    engine = Engine()
    space = engine.validate(MVAU_DESIGN_SPACE_SPEC)
    point = engine.start(
        space,
        {
            str(MVAUDesignPaths.REPETITIONS): 2,
            str(MVAUDesignPaths.MATRIX_WIDTH): 4,
            str(MVAUDesignPaths.MATRIX_HEIGHT): 4,
            str(MVAUDesignPaths.ACTIVATION_ELEMENT_TYPE): INT8,
            str(MVAUDesignPaths.WEIGHT_ELEMENT_TYPE): INT8,
            str(MVAUDesignPaths.OUTPUT_ELEMENT_TYPE): INT16,
        },
    )
    point = engine.commit_assignments(point, {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 2}).point
    expected = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    assert engine.query_property(point, MVAUDesignPaths.REGION) == Decided(expected)
    assert engine.check_readiness(point, "model_structural").ready is True
