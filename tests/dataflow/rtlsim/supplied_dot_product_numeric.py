############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""D6a RTL parity for DotProduct with FINN RTL memstream weight supply."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import cast

import numpy as np  # type: ignore[import-not-found]

from dataflow.rtlsim.composed_mvau_equiv import record_identity
from dataflow.rtlsim.composed_mvau_numeric import (
    Case,
    MVAUDspBlock,
    _activations,
    _model,
    _weights,
    activation_beats,
    golden,
    unpack_output,
)
from dataflow.rtlsim.rtl_transport import drive
from finn.dataflow.design import Decided, Engine
from finn.dataflow.mvau.associations import MVAUSourceAssociation
from finn.dataflow.mvau.designs.dot_product import (
    MVAU_DOT_PRODUCT_DESIGN,
    compose_dot_product_design,
)
from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    write_decomposed_artifact,
)
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.supplied_artifacts import (
    build_supplied_artifact_requirements,
)
from finn.dataflow.mvau.input_supply import FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.mvau.source import MVAUResolvedDesign, MVAUSourceProjection
from finn.dataflow.mvau_problem import (
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.ops.mvau import NetworkRef
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle

CASE = Case(
    "dot_product_memstream",
    MVAUDspBlock.DSP58,
    2,
    4,
    6,
    2,
    2,
    "INT8",
    "INT8",
    "INT32",
    "negative",
    "narrow",
    "the supplied path must preserve exact MVAU arithmetic and backpressure",
)


def requirements_for(
    weights: np.ndarray,
    root: Path,
    case: Case = CASE,
    *,
    pumped_memory: bool = False,
) -> MVAUDecomposedArtifactRequirements:
    """Build the production supplied artifact used by every D6a hardware gate."""

    assembly = MVAU_DOT_PRODUCT_DESIGN
    source_description = MVAUSourceDescription(
        "mvau_dot_product_memstream",
        "activation",
        "weights",
        "output",
        (case.repetitions,),
    )
    facts = {
        MVAUProblemPaths.REPETITIONS: case.repetitions,
        MVAUProblemPaths.MATRIX_WIDTH: case.matrix_width,
        MVAUProblemPaths.MATRIX_HEIGHT: case.matrix_height,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: case.activation_type,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: case.weight_type,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: case.accumulator_type,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: case.accumulator_type,
        MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUProblemPaths.SOURCE_DESCRIPTION: source_description,
        MVAUProblemPaths.TARGET_DSP_BLOCK: case.target,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 4.0,
        MVAUProblemPaths.TARGET_FPGA_PART: case.fpga_part,
    }
    engine = Engine()
    point = engine.start(engine.validate(assembly.specification), facts)
    point = engine.commit_assignments(
        point,
        {
            assembly.semantics.pe.path: case.pe,
            assembly.semantics.simd.path: case.simd,
            assembly.compute_pumping.path: case.pumping,
            assembly.input_supply.declaration.choice.path: FINN_RTL_MEMSTREAM_SUPPLY,
            assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
            assembly.input_supply.settings.pumped_memory.path: pumped_memory,
        },
    ).point
    realized = assembly.inventory.realize(engine, point)
    if not isinstance(realized, Decided):
        print(f"realization failed: {realized.findings}")
        return 1
    realization = realized.value
    association_answer = engine.query_property(point, assembly.source_association.path)
    if not isinstance(association_answer, Decided):
        print(f"source association failed: {association_answer.findings}")
        return 1
    association = cast(MVAUSourceAssociation, association_answer.value)
    resolved = MVAUResolvedDesign(
        engine,
        point,
        NetworkRef("mvau", realization.network, association),
        association,
        "mvau_dot_product_memstream",
        MVAUSourceProjection(source_description, facts, {}),
    )
    elaboration = compose_dot_product_design(resolved, realization)
    return build_supplied_artifact_requirements(
        resolved,
        realization,
        elaboration,
        weights,
        root,
    )


def main() -> int:
    root = Path(os.environ["FINN_ROOT"])
    record_identity(str(root), str(finnlib_root(root)))
    generator = np.random.RandomState(0xD6A)
    weights = _weights(CASE, generator)
    activations = _activations(CASE, generator)
    expected = golden(_model(CASE, weights), activations, weights)
    expected_beats = CASE.repetitions * CASE.neuron_folds
    for pumped_memory in (False, True):
        requirements = requirements_for(weights, root, pumped_memory=pumped_memory)
        values = dict(requirements.parameters)
        output_width = int(values["ACCU_WIDTH"])
        activation_width = int(values["ACTIVATION_WIDTH"])
        pumping = "pumped" if pumped_memory else "unpumped"
        for stalls in (False, True):
            mode = "stalled" if stalls else "free-running"
            with tempfile.TemporaryDirectory() as scratch:
                written = write_decomposed_artifact(requirements, scratch)
                sources = [path for path in written if Path(path).suffix in {".v", ".sv"}]
                observed_beats = drive(
                    requirements.top_module_name,
                    sources,
                    {"in0": activation_beats(CASE, activations, activation_width)},
                    expected_beats,
                    stalls=stalls,
                    data_files=dict(requirements.data_files),
                )
            observed = unpack_output(CASE, observed_beats, output_width)
            if not np.array_equal(observed, expected):
                print(f"{pumping} {mode}: FAIL\nexpected={expected}\nobserved={observed}")
                return 1
            print(f"{pumping} {mode}: PASS ({expected_beats} output beats)")
    print("RESULT: D6A MEMSTREAM NUMERIC PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
