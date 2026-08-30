############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 8: the composed MVAU computes the right numbers.

Runs INSIDE the FINN Docker container; needs Vivado's ``xelab`` and ``finn_xsi``.

Fixture 5 drives the composed RTL and the fused core with the same random
stream and requires bit-identical output.  That is a strong claim about
continuity and, in its own words, "not to a numeric golden" -- seven
configurations agree with a reference that has itself never been checked
against arithmetic.  This fixture closes that: the golden is
``MvauDataflowOp.execute_node`` on the *same* ``ModelWrapper`` the design point
was projected from.

Both fixtures stay.  Fused-versus-composed is the only check that the two
physical readings of one Network agree, which is Phase 4's claim;
composed-versus-arithmetic is the only check that either is right.

**The golden is not computed here.**  A fixture that writes its own expectation
tests the fixture: every accumulator-width and sign subtlety would have to be
reimplemented correctly to be worth anything, and a disagreement would be with
this file rather than with FINN.

**The packing is computed here, and that is the risk.**  Fixture 5 never had to
interpret a beat -- it compared two DUTs fed the same opaque words.  Reading a
matrix into streams and back is new, and a packing error looks exactly like an
RTL error.  So the matrix opens with ``identity``: an identity weight matrix
and distinct small activations, whose correct output is the input.  Any
transposition, reversal or lane swap in the packing shows up there, on a case
where the expected answer needs no arithmetic to see.  Only then do the random
cases run.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_numeric.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zlib
from dataclasses import dataclass

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from dataflow.rtlsim.composed_mvau_equiv import CLOCK_PERIOD_NS, finn_root, record_identity
from dataflow.rtlsim.rtl_transport import drive
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import ActivationReplayKernel, DotProductKernel
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    write_decomposed_artifact,
)
from finn.dataflow.mvau.providers import elaborate_mvau
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.ops.mvau import MVAU_WEIGHT_SUPPLY_SELECTION
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp

#: A part per DSP family.  Repeated from fixture 5 rather than imported,
#: because fixture 5's table is keyed by its own ``Config`` and this one adds
#: DSP48E1 -- the generation Phase 4 corrected a rule for and never measured.
PART_FOR_TARGET = {
    MVAUDspBlock.DSP48E1: "xc7z020clg400-1",
    MVAUDspBlock.DSP48E2: "xczu3eg-sbva484-1-e",
    MVAUDspBlock.DSP58: "xcvc1902-vsva2197-2MP-e-S",
}

PASS, FAIL, SKIP = 0, 1, 2


# -- the cases -----------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One numerical case: a design point, and the stimulus that discriminates.

    Datatypes are named rather than resolved at class scope so the table reads
    as the graph annotations it becomes.  The accumulator and the output are
    one field: the core drives the accumulator straight out and coverage
    refuses anything else, so two fields would be able to disagree.
    """

    label: str
    target: MVAUDspBlock
    repetitions: int
    matrix_width: int
    matrix_height: int
    pe: int
    simd: int
    activation: str
    weight: str
    accumulator: str
    stimulus: str
    why: str
    pumping: bool = False
    part_override: str | None = None

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe

    @property
    def fpga_part(self) -> str:
        return self.part_override or PART_FOR_TARGET[self.target]

    @property
    def activation_type(self) -> object:
        return DataType[self.activation]

    @property
    def weight_type(self) -> object:
        return DataType[self.weight]

    @property
    def accumulator_type(self) -> object:
        return DataType[self.accumulator]


#: The matrix, harness-trust first.
#:
#: ``INT32`` accumulators throughout except where the case is about the
#: accumulator: an ``INT16`` accumulator over ``INT8`` operands and a
#: four-deep dot product can reach 65024, which does not fit, and a fixture
#: that overflowed would be measuring wrap-around while claiming to measure
#: arithmetic.  Fixture 5 runs ``INT16`` and never notices, because two DUTs
#: wrap identically.
CASES = (
    Case(
        "identity",
        MVAUDspBlock.DSP58,
        1,
        4,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "identity",
        "the harness itself: with an identity matrix the output is the input, "
        "so a packing error is visible without arithmetic",
    ),
    Case(
        "signed_random",
        MVAUDspBlock.DSP58,
        2,
        8,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "random",
        "the happy path, over several repetitions and both folds",
    ),
    Case(
        "signed_random_softvec",
        MVAUDspBlock.DSP48E2,
        2,
        8,
        6,
        3,
        2,
        "INT8",
        "INT8",
        "INT32",
        "random",
        "the same, on the soft-vector core rather than the packed one",
    ),
)

CASES_BY_LABEL = {case.label: case for case in CASES}


# -- the model, and the golden -------------------------------------------------


class _BuildConfig:
    """The narrow slice of a build configuration the MVAU operation reads."""

    def __init__(self, part: str) -> None:
        self.synth_clk_period_ns = CLOCK_PERIOD_NS
        self.fpga_part = part

    def _resolve_fpga_part(self) -> str:
        return self.fpga_part


def _weights(case: Case, generator: np.random.RandomState) -> np.ndarray:
    """The weight matrix this case is about, in ONNX ``(MW, MH)`` orientation."""

    shape = (case.matrix_width, case.matrix_height)
    if case.stimulus == "identity":
        assert case.matrix_width == case.matrix_height, "identity needs a square matrix"
        return np.eye(case.matrix_width, dtype=np.float32)
    low, high = int(case.weight_type.min()), int(case.weight_type.max())  # type: ignore[attr-defined]
    return generator.randint(low, high + 1, size=shape).astype(np.float32)


def _activations(case: Case, generator: np.random.RandomState) -> np.ndarray:
    """The activation matrix, in ``(repetitions, MW)``."""

    shape = (case.repetitions, case.matrix_width)
    if case.stimulus == "identity":
        # Distinct, small, and non-symmetric, so a reversed or transposed
        # packing produces a visibly different vector rather than the same one.
        return (
            (np.arange(case.repetitions * case.matrix_width) + 1).reshape(shape).astype(np.float32)
        )
    low, high = int(case.activation_type.min()), int(case.activation_type.max())  # type: ignore[attr-defined]
    return generator.randint(low, high + 1, size=shape).astype(np.float32)


def _model(case: Case, weights: np.ndarray) -> ModelWrapper:
    node_id = f"mvau_{case.label}"
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weights"],
        ["output"],
        name=node_id,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=f"{node_id}_scope",
        accDataType=case.accumulator,
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [node],
                f"numeric-{case.label}",
                [
                    helper.make_tensor_value_info(
                        "activation", TensorProto.FLOAT, [case.repetitions, case.matrix_width]
                    ),
                    helper.make_tensor_value_info(
                        "weights", TensorProto.FLOAT, [case.matrix_width, case.matrix_height]
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "output", TensorProto.FLOAT, [case.repetitions, case.matrix_height]
                    )
                ],
            ),
            producer_name="fixture8",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", case.activation_type)
    model.set_tensor_datatype("weights", case.weight_type)
    # The output *is* the accumulator; the core drives it straight out.
    model.set_tensor_datatype("output", case.accumulator_type)
    model.set_initializer("weights", weights)
    return model


def _operation(model: ModelWrapper, case: Case) -> MvauDataflowOp:
    context = MVAUDataflowBuildContext(_BuildConfig(case.fpga_part))
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    operation.initialize_dataflow_scope_id()
    operation.commit_dataflow_assignments(
        context,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            DECOMPOSED_MVAU_KERNELS.pe.path: case.pe,
            DECOMPOSED_MVAU_KERNELS.simd.path: case.simd,
            DECOMPOSED_MVAU_KERNELS.compute_pumping.path: case.pumping,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
        },
    )
    return operation


def requirements_for(case: Case, model: ModelWrapper) -> MVAUDecomposedArtifactRequirements:
    """What the compiler says this case is built from -- the production path."""

    operation = _operation(model, case)
    context = MVAUDataflowBuildContext(_BuildConfig(case.fpga_part))
    resolved = operation.resolve_dataflow(context)
    root = finn_root()
    return build_decomposed_artifact_requirements(
        resolved, elaborate_mvau(resolved), root, finnlib_root(root)
    )


def golden(model: ModelWrapper, activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``MvauDataflowOp.execute_node``, which is what FINN runs this node with.

    Not a matmul written here.  A disagreement with this is a disagreement with
    FINN's own execution of the operation, which is the only kind worth
    stopping a phase for.
    """

    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    shape = model.get_tensor_shape("output")
    context = {
        "activation": activations,
        "weights": weights,
        "output": np.zeros(shape, dtype=np.float32),
    }
    operation.execute_node(context, model.graph)
    return np.asarray(context["output"])


# -- packing -------------------------------------------------------------------
#
# The one thing this fixture derives from the RTL rather than from the compiler,
# and therefore the one thing that can be wrong here rather than there.
#
# ``dotp_axi`` declares
#
#     typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  dotp_w_t;
#     typedef logic [ 0:0  ][SIMD-1:0][ACTIVATION_WIDTH-1:0]  dotp_a_t;   // broadcasting
#     typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
#
# and a SystemVerilog packed array puts index 0 in the least significant bits,
# innermost dimension fastest.  So element ``[pe][simd]`` of a weight beat sits
# at bit ``(pe*SIMD + simd) * WEIGHT_WIDTH``, activation element ``[simd]`` at
# ``simd * ACTIVATION_WIDTH``, and output lane ``[pe]`` at ``pe * ACCU_WIDTH``.
#
# The beat *order* comes from the replay: ``LEN = SF``, ``REP = NF``, so the
# activation is streamed once per repetition and replayed for each neuron fold.
# Weight beats therefore run ``(repetition, neuron fold, synapse fold)``.


def _encode(value: int, width: int) -> int:
    """Two's complement in ``width`` bits, as the RTL reads it off the bus."""

    return value & ((1 << width) - 1)


def _decode(value: int, width: int) -> int:
    """The inverse, sign-extending the accumulator lanes on the way back."""

    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def activation_beats(case: Case, activations: np.ndarray, width: int) -> list[int]:
    beats = []
    for repetition in range(case.repetitions):
        for fold in range(case.synapse_folds):
            beat = 0
            for lane in range(case.simd):
                element = int(activations[repetition, fold * case.simd + lane])
                beat |= _encode(element, width) << (lane * width)
            beats.append(beat)
    return beats


def weight_beats(case: Case, weights: np.ndarray, width: int) -> list[int]:
    beats = []
    for _repetition in range(case.repetitions):
        for neuron in range(case.neuron_folds):
            for synapse in range(case.synapse_folds):
                beat = 0
                for lane in range(case.pe):
                    for element in range(case.simd):
                        value = int(weights[synapse * case.simd + element, neuron * case.pe + lane])
                        beat |= _encode(value, width) << ((lane * case.simd + element) * width)
                beats.append(beat)
    return beats


def unpack_output(case: Case, beats: list[int], width: int) -> np.ndarray:
    result = np.zeros((case.repetitions, case.matrix_height), dtype=np.int64)
    index = 0
    for repetition in range(case.repetitions):
        for neuron in range(case.neuron_folds):
            beat = beats[index]
            index += 1
            for lane in range(case.pe):
                raw = (beat >> (lane * width)) & ((1 << width) - 1)
                result[repetition, neuron * case.pe + lane] = _decode(raw, width)
    return result


# -- the run -------------------------------------------------------------------


def run_one(case: Case) -> int:
    print(f"\n========== fixture 8: {case.label} ==========")
    print(f"  why:   {case.why}")
    # Seeded from the label, not from ``hash``: ``PYTHONHASHSEED`` is
    # randomized per process, so a hashed seed would give a different
    # stimulus on every run and a failure nobody could reproduce.
    generator = np.random.RandomState(zlib.crc32(case.label.encode()) % (2**31))
    weights = _weights(case, generator)
    activations = _activations(case, generator)
    model = _model(case, weights)
    expectation = golden(model, activations, weights)

    requirements = requirements_for(case, model)
    values = dict(requirements.parameters)
    accumulator_width = int(values["ACCU_WIDTH"])  # type: ignore[arg-type]
    activation_width = int(values["ACTIVATION_WIDTH"])  # type: ignore[arg-type]
    weight_width = int(values["WEIGHT_WIDTH"])  # type: ignore[arg-type]

    # A wrapped accumulator would make this a test of overflow behaviour while
    # claiming to be a test of arithmetic.  Say so here rather than discover it
    # as a mismatch.
    limit = 1 << (accumulator_width - 1)
    if not (-limit <= expectation.min() and expectation.max() < limit):
        print(f"  {case.label.upper()}: FAIL (the golden does not fit ACCU_WIDTH)")
        return FAIL

    stimulus = {
        "in0": activation_beats(case, activations, activation_width),
        "in1": weight_beats(case, weights, weight_width),
    }
    expected_beats = case.repetitions * case.neuron_folds
    print(f"  top:   {requirements.top_module_name} on {requirements.target_fpga_part}")
    print(
        f"  types: {case.activation} x {case.weight} -> {case.accumulator}"
        f"  (PE={case.pe} SIMD={case.simd} R={case.repetitions})"
    )
    print(f"  beats: in0={len(stimulus['in0'])} in1={len(stimulus['in1'])} out={expected_beats}")

    ok = True
    for stalls in (False, True):
        mode = "stalled" if stalls else "free-running"
        with tempfile.TemporaryDirectory() as scratch:
            sources = list(write_decomposed_artifact(requirements, scratch))
            collected = drive(
                requirements.top_module_name,
                sources,
                stimulus,
                expected_beats,
                stalls=stalls,
            )
        if len(collected) != expected_beats:
            print(f"  {mode:12} FAIL: {len(collected)} beats, expected {expected_beats}")
            ok = False
            continue
        measured = unpack_output(case, collected, accumulator_width)
        if np.array_equal(measured, expectation.astype(np.int64)):
            print(f"  {mode:12} matches execute_node on {expectation.size} values")
            continue
        ok = False
        print(f"  {mode:12} MISMATCH")
        print(f"    expected {expectation.astype(np.int64).tolist()}")
        print(f"    measured {measured.tolist()}")
    print(f"  {case.label.upper()}: {'PASS' if ok else 'FAIL'}")
    return PASS if ok else FAIL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES_BY_LABEL))
    arguments = parser.parse_args(argv)

    root = finn_root()
    library_root = str(finnlib_root(root))
    if not os.path.isdir(os.path.join(library_root, "rtl")):
        print(f"FinnLib RTL not found under {library_root}; set FINNLIB_ROOT or fetch-repos.sh")
        return FAIL
    record_identity(root, library_root)

    cases = [CASES_BY_LABEL[arguments.case]] if arguments.case else list(CASES)
    results = [run_one(case) for case in cases]
    passed = results.count(PASS)
    failed = results.count(FAIL)
    skipped = results.count(SKIP)
    ok = failed == 0 and passed > 0
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    print("RESULT:", "FIXTURE 8 PASS" if ok else "FIXTURE 8 FAIL")
    return PASS if ok else FAIL


if __name__ == "__main__":
    sys.exit(main())
