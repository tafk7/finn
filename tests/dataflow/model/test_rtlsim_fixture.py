# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks for the Kernel-only RTL evidence harness."""

from __future__ import annotations

import ast
from pathlib import Path

from dataflow.model.rtlsim.dotp_axi_numeric import CASES, CASES_BY_LABEL, DspBlock
from dataflow.model.rtlsim.dotp_axi_synth import DEFAULT_LABELS
from dataflow.model.rtlsim.replay_buffer_numeric import CASES as REPLAY_CASES
from dataflow.model.rtlsim.replay_buffer_numeric import CASES_BY_LABEL as REPLAY_BY_LABEL
from dataflow.model.rtlsim.replay_buffer_synth import DEFAULT_LABELS as REPLAY_SYNTH_LABELS


def test_numeric_matrix_covers_all_dsp_generations_and_pumping() -> None:
    assert {case.target for case in CASES} == set(DspBlock)
    assert any(case.pumping for case in CASES)
    assert len(CASES) == 16


def test_numeric_harness_constructs_no_operation_or_design() -> None:
    path = Path(__file__).parent / "rtlsim/dotp_axi_numeric.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = (
        "finn.dataflow.op",
        "finn.dataflow.ops",
        "finn.dataflow.authoring",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )


def test_synthesis_matrix_covers_each_dsp_generation() -> None:
    selected = {CASES_BY_LABEL[label].target for label in DEFAULT_LABELS}
    assert selected == set(DspBlock)


def test_replay_matrix_covers_identity_and_repeated_neuron_folds() -> None:
    assert {case.label for case in REPLAY_CASES} >= {"identity", "three_folds"}
    assert any(case.neuron_folds == 1 for case in REPLAY_CASES)
    assert any(case.neuron_folds > 2 for case in REPLAY_CASES)
    assert any(case.repetitions > 1 for case in REPLAY_CASES)
    assert {case.activation for case in REPLAY_CASES} >= {"INT8", "UINT8"}


def test_replay_synthesis_matrix_is_a_subset_of_the_numeric_matrix() -> None:
    assert set(REPLAY_SYNTH_LABELS) <= set(REPLAY_BY_LABEL)


def test_replay_harness_constructs_no_operation_or_design() -> None:
    path = Path(__file__).parent / "rtlsim/replay_buffer_numeric.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = (
        "finn.dataflow.op",
        "finn.dataflow.ops",
        "finn.dataflow.authoring",
        "finn.dataflow.model.design",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )
