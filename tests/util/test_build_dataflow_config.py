# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reading a DataflowBuildConfig refuses the keys it does not declare, naming them.

A dropped key is a silent change of build: a configuration that still said a field's
old name would build with the new field's default. The kernel path's fields
(``kernel_exploration``, ``kernel_completion``) are its own configuration's,
KernelBuildConfig, and are refused here."""

import pytest

import json
from dataclasses_json.undefined import UndefinedParameterError
from pathlib import Path

import finn.builder
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.toolchain import Selection

STATED = {"output_dir": "out", "synth_clk_period_ns": 5.0, "generate_outputs": []}


def test_a_kernel_path_field_is_refused_from_json_naming_it():
    stated = {**STATED, "kernel_exploration": [{"strategy": "size_fifos"}]}
    with pytest.raises(UndefinedParameterError, match="kernel_exploration"):
        DataflowBuildConfig.from_json(json.dumps(stated))


def test_an_undeclared_key_is_refused_from_a_dict_naming_it():
    with pytest.raises(UndefinedParameterError, match="kernel_completion"):
        DataflowBuildConfig.from_dict({**STATED, "kernel_completion": "baseline"})


def test_an_undeclared_toolchain_key_is_refused_naming_it():
    stated = {**STATED, "toolchain": {"hls_frontned": "vitis-run"}}
    with pytest.raises(UndefinedParameterError, match="toolchain.*hls_frontned"):
        DataflowBuildConfig.from_json(json.dumps(stated))


def test_declared_keys_are_read():
    stated = {
        **STATED,
        "target_fps": 100000,
        "toolchain": {"settings": ["/tools/settings64.sh"], "hls_frontend": "vitis-run"},
    }
    cfg = DataflowBuildConfig.from_json(json.dumps(stated))
    assert cfg.target_fps == 100000
    assert cfg.toolchain == Selection(settings=("/tools/settings64.sh",), hls_frontend="vitis-run")


def test_the_example_configuration_is_read():
    """The build_dataflow example FINN ships (``finn.qnn-data``) states only declared keys."""
    example = Path(finn.builder.__file__).parents[1] / "qnn-data/build_dataflow"
    stated = (example / "dataflow_build_config.json").read_text()
    assert DataflowBuildConfig.from_json(stated).output_dir
