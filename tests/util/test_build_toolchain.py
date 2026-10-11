# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One toolchain per build: the selection a build configuration names, laid over the
machine's (or the machine's when it names none), prepared once, in the process that
builds; a build directory's configuration, its relative paths naming the directory's
files. No Vivado and no Vitis HLS."""

from __future__ import annotations

import pytest

import json
import os
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn import resources
from finn.builder import build_dataflow
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.platform import TargetRequest
from finn.toolchain.launcher import Selection, host_selection
from finn.toolchain.vivado import vivado_jobs
from finn.transformation.prepare import GraphPreparation

pytestmark = pytest.mark.util

#: A kernel-path target that needs nothing but the registry.
TARGET = TargetRequest(period_ns=5.0, part="xczu3eg-sbva484-1-e")


def identity_model():
    """A model with no hardware nodes."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["inp"], ["outp"], name="Identity_0")
    return ModelWrapper(qonnx_make_model(helper.make_graph([node], "identity", [inp], [outp])))


def builder_config(**settings):
    return DataflowBuildConfig(**{"output_dir": "out", "target": TARGET, **settings})


def test_the_toolchain_is_the_configured_selection_prepared_on_first_use(monkeypatch):
    prepared = []

    def prepare(selection):
        prepared.append((selection, object()))
        return prepared[-1][1]

    monkeypatch.setattr(Selection, "prepare", prepare)
    # A stated selection is laid over the machine's: what it states wins (its settings
    # and frontend), and the machine's command directory and jobs stay.
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", "/site/tools")
    monkeypatch.setenv("FINN_XILINX_VERSION", "2024.2")
    monkeypatch.setenv("FINN_VIVADO_JOBS", "6")
    selection = Selection(settings=("/opt/xilinx/settings64.sh",), hls_frontend="vitis-run")
    cfg = builder_config(toolchain=selection)
    assert prepared == []
    assert cfg._resolve_toolchain() is cfg._resolve_toolchain() is prepared[0][1]
    assert [named for named, _ in prepared] == [
        Selection(
            settings=("/opt/xilinx/settings64.sh",),
            command_dir="/site/tools",
            hls_frontend="vitis-run",
            vivado_jobs=6,
        )
    ]
    # Unset, it is the machine's: the environment as configured, under the site
    # command directory, with the release's frontend.
    unset = builder_config()
    assert unset.toolchain is None
    assert unset._resolve_selection() == Selection(
        command_dir="/site/tools", hls_frontend="vitis_hls", vivado_jobs=6
    )
    assert unset._resolve_toolchain() is prepared[1][1]


def test_a_stated_selection_drops_none_of_the_machines_fields(monkeypatch):
    """Stating one field of the toolchain changes that field only: the machine's tool
    route (ON4) and HLS frontend stay, and a field the selection states wins over the
    machine's, an empty command directory included."""
    machine = {"FINN_XILINX_ENV": "", "FINN_TOOL_DIR_OVERRIDE": "/site/tools"}
    machine |= {"FINN_XILINX_VERSION": "2025.2", "FINN_VIVADO_JOBS": "8"}
    for name, value in machine.items():
        monkeypatch.setenv(name, value)
    jobs = builder_config(toolchain=Selection(vivado_jobs=2))
    assert jobs._resolve_selection() == Selection(
        command_dir="/site/tools", hls_frontend="vitis-run", vivado_jobs=2
    )
    local = Selection(command_dir="", hls_frontend="vitis_hls")
    assert host_selection(stated=local) == Selection(
        command_dir="", hls_frontend="vitis_hls", vivado_jobs=8
    )
    assert host_selection(stated=Selection()) == host_selection()


def test_the_toolchain_selection_round_trips_through_the_json_config(monkeypatch):
    selection = Selection(command_dir="/site/bin", launcher=("ssh", "build"))
    cfg = builder_config(toolchain=selection)
    stated = json.loads(cfg.to_json())["toolchain"]
    assert stated == {
        "settings": [],
        "command_dir": "/site/bin",
        "launcher": ["ssh", "build"],
        "hls_frontend": None,
        "vivado_jobs": None,
    }
    restored = DataflowBuildConfig.from_json(cfg.to_json())
    assert restored.toolchain == selection and restored == cfg
    # The prepared toolchain is not serialized with it.
    monkeypatch.setattr(Selection, "prepare", lambda selection: object())
    cfg._resolve_toolchain()
    assert DataflowBuildConfig.from_json(cfg.to_json()) == restored
    # Unset, it stays unset: the machine's selection is never written into the
    # configuration, which another machine may build from.
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", "/site/tools")
    unset = builder_config()
    unset._resolve_toolchain()
    assert json.loads(unset.to_json())["toolchain"] is None
    assert DataflowBuildConfig.from_json(unset.to_json()) == unset


def test_a_build_directorys_relative_paths_name_its_files(tmp_path):
    """read_build_config resolves each relative path a build directory's configuration
    states against the directory, so it names the same file from any working directory;
    an absolute path, and an empty command directory, stay as stated."""
    directory = tmp_path / "build"
    directory.mkdir()
    cfg = builder_config(
        verify_input_npy="inputs/input.npy",
        preparation=GraphPreparation(preprocessing="preproc.onnx"),
        kernel_exploration=[
            {"strategy": "pinned", "path": "kernel_choices.json"},
            {"strategy": "pinned", "path": "/elsewhere/kernel_choices.json"},
            {"strategy": "size_fifos", "margin": 2},
        ],
        toolchain=Selection(settings=("settings64.sh", "/opt/site.sh"), command_dir="tools"),
    )
    (directory / build_dataflow.CONFIG_FILE).write_text(cfg.to_json())
    read = build_dataflow.read_build_config(str(directory))
    assert read.output_dir == str(directory / "out")
    assert read.verify_input_npy == str(directory / "inputs/input.npy")
    assert read.preparation.preprocessing == str(directory / "preproc.onnx")
    assert read.kernel_exploration == [
        {"strategy": "pinned", "path": str(directory / "kernel_choices.json")},
        {"strategy": "pinned", "path": "/elsewhere/kernel_choices.json"},
        {"strategy": "size_fifos", "margin": 2},
    ]
    assert read.toolchain == Selection(
        settings=(str(directory / "settings64.sh"), "/opt/site.sh"),
        command_dir=str(directory / "tools"),
    )
    # Absolute as stated, unset, or the empty command directory: unchanged.
    stated = builder_config(output_dir="/results", toolchain=Selection(command_dir=""))
    (directory / build_dataflow.CONFIG_FILE).write_text(stated.to_json())
    read = build_dataflow.read_build_config(str(directory))
    assert read.output_dir == "/results" and read.toolchain == Selection(command_dir="")
    assert read.preparation.preprocessing is None


def test_a_build_directory_builds_in_process_its_toolchain_prepared_once(
    monkeypatch, tmp_path, capsys
):
    """build_dataflow_directory builds in the calling process, its working directory and
    environment unchanged: the toolchain its configuration names (a settings script the
    configuration states relative to the build directory) is prepared once, for the
    whole build, and kept off os.environ; a relative FINN_BUILD_DIR is the caller's."""
    directory = tmp_path / "build"
    directory.mkdir()
    (directory / "settings64.sh").write_text("export SELECTED_BY_SETTINGS=1\n")
    (directory / "model.onnx").write_bytes(identity_model().model.SerializeToString())
    cfg = builder_config(toolchain=Selection(settings=("settings64.sh",)), steps=[])
    (directory / build_dataflow.CONFIG_FILE).write_text(cfg.to_json())
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("FINN_BUILD_DIR", "scratch")
    monkeypatch.delenv("SELECTED_BY_SETTINGS", raising=False)
    prepared = []
    prepare = Selection.prepare

    def counted(selection, *args, **kwargs):
        prepared.append(prepare(selection, *args, **kwargs))
        return prepared[-1]

    monkeypatch.setattr(Selection, "prepare", counted)
    environment = dict(os.environ)
    assert build_dataflow.build_dataflow_directory("../build") == 0
    (toolchain,) = prepared
    assert toolchain.environment["SELECTED_BY_SETTINGS"] == "1"
    assert os.path.samefile(os.getcwd(), caller) and dict(os.environ) == environment
    assert (directory / "out/time_per_step.json").is_file()
    scratch = (caller / "scratch").resolve()
    assert resources.scratch() == scratch
    assert f"Intermediate outputs will be generated in {scratch}" in capsys.readouterr().out


def test_vivados_jobs_are_a_machine_setting_of_the_selection(monkeypatch):
    """SZ7, SZ11 (b): how many runs Vivado launches at once is a machine setting
    (FINN_VIVADO_JOBS), read into the machine's selection, a positive number or None (the
    machine's cores); one check refuses anything else, stated or set."""
    assert Selection(vivado_jobs=3).vivado_jobs == 3 and Selection().vivado_jobs is None
    for refused in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="positive number"):
            Selection(vivado_jobs=refused)
        with pytest.raises(ValueError, match="positive number"):
            vivado_jobs(refused)
    no_file = {"FINN_XILINX_ENV": ""}
    assert host_selection(no_file).vivado_jobs is None
    assert host_selection({**no_file, "FINN_VIVADO_JOBS": "12"}).vivado_jobs == 12
    with pytest.raises(ValueError, match="FINN_VIVADO_JOBS: Vivado's jobs must be a positive"):
        host_selection({**no_file, "FINN_VIVADO_JOBS": "0"})
    with pytest.raises(ValueError, match="FINN_VIVADO_JOBS=four is not a number of runs"):
        host_selection({**no_file, "FINN_VIVADO_JOBS": "four"})
