# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One toolchain per build: the selection a build configuration names, laid over the
machine's (or the machine's when it names none), prepared once, which also prepares
build_dataflow_directory's build process. No Vivado and no Vitis HLS."""

from __future__ import annotations

import pytest

import json
import subprocess
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.builder import build_dataflow
from finn.builder.kernel_build_config import KernelBuildConfig
from finn.platform import TargetRequest
from finn.util.toolchain import Selection, machine_selection
from finn.util.vivado import vivado_jobs

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
    return KernelBuildConfig(**{"output_dir": "out", "target": TARGET, **settings})


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
    assert machine_selection(stated=local) == Selection(
        command_dir="", hls_frontend="vitis_hls", vivado_jobs=8
    )
    assert machine_selection(stated=Selection()) == machine_selection()


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
    restored = KernelBuildConfig.from_json(cfg.to_json())
    assert restored.toolchain == selection and restored == cfg
    # The prepared toolchain is not serialized with it.
    monkeypatch.setattr(Selection, "prepare", lambda selection: object())
    cfg._resolve_toolchain()
    assert KernelBuildConfig.from_json(cfg.to_json()) == restored
    # Unset, it stays unset: the machine's selection is never written into the
    # configuration, which another machine may build from.
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", "/site/tools")
    unset = builder_config()
    unset._resolve_toolchain()
    assert json.loads(unset.to_json())["toolchain"] is None
    assert KernelBuildConfig.from_json(unset.to_json()) == unset


#: The parent's toolchain variables: none, or another Vivado than the selected one.
PARENT_TOOLCHAINS = {
    "clean": {},
    "stale": {"XILINX_VIVADO": "/parent/Vivado"},
}


@pytest.mark.parametrize("parent", sorted(PARENT_TOOLCHAINS))
def test_the_build_process_runs_in_the_configured_selection(monkeypatch, tmp_path, parent):
    """build_dataflow_directory prepares its build process's environment from the
    selection its JSON configuration names: the settings script sourced over the
    parent's environment, the selected Vivado's simulator libraries on the loader path."""
    vivado = tmp_path / "Vivado"
    (vivado / "lib/lnx64.o").mkdir(parents=True)
    settings = tmp_path / "settings64.sh"
    settings.write_text(f"export XILINX_VIVADO={vivado}\nexport SELECTED_BY_SETTINGS=1\n")
    selection = Selection(settings=(str(settings),), hls_frontend="vitis-run")
    directory = tmp_path / "build"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(identity_model().model.SerializeToString())
    cfg = builder_config(toolchain=selection)
    (directory / "kernel_build_config.json").write_text(cfg.to_json())
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "finn_build"))
    monkeypatch.setenv("FINN_RESOURCES_FINNLIB", "/parent/finnlib")
    for variable in ("XILINX_VIVADO", "XILINX_VITIS", "XILINX_HLS"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in PARENT_TOOLCHAINS[parent].items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    children = []

    def run(argv, cwd, env):
        children.append((cwd, env))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(build_dataflow.subprocess, "run", run)
    assert build_dataflow.build_dataflow_directory(str(directory)) == 0
    ((cwd, env),) = children
    assert cwd == str(directory)
    assert env["SELECTED_BY_SETTINGS"] == "1"
    assert env["XILINX_VIVADO"] == str(vivado)
    assert env["LD_LIBRARY_PATH"] == str(vivado / "lib/lnx64.o")
    assert env["FINN_BUILD_DIR"] == str(tmp_path / "finn_build")
    assert env["FINN_RESOURCES_FINNLIB"] == "/parent/finnlib"


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
    assert machine_selection(no_file).vivado_jobs is None
    assert machine_selection({**no_file, "FINN_VIVADO_JOBS": "12"}).vivado_jobs == 12
    with pytest.raises(ValueError, match="FINN_VIVADO_JOBS: Vivado's jobs must be a positive"):
        machine_selection({**no_file, "FINN_VIVADO_JOBS": "0"})
    with pytest.raises(ValueError, match="FINN_VIVADO_JOBS=four is not a number of runs"):
        machine_selection({**no_file, "FINN_VIVADO_JOBS": "four"})
