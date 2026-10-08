# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration (``KernelBuildConfig``): read from and written to
JSON as DataflowBuildConfig is, refusing what it does not declare, and dispatched by its
type through the one ``build_dataflow`` entry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from dataclasses_json.undefined import UndefinedParameterError
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow import (
    build_dataflow_directory,
    read_build_config,
)
from finn.builder.build_dataflow_checks import Severity, run_all_config_checks
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.kernel_build_config import (
    KernelBuildConfig,
    KernelOutputType,
    KernelVerificationStepType,
)
from finn.custom_op.kernels.base import read_target
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.platform import TargetRequest, resolve_target
from finn.shells.pynq.runner import PynqOptions
from finn.transformation.kernels import kernel_choices_config
from finn.util.toolchain import Selection
from kernel_ops.tfc import ULTRA96

STATED = {"output_dir": "out", "target": {"period_ns": 5.0, "board": "Ultra96"}}


def test_the_kernel_path_builds_on_ip_explores_nothing_and_completes_by_baseline() -> None:
    cfg = KernelBuildConfig.from_json(json.dumps(STATED))
    assert cfg.target == TargetRequest(period_ns=5.0, board="Ultra96", shell="ip")
    assert cfg.generate_outputs == [] and cfg.verify_steps == []
    assert cfg.kernel_exploration == [] and cfg.kernel_completion == "baseline"
    assert cfg.steps is None and cfg.toolchain is None


def test_a_configuration_holds_through_json() -> None:
    cfg = KernelBuildConfig(
        output_dir="out",
        target=TargetRequest(period_ns=5.0, board="Ultra96", shell="pynq"),
        generate_outputs=list(KernelOutputType),
        kernel_exploration=[
            {"strategy": "target_throughput", "fps": 1_000_000},
            {"strategy": "size_fifos"},
        ],
        kernel_exploration_fresh=True,
        kernel_completion="placeholder",
        verify_steps=list(KernelVerificationStepType),
        verify_input_npy="frames.npy",
        toolchain=Selection(
            settings=("/tools/settings64.sh",), hls_frontend="vitis-run", vivado_jobs=4
        ),
        shell_options={"enable_hw_debug": True},
        steps=["phase_kernel_path"],
        start_step="phase_kernel_path",
        stop_step="phase_kernel_path",
        save_intermediate_models=False,
        enable_build_pdb_debug=False,
        verbose=True,
    )
    written = cfg.to_json()
    assert KernelBuildConfig.from_json(written) == cfg
    stated = json.loads(written)
    assert stated["target"] == {
        "period_ns": 5.0,
        "board": "Ultra96",
        "part": None,
        "shell": "pynq",
    }
    assert stated["generate_outputs"] == [
        "stitched_ip",
        "ooc_synth",
        "bitfile",
        "pynq_driver",
        "deployment_package",
    ]
    assert stated["verify_steps"] == [
        "kernel_partition_python",
        "kernel_partition_elaboration",
        "stitched_ip_testbench",
    ]
    # How many runs Vivado launches at once is the toolchain's, a machine setting.
    assert stated["toolchain"]["vivado_jobs"] == 4
    # Debug cores are an option of the pynq shell's build (SZ7).
    assert stated["shell_options"] == {"enable_hw_debug": True}
    assert cfg._resolve_shell_options() == PynqOptions(enable_hw_debug=True)


@pytest.mark.parametrize(
    "key",
    [
        "synth_clk_period_ns",
        "board",
        "shell_flow_type",
        "target_fps",
        "fpga_part",
        # SZ7: the toolchain's selection states Vivado's jobs; nothing mutes the checks;
        # debug cores are the pynq shell's build's option (shell_options).
        "vivado_jobs",
        "mute_config_assertions",
        "enable_hw_debug",
    ],
)
def test_a_dataflow_build_config_field_is_refused_naming_it(key: str) -> None:
    with pytest.raises(UndefinedParameterError, match=key):
        KernelBuildConfig.from_json(json.dumps({**STATED, key: None}))


def test_an_undeclared_target_or_toolchain_key_is_refused_naming_it() -> None:
    target = {"period_ns": 5.0, "board": "Ultra96", "shell_flow_type": "vivado_zynq"}
    with pytest.raises(UndefinedParameterError, match="target: .*shell_flow_type"):
        KernelBuildConfig.from_json(json.dumps({**STATED, "target": target}))
    toolchain = {"hls_frontned": "vitis-run"}
    with pytest.raises(UndefinedParameterError, match="toolchain: .*hls_frontned"):
        KernelBuildConfig.from_json(json.dumps({**STATED, "toolchain": toolchain}))


@pytest.mark.parametrize(
    "shell, options, refused",
    [
        ("ip", {"enable_hw_debug": True}, "the 'ip' shell has no build to take enable_hw_debug"),
        ("pynq", {"enable_debug": True}, "the pynq shell's build has no option enable_debug"),
        ("pynq", {"enable_hw_debug": "yes"}, "the pynq shell's enable_hw_debug is a bool"),
    ],
)
def test_shell_options_the_shells_build_does_not_take_are_refused(
    shell: str, options: dict[str, object], refused: str
) -> None:
    """Before the build: the configuration's check names the option its shell refuses."""
    target = {"period_ns": 5.0, "board": "Ultra96", "shell": shell}
    cfg = KernelBuildConfig.from_json(
        json.dumps({**STATED, "target": target, "shell_options": options})
    )
    with pytest.raises(ValueError, match=refused):
        cfg._resolve_shell_options()
    (check,) = [
        check for check in run_all_config_checks(cfg).checks if check.name == "kernel_shell_options"
    ]
    assert (check.severity, check.passed) == (Severity.ERROR, False)
    assert refused in check.message


def test_an_output_the_kernel_path_does_not_make_is_refused() -> None:
    for output in ("rtlsim_performance", "estimate_reports", "cpp_driver"):
        with pytest.raises(ValueError, match=output):
            KernelBuildConfig.from_json(json.dumps({**STATED, "generate_outputs": [output]}))
    stated = {**STATED, "generate_outputs": ["stitched_ip", "ooc_synth"]}
    assert KernelBuildConfig.from_json(json.dumps(stated)).generate_outputs == [
        KernelOutputType.STITCHED_IP,
        KernelOutputType.OOC_SYNTH,
    ]


def test_a_build_directory_states_one_configuration_its_file_naming_its_type(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernel_build_config.json").write_text(json.dumps(STATED))
    assert isinstance(read_build_config(str(tmp_path)), KernelBuildConfig)
    dataflow = {"output_dir": "out", "synth_clk_period_ns": 5.0, "generate_outputs": []}
    (tmp_path / "dataflow_build_config.json").write_text(json.dumps(dataflow))
    with pytest.raises(FileNotFoundError, match="states one build configuration"):
        read_build_config(str(tmp_path))
    (tmp_path / "kernel_build_config.json").unlink()
    assert isinstance(read_build_config(str(tmp_path)), DataflowBuildConfig)
    (tmp_path / "dataflow_build_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="it has none"):
        read_build_config(str(tmp_path))


@pytest.mark.slow
def test_tfc_builds_on_ip_through_the_directory_entry(
    tmp_path: Path, tfc_streamlined: Path
) -> None:
    """build_dataflow_directory (the ``build_dataflow`` command's entry) builds a
    directory's KernelBuildConfig through the kernel path in its build process: TFC on
    the default ip shell, named by its board, explored as its Zynq build is, to its
    verified partition, whose target is the part's on ip."""
    directory = tmp_path / "build"
    directory.mkdir()
    shutil.copyfile(tfc_streamlined, directory / "model.onnx")
    cfg = KernelBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=5.0, board="Ultra96"),
        kernel_exploration=[{"strategy": "target_throughput", "fps": 1_000_000}],
        enable_build_pdb_debug=False,
    )
    (directory / "kernel_build_config.json").write_text(cfg.to_json())
    assert build_dataflow_directory(str(directory)) == 0
    output = tmp_path / "output"
    built = ModelWrapper(str(output / "intermediate_models" / "step_verify_kernel_partition.onnx"))
    _, body, _ = partition_body(built)
    assert read_target(body) == resolve_target(part=ULTRA96.part, period_ns=5.0)
    assert json.loads((output / "kernel_choices.json").read_text()) == kernel_choices_config(body)
    report = json.loads((output / "report" / "kernel_exploration.json").read_text())
    assert report["bottleneck"]["cycles"] == 196
