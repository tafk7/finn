# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One toolchain per flow: the dataflow builder, PrepareForLinking and FIFO sizing
hand the toolchain they are given (or the one they prepare) to each
transformation that runs a vendor tool. The builder's is the selection its
configuration names, or the machine's when it names none, which also prepares
build_dataflow_directory's build process.

The tool steps are replaced by recorders, each checking its call against the
real constructor; where a real run is needed, the tools in the toolchain's
command directory are fakes (tests/util/conftest.py). No Vivado and no Vitis HLS. ZynqBuild's own
threading is tests/kernel_ops/test_zynq_build.py.
"""

from __future__ import annotations

import pytest

import inspect
import json
import numpy as np
import os
import subprocess
import sys
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model

from finn.builder import build_dataflow, build_dataflow_steps
from finn.builder.build_dataflow_config import (
    AutoFIFOSizingMethod,
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)
from finn.transformation.fpgadataflow import alveo_build, set_fifo_depths
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util import hls
from finn.util.toolchain import Selection, machine_selection

pytestmark = pytest.mark.util

ALVEO_PART = "xcu250-figd2104-2L-e"


def recorder(module, name, seen, tmp_path):
    """A stand-in for ``module.name`` that checks its arguments against the real
    constructor, records the toolchain it is given, and leaves the metadata the
    builder reads after it (a stitched-IP project, a driver directory)."""
    signature = inspect.signature(getattr(module, name).__init__)

    class Recorded(Transformation):
        def __init__(self, *args, **kwargs):
            super().__init__()
            bound = signature.bind(None, *args, **kwargs)
            seen.append((name, bound.arguments.get("toolchain")))

        def apply(self, model):
            if name in ("CreateStitchedIP", "MakeCPPDriver"):
                directory = tmp_path / name
                directory.mkdir(exist_ok=True)
                key = "vivado_stitch_proj" if name == "CreateStitchedIP" else "cpp_driver_dir"
                model.set_metadata_prop(key, str(directory))
            return model, False

    return Recorded


class Passed(Transformation):
    """A step that changes nothing (code generation, the export after stitching)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def apply(self, model):
        return model, False


def identity_model():
    """A model with no hardware nodes: every step runs, only the tool steps act."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["inp"], ["outp"], name="Identity_0")
    return ModelWrapper(qonnx_make_model(helper.make_graph([node], "identity", [inp], [outp])))


def mvau_model():
    """One MVAU, 8 inputs to 4 outputs, INT2 weights and inputs, no activation."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["outp"],
        name="MVAU_0",
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=8,
        MH=4,
        SIMD=2,
        PE=2,
        inputDataType="INT2",
        weightDataType="INT2",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        preferred_impl_style="hls",
        mem_mode="internal_embedded",
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "mvau", [inp], [outp])))
    model.set_tensor_datatype("inp", DataType["INT2"])
    model.set_tensor_datatype("outp", DataType["INT32"])
    model.set_initializer("weights", np.ones((8, 4), dtype=np.float32))
    model.set_tensor_datatype("weights", DataType["INT2"])
    return model


def builder_config(tmp_path, **settings):
    return DataflowBuildConfig(
        **{
            "output_dir": str(tmp_path / "output"),
            "synth_clk_period_ns": 5.0,
            "board": "U250",
            "shell_flow_type": ShellFlowType.VITIS_ALVEO,
            "generate_outputs": [],
            **settings,
        }
    )


#: The builder's transformations that run a vendor tool.
BUILDER_TOOL_STEPS = (
    "HLSSynthIP",
    "InsertAndSetFIFODepths",
    "CreateStitchedIP",
    "ZynqBuild",
    "PrepareForLinking",
    "VitisLink",
    "SlashLink",
    "MakeCPPDriver",
)

#: The tool steps the builder run below reaches, in order.
BUILDER_ORDER = [
    "HLSSynthIP",  # step_hw_ipgen
    "HLSSynthIP",  # step_set_fifo_depths, characterized
    "HLSSynthIP",
    "HLSSynthIP",  # step_set_fifo_depths, sizes from the folding config
    "InsertAndSetFIFODepths",  # step_set_fifo_depths, by simulation
    "HLSSynthIP",
    "CreateStitchedIP",  # step_create_stitched_ip
    "CreateStitchedIP",  # step_export_portable_rtl
    "MakeCPPDriver",  # step_make_driver
    "ZynqBuild",  # step_synthesize_bitfile: Zynq
    "PrepareForLinking",  # Vitis
    "VitisLink",
    "PrepareForLinking",  # SLASH
    "SlashLink",
    "HLSSynthIP",  # step_loop_body_set_fifo_depths
    "InsertAndSetFIFODepths",
    "HLSSynthIP",  # step_loop_body_ipgen_and_stitch
    "CreateStitchedIP",
]


def test_a_build_prepares_one_toolchain_and_runs_every_tool_step_by_it(monkeypatch, tmp_path):
    """Every builder step that runs a tool, in each branch that runs one, over one
    build configuration: each tool step receives the configuration's toolchain,
    prepared once."""
    seen = []
    for name in BUILDER_TOOL_STEPS:
        monkeypatch.setattr(
            build_dataflow_steps, name, recorder(build_dataflow_steps, name, seen, tmp_path)
        )
    # Code generation, and the characterization and FIFO insertion around HLS
    # synthesis (which a model without hardware nodes cannot run).
    for name in (
        "PrepareIP",
        "ExportPortableRTL",
        "PrepareRTLSim",
        "DeriveCharacteristic",
        "DeriveFIFOSizes",
        "InsertFIFO",
    ):
        monkeypatch.setattr(build_dataflow_steps, name, Passed)
    monkeypatch.setattr(
        build_dataflow_steps, "dataflow_performance", lambda model: {"max_cycles": 0}
    )
    # The bitfile step's reports, copied from where the shell build left them.
    monkeypatch.setattr(build_dataflow_steps, "copy", lambda *args: None)
    monkeypatch.setattr(build_dataflow_steps, "post_synth_res", lambda model: {})
    monkeypatch.setattr(build_dataflow_steps, "delivered_clock", lambda *args: {})
    toolchain = object()
    prepared = []

    def prepare(selection):
        prepared.append(selection)
        return toolchain

    monkeypatch.setattr(Selection, "prepare", prepare)
    steps = build_dataflow_steps
    cfg = builder_config(tmp_path)

    def run(step, **settings):
        for key, value in settings.items():
            setattr(cfg, key, value)
        step(identity_model(), cfg)

    run(steps.step_hw_ipgen)
    run(
        steps.step_set_fifo_depths,
        auto_fifo_depths=True,
        auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
    )
    run(steps.step_set_fifo_depths, auto_fifo_depths=False)
    run(
        steps.step_set_fifo_depths,
        auto_fifo_depths=True,
        auto_fifo_strategy=AutoFIFOSizingMethod.LARGEFIFO_RTLSIM,
    )
    run(steps.step_create_stitched_ip, generate_outputs=[DataflowOutputType.STITCHED_IP])
    run(steps.step_export_portable_rtl, generate_outputs=[DataflowOutputType.PORTABLE_RTL])
    run(steps.step_make_driver, generate_outputs=[DataflowOutputType.CPP_DRIVER])
    for flow, board in (
        (ShellFlowType.VIVADO_ZYNQ, "Pynq-Z1"),
        (ShellFlowType.VITIS_ALVEO, "U250"),
        (ShellFlowType.SLASH_ALVEO, "U250"),
    ):
        run(
            steps.step_synthesize_bitfile,
            generate_outputs=[DataflowOutputType.BITFILE],
            shell_flow_type=flow,
            board=board,
            enable_hw_sim=True,
        )
    run(steps.step_loop_body_set_fifo_depths)
    run(steps.step_loop_body_ipgen_and_stitch)
    assert [name for name, _ in seen] == BUILDER_ORDER
    assert all(given is toolchain for _, given in seen)
    assert prepared == [machine_selection()]


def test_the_toolchain_is_the_configured_selection_prepared_on_first_use(monkeypatch):
    prepared = []

    def prepare(selection):
        prepared.append((selection, object()))
        return prepared[-1][1]

    monkeypatch.setattr(Selection, "prepare", prepare)
    # A stated selection is used as stated: the machine's command directory and
    # the legacy frontend variable select nothing.
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", "/site/tools")
    monkeypatch.setenv("FINN_HLS_FRONTEND", "vivado_hls")
    selection = Selection(settings=("/opt/xilinx/settings64.sh",), hls_frontend="vitis-run")
    cfg = DataflowBuildConfig(
        output_dir="out", synth_clk_period_ns=5.0, generate_outputs=[], toolchain=selection
    )
    assert prepared == []
    assert cfg._resolve_toolchain() is cfg._resolve_toolchain() is prepared[0][1]
    assert [named for named, _ in prepared] == [selection]
    # Unset, it is the machine's: the environment as configured, under the site
    # command directory.
    unset = DataflowBuildConfig(output_dir="out", synth_clk_period_ns=5.0, generate_outputs=[])
    assert unset.toolchain is None
    assert unset._resolve_selection() == Selection(command_dir="/site/tools")
    assert unset._resolve_toolchain() is prepared[1][1]


def test_the_toolchain_selection_round_trips_through_the_json_config(monkeypatch):
    selection = Selection(command_dir="/site/bin", launcher=("ssh", "build"))
    cfg = DataflowBuildConfig(
        output_dir="out", synth_clk_period_ns=5.0, generate_outputs=[], toolchain=selection
    )
    stated = json.loads(cfg.to_json())["toolchain"]
    assert stated == {
        "settings": [],
        "command_dir": "/site/bin",
        "launcher": ["ssh", "build"],
        "hls_frontend": "vitis_hls",
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
    unset = DataflowBuildConfig(output_dir="out", synth_clk_period_ns=5.0, generate_outputs=[])
    unset._resolve_toolchain()
    assert json.loads(unset.to_json())["toolchain"] is None
    assert DataflowBuildConfig.from_json(unset.to_json()) == unset


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
    cfg = builder_config(tmp_path, toolchain=selection)
    (directory / "dataflow_build_config.json").write_text(cfg.to_json())
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


#: A Vitis HLS that reports 2024.2 and, given a node's script, makes the IP
#: directory HLSBackend checks for and notes that it ran.
FAKE_VITIS_HLS = """
import os, sys
if sys.argv[1] == "-version":
    print("Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)")
    sys.exit()
name = os.path.basename(sys.argv[2])[len("hls_syn_") : -len(".tcl")]
os.makedirs(f"project_{name}/sol1/impl/ip")
open("synthesized_here", "w").close()
"""


def legacy_refused():
    # An Exception, not pytest.fail: it is raised in a pool worker, which passes
    # an Exception back to the parent and dies on a BaseException.
    raise AssertionError("the legacy toolchain was prepared")


def test_the_builder_runs_hls_synthesis_in_its_prepared_toolchain(monkeypatch, tmp_path):
    """step_hw_codegen and step_hw_ipgen for real on an MVAU, in two workers (the
    toolchain reaches them pickled): its synthesis runs in the toolchain the build
    configuration names, whose command directory holds only a fake Vitis HLS."""
    tools = tmp_path / "tools"
    tools.mkdir()
    vitis_hls = tools / "vitis_hls"
    vitis_hls.write_text("#!" + sys.executable + "\n" + FAKE_VITIS_HLS)
    vitis_hls.chmod(0o755)
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "2")
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setattr(hls, "legacy_toolchain", legacy_refused)
    cfg = builder_config(
        tmp_path, fpga_part=ALVEO_PART, toolchain=Selection(command_dir=str(tools))
    )
    model = mvau_model().transform(SpecializeLayers(ALVEO_PART))
    model = build_dataflow_steps.step_hw_codegen(model, cfg)
    model = build_dataflow_steps.step_hw_ipgen(model, cfg)
    (node,) = model.graph.node
    mvau = getCustomOp(node)
    code = Path(mvau.get_nodeattr("code_gen_dir_ipgen"))
    assert (code / "synthesized_here").is_file()
    assert mvau.get_nodeattr("ipgen_path") == f"{code}/project_{node.name}"


#: PrepareForLinking's tool steps, per partition (an IODMA, the MVAU, an IODMA):
#: HLS synthesis, the stitched IP, and for Vitis its object file.
LINKING_ORDER = {
    "vitis-xrt": ["HLSSynthIP", "CreateStitchedIP", "CreateVitisXO"] * 3,
    "slash-vrt": ["HLSSynthIP", "CreateStitchedIP"] * 3,
}


def prepared_for_linking(monkeypatch, tmp_path, platform, toolchain):
    """PrepareForLinking over one MVAU for ``platform``, its tool steps recorded:
    the order and toolchain of each."""
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    seen = []
    for name in ("HLSSynthIP", "CreateStitchedIP", "CreateVitisXO"):
        monkeypatch.setattr(alveo_build, name, recorder(alveo_build, name, seen, tmp_path))
    monkeypatch.setattr(alveo_build, "PrepareIP", Passed)
    preparation = PrepareForLinking(
        ALVEO_PART,
        5.0,
        platform,
        partition_model_dir=str(tmp_path / "partitions"),
        toolchain=toolchain,
    )
    mvau_model().transform(preparation)
    return seen


@pytest.mark.parametrize("platform", sorted(LINKING_ORDER))
def test_linking_runs_its_tools_through_the_toolchain_it_is_given(monkeypatch, tmp_path, platform):
    given = object()
    monkeypatch.setattr(alveo_build, "legacy_toolchain", lambda: pytest.fail("prepared"))
    seen = prepared_for_linking(monkeypatch, tmp_path, platform, given)
    assert seen == [(name, given) for name in LINKING_ORDER[platform]]


@pytest.mark.parametrize("platform", sorted(LINKING_ORDER))
def test_linking_prepares_its_default_toolchain_once(monkeypatch, tmp_path, platform):
    prepared = []

    def legacy_toolchain():
        prepared.append(object())
        return prepared[-1]

    monkeypatch.setattr(alveo_build, "legacy_toolchain", legacy_toolchain)
    seen = prepared_for_linking(monkeypatch, tmp_path, platform, None)
    assert len(prepared) == 1
    assert [name for name, _ in seen] == LINKING_ORDER[platform]
    assert all(toolchain is prepared[0] for _, toolchain in seen)


class Simulated(Exception):
    pass


@pytest.mark.parametrize("given", [True, False], ids=["given", "default"])
def test_fifo_sizing_synthesizes_and_stitches_in_one_toolchain(monkeypatch, tmp_path, given):
    """InsertAndSetFIFODepths up to its simulation: its HLS synthesis, its
    stitched IP and its simulation receive the toolchain it is given, or the one
    it prepares."""
    seen = []
    for name in ("HLSSynthIP", "CreateStitchedIP"):
        monkeypatch.setattr(set_fifo_depths, name, recorder(set_fifo_depths, name, seen, tmp_path))
    monkeypatch.setattr(set_fifo_depths, "PrepareIP", Passed)

    def xsi_fifosim(*args, toolchain, **kwargs):
        seen.append(("xsi_fifosim", toolchain))
        raise Simulated

    monkeypatch.setattr(set_fifo_depths, "xsi_fifosim", xsi_fifosim)
    prepared = []

    def legacy_toolchain():
        prepared.append(object())
        return prepared[-1]

    monkeypatch.setattr(set_fifo_depths, "legacy_toolchain", legacy_toolchain)
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    model = mvau_model().transform(SpecializeLayers(ALVEO_PART))
    toolchain = object() if given else None
    with pytest.raises(Simulated):
        model.transform(
            set_fifo_depths.InsertAndSetFIFODepths(ALVEO_PART, 5.0, toolchain=toolchain)
        )
    expected = toolchain if given else prepared[0]
    assert len(prepared) == (0 if given else 1)
    assert seen == [
        ("HLSSynthIP", expected),
        ("CreateStitchedIP", expected),
        ("xsi_fifosim", expected),
    ]


@pytest.fixture
def two_routes(fake_tools):
    """A build configuration's command directory and the machine's
    (FINN_TOOL_DIR_OVERRIDE, set), each of fake tools."""
    return fake_tools("configured"), fake_tools("machine", machine=True)


def simulation_config(tmp_path, configured, *verification, **settings):
    return builder_config(
        tmp_path,
        fpga_part=ALVEO_PART,
        toolchain=Selection(command_dir=str(configured.directory)),
        verify_steps=list(verification),
        verify_input_npy="unused.npy",
        verify_expected_output_npy="unused.npy",
        **settings,
    )


def stitched(model, tmp_path):
    """The metadata a stitched IP project leaves, for a project of one wrapper."""
    project = tmp_path / "stitched"
    project.mkdir()
    wrapper = project / "finn_design_wrapper.v"
    wrapper.write_text("")
    (project / "all_verilog_srcs.txt").write_text(str(wrapper))
    model.set_metadata_prop("vivado_stitch_proj", str(project))
    model.set_metadata_prop("wrapper_filename", str(wrapper))
    model.set_metadata_prop(
        "vivado_stitch_ifnames",
        json.dumps({"s_axis": [["s_axis_0", 4]], "m_axis": [["m_axis_0", 32]], "aximm": []}),
    )
    return model


def test_the_builder_simulates_in_its_toolchain_not_the_machine_default(
    monkeypatch, tmp_path, two_routes
):
    """Each builder step that compiles a simulation (cppsim, node-by-node and
    stitched-IP rtlsim, the rtlsim performance run) compiles it by the build
    configuration's toolchain; the machine's command directory sees no call.
    Verification itself is not run: the fake tools build nothing that runs."""
    configured, machine = two_routes
    verified = []
    monkeypatch.setattr(
        build_dataflow_steps, "verify_step", lambda model, cfg, name, **kw: verified.append(name)
    )
    steps = build_dataflow_steps
    cfg = simulation_config(
        tmp_path,
        configured,
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
        minimize_bit_width=False,
    )
    model = mvau_model().transform(SpecializeLayers(ALVEO_PART))
    model = steps.step_minimize_bit_width(model, cfg)
    assert configured.calls == ["g++"]
    model = steps.step_hw_codegen(model, cfg)
    model = steps.step_hw_ipgen(model, cfg)
    assert configured.calls[1:] == ["vitis_hls", "vitis_hls", "vivado", "xelab"]
    model = steps.step_create_stitched_ip(stitched(model, tmp_path), cfg)
    assert configured.calls[5:] == ["xelab"]
    cfg.generate_outputs = [DataflowOutputType.STITCHED_IP, DataflowOutputType.RTLSIM_PERFORMANCE]
    monkeypatch.setattr(steps, "CreateStitchedIP", Passed)
    monkeypatch.setattr(steps, "copy", lambda *args: None)
    monkeypatch.setattr(steps.shutil, "copytree", lambda *args, **kwargs: None)
    # With a waveform, too, the step leaves the process's environment alone.
    cfg.verify_save_rtlsim_waveforms = True
    environment = dict(os.environ)
    steps.step_measure_rtlsim_performance(model, cfg)
    assert dict(os.environ) == environment
    assert configured.calls[6:] == ["xelab", "g++"]
    report = json.loads((Path(cfg.output_dir) / "report/rtlsim_performance.json").read_text())
    assert report["cycles"] == 100
    assert verified == ["folded_hls_cppsim", "node_by_node_rtlsim", "stitched_ip_rtlsim"]
    assert machine.calls == []


def test_a_build_with_no_toolchain_stated_runs_under_the_site_directory(
    monkeypatch, tmp_path, fake_tools
):
    """Unset, the configuration's toolchain is the machine's: its cppsim compile
    runs under FINN_TOOL_DIR_OVERRIDE's command directory."""
    site = fake_tools("site", machine=True)
    monkeypatch.setattr(build_dataflow_steps, "verify_step", lambda *args, **kwargs: None)
    cfg = builder_config(
        tmp_path,
        fpga_part=ALVEO_PART,
        verify_steps=["folded_hls_cppsim"],
        verify_input_npy="unused.npy",
        verify_expected_output_npy="unused.npy",
        minimize_bit_width=False,
    )
    model = mvau_model().transform(SpecializeLayers(ALVEO_PART))
    build_dataflow_steps.step_minimize_bit_width(model, cfg)
    assert site.calls == ["g++"]
