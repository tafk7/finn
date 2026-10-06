"""Synthetic execution tests; these do not claim licensed AMD operation coverage."""
import pytest

import concurrent.futures
import dataclasses
import json
import os
import pickle
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from finn import resources
from finn.util._legacy_build_env import build_environment
from finn.util._legacy_build_env import toolchain as legacy_toolchain
from finn.util.hls import CallHLS
from finn.util.resources import tcl_quote
from finn.util.toolchain import Selection, Toolchain, run_process


def executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def test_cpp_builder_argv_and_failures(tmp_path):
    from finn.util.basic import CppBuilder  # noqa: PLC0415

    tools = tmp_path / "tool dir"
    tools.mkdir()
    log = tmp_path / "compiler.json"
    executable(
        tools / "g++",
        "import json,sys\n" + f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
    )
    tc = Selection(command_dir=str(tools)).prepare()
    builder = CppBuilder(toolchain=tc)
    source = tmp_path / "source $ (literal).cpp"
    source.write_text("int main() { return 0; }\n")
    builder.append_sources(source)
    builder.append_includes(["-I" + str(tmp_path / "headers $ [1]"), "-O3"])
    builder.set_executable_path(tmp_path / "result program")
    builder.build(tmp_path)
    assert json.loads(log.read_text()) == [
        "-o",
        str(tmp_path / "result program"),
        str(source),
        "-I" + str(tmp_path / "headers $ [1]"),
        "-O3",
    ]
    executable(tools / "g++", 'import sys; print("compile failed"); sys.exit(17)\n')
    with pytest.raises(subprocess.CalledProcessError) as exc:
        builder.build(tmp_path)
    assert exc.value.returncode == 17
    assert "compile failed" in (tmp_path / "compile.sh.stdout.log").read_text()


def test_zynq_and_vitis_direct_operations_preserve_scope(tmp_path, monkeypatch):
    import onnx.helper as oh  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    from finn.transformation.fpgadataflow import (  # noqa: PLC0415
        alveo_build,
        make_zynq_proj,
    )

    project = tmp_path / "project $ [1]"
    project.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "calls.jsonl"
    body = (
        "import sys,json; from pathlib import Path\n"
        f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if '-version' in sys.argv: print('AMD Vivado v2024.2'); sys.exit(0)\n"
        "for name in ['finn_zynq_link.runs/impl_1/top_wrapper.bit', "
        "'finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh', "
        "'kernel.xo', 'a.xclbin', 'synth_report.xml']:\n"
        " p=Path(name); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('artifact')\n"
    )
    executable(tools / "vivado", body)
    executable(tools / "v++", body)
    wrapper = tmp_path / "site wrapper"
    wrapper.write_text('#!/bin/bash\nexec "$@"\n')
    wrapper.chmod(0o755)
    tc = Selection(command_dir=str(tools), launcher=(str(wrapper),)).prepare(
        {
            "PATH": os.defpath,
            "XILINX_VITIS": "/selected/vitis",
            "XILINX_XRT": "/selected/xrt",
            "PLATFORM_REPO_PATHS": "/selected/platforms",
        }
    )
    monkeypatch.setattr(make_zynq_proj, "make_build_dir", lambda **_: str(project))
    monkeypatch.setattr(alveo_build, "make_build_dir", lambda **_: str(project))
    boards = []
    for number, resource in enumerate(
        r for r in resources.declarations().values() if "vivado-boards" in r.kind
    ):
        boards.append(tmp_path / f"boards {number} $[%]")
        boards[-1].mkdir()
        monkeypatch.setenv(resource.env, str(boards[-1]))
    before = dict(os.environ), os.getcwd()
    model = ModelWrapper(oh.make_model(oh.make_graph([], "empty", [], [])))
    make_zynq_proj.MakeZYNQProject("Pynq-Z1", 10, toolchain=tc).apply(model)
    # Every board repository is a Vivado board path, each a quoted Tcl word.
    config = (project / "ip_config.tcl").read_text()
    assert f"lappend paths_prop {' '.join(tcl_quote(b) for b in boards)}\n" in config
    model.set_metadata_prop("vivado_stitch_proj", str(project))
    model.set_metadata_prop(
        "vivado_stitch_ifnames",
        json.dumps(
            {
                "axilite": [],
                "aximm": [],
                "s_axis": [],
                "m_axis": [],
            }
        ),
    )
    alveo_build.CreateVitisXO("kernel", toolchain=tc).apply(model)
    alveo_build.VitisLink("platform $ literal", 10, toolchain=tc).apply(model)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    link = next(call for call in calls if "--link" in call)
    assert link[link.index("--platform") + 1] == "platform $ literal"
    assert (dict(os.environ), os.getcwd()) == before
    for script in ("synth_project.sh", "gen_xo.sh", "run_vitis_link.sh", "gen_report_xml.sh"):
        assert shlex.quote(str(wrapper)) in (project / script).read_text()


def test_xelab_compiles_without_native_import(tmp_path):
    from finn.xsi.compile import compile_sim_obj  # noqa: PLC0415

    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "vivado", "print('AMD Vivado v2024.2')\n")
    executable(
        tools / "xelab",
        "from pathlib import Path\n"
        "p=Path('xsim.dir/top/xsimk.so'); p.parent.mkdir(parents=True); p.write_bytes(b'design')\n",
    )
    source = tmp_path / "input.sv"
    source.write_text("module top(); endmodule\n")
    modules = set(sys.modules)
    directory, relative = compile_sim_obj(
        "top",
        [str(source)],
        str(tmp_path),
        toolchain=Selection(command_dir=str(tools)).prepare(),
    )
    assert Path(directory, relative).is_file()
    assert Path(directory, relative + ".finn.json").is_file()
    assert "xsi" not in set(sys.modules) - modules


def test_xclbin_inspection_uses_selected_route(tmp_path, monkeypatch):
    import onnx.helper as oh  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    from finn.transformation.fpgadataflow import make_driver  # noqa: PLC0415

    tool = executable(
        tmp_path / "xclbinutil",
        "import json,sys; from pathlib import Path\n"
        "Path('argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "Path('ip_layout.json').write_text(json.dumps({'ip_layout':{'m_ip_data':[]}}))\n",
    )
    tc = Selection(command_dir=str(tool.parent)).prepare()
    model = ModelWrapper(oh.make_model(oh.make_graph([], "empty", [], [])))
    bitfile = tmp_path / "kernel $ [1].xclbin"
    bitfile.write_bytes(b"synthetic")
    model.set_metadata_prop("bitfile", str(bitfile))
    monkeypatch.setattr(
        make_driver, "get_driver_shapes", lambda _: {"idma_names": [], "odma_names": []}
    )
    before = dict(os.environ), os.getcwd()
    make_driver.MakeCPPDriver("vitis-xrt", "HEAD", toolchain=tc)._build_vitis_config(model)
    assert json.loads((tmp_path / "argv.json").read_text())[1] == str(bitfile)
    assert (dict(os.environ), os.getcwd()) == before


def test_settings_are_scoped_and_disable_bash_startup(tmp_path, monkeypatch):
    bad = tmp_path / "bashenv"
    bad.write_text("echo startup-ran > " + shlex.quote(str(tmp_path / "bad")) + "\nexit 51\n")
    monkeypatch.setenv("BASH_ENV", str(bad))
    original = dict(os.environ)
    scripts = []
    for name in ("one with spaces", "two $ [brackets]"):
        script = tmp_path / name
        script.write_text(
            "export CHOICE="
            + shlex.quote(name)
            + '\nexport MULTILINE="one\ntwo"\necho noisy settings\n'
        )
        scripts.append(script)

    def run(script):
        tc = Selection(settings=(script,)).prepare()
        result = run_process(
            [sys.executable, "-c", 'import os; print(os.environ["CHOICE"])'], env=tc.environment
        )
        assert tc.environment["MULTILINE"] == "one\ntwo"
        return result.stdout.decode().strip()

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run, scripts)) == [s.name for s in scripts]
    assert dict(os.environ) == original
    assert not (tmp_path / "bad").exists()


def test_environment_and_selection_are_defensive_snapshots(tmp_path):
    source = {"PATH": os.defpath, "CHOICE": "before"}
    launch = ["site"]
    selection = Selection(launcher=launch)
    launch.append("mutated")
    assert dataclasses.asdict(selection)["launcher"] == ("site",)
    tc = Toolchain(Selection(), source)
    source["CHOICE"] = "after"
    assert tc.environment["CHOICE"] == "before"
    with pytest.raises(TypeError):
        tc.environment["CHOICE"] = "changed"
    executable(tmp_path / "vivado", 'import os; print(os.environ["CHOICE"])\n')
    tc = Toolchain(Selection(command_dir=str(tmp_path)), dict(tc.environment))
    assert tc.run("vivado", env={"CHOICE": "child"}).stdout == b"child\n"
    assert tc.environment["CHOICE"] == "before"


def test_a_toolchain_pickles_as_the_same_read_only_snapshot():
    # NodeLocalTransformation's workers receive the transformation, and the
    # toolchain it holds, pickled.
    tc = Toolchain(Selection(command_dir="/site", launcher=("ssh", "host")), {"PATH": "/bin"})
    copied = pickle.loads(pickle.dumps(tc))
    assert copied == tc
    assert copied.environment == {"PATH": "/bin"}
    with pytest.raises(TypeError):
        copied.environment["PATH"] = "/usr/bin"


LICENCE = "2100@licsrv.example"


def machine_file_with(tmp_path, monkeypatch, licensed):
    """This process reads a synthetic machine file and names no licence itself."""
    lines = ["FINN_XILINX_VERSION=2025.2"]
    if licensed:
        lines += ["FINN_LICENSE_HOST=licsrv.example", "FINN_LICENSE_PORT=2100"]
    path = tmp_path / "xilinx.env"
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FINN_XILINX_ENV", str(path))
    for name in (
        "XILINXD_LICENSE_FILE",
        "LM_LICENSE_FILE",
        "FINN_LICENSE_HOST",
        "FINN_LICENSE_PORT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("licensed", (True, False))
def test_the_machine_files_licence_reaches_every_launch(tmp_path, monkeypatch, capfd, licensed):
    """A process that never sourced activate.sh: every prepared environment, and
    launch_process_helper's default one, carries the machine file's licence
    server; without licence lines the variable stays unset and nothing is said."""
    from finn.util.basic import launch_process_helper  # noqa: PLC0415

    machine_file_with(tmp_path, monkeypatch, licensed)
    settings = tmp_path / "settings64.sh"
    settings.write_text("export CHOICE=selected\n")
    expected = LICENCE if licensed else None
    for toolchain in (
        Selection().prepare(),
        Selection(settings=(settings,)).prepare(),
        Selection().prepare({"PATH": os.defpath}),
    ):
        assert toolchain.environment.get("XILINXD_LICENSE_FILE") == expected
    out, _ = launch_process_helper(
        [sys.executable, "-c", 'import os; print(os.environ.get("XILINXD_LICENSE_FILE"))']
    )
    assert out.strip() == str(expected)
    assert "XILINXD_LICENSE_FILE" not in os.environ
    # A site route owns its activation, licence included.
    route = Selection(launcher=("site",)).prepare({})
    assert "XILINXD_LICENSE_FILE" not in route.environment
    assert "licen" not in capfd.readouterr().err.lower()


@pytest.mark.parametrize("variable", ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE"))
def test_a_licence_in_the_environment_wins_over_the_machine_file(tmp_path, monkeypatch, variable):
    machine_file_with(tmp_path, monkeypatch, licensed=True)
    monkeypatch.setenv(variable, "27000@lic.example")
    settings = tmp_path / "settings64.sh"
    settings.write_text(":\n")
    for toolchain in (
        Selection().prepare(),
        Selection(settings=(settings,)).prepare(),
        Selection().prepare({variable: "27000@lic.example"}),
    ):
        assert toolchain.environment[variable] == "27000@lic.example"
        assert variable == "XILINXD_LICENSE_FILE" or "XILINXD_LICENSE_FILE" not in (
            toolchain.environment
        )


def test_site_probe_preserves_override_remote_name_and_argv(tmp_path):
    log = tmp_path / "calls.jsonl"
    wrapper = executable(
        tmp_path / "site wrapper",
        "import json,sys\n"
        f'with open({str(log)!r}, "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'print("AMD Vivado v2024.2")\n',
    )
    tc = Selection(command_dir="/remote/site tools", launcher=(str(wrapper),)).prepare({})
    assert tc.probe("vivado") == (2024, 2)
    tc.run("vivado", ["a b", "$literal", ";not-shell"])
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        ["/remote/site tools/vivado", "-version"],
        ["/remote/site tools/vivado", "a b", "$literal", ";not-shell"],
    ]
    assert Selection(launcher=(str(wrapper),)).prepare({}).command("vivado")[-1] == "vivado"
    with pytest.raises(ValueError, match="Site routes"):
        Selection(settings=("settings64.sh",), launcher=("site",))


def test_probes_run_once_per_selection(tmp_path):
    log = tmp_path / "calls"
    executable(
        tmp_path / "vitis-run",
        f"import sys\nopen({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "print('vitis-run v2025.2 (hls)')\n",
    )
    tc = Selection(command_dir=str(tmp_path), hls_frontend="vitis-run").prepare({})
    for _ in range(3):
        frontend, args = tc.hls_command("script.tcl")
        assert (frontend, args) == ("vitis-run", ["--mode", "hls", "--tcl", "script.tcl"])
    assert log.read_text().splitlines() == ["--version", "--help"]


def test_failure_logs_status_and_settings_errors(tmp_path, caplog):
    executable(
        tmp_path / "vivado",
        'import sys; print("useful log"); sys.stderr.write("failure"); sys.exit(7)\n',
    )
    tc = Selection(command_dir=str(tmp_path)).prepare({"SECRET": "do-not-log"})
    with caplog.at_level("INFO"), pytest.raises(subprocess.CalledProcessError) as exc:
        tc.run("vivado")
    assert exc.value.returncode == 7
    assert exc.value.stdout == b"useful log\n"
    assert exc.value.stderr == b"failure"
    assert "do-not-log" not in caplog.text
    assert "result=7" in caplog.text
    script = tmp_path / "bad-settings"
    script.write_text("return 13\n")
    with pytest.raises(RuntimeError, match="capture vendor settings"):
        Selection(settings=(script,)).prepare()
    script.write_text("sleep 30\n")
    with pytest.raises(RuntimeError, match="capture vendor settings"):
        Selection(settings=(script,)).prepare(timeout=0.1)
    with pytest.raises(RuntimeError, match="Could not probe"):
        tc.probe("missing")


@pytest.mark.parametrize("cancelled", [False, True])
def test_timeout_and_cancellation_kill_descendants(tmp_path, cancelled):
    pidfile = tmp_path / "pid"
    code = (
        'import subprocess,time; p=subprocess.Popen(["sleep", "60"]); '
        f'open({str(pidfile)!r}, "w").write(str(p.pid)); time.sleep(60)'
    )
    cancel = threading.Event()

    def trigger():
        # Wait for the descendant to exist before cancelling.
        for _ in range(100):
            if pidfile.exists():
                cancel.set()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=trigger) if cancelled else None
    if thread:
        thread.start()
    with pytest.raises(InterruptedError if cancelled else subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", code], timeout=1, cancel=cancel if cancelled else None)
    if thread:
        thread.join()
    pid = pidfile.read_text()
    status = Path("/proc") / pid / "stat"
    for _ in range(100):
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("descendant survived cancellation")


@pytest.mark.parametrize(
    "frontend,version", [("vivado_hls", "2019.2"), ("vitis_hls", "2024.2"), ("vitis-run", "2025.1")]
)
def test_hls_build_uses_selected_frontend(tmp_path, frontend, version):
    executable(
        tmp_path / frontend,
        "import json,os,sys\n"
        f'if "-version" in sys.argv or "--version" in sys.argv: print("AMD {version}")\n'
        'elif "--help" in sys.argv: print("--mode hls")\n'
        'else: open("operation.json", "w").write(json.dumps(sys.argv[1:]))\n',
    )
    tc = Selection(command_dir=str(tmp_path), hls_frontend=frontend).prepare({})
    build = tmp_path / "build with spaces"
    build.mkdir()
    cwd = os.getcwd()
    caller = CallHLS(tc)
    caller.append_tcl("source with spaces.tcl")
    caller.build(str(build))
    assert os.getcwd() == cwd
    args = json.loads((build / "operation.json").read_text())
    assert args == (
        ["--mode", "hls", "--tcl", "source with spaces.tcl"]
        if frontend == "vitis-run"
        else ["-f", "source with spaces.tcl"]
    )
    assert "$FINN_ROOT" not in Path(caller.ipgen_script).read_text()


def test_frontend_availability_does_not_override_compatibility(tmp_path):
    executable(tmp_path / "vitis_hls", 'print("AMD 2025.1")\n')
    tc = Selection(command_dir=str(tmp_path)).prepare({})
    with pytest.raises(ValueError, match="incompatible"):
        tc.hls_command("build.tcl")


def test_legacy_precedence_and_worker_inheritance(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    env = {
        "FINN_ROOT": "/wrong",
        "FINN_BUILD_DIR": "/wrong/build",
        "PATH": os.defpath,
        "XILINX_VIVADO": str(root),
    }
    child = build_environment(env, root=root, build_dir=tmp_path / "scratch")
    assert env["FINN_ROOT"] == "/wrong"
    assert child["VIVADO_PATH"] == str(root)
    assert "VIVADO_PATH" not in env
    result = run_process(
        [
            sys.executable,
            "-c",
            "import multiprocessing,os; "
            'p=multiprocessing.Process(target=lambda: print(os.environ["FINN_BUILD_DIR"])); '
            "p.start(); p.join()",
        ],
        env=child,
    )
    assert result.stdout.decode().strip() == str(tmp_path / "scratch")


def test_the_hls_installation_is_the_one_the_environment_names(tmp_path):
    hls, vitis = tmp_path / "hls", tmp_path / "vitis"
    named = {"PATH": os.defpath, "XILINX_HLS": str(hls), "XILINX_VITIS": str(vitis)}
    assert Selection().prepare(named).hls_installation() == hls
    del named["XILINX_HLS"]
    assert Selection().prepare(named).hls_installation() == vitis
    with pytest.raises(LookupError, match="XILINX_HLS or XILINX_VITIS"):
        Selection().prepare({"PATH": os.defpath}).hls_installation()


@pytest.mark.parametrize(
    "primary, alias", [("XILINX_HLS", "HLS_PATH"), ("XILINX_VITIS", "VITIS_PATH")]
)
def test_legacy_path_aliases_name_the_hls_installation(tmp_path, primary, alias):
    # A *_PATH alias is translated to its XILINX_* root once, by the legacy
    # translation; the named root (here one without a settings script) wins.
    root = tmp_path / "root"
    legacy = {"PATH": os.defpath, alias: str(root)}
    assert legacy_toolchain(legacy).hls_installation() == root
    assert legacy_toolchain(legacy).environment[primary] == str(root)
    named = {**legacy, primary: str(tmp_path / "named")}
    assert legacy_toolchain(named).hls_installation() == tmp_path / "named"


def test_stitched_vivado_operation_uses_selected_route(tmp_path, monkeypatch):
    import onnx.helper as oh  # noqa: PLC0415
    from onnx import TensorProto  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    from finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl import (  # noqa: PLC0415
        StreamingFIFO_rtl,
    )
    from finn.transformation.fpgadataflow.create_stitched_ip import (  # noqa: PLC0415
        CreateStitchedIP,
    )

    executable(
        tmp_path / "vivado",
        "import pathlib,sys\n"
        'if "-version" in sys.argv: print("Vivado v2024.2")\n'
        "else:\n"
        ' p=pathlib.Path("finn_vivado_stitch_proj.srcs/sources_1/bd/finn_design/hdl/")\n'
        ' p=p / "finn_design_wrapper.v"\n'
        ' p.parent.mkdir(parents=True); p.write_text("module finn_design_wrapper(); endmodule")\n',
    )
    tc = Selection(command_dir=str(tmp_path)).prepare({})
    output = tmp_path / "rtl"
    output.mkdir()
    node = oh.make_node(
        "StreamingFIFO_rtl",
        ["inp"],
        ["out"],
        name="fifo",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=4,
        folded_shape=[1, 4],
        dataType="INT8",
        impl_style="rtl",
        code_gen_dir_ipgen=str(output),
    )
    model = ModelWrapper(
        oh.make_model(
            oh.make_graph(
                [node],
                "fifo",
                [oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
                [oh.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])],
            )
        )
    )
    StreamingFIFO_rtl(model.graph.node[0]).generate_hdl(model, "xc7z020clg400-1", 10)
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "scratch"))
    before = dict(os.environ)
    model, _ = CreateStitchedIP("xc7z020clg400-1", 10, toolchain=tc).apply(model)
    assert dict(os.environ) == before
    assert Path(model.get_metadata_prop("wrapper_filename")).is_file()
    tcl = Path(model.get_metadata_prop("vivado_stitch_proj"), "make_project.tcl").read_text()
    assert "FINN_ROOT" not in tcl
    assert "sim_ctrl.v" in tcl


@pytest.mark.slow
def test_public_build_entry_point_preserves_cwd_and_failure_status(tmp_path):
    import onnx  # noqa: PLC0415
    import onnx.helper as oh  # noqa: PLC0415

    from finn.builder.build_dataflow import build_dataflow_directory  # noqa: PLC0415

    project = tmp_path / "project with spaces"
    project.mkdir()
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), project / "model.onnx")
    config = {
        "output_dir": "output",
        "synth_clk_period_ns": 10,
        "generate_outputs": [],
        "steps": [],
        "enable_build_pdb_debug": False,
    }
    (project / "dataflow_build_config.json").write_text(json.dumps(config))
    cwd, env = os.getcwd(), dict(os.environ)
    assert build_dataflow_directory(str(project)) == 0
    assert (project / "output/time_per_step.json").is_file()
    assert os.getcwd() == cwd and dict(os.environ) == env
    config["steps"] = ["not_a_step"]
    (project / "dataflow_build_config.json").write_text(json.dumps(config))
    proc = subprocess.run(
        [str(Path(sys.executable).parent / "build_dataflow"), str(project)],
        cwd=tmp_path,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert os.getcwd() == cwd and dict(os.environ) == env


def test_checkpoint_reuse_requires_original_intermediate_path(tmp_path):
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    from finn.builder.build_dataflow import build_dataflow_cfg  # noqa: PLC0415
    from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: PLC0415

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "identity",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    source = tmp_path / "model.onnx"
    onnx.save(model, source)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        synth_clk_period_ns=10,
        generate_outputs=[],
        steps=["step_qonnx_to_finn", "step_tidy_up"],
        stop_step="step_qonnx_to_finn",
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source), cfg) == 0
    cfg.start_step, cfg.stop_step = "step_tidy_up", None
    assert build_dataflow_cfg(str(source), cfg) == 0
    intermediates = Path(cfg.output_dir) / "intermediate_models"
    assert (intermediates / "step_tidy_up.onnx").is_file()
    (intermediates / "step_qonnx_to_finn.onnx").unlink()
    with pytest.raises((FileNotFoundError, AssertionError)):
        build_dataflow_cfg(str(source), cfg)
