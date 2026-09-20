"""Synthetic execution tests; these do not claim licensed AMD operation coverage."""
import pytest

import concurrent.futures
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from finn.util._legacy_build_env import build_environment, external_path
from finn.util._toolchain import Selection, Toolchain, run_process
from finn.util.hls import CallHLS


def executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


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
    (root / "deps/finn-hlslib").mkdir(parents=True)
    env = {
        "FINN_ROOT": "/wrong",
        "FINN_BUILD_DIR": "/wrong/build",
        "PATH": os.defpath,
        "XILINX_VIVADO": str(root),
    }
    assert external_path("hlslib", root=root, environ=env) == str(root / "deps/finn-hlslib")
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
    with pytest.raises(FileNotFoundError, match="FINN_BOARD_FILES_PATH"):
        external_path("boards", environ={})


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
