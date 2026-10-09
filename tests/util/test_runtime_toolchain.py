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

from finn.util.hls import CallHLS
from finn.util.toolchain import Selection, Toolchain, run_process


def executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


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


def test_the_clean_base_keeps_the_hosts_preload(tmp_path, monkeypatch):
    # The image's libudev.so.1 preload keeps Vivado's licence library alive; a
    # settings-scripts selection must not drop it with the other toolchain's state.
    settings = tmp_path / "settings64.sh"
    settings.write_text("export SELECTED=1\n")
    monkeypatch.setenv("LD_PRELOAD", "/lib/x86_64-linux-gnu/libudev.so.1")
    monkeypatch.setenv("XILINX_VIVADO", "/another/Vivado")
    environment = Selection(settings=(settings,)).prepare().environment
    assert environment["LD_PRELOAD"] == "/lib/x86_64-linux-gnu/libudev.so.1"
    assert environment["SELECTED"] == "1"
    assert "XILINX_VIVADO" not in environment


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
    """A process that never sourced activate.sh: every prepared environment carries
    the machine file's licence server, and so does a process launched in it; without
    licence lines the variable stays unset and nothing is said."""
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
    launched = run_process(
        [sys.executable, "-c", 'import os; print(os.environ.get("XILINXD_LICENSE_FILE"))'],
        env=Selection().prepare().environment,
    )
    assert launched.stdout.decode().strip() == str(expected)
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
        # The descendant may exit between opening and reading its status: gone.
        try:
            if status.read_text().split()[2] == "Z":
                break
        except (FileNotFoundError, ProcessLookupError):
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
    tc = Selection(command_dir=str(tmp_path), hls_frontend="vitis_hls").prepare({})
    with pytest.raises(ValueError, match="incompatible"):
        tc.hls_command("build.tcl")


def test_the_simulation_environment_loads_the_simulator_libraries_in_workers(tmp_path):
    """The simulator libraries that exist come first on the loader path, ahead of
    the environment's own, in a new process and the workers it starts; the
    toolchain's environment is left as it was."""
    vivado, vitis = tmp_path / "Vivado", tmp_path / "Vitis"
    (vivado / "lib/lnx64.o").mkdir(parents=True)
    vitis.mkdir()  # no floating-point operator libraries: not on the path
    env = {
        "PATH": os.defpath,
        "XILINX_VIVADO": str(vivado),
        "XILINX_VITIS": str(vitis),
        "LD_LIBRARY_PATH": "/parent/lib",
    }
    toolchain = Selection().prepare(env)
    child = toolchain.simulation_environment()
    assert child["LD_LIBRARY_PATH"] == f"{vivado}/lib/lnx64.o:/parent/lib"
    assert toolchain.environment["LD_LIBRARY_PATH"] == "/parent/lib"
    result = run_process(
        [
            sys.executable,
            "-c",
            "import multiprocessing,os; "
            'p=multiprocessing.Process(target=lambda: print(os.environ["LD_LIBRARY_PATH"])); '
            "p.start(); p.join()",
        ],
        env=child,
    )
    assert result.stdout.decode().strip() == child["LD_LIBRARY_PATH"]
    assert (
        "LD_LIBRARY_PATH" not in Selection().prepare({"PATH": os.defpath}).simulation_environment()
    )


def test_the_hls_installation_is_the_one_the_environment_names(tmp_path):
    hls, vitis = tmp_path / "hls", tmp_path / "vitis"
    named = {"PATH": os.defpath, "XILINX_HLS": str(hls), "XILINX_VITIS": str(vitis)}
    assert Selection().prepare(named).hls_installation() == hls
    del named["XILINX_HLS"]
    assert Selection().prepare(named).hls_installation() == vitis
    with pytest.raises(LookupError, match="XILINX_HLS or XILINX_VITIS"):
        Selection().prepare({"PATH": os.defpath}).hls_installation()


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
        "target": {"period_ns": 10.0, "part": "xczu3eg-sbva484-1-e"},
        "steps": [],
        "enable_build_pdb_debug": False,
    }
    (project / "kernel_build_config.json").write_text(json.dumps(config))
    cwd, env = os.getcwd(), dict(os.environ)
    assert build_dataflow_directory(str(project)) == 0
    assert (project / "output/time_per_step.json").is_file()
    assert os.getcwd() == cwd and dict(os.environ) == env
    config["steps"] = ["not_a_step"]
    (project / "kernel_build_config.json").write_text(json.dumps(config))
    proc = subprocess.run(
        [str(Path(sys.executable).parent / "build_dataflow"), str(project)],
        cwd=tmp_path,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert os.getcwd() == cwd and dict(os.environ) == env


def step_first(model, cfg):
    return model


def step_second(model, cfg):
    return model


def test_checkpoint_reuse_requires_original_intermediate_path(tmp_path):
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    from finn.builder.build_dataflow import build_dataflow_cfg  # noqa: PLC0415
    from finn.builder.kernel_build_config import KernelBuildConfig  # noqa: PLC0415
    from finn.platform import TargetRequest  # noqa: PLC0415

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
    cfg = KernelBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[step_first, step_second],
        stop_step="step_first",
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source), cfg) == 0
    cfg.start_step, cfg.stop_step = "step_second", None
    assert build_dataflow_cfg(str(source), cfg) == 0
    intermediates = Path(cfg.output_dir) / "intermediate_models"
    assert (intermediates / "step_second.onnx").is_file()
    (intermediates / "step_first.onnx").unlink()
    with pytest.raises((FileNotFoundError, AssertionError)):
        build_dataflow_cfg(str(source), cfg)
