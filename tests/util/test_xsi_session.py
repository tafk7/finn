"""Synthetic session protocol/lifecycle evidence; no AMD simulation is claimed."""

import pytest

import concurrent.futures
import dataclasses
import json
import os
import sys
import threading
from pathlib import Path

from finn.xsi._artifacts import write_record
from finn.xsi._session import SessionError, SessionRequest, run_session


@pytest.fixture
def session(tmp_path):
    bridge = tmp_path / "xsi.py"
    bridge.write_text(
        """
import os
from pathlib import Path
print("bridge imported with " + os.environ.get("LD_LIBRARY_PATH", ""), flush=True)
class Kernel:
    def __init__(self, path): pass
class Port:
    def set(self, value): return self
    def write_back(self): return self
class Design:
    def __init__(self, kernel, design, log, trace):
        self.log = Path(log)
    def getPort(self, name):
        return None if name == "ap_clk2x" else Port()
    def ports(self): return []
    def run(self, cycles): pass
    def trace_all(self): pass
    def close(self): self.log.with_suffix(".closed").write_text("closed")
"""
    )
    design = tmp_path / "xsimk.so"
    design.write_bytes(b"synthetic design")
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"synthetic kernel")
    source = tmp_path / "source.sv"
    source.write_text("// synthetic compile input\n")
    for artifact, kind in ((bridge, "bridge"), (design, "design")):
        write_record(artifact, kind=kind, tool="synthetic-2024.2", sources=[source])
    bench = tmp_path / "testbench.py"
    bench.write_text(
        """
import json, os, signal, time
from pathlib import Path
def run(sim, io, request):
    mode = request["arguments"].get("mode")
    print("testbench started", flush=True)
    if mode == "failure": raise RuntimeError("recoverable testbench failure")
    if mode == "timeout": time.sleep(30)
    if mode == "crash": os.kill(os.getpid(), signal.SIGKILL)
    if mode == "incomplete": os._exit(0)
    if mode == "stale":
        sim.log = sim.top.log
        stale = {"status":"success", "session":"old", "artifacts":{}}
        (sim.log.parent / "result.json").write_text(json.dumps(stale))
        os._exit(0)
    io["outputs"]["out"] = list(io["inputs"]["in"])
    return {"cycles": len(io["outputs"]["out"]), "pid": os.getpid()}
"""
    )
    return SessionRequest(
        str(bridge),
        str(kernel),
        str(design),
        str(tmp_path),
        "synthetic-2024.2",
        {"out": 2},
        testbench=str(bench),
    )


def execute(request, tmp_path, **kwargs):
    return run_session(
        request,
        {"in": [7, 2**130 + 3]},
        environment={**os.environ, "LD_LIBRARY_PATH": "/selected"},
        output_root=tmp_path,
        timeout=kwargs.pop("timeout", 10),
        **kwargs,
    )


def test_session_isolation_bulk_roundtrip_and_concurrent_outputs(session, tmp_path):
    before = dict(os.environ), os.getcwd(), set(sys.modules)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: execute(session, tmp_path), range(2)))
    assert results[0]["directory"] != results[1]["directory"]
    for result in results:
        assert result["outputs"] == {"out": [7, 2**130 + 3]}
        assert result["metrics"]["pid"] != os.getpid()
        directory = Path(result["directory"])
        assert (directory / "xsi.closed").read_text() == "closed"
        assert "bridge imported with /selected" in (directory / "stdout.log").read_text()
        assert (directory / "result.json").is_file()
    assert (dict(os.environ), os.getcwd()) == before[:2]
    assert "xsi" not in set(sys.modules) - before[2]
    assert "finn_xsi.sim_engine" not in set(sys.modules) - before[2]


@pytest.mark.parametrize("mode", ["failure", "timeout", "crash", "incomplete", "stale"])
def test_session_failure_never_publishes_a_success(session, tmp_path, mode):
    request = dataclasses.replace(session, arguments={"mode": mode})
    with pytest.raises(SessionError) as exc:
        execute(request, tmp_path, timeout=1 if mode == "timeout" else 10)
    directory = exc.value.directory
    assert (directory / "stdout.log").is_file()
    if mode == "failure":
        assert (directory / "xsi.closed").is_file()
        assert "recoverable testbench failure" in (directory / "stderr.log").read_text()
    if mode not in {"stale"}:
        assert not (directory / "result.json").exists()


def test_session_cancellation_and_site_python_requirement(session, tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SessionError) as exc:
        execute(session, tmp_path, cancel=cancel)
    assert isinstance(exc.value.__cause__, InterruptedError)
    with pytest.raises(ValueError, match="site Python"):
        execute(session, tmp_path, launcher=["site-driver"])
    wrapper = tmp_path / "site launcher"
    wrapper.write_text('#!/bin/bash\nexec "$@"\n')
    wrapper.chmod(0o755)
    result = execute(session, tmp_path, launcher=[str(wrapper)], python_command=[sys.executable])
    assert result["outputs"] == {"out": [7, 2**130 + 3]}


def test_reuse_rejects_changed_source_tool_and_abi(session, tmp_path):
    with pytest.raises(ValueError, match="tool selection"):
        execute(dataclasses.replace(session, tool_identity="different-tool"), tmp_path)
    record_path = Path(session.bridge + ".finn.json")
    original = record_path.read_text()
    record = json.loads(original)
    record["python_abi"] = "wrong-abi"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="ABI"):
        execute(session, tmp_path)
    record_path.write_text(original)
    (tmp_path / "source.sv").write_text("// incompatible new source\n")
    with pytest.raises(ValueError, match="compile input changed"):
        execute(session, tmp_path)


def test_compilation_helpers_import_without_native_bridge(tmp_path):
    import subprocess  # noqa: PLC0415

    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from finn import xsi; assert callable(xsi.compile_sim_obj); "
            "import finn_xsi.adapter; import finn.core.rtlsim_exec; "
            "assert 'xsi' not in sys.modules; assert 'finn_xsi.sim_engine' not in sys.modules",
        ],
        cwd=tmp_path,
        check=True,
    )
