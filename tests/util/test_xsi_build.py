"""finn_xsi is built on first use: keyed, concurrency-safe, verified, and loud on failure.

Synthetic: the compiler step is replaced, so no Vivado is needed.
"""
import pytest

import multiprocessing
import os
from pathlib import Path

from finn import xsi
from finn.xsi import paths
from finn.xsi import setup as xsi_setup
from finn.xsi._artifacts import write_record


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    vivado = tmp_path / "Vivado"
    vivado.mkdir()
    monkeypatch.setenv("XILINX_VIVADO", str(vivado))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("FINN_XSI_BUILD_DIR", raising=False)
    monkeypatch.setattr(xsi_setup, "check_prerequisites", lambda: [])
    log = tmp_path / "builds.log"

    def fake_compile(artifact, verbose):
        with open(log, "a") as handle:
            handle.write(f"{os.getpid()}\n")
        partial = artifact.with_name(".partial")
        partial.write_bytes(b"native bridge")
        os.replace(partial, artifact)
        write_record(artifact, kind="bridge", tool=xsi_setup.bridge_identity(), sources=[])

    monkeypatch.setattr(xsi_setup, "_compile", fake_compile)
    return vivado, log


def test_artifacts_are_keyed_by_vivado_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_HOME", str(tmp_path))
    monkeypatch.delenv("FINN_XSI_BUILD_DIR", raising=False)
    monkeypatch.setenv("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
    first = paths.xsi_artifact_dir()
    monkeypatch.setenv("XILINX_VIVADO", "/tools/Xilinx/2025.1/Vivado")
    second = paths.xsi_artifact_dir()
    assert first != second and first.parent == second.parent == tmp_path / "xsi"
    monkeypatch.setenv("FINN_XSI_BUILD_DIR", str(tmp_path / "exact"))
    assert paths.xsi_artifact_dir() == tmp_path / "exact"


def test_missing_toolchain_is_reported_not_skipped(monkeypatch):
    monkeypatch.delenv("XILINX_VIVADO", raising=False)
    assert not xsi.is_available()
    with pytest.raises(RuntimeError, match="XILINX_VIVADO"):
        xsi_setup.ensure_built()


def test_first_use_builds_once_and_reuses(toolchain):
    _, log = toolchain
    artifact = xsi_setup.ensure_built()
    assert artifact.read_bytes() == b"native bridge"
    assert xsi_setup.ensure_built() == artifact
    assert len(log.read_text().splitlines()) == 1


def test_changed_artifact_is_rebuilt(toolchain):
    _, log = toolchain
    artifact = xsi_setup.ensure_built()
    artifact.write_bytes(b"corrupted")
    assert xsi_setup.ensure_built().read_bytes() == b"native bridge"
    assert len(log.read_text().splitlines()) == 2


def _build(queue):
    queue.put(str(xsi_setup.ensure_built()))


def test_concurrent_first_use_builds_once(toolchain):
    _, log = toolchain
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=_build, args=(queue,)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    results = {queue.get(timeout=5) for _ in workers}
    assert len(results) == 1 and Path(results.pop()).is_file()
    assert len(log.read_text().splitlines()) == 1
