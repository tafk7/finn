"""finn_xsi is built on first use: keyed, concurrency-safe, verified, and loud on failure.

Synthetic: the compiler step is replaced, so no Vivado is needed.
"""
import pytest

import multiprocessing
import os
from pathlib import Path

from finn import xsi
from finn.toolchain.launcher import Selection, ToolUnavailable, host_toolchain, identity
from finn.xsi import paths
from finn.xsi import setup as xsi_setup
from finn.xsi._artifacts import write_record


@pytest.fixture
def toolchain(tmp_path, monkeypatch, vivado_installation):
    vivado = vivado_installation(tmp_path / "Vivado")
    monkeypatch.setenv("XILINX_VIVADO", str(vivado))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("FINN_XSI_BUILD_DIR", raising=False)
    monkeypatch.setattr(xsi_setup, "check_prerequisites", lambda toolchain: [])
    log = tmp_path / "builds.log"

    def fake_compile(artifact, verbose, toolchain):
        with open(log, "a") as handle:
            handle.write(f"{os.getpid()}\n")
        partial = artifact.with_name(".partial")
        partial.write_bytes(b"native bridge")
        os.replace(partial, artifact)
        write_record(artifact, kind="bridge", tool=identity(toolchain, "xsim"), sources=[])

    monkeypatch.setattr(xsi_setup, "_compile", fake_compile)
    return vivado, log


def test_artifacts_are_keyed_by_vivado_installation(tmp_path, monkeypatch, vivado_installation):
    """By what the installation is, not where: two releases build apart, and one
    release installed twice shares its build."""
    monkeypatch.setenv("FINN_HOME", str(tmp_path))
    monkeypatch.delenv("FINN_XSI_BUILD_DIR", raising=False)

    def directory(path, release):
        installation = vivado_installation(path, release)
        return paths.xsi_artifact_dir(
            Selection().prepare({"PATH": os.defpath, "XILINX_VIVADO": str(installation)})
        )

    first = directory(tmp_path / "2022.2/Vivado", "2022.2")
    second = directory(tmp_path / "2025.1/Vivado", "2025.1")
    assert first != second and first.parent == second.parent == tmp_path / "xsi"
    assert directory(tmp_path / "copy/Vivado", "2025.1") == second
    monkeypatch.setenv("FINN_XSI_BUILD_DIR", str(tmp_path / "exact"))
    assert paths.xsi_artifact_dir(host_toolchain()) == tmp_path / "exact"


def test_missing_toolchain_is_reported_not_skipped(monkeypatch):
    monkeypatch.delenv("XILINX_VIVADO", raising=False)
    assert not xsi.is_available()
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO") as missing:
        xsi_setup.ensure_built()
    assert missing.value.tool == "vivado"


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
