"""Installation journeys; requires prepared build requirements and dependencies.

No proprietary tools, network downloads, root variables or import hooks are used.
"""
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(__file__).with_name("runtime_resource_smoke.py")


def run(argv, cwd, env=None):
    return subprocess.run(
        list(map(str, argv)),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def clean_env():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("FINN_") and k not in {"PYTHONPATH", "BASH_ENV"}
    }
    # FINN's per-user state goes to temporary space, not the user's ~/.finn.
    env["FINN_HOME"] = os.path.join(tempfile.gettempdir(), f"finn-home-test-{os.getuid()}")
    return env


def snapshot(destination):
    for name in (
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "VERSION",
        "LICENSE.txt",
        "README.md",
    ):
        shutil.copy2(ROOT / name, destination / name)
    for name in ("src",):
        shutil.copytree(
            ROOT / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.so", "*.pyc"),
        )


def python_env(path):
    # Prepare this environment from the caller's installed dependency set,
    # excluding FINN and startup hooks. No network or nearby-checkout lookup.
    # Create it from the base interpreter: on Python 3.10 a venv created from inside
    # another venv records the outer venv as its home, which relocatable (e.g.
    # uv-managed) interpreters cannot start from.
    # Without pip: the caller's own pip and build backend are linked in below, so
    # the environment has exactly the caller's tested versions.
    base = Path(sys.base_prefix) / "bin" / ("python%d.%d" % sys.version_info[:2])
    subprocess.run([base, "-m", "venv", "--without-pip", path], check=True)
    site = next((path / "lib").glob("python*/site-packages"))
    for item in Path(sysconfig.get_path("purelib")).iterdir():
        if (
            item.name.startswith(("finn", "_finn", "__editable__"))
            or item.suffix == ".pth"
            or (site / item.name).exists()
        ):
            continue
        (site / item.name).symlink_to(item, target_is_directory=item.is_dir())
    return path / "bin/python"


def install(python, cwd, *args):
    run(
        [python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation", *args],
        cwd,
    )


def test_checkout_and_sdist_wheels_have_same_assets_and_work_without_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    snapshot(checkout)
    run(["git", "init", "--quiet"], checkout)
    run(["git", "add", "VERSION"], checkout)
    run(
        [
            "git",
            "-c",
            "user.name=FINN test",
            "-c",
            "user.email=test@localhost",
            "commit",
            "--quiet",
            "-m",
            "package fixture",
        ],
        checkout,
    )
    revision = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
    run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--sdist"], checkout)
    wheel = next((checkout / "dist").glob("*.whl"))
    sdist = next((checkout / "dist").glob("*.tar.gz"))
    unpack = tmp_path / "unpack"
    unpack.mkdir()
    with tarfile.open(sdist) as archive:
        archive.extractall(unpack, filter="data")
    extracted = next(unpack.iterdir())
    run([sys.executable, "-m", "build", "--no-isolation", "--wheel"], extracted)
    rebuilt = next((extracted / "dist").glob("*.whl"))

    def contents(path):
        with zipfile.ZipFile(path) as archive:
            return {
                name: archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/RECORD")
            }

    data = contents(wheel)
    assert data == contents(rebuilt)
    assert json.loads(data["finn/_build_info.json"])["revision"] == revision
    assert not any(name.startswith("_finn_") for name in data)
    assert "finn/rtllib/sim/hdl/sim_ctrl.v" in data
    assert (
        "finn/rtllib/memstream/component.xml" not in data
        or data["finn/rtllib/memstream/component.xml"]
    )
    assert "finn/custom_hls/CNPY_LICENSE" in data and "finn/custom_hls/cnpy.cpp" in data
    assert "finn/xsi/src/xsi_finn.cpp" in data
    assert "finn/deploy/data/mdd/finn_design.mdd" in data
    assert "finn/deploy/data/pynq_driver/driver_base.py" in data
    assert "finn/resources.toml" in data and "finn/resources/_cli.py" in data
    assert not any(
        "/testcase/" in name or "_tb." in name or "/build_dataflow/" in name for name in data
    )
    installed = tmp_path / "environment"
    python = python_env(installed)
    install(python, tmp_path, wheel)
    package = next((installed / "lib").glob("python*/site-packages/finn"))
    for resources in ("rtllib", "custom_hls", "xsi/src", "deploy/data"):
        for path in (package / resources).rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        (package / resources).chmod(0o555)
    smoke = tmp_path / "smoke.py"
    shutil.copy2(SMOKE, smoke)
    # Remove the exact source trees used for both wheels. This is an installed
    # resource test, not a claim about relocating checkpoints/projects.
    shutil.rmtree(checkout)
    shutil.rmtree(unpack)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    result = run([python, "-I", smoke, tmp_path / "outputs"], unrelated, clean_env())
    assert json.loads(result.stdout)["rtl"].endswith("fifo.v")
    identity = run(
        [
            python,
            "-I",
            "-c",
            "from finn.util.resources import resource_path; "
            "from importlib.metadata import version; "
            'print(version("finn")); print(resource_path("rtllib"))',
        ],
        unrelated,
        clean_env(),
    ).stdout
    assert "0.11.0.dev0" in identity
    assert str(installed) in identity
    assert (installed / "bin/build_dataflow").exists()
    run([installed / "bin/build_dataflow", "--help"], unrelated, clean_env())
    # The declarations ship in the wheel.
    listing = run([installed / "bin/finn-resources", "list"], unrelated, clean_env()).stdout
    assert listing.splitlines()[1].startswith("hlslib ")


def test_two_editable_environments_observe_only_selected_code_and_resources(tmp_path):
    environments = []
    for number in (1, 2):
        source = tmp_path / f"source{number}"
        source.mkdir()
        snapshot(source)
        python = python_env(tmp_path / f"env{number}")
        install(python, tmp_path, "-e", source)
        environments.append((python, source))
    first = environments[0][1]
    (first / "src/finn/util/runtime_edit_probe.py").write_text('value = "selected change"\n')
    asset = first / "src/finn/rtllib/fifo/hdl/fifo.sv"
    replacement = asset.with_suffix(".new")
    replacement.write_text(asset.read_text() + "\n// selected resource change\n")
    replacement.replace(asset)
    probe = (
        "import importlib.util; from importlib.metadata import distribution; "
        "from finn.util.resources import resource_path; from pathlib import Path; "
        'print(importlib.util.find_spec("finn.util.runtime_edit_probe") is not None); '
        'p = Path(resource_path("rtllib", "fifo/hdl/fifo.sv")); '
        'print("selected resource change" in p.read_text()); '
        'print(distribution("finn").read_text("direct_url.json"))'
    )
    for number, (python, source) in enumerate(environments):
        # Starting in an unrelated checkout does not choose its code.
        other = environments[1 - number][1]
        result = run([python, "-I", "-c", probe], other, clean_env()).stdout.splitlines()
        assert result[:2] == (["True", "True"] if number == 0 else ["False", "False"])
        metadata = json.loads(result[2])
        assert metadata["dir_info"]["editable"] is True
        assert metadata["url"] == source.as_uri()
        smoke = tmp_path / f"smoke{number}.py"
        shutil.copy2(SMOKE, smoke)
        run([python, "-I", smoke, tmp_path / f"output{number}"], other, clean_env())


def test_imports_do_not_create_scratch_or_change_environment(tmp_path):
    env = clean_env()
    env["FINN_BUILD_DIR"] = str(tmp_path / "must-not-exist")
    run(
        [
            sys.executable,
            "-I",
            "-c",
            "import os; before=dict(os.environ); import finn.util.basic, finn.xsi.paths; "
            "assert dict(os.environ)==before",
        ],
        tmp_path,
        env,
    )
    assert not Path(env["FINN_BUILD_DIR"]).exists()


def test_legacy_interpretation_allowlist():
    # FINN_LEGACY_COMPAT: resource consumers may not reconstruct checkout paths.
    sites = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if "FINN_ROOT" in p.read_text()
    }
    assert sites == {"src/finn/util/_legacy_build_env.py"}


def test_editable_finn_plus_real_qonnx(tmp_path):
    import pytest  # noqa: PLC0415

    selected = os.environ.get("FINN_TEST_QONNX_CHECKOUT")
    if not selected:
        pytest.skip("set FINN_TEST_QONNX_CHECKOUT to a prepared versioned QONNX checkout")
    qonnx = tmp_path / "qonnx"
    shutil.copytree(
        selected, qonnx, ignore=shutil.ignore_patterns("build", "__pycache__", "*.egg-info")
    )
    source = tmp_path / "finn"
    source.mkdir()
    snapshot(source)
    python = python_env(tmp_path / "editable")
    site = next((python.parent.parent / "lib").glob("python*/site-packages"))
    # Remove the prepared dependency links before installing its editable source.
    for item in site.glob("qonnx*"):
        if item.is_symlink():
            item.unlink()
    install(python, tmp_path, "-e", qonnx, "-e", source)
    (qonnx / "src/qonnx/runtime_edit_probe.py").write_text('value = "QONNX selected edit"\n')
    probe = (
        "import qonnx, qonnx.runtime_edit_probe as p; "
        "from importlib.metadata import distribution; "
        'assert p.value == "QONNX selected edit"; '
        'print(qonnx.__path__[0]); print(distribution("qonnx").read_text("direct_url.json")); '
        'print(distribution("qonnx").version)'
    )
    result = run([python, "-I", "-c", probe], tmp_path, clean_env()).stdout.splitlines()
    assert result[0] == str(qonnx / "src/qonnx")
    assert json.loads(result[1])["url"] == qonnx.as_uri()
    assert result[2] != "0.0.0"
    smoke = tmp_path / "smoke.py"
    shutil.copy2(SMOKE, smoke)
    run([python, "-I", smoke, tmp_path / "outputs"], tmp_path, clean_env())
