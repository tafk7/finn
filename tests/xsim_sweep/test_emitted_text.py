# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""scripts/emitted_text.py: a job's key follows what its simulation consumes, and only that.

The keys are computed for real, without Vivado, on a copy of this checkout and of
FinnLib: a change to an input of a job's simulation (a FinnLib file its design
includes, its harness's code, the sweep script, a result of construction code
the sweep reads besides the design) changes the job's key; a file the job does
not consume (another kernel's RTL, an unrelated test, a comment or an import's
module path, construction code whose effect is unchanged) does not, nor does where
the checkout is or what the machine put inside it (a ``.venv``). The selection
runs exactly the jobs whose key differs from a passed baseline's, or that no run
under the baseline's key passed, and a skip cites the commit where that run was.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import site
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("emitted_text", ROOT / "scripts/emitted_text.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module
    spec.loader.exec_module(module)
    return module


tool = _load()

# One conformance job and one numeric sweep: both capture kinds, each a few seconds.
MEMSTREAM = tool.conformance_job(
    "tests/kernels/test_conformance.py::test_the_kernel_conforms_in_xsim[memstream]"
)
ADAPTERS = tool.Job("sweep-adapters", "sweep", ("kernels.sweeps.adapter_numeric",))
# A MatMul sweep: it reads its assembly's beat counts after the simulation.
PACKED = tool.Job("sweep-packed", "sweep", ("kernels.sweeps.matmul_numeric", "--case", "packed"))
# A pytest group: the XSim tests of tests/kernels outside conformance.
(KERNELS_REST,) = (
    tool.Job(name, "pytest", args) for name, args in tool.PYTEST_GROUPS if name == "kernels-rest"
)
COPIED = ("src", "tests", "scripts", "docker", ".pytest.ini", "uv.lock", "pyproject.toml")


# -- the code digest -------------------------------------------------------------------------


def _code(tmp_path: Path, text: str) -> str:
    path = tmp_path / "module.py"
    path.write_text(text)
    return str(tool.code_digest(path))


BASE = '''"""A module."""
from a.b import f, g
import os

def h(x):
    """Doubles."""
    return f(x) * 2  # twice
'''


def test_the_code_digest_ignores_how_code_is_written_down(tmp_path: Path) -> None:
    same = '''"""Another docstring."""
import os
from a.renamed import g, f


def h(x):  # a comment
    return f(x) * 2
'''
    assert _code(tmp_path, same) == _code(tmp_path, BASE)


@pytest.mark.parametrize(
    "changed",
    [
        BASE.replace("* 2", "* 3"),  # what h computes
        BASE.replace("f, g", "f as g, g as f"),  # what a name binds
        BASE + "\nLIMIT = 4\n",  # a module-level statement
    ],
)
def test_the_code_digest_follows_what_code_does(tmp_path: Path, changed: str) -> None:
    assert _code(tmp_path, changed) != _code(tmp_path, BASE)


def test_a_file_that_is_not_python_counts_by_content(tmp_path: Path) -> None:
    path = tmp_path / "x.sv"
    path.write_text("module x; endmodule\n")
    before = tool.code_digest(path)
    path.write_text("module x; endmodule  // y\n")
    assert tool.code_digest(path) != before


def test_hashes_are_blanked_in_identifiers_only() -> None:
    text = "module finn_root__0123456789abcdef; 64'h0123456789abcdef memstream_fedcba9876543210.dat"
    assert tool.blank_hashes(text) == (
        "module finn_root__################; 64'h0123456789abcdef memstream_################.dat"
    )


# -- keys, computed for real -----------------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A copy of this checkout's inputs (src, tests, scripts, ...) and of FinnLib."""
    base = tmp_path_factory.mktemp("sandbox")
    root = base / "finn"
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, root / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            root.mkdir(exist_ok=True)
            shutil.copy(source, root / name)
    # A checkout: its keys hold the files git tracks (``checkout_files``).
    for command in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    finnlib = base / "finnlib"
    shutil.copytree(
        tool.Target(ROOT).finnlib_path(base),
        finnlib,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return root, finnlib


Edit = Callable[[Path, Path], None]


def _keys(
    root: Path, finnlib: Path, work: Path, jobs: tuple[Any, ...] = (MEMSTREAM, ADAPTERS)
) -> dict[str, Any]:
    found = tool.compute_keys(tool.Target(root, finnlib), list(jobs), work)
    for entry in found["jobs"].values():
        assert entry.get("key"), entry
        assert entry.get("simulations", entry.get("tests")) > 0
    return dict(found["jobs"])


def _append(path: str, text: str) -> Edit:
    def edit(root: Path, finnlib: Path) -> None:
        target = (finnlib / path[len("finnlib/") :]) if path.startswith("finnlib/") else root / path
        target.write_text(target.read_text() + text)

    return edit


@pytest.fixture(scope="module")
def unedited(
    sandbox: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The sandbox's keys: a key does not depend on where the checkout is (tested below)."""
    return _keys(*sandbox, tmp_path_factory.mktemp("unedited"))


def _edited(
    sandbox: tuple[Path, Path],
    tmp_path: Path,
    edit: Edit,
    jobs: tuple[Any, ...] = (MEMSTREAM, ADAPTERS),
) -> dict[str, Any]:
    """The keys after ``edit``, on a fresh copy of the sandbox."""
    root, finnlib = tmp_path / "finn", tmp_path / "finnlib"
    shutil.copytree(sandbox[0], root, symlinks=True)
    shutil.copytree(sandbox[1], finnlib, symlinks=True)
    edit(root, finnlib)
    return _keys(root, finnlib, tmp_path / "keys", jobs)


BOTH = {MEMSTREAM.name, ADAPTERS.name}
EDITS = [
    # FinnLib by content: memstream's RTL is in both jobs' designs (the adapters are
    # fed by a memory), thresholding's in the adapters' only.
    ("finnlib-memstream", "finnlib/rtl/infra/memstream.sv", "\n// edited\n", BOTH),
    ("finnlib-thresholding", "finnlib/rtl/nonlin/thresholding.sv", "\n// e\n", {ADAPTERS.name}),
    # The harness: conformance's code, the adapters' stimulus module, the XSI runtime.
    ("conformance-code", "tests/kernels/conformance.py", "\nLIMIT = 1\n", {MEMSTREAM.name}),
    # test_conformance.py imports kernels.adapted (the channel-stage cases), so every
    # conformance job consumes it, as well as the adapters sweep.
    ("adapter-stimulus", "tests/kernels/adapted.py", "\nLIMIT = 1\n", BOTH),
    ("xsi-runtime", "src/finn/xsi/compile.py", "\nLIMIT = 1\n", {ADAPTERS.name}),
    # The testbench and the harness package: both jobs run the testbench writer's or its
    # toolchain's code.
    ("xsim-rtl", "src/finn/core/executors/xsim/rtl.py", "\nLIMIT = 1\n", BOTH),
    ("harness-toolchain", "src/finn/harness/toolchain.py", "\nLIMIT = 1\n", BOTH),
    ("sweep-script", "scripts/xsim-sweep.sh", "\n# edited\n", BOTH),
    # Not consumed: another kernel's RTL, an unrelated test, a comment in the harness,
    # FINN code no design reaches.
    ("finnlib-dotp", "finnlib/rtl/linalg/dotp.sv", "\n// edited\n", set()),
    ("unrelated-test", "tests/kernels/test_dotp.py", "\nLIMIT = 1\n", set()),
    # The XSim executor: beside the testbench, imported by no captured job.
    ("xsim-executor", "src/finn/core/executors/xsim/executor.py", "\nLIMIT = 1\n", set()),
    ("harness-comment", "tests/kernels/conformance.py", "\n# a comment\n", set()),
    # Construction code whose effect is unchanged: its designs are the same.
    ("construction-code", "tests/kernels/helpers.py", "\nLIMIT = 1\n", set()),
    ("builder", "src/finn/builder/build_dataflow.py", "\nLIMIT = 1\n", set()),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("path", "text", "changed"),
    [pytest.param(path, text, changed, id=label) for label, path, text, changed in EDITS],
)
def test_a_job_key_changes_with_exactly_what_the_job_consumes(
    sandbox: tuple[Path, Path],
    unedited: dict[str, Any],
    tmp_path: Path,
    path: str,
    text: str,
    changed: set[str],
) -> None:
    before, after = unedited, _edited(sandbox, tmp_path, _append(path, text))
    assert {name for name in before if before[name]["key"] != after[name]["key"]} == changed


def _rename(path: str, old: str, new: str) -> Edit:
    def edit(root: Path, finnlib: Path) -> None:
        target = root / path
        assert old in target.read_text()
        target.write_text(target.read_text().replace(old, new))

    return edit


# The verdict's weight beats differ; the design does not.
FEWER_BEATS = """
import dataclasses as _dataclasses
_assembled = matmul_assembly

def matmul_assembly(**arguments):
    built = _assembled(**arguments)
    return _dataclasses.replace(built, weight_beats=built.weight_beats - 1)
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("edit", "changed"),
    [
        pytest.param(_rename("tests/kernels/helpers.py", "_frozen", "_tupled"), False, id="rename"),
        pytest.param(_append("tests/kernels/helpers.py", FEWER_BEATS), True, id="verdict-value"),
    ],
)
def test_a_sweep_is_keyed_by_what_construction_hands_it_not_by_its_code(
    sandbox: tuple[Path, Path], tmp_path: Path, edit: Edit, changed: bool
) -> None:
    before = _keys(*sandbox, tmp_path / "unedited", (PACKED,))[PACKED.name]
    after = _edited(sandbox, tmp_path / "edited", edit, (PACKED,))[PACKED.name]
    assert after["inputs"]["designs"] == before["inputs"]["designs"]
    assert (after["key"] != before["key"]) is changed
    assert not any(name.endswith("helpers.py") for name in after["inputs"])


def _venv_inside(root: Path) -> Path:
    """An environment inside the checkout, untracked, as a host checkout's ``.venv`` is:
    its interpreter, which the jobs run on, imports a module of its own at startup and
    reaches this environment's packages; and a bridge compiled into the XSI runtime's
    package on this machine. Its interpreter."""
    venv = root / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    (packages,) = venv.glob("lib/python*/site-packages")
    # A .pth line that imports runs at startup: the environment's own module, which adds
    # the packages the jobs need (this interpreter's).
    (packages / "machine_local.py").write_text(
        f"import site\nfor folder in {site.getsitepackages()!r}:\n    site.addsitedir(folder)\n"
    )
    (packages / "machine_local.pth").write_text("import machine_local\n")
    (root / "src/finn_xsi/bridge.so").write_bytes(b"compiled on this machine")
    return venv / "bin" / "python"


@pytest.mark.slow
@pytest.mark.parametrize("venv", [False, True], ids=["moved", "moved-with-venv"])
def test_a_key_does_not_depend_on_where_the_checkout_is(
    sandbox: tuple[Path, Path],
    unedited: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    venv: bool,
) -> None:
    """Nor on what the machine put inside it: a ``.venv`` the jobs run on, a bridge."""
    here = unedited
    moved = tmp_path / "moved"
    shutil.copytree(sandbox[0], moved / "finn", symlinks=True)
    shutil.copytree(sandbox[1], moved / "finnlib", symlinks=True)
    if venv:
        monkeypatch.setattr(sys, "executable", str(_venv_inside(moved / "finn")))
    there = _keys(moved / "finn", moved / "finnlib", tmp_path / "there")
    assert {name: entry["key"] for name, entry in here.items()} == {
        name: entry["key"] for name, entry in there.items()
    }


def test_only_the_files_a_checkout_tracks_under_its_import_roots_are_its_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    files = ("src/pkg/a.py", "tests/t.py", "scripts/s.py", ".venv/lib/pkg/a.py")
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("")
    for command in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    (root / "src/pkg/built.so").write_bytes(b"")  # untracked
    paths = [root / name for name in (*files, "src/pkg/built.so")]
    assert tool.checkout_files(root, paths) == ["src/pkg/a.py", "tests/t.py"]
    (tmp_path / "loose").mkdir()
    with pytest.raises(RuntimeError, match="no git checkout"):
        tool.checkout_files(tmp_path / "loose", [])


def test_vivado_is_identified_by_its_release_and_version_not_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = []
    for place in ("here", "there"):
        data = tmp_path / place / "Vivado/data"
        data.mkdir(parents=True)
        (data / "version.dat").write_bytes(b"\x00release bytes")
        (data / "version.sh").write_text("XILINX_VERSION_DEFAULT=2025.2\nOTHER=1\n")
        monkeypatch.setenv("XILINX_VIVADO", str(data.parent))
        identities.append(tool.vivado_identity())
    assert identities[0] == identities[1] and identities[0].startswith("2025.2 ")
    (data / "version.sh").unlink()
    assert tool.vivado_identity().startswith("unstated ")
    monkeypatch.delenv("XILINX_VIVADO")
    assert tool.vivado_identity() == "none"


GROUP_EDITS = [
    # A file of a selected test, FINN code it imports, a package's data beside that
    # code, and FinnLib, which the group's key takes whole.
    ("selected-test", "tests/kernels/test_fifo_sizing.py", "\nLIMIT = 1\n", True),
    ("imported-code", "src/finn/kernels/fifo.py", "\nLIMIT = 1\n", True),
    ("package-data", "src/finn/resources.toml", "\n# edited\n", True),
    ("finnlib", "finnlib/rtl/linalg/dotp.sv", "\n// edited\n", True),
    # Not consumed: a test file with no XSim test, code no selected test imports,
    # a comment in the harness.
    ("deselected-test", "tests/kernels/test_dotp.py", "\nLIMIT = 1\n", False),
    ("builder", "src/finn/builder/build_dataflow.py", "\nLIMIT = 1\n", False),
    ("harness-comment", "tests/kernels/xsim.py", "\n# a comment\n", False),
]


@pytest.fixture(scope="module")
def unedited_group(
    sandbox: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    return _keys(*sandbox, tmp_path_factory.mktemp("unedited-group"), (KERNELS_REST,))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("path", "text", "changed"),
    [pytest.param(path, text, changed, id=label) for label, path, text, changed in GROUP_EDITS],
)
def test_a_pytest_group_is_keyed_by_the_code_its_selected_tests_import(
    sandbox: tuple[Path, Path],
    unedited_group: dict[str, Any],
    tmp_path: Path,
    path: str,
    text: str,
    changed: bool,
) -> None:
    before = unedited_group[KERNELS_REST.name]
    after = _edited(sandbox, tmp_path, _append(path, text), (KERNELS_REST,))[KERNELS_REST.name]
    assert (after["key"] != before["key"]) is changed


# The Pool kernel's conformance job: its designs hold HLS build requests.
POOL = tool.conformance_job(
    "tests/kernels/test_conformance.py::test_the_kernel_conforms_in_xsim[pool]"
)


def _installation(directory: Path, release: str) -> Path:
    data = directory / "data"
    data.mkdir(parents=True)
    (data / "version.dat").write_bytes(release.encode())
    (data / "version.sh").write_text(f"XILINX_VERSION_DEFAULT={release}\n")
    return directory


@pytest.mark.slow
def test_an_hls_job_is_keyed_by_its_requests_their_part_and_the_hls_tool(
    sandbox: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture stages each request (its top, its FinnLib headers by content) and
    synthesizes nothing; only a job whose designs hold one is keyed by the part and the
    HLS installation."""
    jobs = (MEMSTREAM, POOL)
    monkeypatch.setenv("XILINX_HLS", str(_installation(tmp_path / "hls-a", "2025.2")))
    before = _keys(*sandbox, tmp_path / "before", jobs)
    inputs = before[POOL.name]["inputs"]
    assert inputs["hls"].startswith("2025.2 ")
    parts = {value for name, value in inputs.items() if name.startswith("hls-part:")}
    assert parts == {"xczu3eg-sbva484-1-i"}
    assert not any(name.startswith("hls") for name in before[MEMSTREAM.name]["inputs"])
    requests = sorted((tmp_path / "before" / POOL.name / "designs" / "hls").iterdir())
    assert len(requests) == 3  # PE 1, 3 and 6; the adapter sample is PE 3's again
    for request in requests:
        assert (request / "script.tcl").is_file()
        assert (request / "hls/nonlin/pooling.hpp").is_file()

    edited = _edited(
        sandbox, tmp_path / "edited", _append("finnlib/hls/nonlin/pooling.hpp", "\n// e\n"), jobs
    )
    assert {name for name in before if before[name]["key"] != edited[name]["key"]} == {POOL.name}

    monkeypatch.setenv("XILINX_HLS", str(_installation(tmp_path / "hls-b", "2026.1")))
    other = _keys(*sandbox, tmp_path / "other", jobs)
    assert {name for name in before if before[name]["key"] != other[name]["key"]} == {POOL.name}


def test_the_import_closure_follows_every_import_a_file_states(tmp_path: Path) -> None:
    files = {
        "tests/group/__init__.py": "",
        "tests/group/test_a.py": "from . import helper\n",
        "tests/group/helper.py": "def run():\n    from pkg import deep, inner\n",
        "tests/group/table.json": "{}",
        "src/pkg/__init__.py": "",
        "src/pkg/inner.py": "import importlib\nimportlib.import_module('pkg.late')\n",
        # A custom-op domain, which qonnx's registry imports; a dotted name that is no
        # module of the checkout.
        "src/pkg/deep.py": "import numpy\nDOMAIN = 'pkg.ops'\nOTHER = 'numpy.linalg'\n",
        "src/pkg/ops.py": "",
        "src/pkg/late.py": "",
        "src/pkg/unused.py": "",
        "src/pkg/hdl/top.sv": "module top; endmodule\n",
        "src/pkg/sub/__init__.py": "",
        "src/pkg/sub/other.sv": "",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    closure = tool.import_closure(tmp_path, ["tests/group/test_a.py"])
    assert closure == sorted(
        [
            "tests/group/__init__.py",
            "tests/group/test_a.py",
            "tests/group/helper.py",
            "src/pkg/__init__.py",
            "src/pkg/inner.py",
            "src/pkg/deep.py",
            "src/pkg/late.py",
            "src/pkg/ops.py",
        ]
    )
    # A package's data, not a package it contains (sub is a package; nothing imports it).
    assert tool.package_data(tmp_path, closure) == ["src/pkg/hdl/top.sv", "tests/group/table.json"]


# -- the selection ---------------------------------------------------------------------------


def _baseline(**rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "commit": "0123456789abcdef",
        "exit": 0,
        "smoke": False,
        "jobs": [{"job": name, **row} for name, row in rows.items()],
    }


def _current(**keys: str | None) -> dict[str, Any]:
    return {
        "jobs": {
            name: {"key": key, "inputs": {"designs": key or "", "vivado": "v"}}
            if key
            else {"key": None, "error": "capture failed"}
            for name, key in keys.items()
        }
    }


def test_a_job_runs_unless_a_run_under_the_same_key_passed_it() -> None:
    """A skip cites the commit where the job last ran and passed: the baseline's for a
    job it ran, the one a row it skipped carries for a job it skipped."""
    baseline = _baseline(
        same={"exit": 0, "key": "k1", "inputs": {"designs": "k1", "vivado": "v"}},
        changed={"exit": 0, "key": "k2", "inputs": {"designs": "k2", "vivado": "v"}},
        failed={"exit": 1, "key": "k3"},
        carried={
            "exit": None,
            "skipped": "unchanged since fedcba987654",
            "key": "k4",
            "ran_at": "fedcba9876543210",
        },
        # A skipped row of an older summary names no run: nothing vouches for it.
        uncited={"exit": None, "skipped": "unchanged since 89abcdef", "key": "k8"},
        unkeyed={"exit": 0, "key": None},
        broken={"exit": 0, "key": "k6"},
    )
    current = _current(
        same="k1",
        changed="k2'",
        failed="k3",
        carried="k4",
        uncited="k8",
        unkeyed="k5",
        broken=None,
        new="k7",
    )
    decided = {d.job: (d.run, d.reason) for d in tool.select(baseline, current)}
    assert decided == {
        "same": (False, "unchanged since 0123456789ab"),
        "changed": (True, "changed: designs"),
        "failed": (True, "the baseline did not pass it (exit=1)"),
        "carried": (False, "unchanged since fedcba987654"),
        "uncited": (True, "the baseline skipped it and names no run that passed it"),
        "unkeyed": (True, "the baseline has no key for it"),
        "broken": (True, "no key: capture failed"),
        "new": (True, "not in the baseline"),
    }


def test_an_older_checkout_has_only_the_sweeps_whose_modules_it_has(tmp_path: Path) -> None:
    """compare emits an older commit's text: a sweep added since is no job of it, and so
    runs as one not in the baseline."""
    sweep = tool.Job("sweep-new", "sweep", ("kernels.sweeps.new_numeric",))
    module = tmp_path / "tests/kernels/sweeps/new_numeric.py"
    assert not tool._has_module(tmp_path, sweep)
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert tool._has_module(tmp_path, sweep)


def test_a_summary_reports_skipped_jobs_and_fails_on_a_baseline_that_did_not_pass(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "jobs.tsv").write_text("ran\tsweep\tm\nskipped\tsweep\tm\ncarried\tsweep\tm\n")
    (out / "selection.tsv").write_text(
        "ran\trun\tchanged: designs\nskipped\tskip\tunchanged since abc\n"
        "carried\tskip\tunchanged since fedcba\n"
    )
    (out / "keys.json").write_text(json.dumps(_current(ran="k1", skipped="k2", carried="k3")))
    (out / "logs" / "ran.log").write_text("PASS one\nexit=0\n")
    # The baseline ran ``skipped`` and passed it, and skipped ``carried``, last run elsewhere.
    before: dict[str, dict[str, Any]] = {
        "skipped": {"exit": 0, "key": "k2", "ran_at": "0123456789abcdef"},
        "carried": {"exit": None, "skipped": "x", "key": "k3", "ran_at": "fedcba9876543210"},
    }
    for baseline, expected in (
        (_baseline(**before), 0),
        ({**_baseline(**before), "smoke": True}, 1),
    ):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(baseline))
        status = tool.summarize(
            out,
            status=0,
            identity="finn x",
            smoke=False,
            dirty=False,
            collect_code=0,
            baseline=path,
        )
        assert status == expected
        summary = json.loads((out / "summary.json").read_text())
        rows = {row["job"]: row for row in summary["jobs"]}
        assert rows["ran"]["exit"] == 0 and rows["ran"]["key"] == "k1"
        # Where each job last ran: here, or where the baseline's run of it was.
        assert rows["ran"]["ran_at"] == summary["commit"]
        assert rows["skipped"]["ran_at"] == "0123456789abcdef"
        assert rows["carried"]["ran_at"] == "fedcba9876543210"
        assert rows["skipped"]["skipped"] == "unchanged since abc"
        assert rows["skipped"]["key"] == "k2" and rows["skipped"]["exit"] is None
        assert "skipped skipped (unchanged since abc) key=k2" in (out / "summary.log").read_text()
