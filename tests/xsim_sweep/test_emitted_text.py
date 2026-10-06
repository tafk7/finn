# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""scripts/emitted_text.py: a job's key follows what its simulation consumes, and only that.

The keys are computed for real, without Vivado, on a copy of this checkout and of
FinnLib: a change to an input of a job's simulation (a FinnLib file its design
includes, its harness's code, the sweep script) changes the job's key; a file
the job does not consume (another kernel's RTL, an unrelated test, a comment or
an import's module path) does not. The selection runs exactly the jobs whose key
differs from a passed baseline's, or that the baseline did not pass.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
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
    finnlib = base / "finnlib"
    shutil.copytree(
        tool.Target(ROOT).finnlib_path(base),
        finnlib,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return root, finnlib


Edit = Callable[[Path, Path], None]


def _keys(root: Path, finnlib: Path, work: Path) -> dict[str, Any]:
    found = tool.compute_keys(tool.Target(root, finnlib), [MEMSTREAM, ADAPTERS], work)
    for entry in found["jobs"].values():
        assert entry.get("key"), entry
        assert entry["simulations"] > 0
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


def _edited(sandbox: tuple[Path, Path], tmp_path: Path, edit: Edit) -> dict[str, Any]:
    """The keys after ``edit``, on a fresh copy of the sandbox."""
    root, finnlib = tmp_path / "finn", tmp_path / "finnlib"
    shutil.copytree(sandbox[0], root, symlinks=True)
    shutil.copytree(sandbox[1], finnlib, symlinks=True)
    edit(root, finnlib)
    return _keys(root, finnlib, tmp_path / "keys")


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
    ("sweep-script", "scripts/xsim-sweep.sh", "\n# edited\n", BOTH),
    # Not consumed: another kernel's RTL, an unrelated test, a comment in the harness,
    # FINN code no design reaches.
    ("finnlib-dotp", "finnlib/rtl/linalg/dotp.sv", "\n// edited\n", set()),
    ("unrelated-test", "tests/kernels/test_dotp.py", "\nLIMIT = 1\n", set()),
    ("harness-comment", "tests/kernels/conformance.py", "\n# a comment\n", set()),
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


@pytest.mark.slow
def test_a_key_does_not_depend_on_where_the_checkout_is(
    sandbox: tuple[Path, Path], unedited: dict[str, Any], tmp_path: Path
) -> None:
    here = unedited
    moved = tmp_path / "moved"
    shutil.copytree(sandbox[0], moved / "finn", symlinks=True)
    shutil.copytree(sandbox[1], moved / "finnlib", symlinks=True)
    there = _keys(moved / "finn", moved / "finnlib", tmp_path / "there")
    assert {name: entry["key"] for name, entry in here.items()} == {
        name: entry["key"] for name, entry in there.items()
    }


def test_a_pytest_group_is_keyed_by_every_tracked_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("A = 1\n")
    (tmp_path / "README").write_text("text\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    before = tool.tracked_digest(tmp_path, tool.TREES)
    (tmp_path / "README").write_text("other text\n")
    assert tool.tracked_digest(tmp_path, tool.TREES) == before
    (tmp_path / "src" / "a.py").write_text("A = 2\n")
    assert tool.tracked_digest(tmp_path, tool.TREES) != before


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


def test_a_job_runs_unless_the_baseline_passed_it_under_the_same_key() -> None:
    baseline = _baseline(
        same={"exit": 0, "key": "k1", "inputs": {"designs": "k1", "vivado": "v"}},
        changed={"exit": 0, "key": "k2", "inputs": {"designs": "k2", "vivado": "v"}},
        failed={"exit": 1, "key": "k3"},
        carried={"exit": None, "skipped": "unchanged since 89abcdef", "key": "k4"},
        unkeyed={"exit": 0, "key": None},
        broken={"exit": 0, "key": "k6"},
    )
    current = _current(
        same="k1", changed="k2'", failed="k3", carried="k4", unkeyed="k5", broken=None, new="k7"
    )
    decided = {d.job: (d.run, d.reason) for d in tool.select(baseline, current)}
    assert decided == {
        "same": (False, "unchanged since 0123456789ab"),
        "changed": (True, "changed: designs"),
        "failed": (True, "the baseline did not pass it (exit=1)"),
        "carried": (False, "unchanged since 0123456789ab"),
        "unkeyed": (True, "the baseline has no key for it"),
        "broken": (True, "no key: capture failed"),
        "new": (True, "not in the baseline"),
    }


def test_a_summary_reports_skipped_jobs_and_fails_on_a_baseline_that_did_not_pass(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "jobs.tsv").write_text("ran\tsweep\tm\nskipped\tsweep\tm\n")
    (out / "selection.tsv").write_text(
        "ran\trun\tchanged: designs\nskipped\tskip\tunchanged since abc\n"
    )
    (out / "keys.json").write_text(json.dumps(_current(ran="k1", skipped="k2")))
    (out / "logs" / "ran.log").write_text("PASS one\nexit=0\n")
    for baseline, expected in ((_baseline(), 0), ({**_baseline(), "smoke": True}, 1)):
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
        assert rows["skipped"]["skipped"] == "unchanged since abc"
        assert rows["skipped"]["key"] == "k2" and rows["skipped"]["exit"] is None
        assert "skipped skipped (unchanged since abc) key=k2" in (out / "summary.log").read_text()
