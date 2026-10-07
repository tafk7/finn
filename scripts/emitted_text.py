# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The text FINN emits for the kernel layer's evidence, and the XSim sweep's jobs keyed by it.

Nothing here runs Vivado. Each simulation the sweep would run is materialized
by the job's own code with the simulator replaced by a capture:

- a conformance job (one pytest id of tests/kernels/test_conformance.py) runs
  under pytest with ``kernels.xsim.simulate`` capturing the simulation
  directory: the staged module sources (FinnLib's included, copied by
  content), each memory's INIT_FILE, and the testbench ``check.sv``, which
  carries the stimulus and the expected words;
- a numeric sweep (``python -m kernels.sweeps.<module> ARGS``) runs its
  ``main`` with ``rtl_transport._run_worker`` capturing the request it would
  hand the simulation process: its sources by content (with every file of a
  header's directory), stimulus, expected counts and observations. A run of
  one design simulates it once per transport mode; the first request is
  captured and the run ends there, so the other modes, which differ only in
  the harness's flags, are covered by the harness's files.

Commands (run from any FINN checkout; ``--root`` names the checkout whose text
is emitted, by default this one):

  emitted_text.py emit OUT [--blank-hashes]       the text, one file per item, under OUT:
      designs/<job>/...   what each XSim job simulates
      ipxact/<job>/<sample>/   each conformance sample's IP-XACT text (interface.tcl, ifnames.json)
      package/                 PackagePartition's package.tcl and metadata (Chain, TFC), no Vivado
  emitted_text.py jobs [--smoke] [--collect-log F]  the sweep's jobs, one per line: name, kind, args
  emitted_text.py keys --work DIR [--jobs F]        each job's key, as JSON on stdout
  emitted_text.py select BASELINE KEYS              run or skip, with the reason, per job
  emitted_text.py summarize OUT ...                 summary.log and summary.json (xsim-sweep.sh)
  emitted_text.py compare BASE HEAD --work DIR      the selection between two commits

``--blank-hashes`` replaces each 16-hex module hash (an identifier's ``_<hash>``)
with ``#``, in file names and contents, so that two commits' outputs compare
by ``diff -r`` on what changed beside the hash.

A job's key is a digest of everything its simulation consumes: the captured
designs, the files of the test tree the job imports (harness, stimulus and
reference code) except construction modules, a numeric sweep's construction
results, the simulator runtime (``finn.xsi``, ``finn_xsi`` and the modules they
load) when the job drives XSI, the sweep's own scripts, the pytest
configuration, the Python environment, the selected Vivado and this tool. The
three pytest groups (``kernels-rest``, ``kernel-ops-xsim``,
``kernel-ops-vivado``) simulate from inside test bodies that no single stub
reaches, and ``kernels-rest`` also runs Python-only tests: their key is the
digest of every tracked file under src/, tests/ and scripts/ and of the whole
FinnLib tree, so any change to FINN's code or tests runs them.

A construction module (one that declares ``XSIM_KEY = "construction"``, as
tests/kernels/helpers.py does) builds kernels, configurations and samples. Its
code is not an input: what it builds reaches a simulation through the captured
designs, and what a numeric sweep reads from it besides (a MatMul assembly's beat
counts and result type, after its simulation) is recorded as the job's
construction results: each call the sweep makes to it and the plain data it
returned. So a rename in construction code changes no key unless what it builds
changes. A module without the declaration (any module of an older checkout) is
keyed by its code.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL = Path(__file__).resolve()
HERE = TOOL.parents[1]
CONFORMANCE = "tests/kernels/test_conformance.py"

# The numeric sweeps: (job name, module, arguments). Each runs as
# ``python -m <module> <arguments> --output <OUT>/sim-<name without sweep->``.
SWEEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sweep-dense", "kernels.sweeps.matmul_numeric", ()),
    (
        "sweep-fifo-packed",
        "kernels.sweeps.matmul_numeric",
        ("--case", "packed", "--weight-fifo-depth", "2"),
    ),
    (
        "sweep-fifo-int8-pumped",
        "kernels.sweeps.matmul_numeric",
        ("--case", "int8_pumped", "--weight-fifo-depth", "2"),
    ),
    ("sweep-depthwise", "kernels.sweeps.matmul_numeric", ("--depthwise",)),
    ("sweep-memstream", "kernels.sweeps.matmul_numeric", ("--delivery", "memstream")),
    (
        "sweep-memstream-depthwise",
        "kernels.sweeps.matmul_numeric",
        ("--depthwise", "--delivery", "memstream"),
    ),
    ("sweep-pumped-memory", "kernels.sweeps.matmul_numeric", ("--pumped-memory",)),
    ("sweep-sets", "kernels.sweeps.matmul_numeric", ("--sets", "3")),
    ("sweep-dotp", "kernels.sweeps.pure_dot_product_numeric", ()),
    ("sweep-dotp-stress", "kernels.sweeps.pure_dot_product_numeric", ("--stress",)),
    ("sweep-adapters", "kernels.sweeps.adapter_numeric", ()),
    ("sweep-thresholds", "kernels.sweeps.threshold_numeric", ()),
)
SMOKE_SWEEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sweep-packed", "kernels.sweeps.matmul_numeric", ("--case", "packed")),
)
# The pytest groups: (job name, pytest arguments), run with Vivado selected.
PYTEST_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "kernels-rest",
        ("--confcutdir=tests/kernels", "tests/kernels", "--ignore=" + CONFORMANCE),
    ),
    ("kernel-ops-xsim", ("--confcutdir=tests/kernel_ops", "tests/kernel_ops", "-m", "xsim")),
    ("kernel-ops-vivado", ("--confcutdir=tests/kernel_ops", "tests/kernel_ops", "-m", "vivado")),
)
# What every job's run reads besides its own inputs: the sweep, the environment it
# applies, the pytest configuration and the locked Python environment.
SWEEP_FILES = (
    "scripts/xsim-sweep.sh",
    "scripts/activate.sh",
    "docker/finn-toolchain.sh",
    ".pytest.ini",
    "uv.lock",
)
# The tracked trees a pytest group's key covers, beside SWEEP_FILES.
TREES = ("src", "tests", "scripts", "pyproject.toml")
HASH = re.compile(r"(?<=_)[0-9a-f]{16}(?![0-9A-Za-z])")


@dataclass(frozen=True)
class Job:
    """One job of the sweep. ``kind``: conformance, sweep or pytest.

    ``args`` are what the sweep runs: pytest's arguments, or a sweep's module and
    its arguments (``--output`` is the sweep's).
    """

    name: str
    kind: str
    args: tuple[str, ...]

    def line(self) -> str:
        return "\t".join((self.name, self.kind, *self.args))

    @classmethod
    def parse(cls, line: str) -> Job:
        name, kind, *args = line.rstrip("\n").split("\t")
        return cls(name, kind, tuple(args))


# -- the jobs ------------------------------------------------------------------------------


def _environment(root: Path, finnlib: Path | None = None) -> dict[str, str]:
    """The environment a job's code runs in at ``root``: its src and tests, nothing else."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'src'}:{root / 'tests'}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FINN_ROOT"] = str(root)
    env.pop("FORCE_COLOR", None)
    if finnlib is not None:
        env["FINN_RESOURCES_FINNLIB"] = str(finnlib)
    return env


def conformance_ids(root: Path, log: Path | None = None, finnlib: Path | None = None) -> list[str]:
    """The conformance XSim test ids, as the sweep collects them.

    With the project's addopts cleared and one -q, pytest prints one path::id
    line per test. A collection that fails or finds nothing is an error: an empty
    list would run no conformance job and could still pass.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "--confcutdir=tests/kernels",
            CONFORMANCE,
            "-m",
            "xsim",
        ],
        cwd=root,
        env=_environment(root, finnlib),
        capture_output=True,
        text=True,
    )
    if log is not None:
        log.write_text(result.stdout + result.stderr)
    ids = [line for line in result.stdout.splitlines() if line.startswith(CONFORMANCE + "::")]
    if result.returncode != 0 or not ids:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        raise RuntimeError(
            f"conformance collection failed (exit={result.returncode}, {len(ids)} jobs):\n{tail}"
        )
    return ids


def conformance_job(test_id: str) -> Job:
    name = "conformance-" + re.sub(r"[^A-Za-z0-9_-]", "_", test_id.split("::", 1)[1])
    return Job(name, "conformance", ("--confcutdir=tests/kernels", test_id))


def jobs(
    root: Path, smoke: bool = False, log: Path | None = None, finnlib: Path | None = None
) -> list[Job]:
    """The sweep's jobs at ``root``, in the order it starts them."""
    ids = conformance_ids(root, log, finnlib)
    if smoke:
        return [conformance_job(ids[0])] + [
            Job(name, "sweep", (module, *args)) for name, module, args in SMOKE_SWEEPS
        ]
    return [
        *map(conformance_job, ids),
        *(Job(name, "pytest", args) for name, args in PYTEST_GROUPS),
        *(Job(name, "sweep", (module, *args)) for name, module, args in SWEEPS),
    ]


# -- digests -------------------------------------------------------------------------------


def file_digest(path: Path) -> str:
    """A file's content digest; ``missing`` for a file that does not exist."""
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Code(ast.NodeTransformer):
    """A syntax tree less its docstrings and imports; the names each import binds, collected."""

    DEFINITIONS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

    def __init__(self) -> None:
        self.bindings: set[tuple[str, str]] = set()

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not all(isinstance(item, ast.stmt) for item in body):
            return node
        kept = []
        for statement in body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self.bindings |= {
                    (alias.asname or alias.name, alias.name) for alias in statement.names
                }
            else:
                kept.append(statement)
        if (
            isinstance(node, self.DEFINITIONS)
            and kept
            and isinstance(kept[0], ast.Expr)
            and isinstance(kept[0].value, ast.Constant)
            and isinstance(kept[0].value.value, str)
        ):
            kept = kept[1:]
        setattr(node, "body", kept or [ast.Pass()])
        return node


def code_digest(path: Path) -> str:
    """A Python file's code: what it executes, not how it is written down.

    The digest of its syntax tree less docstrings, with each import reduced to
    the names it binds: comments, formatting, docstrings and the module an import
    names (a renamed or reordered import) do not count. What an imported name
    does counts where it matters: a test-tree module is an input of its own, and
    FINN's code reaches a simulation through the designs it emits. Any other file,
    or one that does not parse, by content.
    """
    if path.suffix != ".py" or not path.is_file():
        return file_digest(path)
    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError:
        return file_digest(path)
    code = _Code()
    stripped = ast.dump(code.visit(tree), include_attributes=False)
    imports = repr(sorted(code.bindings))
    return hashlib.sha256(f"{stripped}\0{imports}".encode()).hexdigest()


def _files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
            yield path


def tree_digest(directory: Path) -> str:
    """A digest of every file's relative path and content under ``directory``."""
    digest = hashlib.sha256()
    for path in _files(directory):
        digest.update(f"{path.relative_to(directory)}\0{file_digest(path)}\n".encode())
    return digest.hexdigest()


def tracked_digest(root: Path, paths: Sequence[str]) -> str:
    """A digest of the tracked files under ``paths`` at ``root``, as they are on disk."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *paths], cwd=root, capture_output=True, check=True
    ).stdout.decode()
    digest = hashlib.sha256()
    for name in sorted(filter(None, listed.split("\0"))):
        digest.update(f"{name}\0{file_digest(root / name)}\n".encode())
    return digest.hexdigest()


def vivado_identity() -> str:
    """The selected Vivado: its installation and its version file's digest; ``none``."""
    selected = os.environ.get("XILINX_VIVADO")
    if not selected:
        return "none"
    vivado = Path(selected).resolve()
    return f"{vivado} {file_digest(vivado / 'data' / 'version.dat')[:16]}"


def key_of(inputs: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(inputs), sort_keys=True).encode()).hexdigest()


def common_inputs(root: Path) -> dict[str, str]:
    """What every job's run consumes besides its designs, harness and simulator."""
    inputs = {f"sweep:{name}": file_digest(root / name) for name in SWEEP_FILES}
    inputs["python"] = sys.version.split()[0]
    inputs["vivado"] = vivado_identity()
    inputs["key-tool"] = code_digest(TOOL)
    return inputs


# -- capture (runs in the job's environment, at its root) ---------------------------------


class _Captured(BaseException):
    """A numeric run's first simulation request is captured: end the run there."""


CONSTRUCTION = "construction"
"""A module's ``XSIM_KEY`` when it only constructs (see the module docstring)."""


def _imported(root: Path, construction: bool = False) -> list[str]:
    """The files under ``root`` the process imported, relative to it; with
    ``construction``, only the construction modules'."""
    found = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if (
            not name
            or Path(name).resolve() == TOOL
            or not Path(name).resolve().is_relative_to(root)
        ):
            continue
        if not construction or getattr(module, "XSIM_KEY", None) == CONSTRUCTION:
            found.add(str(Path(name).resolve().relative_to(root)))
    return sorted(found)


_OPAQUE = object()
"""What ``_plain`` returns for a value that is not plain data."""


def _plain(value: Any) -> Any:
    """``value`` as JSON when it is plain data (a scalar, an enum member, a QONNX
    datatype, a sequence of those); otherwise ``_OPAQUE``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    canonical = getattr(value, "get_canonical_name", None)  # a QONNX datatype
    if callable(canonical):
        return f"datatype {canonical()}"
    if isinstance(value, (tuple, list)):
        items = [_plain(item) for item in value]
        return _OPAQUE if any(item is _OPAQUE for item in items) else items
    return _OPAQUE


def _result(value: Any) -> Any:
    """A construction call's result as recorded: plain data as such; a dataclass by its
    fields, each one that is not plain data by its type (a module reaches the
    designs); anything else by its type."""
    plain = _plain(value)
    if plain is not _OPAQUE:
        return plain
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        recorded = {}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            found = _plain(item)
            recorded[field.name] = type(item).__name__ if found is _OPAQUE else found
        return {"type": type(value).__name__, "fields": recorded}
    return {"type": type(value).__name__}


def _record_construction(module: Any, calls: list[dict[str, Any]]) -> None:
    """Each construction function ``module`` names records its result into ``calls``."""

    def recorded(function: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            result = function(*args, **kwargs)
            calls.append({"call": function.__name__, "result": _result(result)})
            return result

        return run

    for name, value in list(vars(module).items()):
        owner = sys.modules.get(getattr(value, "__module__", ""))
        if inspect.isfunction(value) and getattr(owner, "XSIM_KEY", None) == CONSTRUCTION:
            setattr(module, name, recorded(value))


def _copy_tree(source: Path, target: Path) -> None:
    for path in _files(source):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _write_ipxact(module: Any, target: Path) -> None:
    """The module's IP-XACT interface Tcl and interface names, as packaging writes them.

    A checkout from before FINN wrote IP-XACT text says so in ``unavailable.txt``.
    """
    target.mkdir(parents=True, exist_ok=True)
    try:
        from finn.kernels.artifacts.ipxact import (  # noqa: PLC0415 - the checkout's
            interface_names,
            interface_tcl,
        )

        pins = module.abi.pins
    except (ImportError, AttributeError) as error:  # the checkout lacks the API
        (target / "unavailable.txt").write_text(f"{type(error).__name__}: {error}\n")
        return
    (target / "interface.tcl").write_text("\n".join(interface_tcl(pins, 5.0)) + "\n")
    (target / "ifnames.json").write_text(json.dumps(interface_names(pins), indent=1) + "\n")


def capture_test(dest: Path, root: Path, pytest_args: Sequence[str], ipxact: Path | None) -> int:
    """Run one conformance test with every simulation captured into ``dest/designs``."""
    # The checkout's harness, importable only in its environment (Target.run).
    import kernels.conformance as conformance  # noqa: PLC0415
    import kernels.xsim as xsim  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    designs = dest / "designs"
    state: dict[str, Any] = {"tmp": None, "simulations": 0, "started": 0, "completed": 0}

    def where(directory: Path) -> Path:
        tmp = state["tmp"]
        return directory.relative_to(tmp) if tmp is not None else Path(directory.name)

    def simulate(sources: Sequence[str | Path], testbench: str, directory: Path) -> None:
        directory = Path(directory)
        target = designs / where(directory)
        _copy_tree(directory, target)
        (target / "check.sv").write_text(testbench)
        for source in map(Path, sources):  # a source outside the directory: by content
            if not source.resolve().is_relative_to(directory.resolve()):
                outside = target / "outside" / source.name
                outside.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, outside)
        state["simulations"] += 1

    def counted(function: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            state["started"] += 1
            result = function(*args, **kwargs)
            state["completed"] += 1
            return result

        return run

    xsim.vivado_simulator = lambda: True  # selected or not: nothing here simulates
    xsim.simulate = simulate
    conformance.conformance = counted(conformance.conformance)
    if ipxact is not None:
        streamed = conformance.stream_through

        def stream_through(module: Any, directory: Path, **kwargs: Any) -> None:
            _write_ipxact(module, ipxact / where(Path(directory)).parent)
            streamed(module, directory, **kwargs)

        conformance.stream_through = stream_through

    class Plugin:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item: Any) -> Iterator[None]:
            tmp = item.funcargs.get("tmp_path")
            state["tmp"] = Path(tmp) if tmp is not None else None
            yield

    code = pytest.main(
        ["-q", "-p", "no:cacheprovider", f"--basetemp={dest / 'tmp'}", *pytest_args],
        plugins=[Plugin()],
    )
    # The test's verdict is not the question (a wrong-order test fails when nothing
    # simulates); that every conformance call ran to its end and simulated is.
    meta = {
        "pytest_exit": int(code),
        "simulations": state["simulations"],
        "conformance_calls": [state["started"], state["completed"]],
        "imported": _imported(root),
        "construction": _imported(root, construction=True),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    complete = state["started"] == state["completed"] >= 1 and state["simulations"] >= 1
    if not complete:
        print(f"incomplete capture: {meta}", file=sys.stderr)
    return 0 if complete else 1


def capture_sweep(dest: Path, root: Path, module_name: str, args: Sequence[str]) -> int:
    """Run a numeric sweep module with each run's first simulation request captured."""
    # The checkout's harness, importable only in its environment (Target.run).

    from finn import resources  # noqa: PLC0415

    designs, evidence = dest / "designs", dest / "evidence"
    finnlib = Path(resources.path("finnlib")).resolve()
    prefixes = ((evidence.resolve(), "OUTPUT"), (finnlib, "FINNLIB"), (root, "ROOT"))
    captured: list[str] = []

    def mapped(path: str) -> str:
        resolved = Path(path).resolve()
        for prefix, name in prefixes:
            if resolved.is_relative_to(prefix):
                return str(Path(name) / resolved.relative_to(prefix))
        raise ValueError(f"a simulation reads {path}, outside the run's known roots")

    def normalized(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("/"):
            return mapped(value)
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        return value

    def run_worker(top_module: str, request: Path, *_: Any, **__: Any) -> int:
        data = json.loads(Path(request).read_text())
        target = designs / f"{len(captured):03d}-{top_module}"
        sources = [Path(source) for source in data["sources"]]
        # A header's directory is an include directory: every file in it may be read.
        headers = {source.parent for source in sources if source.suffix in (".svh", ".vh")}
        for path in [*sources, *(file for folder in headers for file in _files(folder))]:
            copy = target / "sources" / mapped(str(path))
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
        target.mkdir(parents=True, exist_ok=True)
        (target / "request.json").write_text(
            json.dumps(normalized(data), indent=1, sort_keys=True) + "\n"
        )
        captured.append(top_module)
        raise _Captured

    # The sweeps' transport sits beside them (kernels.sweeps.rtl_transport).
    transport = importlib.import_module(module_name.rpartition(".")[0] + ".rtl_transport")
    transport._run_worker = run_worker
    module = importlib.import_module(module_name)
    calls: list[dict[str, Any]] = []
    _record_construction(module, calls)
    depth = [0]

    def guarded(function: Callable[..., Any]) -> Callable[..., Any]:
        """The outermost of the module's functions ends its run at the capture."""

        def run(*args: Any, **kwargs: Any) -> Any:
            depth[0] += 1
            try:
                return function(*args, **kwargs)
            except _Captured:
                if depth[0] > 1:
                    raise
                return None
            finally:
                depth[0] -= 1

        return run

    for name, value in list(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module.__name__ and name != "main":
            setattr(module, name, guarded(value))
    sys.argv = [module_name, *args, "--output", str(evidence)]
    try:
        module.main()
    except _Captured:
        print(f"{module_name}: main simulates outside a run; not captured", file=sys.stderr)
        return 1
    meta = {
        "simulations": len(captured),
        "imported": _imported(root),
        "construction": _imported(root, construction=True),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (dest / "construction.json").write_text(json.dumps(calls, indent=1) + "\n")
    return 0 if captured else 1


def simulator_closure(root: Path) -> list[str]:
    """The files the XSI runtime loads: finn.xsi's and finn_xsi's packages and their imports."""
    found = set()
    for name in ("finn.xsi", "finn_xsi"):
        package = importlib.import_module(name)
        for info in pkgutil.walk_packages(package.__path__, name + "."):
            try:
                importlib.import_module(info.name)
            except ImportError:  # the compiled bridge (xsi) is built on first use
                pass
        for folder in package.__path__:
            found |= {str(path.relative_to(root)) for path in _files(Path(folder).resolve())}
    found |= set(_imported(root))
    return sorted(found)


def emit_package(out: Path, work: Path) -> int:
    """What PackagePartition writes for the Chain and the TFC partition, Vivado stubbed.

    PackagePartition runs for real except for the Vivado call: the toolchain is a
    stub that writes ip/component.xml. Per partition (the Chain also with
    run_synth): package.tcl and the stitch metadata, the work prefix removed.
    """
    # The checkout's code, importable only in its environment (Target.run).
    from kernel_ops.models import configure_partition, kernel_model  # noqa: PLC0415
    from kernel_ops.tfc import partitioned  # noqa: PLC0415

    from finn.transformation.kernels import PackagePartition  # noqa: PLC0415

    class StubToolchain:
        def run(self, tool: str, args: Any, *, cwd: Path, replay: Any = None, **_: Any) -> None:
            (Path(cwd) / "ip").mkdir(parents=True, exist_ok=True)
            (Path(cwd) / "ip" / "component.xml").write_text("<stub/>")

    def emit(name: str, model: Any, ip_name: str, run_synth: bool = False) -> None:
        project = work / name
        model = model.transform(
            PackagePartition(
                ip_name, run_synth=run_synth, directory=project, toolchain=StubToolchain()
            )
        )
        (out / f"{name}.package.tcl").write_text((project / "package.tcl").read_text())
        props = [
            f"{prop}={str(model.get_metadata_prop(prop)).replace(str(work), 'WORK')}"
            for prop in ("vivado_stitch_proj", "vivado_stitch_vlnv", "vivado_stitch_ifnames")
        ]
        (out / f"{name}.metadata.txt").write_text("\n".join(props) + "\n")

    out.mkdir(parents=True, exist_ok=True)
    work = work.resolve()
    configure_partition(chain := kernel_model())
    emit("chain", chain, "sdp_1")
    configure_partition(chain := kernel_model())
    emit("chain_synth", chain, "sdp_1", run_synth=True)
    (work / "tfc_build").mkdir(parents=True, exist_ok=True)
    _, parent, body = partitioned(work / "tfc_build")
    emit("tfc", body, parent.graph.node[1].name)
    return 0


# -- capture and keys (the driver) ---------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A checkout to read, and the FinnLib its jobs compile (None: as FINN resolves it)."""

    root: Path
    finnlib: Path | None = None

    def run(self, *args: str | Path, log: Path) -> int:
        """This tool's command ``args`` in the checkout's environment; its exit status."""
        with log.open("w") as output:
            return subprocess.run(
                [sys.executable, str(TOOL), *map(str, args)],
                cwd=self.root,
                env=_environment(self.root, self.finnlib),
                stdout=output,
                stderr=subprocess.STDOUT,
            ).returncode

    def finnlib_path(self, work: Path) -> Path:
        log = work / "finnlib.log"
        code = self.run("_finnlib", work / "finnlib.txt", log=log)
        if code != 0:
            raise RuntimeError(f"FinnLib does not resolve at {self.root}: {log.read_text()}")
        return Path((work / "finnlib.txt").read_text().strip())


def _capture(target: Target, job: Job, dest: Path, ipxact: Path | None) -> str | None:
    """Capture one job's designs into ``dest``; an error, or None."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    log = dest / "capture.log"
    if job.kind == "conformance":
        extra = ("--ipxact", ipxact / job.name) if ipxact is not None else ()
        code = target.run("_capture-test", dest, *extra, "--", *job.args, log=log)
    else:
        code = target.run("_capture-sweep", dest, "--", *job.args, log=log)
    if code != 0:
        tail = " | ".join(log.read_text().strip().splitlines()[-3:])
        return f"capture failed (exit {code}, {log}): {tail}"
    return None


def compute_keys(
    target: Target,
    selected: Sequence[Job],
    work: Path,
    ipxact: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Each job's key and the inputs it digests; a job whose capture failed has an error."""
    work.mkdir(parents=True, exist_ok=True)
    root = target.root
    finnlib = target.finnlib_path(work)
    common = common_inputs(root)
    simulator: list[str] | None = None
    once = threading.Lock()

    def simulator_files() -> list[str]:
        nonlocal simulator
        with once:
            if simulator is None:
                log = work / "simulator.log"
                if target.run("_simulator-closure", work / "simulator.json", log=log) != 0:
                    raise RuntimeError(f"the XSI runtime does not import at {root}: {log}")
                simulator = json.loads((work / "simulator.json").read_text())
            return simulator

    def key(job: Job) -> dict[str, Any]:
        inputs = {**common, "job": " ".join((job.kind, *job.args))}
        if job.kind == "pytest":
            inputs["tree"] = tracked_digest(root, TREES)
            inputs["finnlib"] = tree_digest(finnlib)
            return {"key": key_of(inputs), "inputs": inputs}
        dest = work / job.name
        error = _capture(target, job, dest, ipxact)
        if error is not None:
            return {"key": None, "error": error}
        meta = json.loads((dest / "meta.json").read_text())
        inputs["designs"] = tree_digest(dest / "designs")
        imported = meta["imported"]
        construction = set(meta["construction"])
        for name in imported:
            if name.startswith("tests/") and name not in construction:
                inputs[f"harness:{name}"] = code_digest(root / name)
        if (dest / "construction.json").is_file():
            inputs["construction"] = file_digest(dest / "construction.json")
        if any(name.startswith(("src/finn/xsi/", "src/finn_xsi/")) for name in imported):
            for name in simulator_files():
                inputs[f"simulator:{name}"] = code_digest(root / name)
        return {"key": key_of(inputs), "inputs": inputs, "simulations": meta["simulations"]}

    with ThreadPoolExecutor(max_workers=workers or min(16, os.cpu_count() or 4)) as pool:
        results = dict(zip((job.name for job in selected), pool.map(key, selected)))
    return {"root": str(root), "commit": _commit(root), "finnlib": str(finnlib), "jobs": results}


def _commit(root: Path) -> str | None:
    found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return found.stdout.strip() if found.returncode == 0 else None


# -- selection -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    job: str
    run: bool
    reason: str


def _differences(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    shown = ", ".join(changed[:6]) + (f" and {len(changed) - 6} more" if len(changed) > 6 else "")
    return f"changed: {shown}" if changed else "changed (the baseline lists no inputs)"


def baseline_passed(baseline: Mapping[str, Any]) -> bool:
    """A baseline is a sweep that passed overall, and not a smoke run."""
    return baseline.get("exit") == 0 and baseline.get("smoke") is False


def select(baseline: Mapping[str, Any], keys: Mapping[str, Any]) -> list[Decision]:
    """Run each job whose key differs from the baseline's, or that the baseline did not pass."""
    commit = str(baseline.get("commit") or "unknown")[:12]
    rows = {row["job"]: row for row in baseline.get("jobs", [])}
    decisions = []
    for name, current in keys["jobs"].items():
        row = rows.get(name)
        if current.get("key") is None:
            decisions.append(Decision(name, True, f"no key: {current.get('error')}"))
        elif row is None:
            decisions.append(Decision(name, True, "not in the baseline"))
        elif not row.get("key"):
            decisions.append(Decision(name, True, "the baseline has no key for it"))
        elif not (row.get("exit") == 0 or row.get("skipped")):
            decisions.append(
                Decision(name, True, f"the baseline did not pass it (exit={row.get('exit')})")
            )
        elif row["key"] != current["key"]:
            reason = _differences(row.get("inputs") or {}, current["inputs"])
            decisions.append(Decision(name, True, reason))
        else:
            decisions.append(Decision(name, False, f"unchanged since {commit}"))
    return decisions


# -- the summary ---------------------------------------------------------------------------


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def summarize(
    out: Path,
    *,
    status: int,
    identity: str,
    smoke: bool,
    dirty: bool,
    collect_code: int,
    baseline: Path | None,
) -> int:
    """Write summary.log and summary.json from OUT's jobs, logs, keys and selection.

    A failed job fails the sweep; so does a baseline that did not pass. A job the
    selection skipped is reported with the key it was skipped under.
    """
    listed = [Job.parse(line) for line in _lines(out / "jobs.tsv")]
    keys = _json(out / "keys.json") or {"jobs": {}}
    skipped = {
        decision[0]: decision[2]
        for decision in (line.split("\t", 2) for line in _lines(out / "selection.tsv"))
        if decision[1] == "skip"
    }
    head = _commit(HERE) or "unknown"
    lines = [f"commit {head} checkout {HERE}", identity]
    collected = [
        line for line in _lines(out / "collect.log") if line.startswith(CONFORMANCE + "::")
    ]
    conformance = len(collected)
    lines.append(f"conformance collected={conformance} collection exit={collect_code}")
    if baseline is not None:
        record = _json(baseline) or {}
        lines.append(f"changed since {record.get('commit')} ({baseline})")
        if not baseline_passed(record):
            lines.append(
                "the baseline did not pass (overall exit=0, not smoke): it vouches for nothing"
            )
            status = 1
    if not keys["jobs"]:
        lines.append(f"keys: none computed (see {out / 'keys.log'})")
    rows = []
    for job in listed:
        current = keys["jobs"].get(job.name, {})
        row: dict[str, Any] = {"job": job.name, "key": current.get("key")}
        if job.name in skipped:
            row |= {"exit": None, "skipped": skipped[job.name]}
            lines.append(f"{job.name} skipped ({skipped[job.name]}) key={str(row['key'])[:12]}")
        else:
            log = out / "logs" / f"{job.name}.log"
            text = log.read_text(errors="replace") if log.is_file() else ""
            codes = re.findall(r"^exit=(\d+)$", text, flags=re.MULTILINE)
            code = int(codes[-1]) if codes else None
            status = status or (0 if code == 0 else 1)
            row |= {
                "exit": code,
                "passes": _count(r"^PASS|PASSED", text),
                "fails": _count(r"^FAIL|FAILED|^Traceback", text),
                "skips": _count(r"SKIPPED", text),
            }
            if job.kind == "sweep":
                tally = f"passes={row['passes']} fails={row['fails']}"
            else:
                tallies = re.findall(r"^.*(?: passed| failed| skipped).*$", text, re.MULTILINE)
                tally = tallies[-1] if tallies else ""
            lines.append(f"{job.name} exit={code} {tally}".rstrip())
        if current.get("inputs"):
            row["inputs"] = current["inputs"]
        if current.get("error"):
            row["key_error"] = current["error"]
        rows.append(row)
    lines.append(f"overall exit={status}")
    (out / "summary.log").write_text("\n".join(lines) + "\n")
    summary = {
        "commit": head,
        "dirty": dirty,
        "smoke": smoke,
        "exit": status,
        "conformance_collected": conformance,
        "baseline": (_json(baseline) or {}).get("commit") if baseline is not None else None,
        "jobs": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))
    return status


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line] if path.is_file() else []


def _json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


# -- emitted text --------------------------------------------------------------------------


def blank_hashes(text: str) -> str:
    return HASH.sub("#" * 16, text)


def _publish(source: Path, target: Path, blank: bool) -> None:
    """Copy a tree, its 16-hex module hashes blanked in names and text when ``blank``."""
    for path in _files(source):
        relative = str(path.relative_to(source))
        destination = target / (blank_hashes(relative) if blank else relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if blank:
            try:
                data = blank_hashes(data.decode()).encode()
            except UnicodeDecodeError:
                pass
        destination.write_bytes(data)


def emit(target: Target, out: Path, work: Path, blank: bool, strict: bool = True) -> dict[str, Any]:
    """The emitted text into ``out`` (designs, ipxact, package); the keys of its jobs.

    Unless ``strict``, an evidence section (IP-XACT, package) the checkout cannot
    emit says why in its ``unavailable.txt`` (an older checkout's API: compare);
    the designs, which the keys digest, are always required.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    listed = jobs(target.root, finnlib=target.finnlib)
    keys = compute_keys(target, listed, work / "keys", ipxact=work / "ipxact")
    failed = {name: entry["error"] for name, entry in keys["jobs"].items() if entry.get("error")}
    if failed:
        raise RuntimeError(f"capture failed: {failed}")
    for job in (job for job in listed if job.kind != "pytest"):
        _publish(work / "keys" / job.name / "designs", out / "designs" / job.name, blank)
    missing = sorted(work.glob("ipxact/**/unavailable.txt"))
    if strict and missing:
        raise RuntimeError(f"IP-XACT text unavailable: {missing[0].read_text().strip()}")
    _publish(work / "ipxact", out / "ipxact", blank)
    log = work / "package.log"
    if target.run("_emit-package", work / "package", work / "package-work", log=log) != 0:
        unavailable = [line for line in _lines(log) if re.match(r"^\w+(Error|Exception): ", line)]
        if strict or not unavailable:
            raise RuntimeError(f"PackagePartition's text failed: {log}")
        (work / "package").mkdir(parents=True, exist_ok=True)
        (work / "package" / "unavailable.txt").write_text(unavailable[-1] + "\n")
    _publish(work / "package", out / "package", blank)
    return keys


# -- compare -------------------------------------------------------------------------------


def _pinned_finnlib(tree: Path, repository: Path, work: Path) -> Path:
    """FinnLib at the commit ``tree`` pins, extracted from a local clone."""
    pin = tomllib.loads((tree / "src/finn/resources.toml").read_text())["resources"]["finnlib"][
        "commit"
    ]
    target = work / f"finnlib-{pin[:12]}"
    if not target.is_dir():
        target.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "-C", str(repository), "archive", pin], capture_output=True, check=True
        )
        subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)
    return target


def compare(
    base: str, head: str, work: Path, finnlib_repository: Path | None, keep: bool
) -> list[Decision]:
    """The jobs a sweep at ``head`` would run against a passed sweep at ``base``.

    Each commit is checked out detached under ``work`` (``git worktree add``,
    removed afterwards unless ``keep``); with ``finnlib_repository``, each reads
    the FinnLib commit it pins, extracted from that clone, else FinnLib as FINN
    resolves it. Both emit their text with hashes blanked into ``work/<sha>/text``
    for ``diff -r``.
    """
    work.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict[str, Any]] = {}
    for revision in (base, head):
        sha = subprocess.run(
            ["git", "rev-parse", "--verify", revision + "^{commit}"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        place = work / sha[:12]
        tree = place / "tree"
        if not tree.is_dir():
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(tree), sha],
                cwd=HERE,
                capture_output=True,
                check=True,
            )
        try:
            finnlib = (
                _pinned_finnlib(tree, finnlib_repository, work) if finnlib_repository else None
            )
            target = Target(tree.resolve(), finnlib)
            keys = emit(target, place / "text", place / "work", blank=True, strict=False)
            (place / "keys.json").write_text(json.dumps(keys, indent=1) + "\n")
            found[revision] = keys
        finally:
            if not keep:
                subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=HERE)
    before, after = found[base], found[head]
    texts = [work / str(keys["commit"])[:12] / "text" for keys in (before, after)]
    difference = subprocess.run(["diff", "-r", *map(str, texts)], capture_output=True, text=True)
    (work / "text.diff").write_text(difference.stdout)
    differing = subprocess.run(
        ["diff", "-rq", *map(str, texts)], capture_output=True, text=True
    ).stdout.splitlines()
    sections: dict[str, int] = {}
    for line in differing:  # "Files A and B differ", or "Only in DIR: NAME"
        found = re.search(rf"(?:{re.escape(str(texts[0]))}|{re.escape(str(texts[1]))})/(\w+)", line)
        section = found.group(1) if found else "?"
        sections[section] = sections.get(section, 0) + 1
    print(
        f"text ({work / 'text.diff'}): "
        + (
            ", ".join(f"{name} {count} differ" for name, count in sorted(sections.items()))
            or "identical"
        )
    )
    baseline = {
        "commit": before["commit"],
        "exit": 0,
        "smoke": False,
        "jobs": [{"job": name, "exit": 0, **entry} for name, entry in before["jobs"].items()],
    }
    return select(baseline, after)


# -- the command line ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    emitting = commands.add_parser("emit", help="the emitted text, one file per item")
    emitting.add_argument("out", type=Path)
    emitting.add_argument("--root", type=Path, default=HERE)
    emitting.add_argument("--blank-hashes", action="store_true")
    emitting.add_argument("--work", type=Path, help="scratch (default: a temporary directory)")

    listing = commands.add_parser("jobs", help="the sweep's jobs: name, kind, args (tab-separated)")
    listing.add_argument("--root", type=Path, default=HERE)
    listing.add_argument("--smoke", action="store_true")
    listing.add_argument("--collect-log", type=Path)

    keying = commands.add_parser("keys", help="each job's key, as JSON")
    keying.add_argument("--root", type=Path, default=HERE)
    keying.add_argument("--jobs", type=Path, help="jobs.tsv (default: the sweep's full list)")
    keying.add_argument("--work", type=Path, required=True)

    selecting = commands.add_parser("select", help="run or skip each job, against a baseline")
    selecting.add_argument("baseline", type=Path, help="summary.json of a passed full sweep")
    selecting.add_argument("keys", type=Path)

    summarizing = commands.add_parser("summarize", help="summary.log and summary.json")
    summarizing.add_argument("out", type=Path)
    summarizing.add_argument("--status", type=int, required=True)
    summarizing.add_argument("--identity", required=True)
    summarizing.add_argument("--smoke", action="store_true")
    summarizing.add_argument("--dirty", action="store_true")
    summarizing.add_argument("--collect-code", type=int, required=True)
    summarizing.add_argument("--baseline", type=Path)

    comparing = commands.add_parser("compare", help="the jobs HEAD would run against BASE")
    comparing.add_argument("base")
    comparing.add_argument("head")
    comparing.add_argument("--work", type=Path, required=True)
    comparing.add_argument("--finnlib-repo", type=Path, help="each commit's pinned FinnLib from it")
    comparing.add_argument("--keep", action="store_true", help="keep the worktrees")

    # Internal: run in a checkout's environment by the commands above.
    test = commands.add_parser("_capture-test")
    test.add_argument("dest", type=Path)
    test.add_argument("--ipxact", type=Path)
    test.add_argument("args", nargs="+")
    sweep = commands.add_parser("_capture-sweep")
    sweep.add_argument("dest", type=Path)
    sweep.add_argument("module")
    sweep.add_argument("args", nargs="*")
    closure = commands.add_parser("_simulator-closure")
    closure.add_argument("out", type=Path)
    package = commands.add_parser("_emit-package")
    package.add_argument("out", type=Path)
    package.add_argument("work", type=Path)
    resolve = commands.add_parser("_finnlib")
    resolve.add_argument("out", type=Path)

    args = parser.parse_args(argv)
    here = Path.cwd().resolve()
    if args.command == "emit":
        work = args.work or Path(tempfile.mkdtemp(prefix="emitted-text-"))
        emit(Target(args.root.resolve()), args.out, work, args.blank_hashes)
        print(args.out)
    elif args.command == "jobs":
        try:
            found = jobs(args.root.resolve(), args.smoke, args.collect_log)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print("\n".join(job.line() for job in found))
    elif args.command == "keys":
        root = args.root.resolve()
        listed = [Job.parse(line) for line in _lines(args.jobs)] if args.jobs else jobs(root)
        print(json.dumps(compute_keys(Target(root), listed, args.work), indent=1))
    elif args.command == "select":
        decisions = select(json.loads(args.baseline.read_text()), json.loads(args.keys.read_text()))
        for decision in decisions:
            print(f"{decision.job}\t{'run' if decision.run else 'skip'}\t{decision.reason}")
    elif args.command == "summarize":
        return summarize(
            args.out,
            status=args.status,
            identity=args.identity,
            smoke=args.smoke,
            dirty=args.dirty,
            collect_code=args.collect_code,
            baseline=args.baseline,
        )
    elif args.command == "compare":
        decisions = compare(args.base, args.head, args.work, args.finnlib_repo, args.keep)
        width = max(len(decision.job) for decision in decisions)
        for decision in decisions:
            verdict = "run " if decision.run else "skip"
            print(f"{decision.job:<{width}}  {verdict}  {decision.reason}")
    elif args.command == "_capture-test":
        return capture_test(args.dest.resolve(), here, args.args, args.ipxact)
    elif args.command == "_capture-sweep":
        return capture_sweep(args.dest.resolve(), here, args.module, args.args)
    elif args.command == "_simulator-closure":
        args.out.write_text(json.dumps(simulator_closure(here)) + "\n")
    elif args.command == "_emit-package":
        return emit_package(args.out, args.work)
    elif args.command == "_finnlib":
        from finn import resources  # noqa: PLC0415 - the checkout's

        args.out.write_text(str(Path(resources.path("finnlib")).resolve()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
