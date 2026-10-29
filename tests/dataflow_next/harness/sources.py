# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Where the harness's cases come from: each source runs FINN as it derives a design, and
the recorder takes every question FINN asks ``finn.dataflow`` meanwhile.

- ``chain-8``, ``chain-64``: the kernel chain the Space oracle compares (8 or 64 kernels),
  every station answered by the public API, default numbering; ``chain-8-permuted``,
  ``chain-64-permuted``, the engine's permuted numbering;
- ``tfc``: TFC's partition after the cut and after each strategy sequence the oracle
  explores (it builds TFC from its export: a minute or so);
- ``conformance``: the conformance test's Python checks, every kernel case placed on its
  channels (no simulation);
- ``sweeps``: each numeric sweep job of the XSim sweep, every case built up to
  its simulation, which is stopped before it starts;
- ``cnv``: the convolution KernelOp's tests, CNV_W2A2 converted included (its export
  takes about a minute), Python checks only;
- ``geometries``: the cases the revision is meant to change (``geometries``);
- ``fifo-sizing``: the line's tests of FIFO sizing (``tests/kernels/test_fifo_sizing.py``, and
  ``tests/kernels/test_explore.py``, whose one ``SizeFifos`` test is refused before any
  channel is sized), which make the edge calls of ``fifo_sizing`` on
  purpose: buffered acceptances, depths above 0, a side that cannot be paced, a producer slower
  than the period. Its table is the ``fifo`` section's.

A source is run in a process of its own (``__main__``): the engine's numbering is read
when a model is made, and a source's imports are its own.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _prepare_chain(kernels: int, permuted: bool) -> Callable[[], None]:
    if permuted:
        os.environ["FINN_SPACE_PERMUTE"] = "1"
    from support.space_oracle import load  # noqa: PLC0415 - after the numbering is set
    from support.space_oracle.targets import dumped, stations  # noqa: PLC0415

    def run() -> None:
        public = load("public")
        for station in stations(f"chain-{kernels}"):
            dumped(station, public)

    return run


def _prepare_tfc() -> Callable[[], None]:
    from support.space_oracle import load  # noqa: PLC0415
    from support.space_oracle.targets import dumped  # noqa: PLC0415
    from support.space_oracle_tfc import tfc  # noqa: PLC0415

    def run() -> None:
        public = load("public")
        for station in tfc():
            dumped(station, public)

    return run


def _prepare_pytest(*arguments: str) -> Callable[[], None]:
    import pytest  # noqa: PLC0415

    def run() -> None:
        code = pytest.main(
            ["-q", "-p", "no:cacheprovider", "-m", "not xsim and not vivado", *arguments]
        )
        if code not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            raise RuntimeError(f"pytest {' '.join(arguments)} exited {code!r}")

    return run


class _Stopped(BaseException):
    """A sweep's run, ended where it would simulate: its design is derived by then."""


class ToolStarted(RuntimeError):
    """A source tried to start an external tool (Vivado, Vitis HLS, XSim's compiler):
    nothing in the harness may, so the source fails, naming the command."""


def forbid_tools() -> None:
    """Every name for FINN's one process launcher (``finn.toolchain.launcher.run_process``)
    replaced by a refusal, in its module and every loaded module that imported it."""
    import sys  # noqa: PLC0415

    import finn.toolchain.launcher as launcher  # noqa: PLC0415

    original = launcher.run_process

    def refused(argv: object, *_: object, **__: object) -> object:
        raise ToolStarted(f"a source started a tool: {argv!r}")

    for module in list(sys.modules.values()):
        for name, value in list(getattr(module, "__dict__", {}).items()):
            if value is original:
                setattr(module, name, refused)


def _sweep_jobs() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """The XSim sweep's numeric sweep jobs, as ``scripts/emitted_text.py`` lists them."""
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "emitted_text", ROOT / "scripts" / "emitted_text.py"
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool  # a dataclass reads its module from sys.modules
    spec.loader.exec_module(tool)
    jobs: tuple[tuple[str, str, tuple[str, ...]], ...] = tool.SWEEPS
    return jobs


def _prepare_sweeps() -> Callable[[], None]:
    """Each numeric sweep job's ``main``, with its simulation stopped before it starts (the
    tests' seam for the sweeps' transport, and the production testbench's ``simulate``),
    as the sweep's keying tool captures them; each of the module's functions ends its run
    at the stop, so the next case is built."""
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from support import capture  # noqa: PLC0415

    import finn.core.rtlsim_bench as bench  # noqa: PLC0415

    jobs = _sweep_jobs()
    modules = {name: importlib.import_module(name) for _, name, _ in jobs}

    def stop(*_: object, **__: object) -> None:
        raise _Stopped

    depth = [0]

    def guarded(function: Callable[..., object]) -> Callable[..., object]:
        """The outermost of the module's functions ends its run at the stop."""

        def run(*args: object, **kwargs: object) -> object:
            depth[0] += 1
            try:
                return function(*args, **kwargs)
            except _Stopped:
                if depth[0] > 1:
                    raise
                return None
            finally:
                depth[0] -= 1

        return run

    def run() -> None:
        capture.replace("sweep.transport", lambda _: stop)
        setattr(bench, "simulate", stop)  # noqa: B010 - replaced as the keying tool replaces it
        # An HLS leaf's product is built before its simulation: stopped there too.
        setattr(bench, "built_hls", stop)  # noqa: B010
        for module in modules.values():
            if hasattr(module, "print_identity"):
                setattr(module, "print_identity", lambda: None)  # noqa: B010 - asks the tools
            for name, value in list(vars(module).items()):
                if (
                    inspect.isfunction(value)
                    and value.__module__ == module.__name__
                    and name != "main"
                ):
                    setattr(module, name, guarded(value))
        evidence = Path(
            tempfile.mkdtemp(prefix="dataflow-next-sweeps-", dir=os.environ.get("FINN_BUILD_DIR"))
        )
        for job, name, arguments in jobs:
            sys.argv = [name, *arguments, "--output", str(evidence / job)]
            modules[name].main()

    return run


def _prepare_geometries() -> Callable[[], None]:
    from dataflow_next.harness import geometries  # noqa: PLC0415

    return geometries.run


PREPARED: Mapping[str, Callable[[], Callable[[], None]]] = {
    "chain-8": lambda: _prepare_chain(8, permuted=False),
    "chain-8-permuted": lambda: _prepare_chain(8, permuted=True),
    "chain-64": lambda: _prepare_chain(64, permuted=False),
    "chain-64-permuted": lambda: _prepare_chain(64, permuted=True),
    "tfc": _prepare_tfc,
    "conformance": lambda: _prepare_pytest(str(ROOT / "tests" / "kernels" / "test_conformance.py")),
    "sweeps": _prepare_sweeps,
    "cnv": lambda: _prepare_pytest(str(ROOT / "tests" / "kernel_ops" / "test_conv.py")),
    "geometries": _prepare_geometries,
    "fifo-sizing": lambda: _prepare_pytest(
        str(ROOT / "tests" / "kernels" / "test_fifo_sizing.py"),
        str(ROOT / "tests" / "kernels" / "test_explore.py"),
    ),
}
"""Each source by name: preparing it imports what it runs (before the recorder wraps the
line's functions, so every module's name for them is wrapped) and returns the run."""
