"""How FINN runs AMD's tools: a selected installation, its prepared environment, and
the one way a tool process is started and reaped.

- ``Selection`` names an installation and holds no environment, so a worker can
  receive it: local settings scripts to source, or an already configured
  environment accepted as it is; a site command directory; a launcher prefix
  (a site route that owns activation remotely).
- ``Selection.prepare()`` captures the environment once, as a ``Toolchain``. Each
  tool launch then runs in that snapshot (``Toolchain.run``), never in the
  parent's environment, and ``Toolchain.probe`` asks a tool its version through
  exactly that route.
- ``run_process`` is the process primitive under both: a new process group,
  killed whole on timeout or cancellation.

A transformation that runs a tool takes a prepared ``Toolchain`` (``toolchain=``),
so that one build runs every tool through one route; without one, it runs by the
machine's (``machine_toolchain``). Nothing is discovered or activated on import.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from finn.util import machine_file

StrPath = str | os.PathLike[str]

_LOG = logging.getLogger(__name__)

# Vendor tools start slowly (a JVM behind vitis-run), and much more slowly when a
# build launches many at once. Probe answers are cached per process, keyed by the
# selection and the environment's PATH, so parallel HLS synthesis probes once.
PROBE_TIMEOUT = 120
_VERSIONS: dict[tuple[Hashable, ...], tuple[int, ...]] = {}
_HLS_CAPABLE: set[tuple[Hashable, ...]] = set()


class Cancel(Protocol):
    """A cancellation flag (``threading.Event``, ``multiprocessing.Event``)."""

    def is_set(self) -> bool:
        ...


def _child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    # Bash reads BASH_ENV even with --noprofile --norc. Exported shell functions
    # and option variables can also change sourcing before our command runs.
    return {
        str(k): str(v)
        for k, v in environment.items()
        if k not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"} and not k.startswith("BASH_FUNC_")
    }


def run_process(
    argv: Sequence[StrPath],
    *,
    env: Mapping[str, str] | None = None,
    cwd: StrPath | None = None,
    timeout: float | None = None,
    cancel: Cancel | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run argv in a new local process group; reap it on timeout or cancellation.

    ``env`` is the child's whole environment (the parent's when None), less Bash's
    startup hooks; it is copied, and neither cwd nor os.environ is changed. Output
    is captured as bytes. On a timeout, a cancellation (``InterruptedError``) or an
    interrupt, the group is killed and the exception carries what the process
    printed (``output``, ``stderr``); with ``check``, a non-zero exit raises
    ``CalledProcessError``. A remote wrapper is responsible for cancelling its
    remote descendants.
    """
    command = [os.fspath(arg) for arg in argv]
    if not command:
        raise ValueError("Empty command")
    environment = _child_environment(os.environ if env is None else env)
    started = time.monotonic()
    _LOG.info("command=%s cwd=%s", shlex.join(command), cwd or os.getcwd())
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Command cancelled: " + shlex.join(command))
            # Wake every 0.1 s to look at the deadline and the cancellation flag.
            wait: float | None = 0.1 if cancel is not None else None
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                wait = min(0.1, remaining)
            try:
                out, err = proc.communicate(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException as exc:
        # Kill the whole group, even if its leader has already exited while a
        # descendant still owns a pipe. SIGKILL gives bounded cleanup.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        if isinstance(exc, (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt)):
            # Only TimeoutExpired declares them; the others carry them for the same reader.
            setattr(exc, "output", out)
            setattr(exc, "stderr", err)
        _LOG.warning(
            "command stopped duration=%.3fs result=%s",
            time.monotonic() - started,
            type(exc).__name__,
        )
        raise
    result = subprocess.CompletedProcess(command, proc.returncode, out, err)
    _LOG.info(
        "command finished duration=%.3fs result=%s", time.monotonic() - started, proc.returncode
    )
    if check:
        result.check_returncode()
    return result


@dataclass(frozen=True)
class Selection:
    """Serializable selection, safe to pass to workers; contains no environment.

    Empty settings explicitly accepts an already-configured environment. A site
    route owns remote activation and remote paths; it must not use local settings.
    Frontend is requested deliberately, never chosen from executable discovery.
    """

    settings: tuple[str, ...] = ()
    command_dir: str = ""
    launcher: tuple[str, ...] = ()
    hls_frontend: str = "vitis_hls"

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", tuple(map(os.fspath, self.settings)))
        object.__setattr__(self, "launcher", tuple(map(os.fspath, self.launcher)))
        object.__setattr__(self, "command_dir", os.fspath(self.command_dir))
        if self.launcher and self.settings:
            raise ValueError(
                "Site routes own activation; do not combine launcher and local settings"
            )
        if self.hls_frontend not in {"vivado_hls", "vitis_hls", "vitis-run"}:
            raise ValueError("Unknown HLS frontend: " + self.hls_frontend)

    def prepare(self, base_env: Mapping[str, str] | None = None, timeout: float = 30) -> Toolchain:
        """The selected installation's environment, captured once.

        With settings scripts, they are sourced in Bash over ``base_env`` (by default
        a clean base: the system PATH and the user, locale, display, licence and
        ``LD_PRELOAD`` variables) and the resulting environment is kept; without, ``base_env``
        (by default os.environ) is taken as configured. A local route that names no
        licence gets the machine file's. Raises ``RuntimeError`` when a script fails.
        """
        if base_env is None:
            if self.settings:
                # A defined clean base, not an attempt to unsource another toolchain.
                # LD_PRELOAD belongs to the host, not to a toolchain: the image
                # preloads its libudev.so.1, without which Vivado's licence library
                # crashes in udev_enumerate_scan_devices (Vivado exits 139).
                keep = {
                    "HOME",
                    "USER",
                    "LOGNAME",
                    "LANG",
                    "LC_ALL",
                    "TMPDIR",
                    "DISPLAY",
                    "XILINXD_LICENSE_FILE",
                    "LM_LICENSE_FILE",
                    "LD_PRELOAD",
                }
                base_env = {k: v for k, v in os.environ.items() if k in keep}
                base_env["PATH"] = os.defpath
            else:
                base_env = os.environ
        environment = _child_environment(base_env)
        if self.settings:
            scripts = [str(Path(path).resolve(strict=True)) for path in self.settings]
            bash = shutil.which("bash", path=os.defpath)
            if bash is None:
                raise RuntimeError("Could not capture vendor settings: no bash on " + os.defpath)
            command = [
                bash,
                "--noprofile",
                "--norc",
                "-c",
                'for script in "$@"; do source "$script" >&2 || exit; done; /usr/bin/env -0',
                "finn-settings",
                *scripts,
            ]
            try:
                result = run_process(command, env=environment, timeout=timeout)
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError(
                    "Could not capture vendor settings: " + ", ".join(scripts)
                ) from exc
            environment = dict(
                item.decode(errors="surrogateescape").split("=", 1)
                for item in result.stdout.split(b"\0")
                if item
            )
        if not self.launcher:
            # The machine file's licence server, so that a process which never
            # sourced activate.sh still launches licensed tools. A licence
            # variable already set wins; a site route owns its own.
            licence = machine_file.license_for(environment, machine_file.settings())
            if licence:
                environment["XILINXD_LICENSE_FILE"] = licence
        return Toolchain(self, environment)


@dataclass(frozen=True)
class Toolchain:
    """A selection with its prepared environment, a read-only snapshot: every tool
    this toolchain runs starts in it. Made by ``Selection.prepare``."""

    selection: Selection
    environment: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "environment", MappingProxyType(_child_environment(self.environment))
        )

    def __reduce__(self) -> tuple[type[Toolchain], tuple[Selection, dict[str, str]]]:
        # A read-only mapping does not pickle; a toolchain crosses into worker
        # processes with the transformation that holds it (NodeLocalTransformation).
        return (Toolchain, (self.selection, dict(self.environment)))

    def command(self, tool: str, *args: StrPath) -> list[str]:
        """The argv that runs ``tool`` (a name, not a path) with ``args`` on this
        route: under the command directory when one is selected, behind the
        launcher when there is one. A local tool not on the environment's PATH is
        ``FileNotFoundError``."""
        if Path(tool).name != tool:
            raise ValueError("Pass a tool name, not a local executable path")
        selection = self.selection
        executable = str(Path(selection.command_dir) / tool) if selection.command_dir else tool
        # A launcher receives the remote name (or site's override), never a
        # locally resolved /opt/... path. It owns remote existence checking.
        if (
            not selection.launcher
            and shutil.which(executable, path=self.environment.get("PATH", "")) is None
        ):
            raise FileNotFoundError(
                f"{executable} not found (command_dir={selection.command_dir!r})"
            )
        return [*selection.launcher, executable, *(os.fspath(arg) for arg in args)]

    def hls_installation(self) -> Path:
        """The local installation whose HLS C++ headers (``include/``) and C
        simulation libraries (``lnx64/``) code compiled by ``g++`` uses: the
        environment's ``XILINX_HLS``, else ``XILINX_VITIS`` (whose installation
        carries HLS from 2025.1). ``LookupError`` when it names neither."""
        for variable in ("XILINX_HLS", "XILINX_VITIS"):
            if self.environment.get(variable):
                return Path(self.environment[variable])
        raise LookupError(
            "This toolchain names no HLS installation (XILINX_HLS or XILINX_VITIS) "
            "for HLS C++ headers and C simulation libraries"
        )

    def run(
        self,
        tool: str,
        args: Iterable[StrPath] = (),
        *,
        cwd: StrPath | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        cancel: Cancel | None = None,
        check: bool = True,
        replay: StrPath | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run ``tool`` in the prepared environment, updated by ``env``
        (``run_process``'s contract otherwise). ``replay`` names a script written
        before the run that repeats the command by hand, beside which the run's
        stdout and stderr are kept (``<replay>.stdout.log``, ``.stderr.log``)."""
        environment = {**self.environment, **(env or {})}
        command = self.command(tool, *args)
        if replay is not None:
            # Replay in the same prepared environment; never persist its secrets.
            Path(replay).write_text(
                "#!/bin/bash\nset -e\n"
                + ("cd -- " + shlex.quote(os.fspath(cwd)) + "\n" if cwd is not None else "")
                + "exec "
                + shlex.join(command)
                + "\n"
            )
        _LOG.info(
            "route=%s settings=%s tool=%s",
            self.selection.launcher or "local",
            self.selection.settings or "configured environment",
            tool,
        )
        try:
            result = run_process(
                command, env=environment, cwd=cwd, timeout=timeout, cancel=cancel, check=check
            )
        except (subprocess.SubprocessError, InterruptedError, KeyboardInterrupt) as exc:
            if replay is not None:
                Path(str(replay) + ".stdout.log").write_bytes(getattr(exc, "output", None) or b"")
                Path(str(replay) + ".stderr.log").write_bytes(getattr(exc, "stderr", None) or b"")
            raise
        if replay is not None:
            Path(str(replay) + ".stdout.log").write_bytes(result.stdout)
            Path(str(replay) + ".stderr.log").write_bytes(result.stderr)
        return result

    def _probe_key(self, *what: Hashable) -> tuple[Hashable, ...]:
        return (*what, self.selection, self.environment.get("PATH", ""))

    def probe(self, tool: str, timeout: float = PROBE_TIMEOUT) -> tuple[int, ...]:
        """The tool's AMD release, (year, minor), asked through exactly the execution
        route; once per selection and PATH in a process. Not a licence test."""
        key = self._probe_key("version", tool)
        if key in _VERSIONS:
            return _VERSIONS[key]
        try:
            result = self.run(
                tool, ["--version" if tool == "vitis-run" else "-version"], timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Could not probe {tool} through {self.selection.launcher or 'local'}; "
                "check selection, site wrapper and probe timeout"
            ) from exc
        banner = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        match = re.search(r"(?<!\d)(20\d{2})\.(\d+)\b", banner)
        if not match:
            raise RuntimeError(f"{tool} did not report a recognizable AMD version")
        version = tuple(map(int, match.groups()))
        _LOG.info("identity tool=%s version=%s.%s", tool, *version)
        _VERSIONS[key] = version
        return version

    def hls_command(self, script: str, timeout: float = PROBE_TIMEOUT) -> tuple[str, list[str]]:
        """The selected HLS frontend and its arguments to run ``script``; refused
        (``ValueError``) when the probed release does not match the frontend."""
        frontend = self.selection.hls_frontend
        version = self.probe(frontend, timeout)
        # These are FINN code-generation constraints, separate from whether a
        # command exists. Preserve the standalone/unified transition deliberately.
        supported = {
            "vivado_hls": version <= (2020, 1),
            "vitis_hls": (2020, 1) <= version <= (2024, 2),
            "vitis-run": version >= (2025, 1),
        }[frontend]
        if not supported:
            raise ValueError(
                f"FINN HLS frontend {frontend} is incompatible with {version}; "
                "select the matching frontend explicitly"
            )
        capability = self._probe_key("hls-capability", frontend)
        if frontend == "vitis-run" and capability not in _HLS_CAPABLE:
            help_result = self.run(frontend, ["--help"], timeout=timeout)
            help_text = (help_result.stdout + help_result.stderr).decode("utf-8", errors="replace")
            if "hls" not in help_text.lower():
                raise RuntimeError("Selected vitis-run does not advertise HLS capability")
            _LOG.info("capability tool=vitis-run hls=advertised")
            _HLS_CAPABLE.add(capability)
        args = ["--mode", "hls", "--tcl", script] if frontend == "vitis-run" else ["-f", script]
        return frontend, args


#: The machine setting that names a site command directory: wrappers for
#: ``vivado``, ``xelab``, ``v++``, ``g++``, ... that may hand a tool to a compute
#: farm (ci/README.md, "Running tools on LSF"). The name is the Jenkins site
#: configuration's.
COMMAND_DIR_SETTING = "FINN_TOOL_DIR_OVERRIDE"


def machine_selection(environ: Mapping[str, str] | None = None) -> Selection:
    """The selection a tool runs by when its caller names none: this machine's
    environment as configured (``scripts/activate.sh``, ``docker/run``,
    ``finn-toolchain.sh``; no settings script is sourced), under the site command
    directory ``FINN_TOOL_DIR_OVERRIDE`` names, if it names one. A selection a
    caller states (a build configuration's ``toolchain``) is used as stated
    instead: its ``command_dir`` wins."""
    environ = os.environ if environ is None else environ
    return Selection(command_dir=environ.get(COMMAND_DIR_SETTING, ""))


def machine_toolchain() -> Toolchain:
    """``machine_selection()`` prepared over this process's environment: the
    toolchain of a transformation called without one."""
    return machine_selection().prepare()
