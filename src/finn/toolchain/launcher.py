# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

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

Only the route lives here: what one tool needs (HLS's frontend and installation,
XSim's commands, Vivado's installation) is its own module's, ``finn.toolchain.hls``,
``xsim`` and ``vivado``.

A transformation that runs a tool takes a prepared ``Toolchain`` (``toolchain=``),
so that one build runs every tool through one route; without one, it runs by the
machine's (``host_toolchain``). Nothing is discovered or activated on import.

A tool's outcome has two errors of its own, whoever runs it: ``ToolUnavailable``
(a ``LookupError``) when the toolchain names no installation of it or has no such
tool on its route, and ``ToolFailed`` (a ``RuntimeError``) when it ran and failed.
``identity`` is an installation by what it is, the one key of what a tool built.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

# Modules, not names: finn.toolchain.vivado and .hls read this module in turn.
from finn.toolchain import hls, host, vivado

StrPath = str | os.PathLike[str]

_LOG = logging.getLogger(__name__)

# Vendor tools start slowly (a JVM behind vitis-run), and much more slowly when a
# build launches many at once. Probe answers are cached per process, keyed by the
# selection and the environment's PATH, so parallel HLS synthesis probes once.
PROBE_TIMEOUT = 120
_VERSIONS: dict[tuple[Hashable, ...], tuple[int, ...]] = {}


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

    A field that is ``None`` is not stated: the selection a build runs by is the one it
    states laid over the machine's (``over``, ``host_selection``), so a field it
    does not state is the machine's, never a default. The machine states the command
    directory, the HLS frontend and Vivado's jobs, and never settings or a launcher.
    """

    settings: tuple[str, ...] = ()
    command_dir: str | None = None
    launcher: tuple[str, ...] = ()
    hls_frontend: str | None = None
    #: How many runs Vivado launches at once (``launch_runs -jobs``); None: the
    #: machine's cores, at most 16 (``finn.toolchain.vivado.vivado_jobs``).
    vivado_jobs: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", tuple(map(os.fspath, self.settings)))
        object.__setattr__(self, "launcher", tuple(map(os.fspath, self.launcher)))
        if self.command_dir is not None:
            object.__setattr__(self, "command_dir", os.fspath(self.command_dir))
        if self.launcher and self.settings:
            raise ValueError(
                "Site routes own activation; do not combine launcher and local settings"
            )
        if self.hls_frontend is not None and self.hls_frontend not in hls.FRONTENDS:
            raise ValueError("Unknown HLS frontend: " + self.hls_frontend)
        if self.vivado_jobs is not None:
            vivado.vivado_jobs(self.vivado_jobs)

    def over(self, base: Selection) -> Selection:
        """This selection laid over ``base`` (the machine's): each field it states
        (not ``None``) replaces ``base``'s, and ``base`` keeps the rest, so stating one
        field drops none of the others, not the command directory, the HLS frontend
        or Vivado's jobs."""
        stated = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if getattr(self, item.name) is not None
        }
        return replace(base, **stated)

    def prepare(self, base_env: Mapping[str, str] | None = None, timeout: float = 30) -> Toolchain:
        """The selected installation's environment, captured once.

        With settings scripts, they are sourced in Bash over ``base_env`` (by default
        a clean base: the system PATH and the user, locale, display, licence and
        ``LD_PRELOAD`` variables) and the resulting environment is kept; without, ``base_env``
        (by default os.environ) is taken as configured. A local route that names no
        licence gets the host file's. Raises ``RuntimeError`` when a script fails.
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
            # The host file's licence server, so that a process which never
            # sourced activate.sh still launches licensed tools. A licence
            # variable already set wins; a site route owns its own.
            licence = host.license_for(environment, host.settings())
            if licence:
                environment["XILINXD_LICENSE_FILE"] = licence
        return Toolchain(self, environment)


#: What selects a tool, as a ``ToolUnavailable`` names it.
SELECTED_BY = (
    "the host file selects the machine's installation (FINN_XILINX_PATH and "
    "FINN_XILINX_VERSION in ~/.config/finn/xilinx.env, or the file FINN_XILINX_ENV names; "
    "scripts/activate.sh, docker/run), and a build configuration's toolchain selects a build's"
)


class ToolUnavailable(LookupError):
    """No installation of a tool, or no such tool on the toolchain's route: ``tool``
    names it, ``reason`` what is missing; the message also says what selects it
    (``SELECTED_BY``)."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}; {SELECTED_BY}")
        self.tool = tool
        self.reason = reason

    def __reduce__(self) -> tuple[type[ToolUnavailable], tuple[str, str]]:
        return (ToolUnavailable, (self.tool, self.reason))


#: How many of a failed tool's error lines a ``ToolFailed`` quotes.
ERROR_LINES = 5
_ERROR_LINE = re.compile(r"^\s*(ERROR|FATAL|CRITICAL WARNING)\b|\berror:|^Error\b")
#: What a transcript says when the tool found no licence (Vivado's words, FlexLM's).
_NO_LICENCE = re.compile(
    r"valid licen[cs]e was not found|licen[cs]e checkout failed|"
    r"no licen[cs]e (is )?available|FLEX(net|lm) licensing error",
    re.IGNORECASE,
)


class ToolFailed(RuntimeError):
    """A tool that ran and failed: ``tool``, its exit ``status`` (None when it exited
    cleanly but left nothing of what it was run for, or did not exit), ``transcript``
    (the log that tells why, when there is one), ``errors`` (the first error lines it
    printed, else its last lines) and a ``code``: ``licence-unavailable`` when what it
    printed says no licence was found, else ``tool-error``. ``what`` says what failed."""

    def __init__(
        self,
        tool: str,
        status: int | None = None,
        transcript: StrPath | None = None,
        errors: Sequence[str] = (),
        code: str = "tool-error",
        what: str = "",
    ) -> None:
        self.tool = tool
        self.status = status
        self.transcript = None if transcript is None else Path(transcript)
        self.errors = tuple(errors)
        self.code = code
        self.what = what
        message = f"{tool} failed" + ("" if status is None else f" with exit status {status}")
        message += f": {what}" if what else ""
        message += f" ({code})" if code != "tool-error" else ""
        message += f"; transcript {self.transcript}" if self.transcript is not None else ""
        super().__init__("\n  ".join((message, *self.errors)))

    def __reduce__(self) -> tuple[type[ToolFailed], tuple[object, ...]]:
        # Crosses into the parent of a worker process as it was raised.
        return (
            ToolFailed,
            (self.tool, self.status, self.transcript, self.errors, self.code, self.what),
        )

    @classmethod
    def of(
        cls,
        tool: str,
        printed: str | bytes = "",
        *,
        status: int | None = None,
        transcript: StrPath | None = None,
        what: str = "",
    ) -> ToolFailed:
        """The failure of ``tool``, read from what it ``printed`` and from its
        ``transcript`` when that file exists: its first error lines and its code."""
        if isinstance(printed, bytes):
            printed = printed.decode(errors="replace")
        texts = [printed]
        if transcript is not None and Path(transcript).is_file():
            texts.append(Path(transcript).read_text(errors="replace"))
        # A tool's log often repeats what it printed: each line once, in order.
        lines = list(dict.fromkeys(line.rstrip() for text in texts for line in text.splitlines()))
        lines = [line for line in lines if line.strip()]
        errors = [line for line in lines if _ERROR_LINE.search(line)][:ERROR_LINES]
        if not errors:
            errors = lines[-ERROR_LINES:]
        code = "licence-unavailable" if any(map(_NO_LICENCE.search, lines)) else "tool-error"
        return cls(tool, status=status, transcript=transcript, errors=errors, code=code, what=what)

    @classmethod
    def of_run(
        cls,
        tool: str,
        result: subprocess.CompletedProcess[bytes],
        *,
        transcript: StrPath | None = None,
        what: str = "",
    ) -> ToolFailed:
        """The failure of a run of ``tool`` that exited non-zero (``ToolFailed.of``)."""
        return cls.of(
            tool,
            (result.stdout or b"") + (result.stderr or b""),
            status=result.returncode,
            transcript=transcript,
            what=what,
        )


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
        ``ToolUnavailable``."""
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
            where = (
                f"in the command directory {selection.command_dir}"
                if selection.command_dir
                else "on the PATH of its environment"
            )
            raise ToolUnavailable(tool, f"not on this toolchain's route ({where})")
        return [*selection.launcher, executable, *(os.fspath(arg) for arg in args)]

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
        transcript: StrPath | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run ``tool`` in the prepared environment, updated by ``env``
        (``run_process``'s contract otherwise; with ``check``, a non-zero exit is
        ``ToolFailed``). ``replay`` names a script written before the run that repeats
        the command by hand, beside which the run's stdout and stderr are kept
        (``<replay>.stdout.log``, ``.stderr.log``). ``transcript`` is the log the tool
        writes itself (``vivado -log``), which a ``ToolFailed`` names; by default the
        replay's stdout. A tool not on the route is ``ToolUnavailable``."""
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
                command, env=environment, cwd=cwd, timeout=timeout, cancel=cancel, check=False
            )
        except (subprocess.SubprocessError, InterruptedError, KeyboardInterrupt) as exc:
            if replay is not None:
                Path(str(replay) + ".stdout.log").write_bytes(getattr(exc, "output", None) or b"")
                Path(str(replay) + ".stderr.log").write_bytes(getattr(exc, "stderr", None) or b"")
            raise
        if replay is not None:
            Path(str(replay) + ".stdout.log").write_bytes(result.stdout)
            Path(str(replay) + ".stderr.log").write_bytes(result.stderr)
        if check and result.returncode != 0:
            if transcript is None and replay is not None:
                transcript = str(replay) + ".stdout.log"
            raise ToolFailed.of_run(tool, result, transcript=transcript)
        return result

    def probe_key(self, *what: Hashable) -> tuple[Hashable, ...]:
        """The key of what a probe learns of this route (``what``), cached per process:
        by the selection and the environment's PATH."""
        return (*what, self.selection, self.environment.get("PATH", ""))

    def probe(self, tool: str, timeout: float = PROBE_TIMEOUT) -> tuple[int, ...]:
        """The tool's AMD release, (year, minor), asked through exactly the execution
        route; once per selection and PATH in a process. Not a licence test.
        ``ToolUnavailable`` when the tool is not on the route, or does not answer
        with a release."""
        key = self.probe_key("version", tool)
        if key in _VERSIONS:
            return _VERSIONS[key]
        route = shlex.join(self.selection.launcher) if self.selection.launcher else "local"
        try:
            result = self.run(
                tool, ["--version" if tool == "vitis-run" else "-version"], timeout=timeout
            )
        except (OSError, subprocess.SubprocessError, ToolFailed) as exc:
            raise ToolUnavailable(
                tool,
                f"could not probe it through the {route} route ({exc}); check the "
                "selection, site wrapper and probe timeout",
            ) from exc
        banner = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        match = re.search(r"(?<!\d)(20\d{2})\.(\d+)\b", banner)
        if not match:
            raise ToolUnavailable(tool, "its version probe reports no recognizable AMD release")
        version = tuple(map(int, match.groups()))
        _LOG.info("identity tool=%s version=%s.%s", tool, *version)
        _VERSIONS[key] = version
        return version


def identity(toolchain: Toolchain, tool: str) -> str:
    """The installation ``tool`` runs from, by what it is, never where it is installed:
    ``tool``, the release its ``data/version.sh`` states (``XILINX_VERSION_DEFAULT``,
    ``unstated`` without one) and its ``data/version.dat``'s digest (``none`` without
    one). An HLS frontend's installation is the HLS installation
    (``finn.toolchain.hls.installation``), any other tool's Vivado's
    (``finn.toolchain.vivado.installation``). The one key of what a tool built: the
    HLS cache's, a compiled XSI design's and the XSI bridge's. Reads two files and
    runs nothing; ``ToolUnavailable`` when the toolchain names no such installation."""
    if tool in hls.FRONTENDS:
        installation = hls.installation(toolchain)
    else:
        installation = vivado.installation(toolchain)
    data = installation / "data"
    release = "unstated"
    if (data / "version.sh").is_file():
        stated = re.search(
            r"^XILINX_VERSION_DEFAULT=(\S+)$", (data / "version.sh").read_text(), re.MULTILINE
        )
        release = stated.group(1) if stated else release
    version = data / "version.dat"
    stamp = hashlib.sha256(version.read_bytes()).hexdigest()[:16] if version.is_file() else "none"
    return f"{tool} {release} {stamp}"


#: The machine setting that names a site command directory: wrappers for
#: ``vivado``, ``xelab``, ``v++``, ``g++``, ... that may hand a tool to a compute
#: farm (docs/environment.md, "Running tools on LSF").
COMMAND_DIR_SETTING = "FINN_TOOL_DIR_OVERRIDE"

#: The machine setting that says how many runs Vivado launches at once
#: (``Selection.vivado_jobs``), a positive number; unset, the machine's cores, capped.
VIVADO_JOBS_SETTING = "FINN_VIVADO_JOBS"


def host_selection(
    environ: Mapping[str, str] | None = None, *, stated: Selection | None = None
) -> Selection:
    """The selection a tool runs by: this machine's environment as configured
    (``scripts/activate.sh``, ``docker/run``, ``finn-toolchain.sh``; no settings
    script is sourced), under the site command directory ``FINN_TOOL_DIR_OVERRIDE``
    names, if it names one, with the HLS frontend of the host file's
    ``FINN_XILINX_VERSION`` (``finn.toolchain.hls.host_frontend``) and the Vivado jobs
    ``FINN_VIVADO_JOBS`` says, if it says. Each is read here, once a call. A selection
    a caller states (a build configuration's ``toolchain``) is laid over it
    (``Selection.over``): each field it states wins, and the machine's keep the rest."""
    environ = os.environ if environ is None else environ
    jobs = environ.get(VIVADO_JOBS_SETTING)
    if jobs is not None and not re.fullmatch(r"[0-9]+", jobs):
        raise ValueError(f"{VIVADO_JOBS_SETTING}={jobs} is not a number of runs")
    frontend = hls.host_frontend(environ)
    try:
        machine = Selection(
            command_dir=environ.get(COMMAND_DIR_SETTING) or None,
            hls_frontend=frontend,
            vivado_jobs=None if jobs is None else int(jobs),
        )
    except ValueError as error:
        raise ValueError(f"{VIVADO_JOBS_SETTING}: {error}") from error
    return machine if stated is None else stated.over(machine)


def host_toolchain() -> Toolchain:
    """``host_selection()`` prepared over this process's environment: the
    toolchain of a transformation called without one."""
    return host_selection().prepare()
