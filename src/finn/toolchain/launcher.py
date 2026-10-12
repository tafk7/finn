# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How FINN runs AMD's tools: a ``Selection`` names an installation,
``Selection.prepare()`` captures its environment once as a ``Toolchain``, and every
tool runs in that snapshot through ``run_process``.

- ``Selection`` holds no environment, so it can be passed to worker processes. It
  names settings scripts to source (or none, to accept the environment as
  configured), a site command directory, or a launcher prefix (a site route that
  activates the tools remotely).
- ``Toolchain.run`` launches a tool in the captured environment, never in this
  process's, and ``Toolchain.probe`` asks a tool its version through the same route.
- ``run_process`` starts each process in a new process group and kills the whole
  group on timeout or cancellation.

A tool run's transcript is two files the tool writes as it runs, ``<stem>.stdout.log``
and ``<stem>.stderr.log`` (``transcript_files``). The stem is the one the caller names
(``transcript=``); else, in a build, a fresh one in the step's log directory
``output_dir/logs/<step>`` (``Toolchain.transcripts_in``); else the replay script's. A
run with none of them (a version probe) is captured in memory.

This module holds only the route. What one tool needs is in its own module:
``finn.toolchain.hls``, ``xsim`` and ``vivado``.

A function that runs a tool requires a prepared ``Toolchain`` (``toolchain=``), so that
a build runs every tool through one route and nothing runs on another installation
unnoticed. Only the entry points a user calls default to the machine's
(``host_toolchain``), prepared once and passed down: ``PackagePartition``,
``ElaboratePartition`` and the XSim executor when they are made, the pynq shell's
``BuildPynqBitfile``, and the command lines (``python -m finn.platform.generate``,
``python -m finn.xsi.setup``). Nothing is discovered or activated on import, and a
tool's installation is read only through a toolchain, never from this process's
environment.

Running a tool has two errors of its own: ``ToolUnavailable`` (a ``LookupError``) when
the toolchain names no installation of it or the tool is not on its route, and
``ToolFailed`` (a ``RuntimeError``) when it ran and failed. ``identity`` names an
installation by its content: the key for what a tool built.
"""

from __future__ import annotations

import hashlib
import itertools
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

# Modules, not names: finn.toolchain.vivado and .hls import this module in turn.
from finn.toolchain import hls, host, vivado

StrPath = str | os.PathLike[str]

_LOG = logging.getLogger(__name__)

# Vendor tools start slowly (a JVM behind vitis-run), and much more slowly when a
# build launches many at once. Probe answers are cached per process, keyed by the
# selection and the environment's PATH, so parallel HLS synthesis probes once.
PROBE_TIMEOUT = 120
_VERSIONS: dict[tuple[Hashable, ...], tuple[int, ...]] = {}

#: The ``extra`` key of a tool run's log records (``run_process``): a mapping with the
#: run's number in this process (``run``) and, when it starts, ``tool``, ``command``,
#: ``cwd``, ``transcript`` (its standard output's file) and ``status`` ``running``;
#: when it ends, ``status`` (``completed``, ``failed``, or ``stopped`` with the
#: ``error``), ``seconds`` and ``exit``. A build's manifest indexes them
#: (finn.builder.build_dataflow_manifest); the log shows the same records.
TOOL_RUN = "finn_tool_run"
_RUNS = itertools.count(1)


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


def transcript_files(stem: StrPath) -> tuple[Path, Path]:
    """The two files of the transcript ``stem``: what the tool printed on its standard
    output (``<stem>.stdout.log``) and on its standard error (``<stem>.stderr.log``)."""
    return Path(f"{os.fspath(stem)}.stdout.log"), Path(f"{os.fspath(stem)}.stderr.log")


class ToolRun(subprocess.CompletedProcess[bytes]):
    """A finished run (``subprocess.CompletedProcess``): ``stdout`` and ``stderr`` are
    what it printed, and ``transcript`` is the file its standard output went to as it
    ran (``transcript_files``; its standard error is beside it), or None if the output
    was captured in memory."""

    def __init__(
        self,
        args: Sequence[str],
        returncode: int,
        stdout: bytes,
        stderr: bytes,
        transcript: Path | None = None,
    ) -> None:
        super().__init__(args, returncode, stdout, stderr)
        self.transcript = transcript


def run_process(
    argv: Sequence[StrPath],
    *,
    env: Mapping[str, str] | None = None,
    cwd: StrPath | None = None,
    timeout: float | None = None,
    cancel: Cancel | None = None,
    check: bool = True,
    transcript: StrPath | None = None,
    tool: str | None = None,
) -> ToolRun:
    """Run argv in a new local process group; reap it on timeout or cancellation.

    ``env`` is the child's whole environment (the parent's when None), less Bash's
    startup hooks; it is copied, and neither cwd nor os.environ is changed. With a
    ``transcript`` stem, the process writes its standard output and error straight into
    the transcript's two files as it runs (``transcript_files``; their directory is
    made), and what it printed is read back from them when it ends; without, output is
    captured in memory. Either way the result holds it as bytes. On a timeout, a
    cancellation (``InterruptedError``) or an interrupt, the group is killed and the
    exception carries what the process printed (``output``, ``stderr``); with
    ``check``, a non-zero exit raises ``CalledProcessError``. A remote wrapper is
    responsible for cancelling its remote descendants.

    A run that keeps a transcript is a tool run a build indexes: its log records, when
    it starts and when it finishes or is stopped, carry ``TOOL_RUN``'s fields, with the
    run named ``tool`` (by default, its command's first word).
    """
    command = [os.fspath(arg) for arg in argv]
    if not command:
        raise ValueError("Empty command")
    environment = _child_environment(os.environ if env is None else env)
    files = None if transcript is None else transcript_files(transcript)
    started = time.monotonic()
    run = next(_RUNS)

    def indexed(**state: object) -> dict[str, dict[str, object]] | None:
        # The structured fields of this run's records, for a build's manifest; None
        # for a run with no transcript (a probe, a settings capture).
        if files is None:
            return None
        return {TOOL_RUN: {"run": run, **state}}

    _LOG.info(
        "command=%s cwd=%s transcript=%s",
        shlex.join(command),
        cwd or os.getcwd(),
        "none" if files is None else files[0],
        extra=indexed(
            tool=tool or Path(command[0]).name,
            command=shlex.join(command),
            cwd=os.fspath(cwd or os.getcwd()),
            transcript=None if files is None else os.fspath(files[0]),
            status="running",
        ),
    )
    if files is None:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        files[0].parent.mkdir(parents=True, exist_ok=True)
        # The child writes the files itself; this process keeps no handle on them.
        with open(files[0], "wb") as stdout, open(files[1], "wb") as stderr:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
            )

    def printed() -> tuple[bytes, bytes]:
        # What a run streaming into its transcript printed, read back once it ended.
        assert files is not None
        return files[0].read_bytes(), files[1].read_bytes()

    out = err = b""
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
                if files is None:
                    out, err = proc.communicate(timeout=wait)
                else:
                    proc.wait(timeout=wait)
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
        if files is None:
            out, err = proc.communicate()
        else:
            proc.wait()
            out, err = printed()
        if isinstance(exc, (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt)):
            # Only TimeoutExpired declares them; the others carry them for the same reader.
            setattr(exc, "output", out)
            setattr(exc, "stderr", err)
        seconds = time.monotonic() - started
        _LOG.warning(
            "command stopped duration=%.3fs result=%s",
            seconds,
            type(exc).__name__,
            extra=indexed(status="stopped", seconds=round(seconds, 3), error=type(exc).__name__),
        )
        raise
    if files is not None:
        out, err = printed()
    result = ToolRun(command, proc.returncode, out, err, None if files is None else files[0])
    seconds = time.monotonic() - started
    _LOG.info(
        "command finished duration=%.3fs result=%s",
        seconds,
        proc.returncode,
        extra=indexed(
            status="completed" if proc.returncode == 0 else "failed",
            seconds=round(seconds, 3),
            exit=proc.returncode,
        ),
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

    A field that is ``None`` is not set: a build's selection is laid over the machine's
    (``over``, ``host_selection``), so an unset field takes the machine's value, never a
    default. The machine sets the command directory, the HLS frontend and Vivado's jobs,
    never settings or a launcher.
    """

    settings: tuple[str, ...] = ()
    command_dir: str | None = None
    launcher: tuple[str, ...] = ()
    hls_frontend: str | None = None
    #: How many runs Vivado launches at once (``launch_runs -jobs``). None means the
    #: machine's default (``finn.toolchain.vivado.vivado_jobs``).
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
        """This selection laid over ``base`` (the machine's): each field set here (not
        ``None``) wins, and ``base`` keeps the rest."""
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
#: An error line of a tool that does not print Vivado's messages (a compiler, a script).
_ERROR_LINE = re.compile(r"\berror:|^Error\b")


class ToolFailed(RuntimeError):
    """A tool that ran and failed.

    - ``tool``: its name;
    - ``status``: its exit status, or None if it exited cleanly without producing what
      it was run for, or did not exit;
    - ``transcript``: the log that says why, if there is one;
    - ``errors``: the first error lines it printed (by ``finn.toolchain.vivado.messages``:
      its errors, then its critical warnings; else a compiler's ``error:`` lines; else
      its last lines);
    - ``code``: ``licence-unavailable`` if its output says no licence was found, else
      ``tool-error``;
    - ``what``: what failed."""

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
        # Pickled whole, so it reaches the parent of a worker process as raised.
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
        read = vivado.messages(lines)
        errors = list(read.errors[:ERROR_LINES])
        if not errors:
            errors = [line for line in lines if _ERROR_LINE.search(line)][:ERROR_LINES]
        if not errors:
            errors = lines[-ERROR_LINES:]
        return cls(
            tool, status=status, transcript=transcript, errors=errors, code=read.code, what=what
        )

    @classmethod
    def of_run(
        cls,
        tool: str,
        result: subprocess.CompletedProcess[bytes],
        *,
        transcript: StrPath | None = None,
        what: str = "",
    ) -> ToolFailed:
        """The failure of a run of ``tool`` that exited non-zero (``ToolFailed.of``),
        naming the run's own transcript (``ToolRun.transcript``) if it kept one, else
        ``transcript``, the log the tool writes itself (such as ``xvlog.log``)."""
        return cls.of(
            tool,
            (result.stdout or b"") + (result.stderr or b""),
            status=result.returncode,
            transcript=getattr(result, "transcript", None) or transcript,
            what=what,
        )


@dataclass(frozen=True)
class Toolchain:
    """A selection with its prepared environment, a read-only snapshot in which every
    tool this toolchain runs starts. Made by ``Selection.prepare``.

    ``transcripts`` is the directory a build step writes its runs' transcripts in
    (``output_dir/logs/<step>``), or None outside a build. The step takes its toolchain
    from the build's configuration, which sets it (``transcripts_in``; the builder's
    ``DataflowBuildConfig.step_toolchain``), and passes it down to every transformation
    that runs a tool."""

    selection: Selection
    environment: Mapping[str, str] = field(repr=False)
    transcripts: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "environment", MappingProxyType(_child_environment(self.environment))
        )
        if self.transcripts is not None:
            object.__setattr__(self, "transcripts", Path(self.transcripts))

    def __reduce__(
        self,
    ) -> tuple[type[Toolchain], tuple[Selection, dict[str, str], Path | None]]:
        # A read-only mapping does not pickle; a toolchain crosses into worker
        # processes with the transformation that holds it (NodeLocalTransformation).
        return (Toolchain, (self.selection, dict(self.environment), self.transcripts))

    def transcripts_in(self, directory: StrPath | None) -> Toolchain:
        """This toolchain with its runs' transcripts written in ``directory`` (a build
        step's ``output_dir/logs/<step>``, made when a run first writes there). None
        leaves them where each run's caller says (``run``)."""
        return replace(self, transcripts=None if directory is None else Path(directory))

    def transcript(self, name: str) -> Path | None:
        """A fresh transcript stem for a run called ``name`` in ``transcripts``:
        ``<transcripts>/<name>``, else ``<name>.2``, ``.3``, ... when an earlier run of
        the step took it (claimed by creating its standard output file, so parallel
        runs never share one). None without a directory."""
        if self.transcripts is None:
            return None
        self.transcripts.mkdir(parents=True, exist_ok=True)
        for number in itertools.count(1):
            stem = self.transcripts / (name if number == 1 else f"{name}.{number}")
            try:
                with open(transcript_files(stem)[0], "xb"):
                    return stem
            except FileExistsError:
                continue
        raise AssertionError("unreachable")

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
    ) -> ToolRun:
        """Run ``tool`` in the prepared environment, updated by ``env``
        (``run_process``'s contract otherwise; with ``check``, a non-zero exit is
        ``ToolFailed``, naming the transcript). ``replay`` names a script written before
        the run that repeats the command by hand. A tool not on the route is
        ``ToolUnavailable``.

        The tool writes what it prints into its transcript as it runs
        (``transcript_files``): the stem ``transcript`` when the caller names one; else,
        when this toolchain files a build step's transcripts, a fresh stem there named
        after the replay script, or the tool (``transcript``); else the replay's
        (``<replay>.stdout.log``, ``.stderr.log``). With none, what it prints is
        captured in memory. The result names the transcript (``ToolRun.transcript``)."""
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
        if transcript is None:
            transcript = self.transcript(tool if replay is None else Path(replay).name) or replay
        _LOG.info(
            "route=%s settings=%s tool=%s",
            self.selection.launcher or "local",
            self.selection.settings or "configured environment",
            tool,
        )
        result = run_process(
            command,
            env=environment,
            cwd=cwd,
            timeout=timeout,
            cancel=cancel,
            check=False,
            transcript=transcript,
            tool=tool,
        )
        if check and result.returncode != 0:
            raise ToolFailed.of_run(tool, result)
        return result

    def probe_key(self, *what: Hashable) -> tuple[Hashable, ...]:
        """The key of what a probe learns of this route (``what``), cached per process:
        by the selection and the environment's PATH."""
        return (*what, self.selection, self.environment.get("PATH", ""))

    def probed(self, tool: str) -> tuple[int, ...] | None:
        """The release ``probe`` found for ``tool`` on this route in this process, or
        None if it has not asked; never runs the tool."""
        return _VERSIONS.get(self.probe_key("version", tool))

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
            # A probe asks a version, and keeps no transcript.
            result = self.transcripts_in(None).run(
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
    """The machine's selection: its environment as configured (``scripts/activate.sh``,
    ``docker/run``, ``finn-toolchain.sh``; no settings script is sourced), with the site
    command directory ``FINN_TOOL_DIR_OVERRIDE`` names, the HLS frontend for the host
    file's ``FINN_XILINX_VERSION`` (``finn.toolchain.hls.host_frontend``) and the Vivado
    jobs ``FINN_VIVADO_JOBS`` gives, each read on every call. A selection the caller
    gives (a build configuration's ``toolchain``) is laid over it (``Selection.over``)."""
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
    """``host_selection()`` prepared over this process's environment: the default
    toolchain of the entry points a user calls (see the module docstring)."""
    return host_selection().prepare()
