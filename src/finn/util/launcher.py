# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Running external tools: check the exit status, keep a transcript, quote the errors.

``run_tool`` runs a tool's command line itself, not through a shell script, so FINN
sees the exit status of the tool. The tool writes its standard output and standard
error into two transcript files while it runs (``transcript_files``). If the tool
fails, ``ToolFailed`` names the transcript and quotes the first error lines.
``ToolUnavailable`` is a tool that is not installed or not set up.

The tool runs in its own session and process group, so that FINN can kill the tool
with the descendants that stay in its process group (``run_process``).
"""

import contextlib
import errno
import functools
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from finn.util import vivado
from finn.util.build_log import FILE_ONLY, in_run

StrPath = Union[str, os.PathLike]

log = logging.getLogger(__name__)

# How many error lines a ToolFailed quotes.
ERROR_LINES = 5
# An error line of a tool that does not print Vivado's messages (a compiler, a script).
_ERROR_LINE = re.compile(
    r"\berror:|\berror while loading shared libraries:|\bcannot find -l|"
    r"\bundefined reference to\b|^Error\b"
)
# Vivado echoes each line of a sourced Tcl script with this prefix.
_ECHOED_TCL = "# "
# Tcl prints the message of an uncaught error, then this line, then the trace.
_TCL_TRACE = "while executing"
# The dynamic loader's notice that it skipped an LD_PRELOAD library, printed by
# every process of a tool run on a host that lacks the library (e.g. a compute
# farm host behind FINN_TOOL_DIR_OVERRIDE, inheriting the container's LD_PRELOAD).
# The loader carries on, so it is not an error of the tool.
_LOADER_NOTICE = re.compile(r"^ERROR: ld\.so: .* ignored\.$")


def transcript_files(stem: StrPath) -> Tuple[Path, Path]:
    """Return the two files of the transcript ``stem``: ``<stem>.stdout.log`` and
    ``<stem>.stderr.log``."""
    stem = os.fspath(stem)
    return Path(stem + ".stdout.log"), Path(stem + ".stderr.log")


class ToolRun(subprocess.CompletedProcess):
    """A finished run. Like ``subprocess.CompletedProcess``, with ``transcript``: the
    file that received the standard output (the standard error file is beside it), or
    None if the output was kept in memory only."""

    def __init__(self, args, returncode, stdout, stderr, transcript=None):
        super().__init__(args, returncode, stdout, stderr)
        self.transcript = transcript


def resolve_executable(
    name: StrPath,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the path of the executable that a child process started in ``cwd`` with
    the environment ``env`` runs for ``name``, as ``subprocess.Popen`` finds it. A name
    with a slash is relative to ``cwd``. Otherwise, the directories of PATH are
    searched, and an empty or relative directory is relative to ``cwd``. Returns None
    if there is no such executable."""
    name = os.fspath(name)
    base = os.path.abspath(cwd if cwd is not None else os.curdir)
    if os.sep in name:
        return os.path.join(base, name)
    for directory in os.get_exec_path(env):
        candidate = os.path.join(base, directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# The AMD tools that run_tool finds by name (xilinx_tool).
XILINX_TOOLS = frozenset({"vivado", "vitis_hls", "vitis-run", "v++", "xelab", "slashkit"})
# A directory of wrappers that replaces PATH for XILINX_TOOLS, such as a site's LSF
# dispatcher. It must hold a wrapper for each tool that a build runs.
TOOL_DIR_ENV = "FINN_TOOL_DIR_OVERRIDE"


# FINN reads the AMD tools' own environment, as a release's settings64.sh (or a site's
# environment module) sets it, and never builds it: whoever starts FINN does.
AMD_VARIABLES = ("XILINX_VIVADO", "XILINX_HLS", "XILINX_VITIS")
# Where AMD's installer puts releases. FINN only looks here to tell a user without the
# tools which settings64.sh to source; it never uses what it finds.
INSTALL_ROOTS = ("/tools/Xilinx", "/opt/Xilinx", "~/Xilinx")
# A release in a path: the last one, since a site's install root may name a build of
# it too (/proj/xbuilds/SWIP/2022.2_1014_8888/installs/lin64/Vivado/2022.2).
_RELEASE = re.compile(r"(?<![\d.])(20\d\d)\.([12])(?![\d.])")


def amd_path(name: str) -> Optional[str]:
    """Return the directory that AMD's variable ``name`` (one of ``AMD_VARIABLES``)
    names, or None."""
    return os.environ.get(name) or None


def release_of(path: StrPath) -> Optional[Tuple[int, int]]:
    """Return the AMD release (year, half) that ``path`` lies in, or None."""
    found = _RELEASE.findall(os.fspath(path))
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def installed_releases(
    roots: Optional[Sequence[str]] = None,
) -> List[Tuple[Tuple[int, int], str]]:
    """Return ``(release, settings64.sh)`` for each Vivado release under ``roots``, in
    both of AMD's layouts, newest first."""
    found: Dict[Tuple[int, int], str] = {}
    for root in INSTALL_ROOTS if roots is None else roots:
        root = os.path.expanduser(root)
        for pattern in ("*/Vivado/settings64.sh", "Vivado/*/settings64.sh"):
            for script in sorted(Path(root).glob(pattern)):
                release = release_of(script.relative_to(root))
                if release:
                    found.setdefault(release, str(script))
    return sorted(found.items(), reverse=True)


def setup_hint(roots: Optional[Sequence[str]] = None) -> str:
    """Tell a user whose environment has no AMD tools what to source."""
    releases = installed_releases(roots)
    if not releases:
        return (
            "Install the AMD tools and source a release's settings64.sh "
            "(<root>/<release>/Vivado/settings64.sh), or load the site's Vivado module."
        )
    (year, half), script = releases[0]
    hint = f"To use Vivado {year}.{half}: source {script}"
    if len(releases) > 1:
        others = ", ".join(f"{y}.{h}" for (y, h), _ in releases[1:])
        hint += f" (also installed: {others})"
    return hint


class ToolchainMismatch(RuntimeError):
    """AMD tools from different releases in one environment."""


def expected_release() -> Optional[Tuple[int, int]]:
    """Return the release of the AMD tools set up in this environment, or None.
    Raises ``ToolchainMismatch`` if the AMD variables name different releases."""
    releases = {}
    for name in AMD_VARIABLES:
        path = amd_path(name)
        release = release_of(path) if path else None
        if release:
            releases[name] = (release, path)
    if len({release for release, _ in releases.values()}) > 1:
        raise ToolchainMismatch(
            "the AMD tools in this environment come from different releases: "
            + ", ".join(f"{name}={path}" for name, (_, path) in releases.items())
            + ". Start FINN from a shell that sourced one release's settings64.sh."
        )
    return next(iter(releases.values()))[0] if releases else None


def not_set_up(tool: str, reason: str) -> "ToolUnavailable":
    """``ToolUnavailable`` for ``tool``, adding what to source when the environment
    has no AMD tools at all."""
    if not any(amd_path(name) for name in AMD_VARIABLES):
        reason = f"{reason}. No AMD tools are set up in this environment. {setup_hint()}"
    return ToolUnavailable(tool, reason)


def hls_root() -> str:
    """Return the Vitis HLS installation: the one of XILINX_HLS and XILINX_VITIS that
    holds the HLS headers (from 2024.2 they are part of Vitis)."""
    paths = [path for path in (amd_path("XILINX_HLS"), amd_path("XILINX_VITIS")) if path]
    for path in paths:
        if os.path.isfile(os.path.join(path, "include", "hls_stream.h")):
            return path
    if paths:
        raise ToolUnavailable(
            "vitis_hls",
            "no HLS headers (include/hls_stream.h) in " + " or ".join(paths),
        )
    raise not_set_up("vitis_hls", "XILINX_HLS and XILINX_VITIS are not set")


@functools.lru_cache(maxsize=None)
def xilinx_tool(name: str) -> str:
    """Return the path of the AMD tool ``name``: ``<FINN_TOOL_DIR_OVERRIDE>/<name>`` if
    that variable is set, else ``name`` on PATH. Raises ``ToolUnavailable`` if there is
    no such executable, and ``ToolchainMismatch`` if it belongs to another release than
    the AMD variables name.

    The tools are set up once, before FINN starts, so each name is resolved once per
    process. ``xilinx_tool.cache_clear()`` forgets the results, for a test that
    changes the environment. A failed lookup is not kept."""
    expected = expected_release()
    override = os.environ.get(TOOL_DIR_ENV)
    if override:
        path = os.path.abspath(os.path.join(override, name))
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise ToolUnavailable(name, f"no executable {path} ({TOOL_DIR_ENV}={override})")
        return path
    found_path = shutil.which(name)
    if found_path is None:
        raise not_set_up(name, "not found in PATH")
    path = os.path.abspath(found_path)
    # A wrapper (a site's, or the container's shim) names no release; a tool that
    # names another release than the variables would run a different Vivado.
    found = release_of(os.path.realpath(path)) or release_of(path)
    if expected and found and found != expected:
        raise ToolchainMismatch(
            f"{name} on PATH is {path} (release {found[0]}.{found[1]}), but the AMD "
            f"variables name release {expected[0]}.{expected[1]}. Start FINN from a shell "
            "that sourced one release's settings64.sh."
        )
    return path


def _popen(command, **kwargs):
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        if exc.errno != errno.ENOEXEC:
            raise
    # ENOEXEC: a script without a "#!" line, such as a wrapper in FINN_TOOL_DIR_OVERRIDE.
    log.debug("%s has no #! line, running it with bash", command[0])
    return subprocess.Popen(["bash", *command], **kwargs)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# Signals that end this process, for example when a multiprocessing pool terminates
# its workers, or when the terminal closes.
_TERMINATING_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@contextlib.contextmanager
def _killed_on_termination(started):
    """While the body runs, SIGTERM and SIGHUP first kill the process group of the
    process in ``started`` (a list that holds it once it runs), then have the effect
    of the handler that was installed before. A signal that is ignored (SIGHUP under
    nohup) stays ignored and kills nothing. Signal handlers can only be changed in
    the main thread, so other threads get no such cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def terminating(signum, frame):
        for proc in started:
            _kill_group(proc)
        handler = previous[signum]
        if callable(handler):
            handler(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for signum in _TERMINATING_SIGNALS:
        if signal.getsignal(signum) != signal.SIG_IGN:
            previous[signum] = signal.signal(signum, terminating)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


@contextlib.contextmanager
def _signals_held():
    """Hold SIGINT, SIGTERM and SIGHUP while the body runs, and deliver them after it.
    Thus a process that the body starts and records is recorded before a signal can
    stop the body, and the cleanup finds it. Main thread only, as
    ``_killed_on_termination``."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held = []
    previous = {}
    for signum in (signal.SIGINT, *_TERMINATING_SIGNALS):
        if signal.getsignal(signum) not in (signal.SIG_IGN, None):
            previous[signum] = signal.signal(signum, lambda signum, frame: held.append(signum))
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for signum in dict.fromkeys(held):
            signal.raise_signal(signum)


def run_process(
    argv: Sequence[StrPath],
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[StrPath] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    transcript: Optional[StrPath] = None,
) -> ToolRun:
    """Run ``argv`` in a new session and wait for it.

    The executable is ``resolve_executable(argv[0], cwd, env)``. If it is a script
    without a ``#!`` line, it runs with bash. With a ``transcript`` stem, the process
    writes its output into the two transcript files while it runs. Without one, the
    output is kept in memory. The result holds the output in both cases. With
    ``check``, a nonzero exit status raises ``subprocess.CalledProcessError``.

    FINN kills the process group of the tool (SIGKILL) on a timeout
    (``subprocess.TimeoutExpired``), on any exception while it waits, such as
    KeyboardInterrupt, and, in the main thread, on SIGTERM and SIGHUP unless this
    process ignores them. A signal that arrives while the tool starts waits until
    FINN holds the tool. The exception carries the output so far in ``output`` and
    ``stderr``. This does not cover a SIGKILL of this process, or descendants that
    leave the process group of the tool (a new process group or session).
    In a non-main thread, terminal Ctrl-C does not directly reach the tool's
    separate process group, and no termination signal handlers are installed.
    Timeouts and exceptions in the waiting thread still clean up its group.
    """
    command = [os.fspath(arg) for arg in argv]
    if not command:
        raise ValueError("Empty command")
    command[0] = resolve_executable(command[0], cwd, env) or command[0]
    files = None if transcript is None else transcript_files(transcript)
    log.info(
        "Running %s in %s, transcript %s",
        shlex.join(command),
        os.fspath(cwd) if cwd is not None else os.getcwd(),
        "none" if files is None else files[0],
    )
    started = time.monotonic()
    running: List[subprocess.Popen] = []

    def printed():
        return files[0].read_bytes(), files[1].read_bytes()

    with _killed_on_termination(running):
        try:
            if files is None:
                with _signals_held():
                    running.append(
                        _popen(
                            command,
                            cwd=cwd,
                            env=env,
                            start_new_session=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    )
                out, err = running[0].communicate(timeout=timeout)
            else:
                files[0].parent.mkdir(parents=True, exist_ok=True)
                # The child writes the files itself, so they grow while it runs.
                with open(files[0], "wb") as stdout, open(files[1], "wb") as stderr:
                    with _signals_held():
                        running.append(
                            _popen(
                                command,
                                cwd=cwd,
                                env=env,
                                start_new_session=True,
                                stdout=stdout,
                                stderr=stderr,
                            )
                        )
                running[0].wait(timeout=timeout)
                out, err = printed()
        except BaseException as exc:
            if not running:
                raise
            proc = running[0]
            # Kill the whole group: a tool's descendants (Vivado's synthesis runs, a
            # JVM behind vitis-run) can outlive its leader.
            _kill_group(proc)
            if files is None:
                out, err = proc.communicate()
            else:
                proc.wait()
                out, err = printed()
            if isinstance(exc, (subprocess.TimeoutExpired, KeyboardInterrupt)):
                # TimeoutExpired has these attributes, KeyboardInterrupt gets them
                setattr(exc, "output", out)
                setattr(exc, "stderr", err)
            log.warning(
                "Stopped %s after %.1f s (%s)",
                command[0],
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise
    proc = running[0]
    log.info(
        "Finished %s after %.1f s with exit status %d",
        command[0],
        time.monotonic() - started,
        proc.returncode,
    )
    result = ToolRun(command, proc.returncode, out, err, None if files is None else files[0])
    if check:
        result.check_returncode()
    return result


def _text(stream):
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream or ""


def emit_output(result: subprocess.CompletedProcess, logger: logging.Logger) -> Tuple[str, str]:
    """Decode captured output, log it in a build or write it through outside one.

    Output is emitted before the caller checks the status, so failed runs retain
    their diagnostics. Return both streams as UTF-8 strings with replacement.
    """
    out, err = _text(result.stdout), _text(result.stderr)
    if in_run() and logger.isEnabledFor(logging.INFO):
        for stream in (out, err):
            for line in stream.splitlines():
                logger.info(line, extra=FILE_ONLY)
    else:
        sys.stdout.write(out)
        sys.stderr.write(err)
    return out, err


class ToolUnavailable(LookupError):
    """A tool that FINN cannot run, because it is not installed or not set up."""

    def __init__(self, tool: str, reason: str):
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason

    def __reduce__(self):
        return (type(self), (self.tool, self.reason))


class ToolFailed(RuntimeError):
    """A tool that ran and failed.

    Attributes:
        tool: the name of the tool.
        command: its command line, if known.
        status: its exit status, or None if it did not exit with one.
        transcript: the file that holds its output, if there is one.
        errors: the first error lines it printed (see ``ToolFailed.of``).
        code: ``vivado.LICENSE_UNAVAILABLE`` if its output says that no license was
            found, else ``vivado.TOOL_ERROR``.
    """

    def __init__(
        self,
        tool: str,
        status: Optional[int] = None,
        transcript: Optional[StrPath] = None,
        errors: Sequence[str] = (),
        code: str = vivado.TOOL_ERROR,
        command: Sequence[str] = (),
    ):
        self.tool = tool
        self.command = tuple(command)
        self.status = status
        self.transcript = None if transcript is None else Path(transcript)
        self.errors = tuple(errors)
        self.code = code
        message = f"{tool} failed"
        if status is not None:
            message += f" with exit status {status}"
        if code != vivado.TOOL_ERROR:
            message += f" ({code})"
        if self.transcript is not None:
            message += f"; transcript {self.transcript}"
        super().__init__("\n  ".join((message, *self.errors)))

    def __reduce__(self):
        # Keep all fields across a multiprocessing pool (NodeLocalTransformation).
        return (
            type(self),
            (
                self.tool,
                self.status,
                self.transcript,
                self.errors,
                self.code,
                self.command,
            ),
        )

    @classmethod
    def of(
        cls,
        tool: str,
        printed: Union[str, bytes],
        *,
        status: Optional[int] = None,
        transcript: Optional[StrPath] = None,
        command: Sequence[str] = (),
    ) -> "ToolFailed":
        """The failure of ``tool``, read from what it ``printed``.

        The quoted lines are its errors, in the order printed: Vivado-style errors
        (``vivado.messages``), the message of an uncaught Tcl error, and other
        ``error:`` lines (a compiler's), then its critical warnings. If it printed no
        errors, its last lines.
        """
        if isinstance(printed, bytes):
            printed = printed.decode(errors="replace")
        lines = [
            line
            for line in (raw.strip() for raw in printed.splitlines())
            if line and not _LOADER_NOTICE.match(line)
        ]
        read = vivado.messages(lines)
        structured = set(read.errors)
        critical = set(read.critical_warnings)
        errors = [
            line
            for line, after in zip(lines, lines[1:] + [""])
            if line in structured
            or after == _TCL_TRACE
            or (
                line not in critical
                and not line.startswith(_ECHOED_TCL)
                and _ERROR_LINE.search(line)
            )
        ]
        if errors:
            quoted = list(dict.fromkeys(errors + list(read.critical_warnings)))[:ERROR_LINES]
        else:
            quoted = lines[-ERROR_LINES:]
        return cls(
            tool,
            status=status,
            transcript=transcript,
            errors=quoted,
            code=read.code,
            command=command,
        )

    @classmethod
    def of_run(cls, tool: str, result: subprocess.CompletedProcess) -> "ToolFailed":
        """The failure of a run of ``tool`` that exited with a nonzero status."""
        transcript = getattr(result, "transcript", None)
        if transcript is not None and not result.stdout and result.stderr:
            path = Path(transcript)
            if path.name.endswith(".stdout.log"):
                transcript = path.with_name(path.name[: -len(".stdout.log")] + ".stderr.log")
        return cls.of(
            tool,
            # One line break between the streams, so that their lines stay apart.
            "\n".join(_text(stream) for stream in (result.stdout, result.stderr)),
            status=result.returncode,
            transcript=transcript,
            command=result.args,
        )


def run_tool(
    command: Sequence[StrPath],
    *,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    replay: Optional[StrPath] = None,
    transcript: Optional[StrPath] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    log_output: bool = False,
) -> ToolRun:
    """Run an external tool with ``run_process``, and raise ``ToolFailed`` if it exits
    with a nonzero status.

    Args:
        command: the command line, the tool first. A name in ``XILINX_TOOLS``, such
            as ``"vivado"``, runs the executable that ``xilinx_tool`` finds.
        cwd: the working directory of the tool.
        env: the whole environment of the tool. None means this process's
            environment.
        timeout: the seconds after which the tool is killed.
        replay: a bash script to write before the run, which repeats the run by hand.
            It runs the same executable and exports ``env_overrides``. It does not
            record the rest of ``env``: run it in the same environment.
        transcript: the stem of the transcript files. By default, the replay path
            without its suffix. Without either, the output is kept in memory only.
        env_overrides: variables to override in ``env`` (or this process's
            environment) and export in the replay. Other variables are not saved.
        log_output: also emit captured output with ``emit_output`` before checking
            the exit status. Defaults to transcript-only output.
    """
    argv = [os.fspath(arg) for arg in command]
    if not argv:
        raise ValueError("Empty command")
    if argv[0] in XILINX_TOOLS and "XILINX_LOCAL_USER_DATA" not in (env or os.environ):
        # AMD's setting for concurrent Vivado runs (support article 63253).
        env_overrides = {"XILINX_LOCAL_USER_DATA": "no", **(env_overrides or {})}
    if env_overrides:
        for name in env_overrides:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"Invalid environment variable name: {name!r}")
        env = {**(os.environ if env is None else env), **env_overrides}
    # Preserve working-directory errors instead of blaming a missing tool.
    if cwd is not None:
        os.stat(cwd)
        if not os.path.isdir(cwd):
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), os.fspath(cwd))
    if argv[0] in XILINX_TOOLS:
        argv[0] = xilinx_tool(argv[0])
    executable = resolve_executable(argv[0], cwd, env)
    if executable is None:
        raise ToolUnavailable(argv[0], "not found in PATH")
    if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
        raise ToolUnavailable(argv[0], f"no executable {executable}")
    argv[0] = executable
    if replay is not None:
        lines = ["#!/bin/bash", "set -e"]
        if cwd is not None:
            lines.append("cd -- " + shlex.quote(os.fspath(cwd)))
        for name, value in (env_overrides or {}).items():
            lines.append("export " + name + "=" + shlex.quote(value))
        lines.append("exec " + shlex.join(argv))
        Path(replay).write_text("\n".join(lines) + "\n")
        if transcript is None:
            transcript = os.path.splitext(os.fspath(replay))[0]
    try:
        result = run_process(
            argv,
            env=env,
            cwd=cwd,
            timeout=timeout,
            check=False,
            transcript=transcript,
        )
    except OSError as exc:
        if exc.filename == argv[0] and exc.errno in (
            errno.ENOENT,
            errno.EACCES,
            errno.ENOEXEC,
        ):
            raise ToolUnavailable(os.path.basename(argv[0]), str(exc)) from exc
        raise
    if log_output:
        emit_output(result, log)
    if result.returncode != 0:
        raise ToolFailed.of_run(os.path.basename(argv[0]), result)
    return result
