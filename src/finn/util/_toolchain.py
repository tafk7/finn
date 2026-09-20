"""Internal AMD command execution. No discovery or activation occurs on import."""

import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_LOG = logging.getLogger(__name__)


def _child_environment(environment):
    # Bash reads BASH_ENV even with --noprofile --norc. Exported shell functions
    # and option variables can also change sourcing before our command runs.
    return {
        str(k): str(v)
        for k, v in environment.items()
        if k not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"} and not k.startswith("BASH_FUNC_")
    }


def run_process(argv, *, env=None, cwd=None, timeout=None, cancel=None, check=True):
    """Run argv in a new local process group; reap it on timeout or cancellation.

    A remote wrapper is responsible for cancelling its remote descendants.
    Environment mappings are copied; neither cwd nor os.environ is changed.
    """
    argv = [os.fspath(arg) for arg in argv]
    if not argv:
        raise ValueError("Empty command")
    environment = _child_environment(os.environ if env is None else env)
    started = time.monotonic()
    _LOG.info("command=%s cwd=%s", shlex.join(argv), cwd or os.getcwd())
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Command cancelled: " + shlex.join(argv))
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                out, err = proc.communicate(
                    timeout=min(0.1, remaining)
                    if remaining is not None
                    else (0.1 if cancel is not None else None)
                )
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
        if isinstance(exc, subprocess.TimeoutExpired):
            exc.output, exc.stderr = out, err
        _LOG.warning(
            "command stopped duration=%.3fs result=%s",
            time.monotonic() - started,
            type(exc).__name__,
        )
        raise
    result = subprocess.CompletedProcess(argv, proc.returncode, out, err)
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

    settings: tuple = ()
    command_dir: str = ""
    launcher: tuple = ()
    hls_frontend: str = "vitis_hls"

    def __post_init__(self):
        object.__setattr__(self, "settings", tuple(map(os.fspath, self.settings)))
        object.__setattr__(self, "launcher", tuple(map(os.fspath, self.launcher)))
        if self.launcher and self.settings:
            raise ValueError(
                "Site routes own activation; do not combine launcher and local settings"
            )
        if self.hls_frontend not in {"vivado_hls", "vitis_hls", "vitis-run"}:
            raise ValueError("Unknown HLS frontend: " + self.hls_frontend)

    def prepare(self, base_env=None, timeout=30):
        if base_env is None:
            if self.settings:
                # A defined clean base, not an attempt to unsource another toolchain.
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
                }
                base_env = {k: v for k, v in os.environ.items() if k in keep}
                base_env["PATH"] = os.defpath
            else:
                base_env = os.environ
        environment = _child_environment(base_env)
        if self.settings:
            scripts = [str(Path(path).resolve(strict=True)) for path in self.settings]
            bash = shutil.which("bash", path=os.defpath)
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
        return Toolchain(self, environment)


@dataclass(frozen=True)
class Toolchain:
    selection: Selection
    environment: object

    def __post_init__(self):
        object.__setattr__(
            self, "environment", MappingProxyType(_child_environment(self.environment))
        )

    def command(self, tool, *args):
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
                f"{executable} not found (FINN_TOOL_DIR_OVERRIDE={selection.command_dir!r})"
            )
        return [*selection.launcher, executable, *map(os.fspath, args)]

    def run(self, tool, args=(), *, cwd=None, env=None, timeout=None, cancel=None, check=True):
        environment = dict(self.environment)
        environment.update({} if env is None else dict(env))
        _LOG.info(
            "route=%s settings=%s tool=%s",
            self.selection.launcher or "local",
            self.selection.settings or "configured environment",
            tool,
        )
        return run_process(
            self.command(tool, *args),
            env=environment,
            cwd=cwd,
            timeout=timeout,
            cancel=cancel,
            check=check,
        )

    def probe(self, tool, timeout=10):
        """Query identity through exactly the execution route. Not a licence test."""
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
        return version

    def hls_command(self, script, timeout=10):
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
        if frontend == "vitis-run":
            help_result = self.run(frontend, ["--help"], timeout=timeout)
            help_text = (help_result.stdout + help_result.stderr).decode("utf-8", errors="replace")
            if "hls" not in help_text.lower():
                raise RuntimeError("Selected vitis-run does not advertise HLS capability")
            _LOG.info("capability tool=vitis-run hls=advertised")
        args = ["--mode", "hls", "--tcl", script] if frontend == "vitis-run" else ["-f", script]
        return frontend, args
