"""Internal, bulk-data XSI sessions. The parent never imports the native bridge.

This direct entry point is independent of build orchestration. Native support
requires installation-backed validation before integration into FINN build flows.
"""

import json
import numpy as np
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from finn.util.toolchain import run_process
from finn.xsi._artifacts import digest, validate_record


@dataclass(frozen=True)
class SessionRequest:
    bridge: str
    kernel: str
    design: str
    design_cwd: str
    tool_identity: str
    output_counts: dict
    stream_suffix: str = "_V_V"
    reset_cycles: int = 16
    liveness_threshold: int = 10000
    trace: bool = False
    # Optional explicit script defining run(sim, io, request) -> metrics.
    # It runs inside this session; no parent closures or live handles cross exec.
    testbench: str = ""
    arguments: dict = field(default_factory=dict)


class SessionError(RuntimeError):
    def __init__(self, message, directory):
        self.directory = Path(directory)
        super().__init__(f"{message}; session logs/artifacts: {directory}")


def pack_streams(streams):
    """Portable, pickle-free arrays preserving integers wider than uint64."""
    packed = {}
    for name, words in streams.items():
        values = [int(word) for word in words]
        if not isinstance(name, str) or not name or any(word < 0 for word in values):
            raise ValueError("Streams need nonempty names and unsigned packed words")
        packed[name] = np.asarray([format(word, "x") for word in values], dtype=str)
    return packed


def unpack_streams(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: [int(word, 16) for word in arrays[name].tolist()] for name in arrays.files}


def run_session(
    request,
    inputs,
    *,
    environment,
    output_root,
    timeout,
    cancel=None,
    python_command=None,
    launcher=(),
):
    """Execute one complete simulation in a fresh executable image.

    Paths must be visible unchanged to the selected route. Site routes require an
    explicit site Python/driver command and own remote descendant cancellation.
    Logs and possibly partial waveforms remain after failures/timeouts. Successful
    results are read only after exit and carry this invocation's artifact hashes.
    """
    if timeout is None or timeout <= 0:
        raise ValueError("A positive external wall-clock timeout is required")
    if launcher and not python_command:
        raise ValueError("Site execution requires an explicit site Python command")
    data = asdict(request)  # Defensive deep copy of selections/arguments.
    if data["reset_cycles"] < 1 or data["liveness_threshold"] < 1:
        raise ValueError("Reset cycles and watchdog limits must be positive")
    if not data["output_counts"] or any(
        not isinstance(count, int) or count < 1 for count in data["output_counts"].values()
    ):
        raise ValueError("Expected output counts must be positive integers")
    for key in ("bridge", "kernel", "design", "design_cwd", "testbench"):
        if data[key]:
            data[key] = str(Path(data[key]).resolve(strict=True))
    if not Path(data["design_cwd"]).is_dir():
        raise ValueError("design_cwd must be the compiled design's working directory")
    for key in ("bridge", "design"):
        record = validate_record(
            data[key], kind=key, tool=data["tool_identity"], check_abi=not python_command
        )
        if key == "bridge":
            data["python_abi"] = record["python_abi"]
    data["artifacts"] = {key: digest(data[key]) for key in ("bridge", "kernel", "design")}
    if data["testbench"]:
        data["artifacts"]["testbench"] = digest(data["testbench"])
    directory = Path(tempfile.mkdtemp(prefix="xsi-session-", dir=output_root)).resolve()
    data["session"] = directory.name
    np.savez(directory / "inputs.npz", **pack_streams(inputs))
    (directory / "request.json").write_text(json.dumps(data, sort_keys=True) + "\n")
    command = [
        *launcher,
        *(python_command or [sys.executable]),
        "-I",
        "-m",
        "finn.xsi._session_worker",
        str(directory),
    ]
    try:
        result = run_process(
            command,
            env=dict(environment),
            cwd=data["design_cwd"],
            timeout=timeout,
            cancel=cancel,
        )
    except (OSError, subprocess.SubprocessError, InterruptedError, KeyboardInterrupt) as exc:
        (directory / "stdout.log").write_bytes(getattr(exc, "output", None) or b"")
        (directory / "stderr.log").write_bytes(getattr(exc, "stderr", None) or b"")
        raise SessionError(
            f"XSI session failed ({type(exc).__name__}); waveforms may be partial", directory
        ) from exc
    (directory / "stdout.log").write_bytes(result.stdout)
    (directory / "stderr.log").write_bytes(result.stderr)
    try:
        record = json.loads((directory / "result.json").read_text())
        if (
            record["status"] != "success"
            or record["session"] != data["session"]
            or record["artifacts"] != data["artifacts"]
        ):
            raise ValueError("Mismatched or incomplete session result")
        outputs = unpack_streams(directory / "outputs.npz")
        if {key: len(value) for key, value in outputs.items()} != data["output_counts"]:
            raise ValueError("Output buffers disagree with requested stream counts")
    except (OSError, ValueError, KeyError) as exc:
        raise SessionError("Invalid or missing XSI session result", directory) from exc
    return {
        "outputs": outputs,
        "metrics": record["metrics"],
        "directory": str(directory),
        "trace": record["trace"],
        "tool_identity": data["tool_identity"],
    }
