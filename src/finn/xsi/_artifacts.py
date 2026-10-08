"""Small adjacent compatibility records for native bridge and compiled designs."""

import hashlib
import json
import sysconfig
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tool_identity(toolchain):
    return json.dumps(
        {
            "version": toolchain.probe("vivado"),
            "installation": toolchain.environment.get("XILINX_VIVADO", ""),
            "command_dir": toolchain.selection.command_dir or "",
        },
        sort_keys=True,
    )


def write_record(artifact, *, kind, tool, sources, arguments=()):
    artifact = Path(artifact).resolve(strict=True)
    record = {
        "kind": kind,
        "tool": tool,
        "artifact": digest(artifact),
        "python_abi": sysconfig.get_config_var("SOABI") if kind == "bridge" else None,
        "sources": {str(Path(path).resolve(strict=True)): digest(path) for path in sources},
        "arguments": list(map(str, arguments)),
    }
    Path(str(artifact) + ".finn.json").write_text(json.dumps(record, sort_keys=True) + "\n")


def validate_record(artifact, *, kind, tool, check_abi=True):
    record = json.loads(Path(str(artifact) + ".finn.json").read_text())
    if record["kind"] != kind or record["tool"] != tool or record["artifact"] != digest(artifact):
        raise ValueError(f"Incompatible {kind} artifact/tool selection: {artifact}; rebuild")
    if check_abi and kind == "bridge" and record["python_abi"] != sysconfig.get_config_var("SOABI"):
        raise ValueError("XSI bridge Python ABI changed; rebuild")
    for path, expected in record["sources"].items():
        if digest(path) != expected:
            raise ValueError(f"Native compile input changed: {path}; rebuild")
    return record
