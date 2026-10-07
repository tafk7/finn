"""FINN_LEGACY_COMPAT: remaining legacy inputs, interpreted only on demand.

See docs/legacy-build-env-ledger.md. No function mutates the parent environment.
"""
import os
import re
from pathlib import Path

from finn import resources
from finn.util.toolchain import Selection

# Settings scripts are sourced in this order: each prepends to PATH, so a
# later one shadows an earlier one's tools.
_ROOTS = ("XILINX_VITIS", "XILINX_VIVADO", "XILINX_HLS")


def checkout_root(root=None, environ=None):
    env = os.environ if environ is None else environ
    value = root if root is not None else env.get("FINN_ROOT")
    if not value:
        raise RuntimeError("This legacy checkout operation requires an explicit root or FINN_ROOT")
    return str(Path(value).resolve())


def build_directory(path=None, environ=None):
    env = os.environ if environ is None else environ
    value = path if path is not None else env.get("FINN_BUILD_DIR")
    return str(Path(value or resources.home(env) / "build").expanduser().resolve())


def toolchain(environ=None):
    """Translate legacy tool inputs once for unmigrated public operations."""
    env = dict(os.environ if environ is None else environ)
    frontend = env.get("FINN_HLS_FRONTEND")
    if frontend is None:
        match = re.search(r"\b(20\d{2})\.(\d+)\b", env.get("XILINX_VIVADO", ""))
        version = tuple(map(int, match.groups())) if match else None
        frontend = "vitis-run" if version and version > (2024, 2) else "vitis_hls"
    scripts = []
    # Site command-directory wrappers own their activation. Do not turn them
    # into local tools or require a local AMD installation before dispatching.
    command_dir = env.get("FINN_TOOL_DIR_OVERRIDE", "")
    if not command_dir:
        for variable in _ROOTS:
            root = env.get(variable)
            if root:
                script = str(Path(root) / "settings64.sh")
                if Path(script).is_file() and script not in scripts:
                    scripts.append(script)
    selection = Selection(settings=tuple(scripts), command_dir=command_dir, hls_frontend=frontend)
    # Legacy ambient mode accepts its inherited base; explicit Selection.prepare
    # has a clean default base instead. This does not claim to unsource anything.
    return selection.prepare(env)


def build_environment(selection, environ=None, *, root=None, build_dir=None):
    """Prepare the child tree of an explicit build entry point: the parent's
    environment, FINN_ROOT and FINN_BUILD_DIR resolved, with the toolchain
    ``selection`` (a ``finn.util.toolchain.Selection``, the build configuration's)
    prepared over it, and the loader paths of its simulator libraries."""
    env = dict(os.environ if environ is None else environ)
    if root is not None:
        env["FINN_ROOT"] = checkout_root(root)
    env["FINN_BUILD_DIR"] = build_directory(build_dir, env)
    Path(env["FINN_BUILD_DIR"]).mkdir(parents=True, exist_ok=True)
    # Loader paths must exist BEFORE Python starts. Retain the XSI limitation
    # here; ordinary imports and resource operations never call this function.
    # Prepared over the parent's environment, not a clean base: the child keeps
    # FINN's own variables (resources, build directory) beside the tools'.
    env = dict(selection.prepare(env).environment)
    libraries = []
    for variable, suffix in (
        ("XILINX_VIVADO", "lib/lnx64.o"),
        ("XILINX_VITIS", "lnx64/tools/fpo_v7_1"),
        ("XILINX_HLS", "lnx64/tools/fpo_v7_1"),
    ):
        if env.get(variable):
            directory = Path(env[variable]) / suffix
            if directory.is_dir():
                libraries.append(str(directory))
    if libraries:
        env["LD_LIBRARY_PATH"] = ":".join(
            libraries + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    return env
