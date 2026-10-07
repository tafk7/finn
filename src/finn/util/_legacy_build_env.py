"""FINN_LEGACY_COMPAT: remaining legacy inputs, interpreted only on demand.

See docs/legacy-build-env-ledger.md. No function mutates the parent environment.
"""
import os
from pathlib import Path

from finn import resources


def build_directory(path=None, environ=None):
    env = os.environ if environ is None else environ
    value = path if path is not None else env.get("FINN_BUILD_DIR")
    return str(Path(value or resources.home(env) / "build").expanduser().resolve())


def build_environment(selection, environ=None, *, build_dir=None):
    """Prepare the child tree of an explicit build entry point: the parent's
    environment, FINN_BUILD_DIR resolved, with the toolchain
    ``selection`` (a ``finn.util.toolchain.Selection``, the build configuration's)
    prepared over it, and the loader paths of its simulator libraries."""
    env = dict(os.environ if environ is None else environ)
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
