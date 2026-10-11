############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Setup script for FINN XSI (RTL simulation) support.

This script builds and configures the finn_xsi C++ extension module
required for RTL simulation in FINN.

Usage:
    python -m finn.xsi.setup [options]

Options:
    --force    Force rebuild even if already built
    --clean    Clean build artifacts
    --check    Only check build prerequisites; does not inspect the extension

RTL simulation builds the extension on first use (ensure_built); running this
module only builds it ahead of time. Each function takes the toolchain whose Vivado
the extension is built against (its XSim headers); without a Vivado,
``ToolUnavailable``. The command line builds against the machine's (``host_toolchain``).
"""

import argparse
import fcntl
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple

from finn.toolchain import vivado
from finn.toolchain.launcher import (
    Toolchain,
    ToolUnavailable,
    host_toolchain,
    identity,
    run_process,
)
from finn.xsi._artifacts import validate_record, write_record
from finn.xsi.paths import find_xsi_so, xsi_artifact_dir, xsi_source_dir


def get_build_paths(toolchain: Toolchain) -> Tuple[List[str], str, List[str]]:
    """Get include paths and compiler for building the extension against the Vivado
    ``toolchain`` selects.

    Returns:
        Tuple of (include_dirs, compiler, extra_compile_args)
    """
    include_dirs = []

    # Get Python include directory
    python_include = sysconfig.get_path("include")
    if python_include:
        include_dirs.append(python_include)

    # Try to get pybind11 include directory
    # flake8: noqa
    try:
        import pybind11

        pybind11_include = pybind11.get_include()
        include_dirs.append(pybind11_include)

        # Also get the user-specific include
        pybind11_user_include = pybind11.get_include(user=True)
        if pybind11_user_include != pybind11_include:
            include_dirs.append(pybind11_user_include)
    except ImportError:
        # Will be caught later in prerequisites check
        pass

    # The XSim headers of the selected Vivado
    xilinx_include = vivado.installation(toolchain) / "data" / "xsim" / "include"
    if xilinx_include.exists():
        include_dirs.append(str(xilinx_include))

    # Determine compiler
    compiler = "g++"
    if not shutil.which("g++") and shutil.which("clang++"):
        compiler = "clang++"

    # Compile flags
    extra_compile_args = ["--std=c++17", "-Wall", "-O3", "-shared", "-fPIC"]

    return include_dirs, compiler, extra_compile_args


def check_prerequisites(toolchain: Toolchain) -> List[str]:
    """Check if required tools are available: a compiler, pybind11, and a Vivado
    with XSim headers in ``toolchain``."""
    errors = []

    # Check for C++ compiler
    if not shutil.which("g++") and not shutil.which("clang++"):
        errors.append("No C++ compiler found. Please install g++ or clang++.")

    # Check for the selected Vivado and its XSim headers
    try:
        toolchain.command("vivado")
        headers = vivado.installation(toolchain) / "data" / "xsim" / "include"
        if not headers.exists():
            errors.append(f"Xilinx XSim headers not found at {headers}")
    except ToolUnavailable as missing:
        errors.append(str(missing))

    # Check for pybind11
    try:
        import pybind11
    except ImportError:
        errors.append("pybind11 not found. Please install it with: pip install pybind11")

    return errors


SOURCE_FILES = ["xsi_bind.cpp", "xsi_finn.cpp"]


def _usable(artifact: Path, toolchain: Toolchain) -> bool:
    if not artifact.is_file():
        return False
    try:
        validate_record(artifact, kind="bridge", tool=identity(toolchain, "xsim"))
    except (OSError, ValueError, KeyError):
        return False
    return True


def _compile(artifact: Path, verbose: bool, toolchain: Toolchain) -> None:
    """Compile xsi.so to artifact against ``toolchain``, atomically, and record its
    inputs."""
    xsi_path = xsi_source_dir()
    include_dirs, compiler, compile_args = get_build_paths(toolchain)
    partial = artifact.with_name(f".{artifact.name}.{os.getpid()}")
    cmd = [compiler, *compile_args]
    for inc_dir in include_dirs:
        cmd.extend(["-I", inc_dir])
    cmd.extend(["-o", str(partial), *SOURCE_FILES, "-ldl", "-lrt"])
    if verbose:
        print(f"Building finn_xsi: {' '.join(cmd)}")
    result = run_process(cmd, cwd=xsi_path, env=toolchain.environment, check=False)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Building the finn_xsi extension failed:\n"
            + (result.stderr or result.stdout or b"").decode(errors="replace")
            + "\nCommand: "
            + " ".join(cmd)
        )
    os.replace(partial, artifact)
    sources = [xsi_path / name for name in SOURCE_FILES] + list(xsi_path.glob("*.hpp"))
    sources += [
        path
        for directory in include_dirs
        for path in Path(directory).rglob("*")
        if path.suffix in {".h", ".hpp"} and path.is_file()
    ]
    write_record(
        artifact, kind="bridge", tool=identity(toolchain, "xsim"), sources=sources, arguments=cmd
    )


def ensure_built(toolchain: Toolchain, verbose: bool = False, force: bool = False) -> Path:
    """Return the xsi.so for ``toolchain``, building it on first use; ``ToolUnavailable``
    when it selects no Vivado.

    Safe for concurrent callers (e.g. pytest-xdist workers): one builds, the
    others wait and reuse its result.
    """
    artifact = xsi_artifact_dir(toolchain) / "xsi.so"
    if not force and _usable(artifact, toolchain):
        return artifact
    errors = check_prerequisites(toolchain)
    if errors:
        raise RuntimeError(
            "RTL simulation needs the finn_xsi extension, which cannot be built here:\n  "
            + "\n  ".join(errors)
        )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact.parent / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if force or not _usable(artifact, toolchain):
            _compile(artifact, verbose, toolchain)
    return artifact


def build_xsi(toolchain: Toolchain, force: bool = False, verbose: bool = True) -> bool:
    """Build (or reuse) the finn_xsi extension for ``toolchain``; report failures instead
    of raising."""
    try:
        artifact = ensure_built(toolchain, verbose=verbose, force=force)
    except (RuntimeError, ToolUnavailable) as error:
        print(error)
        print("\nCommon issues:")
        print("  - Ensure Xilinx Vivado is properly sourced")
        print("  - Check that pybind11 is installed in your Python environment")
        print("  - Verify C++ compiler is installed")
        return False
    if verbose:
        print(f"finn_xsi is built: {artifact}")
    return True


def verify_installation(toolchain: Toolchain) -> bool:
    """Verify that finn_xsi, built for ``toolchain``, can be imported and works."""
    # Check if xsi.so exists
    xsi_so = find_xsi_so(toolchain)
    if xsi_so is None:
        print(f"\n✗ Compiled extension xsi.so not found in {xsi_artifact_dir(toolchain)}")
        return False

    # The adapter is installed normally; only the native artifact needs a path.
    sys.path.insert(0, str(xsi_so.parent))

    try:
        # Import the compiled C++ extension
        import xsi

        print("\n✓ xsi C++ extension module imports successfully")

        # Import the Python package
        import finn_xsi.adapter

        print("✓ finn_xsi.adapter imports successfully")

        # Check for basic functionality
        if hasattr(finn_xsi.adapter, "load_sim_obj"):
            print("✓ RTL simulation functions available")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import modules: {e}")
        return False
    finally:
        sys.path.pop(0)


def clean_build(toolchain: Toolchain) -> bool:
    """Clean the build artifacts of ``toolchain``'s Vivado."""
    # Only clean the explicitly writable artifact directory.
    removed = False
    directory = xsi_artifact_dir(toolchain)
    for xsi_so in (directory / "xsi.so", directory / "xsi.so.finn.json"):
        if not xsi_so.exists():
            continue
        try:
            xsi_so.unlink()
            print(f"Removed {xsi_so}")
            removed = True
        except Exception as e:
            print(f"Failed to remove {xsi_so}: {e}")
            return False

    if not removed:
        print("No artifacts to clean.")
    return True


def main() -> int:
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup FINN XSI (RTL simulation) support")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if already built")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts and exit")
    parser.add_argument("--check", action="store_true", help="Only check prerequisites")
    parser.add_argument("--quiet", action="store_true", help="Suppress build output")

    args = parser.parse_args()
    # A command a user runs: the machine's toolchain.
    toolchain = host_toolchain()

    # Clean and exit if requested
    if args.clean:
        try:
            cleaned = clean_build(toolchain)
        except ToolUnavailable as missing:
            print(missing)
            cleaned = False
        if cleaned:
            print("Clean completed successfully.")
            return 0
        else:
            print("Clean failed.")
            return 1

    # Check prerequisites
    if not args.quiet:
        print("Checking prerequisites...")
    errors = check_prerequisites(toolchain)

    if errors:
        print("Prerequisite check failed:")
        for error in errors:
            print(f"  ✗ {error}")
        print("Please resolve these issues and try again.")
        return 1

    if not args.quiet:
        print("✓ All prerequisites satisfied")

    if args.check:
        return 0

    # Build finn_xsi
    if not args.quiet:
        print("Building finn_xsi extension...")
    if not build_xsi(toolchain, force=args.force, verbose=not args.quiet):
        print("Build failed. Please check the error messages above.")
        return 1

    # Verify installation
    verification_result = verify_installation(toolchain) if not args.quiet else True

    if verification_result:
        if not args.quiet:
            print("\nFINN XSI setup completed successfully!")
        return 0
    else:
        print("\nSetup completed but verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
