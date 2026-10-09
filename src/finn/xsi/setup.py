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
module only builds it ahead of time.
"""

import argparse
import fcntl
import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple

from finn.util.toolchain import Selection, run_process
from finn.xsi._artifacts import validate_record, write_record
from finn.xsi.paths import find_xsi_so, xsi_artifact_dir, xsi_source_dir


def get_build_paths() -> Tuple[List[str], str, List[str]]:
    """Get include paths and compiler for building the extension.

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

    # Get Xilinx Vivado include directory
    xilinx_vivado = os.environ.get("XILINX_VIVADO")
    if xilinx_vivado:
        xilinx_include = os.path.join(xilinx_vivado, "data", "xsim", "include")
        if os.path.exists(xilinx_include):
            include_dirs.append(xilinx_include)

    # Determine compiler
    compiler = "g++"
    if not shutil.which("g++") and shutil.which("clang++"):
        compiler = "clang++"

    # Compile flags
    extra_compile_args = ["--std=c++17", "-Wall", "-O3", "-shared", "-fPIC"]

    return include_dirs, compiler, extra_compile_args


def check_prerequisites() -> List[str]:
    """Check if required tools are available."""
    errors = []

    # Check for C++ compiler
    if not shutil.which("g++") and not shutil.which("clang++"):
        errors.append("No C++ compiler found. Please install g++ or clang++.")

    # Check for Xilinx tools
    if not shutil.which("vivado"):
        errors.append("'vivado' not found. Ensure Xilinx tools are in PATH.")

    # Check Xilinx Vivado environment variable
    xilinx_vivado = os.environ.get("XILINX_VIVADO")
    if not xilinx_vivado:
        errors.append("XILINX_VIVADO environment variable not set. Please source Vivado settings.")
    elif not os.path.exists(os.path.join(xilinx_vivado, "data", "xsim", "include")):
        errors.append(f"Xilinx XSim headers not found at {xilinx_vivado}/data/xsim/include")

    # Check for pybind11
    try:
        import pybind11
    except ImportError:
        errors.append("pybind11 not found. Please install it with: pip install pybind11")

    return errors


SOURCE_FILES = ["xsi_bind.cpp", "xsi_finn.cpp"]


def bridge_identity() -> str:
    """The toolchain an xsi.so is built against: the selected Vivado installation."""
    vivado = os.environ.get("XILINX_VIVADO", "")
    return json.dumps({"installation": os.path.realpath(vivado) if vivado else ""})


def _usable(artifact: Path) -> bool:
    if not artifact.is_file():
        return False
    try:
        validate_record(artifact, kind="bridge", tool=bridge_identity())
    except (OSError, ValueError, KeyError):
        return False
    return True


def _compile(artifact: Path, verbose: bool) -> None:
    """Compile xsi.so to artifact, atomically, and record its inputs."""
    xsi_path = xsi_source_dir()
    toolchain = Selection().prepare()
    include_dirs, compiler, compile_args = get_build_paths()
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
    write_record(artifact, kind="bridge", tool=bridge_identity(), sources=sources, arguments=cmd)


def ensure_built(verbose: bool = False, force: bool = False) -> Path:
    """Return the xsi.so for the selected toolchain, building it on first use.

    Safe for concurrent callers (e.g. pytest-xdist workers): one builds, the
    others wait and reuse its result.
    """
    artifact = xsi_artifact_dir() / "xsi.so"
    if not force and _usable(artifact):
        return artifact
    errors = check_prerequisites()
    if errors:
        raise RuntimeError(
            "RTL simulation needs the finn_xsi extension, which cannot be built here:\n  "
            + "\n  ".join(errors)
        )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact.parent / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if force or not _usable(artifact):
            _compile(artifact, verbose)
    return artifact


def build_xsi(force: bool = False, verbose: bool = True) -> bool:
    """Build (or reuse) the finn_xsi extension; report failures instead of raising."""
    try:
        artifact = ensure_built(verbose=verbose, force=force)
    except RuntimeError as error:
        print(error)
        print("\nCommon issues:")
        print("  - Ensure Xilinx Vivado is properly sourced")
        print("  - Check that pybind11 is installed in your Python environment")
        print("  - Verify C++ compiler is installed")
        return False
    if verbose:
        print(f"finn_xsi is built: {artifact}")
    return True


def verify_installation() -> bool:
    """Verify that finn_xsi can be imported and works."""
    # Check if xsi.so exists
    xsi_so = find_xsi_so()
    if xsi_so is None:
        print(f"\n✗ Compiled extension xsi.so not found in {xsi_artifact_dir()}")
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


def clean_build() -> bool:
    """Clean build artifacts."""
    # Only clean the explicitly writable artifact directory.
    removed = False
    for xsi_so in (xsi_artifact_dir() / "xsi.so", xsi_artifact_dir() / "xsi.so.finn.json"):
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

    # Clean and exit if requested
    if args.clean:
        if clean_build():
            print("Clean completed successfully.")
            return 0
        else:
            print("Clean failed.")
            return 1

    # Check prerequisites
    if not args.quiet:
        print("Checking prerequisites...")
    errors = check_prerequisites()

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
    if not build_xsi(force=args.force, verbose=not args.quiet):
        print("Build failed. Please check the error messages above.")
        return 1

    # Verify installation
    verification_result = verify_installation() if not args.quiet else True

    if verification_result:
        if not args.quiet:
            print("\nFINN XSI setup completed successfully!")
        return 0
    else:
        print("\nSetup completed but verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
