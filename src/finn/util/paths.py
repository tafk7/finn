# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's two directories. Standard library only: the resource cache and docker/run use
them without FINN's dependencies."""
import os
from pathlib import Path


def home(environ=None) -> Path:
    """FINN's persistent per-user directory: FINN_HOME, by default ``~/.finn``.

    It holds content-keyed state that outlives builds: fetched resources
    (``resources/``, or FINN_RESOURCES_DIR), finn_xsi builds (``xsi/``, or
    FINN_XSI_PATH) and golden references (``golden/``, or FINN_GOLDEN_DIR).
    Deleting it only costs rebuilds and downloads.
    """
    env = os.environ if environ is None else environ
    return Path(env.get("FINN_HOME") or Path.home() / ".finn").expanduser()


def scratch(environ=None) -> Path:
    """FINN's build directory, where builds and tests allocate their working
    directories: FINN_BUILD_DIR, by default ``/tmp/finn_build_<uid>``. Nothing there
    is reused, so it may be wiped at any time. Absolute, and not created here."""
    env = os.environ if environ is None else environ
    value = env.get("FINN_BUILD_DIR")
    return Path(value).expanduser().resolve() if value else Path(f"/tmp/finn_build_{os.getuid()}")
