############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Path policy for the finn_xsi extension.

``$FINN_ROOT/finn_xsi`` holds two different kinds of thing: the C++ sources and
the ``finn_xsi`` Python package (both source, both under version control), and
historically also ``xsi.so`` (a build artifact).

Writing the artifact back into the source tree makes the checkout stateful: a
mounted workspace gets mutated by merely starting a container, two containers
racing the same tree can interleave writes, and the artifact outlives the Vivado
version it was compiled against. So the artifact now lands in a build directory
instead, and only the sources stay in the workspace.

A pre-existing ``xsi.so`` in the source tree is still honoured, so checkouts
built before this split keep working without a forced rebuild.
"""

import os
from pathlib import Path
from typing import Optional


def xsi_source_dir() -> Path:
    """Directory holding the finn_xsi C++ sources and Python package.

    LIMITATION(finn-root-absolute): reached through FINN_ROOT because the
    workspace has no fixed path. See docker/finn_paths.py.
    """
    return Path(os.environ["FINN_ROOT"]) / "finn_xsi"


def xsi_artifact_dir() -> Path:
    """Directory the compiled ``xsi.so`` is written to and loaded from.

    Defaults under ``$FINN_BUILD_DIR`` so the artifact is scoped to the same
    build tree as everything else FINN generates, and never to the workspace.
    """
    override = os.environ.get("FINN_XSI_BUILD_DIR")
    if override:
        return Path(override)
    build_dir = os.environ.get("FINN_BUILD_DIR")
    if build_dir:
        return Path(build_dir) / "finn_xsi"
    return Path("/tmp/finn_xsi")


def find_xsi_so() -> Optional[Path]:
    """Return the usable ``xsi.so``, or None if it has not been built.

    The artifact directory wins over the legacy in-tree location, so a stale
    ``xsi.so`` left in a workspace by an older FINN cannot shadow a freshly
    built one.
    """
    for candidate in (xsi_artifact_dir() / "xsi.so", xsi_source_dir() / "xsi.so"):
        if candidate.exists():
            return candidate
    return None
