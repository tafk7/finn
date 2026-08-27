"""Resolve FINN and its co-developed dependencies from the mounted workspace.

Installed into site-packages alongside ``finn-live.pth``, which imports this
module at interpreter startup. It replaces the four ``pip install -e`` calls the
container entrypoint used to run on every start: those cost ~33 s, reached out to
PyPI mid-start (which a closed-egress sandbox cannot allow), and mutated the
mounted workspace. What they actually produced was a four-line ``easy-install.pth``
naming exactly these directories, so this reproduces the mechanism without the
cost. The image carries the dependency closure, metadata and console scripts as
ordinary wheels; this module decides whether the workspace shadows them.

LIMITATION(finn-root-absolute)
------------------------------
This module exists because FINN has no fixed workspace path. ``run-docker.sh``
mirrors the host path (``-v $SCRIPTPATH:$SCRIPTPATH``) and sbx cannot remap a
mount at all - it attaches every mount at its host path by design - so the
workspace lives at a different absolute path for every developer and is only
knowable at run time.

A normal editable install cannot express that. Editable installs, from pip or
uv, record an *absolute* path (a ``.pth`` line, or a MAPPING in a PEP 660
``__editable___*_finder.py``). Baking one at image build time would hardcode the
building developer's home directory into a shared image - the same unsoundness
as the per-user images this redesign removed. So path resolution has to happen at
import time, from an environment variable, which is what this module does.

This is a known structural flaw in the repo and fixing it is out of scope here.
It is confined to five sites, all tagged ``LIMITATION(finn-root-absolute)``:

    docker/finn_paths.py        this module (workspace_root)
    docker/finn_entrypoint.sh   derives FINN_ROOT from $PWD
    run-docker.sh               the host-path-mirroring mount
    src/finn/util/basic.py      FINN_HLSLIB_PATH / FINN_BOARD_FILES_PATH
    src/finn/xsi/paths.py       finn_xsi source location

EXTENSION POINT
---------------
If the workspace ever gets a fixed path, this module becomes unnecessary rather
than merely simplified. The migration is:

    1. Bake ``pip install -e <fixed>/deps/<dep>`` (or uv equivalents) into the
       dev stage of docker/Dockerfile.finn, alongside the existing wheels.
    2. Delete this module and finn-live.pth, and drop the COPY/install of both.
    3. Keep FINN_DEPS as the live/frozen switch by selecting between the
       editable install and the wheel, rather than by prepending sys.path.

That would also fix the one wart this approach has: in live mode the *code*
comes from the workspace but the *declared version* comes from the baked wheel,
so importlib.metadata can report a commit that is not the one running. Real
editable installs report both consistently. Nothing in FINN reads that metadata
today, which is why the trade is currently worth making.

Runtime contract
----------------
``FINN_ROOT`` unset means do nothing at all, so the image stays usable with no
mount. ``FINN_DEPS`` selects live (default) or frozen; see workspace_root and
dep_src_dirs below.

This runs at the start of every Python process in the image, so it must stay
cheap and must never raise.
"""

import os
import sys
import tempfile

# Dependency source layout, relative to the workspace root.
#
# Kept in sync with deps.env's FINN_DEP_GROUP_PYTHON by a build-time assertion in
# docker/Dockerfile.finn, which fails the build on drift - so deps.env stays the
# single place a dependency is declared, while this list stays readable and this
# module stays usable outside the image.
DEP_SRC_DIRS = (
    "deps/qonnx/src",
    "deps/finn-experimental/src",
    "deps/brevitas/src",
)


def workspace_root():
    """The mounted FINN workspace, or None when there is no mount.

    Resolution order, first hit wins:

    ``FINN_ROOT``
        Explicit, and what run-docker.sh and finn_entrypoint.sh set.
    ``WORKSPACE_DIR``
        What sbx sets. This matters because ``sbx exec`` starts a process
        directly in the container and does NOT run the image ENTRYPOINT, so
        nothing finn_entrypoint.sh exports is present in an exec session.
        Without this fallback every such session fails at ``import finn``.
    cwd containing ``src/finn``
        Last resort, and only when it really looks like a FINN checkout, so a
        stray cwd cannot silently point the shadowing machinery at nonsense.

    FINN_ROOT is exported once resolved, so subprocesses FINN launches -
    Vivado, Vitis HLS, g++ - and the generated Tcl that reads
    ``$::env(FINN_ROOT)`` all agree with whatever Python resolved.

    LIMITATION(finn-root-absolute): this indirection is the whole reason this
    module exists. With a fixed workspace path it collapses to a constant and
    the module can be replaced by ordinary editable installs.
    """
    root = os.environ.get("FINN_ROOT") or os.environ.get("WORKSPACE_DIR")
    if not root:
        cwd = os.getcwd()
        if os.path.isdir(os.path.join(cwd, "src", "finn")):
            root = cwd
    if not root:
        return None
    os.environ.setdefault("FINN_ROOT", root)
    return root


def deps_are_live():
    """Whether the workspace shadows the baked dependency wheels.

    live (default) resolves qonnx, brevitas and finn-experimental from the
    workspace, so edits and branch switches take effect with no reinstall.
    frozen leaves them to the wheels, pinning them to the deps.env commits
    regardless of what is checked out.
    """
    return os.environ.get("FINN_DEPS", "live").lower() != "frozen"


def source_dirs():
    """Workspace directories to place ahead of site-packages, in order."""
    root = workspace_root()
    if not root:
        return []

    # FINN itself is never baked into the image, so it is live in both modes -
    # there is no wheel for frozen to fall back to, by design.
    relative = ["src"]
    if deps_are_live():
        relative.extend(DEP_SRC_DIRS)

    return [os.path.join(root, rel) for rel in relative]


def ensure_build_dir():
    """Guarantee ``FINN_BUILD_DIR`` exists for sessions the entrypoint missed.

    ``finn.util.basic`` reads it as a bare ``os.environ[...]``, so its absence
    is a ``KeyError`` at first use rather than a diagnosable error. The
    entrypoint defaults it, but ``sbx exec`` bypasses the ENTRYPOINT entirely,
    so every exec session would otherwise fail the moment a test touches a
    build directory.

    Scoped per-uid under /tmp, matching the entrypoint's default. An explicit
    value always wins, so run-docker.sh's mounted FINN_HOST_BUILD_DIR is
    untouched.
    """
    if os.environ.get("FINN_BUILD_DIR"):
        return
    build_dir = os.path.join(tempfile.gettempdir(), "finn_build_%d" % os.getuid())
    try:
        os.makedirs(build_dir, exist_ok=True)
    except OSError:
        return
    os.environ["FINN_BUILD_DIR"] = build_dir


def install():
    """Prepend the workspace source directories to ``sys.path``."""
    # Insert at the front so the workspace shadows the baked wheels, matching
    # what an editable install does. Reversed so the listed order is preserved.
    for path in reversed(source_dirs()):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    ensure_build_dir()


try:
    install()
except Exception:  # pragma: no cover - startup code must never break Python
    pass
