"""External build data for hardware flows: finn-hlslib headers and Vivado board files.

finn-hlslib is the ``finn-hlslib`` package (``finn[hw]``); a FINN development
environment installs it editable from the ``packages/finn-hlslib`` submodule.

Board files come from several third-party repositories that FINN does not
redistribute. They are fetched from upstream at the pinned commits below on first
use, verified against a content digest, and cached. Images fetch them at build
time. Either location can be overridden with FINN_HLSLIB_PATH or
FINN_BOARD_FILES_PATH.

This module uses only the standard library so an image build can run it
directly, before FINN is installed:

    python src/finn/util/external.py fetch-boards --dest DIR
"""

import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# (repository, commit, directory in the repository, directory in the assembled tree).
# An empty path means the repository root. Changing any entry changes the content,
# so update BOARD_FILES_DIGEST with `python -m finn.util.external digest <dir>`.
BOARD_SOURCES = (
    ("https://github.com/Avnet/bdf.git", "2d49cfc25766f07792c0b314489f21fe916b639b", "", ""),
    (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        "boards/Xilinx/rfsoc2x2",
        "rfsoc2x2",
    ),
    (
        "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        "board_files/rfsoc4x2",
        "rfsoc4x2",
    ),
    (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        "boards/Xilinx/kv260_som",
        "kv260_som",
    ),
    (
        "https://github.com/RealDigitalOrg/aup-zu3-bsp.git",
        "b595ecdf37c7204129517de1773b0895bcdcc2ed",
        "board-files/aup-zu3-8gb",
        "aup-zu3-8gb",
    ),
)
BOARD_FILES_DIGEST = "89ebe8049a4ffcbdc010dd860c1303ec349cc2ed0efa40639addf3e504b6960e"
MARKER = ".finn-board-files"


def hlslib_path():
    """Return the finn-hlslib directory to pass to compilers with -I."""
    override = os.environ.get("FINN_HLSLIB_PATH")
    if override:
        return _existing(override, "FINN_HLSLIB_PATH")
    try:
        from finn_hlslib import include_dir  # noqa: PLC0415
    except ImportError:
        raise FileNotFoundError(
            "finn-hlslib is not installed. Install finn[hw], or set FINN_HLSLIB_PATH "
            "to a finn-hlslib checkout."
        ) from None
    return include_dir()


def board_files_path(fetch=True):
    """Return the board-file directory, fetching the pinned files on first use."""
    override = os.environ.get("FINN_BOARD_FILES_PATH")
    if override:
        return _existing(override, "FINN_BOARD_FILES_PATH")
    cache = _cache_root() / f"board-files-{BOARD_FILES_DIGEST[:16]}"
    if _complete(cache):
        return str(cache)
    if not fetch:
        raise FileNotFoundError(f"Board files have not been fetched into {cache}")
    return fetch_board_files(cache)


def fetch_board_files(dest, sources=None, digest=None):
    """Assemble the pinned board files into dest, verified, atomically."""
    sources = BOARD_SOURCES if sources is None else sources
    digest = BOARD_FILES_DIGEST if digest is None else digest
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f".{dest.name}.lock", "w") as lock:
        # Parallel workers wait for a single fetch instead of racing.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _complete(dest, digest):
            return str(dest)
        print(f"Fetching pinned board files into {dest}", file=sys.stderr)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}.") as work:
            tree = Path(work) / "tree"
            tree.mkdir()
            for number, (url, commit, subdir, target) in enumerate(sources):
                checkout = Path(work) / f"source{number}"
                _sparse_checkout(url, commit, subdir, checkout)
                shutil.copytree(
                    checkout / subdir if subdir else checkout,
                    tree / target if target else tree,
                    ignore=shutil.ignore_patterns(".git"),
                    dirs_exist_ok=True,
                )
            actual = tree_digest(tree)
            if actual != digest:
                raise RuntimeError(
                    f"Fetched board files have digest {actual}, expected {digest}. "
                    "If the pinned sources were changed on purpose, update "
                    "BOARD_FILES_DIGEST in finn/util/external.py."
                )
            (tree / MARKER).write_text(digest + "\n")
            if dest.exists():
                shutil.rmtree(dest)
            tree.rename(dest)
    return str(dest)


def tree_digest(path):
    """Content digest of a directory tree: relative paths and file contents."""
    path = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and p.name != MARKER):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _sparse_checkout(url, commit, subdir, checkout):
    def git(*args):
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)

    checkout.mkdir()
    git("init", "--quiet")
    git("remote", "add", "origin", url)
    if subdir:
        git("sparse-checkout", "set", subdir)
    # Fetch only the pinned commit; blobs outside the sparse paths are never downloaded.
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "--quiet", "FETCH_HEAD")


def _complete(path, digest=None):
    marker = Path(path) / MARKER
    return marker.is_file() and marker.read_text().strip() == (digest or BOARD_FILES_DIGEST)


def _cache_root():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "finn"


def _existing(path, variable):
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{variable}={path} is not a directory")
    return str(Path(path).resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description="FINN external build data")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-boards", help="fetch the pinned board files")
    fetch.add_argument("--dest", help="directory to assemble into (default: the user cache)")
    digest = commands.add_parser("digest", help="print the content digest of a directory")
    digest.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "digest":
        print(tree_digest(args.directory))
    elif args.dest:
        print(fetch_board_files(args.dest))
    else:
        print(board_files_path())


if __name__ == "__main__":
    main()
