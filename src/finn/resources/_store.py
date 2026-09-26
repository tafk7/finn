"""Digest-keyed caches and the fetchers that fill them.

Each cache entry is ``<root>/<name>-<digest16>/`` with a marker file holding the
full digest; an entry without a matching marker is incomplete. A fetch builds
the tree in a temporary directory beside the entry, verifies its digest, writes
the marker and renames it into place, under a file lock, so a failed or
concurrent fetch never publishes a partial or unverified tree.
"""
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from ._declare import PREFIX, ResourceError

MARKER = ".finn-resource"
DEFAULT_SYSTEM_CACHE = "/opt/finn/resources"


def tree_digest(path):
    """Content digest of a directory tree: sorted relative paths and file hashes.

    Symbolic links contribute their target, not the content they point to. The
    top-level marker file is excluded.
    """
    root = Path(path)
    if not root.is_dir():
        raise ResourceError(f"{root} is not a directory")
    items = []
    for directory, dirs, files in os.walk(root):
        # os.walk lists symbolic links to directories in dirs without entering them.
        for name in files + [d for d in dirs if os.path.islink(os.path.join(directory, d))]:
            item = Path(directory, name)
            relative = item.relative_to(root).as_posix()
            if relative != MARKER:
                items.append((relative.split("/"), relative, item))
    digest = hashlib.sha256()
    for _, relative, item in sorted(items):
        digest.update(relative.encode() + b"\0")
        if item.is_symlink():
            digest.update(hashlib.sha256(b"symlink\0" + os.readlink(item).encode()).digest())
        else:
            digest.update(_file_sha256(item).digest())
    return "sha256:" + digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest


def roots():
    """Cache roots as (path, writable), in search order."""
    result = []
    site = os.environ.get(PREFIX + "CACHE")
    if site:
        result.append((Path(site), True))
    system = os.environ.get(PREFIX + "SYSTEM_CACHE") or DEFAULT_SYSTEM_CACHE
    if Path(system).is_dir():
        result.append((Path(system), False))
    user = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    result.append((user / "finn" / "resources", True))
    return result


def fetch_root():
    """The root new fetches go into: the first writable one."""
    return next(path for path, writable in roots() if writable)


def complete(entry, digest):
    marker = Path(entry) / MARKER
    try:
        return marker.read_text().strip() == digest
    except OSError:
        return False


def lookup(resource):
    """The first complete cached copy of a resource, or None."""
    for root, _ in roots():
        entry = root / resource.entry
        if complete(entry, resource.digest):
            return entry
    return None


def fetch(resource, root):
    """Fetch, verify and publish a resource into a cache root; return the entry."""
    root = Path(root)
    entry = root / resource.entry
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = open(root / f".{resource.entry}.lock", "w")
    except OSError as error:
        raise ResourceError(f"Cannot write to the resource cache {root}: {error}") from None
    with lock:
        # Parallel first uses wait for a single fetch instead of racing.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(entry, resource.digest):
            return entry
        print(f"finn: fetching resource {resource.name} from {resource.source}", file=sys.stderr)
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{resource.entry}.") as work:
            tree = assemble(resource, Path(work))
            actual = tree_digest(tree)
            if actual != resource.digest:
                raise ResourceError(
                    f"Resource {resource.name} from {resource.source} has digest {actual}, "
                    f"but {resource.origin} declares {resource.digest}. If the source was "
                    "changed on purpose, update the digest (`finn-resources update`)."
                )
            (tree / MARKER).write_text(resource.digest + "\n")
            if entry.exists():
                # A leftover without a valid marker, e.g. from an interrupted copy.
                shutil.rmtree(entry)
            tree.rename(entry)
    return entry


def assemble(resource, work):
    """Fetch a resource's source into work and lay out its tree, unverified."""
    if resource.git:
        source = git_checkout(resource.git, resource.commit, resource.subdir, work / "source")
    else:
        source = archive(resource.url, resource.sha256, work / "source")
    if resource.subdir:
        source = source / resource.subdir
        if not source.is_dir():
            raise ResourceError(f"{resource.subdir} is not a directory in {resource.source}")
    tree = work / "tree"
    shutil.copytree(
        source,
        tree / resource.into if resource.into else tree,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return tree


def git_checkout(url, commit, subdir, dest):
    """Check out one commit, fetching only the blobs under subdir; return the checkout."""
    if shutil.which("git") is None:
        raise ResourceError(f"git is required to fetch {url}")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(dest), *args], env=env, capture_output=True, text=True
        )
        if result.returncode:
            raise ResourceError(f"git {args[0]} failed for {url}: {result.stderr.strip()}")

    dest.mkdir(parents=True)
    git("init", "--quiet")
    git("remote", "add", "origin", url)
    if subdir:
        git("sparse-checkout", "set", subdir)
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "--quiet", "FETCH_HEAD")
    return dest


def archive(url, sha256, dest):
    """Download, check and safely extract an archive; return its root directory.

    An archive holding a single top-level directory has that directory as its
    root, as source archives usually do.
    """
    dest.mkdir(parents=True)
    download = dest / "archive"
    try:
        # urllib honours the usual proxy variables.
        with urllib.request.urlopen(url, timeout=60) as response, open(download, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as error:
        raise ResourceError(f"Cannot download {url}: {error}") from None
    actual = _file_sha256(download).hexdigest()
    if actual != sha256:
        raise ResourceError(f"{url} has sha256 {actual}, expected {sha256}")
    tree = dest / "tree"
    tree.mkdir()
    if tarfile.is_tarfile(download):
        if not hasattr(tarfile, "data_filter"):
            raise ResourceError("Extracting archives safely needs Python 3.11.4 or later")
        with tarfile.open(download) as tar:
            try:
                tar.extractall(tree, filter="data")
            except tarfile.FilterError as error:
                raise ResourceError(f"Unsafe archive {url}: {error}") from None
    elif zipfile.is_zipfile(download):
        with zipfile.ZipFile(download) as zip_file:
            for name in zip_file.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ResourceError(f"Unsafe archive {url}: member {name}")
            zip_file.extractall(tree)
    else:
        raise ResourceError(f"{url} is not a tar or zip archive")
    download.unlink()
    children = list(tree.iterdir())
    if len(children) == 1 and children[0].is_dir() and not children[0].is_symlink():
        return children[0]
    return tree
