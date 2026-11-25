# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's CI image: a content-addressed tag, a Bake build under it, and
transport over shared storage.

The tag hashes the files the image is built from (INPUTS), the archives the
selected runtime manifests supply, FINN_RUNTIMES and UBUNTU_TAG. FINN's own
sources are not inputs: containers install the mounted checkout. One build
publishes the image to a shared directory; every other agent loads the same
bytes, verified, and runs them by immutable ID:

    FINN_IMAGE=$(python3 -m finn_ci image load DIR) docker/run --no-build -- ...
"""

import fcntl
import glob
import gzip
import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The image's inputs, relative to the checkout. The selected runtimes' supplied
# packages are added per runtime set (supplied_packages).
INPUTS = (
    ".dockerignore",
    "docker-bake.hcl",
    "docker/Dockerfile",
    "docker/entrypoint.sh",
    "docker/profile.sh",
    "docker/identity.py",
    "docker/install-runtimes.sh",
    "docker/runtimes/*.env",
    "packaging/finn-experimental/pyproject.toml",
    "packaging/finn-experimental/setup.py",
    "pyproject.toml",
    "uv.lock",
    ".uv-version",
    "src/finn/resources/*.py",
    "src/finn/resources.toml",
    "src/finn/util/paths.py",
)
GENERATED = re.compile(r"(^|/)__pycache__/|\.pyc$|\.so$|\.egg-info/")

SCHEMA = 2
FILES = ("finn-docker-tag.txt", "finn-image-digest.txt", "finn-image-provenance.json")
ARCHIVE = "finn-docker-image.tar.gz"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


class ImageError(Exception):
    """An image input, publication or Docker operation CI cannot proceed with."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_runtimes(runtimes):
    """The runtime set ("xrt,slash" or a list) in canonical, duplicate-free order."""
    if isinstance(runtimes, str):
        runtimes = runtimes.replace(",", " ").split()
    return sorted(set(name for name in runtimes if name))


# --------------------------------------------------------------------------
# Identity.
# --------------------------------------------------------------------------


def read_manifest(path, names):
    """Values of a runtime manifest, which is a shell declaration file."""
    script = 'unset %s; . "$1"; printf "%%s\\0" %s' % (
        " ".join(names),
        " ".join('"${%s:-}"' % name for name in names),
    )
    try:
        output = subprocess.check_output(["sh", "-e", "-c", script, "runtime-manifest", str(path)])
    except (OSError, subprocess.CalledProcessError) as error:
        raise ImageError("cannot read runtime manifest %s: %s" % (path, error))
    return dict(zip(names, output.decode().split("\0")))


def supplied_packages(runtimes, root=REPO):
    """The archives in docker/packages that the selected `SOURCE=supply` manifests install."""
    root = Path(root)
    packages = set()
    for name in normalize_runtimes(runtimes):
        manifest = root / "docker/runtimes" / (name + ".env")
        values = read_manifest(manifest, ("SOURCE", "FILES"))
        if values["SOURCE"] == "fetch":
            continue
        if values["SOURCE"] != "supply" or not values["FILES"].split():
            raise ImageError("invalid runtime SOURCE/FILES in %s" % manifest)
        for filename in values["FILES"].split():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ImageError("runtime package must be relative to docker/packages: " + filename)
            path = Path("docker/packages") / relative
            if not (root / path).is_file():
                raise ImageError("runtime %r requires readable package %s" % (name, root / path))
            packages.add(path.as_posix())
    return sorted(packages)


def revision(runtimes, root=REPO, inputs=INPUTS):
    """16 hex digits of a hash over the image inputs, the selected supplied
    packages, FINN_RUNTIMES and UBUNTU_TAG (empty: Bake's default, which
    docker-bake.hcl holds).

    Git records the owner's executable bit, not agent umasks, so modes are
    canonical and the hash is stable across Jenkins agents.
    """
    root = Path(root)
    runtimes = normalize_runtimes(runtimes)
    digest = hashlib.sha256(b"finn-ci-image-v1\n")
    for pattern in list(inputs) + supplied_packages(runtimes, root):
        matches = glob.glob(os.path.join(glob.escape(str(root)), pattern))
        paths = sorted(os.path.relpath(match, str(root)) for match in matches)
        paths = [p for p in paths if (root / p).is_file() and not GENERATED.search(p)]
        if not paths:
            raise ImageError("image input matched no files: " + pattern)
        for path in paths:
            mode = "100755" if (root / path).stat().st_mode & 0o100 else "100644"
            try:
                checksum = sha256(root / path)
            except OSError as error:
                raise ImageError("cannot read image input %s: %s" % (path, error))
            digest.update(("path=%s mode=%s sha256=%s\n" % (path, mode, checksum)).encode())
    digest.update(("arg=FINN_RUNTIMES=%s\n" % ",".join(runtimes)).encode())
    digest.update(("arg=UBUNTU_TAG=%s\n" % os.environ.get("UBUNTU_TAG", "")).encode())
    return digest.hexdigest()[:16]


def tag(runtimes, root=REPO):
    """The image's tag: finn:ci-<revision>, with the runtimes as Bake names them."""
    runtimes = normalize_runtimes(runtimes)
    suffix = "".join("-" + name for name in runtimes)
    return "finn:ci-%s%s" % (revision(runtimes, root), suffix)


# --------------------------------------------------------------------------
# Docker.
# --------------------------------------------------------------------------


def docker(*args):
    """Run the Docker CLI, keeping stdout available for machine data."""
    return subprocess.check_output(["docker", *args], text=True).strip()


def inspect(reference):
    """The daemon's description of an image, or None if it has none."""
    result = subprocess.run(
        ["docker", "image", "inspect", reference], capture_output=True, text=True
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def image_id(reference):
    inspected = inspect(reference)
    if inspected is None:
        raise ImageError("no Docker image " + reference)
    if not IMAGE_ID.fullmatch(inspected["Id"]):
        raise ImageError("invalid Docker image ID: " + inspected["Id"])
    return inspected["Id"]


def build(runtimes, if_missing=False):
    """Build Bake's `finn` target under the image's tag; return its ID.

    With if_missing, an image already under the tag is used as it is.
    """
    runtimes = normalize_runtimes(runtimes)
    reference = tag(runtimes)
    if if_missing and inspect(reference) is not None:
        return image_id(reference)
    argv = ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--load"]
    argv += ["--set", "finn.tags=" + reference, "finn"]
    print("finn_ci image: building %s" % reference, file=sys.stderr)
    try:
        # Bake's output goes to stderr: stdout is the image ID.
        result = subprocess.run(
            argv,
            cwd=str(REPO),
            env=dict(os.environ, FINN_RUNTIMES=",".join(runtimes)),
            stdout=sys.stderr,
        )
    except FileNotFoundError:
        result = None
    if result is None or result.returncode:
        raise ImageError(
            "docker buildx bake finn failed. Docker Buildx with Bake is required; install "
            "docker-buildx-plugin and check 'docker buildx bake --help'."
        )
    return image_id(reference)


# --------------------------------------------------------------------------
# Metadata.
# --------------------------------------------------------------------------


def git(*args):
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def image_metadata(reference, runtimes):
    """Describe the image's bytes, its inputs and the source it was built from."""
    runtimes = normalize_runtimes(runtimes)
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "schema": SCHEMA,
        "image_id": image_id(reference),
        "environment_revision": revision(runtimes),
        "lock_sha256": sha256(REPO / "uv.lock"),
        "resources_sha256": sha256(REPO / "src/finn/resources.toml"),
        "source_revision": git("rev-parse", "HEAD") or "unknown",
        "source_dirty": None if status is None else bool(status),
        "runtimes": runtimes,
        "ci": {
            key: os.environ[key]
            for key in ("JOB_NAME", "BUILD_NUMBER", "BUILD_URL")
            if key in os.environ
        },
    }


def write_metadata(dest, reference, info):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / FILES[0]).write_text(reference + "\n")
    (dest / FILES[1]).write_text(info["image_id"] + "\n")
    (dest / FILES[2]).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def read_metadata(directory, runtimes):
    """Fail closed on missing, partial, stale or foreign publications."""
    directory = Path(directory)
    runtimes = normalize_runtimes(runtimes)
    try:
        info = json.loads((directory / FILES[2]).read_text())
        published_id = (directory / FILES[1]).read_text().strip()
        reference = (directory / FILES[0]).read_text().strip()
    except (OSError, ValueError) as error:
        raise ImageError(
            "incomplete image publication at %s: %s. Publish it again with "
            "`python3 -m finn_ci image publish`." % (directory, error)
        )
    if (
        info.get("schema") != SCHEMA
        or not IMAGE_ID.fullmatch(published_id)
        or info.get("image_id") != published_id
    ):
        raise ImageError("malformed or mismatched image identity at %s" % directory)
    if not reference or "\n" in reference:
        raise ImageError("malformed published image reference at %s" % directory)
    for key, expected in (
        ("environment_revision", revision(runtimes)),
        ("lock_sha256", sha256(REPO / "uv.lock")),
        ("runtimes", runtimes),
    ):
        if info.get(key) != expected:
            raise ImageError("published %s=%r; requested %r" % (key, info.get(key), expected))
    return published_id, reference, info


# --------------------------------------------------------------------------
# Publication.
# --------------------------------------------------------------------------


def blocks(stream):
    """A pipe's bytes as they arrive. A blocking read would hold a SIGTERM that
    lands between two reads until the next data comes; select's timeout returns
    to Python, which then runs the handler."""
    fd = stream.fileno()
    while True:
        if not select.select([fd], [], [], 1)[0]:
            continue
        block = os.read(fd, 1 << 20)
        if not block:
            return
        yield block


def compress(source, output):
    """gzip a stream into output, hashing the compressed bytes as they are written.

    pigz uses every core; without it, Python's gzip is a slower fallback.
    """
    digest = hashlib.sha256()
    pigz = shutil.which("pigz")
    if pigz:
        process = subprocess.Popen(
            [pigz, "-1", "-n", "-p", str(os.cpu_count() or 1)], stdin=source, stdout=subprocess.PIPE
        )
        try:
            for block in blocks(process.stdout):
                digest.update(block)
                output.write(block)
        except BaseException:
            # pigz's own SIGTERM cleanup can stall with threads running; it
            # holds nothing worth keeping.
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if process.wait():
            raise RuntimeError("pigz failed")
        return digest.hexdigest()

    class Hashing:
        def write(self, block):
            digest.update(block)
            return output.write(block)

        def flush(self):
            output.flush()

    with gzip.GzipFile(fileobj=Hashing(), mode="wb", compresslevel=1, mtime=0) as compressed:
        for block in blocks(source):
            compressed.write(block)
    return digest.hexdigest()


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def publish(directory, reference, workspace, runtimes):
    """Publish the image as one complete directory, by rename; copy its metadata
    to workspace. A retry reuses an intact publication and cannot replace it."""
    directory = Path(directory)
    runtimes = normalize_runtimes(runtimes)
    info = image_metadata(reference, runtimes)
    directory.parent.mkdir(parents=True, exist_ok=True)

    def previous_publication():
        previous_id, previous_ref, previous = read_metadata(directory, runtimes)
        if (
            previous_id != info["image_id"]
            or previous["source_revision"] != info["source_revision"]
        ):
            raise ImageError(
                "build directory already contains another image/source: %s" % directory
            )
        # Shards load the archive, so a retry must not accept a damaged one.
        try:
            intact = sha256(directory / ARCHIVE) == previous.get("archive_sha256")
        except OSError:
            intact = False
        if not intact:
            raise ImageError(
                "existing publication at %s has a missing or damaged archive; "
                "remove the directory and publish again" % directory
            )
        write_metadata(workspace, previous_ref, previous)

    if directory.is_dir() and any(directory.iterdir()):
        return previous_publication()
    with tempfile.TemporaryDirectory(dir=str(directory.parent), prefix=".finn-image-") as work:
        prepared = Path(work) / "publication"
        prepared.mkdir()
        # A private tag keeps docker save's tag in the archive without racing a
        # rebuild that moves the human-facing tag during the save.
        transport = "finn-transport:" + uuid.uuid4().hex
        docker("tag", info["image_id"], transport)
        try:
            with (prepared / ARCHIVE).open("wb") as output:
                process = subprocess.Popen(["docker", "save", transport], stdout=subprocess.PIPE)
                try:
                    info["archive_sha256"] = compress(process.stdout, output)
                except BaseException:
                    stop(process)
                    raise
                finally:
                    process.stdout.close()
                if process.wait():
                    raise RuntimeError("docker save failed")
                output.flush()
                os.fsync(output.fileno())
        finally:
            docker("image", "rm", transport)
        info["requested_reference"] = reference
        info["transport_reference"] = transport
        write_metadata(prepared, transport, info)
        for file in prepared.iterdir():
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
        write_metadata(workspace, transport, info)
        try:
            prepared.rename(directory)
        except OSError:
            # A concurrent publisher won. The workspace metadata follows the
            # archive actually published, its transport tag and checksum.
            previous_publication()


# --------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------


def _loaded(published_id):
    inspected = inspect(published_id)
    if inspected is None:
        return False
    # A matching tag is insufficient; the content-addressed ID must match.
    if inspected["Id"] != published_id:
        raise ImageError("loaded image does not match the published identity")
    return True


def _docker_load(archive, expected):
    """Stream an archive into docker load, checking its checksum on the way."""
    digest = hashlib.sha256()
    process = subprocess.Popen(["docker", "load"], stdin=subprocess.PIPE, stdout=sys.stderr)
    try:
        with open(archive, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                process.stdin.write(block)
        if digest.hexdigest() != expected:
            # Abort before docker load sees the end of the stream.
            stop(process)
            raise ImageError("published image archive checksum mismatch")
        process.stdin.close()
    except BaseException:
        if process.poll() is None:
            stop(process)
        raise
    if process.wait():
        raise RuntimeError("docker load failed")


def load(directory, runtimes):
    """Reuse verified image bytes, or load the archive once per host under a
    lock; tag the image finn:ci-<revision> and return its ID."""
    runtimes = normalize_runtimes(runtimes)
    published_id, _, info = read_metadata(directory, runtimes)
    archive = Path(directory) / ARCHIVE
    if not archive.is_file():
        raise ImageError("incomplete image publication: the archive is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", info.get("archive_sha256", "")):
        raise ImageError("malformed archive checksum metadata")
    if not _loaded(published_id):
        # Host-local, not an NFS lock: agents with their own daemons load in parallel.
        lock = os.environ.get("FINN_DOCKER_LOAD_LOCK", "/tmp/finn-docker-load.lock")
        with open(lock, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if not _loaded(published_id):
                _docker_load(archive, info["archive_sha256"])
                if not _loaded(published_id):
                    raise ImageError("loaded image does not match the published identity")
    docker("tag", published_id, tag(runtimes))
    return published_id
