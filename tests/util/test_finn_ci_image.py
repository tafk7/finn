# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The CI image in ci/finn_ci/image.py: identity, Bake build and transport."""
import pytest

import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from finn_ci import __main__ as cli
from finn_ci import image
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGE_ID = "sha256:" + "a" * 64

pytestmark = pytest.mark.util


def has_bake():
    if not shutil.which("docker"):
        return False
    return (
        subprocess.run(["docker", "buildx", "bake", "--help"], capture_output=True).returncode == 0
    )


@pytest.fixture
def clean(monkeypatch):
    for name in ("FINN_RUNTIMES", "UBUNTU_TAG", "FINN_DOCKER_LOAD_LOCK"):
        monkeypatch.delenv(name, raising=False)


def image_root(tmp_path):
    root = tmp_path / "image-root"
    (root / "docker/runtimes").mkdir(parents=True)
    (root / "docker/runtimes/xrt.env").write_text("SOURCE=fetch\n")
    (root / "docker/runtimes/slashkit.env").write_text("SOURCE=supply\nFILES=slashkit.deb\n")
    (root / "image.txt").write_text("environment\n")
    (root / "src").mkdir()
    (root / "src/finn.py").write_text("source v1\n")
    return root


INPUTS = ("image.txt", "docker/runtimes/*.env")


def revision(root, runtimes=(), inputs=INPUTS):
    return image.revision(list(runtimes), root, inputs)


def fake_docker(tools, script):
    tools.mkdir(exist_ok=True)
    (tools / "docker").write_text(
        "#!%s\nimport json, os, pathlib, sys, time\n%s" % (sys.executable, script)
    )
    (tools / "docker").chmod(0o755)
    return str(tools) + os.pathsep + os.environ["PATH"]


# --------------------------------------------------------------------------
# Identity.
# --------------------------------------------------------------------------


def test_revision_tracks_inputs_runtimes_and_build_args_not_sources(clean, tmp_path, monkeypatch):
    root = image_root(tmp_path)
    original = revision(root)
    assert re.fullmatch(r"[0-9a-f]{16}", original)
    (root / "src/finn.py").write_text("source v2\n")
    assert revision(root) == original  # the checkout is mounted, not built in
    assert revision(root, ["xrt"]) != original
    (root / "image.txt").write_text("changed environment\n")
    changed = revision(root)
    assert changed != original
    monkeypatch.setenv("UBUNTU_TAG", "jammy-override")
    assert revision(root) != changed


def test_revision_ignores_umask_but_tracks_the_executable_bit(clean, tmp_path):
    root = image_root(tmp_path)
    source = root / "image.txt"
    source.chmod(0o644)
    original = revision(root)
    source.chmod(0o660)
    assert revision(root) == original
    source.chmod(0o755)
    assert revision(root) != original


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 fixtures")
def test_unreadable_or_missing_inputs_fail(clean, tmp_path):
    root = image_root(tmp_path)
    (root / "image.txt").chmod(0)
    try:
        with pytest.raises(image.ImageError, match="image.txt"):
            revision(root)
    finally:
        (root / "image.txt").chmod(0o644)
    with pytest.raises(image.ImageError, match="matched no files"):
        revision(root, inputs=INPUTS + ("missing/*.txt",))
    with pytest.raises(image.ImageError, match="runtime manifest"):
        revision(root, ["nonexistent"])


@pytest.mark.parametrize("runtimes", [(), ("xrt",)])
def test_only_selected_supplied_packages_change_identity(clean, tmp_path, runtimes):
    root = image_root(tmp_path)
    baseline = revision(root, runtimes)
    packages = root / "docker/packages"
    packages.mkdir()
    (packages / "slashkit.deb").write_bytes(b"leftover private package")
    (packages / "unused.rpm").write_bytes(b"unselected archive")
    (packages / "unused.rpm").chmod(0)
    assert revision(root, runtimes) == baseline
    selected = revision(root, ("slashkit", "xrt", "slashkit"))
    assert selected == revision(root, ("xrt", "slashkit"))
    (packages / "slashkit.deb").write_bytes(b"different selected package")
    assert revision(root, ("slashkit", "xrt")) != selected
    assert revision(root, runtimes) == baseline
    (packages / "slashkit.deb").unlink()
    with pytest.raises(image.ImageError, match="slashkit.deb"):
        revision(root, ("slashkit",))


def test_inputs_cover_the_dockerfile_context(clean):
    """Every file the Dockerfile takes from the build context changes the tag."""
    dockerfile = (REPO / "docker/Dockerfile").read_text().replace("\\\n", " ")
    sources = set(re.findall(r"--mount=type=bind,source=([^,\s]+)", dockerfile))
    for line in dockerfile.splitlines():
        words = line.split()
        if words[:1] == ["COPY"] and not any(w.startswith("--from") for w in words):
            sources.update(w for w in words[1:-1] if not w.startswith("--"))
    # The release stage's whole checkout is not an input of the development image;
    # supplied packages count only when a selected runtime names them.
    sources -= {".", "docker/packages"}
    assert sources
    for source in sorted(sources):
        if (REPO / source).is_dir():
            assert any(pattern.startswith(source + "/") for pattern in image.INPUTS), source
        else:
            assert source in image.INPUTS, source
    assert re.fullmatch(r"finn:ci-[0-9a-f]{16}-xrt", image.tag(["xrt"]))


def test_cli_and_library_identity_agree(clean, capsys, monkeypatch):
    """The committed tree hashes the same in-process and through the CLI."""
    proc = subprocess.run(
        [sys.executable, "-m", "finn_ci", "image", "tag", "--runtime", "xrt"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": str(REPO / "ci")},
        cwd="/",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == image.tag(["xrt"])
    monkeypatch.setenv("FINN_RUNTIMES", "xrt")
    assert cli.main(["image", "tag"]) == 0
    assert capsys.readouterr().out.strip() == image.tag(["xrt"])


# --------------------------------------------------------------------------
# Building with Bake.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not has_bake(), reason="needs Docker Buildx")
def test_bake_takes_the_tag_and_runtimes(clean):
    proc = subprocess.run(
        ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--print"]
        + ["--set", "finn.tags=finn:ci-0123456789abcdef-xrt", "finn"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=dict(os.environ, FINN_RUNTIMES="xrt"),
    )
    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout[proc.stdout.index("{") :])["target"]["finn"]
    assert printed["tags"] == ["finn:ci-0123456789abcdef-xrt"]
    assert printed["args"]["FINN_RUNTIMES"] == "xrt"


def test_build_bakes_the_finn_target_under_the_tag(clean, tmp_path, monkeypatch):
    built = tmp_path / "built"
    monkeypatch.setenv(
        "PATH",
        fake_docker(
            tmp_path / "bin",
            "built = pathlib.Path(%r)\n"
            'if sys.argv[1:3] == ["buildx", "version"]:\n'
            "    sys.exit(0)\n"
            'if sys.argv[1:3] == ["image", "inspect"]:\n'
            "    if not built.exists(): sys.exit(1)\n"
            "    print(json.dumps([{'Id': %r}]))\n"
            "else:\n"
            "    built.write_text(json.dumps([sys.argv[1:], os.environ['FINN_RUNTIMES']]))\n"
            % (str(built), IMAGE_ID),
        ),
    )
    reference = image.tag(["xrt"])
    assert image.build(["xrt"], if_missing=True) == IMAGE_ID
    argv, runtimes = json.loads(built.read_text())
    assert argv[:2] == ["buildx", "bake"]
    assert argv[-3:] == ["--set", "finn.tags=" + reference, "finn"]
    assert runtimes == "xrt"
    built.write_text("present")
    assert image.build(["xrt"], if_missing=True) == IMAGE_ID
    assert built.read_text() == "present"  # not rebuilt


def test_without_buildx_docker_build_builds_the_same_target(clean, tmp_path, monkeypatch):
    built = tmp_path / "built"
    monkeypatch.setenv(
        "PATH",
        fake_docker(
            tmp_path / "bin",
            "built = pathlib.Path(%r)\n"
            'if sys.argv[1] == "buildx":\n'
            '    sys.stderr.write("docker: unknown command: docker buildx\\n"); sys.exit(1)\n'
            'if sys.argv[1:3] == ["image", "inspect"]:\n'
            "    if not built.exists(): sys.exit(1)\n"
            "    print(json.dumps([{'Id': %r}]))\n"
            "else:\n"
            "    built.write_text(json.dumps([sys.argv[1:], os.environ['DOCKER_BUILDKIT']]))\n"
            % (str(built), IMAGE_ID),
        ),
    )
    reference = image.tag(["xrt"])
    assert image.build(["xrt"]) == IMAGE_ID
    argv, buildkit = json.loads(built.read_text())
    assert buildkit == "1"
    assert argv == image.docker_build_argv(reference, ["xrt"])[1:]
    assert argv[:6] == ["build", "-f", "docker/Dockerfile", "--target", "finn", "--platform"]
    assert argv[argv.index("--tag") + 1] == reference and argv[-1] == "."
    args = [argv[i + 1] for i, a in enumerate(argv) if a == "--build-arg"]
    assert args == [
        "UBUNTU_TAG=" + image.bake_default("UBUNTU_TAG"),
        "FINN_RUNTIMES=xrt",
        "FINN_VERSION=",
    ]
    monkeypatch.setenv("UBUNTU_TAG", "noble-20250101")
    assert "UBUNTU_TAG=noble-20250101" in image.docker_build_argv(reference, ["xrt"])


@pytest.mark.skipif(not has_bake(), reason="needs Docker Buildx")
def test_docker_build_matches_bakes_finn_target(clean):
    """The fallback builds what Bake would: Dockerfile, target, platform, args, labels."""
    reference = "finn:ci-0123456789abcdef-slash-xrt"
    proc = subprocess.run(
        ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--print"]
        + ["--set", "finn.tags=" + reference, "finn"],
        cwd=REPO,
        capture_output=True,
        text=True,
        # build() hands Bake the runtimes normalized, as it does here.
        env=dict(os.environ, FINN_RUNTIMES="slash,xrt"),
    )
    assert proc.returncode == 0, proc.stderr
    baked = json.loads(proc.stdout[proc.stdout.index("{") :])["target"]["finn"]
    argv = image.docker_build_argv(reference, ["xrt", "slash"])

    def values(flag):
        return dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == flag)

    assert argv[argv.index("-f") + 1] == baked["dockerfile"]
    assert argv[argv.index("--target") + 1] == baked["target"]
    assert [argv[argv.index("--platform") + 1]] == baked["platforms"]
    assert [argv[argv.index("--tag") + 1]] == baked["tags"]
    assert values("--build-arg") == baked["args"]
    assert values("--label") == baked["labels"]


def test_missing_bake_has_actionable_error(clean, tmp_path):
    path = fake_docker(
        tmp_path / "bin", 'sys.stderr.write("docker: buildx unavailable\\n")\nsys.exit(1)\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "finn_ci", "image", "build"],
        capture_output=True,
        text=True,
        env={"PATH": path, "PYTHONPATH": str(REPO / "ci")},
    )
    assert proc.returncode == 1
    assert "docker-buildx-plugin" in proc.stderr
    assert "Traceback" not in proc.stderr


# --------------------------------------------------------------------------
# Publication and loading.
# --------------------------------------------------------------------------


@pytest.fixture
def fixed(clean, monkeypatch):
    """A fixed image revision, independent of the checkout's inputs."""
    monkeypatch.setattr(image, "revision", lambda runtimes, root=None, inputs=None: "review")


def publication(directory, archive=b"archive", **changes):
    info = {
        "schema": image.SCHEMA,
        "image_id": IMAGE_ID,
        "environment_revision": "review",
        "lock_sha256": image.sha256(REPO / "uv.lock"),
        "runtimes": [],
        "source_revision": "a" * 40,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    info.update(changes)
    image.write_metadata(directory, "descriptive-tag", info)
    (directory / image.ARCHIVE).write_bytes(archive)
    return info


def test_metadata_rejects_partial_stale_and_foreign_publications(fixed, tmp_path):
    directory = tmp_path / "publication"
    publication(directory)
    assert image.read_metadata(directory, [])[0] == IMAGE_ID
    with pytest.raises(image.ImageError, match="published runtimes"):
        image.read_metadata(directory, ["xrt"])
    publication(directory, environment_revision="another-environment")
    with pytest.raises(image.ImageError, match="published environment_revision"):
        image.read_metadata(directory, [])
    publication(directory, schema=1)
    with pytest.raises(image.ImageError, match="identity"):
        image.read_metadata(directory, [])
    publication(directory)
    (directory / image.FILES[1]).write_text("sha256:" + "c" * 64)
    with pytest.raises(image.ImageError, match="identity"):
        image.read_metadata(directory, [])
    (directory / image.FILES[2]).unlink()
    with pytest.raises(image.ImageError, match="incomplete image publication"):
        image.read_metadata(directory, [])


def test_republication_accepts_only_an_intact_archive(fixed, tmp_path, monkeypatch):
    """A retry reuses the existing publication only if shards can load it."""
    directory = tmp_path / "publication"
    info = publication(directory)
    monkeypatch.setattr(image, "image_metadata", lambda reference, runtimes: dict(info))
    monkeypatch.setattr(image, "docker", lambda *args: pytest.fail("republished"))
    workspace = tmp_path / "workspace"
    image.publish(directory, IMAGE_ID, workspace, [])
    assert json.loads((workspace / image.FILES[2]).read_text()) == info
    shutil.rmtree(workspace)
    (directory / image.ARCHIVE).write_bytes(b"truncated")
    with pytest.raises(image.ImageError, match="missing or damaged archive"):
        image.publish(directory, IMAGE_ID, workspace, [])
    (directory / image.ARCHIVE).unlink()
    with pytest.raises(image.ImageError, match="missing or damaged archive"):
        image.publish(directory, IMAGE_ID, workspace, [])
    assert not workspace.exists()


def test_cached_image_avoids_archive_read_and_is_tagged(fixed, tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    publication(directory)
    calls = []
    monkeypatch.setattr(image, "_loaded", lambda published: True)
    monkeypatch.setattr(image, "_docker_load", lambda *args: pytest.fail("loaded"))
    monkeypatch.setattr(image, "docker", lambda *args: calls.append(args))
    assert image.load(directory, []) == IMAGE_ID
    assert calls == [("tag", IMAGE_ID, "finn:ci-review")]
    (directory / image.ARCHIVE).unlink()
    with pytest.raises(image.ImageError, match="archive is missing"):
        image.load(directory, [])


def test_concurrent_misses_load_archive_once(fixed, tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    publication(directory)
    monkeypatch.setenv("FINN_DOCKER_LOAD_LOCK", str(tmp_path / "load.lock"))
    monkeypatch.setattr(image, "docker", lambda *args: None)
    state = {"loaded": False, "loads": 0}
    monkeypatch.setattr(image, "_loaded", lambda published: state["loaded"])

    def load_once(archive, expected):
        state["loads"] += 1
        time.sleep(0.05)
        state["loaded"] = True

    monkeypatch.setattr(image, "_docker_load", load_once)
    with ThreadPoolExecutor(max_workers=6) as executor:
        loads = list(executor.map(lambda _: image.load(directory, []), range(6)))
    assert loads == [IMAGE_ID] * 6
    assert state["loads"] == 1


def test_load_streams_and_verifies_the_archive(tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    directory.mkdir()
    (directory / image.ARCHIVE).write_bytes(b"archive bytes")
    received = tmp_path / "received"
    monkeypatch.setenv(
        "PATH",
        fake_docker(
            tmp_path / "bin",
            "pathlib.Path(%r).write_bytes(sys.stdin.buffer.read())\n" % str(received),
        ),
    )
    image._docker_load(directory / image.ARCHIVE, hashlib.sha256(b"archive bytes").hexdigest())
    assert received.read_bytes() == b"archive bytes"
    received.unlink()
    with pytest.raises(image.ImageError, match="checksum mismatch"):
        image._docker_load(directory / image.ARCHIVE, "0" * 64)


@pytest.mark.parametrize("pigz", [True, False])
def test_compression_is_gzip_and_hashed_once(tmp_path, monkeypatch, pigz):
    if pigz and not shutil.which("pigz"):
        pytest.skip("pigz is not installed")
    if not pigz:
        monkeypatch.setattr(image.shutil, "which", lambda name: None)
    payload = os.urandom(3 << 20) + b"x" * (5 << 20)
    source = tmp_path / "payload"
    source.write_bytes(payload)
    with source.open("rb") as handle, (tmp_path / "out.gz").open("wb") as output:
        digest = image.compress(handle, output)
    assert digest == image.sha256(tmp_path / "out.gz")
    assert gzip.decompress((tmp_path / "out.gz").read_bytes()) == payload


def test_publication_sigterm_cleans_partial_archive(clean, tmp_path):
    path = fake_docker(
        tmp_path / "bin",
        "args = sys.argv[1:]\n"
        'if args[:2] == ["image", "inspect"]: print(json.dumps([{"Id": %r}]))\n' % IMAGE_ID
        + 'elif args[:1] == ["save"]:\n'
        "    sys.stdout.buffer.write(os.urandom(2000000)); sys.stdout.flush(); time.sleep(60)\n"
        'elif args[:2] == ["image", "rm"]: pathlib.Path(%r).touch()\n'
        % str(tmp_path / "tag-removed"),
    )
    target = tmp_path / "images/job/1"
    process = subprocess.Popen(
        [sys.executable, "-m", "finn_ci", "image", "publish", str(target), IMAGE_ID]
        + ["--workspace", str(tmp_path / "workspace")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": path, "HOME": str(tmp_path), "PYTHONPATH": str(REPO / "ci")},
    )
    try:
        for _ in range(200):
            partials = list(target.parent.glob(".finn-image-*/publication/" + image.ARCHIVE))
            if partials and partials[0].stat().st_size > 0:
                break
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            time.sleep(0.05)
        else:
            pytest.fail("Publication did not start")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 143, (stdout, stderr)
        assert not list(target.parent.glob(".finn-image-*"))
        assert (tmp_path / "tag-removed").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_TRANSPORT") != "1", reason="opt-in Docker transport"
)
def test_transport_uses_published_identity_and_rejects_corruption(clean, tmp_path, monkeypatch):
    """Real save/load, a stale tag, concurrent loads and publication retries."""
    key = uuid.uuid4().hex
    monkeypatch.setattr(
        image, "revision", lambda runtimes, root=None, inputs=None: "fixture-" + key
    )
    tag = "finn-transport-test:" + key
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\nCOPY marker /marker\n")
    publication = tmp_path / "shared"
    workspace = tmp_path / "workspace"
    created = []
    published_tag = None

    def build(text):
        (context / "marker").write_text(text)
        subprocess.run(
            ["docker", "build", "-t", tag, str(context)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        identity = image.docker("image", "inspect", "--format", "{{.Id}}", tag)
        created.append(identity)
        return identity

    try:
        original = build("original artifact\n")
        image.publish(publication, tag, workspace, [])
        published_tag = (publication / image.FILES[0]).read_text().strip()
        replacement = build("stale local artifact\n")
        assert original != replacement
        # Remove the published identity, then create a same-tag collision.
        # Loads must recover the archive's original bytes.
        subprocess.run(["docker", "image", "rm", original], capture_output=True)
        image.docker("tag", replacement, published_tag)
        with ThreadPoolExecutor(max_workers=2) as executor:
            loads = list(executor.map(lambda _: image.load(publication, []), range(2)))
        assert loads == [original] * 2
        assert image.docker("image", "inspect", "--format", "{{.Id}}", published_tag) == original
        assert image.docker("image", "inspect", "--format", "{{.Id}}", image.tag([])) == original
        assert image.docker("image", "inspect", "--format", "{{.Id}}", tag) == replacement
        image.publish(publication, original, workspace, [])
        assert json.loads((workspace / image.FILES[2]).read_text()) == json.loads(
            (publication / image.FILES[2]).read_text()
        )
        with pytest.raises(image.ImageError, match="another image/source"):
            image.publish(publication, replacement, workspace, [])
        corrupt = tmp_path / "corrupt"
        shutil.copytree(publication, corrupt)
        with (corrupt / image.ARCHIVE).open("ab") as handle:
            handle.write(b"corrupted")
        # Reuse is safe by immutable ID; archive bytes matter only on a cache miss.
        assert image.load(corrupt, []) == original
        image.docker("image", "rm", published_tag, image.tag([]))
        with pytest.raises(image.ImageError, match="checksum mismatch"):
            image.load(corrupt, [])
        (corrupt / image.FILES[1]).unlink()
        with pytest.raises(image.ImageError, match="incomplete image publication"):
            image.load(corrupt, [])
    finally:
        for reference in [tag, published_tag, image.tag([]), *created]:
            if reference:
                subprocess.run(["docker", "image", "rm", reference], capture_output=True)
