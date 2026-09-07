"""Characterization tests for the Jenkins <-> container image transport.

These exercise the real shell scripts through ``subprocess`` with a temporary
``PATH`` holding fake ``docker`` and ``flock`` commands. Nothing here needs a
Docker daemon, an NFS mount or network access.

The point is to pin the *compatibility baseline* that Jenkins depends on --
exit codes and the essential message, not every line of output -- so the
loader's diagnostics stay free to improve.
"""

import pytest

import gzip
import os
import re
import subprocess
from pathlib import Path

pytestmark = pytest.mark.util

REPO = Path(__file__).resolve().parents[2]
LOADER = REPO / "ci/scripts/load-shared-image.sh"
JENKINSFILE = REPO / "ci/Jenkinsfile"

# Docker image IDs are `sha256:` plus 64 lowercase hex characters. Two distinct
# well-formed ones, so a mismatch test is a genuine mismatch and not a typo.
ID_A = "sha256:" + "a1" * 32
ID_B = "sha256:" + "b2" * 32

# The lock the loader has always used to serialise same-host loads. Jenkins
# agents run several shards at once and a concurrent multi-gigabyte
# `docker load` of the same archive is the reason it exists.
LOCK_PATH = "/tmp/finn-docker-load.lock"

FAKE_DOCKER = r"""#!/usr/bin/env bash
# Minimal `docker` stand-in. Keeps a tag->image-ID registry in a text file so a
# load, a re-tag and an inspect can be reasoned about without a daemon.
set -u
STATE="${FAKE_DOCKER_STATE:?}"
REG="$STATE/images"
touch "$REG"
printf '%s\n' "$*" >> "$STATE/calls"

lookup () {
  local want="$1" tag id
  while IFS=$'\t' read -r tag id; do
    if [ "$tag" = "$want" ]; then printf '%s\n' "$id"; return 0; fi
  done < "$REG"
  return 1
}

case "${1:-}" in
  image)
    shift
    [ "${1:-}" = "inspect" ] || { echo "fake docker: unsupported: image $*" >&2; exit 64; }
    shift
    fmt=""
    if [ "${1:-}" = "--format" ]; then fmt="$2"; shift 2; fi
    # Fails only the --format inspect, i.e. the post-load identity read, so a
    # test can distinguish "load worked but inspection did not" from
    # "image is absent".
    if [ -n "$fmt" ] && [ "${FAKE_DOCKER_INSPECT_FORMAT_FAIL:-0}" = "1" ]; then
      echo "fake docker: refusing to inspect" >&2
      exit 1
    fi
    id=$(lookup "${1:-}") || { echo "Error: No such image: ${1:-}" >&2; exit 1; }
    if [ -n "$fmt" ]; then printf '%s\n' "$id"; else printf '[{"Id":"%s"}]\n' "$id"; fi
    ;;
  load)
    cat > /dev/null
    if [ "${FAKE_DOCKER_LOAD_FAIL:-0}" = "1" ]; then
      echo "fake docker: load failed" >&2
      exit 1
    fi
    printf '%s\t%s\n' "${FAKE_DOCKER_LOAD_TAG:?}" "${FAKE_DOCKER_LOAD_ID:?}" >> "$REG"
    echo "Loaded image: ${FAKE_DOCKER_LOAD_TAG}"
    ;;
  tag)
    id=$(lookup "$2") || { echo "Error: No such image: $2" >&2; exit 1; }
    printf '%s\t%s\n' "$3" "$id" >> "$REG"
    ;;
  *)
    echo "fake docker: unsupported command: $*" >&2
    exit 64
    ;;
esac
exit 0
"""

FAKE_FLOCK = r"""#!/usr/bin/env bash
# Records the lock path, then runs the command. No real locking: the tests
# assert which lock the loader asks for, and a shared /tmp lock would
# serialise the test session itself.
set -u
printf '%s\n' "$1" >> "${FAKE_DOCKER_STATE:?}/flock"
shift
exec "$@"
"""


class Transport:
    """One temporary shared-image directory plus its fake Docker daemon."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.image_dir = tmp_path / "image"
        self.image_dir.mkdir()
        self.state = tmp_path / "state"
        self.state.mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("flock", FAKE_FLOCK)):
            script = bindir / name
            script.write_text(body)
            script.chmod(0o755)
        self.bindir = bindir
        self.env = {}

    def publish(self, tag, digest=None, archive=True, tag_file=True):
        """Write a shared-image directory in the published NFS layout."""
        if archive:
            (self.image_dir / "finn-docker-image.tar.gz").write_bytes(
                gzip.compress(b"not a real image, only the transport is under test")
            )
        if tag_file:
            (self.image_dir / "finn-docker-tag.txt").write_text(tag + "\n")
        if digest is not None:
            (self.image_dir / "finn-image-digest.txt").write_text(digest)

    def local_image(self, tag, image_id):
        """Pretend `tag` is already loaded on this agent's daemon."""
        with (self.state / "images").open("a") as handle:
            handle.write("%s\t%s\n" % (tag, image_id))

    def run(self, requested_tag, loads=None, shared_dir=True, **env):
        environ = dict(os.environ)
        environ["PATH"] = "%s:%s" % (self.bindir, environ["PATH"])
        environ["FAKE_DOCKER_STATE"] = str(self.state)
        environ["FINN_DOCKER_SHARED_IMAGE_DIR"] = str(self.image_dir) if shared_dir else ""
        environ["FINN_DOCKER_PREBUILT"] = "0"
        # What a successful `docker load` would put on the daemon. Defaults to
        # the requested tag resolving to ID_A.
        environ["FAKE_DOCKER_LOAD_TAG"] = loads[0] if loads else requested_tag
        environ["FAKE_DOCKER_LOAD_ID"] = loads[1] if loads else ID_A
        environ.update({key: str(value) for key, value in env.items()})
        return subprocess.run(
            [str(LOADER), requested_tag],
            capture_output=True,
            text=True,
            cwd=REPO,
            env=environ,
        )

    def docker_calls(self):
        path = self.state / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def lock_paths(self):
        path = self.state / "flock"
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def transport(tmp_path):
    return Transport(tmp_path)


# ---------------------------------------------------------------------------
# Baseline: the Jenkins compatibility bridge
# ---------------------------------------------------------------------------


def test_root_bridge_delegates_to_the_docker_launcher():
    # test_container_cli.py pins the bridge's size and its "not a supported
    # user interface" banner. What matters for transport compatibility is that
    # it still forwards an unmodified argv to the same launcher, so a Jenkins
    # caller's historical command syntax keeps working.
    body = (REPO / "run-docker.sh").read_text()
    assert 'exec "$ROOT/docker/run-docker" "$@"' in body
    assert (REPO / "docker/run-docker").is_file()


def test_loader_diagnostics_are_not_shell_escaped_twice(transport):
    # The colour setup had been through a shell-quoting layer one time too
    # many, so every message arrived as  "'\033[0;32m'"Loading...  Console
    # output is the only diagnostic a Jenkins operator has for this path.
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 0, proc.stderr
    assert "Verified" in proc.stdout
    assert "\\033" not in proc.stdout
    assert "\"'" not in proc.stdout


def test_docker_launcher_delegates_shared_image_loading_to_this_script():
    # The loader is only reachable through run-docker, so a rename here would
    # break every Jenkins shard without any test noticing.
    body = (REPO / "docker/run-docker").read_text()
    assert "./ci/scripts/load-shared-image.sh" in body


# ---------------------------------------------------------------------------
# Baseline: unconfigured and prebuilt-without-a-source behavior
# ---------------------------------------------------------------------------


def test_no_shared_dir_and_no_prebuilt_request_is_a_noop(transport):
    proc = transport.run("xilinx/finn:local", shared_dir=False)
    assert proc.returncode == 0, proc.stderr
    # Callers invoke this unconditionally; it must not touch the daemon.
    assert transport.docker_calls() == []


def test_prebuilt_without_shared_dir_or_local_image_fails(transport):
    proc = transport.run("xilinx/finn:pinned", shared_dir=False, FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 1
    assert "FINN_DOCKER_PREBUILT=1" in proc.stderr


def test_prebuilt_without_shared_dir_accepts_an_already_loaded_image(transport):
    transport.local_image("xilinx/finn:pinned", ID_A)
    proc = transport.run("xilinx/finn:pinned", shared_dir=False, FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Baseline: legacy archives, missing files, tag mismatch, locking
# ---------------------------------------------------------------------------


def test_legacy_archive_without_a_digest_still_loads(transport):
    transport.publish("xilinx/finn:abc123")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 0, proc.stderr
    assert "Loading Docker image from shared storage" in proc.stdout
    # The absent-digest warning itself is pinned by
    # test_absent_digest_warns_and_keeps_the_historical_path. What matters here
    # is that the archive still loads.
    assert any(call.startswith("load") for call in transport.docker_calls())


def test_missing_archive_falls_back_when_not_prebuilt(transport):
    transport.publish("xilinx/finn:abc123", archive=False)
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 0, proc.stderr
    assert "Falling back to local build" in proc.stdout


def test_missing_archive_is_fatal_when_prebuilt(transport):
    transport.publish("xilinx/finn:abc123", archive=False)
    proc = transport.run("xilinx/finn:abc123", FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 1
    assert "no usable shared Docker image" in proc.stderr


def test_missing_tag_file_is_fatal_when_prebuilt(transport):
    transport.publish("xilinx/finn:abc123", tag_file=False)
    proc = transport.run("xilinx/finn:abc123", FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 1
    assert "no usable shared Docker image" in proc.stderr


def test_tag_mismatch_is_fatal_in_prebuilt_mode(transport):
    transport.publish("xilinx/finn:published", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:requested", FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 1
    assert "does not match requested tag" in proc.stderr
    # Fatal *before* spending a multi-gigabyte load.
    assert not any(call.startswith("load") for call in transport.docker_calls())


def test_load_is_serialised_on_the_historical_lock_path(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 0, proc.stderr
    assert transport.lock_paths() == [LOCK_PATH]


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------


def test_absent_digest_warns_and_keeps_the_historical_path(transport):
    transport.publish("xilinx/finn:abc123")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 0, proc.stderr
    assert "legacy" in proc.stdout.lower()
    # Nothing is verified, so nothing claims to be.
    assert "Verified" not in proc.stdout


def test_matching_digest_verifies(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123", loads=("xilinx/finn:abc123", ID_A))
    assert proc.returncode == 0, proc.stderr
    assert "Verified" in proc.stdout
    assert ID_A in proc.stdout


def test_mismatching_digest_fails_before_the_container_runs(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123", loads=("xilinx/finn:abc123", ID_B))
    assert proc.returncode == 1
    assert "image identity mismatch" in proc.stderr.lower()
    assert ID_A in proc.stderr and ID_B in proc.stderr
    assert "\\033" not in proc.stderr


def test_empty_digest_file_fails(transport):
    transport.publish("xilinx/finn:abc123", digest="")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 1
    assert "finn-image-digest.txt" in proc.stderr


def test_multiple_digest_lines_fail(transport):
    transport.publish("xilinx/finn:abc123", digest="%s\n%s\n" % (ID_A, ID_B))
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 1
    assert "exactly one" in proc.stderr


def test_malformed_digest_fails(transport):
    for bad in ("not-a-digest\n", "sha256:zzzz\n", "a1" * 32 + "\n", "sha256:" + "a1" * 31 + "\n"):
        transport.publish("xilinx/finn:abc123", digest=bad)
        proc = transport.run("xilinx/finn:abc123")
        assert proc.returncode == 1, bad
        assert "sha256:" in proc.stderr, bad


def test_malformed_digest_fails_before_the_load(transport):
    # A malformed sidecar is a publisher bug. Detect it without first pulling
    # several gigabytes off NFS.
    transport.publish("xilinx/finn:abc123", digest="not-a-digest\n")
    proc = transport.run("xilinx/finn:abc123")
    assert proc.returncode == 1
    assert not any(call.startswith("load") for call in transport.docker_calls())


def test_inspect_failure_after_a_successful_load_fails(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123", FAKE_DOCKER_INSPECT_FORMAT_FAIL=1)
    assert proc.returncode == 1
    assert "could not read the image ID" in proc.stderr


def test_compatibility_retag_is_verified_through_the_requested_tag(transport):
    # The published tag differs from the tag Compose will use, so the re-tag
    # runs first and verification must follow the tag the container actually
    # references.
    transport.publish("xilinx/finn:published", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:requested", loads=("xilinx/finn:published", ID_A))
    assert proc.returncode == 0, proc.stderr
    assert "Tagging xilinx/finn:published as xilinx/finn:requested" in proc.stdout
    assert "Verified" in proc.stdout
    assert "tag xilinx/finn:published xilinx/finn:requested" in transport.docker_calls()


def test_verification_reads_the_requested_tag_not_the_recorded_one(transport):
    # A stale same-name image left by an earlier build must not satisfy the
    # check: it is the requested tag's current target that gets run.
    transport.publish("xilinx/finn:published", digest=ID_A + "\n")
    transport.local_image("xilinx/finn:requested", ID_B)
    proc = transport.run("xilinx/finn:requested", loads=("xilinx/finn:published", ID_A))
    # The local image pre-exists, so the loader skips the shared path entirely
    # in optional-cache mode. That is the historical behavior, unchanged.
    assert proc.returncode == 0, proc.stderr
    assert "Loading Docker image from shared storage" not in proc.stdout


def test_optional_shared_cache_load_failure_still_falls_back(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123", FAKE_DOCKER_LOAD_FAIL=1)
    assert proc.returncode == 0, proc.stderr
    assert "Falling back to local build" in proc.stdout
    # A load that never happened has no identity to verify.
    assert "Verified" not in proc.stdout


def test_authoritative_prebuilt_load_failure_remains_fatal(transport):
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    proc = transport.run("xilinx/finn:abc123", FINN_DOCKER_PREBUILT=1, FAKE_DOCKER_LOAD_FAIL=1)
    assert proc.returncode == 1
    assert "no usable shared Docker image" in proc.stderr


def test_a_load_is_never_skipped_because_the_id_is_already_present(transport):
    # Skipping an already-satisfied load is a real optimisation, but it changes
    # concurrency behavior and is deliberately deferred. Pin the current
    # behavior so the change is a deliberate one.
    transport.publish("xilinx/finn:abc123", digest=ID_A + "\n")
    transport.local_image("xilinx/finn:abc123", ID_A)
    proc = transport.run("xilinx/finn:abc123", FINN_DOCKER_PREBUILT=1)
    assert proc.returncode == 0, proc.stderr
    assert any(call.startswith("load") for call in transport.docker_calls())


# ---------------------------------------------------------------------------
# The published NFS layout
# ---------------------------------------------------------------------------


def test_publisher_writes_the_historical_archive_and_tag_filenames():
    body = (REPO / "ci/scripts/publish_docker_image.sh").read_text()
    assert 'final_img="${image_dir}/finn-docker-image.tar.gz"' in body
    assert 'final_tag="${image_dir}/finn-docker-tag.txt"' in body


def test_builder_writes_the_metadata_filenames_the_loader_reads():
    body = (REPO / "ci/scripts/build-images.sh").read_text()
    assert '"$OUTDIR/finn-image-provenance.json"' in body
    assert '"$OUTDIR/finn-image-digest.txt"' in body


# ---------------------------------------------------------------------------
# Jenkins metadata archival (static assertions against ci/Jenkinsfile)
# ---------------------------------------------------------------------------


def test_publish_still_reads_the_tag_from_the_nfs_provenance_file():
    body = JENKINSFILE.read_text()
    # The external contract is unchanged: publishSharedDockerImage reads the
    # provenance from the image directory, not from the workspace copy.
    assert 'def provenance = "${imageDir}/finn-image-provenance.json"' in body
    assert "json.load(open('${provenance}'))['tag']" in body


def build_and_publish_body():
    """The source of buildAndPublishDockerImage(), with `+` continuations joined."""
    body = JENKINSFILE.read_text()
    start = body.index("void buildAndPublishDockerImage()")
    end = body.index("// Build-scoped image directory", start)
    return re.sub(r"\+\s*\n\s*", "", body[start:end])


def test_jenkins_archives_deterministic_workspace_paths():
    fn = build_and_publish_body()
    assert "${CI_IMAGE_METADATA_DIR}" in fn
    # Exact paths, no recursive wildcard: a workspace-relative `**/` glob over
    # files that build-images.sh wrote to NFS matched nothing, and
    # allowEmptyArchive turned archiving nothing into a silent success.
    code = [line for line in fn.splitlines() if not line.strip().startswith("//")]
    assert not any("**/finn-image" in line for line in code), code
    archive = [line for line in code if "archiveArtifacts" in line]
    assert len(archive) == 1, archive
    assert "finn-image-provenance.json" in archive[0]
    assert "finn-image-digest.txt" in archive[0]
    # The build step immediately above is required to have written both files,
    # so an empty archive here is a container integration failure.
    assert "allowEmptyArchive" not in archive[0]


def test_jenkins_copies_both_metadata_files_to_the_nfs_image_dir():
    fn = build_and_publish_body()
    copy = [line for line in fn.splitlines() if "cp " in line]
    assert len(copy) == 1, copy
    assert "finn-image-provenance.json" in copy[0]
    assert "finn-image-digest.txt" in copy[0]
    assert "${shellQuote(imageDir)}/" in copy[0]
    # The copy has to land before the publisher, which reads the tag back out
    # of the provenance file in the image directory.
    assert fn.index("cp ") < fn.index("publishSharedDockerImage(imageDir)")


def test_local_fallback_archives_metadata_without_an_nfs_path():
    fn = build_and_publish_body()
    # Archival happens before the NFS branch, so local fallback mode still
    # produces build artifacts.
    assert fn.index("archiveArtifacts") < fn.index("if (imageDir)"), fn
    assert "local fallback mode, skipping image publish" in fn
