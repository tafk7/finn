# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""docker/install-runtimes.sh, without Docker, network or apt: its storage paths are
redirected, and wget, sha256sum and apt-get are stand-ins on PATH."""
import pytest

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.util

PACKAGE = b"runtime package\n"


def installer(tmp_path):
    """The installer, writing only below tmp_path, with one fetched runtime, xrt."""
    for name in ("runtimes", "packages", "cache", "bin", "verified"):
        (tmp_path / name).mkdir()
    (tmp_path / "runtimes/xrt.env").write_text(
        "SOURCE=fetch\nDEB=xrt\nURL=https://unused.invalid/xrt.deb\n"
        f"SHA256={hashlib.sha256(PACKAGE).hexdigest()}\n"
    )
    script = (REPO / "docker/install-runtimes.sh").read_text()
    for old, new in (
        ("RUNTIME_DIR=/tmp/runtimes", f"RUNTIME_DIR={tmp_path}/runtimes"),
        ("PACKAGE_DIR=/tmp/packages", f"PACKAGE_DIR={tmp_path}/packages"),
        ("/var/cache/finn-runtimes", f"{tmp_path}/cache"),
        ("rm -rf /var/lib/apt/lists/*", ":"),
    ):
        assert old in script
        script = script.replace(old, new)
    (tmp_path / "install.sh").write_text(script)
    return tmp_path / "install.sh"


def stand_in(tmp_path, name, body):
    path = tmp_path / "bin" / name
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def test_concurrent_builds_share_the_download_cache(tmp_path):
    """Two builds fetching one package into BuildKit's shared cache mount both
    install it, even when both have verified a download before either publishes."""
    install = installer(tmp_path)
    stand_in(
        tmp_path,
        "wget",
        "import pathlib, sys\n"
        f"pathlib.Path(sys.argv[sys.argv.index('-O') + 1]).write_bytes({PACKAGE!r})\n",
    )
    stand_in(tmp_path, "apt-get", "")
    # The real check, then wait until both builds have passed it.
    stand_in(
        tmp_path,
        "sha256sum",
        f"""import os, pathlib, subprocess, sys, time
check = subprocess.run(["/usr/bin/sha256sum", *sys.argv[1:]], input=sys.stdin.buffer.read())
if check.returncode:
    sys.exit(check.returncode)
verified = pathlib.Path({str(tmp_path / "verified")!r})
(verified / os.environ["BUILD"]).touch()
deadline = time.monotonic() + 10
while len(list(verified.iterdir())) < 2 and time.monotonic() < deadline:
    time.sleep(0.01)
""",
    )
    builds = [
        subprocess.Popen(
            ["sh", install],
            env={"PATH": f"{tmp_path}/bin:/usr/bin:/bin", "FINN_RUNTIMES": "xrt", "BUILD": b},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for b in ("one", "two")
    ]
    outputs = [build.communicate(timeout=30)[0] for build in builds]
    assert [build.returncode for build in builds] == [0, 0], outputs
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["xrt.deb"]
    assert (tmp_path / "cache/xrt.deb").read_bytes() == PACKAGE
    assert (tmp_path / "cache/xrt.deb").stat().st_mode & 0o777 == 0o644


def test_a_download_that_fails_its_check_leaves_nothing_in_the_cache(tmp_path):
    install = installer(tmp_path)
    stand_in(
        tmp_path,
        "wget",
        "import pathlib, sys\npathlib.Path(sys.argv[sys.argv.index('-O') + 1]).write_text('x')\n",
    )
    build = subprocess.run(
        ["sh", install],
        env={"PATH": f"{tmp_path}/bin:/usr/bin:/bin", "FINN_RUNTIMES": "xrt"},
        capture_output=True,
        text=True,
    )
    assert build.returncode != 0 and "FAILED" in build.stdout
    assert list((tmp_path / "cache").iterdir()) == []
