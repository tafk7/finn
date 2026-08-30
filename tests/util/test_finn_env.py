"""Tests for docker/finn-env, the single host/container environment resolver.

These run outside a container against synthetic Xilinx trees, so they cover the
layouts and licence forms that a single development host cannot exercise. That
is the whole point: the defects this resolver replaces were all cases where a
code path was only ever run against one host's configuration.

In particular, the pre-2024.2 Xilinx layout is unrepresented on the machine this
was developed on, and that is exactly where the sbx mount silently resolved to
nothing.
"""

import pytest

import importlib.util
import json
import os
import subprocess
import sys

FINN_ENV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docker", "finn-env")


def _load():
    """Import finn-env as a module despite the hyphen and missing suffix."""
    spec = importlib.util.spec_from_loader(
        "finn_env",
        importlib.machinery.SourceFileLoader("finn_env", FINN_ENV))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finn_env = _load()


def _make_tree(base, layout, version):
    """Build a synthetic Xilinx install in one of the two layouts."""
    if layout == "new":
        dirs = ["%s/Vivado" % version, "%s/Vitis" % version]
    else:
        dirs = ["Vivado/%s" % version, "Vitis/%s" % version,
                "Vitis_HLS/%s" % version]
    for d in dirs:
        os.makedirs(os.path.join(base, d))
    return base


# --------------------------------------------------------------------------
# Layout resolution -- defect 2.
# --------------------------------------------------------------------------

def test_new_layout_resolves(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    found = finn_env.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "2025.1", "Vivado")
    assert found["XILINX_VITIS"] == os.path.join(root, "2025.1", "Vitis")


def test_old_layout_resolves(tmp_path):
    """The layout that the sbx path used to miss entirely.

    2022.2 is FINN's documented default version and uses $ROOT/Vivado/2022.2.
    The previous sbx code probed $ROOT/2022.2, found nothing, and created the
    sandbox with no toolchain mounted while still exporting VIVADO_PATH.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2")
    found = finn_env.xilinx_layout(root, "2022.2")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2")
    assert found["XILINX_HLS"] == os.path.join(root, "Vitis_HLS", "2022.2")


def test_layout_is_probed_not_assumed(tmp_path):
    """A site whose tree disagrees with its version string still resolves.

    Deciding purely from the version number gives a confidently wrong answer on
    a symlinked or relocated install. Probing detects it.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2025.1")
    found = finn_env.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2025.1")


def test_missing_toolchain_yields_nothing(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    assert finn_env.xilinx_layout(empty, "2022.2") == {}


def test_malformed_version_still_probes(tmp_path, capsys):
    """An unparseable version warns but does not give up.

    The version string only orders the two candidate layouts; it is never the
    thing that decides. A site running a patched or vendor-relabelled release
    still resolves, and gets told its version string is unusual.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2-patched")
    found = finn_env.xilinx_layout(root, "2022.2-patched")
    assert "not YYYY.1" in capsys.readouterr().err
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2-patched")


# --------------------------------------------------------------------------
# Licence classification.
# --------------------------------------------------------------------------

def test_floating_license():
    servers, files = finn_env.classify_license("2100@licsrv.example")
    assert servers == [{"host": "licsrv.example", "advertised_port": "2100"}]
    assert files == []


def test_node_locked_license():
    servers, files = finn_env.classify_license("/opt/lic/Xilinx.lic")
    assert servers == []
    assert files == ["/opt/lic/Xilinx.lic"]


def test_mixed_license_list():
    servers, files = finn_env.classify_license(
        "2100@a.example:/opt/lic/x.lic::27000@b.example")
    assert [s["host"] for s in servers] == ["a.example", "b.example"]
    assert files == ["/opt/lic/x.lic"]


def test_empty_license():
    assert finn_env.classify_license("") == ([], [])
    assert finn_env.classify_license(None) == ([], [])


# --------------------------------------------------------------------------
# Host resolution and the dev contract.
# --------------------------------------------------------------------------

def _inspect(env, tier, backend="docker"):
    """Run the real CLI in a clean environment, so nothing leaks in."""
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    base.update(env)
    proc = subprocess.run(
        [sys.executable, FINN_ENV, "inspect", "--tier", tier,
         "--backend", backend, "--format", "json"],
        capture_output=True, env=base, text=True)
    return proc


def test_dev_requires_nothing(tmp_path):
    """D2's contract, asserted rather than assumed.

    The Xilinx variables are deliberately set here: dev must ignore them. The
    original bug class was a tier widening because a variable happened to be
    exported in the caller's shell.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "XILINXD_LICENSE_FILE": "2100@licsrv.example",
                     "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms"}, "dev")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["mounts"] == []
    assert data["egress"] == []
    assert data["dev_contract"] == {
        "toolchain": False, "license": False, "egress": False}
    for leaked in ("XILINX_VIVADO", "XILINXD_LICENSE_FILE",
                   "PLATFORM_REPO_PATHS"):
        assert leaked not in data["env"]


def test_build_mounts_root_read_only(tmp_path):
    """Defect 1: the root is mounted, and it is mounted :ro."""
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2")
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2022.2"}, "build")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    toolchain = [m for m in data["mounts"] if m["reason"] == "xilinx-toolchain"]
    assert len(toolchain) == 1
    assert toolchain[0]["source"] == root
    assert toolchain[0]["target"] == root
    assert toolchain[0]["mode"] == "ro"


def test_platform_repo_is_mounted(tmp_path):
    """Defect 3: it used to be exported as env with no mount."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    platforms = str(tmp_path / "platforms")
    os.makedirs(platforms)
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "PLATFORM_REPO_PATHS": platforms}, "build")
    data = json.loads(proc.stdout)
    assert any(m["source"] == platforms and m["mode"] == "ro"
               for m in data["mounts"])
    assert data["env"]["PLATFORM_REPO_PATHS"] == platforms


def test_platform_repo_inside_root_is_not_double_mounted(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    platforms = os.path.join(root, "platforms")
    os.makedirs(platforms)
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "PLATFORM_REPO_PATHS": platforms}, "build")
    data = json.loads(proc.stdout)
    assert [m["source"] for m in data["mounts"]] == [root]


def test_node_locked_license_dir_is_mounted(tmp_path):
    """The mount the old sbx path lost to a subshell."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    licdir = tmp_path / "lic"
    licdir.mkdir()
    (licdir / "Xilinx.lic").write_text("x")
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "XILINXD_LICENSE_FILE": str(licdir / "Xilinx.lic")},
                    "build", backend="sbx")
    data = json.loads(proc.stdout)
    assert any(m["source"] == str(licdir) and m["reason"] == "license-file"
               for m in data["mounts"])
    assert data["egress"] == []


def test_floating_license_grants_host_not_port(tmp_path):
    """A port-scoped grant lets lmstat pass and every real checkout fail."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "XILINXD_LICENSE_FILE": "2100@licsrv.example"},
                    "build", backend="sbx")
    data = json.loads(proc.stdout)
    assert data["egress"] == [{"host": "licsrv.example", "reason": "flexlm",
                               "advertised_port": "2100"}]
    assert not any(m["reason"] == "license-file" for m in data["mounts"])


def test_missing_xilinx_path_is_an_error(tmp_path):
    proc = _inspect({}, "build")
    assert proc.returncode == 3
    assert "FINN_XILINX_PATH" in proc.stderr


def test_unknown_tier_is_rejected():
    proc = subprocess.run(
        [sys.executable, FINN_ENV, "inspect", "--tier", "nonsense"],
        capture_output=True, text=True)
    assert proc.returncode != 0


# --------------------------------------------------------------------------
# Workspace policy -- D4.
# --------------------------------------------------------------------------

def test_sbx_always_mirrors(tmp_path):
    """sbx has no mount remapping, so the policy is not a free choice there."""
    proc = _inspect({"FINN_ROOT": "/somewhere/finn"}, "dev", backend="sbx")
    data = json.loads(proc.stdout)
    assert data["workspace"]["policy"] == "mirror"
    assert data["workspace"]["target"] == "/somewhere/finn"


def test_fpga_tiers_always_mirror(tmp_path):
    """LIMITATION(finn-root-absolute): generated .xpr files embed FINN_ROOT."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect({"FINN_XILINX_PATH": root,
                     "FINN_XILINX_VERSION": "2025.1",
                     "FINN_ROOT": "/somewhere/finn"}, "build")
    data = json.loads(proc.stdout)
    assert data["workspace"]["policy"] == "mirror"


def test_env_root_matches_workspace_target(tmp_path):
    """FINN_ROOT must be the CONTAINER path, never the host one."""
    proc = _inspect({"FINN_ROOT": "/somewhere/finn"}, "dev")
    data = json.loads(proc.stdout)
    assert data["env"]["FINN_ROOT"] == data["workspace"]["target"]


# --------------------------------------------------------------------------
# Output discipline.
# --------------------------------------------------------------------------

def test_stdout_is_pure_json_even_with_warnings(tmp_path):
    """Diagnostics go to stderr so callers can pipe stdout directly."""
    empty = str(tmp_path / "Xilinx")
    os.makedirs(empty)
    proc = _inspect({"FINN_XILINX_PATH": empty,
                     "FINN_XILINX_VERSION": "2022.2"}, "build")
    assert proc.returncode == 0
    json.loads(proc.stdout)          # must not raise
    assert "no Vivado/Vitis/HLS found" in proc.stderr


def test_sh_format_is_shell_assignments(tmp_path):
    """The .env compose consumes.

    Under the dev policy the SOURCE is the host checkout and the TARGET is the
    fixed path -- they are deliberately different, and conflating them is what
    would mount the workspace in the wrong place.
    """
    proc = subprocess.run(
        [sys.executable, FINN_ENV, "inspect", "--tier", "dev",
         "--format", "sh"],
        capture_output=True, text=True,
        env={"PATH": os.environ["PATH"], "FINN_ROOT": "/w/finn"})
    assert "FINN_WORKSPACE_SOURCE=/w/finn" in proc.stdout
    assert "FINN_WORKSPACE_TARGET=%s" % finn_env.FIXED_WORKSPACE in proc.stdout
    # FINN_ROOT is the CONTAINER path, so it tracks the target, not the source.
    assert "FINN_ROOT=%s" % finn_env.FIXED_WORKSPACE in proc.stdout
    # Compose cannot get these from the shell; see the comment in finn-env.
    assert "FINN_UID=" in proc.stdout
    assert "FINN_GID=" in proc.stdout


# --------------------------------------------------------------------------
# Idempotence.
# --------------------------------------------------------------------------

def test_dedupe_paths_collapses_repeats():
    """settings64.sh prepends unconditionally, so applying it twice grows PATH.

    Observed in a live sandbox at four copies of the full Xilinx PATH -- about
    3 kB -- because the entrypoint applied it, then the sbx kit's startup
    command applied it again on top of the result.
    """
    env = {"PATH": "/a:/b:/a:/c:/b", "LD_PRELOAD": "/x:/x"}
    out = finn_env.dedupe_paths(dict(env))
    assert out["PATH"] == "/a:/b:/c"      # order preserved, first wins
    assert out["LD_PRELOAD"] == "/x"


def test_dedupe_paths_drops_empty_segments():
    out = finn_env.dedupe_paths({"PATH": "/a::/b:"})
    assert out["PATH"] == "/a:/b"


def test_dedupe_paths_ignores_absent_vars():
    out = finn_env.dedupe_paths({"PATH": "/a"})
    assert out == {"PATH": "/a"}


def test_dev_uses_the_fixed_workspace_path(tmp_path):
    """D4, as flipped in stage 7.

    dev is Python-only and generates no Vivado projects, so mirroring the host
    path buys it nothing while costing remote-daemon support and reproducible
    diagnostics. The FPGA tiers still mirror because generated .xpr files embed
    $::env(FINN_ROOT) as an absolute path.
    """
    proc = _inspect({"FINN_ROOT": "/somewhere/finn"}, "dev")
    data = json.loads(proc.stdout)
    assert data["workspace"]["policy"] == "fixed"
    assert data["workspace"]["source"] == "/somewhere/finn"
    assert data["workspace"]["target"] == finn_env.FIXED_WORKSPACE


# --------------------------------------------------------------------------
# Host path hygiene.
# --------------------------------------------------------------------------

def test_hostpath_expands_tilde(monkeypatch):
    """A tilde in a variable is never expanded by a shell or by Compose.

    This repo contains a stray directory literally named `~`, holding HLS output
    from a run where something passed `~/builds` through a variable. `.gitignore`
    has `*~` for editor backups, which hides it from `git status`.

    Every path finn-env emits becomes a mount argument, so every one is expanded
    here.
    """
    monkeypatch.setenv("HOME", "/home/someone")
    assert finn_env.hostpath("~/builds") == "/home/someone/builds"


def test_hostpath_makes_relative_absolute():
    assert finn_env.hostpath("x").startswith("/")


def test_hostpath_passes_through_empty():
    assert finn_env.hostpath("") == ""
    assert finn_env.hostpath(None) is None


def test_emitted_paths_have_no_tilde(tmp_path):
    """End to end: a tilde in the environment must not reach the .env file."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = subprocess.run(
        [sys.executable, FINN_ENV, "inspect", "--tier", "build", "--format", "sh"],
        capture_output=True, text=True,
        env={"PATH": os.environ["PATH"], "HOME": "/home/someone",
             "FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.1",
             "FINN_HOST_BUILD_DIR": "~/builds", "FINN_ROOT": "~/finn"})
    assert proc.returncode == 0, proc.stderr
    for line in proc.stdout.splitlines():
        assert "~" not in line, "tilde survived into the .env: %s" % line
    assert "FINN_HOST_BUILD_DIR=/home/someone/builds" in proc.stdout
