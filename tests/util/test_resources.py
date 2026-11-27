"""External resources: declarations, caches, fetchers and overrides.

Sources are local git repositories; no network is used.
"""
import pytest

import importlib.util
import multiprocessing
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from finn import resources
from finn.resources import _declare, _store, prepare

pytestmark = pytest.mark.util

REPO = Path(__file__).resolve().parents[2]
SRC = str(REPO / "src")


@contextmanager
def fault(
    code: str, error: type[resources.ResourceError] = resources.ResourceError, message=None
) -> Iterator[pytest.ExceptionInfo[resources.ResourceError]]:
    """Expect ``error`` carrying ``code`` (and its message matching ``message``, if given):
    a resource fault is asserted by its code, each message's text in one test."""
    with pytest.raises(error, match=message) as caught:
        yield caught
    assert caught.value.code == code, f"raised {caught.value.code!r}: {caught.value}"


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository(path, files):
    """A git repository with one commit of files; returns the commit."""
    path.mkdir()
    git(path, "init", "--quiet")
    # Allow fetching an exact commit, as GitHub does.
    git(path, "config", "uploadpack.allowReachableSHA1InWant", "true")
    return commit(path, files)


def commit(path, files):
    write(path, files)
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "fixture")
    return git(path, "rev-parse", "HEAD")


def write(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def digest_of(tmp_path, files):
    """The digest of a tree with exactly these files."""
    root = tmp_path / f"expected-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    return _store.tree_digest(write(root, files))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Private caches, and a declare(text) that replaces FINN's declarations."""
    for variable in list(os.environ):
        if variable.startswith(("FINN_RESOURCES_", "FINN_BOARD_FILES_PATH")):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "finn-home"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "system-cache"))
    (tmp_path / "system-cache").mkdir()
    monkeypatch.chdir(tmp_path)
    file = tmp_path / "resources.toml"

    def declare(text):
        file.write_text(textwrap.dedent(text))
        monkeypatch.setattr(_declare, "FINN_FILE", file)
        monkeypatch.setattr(resources, "_declared", None)
        return file

    monkeypatch.setattr(resources, "_declared", None)
    yield declare


@pytest.fixture
def boards(tmp_path, project):
    """A board repository declared as a sparse resource placed under 'kv260'."""
    repo = tmp_path / "store"
    revision = repository(
        repo,
        {
            "README.md": "root\n",
            "boards/Xilinx/kv260/board.xml": "k\n",
            "boards/Xilinx/other/board.xml": "o\n",
        },
    )
    digest = digest_of(tmp_path, {"kv260/board.xml": "k\n"})
    project(
        f"""
        # Test resources.
        [resources.kv260-boards]
        kind = ["test-boards"]
        git = "{repo.as_uri()}"
        commit = "{revision}"
        subdir = "boards/Xilinx/kv260"
        into = "kv260"
        digest = "{digest}"
        """
    )
    return repo


def test_finn_declares_its_trees_hlslib_and_board_files(project):
    declared = resources.declarations()
    own = {r.name for r in declared.values() if r.package}
    assert own == {"rtllib", "finn-xsi", "qnn-data", "custom-hls"}
    assert declared["hlslib"].kind == ("hls-include",) and declared["hlslib"].redistributable
    # FINN's own HLS headers come after finn-hlslib's on the include path.
    hls = [r.name for r in declared.values() if "hls-include" in r.kind]
    assert hls == ["hlslib", "custom-hls"]
    boards = [r for r in declared.values() if "vivado-boards" in r.kind]
    assert len(boards) == 5
    assert not any(r.redistributable for r in boards)
    assert all(r.origin == str(_declare.FINN_FILE) for r in declared.values())


def test_sparse_fetch_places_subdir_into_root_and_caches(boards, tmp_path, monkeypatch):
    path = Path(resources.path("kv260-boards"))
    assert path.parent == tmp_path / "finn-home/resources"
    assert path.name.startswith("kv260-boards-")
    assert (path / "kv260/board.xml").read_text() == "k\n"
    assert sorted(p.name for p in path.iterdir()) == [".finn-resource", "kv260"]
    assert resources.status("kv260-boards").state == "cached"

    # Once fetched, lookups never fetch again.
    def refetch(*args):
        raise AssertionError("fetched again")

    monkeypatch.setattr(_store, "assemble", refetch)
    assert resources.path("kv260-boards") == str(path)
    assert resources.paths("test-boards") == [str(path)]


def test_digest_mismatch_publishes_nothing(boards):
    declared = resources.declarations()["kv260-boards"]
    wrong = _declare.Resource(**{**declared.__dict__, "digest": "sha256:" + "0" * 64})
    root = _store.fetch_root()
    with fault("fetch-failed", message="has digest"):
        _store.fetch(wrong, root)
    assert [p.name for p in root.iterdir()] == [f".{wrong.entry}.lock"]


def test_caches_are_searched_in_order_and_the_system_cache_is_only_read(
    boards, tmp_path, monkeypatch
):
    resource = resources.declarations()["kv260-boards"]
    system = tmp_path / "system-cache"
    _store.fetch(resource, system)
    system.chmod(0o555)
    try:
        # The system cache serves the resource; nothing is written elsewhere.
        assert Path(resources.path("kv260-boards")).parent == system
        assert not (tmp_path / "finn-home").exists()

        # FINN_RESOURCES_DIR replaces $FINN_HOME/resources; fetches go there.
        site = tmp_path / "site-cache"
        monkeypatch.setenv("FINN_RESOURCES_DIR", str(site))
        assert Path(resources.path("kv260-boards")).parent == system
        _store.fetch(resource, site)
        assert Path(resources.path("kv260-boards")).parent == site
        assert _store.fetch_root() == site

        # An incomplete entry (no valid marker) is not a copy.
        (site / resource.entry / ".finn-resource").write_text("sha256:bad\n")
        assert Path(resources.path("kv260-boards")).parent == system
    finally:
        system.chmod(0o755)


def test_assembled_board_files_are_read_from_a_read_only_cache(boards, tmp_path, monkeypatch):
    declared = resources.declarations()["kv260-boards"]
    boards = _declare.Resource(**{**declared.__dict__, "kind": ("vivado-boards",)})
    monkeypatch.setattr(resources, "_declared", {"kv260-boards": boards})
    assembled = prepare.board_files()
    assert (Path(assembled) / "kv260/board.xml").read_text() == "k\n"
    cache = tmp_path / "finn-home/resources"
    paths = [cache, *cache.rglob("*")]
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        assert prepare.board_files() == assembled
    finally:
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_offline_and_no_fetch_report_the_fetch_command(boards, monkeypatch):
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    with fault("resource-not-cached", message="finn resources fetch kv260-boards") as error:
        resources.path("kv260-boards")
    # The message names settings that exist.
    assert "set FINN_RESOURCES_DIR=DIR" in str(error.value)
    assert "FINN_RESOURCES_SYSTEM_CACHE=DIR" in str(error.value)
    assert resources.status("kv260-boards").state == "missing"


def test_an_override_replaces_a_resource_with_a_local_directory(boards, tmp_path, monkeypatch):
    local = tmp_path / "local $ checkout"
    local.mkdir()
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS", str(local))
    assert resources.path("kv260-boards") == str(local)
    assert resources.status("kv260-boards") == resources.Status(
        "kv260-boards", "override", str(local)
    )
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS", str(tmp_path / "missing"))
    with fault("override-not-directory", message="FINN_RESOURCES_KV260_BOARDS"):
        resources.path("kv260-boards")


def test_variables_of_the_former_checkout_layout_select_nothing(project, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_HLSLIB_PATH", str(tmp_path))
    assert resources.status("hlslib").state != "override"


def test_git_is_required_to_fetch(boards, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    with fault("fetch-failed", message="git is required"):
        resources.path("kv260-boards")


def _fetch_in_child(queue):
    queue.put(resources.path("kv260-boards"))


def test_parallel_first_use_fetches_once(boards, tmp_path, monkeypatch):
    log = tmp_path / "fetches"
    assemble = _store.assemble

    def counted(resource, *args):
        with open(log, "a") as file:
            file.write(resource.name + "\n")
        return assemble(resource, *args)

    monkeypatch.setattr(_store, "assemble", counted)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=_fetch_in_child, args=(queue,)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert len({queue.get() for _ in workers}) == 1
    assert log.read_text() == "kv260-boards\n"


def test_fetching_into_another_cache_copies_a_verified_copy(boards, tmp_path):
    fetched = Path(resources.path("kv260-boards"))

    def no_network(*args):
        raise AssertionError("fetched from the source")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "assemble", no_network)
        (carried,) = resources.fetch(["kv260-boards"], tmp_path / "carry")
    assert Path(carried).parent == tmp_path / "carry"
    assert _store.tree_digest(carried) == _store.tree_digest(fetched)

    # A copy that no longer matches its digest is not propagated: the source is used.
    (fetched / "kv260/board.xml").write_text("tampered\n")
    (again,) = resources.fetch(["kv260-boards"], tmp_path / "carry-again")
    assert (Path(again) / "kv260/board.xml").read_text() == "k\n"


VALID = {"git": "u", "commit": "c" * 40, "digest": "sha256:" + "0" * 64}


@pytest.mark.parametrize(
    "name, fields, code, message",
    [
        ("acme", {"git": "u", "commit": "c" * 40}, "digest-invalid", "digest must be"),
        ("acme", {"git": "u"}, "source-invalid", "40-character commit"),
        ("acme", {**VALID, "digest": "sha256:0"}, "digest-invalid", "digest must be"),
        ("acme", {**VALID, "commit": "main"}, "source-invalid", "40-character commit"),
        ("acme", {**VALID, "subdir": "../x"}, "path-not-relative", "relative"),
        ("acme", {**VALID, "kind": "rtl"}, "field-type-invalid", "kind must be a list"),
        ("acme", {**VALID, "url": "u"}, "field-unknown", "unknown field"),
        ("acme", {"package": "a-b"}, "package-invalid", "importable module"),
        ("acme", {"package": "a", "digest": VALID["digest"]}, "package-fields-invalid", "subdir"),
        ("Acme", VALID, "name-invalid", "lower-case"),
        ("acme_rtl", VALID, "name-invalid", "lower-case"),
        # names whose override variable would be one of the settings
        ("dir", VALID, "name-invalid", "lower-case"),
        ("offline", VALID, "name-invalid", "lower-case"),
    ],
)
def test_invalid_declarations_are_rejected(name, fields, code, message):
    with fault(code, resources.DeclarationError, message):
        _declare.parse({name: fields}, "test")


def test_the_tree_digest_sorts_by_path_components_and_skips_the_marker(tmp_path):
    # "a/x" before "a-b/x", although "-" sorts before "/" as a string.
    tree = write(tmp_path / "t", {"a-b/x": "1", "a/x": "2", ".finn-resource": "ignored"})
    expected = sha256()
    for name in ("a/x", "a-b/x"):
        expected.update(name.encode() + b"\0" + sha256((tree / name).read_bytes()).digest())
    assert _store.tree_digest(tree) == "sha256:" + expected.hexdigest()


def test_resources_import_only_the_standard_library(tmp_path):
    check = textwrap.dedent(
        """
        import sys
        before = set(sys.modules)
        from finn import resources
        resources.declarations()
        resources.status("hlslib")
        loaded = {name.partition(".")[0] for name in set(sys.modules) - before}
        foreign = sorted(loaded - set(sys.stdlib_module_names) - {"finn"})
        assert not foreign, foreign
        allowed = ("finn.resources", "finn.util", "finn.util.paths")
        assert not any(n.startswith("finn.") and not n.startswith(allowed)
                       for n in sys.modules), sorted(sys.modules)
        """
    )
    # -S: no site-packages at all, so any non-stdlib import fails outright.
    subprocess.run(
        [sys.executable, "-S", "-c", check],
        env={"PYTHONPATH": SRC, "HOME": str(tmp_path), "PATH": os.defpath},
        cwd=tmp_path,
        check=True,
    )


def cli(capsys, *args):
    from finn.resources import _cli  # noqa: PLC0415

    code = _cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_list_fetch_and_path(boards, tmp_path, capsys):
    row = cli(capsys, "list")[1].splitlines()
    assert ["kv260-boards", "missing", "test-boards"] == next(
        line.split()[:3] for line in row if line.startswith("kv260-boards")
    )
    code, out, err = cli(capsys, "fetch", "--kind", "test-boards", "--redistributable-only")
    assert (code, out, err) == (0, "", "")
    dest = tmp_path / "carry"
    code, out, _ = cli(capsys, "fetch", "kv260-boards", "--dest", str(dest))
    assert code == 0 and out.strip() == str(dest / resources.declarations()["kv260-boards"].entry)
    assert cli(capsys, "fetch")[0] == 1
    assert "no resource named 'nope'" in cli(capsys, "fetch", "nope")[2].lower()
    code, out, _ = cli(capsys, "path", "kv260-boards")
    assert code == 0 and (Path(out.strip()) / "kv260/board.xml").read_text() == "k\n"


def update_script(monkeypatch, *args):
    """Run scripts/update-resource.py in this process, so it sees the test's declarations."""
    spec = importlib.util.spec_from_file_location(
        "update_resource", REPO / "scripts/update-resource.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["update-resource.py", *args])
    module.main()
    return module


def test_update_script_rewrites_only_the_pin(boards, capsys, monkeypatch):
    file = _declare.FINN_FILE
    before = file.read_text()
    file.write_text(before + '\n[other]\ncommit = "keep"\n')
    old = resources.declarations()["kv260-boards"]
    revision = commit(boards, {"boards/Xilinx/kv260/board.xml": "k2\n"})
    git(boards, "branch", "release")

    update_script(monkeypatch, "kv260-boards", "--ref", "release")
    assert revision[:12] in capsys.readouterr().out
    text = file.read_text()
    monkeypatch.setattr(resources, "_declared", None)
    new = resources.declarations()["kv260-boards"]
    assert new.commit == revision and new.digest != old.digest
    assert text.replace(revision, old.commit).replace(new.digest, old.digest) == (
        before + '\n[other]\ncommit = "keep"\n'
    )
    # The script published the new tree, so no fetch is needed.
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    assert (Path(resources.path("kv260-boards")) / "kv260/board.xml").read_text() == "k2\n"
    update_script(monkeypatch, "kv260-boards", "--ref", revision)
    assert "already at" in capsys.readouterr().out
    entry = resources.path("kv260-boards")
    update_script(monkeypatch, "--digest", entry)
    assert capsys.readouterr().out.strip() == _store.tree_digest(entry)


def test_update_refuses_edits_it_cannot_confine(tmp_path, monkeypatch):
    module = update_script(monkeypatch, "--digest", str(tmp_path))
    file = tmp_path / "resources.toml"
    inline = '[resources]\nacme = { git = "u", commit = "%s", digest = "d" }\n' % ("a" * 40)
    file.write_text(inline)
    with pytest.raises(SystemExit, match="edit it by hand"):
        module.rewrite(file, "acme", {"commit": "b" * 40})
    # A multi-line string that looks like a field would change another value.
    tricky = '[resources.acme]\nnote = """\ncommit = "x"\n"""\ncommit = "%s"\n' % ("a" * 40)
    file.write_text(tricky)
    with pytest.raises(SystemExit, match="edit it by hand"):
        module.rewrite(file, "acme", {"commit": "b" * 40})
    assert file.read_text() == tricky


def test_cli_runs_as_a_module(project):
    result = subprocess.run(
        [sys.executable, "-m", "finn.cli", "resources", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert any(line.startswith("hlslib ") for line in result.stdout.splitlines())


def test_a_resource_fault_names_its_reason_by_a_code_and_keeps_its_message():
    with pytest.raises(resources.ResourceError, match="No resource named") as error:
        resources.path("no-such-resource")
    assert error.value.code == "resource-undeclared"
    assert str(error.value) == "No resource named 'no-such-resource' is declared"
    declaration = resources.DeclarationError("name-invalid", "bad name")
    assert isinstance(declaration, resources.ResourceError)
    assert (declaration.code, str(declaration)) == ("name-invalid", "bad name")


def test_package_sources_are_used_in_place(project, tmp_path, monkeypatch):
    write(tmp_path / "lib", {"acme/rtl/mvu/mvu.sv": "module mvu;\n", "acme/__init__.py": ""})
    monkeypatch.syspath_prepend(str(tmp_path / "lib"))
    project(
        """
        [resources.acme-rtl]
        kind = ["hls-include"]
        package = "acme"
        subdir = "rtl"
        """
    )
    rtl = str((tmp_path / "lib/acme/rtl").resolve())
    assert resources.status("acme-rtl") == resources.Status("acme-rtl", "package", rtl)
    assert resources.paths("hls-include") == [rtl]
    # Used in place: nothing is fetched, verified or cached.
    assert resources.fetch(["acme-rtl"]) == []
    assert not (tmp_path / "finn-home/resources").exists()
    assert resources.join("acme-rtl", "mvu", "mvu.sv") == rtl + "/mvu/mvu.sv"
    assert resources.join("acme-rtl", "mvu/").endswith("/mvu/")
    for escape in ("../x", "/etc/passwd", "mvu/../../x"):
        with pytest.raises(ValueError, match="leaves resource acme-rtl"):
            resources.join("acme-rtl", escape)
    with fault("resource-file-missing", message="in resource acme-rtl"):
        resources.join("acme-rtl", "mvu", "absent.sv")


def test_finn_trees_are_package_data_in_a_checkout_too():
    """src/finn/_sources links to the checkout's trees, which a wheel carries as files."""
    for name, tree in (
        ("rtllib", "finn-rtllib"),
        ("finn-xsi", "finn_xsi"),
        ("custom-hls", "custom_hls"),
    ):
        status = resources.status(name)
        assert status.state == "package"
        # From this checkout (not a wheel installed beside it): the links' targets.
        if status.path.startswith(str(REPO)):
            assert status.path == str((REPO / tree).resolve())


def test_a_package_may_only_add_resources(tmp_path):
    finn = _declare.parse({"own": {"package": "x"}}, "finn")
    package = _declare.parse({"own": {"package": "y"}}, "acme")
    with fault("resource-redeclared", resources.DeclarationError, "already declared"):
        _declare.merge(finn, package)


def installed(site, dist, module, declarations):
    """An installed distribution declaring resources through the finn.resources group."""
    info = f"{dist.replace('-', '_')}-1.0.dist-info"
    return write(
        site,
        {
            f"{module}/__init__.py": "",
            f"{module}/resources.toml": textwrap.dedent(declarations),
            f"{module}/rtl/core.v": "module core;\n",
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0\n",
            f"{info}/entry_points.txt": f"[finn.resources]\n{module} = {module}\n",
        },
    )


def test_package_declarations_in_a_virtual_environment(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    (site / "finn-src.pth").write_text(SRC + "\n")
    installed(
        site,
        "acme-finn",
        "acme_finn",
        '[resources.acme-rtl]\nkind = ["rtl"]\npackage = "acme_finn"\nsubdir = "rtl"\n',
    )

    def run(*args):
        return subprocess.run(
            [venv / "bin/python", "-I", "-m", "finn.resources", *args],
            cwd=tmp_path,
            env={"PATH": os.defpath, "HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert ["acme-rtl", "package", "rtl"] in [line.split()[:3] for line in run("list").splitlines()]
    assert run("path", "acme-rtl").strip() == str(site / "acme_finn/rtl")
