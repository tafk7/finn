"""External build data lookup: overrides, the finn-hlslib package and board-file fetching.

Board fetching runs against local git repositories; no network is used.
"""
import pytest

import subprocess
from pathlib import Path

from finn.util import external


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def repository(path, files):
    path.mkdir()
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    git(path, "init", "--quiet")
    # Allow fetching an exact commit, as GitHub does.
    git(path, "config", "uploadpack.allowReachableSHA1InWant", "true")
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "fixture")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def sources(tmp_path):
    whole = tmp_path / "whole"
    whole_commit = repository(whole, {"README.md": "root\n", "zedboard/board.xml": "z\n"})
    store = tmp_path / "store"
    store_commit = repository(
        store, {"boards/Xilinx/kv260/board.xml": "k\n", "boards/Xilinx/other/board.xml": "o\n"}
    )
    return (
        (whole.as_uri(), whole_commit, "", ""),
        (store.as_uri(), store_commit, "boards/Xilinx/kv260", "kv260_som"),
    )


def test_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_HLSLIB_PATH", str(tmp_path))
    monkeypatch.setenv("FINN_BOARD_FILES_PATH", str(tmp_path))
    assert external.hlslib_path() == str(tmp_path)
    assert external.board_files_path() == str(tmp_path)
    monkeypatch.setenv("FINN_BOARD_FILES_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="FINN_BOARD_FILES_PATH"):
        external.board_files_path()


def test_hlslib_comes_from_the_package(monkeypatch):
    pytest.importorskip("finn_hlslib")
    monkeypatch.delenv("FINN_HLSLIB_PATH", raising=False)
    assert (Path(external.hlslib_path()) / "bnn-library.h").is_file()


def test_unverified_board_files_are_never_published(tmp_path, sources):
    dest = tmp_path / "boards"
    with pytest.raises(RuntimeError, match="digest"):
        external.fetch_board_files(dest, sources, digest="0" * 64)
    assert not dest.exists()
    assert not any(p.name.startswith(".boards.") and p.is_dir() for p in tmp_path.iterdir())


def test_board_files_layout_and_cache(tmp_path, sources, monkeypatch):
    # Learn the digest from one assembly, then check the real behaviour with it.
    probe = tmp_path / "probe"
    with pytest.raises(RuntimeError) as error:
        external.fetch_board_files(probe, sources, digest="0" * 64)
    digest = str(error.value).split("digest ")[1].split(",")[0]

    monkeypatch.delenv("FINN_BOARD_FILES_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(external, "BOARD_SOURCES", sources)
    monkeypatch.setattr(external, "BOARD_FILES_DIGEST", digest)
    path = Path(external.board_files_path())
    assert path.parent == tmp_path / "cache" / "finn"
    assert (path / "zedboard/board.xml").read_text() == "z\n"
    assert (path / "kv260_som/board.xml").read_text() == "k\n"
    assert not (path / "other").exists() and not (path / ".git").exists()

    # Once fetched, lookups never touch git again.
    def no_git(*args, **kwargs):
        raise AssertionError("fetched again")

    monkeypatch.setattr(external, "_sparse_checkout", no_git)
    assert external.board_files_path() == str(path)
    with pytest.raises(FileNotFoundError):
        monkeypatch.setattr(external, "BOARD_FILES_DIGEST", "1" * 64)
        external.board_files_path(fetch=False)
