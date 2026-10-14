# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged-IP store: Vivado's packaging of a staged project, kept under
``$FINN_HOME/ip`` (``ip_store``) and shared by every build.

A staged project is a directory holding ``package.tcl`` and the files it reads, by
paths relative to the script (``finn.kernels.artifacts.ipxact.package_tcl`` writes such
a script, with the IP's name, the part and the clock inside it). Its key
(``packaging_key``) is the project's content, Vivado's identity
(``finn.toolchain.launcher.identity``) and ``run_synth``: what Vivado is given, never
where. ``packaged`` returns the store's entry for a project, ``<name>-<key16>``, and
packages it only when the store holds none: Vivado runs in a temporary directory beside
the entry, and its product is marked with the full key and renamed into place under the
entry's lock (``finn.resources``' store), so a failed or concurrent packaging never
publishes a partial IP, and a second caller waits for the first. A failed packaging is
kept beside the entry (``<entry>.failed``), and the ``ToolFailed`` it raises names its
transcript.

An entry holds the staged project, the IP (``ip/``), Vivado's log and the run's replay
script (``package.sh``), with its transcript (``.stdout.log``, ``.stderr.log``) unless
the run was a build's, whose step's log directory holds it (``Toolchain.run``); with
``run_synth``, the checkpoint and reports the script writes beside them. Vivado's own
project (``project/``) is not kept. Vivado records the IP's include directories by
absolute path; the store rewrites each one inside the project relative to the IP, so an
entry, and every copy of it, can move. An IP whose ``component.xml`` still names the
directory it was packaged in is refused. The entry's files are read-only: ``place``
copies an entry into a build's directory by hard links where it can, so a write into
the copy would change the store.

It takes directories, never a kernel value: ``finn.toolchain`` imports no kernel layer.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from finn.resources import home, tree_digest
from finn.resources._store import MARKER, complete, locked, publish
from finn.toolchain import launcher

if TYPE_CHECKING:
    from finn.toolchain.launcher import Toolchain

#: The staged project's script.
PACKAGE_SCRIPT = "package.tcl"
#: The IP's directory in a packaged project, and its description.
IP_DIR, COMPONENT = "ip", "component.xml"
#: What Vivado leaves in the project that an entry does not keep: its project and its
#: scratch directory.
_UNKEPT = ("project", ".Xil")
_DEPENDENCY = re.compile(r"<spirit:dependency>([^<]*)</spirit:dependency>")


def ip_store() -> Path:
    """Where packaged IPs are stored: ``$FINN_HOME/ip``."""
    root: Path = home()  # type: ignore[no-untyped-call]
    return root / "ip"


def packaging_key(project: Path, *, run_synth: bool, toolchain: Toolchain) -> str:
    """The key of a staged project's IP, ``sha256:<hex>``: its content, Vivado's
    identity and ``run_synth``."""
    preimage = "\0".join(
        (
            "packaged-ip-v1",
            tree_digest(project),  # type: ignore[no-untyped-call]
            launcher.identity(toolchain, "vivado"),
            f"run_synth={run_synth}",
        )
    )
    return "sha256:" + hashlib.sha256(preimage.encode()).hexdigest()


@dataclass(frozen=True)
class PackagedIp:
    """A stored packaging: its entry (``directory``) and its ``key``."""

    directory: Path
    key: str

    @property
    def ip(self) -> Path:
        """The IP: its ``component.xml`` and the files it names."""
        return self.directory / IP_DIR


def packaged(
    project: Path,
    *,
    name: str,
    run_synth: bool,
    toolchain: Toolchain,
    store: Path | None = None,
) -> PackagedIp:
    """The stored packaging of the staged ``project``, packaged by Vivado in
    ``toolchain`` unless the store holds it.

    ``name`` names the entry, ``<name>-<key16>`` under ``store`` (``ip_store()`` by
    default). Raises ``ToolFailed`` when Vivado fails or packages no IP,
    ``ToolUnavailable`` when the toolchain names no Vivado."""
    store = (ip_store() if store is None else Path(store)).resolve()
    project = Path(project)
    if not (project / PACKAGE_SCRIPT).is_file():
        raise ValueError(f"{project} holds no {PACKAGE_SCRIPT}: it is not a staged project")
    key = packaging_key(project, run_synth=run_synth, toolchain=toolchain)
    entry_name = f"{name}-{key.removeprefix('sha256:')[:16]}"
    with locked(store, entry_name) as entry:
        if complete(entry, key):  # type: ignore[no-untyped-call]
            return PackagedIp(entry, key)
        work = Path(tempfile.mkdtemp(dir=store, prefix=f".{entry_name}."))
        failed = store / f"{entry_name}.failed"
        try:
            _package(project, work, toolchain, kept=failed, name=name)
        except BaseException:
            shutil.rmtree(failed, ignore_errors=True)
            work.rename(failed)
            raise
        _read_only(work)
        publish(work, entry, key)  # type: ignore[no-untyped-call]
    return PackagedIp(entry, key)


def _package(project: Path, work: Path, toolchain: Toolchain, *, kept: Path, name: str) -> None:
    """Package ``project`` in ``work`` and make the result movable; a failure names the
    transcript where the run is ``kept`` (``work``, once moved there)."""
    shutil.copytree(project, work, dirs_exist_ok=True)
    replay = work / "package.sh"
    ran = toolchain.run(
        "vivado",
        ["-mode", "batch", "-nojournal", "-log", "package.log", "-source", PACKAGE_SCRIPT],
        cwd=work,
        check=False,
        replay=replay,
    )
    # In a build, the transcript is in the step's log directory; else beside the replay,
    # where the run is kept.
    transcript = getattr(ran, "transcript", None)
    if transcript is not None and Path(transcript).is_relative_to(work):
        transcript = kept / Path(transcript).relative_to(work)
    printed = (ran.stdout or b"") + (ran.stderr or b"")
    if ran.returncode != 0:
        raise launcher.ToolFailed.of(
            "vivado", printed, status=ran.returncode, transcript=transcript
        )
    component = work / IP_DIR / COMPONENT
    if not component.is_file():
        raise launcher.ToolFailed.of(
            "vivado", printed, transcript=transcript, what=f"{name}: no IP packaged"
        )
    for unkept in _UNKEPT:
        shutil.rmtree(work / unkept, ignore_errors=True)
    # The replay runs where the entry is, not in the temporary directory.
    if replay.is_file():
        replay.write_text(
            re.sub(r"^cd -- .*$", 'cd -- "$(dirname -- "$0")"', replay.read_text(), flags=re.M)
        )
    _relative_dependencies(component, work)
    if str(work) in component.read_text():
        raise launcher.ToolFailed.of(
            "vivado",
            transcript=transcript,
            what=f"{name}: the packaged IP's {COMPONENT} names the directory it was packaged in",
        )


def _relative_dependencies(component: Path, work: Path) -> None:
    """Rewrite each include directory ``component`` names inside ``work`` relative to
    the IP's directory."""
    ip = component.parent

    def relative(match: re.Match[str]) -> str:
        named = Path(match.group(1))
        if not named.is_relative_to(work):
            return match.group(0)
        return f"<spirit:dependency>{os.path.relpath(named, ip)}</spirit:dependency>"

    component.write_text(_DEPENDENCY.sub(relative, component.read_text()))


def _read_only(directory: Path) -> None:
    """Make every file below ``directory`` read-only."""
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _link(source: str, target: str) -> None:
    """A hard link of ``source`` at ``target``, else a copy."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def place(entry: Path, directory: Path) -> None:
    """Copy a stored entry's files into ``directory`` by hard links where possible,
    replacing what ``directory`` holds of the same names; the rest of ``directory``
    stays."""
    directory.mkdir(parents=True, exist_ok=True)
    for item in sorted(Path(entry).iterdir()):
        if item.name == MARKER:
            continue
        target = directory / item.name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True, copy_function=_link)
        else:
            _link(str(item), str(target))


__all__ = ["PACKAGE_SCRIPT", "PackagedIp", "ip_store", "packaged", "packaging_key", "place"]
