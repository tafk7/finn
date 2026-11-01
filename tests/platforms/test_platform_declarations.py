# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform's declarations (``finn.platform.declarations``): FINN's own file, the
three sources and their order, the refusals of a malformed or repeated declaration,
``based_on``, code imported only on use, the checks (``finn.platform.check``) and the
``finn-platform`` command; and a made-up platform (``support.example_platform``) declared
outside FINN, which needs no change to FINN."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from support.example_platform import DIRECTORY as EXAMPLE
from support.example_platform import IMAGE, ExamplePush, RecordingRuntime
from support.refusals import refused

from finn.platform import _cli
from finn.platform.catalog import catalog
from finn.platform.check import check
from finn.platform.declarations import (
    FINN_FILE,
    GROUP,
    PATH_SETTING,
    Declarations,
    Origin,
    code,
    declarations,
    load,
)
from finn.platform.refusal import TargetRefused
from finn.platform.resolve import resolve_target
from finn.platform.shells import MOVED, PUSHED, Versions, shell_row, shell_rows

FINN_ONLY: dict[str, str] = {}
"""An environment with no ``FINN_PLATFORM_PATH``."""


def declared(tmp_path: Path, text: str, name: str = "platform.toml") -> Path:
    """``text`` written as a declaration file in ``tmp_path``; its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


def findings(declared_: Declarations) -> list[tuple[str, str]]:
    return [(finding.code, finding.owner.split(": ")[-1]) for finding in check(declared_)]


# -- FINN's own --------------------------------------------------------------------------


def test_finn_declares_its_boards_shells_linker_runtime_and_mover_kind() -> None:
    finn = load(environ=FINN_ONLY)
    assert sorted(finn.shells) == ["ip", "pynq"]
    assert list(finn.linkers) == ["vivado-block-design"]
    assert list(finn.runtimes) == ["pynq"] and list(finn.movers) == ["iodma_hls"]
    assert len(finn.boards) == 18
    assert {
        str(finn.origin(kind, name))
        for kind in ("boards", "shells")
        for name in ("V80", "ip")
        if name in getattr(finn, kind)
    } == {"FINN"}
    assert finn.origin("shells", "pynq") == Origin("finn", FINN_FILE)
    assert str(finn.origin("shells", "pynq")) == "FINN"
    assert finn.shell("ip").any_board and finn.supporting("ZCU111") == ("ip",)
    assert finn.supporting("Ultra96") == ("pynq", "ip")


def test_finns_declarations_check() -> None:
    assert check(load(environ=FINN_ONLY)) == []


def test_reading_and_resolving_import_no_declared_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declaration names its code; only ``code`` (and the checks) import it."""
    monkeypatch.delitem(sys.modules, "support.example_platform", raising=False)
    found = load([EXAMPLE], environ=FINN_ONLY)
    shell_rows(found)
    resolve_target(board="Example-ZU3", period_ns=5.0, shell="example", declared=found)
    assert "support.example_platform" not in sys.modules
    assert (
        code(found.runtimes["example-run"].code)
        is sys.modules["support.example_platform"].RecordingRuntime
    )


# -- sources and their refusals ----------------------------------------------------------


def test_a_name_is_declared_once_across_every_source(tmp_path: Path) -> None:
    again = declared(tmp_path, '[boards.Ultra96]\npart = "xczu3eg-sbva484-1-i"\n')
    with refused(
        "platform-name-taken",
        error=TargetRefused,
        owner=f"{again}: boards.Ultra96",
        message=f"^boards.Ultra96 is declared by FINN and again by {again}: a source only adds",
    ):
        load([again], environ=FINN_ONLY)
    on_path = tmp_path / "on-path"
    on_path.mkdir()
    declared(on_path, '[shells.pynq]\nboards = "any"\nclocks = {}\n')
    with refused(
        "platform-name-taken",
        error=TargetRefused,
        message=f"^shells.pynq is declared by FINN and again by {PATH_SETTING} {on_path}",
    ):
        load(environ={PATH_SETTING: str(on_path)})


def test_the_path_lists_directories_each_declaring(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    for directory, board in ((first, "One"), (second, "Two")):
        directory.mkdir()
        declared(directory, f'[boards.{board}]\npart = "xczu3eg-sbva484-1-e"\n')
    found = load(environ={PATH_SETTING: f"{first}:{second}"})
    assert [str(found.origin("boards", name)) for name in ("One", "Two")] == [
        f"{PATH_SETTING} {first}",
        f"{PATH_SETTING} {second}",
    ]
    assert [directory for _, directory in found.directories] == [first, second]
    with refused("declaration-invalid", error=TargetRefused, owner=PATH_SETTING):
        load(environ={PATH_SETTING: str(tmp_path / "absent")})


def test_an_added_file_or_directory_that_declares_nothing_is_refused(tmp_path: Path) -> None:
    """A file a caller adds is read or refused, never skipped: one that is not there, or a
    directory holding neither declarations nor a parts overlay."""
    for absent in (tmp_path / "absent.toml", tmp_path):
        with refused(
            "declaration-invalid",
            error=TargetRefused,
            owner=str(absent),
            message=f"^{absent}: a platform.toml, or a directory holding",
        ):
            load([absent], environ=FINN_ONLY)
    (tmp_path / "parts.json").write_text('{"schema": 1, "devices": {}, "parts": {}}')
    assert load([tmp_path], environ=FINN_ONLY).directories[-1][1] == tmp_path


@dataclass(frozen=True)
class _Dist:
    name: str
    version: str
    files: None = None  # recorded none: an editable install's


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: str
    dist: _Dist


def test_an_installed_package_declares_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package names its declarations' package in the group ``finn.platform``; they are
    read after FINN's, with the distribution as their origin."""
    point = _EntryPoint("example", "support.example_platform", _Dist("example-platform", "0.1"))

    def entry_points(*, group: str) -> list[_EntryPoint]:
        return [point] if group == GROUP else []

    monkeypatch.setattr("finn.platform.declarations.entry_points", entry_points)
    found = load(environ=FINN_ONLY)
    assert str(found.origin("shells", "example")) == "package example-platform 0.1"
    assert found.directories == ((found.origin("shells", "example"), EXAMPLE),)
    assert check(found) == []


_INITIALIZER = """\
import os, pathlib
pathlib.Path(os.environ["MARKER"]).write_text("ran")
print("the package was imported")
"""

_LIST_BOARDS = """\
from finn.platform import _cli
from finn.platform.declarations import declarations
print(declarations().origin("boards", "Marked"))
_cli.main(["list", "boards"])
"""


def _installed(site: Path, package: str, *, declares: bool) -> None:
    """``package`` installed in ``site`` as a wheel installs it: its code, which marks that
    it ran, its ``platform.toml`` (if it ``declares``), and its distribution's metadata,
    recording its files and naming it in the group ``finn.platform``."""
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text(_INITIALIZER)
    files = [f"{package}/__init__.py"]
    if declares:
        (site / package / "platform.toml").write_text(
            '[boards.Marked]\npart = "xczu3eg-sbva484-1-e"\n'
        )
        files.append(f"{package}/platform.toml")
    info = site / f"{package}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[{GROUP}]\nmarked = {package}\n")
    (info / "RECORD").write_text("".join(f"{each},,\n" for each in files))


def _fresh(site: Path, script: str, marker: Path) -> subprocess.CompletedProcess[str]:
    """``script`` run in a fresh interpreter with ``site`` on its path."""
    environ = {key: value for key, value in os.environ.items() if key != PATH_SETTING}
    environ["PYTHONPATH"] = os.pathsep.join(
        each for each in (str(site), os.environ.get("PYTHONPATH")) if each
    )
    environ["MARKER"] = str(marker)
    return subprocess.run(
        [sys.executable, "-c", script], env=environ, capture_output=True, text=True, check=False
    )


def test_an_installed_packages_declarations_are_read_without_running_its_code(
    tmp_path: Path,
) -> None:
    """Found through its distribution's metadata in a fresh process, an installed package's
    declarations are read, listed and shown, and the package's own code never runs; one that
    holds no declarations is refused, owned by its distribution."""
    site, marker = tmp_path / "site", tmp_path / "marker"
    _installed(site, "marked_platform", declares=True)
    ran = _fresh(site, _LIST_BOARDS, marker)
    assert ran.returncode == 0, ran.stderr
    origin, *listed = ran.stdout.splitlines()
    assert origin == "package marked_platform 1.0"
    assert ["Marked", "package", "marked_platform", "1.0"] in [line.split() for line in listed]
    assert "imported" not in ran.stdout and not marker.exists()

    bare = tmp_path / "bare"
    _installed(bare, "bare_platform", declares=False)
    ran = _fresh(bare, _LIST_BOARDS, marker)
    assert ran.returncode != 0 and not marker.exists()
    assert (
        "package bare_platform 1.0: entry point 'marked': declaration-invalid: "
        "bare_platform holds no platform.toml"
    ) in ran.stderr


@pytest.mark.parametrize(
    "text, where, message",
    [
        ('[boards.B]\npart = "x"\nvivado = "2024.1"\n', "boards.B", "states vivado, which"),
        ("[board.B]\n", "board", "a declaration is one of boards, linkers"),
        ('[boards."a b"]\npart = "x"\n', "boards.a b", "is no name"),
        (
            '[movers.m]\ncode = "no colon"\nkinds = ["moved"]\n',
            "movers.m",
            "movers.m.code names code as module:attr",
        ),
        (
            '[runtimes.r]\ncode = "a:b"\nimages = []\nmovers = []\nkinds = []\nschemas = [0]\n',
            "runtimes.r",
            "runtimes.r.schemas is a list of schema numbers",
        ),
        ("[shells.s]\nclocks = {}\n", "shells.s", "states no boards"),
        (
            "[shells.s]\nboards = { B = {} }\nclocks = {}\n",
            "shells.s",
            "a shell with no linker supports any part",
        ),
        ('[shells.s]\nbased_on = "s"\nboards = "any"\n', "shells.s.based_on", "based on itself"),
        ('[shells.s]\nbased_on = "t"\nboards = "any"\n', "shells.s.based_on", "not a declared"),
    ],
    ids=[
        "an unknown key",
        "an unknown kind",
        "no name",
        "code not module:attr",
        "a schema 0",
        "a shell with no boards",
        "boards with no linker",
        "based on itself",
        "based on nothing",
    ],
)
def test_a_malformed_declaration_is_refused_where_it_is(
    tmp_path: Path, text: str, where: str, message: str
) -> None:
    path = declared(tmp_path, text)
    with refused(
        "declaration-invalid", error=TargetRefused, owner=f"{path}: {where}", message=message
    ):
        load([path], environ=FINN_ONLY)


def test_a_shell_based_on_another_replaces_each_key_it_states(tmp_path: Path) -> None:
    path = declared(
        tmp_path,
        '[shells.narrow]\nbased_on = "pynq"\ncontrol_budget = 4\n'
        '[shells.narrow.boards.Ultra96]\nboard_part = "avnet.com:ultra96v1:part0:1.2"\n',
    )
    found = load([path], environ=FINN_ONLY)
    narrow, pynq = shell_row("narrow", "Ultra96", found), shell_row("pynq", "Ultra96", found)
    assert (narrow.control_budget, pynq.control_budget) == (4, 9)
    assert (narrow.fixed, narrow.supply, narrow.linker) == (pynq.fixed, pynq.supply, pynq.linker)
    # boards is a key like any other: the base's whole table is replaced.
    assert list(found.shell("narrow").boards) == ["Ultra96"]
    assert found.shell("narrow").based_on == "pynq"


@pytest.mark.parametrize(
    "fact, named",
    [
        ('control_budget = "nine"', "control_budget is a whole number"),
        ('supply = [{ kind = "moved", movers = [{}] }]', "supply[0].movers[0] is a non-empty"),
        ('caveat = "on {board.missing}"', "caveat names {board} and {part} only"),
        ('clocks = { main = "", set_at = "build", doubled = false }', "main is a non-empty"),
        (
            'clocks = { main = "c", set_at = "build", doubled = false, default_mhz = -100.0 }',
            "default_mhz is a finite number above zero",
        ),
        (
            'clocks = { main = "c", set_at = "build", doubled = false, default_mhz = inf }',
            "default_mhz is a finite number above zero",
        ),
        ('memory = [{ name = "", width = 64 }]', "memory[0].name is a non-empty"),
        ('memory = [{ name = "m", width = 9 }]', "memory[0].width: 9 bits is no whole bytes"),
    ],
    ids=[
        "a count not a number",
        "a mover kind not a name",
        "a caveat naming another field",
        "a clock with no name",
        "a negative frequency",
        "an infinite frequency",
        "a memory target with no name",
        "a memory width of no whole bytes",
    ],
)
def test_a_fact_not_of_its_type_is_refused_when_its_row_is_read(
    tmp_path: Path, fact: str, named: str
) -> None:
    path = declared(tmp_path, f'[shells.broken]\nbased_on = "pynq"\n{fact}\n')
    found = load([path], environ=FINN_ONLY)
    with refused(
        "declaration-invalid", error=TargetRefused, owner=f"{path}: shells.broken"
    ) as caught:
        shell_row("broken", "Ultra96", found)
    assert named in str(caught.error)
    assert ("declaration-invalid", "shells.broken") in findings(found)


def test_checking_goes_on_past_a_malformed_shell_and_code_that_fails_to_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each malformed declaration is a finding owned by it, and code whose module raises
    as it is imported is ``code-unimportable``, whatever it raises: the rest is checked."""
    (tmp_path / "failing_platform_code.py").write_text('raise RuntimeError("it fails")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    path = declared(
        tmp_path,
        '[movers.failing]\ncode = "failing_platform_code:Mover"\nkinds = ["moved"]\n'
        '[shells.unhashable]\nbased_on = "pynq"\nsupply = [{ kind = "moved", movers = [{}] }]\n'
        '[shells.misspelt]\nbased_on = "pynq"\ncaveat = "on {board.missing}"\n',
    )
    found = findings(load([path], environ=FINN_ONLY))
    assert {
        ("code-unimportable", "movers.failing"),
        ("declaration-invalid", "shells.unhashable"),
        ("declaration-invalid", "shells.misspelt"),
    } == set(found)
    with refused("code-unimportable", error=TargetRefused, message="it fails"):
        code("failing_platform_code:Mover")


# -- resolving through the declarations --------------------------------------------------


def test_an_undeclared_shell_is_refused_listing_the_declared_ones() -> None:
    with refused(
        "unknown-shell",
        error=TargetRefused,
        owner="configuration.target.shell",
        message=r"^'xrt' is not a declared shell \(one of \['ip', 'pynq'\]\)$",
    ):
        shell_row("xrt", None)


def test_a_known_board_its_shell_does_not_support_names_those_that_do() -> None:
    with refused(
        "board-unsupported",
        error=TargetRefused,
        message=r"^the 'pynq' shell does not support board 'V80' \(xcv80-lsva4737-2MHP-e-S\); "
        r"only 'ip' supports it$",
    ):
        shell_row("pynq", "V80")


def test_a_board_entry_without_its_linkers_keys_is_refused(tmp_path: Path) -> None:
    path = declared(tmp_path, '[shells.bare]\nbased_on = "pynq"\nboards = { Ultra96 = {} }\n')
    found = load([path], environ=FINN_ONLY)
    with refused("declaration-invalid", error=TargetRefused, message="states no board_part"):
        shell_row("bare", "Ultra96", found)


# -- the checks --------------------------------------------------------------------------


def test_every_check_names_its_declaration(tmp_path: Path) -> None:
    path = declared(
        tmp_path,
        """
[boards.Nowhere]
part = "xcnothing-0"

[linkers.lost]
code = "support.no_such_module:link"
tool = "vivado"
board_keys = []
image = "rom"

[runtimes.idle]
code = "support.example_platform:IMAGE"
images = ["bit"]
movers = []
kinds = []
schemas = [1]

[shells.odd]
linker = "lost"
runtimes = ["idle", "absent"]
clocks = { main = "c", set_at = "build", doubled = false }
supply = [{ kind = "moved", movers = ["iodma_hls"] }]
boards = { Ultra96 = {}, Elsewhere = {} }

[shells.unlinked]
linker = "gone"
clocks = { main = "c", set_at = "build", doubled = false }
boards = { Ultra96 = {} }
""",
    )
    found = load([path], environ=FINN_ONLY)
    assert sorted(set(findings(found))) == sorted(
        {
            ("unknown-part", "boards.Nowhere"),
            ("runtime-undeclared", "shells.odd"),
            ("board-undeclared", "shells.odd"),
            ("image-unloaded", "shells.odd"),
            ("kind-unserved", "shells.odd"),
            ("mover-undriven", "shells.odd"),
            ("linker-undeclared", "shells.unlinked"),
            ("code-unimportable", "linkers.lost"),
            ("code-not-callable", "runtimes.idle"),
        }
    )
    owners = {finding.owner for finding in check(found)}
    assert f"{path}: shells.odd" in owners and f"{FINN_FILE}: shells.pynq" not in owners


# -- the command -------------------------------------------------------------------------


@pytest.fixture
def example_on_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The made-up platform's directory on ``FINN_PLATFORM_PATH``, the process's
    declarations and catalog read again."""
    monkeypatch.setenv(PATH_SETTING, str(EXAMPLE))
    declarations.cache_clear()
    catalog.cache_clear()
    yield EXAMPLE
    monkeypatch.delenv(PATH_SETTING)
    declarations.cache_clear()
    catalog.cache_clear()


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    status = _cli.main(argv)
    out, err = capsys.readouterr()
    return status, out, err


def test_the_command_lists_shows_and_checks(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(["list", "shells"], capsys)
    assert status == 0 and out.splitlines() == [
        "shells:",
        f"  {'ip':<24} FINN",
        f"  {'pynq':<24} FINN",
    ]
    status, out, _ = run(["show", "board", "AUP-ZU3_8GB"], capsys)
    assert status == 0
    assert out.splitlines()[0] == "board AUP-ZU3_8GB: part xczu3eg-sfvc784-2-e (FINN)"
    assert '  shell pynq: {"board_part": ' in out and "  shell ip (any part)" in out
    status, out, _ = run(["show", "shell", "pynq", "--board", "Ultra96"], capsys)
    stated = json.loads("\n".join(line[4:] for line in out.splitlines()[2:]))
    assert stated["control"]["aperture"] == "0x10000000"
    assert stated["fixed"]["control_base"] == "0xa0000000"
    assert stated["board_facts"] == {"board_part": "avnet.com:ultra96v1:part0:1.2"}
    status, out, _ = run(["show", "part", "xczu3eg-sbva484-1-e"], capsys)
    assert status == 0 and json.loads(out)["name"] == "xczu3eg-sbva484-1-e"
    status, out, _ = run(["check"], capsys)
    assert status == 0 and out.startswith("finn-platform: every declaration checks (18 boards, ")
    status, _, err = run(["show", "shell", "xrt"], capsys)
    assert status == 1 and err.startswith(
        "finn-platform: configuration.target.shell: unknown-shell"
    )


def test_the_command_reports_each_finding_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    declared(tmp_path, '[boards.Nowhere]\npart = "xcnothing-0"\n')
    monkeypatch.setenv(PATH_SETTING, str(tmp_path))
    declarations.cache_clear()
    try:
        status, out, _ = run(["check"], capsys)
    finally:
        declarations.cache_clear()
    assert status == 1
    assert out.startswith(f"{tmp_path / 'platform.toml'}: boards.Nowhere: unknown-part: ")


# -- a made-up platform, declared outside FINN -------------------------------------------


def test_a_made_up_platform_declares_checks_and_resolves_with_no_change_to_finn(
    example_on_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``support.example_platform`` declares a board, a linker, a runtime, a pushed mover
    kind and a shell based on ``pynq``: they check, a build's target resolves on them, and
    the command shows them, each with its origin."""
    found = declarations()
    assert check(found) == []
    origin = f"{PATH_SETTING} {example_on_path}"
    assert str(found.origin("shells", "example")) == origin
    target = resolve_target(board="Example-ZU3", period_ns=5.0, shell="example")
    assert (target.shell, target.board, target.part) == (
        "example",
        "Example-ZU3",
        "xczu3eg-sbva484-1-e",
    )
    row = shell_row("example", "Example-ZU3")
    pynq = shell_row("pynq", "Ultra96")
    assert (row.linker, row.runtimes, row.board_facts) == (
        "example-link",
        ("example-run",),
        {"slot": "B"},
    )
    assert row.versions == Versions(tool="example-tool", spec=">=1.0")
    assert [offer.kind for offer in row.supply] == [MOVED, PUSHED]
    assert [mover.name for mover in row.supply[1].movers] == ["example_push"]
    assert (row.fixed, row.control, row.delivery) == (pynq.fixed, pynq.control, pynq.delivery)
    assert row.caveat == (
        "the 'example' shell is made up: on Example-ZU3 (xczu3eg-sbva484-1-e) nothing is measured"
    )
    # Its code is a package's own, imported where it is used.
    linker: Any = code(found.linkers["example-link"].code)
    image = linker({"shell": "example"}, tmp_path / "linked")
    assert image.name == IMAGE and image.is_file()
    made: Any = code(found.runtimes["example-run"].code)
    runtime = made()
    assert isinstance(runtime, RecordingRuntime)
    runtime.load(image)
    assert runtime.operations == [("load", IMAGE)]
    assert code(found.movers["example_push"].code) is ExamplePush
    status, out, _ = run(["list"], capsys)
    assert status == 0 and f"  {'example':<24} {origin}" in out
    status, out, _ = run(["show", "board", "Ultra96"], capsys)
    assert '  shell example: {"slot": "A"}' in out
