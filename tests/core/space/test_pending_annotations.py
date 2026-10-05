# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Suppliers of a formal whose annotation does not resolve yet, and forwards' Space classes.

Two Space classes that place each other across an import cycle (a channel whose
source is a memory kernel, whose ports reference channels) leave a formal's
annotation unresolved while the other module's class bodies run. A supplier of
such a formal is checked when linking, where every annotation resolves: the
formal's kind, its Space class (a forward's too) and its value semantics. A literal
still needs its type at the call.

The cycle is a toy package written to ``tmp_path``. Each (variant, entry
module) runs in a fresh interpreter, because the outcome must not depend on
which module of the cycle is imported first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from finn.core.space import Decision, DefinitionError, Param, Space, derived, design_space

SRC = Path(__file__).resolve().parents[3] / "src"

# The port names the channel Space class by its full path, without importing it: the
# channel module imports the port (as finn.kernels.channels does through its FIFO).
PORT = """
from __future__ import annotations
from typing import TYPE_CHECKING
import toy
from finn.core.space import Param, Space, derived
if TYPE_CHECKING:
    import toy.chan

class Port(Space):
    channel: {port_annotation} = Param(required=False)

    @derived
    def width(self) -> int:
        return self.channel.width
"""

# The memory kernel imports the channel module last: its Space classes come first.
MEM = """
from __future__ import annotations
from finn.core.space import Decision, Param, Space
from toy.port import Port

class Other(Space):
    size: int = Param(default=1)

class Mem(Space):
    output: chan.Chan = Param(required=False)
    index: chan.Chan = Param(required=False)
    count: int = Param(default=1)
    other: Other = Param(required=False)
    out_port = Port(channel=output)
    set_port = Port(channel={set_port_supplier})
{mem_extra}
from toy import chan  # noqa: E402
"""

CHAN = """
from __future__ import annotations
from finn.core.space import Decision, Param, Space, supplied
from toy.mem import Mem
from toy.port import Port

class Chan(Space):
    width: int = Param()
    contents: int = Param(required=False)
    index: Chan = Param(required=False)
    valued = supplied(contents)
    source: Mem = Decision({{"mem": Mem}}, when=valued, index=index)

class Width:
    pass
{chan_extra}
"""

ENTRY = """
from __future__ import annotations
import importlib, json, sys
stage = "import"
try:
    importlib.import_module(sys.argv[1])
    import toy.chan, toy.mem, toy.port
    from finn.core.space import Param, Space, design_space, inspection
    from toy.chan import Chan
    from toy.port import Port

    class Consumer(Space):
        x: Chan = Param()
        p = Port(channel=x)

    class Root(Space):
        idx = Chan(width=2)
        c = Chan(width=8, contents=5, index=idx)
        k = Consumer(x=c)

    stage = "design_space"
    point = design_space(Root())
    keys = sorted(d.key for d in inspection.decisions(point))
    print(json.dumps({"stage": "ok", "decisions": keys, "width": point.k.p.width}))
except Exception as error:
    print(json.dumps({"stage": stage, "error": type(error).__name__, "message": str(error)}))
"""

DEFAULTS = {
    "port_annotation": "toy.chan.Chan",
    "set_port_supplier": "index",
    "mem_extra": "",
    "chan_extra": "",
}

ENTRIES = ("toy.port", "toy.mem", "toy.chan")


def run_cycle(tmp_path: Path, entry: str, **variant: str) -> dict[str, object]:
    """Write the toy with ``variant`` applied and run its root, importing ``entry`` first."""
    values = {**DEFAULTS, **variant}
    package = tmp_path / "toy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name, text in (("port", PORT), ("mem", MEM), ("chan", CHAN)):
        (package / f"{name}.py").write_text(textwrap.dedent(text.format(**values)))
    (tmp_path / "entry.py").write_text(textwrap.dedent(ENTRY))
    environment = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{SRC}")
    done = subprocess.run(
        [sys.executable, str(tmp_path / "entry.py"), entry],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    outcome: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return outcome


# -- accepted -----------------------------------------------------------------------------

ACCEPTED = {
    "a forward into a pending reference formal": {},
    "a node supplied to a pending formal, in the cycle's module": {
        "chan_extra": "class Pair(Space):\n    c = Chan(width=1)\n    p = Port(channel=c)"
    },
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("variant", ACCEPTED)
def test_a_pending_supplier_is_accepted_in_every_entry_order(
    tmp_path: Path, variant: str, entry: str
) -> None:
    outcome = run_cycle(tmp_path, entry, **ACCEPTED[variant])
    assert outcome == {"stage": "ok", "decisions": ["c.source"], "width": 8}


# -- refused, each where it is first known ------------------------------------------------

REFUSED = {
    "a value formal forwarded into a pending reference formal": (
        {"set_port_supplier": "count"},
        "design_space",
        "c.source.mem.set_port.channel: forwards count (declared at mem.py:12), which is "
        "not a reference input of Mem",
    ),
    "a reference of another Space class forwarded into a pending formal": (
        {"set_port_supplier": "other"},
        "design_space",
        "c.source.mem.set_port.channel: forwards other (declared at mem.py:13), a Other "
        "input of Mem; Port.channel takes a Chan",
    ),
    "a reference forwarded into a pending formal that takes a value": (
        {"port_annotation": "toy.chan.Width"},
        "design_space",
        "k.p.channel: forwards the reference input x (declared at entry.py:13), but the "
        "formal takes a value",
    ),
    "a node of another Space class supplied to a pending formal": (
        {"mem_extra": "    extra = Port(channel=Other())"},
        "design_space",
        "c.source.mem.extra.channel: expected a Chan node, got Other (declared at mem.py:16)",
    ),
    "a typo in a pending annotation": (
        {"port_annotation": "toy.chan.Chann"},
        "design_space",
        "Port.channel (declared at port.py:10): cannot resolve the annotation "
        "'toy.chan.Chann': module 'toy.chan' has no attribute 'Chann'",
    ),
    "a literal supplied to a pending formal": (
        {"mem_extra": "    extra = Port(channel=3)"},
        "import",
        "the annotation of channel does not resolve yet, so a literal cannot be recognized "
        "here; bind a member, or assign the value where the Space class is defined",
    ),
    "a Decision over nodes supplied to a pending formal": (
        {"mem_extra": '    extra = Port(channel=Decision({"o": Other}))'},
        "import",
        "a Decision over nodes is not a supplier; bind a member",
    ),
    "a reach through a pending reference in a class body": (
        {"mem_extra": "    width_seen = output.width"},
        "import",
        "width: the reference's annotation does not resolve yet (its Space class is still being "
        "defined); read the member in a method",
    ),
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("variant", REFUSED)
def test_a_pending_supplier_is_refused_by_name_in_every_entry_order(
    tmp_path: Path, variant: str, entry: str
) -> None:
    overrides, stage, message = REFUSED[variant]
    outcome = run_cycle(tmp_path, entry, **overrides)
    assert outcome["stage"] == stage, outcome
    assert message in str(outcome["message"]), outcome


# -- without a cycle ----------------------------------------------------------------------


class Chan(Space):
    width: int = Param()


class Other(Space):
    size: int = Param(default=3)


class Port(Space):
    channel: Chan = Param()

    @derived
    def width(self) -> int:
        return self.channel.width


def test_a_forward_of_another_space_class_is_refused_when_linking() -> None:
    """Every annotation resolves; the forward's Space class is still checked when linking."""

    class Kernel(Space):
        x: Other = Param()
        port = Port(channel=x)  # type: ignore[arg-type]

    class Root(Space):
        o = Other()
        k = Kernel(x=o)

    with pytest.raises(
        DefinitionError,
        match=r"k\.port\.channel: forwards x .*, a Other input of .*Kernel; Port\.channel "
        r"takes a Chan",
    ):
        design_space(Root())


def test_a_forward_of_a_subclass_is_accepted() -> None:
    class Wide(Chan):
        pass

    class Kernel(Space):
        x: Wide = Param()
        port = Port(channel=x)

    class Root(Space):
        w = Wide(width=5)
        k = Kernel(x=w)

    assert design_space(Root()).k.port.width == 5


def test_a_node_of_another_space_class_is_still_refused_at_the_call() -> None:
    with pytest.raises(DefinitionError, match="expected a Chan node, got Other"):
        Port(channel=Other())  # type: ignore[arg-type]


def test_a_value_forward_of_another_type_is_still_refused_in_the_class_body() -> None:
    with pytest.raises(DefinitionError, match="binding has incompatible value semantics"):

        class Wrong(Space):
            label: str = Param()
            child = Chan(width=label)  # type: ignore[arg-type]


def test_only_an_unresolved_annotation_is_deferred() -> None:
    """Any other error of a formal or Decision is reported where its class is defined."""
    with pytest.raises(DefinitionError, match="a reference input has no value default"):

        class Holder(Space):
            c: Chan = Param(default=None)  # type: ignore[assignment]

    with pytest.raises(DefinitionError, match="Bare.choice .*annotate the decision"):

        class Bare(Space):
            choice = Decision(values=(1, 2))
