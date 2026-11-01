# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Whether a load of declarations (``finn.platform.declarations``) agrees with itself
and with the part catalog, its parts overlays laid over it: every finding by code, owned
by the declaration it is about (``<file>: <kind>.<name>``). ``finn-platform check``
prints them.

- ``unknown-part``: a board's part is not in the catalog;
- ``linker-undeclared``, ``runtime-undeclared``: a shell names a linker or a runtime
  nothing declares (a mover kind nothing declares is ``declaration-invalid``, when its row
  is read);
- ``board-undeclared``: a shell supports a board nothing declares;
- ``declaration-invalid``: a fact a shell states does not read, whatever boards it
  supports, or its row on one of its boards does not (a fact not of its type, a board
  entry without a key its linker needs, or no board for a linker that needs one);
- ``image-unloaded``: a runtime a shell offers does not load its linker's image;
- ``mover-undriven``: a runtime a shell offers does not drive a mover kind the shell offers;
- ``kind-unserved``: a runtime a shell offers, or a mover kind it offers for a kind, does
  not serve that kind;
- ``offer-unplaced``: a shell offers what a core does not place yet
  (``finn.platform.shells.placed_offer``): a kind other than moved, different offers on
  its inputs and its outputs, no width cap, more than one mover kind, or a mover kind
  other than those a core places (``finn.kernels.movers.PLACED_KINDS``);
- ``code-unimportable``: a linker's, runtime's or mover kind's ``code`` does not import;
- ``code-not-callable``: it imports, and is not a class or a function;
- ``mover-offer-invalid``: a moved mover kind's code has no ``offer``, or its offer at the
  width cap of a shell offering it is not a mover's offer of its kind at that cap
  (``finn.platform.shells.mover_offer``), which a core's boundary takes.

The code is checked for what FINN calls of it now: a linker's and a runtime's to be
callable, and a moved mover kind's offer. That a linker or a runtime does its work is
not checked until FINN calls them through their declarations.

Checking imports each declaration's code (``finn.platform.declarations.code``); reading
and resolving imports none.
"""

from __future__ import annotations

from finn.core.reporting.findings import Finding, FindingKind
from finn.platform.catalog import catalog_for
from finn.platform.declarations import Declarations, code
from finn.platform.refusal import TargetRefused
from finn.platform.shells import MOVED, MoverKind, mover_offer, own_facts, placed_offer, row


def _finding(declared: Declarations, kind: str, name: str, reason: str, message: str) -> Finding:
    owner = f"{declared.origin(kind, name).file}: {kind}.{name}"
    return Finding(FindingKind.REJECTION, reason, owner, message)


def _refusal(refused: TargetRefused) -> list[Finding]:
    return list(refused.findings)


def check(declared: Declarations) -> list[Finding]:
    """Every finding of ``declared`` (the module docstring), in order of kind and name."""
    findings: list[Finding] = []
    parts = catalog_for(declared)
    for name, declared_board in sorted(declared.boards.items()):
        try:
            parts.part(declared_board.part)
        except TargetRefused as refused:
            findings.append(
                _finding(declared, "boards", name, "unknown-part", str(refused.findings[0]))
            )
    unplaced: dict[tuple[str, str], None] = {}
    capped: dict[tuple[str, MoverKind, int], None] = {}
    for name, shell in sorted(declared.shells.items()):
        if shell.linker is not None and shell.linker not in declared.linkers:
            findings.append(
                _finding(
                    declared,
                    "shells",
                    name,
                    "linker-undeclared",
                    f"its linker {shell.linker!r} is not declared "
                    f"(one of {sorted(declared.linkers)})",
                )
            )
        for named in shell.runtimes:
            if named not in declared.runtimes:
                findings.append(
                    _finding(
                        declared,
                        "shells",
                        name,
                        "runtime-undeclared",
                        f"its runtime {named!r} is not declared "
                        f"(one of {sorted(declared.runtimes)})",
                    )
                )
        boards: list[str | None] = [None]
        if not isinstance(shell.boards, str):
            boards = list(sorted(shell.boards))
        unknown = [each for each in boards if each is not None and each not in declared.boards]
        for board in unknown:
            findings.append(
                _finding(
                    declared,
                    "shells",
                    name,
                    "board-undeclared",
                    f"it supports board {board!r}, which is not declared",
                )
            )
        rows = []
        try:
            own_facts(shell, declared)
        except TargetRefused as refused:
            findings += _refusal(refused)
            boards = []
        if shell.linker is not None and shell.linker not in declared.linkers:
            boards = []  # its rows need their linker: linker-undeclared says so
        refused_rows: dict[Finding, None] = {}  # one refusal of every board's row, once
        for each in boards:
            if each in unknown:
                continue
            try:
                rows.append(row(shell, each, declared))
            except TargetRefused as refused:
                refused_rows.update(dict.fromkeys(_refusal(refused)))
        findings += refused_rows
        for runtime_name in shell.runtimes:
            runtime = declared.runtimes.get(runtime_name)
            if runtime is None:
                continue
            linker = None if shell.linker is None else declared.linkers.get(shell.linker)
            if linker is not None and linker.image not in runtime.images:
                findings.append(
                    _finding(
                        declared,
                        "shells",
                        name,
                        "image-unloaded",
                        f"its runtime {runtime_name!r} loads {list(runtime.images)}, not its "
                        f"linker {linker.name!r}'s {linker.image!r}",
                    )
                )
            offers = {offer for each in rows for offer in each.offers}
            for offer in sorted(offers, key=lambda offer: offer.kind):
                if offer.kind not in runtime.kinds:
                    findings.append(
                        _finding(
                            declared,
                            "shells",
                            name,
                            "kind-unserved",
                            f"its runtime {runtime_name!r} serves {list(runtime.kinds)}, not "
                            f"the {offer.kind!r} kind it offers",
                        )
                    )
                for mover in offer.movers:
                    if mover.name not in runtime.movers:
                        findings.append(
                            _finding(
                                declared,
                                "shells",
                                name,
                                "mover-undriven",
                                f"its runtime {runtime_name!r} drives {list(runtime.movers)}, "
                                f"not the {mover.name!r} mover kind it offers",
                            )
                        )
        for read in rows:
            try:
                placed_offer(read)
            except TargetRefused as refused:
                unplaced.setdefault((name, refused.findings[0].message), None)
            for offer in read.offers:
                if offer.kind == MOVED and offer.width_cap is not None:
                    for mover in offer.movers:
                        capped.setdefault((name, mover, offer.width_cap), None)
        offers = {offer for each in rows for offer in each.offers}
        for offer in sorted(offers, key=lambda offer: offer.kind):
            for mover in offer.movers:
                kinds = declared.movers[mover.name].kinds
                if offer.kind not in kinds:
                    findings.append(
                        _finding(
                            declared,
                            "shells",
                            name,
                            "kind-unserved",
                            f"its mover kind {mover.name!r} serves {list(kinds)}, not the "
                            f"{offer.kind!r} kind it is offered for",
                        )
                    )
    for name, message in unplaced:
        findings.append(_finding(declared, "shells", name, "offer-unplaced", message))
    coded: list[tuple[str, str, str]] = [
        *(("linkers", name, each.code) for name, each in declared.linkers.items()),
        *(("runtimes", name, each.code) for name, each in declared.runtimes.items()),
        *(("movers", name, each.code) for name, each in declared.movers.items()),
    ]
    offering: set[str] = set()
    for kind, name, reference in sorted(coded):
        try:
            found = code(reference)
        except TargetRefused as refused:
            findings.append(
                _finding(declared, kind, name, "code-unimportable", str(refused.findings[0]))
            )
            continue
        if not callable(found):
            findings.append(
                _finding(
                    declared,
                    kind,
                    name,
                    "code-not-callable",
                    f"{reference} is {type(found).__name__}, not a class or a function",
                )
            )
        elif kind == "movers" and MOVED in declared.movers[name].kinds:
            if callable(getattr(found, "offer", None)):
                offering.add(name)
            else:
                findings.append(
                    _finding(
                        declared,
                        kind,
                        name,
                        "mover-offer-invalid",
                        f"{reference} has no offer(width_cap), which a moved mover kind's "
                        "code has to make what it offers a boundary side",
                    )
                )
    for name, mover, cap in capped:
        if mover.name in offering:
            try:
                mover_offer(mover, cap)
            except TargetRefused as refused:
                message = refused.findings[0].message
                findings.append(_finding(declared, "shells", name, "mover-offer-invalid", message))
    return findings


__all__ = ["check"]
