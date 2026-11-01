# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shell rows: what a declared shell gives a partition on one of its boards, as typed
facts, and what it costs.

A row (``ShellRow``) is what one shell declaration (``finn.platform.declarations``)
gives on one of its boards: the shell's facts, with any fact its entry for that board
states in their place. The facts:

- ``clocks`` (``Clocks``): the clock it gives ``ap_clk``, who sets its frequency, and
  whether it gives an aligned doubled clock (``ap_clk2x``);
- ``control`` (``Control``): how the platform reaches a unit's control port, the least
  window a unit takes and the aperture all units share (``None``: nothing is addressed);
- ``memory``: the memory targets a mover may bind (``MemoryTarget``);
- ``units``: the most units the integration may place (``None``: no bound);
- ``supply`` and ``delivery``: the kinds offered on input and output edge ports, each
  with its mover kinds (``Offer``);
- ``linker`` and ``runtimes``: the declared linker that links it (``None``: its user
  does) and the runtimes it offers, the first the default;
- ``versions``: the linker's tool releases, narrowed for this board (``Versions``);
- ``board_facts``: what the linker needs of this board (its ``board_keys``);
- ``fixed``: its own logic and that logic's cost (``FixedShell``);
- ``characterised`` and ``caveat``: where its cost was characterised and the shell built
  and timed, and what that does not cover on this board;
- ``control_budget`` and ``memory_ports``: the AXI-Lite buses the partition and its
  movers may present together, and the memory ports the partition may initiate beside
  its movers' (``None``: no bound), which the partition's shell rules read
  (``finn.kernels.budgets``).

``ip`` is the packaged IP for the user to integrate, on any part. ``pynq`` is the Zynq
block design, one row per board it supports. Any other shell is a declaration's.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from string import Formatter
from typing import Any, TypeVar

from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.module import Fragment
from finn.kernels.artifacts.top import doubled, top_buses
from finn.kernels.movers import PLACED_KINDS, MoverOffer
from finn.kernels.utilization import Fit
from finn.platform import _stated as valid
from finn.platform.capabilities import RESOURCE_NAMES, Resources
from finn.platform.declarations import Declarations, Shell, code, declarations
from finn.platform.refusal import TargetRefused

IP, PYNQ = "ip", "pynq"
"""FINN's own shells: the packaged IP, the default, and the Zynq block design."""

VIVADO_BLOCK_DESIGN = "vivado-block-design"
"""The name FINN's own linker of the Zynq block design is declared under."""

MOVED, FETCHED, PUSHED, CHAINED = "moved", "fetched", "pushed", "chained"
"""How an edge port's data reaches the core or leaves it: moved by a mover once a frame,
fetched every frame from memory written while idle, pushed by the host or a peer, or
chained to another core's edge port (stated, never offered)."""

SUPPLY_KINDS = (MOVED, FETCHED, PUSHED)
"""The kinds a shell may offer an input edge port."""
DELIVERY_KINDS = (MOVED, PUSHED)
"""The kinds a shell may offer an output edge port."""

BUILD, RUNTIME = "build", "runtime"
"""Who sets a shell's clock: the build, or the runtime at load (``Clocks.set_at``)."""


@dataclass(frozen=True, kw_only=True)
class Clocks:
    """The clock a shell gives ``ap_clk`` (``main``), who sets its frequency (``set_at``:
    ``build`` or ``runtime``), what it runs at unless set (``default_mhz``; ``None``:
    unstated), and whether it gives an aligned ``ap_clk2x`` (``doubled``)."""

    main: str
    set_at: str
    default_mhz: float | None
    doubled: bool


@dataclass(frozen=True, kw_only=True)
class Control:
    """How a shell addresses units: whether the platform reaches a unit by one control port
    (``one_port_per_unit``), the least window a unit takes, aligned to itself
    (``least_window``, bytes), and the bytes the host addresses for all units together
    (``aperture``)."""

    one_port_per_unit: bool
    least_window: int
    aperture: int


@dataclass(frozen=True, kw_only=True)
class MemoryTarget:
    """A memory target a mover may bind: its ``name``, its port's ``width`` (bits, a
    mover's width cap) and the most mover ports it takes (``masters``; ``None``: no
    bound)."""

    name: str
    width: int
    masters: int | None


@dataclass(frozen=True, kw_only=True)
class MoverKind:
    """A mover kind a shell offers: its declared ``name`` and its ``code``
    (``module:attr``), imported where a step places it."""

    name: str
    code: str


@dataclass(frozen=True, kw_only=True)
class Offer:
    """A kind a shell offers its edge ports (``kind``: ``moved``, ``fetched`` or
    ``pushed``), the mover kinds that serve it there, the first the default (``movers``),
    and the widest memory port they may take (``width_cap``, bits; ``None``: no cap)."""

    kind: str
    movers: tuple[MoverKind, ...]
    width_cap: int | None


@dataclass(frozen=True, kw_only=True)
class Versions:
    """The releases of a ``tool`` a row builds with: ``spec``, comparisons of a release
    (``year.minor``) joined by commas, each of which it meets (``>=2024.1``)."""

    tool: str
    spec: str

    def admits(self, release: tuple[int, int]) -> bool:
        """Whether ``release`` meets every comparison of ``spec``."""
        return all(_compare(release, clause) for clause in valid.clauses(self.spec))


def _compare(release: tuple[int, int], clause: tuple[str, tuple[int, int]]) -> bool:
    operator, against = clause
    return {
        ">=": release >= against,
        "<=": release <= against,
        "==": release == against,
        "!=": release != against,
        ">": release > against,
        "<": release < against,
    }[operator]


@dataclass(frozen=True, kw_only=True)
class Scaled:
    """An IP's resources by the count that scales it: ``fixed``, plus LUTs and FFs by a
    ``Fit`` over the count (``lut``, ``ff``) from ``scales_from`` on, none below it."""

    fixed: Resources = Resources()
    lut: Fit = Fit(0, (0,))
    ff: Fit = Fit(0, (0,))
    scales_from: int = 1

    def at(self, count: int) -> Resources:
        if count < self.scales_from:
            return self.fixed
        return self.fixed + Resources(lut=self.lut.at(count), ff=self.ff.at(count))


@dataclass(frozen=True, kw_only=True)
class FixedShell:
    """A shell's own logic beside the partition and its movers, each IP with its
    resources out of context. The memory interconnect takes one master per memory
    port, and the control interconnect gives one slave per AXI-Lite bus. The processor
    addresses the AXI-Lite buses from ``control_base``, in connection order, each
    aligned to its window, at least the row's ``Control.least_window``."""

    processor: str
    reset: str
    memory_interconnect: str
    control_interconnect: str
    control_base: int
    processor_use: Resources
    reset_use: Resources
    memory_interconnect_use: Scaled
    control_interconnect_use: Scaled

    def resources(self, *, masters: int, slaves: int) -> tuple[tuple[str, Resources], ...]:
        """What each IP uses, by its Vivado IP name, with ``masters`` memory ports and
        ``slaves`` AXI-Lite buses connected. The template connects at least one of each."""
        if masters < 1 or slaves < 1:
            raise ValueError(
                f"a fixed shell connects at least one memory port and one AXI-Lite bus, "
                f"not {masters} and {slaves}"
            )
        return (
            (self.processor, self.processor_use),
            (self.reset, self.reset_use),
            (self.memory_interconnect, self.memory_interconnect_use.at(masters)),
            (self.control_interconnect, self.control_interconnect_use.at(slaves)),
        )


@dataclass(frozen=True, kw_only=True)
class ShellRow:
    """What a shell gives and takes on one board (``board``: ``None`` for the row of any
    part): its typed facts (the module docstring)."""

    shell: str
    board: str | None
    board_facts: Mapping[str, str]
    clocks: Clocks
    control: Control | None
    memory: tuple[MemoryTarget, ...]
    units: int | None
    supply: tuple[Offer, ...]
    delivery: tuple[Offer, ...]
    linker: str | None
    runtimes: tuple[str, ...]
    versions: Versions | None
    fixed: FixedShell | None
    characterised: str | None
    caveat: str | None
    control_budget: int | None
    memory_ports: int | None

    @property
    def offers(self) -> tuple[Offer, ...]:
        """Its supply offers, then its delivery offers."""
        return self.supply + self.delivery


def bus_counts(fragment: Fragment, boundary: Iterable[Bus]) -> tuple[int, int, bool]:
    """What the top of ``fragment`` and its ``boundary`` buses asks of a shell, counted
    from the pins it would have (``finn.kernels.artifacts.top``) without building it: the
    AXI-Lite buses it presents, the AXI memory ports it initiates (a bus beside its streams
    that it initiates: a boundary bus, since each presented bus is a target port), and
    whether it takes an aligned doubled clock (an instance's doubled clock, ``doubled``).
    The budgets are rules of the partition in its shell, over its members' derived values
    (``finn.kernels.budgets``); emission checks this count against theirs, and the fixed
    shell is sized by it."""
    buses = top_buses(fragment, boundary)
    control = sum(bus.protocol is StandardProtocol.AXILITE for bus in buses)
    memory = sum(
        bus.protocol is not StandardProtocol.AXIS and bus.endpoint is Endpoint.INITIATOR
        for bus in buses
    )
    return control, memory, doubled(fragment)


# -- a row from its declaration ----------------------------------------------------------

T = TypeVar("T")

_FACTS = {
    "clocks",
    "control",
    "memory",
    "units",
    "supply",
    "delivery",
    "fixed",
    "characterised",
    "timed",
    "caveat",
    "control_budget",
    "memory_ports",
}
"""The facts a shell states, and its entry for a board may state in their place."""


def _clocks(stated: object, where: str) -> Clocks:
    table = valid.expect(stated, where, {"main", "set_at", "doubled"}, {"default_mhz"})
    set_at = table["set_at"]
    if set_at not in (BUILD, RUNTIME):
        raise valid.Invalid(f"{where}.set_at is {BUILD!r} or {RUNTIME!r}, not {set_at!r}")
    default = table.get("default_mhz")
    return Clocks(
        main=valid.text(table["main"], f"{where}.main"),
        set_at=set_at,
        default_mhz=None if default is None else valid.positive(default, f"{where}.default_mhz"),
        doubled=valid.typed(table["doubled"], bool, f"{where}.doubled"),
    )


def _control(stated: object, where: str) -> Control:
    """A shell's control: a unit's least window and the host's aperture, each a power of
    two of bytes (a window is aligned to itself), the aperture holding one window at least."""
    table = valid.expect(stated, where, {"one_port_per_unit", "least_window", "aperture"}, set())
    least = valid.power_of_two(table["least_window"], f"{where}.least_window")
    aperture = valid.power_of_two(table["aperture"], f"{where}.aperture")
    if aperture < least:
        raise valid.Invalid(
            f"{where}.aperture: {aperture} bytes hold no window of the least {least}"
        )
    return Control(
        one_port_per_unit=valid.typed(
            table["one_port_per_unit"], bool, f"{where}.one_port_per_unit"
        ),
        least_window=least,
        aperture=aperture,
    )


def _memory(stated: object, where: str) -> tuple[MemoryTarget, ...]:
    """A shell's memory targets: each named once, its width an AXI data width (a mover's
    memory port binds it)."""
    targets: list[MemoryTarget] = []
    for index, each in enumerate(valid.typed(stated, list, where)):
        here = f"{where}[{index}]"
        table = valid.expect(each, here, {"name", "width"}, {"masters"})
        masters = table.get("masters")
        name = valid.text(table["name"], f"{here}.name")
        named = [at for at, target in enumerate(targets) if target.name == name]
        if named:
            raise valid.Invalid(f"{here}.name: {name!r} names {where}[{named[0]}] too")
        targets.append(
            MemoryTarget(
                name=name,
                width=valid.axi_width(table["width"], f"{here}.width"),
                masters=None
                if masters is None
                else valid.count(masters, f"{here}.masters", least=1),
            )
        )
    return tuple(targets)


def _offers(
    stated: object, where: str, kinds: tuple[str, ...], declared: Declarations
) -> tuple[Offer, ...]:
    offers: list[Offer] = []
    for index, each in enumerate(valid.typed(stated, list, where)):
        here = f"{where}[{index}]"
        table = valid.expect(each, here, {"kind", "movers"}, {"width_cap"})
        kind = table["kind"]
        if kind not in kinds:
            raise valid.Invalid(f"{here}.kind is one of {', '.join(kinds)}, not {kind!r}")
        if kind in (offer.kind for offer in offers):
            raise valid.Invalid(f"{here}: {where} offers the {kind} kind once")
        movers = []
        for name in valid.texts(table["movers"], f"{here}.movers"):
            if name not in declared.movers:
                raise valid.Invalid(
                    f"{here}.movers: {name!r} is not a declared mover kind "
                    f"(one of {sorted(declared.movers)})"
                )
            movers.append(MoverKind(name=name, code=declared.movers[name].code))
        cap = table.get("width_cap")
        if cap is not None:
            valid.axi_width(cap, f"{here}.width_cap")
        offers.append(Offer(kind=kind, movers=tuple(movers), width_cap=cap))
    return tuple(offers)


def _resources(table: Mapping[str, Any], where: str) -> Resources:
    counts = {key: table[key] for key in RESOURCE_NAMES if key in table}
    return Resources(**{key: valid.count(value, f"{where}.{key}") for key, value in counts.items()})


def _fit(stated: object, where: str) -> Fit:
    """A scaled IP's fit over its one feature, the count that scales it: ``[constant,
    slope]``, each a finite number."""
    numbers = valid.typed(stated, list, where)
    if len(numbers) != 2 or not all(
        isinstance(each, (int, float)) and not isinstance(each, bool) and math.isfinite(each)
        for each in numbers
    ):
        raise valid.Invalid(
            f"{where} is a fit over the count that scales it, [constant, slope], two finite "
            f"numbers, not {stated!r}"
        )
    return Fit(float(numbers[0]), (float(numbers[1]),))


def _use(stated: object, where: str) -> tuple[str, Resources]:
    table = valid.expect(stated, where, {"ip"}, set(RESOURCE_NAMES))
    return valid.text(table["ip"], f"{where}.ip"), _resources(table, where)


def _scaled(stated: object, where: str) -> tuple[str, Scaled]:
    table = valid.expect(stated, where, {"ip"}, {"fixed", "lut", "ff", "scales_from"})
    fixed = table.get("fixed", {})
    valid.expect(fixed, f"{where}.fixed", set(), set(RESOURCE_NAMES))
    return valid.text(table["ip"], f"{where}.ip"), Scaled(
        fixed=_resources(fixed, f"{where}.fixed"),
        lut=_fit(table["lut"], f"{where}.lut") if "lut" in table else Fit(0, (0,)),
        ff=_fit(table["ff"], f"{where}.ff") if "ff" in table else Fit(0, (0,)),
        scales_from=valid.count(table.get("scales_from", 1), f"{where}.scales_from", least=1),
    )


def _fixed(stated: object, where: str) -> FixedShell:
    table = valid.expect(
        stated,
        where,
        {"control_base", "processor", "reset", "memory_interconnect", "control_interconnect"},
        set(),
    )
    processor, processor_use = _use(table["processor"], f"{where}.processor")
    reset, reset_use = _use(table["reset"], f"{where}.reset")
    memory, memory_use = _scaled(table["memory_interconnect"], f"{where}.memory_interconnect")
    control, control_use = _scaled(table["control_interconnect"], f"{where}.control_interconnect")
    return FixedShell(
        processor=processor,
        reset=reset,
        memory_interconnect=memory,
        control_interconnect=control,
        control_base=valid.count(table["control_base"], f"{where}.control_base"),
        processor_use=processor_use,
        reset_use=reset_use,
        memory_interconnect_use=memory_use,
        control_interconnect_use=control_use,
    )


def _caveat(stated: object, where: str) -> str:
    """A shell's ``caveat``: a template naming its board (``{board}``) and its part
    (``{part}``), and nothing else."""
    template = valid.text(stated, where)
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None and (name not in ("board", "part") or spec or conversion):
            raise valid.Invalid(
                f"{where} names {{board}} and {{part}} only, not {{{name}}} in {template!r}"
            )
    return template


def _facts(stated: Mapping[str, Any], where: str, declared: Declarations) -> dict[str, Any]:
    """Each of the facts ``stated`` (a shell's, or a board entry's), read for its type."""
    readers: dict[str, Callable[[object, str], object]] = {
        "clocks": _clocks,
        "control": _control,
        "memory": _memory,
        "units": valid.count,
        "supply": partial(_offers, kinds=SUPPLY_KINDS, declared=declared),
        "delivery": partial(_offers, kinds=DELIVERY_KINDS, declared=declared),
        "fixed": _fixed,
        "characterised": valid.text,
        "timed": valid.texts,
        "caveat": _caveat,
        "control_budget": valid.count,
        "memory_ports": valid.count,
    }
    unknown = sorted(set(stated) - _FACTS)
    if unknown:
        raise valid.Invalid(f"{where} states {', '.join(unknown)}, which a shell does not have")
    return {key: readers[key](value, f"{where}.{key}") for key, value in stated.items()}


def _bound(facts: Mapping[str, Any], where: str) -> None:
    """Refuse a moved offer no memory target of the row can bind: a row with none, or a
    width cap wider than every target's port (a mover's memory port binds one)."""
    targets: tuple[MemoryTarget, ...] = facts.get("memory", ())
    widest = max((target.width for target in targets), default=None)
    for side in ("supply", "delivery"):
        for index, offer in enumerate(facts.get(side, ())):
            if offer.kind != MOVED:
                continue
            here = f"{where}.{side}[{index}]"
            if widest is None:
                raise valid.Invalid(f"{here}: a moved offer, and no memory target to bind it")
            if offer.width_cap is not None and offer.width_cap > widest:
                listed = ", ".join(f"{target.name} {target.width}" for target in targets)
                raise valid.Invalid(
                    f"{here}.width_cap: {offer.width_cap} bits is wider than every memory "
                    f"target ({listed} bits)"
                )


def _row(shell: Shell, board: str | None, declared: Declarations) -> ShellRow:
    """The row of ``shell`` on ``board`` (``None`` on a shell of any part), its board
    entry's facts in place of the shell's."""
    where = f"shells.{shell.name}"
    facts = _facts(shell.facts, where, declared)
    linker = None if shell.linker is None else declared.linkers.get(shell.linker)
    board_facts: dict[str, str] = {}
    narrowed = None
    if shell.linker is not None and linker is None:
        raise valid.Invalid(f"{where}: its linker {shell.linker!r} is not declared")
    if board is None and linker is not None and linker.board_keys:
        raise valid.Invalid(
            f"{where} supports any part, and its linker {linker.name!r} needs "
            f"{', '.join(linker.board_keys)} of each board: its boards are a table of entries"
        )
    if board is not None and not isinstance(shell.boards, str):
        entry = shell.boards[board]
        here = f"{where}.boards.{board}"
        keys = set() if linker is None else set(linker.board_keys)
        valid.expect(entry, here, keys, _FACTS | {"versions"})
        board_facts = {key: valid.text(entry[key], f"{here}.{key}") for key in sorted(keys)}
        narrowed = (
            None
            if "versions" not in entry
            else valid.releases(entry["versions"], f"{here}.versions")
        )
        stated = {key: value for key, value in entry.items() if key in _FACTS}
        facts.update(_facts(stated, here, declared))
    if "clocks" not in facts:
        raise valid.Invalid(f"{where} states no clocks")
    _bound(facts, where)
    specs = [each for each in (None if linker is None else linker.versions, narrowed) if each]
    versions = None
    if linker is not None and specs:
        versions = Versions(tool=linker.tool, spec=",".join(specs))
    template = facts.get("caveat")
    caveat = None
    if board is not None and board not in facts.get("timed", ()) and template is not None:
        caveat = template.format(board=board, part=declared.board(board).part)
    return ShellRow(
        shell=shell.name,
        board=board,
        board_facts=board_facts,
        clocks=facts["clocks"],
        control=facts.get("control"),
        memory=facts.get("memory", ()),
        units=facts.get("units"),
        supply=facts.get("supply", ()),
        delivery=facts.get("delivery", ()),
        linker=shell.linker,
        runtimes=shell.runtimes,
        versions=versions,
        fixed=facts.get("fixed"),
        characterised=facts.get("characterised"),
        caveat=caveat,
        control_budget=facts.get("control_budget"),
        memory_ports=facts.get("memory_ports"),
    )


def _refusing(shell: Shell, declared: Declarations, read: Callable[[], T]) -> T:
    try:
        return read()
    except (ValueError, KeyError) as error:
        origin = declared.origin("shells", shell.name)
        raise TargetRefused(
            "declaration-invalid", f"{origin.file}: shells.{shell.name}", str(error)
        ) from None


def row(shell: Shell, board: str | None, declared: Declarations) -> ShellRow:
    """The row of the declared ``shell`` on ``board`` (one it supports, or ``None`` on a
    shell of any part); a fact not of its type is refused (``declaration-invalid``), the
    shell's own read apart from the board's, so a board entry stating a fact in its place
    does not hide it. A shell of any part whose linker needs keys of a board is refused."""
    return _refusing(shell, declared, partial(_row, shell, board, declared))


def own_facts(shell: Shell, declared: Declarations) -> dict[str, Any]:
    """The facts ``shell`` itself states, apart from any board, each read for its type;
    refused where one is not (``declaration-invalid``). ``finn.platform.check`` reads them
    whatever boards the shell supports."""
    return _refusing(
        shell, declared, partial(_facts, shell.facts, f"shells.{shell.name}", declared)
    )


# -- the offer a core places --------------------------------------------------------------


def placed_offer(row: ShellRow) -> Offer | None:
    """The offer a core places on its boundary channels in ``row`` (``None``: it offers
    none). Until the movers are kernels of their own, a core places one offer on every
    boundary side, inputs and outputs alike: one mover kind it places (``PLACED_KINDS``),
    moved, at a bounded width cap. A row offering anything else is refused
    (``offer-unplaced``), naming what it offers, rather than placed as something it does
    not offer."""
    if not row.offers:
        return None
    why = []
    others = sorted({offer.kind for offer in row.offers if offer.kind != MOVED})
    if others:
        why.append(f"it offers the {', '.join(others)} kind, which a core does not place yet")
    supply = [offer for offer in row.supply if offer.kind == MOVED]
    delivery = [offer for offer in row.delivery if offer.kind == MOVED]
    if supply != delivery:
        why.append(
            f"its inputs are offered {_said(supply)} and its outputs {_said(delivery)}: a "
            "core places one offer on both"
        )
    for offer in {*supply, *delivery}:
        if offer.width_cap is None:
            why.append(f"{_said([offer])} has no width cap, and a core's mover needs one")
        if len(offer.movers) != 1:
            why.append(f"{_said([offer])} names {len(offer.movers)} mover kinds, not one")
        unplaced = [mover.name for mover in offer.movers if mover.name not in PLACED_KINDS]
        if unplaced:
            why.append(
                f"a core places {', '.join(PLACED_KINDS)} movers alone yet, not "
                f"{', '.join(unplaced)}"
            )
    if why:
        raise TargetRefused("offer-unplaced", f"shells.{row.shell}", "; ".join(why))
    return supply[0] if supply else None


def _said(offers: list[Offer]) -> str:
    if not offers:
        return "nothing moved"
    return " and ".join(
        f"{offer.kind} by {', '.join(mover.name for mover in offer.movers)}"
        + ("" if offer.width_cap is None else f" up to {offer.width_cap} bits")
        for offer in offers
    )


def mover_offer(mover: MoverKind, width_cap: int) -> MoverOffer:
    """What a boundary side is offered by ``mover``, a moved mover kind, at ``width_cap``:
    its declared code's ``offer(width_cap)`` (``finn.platform.declarations.code``), a
    ``MoverOffer`` of its kind at that cap. Code that does not keep that contract, or
    raises, is refused (``mover-offer-invalid``)."""
    found = code(mover.code)
    try:
        made = getattr(found, "offer")(width_cap)
    except Exception as error:  # declared code, which may raise anything
        raise TargetRefused(
            "mover-offer-invalid",
            mover.code,
            f"{mover.code}.offer({width_cap}) does not make an offer: "
            f"{type(error).__name__}: {error}",
        ) from None
    if not (
        isinstance(made, MoverOffer) and made.kind == mover.name and made.width_cap == width_cap
    ):
        raise TargetRefused(
            "mover-offer-invalid",
            mover.code,
            f"{mover.code}.offer({width_cap}) made {made!r}, not a mover's offer of the "
            f"{mover.name!r} kind at {width_cap} bits",
        )
    return made


def shell_row(shell: str, board: str | None, declared: Declarations | None = None) -> ShellRow:
    """The row of ``shell`` for ``board``, from ``declared`` (the process's declarations
    by default). Refuses a shell nothing declares (``unknown-shell``), a missing board
    on a shell built for one (``board-required``), and a board the shell does not support
    (``board-unsupported``, naming the shells that do). On a shell of any part the board
    is not read."""
    declared = declarations() if declared is None else declared
    found = declared.shell(shell)
    if found.any_board:
        return row(found, None, declared)
    if board is None:
        supported = sorted(found.boards)
        raise TargetRefused(
            "board-required",
            "configuration.target.board",
            f"the {shell!r} shell is built for a board: name one of {supported}",
        )
    known = declared.board(board)
    if not found.supports(board):
        others = [name for name in declared.supporting(board) if name != shell]
        raise TargetRefused(
            "board-unsupported",
            "configuration.target.board",
            f"the {shell!r} shell does not support board {board!r} ({known.part}); "
            f"{_listed(others)} supports it",
        )
    return row(found, board, declared)


def _listed(shells: list[str]) -> str:
    if not shells:
        return "no declared shell"
    quoted = [repr(each) for each in shells]
    if len(quoted) == 1:
        return f"only {quoted[0]}"
    return "each of " + ", ".join(quoted)


def shell_rows(declared: Declarations | None = None) -> dict[tuple[str, str | None], ShellRow]:
    """Every row of ``declared`` (the process's by default), by (shell, board): a shell of
    any part has one, its board ``None``."""
    declared = declarations() if declared is None else declared
    rows: dict[tuple[str, str | None], ShellRow] = {}
    for name, shell in declared.shells.items():
        boards = [None] if isinstance(shell.boards, str) else list(shell.boards)
        for board in boards:
            rows[name, board] = row(shell, board, declared)
    return rows


__all__ = [
    "BUILD",
    "CHAINED",
    "DELIVERY_KINDS",
    "FETCHED",
    "IP",
    "MOVED",
    "PUSHED",
    "PYNQ",
    "RUNTIME",
    "SUPPLY_KINDS",
    "VIVADO_BLOCK_DESIGN",
    "Clocks",
    "Control",
    "FixedShell",
    "MemoryTarget",
    "MoverKind",
    "Offer",
    "Scaled",
    "ShellRow",
    "Versions",
    "bus_counts",
    "mover_offer",
    "own_facts",
    "placed_offer",
    "row",
    "shell_row",
    "shell_rows",
]
