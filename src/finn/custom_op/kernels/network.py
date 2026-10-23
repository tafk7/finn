# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition of KernelOp nodes a graph binds, one design space, and its choices
written back.

``network(model, nodes, movers=...)`` binds a ``Partition`` straight from the graph, for
any selection of KernelOp nodes (a partition's body, a slice of it, one node), and changes
no graph: it reads the graph once, its tensors from one index
(``finn.core.graph.facts.reading``), into a declaration (``declaration``), which the
composer (``finn.kernels.composer``) makes one Space class of, and binds its stored choices
on one design space of that class. The movers offered on its boundary (``movers``, each a
``finn.kernels.movers.MoverOffer``; none by default) are a fact its caller gives: a shell's
row offers them, and ``finn.custom_op.kernels.shell`` binds a partition in the shell of the
model's target (``in_shell``).

**The declaration** (``declaration``):

- **kernels**, one per node, named as the node: its kernel's facts, read once
  (``KernelOp.facts``);
- **channels**, one per ONNX tensor, named as the tensor, in node order: a node's graph
  inputs, the parameter channels it owns (named after the initializer), its outputs. Each
  has its two sides, a kernel's port (``Port``: the op's ``wiring``, an input on the
  consumer's side, an output on the producer's) or, where no port of the selection is on
  it, the channel's source (a parameter channel, whose kernel states its value) or the
  boundary (``Side.boundary``: its AXIS port, offered the movers given); its tensor (the
  graph's, ``edge_tensor``, at the graph's rank; a parameter channel's, the tensor and
  contents its kernel states for it, ``Stated``); and the target's platform. Only the
  selection's ONNX inputs and outputs are boundaries, named by the shell's convention
  ``s_axis_<i>`` and ``m_axis_<j>``. A channel has one consumer, so a tensor read more
  than once is refused by name before anything is declared (``tensor-fan-out``): read by
  two of the nodes (an edge, or a parameter both own), or read by one and at an output
  boundary too (a graph output, or read outside the nodes). Fan-out waits for a channel
  that forks. A node and a tensor, or two nodes, of one name are refused (``name-shared``):
  a member is named once, as the choices log ``kernel_choices.json`` names it; and so is
  what the composer refuses (``PartitionRefused``: its findings, as a ``KernelOpError``);
- **its order**: the channels are declared input boundary channels, the other channels,
  output boundary channels, which the composer ranks (each after the members it reads); its
  module places them input boundary channels, the other channels, the kernels, output
  boundary channels (``Reshape_0_out0``, ..., ``MatMul_0``, ...), its instance order; each
  member's node in the module is its member name (``member_name``: ``\\W`` as ``_``).

**Its stored choices** (bind is assignment): every node's attributes and every channel's
``finn.channel`` on its tensor (in a partition's body only, ``refuse_outside_body``), bound
with the engine's ``bind`` on the design space of the graph's facts, never refused there. A
stored value outside its domain over the current facts (or a key that is no choice) is
dropped as stale, with why (``Partition.stale``). A stored set that breaks a rule binds, and
the partition reads refused: the member's validity names the rule, and so do its readers
(``configured_network``, ``verify_partition``, and the seam's step, ``Seam.refusals``). A
channel whose choices wait on an open kernel choice holds them, waiting
(``inspection.Waiting``). Nothing is filled in. With every KernelOp of the model selected,
choices a tensor states that is no channel of it are stale too. Nothing is written:
``persist``, the one writer of choices, writes a partition's choices back, each owner's
whole.

**Its outputs**: each node's output is its kernel's product. Where the graph's annotation of
an output (its shape's element count and last axis, its datatype and its range) differs from
what its producer states, the graph is stale, refused by name (``tensor-stale``: "run
InferKernelTensors"), before any channel's own ``channel-tensor`` is read.

**Its module** is the partition's IP, the cut's (``PARTITION``): its members' netlists,
named as the graph's nodes and tensors, with the boundary's pins, named after the partition
(its stem ``finn_<name>``) and identified as the partition's IP (``PRODUCER``), since a
mover is integrated beside the IP, not in it. Its views are read out of the design space,
composed from its members' (the netlist with each channel's outer hops from the kernels'
contract views, FIFO sizing's paces, resources), never bound into it.

A partition's resources (``Partition.own_resources``) are its members'. The shell's budgets
and resources over a partition (its members', its movers' and its static region's), and the
cut's choices (``make_cut_choices``) on the partition in its shell, are
``finn.custom_op.kernels.shell``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, TypeVar

from onnx import NodeProto
from qonnx.core.metadata import MetadataError

from finn.core.graph.annotations.channel import (
    CHANNEL,
    CHANNEL_KEYS,
    channel_choices,
    refuse_outside_body,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor, initializer, reading
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    BrokenRule,
    ChangeRequest,
    ConfigurationError,
    Inapplicable,
    QueryResult,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ViewKey,
    bind,
    inspection,
    reject,
)
from finn.custom_op.kernels.base import KernelOp, kernel_op, typed
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed as Module
from finn.kernels.artifacts.module import Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose as top
from finn.kernels.base import BOUNDARY, NETLIST, RESOURCES, PortContract, merged
from finn.kernels.channels import Channel, Neighbours, OuterHops, outer_hops
from finn.kernels.composer import (
    ChannelDeclaration,
    Composed,
    Key,
    Names,
    PartitionDeclaration,
    PartitionRefused,
    Port,
    Stated,
    compose,
)
from finn.kernels.control import CONTROL, Control
from finn.kernels.control import presented as presented_buses
from finn.kernels.cut import Cut
from finn.kernels.explore import Trial, dotted
from finn.kernels.movers import MoverOffer
from finn.kernels.policies import Owner, Unstated
from finn.kernels.sides import Side
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

T = TypeVar("T")

PARTITION = "partition"
"""The partition's name: the cut's node and body file, and the IP made from it, which
is its module (``finn_partition``)."""

PRODUCER = ProducerIdentity("finn.custom_op.kernels.partition", "1")
"""What derives a partition's module: identified as the partition's, so the IP's name and
digest are the partition's."""


def _primitive(value: object) -> bool:
    """Whether a finding can hold ``value`` as a detail."""
    return type(value) in (bool, int, float, str)


def stale_text(key: str, found: Iterable[Finding]) -> str:
    """A stale choice as a line, its key (dotted, ``first.pe``) named once:
    ``first.pe = 3: not one of 1, 2, 4 (domain-membership)``, the value
    where the findings hold it, each finding's message and its code (a message that is its
    code, its code only)."""
    findings = tuple(found)
    values = [dict(each.details)["value"] for each in findings if "value" in dict(each.details)]
    named = f"{key} = {values[0]}" if values else key
    said = []
    for finding in findings:
        message = finding.message
        if values:
            message = message.removeprefix(f"{values[0]!r}: ")
        said.append(message if message == finding.code else f"{message} ({finding.code})")
    return f"{named}: " + "; ".join(said)


def _together(results: Iterable[QueryResult[Any]]) -> Rejected | Unresolved:
    """Results that are not available, as one: refused where one is, else waiting."""
    held = list(results)
    findings = tuple(
        dict.fromkeys(finding for each in held for finding in getattr(each, "findings", ()))
    )
    if any(isinstance(each, Rejected) for each in held):
        return Rejected(findings)
    return Unresolved(findings)


# -- the partition --------------------------------------------------------------------------


class Partition:
    """A partition bound from a graph's nodes (see the module docstring): one design space
    (``point``) of the class its declaration composes (``composed``: the class and its
    ``Names``), and what FINN reads of it, each choice keyed by its owner's graph name and its
    key there, ``(owner, key)`` (``Names`` maps a key to its path in the class).

    Its ``name`` (its module's), its ``boundary`` (each boundary tensor and its port, inputs
    then outputs), the boundary channels offered movers (``movers``), the stored choices
    dropped as stale where it was bound, by key, with the kernel's findings (``stale``: each
    a value outside its domain, keeping its code, its owner and its details, which add the
    value stored, ``value``, and the values the choice could take, ``candidates``;
    ``stale_text`` makes a line of them), and whether it holds every KernelOp of its model
    (``whole``). A rule its stored choices break leaves it reading refused: each member's
    ``validity`` names it.

    A member's answers are read as its own space's were: its keys, and the owners of its
    findings, relative to it (``pe``, ``transport.fifo.depth``); a finding of another member
    a member's answer carries (a channel waiting on its kernel's open choice) is owned by its
    key in that member and names the member (``node`` or ``tensor``, as ``located`` does).

    Immutable, as a design space: a change (``trial``) returns another, which keeps these
    but ``dropped``."""

    __slots__ = (
        "_composed",
        "_decisions",
        "_locator",
        "_memo",
        "_order",
        "_owners",
        "_point",
        "_sides",
        "boundary",
        "dropped",
        "movers",
        "name",
        "stale",
        "whole",
    )

    name: str
    boundary: tuple[tuple[str, str], ...]
    movers: tuple[str, ...]
    stale: Mapping[str, tuple[Finding, ...]]
    dropped: Mapping[Key, tuple[Finding, ...]]
    whole: bool
    _composed: Composed
    _point: Space
    _order: tuple[str, ...]
    _sides: Mapping[str, tuple[Port | Side, Port | Side]]
    _owners: Mapping[str, Owner]
    _decisions: Mapping[Key, inspection.DecisionInfo[object]]
    _locator: _Locator
    _memo: dict[str, object]

    def __init__(
        self,
        composed: Composed,
        point: Space,
        *,
        name: str,
        order: tuple[str, ...],
        sides: Mapping[str, tuple[Port | Side, Port | Side]],
        boundary: tuple[tuple[str, str], ...],
        movers: tuple[MoverOffer, ...],
        owners: Mapping[str, Owner],
        dropped: Mapping[Key, tuple[Finding, ...]],
        stray: Mapping[str, tuple[Finding, ...]],
        stored: Mapping[Key, object],
        whole: bool,
    ) -> None:
        self._composed, self._point = composed, point
        self._order = order
        self._sides = MappingProxyType(dict(sides))
        self._owners = MappingProxyType(dict(owners))
        self._locator = _Locator(composed.names, frozenset(sides))
        self._decisions = MappingProxyType(_decisions(composed, point))
        self._memo = {}
        self.name, self.boundary, self.whole = name, boundary, whole
        self.movers = tuple(tensor for tensor, _ in boundary) if movers else ()
        self.dropped = MappingProxyType(
            {
                key: tuple(self.local(key[0], each) for each in found)
                for key, found in dropped.items()
            }
        )
        self.stale = MappingProxyType(self._described(stored) | dict(stray))

    def at(self, point: Space) -> Self:
        """This partition at another design space of its class (a change's, or one its
        members' policies made, ``make_cut_choices``): what it keeps carried, ``dropped``
        none, what it read of the design space read again."""
        if type(point) is not self._composed.space_type:
            raise ValueError(f"{point!r} is no design space of {self!r}")
        made = object.__new__(type(self))
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if hasattr(self, slot):
                    object.__setattr__(made, slot, getattr(self, slot))
        made._point = point
        made._memo = {}
        made.dropped = MappingProxyType({})
        return made

    def _read(self, question: str, answer: Callable[[], Any]) -> Any:
        """``answer()``, read once of this partition (it is immutable)."""
        if question not in self._memo:
            self._memo[question] = answer()
        return self._memo[question]

    # -- its members ------------------------------------------------------------------------

    @property
    def composed(self) -> Composed:
        """Its class, its ``Names`` and the facts its root Params are given."""
        return self._composed

    @property
    def point(self) -> Space:
        """Its design space: the composed class's, its stored choices bound."""
        return self._point

    @property
    def members(self) -> tuple[str, ...]:
        """Every member, ranked: the kernels in graph order, then each channel after the
        members it reads."""
        return tuple(self._composed.names.members)

    @property
    def order(self) -> tuple[str, ...]:
        """Every member in its module's instance order."""
        return self._order

    @property
    def kernels(self) -> tuple[str, ...]:
        return tuple(name for name in self.members if name not in self._sides)

    @property
    def channels(self) -> tuple[str, ...]:
        """Its channels, ranked."""
        return tuple(name for name in self.members if name in self._sides)

    def node(self, name: str) -> str:
        """Member ``name``'s node in the module, its member name: the prefix of its
        instances and of the ports it presents."""
        return self._composed.names.members[name]

    def path(self, key: Key) -> str:
        """``key``'s path in the design space (``first.compute.packed.pe``)."""
        return self._composed.names.path(key)

    def pair(self, name: str, key: str) -> Key:
        """The key, by owner, of ``key``, a path in the design space a rule or a stored
        choice of an answer about member ``name`` is named by (another member's, where the
        rule or the choice is: a fold reaching its channel's rule)."""
        return self._composed.names.pair(key)

    def member(self, name: str) -> Any:
        """Member ``name``'s Space inside the design space."""
        member = self._composed.names.members.get(name)
        if member is None:
            raise ValueError(f"{name} is no member of the partition")
        return getattr(self._point, member)

    def owner(self, name: str) -> Owner:
        """The owner of member ``name``'s policies (``finn.kernels.policies.policy``): a
        channel's the cut (``finn.kernels.cut.Cut``), a kernel's its KernelOp class, or
        ``Unstated`` where it was given none."""
        if name in self._sides:
            return Cut
        if name not in self._composed.names.members:
            raise ValueError(f"{name} is no member of the partition")
        return self._owners.get(name, Unstated)

    def located(self, name: str, finding: Finding) -> Finding:
        """A finding of member ``name``'s, its owner kept, with the member named in its
        details (a finding of another member's names that one already): a channel as its
        ``tensor``, a kernel as its ``node``."""
        return finding.about(**{"tensor" if name in self._sides else "node": name})

    # -- a member's answers, relative to it --------------------------------------------------

    def local(self, name: str, finding: Finding) -> Finding:
        """``finding``, of an answer about member ``name`` read on its Space (a policy's
        ``choice-no-valid-value``), as the member's own space gives it: owned by its key in
        the member it is below, naming that member where it is another (this class's
        docstring)."""
        return self._locator.local(name, finding)

    def _result(self, name: str, result: QueryResult[T]) -> QueryResult[T]:
        if isinstance(result, Rejected):
            return Rejected(tuple(self.local(name, each) for each in result.findings))
        if isinstance(result, Unresolved):
            return Unresolved(tuple(self.local(name, each) for each in result.findings))
        return result

    def query(self, name: str, key: ViewKey[T]) -> QueryResult[T]:
        """Member ``name``'s export ``key`` (``CYCLES``, ``NETLIST``, ...);
        ``Inapplicable`` where the member exports none."""
        point = self.member(name)
        exports = type(point).exports
        if key not in exports:
            return Inapplicable()
        found: QueryResult[T] = point.query(exports[key])
        return self._result(name, found)

    def waits(self, name: str, answer: object) -> tuple[Key, ...]:
        """The open Decisions an answer of member ``name`` (``query``'s) waits on, by key:
        its own, or another member's (a channel's on its kernel's choice), which the
        finding names (``node`` or ``tensor``)."""
        if not isinstance(answer, Unresolved):
            return ()
        members = self._composed.names.members
        found: dict[Key, None] = {}
        for item in answer.findings:
            if item.code != "decision-unassigned":
                continue
            details = dict(item.details)
            other = details.get("tensor" if "tensor" in details else "node")
            owner = other if isinstance(other, str) and other in members else name
            found[(owner, item.owner)] = None
        return tuple(found)

    def validity(self, name: str) -> inspection.Validity:
        """Member ``name``'s own validity inside the design space (``inspection.validity``):
        its pieces and findings relative to it."""
        found = inspection.validity(self.member(name))
        pieces = tuple(
            inspection.Piece(
                self._locator.relative(name, piece.owner),
                Rejected(tuple(self.local(name, each) for each in piece.result.findings))
                if isinstance(piece.result, Rejected)
                else Unresolved(tuple(self.local(name, each) for each in piece.result.findings)),
            )
            for piece in found.pieces
        )
        return inspection.Validity(found.status, pieces, self._result(name, found.result))

    # -- its choices ------------------------------------------------------------------------

    def _states(self) -> dict[str, list[tuple[str, inspection.ChoiceState]]]:
        """Every choice's state, read once, by member, each member's in the flow order."""

        def read() -> dict[str, list[tuple[str, inspection.ChoiceState]]]:
            found: dict[str, list[tuple[str, inspection.ChoiceState]]] = {}
            for path, state in inspection.states(self._point):
                owner, key = self._composed.names.pair(path)
                found.setdefault(owner, []).append((key, state))
            return found

        states: dict[str, list[tuple[str, inspection.ChoiceState]]] = self._read("states", read)
        return states

    def _wanted(self, names: Iterable[str] | None) -> list[str]:
        """``names`` (every member where None) ranked, those of the partition only."""
        if names is None:
            return list(self.members)
        wanted = set(names)
        return [name for name in self.members if name in wanted]

    def _state(self, name: str, state: inspection.ChoiceState) -> inspection.ChoiceState:
        """A state of member ``name``'s choice, what it waits on relative to it."""
        if isinstance(state, inspection.Waiting):
            return inspection.Waiting(
                state.value, tuple(self._locator.relative(name, each) for each in state.waits)
            )
        if isinstance(state, inspection.Open) and state.waits:
            return inspection.Open(
                tuple(self._locator.relative(name, each) for each in state.waits)
            )
        return state

    def states(
        self, names: Iterable[str] | None = None
    ) -> tuple[tuple[Key, inspection.ChoiceState], ...]:
        """Every choice of each member (of ``names`` only, where named), by key, with its
        state (``inspection.states``: set, waiting, dormant, open or absent): by member in
        rank order, each member's choices in its flow order (the seam's order,
        ``finn.kernels.explore.Composition``)."""
        states = self._states()
        return tuple(
            ((name, key), self._state(name, state))
            for name in self._wanted(names)
            for key, state in states.get(name, ())
        )

    def decisions(self) -> Mapping[Key, inspection.DecisionInfo[object]]:
        """Every Decision of every member, by key, applicable or not, by member in rank
        order, each member's in its flow order. Each is the engine's, keyed by its path."""
        return self._decisions

    def decision(self, key: Key) -> inspection.DecisionInfo[object] | None:
        """The Decision ``key`` names, or None."""
        return self._decisions.get(key)

    def undecided(self) -> tuple[Key, ...]:
        """Every open choice in play, by key, in the seam's order (``states``): unset in a
        scope that is present (``inspection.Open``, waiting on nothing). A choice whose
        presence waits on another open choice is not in play yet, and a stored one is not
        open, whatever its state."""
        return tuple(
            key
            for key, state in self.states()
            if isinstance(state, inspection.Open) and not state.waits
        )

    def chosen(self, names: Iterable[str] | None = None) -> dict[Key, object]:
        """Every choice stored (of ``names`` only, where named), by key, in the seam's
        order, whatever its state (set, waiting or dormant): what saving writes
        (``persist``)."""
        return {
            key: state.value
            for key, state in self.states(names)
            if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
        }

    def viable(self, keys: Iterable[Key]) -> tuple[tuple[Key, inspection.Viable], ...]:
        """Each choice ``keys`` names, by key, in the order given, with its listed values,
        each possible or refused as setting it alone would be (``inspection.viable``, on
        the design space): each refused value with the rules it would break and the stored
        choices standing in their way, each by its path (``pair`` gives its key), the
        rules' findings relative to the member each is below. A key that is no choice is a
        ``ValueError``."""
        wanted = list(keys)
        references = []
        for key in wanted:
            info = self._decisions.get(key)
            if info is None:
                raise ValueError(f"{dotted(key)}: not a choice of the partition")
            references.append(info.reference)
        found = inspection.viable(self._point, references) if references else ()
        return tuple(
            (key, self._viable(key, item)) for key, item in zip(wanted, found, strict=True)
        )

    def _viable(self, key: Key, item: inspection.Viable) -> inspection.Viable:
        if item.values is None:
            return replace(item, key=key[1])
        values = tuple(
            inspection.Refused(each.value, tuple(map(self._locator.rule, each.broken)))
            if isinstance(each, inspection.Refused)
            else each
            for each in item.values
        )
        return inspection.Viable(key[1], values, item.waits)

    # -- a change ---------------------------------------------------------------------------

    def trial(
        self, changes: Mapping[Key, object], clear: Collection[Key] = (), *, check: bool = True
    ) -> Trial[Self]:
        """``changes`` set and ``clear`` cleared, as one change of the design space
        (``try_with_choices``), each key mapped to its path: refused where a value is
        outside its domain (by the choice's key) or a rule the change makes newly false (by
        the rule's key, another member's where the change reaches it), with the stored
        choices standing in its way; or a key that names no choice (``not-a-decision``).
        Nothing is rebuilt: the trial's ``rederived`` and ``fallbacks`` are empty, and
        ``check`` (a hand-built network's check of the channels it rebuilt) reads nothing. A
        request the engine refuses as malformed (a value of another type) is refused, each
        key named (``request-refused``)."""
        refused: dict[Key, tuple[Finding, ...]] = {}
        values: dict[object, object] = {}
        for key, value in changes.items():
            info = self._decisions.get(key)
            if info is None:
                refused[key] = _not_a_decision(key)
            else:
                values[info.reference] = value
        clears: list[ChangeRequest] = []
        for key in clear:
            info = self._decisions.get(key)
            if info is None:
                refused[key] = _not_a_decision(key)
            else:
                clears.append(self._point.field(info.reference).clear())
        if refused:
            return Trial(self, {}, refused)
        try:
            report = self._point.try_with_choices(values, *clears)
        except (RequestError, ConfigurationError) as error:
            # A request the engine refuses as malformed (a value of another type).
            return Trial(
                self,
                {},
                {
                    key: (Finding(FindingKind.REJECTION, "request-refused", key[1], str(error)),)
                    for key in (*changes, *clear)
                },
            )
        if report.accepted:
            return Trial(self.at(report.instance), {}, {})
        standing: dict[Key, tuple[Key, ...]] = {}
        for outcome in report.outcomes:
            if outcome.status == "refused" and isinstance(outcome.result, (Rejected, Unresolved)):
                owner, below = self._composed.names.pair(outcome.owner)
                refused[(owner, below)] = tuple(
                    self.local(owner, each) for each in outcome.result.findings
                )
        if not refused:
            for rule in report.broken:
                key = self._composed.names.pair(rule.rule)
                refused[key] = self._locator.rule(rule).result.findings
                standing[key] = tuple(self._composed.names.pair(each) for each in rule.standing)
        return Trial(self, {}, refused, standing)

    def with_choices(self, changes: Mapping[Key, object]) -> Self:
        """``changes`` set as one change (``trial``); a refusal is a ``ValueError`` naming
        each refused value or rule, with the stored choices standing in its way."""
        trial = self.trial(changes)
        if not trial.accepted:
            raise ValueError(
                "choices are not accepted: "
                + "; ".join(
                    f"{dotted(key)}: "
                    + "; ".join(map(str, found))
                    + (
                        f" (standing: {', '.join(map(dotted, trial.standing[key]))})"
                        if trial.standing.get(key)
                        else ""
                    )
                    for key, found in trial.refused.items()
                )
            )
        return trial.network

    # -- what it is -------------------------------------------------------------------------

    def neighbours(self, name: str) -> Neighbours:
        """The ports beside channel ``name`` (each kernel's ``<p>_contract`` view, labelled
        below its member's node): the producing kernel's and the consuming kernel's, by the
        side each is declared on. FIFO sizing reads their paces
        (``finn.kernels.fifo_sizing``), and the netlist its hops to them (``hops``)."""
        found: dict[str, PortContract] = {}
        for side, declared in zip(("producer", "consumer"), self._sides[name], strict=True):
            if not isinstance(declared, Port):
                continue
            point = self.member(declared.member)
            answer = point.query(getattr(type(point), f"{declared.port}_contract"))
            if isinstance(answer, Available) and isinstance(answer.value, PortContract):
                found[side] = answer.value.under(self.node(declared.member))
        return Neighbours(found.get("producer"), found.get("consumer"))

    def hops(self, name: str) -> OuterHops | Rejected:
        """Channel ``name``'s hops to the ports beside it (``outer_hops``, from
        ``neighbours``), of a channel whose netlist is answered, its refusals owned by
        ``name``."""
        return outer_hops(self.member(name), self.neighbours(name), owner=name)

    def netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        """What its top is made of, read once: each member's netlist under its node, in its
        module's order, a channel's with its hops to the ports beside it around its own
        links (``hops``: the producer's first, the consumer's last), merged with each
        control bus its kernels present (``CONTROL``, at the port each was given), and each
        boundary channel's AXIS bus; or why not, every member's refusal (or, where none
        refuses, what they wait on; a channel's hops' own, ``channel-port`` or
        ``channel-link``), each finding naming its member (``located``), or the merge's own
        (``kernel-netlist``). The shell's counts and its module read the same."""
        found: QueryResult[tuple[Fragment, tuple[Bus, ...]]] = self._read("netlist", self._netlist)
        return found

    def _netlist(self) -> QueryResult[tuple[Fragment, tuple[Bus, ...]]]:
        netlists: list[tuple[str, Fragment]] = []
        boundary: list[Bus] = []
        controls: list[tuple[str, Control]] = []
        failed: list[tuple[str, QueryResult[object]]] = []
        for name in self._order:
            netlist = self.query(name, NETLIST)
            if name in self._sides and isinstance(netlist, Available):
                hops = self.hops(name)
                netlist = (
                    hops if isinstance(hops, Rejected) else Available(hops.around(netlist.value))
                )
            if isinstance(netlist, Available):
                netlists.append((self.node(name), netlist.value))
            else:
                failed.append((name, netlist))
            if name not in self._sides:
                control = self.query(name, CONTROL)
                if isinstance(control, Available):
                    controls.append((self.node(name), control.value))
                elif not isinstance(control, Inapplicable):
                    failed.append((name, control))
            else:
                buses = self.query(name, BOUNDARY)
                if isinstance(buses, Available):
                    boundary.extend(buses.value)
                else:
                    failed.append((name, buses))
        if failed:
            rejected = [(name, each) for name, each in failed if isinstance(each, Rejected)]
            findings = tuple(
                dict.fromkeys(
                    self.located(name, finding)
                    for name, each in (rejected or failed)
                    for finding in getattr(each, "findings", ())
                )
            )
            if rejected:
                return Rejected(findings)
            return Unresolved(findings) if findings else Inapplicable()
        exported = presented_buses(controls)
        if isinstance(exported, Rejected):
            return exported
        fragment = merged(netlists, exported)
        if isinstance(fragment, Rejected):
            return fragment
        return Available((fragment, tuple(boundary)))

    def module(self, producer: ProducerIdentity, stem: str) -> QueryResult[Module]:
        """Its composed module (``finn.kernels.artifacts.top.compose``) of its ``netlist``;
        or why not, the netlist's refusal or wait, or the top's own (``kernel-netlist``)."""
        found = self.netlist()
        if not isinstance(found, Available):
            return found
        fragment, boundary = found.value
        try:
            return Available(top(producer, stem, fragment, boundary))
        except ArtifactError as error:
            return reject("kernel-netlist", str(error))

    def module_of(self) -> QueryResult[Module]:
        """Its module, the partition's IP (``finn_<name>``), as far as its members
        decide it, read once."""
        found: QueryResult[Module] = self._read(
            "module", lambda: self.module(PRODUCER, f"finn_{self.name.lower()}")
        )
        return found

    def own_resources(self) -> QueryResult[Resources]:
        """What its members use, their own statements' sum; or what they wait on, or why
        one is refused. A shell adds its movers' and its static region's
        (``finn.custom_op.kernels.shell``)."""

        def read() -> QueryResult[Resources]:
            used = [self.query(name, RESOURCES) for name in self.members]
            missing = [each for each in used if not isinstance(each, Available)]
            if missing:
                return _together(missing)
            return Available(total(each.value for each in used if isinstance(each, Available)))

        found: QueryResult[Resources] = self._read("members' resources", read)
        return found

    # -- what its bind dropped ----------------------------------------------------------------

    def _described(self, stored: Mapping[Key, object]) -> dict[str, tuple[Finding, ...]]:
        """What the bind dropped of the stored choices (each a value outside its domain), by
        dotted key, each with the kernel's findings kept: the value stored, and the values
        the choice could take now (its possible ones, ``viable``, read where the dropped
        choice is open), which its message names."""
        if not self.dropped:
            return {}
        listed = [key for key in self.dropped if key in self._decisions]
        cases = {key: item.cases for key, item in self.viable(listed)}

        def described(key: Key, finding: Finding) -> Finding:
            details = dict(finding.details)
            message = finding.message
            if key in stored and _primitive(stored[key]):
                details["value"] = stored[key]
            viable = cases.get(key, ())
            if finding.code == "domain-membership" and viable and all(map(_primitive, viable)):
                details["candidates"] = viable
                message = "not one of " + ", ".join(map(str, viable))
            return replace(finding, message=message, details=tuple(details.items()))

        return {
            dotted(key): tuple(described(key, finding) for finding in found)
            for key, found in self.dropped.items()
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self._order)})"


class _Locator:
    """Paths in a composed class as a member's own space names them (``Partition``'s
    docstring): ``names``, its ``Names``; ``channels``, its channels' graph names."""

    __slots__ = ("channels", "names")

    def __init__(self, names: Names, channels: frozenset[str]) -> None:
        self.names, self.channels = names, channels

    def below(self, path: str) -> Key | None:
        """The member a path is below, by graph name, and its key there; None for a root
        Param's or a side's."""
        member, _, below = path.partition(".")
        owner = self.names.owners.get(member)
        return None if owner is None else (owner, below)

    def relative(self, name: str, path: str) -> str:
        """``path`` as an answer about member ``name`` names it: relative to ``name`` where
        it is below it, else as it is."""
        found = self.below(path)
        return found[1] if found is not None and found[0] == name and found[1] else path

    def local(self, name: str, finding: Finding) -> Finding:
        """``finding``, of an answer about member ``name``, as the member's own space gave
        it: owned by its key in the member it is below, naming that member where it is
        another (``tensor`` for a channel, ``node`` for a kernel); its causes the same."""
        causes = tuple(self.local(name, cause) for cause in finding.causes)
        found = self.below(finding.owner)
        if found is None or not found[1]:
            return finding if causes == finding.causes else replace(finding, causes=causes)
        owner, below = found
        made = replace(finding, owner=below, causes=causes)
        if owner == name:
            return made
        return made.about(**{"tensor" if owner in self.channels else "node": owner})

    def rule(self, rule: BrokenRule) -> BrokenRule:
        """A rule broken, its findings relative to the member it is below (its path and the
        stored choices standing kept, ``Partition.pair`` their keys)."""
        found = self.below(rule.rule)
        name = rule.rule if found is None else found[0]
        return replace(
            rule, result=Rejected(tuple(self.local(name, each) for each in rule.result.findings))
        )


def _decisions(composed: Composed, point: Space) -> dict[Key, inspection.DecisionInfo[object]]:
    """Every Decision of ``point``, by key, by member in rank order, each member's in the
    flow order."""
    found: dict[str, dict[Key, inspection.DecisionInfo[object]]] = {
        name: {} for name in composed.names.members
    }
    for item in inspection.decisions(point):
        owner, key = composed.names.pair(item.key)
        found[owner][(owner, key)] = item
    return {key: item for each in found.values() for key, item in each.items()}


def _not_a_decision(key: Key) -> tuple[Finding, ...]:
    return (
        Finding(
            FindingKind.REJECTION,
            "not-a-decision",
            dotted(key),
            "no choice of the partition" if key[1] else "no member owns it",
        ),
    )


# -- reading the graph --------------------------------------------------------------------


def _boundary(model: ModelWrapper, nodes: list[NodeProto], owned: set[str]) -> dict[str, str]:
    """The port of each boundary tensor, inputs then outputs, in node order: an input
    no node produces (and not a parameter: owned, or an initializer the op reads as a
    fact) is ``s_axis_<i>``; an output read outside ``nodes`` is ``m_axis_<j>``."""
    produced = {tensor for node in nodes for tensor in node.output}
    inside = {id(node) for node in nodes}
    used_outside = {
        tensor for node in model.graph.node if id(node) not in inside for tensor in node.input
    } | {output.name for output in model.graph.output}
    inputs = [
        tensor
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced and tensor not in owned and initializer(model, tensor) is None
    ]
    outputs = [tensor for node in nodes for tensor in node.output if tensor in used_outside]
    ports = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    return ports | {tensor: f"m_axis_{index}" for index, tensor in enumerate(outputs)}


def _fan_out(
    model: ModelWrapper,
    nodes: list[NodeProto],
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
) -> list[Finding]:
    """Each tensor whose channel would have more than one consumer, in node order: read
    by two of ``nodes`` (an input that is not an initializer, or a parameter a node owns),
    or by one and at its output port too (``m_axis_<j>``)."""
    readers: dict[str, list[str]] = {}
    for node, tensors in zip(nodes, owned):
        for tensor in node.input:
            if tensor in tensors.values() or initializer(model, tensor) is None:
                readers.setdefault(tensor, []).append(node.name)
    found = []
    for tensor, names in readers.items():
        port = ports.get(tensor, "")
        if port.startswith("m_axis_"):
            message = f"{tensor} is read by {', '.join(names)} and leaves the partition at {port}"
        elif len(names) > 1:
            message = f"{tensor} is read by {', '.join(names)}"
        else:
            continue
        details = (("tensor", tensor),)
        found.append(Finding(FindingKind.REJECTION, "tensor-fan-out", "fan_out", message, details))
    return found


def _tensors(model: ModelWrapper, nodes: list[NodeProto], owned: list[dict[str, str]]) -> list[str]:
    """Each tensor a channel carries, in node order: a node's inputs on an edge or the
    boundary (an initializer it reads as a fact carries none), the parameters it owns,
    its outputs."""
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    found: dict[str, None] = {}
    for node, tensors in zip(nodes, owned):
        for tensor in node.input:
            if tensor not in parameters and initializer(model, tensor) is None:
                found.setdefault(tensor)
        for tensor in tensors.values():
            found.setdefault(tensor)
        for tensor in node.output:
            found.setdefault(tensor)
    return list(found)


def _named(nodes: list[NodeProto], tensors: Iterable[str]) -> None:
    """Refuse two members of one name: a node and a tensor, or two nodes."""
    kinds: dict[str, str] = dict.fromkeys(tensors, "tensor")
    for node in nodes:
        if node.name in kinds:
            other = "another node" if kinds[node.name] == "node" else "a tensor"
            raise KernelOpError.refused(
                "name-shared",
                "unique_names",
                f"{node.name}: a node and {other} are both named {node.name}",
                (node.name,),
                node=node.name,
            )
        kinds[node.name] = "node"


def _stored(
    model: ModelWrapper,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    tensors: list[str],
    whole: bool,
) -> tuple[dict[Key, object], dict[str, tuple[Finding, ...]]]:
    """The choices the partition binds, by key: each node's attributes, each channel's as
    its tensor states them; and, where ``whole``, those of a tensor that states some and is
    no channel of it, by dotted key, each with its finding (``channel-choice-stray``)."""
    found: dict[Key, object] = {}
    for node, op in zip(nodes, ops):
        found |= {(node.name, key): value for key, value in op.stored().items()}
    stated = model.tensors_stating(CHANNEL)
    refuse_outside_body(model, stated)
    for tensor in tensors:
        if tensor in stated:
            values = channel_choices(model, tensor)
            found |= {(tensor, key): value for key, value in typed(Channel, values).items()}
    channels = set(tensors)
    stray: dict[str, tuple[Finding, ...]] = {
        dotted((tensor, key)): (
            Finding(
                FindingKind.REJECTION,
                "channel-choice-stray",
                key,
                "stated on a tensor that is no channel of the partition",
                (("tensor", tensor),),
            ),
        )
        for tensor in stated
        if whole and tensor not in channels
        for key in channel_choices(model, tensor)
    }
    return found, stray


def _stale(
    model: ModelWrapper, bound: Partition, nodes: list[NodeProto], ops: list[KernelOp]
) -> None:
    """Refuse a graph whose annotation of an output differs from what its producer
    states (its element count, its last axis, or its element: the datatype and its
    range), by name: the graph is stale (``tensor-stale``)."""
    findings: list[Finding] = []
    for node, op in zip(nodes, ops):
        kernel = bound.member(node.name)
        for port, tensor in zip(op.outputs, node.output):
            stated = kernel.query(getattr(type(kernel), op.wiring[port].tensor))
            if not isinstance(stated, Available):
                continue  # what it waits on, or why it refuses, is its channel's to name
            graph = edge_tensor(model, tensor, op.label)
            produced = stated.value
            if (
                graph.size == produced.size
                and graph.shape[-1] == produced.shape[-1]
                and graph.element == produced.element
            ):
                continue
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "tensor-stale",
                    "stale_outputs",
                    f"the graph states {tensor} {list(graph.shape)} {graph.element}, "
                    f"its producer {node.name} {list(produced.shape)} {produced.element}: "
                    "run InferKernelTensors",
                    (("tensor", tensor), ("node", node.name)),
                )
            )
    if findings:
        raise KernelOpError("; ".join(map(str, findings)), findings=tuple(findings))


class _Read:
    """What the graph states of a selection: its declaration, its module's order, its
    boundary's ports, its KernelOps, the choices it stores and those stray, and whether it
    holds every KernelOp of the model. A record read once, never a value a Space holds."""

    __slots__ = ("declared", "ops", "order", "ports", "stored", "stray", "whole")

    def __init__(
        self,
        declared: PartitionDeclaration,
        order: tuple[str, ...],
        ports: dict[str, str],
        ops: list[KernelOp],
        stored: dict[Key, object],
        stray: dict[str, tuple[Finding, ...]],
        whole: bool,
    ) -> None:
        self.declared, self.order, self.ports, self.ops = declared, order, ports, ops
        self.stored, self.stray, self.whole = stored, stray, whole


def _graph(
    model: ModelWrapper, nodes: list[NodeProto], name: str, movers: tuple[MoverOffer, ...]
) -> _Read:
    """The selection ``nodes`` of ``model`` as the graph states it (``_Read``; the module
    docstring), refused where a tensor is read more than once (``tensor-fan-out``) or two
    members share a name (``name-shared``)."""
    ops = [kernel_op(model, node) for node in nodes]
    owned = [op.owned() for op in ops]
    ports = _boundary(model, nodes, {tensor for tensors in owned for tensor in tensors.values()})
    fanned = _fan_out(model, nodes, owned, ports)
    if fanned:
        raise KernelOpError(
            f"{name}: a channel has one consumer, and fan-out is not designed yet (a channel "
            "that forks): " + "; ".join(map(str, fanned)),
            findings=fanned,
        )
    tensors = _tensors(model, nodes, owned)
    _named(nodes, tensors)
    inside = {id(node) for node in nodes}
    whole = all(id(node) in inside for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN)
    stored, stray = _stored(model, nodes, ops, tensors, whole)
    platform = read_target(model).platform
    sides: dict[str, dict[str, Port | Side]] = {tensor: {} for tensor in tensors}
    facts: dict[str, dict[str, object]] = {tensor: {} for tensor in tensors}
    readers: dict[str, str] = {}
    for node, op, held in zip(nodes, ops, owned):
        wired = [("consumer", port, tensor) for port, tensor in op.inputs().items()]
        wired += [("producer", port, tensor) for port, tensor in zip(op.outputs, node.output)]
        for side, port, tensor in wired:
            wiring = op.wiring[port]
            sides[tensor][side] = Port(node.name, wiring.contract.removesuffix("_contract"))
            readers.setdefault(tensor, op.label)
            if port in held:
                facts[tensor]["tensor"] = Stated(node.name, str(wiring.carried))
                facts[tensor]["contents"] = Stated(node.name, str(wiring.stored))
    channels: dict[str, ChannelDeclaration] = {}
    for tensor in tensors:
        given = facts[tensor]
        if "tensor" not in given:
            given["tensor"] = edge_tensor(model, tensor, readers[tensor])
        given["platform"] = platform
        ends = sides[tensor]
        if "contents" in given and "producer" not in ends:
            ends["producer"] = Side.source()
        for each in ("producer", "consumer"):
            if each in ends:
                continue
            if tensor not in ports:
                raise KernelOpError.refused(
                    "channel-boundary",
                    tensor,
                    f"{tensor}: a boundary of its partition, but no port names it: only an ONNX "
                    "input or output of a partition crosses its boundary",
                    tensor=tensor,
                )
            ends[each] = Side.boundary(ports[tensor], movers)
        channels[tensor] = ChannelDeclaration(ends["producer"], ends["consumer"], given)

    def on(prefix: str) -> list[str]:
        """The channels at the ports ``<prefix><i>`` (``""``: no port, none on the
        boundary)."""
        return [
            tensor for tensor in tensors if ports.get(tensor, "").rstrip("0123456789") == prefix
        ]

    kernels = {node.name: op.facts() for node, op in zip(nodes, ops)}
    declared_order = [*on("s_axis_"), *on(""), *on("m_axis_")]
    declared = PartitionDeclaration(
        kernels, {tensor: channels[tensor] for tensor in declared_order}
    )
    order = (*on("s_axis_"), *on(""), *kernels, *on("m_axis_"))
    return _Read(declared, order, ports, ops, stored, stray, whole)


def declaration(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, movers: tuple[MoverOffer, ...] = ()
) -> PartitionDeclaration:
    """The partition of ``nodes``, KernelOp nodes of ``model``, as the graph declares it
    (the module docstring): each node's kernel by its facts, in graph order, and each
    channel by its tensor with its sides and facts, its boundary offered ``movers``."""
    with reading(model):
        return _graph(model, list(nodes), PARTITION, movers).declared


def network(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str = PARTITION,
    movers: tuple[MoverOffer, ...] = (),
) -> Partition:
    """The partition of ``nodes``, KernelOp nodes of ``model``, its module the partition
    ``name``'s IP, on the platform of the model's target: ``movers`` offered on the boundary
    channels (none by default; a shell's row offers them, ``in_shell``); see the module
    docstring. A tensor read more than once is refused (``tensor-fan-out``), and so is a
    stale output (``tensor-stale``)."""
    return _network(model, list(nodes), name, movers, Partition)


P = TypeVar("P", bound=Partition)


def _network(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    movers: tuple[MoverOffer, ...],
    kind: Callable[..., P],
) -> P:
    """``network``, made by ``kind`` (``Partition``, or a shell's partition over it), the
    graph's tensors read from one index."""
    with reading(model):
        read = _graph(model, nodes, name, movers)
        try:
            composed = compose(read.declared)
        except PartitionRefused as error:
            message = f"{name}: " + "; ".join(map(str, error.findings))
            raise KernelOpError(message, findings=error.findings) from error
        point = composed.design_space()
        decisions = {item.key: item.reference for item in inspection.decisions(point)}
        unknown: dict[Key, tuple[Finding, ...]] = {}
        values: dict[object, object] = {}
        for key, value in read.stored.items():
            path = composed.names.path(key)
            if path in decisions:
                values[decisions[path]] = value
            else:
                # Owned by its path, as the engine's findings are: relative to the member
                # once the partition holds it.
                unknown[key] = (
                    Finding(
                        FindingKind.REJECTION,
                        "not-a-decision",
                        path,
                        "not a Decision of this member",
                    ),
                )
        report = bind(point, values)
        dropped = dict(unknown)
        for outcome in report.outcomes:
            if outcome.status == "dropped":
                result = outcome.result
                findings = result.findings if isinstance(result, (Rejected, Unresolved)) else ()
                dropped[composed.names.pair(outcome.owner)] = findings
        made = kind(
            composed,
            report.instance,
            name=name,
            order=read.order,
            sides={
                tensor: (each.producer, each.consumer)
                for tensor, each in read.declared.channels.items()
            },
            boundary=tuple(read.ports.items()),
            movers=movers,
            owners={node.name: type(op) for node, op in zip(nodes, read.ops)},
            dropped=dropped,
            stray=read.stray,
            stored=read.stored,
            whole=read.whole,
        )
        _stale(model, made, nodes, read.ops)
        return made


# -- writing choices back -----------------------------------------------------------------


def _encoded(tensor: str, values: Mapping[str, object]) -> None:
    """Refuse a channel choice ``CHANNEL`` does not declare or admit, by name."""
    unknown = sorted(set(values) - set(CHANNEL_KEYS))
    if unknown:
        raise KernelOpError.refused(
            "channel-choice-unknown",
            CHANNEL.name,
            f"{tensor}: {unknown} are not a channel's choices ({CHANNEL.name})",
            tuple(unknown),
            tensor=tensor,
        )
    for key, value in values.items():
        try:
            CHANNEL_KEYS[key].encode(value)
        except MetadataError as error:
            raise KernelOpError.refused(
                "channel-choice-invalid",
                CHANNEL.name,
                f"{tensor}: {error}",
                (key,),
                tensor=tensor,
            ) from error


def _write_channel(model: ModelWrapper, tensor: str, choices: Mapping[str, object]) -> None:
    """State ``choices`` on ``tensor``'s channel, its whole: what it stated before is
    cleared."""
    model.clear(CHANNEL, tensor=tensor)
    for key, value in sorted(choices.items()):
        model.set(CHANNEL_KEYS[key], value, tensor=tensor)


def persist(model: ModelWrapper, partition: Partition) -> dict[str, dict[str, object]]:
    """Write ``partition``'s choices, a partition of ``model``'s nodes (``network``), back
    where they are stated, each owner's whole: every choice its design space stores,
    whatever its state (set, waiting or dormant), split by member, on its owner, a kernel's
    node or a channel's tensor; a choice an owner holds that the partition does not (one the
    bind dropped as stale) cleared, and, where it holds every KernelOp of the model, so are
    the channel choices of a tensor that is no channel of it.

    The one writer of a partition's choices: a route to one is ``Hold`` in a step after the
    cut, this function for the compiler (``persist_kernel`` beside it for conversion, one
    node's kernel alone), and ``support.configure.configured`` for a test. Channel
    choices are a partition's body's only: written on a graph that is not (host nodes
    beside the KernelOps) they are refused, by name, before anything is written
    (``channel-choices-outside-body``). Every value is checked and encoded before any is
    written, so a write cannot fail halfway; none is set again on a fresh point, since the
    partition has bound each (its choices are checked as they are set).
    Returns what each owner now holds, by graph name."""
    names = list(partition.members)
    kernels = set(partition.kernels)
    held_by: dict[str, dict[str, object]] = {name: {} for name in names}
    for (owner, key), value in partition.chosen(names).items():
        held_by[owner][key] = value
    refuse_outside_body(
        model, [name for name, values in held_by.items() if values and name not in kernels]
    )
    nodes = {node.name: node for node in model.graph.node}
    stated = model.tensors_stating(CHANNEL)
    writes: list[Callable[[], None]] = []
    for name, values in held_by.items():
        if name in kernels:
            op = kernel_op(model, nodes[name])
            if values or op.choices():
                writes.append(_kernel_write(op, op._saved(values)))
        elif values or name in stated:
            _encoded(name, values)
            writes.append(_channel_write(model, name, values))
    stray = [tensor for tensor in stated if partition.whole and tensor not in held_by]
    for write in writes:
        write()
    for tensor in stray:
        model.clear(CHANNEL, tensor=tensor)
    return held_by


def persist_kernel(model: ModelWrapper, node: NodeProto, point: Space) -> dict[str, object]:
    """Write ``point``, the KernelOp ``node``'s kernel alone, on the node, its whole: every
    choice it stores, by key, whatever its state (``inspection.states``: set, waiting or
    dormant). Conversion's write (``finn.transformation.kernels.convert``): one node, no
    partition, nothing on a tensor. As ``persist``, every value is checked and encoded
    before any is written. Returns what the node now holds."""
    held = {
        key: state.value
        for key, state in inspection.states(point)
        if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
    }
    op = kernel_op(model, node)
    op._write(op._saved(held))
    return held


def _kernel_write(op: KernelOp, merged: Mapping[str, int | str]) -> Callable[[], None]:
    return lambda: op._write(merged)


def _channel_write(
    model: ModelWrapper, tensor: str, values: Mapping[str, object]
) -> Callable[[], None]:
    return lambda: _write_channel(model, tensor, values)


__all__ = [
    "PARTITION",
    "PRODUCER",
    "Partition",
    "declaration",
    "network",
    "persist",
    "persist_kernel",
    "stale_text",
]
