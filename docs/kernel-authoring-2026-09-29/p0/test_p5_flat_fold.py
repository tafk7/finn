"""P0.5: a flat (streamless) build commits a fold whose domain reads a stream's extent (G0.4b).

The engine cannot commit it: `divisors_of(extent_of(c))` is refused flat
(`kernel-extents`), and a domain reading the absent stream directly is unresolved
(`input-unsupplied`). The plan's fallback works: a domain over the kernel's
`extents` mapping (always available, empty when flat) that is the RTL's own bound
(1 <= PE < 2^32) for an unbound index and the divisors of its extent for a bound one.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from _bind import BoundKernel, extent_of
from _pool import stream

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    default_semantics,
    derived,
    design_space,
    divisors_of,
    domain,
    reject,
)
from finn.core.space.domains import Domain
from finn.core.space.results import Available, QueryResult, Unresolved
from finn.dataflow.schedule import SCHEDULE, Index, Schedule
from finn.dataflow.stream import Stream
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.configure import commit
from finn.kernels.port import ScheduledPort

a, c = Index("a"), Index("c")
RTL_BOUND = 1 << 32  # binopi's PE is a 32-bit generic


class Elt(BoundKernel):
    """An eltwise-like streaming module: its parameters need no extents, only PE."""

    id = "probe.elt"
    module = "binopi"
    lhs_stream: Stream = Param(required=False)
    out_stream: Stream = Param(required=False)

    channels = extent_of(c)
    pe: int = Decision(domain=divisors_of(channels))

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        return Schedule(self.extents, folds={c: self.pe}, beats=(a, c))

    lhs = ScheduledPort(
        name="s_axis_lhs",
        endpoint=Endpoint.TARGET,
        stream=lhs_stream,
        schedule=schedule,
        index=(a, c),
        lanes=(c,),
        idle_lanes=pe,
    )
    out = ScheduledPort(
        name="m_axis_out",
        endpoint=Endpoint.INITIATOR,
        stream=out_stream,
        schedule=schedule,
        index=(a, c),
        lanes=(c,),
        idle_lanes=pe,
    )

    def parameters(self) -> Mapping[str, int | str]:
        return {"PE": self.pe}


class ReadsStream(Elt):
    """The same fold with its domain read straight off the absent stream."""

    id = "probe.elt_reads_stream"

    @derived
    def channels(self) -> int:  # type: ignore[override]
        return self.lhs_stream.tensor.shape[-1]

    pe: int = Decision(domain=divisors_of(channels))


# -- what the engine does -----------------------------------------------------------------


def test_the_engine_cannot_commit_a_divisor_fold_flat() -> None:
    flat = design_space(Elt())
    assert flat.extents == {}
    refused = flat.field(Elt.pe).candidates()
    assert isinstance(refused, Rejected)
    assert [(f.code, f.message) for f in refused.findings] == [
        ("kernel-extents", "c is bound by no placed port")
    ]
    with pytest.raises(ValueError, match="channels: kernel-extents: c is bound by no placed port"):
        commit(flat, {"pe": 2})


def test_nor_with_the_domain_read_off_the_absent_stream() -> None:
    flat = design_space(ReadsStream())
    unresolved = flat.field(ReadsStream.pe).candidates()
    assert isinstance(unresolved, Unresolved)
    assert [(f.code, f.owner) for f in unresolved.findings] == [("input-unsupplied", "lhs_stream")]
    with pytest.raises(ValueError, match="lhs_stream: input-unsupplied"):
        commit(flat, {"pe": 2})


# -- the fallback: the RTL's own bound unplaced, divisors placed ---------------------------


def fold_domain(index: Index, bound: int = RTL_BOUND) -> Domain[int]:
    """Divisors of `index`'s bound extent; the RTL's own bound while nothing binds it."""
    divisors = divisors_of(1).candidates
    assert divisors is not None

    def accepts(*, candidate: int, extents: Mapping[Index, int]) -> bool:
        if type(candidate) is not int or candidate < 1:
            return False
        return extents[index] % candidate == 0 if index in extents else candidate < bound

    def candidates(*, extents: Mapping[Index, int]) -> tuple[int, ...] | QueryResult[object]:
        if index in extents:
            return tuple(divisors(extent=extents[index]))
        return reject("fold-unplaced", f"{index!r} is unbound: any 1 <= fold < {bound}")

    return domain(
        accepts=accepts,
        candidates=candidates,  # type: ignore[arg-type]
        semantics=default_semantics(int),
        extents=BoundKernel.extents,
    )


class FlatElt(Elt):
    id = "probe.elt_flat"
    pe: int = Decision(domain=fold_domain(c))


def test_the_fallback_commits_flat_within_the_rtl_bound() -> None:
    flat = design_space(FlatElt())
    enumerated = flat.field(FlatElt.pe).candidates()
    assert isinstance(enumerated, Rejected)  # advisory: no finite enumeration unplaced
    assert [f.code for f in enumerated.findings] == ["fold-unplaced"]
    point = commit(flat, {"pe": 3})  # no extent to divide: any PE the RTL takes
    assert point.pe == 3 and point.parameters() == {"PE": 3}
    assert (point.lhs.lane_count, point.out.lane_count) == (3, 3)  # idle lanes from the fold
    for outside in (0, RTL_BOUND):
        with pytest.raises(ValueError, match="domain-membership: candidate is outside the domain"):
            commit(flat, {"pe": outside})


def test_the_fallback_is_the_divisors_once_placed() -> None:
    class Placed(Space):
        lhs = stream((2, 8), "INT4", "in0_V")
        out = stream((2, 8), "INT4", "out0_V")
        elt = FlatElt(lhs_stream=lhs, out_stream=out)

    point = design_space(Placed())
    assert point.elt.field(FlatElt.pe).candidates() == Available((1, 2, 4, 8))
    committed = commit(point, {"elt.pe": 4})
    assert committed.elt.lhs.sequence.form.lanes == 4
    with pytest.raises(ValueError, match="domain-membership: candidate is outside the domain"):
        commit(point, {"elt.pe": 3})
