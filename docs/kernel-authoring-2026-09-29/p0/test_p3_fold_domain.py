"""P0.3: a fold Decision whose domain is the divisors of a bound extent (plan D4, G0.4).

`extent_of(c)` is a derived member reading the derived `extents` mapping
(`_bind.extent_of`). Inline, `divisors_of(extent_of(c))` is refused at link time;
the one-line named member `channels = extent_of(c)` is the fallback, and is the
spelling the plan's target API already uses.
"""

from __future__ import annotations

import pytest
from _bind import extent_of
from _pool import Pool, c, stream

from finn.core.space import Decision, Space, design_space, divisors_of, inspection
from finn.core.space.errors import DefinitionError
from finn.kernels.configure import commit, undecided


def placed(family: type[Pool] = Pool) -> Space:
    class Placed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        pool = family(x_stream=x, y_stream=y)

    return design_space(Placed())


def test_inline_extent_of_in_a_domain_is_refused_at_link_time() -> None:
    class Inline(Pool):
        id = "probe.inline"
        pe: int = Decision(domain=divisors_of(extent_of(c)))

    with pytest.raises(DefinitionError, match="is not a member of this scope; name it"):
        placed(Inline)


def test_the_named_member_domain_enumerates_and_commits() -> None:
    point = placed()
    assert point.pool.channels == 8
    assert point.pool.field(Pool.pe).candidates().value == (1, 2, 4, 8)
    assert undecided(point, "pool.pe") == ["pool.pe"]
    committed = commit(point, {"pool.pe": 4})
    assert committed.pool.schedule.folds == ((c, 4),)
    with pytest.raises(ValueError, match="outside"):
        commit(point, {"pool.pe": 3})


def test_a_parent_pins_the_fold_by_key() -> None:
    class Pinned(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        pool = Pool(x_stream=x, y_stream=y)
        pool.pe = 4

    point = design_space(Pinned())
    assert [item.key for item in inspection.pinned(point)] == ["pool.pe"]
    assert "pool.pe" not in {item.key for item in inspection.choices(point)}
    assert point.pool.schedule.folds == ((c, 4),)


def test_a_parent_narrows_the_fold() -> None:
    class Narrowed(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        pool = Pool(x_stream=x, y_stream=y)
        pool.pe = Decision(values=(2, 4))

    point = design_space(Narrowed())
    assert point.pool.field(Pool.pe).candidates().value == (2, 4)


def test_a_pin_outside_the_domain_is_a_rejection() -> None:
    class Bad(Space):
        x = stream((1, 4, 8), "INT4", "in0_V")
        y = stream((1, 8), "INT8", "out0_V")
        pool = Pool(x_stream=x, y_stream=y)
        pool.pe = 3

    refused = design_space(Bad()).pool.query(Pool.build_requirements)
    assert [(f.code, f.owner) for f in refused.findings] == [("domain-membership", "pool.pe")]
