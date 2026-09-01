# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# hypothesis is not a declared dependency yet (see the docstring), so mypy
# cannot see it and reads every `@given` as an untyped decorator.  Waived here
# and nowhere else; when the dependency is declared, delete this line.
# mypy: disable-error-code="import-not-found, untyped-decorator"

"""A1: the projection's two properties, over generated values rather than chosen ones.

The parametrized tests next door pin the cases somebody thought of.  These pin
the claim itself, which is what §6.1 rests on: *for any* ``Derivation``, the
preimage carries no mapping and no untagged number, and the key is a function
of the declared inputs and nothing else.

``hypothesis`` is not yet declared in the repository's requirements -- doing so
means editing a file this effort does not own -- so the module skips where it
is absent.  Everything it asserts is also covered by a hand-written case in
``test_projection.py``; these find the cases nobody wrote.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from finn.dataflow.artifacts.derivation import (  # noqa: E402
    ArtifactRef,
    ContentRef,
    Derivation,
    ProducerIdentity,
    Scalar,
    build_key,
)
from finn.dataflow.artifacts.projection import project  # noqa: E402

#: ``derandomize`` so a gate run is reproducible rather than a different search
#: each time, and ``database=None`` so no ``.hypothesis`` directory appears in
#: the repository -- gitignoring one would mean editing a file this effort does
#: not own.
_SETTINGS = {"max_examples": 50, "derandomize": True, "database": None}

TAGS = frozenset({"none", "bool", "int", "float", "str", "bytes", "enum"})

digests = st.from_regex(r"\A[0-9a-f]{64}\Z")
names = st.text(min_size=1, max_size=12)

scalars: st.SearchStrategy[Scalar] = st.one_of(
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)

references = st.one_of(
    digests.map(ContentRef),
    st.tuples(st.sampled_from(["kernel-source", "composed-source"]), digests).map(
        lambda pair: ArtifactRef(*pair)
    ),
)


def _table(values: st.SearchStrategy[object]) -> st.SearchStrategy[tuple[object, ...]]:
    """Name/value pairs with the names kept unique, since a repeat is refused."""

    return st.lists(names, unique=True, max_size=6).flatmap(
        lambda keys: st.tuples(*[st.tuples(st.just(key), values) for key in keys])
    )


derivations = st.builds(
    Derivation,
    kind=st.text(min_size=1, max_size=16),
    schema_version=st.text(min_size=1, max_size=16),
    producer=st.builds(
        ProducerIdentity,
        producer_id=st.text(min_size=1, max_size=16),
        contract_version=st.text(min_size=1, max_size=8),
    ),
    templates=st.lists(digests.map(ContentRef), max_size=3).map(tuple),
    inputs=_table(references),
    options=_table(scalars),
)


@given(derivations)
@settings(**_SETTINGS)
def test_the_preimage_of_any_derivation_carries_no_mapping_and_no_bare_number(
    derivation: Derivation,
) -> None:
    """Every leaf is ``(path, tag, text)``, so there is nothing left to order."""

    for pair in project(derivation):
        assert len(pair) == 3
        path, tag, text = pair
        assert isinstance(path, str)
        assert tag in TAGS
        assert isinstance(text, str)


@given(derivations)
@settings(**_SETTINGS)
def test_a_key_is_a_function_of_the_declared_inputs(derivation: Derivation) -> None:
    """Recomputing it twice in one process is the weakest half of determinism.

    The other half -- two processes, several hash seeds -- is a subprocess test
    in ``test_existing_identities``, because it is the only place a set or dict
    iteration would show.
    """

    assert build_key(derivation) == build_key(derivation)


@given(derivations, st.randoms())
@settings(**_SETTINGS)
def test_an_option_table_written_in_another_order_is_the_same_table(
    derivation: Derivation, random: object
) -> None:
    """Options are named, so their order is not a fact about the build."""

    shuffled = list(derivation.options)
    random.shuffle(shuffled)  # type: ignore[attr-defined]
    reordered = Derivation(
        kind=derivation.kind,
        schema_version=derivation.schema_version,
        producer=derivation.producer,
        templates=derivation.templates,
        inputs=derivation.inputs,
        options=tuple(shuffled),
    )
    assert build_key(reordered) == build_key(derivation)


@given(derivations, names, scalars)
@settings(**_SETTINGS)
def test_adding_a_consumed_input_moves_the_key(
    derivation: Derivation, name: str, value: Scalar
) -> None:
    """An input that can leave the key without moving it is a wrong hit waiting."""

    if any(existing == name for existing, _ in derivation.options):
        return
    widened = Derivation(
        kind=derivation.kind,
        schema_version=derivation.schema_version,
        producer=derivation.producer,
        templates=derivation.templates,
        inputs=derivation.inputs,
        options=derivation.options + ((name, value),),
    )
    assert build_key(widened) != build_key(derivation)
