# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every field of the oracle's problem, and what became of it.

**The table is the deliverable, not the diff.**  "Compare the fields present in
both" is the comparison that always passes: it silently drops whatever the new
stack stopped projecting, which is precisely the set worth arguing about.  So
every field of ``MVAUProblem`` and of ``MVAUSourceDescription`` carries exactly
one mark, and a field with no entry is a failure rather than an omission.

The five marks, and what each obliges:

``EQUAL``
    Same value, same meaning.  Compared numerically.

``REPRESENTED_BY``
    Renamed or re-spelled.  Compared against the named replacement.

``DERIVED_FROM``
    Not stored.  Reconstructed from what is, and compared.

``MOVED_TO``
    Belongs to a later phase or another layer.  **Not parity**: an open
    obligation that the Gate report lists as such.  Summing these with ``EQUAL``
    is the failure the marking exists to prevent, so nothing here supplies a
    comparison at all -- there is deliberately no way to accidentally count one.

``REMOVED_BECAUSE``
    Deliberately gone, with the argument recorded here rather than in a commit
    message nobody will read next to the table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EQUAL = "equal"
REPRESENTED_BY = "represented_by"
DERIVED_FROM = "derived_from"
MOVED_TO = "moved_to"
REMOVED_BECAUSE = "removed_because"

MARKS = (EQUAL, REPRESENTED_BY, DERIVED_FROM, MOVED_TO, REMOVED_BECAUSE)


#: A value neither stack produced, distinct from ``None``, which several fields
#: use to mean "nobody promised anything".
class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<missing>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing)

    def __hash__(self) -> int:
        return hash("<missing>")


MISSING = _Missing()


def encode(value: Any) -> Any:
    """The structural spelling the oracle probe writes, for the local side.

    Deliberately a second copy of ``oracle_probe._encode`` rather than a shared
    import: the probe runs in an interpreter where this package is not
    importable, which is the property that makes the comparison meaningful.
    The two are kept honest by ``test_the_two_encoders_agree``.
    """

    import numpy  # noqa: PLC0415

    if value is MISSING:
        return MISSING
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "allowed") and hasattr(value, "bitwidth"):
        return {"datatype": str(value)}
    if isinstance(value, numpy.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return {"enum": value.value}
    return {"repr": repr(value)}


@dataclass(frozen=True)
class Entry:
    """One oracle field, its mark, and -- unless it moved -- how to compare it."""

    oracle: str
    mark: str
    detail: str
    #: Reads the encoded oracle value out of one fixture's probe entry.
    read_oracle: Callable[[dict[str, Any]], Any] | None = None
    #: Reads the same fact off a bound occurrence of this stack.
    read_local: Callable[[Any], Any] | None = None
    #: For ``MOVED_TO``: the phase that owes it.
    target: str | None = None
    #: How the two encoded values are compared, when equality is the wrong
    #: question -- the computation profile is one value on one side and a pair
    #: on the other, and *that* is its disposition rather than a mismatch.
    equal: Callable[[Any, Any], bool] | None = None
    #: The test that compares this field, when the comparison is a table of its
    #: own.  An entry may delegate, but it may not simply have no comparison.
    compared_in: str | None = None

    def __post_init__(self) -> None:
        if self.mark not in MARKS:
            raise ValueError(f"{self.oracle}: {self.mark!r} is not one of the five marks")
        readable = self.read_oracle is not None and self.read_local is not None
        if self.mark == MOVED_TO:
            if readable or self.compared_in:
                raise ValueError(f"{self.oracle}: a moved field is not compared, it is owed")
            if not self.target:
                raise ValueError(f"{self.oracle}: a moved field names the phase that owes it")
        elif not readable and not self.compared_in:
            raise ValueError(f"{self.oracle}: a field that is not moved must be comparable")

    def agrees(self, oracle_value: Any, local_value: Any) -> bool:
        if self.equal is not None:
            return bool(self.equal(oracle_value, local_value))
        return bool(oracle_value == local_value)


# -- reading the oracle side --------------------------------------------------


def _problem(path: str) -> Callable[[dict[str, Any]], Any]:
    def read(entry: dict[str, Any]) -> Any:
        return entry["problem"].get(path, MISSING)

    return read


def _derived(name: str) -> Callable[[dict[str, Any]], Any]:
    def read(entry: dict[str, Any]) -> Any:
        return entry["derived"].get(name, MISSING)

    return read


def _description(field: str) -> Callable[[dict[str, Any]], Any]:
    def read(entry: dict[str, Any]) -> Any:
        description = entry["derived"].get("source_description")
        if not isinstance(description, dict) or "fields" not in description:
            return MISSING
        return description["fields"].get(field, MISSING)

    return read


# -- reading this stack -------------------------------------------------------


def _member(name: str) -> Callable[[Any], Any]:
    def read(bound: Any) -> Any:
        try:
            return encode(getattr(bound, name))
        except Exception:  # noqa: BLE001 - an absent optional fact is a value here
            return MISSING

    return read


def _operand(operand: str, facet: str) -> Callable[[Any], Any]:
    def read(bound: Any) -> Any:
        source = bound.source
        if not source.has(operand):
            return MISSING
        value = getattr(source.operand(operand), facet)
        return MISSING if value is None else encode(value)

    return read


#: What the old three-valued enum could name, against the pair.
#:
#: Two of the three are *many-to-one*, and that is the whole finding: the oracle
#: had no value for a bipolar popcount and no value for any popcount that was
#: then thresholded, so its profile said "integer" or "fused_threshold" for
#: nodes it nevertheless executed as popcounts.  Parity for those rows is
#: therefore against the oracle's **execution**, which
#: ``test_computation_profile_mapping`` runs on both stacks.
_PROFILE_CORRESPONDENCE = {
    "accumulator_integer": {("integer", "none"), ("bipolar_popcount", "none")},
    "bipolar_xnor_accumulator": {("xnor_popcount", "none")},
    "fused_threshold": {
        ("integer", "multithreshold"),
        ("xnor_popcount", "multithreshold"),
        ("bipolar_popcount", "multithreshold"),
    },
}


def _enum_value(value: Any) -> str:
    """The value of an encoded enum, whichever spelling ``encode`` produced.

    A ``str`` Enum encodes as a bare string, an ordinary one as ``{"enum": ...}``
    -- and both stacks have one of each, so normalizing here is cheaper than
    pinning either Enum's base class as though it were part of the contract.
    """

    if isinstance(value, dict) and "enum" in value:
        return str(value["enum"])
    # A ``str`` Enum survives ``encode`` as itself and survives JSON as its
    # *value*, so the local side needs the value taken explicitly.
    return str(getattr(value, "value", value))


def _profiles_correspond(oracle_value: Any, local_value: Any) -> bool:
    if not isinstance(local_value, (list, tuple)) or len(local_value) != 2:
        return False
    pair = (_enum_value(local_value[0]), _enum_value(local_value[1]))
    return pair in _PROFILE_CORRESPONDENCE.get(_enum_value(oracle_value), set())


def _profile_pair(bound: Any) -> Any:
    profile = bound.profile
    return [encode(profile.accumulation), encode(profile.activation)]


def _association(field: str) -> Callable[[Any], Any]:
    def read(bound: Any) -> Any:
        answer = bound.association
        value = getattr(answer, "value", None)
        if value is None:
            return MISSING
        if field == "scope_id":
            return encode(value.scope_id)
        if field == "origin_nodes":
            return encode(value.origin_nodes)
        for item in value.operands:
            if item.operand == field:
                return encode(item.tensor)
        return MISSING

    return read


def _tensor(operand: str, *, optional: bool = False) -> Callable[[Any], Any]:
    """The graph name of one operand, read from the frozen source.

    From the source rather than from the association, deliberately.  The
    oracle's description was a *source* projection: it existed whether or not a
    Design applied.  ``SourceAssociation`` is read off a resolved Network, and a
    node with no applicable Design -- a fused-threshold one, here -- has none.
    Comparing the source reading keeps every fixture in the comparison;
    ``test_the_association_carries_the_same_identities`` then checks that the
    association agrees with it wherever a Network does resolve.
    """

    def read(bound: Any) -> Any:
        source = bound.source
        if not source.has(operand):
            return None if optional else MISSING
        return encode(source.operand(operand).tensor)

    return read


def _scope_id(bound: Any) -> Any:
    return encode(bound.binding.node_identity)


def _origin_nodes(bound: Any) -> Any:
    from finn.dataflow.ops.mvau.op import origin_nodes  # noqa: PLC0415

    return encode(origin_nodes(bound.source))


def _threshold_operand_id(bound: Any) -> Any:
    source = bound.source
    return encode(source.operand("threshold").tensor) if source.has("threshold") else None


def _leading_shape(bound: Any) -> Any:
    return encode(tuple(bound.source.operand("activation").shape[:-1]))


def _threshold_shape(bound: Any) -> Any:
    source = bound.source
    return encode(tuple(source.operand("threshold").shape)) if source.has("threshold") else None


def _output_element_type(bound: Any) -> Any:
    """The oracle's one field that this stack answers from two places.

    A fused node's output datatype is a *source fact* it reads and owns; a plain
    node's output is the accumulator it produces, derived rather than read.  The
    two-case split is the disposition, not a gap in it: one field there answers
    two different questions, and which one it was answering depended on the node.
    """

    if bound.profile.fuses_activation:
        return encode(bound.output_type)
    return encode(bound.accumulator_type)


#: Every field of the oracle's ``MVAUProblem``.  A field of that dataclass with
#: no entry here fails ``test_the_table_covers_every_oracle_field``.
PROBLEM_TABLE: tuple[Entry, ...] = (
    Entry("repetitions", EQUAL, "repetitions", _derived("repetitions"), _member("repetitions")),
    Entry("matrix_width", EQUAL, "matrix_width", _derived("matrix_width"), _member("matrix_width")),
    Entry(
        "matrix_height",
        EQUAL,
        "matrix_height",
        _derived("matrix_height"),
        _member("matrix_height"),
    ),
    Entry(
        "activation_element_type",
        REPRESENTED_BY,
        "the activation operand's datatype facet",
        _problem("problem.mvau.activation.datatype"),
        _operand("activation", "datatype"),
    ),
    Entry(
        "weight_element_type",
        REPRESENTED_BY,
        "the weight operand's datatype facet",
        _problem("problem.mvau.weight.datatype"),
        _operand("weight", "datatype"),
    ),
    Entry(
        "accumulator_element_type",
        REPRESENTED_BY,
        "accumulator_type, the accDataType attribute",
        _problem("problem.mvau.accumulator_element_type"),
        _member("accumulator_type"),
    ),
    Entry(
        "output_element_type",
        REPRESENTED_BY,
        "output_type when the node fuses a threshold; accumulator_type when it does not",
        _problem("problem.mvau.output.datatype"),
        _output_element_type,
    ),
    Entry(
        "threshold_element_type",
        REPRESENTED_BY,
        "the threshold operand's datatype facet; absent when the operand is",
        _problem("problem.mvau.threshold.datatype"),
        _operand("threshold", "datatype"),
    ),
    Entry(
        "threshold_initializer_available",
        REPRESENTED_BY,
        "the threshold operand's initializer facet",
        _problem("problem.mvau.threshold.initializer.present"),
        _operand("threshold", "initializer"),
    ),
    Entry(
        "computation_profile",
        REPRESENTED_BY,
        "accumulation x activation -- one value on the oracle's side and a pair on this "
        "one, compared through the six-row mapping",
        _derived("computation_profile"),
        _profile_pair,
        equal=lambda oracle_value, local_value: _profiles_correspond(oracle_value, local_value),
    ),
    Entry(
        "weight_initializer_available",
        REPRESENTED_BY,
        "the weight operand's initializer facet",
        _problem("problem.mvau.weight.initializer.present"),
        _operand("weight", "initializer"),
    ),
    Entry(
        "weight_initializer_fingerprint",
        REPRESENTED_BY,
        "the weight operand's initializer_digest facet, compared as an equivalence",
        _problem("problem.mvau.weight.initializer.fingerprint"),
        _operand("weight", "initializer_digest"),
    ),
    Entry(
        "threshold_initializer_fingerprint",
        REPRESENTED_BY,
        "the threshold operand's initializer_digest facet, compared as an equivalence",
        _problem("problem.mvau.threshold.initializer.fingerprint"),
        _operand("threshold", "initializer_digest"),
    ),
    Entry(
        "source_description",
        REPRESENTED_BY,
        "the wrapper is retired; SourceAssociation and the source reading own its eight "
        "fields, each disposed of in DESCRIPTION_TABLE",
        compared_in="test_association_parity",
    ),
    Entry(
        "initializer_excludes_minimum",
        REPRESENTED_BY,
        "weight_excludes_minimum, the same InitializerAnalysis",
        _problem("problem.mvau.initializer_excludes_minimum"),
        _member("weight_excludes_minimum"),
    ),
    Entry(
        "runtime_weight_range_contract",
        REPRESENTED_BY,
        "the runtime_weight_range_contract build fact",
        _problem("problem.mvau.runtime_weight_range_contract"),
        _member("runtime_weight_range_contract"),
    ),
    Entry(
        "runtime_writable",
        REPRESENTED_BY,
        "the runtime_writable_weights build fact",
        _problem("problem.cyclic_parameter.runtime_writable"),
        _member("runtime_writable_weights"),
    ),
    Entry(
        "external_weight_sequence",
        MOVED_TO,
        "an external supply sequence is a physical-composition fact and no Design "
        "in this round consumes one",
        target="U6",
    ),
    Entry(
        "accumulator_type_analysis_owner",
        MOVED_TO,
        "provenance of the accumulator-width analysis belongs to the pass that performs "
        "it; this stack performs none",
        target="U7",
    ),
    Entry(
        "target_dsp_block",
        REPRESENTED_BY,
        "the target_dsp build fact",
        _problem("problem.target.dsp_block"),
        _member("target_dsp"),
    ),
    Entry(
        "target_fpga_part",
        MOVED_TO,
        "the part reaches the physical layer, not the design space; target_dsp is what "
        "the Kernels read",
        target="U6",
    ),
    Entry(
        "target_clock_period_ns",
        REPRESENTED_BY,
        "the clock_period_ns build fact",
        _problem("problem.target.clock_period_ns"),
        _member("clock_period_ns"),
    ),
    Entry(
        "target_memory_capabilities",
        MOVED_TO,
        "a memory-capability record is consumed by the parameter composition, which this "
        "round does not build",
        target="U6",
    ),
)


#: Every field of the oracle's ``MVAUSourceDescription``.  The wrapper itself is
#: retired -- ``SourceAssociation`` owns what it carried -- so there is no
#: structural counterpart to compare, only these.
DESCRIPTION_TABLE: tuple[Entry, ...] = (
    Entry(
        "source_node_id",
        REPRESENTED_BY,
        "SourceAssociation.scope_id, and the frozen binding's node identity",
        _description("source_node_id"),
        _scope_id,
    ),
    Entry(
        "activation_operand_id",
        REPRESENTED_BY,
        "the activation OperandAssociation's tensor",
        _description("activation_operand_id"),
        _tensor("activation"),
    ),
    Entry(
        "weight_operand_id",
        REPRESENTED_BY,
        "the weight OperandAssociation's tensor",
        _description("weight_operand_id"),
        _tensor("weight"),
    ),
    Entry(
        "output_operand_id",
        REPRESENTED_BY,
        "the output OperandAssociation's tensor",
        _description("output_operand_id"),
        _tensor("output"),
    ),
    Entry(
        "threshold_operand_id",
        REPRESENTED_BY,
        "the threshold operand's tensor, present exactly when the operand is",
        _description("threshold_operand_id"),
        _threshold_operand_id,
    ),
    Entry(
        "leading_shape",
        DERIVED_FROM,
        "the activation operand's shape, less its last dimension",
        _description("leading_shape"),
        _leading_shape,
    ),
    Entry(
        "threshold_shape",
        DERIVED_FROM,
        "the threshold operand's reading",
        _description("threshold_shape"),
        _threshold_shape,
    ),
    Entry(
        "fused_source_node_ids",
        REPRESENTED_BY,
        "SourceAssociation.origin_nodes -- the one field with no other home once nodes are fused",
        _description("fused_source_node_ids"),
        _origin_nodes,
    ),
)


#: Facts this stack projects that the oracle had no field for.  Recorded rather
#: than left out: a table that listed only correspondences would read as though
#: nothing was added.
ADDITIONS = (
    "OperandAssociation.destination -- where each operand actually goes, which the "
    "oracle's description could not say because it named boundaries only",
    "OperandAssociation.correspondence and selected_shape -- the coordinate mapping "
    "between the source tensor and the selected port",
    "the operand presence facets -- a threshold's absence is an ordinary reading here, "
    "where the oracle made it a schema rule and had nothing to report",
)


def moved(table: tuple[Entry, ...] = PROBLEM_TABLE) -> tuple[Entry, ...]:
    return tuple(item for item in table if item.mark == MOVED_TO)


def comparable(table: tuple[Entry, ...]) -> tuple[Entry, ...]:
    """The entries this table compares itself, field by field."""

    return tuple(
        item
        for item in table
        if item.mark != MOVED_TO and item.read_oracle is not None and item.read_local is not None
    )


def delegated(table: tuple[Entry, ...]) -> tuple[Entry, ...]:
    """The entries that name another test as their comparison."""

    return tuple(item for item in table if item.compared_in)


__all__ = [
    "ADDITIONS",
    "DERIVED_FROM",
    "DESCRIPTION_TABLE",
    "EQUAL",
    "MARKS",
    "MISSING",
    "MOVED_TO",
    "PROBLEM_TABLE",
    "REMOVED_BECAUSE",
    "REPRESENTED_BY",
    "Entry",
    "comparable",
    "encode",
    "moved",
]
