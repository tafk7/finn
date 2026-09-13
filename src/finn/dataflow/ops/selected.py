# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached selected ONNX snapshots and compact logical declarations.

This module is the ONNX-facing compiler adapter. The canonical Region/Network
model remains independent of ONNX, while operation-specific construction rules
remain outside this generic decoder.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, NoReturn, TypeVar, cast

from onnx import (  # type: ignore[import-not-found]
    ModelProto,
    TensorProto,
    checker,
    helper as onnx_helper,
)
from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]
    initializer_value_summaries,
)
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Finding
from finn.dataflow.model.datatypes import QONNXDataType, resolve_qonnx_datatype_name
from finn.dataflow.model.maps import (
    AffineRankMap,
    CoordinateMap,
    CoordinateSet,
    ExplicitCoordinateMap,
    IdentityCoordinateMap,
    RectangularDomain,
    decode_coordinate_map,
    encode_coordinate_map,
    encoding_is_json_shaped,
)
from finn.dataflow.model.network import (
    DataflowNetwork,
    RegionEndpoint,
)
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.region import (
    InputInterface,
    InternalInput,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer
from finn.dataflow.ops.native import (
    ChoiceAttribute,
    choice_subset,
    decode_choice_value,
    encode_choice_value,
)
from finn.dataflow.space.declarations import ValueSource

SELECTED_METADATA_KEY = "finn.dataflow.selected"
SELECTED_DECLARATION_ID = "finn.dataflow.selected_graph"
SELECTED_DECLARATION_VERSION = 2

S = TypeVar("S")
P = TypeVar("P")


class SelectedGraphError(ValueError):
    """An attributable selected-graph construction or decoding failure."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{path}: {message} [{code}]")
        self.code = code
        self.path = path


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise SelectedGraphError(code, path, message)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _canonical_json(value: object) -> str:
    thawed = _thaw_json(value)
    if not encoding_is_json_shaped(thawed):
        raise TypeError("selected declaration values must be JSON-shaped")
    return json.dumps(thawed, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_json(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_json_object,
        parse_constant=_reject_constant,
    )


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("selected declaration mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("selected declaration floats must be finite")
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _is_deeply_immutable(value: object) -> bool:
    if value is None or type(value) in (str, int, float, bool, bytes):
        return True
    if isinstance(value, Enum):
        return True
    if isinstance(value, tuple):
        return all(_is_deeply_immutable(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_deeply_immutable(item) for item in value)
    if isinstance(value, MappingProxyType):
        return all(type(key) is str and _is_deeply_immutable(item) for key, item in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        return bool(parameters and parameters.frozen) and all(
            _is_deeply_immutable(getattr(value, field.name)) for field in fields(value)
        )
    return False


def _fields(value: object, expected: set[str], path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail("selected.declaration.type", path, "expected an object")
    if set(value) != expected:
        _fail(
            "selected.declaration.fields",
            path,
            f"expected fields {sorted(expected)}, found {sorted(value)}",
        )
    return cast("Mapping[str, object]", value)


def _string(value: object, path: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        _fail("selected.declaration.type", path, "expected a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        _fail("selected.declaration.type", path, "expected an integer")
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _fail("selected.declaration.type", path, "expected an array")
    return cast("list[object]", value)


class InterfaceDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class GraphSlotKind(str, Enum):
    GRAPH_INPUT = "graph_input"
    GRAPH_OUTPUT = "graph_output"
    NODE_INPUT = "node_input"
    NODE_OUTPUT = "node_output"
    INITIALIZER = "initializer"


class RelationKind(str, Enum):
    DIRECT = "direct"
    ROW_MAJOR_RESHAPE = "row_major_reshape"
    TRANSPOSE_2D = "transpose_2d"


class SourceDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class OwnerKind(str, Enum):
    REGION = "region"
    SOURCE_BOUNDARY = "source_boundary"


@dataclass(frozen=True, slots=True)
class QualifiedInterfaceRef:
    node_id: str
    direction: InterfaceDirection
    operand_id: str
    interface_id: str | None


@dataclass(frozen=True, slots=True)
class GraphSlotRef:
    kind: GraphSlotKind
    owner: str
    index: int | None


@dataclass(frozen=True, slots=True, init=False)
class PositionRelation:
    coordinate_map: CoordinateMap

    def __init__(self, coordinate_map: CoordinateMap) -> None:
        if not isinstance(
            coordinate_map,
            (IdentityCoordinateMap, AffineRankMap, ExplicitCoordinateMap),
        ):
            raise TypeError("selected bindings require a reusable compact CoordinateMap")
        if isinstance(coordinate_map, IdentityCoordinateMap):
            if not coordinate_map.domain.is_full:
                raise ValueError("an identity binding relation must cover its source domain")
            if coordinate_map.domain.ambient != coordinate_map.target_domain:
                raise ValueError("an identity binding relation needs equal source/target domains")
        elif isinstance(coordinate_map, ExplicitCoordinateMap):
            source = coordinate_map.source_domain
            target = coordinate_map.target_domain
            if source is None or target is None:
                raise ValueError("an explicit binding relation needs typed domains")
            sources = tuple(item[0] for item in coordinate_map.entries)
            if len(sources) != len(set(sources)):
                raise ValueError("an explicit binding relation repeats a source position")
            if len(sources) != source.cardinality:
                raise ValueError("an explicit binding relation must cover every source position")
            if any(not source.contains(item) for item in sources):
                raise ValueError("an explicit binding relation has an out-of-domain source")
            targets = tuple(item[1] for item in coordinate_map.entries)
            if any(not target.contains(item) for item in targets):
                raise ValueError("an explicit binding relation has an out-of-domain target")
            if len(targets) != len(set(targets)) or len(targets) != target.cardinality:
                raise ValueError("an explicit binding relation must be bijective")
        else:
            low = coordinate_map.offset + sum(
                min(0, coefficient * (extent - 1))
                for coefficient, extent in zip(
                    coordinate_map.coefficients, coordinate_map.view_extents
                )
            )
            high = coordinate_map.offset + sum(
                max(0, coefficient * (extent - 1))
                for coefficient, extent in zip(
                    coordinate_map.coefficients, coordinate_map.view_extents
                )
            )
            if low < 0 or high >= coordinate_map.target.cardinality:
                raise ValueError("an affine binding relation reaches outside its target domain")
            if not coordinate_map.is_bijection:
                raise ValueError("an affine binding relation must be bijective")
        object.__setattr__(self, "coordinate_map", coordinate_map)

    @classmethod
    def direct(
        cls, source_domain: RectangularDomain, target_domain: RectangularDomain
    ) -> PositionRelation:
        return cls(IdentityCoordinateMap(CoordinateSet.full(source_domain), target_domain))

    @classmethod
    def row_major_reshape(
        cls, source_domain: RectangularDomain, target_domain: RectangularDomain
    ) -> PositionRelation:
        return cls(AffineRankMap.row_major_reshape(source_domain, target_domain))

    @classmethod
    def transpose_2d(
        cls, source_domain: RectangularDomain, target_domain: RectangularDomain
    ) -> PositionRelation:
        if len(source_domain.extents) != 2 or target_domain.extents != tuple(
            reversed(source_domain.extents)
        ):
            raise ValueError("transpose_2d requires reversed rank-two extents")
        rows, columns = source_domain.extents
        return cls(
            AffineRankMap.from_mixed_radix(
                source_domain,
                view_extents=(rows, columns),
                target=target_domain,
                offset=0,
                coefficients=(1, rows),
            )
        )

    @property
    def source_domain(self) -> RectangularDomain:
        if isinstance(self.coordinate_map, IdentityCoordinateMap):
            return self.coordinate_map.domain.ambient
        if isinstance(self.coordinate_map, ExplicitCoordinateMap):
            source = self.coordinate_map.source_domain
            if source is None:
                raise ValueError("explicit relation has no source domain")
            return source
        return self.coordinate_map.source

    @property
    def target_domain(self) -> RectangularDomain:
        if isinstance(self.coordinate_map, IdentityCoordinateMap):
            target = self.coordinate_map.target_domain
            if target is None:
                raise ValueError("identity relation has no target domain")
            return target
        if isinstance(self.coordinate_map, ExplicitCoordinateMap):
            target = self.coordinate_map.target_domain
            if target is None:
                raise ValueError("explicit relation has no target domain")
            return target
        return self.coordinate_map.target

    @property
    def kind(self) -> RelationKind | None:
        if isinstance(self.coordinate_map, IdentityCoordinateMap):
            return RelationKind.DIRECT
        if isinstance(self.coordinate_map, ExplicitCoordinateMap):
            return None
        reshape = AffineRankMap.row_major_reshape(self.source_domain, self.target_domain)
        if self.coordinate_map == reshape:
            return RelationKind.ROW_MAJOR_RESHAPE
        if len(self.source_domain.extents) == 2 and self.target_domain.extents == tuple(
            reversed(self.source_domain.extents)
        ):
            rows, columns = self.source_domain.extents
            transpose = AffineRankMap.from_mixed_radix(
                self.source_domain,
                view_extents=(rows, columns),
                target=self.target_domain,
                offset=0,
                coefficients=(1, rows),
            )
            if self.coordinate_map == transpose:
                return RelationKind.TRANSPOSE_2D
        return None

    def mapped(self, coordinate: tuple[int, ...]) -> tuple[int, ...]:
        return self.coordinate_map.mapped(coordinate)


@dataclass(frozen=True, slots=True)
class InterfaceBinding:
    interface: QualifiedInterfaceRef
    graph_value: str
    relation: PositionRelation
    anchors: tuple[GraphSlotRef, ...]


@dataclass(frozen=True, slots=True)
class SourceOperandKey:
    operand_id: str
    direction: SourceDirection
    index: int


@dataclass(frozen=True, slots=True)
class SourceValueRef:
    key: SourceOperandKey
    shape: tuple[int, ...]
    carrier_dtype: int
    logical_datatype: str
    initializer_content_digest: str | None

    @property
    def initializer_present(self) -> bool:
        return self.initializer_content_digest is not None


@dataclass(frozen=True, slots=True)
class ConstructionInputs:
    initializers: tuple[tuple[SourceOperandKey, FrozenInitializer], ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.initializers)
        keys = tuple(key for key, _value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("construction initializer keys must be unique")
        object.__setattr__(
            self, "initializers", tuple(sorted(values, key=lambda item: repr(item[0])))
        )

    def initializer(self, key: SourceOperandKey) -> FrozenInitializer:
        matches = tuple(value for candidate, value in self.initializers if candidate == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]


@dataclass(frozen=True, slots=True)
class SourceValueBinding:
    source: SourceOperandKey
    graph_value: str
    relation: PositionRelation
    anchors: tuple[GraphSlotRef, ...]


@dataclass(frozen=True, slots=True)
class RequiredSupply:
    required_input: QualifiedInterfaceRef
    root_graph_value: str
    source: SourceOperandKey
    derivation_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComputationOwner:
    kind: OwnerKind
    owner_id: str
    node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphNodeBinding:
    node_id: str
    index: int


@dataclass(frozen=True, slots=True)
class ConstructionIdentity:
    family: str
    version: str
    form: str = "canonical"
    form_version: int = 1
    form_arguments: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.family or not self.version or not self.form:
            raise ValueError("construction identity fields must be non-empty")
        if type(self.form_version) is not int or self.form_version < 1:
            raise ValueError("construction form version must be a positive integer")
        arguments = tuple(self.form_arguments)
        names = tuple(name for name, _value in arguments)
        if len(names) != len(set(names)) or names != tuple(sorted(names)):
            raise ValueError("construction form arguments must have unique sorted names")
        frozen_arguments = []
        for name, value in arguments:
            if type(name) is not str or not name:
                raise ValueError("construction form argument names must be non-empty strings")
            frozen_arguments.append((name, _freeze_json(value)))
        object.__setattr__(self, "form_arguments", tuple(frozen_arguments))


@dataclass(frozen=True, slots=True)
class RecordedChoice:
    path: str
    value: object
    encoding: object | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("recorded choice path must be non-empty")
        if self.encoding is not None:
            object.__setattr__(self, "encoding", _freeze_json(self.encoding))


@dataclass(frozen=True, slots=True)
class EncodedSourceSemantics:
    identity: str
    version: int
    payload: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json(self.payload))


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    schema_version: int
    problem_fingerprint: str
    scope_id: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("native schema origin must be a positive integer")
        if type(self.problem_fingerprint) is not str or not self.problem_fingerprint:
            raise ValueError("origin problem fingerprint must be a non-empty string")
        if self.scope_id is not None and (type(self.scope_id) is not str or not self.scope_id):
            raise ValueError("origin scope_id must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    family: str
    family_version: str
    operands: tuple[SourceValueRef, ...]
    semantics: EncodedSourceSemantics
    origin: SourceOrigin
    semantic_fingerprint: str

    @property
    def schema_version(self) -> int:
        return self.origin.schema_version

    @property
    def problem_fingerprint(self) -> str:
        return self.origin.problem_fingerprint

    @property
    def scope_id(self) -> str | None:
        return self.origin.scope_id

    @classmethod
    def create(
        cls,
        *,
        family: str,
        family_version: str,
        schema_version: int,
        problem_fingerprint: str,
        scope_id: str | None,
        operands: Iterable[SourceValueRef],
        semantics: EncodedSourceSemantics,
    ) -> SourceProvenance:
        result = cls(
            family,
            family_version,
            tuple(operands),
            semantics,
            SourceOrigin(schema_version, problem_fingerprint, scope_id),
            "",
        )
        return replace(result, semantic_fingerprint=_source_semantic_fingerprint(result))


@dataclass(frozen=True, slots=True)
class SelectionFacts(Generic[S, P]):
    construction: ConstructionIdentity
    source: SourceProvenance
    source_semantics: S
    choices: tuple[RecordedChoice, ...]
    parameters: P

    @property
    def selection_fingerprint(self) -> str:
        return _digest_json(
            {
                "construction": _encode_construction(self.construction),
                "source_semantic_fingerprint": self.source.semantic_fingerprint,
                "choices": [_encode_choice(choice) for choice in self.choices],
            }
        )


@dataclass(frozen=True, slots=True)
class SelectedInitializerInput:
    """One declared payload input and its operation-owned mode predicate."""

    key: SourceOperandKey
    source: ValueSource[FrozenInitializer]
    required: Callable[[SelectionFacts[Any, Any]], bool]


@dataclass(frozen=True, slots=True)
class SelectedGraphDeclaration:
    identity: str
    version: int
    graph_digest: str
    construction: ConstructionIdentity
    source: SourceProvenance
    choices: tuple[RecordedChoice, ...]
    graph_nodes: tuple[GraphNodeBinding, ...]
    interface_bindings: tuple[InterfaceBinding, ...]
    source_bindings: tuple[SourceValueBinding, ...]
    supplies: tuple[RequiredSupply, ...]
    ownership: tuple[ComputationOwner, ...]


@dataclass(frozen=True, slots=True)
class DecodedSelectedGraph(Generic[S, P]):
    snapshot: SelectedGraphSnapshot
    declaration: SelectedGraphDeclaration
    network: DataflowNetwork
    selection_facts: SelectionFacts[S, P]


@dataclass(frozen=True, slots=True)
class SelectedConstruction(Generic[S, P]):
    family: str
    version: str
    source_semantics_identity: str
    source_semantics_version: int
    admitted_forms: tuple[str, ...]
    choice_paths: tuple[str, ...]
    initializer_inputs: tuple[SelectedInitializerInput, ...]
    decode_source_semantics: Callable[[EncodedSourceSemantics], S]
    derive_facts: Callable[
        [ConstructionIdentity, SourceProvenance, S, tuple[RecordedChoice, ...]],
        SelectionFacts[S, P],
    ]
    project: Callable[[SelectionFacts[S, P]], DataflowNetwork]
    construct: Callable[[SelectionFacts[S, P], ConstructionInputs], SelectedGraphSnapshot]
    verify: Callable[[SelectedGraphSnapshot, SelectionFacts[S, P]], tuple[Finding, ...]]


@dataclass(frozen=True, slots=True)
class ConstructionRegistry:
    entries: Mapping[tuple[str, str], SelectedConstruction[Any, Any]]
    choice_schemas: Mapping[tuple[str, str], tuple[ChoiceAttribute, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "choice_schemas", MappingProxyType(dict(self.choice_schemas)))
        if not set(self.choice_schemas).issubset(self.entries):
            raise ValueError("choice schemas must belong to registered constructions")
        for key, construction in self.entries.items():
            schema = self.choice_schemas.get(key, ())
            if construction.choice_paths and not schema:
                raise ValueError("registered construction has no compiled choice schema")
            choice_subset(schema, construction.choice_paths)

    def resolve(self, identity: ConstructionIdentity) -> SelectedConstruction[Any, Any]:
        try:
            construction = self.entries[(identity.family, identity.version)]
        except KeyError:
            _fail(
                "selected.construction.unsupported",
                "declaration.construction",
                f"unsupported construction {identity.family!r} version {identity.version!r}",
            )
        if identity.form not in construction.admitted_forms:
            _fail(
                "selected.construction.form_unsupported",
                "declaration.construction.form",
                f"unsupported construction form {identity.form!r}",
            )
        return construction

    def resolve_choice_schema(self, identity: ConstructionIdentity) -> tuple[ChoiceAttribute, ...]:
        construction = self.resolve(identity)
        key = (identity.family, identity.version)
        try:
            schema = self.choice_schemas[key]
        except KeyError:
            if construction.choice_paths:
                _fail(
                    "selected.construction.choice_schema",
                    "declaration.construction",
                    "construction registry has no compiled choice schema",
                )
            return ()
        try:
            return choice_subset(schema, construction.choice_paths)
        except (TypeError, ValueError) as error:
            _fail(
                "selected.construction.choice_schema",
                "declaration.construction",
                str(error),
            )


EMPTY_CONSTRUCTION_REGISTRY = ConstructionRegistry({})


def encode_selected_choices(
    schema: tuple[ChoiceAttribute, ...], choices: tuple[RecordedChoice, ...]
) -> tuple[RecordedChoice, ...]:
    """Validate typed choices and attach their shared canonical encodings."""

    if tuple(item.path for item in choices) != tuple(item.choice.path for item in schema):
        raise ValueError("selected choices do not match the declared ordered choice_paths")
    result = []
    for item, choice in zip(schema, choices):
        encoded = encode_choice_value(item, choice.value)
        decoded = decode_choice_value(item, encoded)
        result.append(RecordedChoice(choice.path, decoded, _freeze_json(encoded)))
    return tuple(result)


def decode_selected_choices(
    schema: tuple[ChoiceAttribute, ...], choices: tuple[RecordedChoice, ...]
) -> tuple[RecordedChoice, ...]:
    """Decode one exact selected subset through compiled declaration codecs."""

    if tuple(item.path for item in choices) != tuple(item.choice.path for item in schema):
        _fail(
            "selected.construction.choice_subset",
            "declaration.choices",
            "choices do not match the construction's ordered choice_paths",
        )
    decoded = []
    for item, choice in zip(schema, choices):
        encoded = choice.value if choice.encoding is None else choice.encoding
        try:
            value = decode_choice_value(item, _thaw_json(encoded))
        except (TypeError, ValueError) as error:
            _fail(
                "selected.construction.choice_value",
                f"declaration.choices.{choice.path}",
                str(error),
            )
        decoded.append(RecordedChoice(choice.path, value, _freeze_json(encoded)))
    return tuple(decoded)


@dataclass(frozen=True, slots=True)
class _GraphValueFact:
    shape: tuple[int, ...]
    carrier_dtype: int
    logical_datatype: QONNXDataType | None


@dataclass(frozen=True, slots=True)
class SelectedGraphSnapshot:
    model_bytes: bytes

    def __post_init__(self) -> None:
        model = ModelProto()
        try:
            model.ParseFromString(bytes(self.model_bytes))
        except Exception as error:
            raise SelectedGraphError(
                "selected.graph.parse", "snapshot", f"invalid ModelProto bytes: {error}"
            ) from error
        object.__setattr__(self, "model_bytes", bytes(self.model_bytes))

    @classmethod
    def capture(cls, model: ModelWrapper) -> SelectedGraphSnapshot:
        copied = ModelProto()
        copied.ParseFromString(model.model.SerializeToString(deterministic=True))
        metadata = sorted(copied.graph.metadata_props, key=lambda item: (item.key, item.value))
        del copied.graph.metadata_props[:]
        copied.graph.metadata_props.extend(metadata)
        encoded = copied.SerializeToString(deterministic=True)
        reparsed = ModelProto()
        reparsed.ParseFromString(encoded)
        if reparsed.SerializeToString(deterministic=True) != encoded:
            _fail("selected.graph.nondeterministic", "snapshot", "protobuf did not round trip")
        return cls(encoded)

    def model_copy(self) -> ModelWrapper:
        model = ModelProto()
        model.ParseFromString(self.model_bytes)
        return ModelWrapper(model)

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(self.model_bytes).hexdigest()

    @property
    def graph_digest(self) -> str:
        return _graph_digest(self.model_copy())

    @property
    def declaration(self) -> SelectedGraphDeclaration:
        return _read_declaration(self.model_copy())


def _metadata_values(model: ModelWrapper, key: str) -> tuple[str, ...]:
    return tuple(item.value for item in model.graph.metadata_props if item.key == key)


def _drop_metadata(model: ModelWrapper, key: str) -> None:
    kept = [item for item in model.graph.metadata_props if item.key != key]
    del model.graph.metadata_props[:]
    model.graph.metadata_props.extend(kept)


def _set_metadata(model: ModelWrapper, key: str, value: str) -> None:
    from onnx import StringStringEntryProto  # noqa: PLC0415

    _drop_metadata(model, key)
    item = StringStringEntryProto(key=key, value=value)
    model.graph.metadata_props.append(item)


def _graph_digest(model: ModelWrapper) -> str:
    copied = model.model.__class__()
    copied.ParseFromString(model.model.SerializeToString(deterministic=True))
    wrapper = ModelWrapper(copied)
    _drop_metadata(wrapper, SELECTED_METADATA_KEY)
    metadata = sorted(copied.graph.metadata_props, key=lambda item: (item.key, item.value))
    del copied.graph.metadata_props[:]
    copied.graph.metadata_props.extend(metadata)
    return hashlib.sha256(copied.SerializeToString(deterministic=True)).hexdigest()


def build_selected_snapshot(
    model: ModelWrapper, declaration: SelectedGraphDeclaration
) -> SelectedGraphSnapshot:
    copied = ModelProto()
    copied.ParseFromString(model.model.SerializeToString(deterministic=True))
    wrapper = ModelWrapper(copied)
    _drop_metadata(wrapper, SELECTED_METADATA_KEY)
    current = replace(declaration, graph_digest=_graph_digest(wrapper))
    _set_metadata(wrapper, SELECTED_METADATA_KEY, encode_selected_declaration(current))
    return SelectedGraphSnapshot.capture(wrapper)


def encode_selected_declaration(declaration: SelectedGraphDeclaration) -> str:
    return _canonical_json(_encode_declaration(declaration))


def decode_selected_declaration(text: str) -> SelectedGraphDeclaration:
    try:
        value = _parse_json(text)
        if isinstance(value, Mapping):
            identity = _string(value.get("identity"), "declaration.identity")
            version = _integer(value.get("version"), "declaration.version")
            if identity == SELECTED_DECLARATION_ID and version == 1:
                _fail(
                    "selected.declaration.v1_unsupported",
                    "declaration.version",
                    "selected declaration v1 is unsupported; republish from native source",
                )
            if identity != SELECTED_DECLARATION_ID:
                _fail(
                    "selected.declaration.identity",
                    "declaration.identity",
                    "unknown selected declaration identity",
                )
            if version != SELECTED_DECLARATION_VERSION:
                _fail(
                    "selected.declaration.version",
                    "declaration.version",
                    "unsupported selected declaration version",
                )
        return _decode_declaration(value)
    except SelectedGraphError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise SelectedGraphError(
            "selected.declaration.invalid", "declaration", str(error)
        ) from error


def _read_declaration(model: ModelWrapper) -> SelectedGraphDeclaration:
    values = _metadata_values(model, SELECTED_METADATA_KEY)
    if len(values) != 1:
        _fail(
            "selected.declaration.missing_or_duplicate",
            "model.metadata",
            f"expected one {SELECTED_METADATA_KEY!r} entry, found {len(values)}",
        )
    return decode_selected_declaration(values[0])


def decode_selected_graph(
    snapshot: SelectedGraphSnapshot,
    *,
    constructions: ConstructionRegistry | None = None,
    expected_network: DataflowNetwork | None = None,
    expected_source: SourceProvenance | None = None,
) -> DecodedSelectedGraph[Any, Any]:
    if constructions is None:
        from finn.dataflow.ops.selected_registry import (  # noqa: PLC0415 - avoid recipe cycles
            DEFAULT_SELECTED_CONSTRUCTIONS,
        )

        constructions = DEFAULT_SELECTED_CONSTRUCTIONS
    model = snapshot.model_copy()
    declaration = _read_declaration(model)
    if declaration.identity != SELECTED_DECLARATION_ID:
        _fail("selected.declaration.identity", "declaration.identity", "unknown identity")
    if declaration.version != SELECTED_DECLARATION_VERSION:
        _fail("selected.declaration.version", "declaration.version", "unsupported version")
    if declaration.graph_digest != snapshot.graph_digest:
        _fail("selected.graph.digest_mismatch", "declaration.graph_digest", "stale graph digest")
    try:
        checker.check_model(model.model)
    except Exception as error:
        raise SelectedGraphError("selected.graph.onnx_invalid", "model", str(error)) from error

    _validate_declaration_uniqueness(declaration)
    _validate_source(declaration.source)
    if expected_source is not None and declaration.source != expected_source:
        _fail(
            "selected.source.expected_mismatch",
            "declaration.source",
            "selected source provenance differs from the construction facts",
        )
    construction = constructions.resolve(declaration.construction)
    choice_schema = constructions.resolve_choice_schema(declaration.construction)
    decoded_choices = decode_selected_choices(choice_schema, declaration.choices)
    encoded = declaration.source.semantics
    if (
        encoded.identity != construction.source_semantics_identity
        or encoded.version != construction.source_semantics_version
    ):
        _fail(
            "selected.source.semantics_version",
            "declaration.source.semantics",
            "source semantics are incompatible with the construction",
        )
    semantics = construction.decode_source_semantics(encoded)
    facts = construction.derive_facts(
        declaration.construction,
        declaration.source,
        semantics,
        decoded_choices,
    )
    if not _is_deeply_immutable(facts.source_semantics):
        _fail(
            "selected.construction.mutable_semantics",
            "selection_facts.source_semantics",
            "decoded source semantics must be deeply immutable",
        )
    if not _is_deeply_immutable(facts.parameters):
        _fail(
            "selected.construction.mutable_parameters",
            "selection_facts.parameters",
            "derived construction parameters must be deeply immutable",
        )
    try:
        network = construction.project(facts)
        network_report = validate_network(network)
        if network_report.issues:
            issue = network_report.issues[0]
            _fail(
                f"selected.projection.network.{issue.code}",
                f"network.{issue.path}",
                issue.message,
            )
    except SelectedGraphError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise SelectedGraphError(
            "selected.projection.invalid", "declaration.construction", str(error)
        ) from error
    if expected_network is not None and network != expected_network:
        _fail(
            "selected.projection.network_mismatch",
            "declaration.construction",
            "derived Network differs from the expected projection",
        )

    nodes = _resolve_graph_nodes(model, declaration.graph_nodes)
    supply_roots = {item.root_graph_value for item in declaration.supplies}
    initializer_names = {item.name for item in model.graph.initializer}
    values = _graph_value_facts(model, logical_type_exempt=initializer_names - supply_roots)
    _validate_source_bindings(declaration, model, nodes, values)
    externally_derived_sources = _validate_source_boundary_paths(declaration, model, nodes, values)
    _validate_interface_bindings(declaration, network, model, nodes, values)
    _validate_edge_bindings(declaration, network)
    _validate_supplies_and_constants(
        declaration,
        network,
        model,
        nodes,
        values,
        externally_derived_sources,
    )
    _validate_ownership(declaration, network, nodes)
    findings = construction.verify(snapshot, facts)
    if findings:
        first = findings[0]
        _fail(
            "selected.construction.verification",
            "declaration.construction",
            f"{first.path}: {first.message} [{first.code}]",
        )
    return DecodedSelectedGraph(snapshot, declaration, network, facts)


def _validate_declaration_uniqueness(declaration: SelectedGraphDeclaration) -> None:
    for path, values in (
        ("declaration.choices", tuple(item.path for item in declaration.choices)),
        ("declaration.graph_nodes", tuple(item.node_id for item in declaration.graph_nodes)),
    ):
        if len(values) != len(set(values)):
            _fail("selected.declaration.duplicate", path, "identities must be unique")


def reconstruct_selected_graph(
    source: bytes | str | Path | ModelProto,
    *,
    constructions: ConstructionRegistry | None = None,
) -> DecodedSelectedGraph[Any, Any]:
    if isinstance(source, bytes):
        snapshot = SelectedGraphSnapshot(source)
    elif isinstance(source, ModelProto):
        snapshot = SelectedGraphSnapshot(source.SerializeToString(deterministic=True))
    else:
        snapshot = SelectedGraphSnapshot(
            Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
        )
    return decode_selected_graph(snapshot, constructions=constructions)


def construct_selected_graph(
    construction: SelectedConstruction[Any, Any],
    facts: SelectionFacts[Any, Any],
    construction_inputs: ConstructionInputs,
    *,
    expected_network: DataflowNetwork,
    expected_source: SourceProvenance,
    choice_schema: tuple[ChoiceAttribute, ...] = (),
) -> SelectedGraphSnapshot:
    """Run one detached recipe and cross-check both retained authorities."""

    if facts.source != expected_source:
        _fail(
            "selected.source.fact_mismatch",
            "selection_facts.source",
            "construction facts differ from the expected source provenance",
        )
    if construction.choice_paths and not choice_schema:
        from finn.dataflow.ops.selected_registry import (  # noqa: PLC0415
            DEFAULT_SELECTED_CONSTRUCTIONS,
        )

        choice_schema = DEFAULT_SELECTED_CONSTRUCTIONS.resolve_choice_schema(facts.construction)
    facts = replace(facts, choices=encode_selected_choices(choice_schema, facts.choices))
    snapshot = construction.construct(facts, construction_inputs)
    decoded = decode_selected_graph(
        snapshot,
        constructions=ConstructionRegistry(
            {(construction.family, construction.version): construction},
            {(construction.family, construction.version): choice_schema},
        ),
        expected_network=expected_network,
        expected_source=expected_source,
    )
    if decoded.selection_facts != facts:
        _fail(
            "selected.construction.fact_mismatch",
            "selection_facts",
            "decoded construction facts differ from the supplied facts",
        )
    return snapshot


def validate_construction_inputs(
    facts: SelectionFacts[Any, Any],
    inputs: ConstructionInputs,
    required: Iterable[SourceOperandKey],
) -> Mapping[SourceOperandKey, FrozenInitializer]:
    required_keys = tuple(required)
    supplied = dict(inputs.initializers)
    if set(supplied) != set(required_keys):
        _fail(
            "selected.construction.initializer_keys",
            "construction_inputs.initializers",
            "initializer payload keys do not match the selected mode",
        )
    source = {item.key: item for item in facts.source.operands}
    for key, frozen in supplied.items():
        reference = source.get(key)
        if reference is None or not reference.initializer_present:
            _fail(
                "selected.construction.initializer_source",
                f"construction_inputs.{key.operand_id}",
                "payload has no matching source initializer fact",
            )
        if reference.shape != frozen.shape:
            _fail(
                "selected.construction.initializer_shape",
                f"construction_inputs.{key.operand_id}",
                "payload shape differs from retained source facts",
            )
        if reference.carrier_dtype != frozen.carrier_dtype:
            _fail(
                "selected.construction.initializer_dtype",
                f"construction_inputs.{key.operand_id}",
                "payload carrier dtype differs from retained source facts",
            )
        if reference.initializer_content_digest != frozen.summary.content_digest:
            _fail(
                "selected.construction.initializer_digest",
                f"construction_inputs.{key.operand_id}",
                "payload content differs from retained source facts",
            )
    return MappingProxyType(supplied)


def set_frozen_initializer(model: ModelWrapper, graph_value: str, value: FrozenInitializer) -> None:
    """Copy one checked detached payload into a selected graph."""

    model.set_initializer(graph_value, value.array_copy())
    matches = [item for item in model.graph.initializer if item.name == graph_value]
    if len(matches) != 1 or int(matches[0].data_type) != value.carrier_dtype:
        _fail(
            "selected.construction.initializer_copy",
            f"initializer.{graph_value}",
            "copied initializer carrier dtype changed",
        )


def _resolve_graph_nodes(
    model: ModelWrapper, records: tuple[GraphNodeBinding, ...]
) -> Mapping[str, Any]:
    if len(records) != len(model.graph.node):
        _fail("selected.graph.node_coverage", "declaration.graph_nodes", "node count differs")
    result: dict[str, Any] = {}
    indices: set[int] = set()
    for record in records:
        if record.node_id in result or record.index in indices:
            _fail("selected.graph.node_duplicate", "declaration.graph_nodes", "duplicate node")
        if record.index < 0 or record.index >= len(model.graph.node):
            _fail("selected.graph.node_index", record.node_id, "node index is out of range")
        node = model.graph.node[record.index]
        result[record.node_id] = node
        indices.add(record.index)
    return MappingProxyType(result)


def _explicit_datatype(model: ModelWrapper, name: str, *, required: bool) -> QONNXDataType | None:
    annotations = [item for item in model.graph.quantization_annotation if item.tensor_name == name]
    values = [
        entry.value
        for item in annotations
        for entry in item.quant_parameter_tensor_names
        if entry.key == "finn_datatype"
    ]
    if not values and not required:
        return None
    if len(values) != 1:
        _fail(
            "selected.logical_datatype.missing_or_duplicate",
            f"tensor.{name}",
            f"expected one explicit finn_datatype annotation, found {len(values)}",
        )
    try:
        datatype = resolve_qonnx_datatype_name(values[0])
    except Exception as error:
        raise SelectedGraphError(
            "selected.logical_datatype.invalid", f"tensor.{name}", str(error)
        ) from error
    if datatype.name != values[0]:
        _fail(
            "selected.logical_datatype.noncanonical",
            f"tensor.{name}",
            f"logical datatype {values[0]!r} is not canonical",
        )
    return datatype


def _carrier_dtype(model: ModelWrapper, name: str) -> int:
    info = model.get_tensor_valueinfo(name)
    if info is not None:
        return int(info.type.tensor_type.elem_type)
    matches = [item for item in model.graph.initializer if item.name == name]
    if len(matches) == 1:
        return int(matches[0].data_type)
    _fail("selected.graph.value_info_missing", f"tensor.{name}", "carrier type is absent")


def _graph_value_facts(
    model: ModelWrapper, *, logical_type_exempt: set[str]
) -> Mapping[str, _GraphValueFact]:
    names = {name for node in model.graph.node for name in (*node.input, *node.output) if name}
    names.update(item.name for item in model.graph.input)
    names.update(item.name for item in model.graph.output)
    names.update(item.name for item in model.graph.value_info)
    names.update(item.name for item in model.graph.initializer)
    result = {}
    for name in sorted(names):
        shape = model.get_tensor_shape(name)
        if shape is None or any(type(extent) is not int or extent < 0 for extent in shape):
            _fail("selected.graph.shape_missing", f"tensor.{name}", "static shape is absent")
        result[name] = _GraphValueFact(
            tuple(int(extent) for extent in shape),
            _carrier_dtype(model, name),
            _explicit_datatype(model, name, required=name not in logical_type_exempt),
        )
    return MappingProxyType(result)


def _required_logical_type(fact: _GraphValueFact, path: str) -> QONNXDataType:
    if fact.logical_datatype is None:
        _fail(
            "selected.logical_datatype.missing_or_duplicate",
            path,
            "bound numeric value has no explicit finn_datatype annotation",
        )
    return fact.logical_datatype


def _resolve_slot(
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    slot: GraphSlotRef,
) -> str:
    if slot.kind is GraphSlotKind.GRAPH_INPUT:
        if slot.owner != model.graph.name:
            _fail("selected.binding.graph_owner", slot.owner, "graph input owner is absent")
        values = model.graph.input
    elif slot.kind is GraphSlotKind.GRAPH_OUTPUT:
        if slot.owner != model.graph.name:
            _fail("selected.binding.graph_owner", slot.owner, "graph output owner is absent")
        values = model.graph.output
    elif slot.kind is GraphSlotKind.INITIALIZER:
        matches = [item for item in model.graph.initializer if item.name == slot.owner]
        if slot.index is not None or len(matches) != 1:
            _fail("selected.binding.slot", slot.owner, "initializer anchor does not resolve")
        return slot.owner
    else:
        node = nodes.get(slot.owner)
        if node is None or slot.index is None:
            _fail("selected.binding.slot", slot.owner, "node slot does not resolve")
        values = node.input if slot.kind is GraphSlotKind.NODE_INPUT else node.output
    if slot.index is None or slot.index < 0 or slot.index >= len(values):
        _fail("selected.binding.slot", slot.owner, "slot index is out of range")
    actual = values[slot.index].name if hasattr(values[slot.index], "name") else values[slot.index]
    return str(actual)


def _source_semantic_fingerprint(source: SourceProvenance) -> str:
    return _digest_json(
        {
            "family": source.family,
            "family_version": source.family_version,
            "operands": [_encode_source_value(item) for item in source.operands],
            "semantics": _encode_semantics(source.semantics),
        }
    )


def _validate_source(source: SourceProvenance) -> None:
    if source.schema_version < 1:
        _fail(
            "selected.source.origin",
            "declaration.source.origin.native_schema_version",
            "native schema origin must be a positive integer",
        )
    keys = tuple(item.key for item in source.operands)
    if len(keys) != len(set(keys)):
        _fail("selected.source.operand_duplicate", "declaration.source.operands", "duplicate key")


def _validate_source_bindings(
    declaration: SelectedGraphDeclaration,
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    values: Mapping[str, _GraphValueFact],
) -> None:
    source = {item.key: item for item in declaration.source.operands}
    bound: set[SourceOperandKey] = set()
    for index, binding in enumerate(declaration.source_bindings):
        path = f"declaration.source_bindings.{index}"
        if binding.source in bound or binding.source not in source:
            _fail("selected.source.binding_coverage", path, "source key is duplicate or absent")
        fact = values.get(binding.graph_value)
        if fact is None:
            _fail("selected.source.graph_value", path, "selected graph value is absent")
        reference = source[binding.source]
        if binding.relation.source_domain.extents != reference.shape:
            _fail("selected.source.shape", path, "source relation has the wrong source shape")
        if binding.relation.target_domain.extents != fact.shape:
            _fail("selected.source.shape", path, "source relation has the wrong selected shape")
        if reference.carrier_dtype != fact.carrier_dtype:
            _fail("selected.source.carrier_dtype", path, "carrier datatypes differ")
        if (
            reference.logical_datatype
            != _required_logical_type(fact, f"tensor.{binding.graph_value}").name
        ):
            _fail("selected.source.logical_datatype", path, "logical datatypes differ")
        try:
            binding.relation.coordinate_map
        except (TypeError, ValueError) as error:
            raise SelectedGraphError("selected.source.relation", path, str(error)) from error
        for anchor in binding.anchors:
            if _resolve_slot(model, nodes, anchor) != binding.graph_value:
                _fail("selected.source.anchor_value", path, "anchor and graph value differ")
        if not binding.anchors:
            _fail("selected.source.anchor_missing", path, "source binding has no graph anchor")
        bound.add(binding.source)
    if bound != set(source):
        _fail(
            "selected.source.binding_coverage",
            "declaration.source_bindings",
            "source operands are not bound exactly once",
        )


def _binding_map(
    bindings: tuple[InterfaceBinding, ...],
) -> Mapping[QualifiedInterfaceRef, InterfaceBinding]:
    result = {}
    for binding in bindings:
        if binding.interface in result:
            _fail(
                "selected.binding.duplicate",
                "declaration.interface_bindings",
                f"duplicate interface {binding.interface!r}",
            )
        result[binding.interface] = binding
    return MappingProxyType(result)


def _resolve_interface(network: DataflowNetwork, reference: QualifiedInterfaceRef) -> Any:
    region = network.node(reference.node_id).region
    if reference.direction is InterfaceDirection.INPUT:
        item = region.input(reference.operand_id)
        if reference.interface_id is None:
            if not isinstance(item, InternalInput):
                raise KeyError(reference)
        elif not isinstance(item, InputInterface) or item.port.id != reference.interface_id:
            raise KeyError(reference)
        return item
    matches = [
        item
        for item in region.outputs
        if item.port.operand.id == reference.operand_id and item.port.id == reference.interface_id
    ]
    if len(matches) != 1:
        raise KeyError(reference)
    return matches[0]


def _validate_interface_bindings(
    declaration: SelectedGraphDeclaration,
    network: DataflowNetwork,
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    values: Mapping[str, _GraphValueFact],
) -> None:
    region_owners = {
        owner.owner_id: frozenset(owner.node_ids)
        for owner in declaration.ownership
        if owner.kind is OwnerKind.REGION
    }
    expected = {
        QualifiedInterfaceRef(
            node.id,
            InterfaceDirection.INPUT,
            item.operand.id,
            item.port.id if isinstance(item, InputInterface) else None,
        )
        for node in network.nodes
        for item in node.region.inputs
    } | {
        QualifiedInterfaceRef(
            node.id,
            InterfaceDirection.OUTPUT,
            item.port.operand.id,
            item.port.id,
        )
        for node in network.nodes
        for item in node.region.outputs
    }
    actual = {item.interface for item in declaration.interface_bindings}
    if actual != expected:
        _fail("selected.binding.coverage", "declaration.interface_bindings", "interfaces differ")
    owned_slots: set[tuple[GraphSlotKind, str, int | None]] = set()
    declared_inputs: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for index, binding in enumerate(declaration.interface_bindings):
        path = f"declaration.interface_bindings.{index}"
        try:
            item = _resolve_interface(network, binding.interface)
        except KeyError as error:
            raise SelectedGraphError(
                "selected.binding.interface", path, "interface is absent"
            ) from error
        operand = (
            item.operand if isinstance(item, (InputInterface, InternalInput)) else item.port.operand
        )
        owner_nodes = region_owners.get(binding.interface.node_id)
        if owner_nodes is None:
            _fail("selected.ownership.region", path, "interface Region owner is absent")
        fact = values.get(binding.graph_value)
        if fact is None:
            _fail("selected.binding.graph_value", path, "graph value is absent")
        if binding.relation.source_domain != operand.position_domain:
            _fail("selected.binding.local_domain", path, "binding has the wrong local domain")
        if binding.relation.target_domain.extents != fact.shape:
            _fail("selected.binding.graph_domain", path, "binding has the wrong graph domain")
        if _required_logical_type(fact, f"tensor.{binding.graph_value}") != operand.element_type:
            _fail("selected.binding.logical_datatype", path, "logical datatypes differ")
        try:
            binding.relation.coordinate_map
        except (TypeError, ValueError) as error:
            raise SelectedGraphError("selected.binding.relation", path, str(error)) from error
        node_inputs = tuple(
            anchor for anchor in binding.anchors if anchor.kind is GraphSlotKind.NODE_INPUT
        )
        node_outputs = tuple(
            anchor for anchor in binding.anchors if anchor.kind is GraphSlotKind.NODE_OUTPUT
        )
        if binding.interface.direction is InterfaceDirection.INPUT:
            if node_outputs or any(
                anchor.kind is GraphSlotKind.GRAPH_OUTPUT for anchor in binding.anchors
            ):
                _fail("selected.binding.anchor_direction", path, "input uses an output anchor")
            if not isinstance(item, InternalInput) and not node_inputs:
                _fail(
                    "selected.binding.consuming_anchor_missing",
                    path,
                    "stream input has no consuming node-input anchor",
                )
        else:
            if node_inputs or any(
                anchor.kind in (GraphSlotKind.GRAPH_INPUT, GraphSlotKind.INITIALIZER)
                for anchor in binding.anchors
            ):
                _fail("selected.binding.anchor_direction", path, "output uses an input anchor")
            if not node_outputs:
                _fail(
                    "selected.binding.defining_anchor_missing",
                    path,
                    "output has no defining node-output anchor",
                )
        for anchor in binding.anchors:
            if _resolve_slot(model, nodes, anchor) != binding.graph_value:
                _fail("selected.binding.anchor_value", path, "anchor and graph value differ")
            if anchor.kind in (GraphSlotKind.NODE_INPUT, GraphSlotKind.NODE_OUTPUT):
                if anchor.owner not in owner_nodes:
                    _fail(
                        "selected.binding.anchor_owner",
                        path,
                        "node anchor is not owned by the interface Region",
                    )
                identity = (anchor.kind, anchor.owner, anchor.index)
                if identity in owned_slots:
                    _fail(
                        "selected.binding.anchor_duplicate",
                        path,
                        "a graph node slot is owned by several interfaces",
                    )
                owned_slots.add(identity)
            if anchor.kind is GraphSlotKind.NODE_INPUT and anchor.index is not None:
                declared_inputs.setdefault(
                    (binding.interface.node_id, binding.graph_value), set()
                ).add((anchor.owner, anchor.index))

    for (region_id, graph_value), declared_input_slots in declared_inputs.items():
        owner_nodes = region_owners[region_id]
        actual_input_slots = {
            (node_id, input_index)
            for node_id in owner_nodes
            for input_index, value in enumerate(nodes[node_id].input)
            if value == graph_value
        }
        if declared_input_slots != actual_input_slots:
            _fail(
                "selected.binding.input_slot_coverage",
                f"region.{region_id}.value.{graph_value}",
                "declared consuming anchors do not cover every node input use",
            )

    for binding in declaration.interface_bindings:
        if binding.interface.direction is not InterfaceDirection.OUTPUT:
            continue
        owner_nodes = region_owners[binding.interface.node_id]
        declared_output_slots = {
            (anchor.owner, anchor.index)
            for anchor in binding.anchors
            if anchor.kind is GraphSlotKind.NODE_OUTPUT and anchor.index is not None
        }
        actual_output_slots = {
            (node_id, output_index)
            for node_id in owner_nodes
            for output_index, value in enumerate(nodes[node_id].output)
            if value == binding.graph_value
        }
        if declared_output_slots != actual_output_slots:
            _fail(
                "selected.binding.output_slot_coverage",
                repr(binding.interface),
                "declared defining anchors do not cover the graph value definition",
            )


def _endpoint_binding(
    declaration: SelectedGraphDeclaration,
    network: DataflowNetwork,
    endpoint: RegionEndpoint,
    direction: InterfaceDirection,
) -> InterfaceBinding:
    region = network.node(endpoint.node_id).region
    if direction is InterfaceDirection.INPUT:
        input_interface = region.input_interface(endpoint.port_id)
        operand_id = input_interface.port.operand.id
    else:
        output_interface = region.output_interface(endpoint.port_id)
        operand_id = output_interface.port.operand.id
    reference = QualifiedInterfaceRef(endpoint.node_id, direction, operand_id, endpoint.port_id)
    return _binding_map(declaration.interface_bindings)[reference]


def _validate_edge_bindings(
    declaration: SelectedGraphDeclaration, network: DataflowNetwork
) -> None:
    for edge in network.edges:
        producer = _endpoint_binding(declaration, network, edge.source, InterfaceDirection.OUTPUT)
        for sink in edge.sinks:
            consumer = _endpoint_binding(
                declaration, network, sink.endpoint, InterfaceDirection.INPUT
            )
            if producer.graph_value != consumer.graph_value:
                _fail(
                    "selected.binding.edge_graph_value",
                    f"edge.{edge.id}",
                    "connected interfaces bind different graph values",
                )
            if (
                producer.relation.kind is not RelationKind.DIRECT
                or consumer.relation.kind is not RelationKind.DIRECT
                or not isinstance(sink.position_map.coordinate_map, IdentityCoordinateMap)
            ):
                _fail(
                    "selected.binding.edge_relation_unsupported",
                    f"edge.{edge.id}",
                    "version 1 edge binding validation supports direct identity bindings",
                )


def _validate_source_boundary_paths(
    declaration: SelectedGraphDeclaration,
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    values: Mapping[str, _GraphValueFact],
) -> frozenset[SourceOperandKey]:
    """Check the bounded graph-input/output view paths owned by source boundaries."""

    owners = {
        owner.owner_id: owner
        for owner in declaration.ownership
        if owner.kind is OwnerKind.SOURCE_BOUNDARY
    }
    graph_inputs = {item.name for item in model.graph.input}
    graph_outputs = {item.name for item in model.graph.output}
    initializer_names = {item.name for item in model.graph.initializer}
    interface_bindings = _binding_map(declaration.interface_bindings)
    supplied_values = {
        interface_bindings[item.required_input].graph_value for item in declaration.supplies
    }
    source_values = {item.key: item for item in declaration.source.operands}
    externally_derived: set[SourceOperandKey] = set()

    def path_map(
        *,
        start: str,
        start_domain: RectangularDomain,
        node_ids: tuple[str, ...],
        path: str,
    ) -> tuple[str, RectangularDomain, CoordinateMap]:
        current = start
        current_shape = start_domain.extents
        transpose_count = 0
        reshape_seen = False
        for node_id in node_ids:
            node = nodes[node_id]
            if node.domain or node.op_type not in {"Identity", "Reshape", "Transpose"}:
                _fail(
                    "selected.source.boundary_unsupported",
                    path,
                    f"unsupported source-boundary node {node.domain}::{node.op_type}",
                )
            if not node.input or node.input[0] != current or len(node.output) != 1:
                _fail(
                    "selected.source.boundary_disconnected",
                    path,
                    f"node {node_id!r} does not continue the source-boundary path",
                )
            output = node.output[0]
            if node.op_type == "Identity":
                output_shape = current_shape
            elif node.op_type == "Transpose":
                attributes = {
                    item.name: onnx_helper.get_attribute_value(item) for item in node.attribute
                }
                permutation = tuple(attributes.get("perm", (1, 0)))
                if len(current_shape) != 2 or permutation != (1, 0):
                    _fail(
                        "selected.source.boundary_unsupported",
                        path,
                        "only two-dimensional source-boundary transpose is supported",
                    )
                output_shape = tuple(reversed(current_shape))
                transpose_count += 1
            else:
                output_shape = _resolve_static_reshape_shape(model, node, current_shape, path)
                reshape_seen = True
            output_fact = values.get(output)
            if output_fact is None or output_fact.shape != output_shape:
                _fail(
                    "selected.source.boundary_shape",
                    path,
                    f"node {node_id!r} output shape disagrees with its static view semantics",
                )
            current = output
            current_shape = output_shape
        target_domain = RectangularDomain(current_shape)
        if transpose_count:
            actual = (
                PositionRelation.transpose_2d(start_domain, target_domain).coordinate_map
                if transpose_count % 2
                else IdentityCoordinateMap(CoordinateSet.full(start_domain), target_domain)
            )
        elif reshape_seen:
            actual = AffineRankMap.row_major_reshape(start_domain, target_domain)
        else:
            actual = IdentityCoordinateMap(CoordinateSet.full(start_domain), target_domain)
        return current, target_domain, actual

    for index, binding in enumerate(declaration.source_bindings):
        path = f"declaration.source_bindings.{index}"
        owner = owners.get(binding.source.operand_id)
        if owner is None:
            direct_boundary = (
                binding.graph_value in graph_inputs
                or binding.graph_value in initializer_names
                or binding.graph_value in supplied_values
                if binding.source.direction is SourceDirection.INPUT
                else binding.graph_value in graph_outputs
            )
            if not direct_boundary:
                _fail(
                    "selected.source.boundary_missing",
                    path,
                    "source binding has neither a direct graph boundary nor an owned adapter",
                )
            continue
        if binding.source.direction is SourceDirection.INPUT:
            first = nodes[owner.node_ids[0]]
            root = first.input[0] if first.input else ""
            root_fact = values.get(root)
            if root not in graph_inputs or root_fact is None:
                _fail(
                    "selected.source.boundary_root",
                    path,
                    "source-boundary input adapter has no graph-input root",
                )
            if root_fact.shape != binding.relation.source_domain.extents:
                _fail(
                    "selected.source.boundary_shape",
                    path,
                    "source-boundary graph input differs from the source domain",
                )
            source_value = source_values[binding.source]
            if (
                root_fact.carrier_dtype != source_value.carrier_dtype
                or _required_logical_type(root_fact, f"tensor.{root}").name
                != source_value.logical_datatype
            ):
                _fail(
                    "selected.source.boundary_datatype",
                    path,
                    "source-boundary graph input datatype differs from source facts",
                )
            target, target_domain, actual = path_map(
                start=root,
                start_domain=binding.relation.source_domain,
                node_ids=owner.node_ids,
                path=path,
            )
            if (
                target != binding.graph_value
                or target_domain != binding.relation.target_domain
                or actual != binding.relation.coordinate_map
            ):
                _fail(
                    "selected.source.boundary_relation",
                    path,
                    "source-boundary input path disagrees with the declared relation",
                )
            externally_derived.add(binding.source)
        else:
            target, target_domain, actual = path_map(
                start=binding.graph_value,
                start_domain=binding.relation.target_domain,
                node_ids=owner.node_ids,
                path=path,
            )
            if binding.relation.kind is RelationKind.DIRECT:
                expected = PositionRelation.direct(
                    binding.relation.target_domain,
                    binding.relation.source_domain,
                ).coordinate_map
            elif binding.relation.kind is RelationKind.ROW_MAJOR_RESHAPE:
                expected = PositionRelation.row_major_reshape(
                    binding.relation.target_domain,
                    binding.relation.source_domain,
                ).coordinate_map
            elif binding.relation.kind is RelationKind.TRANSPOSE_2D:
                expected = PositionRelation.transpose_2d(
                    binding.relation.target_domain,
                    binding.relation.source_domain,
                ).coordinate_map
            else:
                _fail(
                    "selected.source.boundary_relation",
                    path,
                    "output boundary relation has no supported inverse",
                )
            if (
                target not in graph_outputs
                or target_domain != binding.relation.source_domain
                or actual != expected
            ):
                _fail(
                    "selected.source.boundary_relation",
                    path,
                    "source-boundary output path disagrees with the declared relation",
                )
    return frozenset(externally_derived)


def _validate_supplies_and_constants(
    declaration: SelectedGraphDeclaration,
    network: DataflowNetwork,
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    values: Mapping[str, _GraphValueFact],
    externally_derived_sources: frozenset[SourceOperandKey],
) -> None:
    summaries = initializer_value_summaries(model)

    supplied: set[QualifiedInterfaceRef] = set()
    roots: set[str] = set()
    source_values = {item.key: item for item in declaration.source.operands}
    for index, supply in enumerate(declaration.supplies):
        path = f"declaration.supplies.{index}"
        if supply.required_input in supplied:
            _fail("selected.supply.duplicate", path, "input has several supplies")
        try:
            item = _resolve_interface(network, supply.required_input)
        except KeyError as error:
            raise SelectedGraphError("selected.supply.input", path, "input is absent") from error
        if not isinstance(item, InternalInput):
            _fail("selected.supply.not_internal", path, "only InternalInput can have a supply")
        source_value = source_values.get(supply.source)
        if source_value is None or (source_value.initializer_content_digest is None):
            _fail("selected.supply.source", path, "supply content source is absent")
        summary = summaries.get(supply.root_graph_value)
        if summary is None or summary.content_digest != source_value.initializer_content_digest:
            _fail("selected.supply.initializer_digest", path, "initializer content is stale")
        fact = values.get(supply.root_graph_value)
        if fact is None or (
            fact.shape,
            fact.carrier_dtype,
            _required_logical_type(fact, f"tensor.{supply.root_graph_value}").name,
        ) != (
            source_value.shape,
            source_value.carrier_dtype,
            source_value.logical_datatype,
        ):
            _fail("selected.supply.initializer_facts", path, "initializer facts differ")
        if any(node_id not in nodes for node_id in supply.derivation_nodes):
            _fail("selected.supply.derivation", path, "derivation node is absent")
        supplied.add(supply.required_input)
        roots.add(supply.root_graph_value)
    expected_internal = {
        QualifiedInterfaceRef(node.id, InterfaceDirection.INPUT, item.operand.id, None)
        for node in network.nodes
        for item in node.region.internal_inputs
    }
    if supplied != expected_internal:
        _fail(
            "selected.supply.coverage",
            "declaration.supplies",
            "InternalInput supplies are incomplete",
        )

    for index, source_binding in enumerate(declaration.source_bindings):
        source_value = source_values[source_binding.source]
        if not source_value.initializer_present:
            continue
        path = f"declaration.source_bindings.{index}"
        candidates = [
            supply
            for supply in declaration.supplies
            if supply.source == source_binding.source
            and _binding_map(declaration.interface_bindings)[supply.required_input].graph_value
            == source_binding.graph_value
        ]
        if candidates:
            candidate_roots = {
                (
                    supply.root_graph_value,
                    supply.source,
                )
                for supply in candidates
            }
            if len(candidate_roots) != 1:
                _fail(
                    "selected.source.initializer_conflict",
                    path,
                    "qualified supplies disagree about the selected initializer root",
                )
            for supply in candidates:
                _validate_supply_derivation(
                    source_binding,
                    root=supply.root_graph_value,
                    derivation_nodes=supply.derivation_nodes,
                    model=model,
                    nodes=nodes,
                    values=values,
                    path=path,
                )
            continue
        selected_summary = summaries.get(source_binding.graph_value)
        if selected_summary is not None:
            if source_value.initializer_content_digest != selected_summary.content_digest:
                _fail(
                    "selected.source.initializer_digest",
                    path,
                    "selected initializer differs from retained source identity",
                )
            _validate_supply_derivation(
                source_binding,
                root=source_binding.graph_value,
                derivation_nodes=(),
                model=model,
                nodes=nodes,
                values=values,
                path=path,
            )
            continue
        if any(anchor.kind is GraphSlotKind.GRAPH_INPUT for anchor in source_binding.anchors):
            continue  # explicit external runtime supply
        if source_binding.source in externally_derived_sources:
            continue  # checked graph-input boundary adapter
        _fail(
            "selected.source.initializer_root_missing",
            path,
            "source initializer has no selected root, derivation, or external input",
        )

    if not roots.issubset(summaries):
        _fail(
            "selected.supply.initializer_root_missing",
            "declaration.supplies",
            "a declared supply root is absent",
        )


def _validate_supply_derivation(
    binding: SourceValueBinding,
    *,
    root: str,
    derivation_nodes: tuple[str, ...],
    model: ModelWrapper,
    nodes: Mapping[str, Any],
    values: Mapping[str, _GraphValueFact],
    path: str,
) -> None:
    target = binding.graph_value
    if root == target:
        if derivation_nodes:
            _fail(
                "selected.source.initializer_derivation",
                path,
                "direct initializer binding declares an unnecessary derivation",
            )
    elif not derivation_nodes:
        _fail(
            "selected.source.initializer_derivation",
            path,
            "derived selected value has no declared initializer derivation",
        )

    root_matches = [item for item in model.graph.initializer if item.name == root]
    if len(root_matches) != 1:
        _fail(
            "selected.source.initializer_root_missing",
            path,
            "declared initializer root is absent",
        )
    source_domain = binding.relation.source_domain
    current_shape = tuple(int(extent) for extent in root_matches[0].dims)
    root_fact = values.get(root)
    if (
        current_shape != source_domain.extents
        or root_fact is None
        or root_fact.shape != current_shape
    ):
        _fail(
            "selected.source.initializer_derivation_shape",
            path,
            "initializer root shape differs from the declared source domain",
        )

    current = root
    operations: list[str] = []
    transpose_count = 0
    reshape_seen = False
    for node_id in derivation_nodes:
        node = nodes[node_id]
        if node.domain or node.op_type not in {"Identity", "Reshape", "Transpose"}:
            _fail(
                "selected.source.initializer_derivation_unsupported",
                path,
                f"unsupported initializer derivation node {node.domain}::{node.op_type}",
            )
        if not node.input or node.input[0] != current or len(node.output) != 1:
            _fail(
                "selected.source.initializer_derivation_disconnected",
                path,
                f"node {node_id!r} does not continue the declared initializer path",
            )
        output = node.output[0]
        if node.op_type == "Identity":
            output_shape = current_shape
        elif node.op_type == "Transpose":
            attributes = {
                item.name: onnx_helper.get_attribute_value(item) for item in node.attribute
            }
            permutation = tuple(attributes.get("perm", (1, 0)))
            if len(current_shape) != 2 or permutation != (1, 0):
                _fail(
                    "selected.source.initializer_derivation_unsupported",
                    path,
                    "only two-dimensional transpose derivation is supported",
                )
            output_shape = tuple(reversed(current_shape))
            transpose_count += 1
        else:
            output_shape = _resolve_static_reshape_shape(model, node, current_shape, path)
            reshape_seen = True
        output_fact = values.get(output)
        if output_fact is None or output_fact.shape != output_shape:
            _fail(
                "selected.source.initializer_derivation_shape",
                path,
                f"node {node_id!r} output shape disagrees with its static view semantics",
            )
        operations.append(node.op_type)
        current = output
        current_shape = output_shape
    if current != target:
        _fail(
            "selected.source.initializer_derivation_disconnected",
            path,
            "declared initializer path does not produce the bound selected value",
        )

    relation_kind = binding.relation.kind
    if relation_kind is None:
        _fail(
            "selected.source.initializer_derivation_relation",
            path,
            "initializer derivation uses an unsupported coordinate map",
        )
    allowed = {
        RelationKind.DIRECT: {"Identity"},
        RelationKind.ROW_MAJOR_RESHAPE: {"Identity", "Reshape"},
        RelationKind.TRANSPOSE_2D: {"Identity", "Transpose"},
    }[relation_kind]
    if any(operation not in allowed for operation in operations):
        _fail(
            "selected.source.initializer_derivation_relation",
            path,
            "derivation operations do not implement the declared position relation",
        )
    if current_shape != binding.relation.target_domain.extents:
        _fail(
            "selected.source.initializer_derivation_shape",
            path,
            "initializer derivation result shape differs from the declared target domain",
        )

    actual_map: CoordinateMap
    if transpose_count:
        if transpose_count % 2:
            actual_map = PositionRelation.transpose_2d(
                source_domain,
                RectangularDomain(current_shape),
            ).coordinate_map
        else:
            actual_map = IdentityCoordinateMap(
                CoordinateSet.full(source_domain), RectangularDomain(current_shape)
            )
    elif reshape_seen:
        actual_map = AffineRankMap.row_major_reshape(
            source_domain, RectangularDomain(current_shape)
        )
    else:
        actual_map = IdentityCoordinateMap(
            CoordinateSet.full(source_domain), RectangularDomain(current_shape)
        )
    if actual_map != binding.relation.coordinate_map:
        _fail(
            "selected.source.initializer_derivation_relation",
            path,
            "initializer derivation does not implement the declared coordinate relation",
        )


def _resolve_static_reshape_shape(
    model: ModelWrapper,
    node: Any,
    input_shape: tuple[int, ...],
    path: str,
) -> tuple[int, ...]:
    if len(node.input) != 2 or not node.input[1]:
        _fail(
            "selected.source.initializer_derivation_unsupported",
            path,
            "Reshape derivation requires one static shape input",
        )
    shape_name = node.input[1]
    matches = [item for item in model.graph.initializer if item.name == shape_name]
    if len(matches) != 1 or matches[0].data_type != TensorProto.INT64:
        _fail(
            "selected.source.initializer_derivation_unsupported",
            path,
            "Reshape derivation shape must be one int64 initializer",
        )
    shape_value = model.get_initializer(shape_name)
    if shape_value is None or shape_value.ndim != 1:
        _fail(
            "selected.source.initializer_derivation_unsupported",
            path,
            "Reshape derivation shape must be a rank-one initializer",
        )
    attributes = {item.name: onnx_helper.get_attribute_value(item) for item in node.attribute}
    if int(attributes.get("allowzero", 0)) != 0:
        _fail(
            "selected.source.initializer_derivation_unsupported",
            path,
            "Reshape derivation with allowzero enabled is unsupported",
        )

    requested = tuple(int(item) for item in shape_value.tolist())
    resolved: list[int] = []
    inferred_index: int | None = None
    for index, extent in enumerate(requested):
        if extent == 0:
            if index >= len(input_shape):
                _fail(
                    "selected.source.initializer_derivation_shape",
                    path,
                    "Reshape zero dimension has no corresponding input dimension",
                )
            resolved.append(input_shape[index])
        elif extent == -1:
            if inferred_index is not None:
                _fail(
                    "selected.source.initializer_derivation_shape",
                    path,
                    "Reshape shape has more than one inferred dimension",
                )
            inferred_index = index
            resolved.append(-1)
        elif extent < -1:
            _fail(
                "selected.source.initializer_derivation_shape",
                path,
                "Reshape shape contains an invalid negative dimension",
            )
        else:
            resolved.append(extent)

    input_count = RectangularDomain(input_shape).cardinality
    known_count = 1
    for index, extent in enumerate(resolved):
        if index != inferred_index:
            known_count *= extent
    if inferred_index is not None:
        if known_count == 0 or input_count % known_count:
            _fail(
                "selected.source.initializer_derivation_shape",
                path,
                "Reshape inferred dimension is not statically determined",
            )
        resolved[inferred_index] = input_count // known_count
    elif known_count != input_count:
        _fail(
            "selected.source.initializer_derivation_shape",
            path,
            "Reshape changes the initializer element count",
        )
    return tuple(resolved)


def _validate_ownership(
    declaration: SelectedGraphDeclaration,
    network: DataflowNetwork,
    nodes: Mapping[str, Any],
) -> None:
    owned = [node_id for owner in declaration.ownership for node_id in owner.node_ids]
    if len(owned) != len(set(owned)) or set(owned) != set(nodes):
        _fail(
            "selected.ownership.coverage",
            "declaration.ownership",
            "every graph node must have exactly one owner",
        )
    region_ids = {item.id for item in network.nodes}
    source_ids = {item.key.operand_id for item in declaration.source.operands}
    for owner in declaration.ownership:
        if owner.kind is OwnerKind.REGION and owner.owner_id not in region_ids:
            _fail("selected.ownership.region", owner.owner_id, "owning Region is absent")
        if owner.kind is OwnerKind.SOURCE_BOUNDARY and owner.owner_id not in source_ids:
            _fail(
                "selected.ownership.source_boundary",
                owner.owner_id,
                "owning source operand is absent",
            )


def _encode_relation(value: PositionRelation) -> dict[str, object]:
    return encode_coordinate_map(value.coordinate_map)


def _decode_relation(value: object, path: str) -> PositionRelation:
    try:
        coordinate_map = decode_coordinate_map(value)
    except (TypeError, ValueError, KeyError) as error:
        _fail("selected.declaration.relation", path, str(error))
    try:
        return PositionRelation(coordinate_map)
    except (TypeError, ValueError) as error:
        _fail("selected.declaration.relation", path, str(error))


def _encode_interface_ref(value: QualifiedInterfaceRef) -> dict[str, object]:
    return {
        "node_id": value.node_id,
        "direction": value.direction.value,
        "operand_id": value.operand_id,
        "interface_id": value.interface_id,
    }


def _decode_interface_ref(value: object, path: str) -> QualifiedInterfaceRef:
    raw = _fields(value, {"node_id", "direction", "operand_id", "interface_id"}, path)
    interface_id = raw["interface_id"]
    if interface_id is not None:
        interface_id = _string(interface_id, f"{path}.interface_id")
    return QualifiedInterfaceRef(
        _string(raw["node_id"], f"{path}.node_id"),
        InterfaceDirection(_string(raw["direction"], f"{path}.direction")),
        _string(raw["operand_id"], f"{path}.operand_id"),
        interface_id,
    )


def _encode_slot(value: GraphSlotRef) -> dict[str, object]:
    return {
        "kind": value.kind.value,
        "owner": value.owner,
        "index": value.index,
    }


def _decode_slot(value: object, path: str) -> GraphSlotRef:
    raw = _fields(value, {"kind", "owner", "index"}, path)
    index = raw["index"]
    if index is not None:
        index = _integer(index, f"{path}.index")
    return GraphSlotRef(
        GraphSlotKind(_string(raw["kind"], f"{path}.kind")),
        _string(raw["owner"], f"{path}.owner", empty=True),
        index,
    )


def _encode_source_key(value: SourceOperandKey) -> dict[str, object]:
    return {
        "operand_id": value.operand_id,
        "direction": value.direction.value,
        "index": value.index,
    }


def _decode_source_key(value: object, path: str) -> SourceOperandKey:
    raw = _fields(value, {"operand_id", "direction", "index"}, path)
    return SourceOperandKey(
        _string(raw["operand_id"], f"{path}.operand_id"),
        SourceDirection(_string(raw["direction"], f"{path}.direction")),
        _integer(raw["index"], f"{path}.index"),
    )


def _encode_source_value(value: SourceValueRef) -> dict[str, object]:
    return {
        "key": _encode_source_key(value.key),
        "shape": list(value.shape),
        "carrier_dtype": value.carrier_dtype,
        "logical_datatype": value.logical_datatype,
        "initializer_content_digest": value.initializer_content_digest,
    }


def _decode_source_value(value: object, path: str) -> SourceValueRef:
    raw = _fields(
        value,
        {
            "key",
            "shape",
            "carrier_dtype",
            "logical_datatype",
            "initializer_content_digest",
        },
        path,
    )
    digest = raw["initializer_content_digest"]
    if digest is not None:
        digest = _string(digest, f"{path}.initializer_content_digest")
    return SourceValueRef(
        _decode_source_key(raw["key"], f"{path}.key"),
        tuple(_integer(item, path) for item in _sequence(raw["shape"], f"{path}.shape")),
        _integer(raw["carrier_dtype"], f"{path}.carrier_dtype"),
        _string(raw["logical_datatype"], f"{path}.logical_datatype"),
        digest,
    )


def _encode_semantics(value: EncodedSourceSemantics) -> dict[str, object]:
    return {"identity": value.identity, "version": value.version, "payload": value.payload}


def _decode_semantics(value: object, path: str) -> EncodedSourceSemantics:
    raw = _fields(value, {"identity", "version", "payload"}, path)
    if not encoding_is_json_shaped(raw["payload"]):
        _fail("selected.declaration.type", f"{path}.payload", "payload is not JSON-shaped")
    return EncodedSourceSemantics(
        _string(raw["identity"], f"{path}.identity"),
        _integer(raw["version"], f"{path}.version"),
        _freeze_json(raw["payload"]),
    )


def _encode_source(value: SourceProvenance) -> dict[str, object]:
    return {
        "family": value.family,
        "family_version": value.family_version,
        "origin": {
            "native_schema_version": value.schema_version,
            "problem_fingerprint": value.problem_fingerprint,
            "scope_id": value.scope_id,
        },
        "operands": [_encode_source_value(item) for item in value.operands],
        "semantics": _encode_semantics(value.semantics),
    }


def _decode_source(value: object, path: str) -> SourceProvenance:
    raw = _fields(
        value,
        {
            "family",
            "family_version",
            "origin",
            "operands",
            "semantics",
        },
        path,
    )
    origin = _fields(
        raw["origin"],
        {"native_schema_version", "problem_fingerprint", "scope_id"},
        f"{path}.origin",
    )
    scope = origin["scope_id"]
    if scope is not None:
        scope = _string(scope, f"{path}.scope_id")
    return SourceProvenance.create(
        family=_string(raw["family"], f"{path}.family"),
        family_version=_string(raw["family_version"], f"{path}.family_version"),
        schema_version=_integer(
            origin["native_schema_version"], f"{path}.origin.native_schema_version"
        ),
        problem_fingerprint=_string(
            origin["problem_fingerprint"], f"{path}.origin.problem_fingerprint"
        ),
        scope_id=scope,
        operands=tuple(
            _decode_source_value(item, f"{path}.operands")
            for item in _sequence(raw["operands"], f"{path}.operands")
        ),
        semantics=_decode_semantics(raw["semantics"], f"{path}.semantics"),
    )


def _encode_construction(value: ConstructionIdentity) -> dict[str, object]:
    return {
        "family": value.family,
        "version": value.version,
        "form": {
            "identity": value.form,
            "version": value.form_version,
            "arguments": {name: item for name, item in value.form_arguments},
        },
    }


def _decode_construction(value: object, path: str) -> ConstructionIdentity:
    raw = _fields(value, {"family", "version", "form"}, path)
    form = _fields(raw["form"], {"identity", "version", "arguments"}, f"{path}.form")
    arguments = form["arguments"]
    if not isinstance(arguments, Mapping):
        _fail("selected.declaration.type", f"{path}.form.arguments", "expected an object")
    return ConstructionIdentity(
        _string(raw["family"], f"{path}.family"),
        _string(raw["version"], f"{path}.version"),
        _string(form["identity"], f"{path}.form.identity"),
        _integer(form["version"], f"{path}.form.version"),
        tuple(
            sorted(
                (
                    _string(name, f"{path}.form.arguments.name"),
                    _freeze_json(item),
                )
                for name, item in arguments.items()
            )
        ),
    )


def _encode_choice(value: RecordedChoice) -> dict[str, object]:
    encoded = value.value if value.encoding is None else value.encoding
    if not encoding_is_json_shaped(_thaw_json(encoded)):
        raise TypeError("choice value is not JSON-shaped")
    return {"path": value.path, "value": encoded}


def _decode_choice(value: object, path: str) -> RecordedChoice:
    raw = _fields(value, {"path", "value"}, path)
    if not encoding_is_json_shaped(raw["value"]):
        _fail("selected.declaration.type", f"{path}.value", "choice is not JSON-shaped")
    encoded = _freeze_json(raw["value"])
    return RecordedChoice(_string(raw["path"], f"{path}.path"), encoded, encoded)


def _encode_declaration(value: SelectedGraphDeclaration) -> dict[str, object]:
    return {
        "identity": value.identity,
        "version": value.version,
        "graph_digest": value.graph_digest,
        "construction": _encode_construction(value.construction),
        "source": _encode_source(value.source),
        "choices": [_encode_choice(item) for item in value.choices],
        "graph_nodes": [
            {
                "node_id": item.node_id,
                "index": item.index,
            }
            for item in value.graph_nodes
        ],
        "interface_bindings": [
            {
                "interface": _encode_interface_ref(item.interface),
                "graph_value": item.graph_value,
                "relation": _encode_relation(item.relation),
                "anchors": [_encode_slot(anchor) for anchor in item.anchors],
            }
            for item in value.interface_bindings
        ],
        "source_bindings": [
            {
                "source": _encode_source_key(item.source),
                "graph_value": item.graph_value,
                "relation": _encode_relation(item.relation),
                "anchors": [_encode_slot(anchor) for anchor in item.anchors],
            }
            for item in value.source_bindings
        ],
        "supplies": [
            {
                "required_input": _encode_interface_ref(item.required_input),
                "initializer": {
                    "graph_value": item.root_graph_value,
                    "source": _encode_source_key(item.source),
                },
                "derivation_nodes": list(item.derivation_nodes),
            }
            for item in value.supplies
        ],
        "ownership": [
            {
                "kind": item.kind.value,
                "owner_id": item.owner_id,
                "node_ids": list(item.node_ids),
            }
            for item in value.ownership
        ],
    }


def _decode_declaration(value: object) -> SelectedGraphDeclaration:
    names = {
        "identity",
        "version",
        "graph_digest",
        "construction",
        "source",
        "choices",
        "graph_nodes",
        "interface_bindings",
        "source_bindings",
        "supplies",
        "ownership",
    }
    raw = _fields(value, names, "declaration")
    source = _decode_source(raw["source"], "declaration.source")
    source_values = {item.key: item for item in source.operands}
    graph_nodes = []
    for item in _sequence(raw["graph_nodes"], "declaration.graph_nodes"):
        node = _fields(item, {"node_id", "index"}, "declaration.graph_node")
        graph_nodes.append(
            GraphNodeBinding(
                _string(node["node_id"], "graph_node.node_id"),
                _integer(node["index"], "graph_node.index"),
            )
        )
    bindings = []
    for item in _sequence(raw["interface_bindings"], "declaration.interface_bindings"):
        entry = _fields(
            item,
            {"interface", "graph_value", "relation", "anchors"},
            "declaration.interface_binding",
        )
        bindings.append(
            InterfaceBinding(
                _decode_interface_ref(entry["interface"], "interface_binding.interface"),
                _string(entry["graph_value"], "interface_binding.graph_value"),
                _decode_relation(entry["relation"], "interface_binding.relation"),
                tuple(
                    _decode_slot(slot, "interface_binding.anchor")
                    for slot in _sequence(entry["anchors"], "interface_binding.anchors")
                ),
            )
        )
    source_bindings = []
    for item in _sequence(raw["source_bindings"], "declaration.source_bindings"):
        entry = _fields(
            item, {"source", "graph_value", "relation", "anchors"}, "declaration.source_binding"
        )
        source_bindings.append(
            SourceValueBinding(
                _decode_source_key(entry["source"], "source_binding.source"),
                _string(entry["graph_value"], "source_binding.graph_value"),
                _decode_relation(entry["relation"], "source_binding.relation"),
                tuple(
                    _decode_slot(slot, "source_binding.anchor")
                    for slot in _sequence(entry["anchors"], "source_binding.anchors")
                ),
            )
        )
    supplies = []
    for item in _sequence(raw["supplies"], "declaration.supplies"):
        entry = _fields(
            item,
            {
                "required_input",
                "initializer",
                "derivation_nodes",
            },
            "declaration.supply",
        )
        initializer = _fields(
            entry["initializer"],
            {"graph_value", "source"},
            "supply.initializer",
        )
        source_key = _decode_source_key(initializer["source"], "initializer.source")
        try:
            source_value = source_values[source_key]
        except KeyError:
            _fail(
                "selected.supply.source",
                "supply.initializer.source",
                "initializer content source is absent",
            )
        if source_value.initializer_content_digest is None:
            _fail(
                "selected.supply.source",
                "supply.initializer.source",
                "initializer content source has no content digest",
            )
        required_input = _decode_interface_ref(entry["required_input"], "supply.required_input")
        matching_bindings = tuple(
            item.graph_value for item in bindings if item.interface == required_input
        )
        if len(matching_bindings) != 1:
            _fail(
                "selected.supply.binding",
                "supply.required_input",
                "required input has no unique interface binding",
            )
        supplies.append(
            RequiredSupply(
                required_input,
                _string(initializer["graph_value"], "initializer.graph_value"),
                source_key,
                tuple(
                    _string(node, "supply.derivation_node")
                    for node in _sequence(entry["derivation_nodes"], "supply.derivation_nodes")
                ),
            )
        )
    ownership = []
    for item in _sequence(raw["ownership"], "declaration.ownership"):
        entry = _fields(item, {"kind", "owner_id", "node_ids"}, "declaration.owner")
        ownership.append(
            ComputationOwner(
                OwnerKind(_string(entry["kind"], "owner.kind")),
                _string(entry["owner_id"], "owner.owner_id"),
                tuple(
                    _string(node, "owner.node_id")
                    for node in _sequence(entry["node_ids"], "owner.node_ids")
                ),
            )
        )
    return SelectedGraphDeclaration(
        _string(raw["identity"], "declaration.identity"),
        _integer(raw["version"], "declaration.version"),
        _string(raw["graph_digest"], "declaration.graph_digest", empty=True),
        _decode_construction(raw["construction"], "declaration.construction"),
        source,
        tuple(
            _decode_choice(item, "declaration.choice")
            for item in _sequence(raw["choices"], "declaration.choices")
        ),
        tuple(graph_nodes),
        tuple(bindings),
        tuple(source_bindings),
        tuple(supplies),
        tuple(ownership),
    )


__all__ = [
    "ComputationOwner",
    "ConstructionIdentity",
    "ConstructionInputs",
    "ConstructionRegistry",
    "DecodedSelectedGraph",
    "EMPTY_CONSTRUCTION_REGISTRY",
    "EncodedSourceSemantics",
    "GraphNodeBinding",
    "GraphSlotKind",
    "GraphSlotRef",
    "InterfaceBinding",
    "InterfaceDirection",
    "OwnerKind",
    "PositionRelation",
    "QualifiedInterfaceRef",
    "RecordedChoice",
    "RelationKind",
    "RequiredSupply",
    "SELECTED_DECLARATION_ID",
    "SELECTED_DECLARATION_VERSION",
    "SELECTED_METADATA_KEY",
    "SelectionFacts",
    "SelectedConstruction",
    "SelectedGraphDeclaration",
    "SelectedGraphError",
    "SelectedGraphSnapshot",
    "SelectedInitializerInput",
    "SourceDirection",
    "SourceOrigin",
    "SourceOperandKey",
    "SourceProvenance",
    "SourceValueBinding",
    "SourceValueRef",
    "build_selected_snapshot",
    "construct_selected_graph",
    "decode_selected_declaration",
    "decode_selected_choices",
    "decode_selected_graph",
    "encode_selected_declaration",
    "encode_selected_choices",
    "reconstruct_selected_graph",
    "set_frozen_initializer",
    "validate_construction_inputs",
]
