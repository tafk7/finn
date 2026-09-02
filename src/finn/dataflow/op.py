# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-aware FINN ``DataflowOp`` base and node-attribute persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from importlib import import_module
import json
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, TypeVar, cast
from uuid import uuid4

from onnx import AttributeProto  # type: ignore[import-not-found]
from qonnx.custom_op.base import CustomOp  # type: ignore[import-not-found]

from finn.dataflow.authoring.compiler import (
    CompiledDataflowOperation,
    compile_dataflow_operation,
)
from finn.dataflow.authoring.composition import PhysicalCompositionContext
from finn.dataflow.authoring.persistence import DecisionStorageCodec
from finn.dataflow.authoring.projection import project_build, project_graph
from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.design import (
    Decided,
    DesignPoint,
    DesignSpace,
    DesignSpaceSpec,
    Engine,
    Finding,
    FindingKind,
    ItemOutcome,
    QualifiedPath,
    RequestError,
)
from finn.dataflow.datatypes import encode_datatype, is_qonnx_datatype
from finn.dataflow.op_contracts import DataflowOpError, NodeAttrCodec, NodeAttributeType
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.resolution import ResolvedDataflowOp

if TYPE_CHECKING:
    from finn.dataflow.authoring.inventory import DataflowDesignDeclaration, DataflowOpAuthoring
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]


AssignmentMapping: TypeAlias = (
    Mapping[QualifiedPath, object] | Mapping[str, object] | Mapping[QualifiedPath | str, object]
)
AssociationT = TypeVar("AssociationT")


class DataflowBuildConfigView(Protocol):
    """Build facts used by the generic operation boundary.

    Concrete operation subclasses may require a richer protocol.  FINN's
    ``DataflowBuildConfig`` satisfies this base view.
    """

    @property
    def synth_clk_period_ns(self) -> float: ...


@dataclass(frozen=True)
class DataflowAssignmentCommit:
    """Successful transactional node-backed assignment update."""

    point: DesignPoint
    outcomes: tuple[ItemOutcome, ...]


_ERROR_PATH = QualifiedPath("compiler.dataflow_op")


def _finding(
    kind: FindingKind,
    code: str,
    message: str,
    *,
    path: QualifiedPath = _ERROR_PATH,
    **values: object,
) -> Finding:
    return Finding(kind, code, path, message, tuple(values.items()))


def _canonical_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value, QualifiedPath):
        return {"qualified_path": value.value}
    # Before the dataclass branch: a QONNX datatype is not a dataclass, but
    # being explicit here says that a datatype is fingerprinted by its canonical
    # name and by nothing else -- never by its class or its private width.
    if is_qonnx_datatype(value):
        return encode_datatype(value)
    if isinstance(value, Enum):
        kind = type(value)
        type_token = getattr(
            kind,
            "__dataflow_identity_token__",
            f"{kind.__module__}.{kind.__qualname__}",
        )
        if not isinstance(type_token, str) or not type_token:
            raise TypeError(f"invalid stable identity token for {kind.__name__}")
        return {
            "enum_type": type_token,
            "value": _canonical_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_kind = type(value)
        type_token = getattr(
            dataclass_kind,
            "__dataflow_identity_token__",
            f"{dataclass_kind.__module__}.{dataclass_kind.__qualname__}",
        )
        if not isinstance(type_token, str) or not type_token:
            raise TypeError(f"invalid stable identity token for {dataclass_kind.__name__}")
        return {
            "dataclass_type": type_token,
            "fields": [
                [field.name, _canonical_value(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        pairs = [[_canonical_value(key), _canonical_value(item)] for key, item in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return {"sequence": [_canonical_value(item) for item in value]}
    raise TypeError(f"unsupported problem fingerprint value: {type(value).__name__}")


def dataflow_problem_fingerprint(problem: Mapping[QualifiedPath, object]) -> str:
    """Return a deterministic digest of one complete projected problem."""

    payload = [
        [path.value, _canonical_value(value)]
        for path, value in sorted(problem.items(), key=lambda item: item[0])
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(serialized).hexdigest()


def _compiled_inventory(
    resolved: ResolvedDataflowOp[AssociationT],
) -> tuple[CompiledDataflowOperation, DataflowDesignDeclaration]:
    compiled = resolved.compiled
    if not isinstance(compiled, CompiledDataflowOperation) or compiled.inventory is None:
        raise DataflowOpError(
            (
                _finding(
                    FindingKind.REQUEST,
                    "dataflow-realization-unavailable",
                    "this result has no compiler-owned Design inventory",
                ),
            )
        )
    try:
        declaration = compiled.inventory.declaration(resolved.selected_design_id)
    except KeyError as exc:
        raise DataflowOpError(
            (
                _finding(
                    FindingKind.AUTHORING,
                    "dataflow-selected-design-unknown",
                    f"the resolved Design {resolved.selected_design_id!r} is not in its inventory",
                ),
            )
        ) from exc
    return compiled, declaration


def realize_resolved_dataflow(resolved: ResolvedDataflowOp[AssociationT]) -> DesignRealization:
    """Exactly bind the Design selected by an already-resolved operation."""

    _compiled, declaration = _compiled_inventory(resolved)
    answer = declaration.realize(resolved.engine, resolved.point)
    if not isinstance(answer, Decided):
        raise DataflowOpError(answer.findings)
    if not isinstance(answer.value, DesignRealization):
        raise DataflowOpError(
            (
                _finding(
                    FindingKind.AUTHORING,
                    "dataflow-realization-type-invalid",
                    "the selected Design did not produce a DesignRealization",
                ),
            )
        )
    return answer.value


def physical_composition_context(
    resolved: ResolvedDataflowOp[AssociationT],
    *,
    realization: DesignRealization | None = None,
) -> PhysicalCompositionContext:
    """Project only declared inputs into a Design's composition capability."""

    compiled, declaration = _compiled_inventory(resolved)
    selected_realization = realization or realize_resolved_dataflow(resolved)
    if (
        selected_realization.design_id != declaration.id
        or selected_realization.network != resolved.network
    ):
        raise DataflowOpError(
            (
                _finding(
                    FindingKind.REQUEST,
                    "dataflow-composition-realization-mismatch",
                    "the supplied realization does not match the resolved Design and Network",
                ),
            )
        )
    facts: dict[str, object] = {}
    for name, ref in declaration.composition_inputs.items():
        if ref.path in resolved.point.problem:
            facts[name] = resolved.point.problem[ref.path]
            continue
        answer = resolved.engine.query_property(resolved.point, ref.path)
        if not isinstance(answer, Decided):
            raise DataflowOpError(answer.findings)
        facts[name] = answer.value
    return PhysicalCompositionContext(
        family_id=getattr(compiled.owner, "dataflow_family_id")(),
        family_version=getattr(compiled.owner, "dataflow_family_version")(),
        problem_fingerprint=dataflow_problem_fingerprint(resolved.point.problem),
        assignments=tuple(
            (str(path), value) for path, value in sorted(resolved.point.assignments.items())
        ),
        source_scope_id=resolved.source_scope_id,
        selected_design_id=declaration.id,
        selected_design_version=declaration.version,
        network=resolved.network,
        source_association=resolved.source_association,
        facts=facts,
        kernel_origins=tuple(
            (name, kernel.origin()) for name, kernel in sorted(selected_realization.kernels.items())
        ),
        realization=selected_realization,
    )


def compose_resolved_dataflow(resolved: ResolvedDataflowOp[AssociationT]) -> object:
    """Dispatch composition through the selected Design's compiled declaration."""

    _compiled, declaration = _compiled_inventory(resolved)
    composer = declaration.composer
    if composer is None:
        raise DataflowOpError(
            (
                _finding(
                    FindingKind.LIMITATION,
                    "dataflow-composition-unsupported",
                    f"Design {declaration.id!r} declares no physical composer",
                ),
            )
        )
    if isinstance(composer, str):
        module_name, attribute_name = composer.split(":", 1)
        composer = cast(
            Callable[[PhysicalCompositionContext], object],
            getattr(import_module(module_name), attribute_name),
        )
    return composer(physical_composition_context(resolved))


class DataflowOp(CustomOp):  # type: ignore[misc]
    """Base for logical FINN operations backed by a static design space."""

    wants_model = True

    family_id: ClassVar[str] = ""
    family_version: ClassVar[str] = ""
    declaration_namespace: ClassVar[str] = ""
    uses_class_authoring: ClassVar[bool] = False
    result_member: ClassVar[str] = "result"
    source_association_member: ClassVar[str] = "source_association"
    persistence: ClassVar[tuple[object, ...]] = ()
    selection_constraints: ClassVar[str | None] = None
    structural_constraint_set: ClassVar[str | None] = None
    structural_readiness: ClassVar[str | None] = None
    artifact_readiness: ClassVar[str | None] = None
    feasibility_constraints: ClassVar[tuple[str, ...]] = ()

    SCOPE_ID_ATTR: ClassVar[str] = "dataflow_scope_id"
    FAMILY_ID_ATTR: ClassVar[str] = "dataflow_family_id"
    FAMILY_VERSION_ATTR: ClassVar[str] = "dataflow_family_version"
    PERSISTENCE_FORMAT_ATTR: ClassVar[str] = "dataflow_persistence_format"
    PERSISTENCE_FORMAT_VERSION: ClassVar[str] = "1"
    PROBLEM_FINGERPRINT_ATTR: ClassVar[str] = "dataflow_problem_fingerprint"
    RESERVED_NODEATTRS: ClassVar[frozenset[str]] = frozenset(
        {
            SCOPE_ID_ATTR,
            FAMILY_ID_ATTR,
            FAMILY_VERSION_ATTR,
            PERSISTENCE_FORMAT_ATTR,
            PROBLEM_FINGERPRINT_ATTR,
        }
    )
    _space_cache: ClassVar[dict[type[DataflowOp], DesignSpace]] = {}
    _compiled_cache: ClassVar[dict[type[DataflowOp], CompiledDataflowOperation]] = {}
    _space_cache_lock: ClassVar[RLock] = RLock()

    @classmethod
    def dataflow_family_id(cls) -> str:
        """Return the stable operation-family identity."""

        if cls.family_id:
            return cls.family_id
        raise NotImplementedError(f"{cls.__name__} does not declare a dataflow family id")

    @classmethod
    def dataflow_family_version(cls) -> str:
        """Return the version covering paths, meanings, and persistence codecs."""

        if cls.family_version:
            return cls.family_version
        raise NotImplementedError(f"{cls.__name__} does not declare a dataflow family version")

    @classmethod
    def compiled_dataflow_operation(cls) -> CompiledDataflowOperation | None:
        """Return the cached direct class compilation, when this family uses it."""

        if not cls.uses_class_authoring:
            return None
        with cls._space_cache_lock:
            compiled = cls._compiled_cache.get(cls)
            if compiled is None:
                compiled = compile_dataflow_operation(cls)
                cls._compiled_cache[cls] = compiled
            return compiled

    @classmethod
    def dataflow_authoring(cls) -> DataflowOpAuthoring | None:
        """Return the inventory-backed declaration assembly, when adopted."""

        return None

    @classmethod
    def _required_authoring(cls) -> DataflowOpAuthoring:
        authored = cls.dataflow_authoring()
        if authored is None:
            raise NotImplementedError(f"{cls.__name__} does not declare dataflow authoring")
        return authored

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        """Return the node-independent static superspace for this operation."""

        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.specification
        return cls._required_authoring().specification

    @classmethod
    def result_path(cls) -> QualifiedPath:
        """Return the selected ``DataflowNetwork`` property path."""

        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.result.path
        return cls._required_authoring().result.path

    @classmethod
    def source_association_path(cls) -> QualifiedPath:
        """Return the source-association property path."""

        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.source_association.path
        return cls._required_authoring().source_association.path

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, DecisionStorageCodec]:
        """Map every persistent decision to one stable node attribute."""

        compiled = cls.compiled_dataflow_operation()
        return {} if compiled is None else compiled.persistence

    # -- selection contract ------------------------------------------------
    #
    # A selection policy must not have to know which operation it is looking
    # at.  These classmethods are how an operation family names its own
    # Kernel pools and the constraint sets and readiness profiles a caller
    # should ask about, so the generic selection transform needs no
    # per-operation configuration.

    @classmethod
    def selection_constraint_set(cls) -> str | None:
        """Return the constraint set a complete design point must satisfy."""

        authored = cls.dataflow_authoring()
        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.selection_constraint_set
        return None if authored is None else authored.selection_constraint_set

    @classmethod
    def structural_readiness_profile(cls) -> str | None:
        """Return the profile answering whether the semantic result is ready."""

        authored = cls.dataflow_authoring()
        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.structural_readiness_profile
        return None if authored is None else authored.structural_readiness_profile

    @classmethod
    def artifact_readiness_profile(cls) -> str | None:
        """Return the profile answering whether artifacts can be built."""

        authored = cls.dataflow_authoring()
        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.artifact_readiness_profile
        return None if authored is None else authored.artifact_readiness_profile

    @classmethod
    def feasibility_constraint_sets(cls) -> tuple[str, ...]:
        """Return the per-pool feasibility sets, reported separately."""

        authored = cls.dataflow_authoring()
        compiled = cls.compiled_dataflow_operation()
        if compiled is not None:
            return compiled.feasibility_constraint_sets
        if authored is not None:
            return authored.feasibility_constraint_sets
        return ()

    @classmethod
    def source_nodeattr_types(cls) -> Mapping[str, NodeAttributeType]:
        """Return source-semantic node attributes declared by the subclass."""

        compiled = cls.compiled_dataflow_operation()
        return {} if compiled is None else compiled.projection.source_nodeattrs

    @classmethod
    def validated_design_space(cls) -> DesignSpace:
        """Build and validate the class superspace once per concrete subclass."""

        with cls._space_cache_lock:
            cached = cls._space_cache.get(cls)
            if cached is not None:
                return cached
            family_id = cls.dataflow_family_id()
            family_version = cls.dataflow_family_version()
            if not family_id or not family_version:
                raise TypeError("dataflow family ID and version must not be empty")
            spec = cls.build_design_space_spec()
            space = Engine().validate(spec)
            codecs = dict(cls.decision_nodeattrs())
            attribute_names = tuple(codec.attribute_name for codec in codecs.values())
            if len(set(attribute_names)) != len(attribute_names):
                raise TypeError("persistent decision node-attribute names must be unique")
            source_names = set(cls.source_nodeattr_types())
            collisions = (set(attribute_names) | source_names) & set(cls.RESERVED_NODEATTRS)
            if collisions:
                raise TypeError(
                    "contribution node attributes collide with reserved dataflow metadata: "
                    + ", ".join(sorted(collisions))
                )
            if set(attribute_names) & source_names:
                raise TypeError("source and decision node-attribute names must be disjoint")
            for path, codec in codecs.items():
                declaration = space.decisions.get(path)
                if declaration is None:
                    raise TypeError(f"persistent decision path is not declared: {path}")
                if declaration.value_semantics.type_token is not codec.value_type:
                    raise TypeError(
                        f"node-attribute codec for {path} has incompatible value semantics"
                    )
            for path in (cls.result_path(), cls.source_association_path()):
                if path not in space.properties:
                    raise TypeError(f"required dataflow property path is not declared: {path}")
            cls._space_cache[cls] = space
            return space

    @classmethod
    def clear_validated_space_cache(cls) -> None:
        """Clear this subclass's cache entry, primarily for conformance tests."""

        with cls._space_cache_lock:
            cls._space_cache.pop(cls, None)
            cls._compiled_cache.pop(cls, None)

    def attach_model(self, model: ModelWrapper) -> DataflowOp:
        self._model = model
        return self

    def _attached_model(self) -> ModelWrapper:
        model = getattr(self, "_model", None)
        if model is None:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REQUEST,
                        "dataflow-model-required",
                        "dataflow evaluation requires ModelWrapper.get_customop_wrapper(node)",
                    ),
                )
            )
        return cast("ModelWrapper", model)

    def project_graph_problem(self) -> Mapping[QualifiedPath, object]:
        """Project problem facts owned by the attached graph and target node."""

        compiled = type(self).compiled_dataflow_operation()
        if compiled is None:
            raise NotImplementedError(f"{type(self).__name__} does not declare graph projection")
        projected = project_graph(
            compiled.projection,
            self,
            self._attached_model(),
            self.onnx_node,
            self.dataflow_scope_id(),
        )
        if projected.findings:
            raise DataflowOpError(projected.findings)
        compiled.problem_provenance.check_graph_projection(projected.values)
        return projected.values

    def project_build_problem(
        self, config: DataflowBuildConfigView
    ) -> Mapping[QualifiedPath, object]:
        """Project problem facts owned by the build invocation."""

        compiled = type(self).compiled_dataflow_operation()
        if compiled is None:
            raise NotImplementedError(f"{type(self).__name__} does not declare build projection")
        projected = project_build(compiled.projection, config)
        if projected.findings:
            raise DataflowOpError(projected.findings)
        compiled.problem_provenance.check_build_projection(projected.values)
        return projected.values

    def combine_problem_data(
        self,
        graph_problem: Mapping[QualifiedPath, object],
        build_problem: Mapping[QualifiedPath, object],
    ) -> Mapping[QualifiedPath, object]:
        """Combine disjoint owner projections into one engine problem mapping."""

        return {**graph_problem, **build_problem}

    def problem_instance(self, config: DataflowBuildConfigView) -> Mapping[QualifiedPath, object]:
        """Project and engine-validate one immutable problem snapshot."""

        self._attached_model()
        graph_problem = dict(self.project_graph_problem())
        build_problem = dict(self.project_build_problem(config))
        overlap = graph_problem.keys() & build_problem.keys()
        if overlap:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.AUTHORING,
                        "dataflow-problem-owner-overlap",
                        "graph and build projection hooks produced the same problem field",
                        paths=tuple(sorted(str(path) for path in overlap)),
                    ),
                )
            )
        engine = Engine()
        point = engine.start(
            self.validated_design_space(),
            self.combine_problem_data(graph_problem, build_problem),
        )
        return point.problem

    def validate_declared_source(self) -> Mapping[QualifiedPath, object]:
        """Project only graph-owned declarations for ordinary QONNX verification."""

        self._attached_model()
        return self.project_graph_problem()

    def declared_source_value(self, member: str) -> object:
        """Read one projected class-member fact without exposing model access."""

        compiled = type(self).compiled_dataflow_operation()
        if compiled is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not use class-authored source declarations"
            )
        ref = compiled.declarations.ref(member)
        projected = self.validate_declared_source()
        try:
            return projected[ref.path]
        except KeyError:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-source-value-absent",
                        f"declared source value {member!r} is absent",
                        path=ref.path,
                    ),
                )
            ) from None

    def declared_tensor_shape(self, role: str) -> tuple[int, ...]:
        value = self.declared_source_value(f"{role}.shape")
        if not isinstance(value, tuple) or not all(type(item) is int for item in value):
            raise TypeError(f"declared tensor {role!r} did not project a concrete shape")
        return cast("tuple[int, ...]", value)

    def declared_tensor_datatype(self, role: str) -> object:
        return self.declared_source_value(f"{role}.datatype")

    def get_nodeattr_types(self) -> Mapping[str, NodeAttributeType]:
        attrs = dict(type(self).source_nodeattr_types())
        attrs.update(
            {
                self.SCOPE_ID_ATTR: ("s", False, "", None),
                self.FAMILY_ID_ATTR: ("s", False, "", None),
                self.FAMILY_VERSION_ATTR: ("s", False, "", None),
                self.PROBLEM_FINGERPRINT_ATTR: ("s", False, "", None),
            }
        )
        if type(self).compiled_dataflow_operation() is not None:
            attrs[self.PERSISTENCE_FORMAT_ATTR] = ("s", False, "", None)
        attrs.update(
            {
                codec.attribute_name: codec.nodeattr_definition
                for codec in type(self).decision_nodeattrs().values()
            }
        )
        return MappingProxyType(attrs)

    def _present_attribute(self, name: str) -> AttributeProto | None:
        return next((item for item in self.onnx_node.attribute if item.name == name), None)

    @staticmethod
    def _attribute_scalar(attribute: AttributeProto) -> int | float | str:
        if attribute.type == AttributeProto.INT:
            return int(attribute.i)
        if attribute.type == AttributeProto.STRING:
            return cast(bytes, attribute.s).decode("utf-8")
        if attribute.type == AttributeProto.FLOAT:
            return float(attribute.f)
        raise ValueError("dataflow assignment attributes must be scalar values")

    def read_assignments(self) -> Mapping[QualifiedPath, object]:
        """Decode only physically present decision attributes."""

        assignments: dict[QualifiedPath, object] = {}
        findings: list[Finding] = []
        for path, codec in type(self).decision_nodeattrs().items():
            attribute = self._present_attribute(codec.attribute_name)
            if attribute is None:
                continue
            try:
                assignments[path] = codec.decode(self._attribute_scalar(attribute))
            except (TypeError, ValueError, UnicodeDecodeError) as exc:
                findings.append(
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-node-assignment-malformed",
                        str(exc),
                        path=path,
                        attribute=codec.attribute_name,
                    )
                )
        if findings:
            raise DataflowOpError(tuple(findings))
        return MappingProxyType(assignments)

    def _metadata_value(self, name: str) -> str | None:
        attribute = self._present_attribute(name)
        if attribute is None:
            return None
        try:
            value = self._attribute_scalar(attribute)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-selection-metadata-malformed",
                        "saved dataflow selection metadata is malformed",
                        attribute=name,
                    ),
                )
            ) from exc
        if type(value) is not str or not value:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-selection-metadata-malformed",
                        "saved dataflow selection metadata must be a non-empty string",
                        attribute=name,
                    ),
                )
            )
        return value

    def dataflow_scope_id(self) -> str:
        """Return the stable operation-scope identity stored on the node."""

        value = self._metadata_value(self.SCOPE_ID_ATTR)
        if value is None:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-scope-id-missing",
                        "logical dataflow nodes require a stable scope identity",
                    ),
                )
            )
        return value

    def initialize_dataflow_scope_id(self, scope_id: str | None = None) -> str:
        """Create the operation identity explicitly, independently of selection."""

        existing = self._metadata_value(self.SCOPE_ID_ATTR)
        if existing is not None:
            return existing
        value = f"dataflow_{uuid4().hex}" if scope_id is None else scope_id
        if not isinstance(value, str) or not value:
            raise ValueError("scope_id must be a non-empty string")
        self.set_nodeattr(self.SCOPE_ID_ATTR, value)
        return value

    def renew_dataflow_scope_id(self) -> str:
        """Give a semantic clone a fresh identity and clear copied selections."""

        self.clear_dataflow_assignments()
        value = f"dataflow_{uuid4().hex}"
        self.set_nodeattr(self.SCOPE_ID_ATTR, value)
        return value

    def _selection_metadata(
        self, problem: Mapping[QualifiedPath, object], *, require_present: bool
    ) -> tuple[str, str, str, str] | None:
        scope_id = self._metadata_value(self.SCOPE_ID_ATTR)
        selection_values = tuple(
            self._metadata_value(name)
            for name in (
                self.FAMILY_ID_ATTR,
                self.FAMILY_VERSION_ATTR,
                self.PROBLEM_FINGERPRINT_ATTR,
            )
        )
        if all(value is None for value in selection_values):
            if require_present:
                raise DataflowOpError(
                    (
                        _finding(
                            FindingKind.REJECTION,
                            "dataflow-selection-metadata-missing",
                            "present dataflow assignments require complete identity metadata",
                        ),
                    )
                )
            return None
        if scope_id is None or any(value is None for value in selection_values):
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-selection-metadata-incomplete",
                        "saved dataflow selection identity metadata is incomplete",
                    ),
                )
            )
        family_id, family_version, fingerprint = cast(tuple[str, str, str], selection_values)
        if (family_id, family_version) != (
            type(self).dataflow_family_id(),
            type(self).dataflow_family_version(),
        ):
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-selection-family-mismatch",
                        "saved choices target another dataflow family or family version",
                        actual_family=family_id,
                        actual_version=family_version,
                        expected_family=type(self).dataflow_family_id(),
                        expected_version=type(self).dataflow_family_version(),
                    ),
                )
            )
        if type(self).compiled_dataflow_operation() is not None:
            format_version = self._metadata_value(self.PERSISTENCE_FORMAT_ATTR)
            if format_version != self.PERSISTENCE_FORMAT_VERSION:
                raise DataflowOpError(
                    (
                        _finding(
                            FindingKind.REJECTION,
                            "dataflow-selection-format-mismatch",
                            "saved choices use another persistence format version",
                            actual_version=format_version,
                            expected_version=self.PERSISTENCE_FORMAT_VERSION,
                        ),
                    )
                )
        expected_fingerprint = dataflow_problem_fingerprint(problem)
        if fingerprint != expected_fingerprint:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "dataflow-selection-problem-mismatch",
                        "saved choices do not belong to the current graph and build problem",
                    ),
                )
            )
        return scope_id, family_id, family_version, fingerprint

    @staticmethod
    def _commit_failures(outcomes: tuple[ItemOutcome, ...]) -> tuple[Finding, ...]:
        failures = tuple(
            finding
            for outcome in outcomes
            if outcome.disposition not in {"committed", "unchanged"}
            for finding in outcome.findings
        )
        if failures:
            return failures
        failed_paths = tuple(
            str(outcome.path)
            for outcome in outcomes
            if outcome.disposition not in {"committed", "unchanged"}
        )
        if failed_paths:
            return (
                _finding(
                    FindingKind.REJECTION,
                    "dataflow-assignment-commit-failed",
                    "one or more dataflow assignments could not be committed",
                    paths=failed_paths,
                ),
            )
        return ()

    def _hydrate_problem(self, problem: Mapping[QualifiedPath, object]) -> DesignPoint:
        assignments = self.read_assignments()
        self._selection_metadata(problem, require_present=bool(assignments))
        engine = Engine()
        point = engine.start(self.validated_design_space(), problem)
        if not assignments:
            return point
        try:
            committed = engine.commit_assignments(point, assignments)
        except RequestError as exc:
            raise DataflowOpError(exc.findings) from exc
        failures = self._commit_failures(committed.outcomes)
        if failures:
            raise DataflowOpError(failures)
        return committed.point

    def hydrate_dataflow_point(self, config: DataflowBuildConfigView) -> DesignPoint:
        """Rebuild a possibly partial point from live facts and present attributes."""

        return self._hydrate_problem(self.problem_instance(config))

    def resolve_dataflow(self, config: DataflowBuildConfigView) -> ResolvedDataflowOp[Any]:
        """Hydrate the point and require a selected logical result and association."""

        point = self.hydrate_dataflow_point(config)
        engine = Engine()
        result = engine.query_property(point, type(self).result_path())
        association = engine.query_property(point, type(self).source_association_path())
        findings = tuple(
            finding
            for answer in (result, association)
            if not isinstance(answer, Decided)
            for finding in answer.findings
        )
        if findings or not isinstance(result, Decided) or not isinstance(association, Decided):
            raise DataflowOpError(
                findings
                or (
                    _finding(
                        FindingKind.BLOCKER,
                        "dataflow-result-unresolved",
                        "the dataflow result or source association is not resolved",
                    ),
                )
            )
        selected = result.value
        if isinstance(selected, DataflowNetwork):
            selected_network = selected
            fallback_design_id = "network"
        else:
            raise DataflowOpError(
                (
                    _finding(
                        FindingKind.AUTHORING,
                        "dataflow-result-type-invalid",
                        "the selected Network property did not produce DataflowNetwork",
                    ),
                )
            )
        compiled = type(self).compiled_dataflow_operation()
        selected_design_id = fallback_design_id
        if compiled is not None and compiled.inventory is not None:
            selected_design = compiled.inventory.selected(point)
            if not isinstance(selected_design, Decided):
                raise DataflowOpError(selected_design.findings)
            selected_design_id = selected_design.value.id
        return ResolvedDataflowOp(
            engine,
            point,
            self.dataflow_scope_id(),
            selected_design_id,
            selected_network,
            association.value,
            compiled,
        )

    def realize_dataflow(self, config: DataflowBuildConfigView) -> DesignRealization:
        """Resolve and exactly bind the selected class-authored Design inventory."""

        return realize_resolved_dataflow(self.resolve_dataflow(config))

    def physical_composition_context(
        self, config: DataflowBuildConfigView
    ) -> PhysicalCompositionContext:
        return physical_composition_context(self.resolve_dataflow(config))

    def compose_dataflow(self, config: DataflowBuildConfigView) -> object:
        """Dispatch the selected Design's registered restricted composer."""

        return compose_resolved_dataflow(self.resolve_dataflow(config))

    def _encode_assignments(
        self, assignments: Mapping[QualifiedPath, object]
    ) -> dict[str, int | float | str]:
        codecs = type(self).decision_nodeattrs()
        encoded: dict[str, int | float | str] = {}
        for path, value in assignments.items():
            codec = codecs.get(path)
            if codec is None:
                raise DataflowOpError(
                    (
                        _finding(
                            FindingKind.REQUEST,
                            "dataflow-assignment-not-persistent",
                            "the requested decision has no node-attribute codec",
                            path=path,
                        ),
                    )
                )
            try:
                encoded[codec.attribute_name] = codec.encode(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise DataflowOpError(
                    (
                        _finding(
                            FindingKind.REJECTION,
                            "dataflow-assignment-encode-failed",
                            str(exc),
                            path=path,
                            attribute=codec.attribute_name,
                        ),
                    )
                ) from exc
        return encoded

    def _write_selection(
        self,
        problem: Mapping[QualifiedPath, object],
        assignments: Mapping[QualifiedPath, object],
        *,
        replace: bool,
    ) -> None:
        encoded = self._encode_assignments(assignments)
        scope_id = self.dataflow_scope_id()
        metadata = {
            self.SCOPE_ID_ATTR: scope_id,
            self.FAMILY_ID_ATTR: type(self).dataflow_family_id(),
            self.FAMILY_VERSION_ATTR: type(self).dataflow_family_version(),
            self.PROBLEM_FINGERPRINT_ATTR: dataflow_problem_fingerprint(problem),
        }
        if type(self).compiled_dataflow_operation() is not None:
            metadata[self.PERSISTENCE_FORMAT_ATTR] = self.PERSISTENCE_FORMAT_VERSION
        snapshot = self.onnx_node.SerializeToString(deterministic=True)
        try:
            if replace:
                names = {codec.attribute_name for codec in type(self).decision_nodeattrs().values()}
                kept = [
                    attribute
                    for attribute in self.onnx_node.attribute
                    if attribute.name not in names
                ]
                del self.onnx_node.attribute[:]
                self.onnx_node.attribute.extend(kept)
            for name, value in {**encoded, **metadata}.items():
                self.set_nodeattr(name, value)
        except Exception:
            restored = type(self.onnx_node)()
            restored.ParseFromString(snapshot)
            self.onnx_node.CopyFrom(restored)
            raise

    def _commit_replacement(
        self,
        config: DataflowBuildConfigView,
        assignments: AssignmentMapping,
        *,
        replace: bool,
    ) -> DataflowAssignmentCommit:
        problem = self.problem_instance(config)
        engine = Engine()
        point = (
            engine.start(self.validated_design_space(), problem)
            if replace
            else self._hydrate_problem(problem)
        )
        requested_paths = tuple(QualifiedPath.parse(path) for path in assignments)
        missing_codecs = tuple(
            path for path in requested_paths if path not in type(self).decision_nodeattrs()
        )
        if missing_codecs:
            raise DataflowOpError(
                tuple(
                    _finding(
                        FindingKind.REQUEST,
                        "dataflow-assignment-not-persistent",
                        "the requested decision has no node-attribute codec",
                        path=path,
                    )
                    for path in missing_codecs
                )
            )
        try:
            committed = engine.commit_assignments(point, assignments)
        except RequestError as exc:
            raise DataflowOpError(exc.findings) from exc
        failures = self._commit_failures(committed.outcomes)
        if failures:
            raise DataflowOpError(failures)
        if not committed.point.assignments:
            if replace:
                self.clear_dataflow_assignments()
            return DataflowAssignmentCommit(committed.point, committed.outcomes)
        self._write_selection(problem, committed.point.assignments, replace=replace)
        return DataflowAssignmentCommit(committed.point, committed.outcomes)

    def commit_dataflow_assignments(
        self,
        config: DataflowBuildConfigView,
        assignments: AssignmentMapping,
    ) -> DataflowAssignmentCommit:
        """Validate and append commitments before changing the ONNX node."""

        return self._commit_replacement(config, assignments, replace=False)

    def replace_dataflow_assignments(
        self,
        config: DataflowBuildConfigView,
        assignments: AssignmentMapping,
    ) -> DataflowAssignmentCommit:
        """Validate and replace the complete persistent assignment map."""

        return self._commit_replacement(config, assignments, replace=True)

    def clear_dataflow_assignments(self) -> None:
        """Remove choices and selection metadata while preserving scope identity."""

        names = {
            self.FAMILY_ID_ATTR,
            self.FAMILY_VERSION_ATTR,
            self.PERSISTENCE_FORMAT_ATTR,
            self.PROBLEM_FINGERPRINT_ATTR,
        }
        names.update(codec.attribute_name for codec in type(self).decision_nodeattrs().values())
        kept = [attribute for attribute in self.onnx_node.attribute if attribute.name not in names]
        del self.onnx_node.attribute[:]
        self.onnx_node.attribute.extend(kept)


__all__ = [
    "DataflowAssignmentCommit",
    "AssignmentMapping",
    "DataflowBuildConfigView",
    "DataflowOp",
    "DataflowOpError",
    "NodeAttrCodec",
    "NodeAttributeType",
    "compose_resolved_dataflow",
    "dataflow_problem_fingerprint",
    "physical_composition_context",
    "realize_resolved_dataflow",
]
