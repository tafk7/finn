# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen logical graph contracts and a current-model context provider.

The values in this module are compiler values.  They are deliberately absent
from the native and selected persistence formats: a graph-qualified operation
must obtain them again from a current :class:`GraphContext` before use.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, Protocol, cast

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.model.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    encode_datatype,
)
from finn.dataflow.model.maps import RectangularDomain
from finn.dataflow.model.network import DataflowNetwork, PassCorrespondence
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.mapping import External, OperandMapping
from finn.dataflow.ops.model_effects import (
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    ObservationMutationError,
    checked_model_observation,
    merge_model_read_sets,
    validate_model_read_set,
)
from finn.dataflow.ops.selected import (
    InterfaceDirection,
    QualifiedInterfaceRef,
    RecordedChoice,
    SourceDirection,
    SourceOperandKey,
    SourceOrigin,
)
from finn.dataflow.space.declarations import CanonicalValue, CanonicalValueCodec
from finn.dataflow.space.declarations import AuthoringError
from finn.dataflow.space.occurrence import ProjectionAssessment

if TYPE_CHECKING:
    from finn.dataflow.ops.base import DataflowOp


_CONTEXT_PATH = QualifiedPath("op.graph_context")
_DEFAULT_MAX_CONTRACT_FIELDS = 1_000_000
_active_provider_scopes: ContextVar[tuple[str, ...]] = ContextVar(
    "dataflow_graph_context_scopes", default=()
)


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _source_key_sort(key: SourceOperandKey) -> tuple[str, int, str]:
    return key.direction.value, key.index, key.operand_id


@dataclass(frozen=True, slots=True)
class LogicalBoundaryContract:
    source_shape: tuple[int, ...]
    element_type: QONNXDataType
    beat_sequence: BeatSequence
    pass_correspondence: PassCorrespondence

    def __post_init__(self) -> None:
        shape = tuple(self.source_shape)
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise TypeError("logical boundary source_shape must contain non-negative integers")
        datatype = canonical_qonnx_datatype(self.element_type)
        if not isinstance(self.beat_sequence, BeatSequence):
            raise TypeError("logical boundary beat_sequence must be a BeatSequence")
        if self.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
            raise ValueError("the first graph-context profile requires one-to-one passes")
        sequence = self.beat_sequence.bind_position_domain(RectangularDomain(shape))
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "element_type", datatype)
        object.__setattr__(self, "beat_sequence", sequence)


@dataclass(frozen=True, slots=True)
class GraphInputOrigin:
    interface_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface_id", _nonempty(self.interface_id, "interface_id"))


@dataclass(frozen=True, slots=True)
class OpOutputOrigin:
    producer_scope_id: str
    output: SourceOperandKey
    boundary_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "producer_scope_id",
            _nonempty(self.producer_scope_id, "producer_scope_id"),
        )
        if not isinstance(self.output, SourceOperandKey):
            raise TypeError("Op output origin requires a SourceOperandKey")
        if self.output.direction is not SourceDirection.OUTPUT:
            raise ValueError("Op output origin requires an output source key")
        object.__setattr__(self, "boundary_id", _nonempty(self.boundary_id, "boundary_id"))


@dataclass(frozen=True, slots=True)
class ExternalOperandOrigin:
    interface_id: str
    consumer_scope_id: str
    input: SourceOperandKey
    initializer_content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface_id", _nonempty(self.interface_id, "interface_id"))
        object.__setattr__(
            self,
            "consumer_scope_id",
            _nonempty(self.consumer_scope_id, "consumer_scope_id"),
        )
        if not isinstance(self.input, SourceOperandKey):
            raise TypeError("external operand origin requires a SourceOperandKey")
        if self.input.direction is not SourceDirection.INPUT:
            raise ValueError("external operand origin requires an input source key")
        object.__setattr__(
            self,
            "initializer_content_digest",
            _nonempty(self.initializer_content_digest, "initializer_content_digest"),
        )


ContractOrigin = GraphInputOrigin | OpOutputOrigin | ExternalOperandOrigin


@dataclass(frozen=True, slots=True)
class IncomingLogicalBinding:
    consumer_input: SourceOperandKey
    origin: ContractOrigin
    contract: LogicalBoundaryContract

    def __post_init__(self) -> None:
        if not isinstance(self.consumer_input, SourceOperandKey):
            raise TypeError("incoming binding requires a SourceOperandKey")
        if self.consumer_input.direction is not SourceDirection.INPUT:
            raise ValueError("incoming binding requires an input source key")
        if not isinstance(self.origin, (GraphInputOrigin, OpOutputOrigin, ExternalOperandOrigin)):
            raise TypeError("incoming binding has an unsupported origin")
        if not isinstance(self.contract, LogicalBoundaryContract):
            raise TypeError("incoming binding requires a LogicalBoundaryContract")


@dataclass(frozen=True, slots=True)
class IncomingGraphContext:
    bindings: tuple[IncomingLogicalBinding, ...]

    def __post_init__(self) -> None:
        bindings = tuple(
            sorted(
                tuple(self.bindings),
                key=lambda item: _source_key_sort(item.consumer_input),
            )
        )
        if any(not isinstance(item, IncomingLogicalBinding) for item in bindings):
            raise TypeError("incoming graph context contains a non-binding value")
        keys = tuple(item.consumer_input for item in bindings)
        if len(keys) != len(set(keys)):
            raise ValueError("incoming graph context contains duplicate consumer inputs")
        object.__setattr__(self, "bindings", bindings)


@dataclass(frozen=True, slots=True)
class OutputLogicalContract:
    source_output: SourceOperandKey
    boundary_id: str
    interface: QualifiedInterfaceRef
    contract: LogicalBoundaryContract

    def __post_init__(self) -> None:
        if not isinstance(self.source_output, SourceOperandKey):
            raise TypeError("output logical contract requires a SourceOperandKey")
        if self.source_output.direction is not SourceDirection.OUTPUT:
            raise ValueError("output logical contract requires an output source key")
        object.__setattr__(self, "boundary_id", _nonempty(self.boundary_id, "boundary_id"))
        if not isinstance(self.interface, QualifiedInterfaceRef):
            raise TypeError("output logical contract requires a QualifiedInterfaceRef")
        if self.interface.direction is not InterfaceDirection.OUTPUT:
            raise ValueError("output logical interface must be an output")
        if not isinstance(self.contract, LogicalBoundaryContract):
            raise TypeError("output logical contract requires a LogicalBoundaryContract")


@dataclass(frozen=True, slots=True)
class ContextRead:
    incoming: IncomingGraphContext
    model_reads: ModelReadSet

    def __post_init__(self) -> None:
        if not isinstance(self.incoming, IncomingGraphContext):
            raise TypeError("context read requires an IncomingGraphContext")
        if not isinstance(self.model_reads, ModelReadSet):
            raise TypeError("context read requires a ModelReadSet")


class GraphContext(Protocol):
    def read_inputs(
        self, model: Any, build: object, *, consumer_scope_id: str
    ) -> Answer[ContextRead]: ...


@dataclass(frozen=True, slots=True)
class GraphInputEntry:
    graph_value: str
    interface_id: str
    contract: LogicalBoundaryContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_value", _nonempty(self.graph_value, "graph_value"))
        object.__setattr__(self, "interface_id", _nonempty(self.interface_id, "interface_id"))
        if not isinstance(self.contract, LogicalBoundaryContract):
            raise TypeError("graph input entry requires a LogicalBoundaryContract")


@dataclass(frozen=True, slots=True)
class ExternalOperandEntry:
    graph_value: str
    interface_id: str
    consumer_scope_id: str
    consumer_input: SourceOperandKey
    contract: LogicalBoundaryContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_value", _nonempty(self.graph_value, "graph_value"))
        object.__setattr__(self, "interface_id", _nonempty(self.interface_id, "interface_id"))
        object.__setattr__(
            self,
            "consumer_scope_id",
            _nonempty(self.consumer_scope_id, "consumer_scope_id"),
        )
        if not isinstance(self.consumer_input, SourceOperandKey):
            raise TypeError("external operand entry requires a SourceOperandKey")
        if self.consumer_input.direction is not SourceDirection.INPUT:
            raise ValueError("external operand entry requires an input source key")
        if not isinstance(self.contract, LogicalBoundaryContract):
            raise TypeError("external operand entry requires a LogicalBoundaryContract")


@dataclass(frozen=True, slots=True)
class CurrentGraphContext:
    """Derive input contracts from the current committed graph.

    Graph inputs and initializer-backed streamed operands are explicit entries.
    Dataflow predecessors are rebound recursively from their current native
    choices and their own current inputs.  No derived annotation is cached.
    """

    graph_inputs: tuple[GraphInputEntry, ...] = ()
    external_operands: tuple[ExternalOperandEntry, ...] = ()
    max_contract_fields: int = _DEFAULT_MAX_CONTRACT_FIELDS

    def __post_init__(self) -> None:
        graph_inputs = tuple(sorted(tuple(self.graph_inputs), key=lambda item: item.graph_value))
        external = tuple(
            sorted(
                tuple(self.external_operands),
                key=lambda item: (
                    item.consumer_scope_id,
                    _source_key_sort(item.consumer_input),
                    item.graph_value,
                ),
            )
        )
        if any(not isinstance(item, GraphInputEntry) for item in graph_inputs):
            raise TypeError("graph_inputs must contain GraphInputEntry values")
        if any(not isinstance(item, ExternalOperandEntry) for item in external):
            raise TypeError("external_operands must contain ExternalOperandEntry values")
        if len({item.graph_value for item in graph_inputs}) != len(graph_inputs):
            raise ValueError("graph input entries must have unique graph values")
        external_keys = tuple((item.consumer_scope_id, item.consumer_input) for item in external)
        if len(set(external_keys)) != len(external_keys):
            raise ValueError("external operand entries must have unique consumer slots")
        if type(self.max_contract_fields) is not int or self.max_contract_fields < 0:
            raise TypeError("max_contract_fields must be a non-negative integer")
        object.__setattr__(self, "graph_inputs", graph_inputs)
        object.__setattr__(self, "external_operands", external)

    def read_inputs(
        self, model: Any, build: object, *, consumer_scope_id: str
    ) -> Answer[ContextRead]:
        scope = _nonempty(consumer_scope_id, "consumer_scope_id")
        active = _active_provider_scopes.get()
        if scope in active:
            return _unresolved(
                "graph-context-cycle",
                f"dataflow dependency recursion reached scope {scope!r} twice",
            )
        token = _active_provider_scopes.set((*active, scope))
        try:
            return self._read_current(model, build, scope)
        finally:
            _active_provider_scopes.reset(token)

    def _read_current(self, model: Any, build: object, scope: str) -> Answer[ContextRead]:
        from finn.dataflow.ops.base import DataflowOp, source_declarations  # noqa: PLC0415
        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415
        from finn.dataflow.ops.reconstruction import source_analysis  # noqa: PLC0415
        from finn.dataflow.ops.schema import OpInput  # noqa: PLC0415

        try:
            node = find_node(model, scope)
            unbound = model.get_customop_wrapper(node)
            if not isinstance(unbound, DataflowOp):
                return _unresolved(
                    "graph-context-consumer-not-dataflow",
                    f"scope {scope!r} does not resolve to a DataflowOp",
                )
            with source_analysis(model) as summaries:
                consumer = unbound._bind_with(
                    model,
                    unbound._build_values(build),
                    summaries,
                    recorded=False,
                )
        except ObservationMutationError:
            raise
        except (RequestError, ValueError, KeyError) as error:
            if isinstance(error, AuthoringError):
                raise
            return _unresolved(
                "graph-context-consumer-unavailable",
                f"cannot read source scope {scope!r}: {error}",
            )

        declarations = {
            name: declaration
            for name, declaration in source_declarations(type(consumer))
            if isinstance(declaration, OpInput)
        }
        graph_inputs = {item.graph_value: item for item in self.graph_inputs}
        external = {
            (item.consumer_scope_id, item.consumer_input): item for item in self.external_operands
        }
        initializer_names = {item.name for item in model.graph.initializer}
        graph_input_names = {item.name for item in model.graph.input}
        producer_by_value: dict[str, list[tuple[Any, int]]] = {}
        for producer in model.graph.node:
            for index, value in enumerate(producer.output):
                if value:
                    producer_by_value.setdefault(value, []).append((producer, index))

        bindings: list[IncomingLogicalBinding] = []
        reads: list[ModelReadSet] = []
        for operand in consumer.source.inputs:
            declaration = declarations[operand.id]
            key = SourceOperandKey(operand.id, SourceDirection.INPUT, declaration.index)
            reads.append(
                ModelReadSet(
                    (
                        ModelReadExpectation(
                            ModelReadKind.OPERAND_SLOT,
                            scope,
                            f"input:{declaration.index}",
                            operand.tensor,
                        ),
                    )
                )
            )
            if operand.tensor in initializer_names:
                external_entry = external.get((scope, key))
                if external_entry is None or external_entry.graph_value != operand.tensor:
                    return _unresolved(
                        "graph-context-external-entry-missing",
                        f"initializer input {scope!r}.{key.operand_id!r} needs an explicit "
                        "external operand entry",
                    )
                digest = operand.initializer_digest
                if digest is None:
                    return _unresolved(
                        "graph-context-initializer-digest-missing",
                        f"initializer {operand.tensor!r} has no frozen content digest",
                    )
                bindings.append(
                    IncomingLogicalBinding(
                        key,
                        ExternalOperandOrigin(external_entry.interface_id, scope, key, digest),
                        external_entry.contract,
                    )
                )
                reads.append(
                    ModelReadSet(
                        (
                            ModelReadExpectation(
                                ModelReadKind.INITIALIZER_CONTENT,
                                scope,
                                f"input:{declaration.index}",
                                digest,
                            ),
                        )
                    )
                )
                continue

            producers = producer_by_value.get(operand.tensor, [])
            if producers:
                if len(producers) != 1:
                    return _unresolved(
                        "graph-context-producer-ambiguous",
                        f"graph value {operand.tensor!r} has {len(producers)} producers",
                    )
                producer_node, output_index = producers[0]
                producer = model.get_customop_wrapper(producer_node)
                if not isinstance(producer, DataflowOp):
                    return _unresolved(
                        "graph-context-predecessor-unsupported",
                        f"graph value {operand.tensor!r} is produced by unsupported "
                        f"operator {producer_node.domain!r}:{producer_node.op_type!r}",
                    )
                producer_scope = producer.recorded_scope_id()
                if not producer_scope:
                    return _unresolved(
                        "graph-context-producer-scope-missing",
                        f"producer of {operand.tensor!r} has no dataflow scope id",
                    )
                try:
                    bound = producer.bind(model, build, graph_context=self)
                    accepted_network = bound.graph_dataflow.accepted_answer
                    mappings = bound.operand_mapping
                except ObservationMutationError:
                    raise
                except (RequestError, ValueError, KeyError) as error:
                    if isinstance(error, AuthoringError):
                        raise
                    return _unresolved(
                        "graph-context-producer-unavailable",
                        f"producer {producer_scope!r} is not currently graph-accepted: {error}",
                    )
                if not isinstance(accepted_network, Decided):
                    return cast("Answer[ContextRead]", accepted_network)
                if not isinstance(mappings, Decided):
                    return cast("Answer[ContextRead]", mappings)
                try:
                    outgoing = _output_contracts(
                        bound,
                        accepted_network.value,
                        mappings.value,
                        max_fields=self.max_contract_fields,
                    )
                except (TypeError, ValueError, KeyError) as error:
                    return _unresolved(
                        "graph-context-producer-output-unavailable",
                        f"producer {producer_scope!r} has no usable output contract: {error}",
                    )
                matches = tuple(
                    item for item in outgoing if item.source_output.index == output_index
                )
                if len(matches) != 1:
                    return _unresolved(
                        "graph-context-producer-output-ambiguous",
                        f"producer {producer_scope!r} has {len(matches)} accepted contracts "
                        f"for output slot {output_index}",
                    )
                produced = matches[0]
                bindings.append(
                    IncomingLogicalBinding(
                        key,
                        OpOutputOrigin(
                            producer_scope,
                            produced.source_output,
                            produced.boundary_id,
                        ),
                        produced.contract,
                    )
                )
                reads.append(_operation_current_reads(bound))
                context_read = bound._frozen_context_read()
                if context_read is not None:
                    reads.append(context_read.model_reads)
                reads.append(
                    ModelReadSet(
                        (
                            ModelReadExpectation(
                                ModelReadKind.OPERAND_SLOT,
                                producer_scope,
                                f"output:{output_index}",
                                operand.tensor,
                            ),
                        )
                    )
                )
                continue

            if operand.tensor in graph_input_names:
                graph_entry = graph_inputs.get(operand.tensor)
                if graph_entry is None:
                    return _unresolved(
                        "graph-context-entry-contract-missing",
                        f"graph input {operand.tensor!r} has no explicit logical contract",
                    )
                value_info = tuple(
                    item for item in model.graph.input if item.name == operand.tensor
                )
                if len(value_info) != 1:
                    return _unresolved(
                        "graph-context-entry-ambiguous",
                        f"graph input {operand.tensor!r} is not uniquely declared",
                    )
                bindings.append(
                    IncomingLogicalBinding(
                        key,
                        GraphInputOrigin(graph_entry.interface_id),
                        graph_entry.contract,
                    )
                )
                reads.append(
                    ModelReadSet(
                        (
                            ModelReadExpectation(
                                ModelReadKind.GRAPH_INPUT,
                                operand.tensor,
                                None,
                                value_info[0].SerializeToString(deterministic=True),
                            ),
                        )
                    )
                )
                continue

            return _unresolved(
                "graph-context-source-missing",
                f"input {scope!r}.{key.operand_id!r} names graph value {operand.tensor!r} "
                "with no supported current source",
            )

        try:
            merged = merge_model_read_sets(*reads)
            return Decided(ContextRead(IncomingGraphContext(tuple(bindings)), merged))
        except (TypeError, ValueError) as error:
            return _unresolved("graph-context-read-conflict", str(error))


@dataclass(frozen=True, slots=True)
class AcceptedLogicalCapture:
    source_origin: SourceOrigin
    local_problem_fingerprint: str
    expected_choices: tuple[RecordedChoice, ...]
    source_reads: ModelReadSet
    incoming: IncomingGraphContext
    network: DataflowNetwork
    operand_mappings: tuple[OperandMapping, ...]
    outgoing: tuple[OutputLogicalContract, ...]


@dataclass(frozen=True, slots=True)
class _MappedContract:
    source_key: SourceOperandKey
    boundary_id: str
    interface: QualifiedInterfaceRef
    contract: LogicalBoundaryContract


def checked_context_read(
    graph_context: GraphContext,
    model: Any,
    build: object,
    *,
    consumer_scope_id: str,
) -> Answer[ContextRead]:
    """Invoke a provider through detached and live-wrapper mutation guards."""

    answer = checked_model_observation(
        model,
        lambda detached: graph_context.read_inputs(
            detached,
            build,
            consumer_scope_id=consumer_scope_id,
        ),
        what="graph-context provider",
    )
    if not isinstance(answer, (Decided, Absent, Unresolved)):
        raise TypeError("GraphContext.read_inputs must return an Answer")
    if isinstance(answer, Decided) and not isinstance(answer.value, ContextRead):
        raise TypeError("a decided GraphContext answer must contain ContextRead")
    return answer


def require_context_read(
    graph_context: GraphContext,
    model: Any,
    build: object,
    *,
    consumer_scope_id: str,
) -> ContextRead:
    answer = checked_context_read(
        graph_context,
        model,
        build,
        consumer_scope_id=consumer_scope_id,
    )
    if isinstance(answer, Decided):
        return answer.value
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    findings = answer.findings
    codes = ", ".join(sorted({item.code for item in findings})) or "not decided"
    raise DataflowOpError(f"graph context is unavailable ({codes})", findings)


def graph_dataflow_assessment(operation: DataflowOp) -> ProjectionAssessment[DataflowNetwork]:
    local = operation.dataflow
    answer = local.accepted_answer
    if not isinstance(answer, Decided):
        return ProjectionAssessment(
            "graph_dataflow", local.readiness, local.constraints, local.output, answer
        )
    incoming = operation._frozen_incoming_context()
    if incoming is None:
        graph_answer: Answer[DataflowNetwork] = _unresolved(
            "graph-context-not-bound",
            "this operation was bound without a GraphContext",
        )
    else:
        graph_answer = _validate_graph_contracts(operation, answer.value, incoming)
    return ProjectionAssessment(
        "graph_dataflow", local.readiness, local.constraints, local.output, graph_answer
    )


def outgoing_logical_contracts(
    operation: DataflowOp,
) -> Answer[tuple[OutputLogicalContract, ...]]:
    accepted = operation.graph_dataflow.accepted_answer
    if not isinstance(accepted, Decided):
        return cast("Answer[tuple[OutputLogicalContract, ...]]", accepted)
    mappings = operation.operand_mapping
    if not isinstance(mappings, Decided):
        return cast("Answer[tuple[OutputLogicalContract, ...]]", mappings)
    try:
        return Decided(_output_contracts(operation, accepted.value, mappings.value))
    except (TypeError, ValueError, KeyError) as error:
        return cast(
            "Answer[tuple[OutputLogicalContract, ...]]",
            _rejected("graph-output-contract-invalid", str(error)),
        )


def capture_frozen_op_logical(operation: DataflowOp) -> AcceptedLogicalCapture:
    """Strictly capture one graph-qualified logical candidate at one point."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.native import capture_decided_choices, encode_choice_value  # noqa: PLC0415

    graph = operation.graph_dataflow.accepted_answer
    if not isinstance(graph, Decided):
        findings = graph.findings
        raise DataflowOpError("graph dataflow is not accepted", findings)
    mappings = operation.operand_mapping
    if not isinstance(mappings, Decided):
        findings = mappings.findings
        raise DataflowOpError("operand correspondence is not accepted", findings)
    outgoing = operation.outgoing_logical_contracts
    if not isinstance(outgoing, Decided):
        findings = outgoing.findings
        raise DataflowOpError("outgoing logical contracts are not accepted", findings)
    incoming = operation._frozen_incoming_context()
    if incoming is None:
        raise DataflowOpError("graph-qualified capture has no frozen incoming context")
    captured = capture_decided_choices(operation)
    choices = tuple(
        RecordedChoice(item.choice.path, value, encode_choice_value(item, value))
        for item, value in captured
    )
    scope = operation.recorded_scope_id()
    if not scope:
        raise DataflowOpError("graph-qualified capture requires a source scope id")
    return AcceptedLogicalCapture(
        SourceOrigin(type(operation).schema_version, operation.local_problem_fingerprint, scope),
        operation.local_problem_fingerprint,
        choices,
        _operation_current_reads(operation),
        incoming,
        graph.value,
        mappings.value,
        outgoing.value,
    )


def validate_frozen_op_logical(
    operation: DataflowOp,
    capture: AcceptedLogicalCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> tuple[Finding, ...]:
    """Validate a committed occurrence against one immutable logical capture."""

    from finn.dataflow.ops.persistence import _apply_expected_choices  # noqa: PLC0415

    try:
        validate_model_read_set(model, capture.source_reads)
        current = operation.rebind(model, build, graph_context=graph_context)
        candidate = _apply_expected_choices(current.reconstruct(), capture.expected_choices)
        actual = capture_frozen_op_logical(candidate)
        if actual != capture:
            return _findings(
                "graph-logical-capture-changed",
                "the current source, choices, graph inputs, or logical projection differ "
                "from the frozen capture",
            )
        return ()
    except ObservationMutationError:
        raise
    except AuthoringError:
        raise
    except (ValueError, KeyError) as error:
        findings = getattr(error, "findings", ())
        return tuple(findings) or _findings("graph-logical-validation-failed", str(error))


def _validate_graph_contracts(
    operation: DataflowOp,
    network: DataflowNetwork,
    incoming: IncomingGraphContext,
) -> Answer[DataflowNetwork]:
    mappings = operation.operand_mapping
    if not isinstance(mappings, Decided):
        return cast("Answer[DataflowNetwork]", mappings)
    try:
        keys = _source_keys(operation)
        expected_inputs = {key for key in keys.values() if key.direction is SourceDirection.INPUT}
        by_input = {item.consumer_input: item for item in incoming.bindings}
        if set(by_input) != expected_inputs:
            missing = sorted(expected_inputs - set(by_input), key=_source_key_sort)
            extra = sorted(set(by_input) - expected_inputs, key=_source_key_sort)
            raise ValueError(
                f"incoming bindings do not exactly cover present source inputs; "
                f"missing={missing!r}, extra={extra!r}"
            )
        by_source = {item.source_operand: item for item in mappings.value}
        for name, key in keys.items():
            if key.direction is not SourceDirection.INPUT:
                continue
            mapping = by_source.get(name)
            if mapping is None:
                raise ValueError(f"source input {name!r} has no operand mapping")
            expected = _contract_for_mapping(operation, network, mapping, output=False)
            if by_input[key].contract != expected.contract:
                raise ValueError(f"incoming logical contract for {name!r} is incompatible")
        _output_contracts(operation, network, mappings.value)
        return Decided(network)
    except AuthoringError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        return cast(
            "Answer[DataflowNetwork]",
            _rejected("graph-logical-contract-mismatch", str(error)),
        )


def _output_contracts(
    operation: DataflowOp,
    network: DataflowNetwork,
    mappings: tuple[OperandMapping, ...],
    *,
    max_fields: int = _DEFAULT_MAX_CONTRACT_FIELDS,
) -> tuple[OutputLogicalContract, ...]:
    keys = _source_keys(operation)
    by_source = {item.source_operand: item for item in mappings}
    result = []
    for name, key in keys.items():
        if key.direction is not SourceDirection.OUTPUT:
            continue
        mapping = by_source.get(name)
        if mapping is None:
            raise ValueError(f"source output {name!r} has no operand mapping")
        boundary = _contract_for_mapping(
            operation,
            network,
            mapping,
            output=True,
            max_fields=max_fields,
        )
        result.append(
            OutputLogicalContract(
                boundary.source_key,
                boundary.boundary_id,
                boundary.interface,
                boundary.contract,
            )
        )
    return tuple(sorted(result, key=lambda item: _source_key_sort(item.source_output)))


def _contract_for_mapping(
    operation: DataflowOp,
    network: DataflowNetwork,
    mapping: OperandMapping,
    *,
    output: bool,
    max_fields: int = _DEFAULT_MAX_CONTRACT_FIELDS,
) -> _MappedContract:
    if not isinstance(mapping.placement, External):
        raise ValueError(f"source operand {mapping.source_operand!r} has no selected boundary")
    matches = tuple(
        item
        for item in network.boundaries
        if item.id == mapping.placement.boundary_id
        and item.endpoint.node_id == mapping.placement.node_id
        and item.endpoint.port_id == mapping.placement.port_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"source operand {mapping.source_operand!r} resolves {len(matches)} selected boundaries"
        )
    boundary = matches[0]
    source = operation.source.operand(mapping.source_operand)
    element_type = source.datatype
    if output:
        expected = operation.expected_outputs().get(mapping.source_operand)
        if expected is not None and expected[1] is not None:
            element_type = expected[1]
    sequence = _source_coordinate_sequence(
        mapping,
        boundary.external_beat_sequence,
        max_fields=max_fields,
    )
    direction = InterfaceDirection.OUTPUT if output else InterfaceDirection.INPUT
    key_direction = SourceDirection.OUTPUT if output else SourceDirection.INPUT
    key = _source_keys(operation)[mapping.source_operand]
    if key.direction is not key_direction:
        raise ValueError(f"source operand {mapping.source_operand!r} has the wrong direction")
    interface = QualifiedInterfaceRef(
        mapping.semantic_operand.node_id,
        direction,
        mapping.semantic_operand.operand_id,
        mapping.placement.port_id,
    )
    contract = LogicalBoundaryContract(
        mapping.source_shape,
        element_type,
        sequence,
        boundary.pass_correspondence,
    )
    return _MappedContract(key, boundary.id, interface, contract)


def _source_coordinate_sequence(
    mapping: OperandMapping,
    sequence: BeatSequence,
    *,
    max_fields: int,
) -> BeatSequence:
    count = prod(mapping.source_shape)
    if count > max_fields or sequence.delivered_field_count > max_fields:
        raise ValueError(
            f"logical contract needs more than {max_fields} positions for exact source mapping"
        )
    domain = RectangularDomain(mapping.source_shape)
    inverse: dict[tuple[int, ...], tuple[int, ...]] = {}
    for source in domain.iter_coordinates():
        target = mapping.coordinate_map.mapped(source)
        if target in inverse:
            raise ValueError("source correspondence is not invertible")
        inverse[target] = source
    beats = []
    for beat in sequence.iter_beats(max_fields=max_fields):
        try:
            beats.append(tuple(inverse[position] for position in beat))
        except KeyError as error:
            raise ValueError(
                f"selected boundary presents semantic position {error.args[0]!r} "
                "outside source correspondence"
            ) from error
    return BeatSequence(sequence.elements_per_beat, beats).bind_position_domain(domain)


def _source_keys(operation: DataflowOp) -> Mapping[str, SourceOperandKey]:
    from finn.dataflow.ops.base import source_declarations  # noqa: PLC0415
    from finn.dataflow.ops.schema import OpInput, OpOutput  # noqa: PLC0415

    present = {item.id for item in (*operation.source.inputs, *operation.source.outputs)}
    return {
        name: SourceOperandKey(
            name,
            SourceDirection.OUTPUT if declaration.output else SourceDirection.INPUT,
            declaration.index,
        )
        for name, declaration in source_declarations(type(operation))
        if isinstance(declaration, (OpInput, OpOutput)) and name in present
    }


def _operation_current_reads(operation: DataflowOp) -> ModelReadSet:
    from finn.dataflow.ops.native import RESERVED_ATTRIBUTES, choice_schema  # noqa: PLC0415
    from finn.dataflow.ops.persistence import source_read_set  # noqa: PLC0415

    node = operation._bound_node().materialize()
    names = {item.name for item in choice_schema(operation)} | RESERVED_ATTRIBUTES
    encoded = {
        item.name: item.SerializeToString(deterministic=True)
        for item in node.attribute
        if item.name in names
    }
    return source_read_set(
        operation,
        expected_attributes={name: encoded.get(name) for name in names},
    )


def _unresolved(code: str, message: str) -> Unresolved:
    return Unresolved(_findings(code, message, kind=FindingKind.LIMITATION))


def _rejected(code: str, message: str) -> Absent:
    return Absent(_findings(code, message, kind=FindingKind.REJECTION))


def _findings(
    code: str,
    message: str,
    *,
    kind: FindingKind = FindingKind.REJECTION,
) -> tuple[Finding, ...]:
    return (Finding(kind, code, _CONTEXT_PATH, message),)


def _encode_beat_sequence(value: BeatSequence) -> CanonicalValue:
    if value.is_explicit:
        position_domain = getattr(value, "_position_domain", None)
        return {
            "kind": "explicit",
            "elements_per_beat": value.elements_per_beat,
            "beats": [[list(position) for position in beat] for beat in value.beats],
            "position_domain": None if position_domain is None else list(position_domain.extents),
        }
    affine = value.affine_map
    assert affine is not None
    source, target, offset, terms = affine.normal_form
    return {
        "kind": "affine",
        "elements_per_beat": value.elements_per_beat,
        "beat_count": value.beat_count,
        "source": list(source.extents),
        "target": list(target.extents),
        "offset": offset,
        "terms": [list(item) for item in terms],
    }


def encode_incoming_graph_context(value: IncomingGraphContext) -> CanonicalValue:
    def key(item: SourceOperandKey) -> dict[str, object]:
        return {
            "operand_id": item.operand_id,
            "direction": item.direction.value,
            "index": item.index,
        }

    def origin(item: ContractOrigin) -> dict[str, object]:
        if isinstance(item, GraphInputOrigin):
            return {"kind": "graph_input", "interface_id": item.interface_id}
        if isinstance(item, OpOutputOrigin):
            return {
                "kind": "op_output",
                "producer_scope_id": item.producer_scope_id,
                "output": key(item.output),
                "boundary_id": item.boundary_id,
            }
        return {
            "kind": "external_operand",
            "interface_id": item.interface_id,
            "consumer_scope_id": item.consumer_scope_id,
            "input": key(item.input),
            "initializer_content_digest": item.initializer_content_digest,
        }

    return {
        "bindings": [
            {
                "consumer_input": key(item.consumer_input),
                "origin": origin(item.origin),
                "contract": {
                    "source_shape": list(item.contract.source_shape),
                    "element_type": dict(encode_datatype(item.contract.element_type)),
                    "beat_sequence": _encode_beat_sequence(item.contract.beat_sequence),
                    "pass_correspondence": item.contract.pass_correspondence.value,
                },
            }
            for item in value.bindings
        ]
    }


INCOMING_GRAPH_CONTEXT_CODEC = CanonicalValueCodec(
    "finn.dataflow.incoming_graph_context", 1, encode_incoming_graph_context
)


__all__ = [
    "AcceptedLogicalCapture",
    "ContextRead",
    "ContractOrigin",
    "CurrentGraphContext",
    "ExternalOperandEntry",
    "ExternalOperandOrigin",
    "GraphContext",
    "GraphInputEntry",
    "GraphInputOrigin",
    "INCOMING_GRAPH_CONTEXT_CODEC",
    "IncomingGraphContext",
    "IncomingLogicalBinding",
    "LogicalBoundaryContract",
    "OpOutputOrigin",
    "OutputLogicalContract",
    "capture_frozen_op_logical",
    "checked_context_read",
    "encode_incoming_graph_context",
    "graph_dataflow_assessment",
    "outgoing_logical_contracts",
    "require_context_read",
    "validate_frozen_op_logical",
]
