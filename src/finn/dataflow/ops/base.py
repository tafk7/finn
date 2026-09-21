# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""QONNX graph adapters with an explicitly owned immutable source Space."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from qonnx.custom_op.base import CustomOp  # type: ignore[import-not-found]
from finn.dataflow.space.declarations import Space
from finn.dataflow.ops.space import (
    DATAFLOW_DOMAIN,
    DataflowSpace,
    DataflowOpError,
    source_declarations,
    kernel_logical_network,
    unresolved_reason,
)
from finn.dataflow.ops.native import (
    AttributeCodec,
    RESERVED_ATTRIBUTES,
    SCHEMA_VERSION_ATTRIBUTE,
    SCOPE_ID_ATTRIBUTE,
    proposed_choice_values,
    serialize_choices,
    validate_native_encoding,
)
from finn.dataflow.ops.schema import Attribute, DatatypeAttribute, OpOutput, attribute_name


class DataflowOp(CustomOp):  # type: ignore[misc]
    """A model-facing adapter. The declared Space is a distinct immutable object."""

    wants_model: ClassVar[bool] = True
    space_type: ClassVar[type[DataflowSpace]]
    onnx_node: Any

    def __init__(self, onnx_node: Any, onnx_opset_version: int = 1) -> None:
        CustomOp.__init__(self, onnx_node, onnx_opset_version)
        self._model: Any | None = None
        self._space: DataflowSpace | None = None
        self._build: Any = None
        self._graph_context: Any = None
        self._operator_identity = (onnx_node.domain, onnx_node.op_type)
        self._scope_id = next(
            (
                item.s.decode("utf-8")
                for item in onnx_node.attribute
                if item.name == SCOPE_ID_ATTRIBUTE
            ),
            None,
        )

    @property
    def space(self) -> DataflowSpace:
        if self._space is None:
            raise DataflowOpError("model-aware DataflowOp creation has not completed")
        return self._space

    @space.setter
    def space(self, proposal: DataflowSpace) -> None:
        if not isinstance(proposal, self.space_type):
            raise TypeError("the Op space must use its declared source Space type")
        self._space = proposal

    def _attached_model(self) -> Any:
        if self._model is None:
            raise DataflowOpError("DataflowOp has no attached model")
        return self._model

    def _live_node(self, model: Any) -> Any:
        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415

        if not self._scope_id:
            if not any(node is self.onnx_node for node in model.graph.node):
                raise DataflowOpError("the supplied DataflowOp node does not belong to this model")
            node = self.onnx_node
        else:
            node = find_node(model, self._scope_id)
        if (node.domain, node.op_type) != self._operator_identity:
            raise DataflowOpError("current target operator identity differs from this Op")
        return node

    def recorded_scope_id(self) -> str | None:
        return self._scope_id

    def attach_model(self, model: Any) -> DataflowOp:
        from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415

        if self._model is None and not any(node is self.onnx_node for node in model.graph.node):
            raise DataflowOpError("the supplied DataflowOp node does not belong to this model")
        node = self._live_node(model)
        space = build_space(
            self.space_type,
            model,
            node,
            build=self._build,
            graph_context=self._graph_context,
            opset_version=self.onnx_opset_version,
        )
        self._model, self._space = model, space
        self.onnx_node = node
        return self

    def set_context(self, build: Any = None, graph_context: Any = None) -> DataflowOp:
        """Explicitly rebuild the Space with current graph and declared target facts."""
        from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415

        model = self._attached_model()
        validate_native_encoding(self.space_type, self._live_node(model))
        space = build_space(
            self.space_type,
            model,
            self._live_node(model),
            build=build,
            graph_context=graph_context,
            opset_version=self.onnx_opset_version,
            recorded=False,
        )
        space = space.commit_choices(proposed_choice_values(space, self.space))
        self._build, self._graph_context, self._space = build, graph_context, space
        return self

    def save_space(
        self,
        proposal: Space | None = None,
        *,
        require: Any = None,
        require_graph: bool = False,
    ) -> DataflowSpace:
        """Validate proposed choices against the current attached graph, then save."""
        from finn.dataflow.ops.persistence import CommitmentStage, _apply_graph_effects, find_node  # noqa: PLC0415
        from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415

        model = self._attached_model()
        proposed = self.space if proposal is None else proposal
        if not isinstance(proposed, Space):
            raise TypeError("save_space requires a Space proposal")
        node = self._live_node(model)
        if (node.domain, node.op_type) != self._operator_identity or node.domain != DATAFLOW_DOMAIN:
            raise DataflowOpError("current target operator identity differs from this Op")
        validate_native_encoding(self.space_type, node)
        if not self._scope_id:
            from finn.dataflow.ops.persistence import save_unidentified_space  # noqa: PLC0415

            saved = save_unidentified_space(
                self, proposed, require=require, require_graph=require_graph
            )
            self._scope_id = saved.recorded_scope_id()
            self._space = saved
            self.onnx_node = self._live_node(model)
            return saved
        strong = require_graph or require is CommitmentStage.PHYSICAL
        if strong and self._graph_context is None:
            raise DataflowOpError("graph-required save needs a GraphContext")
        current = build_space(
            self.space_type,
            model,
            node,
            build=self._build,
            graph_context=self._graph_context,
            recorded=False,
            opset_version=self.onnx_opset_version,
            fresh=True,
        )
        candidate = current.commit_choices(proposed_choice_values(current, proposed))
        effects = candidate.graph_effects(require=require, require_graph=require_graph)
        scope = candidate.recorded_scope_id()

        def finish(updated: Any) -> DataflowSpace:
            return build_space(
                self.space_type,
                updated,
                find_node(updated, scope),
                build=self._build,
                graph_context=self._graph_context,
                opset_version=self.onnx_opset_version,
                fresh=True,
            )

        saved = _apply_graph_effects(
            model, effects, finish, graph_context=self._graph_context, build=self._build
        )
        # Mutation of the adapter happens only after successful atomic graph commit.
        self._space = saved
        self.onnx_node = self._live_node(model)
        return saved

    def get_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        result: dict[str, tuple[str, bool, object]] = {
            SCOPE_ID_ATTRIBUTE: ("s", False, ""),
            SCHEMA_VERSION_ATTRIBUTE: ("i", False, 0),
        }
        for name, declaration in source_declarations(self.space_type):
            if isinstance(declaration, DatatypeAttribute):
                result[attribute_name(name, declaration)] = (
                    "s",
                    declaration.default is None,
                    declaration.default or "",
                )
            elif isinstance(declaration, Attribute):
                kind = (
                    "s"
                    if declaration.value_type is str
                    else "f"
                    if declaration.value_type is float
                    else "i"
                )
                result[attribute_name(name, declaration)] = (kind, False, declaration.default)
        if self._space is not None:
            for name, attribute in serialize_choices(self.space).items():
                result[name] = (attribute.kind, False, attribute.value)
        return result

    def make_shape_compatible_op(self, model: Any) -> Any:
        from onnx import helper  # type: ignore[import-not-found] # noqa: PLC0415

        self.attach_model(model)
        outputs = self.space.expected_outputs()
        declaration = next(
            (
                name
                for name, decl in source_declarations(self.space_type)
                if isinstance(decl, OpOutput) and decl.index == 0
            ),
            None,
        )
        entry = outputs.get(declaration) if declaration is not None else None
        if entry is None or entry[0] is None:
            raise DataflowOpError(f"{self.onnx_node.name} cannot state a shape-compatible op")
        return helper.make_node(
            "RandomNormal", [], [self.onnx_node.output[0]], shape=list(entry[0])
        )

    def infer_node_datatype(self, model: Any) -> None:
        self.attach_model(model)
        for name, (_, datatype) in self.space.expected_outputs().items():
            if self.space.source.has(name):
                model.set_tensor_datatype(self.space.source.operand(name).tensor, datatype)

    def verify_node(self) -> list[str]:
        assessment = self.space.assess_source()
        if assessment.verdict is True:
            return []
        return [
            finding.message
            for answer in assessment.answers.values()
            for finding in getattr(answer, "findings", ())
        ]

    def execute_node(self, context: Any, graph: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not execute its source semantics")


__all__ = [
    "DATAFLOW_DOMAIN",
    "DataflowOp",
    "DataflowSpace",
    "DataflowOpError",
    "source_declarations",
    "kernel_logical_network",
    "unresolved_reason",
    "AttributeCodec",
    "RESERVED_ATTRIBUTES",
    "SCOPE_ID_ATTRIBUTE",
    "SCHEMA_VERSION_ATTRIBUTE",
]
